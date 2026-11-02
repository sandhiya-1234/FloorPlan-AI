"""
Evaluation Metrics Suite for FloorPlan-AI.

Implements rigorous spatial, geometric, and classification metrics using
Hungarian assignment (scipy.optimize.linear_sum_assignment):
- Object Detection: Precision, Recall, F1, Micro-F1, Macro-F1 per architectural class
- Geometric Accuracy: Polygon IoU (Shapely), Bounding-box IoU, Wall endpoint matching
- Semantic Accuracy: Room functional type classification accuracy
- Coordinate Errors: Mean Euclidean pixel deviation on matched elements
- JSON Quality: Schema validity rate, required field completeness
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from schemas.floorplan_schema import FloorPlanSchema

logger = logging.getLogger(__name__)


def compute_bbox_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU between two [xmin, ymin, xmax, ymax] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return float(inter / union) if union > 0 else 0.0


def compute_polygon_iou(poly1_pts: List[List[float]], poly2_pts: List[List[float]]) -> float:
    """Computes geometric polygon IoU using Shapely."""
    if len(poly1_pts) < 3 or len(poly2_pts) < 3:
        return 0.0
    from shapely.geometry import Polygon  # lazy: only polygon metrics need shapely
    try:
        p1 = Polygon(poly1_pts).buffer(0)
        p2 = Polygon(poly2_pts).buffer(0)
        if not p1.is_valid or not p2.is_valid or p1.is_empty or p2.is_empty:
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_wall_matching_error(
    w1_start: List[float], w1_end: List[float],
    w2_start: List[float], w2_end: List[float]
) -> float:
    """Computes symmetric endpoint distance between two wall segments."""
    p1_s, p1_e = np.array(w1_start, dtype=np.float32), np.array(w1_end, dtype=np.float32)
    p2_s, p2_e = np.array(w2_start, dtype=np.float32), np.array(w2_end, dtype=np.float32)

    # Orientation 1: start->start, end->end
    d1 = np.linalg.norm(p1_s - p2_s) + np.linalg.norm(p1_e - p2_e)
    # Orientation 2: start->end, end->start
    d2 = np.linalg.norm(p1_s - p2_e) + np.linalg.norm(p1_e - p2_s)
    return float(min(d1, d2) / 2.0)


def match_elements_hungarian(
    pred_items: List[Dict[str, Any]],
    gt_items: List[Dict[str, Any]],
    match_type: str = "bbox",
    threshold: float = 0.5
) -> Tuple[int, int, int, List[float], Dict[str, Any]]:
    """
    Optimal bipartite matching between predictions and ground-truth using the Hungarian algorithm.
    Returns (TP, FP, FN, matched_scores, extra_info).
    """
    n_pred = len(pred_items)
    n_gt = len(gt_items)

    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0, [], {"mean_coord_error": 0.0, "semantic_matches": 0, "semantic_accuracy": 0.0}
    if n_pred == 0:
        return 0, 0, n_gt, [], {"mean_coord_error": 0.0, "semantic_matches": 0, "semantic_accuracy": 0.0}
    if n_gt == 0:
        return 0, n_pred, 0, [], {"mean_coord_error": 0.0, "semantic_matches": 0, "semantic_accuracy": 0.0}

    # Construct cost matrix
    cost_matrix = np.full((n_pred, n_gt), fill_value=1e6, dtype=np.float64)

    for i, pred in enumerate(pred_items):
        for j, gt in enumerate(gt_items):
            if match_type == "bbox":
                iou = compute_bbox_iou(pred["bbox"], gt["bbox"])
                if iou >= threshold:
                    cost_matrix[i, j] = 1.0 - iou
            elif match_type == "polygon":
                iou = compute_polygon_iou(pred["polygon"], gt["polygon"])
                if iou >= threshold:
                    cost_matrix[i, j] = 1.0 - iou
            elif match_type == "wall":
                err = compute_wall_matching_error(pred["start"], pred["end"], gt["start"], gt["end"])
                if err <= threshold:
                    cost_matrix[i, j] = err

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_scores = []
    coord_errors = []
    matched_pairs = []
    semantic_matches = 0
    tp = 0

    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost < 1e5:  # Valid assignment under threshold
            tp += 1
            matched_pairs.append((int(r), int(c)))
            if match_type in ["bbox", "polygon"]:
                score = 1.0 - cost
                matched_scores.append(float(score))
                # Compute center coordinate error
                if match_type == "bbox":
                    p_c = [(pred_items[r]["bbox"][0] + pred_items[r]["bbox"][2]) / 2,
                           (pred_items[r]["bbox"][1] + pred_items[r]["bbox"][3]) / 2]
                    g_c = [(gt_items[c]["bbox"][0] + gt_items[c]["bbox"][2]) / 2,
                           (gt_items[c]["bbox"][1] + gt_items[c]["bbox"][3]) / 2]
                    coord_errors.append(float(np.linalg.norm(np.array(p_c) - np.array(g_c))))
                elif match_type == "polygon":
                    if pred_items[r].get("type", "").lower() == gt_items[c].get("type", "").lower():
                        semantic_matches += 1
            elif match_type == "wall":
                matched_scores.append(float(cost))
                coord_errors.append(float(cost))

    fp = n_pred - tp
    fn = n_gt - tp

    extra_info = {
        "mean_coord_error": float(np.mean(coord_errors)) if coord_errors else 0.0,
        "semantic_matches": semantic_matches,
        "semantic_accuracy": float(semantic_matches / tp) if tp > 0 else 0.0,
        "matched_pairs": matched_pairs,  # (pred_index, gt_index) for every true positive
    }

    return tp, fp, fn, matched_scores, extra_info


def evaluate_single_prediction(
    pred_schema: FloorPlanSchema,
    gt_schema: FloorPlanSchema
) -> Dict[str, Any]:
    """
    Evaluates one prediction against ground truth across all classes using Hungarian matching.
    """
    class_metrics = {}

    # Every class reports "match_scores": per-true-positive IoU (doors/windows/rooms) or
    # endpoint error in px (walls), so aggregation can pool matches instead of averaging averages.

    # 1. Walls (Threshold: 25.0 pixels endpoint distance)
    tp, fp, fn, scores, extra = match_elements_hungarian(
        [w.model_dump() for w in pred_schema.walls],
        [w.model_dump() for w in gt_schema.walls],
        match_type="wall",
        threshold=25.0
    )
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    class_metrics["walls"] = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_endpoint_error_px": round(extra["mean_coord_error"], 2),
        "match_scores": [round(float(s), 3) for s in scores],
        "tp": tp, "fp": fp, "fn": fn,
        "pred_count": len(pred_schema.walls),
        "gt_count": len(gt_schema.walls)
    }

    # 2. Doors (Threshold: IoU >= 0.5)
    tp, fp, fn, scores, extra = match_elements_hungarian(
        [d.model_dump() for d in pred_schema.doors],
        [d.model_dump() for d in gt_schema.doors],
        match_type="bbox",
        threshold=0.5
    )
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    class_metrics["doors"] = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(scores)), 4) if scores else 0.0,
        "mean_coord_error_px": round(extra["mean_coord_error"], 2),
        "match_scores": [round(float(s), 4) for s in scores],
        "tp": tp, "fp": fp, "fn": fn,
        "pred_count": len(pred_schema.doors),
        "gt_count": len(gt_schema.doors)
    }

    # 3. Windows (Threshold: IoU >= 0.5)
    tp, fp, fn, scores, extra = match_elements_hungarian(
        [w.model_dump() for w in pred_schema.windows],
        [w.model_dump() for w in gt_schema.windows],
        match_type="bbox",
        threshold=0.5
    )
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    class_metrics["windows"] = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(scores)), 4) if scores else 0.0,
        "mean_coord_error_px": round(extra["mean_coord_error"], 2),
        "match_scores": [round(float(s), 4) for s in scores],
        "tp": tp, "fp": fp, "fn": fn,
        "pred_count": len(pred_schema.windows),
        "gt_count": len(gt_schema.windows)
    }

    # 4. Rooms (Threshold: Polygon IoU >= 0.5)
    tp, fp, fn, scores, extra = match_elements_hungarian(
        [r.model_dump() for r in pred_schema.rooms],
        [r.model_dump() for r in gt_schema.rooms],
        match_type="polygon",
        threshold=0.5
    )
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    class_metrics["rooms"] = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(scores)), 4) if scores else 0.0,
        "semantic_accuracy": round(extra["semantic_accuracy"], 4),
        "match_scores": [round(float(s), 4) for s in scores],
        # (predicted_type, ground_truth_type) for each matched room -> confusion matrix
        "room_type_pairs": [
            (pred_schema.rooms[p].type.lower(), gt_schema.rooms[g].type.lower())
            for p, g in extra["matched_pairs"]
        ],
        "tp": tp, "fp": fp, "fn": fn,
        "pred_count": len(pred_schema.rooms),
        "gt_count": len(gt_schema.rooms)
    }

    return class_metrics
