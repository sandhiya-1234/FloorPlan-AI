"""
Geometric Postprocessing & Schema Sanitation for FloorPlan-AI.

Implements architectural geometric refinement:
- Orthogonal wall snapping (watertight CAD line networks)
- Polygon topology validation and area re-computation (using Shapely)
- Duplicate entity filtering via IoU suppression
- Door and window snapping to structural wall centerlines
- Markdown/JSON syntax repair and Pydantic validation
"""

import re
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from schemas.floorplan_schema import FloorPlanSchema

logger = logging.getLogger(__name__)


class ModelOutputError(ValueError):
    """The model output could not be parsed into a floor-plan dict. Never replaced by defaults."""


def _opt_float(value: Any) -> Optional[float]:
    """float(value), or None when the model did not provide it (no default is invented)."""
    return None if value is None else float(value)


def extract_json_from_text(raw_text: str) -> str:
    """Extracts raw JSON substring from model output, stripping markdown fences."""
    text = raw_text.strip()
    # Strip ```json ... ``` markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Search for outer curly braces
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        return brace_match.group(1).strip()

    return text


def calculate_bbox_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection over Union for two [xmin, ymin, xmax, ymax] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def remove_duplicate_boxes(items: List[Dict[str, Any]], iou_thresh: float = 0.75) -> List[Dict[str, Any]]:
    """Filters duplicate bounding boxes using Non-Maximum Suppression style IoU."""
    if not items:
        return []
    filtered = []
    for item in items:
        b1 = item["bbox"]
        is_dup = False
        for kept in filtered:
            b2 = kept["bbox"]
            if calculate_bbox_iou(b1, b2) > iou_thresh:
                is_dup = True
                break
        if not is_dup:
            filtered.append(item)
    return filtered


def snap_wall_orthogonality(
    start: List[float], end: List[float], angle_tolerance_deg: float = 4.0
) -> Tuple[List[float], List[float], bool]:
    """
    Snaps wall segments close to horizontal or vertical axes to strict CAD orthogonality.
    Returns (snapped_start, snapped_end, was_snapped). Preserves intentional diagonals.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle_deg = float(np.degrees(np.arctan2(abs(dy), abs(dx))))

    # If within angle_tolerance of horizontal (0 deg)
    if angle_deg <= angle_tolerance_deg:
        avg_y = (start[1] + end[1]) / 2.0
        return [start[0], avg_y], [end[0], avg_y], True

    # If within angle_tolerance of vertical (90 deg)
    if abs(angle_deg - 90.0) <= angle_tolerance_deg:
        avg_x = (start[0] + end[0]) / 2.0
        return [avg_x, start[1]], [avg_x, end[1]], True

    return start, end, False


def validate_and_refine_polygon(pts: List[List[float]], min_area: float = 100.0) -> Tuple[Optional[List[List[float]]], bool]:
    """
    Validates polygon vertices with Shapely and resolves self-intersections.
    Returns (refined_pts, was_repaired).
    """
    if len(pts) < 3:
        return None, False
    from shapely.geometry import Polygon, MultiPolygon  # lazy: only polygon repair needs shapely
    try:
        poly = Polygon(pts)
        repaired = False
        if not poly.is_valid:
            poly = poly.buffer(0)
            repaired = True
        if poly.is_empty:
            return None, False
        if isinstance(poly, MultiPolygon):
            # Pick largest polygon component
            poly = max(poly.geoms, key=lambda p: p.area)
            repaired = True
        if poly.area < min_area:
            return None, False
        coords = list(poly.exterior.coords)[:-1]  # remove duplicate closing coord
        return [[round(c[0], 2), round(c[1], 2)] for c in coords], repaired
    except Exception:
        return pts, False


def snap_element_to_nearest_wall(
    position: List[float],
    walls: List[Dict[str, Any]],
    portal_orientation: Optional[str] = None,
    max_distance: float = 15.0
) -> Tuple[List[float], bool]:
    """
    Snaps door/window midpoint to nearest wall centerline for structural consistency.
    Enforces maximum snap distance (15px) and orientation compatibility to prevent false snapping.
    Returns (snapped_position, was_snapped).
    """
    if not position or not walls:
        return position, False
    pos_pt = np.array(position, dtype=np.float32)
    best_dist = float("inf")
    snapped_pos = position
    was_snapped = False

    for w in walls:
        p1 = np.array(w["start"], dtype=np.float32)
        p2 = np.array(w["end"], dtype=np.float32)
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-4:
            continue

        # Check orientation consistency
        wall_is_horiz = abs(line_vec[0]) >= abs(line_vec[1])
        if portal_orientation == "horizontal" and not wall_is_horiz:
            continue
        if portal_orientation == "vertical" and wall_is_horiz:
            continue

        line_unit = line_vec / line_len
        proj = np.dot(pos_pt - p1, line_unit)
        proj_clamped = max(0.0, min(line_len, proj))
        closest_pt = p1 + proj_clamped * line_unit
        dist = float(np.linalg.norm(pos_pt - closest_pt))

        if dist < best_dist and dist <= max_distance:
            best_dist = dist
            snapped_pos = [round(float(closest_pt[0]), 2), round(float(closest_pt[1]), 2)]
            was_snapped = True

    return snapped_pos, was_snapped


def postprocess_floorplan_json(
    raw_dict_or_str: Any,
    image_width: int,
    image_height: int,
    snap_walls: bool = True,
    filter_duplicates: bool = True,
    return_diagnostics: bool = False
) -> Any:
    """
    Main postprocessing pipeline:
    Parses JSON -> Sanitizes Fields -> Applies Geometric Refinement -> Validates Pydantic Schema.
    Preserves original predictions and tracks diagnostic refinement events.
    """
    diagnostics = {
        "walls_snapped_count": 0,
        "walls_dropped_zero_length": 0,
        "duplicates_removed_count": 0,
        "portals_snapped_to_wall": 0,
        "polygons_repaired_count": 0,
        "events": []
    }

    # 1. Parse JSON strictly: invalid model output is an error, not an empty floor plan
    if isinstance(raw_dict_or_str, str):
        extracted = extract_json_from_text(raw_dict_or_str)
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError as e:
            raise ModelOutputError(f"Model output is not valid JSON: {e}") from e
    elif isinstance(raw_dict_or_str, dict):
        data = raw_dict_or_str
    else:
        raise ModelOutputError(f"Unsupported model output type: {type(raw_dict_or_str).__name__}")
    if not isinstance(data, dict):
        raise ModelOutputError(f"Model output JSON must be an object, got {type(data).__name__}")

    # Ensure required top-level fields
    w = int(data.get("image_width", image_width) or image_width)
    h = int(data.get("image_height", image_height) or image_height)
    img_name = str(data.get("image", "floorplan.png"))

    raw_rooms = data.get("rooms", [])
    raw_walls = data.get("walls", [])
    raw_doors = data.get("doors", [])
    raw_windows = data.get("windows", [])

    # 2. Geometric wall snapping
    refined_walls = []
    for idx, w_elem in enumerate(raw_walls, 1):
        start = w_elem.get("start", [0, 0])
        end = w_elem.get("end", [0, 0])
        if snap_walls:
            s_snapped, e_snapped, snapped = snap_wall_orthogonality(start, end, angle_tolerance_deg=4.0)
            if snapped:
                diagnostics["walls_snapped_count"] += 1
                diagnostics["events"].append({
                    "type": "wall_snapped", "id": w_elem.get("id", f"wall_{idx}"),
                    "from": [start, end], "to": [s_snapped, e_snapped]
                })
        else:
            s_snapped, e_snapped = start, end

        # Check non-zero length
        dx = s_snapped[0] - e_snapped[0]
        dy = s_snapped[1] - e_snapped[1]
        if (dx * dx + dy * dy) > 1.0:
            refined_walls.append({
                "id": str(w_elem.get("id", f"wall_{idx}")),
                "start": [round(float(s_snapped[0]), 2), round(float(s_snapped[1]), 2)],
                "end": [round(float(e_snapped[0]), 2), round(float(e_snapped[1]), 2)],
                "thickness": _opt_float(w_elem.get("thickness")),
                "confidence": _opt_float(w_elem.get("confidence"))
            })
        else:
            diagnostics["walls_dropped_zero_length"] += 1

    # 3. Filter duplicate doors and windows
    if filter_duplicates:
        doors_pre = len(raw_doors)
        refined_doors = remove_duplicate_boxes(raw_doors, iou_thresh=0.75)
        diagnostics["duplicates_removed_count"] += (doors_pre - len(refined_doors))

        wins_pre = len(raw_windows)
        refined_windows = remove_duplicate_boxes(raw_windows, iou_thresh=0.75)
        diagnostics["duplicates_removed_count"] += (wins_pre - len(refined_windows))
    else:
        refined_doors = raw_doors
        refined_windows = raw_windows

    # Snap doors and windows to walls with orientation awareness and max 15px distance
    final_doors = []
    for idx, d_elem in enumerate(refined_doors, 1):
        bbox = d_elem.get("bbox", [0, 0, 10, 10])
        pos = d_elem.get("position", [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        orient = str(d_elem.get("orientation", "horizontal")).lower()
        snapped_p, was_snapped = snap_element_to_nearest_wall(pos, refined_walls, portal_orientation=orient, max_distance=15.0)
        if was_snapped:
            diagnostics["portals_snapped_to_wall"] += 1

        final_doors.append({
            "id": str(d_elem.get("id", f"door_{idx}")),
            "bbox": [round(float(v), 2) for v in bbox],
            "position": snapped_p,
            "orientation": orient if orient in ["horizontal", "vertical", "slanted"] else "horizontal",
            "confidence": _opt_float(d_elem.get("confidence"))
        })

    final_windows = []
    for idx, win_elem in enumerate(refined_windows, 1):
        bbox = win_elem.get("bbox", [0, 0, 10, 10])
        pos = win_elem.get("position", [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        orient = str(win_elem.get("orientation", "horizontal")).lower()
        snapped_p, was_snapped = snap_element_to_nearest_wall(pos, refined_walls, portal_orientation=orient, max_distance=15.0)
        if was_snapped:
            diagnostics["portals_snapped_to_wall"] += 1

        final_windows.append({
            "id": str(win_elem.get("id", f"window_{idx}")),
            "bbox": [round(float(v), 2) for v in bbox],
            "position": snapped_p,
            "orientation": orient if orient in ["horizontal", "vertical", "slanted"] else "horizontal",
            "confidence": _opt_float(win_elem.get("confidence"))
        })

    # 4. Refine room polygons
    refined_rooms = []
    for idx, r_elem in enumerate(raw_rooms, 1):
        poly = r_elem.get("polygon", [])
        valid_poly, was_repaired = validate_and_refine_polygon(poly)
        if was_repaired:
            diagnostics["polygons_repaired_count"] += 1
        if valid_poly and len(valid_poly) >= 3:
            refined_rooms.append({
                "id": str(r_elem.get("id", f"room_{idx}")),
                "type": str(r_elem.get("type", "undivided")),
                "polygon": valid_poly,
                "area_pixels": float(r_elem.get("area_pixels", 0.0)) or None,
                "confidence": _opt_float(r_elem.get("confidence"))
            })

    # 5. Build validated Pydantic FloorPlanSchema
    schema = FloorPlanSchema(
        image=img_name,
        image_width=w,
        image_height=h,
        units="pixels",
        rooms=refined_rooms,
        walls=refined_walls,
        doors=final_doors,
        windows=final_windows
    )

    if return_diagnostics:
        return schema, diagnostics
    return schema
