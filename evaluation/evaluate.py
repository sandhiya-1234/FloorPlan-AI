"""
Evaluation runner for FloorPlan-AI. Scores ONE system from its saved prediction records.

    python evaluation/evaluate.py --gt-dir dataset/test \
        --pred-dir outputs/predictions/<system> --system-name <system> --out-dir outputs/evaluation/<system>

The evaluator never calls a model. It reads ground truth from --gt-dir (<stem>.json) and one
prediction record per image from --pred-dir (<stem>.pred.json, see inference/records.py).

Integrity rules enforced here:
  * An image with a missing or failed record is COUNTED as a failure: all its ground-truth
    elements become false negatives. Nothing is dropped from the averages.
  * JSON validity is computed from the records (null for systems that emit no text).
  * Scores are micro-averaged over all elements in the test set (pooled TP/FP/FN).
  * Matching is Hungarian assignment (see evaluation/metrics.py):
      walls: endpoint error <= 25 px | doors/windows: bbox IoU >= 0.5 | rooms: polygon IoU >= 0.5
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import evaluate_single_prediction  # noqa: E402
from inference.records import load_record, record_path  # noqa: E402
from schemas.floorplan_schema import FloorPlanSchema  # noqa: E402

logger = logging.getLogger(__name__)

CLASSES = ["walls", "doors", "windows", "rooms"]
SMALL_TEST_SET = 50  # below this, results are reported with an explicit limitation warning


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def evaluate_system(gt_dir: Path, pred_dir: Path, system_name: str) -> Dict[str, Any]:
    gt_files = sorted(Path(gt_dir).glob("*.json"))
    if not gt_files:
        raise FileNotFoundError(f"No ground-truth JSON files in {gt_dir}")

    totals = {c: Counter() for c in CLASSES}
    scores: Dict[str, List[float]] = {c: [] for c in CLASSES}
    confusion: Dict[str, Counter] = defaultdict(Counter)  # gt_type -> pred_type -> count
    per_image: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    stage_counts: Counter = Counter()
    n_text = n_json_ok = n_ok = 0

    for gt_path in gt_files:
        stem = gt_path.stem
        gt = FloorPlanSchema(**json.loads(gt_path.read_text(encoding="utf-8")))
        rec_path = record_path(Path(pred_dir), stem)

        failure = None
        pred = None
        rec: Dict[str, Any] = {}
        if not rec_path.is_file():
            failure = ("missing_record", f"no prediction record at {rec_path}")
        else:
            rec = load_record(rec_path)
            if rec.get("raw_output") is not None:
                n_text += 1
                n_json_ok += rec.get("failure_stage") != "json_parse"
            if rec["status"] == "failed":
                failure = (rec["failure_stage"], rec["error"])
            else:
                try:
                    pred = FloorPlanSchema(**rec["prediction"])
                except Exception as e:  # noqa: BLE001 - counted as a failure, not dropped
                    failure = ("schema_validation", f"{type(e).__name__}: {e}")

        if failure is not None:
            stage_counts[failure[0]] += 1
            failures.append({"image": stem, "stage": failure[0], "error": failure[1]})
            pred = FloorPlanSchema(image=gt.image, image_width=gt.image_width, image_height=gt.image_height)
        else:
            n_ok += 1

        m = evaluate_single_prediction(pred, gt)
        per_image.append({"image": stem, "failed": failure is not None,
                          **{c: {k: m[c][k] for k in ("tp", "fp", "fn", "pred_count", "gt_count")} for c in CLASSES}})
        for c in CLASSES:
            for k in ("tp", "fp", "fn"):
                totals[c][k] += m[c][k]
            totals[c]["pred_count"] += m[c]["pred_count"]
            totals[c]["gt_count"] += m[c]["gt_count"]
            scores[c].extend(m[c]["match_scores"])
        for pred_type, gt_type in m["rooms"]["room_type_pairs"]:
            confusion[gt_type][pred_type] += 1

    n_images = len(gt_files)
    class_metrics: Dict[str, Any] = {}
    for c in CLASSES:
        t = totals[c]
        entry = {**_prf(t["tp"], t["fp"], t["fn"]), "tp": t["tp"], "fp": t["fp"], "fn": t["fn"],
                 "n_gt": t["gt_count"], "n_pred": t["pred_count"]}
        if scores[c]:
            key = "mean_endpoint_error_px" if c == "walls" else "mean_iou_of_matches"
            entry[key] = round(float(np.mean(scores[c])), 4)
        class_metrics[c] = entry

    matched_rooms = sum(sum(row.values()) for row in confusion.values())
    correct_types = sum(row[t] for t, row in confusion.items())
    class_metrics["rooms"]["type_accuracy_of_matched"] = (
        round(correct_types / matched_rooms, 4) if matched_rooms else None)
    class_metrics["rooms"]["type_correct_over_all_gt_rooms"] = (
        round(correct_types / totals["rooms"]["gt_count"], 4) if totals["rooms"]["gt_count"] else None)

    return {
        "system": system_name,
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "n_test_images": n_images,
        "n_predictions_ok": n_ok,
        "n_failed": n_images - n_ok,
        "failures_by_stage": dict(stage_counts),
        "json_validity_rate": round(n_json_ok / n_text, 4) if n_text else None,
        "json_validity_note": None if n_text else "not applicable: system emits no text output",
        "macro_f1_over_classes": (
            round(float(np.mean([class_metrics[c]["f1"] for c in CLASSES if class_metrics[c]["n_gt"] > 0])), 4)
            if any(class_metrics[c]["n_gt"] > 0 for c in CLASSES) else None),
        "macro_f1_note": "mean over classes that have at least one ground-truth element",
        "class_metrics": class_metrics,
        "room_type_confusion": {g: dict(row) for g, row in confusion.items()},
        "matching": "Hungarian; walls endpoint error <= 25 px; doors/windows bbox IoU >= 0.5; rooms polygon IoU >= 0.5",
        "limitations": (
            [f"Only {n_images} test images (< {SMALL_TEST_SET}): metrics are indicative, not statistically reliable."]
            if n_images < SMALL_TEST_SET else []),
        "per_image": per_image,
        "failures": failures,
    }


def write_outputs(results: Dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    with open(out_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "class", "precision", "recall", "f1", "tp", "fp", "fn", "n_gt", "n_pred"])
        for c in CLASSES:
            m = results["class_metrics"][c]
            w.writerow([results["system"], c, m["precision"], m["recall"], m["f1"],
                        m["tp"], m["fp"], m["fn"], m["n_gt"], m["n_pred"]])

    types = sorted({t for row in results["room_type_confusion"].values() for t in row}
                   | set(results["room_type_confusion"]))
    with open(out_dir / "room_type_confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_type\\pred_type"] + types)
        for g in types:
            row = results["room_type_confusion"].get(g, {})
            w.writerow([g] + [row.get(t, 0) for t in types])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Score one system from saved prediction records.")
    p.add_argument("--gt-dir", required=True, help="folder with ground-truth <stem>.json files (e.g. dataset/test)")
    p.add_argument("--pred-dir", required=True, help="folder with <stem>.pred.json records")
    p.add_argument("--system-name", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    results = evaluate_system(Path(a.gt_dir), Path(a.pred_dir), a.system_name)
    write_outputs(results, Path(a.out_dir))
    logger.info("%s: %d images, %d failed, macro-F1 %.4f -> %s",
                a.system_name, results["n_test_images"], results["n_failed"],
                results["macro_f1_over_classes"], a.out_dir)


if __name__ == "__main__":
    main()
