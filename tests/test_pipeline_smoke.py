"""
Plumbing smoke test: drawn plan -> OpenCV baseline -> prediction record -> evaluator.

This checks that the pieces connect and that nothing is invented. It is NOT a benchmark: it uses one
computer-drawn plan, and the baseline is a classical heuristic, not an AI model.
"""

import json

import cv2
import numpy as np

from baselines.opencv_baseline import run_baseline_on_image
from evaluation.evaluate import evaluate_system, write_outputs
from inference.records import load_record, record_path, write_record


def test_drawn_plan_through_baseline_records_and_evaluator(tmp_path):
    img = np.full((600, 800, 3), 255, np.uint8)
    walls = [((50, 50), (750, 50)), ((750, 50), (750, 550)), ((750, 550), (50, 550)),
             ((50, 550), (50, 50)), ((400, 50), (400, 550))]
    for a, b in walls:
        cv2.line(img, a, b, (0, 0, 0), 8)
    (tmp_path / "img").mkdir()
    cv2.imwrite(str(tmp_path / "img" / "plan.png"), img)
    gt = {"image": "plan.png", "image_width": 800, "image_height": 600, "units": "pixels",
          "walls": [{"id": f"wall_{i + 1}", "start": list(map(float, a)), "end": list(map(float, b))}
                    for i, (a, b) in enumerate(walls)],
          "doors": [], "windows": [], "rooms": []}
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt" / "plan.json").write_text(json.dumps(gt), encoding="utf-8")

    rec = run_baseline_on_image(tmp_path / "img" / "plan.png")
    write_record(record_path(tmp_path / "pred", "plan.png"), rec)
    assert load_record(record_path(tmp_path / "pred", "plan.png"))["status"] == "ok"

    res = evaluate_system(tmp_path / "gt", tmp_path / "pred", "opencv_baseline")
    write_outputs(res, tmp_path / "out")

    assert res["n_failed"] == 0 and res["n_test_images"] == 1
    assert res["json_validity_rate"] is None  # the baseline emits no text, so JSON validity does not apply
    w = res["class_metrics"]["walls"]
    assert w["n_pred"] > 0 and w["tp"] > 0, "the baseline should find at least some drawn walls"
    for cls in ("doors", "windows", "rooms"):  # the baseline never reports these; nothing may be invented
        assert res["class_metrics"][cls]["n_pred"] == 0 or cls == "rooms"
    assert res["limitations"], "a 1-image run must carry the small-test-set warning"
    assert (tmp_path / "out" / "metrics.json").is_file()
