"""
End-to-end checks for evaluation/evaluate.py: failures are counted (never dropped or replaced by default
geometry), ground truth is only read from --gt-dir, and the aggregate numbers equal hand-computed values.

The room test needs shapely (polygon IoU) and is skipped, not passed, where shapely is not installed.
"""

import csv
import json

import pytest

from evaluation.evaluate import evaluate_system, write_outputs
from inference.records import make_record, record_path, write_record

W, H = 400, 300


def _gt(name, rooms=None):
    return {
        "image": f"{name}.png", "image_width": W, "image_height": H, "units": "pixels",
        "rooms": rooms or [],
        "walls": [{"id": "wall_1", "start": [10, 10], "end": [390, 10]},
                  {"id": "wall_2", "start": [10, 10], "end": [10, 290]}],
        "doors": [{"id": "door_1", "bbox": [100, 5, 140, 15]}],
        "windows": [{"id": "window_1", "bbox": [200, 5, 260, 15]}],
    }


def _write_gt(gt_dir, name, rooms=None):
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{name}.json").write_text(json.dumps(_gt(name, rooms)), encoding="utf-8")


def _ok_record(pred_dir, name, prediction):
    write_record(record_path(pred_dir, name), make_record(
        image=f"{name}.png", system="t", status="ok", raw_output=json.dumps(prediction), prediction=prediction))


def test_failures_are_counted_and_scores_match_hand_computation(tmp_path):
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    for n in "abcd":
        _write_gt(gt_dir, n)

    _ok_record(pred_dir, "a", _gt("a"))  # a perfect prediction for image a
    write_record(record_path(pred_dir, "b"), make_record(  # model produced non-JSON text
        image="b.png", system="t", status="failed", failure_stage="json_parse",
        error="no JSON", raw_output="I cannot analyse this image."))
    # image c: no record at all
    bad = _gt("d")
    bad["walls"][0]["end"] = bad["walls"][0]["start"]  # zero-length wall violates the schema
    _ok_record(pred_dir, "d", bad)

    res = evaluate_system(gt_dir, pred_dir, "t")

    assert res["n_test_images"] == 4 and res["n_predictions_ok"] == 1 and res["n_failed"] == 3
    assert res["failures_by_stage"] == {"json_parse": 1, "missing_record": 1, "schema_validation": 1}
    # raw text exists for a, b, d; only b failed to parse
    assert res["json_validity_rate"] == pytest.approx(2 / 3, abs=1e-4)

    walls = res["class_metrics"]["walls"]
    assert (walls["tp"], walls["fp"], walls["fn"], walls["n_gt"], walls["n_pred"]) == (2, 0, 6, 8, 2)
    assert walls["precision"] == 1.0 and walls["recall"] == 0.25 and walls["f1"] == 0.4
    for cls in ("doors", "windows"):
        m = res["class_metrics"][cls]
        assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 3)

    by_image = {p["image"]: p for p in res["per_image"]}
    assert by_image["a"]["failed"] is False and by_image["a"]["walls"]["fn"] == 0
    for n in "bcd":
        assert by_image[n]["failed"] is True
        assert by_image[n]["walls"]["tp"] == 0 and by_image[n]["walls"]["fn"] == 2  # all ground truth missed
        assert by_image[n]["walls"]["pred_count"] == 0  # no default geometry invented
    assert res["limitations"], "4 test images must carry the small-test-set warning"
    # rooms have no ground truth here, so they must not pull the macro average toward 0
    assert res["macro_f1_over_classes"] == pytest.approx((0.4 + 0.4 + 0.4) / 3, abs=1e-3)


def test_write_outputs_creates_report_files(tmp_path):
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    _write_gt(gt_dir, "a")
    _ok_record(pred_dir, "a", _gt("a"))
    res = evaluate_system(gt_dir, pred_dir, "t")
    write_outputs(res, tmp_path / "out")
    assert json.loads((tmp_path / "out/metrics.json").read_text())["system"] == "t"
    rows = list(csv.reader(open(tmp_path / "out/classification_report.csv")))
    assert [r[1] for r in rows[1:]] == ["walls", "doors", "windows", "rooms"]
    assert (tmp_path / "out/room_type_confusion.csv").is_file()


def test_empty_ground_truth_dir_raises(tmp_path):
    (tmp_path / "gt").mkdir()
    with pytest.raises(FileNotFoundError):
        evaluate_system(tmp_path / "gt", tmp_path / "pred", "t")


def test_room_type_confusion_and_spurious_room():
    pytest.importorskip("shapely", reason="polygon IoU needs shapely; run this on Colab / a full environment")
    import tempfile
    from pathlib import Path

    sq = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]  # noqa: E731
    gt_rooms = [{"id": "room_1", "type": "bedroom", "polygon": sq(10, 20, 110, 120)},
                {"id": "room_2", "type": "kitchen", "polygon": sq(110, 20, 210, 120)}]
    pred = _gt("a", gt_rooms)
    pred["rooms"] = [
        {"id": "room_1", "type": "bedroom", "polygon": sq(10, 20, 110, 120)},   # right place, right type
        {"id": "room_2", "type": "bathroom", "polygon": sq(110, 20, 210, 120)},  # right place, wrong type
        {"id": "room_3", "type": "office", "polygon": sq(300, 200, 380, 280)},   # spurious
    ]
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir, pred_dir = Path(tmp) / "gt", Path(tmp) / "pred"
        _write_gt(gt_dir, "a", gt_rooms)
        _ok_record(pred_dir, "a", pred)
        res = evaluate_system(gt_dir, pred_dir, "t")

    rooms = res["class_metrics"]["rooms"]
    assert (rooms["tp"], rooms["fp"], rooms["fn"]) == (2, 1, 0)
    assert rooms["type_accuracy_of_matched"] == 0.5
    assert rooms["type_correct_over_all_gt_rooms"] == 0.5
    assert res["room_type_confusion"] == {"bedroom": {"bedroom": 1}, "kitchen": {"bathroom": 1}}
