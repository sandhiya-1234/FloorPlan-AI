"""Prediction records cannot carry fabricated results; the OpenCV baseline never invents geometry."""

import cv2
import numpy as np
import pytest

from baselines.opencv_baseline import detect_opencv_baseline, run_baseline_on_image
from inference.records import make_record


def test_failed_record_cannot_carry_a_prediction():
    with pytest.raises(ValueError, match="no default geometry"):
        make_record(image="a.png", system="s", status="failed", failure_stage="json_parse",
                    error="bad json", prediction={"walls": []})


def test_failed_record_needs_stage_and_error():
    with pytest.raises(ValueError):
        make_record(image="a.png", system="s", status="failed", error="x")
    with pytest.raises(ValueError):
        make_record(image="a.png", system="s", status="failed", failure_stage="generate")


def test_ok_record_requires_prediction():
    with pytest.raises(ValueError):
        make_record(image="a.png", system="s", status="ok")


@pytest.mark.parametrize("value", [255, 0])
def test_baseline_returns_nothing_on_blank_image(value):
    """The old detector returned 4 walls, 4 rooms, doors and windows for a blank image."""
    blank = np.full((768, 1024, 3), value, np.uint8)
    out = detect_opencv_baseline(blank)
    assert out["walls"] == [] and out["rooms"] == [] and out["doors"] == [] and out["windows"] == []


def test_baseline_never_reports_confidence_doors_windows_or_room_types(tmp_path):
    img = np.full((768, 1024, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (950, 700), (0, 0, 0), 8)
    cv2.line(img, (500, 50), (500, 700), (0, 0, 0), 6)
    path = tmp_path / "plan.png"
    cv2.imwrite(str(path), img)

    rec = run_baseline_on_image(path)
    assert rec["status"] == "ok" and rec["raw_output"] is None and rec["meta"]["learned"] is False
    pred = rec["prediction"]
    assert pred["walls"], "straight dark lines should be found as walls"
    assert pred["doors"] == [] and pred["windows"] == []
    assert all("confidence" not in e for k in ("walls", "rooms") for e in pred[k])
    assert all(r["type"] == "unknown" for r in pred["rooms"])


def test_unreadable_image_is_a_failed_record_not_an_empty_plan(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    rec = run_baseline_on_image(bad)
    assert rec["status"] == "failed" and rec["prediction"] is None
