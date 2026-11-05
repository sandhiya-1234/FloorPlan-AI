"""
Numpy-side tests for polygon extraction, JSON validity and the evaluation report (no torch needed).
The model itself is exercised by tests/test_seg_torch.py, which needs torch and therefore runs on Colab.
"""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from evaluation.evaluate_segmentation import evaluate_split
from inference.predict_segmentation import SegmentationPrediction, mask_to_entities, mask_to_prediction
from training.seg_common import IGNORE_INDEX, load_config

CFG = load_config()
BG, ROOM, WALL, DOOR, WIN = range(5)


def make_mask():
    m = np.zeros((200, 300), np.uint8)
    m[20:180, 20:280] = ROOM
    m[20:180, 145:155] = WALL           # interior wall splits the room into two regions
    m[20:25, 20:280] = WALL
    m[60:100, 143:157] = DOOR           # door cuts through the interior wall
    m[20:25, 40:70] = WIN
    return m


def test_regions_become_polygons_in_original_pixel_coordinates():
    pred = mask_to_prediction(make_mask(), "p.png", CFG)
    assert (pred.image_width, pred.image_height) == (300, 200)
    assert len(pred.rooms) == 2 and len(pred.doors) == 1 and len(pred.windows) == 1 and len(pred.walls) >= 1
    left = min(pred.rooms, key=lambda r: min(x for x, _ in r.polygon))
    xs, ys = zip(*left.polygon)
    assert 18 <= min(xs) <= 22 and 140 <= max(xs) <= 146 and 24 <= min(ys) <= 26 and 176 <= max(ys) <= 180
    d = pred.doors[0]
    assert d.id == "door_1" and 140 <= min(x for x, _ in d.polygon) and max(x for x, _ in d.polygon) <= 158
    assert d.area_px == pytest.approx(40 * 14, rel=0.15)           # area is measured from the polygon, not invented


def test_no_confidence_fields_and_ids_unique():
    pred = mask_to_prediction(make_mask(), "p.png", CFG)
    dumped = json.loads(pred.model_dump_json())
    ids = [e["id"] for k in ("rooms", "walls", "doors", "windows") for e in dumped[k]]
    assert len(ids) == len(set(ids))
    assert "confidence" not in json.dumps(dumped) and "score" not in json.dumps(dumped)


def test_empty_and_tiny_masks_produce_no_entities_not_defaults():
    pred = mask_to_prediction(np.zeros((100, 100), np.uint8), "e.png", CFG)
    assert pred.rooms == pred.walls == pred.doors == pred.windows == []
    speck = np.zeros((1000, 1000), np.uint8)
    speck[5:8, 5:8] = DOOR                                          # 9 px << min_area_frac * image area
    assert mask_to_prediction(speck, "s.png", CFG).doors == []
    assert mask_to_entities(np.zeros((10, 10), np.uint8), "x", 1.0, 0.01) == []


def test_holes_are_kept_and_subtracted_from_area():
    m = np.zeros((100, 100), np.uint8)
    m[10:90, 10:90] = WALL
    m[40:60, 40:60] = BG
    (e,) = mask_to_entities(m == WALL, "wall", 10.0, 0.01)
    assert len(e["holes"]) == 1 and e["area_px"] == pytest.approx(79 * 79 - 19 * 19, rel=0.02)


def test_schema_rejects_points_outside_the_image_and_degenerate_rings():
    ok = {"id": "room_1", "polygon": [[0, 0], [10, 0], [10, 10]], "area_px": 50.0}
    SegmentationPrediction(image="a", image_width=20, image_height=20, rooms=[ok])
    with pytest.raises(ValidationError):
        SegmentationPrediction(image="a", image_width=5, image_height=5, rooms=[ok])
    with pytest.raises(ValidationError):
        SegmentationPrediction(image="a", image_width=20, image_height=20, rooms=[{**ok, "polygon": [[0, 0], [1, 1]]}])


def _samples(n):
    out = []
    for i in range(n):
        gt = make_mask()
        gt[:, :3 + i] = IGNORE_INDEX                                 # some ignored pixels
        out.append((f"s{i}", np.full(gt.shape + (3,), 200, np.uint8), gt))
    return out


def test_evaluate_split_perfect_prediction_gives_perfect_scores_and_valid_json(tmp_path):
    samples = _samples(3)
    lookup = {id(s[1]): s[2] for s in samples}
    summary = evaluate_split(samples, lambda rgb: (np.where(lookup[id(rgb)] == IGNORE_INDEX, BG, lookup[id(rgb)]).astype(np.uint8), 0.01),
                             CFG, tmp_path, {"architecture": "test"}, n_visuals=2)
    for c in CFG["class_names"]:
        assert summary["per_class"][c]["iou"] == pytest.approx(1.0)
    assert summary["pixel_accuracy"] == pytest.approx(1.0) and summary["json_validity"]["rate"] == 1.0
    assert (tmp_path / "metrics.json").is_file() and (tmp_path / "confusion_matrix.csv").is_file()
    assert len(list((tmp_path / "predictions").glob("*.json"))) == 3 and len(list((tmp_path / "visuals").glob("*.png"))) >= 1
    assert summary["limitations"]                                    # 3 images: small-test-set warning
    assert len(list((tmp_path / "pred_masks").glob("*.png"))) == 3    # predicted class masks are saved


def test_evaluate_split_counts_errors_and_matches_hand_computation(tmp_path):
    gt = np.zeros((10, 10), np.uint8)
    gt[:, :5] = ROOM
    gt[0, :] = IGNORE_INDEX
    pred = np.zeros((10, 10), np.uint8)
    pred[:, :4] = ROOM                                               # misses one room column; no false room pixels
    summary = evaluate_split([("a", np.zeros((10, 10, 3), np.uint8), gt)], lambda rgb: (pred, 0.0), CFG, tmp_path, n_visuals=1)
    r, b = summary["per_class"]["Room"], summary["per_class"]["Background"]
    assert (r["gt_pixels"], r["pred_pixels"]) == (45, 36)            # first row ignored: 9 rows * 5 / 9 * 4
    assert r["iou"] == pytest.approx(36 / 45) and r["precision"] == pytest.approx(1.0) and r["recall"] == pytest.approx(36 / 45)
    assert b["gt_pixels"] == 45 and b["recall"] == pytest.approx(1.0) and b["precision"] == pytest.approx(45 / 54)
    assert summary["per_class"]["Door"]["iou"] is None               # absent from both: undefined, not 0
    assert summary["evaluated_pixels"] == 90


def test_shape_mismatch_between_prediction_and_ground_truth_is_an_error(tmp_path):
    gt = np.zeros((10, 10), np.uint8)
    with pytest.raises(RuntimeError, match="prediction"):
        evaluate_split([("a", np.zeros((10, 10, 3), np.uint8), gt)], lambda rgb: (np.zeros((9, 10), np.uint8), 0.0), CFG, tmp_path)
