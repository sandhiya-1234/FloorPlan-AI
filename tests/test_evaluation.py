"""
Unit tests for Evaluation Metrics and Bipartite Matching.
"""

import pytest
from schemas.floorplan_schema import FloorPlanSchema, RoomElement, WallElement, DoorElement, WindowElement
from evaluation.metrics import (
    compute_bbox_iou, compute_polygon_iou, compute_wall_matching_error,
    match_elements_hungarian, evaluate_single_prediction
)


def test_polygon_iou():
    """Tests exact polygon IoU calculation."""
    pytest.importorskip("shapely", reason="needs shapely (requirements.txt); run on Colab / a full environment")
    poly1 = [[0, 0], [10, 0], [10, 10], [0, 10]]
    poly2 = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert compute_polygon_iou(poly1, poly2) == 1.0

    poly3 = [[20, 20], [30, 20], [30, 30], [20, 30]]
    assert compute_polygon_iou(poly1, poly3) == 0.0


def test_wall_matching_error():
    """Tests symmetric wall endpoint distance."""
    # Exact match in forward direction
    err1 = compute_wall_matching_error([0, 0], [100, 0], [0, 0], [100, 0])
    assert err1 == 0.0

    # Exact match in reversed endpoint order
    err2 = compute_wall_matching_error([0, 0], [100, 0], [100, 0], [0, 0])
    assert err2 == 0.0

    # 10px parallel offset
    err3 = compute_wall_matching_error([0, 0], [100, 0], [0, 10], [100, 10])
    assert err3 == 10.0


def test_bipartite_matching():
    """Tests bipartite matching accuracy on doors."""
    preds = [
        {"id": "p1", "bbox": [100, 100, 200, 150]},
        {"id": "p2", "bbox": [500, 500, 600, 550]},
    ]
    gts = [
        {"id": "g1", "bbox": [102, 101, 198, 149]}, # matches p1
        {"id": "g2", "bbox": [800, 800, 900, 850]}, # unmatched
    ]
    tp, fp, fn, scores, extra = match_elements_hungarian(preds, gts, match_type="bbox", threshold=0.5)
    assert tp == 1
    assert fp == 1
    assert fn == 1
    assert len(scores) == 1


def test_empty_predictions_handling():
    """Tests that empty predictions or empty ground truths do not cause division by zero."""
    preds = []
    gts = [{"id": "g1", "bbox": [10, 10, 50, 50]}]
    tp, fp, fn, scores, extra = match_elements_hungarian(preds, gts, match_type="bbox", threshold=0.5)
    assert tp == 0
    assert fp == 0
    assert fn == 1

    # Empty both
    tp, fp, fn, scores, extra = match_elements_hungarian([], [], match_type="bbox", threshold=0.5)
    assert tp == 0 and fp == 0 and fn == 0
