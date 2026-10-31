"""
Unit tests for Geometric Postprocessing and algorithms.
"""

from inference.postprocess import (
    calculate_bbox_iou,
    remove_duplicate_boxes,
    snap_wall_orthogonality,
    validate_and_refine_polygon,
    snap_element_to_nearest_wall
)


def test_bbox_iou_calculation():
    """Tests exact bounding box IoU calculation."""
    # Identical boxes
    box1 = [100.0, 100.0, 200.0, 200.0]
    box2 = [100.0, 100.0, 200.0, 200.0]
    assert calculate_bbox_iou(box1, box2) == 1.0

    # Non-overlapping
    box3 = [300.0, 300.0, 400.0, 400.0]
    assert calculate_bbox_iou(box1, box3) == 0.0

    # Partial overlap (50% horizontal overlap, identical vertical)
    box4 = [150.0, 100.0, 250.0, 200.0]
    # Inter: 50 * 100 = 5000; Union: 10000 + 10000 - 5000 = 15000 -> 1/3
    assert abs(calculate_bbox_iou(box1, box4) - (1.0 / 3.0)) < 1e-4


def test_duplicate_filtering():
    """Tests that duplicate detections above IoU threshold are removed."""
    items = [
        {"id": "d1", "bbox": [100, 100, 200, 200]},
        {"id": "d2", "bbox": [102, 101, 198, 199]}, # highly overlapping duplicate
        {"id": "d3", "bbox": [500, 500, 600, 600]}, # separate item
    ]
    filtered = remove_duplicate_boxes(items, iou_thresh=0.75)
    assert len(filtered) == 2
    assert filtered[0]["id"] == "d1"
    assert filtered[1]["id"] == "d3"


def test_snap_wall_orthogonality():
    """Tests snapping near-horizontal and near-vertical wall angles to exact axes."""
    # Near horizontal (dx=400, dy=2 -> angle ~ 0.28 deg)
    s1, e1, snapped1 = snap_wall_orthogonality([100.0, 200.0], [500.0, 202.0], angle_tolerance_deg=5.0)
    assert snapped1 is True
    assert s1[1] == e1[1] == 201.0

    # Near vertical (dx=3, dy=400 -> angle ~ 89.5 deg)
    s2, e2, snapped2 = snap_wall_orthogonality([300.0, 100.0], [303.0, 500.0], angle_tolerance_deg=5.0)
    assert snapped2 is True
    assert s2[0] == e2[0] == 301.5


def test_polygon_validation_shapely():
    """Tests topological validation of valid and self-intersecting polygons."""
    # Valid square
    valid_sq = [[100, 100], [300, 100], [300, 300], [100, 300]]
    res, repaired = validate_and_refine_polygon(valid_sq)
    assert res is not None
    assert len(res) == 4
    assert repaired is False

    # Degenerate collinear points
    degenerate = [[100, 100], [200, 100], [300, 100]]
    res_deg, _ = validate_and_refine_polygon(degenerate)
    assert res_deg is None


def test_snap_element_to_wall():
    """Tests door/window midpoint snapping to closest wall centerline."""
    walls = [
        {"id": "w1", "start": [100.0, 200.0], "end": [600.0, 200.0]}
    ]
    # Position slightly off the wall line (y=204 instead of 200)
    pos = [350.0, 204.0]
    snapped, was_snapped = snap_element_to_nearest_wall(pos, walls, portal_orientation="horizontal", max_distance=15.0)
    assert was_snapped is True
    assert snapped[0] == 350.0
    assert snapped[1] == 200.0


def test_diagonal_wall_preserved():
    """Tests that true diagonal walls (e.g. 45 degrees) are NOT snapped to flat axes."""
    s, e, was_snapped = snap_wall_orthogonality([100.0, 100.0], [200.0, 200.0], angle_tolerance_deg=4.0)
    assert was_snapped is False
    assert s == [100.0, 100.0]
    assert e == [200.0, 200.0]


def test_portal_snap_distance_limit():
    """Tests that portals further than max_distance are not falsely snapped."""
    walls = [{"id": "w1", "start": [100.0, 200.0], "end": [600.0, 200.0]}]
    pos = [350.0, 230.0] # 30px away, max_distance=15px
    snapped, was_snapped = snap_element_to_nearest_wall(pos, walls, portal_orientation="horizontal", max_distance=15.0)
    assert was_snapped is False
    assert snapped == [350.0, 230.0]


def test_portal_snap_orientation_mismatch():
    """Tests that a vertical portal is not snapped to a horizontal wall."""
    walls = [{"id": "w1", "start": [100.0, 200.0], "end": [600.0, 200.0]}] # horizontal wall
    pos = [350.0, 202.0] # 2px away, but vertical orientation
    snapped, was_snapped = snap_element_to_nearest_wall(pos, walls, portal_orientation="vertical", max_distance=15.0)
    assert was_snapped is False
