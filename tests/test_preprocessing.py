"""
Unit tests for Dataset Preprocessing and SVG conversion.
"""

from pathlib import Path
import numpy as np
from preprocessing.convert_annotations import (
    parse_svg_points, normalize_room_type, preprocess_image
)


def test_parse_svg_points():
    """Tests SVG point string parsing with commas and spaces."""
    pts_str = "100,200 300,200 300,400 100,400"
    parsed = parse_svg_points(pts_str)
    assert len(parsed) == 4
    assert parsed[0] == [100.0, 200.0]
    assert parsed[2] == [300.0, 400.0]


def test_normalize_room_type():
    """Tests architectural room classification mapping."""
    assert normalize_room_type("MasterBedroom") == "bedroom"
    assert normalize_room_type("Kitchen_Open") == "kitchen"
    assert normalize_room_type("WC_Guest") == "bathroom"
    assert normalize_room_type("LivingRoom_Main") == "living_room"
    assert normalize_room_type("UnknownZone") == "undivided"


def test_preprocess_image_scaling(tmp_path):
    """Tests image preprocessing with resolution normalization and scale factor tracking."""
    import cv2
    dummy_img = np.ones((500, 1000, 3), dtype=np.uint8) * 200
    test_img_path = tmp_path / "test_sample.png"
    cv2.imwrite(str(test_img_path), dummy_img)

    processed, sx, sy = preprocess_image(test_img_path, target_size=(1024, 1024))
    assert processed.shape == (1024, 1024, 3)
    assert abs(sx - 1.024) < 1e-4
    assert abs(sy - 2.048) < 1e-4
