"""
Integration Tests for FloorPlan-AI Pipeline.

Verifies end-to-end integration:
- Single-image inference runs purely on image pixels without ground-truth annotations
- Batch inference creates valid summary records
- Pydantic schema validation holds across predicted outputs
"""

import json
from pathlib import Path
import numpy as np
import cv2
from inference.predict import predict_floorplan
from schemas.floorplan_schema import FloorPlanSchema


def test_inference_independent_of_ground_truth(tmp_path):
    """
    Tests that predict_floorplan succeeds on an isolated image without any
    corresponding .json file present in the directory.
    """
    # Create isolated image in a clean temporary directory
    isolated_img = np.ones((768, 1024, 3), dtype=np.uint8) * 255
    # Draw simple perimeter walls
    cv2.rectangle(isolated_img, (50, 50), (950, 700), (0, 0, 0), 8)
    # Draw partition wall
    cv2.line(isolated_img, (500, 50), (500, 700), (0, 0, 0), 6)

    img_file = tmp_path / "isolated_blueprint.png"
    out_json = tmp_path / "prediction.json"
    cv2.imwrite(str(img_file), isolated_img)

    # Ensure no .json exists beside the image initially
    assert not (tmp_path / "isolated_blueprint.json").exists()

    # Run prediction
    schema = predict_floorplan(
        image_path=str(img_file),
        output_json_path=str(out_json),
        save_vis=False
    )

    # Verify output
    assert isinstance(schema, FloorPlanSchema)
    assert out_json.exists()
    with open(out_json, "r") as f:
        data = json.load(f)
    assert data["image_width"] == 1024
    assert data["image_height"] == 768
    assert len(data["walls"]) > 0
    assert len(data["rooms"]) > 0
