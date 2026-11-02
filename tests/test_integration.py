"""
Integration guards: VLM inference must fail loudly instead of falling back to anything else,
and strict parsing must never turn invalid model output into an empty floor plan.
"""

import json
from pathlib import Path

import numpy as np
import cv2
import pytest

from inference.adapter_check import AdapterError
from inference.postprocess import ModelOutputError, postprocess_floorplan_json
from inference.predict import predict_floorplan

REPO = Path(__file__).resolve().parent.parent


def _image(tmp_path):
    img = np.full((200, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (280, 180), (0, 0, 0), 4)
    p = tmp_path / "plan.png"
    cv2.imwrite(str(p), img)
    return p


def test_predict_without_adapter_raises_and_writes_nothing(tmp_path):
    out = tmp_path / "pred.json"
    with pytest.raises(AdapterError):
        predict_floorplan(str(_image(tmp_path)), adapter_path=str(tmp_path / "missing"),
                          output_json_path=str(out), save_vis=False)
    assert not out.exists()


def test_repository_ships_no_fake_adapter():
    """No directory the dashboard or CLI points at may contain adapter-looking leftovers."""
    adapter_dir = REPO / "artifacts/adapters/final_lora_adapter"
    assert not (adapter_dir / "adapter_config.json").exists() or (adapter_dir / "adapter_model.safetensors").exists()


def test_invalid_model_text_raises_instead_of_returning_empty_plan():
    with pytest.raises(ModelOutputError):
        postprocess_floorplan_json("I cannot analyse this image.", 100, 100)
    with pytest.raises(ModelOutputError):
        postprocess_floorplan_json('{"walls": [', 100, 100)
    with pytest.raises(ModelOutputError):
        postprocess_floorplan_json("[1, 2, 3]", 100, 100)


def test_missing_confidence_and_thickness_stay_missing():
    raw = json.dumps({
        "walls": [{"id": "w1", "start": [0, 0], "end": [50, 0]}],
        "doors": [{"id": "d1", "bbox": [10, 0, 20, 5], "orientation": "horizontal"}],
        "windows": [],
        "rooms": [],
    })
    schema = postprocess_floorplan_json(raw, 100, 100)
    dumped = schema.to_compact_dict()
    assert "confidence" not in dumped["walls"][0] and "thickness" not in dumped["walls"][0]
    assert "confidence" not in dumped["doors"][0]
