"""inspect_adapter must accept only a real, trained, matching LoRA adapter."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from inference.adapter_check import AdapterError, inspect_adapter

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
REPO = Path(__file__).resolve().parent.parent


def write_safetensors(path: Path, tensors: dict) -> None:
    """Minimal safetensors writer (header length + JSON header + raw data) for test fixtures."""
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        blob = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def make_adapter(tmp_path: Path, *, b_value=0.01, base=BASE, r=4, with_weights=True, peft_type="LORA") -> Path:
    d = tmp_path / "adapter"
    d.mkdir()
    cfg = {"peft_type": peft_type, "base_model_name_or_path": base, "r": r, "lora_alpha": 8,
           "target_modules": ["q_proj", "v_proj"]}
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    if with_weights:
        pre = "base_model.model.model.layers.0.self_attn.q_proj"
        write_safetensors(d / "adapter_model.safetensors", {
            f"{pre}.lora_A.weight": np.random.default_rng(0).normal(size=(r, 16)),
            f"{pre}.lora_B.weight": np.full((16, r), b_value),
        })
    return d


def test_valid_adapter_is_accepted(tmp_path):
    info = inspect_adapter(str(make_adapter(tmp_path)), expected_base_model=BASE)
    assert info["r"] == 4 and info["n_lora_pairs"] == 1
    assert info["n_lora_params"] == 4 * 16 + 16 * 4
    assert info["lora_b_nonzero_tensor_fraction"] == 1.0
    assert len(info["weights_sha256"]) == 64


def test_untrained_adapter_with_zero_B_is_rejected(tmp_path):
    with pytest.raises(AdapterError, match="never been trained"):
        inspect_adapter(str(make_adapter(tmp_path, b_value=0.0)))


def test_wrong_base_model_is_rejected(tmp_path):
    with pytest.raises(AdapterError, match="trained for"):
        inspect_adapter(str(make_adapter(tmp_path, base="Qwen/Qwen2-VL-2B-Instruct")), expected_base_model=BASE)


def test_missing_weights_and_missing_dir_are_rejected(tmp_path):
    with pytest.raises(AdapterError, match="adapter_model.safetensors"):
        inspect_adapter(str(make_adapter(tmp_path, with_weights=False)))
    with pytest.raises(AdapterError, match="not found"):
        inspect_adapter(str(tmp_path / "nope"))


def test_non_lora_config_is_rejected(tmp_path):
    with pytest.raises(AdapterError, match="peft_type"):
        inspect_adapter(str(make_adapter(tmp_path, peft_type="PREFIX_TUNING")))


def test_nan_weights_are_rejected(tmp_path):
    with pytest.raises(AdapterError, match="NaN"):
        inspect_adapter(str(make_adapter(tmp_path, b_value=float("nan"))))


def test_previous_toy_adapter_config_is_rejected(tmp_path):
    """The mislabelled config that used to sit in artifacts/adapters/ must never verify."""
    toy = REPO / "docs/audit/invalid_artifacts/toy_surrogate_training/adapter_config.json"
    d = tmp_path / "toy"
    d.mkdir()
    (d / "adapter_config.json").write_text(toy.read_text())
    with pytest.raises(AdapterError, match="base_model_name_or_path"):
        inspect_adapter(str(d), expected_base_model=BASE)
