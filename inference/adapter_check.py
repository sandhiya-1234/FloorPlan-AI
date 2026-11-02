"""
Static verification of a PEFT LoRA adapter directory. Needs only numpy (no torch).

Answers, from the files on disk: is this a real LoRA adapter for the expected base model,
does it contain trained (non-zero) LoRA weights, and which parts of the network does it touch?
Every inference entry point calls ``inspect_adapter`` first and fails loudly on any problem.

Safetensors layout parsed here: 8-byte little-endian header length N, N bytes of JSON header
({tensor: {dtype, shape, data_offsets}}), then the raw data buffer.
"""

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


class AdapterError(RuntimeError):
    """The adapter directory is missing, malformed, mismatched, or untrained."""


_NP_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64}


def _read_header(path: Path) -> Tuple[Dict[str, Any], int]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise AdapterError(f"{path.name}: file too short to be safetensors")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 200_000_000:
            raise AdapterError(f"{path.name}: implausible safetensors header length {n}")
        try:
            header = json.loads(f.read(n))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise AdapterError(f"{path.name}: unreadable safetensors header ({e})") from e
    return header, 8 + n


def _load_tensor(path: Path, data_start: int, info: Dict[str, Any]) -> np.ndarray:
    begin, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        buf = f.read(end - begin)
    dtype = info["dtype"]
    if dtype == "BF16":
        as_u32 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return as_u32.view(np.float32)
    if dtype in _NP_DTYPES:
        return np.frombuffer(buf, dtype=_NP_DTYPES[dtype])
    raise AdapterError(f"unsupported tensor dtype {dtype!r}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_adapter(adapter_dir: str, expected_base_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Verify an adapter directory and return a summary. Raises AdapterError on any problem.

    Checks: config is a LoRA config naming a base model (and it is the expected one);
    weights are safetensors containing paired lora_A/lora_B tensors with consistent rank;
    all values are finite; at least one lora_B tensor is non-zero (LoRA initialises B to
    exactly zero, so an all-zero B means the adapter was never trained).
    """
    d = Path(adapter_dir)
    if not d.is_dir():
        raise AdapterError(f"Adapter directory not found: {d}")

    cfg_path = d / "adapter_config.json"
    if not cfg_path.is_file():
        raise AdapterError(f"{cfg_path} not found")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AdapterError(f"{cfg_path} is not valid JSON: {e}") from e

    if str(cfg.get("peft_type", "")).upper() != "LORA":
        raise AdapterError(f"{cfg_path.name}: peft_type is {cfg.get('peft_type')!r}, expected 'LORA'")
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise AdapterError(f"{cfg_path.name}: no base_model_name_or_path (not a PEFT-saved adapter)")
    if expected_base_model and base != expected_base_model:
        raise AdapterError(
            f"Adapter was trained for {base!r} but inference requested {expected_base_model!r}"
        )
    r = cfg.get("r")
    if not isinstance(r, int) or r <= 0:
        raise AdapterError(f"{cfg_path.name}: invalid LoRA rank r={r!r}")
    if not cfg.get("target_modules"):
        raise AdapterError(f"{cfg_path.name}: target_modules is empty")

    weights = d / "adapter_model.safetensors"
    if not weights.is_file():
        raise AdapterError(
            f"{weights} not found. (adapter_model.bin is deliberately not accepted: it cannot be "
            "verified without torch and may be a non-PEFT state dict.)"
        )

    header, data_start = _read_header(weights)
    header.pop("__metadata__", None)
    a_keys = sorted(k for k in header if ".lora_A." in k)
    b_keys = sorted(k for k in header if ".lora_B." in k)
    if not a_keys or not b_keys:
        raise AdapterError(f"{weights.name}: contains no lora_A/lora_B tensors; not a LoRA adapter")
    if {k.replace(".lora_A.", ".lora_B.") for k in a_keys} != set(b_keys):
        raise AdapterError(f"{weights.name}: lora_A and lora_B tensors are not paired")

    uniform_rank = not cfg.get("rank_pattern")
    for k in a_keys:
        shape = header[k]["shape"]
        if len(shape) != 2 or (uniform_rank and shape[0] != r):
            raise AdapterError(f"{k}: shape {shape} inconsistent with LoRA rank r={r}")

    n_params = int(sum(int(np.prod(v["shape"])) for v in header.values()))

    b_nonzero = 0
    b_abs_sum = 0.0
    b_count = 0
    for k in b_keys:
        arr = _load_tensor(weights, data_start, header[k])
        if not np.isfinite(arr).all():
            raise AdapterError(f"{k}: contains NaN/Inf")
        if np.any(arr != 0):
            b_nonzero += 1
        b_abs_sum += float(np.abs(arr).sum())
        b_count += arr.size
    for k in a_keys:
        if not np.isfinite(_load_tensor(weights, data_start, header[k])).all():
            raise AdapterError(f"{k}: contains NaN/Inf")
    if b_nonzero == 0:
        raise AdapterError("All lora_B tensors are exactly zero: the adapter has never been trained")

    return {
        "adapter_dir": str(d),
        "base_model_name_or_path": base,
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": cfg.get("lora_alpha"),
        "target_modules": cfg.get("target_modules"),
        "n_lora_pairs": len(a_keys),
        "n_lora_params": n_params,
        "lora_b_nonzero_tensor_fraction": round(b_nonzero / len(b_keys), 4),
        "lora_b_mean_abs": b_abs_sum / max(1, b_count),
        "vision_tower_lora_pairs": sum(1 for k in a_keys if ".visual." in k),
        "weights_file": weights.name,
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": _sha256(weights),
    }
