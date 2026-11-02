"""
Single-image inference entry point: Qwen2.5-VL-3B-Instruct + a trained LoRA adapter.

Pipeline: image -> Qwen processor -> base model -> LoRA adapter -> generated JSON ->
parser -> schema validation -> geometric postprocessing.

Status: the adapter integrity gate below is implemented and tested. Token generation
(inference/vlm_infer.py) is not implemented yet; until it is, a verified adapter still ends
in an explicit error. This module never falls back to OpenCV and never fabricates geometry
or confidence values. For the OpenCV comparison baseline see baselines/opencv_baseline.py.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.adapter_check import AdapterError, inspect_adapter  # noqa: E402,F401

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_ADAPTER_DIR = "artifacts/adapters/final_lora_adapter"


def predict_floorplan(
    image_path: str,
    adapter_path: str = DEFAULT_ADAPTER_DIR,
    base_model_name: str = DEFAULT_BASE_MODEL,
    output_json_path: Optional[str] = None,
    save_vis: bool = True,
):
    """
    Predict walls/doors/windows/rooms for one image with the fine-tuned VLM.

    Raises AdapterError if the adapter is missing, mismatched, malformed or untrained, so
    callers can never mistake an unloaded model for a working one.
    """
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    info = inspect_adapter(adapter_path, expected_base_model=base_model_name)
    logger.info("Adapter verified: r=%s, %s LoRA params, sha256=%s...",
                info["r"], info["n_lora_params"], info["weights_sha256"][:12])

    raise NotImplementedError(
        "The adapter passed verification, but VLM generation (inference/vlm_infer.py) "
        "is not implemented yet."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Predict floor-plan elements with Qwen2.5-VL + LoRA.")
    p.add_argument("--image", required=True)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER_DIR)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output", default=None)
    p.add_argument("--no-vis", action="store_true")
    a = p.parse_args()
    predict_floorplan(a.image, a.adapter, a.base_model, a.output, save_vis=not a.no_vis)


if __name__ == "__main__":
    main()
