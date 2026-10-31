"""
Model Loader for FloorPlan-AI: Qwen2.5-VL Pretrained Model & LoRA/QLoRA Integration.

Handles device discovery, 4-bit BitsAndBytes quantization, LoRA configuration,
processor setup, and adapter merging for training and inference.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Any, Optional, Dict
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import ModelConfig, LoRAConfig

logger = logging.getLogger(__name__)


def check_cuda_environment() -> Dict[str, Any]:
    """Audits local compute hardware for VLM training and inference."""
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if has_cuda
        else 0.0
    )
    logger.info(f"Hardware Check: Device={device_name}, CUDA Available={has_cuda}, VRAM={vram_gb:.2f} GB")
    return {
        "has_cuda": has_cuda,
        "device_name": device_name,
        "vram_gb": vram_gb
    }


def get_quantization_config(model_cfg: ModelConfig) -> Optional[Any]:
    """Builds BitsAndBytes 4-bit NF4 quantization config when CUDA is present."""
    if not torch.cuda.is_available() or not model_cfg.load_in_4bit:
        return None

    try:
        from transformers import BitsAndBytesConfig
        compute_dtype = (
            torch.bfloat16 if model_cfg.bnb_4bit_compute_dtype == "bfloat16" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_cfg.use_nested_quant,
        )
    except ImportError:
        logger.warning("bitsandbytes not installed or not supported on this platform. Running without 4-bit quantization.")
        return None


def load_model_and_processor(
    model_cfg: ModelConfig,
    lora_cfg: Optional[LoRAConfig] = None,
    is_training: bool = True
) -> Tuple[Any, Any]:
    """
    Loads base vision-language model, sets up processor, and applies LoRA/QLoRA if training.
    """
    from transformers import AutoProcessor
    model_name = model_cfg.base_model_name

    logger.info(f"Loading processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=model_cfg.min_image_pixels,
        max_pixels=model_cfg.max_image_pixels,
        trust_remote_code=True
    )

    bnb_config = get_quantization_config(model_cfg)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    logger.info(f"Loading base model {model_name} (dtype={torch_dtype}, device_map={device_map})...")

    # Try loading Qwen2_5_VLForConditionalGeneration
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"Could not load with Qwen2_5_VLForConditionalGeneration: {e}. Falling back to AutoModelForVision2Seq...")
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

    if is_training and lora_cfg is not None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if bnb_config is not None:
            logger.info("Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type="CAUSAL_LM",
            target_modules=lora_cfg.target_modules,
        )

        logger.info(f"Attaching LoRA adapter (r={lora_cfg.r}, alpha={lora_cfg.lora_alpha})...")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, processor


def load_adapter_for_inference(
    base_model_name: str,
    adapter_path: str,
    device: str = "auto"
) -> Tuple[Any, Any]:
    """
    Loads pretrained base model and attaches fine-tuned LoRA adapter weights.
    """
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel

    logger.info(f"Loading base model: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = device if device != "auto" else ("auto" if torch.cuda.is_available() else "cpu")

    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    adapter_p = Path(adapter_path)
    if adapter_p.exists():
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_p))
        model.eval()
        return model, processor
    else:
        logger.warning(f"Adapter path {adapter_path} not found. Returning base model.")
        base_model.eval()
        return base_model, processor
