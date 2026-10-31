"""
Training Configuration Management for FloorPlan-AI.

Loads hyperparameter and architecture configurations from config.yaml into
strongly-typed dataclasses for reproducible fine-tuning.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class ModelConfig:
    base_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    fallback_model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    max_image_pixels: int = 1048576
    min_image_pixels: int = 65536
    prompt_template: str = (
        "Analyze this 2D architectural floor plan image and detect all architectural elements: "
        "walls, doors, windows, and rooms.\n"
        "Return ONLY a valid JSON object matching the required schema with keys: "
        "'image_width', 'image_height', 'units', 'rooms', 'walls', 'doors', 'windows'."
    )


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingHyperparameters:
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    fp16: bool = True
    bf16: bool = False
    output_dir: str = "artifacts/checkpoints"
    adapter_output_dir: str = "artifacts/adapters/final_lora_adapter"
    logs_dir: str = "artifacts/training_logs"


@dataclass
class FloorPlanTrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    train_dir: str = "dataset/train"
    val_dir: str = "dataset/validation"
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> "FloorPlanTrainingConfig":
        """Loads configuration from yaml file."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        model_kwargs = raw.get("model", {})
        lora_kwargs = raw.get("lora", {})
        train_kwargs = raw.get("training", {})
        paths = raw.get("paths", {})

        return cls(
            model=ModelConfig(
                base_model_name=model_kwargs.get("base_model_name", "Qwen/Qwen2.5-VL-3B-Instruct"),
                fallback_model_name=model_kwargs.get("fallback_model_name", "Qwen/Qwen2-VL-2B-Instruct"),
                load_in_4bit=model_kwargs.get("load_in_4bit", True),
                bnb_4bit_compute_dtype=model_kwargs.get("bnb_4bit_compute_dtype", "bfloat16"),
                bnb_4bit_quant_type=model_kwargs.get("bnb_4bit_quant_type", "nf4"),
                use_nested_quant=model_kwargs.get("use_nested_quant", True),
                max_image_pixels=model_kwargs.get("max_image_pixels", 1048576),
                min_image_pixels=model_kwargs.get("min_image_pixels", 65536),
                prompt_template=model_kwargs.get("prompt_template", ModelConfig.prompt_template)
            ),
            lora=LoRAConfig(
                r=lora_kwargs.get("r", 16),
                lora_alpha=lora_kwargs.get("lora_alpha", 32),
                lora_dropout=lora_kwargs.get("lora_dropout", 0.05),
                bias=lora_kwargs.get("bias", "none"),
                target_modules=lora_kwargs.get("target_modules", LoRAConfig().target_modules)
            ),
            training=TrainingHyperparameters(
                learning_rate=float(train_kwargs.get("learning_rate", 2e-4)),
                num_train_epochs=int(train_kwargs.get("num_train_epochs", 3)),
                per_device_train_batch_size=int(train_kwargs.get("per_device_train_batch_size", 1)),
                per_device_eval_batch_size=int(train_kwargs.get("per_device_eval_batch_size", 1)),
                gradient_accumulation_steps=int(train_kwargs.get("gradient_accumulation_steps", 8)),
                warmup_ratio=float(train_kwargs.get("warmup_ratio", 0.03)),
                weight_decay=float(train_kwargs.get("weight_decay", 0.01)),
                logging_steps=int(train_kwargs.get("logging_steps", 5)),
                eval_steps=int(train_kwargs.get("eval_steps", 50)),
                save_steps=int(train_kwargs.get("save_steps", 50)),
                save_total_limit=int(train_kwargs.get("save_total_limit", 2)),
                gradient_checkpointing=bool(train_kwargs.get("gradient_checkpointing", True)),
                fp16=bool(train_kwargs.get("fp16", True)),
                bf16=bool(train_kwargs.get("bf16", False)),
                output_dir=paths.get("checkpoints_dir", "artifacts/checkpoints"),
                adapter_output_dir=paths.get("adapters_dir", "artifacts/adapters") + "/final_lora_adapter",
                logs_dir=paths.get("logs_dir", "artifacts/training_logs")
            ),
            train_dir=paths.get("train_dir", "dataset/train"),
            val_dir=paths.get("val_dir", "dataset/validation"),
            seed=raw.get("project", {}).get("seed", 42)
        )
