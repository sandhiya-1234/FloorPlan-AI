"""
PyTorch Vision-Language Dataset for FloorPlan-AI.

Loads floor plan images and architectural ground-truth JSON annotations,
formatting them into multimodal instruction conversations compatible with
Qwen2.5-VL and Qwen2-VL chat templates. Supports prompt masking for training loss.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.floorplan_schema import FloorPlanSchema

logger = logging.getLogger(__name__)


class FloorPlanVLMDataset(Dataset):
    """
    Multimodal Dataset for Vision-Language Floor Plan Architectural Detection.
    Converts (Image, Structured JSON) pairs into Qwen2.5-VL conversational tokens.
    """

    def __init__(
        self,
        data_dir: str,
        processor: Optional[Any] = None,
        prompt_text: Optional[str] = None,
        max_length: int = 4096,
        mask_prompt: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.prompt_text = prompt_text or (
            "Analyze this 2D architectural floor plan image and detect all architectural elements: "
            "walls, doors, windows, and rooms.\n"
            "Return ONLY a valid JSON object matching the required schema with keys: "
            "'image_width', 'image_height', 'units', 'rooms', 'walls', 'doors', 'windows'."
        )

        self.samples = self._discover_samples()
        logger.info(f"Loaded {len(self.samples)} floor plan samples from {self.data_dir}")

    def _discover_samples(self) -> List[Dict[str, Path]]:
        samples = []
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist.")
            return samples

        png_files = sorted(list(self.data_dir.glob("*.png")) + list(self.data_dir.glob("*.jpg")))
        for p in png_files:
            json_file = p.with_suffix(".json")
            if json_file.exists():
                samples.append({
                    "id": p.stem,
                    "image_path": p,
                    "json_path": json_file
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_raw_sample(self, index: int) -> Dict[str, Any]:
        """Returns raw PIL Image and parsed JSON dictionary without tokenization."""
        entry = self.samples[index]
        image = Image.open(entry["image_path"]).convert("RGB")
        with open(entry["json_path"], "r", encoding="utf-8") as f:
            gt_dict = json.load(f)
        # Validate through schema
        schema = FloorPlanSchema(**gt_dict)
        return {
            "id": entry["id"],
            "image": image,
            "json_data": schema.to_compact_dict(),
            "image_path": str(entry["image_path"])
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.samples[index]
        image = Image.open(entry["image_path"]).convert("RGB")

        with open(entry["json_path"], "r", encoding="utf-8") as f:
            gt_dict = json.load(f)
        schema = FloorPlanSchema(**gt_dict)
        target_json_str = json.dumps(schema.to_compact_dict(), separators=(",", ":"))

        # Build Qwen2.5-VL conversational structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_json_str}
                ]
            }
        ]

        if self.processor is None:
            # Fallback raw dictionary
            return {
                "id": entry["id"],
                "image": image,
                "messages": messages,
                "target_text": target_json_str,
                "image_width": schema.image_width,
                "image_height": schema.image_height
            }

        # Tokenize using Qwen2.5-VL processor
        try:
            # Full text with assistant answer
            text_prompt_only = self.processor.apply_chat_template(
                messages[:1], tokenize=False, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Process full input
            inputs = self.processor(
                text=[full_text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]
            labels = input_ids.clone()

            # Mask out the prompt tokens so model only learns to predict the JSON response
            if self.mask_prompt:
                prompt_inputs = self.processor(
                    text=[text_prompt_only],
                    images=[image],
                    return_tensors="pt"
                )
                prompt_len = prompt_inputs["input_ids"].shape[1]
                # Set prompt token labels to -100 (cross entropy ignore index)
                labels[:prompt_len] = -100

            item = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            if "pixel_values" in inputs:
                item["pixel_values"] = inputs["pixel_values"][0]
            if "image_grid_thw" in inputs:
                item["image_grid_thw"] = inputs["image_grid_thw"][0]

            return item

        except Exception as e:
            logger.error(f"Error processing sample {entry['id']}: {e}")
            raise
