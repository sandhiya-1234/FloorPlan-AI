"""
Data Collator for FloorPlan-AI Vision-Language Model Fine-Tuning.

Dynamically pads multimodal token sequences, masks labels with -100 for ignore_index,
and collates vision patches and image_grid_thw tensors for Qwen2.5-VL architecture.
"""

from typing import List, Dict, Any, Optional
import torch


class FloorPlanDataCollator:
    """
    Multimodal batch collator for Qwen2.5-VL / Qwen2-VL architectures.
    Pads input_ids, labels (-100), and concatenates vision embeddings.
    """

    def __init__(self, processor: Any, pad_token_id: Optional[int] = None):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else getattr(self.tokenizer, "pad_token_id", 0) or 0
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Handle cases where items are tokenized dicts
        if "input_ids" in batch[0]:
            input_ids_list = [item["input_ids"] for item in batch]
            attention_mask_list = [item["attention_mask"] for item in batch]
            labels_list = [item["labels"] for item in batch]

            # Dynamic padding to max sequence length in this batch
            max_len = max(x.size(0) for x in input_ids_list)

            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []

            for input_ids, mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
                pad_len = max_len - input_ids.size(0)
                if pad_len > 0:
                    padded_input_ids.append(
                        torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)])
                    )
                    padded_attention_mask.append(
                        torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                    )
                    padded_labels.append(
                        torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
                    )
                else:
                    padded_input_ids.append(input_ids)
                    padded_attention_mask.append(mask)
                    padded_labels.append(labels)

            collated = {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask),
                "labels": torch.stack(padded_labels),
            }

            # Collate pixel values if present
            if "pixel_values" in batch[0]:
                collated["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)

            # Collate image_grid_thw if present
            if "image_grid_thw" in batch[0]:
                collated["image_grid_thw"] = torch.stack([item["image_grid_thw"] for item in batch])

            return collated

        # Handle raw message dicts fallback
        return {k: [item[k] for item in batch] for k in batch[0].keys()}
