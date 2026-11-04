# Training notes

**Status: no training has been executed, by either approach below.** This repository contains no loss curve, no checkpoint and no weights
from a real run. Numbers that earlier versions of this file quoted (losses 7.88 to 7.04) belonged to a toy surrogate network and are archived under
`docs/audit/invalid_artifacts/toy_surrogate_training/`; they must not be cited.

## A. Active approach: U-Net with a pretrained ResNet-18 encoder (semantic segmentation)

Code: `training/seg_model.py`, `training/train_segmentation.py`. Configuration: `config/segmentation.yaml`. Run it with
`notebooks/colab_segmentation.ipynb` (Stage B).

### Why this model instead of the Qwen2.5-VL plan

The original plan was a LoRA-fine-tuned vision-language model that writes JSON. For the deadline (5 November 2026), the available hardware (one Colab T4,
16 GB) and the need for a result that can be *measured*, a segmentation network is the more suitable first model:

- The supervision is dense and exact: CubiCasa5K gives polygons for every wall, door, window and room, which turn directly into per-pixel masks. A VLM would
  have to learn to emit hundreds of coordinates as text.
- Output quality is measurable with standard, well-defined metrics (per-class IoU, Dice/F1, precision, recall, confusion matrix) with no JSON-parsing failure mode in the model.
- Training is cheap and stable on a T4. A 3B VLM with 4-bit QLoRA at high image resolution is slow there, and its earlier draft had never been executed.
- The polygon JSON is derived deterministically from the predicted mask, so the model cannot invent entities that have no pixels behind them.

Costs of this choice: it predicts a class per pixel, not instances, so touching rooms or two doors that merge in the mask become one polygon; walls come out as
regions, not centre-lines; there are no room types (all rooms are one class). These are limitations of the method, not bugs.

### What the code does

| Item | Value (from `config/segmentation.yaml`) |
| :--- | :--- |
| Architecture | torchvision ResNet-18 encoder (ImageNet weights `IMAGENET1K_V1`) + four-stage U-Net decoder with skip connections + 1x1 classifier, 5 classes |
| Trained parameters | all of them: the encoder is fine-tuned end to end, the decoder starts from random initialisation |
| Input | 512x512 letterboxed (aspect ratio preserved), ImageNet normalisation |
| Loss | cross-entropy (optional sqrt-inverse-frequency class weights computed from the training masks) + soft Dice; both skip pixels labelled 255 |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4, linear warmup then cosine decay, gradient clipping 5.0 |
| Batch / epochs | batch size 4, 25 epochs |
| Precision | fp16 autocast with gradient scaling on CUDA (a T4 has no bf16) |
| Augmentation | flips and 90-degree rotations |
| Model selection | the epoch with the best validation mean IoU is saved as `best.pt` |
| Seed | 42 |

Outputs of a run (`runs/seg/<run-name>/`): `best.pt` (state dict, class names, image size, train and validation ids, epoch, validation mIoU), `history.json` (every epoch:
learning rate, train loss, validation loss, validation mIoU and Dice), `config.json` (the resolved configuration and environment), `train_summary.json` (best epoch, optimizer steps,
duration, checkpoint sha256 and size).

### Integrity gates built into the code

1. Training requires `data/seg/manifest.json` and a human-created `overlays_confirmed.txt` (the overlays of the real masks must have been looked at).
2. `--overfit N` trains and validates on the same N images as a pipeline sanity check. The checkpoint is marked `overfit_sanity` and evaluation refuses it.
3. Evaluation aborts if any test image id appears among the checkpoint's recorded training or validation ids.
4. Inference loads the checkpoint strictly (`weights_only=True`, exact architecture, sha256 recorded) and raises if it is missing; there is no pretrained-only or heuristic fallback.

### What is and is not known

| Claim | Status |
| :--- | :--- |
| The model, loss, scheduler, checkpoint writer and loader are implemented | yes |
| Their numpy-side mechanics are unit-tested | yes (17 tests). The torch-dependent tests are **skipped in the authoring sandbox** (no torch there) and have not been run |
| A training run completed | **no** |
| Any validation or test metric | **none exists** |
| The hyperparameters are suitable | **untested**: they are reasonable defaults, not tuned values |

## B. Original plan (not executed): Qwen2.5-VL-3B LoRA / QLoRA

`training/train.py`, `training/model_loader.py`, `training/dataset.py`, `training/collator.py` are an early draft of a Qwen2.5-VL-3B-Instruct LoRA/QLoRA trainer.
They were **never executed** and may contain bugs. They are kept for reference only and are **not** part of the submitted result. `train.py` raises `RuntimeError` without a CUDA GPU
and does not invent a validation loss.

Known gaps in that draft: no learning-rate scheduler (warmup computed but not applied), no explicit fp16 gradient scaling for a T4, no resumable checkpoints, no run manifest,
validation on a tiny set. Planned configuration, unreconciled with the draft: LoRA r=16, alpha=32, dropout 0.05 on `q,k,v,o,gate,up,down_proj`; 4-bit NF4; lr 2e-4; batch 1 with
gradient accumulation 8; loss on assistant tokens only. `inference/adapter_check.py` can verify a LoRA adapter folder, but no adapter exists.
