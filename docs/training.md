# Training notes

**Status: no training has been executed.** There are no loss curves, checkpoints or adapter weights in this
repository, and no result in these notes comes from a run. The earlier version of this file quoted losses
(7.88 to 7.04) that belonged to a toy surrogate network, not to Qwen2.5-VL; they are archived under
`docs/audit/invalid_artifacts/toy_surrogate_training/` and must not be cited.

## What exists

`training/train.py`, `training/model_loader.py`, `training/dataset.py`, `training/collator.py` are a first
draft of a Qwen2.5-VL-3B-Instruct LoRA/QLoRA trainer. They have never been executed on a GPU, so they may contain
bugs. `train.py` raises `RuntimeError` when no CUDA GPU is present; there is deliberately no CPU fallback.
It also raises if validation produces no valid loss, rather than inventing a number.

Known gaps that the Phase 3 rewrite must close before any run counts: no learning-rate scheduler (warmup is computed but not applied),
no explicit fp16 mixed-precision handling (gradient scaling) for a T4, which has no bf16, no resumable checkpoints, no saved run manifest
(package versions, config hash, dataset manifest hash, seed), and the adapter is selected by the best validation loss
of a tiny validation set.

## Planned configuration (from `config.yaml`, not yet reconciled)

| Item | Value |
| :--- | :--- |
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Method | LoRA on `q,k,v,o,gate,up,down_proj`, r=16, alpha=32, dropout 0.05; 4-bit NF4 QLoRA |
| Hardware | Colab T4 (16 GB), fp16 compute |
| Optimiser | AdamW, lr 2e-4, weight decay 0.01 |
| Batching | per-device batch 1, gradient accumulation 8 |
| Loss | next-token cross-entropy on the assistant JSON tokens only (prompt and image tokens masked) |

Planned change to the target format: the model should emit a compact JSON (coordinate arrays without ids,
areas or confidence), expanded deterministically into the full schema afterwards. This shortens the output and
removes fields the model cannot know. Not implemented yet.

## Procedure (Phase 3, not done)

1. Convert and verify a CubiCasa5K subset (Phase 2), letterboxed to 1008x1008.
2. Sanity run: overfit 8 training images. It shows that the pipeline can fit; it says nothing about generalisation.
3. Train on the training split, validate on the validation split, never touch the test split.
4. Save the adapter and run `inference.adapter_check.inspect_adapter` on it: paired LoRA A/B tensors, consistent rank,
   finite values, at least one non-zero B matrix, base-model name match.
5. Record the run manifest next to the adapter.

A model counts as fine-tuned only after steps 3 to 5 exist as artifacts.
