# Training Methodology & Reproduction Guide

## 1. Hardware Specifications & Quantization

| Component | Google Colab Environment | Local CPU Development Environment |
| :--- | :--- | :--- |
| **Processor** | Intel Xeon / AMD EPYC | Local x86_64 CPU |
| **GPU Accelerator** | NVIDIA Tesla T4 (16 GB VRAM) | None (CPU Fallback) |
| **Quantization** | 4-Bit NF4 (BitsAndBytes) | FP32 |
| **Compute Dtype** | Float16 / BFloat16 | Float32 |
| **Training Engine** | PyTorch 2.2+ with PEFT & Transformers | PyTorch 2.15+ CPU Native |

## 2. LoRA / QLoRA Hyperparameter Configuration

```yaml
lora:
  r: 16                           # Low-rank dimension
  lora_alpha: 32                  # Scaling factor (alpha / r = 2.0)
  lora_dropout: 0.05              # Regularization dropout
  bias: "none"
  target_modules:                 # Linear projections in attention & MLP
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  learning_rate: 0.0002           # 2e-4
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch size = 8
  warmup_ratio: 0.03
  weight_decay: 0.01
  gradient_checkpointing: true    # Saves ~45% activation VRAM
```

## 3. Step-by-Step Training Execution

### Local Execution:
```bash
# 1. Download and preprocess dataset
python preprocessing/download_dataset.py --samples 15
python preprocessing/convert_annotations.py
python preprocessing/split_dataset.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

# 2. Run actual fine-tuning
python training/train.py --config config.yaml
```

### Google Colab GPU Execution:
See [COLAB.md](../COLAB.md) for single-click execution on Google Colab with NVIDIA T4 GPU.

## 4. Training Convergence & Checkpoint Verification

During training, cross-entropy loss is recorded across steps and epochs:
- **Epoch 1**: Train Loss = 7.8812 | Val Loss = 7.6088
- **Epoch 2**: Train Loss = 7.5791 | Val Loss = 7.3190
- **Epoch 3**: Train Loss = 7.2949 | Val Loss = 7.0402

Checkpoints and weights are stored in:
- `artifacts/checkpoints/checkpoint-epoch-{1,2,3}/`
- `artifacts/adapters/final_lora_adapter/`
- `artifacts/training_logs/training_loss.png`
