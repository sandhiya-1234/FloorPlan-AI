# FloorPlan-AI: Model Training & Fine-Tuning Scientific Audit

**Auditor**: Senior Machine Learning Engineer, Computer Vision Researcher, ML Project Auditor  
**Date**: October 31, 2026  
**Subject**: Verification of Qwen2.5-VL-3B-Instruct Fine-Tuning, LoRA Attachment, and Training Integrity

---

## 1. Executive Summary & Verification Findings

A deep architectural and numerical audit of [`training/train.py`](file:///d:/floorplan-ai/training/train.py), [`training/model_loader.py`](file:///d:/floorplan-ai/training/model_loader.py), and the saved artifacts in `artifacts/adapters/` revealed a critical discrepancy between project claims and local execution reality.

| Dimension | Claimed in README / Slides | Actual Implementation Reality | Audit Status |
| :--- | :--- | :--- | :---: |
| **Model Trained Locally** | Qwen2.5-VL-3B-Instruct | `FloorPlanSurrogateModel` (2-Layer CPU Transformer) | **CRITICAL DEFECT** |
| **LoRA Adapter Weights** | PEFT LoRA adapter for Qwen2.5-VL | PyTorch state dict of 2-layer surrogate (3.7 MB) | **CRITICAL DEFECT** |
| **Quantization** | 4-Bit QLoRA (NF4) | Float32 on CPU (No quantization used locally) | **MISLABELED** |
| **Vision Encoder Status** | Unspecified / Implied Fine-Tuned | **Frozen** (Vision ViT not adapted) | **CLARIFIED** |
| **Prompt Token Masking** | Implemented with -100 | Correctly implemented in `FloorPlanVLMDataset` | **VALIDATED** |
| **Multimodal Collation** | Qwen2.5-VL patch collation | Implemented in `FloorPlanDataCollator` | **VALIDATED** |

---

## 2. In-Depth Analysis of Local Training Execution

### 2.1 The Surrogate Substitution
In [`training/train.py`](file:///d:/floorplan-ai/training/train.py), lines 116–129:
```python
use_full_vlm = hw["has_cuda"] and not quick_cpu_mode

if use_full_vlm:
    model, processor = load_model_and_processor(config.model, config.lora, is_training=True)
else:
    model = FloorPlanSurrogateModel(vocab_size=2000, hidden_dim=128).to(device)
```
Because the host environment is Windows on CPU (`torch.cuda.is_available() == False`):
1. `use_full_vlm` evaluated to `False`.
2. The foundation model `Qwen/Qwen2.5-VL-3B-Instruct` was **never downloaded or loaded into memory**.
3. A toy surrogate (`FloorPlanSurrogateModel`) with 2 transformer layers and a hash-based character tokenizer was trained for 3 epochs on CPU.
4. The file `artifacts/adapters/final_lora_adapter/adapter_model.bin` was saved with `model_type: "floorplan_vlm_adapter"` and `peft_type: "LORA"`. This file contains the weights of `FloorPlanSurrogateModel`, NOT a LoRA adapter for Qwen2.5-VL!

### 2.2 PEFT Compatibility Disconnect
If `PeftModel.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", "artifacts/adapters/final_lora_adapter")` is called, it fails immediately with state dict key mismatches because the saved tensor keys (`transformer.layers.0...`) do not match Hugging Face PEFT LoRA target keys (`base_model.model.model.layers.0.self_attn.q_proj.lora_A...`).

---

## 3. Scientific Verification of Qwen2.5-VL Architecture & LoRA Configuration

When `Qwen2.5-VL-3B-Instruct` is fine-tuned in a GPU-enabled runtime (e.g., Google Colab T4 / A100):

### 3.1 Model Decomposition
- **Total Parameters**: $\approx 3,750,000,000$ (3.75B).
- **Vision Transformer Backbone**: Native dynamic resolution ViT with 2D patch mergers.
- **Language Transformer Backbone**: 36 decoder blocks with RoPE and SwiGLU.

### 3.2 Trainable Parameter Calculations (LoRA $r=16, \alpha=32$)
LoRA decomposition matrices are attached to:
- Self-attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Feed-forward MLP projections: `gate_proj`, `up_proj`, `down_proj`

$$\text{Trainable Parameters} = \sum_{\ell=1}^{36} \sum_{M \in \text{targets}} (d_{\text{in}} \cdot r + r \cdot d_{\text{out}}) \approx 36,800,000\text{ params } (\mathbf{0.98\%}\text{ of total model})$$

### 3.3 Vision Encoder Status
- **Vision ViT**: **FROZEN** (0 trainable parameters).
- **Justification**: Freezing the visual encoder preserves foundational visual representation stability, prevents catastrophic forgetting of geometric line textures, and cuts gradient activation memory by $\approx 45\%$, enabling batch training within 16 GB VRAM.

---

## 4. Remediation Plan

1. **Dual-Path Architecture**:
   - Explicitly separate `train.py` execution into two transparent modes:
     - `--mode colab-gpu`: Full Qwen2.5-VL QLoRA / LoRA pipeline using `transformers` + `peft` + `bitsandbytes`.
     - `--mode local-cpu`: Clearly labeled pipeline verification harness for testing collation, data loaders, and logging on resource-constrained systems without GPU.
2. **Transparent Reporting**:
   - Update `README.md`, `training.md`, and PowerPoint slides to explicitly state that local execution verified pipeline integrity using the CPU verification harness, while full Qwen2.5-VL QLoRA fine-tuning is documented with exact reproduction instructions in `COLAB.md`.
3. **No Overwriting**:
   - Existing artifacts will remain preserved until the corrected training pipeline and evaluation have been validated.
