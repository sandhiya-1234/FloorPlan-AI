# Google Colab GPU guide (T4)

Colab is where all model training and model inference run, because the development laptop has no NVIDIA GPU.
Do **not** download the 3B base model to the laptop.

**Status:** only the cells in section 2 exist today and none of them trains or runs a model. Training,
inference and evaluation cells will be added here when those phases are implemented *and* verified.
Nothing below has been run on Colab yet, so no outputs are shown.

## 1. Runtime

Runtime > Change runtime type > **T4 GPU** > Save.

## 2. Cells that are valid now

### Cell 1: clone the branch and install

```python
!git clone https://github.com/sandhiya-1234/FloorPlan-AI.git
%cd FloorPlan-AI
!git checkout repair/real-vlm-pipeline
!git log --oneline -1

# Quote version specifiers: an unquoted ">=" is a shell redirect and silently writes a file named "=4.49.0".
!pip install -q -r requirements-train.txt
!pip install -q "pytest>=7.4.0"
```

Colab already ships a CUDA build of torch. If `pip` offers to replace it, keep Colab's version unless a later phase says otherwise.

### Cell 2: check the GPU

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("Device:", p.name, "| VRAM GB:", round(p.total_memory / 1024**3, 1))
    print("bf16 supported:", torch.cuda.is_bf16_supported())  # a T4 reports False: train in fp16
```

### Cell 3: run the test suite (including the Shapely tests)

Three tests are skipped in the sandbox where the code was written, because `shapely` could not be installed there.
With `requirements-train.txt` installed they should run here.

```python
!python -m pytest tests -q -rs
```

Read the summary line. If any test is still reported as skipped, `-rs` prints the reason. Skipped is not passed.

## 3. Cells to be added later (not available yet)

| Phase | What the cells will do |
| :--- | :--- |
| 2 | download a small CubiCasa5K subset, convert annotations, write overlay images, run leakage checks |
| 3 | 8-image overfit sanity run, then LoRA/QLoRA training with a saved run manifest |
| 4 | verify the adapter against the base model, then generate JSON from images and save raw outputs |
| 5 | run zero-shot Qwen, fine-tuned Qwen + LoRA and the OpenCV baseline on the test split, then evaluate |

Until Phase 3 has run, `artifacts/adapters/final_lora_adapter/` is empty and `inference/predict.py` raises on purpose.
