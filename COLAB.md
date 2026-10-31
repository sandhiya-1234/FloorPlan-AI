# Google Colab GPU Training Guide: FloorPlan-AI

This guide provides step-by-step instructions for fine-tuning **Qwen2.5-VL-3B-Instruct** using **QLoRA (4-bit)** on a free Google Colab **NVIDIA Tesla T4 GPU**.

---

## 1. Colab Runtime Setup

1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime** > **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU**.
4. Click **Save**.

---

## 2. Execute Training in Colab

Copy and run the following code blocks in your Colab notebook:

### Cell 1: Clone Repository & Install GPU Dependencies
```python
# Clone the project repository
!git clone https://github.com/your-username/floorplan-ai.git
%cd floorplan-ai

# Install GPU accelerated dependencies
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers>=4.48.0 peft>=0.12.0 accelerate>=0.33.0 bitsandbytes>=0.43.0
!pip install -q pydantic opencv-python pillow shapely pyyaml matplotlib pandas python-pptx pytest
```

### Cell 2: Verify GPU Environment
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
```

### Cell 3: Prepare Dataset
```python
# Download and preprocess 50 CubiCasa5K floor plans
!python preprocessing/download_dataset.py --samples 50
!python preprocessing/convert_annotations.py
!python preprocessing/split_dataset.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### Cell 4: Launch QLoRA Fine-Tuning
```python
# Run fine-tuning with 4-bit quantization and LoRA
!python training/train.py --config config.yaml --epochs 3 --lr 0.0002
```

### Cell 5: Run Evaluation on Unseen Test Split
```python
!python evaluation/evaluate.py
```

### Cell 6: Package & Download Trained LoRA Adapter
```python
from google.colab import files
import shutil

# Zip the trained LoRA adapter weights
shutil.make_archive("final_lora_adapter", "zip", "artifacts/adapters/final_lora_adapter")

# Download to your local machine for local inference
files.download("final_lora_adapter.zip")
```

---

## 3. Using the Trained Adapter Locally

Once downloaded, extract `final_lora_adapter.zip` into your local `artifacts/adapters/final_lora_adapter/` directory.

Run local inference:
```bash
python inference/predict.py --image samples/sample1.png
```

Launch the Streamlit app:
```bash
streamlit run app.py
```
