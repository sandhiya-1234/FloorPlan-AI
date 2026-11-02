# FloorPlan-AI: 2D floor plans to structured JSON with a LoRA-fine-tuned vision-language model

**Goal:** fine-tune a pretrained vision-language model (Qwen2.5-VL-3B-Instruct, LoRA/QLoRA) to detect and
classify **walls, doors, windows and rooms** in 2D floor-plan images and emit structured spatial JSON.

## Current status (read this first)

> **No model has been trained yet, and there are no evaluation results.**
> The branch `repair/real-vlm-pipeline` is replacing an earlier version of this project that *claimed* a
> fine-tuned model but actually used an OpenCV heuristic, a toy surrogate network and synthetic data.
> Those artifacts are quarantined in [`docs/audit/invalid_artifacts/`](docs/audit/invalid_artifacts/README.md)
> and must not be cited. This README only states what has been implemented, executed and verified.

| Component | Implemented | Executed | Verified |
| :--- | :---: | :---: | :--- |
| Repository integrity cleanup (fake results, fake adapter, fake samples removed) | yes | yes | yes, see `tests/test_integration.py` and the archive README |
| LoRA adapter verification (`inference/adapter_check.py`) | yes | unit tests only | tested on synthetic safetensors files; **never run on a real adapter (none exists)** |
| Prediction records (`inference/records.py`): a failed record cannot carry a prediction | yes | unit tests | yes |
| Strict JSON parsing and schema validation (`inference/postprocess.py`) | yes | unit tests | yes; invalid model text raises, it never becomes an empty plan |
| Geometric postprocessing (wall snap, duplicate suppression, portal snap, polygon repair) | yes | unit tests | partly: polygon repair needs `shapely`, which was not installable in the sandbox, so 2 tests were not run there (run them on Colab) |
| Letterbox preprocessing, 1008x1008 canvas, exact coordinate round-trip | yes | unit tests | yes |
| OpenCV baseline (`baselines/opencv_baseline.py`), a comparison point only, **not AI** | yes | unit tests on tiny drawn images | no benchmark run |
| Evaluation (Hungarian matching, failures counted as misses) | yes | unit and end-to-end tests on hand-made fixtures | no run on model output |
| Streamlit dashboard (`app.py`) | yes | compiled only | **never launched** |
| Qwen2.5-VL LoRA training script (`training/`) | written | **never executed** | no. A verified rewrite is planned (Phase 3) |
| Qwen2.5-VL generation (`inference/predict.py`) | **not implemented** | no | the function verifies the adapter and then raises `NotImplementedError` |
| Real dataset (CubiCasa5K subset) | **not downloaded** | no | the annotation parser has not been checked against real files |
| Trained adapter weights | **do not exist** | | |
| Evaluation results of any system | **none** | | |

## What the finished system will do

```
floor-plan image
   -> letterbox to 1008x1008 (aspect ratio kept, coordinates mapped back exactly)
   -> Qwen2.5-VL-3B-Instruct + LoRA adapter (must pass adapter_check against the base model)
   -> raw generated text (always saved)
   -> strict JSON parse + schema validation (failure is recorded, never repaired silently)
   -> optional geometric postprocessing (raw and processed outputs both stored)
   -> structured JSON: walls, doors, windows, rooms
```

Output schema: `schemas/floorplan_schema.py`. Confidence and wall thickness are `None` unless the model
produced them; nothing is filled in with default values.

## Repository layout

```
app.py                       Streamlit dashboard (shows only verified data, labels everything else)
baselines/opencv_baseline.py Classical comparison baseline (walls and rooms only; never doors/windows/confidence)
inference/
  adapter_check.py           Verifies a LoRA adapter folder without torch (paired A/B tensors, rank, finite, B != 0)
  predict.py                 Adapter check, then VLM generation (generation: Phase 4, not implemented)
  postprocess.py             Strict parsing + geometric refinement
  records.py                 <image>.pred.json record format (ok / failed with stage and error)
evaluation/                  Hungarian-matching metrics and evaluate.py (CLI)
preprocessing/               letterbox.py (verified); download/convert scripts still target the old synthetic format
training/                    Qwen2.5-VL LoRA trainer (GPU only, not yet executed)
schemas/                     Pydantic schema
tests/                       pytest suite
docs/                        methodology, dataset, training notes; docs/audit/ holds the audit history
presentation/                Slide generator (only reads verified results; says so when none exist)
```

## Installation: two separate environments

```bash
git clone https://github.com/sandhiya-1234/FloorPlan-AI.git
cd FloorPlan-AI
git checkout repair/real-vlm-pipeline
```

| File | Use | Contents |
| :--- | :--- | :--- |
| `requirements.txt` | dashboard / laptop, no GPU | streamlit, pydantic, numpy, pillow, opencv-python-headless, pandas, pyyaml, shapely |
| `requirements-dev.txt` | tests, evaluation, slide generation | includes `requirements.txt` plus scipy, matplotlib, python-pptx, pytest, tqdm |
| `requirements-train.txt` | Google Colab GPU only | torch, transformers>=4.49.0, peft, accelerate, bitsandbytes, qwen-vl-utils, ... |

The laptop has no NVIDIA GPU: **do not download the 3B base model locally.** Training and model inference run on a
Colab T4 (see [COLAB.md](COLAB.md)).

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Things you can run today

```bash
# Verify that a folder really contains a usable LoRA adapter (fails loudly otherwise)
python -c "from inference.adapter_check import inspect_adapter; print(inspect_adapter('artifacts/adapters/final_lora_adapter', 'Qwen/Qwen2.5-VL-3B-Instruct'))"

# OpenCV comparison baseline over a folder of images (writes <image>.pred.json records)
python -m baselines.opencv_baseline --image-dir <images> --out-dir outputs/predictions/opencv_baseline

# Score a folder of prediction records against ground-truth JSON
python evaluation/evaluate.py --gt-dir <gt_dir> --pred-dir <pred_dir> --system-name opencv_baseline --out-dir outputs/evaluation/opencv_baseline
```

Today the adapter command fails with `AdapterError` because no adapter exists. That is the correct behaviour.

## Evaluation protocol (implemented, not yet applied to any real data)

Three systems will be compared on the same unseen test images: zero-shot Qwen2.5-VL-3B, Qwen2.5-VL-3B + our LoRA
adapter, and the OpenCV baseline.

- Matching: Hungarian optimal assignment (`scipy.optimize.linear_sum_assignment`) per element class.
- Walls: matched if endpoint error is at most 25 px. Doors and windows: bbox IoU of at least 0.5. Rooms: polygon IoU of at least 0.5.
- Micro-averaged precision / recall / F1 per class; room-type confusion matrix.
- A prediction that failed (load, generate, JSON parse, schema validation) counts as all ground truth missed and is
  reported as a failure count; it is never replaced by default geometry.
- A warning is attached to the results when fewer than 50 test images are used.
- Leakage control (Phase 2): official CubiCasa5K split lists plus exact-hash, perceptual-hash and annotation-similarity checks across splits.

## Planned training configuration (NOT executed)

LoRA r=16, alpha=32, dropout 0.05 on `q,k,v,o,gate,up,down_proj`; 4-bit NF4 QLoRA; fp16 on a T4 (no bf16);
learning rate 2e-4; batch 1 with gradient accumulation 8; loss on assistant tokens only. These are the values in
`config.yaml`, which has **not been reconciled yet** with the planned 1008-px letterbox input or the compact output format.
Phase 3 starts with an 8-image overfit sanity run, which proves only that the pipeline can fit, not that it generalises.

## Limitations

- Everything above "Trained adapter weights" in the status table is untested on real data or a real model.
- The dataset has not been downloaded; the annotation converter was written against synthetic files and must be re-checked on real CubiCasa5K SVGs.
- CubiCasa5K is licensed CC BY-NC-SA 4.0 (non-commercial, share-alike). Any released adapter inherits those terms.
- Polygon-related tests need `shapely`, which could not be installed in the sandbox used for this phase.
- Postprocessing assumes mostly orthogonal walls; slanted or curved walls are kept as predicted but not refined.
- The OpenCV baseline cannot detect doors or windows and does not classify rooms. That is by design.

## Roadmap

1. Integrity cleanup. **Done on this branch.**
2. Real CubiCasa5K subset, annotation conversion verified by image overlays, leakage-checked splits.
3. Qwen2.5-VL-3B LoRA/QLoRA training on Colab T4 (overfit-on-8 sanity run first).
4. Real inference module: base/adapter compatibility check, raw generations saved, loud failures.
5. Three-system evaluation on unseen plans.
6. Dashboard, README and slides updated from verified results only.

## License and data

Code: see repository. Data: CubiCasa5K, CC BY-NC-SA 4.0. Base model: Qwen2.5-VL-3B-Instruct under its own licence.
