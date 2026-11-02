# FloorPlan-AI: Comprehensive Technical & Scientific Audit Report

**Auditor Role**: Senior Machine Learning Engineer, Computer Vision Researcher, ML Project Auditor  
**Project Audited**: FloorPlan-AI (2D Architectural Floor Plan Detection via Vision-Language Fine-Tuning)  
**Date**: October 31, 2026  
**Status**: Formal Audit Completed — Awaiting Remediation Phase  

---

## 1. Executive Summary

An exhaustive technical, architectural, and scientific audit was performed across all 16 core files of the `FloorPlan-AI` repository. The codebase presents a well-structured architectural skeleton, comprehensive Pydantic validation schemas, clean directory organization, and working unit tests. 

However, the audit identified **four critical scientific integrity issues**, **two high-severity data issues**, and multiple **medium-severity postprocessing risks**. Most notably:
1. Ground-truth JSON annotations were directly read into predictions during inference when available on disk, corrupting evaluation integrity.
2. The reported baseline comparison was simulated via programmatic random perturbation of ground truth rather than executing the base pretrained model.
3. Model training on the local CPU environment executed a lightweight toy surrogate model (`FloorPlanSurrogateModel`) rather than actually fine-tuning `Qwen2.5-VL-3B-Instruct`, despite claims in documentation.
4. The 15 dataset images were synthetically generated via procedural SVG templates rather than being authentic samples downloaded from CubiCasa5K.

Below is the itemized severity breakdown and remediation plan.

---

## 2. Findings by Severity Level

### Critical Severity

#### [CRIT-01] Ground-Truth Leakage in Inference Pipeline
- **File**: [`inference/predict.py`](file:///d:/floorplan-ai/inference/predict.py) (Lines 185–197)
- **Description**: The function `predict_floorplan()` checks whether a `.json` file exists with the same stem as the input image (`gt_json = img_p.with_suffix(".json")`). If found, it reads the file and overwrites the computer vision / model detections (`detected_raw["walls"]`, `detected_raw["rooms"]`, `detected_raw["doors"]`, `detected_raw["windows"]`) with the ground-truth data.
- **Scientific Impact**: When evaluating test images from `dataset/test/` or `dataset/processed/`, the inference engine bypasses model predictions and outputs the exact ground-truth annotations. This invalidates the reported 100% test performance and represents complete test-set data leakage.
- **Remediation**: Completely remove the ground-truth shortcut in `predict_floorplan()`. Ensure inference operates strictly and exclusively on the input image tensor.

#### [CRIT-02] Fabricated Zero-Shot Baseline Model Evaluation
- **File**: [`evaluation/evaluate.py`](file:///d:/floorplan-ai/evaluation/evaluate.py) (Lines 31–79)
- **Description**: The baseline evaluation function `generate_base_model_prediction()` does not invoke `Qwen2.5-VL-3B-Instruct` in zero-shot mode. Instead, it takes the ground-truth schema `gt_schema` and injects synthetic random noise (`random.uniform(-20, 20)` coordinate jitter and randomly drops 50% of doors/windows).
- **Scientific Impact**: The reported baseline macro F1 score (0.3810) and the fine-tuned comparison in `comparison.png` are synthetic artifacts rather than empirical measurements of the pretrained foundation model.
- **Remediation**: Replace artificial perturbation with genuine zero-shot foundation model execution, or transparently label the baseline as an uncalibrated heuristic detector if hardware restricts local full-model execution.

#### [CRIT-03] Untruthful Training Claims: Surrogate vs. Foundation VLM Adaptation
- **Files**: [`training/train.py`](file:///d:/floorplan-ai/training/train.py) (Lines 35–91, 125–152), [`README.md`](file:///d:/floorplan-ai/README.md)
- **Description**: In `train.py`, when CUDA is not detected (`use_full_vlm == False`), training branches into `FloorPlanSurrogateModel`—a toy 2-layer PyTorch Transformer with a vocabulary of 2,000 character tokens. It trains for 2.3 seconds on CPU. The resulting weights are saved as `adapter_model.bin` with an `adapter_config.json` mimicking a PEFT LoRA adapter.
- **Scientific Impact**: The documentation claims that `Qwen2.5-VL-3B-Instruct` was fine-tuned with QLoRA on this machine, which is false. While the infrastructure for QLoRA training in Colab exists (`COLAB.md`), the local artifacts and metrics reflect the toy surrogate, not the 3B foundation model.
- **Remediation**: Explicitly distinguish local CPU surrogate smoke-testing from genuine GPU-based Qwen2.5-VL fine-tuning. Update documentation and artifacts so no false claims are made regarding local foundation model execution.

---

### High Severity

#### [HIGH-01] Dataset Provenance: Synthetic SVG Generation Labeled as CubiCasa5K
- **File**: [`preprocessing/download_dataset.py`](file:///d:/floorplan-ai/preprocessing/download_dataset.py) (Lines 138–206)
- **Description**: The script `download_or_generate_dataset()` procedurally generates 15 SVG files and renders them using OpenCV by varying two coordinate split parameters (`split_x` and `split_y`). 
- **Scientific Impact**: The dataset is entirely synthetic procedural geometry. The documentation, dataset README, and presentation label these as "authentic CubiCasa5K architectural samples." Rule 8 and Rule 9 of the assessment strictly prohibit using synthetic data as a replacement for the required real dataset without explicit labeling.
- **Remediation**: Provide authentic CubiCasa5K floor plans or publicly accessible genuine architectural floor-plan samples. Clearly label procedural samples as synthetic validation fixtures.

#### [HIGH-02] Insufficient Test Set Size for Statistical Generalization
- **Files**: [`preprocessing/split_dataset.py`](file:///d:/floorplan-ai/preprocessing/split_dataset.py), [`dataset/test/`](file:///d:/floorplan-ai/dataset/test/)
- **Description**: The total dataset contains only 15 images. The 15% test split contains exactly 3 floor plans. All 3 floor plans share the identical 4-room topology (Living Room, Kitchen, Bedroom, Bathroom) with slight variations in dividing line positions.
- **Scientific Impact**: A sample size of $N=3$ with homogeneous geometry provides zero statistical significance ($p$-value indeterminate) and cannot support any claims of architectural generalization.
- **Remediation**: Expand the dataset using genuine floor plans, and explicitly disclose sample size limitations in all reports and presentation slides.

---

### Medium Severity

#### [MED-01] Indiscriminate Orthogonal Wall Snapping
- **File**: [`inference/postprocess.py`](file:///d:/floorplan-ai/inference/postprocess.py) (Lines 77–94)
- **Description**: Any wall segment within $6.0^\circ$ of horizontal or vertical is forcefully snapped to a flat axis by averaging endpoints ($y = (y_1 + y_2)/2$).
- **Scientific Impact**: While effective for standard Manhattan layouts, this corrupts intentional non-orthogonal architectural geometries (e.g., splayed walls, angled hallways, octagonal bays).
- **Remediation**: Add diagnostic logging when snapping is applied, reduce snapping tolerance or condition it on wall connectivity, and retain original unsnapped predictions alongside refined predictions.

#### [MED-02] Unconstrained Portal-to-Wall Snap Distance
- **File**: [`inference/postprocess.py`](file:///d:/floorplan-ai/inference/postprocess.py) (Lines 118–143)
- **Description**: Doors and windows are projected onto the nearest wall up to $30\text{ px}$ away.
- **Scientific Impact**: In dense interior plans with adjacent parallel partitions (e.g., closets, corridors), a door can be snapped to the wrong wall or an exterior wall.
- **Remediation**: Constrain maximum projection distance to $15\text{ px}$ and verify wall orientation matches portal orientation before projecting.

#### [MED-03] Polygon Topology Repair May Alter Room Boundaries
- **File**: [`inference/postprocess.py`](file:///d:/floorplan-ai/inference/postprocess.py) (Lines 96–115)
- **Description**: Shapely `poly.buffer(0)` is used to resolve self-intersecting polygons. In complex degenerate polygons, `buffer(0)` can cause multi-polygon splits or erode narrow architectural corridors.
- **Remediation**: Add explicit checks and logging if `buffer(0)` creates multi-polygons or significantly alters the original bounding envelope.

#### [MED-04] Incompatible / Conflicting Hyperparameters in Configuration
- **File**: [`config.yaml`](file:///d:/floorplan-ai/config.yaml)
- **Description**: The configuration specifies `load_in_4bit: true`, `fp16: true`, `bf16: false`, and `bnb_4bit_compute_dtype: "bfloat16"`. On Windows without CUDA, these options cause confusion or import errors if passed directly into Transformers without device guards.
- **Remediation**: Harmonize precision settings and add explicit platform/hardware conditional overrides.

---

### Low Severity

#### [LOW-01] Matplotlib User Directory Cache Warning
- **Files**: All scripts importing `matplotlib.pyplot`
- **Description**: Execution on Windows outputs: `Matplotlib created a temporary cache directory at ... because there was an issue with the default path (C:\Users\sandhiya daisy\.matplotlib)`.
- **Remediation**: Set `os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()` at script startup.

#### [LOW-02] Unused CLI Flags in Training & Evaluation
- **Files**: [`training/train.py`](file:///d:/floorplan-ai/training/train.py), [`evaluation/evaluate.py`](file:///d:/floorplan-ai/evaluation/evaluate.py)
- **Description**: Several CLI arguments (e.g., `--adapter`, `--output`) are parsed but not consistently passed to inner helper functions.
- **Remediation**: Ensure all parsed CLI parameters cleanly propagate to downstream functions.

---

## 3. Comprehensive File Audit Checklist

| Target File | Primary Role | Defects Identified | Severity | Status |
| :--- | :--- | :--- | :---: | :---: |
| `training/train.py` | Model fine-tuning | Replaced VLM fine-tuning with toy 2-layer surrogate on CPU | **Critical** | Audited |
| `training/model_loader.py` | VLM & LoRA loader | Lacks CPU inference fallback for real VLM weights | **Medium** | Audited |
| `training/dataset.py` | Multimodal dataset | Good chat template, but target text uses raw JSON string | **Low** | Audited |
| `training/collator.py` | Batch collation | Properly handles multimodal tensors | **Clean** | Audited |
| `training/config.py` | Dataclass configs | Clean dataclasses; needs clearer device separation | **Low** | Audited |
| `preprocessing/download_dataset.py` | Dataset downloader | Procedural synthetic SVG generator disguised as CubiCasa5K | **High** | Audited |
| `preprocessing/inspect_dataset.py` | Dataset inspection | Correctly audits SVG files present in directory | **Clean** | Audited |
| `preprocessing/convert_annotations.py`| Annotation converter | Good SVG parser and CLAHE normalization | **Clean** | Audited |
| `preprocessing/split_dataset.py` | Split manager | Deterministic splitting works, but $N=3$ test size is tiny | **High** | Audited |
| `inference/predict.py` | Prediction pipeline | **Directly copies ground-truth JSON into predictions** | **Critical** | Audited |
| `inference/postprocess.py` | Geometric snapping | Over-aggressive orthogonal snapping and portal projection | **Medium** | Audited |
| `evaluation/metrics.py` | Metric suite | Greedy bipartite matching implemented; needs Hungarian assignment | **Medium** | Audited |
| `evaluation/evaluate.py` | Evaluation runner | **Fabricated baseline via noisy ground-truth perturbation** | **Critical** | Audited |
| `schemas/floorplan_schema.py` | Pydantic schema | Robust validation and Shoelace formula implementation | **Clean** | Audited |
| `config.yaml` | Configuration | Mixed precision and quantization flag inconsistencies | **Medium** | Audited |
| `README.md` | Documentation | Contains unsupported claims regarding 100% test performance and local VLM tuning | **Critical** | Audited |

---

## 4. Phase-by-Phase Remediation Roadmap

1. **Phase 2 (Dataset Authenticity)**: Audit current images, document provenance honestly, and acquire/integrate genuine CubiCasa5K floor plans.
2. **Phase 3 (Data Split & Leakage)**: Re-split dataset, verify zero cross-contamination, separate synthetic fixtures from real benchmarks, and acknowledge sample size bounds.
3. **Phase 4 & 5 (Model Training)**: Correct training implementation, clearly document hardware-dependent execution paths (Google Colab T4 GPU for genuine Qwen2.5-VL QLoRA vs. CPU validation mode), and log verifiable training histories.
4. **Phase 6 & 7 (Inference & Postprocessing)**: Eliminate the ground-truth shortcut in `predict.py` completely. Implement genuine vision-based inference. Refine geometric postprocessing with diagnostic logging and coordinate preservation.
5. **Phase 8 & 9 (Evaluation & Generalization)**: Implement genuine evaluation comparing real predictions against ground truth without shortcuts. Implement Hungarian matching. Replace fabricated baseline with actual baseline.
6. **Phase 10–15 (Verification & Presentation)**: Update Pydantic schema tests, update Streamlit UI, author new unit tests, revise `README.md` and PowerPoint slides with true metrics.
