# SUBMISSION_STATUS: FloorPlan-AI

**Read this before anything else.**

> **Model training and evaluation were NOT executed.** No model has been fine-tuned, no checkpoint or weights exist, and there are no
> evaluation metrics, prediction samples or result figures anywhere in this submission. What is delivered is the complete, unit-tested
> **code and Colab procedure** for a genuine fine-tuning experiment, plus honest documentation of its state. The author's development
> environment had no GPU, no PyTorch and no network route to the dataset host (Zenodo), and the Colab run was not carried out before this package was made.

## 1. The task and the chosen approach

Assessment: fine-tune a pretrained model to detect and classify walls, doors, windows and rooms in 2D floor plans and output structured JSON.

Chosen approach (changed from the original Qwen2.5-VL plan, reasons in `docs/training.md`): **5-class semantic segmentation** (Background, Room, Wall, Door, Window)
with a **U-Net whose encoder is an ImageNet-pretrained ResNet-18**, fine-tuned end to end on real **CubiCasa5K** plans (Colab T4), then polygons extracted from the
predicted mask into JSON in original image pixel coordinates (no confidence field, no invented entities).

## 2. Status table

| Item | Implemented | Actually executed | Result |
| :--- | :---: | :--- | :--- |
| Dataset preparation from official CubiCasa5K (`training/prepare_dataset.py`, `seg_common.py`, `config/segmentation.yaml`) | yes | unit tests on small hand-written SVG fixtures only. **Never run on the real dataset** | none |
| Real dataset download, MD5 check, SVG inspection, class-mapping verification, overlays | cells exist in the notebook | **no** | the mapping is taken from the official loader *code* and is **unverified on real files** |
| U-Net / ResNet-18 model, loss, training loop, checkpointing (`training/seg_model.py`, `train_segmentation.py`) | yes | **no**: the torch tests were skipped (no torch in the sandbox); nothing was trained | **no weights exist** |
| Inference checkpoint -> mask -> polygon JSON (`inference/predict_segmentation.py`) | yes | mask-to-JSON, schema validation and failure handling unit-tested; **never run with a trained model** | none |
| Evaluation on held-out test plans (`evaluation/evaluate_segmentation.py`) | yes | metric, leakage-guard and JSON-validity logic unit-tested on hand-made data; **never run on real predictions** | **no metrics** |
| Colab notebook (`notebooks/colab_segmentation.ipynb`) | yes | code cells syntax-checked; cell A5b exercised on a tiny throwaway dataset in the sandbox; **no cell has been run on Colab** | none |
| OpenCV baseline (`baselines/opencv_baseline.py`) | yes, a comparison point only, **not AI** | unit tests and a drawn-plan smoke test | **no benchmark on real plans** |
| Streamlit dashboard (`app.py`) | yes | exercised only against a stand-in for Streamlit; **never launched in real Streamlit** | none |
| Slide deck (`presentation/Floorplan_AI_Assessment.pptx`) | yes, white background, 16 slides | generated, rendered to images and inspected | states "not trained / not evaluated"; every result slide is read from result files that do not exist yet |
| Qwen2.5-VL LoRA draft (`training/train.py` and related) | early draft | **never executed** | abandoned for this submission |

## 3. Tests actually run (development sandbox, Python 3.13, no torch, no shapely)

```
python -m pytest tests -q -rs
65 passed, 4 skipped
```

The 4 **skipped** tests were **not run** and are not counted as passed: `tests/test_seg_torch.py` (whole file, needs torch/torchvision),
and one test each in `tests/test_evaluate_system.py`, `tests/test_evaluation.py`, `tests/test_geometry.py` (need shapely).
They are intended to run on Colab (notebook cell B1). The passing tests prove data-handling, geometry, JSON, metric and guard logic on
hand-made fixtures. They say nothing about model quality or about how the real dataset looks. The same result was re-checked inside the unpacked ZIP (see the final report).

## 4. What is verified and what is not

Verified: Zenodo record page metadata (file `cubicasa5k.zip`, 5.5 GB, md5 `0ce0b203d1e3c125b51087b219bd23b9`, DOI 10.5281/zenodo.2613548) and the two licence statements
(Zenodo: CC BY-NC-SA 4.0; official repository: CC BY-NC 4.0; the stricter reading, non-commercial and share-alike, is followed).

Not verified: the real SVG structure and zip layout; the source-to-target class mapping on real files; image/mask alignment (a person must look at the overlays; training refuses to start
without `overlays_confirmed.txt`); class balance; door and window annotation quality. **Do not assume doors or windows would work**: that is exactly what the unrun inspection and overlay steps decide.

## 5. What remains incomplete

1. Run `notebooks/colab_segmentation.ipynb` on a Colab T4 (steps and expected outputs in `COLAB.md`): download + checksum, inspect, prepare, overlays, class-coverage report.
2. A person checks the overlays, then sets `OVERLAYS_CONFIRMED = True`.
3. Run the torch tests, the overfit-on-8 sanity run, full training, and evaluation on the held-out test split.
4. Copy `results/segmentation/` from the run into the repository, regenerate the deck (`python presentation/generate_presentation.py`), and update the README from those files only.
5. Host the weights outside Git (GitHub Release or Drive) and add the link and sha256. **No weights exist and none are included.**

## 6. What this package contains

- Source: `training/`, `inference/`, `evaluation/`, `baselines/`, `preprocessing/`, `schemas/`, `app.py`
- Colab notebook: `notebooks/colab_segmentation.ipynb`; guide: `COLAB.md`
- Configuration: `config/segmentation.yaml` (learned pipeline), `config.yaml` (old Qwen draft), `requirements*.txt` (dashboard, dev, segmentation experiment, Qwen draft kept separate)
- Documentation: `README.md`, `docs/dataset.md`, `docs/training.md`, `docs/methodology.md`, `docs/challenges.md`, `dataset/README.md`, audit reports in `docs/audit/`
- Presentation: `presentation/Floorplan_AI_Assessment.pptx` and its generator
- Tests: `tests/`
- Sample outputs and existing artifacts: **none exist** (`samples/`, `outputs/`, `artifacts/` contain only `.gitkeep` placeholders)

**Deliberately left out of the ZIP** (still in the Git history of the branch): the quarantined files under `docs/audit/invalid_artifacts/` that are synthetic data or withdrawn numbers
(`opencv_baseline_on_synthetic_fixtures/`, `ground_truth_copies_as_samples/`, `synthetic_fixture_dataset/`, and the loss history and plot of `toy_surrogate_training/`).
Kept: that folder's README and `toy_surrogate_training/adapter_config.json` (a config with no metrics, read by one regression test).
An earlier version of this project claimed a fine-tuned model that did not exist; those claims are withdrawn, as the audit reports in `docs/audit/` describe.

## 7. Repository

https://github.com/sandhiya-1234/FloorPlan-AI, branch `repair/real-vlm-pipeline`. `main` was not modified.
