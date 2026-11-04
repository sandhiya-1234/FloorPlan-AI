# FloorPlan-AI: 2D floor plans to structured JSON with a fine-tuned U-Net (ResNet-18)

**Goal:** fine-tune a pretrained network to detect and classify **walls, doors, windows and rooms** in 2D floor-plan images and emit structured
JSON (polygons in original image pixel coordinates, with entity ids and types) for downstream 2D-to-3D processing.

**Approach:** 5-class semantic segmentation (Background, Room, Wall, Door, Window) with a U-Net whose encoder is an ImageNet-pretrained ResNet-18,
fine-tuned end to end on real [CubiCasa5K](https://zenodo.org/records/2613548) plans on a Colab T4; polygons are then extracted from the predicted mask.
Why a segmentation network and not a vision-language model: [`docs/training.md`](docs/training.md).

## Current status (read this first)

> **The model has not been trained and nothing has been evaluated.** The training, inference and evaluation code is written and its
> hardware-independent parts are unit-tested, but the environment that produced it had no GPU, no PyTorch and no route to the dataset host.
> There are therefore **no weights, no metrics, no prediction samples and no result figures** in this repository. The first real run is
> [`notebooks/colab_segmentation.ipynb`](notebooks/colab_segmentation.ipynb), run on a Colab T4 by the repository owner.
> This README will be updated from the saved result files of that run, and only from them.

| Item | Implemented | Actually executed | Actually trained / evaluated | Not completed |
| :--- | :---: | :---: | :---: | :--- |
| Dataset access to official CubiCasa5K | notebook cell + md5 check | **no** | n/a | download and checksum on Colab; record-page metadata (file, size, md5, DOI, licence) was read |
| Real SVG structure inspection | `prepare_dataset.py inspect` | **no** | n/a | the class mapping comes from the official loader *code*, not from real files; `inspect` must confirm it |
| Mask generation, overlays, subset, official splits, duplicate removal | `prepare_dataset.py prepare` | unit tests on hand-written SVG fixtures only | n/a | real run and a person checking the overlays |
| U-Net + ResNet-18 model, CE+Dice loss, AdamW, AMP, checkpointing | `training/seg_model.py`, `train_segmentation.py` | torch tests written, **skipped here (no torch)** | **not trained** | everything: sanity run, full run, best checkpoint |
| Inference: checkpoint -> mask -> polygon JSON | `inference/predict_segmentation.py` | mask-to-JSON and schema validation unit-tested | **no model run** | run with a real checkpoint |
| Evaluation on held-out test plans | `evaluation/evaluate_segmentation.py` | metric and leakage logic unit-tested | **not evaluated** | IoU, Dice/F1, precision, recall, confusion matrix, JSON validity, inference time |
| Fine-tuned weights | n/a | n/a | **do not exist** | produced by notebook Stage B; to be hosted outside Git |
| OpenCV baseline (`baselines/opencv_baseline.py`) | yes, comparison only, **not AI** | runs end to end on a drawn plan | **no benchmark on real plans** | over-detects walls; never produces doors/windows/rooms types |
| Qwen2.5-VL LoRA plan (`training/train.py` etc.) | early draft | **never executed** | **not trained** | abandoned for this submission; see `docs/training.md` section B |
| Dashboard (`app.py`) | yes | run only against a stand-in for Streamlit | n/a | never launched in real Streamlit; shows no learned-model results |

## Pipeline

```
image
  -> letterbox to 512x512 (aspect ratio kept; padding is white / ignored in the labels)
  -> U-Net, ResNet-18 encoder (ImageNet weights, fine-tuned together with the decoder)
  -> logits -> crop padding -> resize to the original size -> argmax -> class mask
  -> connected regions per class -> Douglas-Peucker -> polygons (+ holes)
  -> JSON in original image pixels: rooms, walls, doors, windows  (schema-validated)
```

Output (`SegmentationPrediction` in `inference/predict_segmentation.py`): `image`, `image_width`, `image_height`, `units: "pixels"`, `generated_by`
(architecture, checkpoint sha256, epoch) and four lists, `rooms`, `walls`, `doors`, `windows`, whose entries have `id`, `polygon` (outer ring `[[x, y], ...]`),
`holes` and `area_px`. **There is no confidence field:** the network is not calibrated, and an invented score would be fabricated data.
Walls are region polygons, not centre-lines; all rooms share one class, so there are no room types; touching rooms with no wall pixel between them merge.

## Dataset

Official CubiCasa5K (Zenodo 2613548, DOI 10.5281/zenodo.2613548, 5.5 GB zip, md5 `0ce0b203d1e3c125b51087b219bd23b9`). Licence: **CC BY-NC-SA 4.0** on Zenodo (the official
repository says CC BY-NC 4.0); this project follows the stricter reading (non-commercial, share-alike), so weights trained on it are non-commercial too.
Subset: 1000 / 150 / 200 train / val / test plans drawn with a fixed seed from the **official** split lists; cross-split duplicates are removed by image hash, mask hash and perceptual hash;
the explicit source-to-target class mapping is in [`config/segmentation.yaml`](config/segmentation.yaml). Full provenance, mapping, preprocessing and the list of what is still
unverified: [`docs/dataset.md`](docs/dataset.md). The 105 GB LMDB of the official training code is not created.

Training will not start until `manifest.json` exists and a person has created `overlays_confirmed.txt` after looking at the image/mask overlays.

## How to run (Google Colab T4)

Open `notebooks/colab_segmentation.ipynb` (steps and expected outputs: [COLAB.md](COLAB.md)). In short:

```
Stage A  clone -> pip install -r requirements-seg.txt -> download + md5 -> inspect -> prepare -> overlays -> (you look) -> confirm
Stage B  pytest -> overfit-8 sanity -> full training -> evaluation on the test split -> package results
```

Command-line equivalents (GPU needed for training):

```bash
python -m training.prepare_dataset inspect --zip cubicasa5k.zip --n 60
python -m training.prepare_dataset prepare --zip cubicasa5k.zip --out data/seg
python training/train_segmentation.py --run-name unet_resnet18
python evaluation/evaluate_segmentation.py --checkpoint runs/seg/unet_resnet18/best.pt --out-dir results/segmentation
python inference/predict_segmentation.py --checkpoint runs/seg/unet_resnet18/best.pt --image-dir <plans> --out-dir outputs/segmentation_predictions --save-masks
```

### Evaluation protocol (implemented, not yet applied)

Held-out test split only (ids asserted disjoint from the checkpoint's training and validation ids; the run aborts otherwise). Pixel-level, at original image resolution, ignore-labelled pixels
excluded: per-class IoU, Dice/F1, precision and recall, the confusion matrix, mean IoU/Dice, pixel accuracy, JSON validity (serialise, re-parse, schema-validate; failures are counted), and inference time
(model forward + resize + argmax, and mask-to-JSON, excluding two warm-up images). A class absent from both ground truth and prediction reports `null`, not 0 or 1.
A note is attached to the results when the test set has fewer than 100 images. Raw `metrics.json`, `confusion_matrix.csv`, `per_image.csv`, prediction JSON files and image/ground-truth/prediction panels are written to `results/segmentation/`.

### Weights

Weights are never committed (`*.pt` is git-ignored). After a real run they will be uploaded outside Git (GitHub Release asset or Drive link) and the link and sha256 will be added here.
**No download link exists yet because no weights exist.**

## Repository layout

```
config/segmentation.yaml        class mapping, subset sizes, hyper-parameters (single source of truth)
training/
  seg_common.py                 SVG parsing, rasterisation, letterbox, hashing, metrics, overlays (numpy/cv2)
  prepare_dataset.py            inspect / prepare commands
  seg_model.py                  U-Net with ResNet-18 encoder (torch)
  train_segmentation.py         training loop, checkpoint, history (torch)
  train.py, model_loader.py, dataset.py, collator.py   early Qwen2.5-VL draft, never executed
inference/predict_segmentation.py   checkpoint -> mask -> polygon JSON
evaluation/evaluate_segmentation.py held-out evaluation
notebooks/colab_segmentation.ipynb  the runnable Colab procedure
baselines/opencv_baseline.py    classical comparison baseline (kept, separate from the learned pipeline)
app.py                          Streamlit dashboard
tests/                          pytest suite
docs/                           dataset, training, methodology; docs/audit/ holds the audit history
presentation/                   slide generator (white background; reads results only if they exist)
```

## Installation: separate environments

| File | Use |
| :--- | :--- |
| `requirements.txt` | dashboard / laptop, no GPU |
| `requirements-dev.txt` | tests, evaluation of the old schema, slide generation |
| `requirements-seg.txt` | the U-Net experiment on Colab (torch and torchvision come pre-installed there) |
| `requirements-train.txt` | the abandoned Qwen2.5-VL plan only |

The laptop has no NVIDIA GPU: do not train or run the model locally.

### Test status

```bash
python -m pytest tests -q -rs
```
Result in the authoring sandbox: **65 passed, 4 skipped**. Skipped is not passed: the 4 skipped tests (the whole torch test file, and three tests that need `shapely`)
**have not been run**; they are expected to run in Colab notebook cell B1. The passing tests prove the data, geometry, JSON and metric logic on hand-made fixtures; they say nothing
about model quality or about the real dataset.

## Limitations (known in advance)

- Nothing learned has been run: no training, no metrics, no sample predictions exist yet.
- The class mapping was derived from the official loader code. Until `inspect` and the overlay check pass on the real files, it is unverified.
- Small subset (about 1000 training plans) and a small test split; the evaluation attaches a limitation note when the test set is under 100 plans.
- Duplicate control uses hashes. It catches re-exports of the same plan but cannot prove that two different files do not show the same building.
- Semantic segmentation: touching instances merge; walls are regions, not centre-lines; no room types; no confidence values.
- Raster polygons are approximations (Douglas-Peucker); no orthogonal snapping or CAD clean-up is applied to the learned output.
- CubiCasa5K is non-commercial / share-alike; so is anything trained on it.
- The OpenCV baseline cannot detect doors or windows and does not classify rooms. That is by design.

## Integrity history

An earlier version of this project claimed a fine-tuned vision-language model but actually ran OpenCV heuristics, a toy surrogate network and synthetic data. Those artifacts and the audit
are quarantined in [`docs/audit/`](docs/audit/invalid_artifacts/README.md) and must not be cited.

## License and data

Code: see repository. Data: CubiCasa5K, CC BY-NC-SA 4.0 (Kalervo et al., SCIA 2019). Encoder initialisation: torchvision ResNet-18 ImageNet weights under torchvision's terms.
