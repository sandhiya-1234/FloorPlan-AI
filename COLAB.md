# Google Colab GPU guide (T4): U-Net / ResNet-18 segmentation

Training and model inference run on Colab because the development laptop has no NVIDIA GPU. The whole procedure is one notebook,
[`notebooks/colab_segmentation.ipynb`](notebooks/colab_segmentation.ipynb).

**Status: this notebook has never been run.** It was written in a sandbox with no GPU, no torch and no route to Zenodo. Its code cells were
syntax-checked and the Python modules it calls are covered by unit tests, but no cell has produced a real output yet. Treat the first run as
the first real test and send back any error text.

## Open it

Colab > File > Open notebook > GitHub tab > `sandhiya-1234/FloorPlan-AI`, branch `repair/real-vlm-pipeline`, `notebooks/colab_segmentation.ipynb`.
Then Runtime > Change runtime type > **T4 GPU**. (Free-tier T4 sessions can disconnect; the notebook copies stage outputs to Google Drive.)

## Stage A: data (about 5.5 GB download)

| Cell | What it does | You should see |
| :--- | :--- | :--- |
| A1 | clone the branch, `pip install -r requirements-seg.txt`, check the GPU | `CUDA: True`, the T4 name |
| A2 | mount Google Drive for backups | a Drive folder `floorplan_ai` |
| A3 | download `cubicasa5k.zip` from Zenodo record 2613548 and check the md5 | `checksum OK` (it stops otherwise) |
| A4 | `training.prepare_dataset inspect`: census of the real SVG selectors, room names, split lists | the selector counts; the cell fails if a selector never occurs |
| A5 | `training.prepare_dataset prepare`: seeded subset from the official splits, masks, duplicate removal, overlays, `manifest.json` | counts per split, rejections, duplicate drops |
| A6 | shows the overlay images | image on the left, ground-truth mask on the right |
| A7 | zips `inspect_report.txt`, `prepare_report.txt`, `manifest.json` and the overlays and downloads them | two zip downloads |
| A8 | the confirmation gate | creates `overlays_confirmed.txt` only if you set `OVERLAYS_CONFIRMED = True` |

**Stop after A7.** Look at the overlays yourself (walls on the drawn walls, doors and windows on the openings, rooms inside the walls). If you want
an independent look, attach the two zips to the chat; the inspect report and overlays are what decide whether the mapping in
`config/segmentation.yaml` is right. Only then set `OVERLAYS_CONFIRMED = True` and run A8.

## Stage B: model (only after the gate)

| Cell | What it does |
| :--- | :--- |
| B1 | `pytest tests`: the whole suite, including the torch tests and the Shapely tests that could not run in the sandbox. Fix failures before training |
| B2 | `--overfit 8` sanity run: trains and validates on the same 8 real images. It shows that the code can learn; it is labelled `overfit_sanity` and evaluation refuses its checkpoint |
| B3 | full training: ImageNet-pretrained ResNet-18 encoder + U-Net decoder, AdamW, CE + Dice, fp16 autocast, 512x512, batch 4, best checkpoint by validation mean IoU |
| B4 | training curves read from `history.json` |
| B5 | evaluation on the held-out test split; aborts on any id overlap with training or validation |
| B6 | image / ground truth / prediction panels |
| B7 | zips `results/segmentation/` (small, Git-safe) and copies `best.pt` to Drive; prints the checkpoint sha256 |

## Where things end up

| Artifact | Location | In Git? |
| :--- | :--- | :---: |
| prepared data and overlays | `data/seg/` (Colab disk) | no |
| checkpoint `best.pt`, `history.json`, `config.json`, `train_summary.json` | `runs/seg/unet_resnet18/` (Colab disk), copy on Drive | **no** (weights stay out of Git) |
| metrics, confusion matrix, per-image table, prediction JSON, visual panels | `results/segmentation/` | yes, after verification |

The weights are small (an estimate from the parameter count, roughly 14 million parameters or about 55-60 MB in fp32; the real size is recorded in `train_summary.json`). They are not committed. After a real run they should be uploaded as a GitHub Release asset or a Drive
link with the sha256 printed by B7; that link and hash then go into the README. Until then no download link exists.

## Run inference on your own plans (after training)

```bash
python inference/predict_segmentation.py --checkpoint runs/seg/unet_resnet18/best.pt \
    --image-dir <folder of png/jpg plans> --out-dir outputs/segmentation_predictions --save-masks
```

It fails loudly if the checkpoint is missing or does not match the architecture. It never falls back to the OpenCV baseline.

## If something fails

Copy the full cell output. Typical, expected stops: md5 mismatch (incomplete download), `inspect` exit 1 (a selector is absent from the real files, so the
mapping must be corrected, not worked around), `UnmappedCategory` / rejections in `prepare` (a room name missing from the config), or the gate refusing to train.
