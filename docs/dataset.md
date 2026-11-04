# Dataset: CubiCasa5K (official), provenance and preprocessing

**Status (read this first):** the dataset has **not been downloaded or inspected by the author of this code.** The environment
that wrote the pipeline could not reach Zenodo. Everything marked *verified* below comes from the Zenodo record page and the
official loader source code. Everything marked *unverified* is checked automatically by the first cells of
`notebooks/colab_segmentation.ipynb`, and training refuses to start until a person has looked at the overlays.

The 15-image synthetic set that used to live in `dataset/` is archived in `docs/audit/invalid_artifacts/` and is **not** used anywhere.

## 1. Provenance

| Item | Value | Status |
| :--- | :--- | :--- |
| Dataset | CubiCasa5K: 5000 annotated floor-plan images with vector (SVG) annotations | verified (record page) |
| Authors | Kalervo, Ylioinas, Häikiö, Karhu, Kannala. *CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis*, SCIA 2019 | verified (record page) |
| Download | https://zenodo.org/records/2613548, DOI 10.5281/zenodo.2613548, Version 1.0, published 2019-03-28 | verified |
| File | `cubicasa5k.zip`, one file, 5.5 GB | verified |
| Checksum | md5 `0ce0b203d1e3c125b51087b219bd23b9` (the notebook recomputes it and stops on a mismatch) | verified from the record page; recomputed on Colab |
| Official code | https://github.com/CubiCasa/CubiCasa5k (loader, split lists, class names) | read, not executed |
| Licence | Zenodo record: **CC BY-NC-SA 4.0**. The official repository's `LICENSE` file says **CC BY-NC 4.0**. The two disagree, so this project follows the stricter reading: attribution, **non-commercial**, **share-alike**. Any released weights are derived from this data and carry the same restrictions. | verified (both texts read) |

The 105 GB LMDB format used by the official training code is **not** created. The pipeline reads the zip directly and only
materialises the selected subset.

## 2. Task and target classes

Five-class semantic segmentation. The saved mask value of each class:

| Value | Class | Meaning |
| :---: | :--- | :--- |
| 0 | Background | everything not listed below, including paper margin |
| 1 | Room | any official `Space` polygon (see the mapping) |
| 2 | Wall | `Wall` polygons |
| 3 | Door | `Door` polygons |
| 4 | Window | `Window` polygons |
| 255 | ignored | excluded from the loss and from every metric |

The model predicts no room *types* (kitchen, bedroom, ...). All 59 official room names are folded into one class, Room, on purpose:
the assessment asks for walls, doors, windows and rooms, and a 5-class model trained on about a thousand plans cannot be expected to
learn 59 room names reliably.

## 3. Source-to-target mapping

The mapping lives in `config/segmentation.yaml` and is the single source of truth. It was written from the official loader code
(`floortrans/loaders/house.py`, `svg_utils.py`), **not** from inspecting real SVGs.

| Source selector in `model.svg` | Target | Notes |
| :--- | :--- | :--- |
| `<g id="Wall">` | Wall (2) | the first direct `<polygon points="...">` child; polygons with an extent under 4 px are dropped, as in the official loader |
| `<g id="Window">` | Window (4) | same geometry rule |
| `<g id="Door">` | Door (3) | same geometry rule |
| `<g class="Space <Name>">`, any of the 59 official names listed in the config | Room (1) | every name is listed explicitly |
| `Space Outdoor`, `Space Below150cm` | ignore (255) | not a room in the sense of the task, but also not background |
| `<g id="Railing">` | ignore (255) | balustrades are neither wall nor background; painted as ignore only on pixels that would otherwise be empty |
| a `Space` name that is **not** in the config | the sample is **rejected and counted** | never guessed, never silently mapped |

Paint order (later overrides earlier): outdoor_ignore, rooms, railing_ignore_on_empty, walls, windows, doors. Doors and windows
therefore win over the wall they cut through, and walls win over the room underneath.

`python -m training.prepare_dataset inspect` is the check that this table matches reality. It samples real files and prints the
selector census, every `Space` name seen, names that are not in the config, SVG-versus-PNG size mismatches and the split-list counts.
It exits non-zero if the wall, door, window or space selector never occurs in the real files. **If it does, stop: the mapping is wrong.**

## 4. Image pairing

The official loader pairs `model.svg` with `F1_scaled.png`. This pipeline uses the same pair, so the SVG polygon coordinates are in the
pixel grid of that PNG. The expected layout is `<category>/<number>/{F1_scaled.png, F1_original.png, model.svg}`; the real layout of the zip
is *unverified* and is discovered by `inspect` (the sample id is the last two path components).

## 5. Splits and duplicate control

- Splits come from the **official** `train.txt`, `val.txt`, `test.txt` inside the zip. Nothing is re-split.
- A seeded random subset is drawn from each official list: 1000 train, 150 val, 200 test (`config/segmentation.yaml`; fewer if samples are rejected).
- Duplicate control across splits: the splits are processed test, then val, then train. A val or train candidate is dropped when it
  matches a sample already accepted from an earlier split by exact image sha256, identical mask hash, or perceptual dHash with
  Hamming distance at most 3. Every drop is listed in `manifest.json` under `duplicate_drops`.
- `manifest.json` asserts that the id sets and the image-hash sets of the three splits are disjoint.
- Evaluation additionally aborts if any test id appears in the checkpoint's recorded training or validation ids.

dHash catches re-scans and re-exports of the same plan. It cannot prove that two *different* files are not the same building; that
limit is stated in the README.

## 6. Preprocessing

| Step | What happens |
| :--- | :--- |
| Read | the image and SVG are read straight from the zip; nothing is extracted wholesale |
| Rasterise | SVG polygons become a class mask at the image's own resolution, using the paint order above |
| Reject | samples with an unmapped room name, no wall pixels, no room pixels or an SVG parse error are rejected and counted, with the reason, in the manifest |
| Letterbox (training and inference) | the image is scaled to fit a 512x512 canvas with the aspect ratio kept; padding is white in the image and **255 (ignore)** in the mask |
| Augmentation (training only) | horizontal and vertical flips and 90-degree rotations |
| Normalisation | ImageNet mean and standard deviation, as the ResNet-18 encoder expects |

At inference the padding is cropped away and the logits are resized back to the original image size before the argmax, so the output
polygons are in the **original image's pixel coordinates**.

## 7. Overlays: the human gate

`prepare` writes `data/seg/overlays/<split>_<id>.png`: the image on the left, the ground-truth mask coloured on the right
(blue Room, red Wall, green Door, yellow Window, hatched grey ignored). A person must check that walls sit on the drawn walls, doors and
windows on the openings and rooms inside the walls. Only then does the notebook create `data/seg/overlays_confirmed.txt`.
`train_segmentation.py` refuses to run without `manifest.json` and that file.

## 8. What is verified today

| Claim | Status |
| :--- | :--- |
| Record, file size, checksum, DOI and the licence text on the Zenodo page | verified |
| The mapping mechanics (parser, rasteriser, paint order, ignore handling, dedupe, manifest) | unit-tested on small hand-written SVG fixtures that follow the documented selectors; this proves the code does what it says, **not** that the real files look like the fixtures |
| The real SVG structure, the zip layout, split-list contents | **unverified** (needs the Colab run) |
| Image/mask alignment on real plans | **unverified** (needs a person looking at the overlays) |
| Class balance, number of rejected samples, number of duplicate drops | **unknown** until `prepare` runs; they will be in `manifest.json` |
