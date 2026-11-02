# Dataset notes

**Status: no real dataset is in the repository yet.** The 15-image set that used to live in `dataset/` was
procedurally generated (one shared 4-room template, annotations not aligned with the images) and is archived
in `docs/audit/invalid_artifacts/synthetic_fixture_dataset/`. It is not CubiCasa5K and was never a valid
train/validation/test corpus.

## Target dataset: CubiCasa5K

- Source: CubiCasa5K, floor plans with vector annotations (Kalervo et al., 2019).
- Licence: CC BY-NC-SA 4.0 (non-commercial, share-alike). Check the licence terms on the download page before use;
  any adapter trained on it inherits the same restrictions.
- Plan: start with a small subset downloaded from the official archive, use the dataset's official train/val/test
  lists, and grow only after the pipeline is proven.

## What is not yet known

The real SVG structure has not been inspected. In particular, how walls, doors, windows and room classes are
encoded (element types, group names, class attributes, transforms) is **unverified**. An earlier version of this
page described a structure that had been inferred from the synthetic fixtures; that description has been removed.
`preprocessing/convert_annotations.py` is therefore marked unverified and must not be used for training data.

## Phase 2 acceptance criteria

1. A small tool prints the real structure of several SVGs (element and attribute inventory).
2. The converter is written against that structure and outputs the project schema in original-image pixels.
3. Overlays of converted walls, doors, windows and rooms drawn on the real images are inspected and match.
   **No training until this is true.**
4. Images are letterboxed to a 1008x1008 canvas with an exact, tested coordinate mapping (`preprocessing/letterbox.py`).
5. A split report shows no exact-duplicate, near-duplicate (perceptual hash) or near-identical annotations across
   train, validation and test. The split follows the official lists.
6. A `DATASET_MANIFEST.json` (counts, file hashes, split lists, licence) is written; the dashboard reads it.

## Target classes

walls (line segments), doors and windows (bounding boxes with orientation), rooms (polygons with a type such as
bedroom, bathroom, kitchen, living_room). The exact room-type mapping from CubiCasa5K classes will be fixed in Phase 2
once the real class names have been read from the data.
