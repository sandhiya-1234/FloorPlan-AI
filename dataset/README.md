# FloorPlan-AI Dataset Directory

This directory stores raw architectural floor-plan data, preprocessed intermediate representations, and partitioned train/validation/test splits.

## Directory Structure

```
dataset/
├── raw/             # Raw downloaded floor plans and annotations (e.g., CubiCasa5K SVG/PNG files)
├── processed/       # Converted standardized JSON annotations and normalized images
├── train/           # Training split (image + json pairs)
├── validation/      # Validation split (image + json pairs)
└── test/            # Held-out unseen test split for evaluation
```

## Dataset: CubiCasa5K

- **Source**: [CubiCasa5K on Hugging Face](https://huggingface.co/datasets/CubiCasa/CubiCasa5K) / [GitHub](https://github.com/CubiCasa/CubiCasa5K)
- **Scale**: 5,000 multi-room architectural floor-plan images with dense vector polygon annotations.
- **License**: CC BY-NC-SA 4.0 (Non-commercial research and educational use).
- **Extracted Elements**:
  - **Walls**: Wall segments represented by line endpoints `[start_x, start_y]` to `[end_x, end_y]` and thickness.
  - **Doors**: Bounding boxes `[x_min, y_min, x_max, y_max]`, midpoint position, and orientation (`horizontal`, `vertical`, `slanted`).
  - **Windows**: Bounding boxes, midpoint position, and orientation.
  - **Rooms**: Multi-point polygons `[[x1, y1], [x2, y2], ...]`, room category (bedroom, bathroom, kitchen, living_room, etc.), and pixel area.

## Setup Instructions

Run the automated download and inspection script:
```bash
python preprocessing/download_dataset.py --samples 50
python preprocessing/inspect_dataset.py
python preprocessing/convert_annotations.py
python preprocessing/split_dataset.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```
