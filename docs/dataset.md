# Dataset Documentation: CubiCasa5K Architectural Corpus

## 1. Dataset Overview

- **Name**: CubiCasa5K
- **Authors**: CubiCasa Inc. / Kalervo et al.
- **Reference**: *CubiCasa5K: A Dataset for Architectural Plan Analysis and 3D Reconstruction*
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Scale**: 5,000 multi-room residential and commercial architectural floor-plan images.

## 2. Raw Structure

Each raw floor plan in CubiCasa5K is structured as an individual directory:
```
dataset/raw/<sample_id>/
├── F1_scaled.png       # Rasterized 2D floor-plan image
├── F1_original.png     # Full-resolution raw scan
└── model.svg           # High-precision vector graphics annotation
```

### SVG Primitives
- `<g id="Wall">`: Contains `<line>`, `<polyline>`, and `<path>` elements defining wall centerlines and thickness.
- `<g id="Door">`: Contains single-leaf and double-leaf door symbols with swing directions.
- `<g id="Window">`: Contains fenestration rects and line dividers along perimeter walls.
- `<g id="Space">`: Contains `<g class="RoomType">` with `<polygon points="...">` delineating closed room perimeters.

## 3. Standardization & Mapping

Our preprocessing pipeline (`preprocessing/convert_annotations.py`) maps the heterogeneous SVG elements into standardized schema categories:

| Source SVG Class | Target Standard Class | Subtype Preserved | Geometry Representation |
| :--- | :--- | :--- | :--- |
| `<g id="Wall">` | `wall` | None | Line segment `[start_x, start_y]` to `[end_x, end_y]` |
| `Door-single`, `Door-double` | `door` | Orientation | Bounding box `[xmin, ymin, xmax, ymax]` + midpoint `[x, y]` |
| `Window`, `Window-bay` | `window` | Orientation | Bounding box `[xmin, ymin, xmax, ymax]` + midpoint `[x, y]` |
| `Bedroom`, `MasterBedroom` | `room` | `bedroom` | Multi-point polygon `[[x1, y1], [x2, y2], ...]` |
| `Bathroom`, `WC`, `Shower` | `room` | `bathroom` | Multi-point polygon |
| `Kitchen`, `Kitchenette` | `room` | `kitchen` | Multi-point polygon |
| `LivingRoom`, `Living` | `room` | `living_room` | Multi-point polygon |
| `DiningRoom`, `Dining` | `room` | `dining_room` | Multi-point polygon |
| `Hall`, `Corridor`, `Entry` | `room` | `corridor` | Multi-point polygon |
| `Balcony`, `Terrace` | `room` | `balcony` | Multi-point polygon |
| `Storage`, `Closet` | `room` | `storage` | Multi-point polygon |

## 4. Train / Validation / Test Splitting

To eliminate data leakage, data partitioning is performed strictly at the **image level** (an entire floor plan image and all its sub-elements belong exclusively to one split).

- **Train Split**: 70% (10 floor plans in base experiment set)
- **Validation Split**: 15% (2 floor plans)
- **Test Split**: 15% (3 held-out unseen floor plans)
- **Deterministic Seed**: `seed = 42`
