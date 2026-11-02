# FloorPlan-AI: Dataset Integrity & Split Audit Report

**Auditor**: Senior Machine Learning Engineer, Computer Vision Researcher, ML Project Auditor  
**Date**: October 31, 2026  
**Scope**: Verification of dataset authenticity, annotation counts, coordinate transformations, data leakage, and split partitioning.

---

## 1. Annotation Count Verification: Reported vs. Actual Ground Truth

The initial dataset inspection summary reported:
- Walls: 180
- Doors: 60
- Windows: 75
- Rooms: 60

An independent re-parsing of all 15 XML/SVG vector annotation files (`dataset/raw/sample_*/model.svg`) yielded the **actual physical element counts**:

| Architectural Element | Reported in Prior Inspection | Actual Ground Truth Count | Discrepancy & Root Cause |
| :--- | :---: | :---: | :--- |
| **Walls** | 180 | **90** | **+100% overcount**. `inspect_dataset.py` counted the `<g id="Wall">` container's children ($+6$), and then subsequently visited each child `<line>` ($+6$), reporting 12 walls per plan instead of 6 ($15 \times 6 = 90$). |
| **Doors** | 60 | **45** | **+33.3% overcount**. `inspect_dataset.py` matched `"door" in elem_id.lower()` on the parent `<g id="Door">` group ($+1$), plus the 3 child `<rect>` tags ($+3$), reporting 4 doors per plan instead of 3 ($15 \times 3 = 45$). |
| **Windows** | 75 | **60** | **+25% overcount**. `inspect_dataset.py` matched `"window" in elem_id.lower()` on the parent `<g id="Window">` group ($+1$), plus the 4 child `<rect>` tags ($+4$), reporting 5 windows per plan instead of 4 ($15 \times 4 = 60$). |
| **Rooms** | 60 | **60** | **Accurate**. Exactly 4 room `<polygon>` tags per floor plan ($15 \times 4 = 60$). |

---

## 2. Dataset Authenticity & Provenance Audit

### Verdict: 100% Synthetic Procedural Generation
Inspection of [`preprocessing/download_dataset.py`](file:///d:/floorplan-ai/preprocessing/download_dataset.py) reveals that **no floor plans were downloaded from the CubiCasa5K repository**. 

Instead, the dataset was procedurally generated using an SVG string template (`SAMPLE_CUBICASA_SVG`) inside `download_or_generate_dataset()`:
```python
split_x = int(450 + (idx % 3) * 50)
split_y = int(380 + (idx % 2) * 60)
```
- **Geometry**: Every sample is a single 1024×768 outer rectangle partitioned into 4 quadrants by one vertical wall and one horizontal wall.
- **Topology**: Every floor plan contains exactly the same 4 rooms: `LivingRoom` (top-left), `Kitchen` (top-right), `Bedroom` (bottom-left), and `Bathroom` (bottom-right).
- **Portals**: Every sample contains 3 interior doors along the partition walls and 4 windows along the exterior perimeter.
- **Visuals**: Rendered onto a blank canvas with synthetic grid lines and computer-generated text via OpenCV.

### Ethical & Assessment Compliance Finding:
The project documentation, README, and presentation slides previously stated:
> *"Dataset: CubiCasa5K (5,000 multi-room architectural floor-plan images under CC BY-NC-SA 4.0)"*

This claim is scientifically invalid. These samples are **synthetic procedural validation fixtures**, NOT authentic CubiCasa5K floor plans. 
Under Rule 8 and Rule 9 of the assessment guidelines, synthetic data must be explicitly labeled as such and cannot masquerade as the authentic benchmark.

---

## 3. Data Split & Leakage Audit

The 15 samples were split as follows:
- **Train Split**: 10 samples (`sample_0003` to `sample_0010`, `sample_0012`, `sample_0013`)
- **Validation Split**: 2 samples (`sample_0014`, `sample_0015`)
- **Test Split**: 3 samples (`sample_0001`, `sample_0002`, `sample_0011`)
- **Partition Mode**: By unique sample directory name with random seed 42.

### Leakage Findings:
1. **File-Level Separation**: There is no direct file-level leakage; no image file appears in more than one split folder.
2. **Structural Near-Duplication**: Because all 15 images were generated from the same 4-room template with only 3 discrete variations of `split_x` ($450, 500, 550$) and 2 variations of `split_y` ($380, 440$), `sample_0001` and `sample_0007` have mathematically identical room geometries. The test split is structurally near-identical to the training split.
3. **Statistical Significance**: A held-out test set of $N=3$ samples is completely inadequate for measuring generalization. Any standard error calculation on $N=3$ yields confidence intervals spanning $[0.0, 1.0]$.

---

## 4. Remediation Steps

1. **Fix `inspect_dataset.py`**: Fix the XML element visitor to count only terminal primitive tags (`<line>`, `<polyline>`, `<rect>`, `<polygon>`) and ignore container `<g>` elements.
2. **Transparent Documentation**: Clearly designate current local data as **Synthetic Procedural Benchmark Fixtures (v1-synthetic)** in all documentation, reports, and presentation slides.
3. **Dataset Ingestion Pipeline**: Provide automated tools to ingest authentic CubiCasa5K samples directly from open Hugging Face / GitHub sources for users with network access, while maintaining reproducible offline execution.
4. **Disclose Sample Size Constraints**: Explicitly state in the evaluation reports and slides that the current test set ($N=3$) serves strictly as an algorithmic pipeline verification suite.
