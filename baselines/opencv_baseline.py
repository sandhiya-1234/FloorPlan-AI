"""
Classical OpenCV BASELINE for floor-plan parsing.

    This is NOT a learned model and NOT Qwen2.5-VL. It exists only as a comparison point.

What it does: finds long straight dark lines (morphology + probabilistic Hough) and reports
them as walls, and reports large enclosed regions as rooms.

What it does not do (and therefore never reports): doors, windows, room types, wall thickness,
or confidence scores. When it finds nothing it returns empty lists. It never inserts default or
placeholder geometry (the previous implementation did; see docs/audit/).

Usage:
    python -m baselines.opencv_baseline --image-dir dataset/test --out-dir outputs/predictions/opencv_baseline
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.records import make_record, record_path, write_record  # noqa: E402
from schemas.floorplan_schema import FloorPlanSchema  # noqa: E402

logger = logging.getLogger(__name__)

SYSTEM_NAME = "opencv_baseline"


def detect_opencv_baseline(
    image_rgb: np.ndarray,
    max_walls: Optional[int] = None,
    min_room_area_px: float = 10000.0,
    max_room_area_frac: float = 0.9,
) -> Dict[str, Any]:
    """Classical detection from pixels only. Returns a schema-shaped dict with empty doors/windows."""
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    walls_mask = cv2.bitwise_or(
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h),
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v),
    )

    walls = []
    lines = cv2.HoughLinesP(walls_mask, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=15)
    if lines is not None:
        for line in lines:
            # OpenCV <5 returns (N,1,4), OpenCV 5 returns (N,4); flatten handles both
            x1, y1, x2, y2 = (float(v) for v in np.asarray(line).reshape(-1)[:4])
            if x1 == x2 and y1 == y2:
                continue
            walls.append({"id": f"wall_{len(walls) + 1}", "start": [x1, y1], "end": [x2, y2]})
            if max_walls is not None and len(walls) >= max_walls:
                break

    rooms = []
    contours, _ = cv2.findContours(cv2.bitwise_not(walls_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_room_area_px < area < w * h * max_room_area_frac:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) >= 3:
                rooms.append({
                    "id": f"room_{len(rooms) + 1}",
                    "type": "unknown",  # this baseline cannot classify rooms
                    "polygon": [[float(p[0][0]), float(p[0][1])] for p in approx],
                })

    return {
        "image_width": w,
        "image_height": h,
        "units": "pixels",
        "rooms": rooms,
        "walls": walls,
        "doors": [],      # not supported by this baseline
        "windows": [],    # not supported by this baseline
    }


def run_baseline_on_image(image_path: Path, apply_postprocess: bool = False) -> Dict[str, Any]:
    """Run the baseline on one image and return a prediction record (never raises per-image)."""
    image_path = Path(image_path)
    meta = {"method": "opencv_hough_morphology", "learned": False, "postprocess": apply_postprocess}
    try:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"cannot read image {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        raw = detect_opencv_baseline(img_rgb)
        raw["image"] = image_path.name
        schema_raw = FloorPlanSchema(**raw)
    except Exception as e:  # noqa: BLE001 - recorded as an explicit failure, not hidden
        return make_record(image=image_path.name, system=SYSTEM_NAME, status="failed",
                           failure_stage="generate", error=f"{type(e).__name__}: {e}", meta=meta)

    final, diagnostics = schema_raw, {}
    if apply_postprocess:
        from inference.postprocess import postprocess_floorplan_json  # lazy: needs shapely
        final, diagnostics = postprocess_floorplan_json(
            schema_raw.to_compact_dict(), schema_raw.image_width, schema_raw.image_height,
            return_diagnostics=True,
        )
    return make_record(
        image=image_path.name, system=SYSTEM_NAME, status="ok",
        prediction_raw=schema_raw.to_compact_dict(), prediction=final.to_compact_dict(),
        postprocess_diagnostics=diagnostics, meta=meta,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Run the OpenCV baseline (NOT an AI model) over a folder of images.")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out-dir", default=f"outputs/predictions/{SYSTEM_NAME}")
    p.add_argument("--postprocess", action="store_true", help="also apply geometric refinement (needs shapely)")
    args = p.parse_args()

    images = sorted(f for ext in ("*.png", "*.jpg", "*.jpeg") for f in Path(args.image_dir).glob(ext))
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")
    n_failed = 0
    for img in images:
        rec = run_baseline_on_image(img, apply_postprocess=args.postprocess)
        write_record(record_path(Path(args.out_dir), img.name), rec)
        n_failed += rec["status"] == "failed"
    logger.info("Wrote %d records to %s (%d failed)", len(images), args.out_dir, n_failed)


if __name__ == "__main__":
    main()
