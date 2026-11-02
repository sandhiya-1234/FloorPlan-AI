"""
Annotation Converter & Image Preprocessor for FloorPlan-AI.

STATUS: written for the synthetic SVG fixtures, NOT verified against real CubiCasa5K files. Do not use
it to build a training set until Phase 2 has inspected the real SVG structure and proven with image
overlays that converted annotations line up with the images.

Converts CubiCasa5K raw SVG vector annotations into standardized JSON according
to FloorPlanSchema. Handles resolution normalization, aspect-ratio preservation,
coordinate scaling, contrast enhancement, and Pydantic validation.
"""

import os
import sys
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.floorplan_schema import (
    FloorPlanSchema, RoomElement, WallElement, DoorElement, WindowElement, OrientationEnum
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOM_CLASS_MAPPING = {
    "livingroom": "living_room",
    "living": "living_room",
    "bedroom": "bedroom",
    "bed": "bedroom",
    "bathroom": "bathroom",
    "bath": "bathroom",
    "wc": "bathroom",
    "kitchen": "kitchen",
    "dining": "dining_room",
    "diningroom": "dining_room",
    "corridor": "corridor",
    "hall": "corridor",
    "balcony": "balcony",
    "terrace": "balcony",
    "storage": "storage",
    "closet": "storage",
    "office": "office",
}


def normalize_room_type(raw_type: str) -> str:
    """Maps varied architectural room labels to standardized RoomTypeEnum."""
    cleaned = re.sub(r"[^a-zA-Z]", "", raw_type).lower()
    for pattern, std_type in ROOM_CLASS_MAPPING.items():
        if pattern in cleaned:
            return std_type
    return "undivided"


def parse_svg_points(points_str: str) -> List[List[float]]:
    """Parses SVG points attribute string 'x1,y1 x2,y2' into [[x1, y1], [x2, y2], ...]."""
    pts = []
    # Handle both space and comma separation
    tokens = re.split(r"[\s,]+", points_str.strip())
    tokens = [t for t in tokens if t]
    for i in range(0, len(tokens) - 1, 2):
        try:
            x = float(tokens[i])
            y = float(tokens[i + 1])
            pts.append([x, y])
        except ValueError:
            continue
    return pts


def parse_cubicasa_svg(svg_path: Path) -> Dict[str, Any]:
    """
    Parses a CubiCasa5K model.svg into raw architectural primitives.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    rooms = []
    walls = []
    doors = []
    windows = []

    # Get root SVG dimensions
    svg_w = 1024.0
    svg_h = 768.0
    viewbox = root.attrib.get("viewBox", "")
    if viewbox:
        parts = [float(p) for p in viewbox.split()]
        if len(parts) == 4:
            svg_w, svg_h = parts[2], parts[3]
    else:
        if "width" in root.attrib and "height" in root.attrib:
            try:
                svg_w = float(re.sub(r"[^\d.]", "", root.attrib["width"]))
                svg_h = float(re.sub(r"[^\d.]", "", root.attrib["height"]))
            except ValueError:
                pass

    # Extract spaces / rooms
    room_idx = 1
    for space_elem in root.iter():
        elem_class = space_elem.attrib.get("class", "")
        elem_id = space_elem.attrib.get("id", "")
        # Find polygon within space group
        if space_elem.tag.endswith("polygon"):
            pts_str = space_elem.attrib.get("points", "")
            if pts_str:
                pts = parse_svg_points(pts_str)
                if len(pts) >= 3:
                    parent_class = space_elem.attrib.get("class", elem_class)
                    room_type = normalize_room_type(parent_class or elem_id)
                    rooms.append({
                        "id": f"room_{room_idx}",
                        "type": room_type,
                        "polygon": pts
                    })
                    room_idx += 1

    # Extract walls
    wall_idx = 1
    for wall_elem in root.iter():
        tag = wall_elem.tag.split("}")[-1]
        elem_id = wall_elem.attrib.get("id", "")
        if tag == "line" or (tag in ["line", "polyline"] and "wall" in elem_id.lower()):
            try:
                x1 = float(wall_elem.attrib.get("x1", 0))
                y1 = float(wall_elem.attrib.get("y1", 0))
                x2 = float(wall_elem.attrib.get("x2", 0))
                y2 = float(wall_elem.attrib.get("y2", 0))
                thick = float(wall_elem.attrib.get("stroke-width", 10))
                if (x1 != x2 or y1 != y2):
                    walls.append({
                        "id": f"wall_{wall_idx}",
                        "start": [x1, y1],
                        "end": [x2, y2],
                        "thickness": thick
                    })
                    wall_idx += 1
            except (ValueError, TypeError):
                continue

    # Extract doors & windows from rect elements
    door_idx = 1
    win_idx = 1
    for rect in root.iter():
        tag = rect.tag.split("}")[-1]
        if tag == "rect":
            r_id = rect.attrib.get("id", "").lower()
            r_class = rect.attrib.get("class", "").lower()
            try:
                rx = float(rect.attrib.get("x", 0))
                ry = float(rect.attrib.get("y", 0))
                rw = float(rect.attrib.get("width", 0))
                rh = float(rect.attrib.get("height", 0))
                if rw <= 0 or rh <= 0:
                    continue

                bbox = [rx, ry, rx + rw, ry + rh]
                orient = "horizontal" if rw >= rh else "vertical"

                if "door" in r_id or "door" in r_class:
                    doors.append({
                        "id": f"door_{door_idx}",
                        "bbox": bbox,
                        "position": [round(rx + rw / 2.0, 2), round(ry + rh / 2.0, 2)],
                        "orientation": orient
                    })
                    door_idx += 1
                elif "window" in r_id or "window" in r_class:
                    windows.append({
                        "id": f"window_{win_idx}",
                        "bbox": bbox,
                        "position": [round(rx + rw / 2.0, 2), round(ry + rh / 2.0, 2)],
                        "orientation": orient
                    })
                    win_idx += 1
            except (ValueError, TypeError):
                continue

    return {
        "svg_width": svg_w,
        "svg_height": svg_h,
        "rooms": rooms,
        "walls": walls,
        "doors": doors,
        "windows": windows
    }


def preprocess_image(
    image_path: Path,
    target_size: Optional[Tuple[int, int]] = (1024, 1024),
    enhance_contrast: bool = True
) -> Tuple[np.ndarray, float, float]:
    """
    Loads, cleans, and resizes floor plan image while tracking scaling factors.
    Returns (processed_image, scale_x, scale_y).
    """
    # Load using OpenCV
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    if enhance_contrast:
        # Convert to LAB for luminance-channel CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    if target_size is not None:
        target_w, target_h = target_size
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)
        return resized, scale_x, scale_y
    else:
        return img, 1.0, 1.0


def convert_dataset_sample(
    raw_sample_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (1024, 1024)
) -> Optional[FloorPlanSchema]:
    """
    Converts one sample folder (containing F1_scaled.png & model.svg) into
    standardized processed PNG and validated FloorPlanSchema JSON.
    """
    img_candidates = list(raw_sample_dir.glob("*.png")) + list(raw_sample_dir.glob("*.jpg"))
    svg_candidates = list(raw_sample_dir.glob("*.svg"))

    if not img_candidates or not svg_candidates:
        logger.warning(f"Skipping {raw_sample_dir}: Missing image or SVG annotation.")
        return None

    img_path = img_candidates[0]
    svg_path = svg_candidates[0]
    sample_id = raw_sample_dir.name

    # Preprocess image
    processed_img, scale_x, scale_y = preprocess_image(img_path, target_size=target_size)
    new_h, new_w = processed_img.shape[:2]

    # Parse SVG primitives
    raw_data = parse_cubicasa_svg(svg_path)

    # Scale coordinates
    scaled_rooms = []
    for r in raw_data["rooms"]:
        scaled_poly = [[round(pt[0] * scale_x, 2), round(pt[1] * scale_y, 2)] for pt in r["polygon"]]
        scaled_rooms.append(RoomElement(id=r["id"], type=r["type"], polygon=scaled_poly))

    scaled_walls = []
    for w in raw_data["walls"]:
        scaled_start = [round(w["start"][0] * scale_x, 2), round(w["start"][1] * scale_y, 2)]
        scaled_end = [round(w["end"][0] * scale_x, 2), round(w["end"][1] * scale_y, 2)]
        scaled_walls.append(WallElement(id=w["id"], start=scaled_start, end=scaled_end, thickness=w["thickness"]))

    scaled_doors = []
    for d in raw_data["doors"]:
        sb = [
            round(d["bbox"][0] * scale_x, 2),
            round(d["bbox"][1] * scale_y, 2),
            round(d["bbox"][2] * scale_x, 2),
            round(d["bbox"][3] * scale_y, 2)
        ]
        sp = [round(d["position"][0] * scale_x, 2), round(d["position"][1] * scale_y, 2)]
        orient = OrientationEnum(d["orientation"]) if d["orientation"] in [e.value for e in OrientationEnum] else OrientationEnum.HORIZONTAL
        scaled_doors.append(DoorElement(id=d["id"], bbox=sb, position=sp, orientation=orient))

    scaled_windows = []
    for win in raw_data["windows"]:
        sb = [
            round(win["bbox"][0] * scale_x, 2),
            round(win["bbox"][1] * scale_y, 2),
            round(win["bbox"][2] * scale_x, 2),
            round(win["bbox"][3] * scale_y, 2)
        ]
        sp = [round(win["position"][0] * scale_x, 2), round(win["position"][1] * scale_y, 2)]
        orient = OrientationEnum(win["orientation"]) if win["orientation"] in [e.value for e in OrientationEnum] else OrientationEnum.HORIZONTAL
        scaled_windows.append(WindowElement(id=win["id"], bbox=sb, position=sp, orientation=orient))

    # Build schema
    target_img_filename = f"{sample_id}.png"
    schema = FloorPlanSchema(
        image=target_img_filename,
        image_width=new_w,
        image_height=new_h,
        units="pixels",
        rooms=scaled_rooms,
        walls=scaled_walls,
        doors=scaled_doors,
        windows=scaled_windows
    )

    # Save output files
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = output_dir / target_img_filename
    out_json_path = output_dir / f"{sample_id}.json"

    # Save RGB image
    bgr_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_img_path), bgr_img)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(schema.to_compact_dict(), f, indent=2)

    return schema


def convert_all(raw_dir: Path, processed_dir: Path, target_size: Tuple[int, int] = (1024, 1024)):
    """Converts all raw floor plans to processed dataset directory."""
    logger.info(f"Converting raw samples from {raw_dir} to {processed_dir}...")
    sample_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    count = 0
    for s_dir in sample_dirs:
        res = convert_dataset_sample(s_dir, processed_dir, target_size=target_size)
        if res:
            count += 1
    logger.info(f"Successfully converted {count} floor plan samples.")


if __name__ == "__main__":
    convert_all(Path("dataset/raw"), Path("dataset/processed"))
