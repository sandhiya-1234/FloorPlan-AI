"""
Dataset Inspector for FloorPlan-AI.

Audits raw architectural floor plan dataset, validating image formats,
resolutions, SVG annotation integrity, and class balance across floor plans.
"""

import os
import sys
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def inspect_svg(svg_path: Path) -> Dict[str, Any]:
    """Inspects a CubiCasa5K model.svg vector annotation file."""
    stats = {
        "walls": 0,
        "doors": 0,
        "windows": 0,
        "rooms": 0,
        "room_types": {}
    }
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Iterate all primitive elements (avoid double counting container groups)
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]  # strip XML namespace
            elem_id = elem.attrib.get("id", "")
            elem_class = elem.attrib.get("class", "")

            # Count walls (primitive lines/polylines only, ignore <g> containers)
            if tag in ["line", "polyline", "path"]:
                if "wall" in elem_id.lower() or "wall" in elem_class.lower() or not elem_id:
                    stats["walls"] += 1

            # Count doors (leaf rect or path elements only)
            if tag in ["rect", "path", "circle"] and ("door" in elem_id.lower() or "door" in elem_class.lower()):
                stats["doors"] += 1

            # Count windows (leaf rect or path elements only)
            if tag in ["rect", "path", "line"] and ("window" in elem_id.lower() or "window" in elem_class.lower()):
                stats["windows"] += 1

            # Count rooms (leaf polygons only)
            if tag == "polygon":
                stats["rooms"] += 1
                rtype = elem_class or "undivided"
                stats["room_types"][rtype] = stats["room_types"].get(rtype, 0) + 1

    except Exception as e:
        logger.warning(f"Error parsing SVG {svg_path}: {e}")

    return stats


def inspect_dataset(raw_dir: Path) -> Dict[str, Any]:
    """Inspects entire raw directory and aggregates dataset metrics."""
    logger.info(f"Inspecting raw dataset directory: {raw_dir}")

    report = {
        "dataset_path": str(raw_dir),
        "total_floorplans": 0,
        "image_resolutions": [],
        "total_walls": 0,
        "total_doors": 0,
        "total_windows": 0,
        "total_rooms": 0,
        "room_type_distribution": {},
        "valid_pairs": 0,
        "missing_images": 0,
        "missing_annotations": 0
    }

    if not raw_dir.exists():
        logger.error(f"Directory {raw_dir} does not exist.")
        return report

    sample_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    report["total_floorplans"] = len(sample_dirs)

    for s_dir in sample_dirs:
        # Check image
        img_candidates = list(s_dir.glob("*.png")) + list(s_dir.glob("*.jpg"))
        svg_candidates = list(s_dir.glob("*.svg"))

        if not img_candidates:
            report["missing_images"] += 1
            continue
        if not svg_candidates:
            report["missing_annotations"] += 1
            continue

        report["valid_pairs"] += 1
        img_path = img_candidates[0]
        svg_path = svg_candidates[0]

        try:
            with Image.open(img_path) as im:
                report["image_resolutions"].append(im.size)
        except Exception as e:
            logger.warning(f"Failed opening image {img_path}: {e}")

        svg_stats = inspect_svg(svg_path)
        report["total_walls"] += svg_stats["walls"]
        report["total_doors"] += svg_stats["doors"]
        report["total_windows"] += svg_stats["windows"]
        report["total_rooms"] += svg_stats["rooms"]

        for k, v in svg_stats["room_types"].items():
            report["room_type_distribution"][k] = report["room_type_distribution"].get(k, 0) + v

    logger.info(f"Inspection Complete: {report['valid_pairs']} valid pairs found.")
    logger.info(f"Walls: {report['total_walls']}, Doors: {report['total_doors']}, Windows: {report['total_windows']}, Rooms: {report['total_rooms']}")
    return report


if __name__ == "__main__":
    raw_path = Path("dataset/raw")
    results = inspect_dataset(raw_path)
    print("\n--- DATASET INSPECTION SUMMARY ---")
    print(json.dumps(results, indent=2))
