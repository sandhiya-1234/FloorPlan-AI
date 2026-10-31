"""
Annotation Visualizer for FloorPlan-AI.

Renders ground-truth or detected architectural elements (walls, doors, windows, rooms)
over floor-plan images with class-specific color coding and semi-transparent room fills.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.floorplan_schema import FloorPlanSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Class color schemes (BGR for OpenCV, RGB for PIL)
COLORS_BGR = {
    "wall": (30, 30, 30),         # Dark charcoal
    "door": (0, 140, 255),        # Vibrant Orange
    "window": (220, 180, 0),      # Cyan / Sky Blue
    "room_fill": {
        "living_room": (240, 215, 230),  # Soft lavender
        "bedroom": (255, 220, 200),      # Soft light blue
        "kitchen": (190, 245, 255),      # Soft yellow
        "bathroom": (210, 255, 220),     # Soft mint green
        "corridor": (230, 230, 230),     # Light grey
        "balcony": (210, 235, 255),      # Peach
        "storage": (220, 220, 220),      # Neutral
        "office": (245, 230, 210),       # Pale blue
        "undivided": (235, 235, 235),
        "other": (230, 230, 230),
    }
}


def draw_floorplan_annotations(
    image: np.ndarray,
    schema: FloorPlanSchema,
    alpha: float = 0.35,
    show_labels: bool = True
) -> np.ndarray:
    """
    Overlays architectural elements from FloorPlanSchema onto floor plan image.
    """
    overlay = image.copy()
    h, w = image.shape[:2]

    # 1. Draw Room Polygons (Semi-transparent fills)
    for room in schema.rooms:
        if len(room.polygon) >= 3:
            pts = np.array(room.polygon, dtype=np.int32).reshape((-1, 1, 2))
            rtype = room.type.lower()
            fill_color = COLORS_BGR["room_fill"].get(rtype, (220, 220, 220))
            cv2.fillPoly(overlay, [pts], fill_color)

    # Blend room fills
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # 2. Draw Room Outlines and Centroid Labels
    for room in schema.rooms:
        if len(room.polygon) >= 3:
            pts = np.array(room.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(120, 120, 120), thickness=2)

            if show_labels:
                # Centroid calculation
                pts_arr = np.array(room.polygon, dtype=np.float32)
                cx = int(np.mean(pts_arr[:, 0]))
                cy = int(np.mean(pts_arr[:, 1]))
                label = f"{room.type.upper()}"
                cv2.putText(image, label, (max(10, cx - 40), cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2, cv2.LINE_AA)

    # 3. Draw Walls
    for wall in schema.walls:
        pt1 = (int(wall.start[0]), int(wall.start[1]))
        pt2 = (int(wall.end[0]), int(wall.end[1]))
        thick = max(2, int(wall.thickness) if wall.thickness else 6)
        cv2.line(image, pt1, pt2, COLORS_BGR["wall"], thick)

    # 4. Draw Doors
    for door in schema.doors:
        xmin, ymin, xmax, ymax = [int(v) for v in door.bbox]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLORS_BGR["door"], 3)
        if show_labels and door.position:
            px, py = int(door.position[0]), int(door.position[1])
            cv2.circle(image, (px, py), 4, COLORS_BGR["door"], -1)
            cv2.putText(image, "D", (px - 5, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS_BGR["door"], 2, cv2.LINE_AA)

    # 5. Draw Windows
    for win in schema.windows:
        xmin, ymin, xmax, ymax = [int(v) for v in win.bbox]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLORS_BGR["window"], 3)
        if show_labels and win.position:
            px, py = int(win.position[0]), int(win.position[1])
            cv2.circle(image, (px, py), 4, COLORS_BGR["window"], -1)
            cv2.putText(image, "W", (px - 5, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS_BGR["window"], 2, cv2.LINE_AA)

    # 6. Add Architectural Legend
    legend_y = 30
    cv2.putText(image, "LEGEND:", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # Wall
    cv2.line(image, (110, legend_y - 5), (140, legend_y - 5), COLORS_BGR["wall"], 4)
    cv2.putText(image, "Wall", (150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # Door
    cv2.rectangle(image, (210, legend_y - 12), (230, legend_y + 2), COLORS_BGR["door"], -1)
    cv2.putText(image, "Door", (240, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # Window
    cv2.rectangle(image, (300, legend_y - 12), (320, legend_y + 2), COLORS_BGR["window"], -1)
    cv2.putText(image, "Window", (330, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def visualize_sample(
    image_path: Path,
    json_path: Path,
    output_path: Path
):
    """Visualizes an image + JSON pair and saves rendered PNG."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    schema = FloorPlanSchema(**data)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Failed to load image {image_path}")
        return

    # Resize image if dimensions mismatch schema
    if img.shape[1] != schema.image_width or img.shape[0] != schema.image_height:
        img = cv2.resize(img, (schema.image_width, schema.image_height))

    rendered = draw_floorplan_annotations(img, schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rendered)
    logger.info(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize floor plan annotations.")
    parser.add_argument("--image", type=str, required=True, help="Floor plan image path")
    parser.add_argument("--json", type=str, required=True, help="Annotation JSON path")
    parser.add_argument("--output", type=str, default="outputs/visualizations/annotation_vis.png", help="Output PNG path")
    args = parser.parse_args()

    visualize_sample(Path(args.image), Path(args.json), Path(args.output))


if __name__ == "__main__":
    main()
