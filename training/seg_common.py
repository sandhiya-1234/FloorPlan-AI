"""
Shared, torch-free utilities for the learned segmentation experiment (U-Net, ResNet-18 encoder).

  * parse_svg / rasterize   CubiCasa5K model.svg  ->  5-class label mask (explicit, config-driven mapping)
  * letterbox_*             aspect-preserving square canvas for images and masks, with exact inverse
  * dhash / hamming         cross-split duplicate detection
  * overlay / side_by_side  visual verification images
  * confusion / metrics     per-class IoU, Dice/F1, precision, recall from a confusion matrix

The SVG selectors come from the OFFICIAL CubiCasa5k loader (floortrans/loaders/house.py, svg_utils.py)
and live in config/segmentation.yaml. Nothing here guesses a category: a room name that is not in the
config mapping raises UnmappedCategory and the sample is skipped and counted by the caller.

Licence note: the CubiCasa5K data is non-commercial (CC BY-NC 4.0 per the official repository LICENSE).
This file contains our own implementation of the documented selectors, not copied official code.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

IGNORE_INDEX = 255
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "segmentation.yaml"

# BGR overlay colours per class index (0 background is not drawn)
CLASS_COLOURS_RGB = {1: (66, 135, 245), 2: (230, 40, 40), 3: (40, 190, 60), 4: (250, 200, 20)}
IGNORE_COLOUR_RGB = (150, 150, 150)


class UnmappedCategory(ValueError):
    """A source category (e.g. a room name) has no entry in the explicit mapping."""

    def __init__(self, names: Sequence[str]):
        self.names = sorted(set(names))
        super().__init__(f"unmapped source categories: {self.names}")


# --------------------------------------------------------------------------------------- config

def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["class_names"] = list(cfg["classes"])
    cfg["n_classes"] = len(cfg["class_names"])
    return cfg


# ------------------------------------------------------------------------------------ SVG parse

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_polygon_points(g: ET.Element) -> Optional[np.ndarray]:
    """Points of the first DIRECT <polygon> child (the official rule), rounded like the official code."""
    for child in g:
        if _local(child.tag) == "polygon":
            pts = []
            for tok in (child.get("points") or "").split():
                if "," not in tok:
                    return None
                x, y = tok.split(",")[:2]
                pts.append((round(float(x)), round(float(y))))
            return np.array(pts, dtype=np.int32) if len(pts) >= 3 else None
    return None


def parse_svg(svg: bytes | str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walk every <g> in document order and collect polygons per source category.
    Returns {"walls","doors","windows","railings": [pts], "rooms": [(name, pts)], "stats": {...}}.
    """
    root = ET.fromstring(svg)
    sel = cfg["mapping"]["selectors"]
    ids = {k: sel[k]["value"] for k in ("wall", "door", "window", "railing")}
    space_token = sel["space"]["first_token"]
    name_idx = sel["space"]["name_token_index"]
    min_extent = cfg["mapping"]["min_wall_extent_px"]

    out: Dict[str, Any] = {"walls": [], "doors": [], "windows": [], "railings": [], "rooms": []}
    stats: Counter = Counter()
    for g in root.iter():
        if _local(g.tag) != "g":
            continue
        gid, gcls = g.get("id") or "", (g.get("class") or "").split()
        kind = next((k for k, v in ids.items() if gid == v), None)
        is_space = bool(gcls) and gcls[0] == space_token
        if kind is None and not is_space:
            continue
        pts = _first_polygon_points(g)
        if pts is None:
            stats["no_usable_polygon_child:" + (kind or "space")] += 1
            continue
        if kind == "wall" and (np.ptp(pts[:, 0]) < min_extent or np.ptp(pts[:, 1]) < min_extent):
            stats["small_wall_dropped"] += 1  # same rule as the official loader ("small wall")
            continue
        if kind:
            out[kind + "s"].append(pts)
            stats[kind] += 1
        else:
            name = gcls[name_idx] if len(gcls) > name_idx else "<missing>"
            out["rooms"].append((name, pts))
            stats["space"] += 1
    out["stats"] = dict(stats)
    return out


def room_name_lookup(cfg: Dict[str, Any]) -> Tuple[Dict[str, int], set]:
    to_room = {name: cfg["class_names"].index(cls) for name, cls in cfg["mapping"]["room_names"].items()}
    return to_room, set(cfg["mapping"]["ignore_room_names"])


def rasterize(parsed: Dict[str, Any], height: int, width: int, cfg: Dict[str, Any]) -> np.ndarray:
    """Paint polygons into a uint8 label mask using the explicit mapping and paint order from the config."""
    names = cfg["class_names"]
    c_room, c_wall, c_door, c_win = (names.index(n) for n in ("Room", "Wall", "Door", "Window"))
    to_room, ignore_names = room_name_lookup(cfg)
    unmapped = [n for n, _ in parsed["rooms"] if n not in to_room and n not in ignore_names]
    if unmapped:
        raise UnmappedCategory(unmapped)

    mask = np.zeros((height, width), np.uint8)

    def paint(polys: List[np.ndarray], value: int, only_where: Optional[Tuple[int, ...]] = None) -> None:
        if not polys:
            return
        layer = np.zeros((height, width), np.uint8)
        cv2.fillPoly(layer, [p.reshape(-1, 1, 2) for p in polys], 1)
        sel = layer == 1
        if only_where is not None:
            sel &= np.isin(mask, only_where)
        mask[sel] = value

    # order is fixed by config.mapping.paint_order and mirrored here; overlaps resolve by this priority
    paint([p for n, p in parsed["rooms"] if n in ignore_names], IGNORE_INDEX)          # outdoor_ignore
    paint([p for n, p in parsed["rooms"] if n in to_room], c_room)                      # rooms
    paint(parsed["railings"], IGNORE_INDEX, only_where=(0, IGNORE_INDEX))               # railing_ignore_on_empty
    paint(parsed["walls"], c_wall)                                                      # walls
    paint(parsed["windows"], c_win)                                                     # windows
    paint(parsed["doors"], c_door)                                                      # doors
    return mask


def svg_declared_size(svg: bytes | str) -> Dict[str, Optional[str]]:
    root = ET.fromstring(svg)
    return {"width": root.get("width"), "height": root.get("height"), "viewBox": root.get("viewBox")}


# ----------------------------------------------------------------------------------- letterbox

def letterbox_meta(orig_w: int, orig_h: int, size: int) -> Dict[str, Any]:
    scale = size / max(orig_w, orig_h)
    new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    return {"orig_w": orig_w, "orig_h": orig_h, "size": size, "scale": scale, "new_w": new_w, "new_h": new_h,
            "pad_x": (size - new_w) // 2, "pad_y": (size - new_h) // 2}


def letterbox_image(img: np.ndarray, size: int, fill: int = 255) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = img.shape[:2]
    m = letterbox_meta(w, h, size)
    interp = cv2.INTER_AREA if m["scale"] < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (m["new_w"], m["new_h"]), interpolation=interp)
    canvas = np.full((size, size) + img.shape[2:], fill, img.dtype)
    canvas[m["pad_y"]:m["pad_y"] + m["new_h"], m["pad_x"]:m["pad_x"] + m["new_w"]] = resized
    return canvas, m


def letterbox_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize; padding is labelled IGNORE_INDEX."""
    h, w = mask.shape[:2]
    m = letterbox_meta(w, h, size)
    resized = cv2.resize(mask, (m["new_w"], m["new_h"]), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size, size), IGNORE_INDEX, np.uint8)
    canvas[m["pad_y"]:m["pad_y"] + m["new_h"], m["pad_x"]:m["pad_x"] + m["new_w"]] = resized
    return canvas


def crop_canvas(arr: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """Remove the letterbox padding from a (size,size[,C]) array -> (new_h,new_w[,C])."""
    return arr[meta["pad_y"]:meta["pad_y"] + meta["new_h"], meta["pad_x"]:meta["pad_x"] + meta["new_w"]]


def canvas_to_original_mask(mask_canvas: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    cropped = crop_canvas(mask_canvas, meta)
    return cv2.resize(cropped, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------------- duplicates

def dhash(img_bgr_or_gray: np.ndarray) -> int:
    gray = img_bgr_or_gray if img_bgr_or_gray.ndim == 2 else cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ------------------------------------------------------------------------------------ visuals

def overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    for cls, col in {**CLASS_COLOURS_RGB, IGNORE_INDEX: IGNORE_COLOUR_RGB}.items():
        sel = mask == cls
        if cls == IGNORE_INDEX:
            sel = sel & (np.indices(mask.shape).sum(0) % 6 < 2)  # sparse diagonal hatch for ignored pixels
        out[sel] = (1 - alpha) * out[sel] + alpha * np.array(col, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def label_strip(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def side_by_side(image_rgb: np.ndarray, gt: np.ndarray, pred: Optional[np.ndarray] = None) -> np.ndarray:
    panels = [label_strip(image_rgb, "image"), label_strip(overlay(image_rgb, gt), "ground truth")]
    if pred is not None:
        panels.append(label_strip(overlay(image_rgb, pred), "prediction"))
    return np.hstack(panels)


LEGEND = "overlay colours: Room=blue  Wall=red  Door=green  Window=yellow  hatched grey=ignored"


# -------------------------------------------------------------------------------------- metrics

def confusion(pred: np.ndarray, gt: np.ndarray, n_classes: int, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """cm[g, p] = number of pixels with ground-truth class g predicted as p. Ignored pixels are excluded."""
    valid = gt != ignore
    idx = gt[valid].astype(np.int64) * n_classes + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def metrics_from_confusion(cm: np.ndarray, class_names: Sequence[str]) -> Dict[str, Any]:
    """Per-class IoU, Dice(=F1), precision, recall. A value is None when its denominator is 0."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt_n, pr_n = cm.sum(1), cm.sum(0)

    def ratio(num, den):
        return [None if d == 0 else float(n / d) for n, d in zip(num, den)]

    per = {
        "iou": ratio(tp, gt_n + pr_n - tp),
        "dice_f1": ratio(2 * tp, gt_n + pr_n),
        "precision": ratio(tp, pr_n),
        "recall": ratio(tp, gt_n),
    }
    out: Dict[str, Any] = {"per_class": {
        name: {k: per[k][i] for k in per} | {"gt_pixels": int(gt_n[i]), "pred_pixels": int(pr_n[i])}
        for i, name in enumerate(class_names)}}
    for k in ("iou", "dice_f1"):
        vals = [v for v in per[k] if v is not None]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else None
    total = cm.sum()
    out["pixel_accuracy"] = float(tp.sum() / total) if total else None
    return out


def class_weights_from_counts(counts: Sequence[int], mode: str, max_weight: float = 5.0) -> Optional[List[float]]:
    if mode == "none":
        return None
    if mode != "sqrt_inv_freq":
        raise ValueError(f"unknown ce_class_weights mode {mode!r}")
    c = np.maximum(np.asarray(counts, np.float64), 1.0)
    w = 1.0 / np.sqrt(c / c.sum())
    w = w / w.mean()
    return [float(min(x, max_weight)) for x in w]
