"""
Inference with the FINE-TUNED U-Net (ResNet-18) checkpoint produced by training/train_segmentation.py.

    python inference/predict_segmentation.py --checkpoint runs/seg/unet_resnet18/best.pt \
        --image-dir <images> --out-dir outputs/segmentation_predictions [--save-masks]

Pipeline: image -> letterbox 512 -> U-Net logits -> crop padding -> bilinear resize to the ORIGINAL size ->
argmax -> class mask -> connected regions -> polygons in ORIGINAL image pixel coordinates -> JSON.

Integrity rules:
  * the checkpoint must load strictly into the architecture it declares; otherwise this fails loudly
  * every coordinate comes from the predicted mask; no confidence scores, no default or invented entities
  * an image that fails is written to failures.json with its error; nothing replaces it
  * the polygon JSON is schema-validated (SegmentationPrediction); invalid output is an error, not repaired

Coordinates are pixel-grid coordinates of the original image (x right, y down); vertices are pixel centres.
Walls are reported as region polygons (outer ring + holes), not as centre-lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.seg_common import (  # noqa: E402
    letterbox_image, letterbox_meta, crop_canvas, load_config, overlay,
)

ENTITY_CLASSES = {"rooms": "Room", "walls": "Wall", "doors": "Door", "windows": "Window"}
DEFAULT_CHECKPOINT = "runs/seg/unet_resnet18/best.pt"


# ------------------------------------------------------------------------------------ schema

class RegionEntity(BaseModel):
    id: str
    polygon: List[List[float]] = Field(..., min_length=3, description="outer ring [[x, y], ...] in original-image pixels")
    holes: List[List[List[float]]] = Field(default_factory=list)
    area_px: float = Field(..., gt=0, description="polygon area (outer minus holes) in px^2, measured from the polygon")

    @model_validator(mode="after")
    def _points_ok(self) -> "RegionEntity":
        for ring in [self.polygon, *self.holes]:
            if len(ring) < 3 or any(len(p) != 2 for p in ring):
                raise ValueError("every ring needs >= 3 [x, y] points")
        return self


class SegmentationPrediction(BaseModel):
    image: str
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    units: str = "pixels"
    generated_by: Dict[str, Any] = Field(default_factory=dict)
    rooms: List[RegionEntity] = Field(default_factory=list)
    walls: List[RegionEntity] = Field(default_factory=list)
    doors: List[RegionEntity] = Field(default_factory=list)
    windows: List[RegionEntity] = Field(default_factory=list)

    @model_validator(mode="after")
    def _inside_image(self) -> "SegmentationPrediction":
        for key in ENTITY_CLASSES:
            for e in getattr(self, key):
                for ring in [e.polygon, *e.holes]:
                    for x, y in ring:
                        if not (0 <= x <= self.image_width and 0 <= y <= self.image_height):
                            raise ValueError(f"{e.id}: point ({x}, {y}) outside the {self.image_width}x{self.image_height} image")
        return self


# ---------------------------------------------------------------------- mask -> polygons (numpy)

def _ring(cnt: np.ndarray, eps_frac: float) -> Optional[np.ndarray]:
    approx = cv2.approxPolyDP(cnt, eps_frac * cv2.arcLength(cnt, True), True)
    return approx.reshape(-1, 2) if len(approx) >= 3 else None


def mask_to_entities(binary: np.ndarray, prefix: str, min_area_px: float, eps_frac: float) -> List[Dict[str, Any]]:
    """Connected regions of a binary mask -> [{id, polygon, holes, area_px}] (outer ring + holes)."""
    contours, hier = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    found = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:          # a hole; handled with its parent
            continue
        holes = [contours[j] for j in range(len(contours)) if hier[j][3] == i]
        area = cv2.contourArea(cnt) - sum(cv2.contourArea(h) for h in holes)
        if area < min_area_px:
            continue
        outer = _ring(cnt, eps_frac)
        if outer is None:
            continue
        hole_rings = [r for h in holes if cv2.contourArea(h) >= min_area_px and (r := _ring(h, eps_frac)) is not None]
        found.append((outer, hole_rings, float(area)))
    found.sort(key=lambda t: (int(t[0][:, 1].min()), int(t[0][:, 0].min())))
    return [{"id": f"{prefix}_{k + 1}", "polygon": o.astype(float).tolist(), "holes": [h.astype(float).tolist() for h in hs],
             "area_px": a} for k, (o, hs, a) in enumerate(found)]


def mask_to_prediction(mask: np.ndarray, image_name: str, cfg: Dict[str, Any],
                       generated_by: Optional[Dict[str, Any]] = None) -> SegmentationPrediction:
    h, w = mask.shape
    inf = cfg["inference"]
    min_area = inf["min_area_frac"] * h * w
    names = cfg["class_names"]
    entities = {key: mask_to_entities(mask == names.index(cls), key[:-1], min_area, inf["simplify_epsilon_frac"])
                for key, cls in ENTITY_CLASSES.items()}
    return SegmentationPrediction(image=image_name, image_width=w, image_height=h, generated_by=generated_by or {}, **entities)


# ------------------------------------------------------------------------- model (needs torch)

def load_model(checkpoint: Path, device: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """Load the fine-tuned checkpoint strictly. Raises if it is missing, inconsistent or not the declared architecture."""
    import torch
    from training.seg_model import UNetResNet18

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint} (train with training/train_segmentation.py first)")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("model_state", "encoder", "n_classes", "class_names", "image_size", "train_ids", "val_ids"):
        if key not in ckpt:
            raise ValueError(f"{checkpoint}: missing '{key}'; this is not a checkpoint written by train_segmentation.py")
    if ckpt["encoder"] != "resnet18" or ckpt["n_classes"] != len(ckpt["class_names"]):
        raise ValueError(f"{checkpoint}: unsupported architecture {ckpt['encoder']!r} / n_classes {ckpt['n_classes']}")
    model = UNetResNet18(ckpt["n_classes"], pretrained=False)   # weights come from the checkpoint, nothing is downloaded
    model.load_state_dict(ckpt["model_state"], strict=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    h = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    meta = {k: ckpt[k] for k in ("class_names", "image_size", "train_ids", "val_ids", "epoch", "val_mean_iou",
                                 "pretrained_encoder", "overfit_sanity", "created_utc") if k in ckpt}
    meta.update({"checkpoint": str(checkpoint), "checkpoint_sha256": h.hexdigest(), "device": str(dev), "architecture": "unet_resnet18"})
    return model, meta


def predict_mask(model: Any, image_rgb: np.ndarray, size: int, amp: bool = True) -> np.ndarray:
    """RGB uint8 image -> class mask (uint8) at the ORIGINAL image resolution."""
    import torch
    import torch.nn.functional as F
    from training.seg_model import normalise

    dev = next(model.parameters()).device
    h, w = image_rgb.shape[:2]
    canvas, meta = letterbox_image(image_rgb, size)
    x = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).unsqueeze(0).to(dev)
    with torch.no_grad():
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            logits = model(normalise(x))
        logits = logits.float()[:, :, meta["pad_y"]:meta["pad_y"] + meta["new_h"], meta["pad_x"]:meta["pad_x"] + meta["new_w"]]
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        mask = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
    return mask


def predict_file(model: Any, meta: Dict[str, Any], image_path: Path, cfg: Dict[str, Any]) -> Tuple[np.ndarray, SegmentationPrediction, Dict[str, float]]:
    import torch

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sync = torch.cuda.synchronize if meta["device"].startswith("cuda") else (lambda: None)
    sync(); t0 = time.perf_counter()
    mask = predict_mask(model, rgb, meta["image_size"])
    sync(); t1 = time.perf_counter()
    pred = mask_to_prediction(mask, image_path.name, cfg, {k: meta[k] for k in ("architecture", "checkpoint_sha256", "epoch")})
    t2 = time.perf_counter()
    return mask, pred, {"model_ms": (t1 - t0) * 1e3, "json_ms": (t2 - t1) * 1e3, "total_ms": (t2 - t0) * 1e3}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out-dir", default="outputs/segmentation_predictions")
    p.add_argument("--config", default="config/segmentation.yaml")
    p.add_argument("--save-masks", action="store_true", help="also write <stem>_mask.png and <stem>_overlay.png")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    model, meta = load_model(Path(args.checkpoint))
    if meta.get("overfit_sanity"):
        print("WARNING: this checkpoint comes from an --overfit sanity run; its outputs are not results.")
    images = sorted(f for ext in ("*.png", "*.jpg", "*.jpeg") for f in Path(args.image_dir).glob(ext))
    if not images:
        raise SystemExit(f"no images in {args.image_dir}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    for img in images:
        try:
            mask, pred, t = predict_file(model, meta, img, cfg)
            (out / f"{img.stem}.json").write_text(pred.model_dump_json(indent=2), encoding="utf-8")
            if args.save_masks:
                cv2.imwrite(str(out / f"{img.stem}_mask.png"), mask)
                rgb = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(out / f"{img.stem}_overlay.png"), cv2.cvtColor(overlay(rgb, mask), cv2.COLOR_RGB2BGR))
            print(f"{img.name}: " + ", ".join(f"{len(getattr(pred, k))} {k}" for k in ENTITY_CLASSES) + f"  ({t['total_ms']:.0f} ms)")
        except Exception as e:  # noqa: BLE001 - recorded, never replaced by default geometry
            failures.append({"image": img.name, "error": f"{type(e).__name__}: {e}"})
            print(f"FAILED {img.name}: {e}")
    (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"done: {len(images) - len(failures)} ok, {len(failures)} failed -> {out}")


if __name__ == "__main__":
    main()
