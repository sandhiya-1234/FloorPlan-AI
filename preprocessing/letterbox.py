"""
Aspect-ratio-preserving letterbox preprocessing and exact coordinate transforms.

Every image shown to the VLM is resized (long side -> ``canvas`` px, no distortion) and centred
on a white square canvas. 1008 = 36 * 28, so Qwen2.5-VL's processor (28-px patch grid,
max_pixels >= 1008^2) leaves the canvas untouched and annotation coordinates stay valid in the
frame the model actually sees. ``LetterboxMeta`` maps coordinates back to the original image.
"""

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Tuple

from PIL import Image

DEFAULT_CANVAS = 1008
Point = Tuple[float, float]


@dataclass(frozen=True)
class LetterboxMeta:
    orig_w: int
    orig_h: int
    canvas: int
    scale: float
    new_w: int
    new_h: int
    pad_x: int
    pad_y: int

    def to_canvas(self, x: float, y: float) -> Point:
        return x * self.scale + self.pad_x, y * self.scale + self.pad_y

    def to_original(self, x: float, y: float) -> Point:
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def letterbox(
    image: Image.Image,
    canvas: int = DEFAULT_CANVAS,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[Image.Image, LetterboxMeta]:
    """Resize preserving aspect ratio and centre on a square ``canvas`` x ``canvas`` image."""
    w, h = image.size
    scale = canvas / float(max(w, h))
    new_w = max(1, min(canvas, round(w * scale)))
    new_h = max(1, min(canvas, round(h * scale)))
    resized = image.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
    pad_x = (canvas - new_w) // 2
    pad_y = (canvas - new_h) // 2
    out = Image.new("RGB", (canvas, canvas), fill)
    out.paste(resized, (pad_x, pad_y))
    return out, LetterboxMeta(w, h, canvas, scale, new_w, new_h, pad_x, pad_y)


def transform_annotation(data: Dict[str, Any], fn: Callable[[float, float], Point]) -> Dict[str, Any]:
    """
    Apply a point transform to every coordinate of a floor-plan annotation dict
    (walls: start/end, doors/windows: bbox corners + position, rooms: polygon vertices).
    The transform must be axis-aligned (scale + translate) so bboxes stay valid boxes.
    Returns a new dict; the input is not modified.
    """
    out = {k: v for k, v in data.items() if k not in ("walls", "doors", "windows", "rooms")}
    out["walls"] = [
        {**w, "start": list(fn(*w["start"])), "end": list(fn(*w["end"]))} for w in data.get("walls", [])
    ]
    for key in ("doors", "windows"):
        items = []
        for it in data.get(key, []):
            x1, y1 = fn(it["bbox"][0], it["bbox"][1])
            x2, y2 = fn(it["bbox"][2], it["bbox"][3])
            new = {**it, "bbox": [x1, y1, x2, y2]}
            if it.get("position") is not None:
                new["position"] = list(fn(*it["position"]))
            items.append(new)
        out[key] = items
    out["rooms"] = [{**r, "polygon": [list(fn(*p)) for p in r["polygon"]]} for r in data.get("rooms", [])]
    return out
