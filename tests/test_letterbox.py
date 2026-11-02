"""Letterbox preprocessing must preserve aspect ratio and map coordinates back exactly."""

import pytest
from PIL import Image

from preprocessing.letterbox import letterbox, transform_annotation


@pytest.mark.parametrize("size", [(2000, 1000), (640, 1280), (1008, 1008), (300, 200)])
def test_canvas_size_and_roundtrip(size):
    img = Image.new("RGB", size, (10, 20, 30))
    out, meta = letterbox(img, canvas=1008)
    assert out.size == (1008, 1008)
    assert meta.new_w <= 1008 and meta.new_h <= 1008 and max(meta.new_w, meta.new_h) == 1008
    # aspect ratio preserved to within rounding of one pixel
    assert abs(meta.new_w / meta.new_h - size[0] / size[1]) < 2.0 / min(meta.new_w, meta.new_h)
    for x, y in [(0, 0), (size[0], size[1]), (size[0] / 3, size[1] / 7)]:
        cx, cy = meta.to_canvas(x, y)
        ox, oy = meta.to_original(cx, cy)
        assert abs(ox - x) < 1e-9 and abs(oy - y) < 1e-9


def test_content_lands_where_the_transform_says():
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    img.paste((0, 0, 0), (100, 0, 200, 100))  # right half black
    out, meta = letterbox(img, canvas=400)
    x, y = meta.to_canvas(150, 50)
    assert out.getpixel((round(x), round(y))) == (0, 0, 0)
    x, y = meta.to_canvas(50, 50)
    assert out.getpixel((round(x), round(y))) == (255, 255, 255)
    assert out.getpixel((200, 5)) == (255, 255, 255)  # padding band is white


def test_transform_annotation_maps_every_coordinate_type():
    _, meta = letterbox(Image.new("RGB", (200, 100)), canvas=400)  # scale 2, pad_y 100
    data = {"image": "a.png", "walls": [{"id": "w", "start": [0, 0], "end": [100, 0]}],
            "doors": [{"id": "d", "bbox": [10, 10, 20, 30], "position": [15, 20]}], "windows": [],
            "rooms": [{"id": "r", "type": "x", "polygon": [[0, 0], [10, 0], [10, 10]]}]}
    out = transform_annotation(data, meta.to_canvas)
    assert out["walls"][0]["start"] == [0.0, 100.0] and out["walls"][0]["end"] == [200.0, 100.0]
    assert out["doors"][0]["bbox"] == [20.0, 120.0, 40.0, 160.0] and out["doors"][0]["position"] == [30.0, 140.0]
    assert out["rooms"][0]["polygon"][1] == [20.0, 100.0]
    assert data["walls"][0]["start"] == [0, 0]  # input untouched
