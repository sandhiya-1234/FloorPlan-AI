"""
Mechanics tests for the segmentation data path (parser, rasteriser, letterbox, metrics, prepare/inspect CLI).

IMPORTANT: the fixtures below are tiny files built from the SELECTORS DOCUMENTED IN THE OFFICIAL CubiCasa5k
LOADER code. They prove the code does what the mapping says; they do NOT prove that the real dataset
files follow that structure. That check is `python -m training.prepare_dataset inspect` on the real zip.
"""

import json
import zipfile

import cv2
import numpy as np
import pytest

from training.prepare_dataset import main as prepare_main, sample_id
from training.seg_common import (
    IGNORE_INDEX, UnmappedCategory, canvas_to_original_mask, class_weights_from_counts, confusion, dhash,
    hamming, letterbox_image, letterbox_mask, load_config, metrics_from_confusion, parse_svg, rasterize,
)

CFG = load_config()
BG, ROOM, WALL, DOOR, WIN = range(5)


def poly(x0, y0, x1, y1):
    return " ".join(f"{x},{y}" for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) + " "


def svg_doc(body, w=200, h=100):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{body}</svg>').encode()


def g(attrs, polygon):
    return f'<g {attrs}><polygon points="{polygon}"/></g>'


def test_config_mapping_is_complete_and_valid():
    rooms = CFG["mapping"]["room_names"]
    assert len(rooms) == 59 and set(rooms.values()) == {"Room"}
    assert set(CFG["mapping"]["ignore_room_names"]) == {"Outdoor", "Below150cm"}
    assert not set(rooms) & set(CFG["mapping"]["ignore_room_names"])
    assert CFG["class_names"] == ["Background", "Room", "Wall", "Door", "Window"]
    assert {k: v for k, v in CFG["mapping"]["selectors"].items() if k != "space"} == {
        "wall": {"attr": "id", "value": "Wall"}, "door": {"attr": "id", "value": "Door"},
        "window": {"attr": "id", "value": "Window"}, "railing": {"attr": "id", "value": "Railing"}}


def test_parse_selects_only_documented_groups_and_drops_small_walls():
    body = (g('id="Wall"', poly(10, 10, 100, 20)) + g('id="Wall"', poly(10, 10, 12, 60))      # 2nd is < 4 px wide
            + g('class="Space Bedroom"', poly(20, 20, 90, 80)) + g('id="Door"', poly(40, 18, 50, 22))
            + g('id="Window"', poly(60, 18, 70, 22)) + g('id="Railing"', poly(0, 90, 50, 95))
            + g('class="FixedFurniture Sink"', poly(25, 25, 30, 30))                          # not a target
            + '<g id="Wall"></g>')                                                           # no polygon child
    parsed = parse_svg(svg_doc(body), CFG)
    assert [len(parsed[k]) for k in ("walls", "doors", "windows", "railings")] == [1, 1, 1, 1]
    assert [n for n, _ in parsed["rooms"]] == ["Bedroom"]
    assert parsed["stats"]["small_wall_dropped"] == 1 and parsed["stats"]["no_usable_polygon_child:wall"] == 1


def test_rasterize_priority_and_ignore_labels():
    body = (g('class="Space Outdoor"', poly(150, 0, 200, 100)) + g('class="Space LivingRoom"', poly(0, 0, 100, 100))
            + g('id="Wall"', poly(0, 40, 100, 50)) + g('id="Window"', poly(30, 38, 40, 52))
            + g('id="Door"', poly(60, 38, 70, 52)) + g('id="Railing"', poly(140, 0, 150, 100)))
    m = rasterize(parse_svg(svg_doc(body), CFG), 100, 200, CFG)
    assert m[20, 20] == ROOM and m[45, 10] == WALL            # wall paints over the room
    assert m[45, 35] == WIN and m[45, 65] == DOOR              # openings paint over the wall
    assert m[50, 175] == IGNORE_INDEX                          # outdoor space is ignored, not Room or Background
    assert m[50, 145] == IGNORE_INDEX                          # railing is ignored, not Wall
    assert m[50, 120] == BG                                    # unannotated pixels are Background
    assert set(np.unique(m)) <= {BG, ROOM, WALL, DOOR, WIN, IGNORE_INDEX}


def test_unknown_room_name_is_rejected_not_guessed():
    parsed = parse_svg(svg_doc(g('class="Space Spaceship"', poly(0, 0, 50, 50))), CFG)
    with pytest.raises(UnmappedCategory) as e:
        rasterize(parsed, 100, 200, CFG)
    assert e.value.names == ["Spaceship"]


def test_letterbox_roundtrip_and_padding():
    m = np.zeros((300, 600), np.uint8)
    m[:, 300:] = WALL
    canvas = letterbox_mask(m, 512)
    assert canvas.shape == (512, 512) and (canvas[:100] == IGNORE_INDEX).all()   # padding rows are ignored
    img, meta = letterbox_image(np.full((300, 600, 3), 7, np.uint8), 512)
    assert img.shape == (512, 512, 3) and meta["new_w"] == 512 and meta["new_h"] == 256 and meta["pad_y"] == 128
    back = canvas_to_original_mask(letterbox_mask(m, 512), {**meta, "orig_w": 600, "orig_h": 300})
    assert back.shape == m.shape and (back == m).mean() > 0.99


def test_metrics_hand_computed():
    gt = np.array([[0, 0, 1, 1], [2, 2, 255, 255]], np.uint8)
    pr = np.array([[0, 1, 1, 1], [2, 0, 0, 0]], np.uint8)
    cm = confusion(pr, gt, 5)
    assert cm.sum() == 6                                       # the two ignored pixels are excluded
    r = metrics_from_confusion(cm, CFG["class_names"])["per_class"]
    assert r["Background"]["iou"] == pytest.approx(1 / 3) and r["Room"]["iou"] == pytest.approx(2 / 3)
    assert r["Wall"]["recall"] == pytest.approx(0.5) and r["Wall"]["precision"] == pytest.approx(1.0)
    assert r["Door"]["iou"] is None                            # class absent from both: undefined, not 0


def test_class_weights_and_dhash():
    w = class_weights_from_counts([9000, 800, 150, 40, 10], "sqrt_inv_freq")
    assert w[4] > w[0] and max(w) <= 5.0 and class_weights_from_counts([1, 1], "none") is None
    a = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    assert hamming(dhash(a), dhash(a)) == 0 and hamming(dhash(a), dhash(255 - a)) > 3
    assert sample_id("/high_quality_architectural/6044/") == "high_quality_architectural/6044"
    assert sample_id("x/cubicasa5k/colorful/12/F1_scaled.png") == "colorful/12"


# ---- end to end on a miniature zip in the official layout -----------------------------------------

def _fake_zip(path, with_door=True, unknown_room=False):
    rng = np.random.default_rng(1)
    entries, lists = {}, {"train": [], "val": [], "test": []}

    def add(split, sid, seed, room="Bedroom", clone_of=None):
        r = np.random.default_rng(seed)
        x0, y0 = int(r.integers(5, 40)), int(r.integers(5, 30))
        x1, y1 = x0 + int(r.integers(80, 120)), y0 + int(r.integers(40, 60))
        img = np.full((100, 200, 3), 255, np.uint8)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 4)
        for _ in range(8):
            cv2.circle(img, (int(r.integers(0, 200)), int(r.integers(0, 100))), int(r.integers(2, 6)), (90, 90, 90), -1)
        body = (g('class="Space ' + room + '"', poly(x0 + 3, y0 + 3, x1 - 3, y1 - 3)) + g('id="Wall"', poly(x0, y0, x1, y0 + 4))
                + g('id="Window"', poly(x0 + 10, y0, x0 + 20, y0 + 4)) + (g('id="Door"', poly(x0 + 30, y0, x0 + 40, y0 + 4)) if with_door else ""))
        png = cv2.imencode(".png", img)[1].tobytes()
        pre = f"cubicasa5k/{sid}/"
        entries[pre + "F1_scaled.png"], entries[pre + "model.svg"] = png, svg_doc(body)
        lists[split].append(f"/{sid}/")
    for i in range(6):
        add("test", f"high_quality/{100 + i}", 100 + i)
    for i in range(4):
        add("val", f"high_quality/{200 + i}", 200 + i)
    for i in range(12):
        add("train", f"colorful/{300 + i}", 300 + i, room="Spaceship" if (unknown_room and i == 0) else "Bedroom")
    # a train entry that is an exact copy of a test image: must be dropped from train
    src = "cubicasa5k/high_quality/100/"
    entries["cubicasa5k/colorful/999/F1_scaled.png"] = entries[src + "F1_scaled.png"]
    entries["cubicasa5k/colorful/999/model.svg"] = entries[src + "model.svg"]
    lists["train"].append("/colorful/999/")
    with zipfile.ZipFile(path, "w") as z:
        for k, v in entries.items():
            z.writestr(k, v)
        for s, name in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
            z.writestr("cubicasa5k/" + name, "\n".join(lists[s]) + "\n")
    return path


def test_prepare_end_to_end_on_miniature_zip(tmp_path, capsys):
    z = _fake_zip(tmp_path / "fake.zip", unknown_room=True)
    out = tmp_path / "seg"
    rc = prepare_main(["prepare", "--zip", str(z), "--out", str(out), "--n-train", "20", "--n-val", "4", "--n-test", "6", "--n-overlays", "2"])
    assert rc == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["accepted"]["val"] == 4 and m["accepted"]["test"] == 6
    # the unknown room name was rejected and counted; the copy of a test image was dropped from train
    assert m["unmapped_room_names"] == {"Spaceship": 1}
    assert m["rejections"]["train"]["unmapped_room_name"] == 1
    assert sum(m["duplicate_drops"]["train"].values()) >= 1
    assert m["accepted"]["train"] == 11  # 12 + 1 listed - 1 unmapped - 1 duplicate
    ids = {s: {m["samples"][n]["source_id"] for n in m["splits"][s]} for s in m["splits"]}
    assert not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"]) and "colorful/999" not in ids["train"]
    name = m["splits"]["test"][0]
    mask = cv2.imread(str(out / "masks" / "test" / f"{name}.png"), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (100, 200) and {ROOM, WALL, DOOR, WIN} <= set(np.unique(mask))
    assert len(list((out / "overlays").glob("*.png"))) == 6 and m["alignment_report"]["median_wall_to_room_gray_ratio"] is not None
    assert prepare_main(["inspect", "--zip", str(z), "--n", "10"]) == 0
    assert "room names NOT in the mapping: {'Spaceship'" in capsys.readouterr().out


def test_inspect_fails_when_an_official_selector_is_missing(tmp_path, capsys):
    z = _fake_zip(tmp_path / "nodoor.zip", with_door=False)
    assert prepare_main(["inspect", "--zip", str(z), "--n", "10"]) == 1
    assert "official selector(s) never found" in capsys.readouterr().out
