"""
Build a small, verified CubiCasa5K subset for the segmentation experiment. Reads straight from the
official zip (no extraction of the whole archive, no LMDB).

    # 1. Look at the REAL annotation structure and check every official selector exists
    python -m training.prepare_dataset inspect --zip /content/cubicasa5k.zip

    # 2. Select a seeded subset from the OFFICIAL train/val/test lists, rasterise masks, write overlays
    python -m training.prepare_dataset prepare --zip /content/cubicasa5k.zip --out data/seg

Output (data/seg):  images/<split>/<id>.png   masks/<split>/<id>.png   overlays/*.png   manifest.json
Look at overlays/ BEFORE training. The training notebook refuses to start until you confirm them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.seg_common import (  # noqa: E402
    IGNORE_INDEX, LEGEND, UnmappedCategory, dhash, hamming, load_config, overlay, parse_svg, rasterize,
    side_by_side, svg_declared_size,
)

SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------------------- zip indexing

def sample_id(path_like: str) -> str:
    """'/high_quality_architectural/6044/' or '.../6044/F1_scaled.png' style paths -> 'high_quality_architectural/6044'."""
    parts = [p for p in path_like.replace("\\", "/").split("/") if p]
    if parts and parts[-1].lower().endswith((".png", ".svg")):
        parts = parts[:-1]
    if len(parts) < 2:
        raise ValueError(f"cannot derive a sample id from {path_like!r}")
    return "/".join(parts[-2:])


def safe_name(sid: str) -> str:
    return sid.replace("/", "__")


def index_zip(zf: zipfile.ZipFile, image_file: str, svg_file: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ({sample_id: dir_prefix_inside_zip}, {split_name: list_member_name}). Fails loudly on ambiguity."""
    names = set(zf.namelist())
    samples: Dict[str, str] = {}
    for n in names:
        if n.endswith("/" + image_file) or n == image_file:
            prefix = n[: -len(image_file)]
            if prefix + svg_file in names:
                sid = sample_id(prefix)
                if sid in samples and samples[sid] != prefix:
                    raise RuntimeError(f"two folders map to sample id {sid}: {samples[sid]} and {prefix}")
                samples[sid] = prefix
    lists: Dict[str, str] = {}
    for split, base in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
        hits = [n for n in names if n.rsplit("/", 1)[-1] == base]
        if len(hits) != 1:
            raise RuntimeError(f"expected exactly one {base} in the zip, found {hits or 'none'}. "
                               f"Top-level entries: {sorted({n.split('/')[0] for n in names})[:10]}")
        lists[split] = hits[0]
    return samples, lists


def read_split_ids(zf: zipfile.ZipFile, member: str) -> List[str]:
    ids = [sample_id(line.strip()) for line in zf.read(member).decode("utf-8").splitlines() if line.strip()]
    if len(set(ids)) != len(ids):
        print(f"WARNING: {member} lists {len(ids) - len(set(ids))} duplicate entries; keeping the first of each")
    return list(dict.fromkeys(ids))


# ------------------------------------------------------------------------------------ inspect

def cmd_inspect(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    sel = cfg["mapping"]["selectors"]
    with zipfile.ZipFile(args.zip) as zf:
        samples, lists = index_zip(zf, cfg["source"]["image_file"], cfg["source"]["svg_file"])
        split_ids = {s: read_split_ids(zf, m) for s, m in lists.items()}
        print(f"samples with image+svg in zip: {len(samples)}")
        for s in SPLITS:
            present = sum(i in samples for i in split_ids[s])
            print(f"official {s}.txt: {len(split_ids[s])} entries, {present} found in zip ({lists[s]})")
        overlap = {a: set(split_ids[a]) & set(split_ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
        print("official list overlaps (must be 0):", {k: len(v) for k, v in overlap.items()})
        print("categories (folder before the number):", dict(Counter(i.split("/")[0] for i in samples)))

        rng = random.Random(cfg["experiment"]["seed"])
        pick = rng.sample(sorted(samples), min(args.n, len(samples)))
        g_ids, space_names, root_attrs, extra_kinds = Counter(), Counter(), Counter(), Counter()
        presence = Counter()
        size_mismatch = 0
        for sid in pick:
            svg = zf.read(samples[sid] + cfg["source"]["svg_file"])
            root = ET.fromstring(svg)
            found = set()
            for g in root.iter():
                if g.tag.rsplit("}", 1)[-1] != "g":
                    continue
                gid, cls = g.get("id") or "", (g.get("class") or "").split()
                if gid:
                    g_ids[gid] += 1
                if cls:
                    extra_kinds[cls[0]] += 1
                if cls and cls[0] == sel["space"]["first_token"]:
                    space_names[cls[1] if len(cls) > 1 else "<missing>"] += 1
                    found.add("space")
                for k in ("wall", "door", "window", "railing"):
                    if gid == sel[k]["value"]:
                        found.add(k)
            presence.update(found)
            decl = svg_declared_size(svg)
            root_attrs[json.dumps(decl, sort_keys=True)[:80]] += 1
            img = cv2.imdecode(np.frombuffer(zf.read(samples[sid] + cfg["source"]["image_file"]), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            vb = (decl["viewBox"] or "").split()
            if len(vb) == 4 and (round(float(vb[2])), round(float(vb[3]))) != (w, h):
                size_mismatch += 1

    to_room, ign = (set(cfg["mapping"]["room_names"]), set(cfg["mapping"]["ignore_room_names"]))
    unmapped = {n: c for n, c in space_names.items() if n not in to_room and n not in ign}
    print(f"\n--- selector census over {len(pick)} random samples ---")
    print("group ids seen (top 25):", dict(g_ids.most_common(25)))
    print("group class first-tokens (top 15):", dict(extra_kinds.most_common(15)))
    print("selector presence (samples containing it):", {k: f"{presence[k]}/{len(pick)}" for k in ("wall", "door", "window", "space", "railing")})
    print("Space room names seen:", dict(space_names.most_common()))
    print("room names NOT in the mapping:", unmapped or "none")
    print("svg root size attributes (first 3):", list(root_attrs.items())[:3])
    print("samples whose svg viewBox size != F1_scaled.png size:", size_mismatch)
    missing = [k for k in ("wall", "door", "window", "space") if presence[k] == 0]
    if missing:
        print(f"\nFAIL: official selector(s) never found in real data: {missing}. Do NOT train; fix the mapping first.")
        return 1
    print("\nOK: every required official selector exists in the real files. Unmapped names (if any) will skip their samples, not be guessed.")
    return 0


# ------------------------------------------------------------------------------------ prepare

def _decode(zf: zipfile.ZipFile, member: str) -> Tuple[bytes, np.ndarray]:
    raw = zf.read(member)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode {member}")
    return raw, img


def cmd_prepare(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    sub = cfg["subset"]
    n_req = {"train": args.n_train or sub["n_train"], "val": args.n_val or sub["n_val"], "test": args.n_test or sub["n_test"]}
    out = Path(args.out)
    for sub_dir in ("images", "masks"):
        for s in SPLITS:
            (out / sub_dir / s).mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    names = cfg["class_names"]
    n_cls = cfg["n_classes"]
    c_wall, c_room = names.index("Wall"), names.index("Room")

    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "zip_file": Path(args.zip).name, "zip_bytes": Path(args.zip).stat().st_size,
        "config_snapshot": {k: cfg[k] for k in ("source", "classes", "mapping", "subset", "image")},
        "requested": n_req, "splits": {s: [] for s in SPLITS}, "samples": {},
        "rejections": {s: Counter() for s in SPLITS}, "unmapped_room_names": Counter(),
        "class_pixels": {s: [0] * n_cls for s in SPLITS}, "ignored_pixels": {s: 0 for s in SPLITS},
        "duplicate_drops": {s: Counter() for s in SPLITS}, "parse_stats": Counter(),
    }
    seen: Dict[str, List[Tuple[str, int, str, str]]] = {"hash": []}  # (split, dhash, sha_img, sha_mask)
    n_overlays = Counter()

    with zipfile.ZipFile(args.zip) as zf:
        samples, lists = index_zip(zf, cfg["source"]["image_file"], cfg["source"]["svg_file"])
        official = {s: read_split_ids(zf, m) for s, m in lists.items()}
        manifest["official_list_sizes"] = {s: len(v) for s, v in official.items()}
        manifest["official_list_sha256"] = {s: hashlib.sha256(zf.read(m)).hexdigest() for s, m in lists.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if set(official[a]) & set(official[b]):
                raise RuntimeError(f"official lists {a}/{b} overlap; refusing to continue")

        rng = random.Random(sub["seed"])
        # test first, then val, then train: later splits are filtered against earlier ones, so the
        # official val/test boundaries are never altered by duplicate removal.
        for split in ("test", "val", "train"):
            cands = [i for i in official[split] if i in samples]
            manifest["rejections"][split]["listed_but_missing_in_zip"] = len(official[split]) - len(cands)
            rng.shuffle(cands)
            for sid in cands:
                if len(manifest["splits"][split]) >= n_req[split]:
                    break
                prefix, why = samples[sid], None
                try:
                    raw_img, img = _decode(zf, prefix + cfg["source"]["image_file"])
                    svg = zf.read(prefix + cfg["source"]["svg_file"])
                    parsed = parse_svg(svg, cfg)
                    h, w = img.shape[:2]
                    mask = rasterize(parsed, h, w, cfg)
                except UnmappedCategory as e:
                    manifest["unmapped_room_names"].update(e.names)
                    why = "unmapped_room_name"
                except Exception as e:  # noqa: BLE001 - recorded, not hidden
                    why = f"parse_error:{type(e).__name__}"
                manifest["parse_stats"].update(parsed["stats"] if why is None else {})
                if why is None and not (mask == c_wall).any():
                    why = "no_wall_pixels"
                if why is None and not (mask == c_room).any():
                    why = "no_room_pixels"
                if why:
                    manifest["rejections"][split][why] += 1
                    continue

                sha_img = hashlib.sha256(raw_img).hexdigest()
                sha_mask = hashlib.sha256(mask.tobytes()).hexdigest()
                dh = dhash(img)
                dup_of = next((o for o in seen["hash"] if o[0] != split and
                               (o[2] == sha_img or o[3] == sha_mask or hamming(o[1], dh) <= sub["dhash_max_hamming"])), None)
                if dup_of is not None:
                    kind = "exact_image" if dup_of[2] == sha_img else "same_annotation" if dup_of[3] == sha_mask else "near_duplicate_dhash"
                    manifest["duplicate_drops"][split][f"{kind}_vs_{dup_of[0]}"] += 1
                    continue
                seen["hash"].append((split, dh, sha_img, sha_mask))

                name = safe_name(sid)
                (out / "images" / split / f"{name}.png").write_bytes(raw_img)
                cv2.imwrite(str(out / "masks" / split / f"{name}.png"), mask)
                counts = np.bincount(mask[mask != IGNORE_INDEX].ravel(), minlength=n_cls).tolist()
                manifest["class_pixels"][split] = [a + b for a, b in zip(manifest["class_pixels"][split], counts)]
                manifest["ignored_pixels"][split] += int((mask == IGNORE_INDEX).sum())
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                wall_g, room_g = gray[mask == c_wall].mean(), gray[mask == c_room].mean()
                decl = svg_declared_size(svg)
                manifest["splits"][split].append(name)
                manifest["samples"][name] = {
                    "source_id": sid, "split": split, "height": h, "width": w, "class_pixels": counts,
                    "svg_declared": decl, "sha256_image": sha_img,
                    # aligned annotations: wall pixels should be darker than room pixels (report-only)
                    "wall_to_room_gray_ratio": round(float(wall_g / max(room_g, 1e-6)), 4),
                }
                if n_overlays[split] < args.n_overlays:
                    n_overlays[split] += 1
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    panel = side_by_side(rgb, mask)
                    cv2.imwrite(str(out / "overlays" / f"{split}_{name}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    ratios = [v["wall_to_room_gray_ratio"] for v in manifest["samples"].values()]
    manifest["alignment_report"] = {
        "median_wall_to_room_gray_ratio": float(np.median(ratios)) if ratios else None,
        "fraction_ratio_below_0.9": float(np.mean([r < 0.9 for r in ratios])) if ratios else None,
        "note": "Automatic hint only. Aligned data usually has ratio < 1 (walls darker than rooms). "
                "The overlays must still be inspected by a person before training.",
    }
    # train/val/test id disjointness (by source id and by hash) is asserted, not assumed
    ids = {s: {manifest["samples"][n]["source_id"] for n in manifest["splits"][s]} for s in SPLITS}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]), "split id leakage"
    hashes = {s: {manifest["samples"][n]["sha256_image"] for n in manifest["splits"][s]} for s in SPLITS}
    assert not (hashes["train"] & hashes["val"] or hashes["train"] & hashes["test"] or hashes["val"] & hashes["test"]), "split hash leakage"
    manifest["accepted"] = {s: len(manifest["splits"][s]) for s in SPLITS}
    for k in ("rejections", "duplicate_drops"):
        manifest[k] = {s: dict(v) for s, v in manifest[k].items()}
    manifest["unmapped_room_names"] = dict(manifest["unmapped_room_names"])
    manifest["parse_stats"] = dict(manifest["parse_stats"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({k: manifest[k] for k in ("requested", "accepted", "rejections", "duplicate_drops",
                                               "unmapped_room_names", "ignored_pixels", "alignment_report")}, indent=2))
    print(f"\n{LEGEND}\nOverlays: {out/'overlays'}  (LOOK AT THEM before training)\nManifest: {out/'manifest.json'}")
    short = [s for s in SPLITS if manifest["accepted"][s] < n_req[s]]
    if short:
        print(f"WARNING: fewer samples than requested for {short}; see rejections in the manifest.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "prepare"):
        q = sp.add_parser(name)
        q.add_argument("--zip", required=True, help="path to the official cubicasa5k zip")
        q.add_argument("--config", default="config/segmentation.yaml")
    sp.choices["inspect"].add_argument("--n", type=int, default=40, help="random samples to census")
    pr = sp.choices["prepare"]
    pr.add_argument("--out", default="data/seg")
    pr.add_argument("--n-train", type=int, default=0)
    pr.add_argument("--n-val", type=int, default=0)
    pr.add_argument("--n-test", type=int, default=0)
    pr.add_argument("--n-overlays", type=int, default=8, help="overlay panels written per split")
    args = p.parse_args(argv)
    return cmd_inspect(args) if args.cmd == "inspect" else cmd_prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
