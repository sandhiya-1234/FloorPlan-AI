"""
Evaluate the fine-tuned U-Net on the HELD-OUT test split of real CubiCasa5K floor plans.

    python evaluation/evaluate_segmentation.py --checkpoint runs/seg/unet_resnet18/best.pt \
        --data-dir data/seg --out-dir results/segmentation

Reports (pixel-level, original image resolution, ignore-labelled pixels excluded):
  per-class IoU, Dice/F1, precision, recall, confusion matrix, mean IoU/Dice, pixel accuracy,
  JSON validity of the generated polygon files, and inference time.

Integrity checks that abort the run:
  * the test split must be disjoint (by id) from the checkpoint's recorded training and validation ids
  * checkpoints from --overfit sanity runs are refused
Nothing is averaged over classes that are absent from both ground truth and prediction (IoU None).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.predict_segmentation import SegmentationPrediction, mask_to_prediction  # noqa: E402
from training.seg_common import (  # noqa: E402
    IGNORE_INDEX, confusion, load_config, metrics_from_confusion, side_by_side,
)

WARMUP_IMAGES = 2   # excluded from timing statistics (CUDA / cuDNN warm-up)


def evaluate_split(
    samples: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, float]],
    cfg: Dict[str, Any],
    out_dir: Path,
    generated_by: Optional[Dict[str, Any]] = None,
    n_visuals: int = 12,
) -> Dict[str, Any]:
    """
    samples:    (name, image_rgb, gt_mask) with gt_mask at the image's original resolution
    predict_fn: image_rgb -> (predicted class mask at original resolution, model seconds)
    Writes predictions/*.json, visuals/*.png, metrics.json, confusion_matrix.csv, per_image.csv. Returns the summary.
    """
    names, n_cls = cfg["class_names"], cfg["n_classes"]
    out_dir = Path(out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "visuals").mkdir(parents=True, exist_ok=True)

    total_cm = np.zeros((n_cls, n_cls), np.int64)
    per_image: List[Dict[str, Any]] = []
    preds: Dict[str, np.ndarray] = {}
    json_failures: List[Dict[str, str]] = []
    model_ms, json_ms = [], []
    for k, (name, rgb, gt) in enumerate(samples):
        pred_mask, model_s = predict_fn(rgb)
        if pred_mask.shape != gt.shape:
            raise RuntimeError(f"{name}: prediction {pred_mask.shape} != ground truth {gt.shape}")
        cm = confusion(pred_mask, gt, n_cls)
        total_cm += cm
        ious = metrics_from_confusion(cm, names)
        t0 = time.perf_counter()
        valid_json = True
        try:
            pred_obj = mask_to_prediction(pred_mask, name + ".png", cfg, generated_by)
            text = pred_obj.model_dump_json(indent=2)
            SegmentationPrediction.model_validate(json.loads(text))      # round-trip: parse + schema-validate
            (out_dir / "predictions" / f"{name}.json").write_text(text, encoding="utf-8")
            counts = {c: len(getattr(pred_obj, c)) for c in ("rooms", "walls", "doors", "windows")}
        except Exception as e:  # noqa: BLE001 - counted as invalid JSON, never dropped
            valid_json, counts = False, {}
            json_failures.append({"image": name, "error": f"{type(e).__name__}: {e}"})
        t_json = time.perf_counter() - t0
        if k >= WARMUP_IMAGES:
            model_ms.append(model_s * 1e3)
            json_ms.append(t_json * 1e3)
        per_image.append({"image": name, "mean_iou": ious["mean_iou"], "valid_json": valid_json, **counts,
                          **{f"iou_{c}": ious["per_class"][c]["iou"] for c in names}})
        preds[name] = pred_mask

    summary_metrics = metrics_from_confusion(total_cm, names)
    ranked = sorted((p for p in per_image if p["mean_iou"] is not None), key=lambda p: p["mean_iou"])
    chosen = [p["image"] for p in ranked[:n_visuals // 2]] + [p["image"] for p in per_image[:n_visuals - n_visuals // 2]]
    by_name = {n: (rgb, gt) for n, rgb, gt in samples}
    for n in dict.fromkeys(chosen):
        rgb, gt = by_name[n]
        panel = side_by_side(rgb, gt, preds[n])
        scale = min(1.0, 2400 / panel.shape[1])
        if scale < 1:
            panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / "visuals" / f"{n}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    def stats(v: List[float]) -> Optional[Dict[str, float]]:
        return None if not v else {"mean": float(np.mean(v)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                                   "n_timed_images": len(v)}

    n = len(samples)
    n_valid = sum(p["valid_json"] for p in per_image)
    summary: Dict[str, Any] = {
        "n_test_images": n, "class_names": names,
        **summary_metrics,
        "confusion_matrix_counts": total_cm.tolist(),
        "json_validity": {"valid": n_valid, "total": n, "rate": n_valid / n if n else None, "failures": json_failures,
                          "definition": "polygon JSON serialised, re-parsed and validated against SegmentationPrediction"},
        "inference_time_ms": {"model_forward_resize_argmax": stats(model_ms), "mask_to_json": stats(json_ms),
                              "warmup_images_excluded": min(WARMUP_IMAGES, n)},
        "evaluated_pixels": int(total_cm.sum()), "ignored_label": IGNORE_INDEX,
        "limitations": ([f"Only {n} test images: indicative, wide uncertainty."] if n < 100 else []),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + names + ["gt_total", "recall_row_normalised"])
        for i, row in enumerate(total_cm):
            tot = int(row.sum())
            w.writerow([names[i]] + row.tolist() + [tot, "" if tot == 0 else round(float(row[i] / tot), 4)])
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0]) if per_image else ["image"])
        w.writeheader()
        w.writerows(per_image)
    return summary


# --------------------------------------------------------------------------- CLI (needs torch)

def _load_real_samples(data_dir: Path, split: str, names: List[str]):
    for name in names:
        bgr = cv2.imread(str(data_dir / "images" / split / f"{name}.png"), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(data_dir / "masks" / split / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or gt is None or bgr.shape[:2] != gt.shape[:2]:
            raise RuntimeError(f"bad test sample {name}")
        yield name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="runs/seg/unet_resnet18/best.pt")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", default="test", choices=["test", "val"], help="use 'val' only for debugging; reported results use 'test'")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--config", default="config/segmentation.yaml")
    p.add_argument("--n-visuals", type=int, default=12)
    args = p.parse_args()

    from inference.predict_segmentation import load_model, predict_mask
    import torch

    cfg = load_config(Path(args.config))
    data_dir = Path(args.data_dir or cfg["paths"]["data_dir"])
    out_dir = Path(args.out_dir or cfg["paths"]["results_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    model, meta = load_model(Path(args.checkpoint))
    if meta.get("overfit_sanity"):
        raise SystemExit("refusing to evaluate a checkpoint from an --overfit sanity run")
    names = manifest["splits"][args.split]
    src_ids = {manifest["samples"][n]["source_id"] for n in names}
    seen_ids = {n for n in meta["train_ids"]} | {n for n in meta["val_ids"]}
    leaked = [n for n in names if n in seen_ids]
    if args.split == "test" and leaked:
        raise SystemExit(f"LEAKAGE: {len(leaked)} test images appear in the checkpoint's training/validation ids: {leaked[:5]}")
    print(f"evaluating {len(names)} held-out '{args.split}' images (no overlap with the {len(meta['train_ids'])} train / {len(meta['val_ids'])} val ids)")

    def predict_fn(rgb: np.ndarray):
        if meta["device"].startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = predict_mask(model, rgb, meta["image_size"])
        if meta["device"].startswith("cuda"):
            torch.cuda.synchronize()
        return m, time.perf_counter() - t0

    gen = {"architecture": meta["architecture"], "checkpoint_sha256": meta["checkpoint_sha256"], "epoch": meta["epoch"]}
    summary = evaluate_split(list(_load_real_samples(data_dir, args.split, names)), predict_fn, cfg, out_dir, gen, args.n_visuals)

    run_dir = Path(args.checkpoint).parent
    for fname in ("history.json", "train_summary.json", "config.json"):
        if (run_dir / fname).is_file():
            (out_dir / fname).write_text((run_dir / fname).read_text(encoding="utf-8"), encoding="utf-8")
    summary.update({
        "split": args.split, "held_out": True, "checkpoint_sha256": meta["checkpoint_sha256"], "checkpoint_epoch": meta["epoch"],
        "device": meta["device"] + (f" ({torch.cuda.get_device_name(0)})" if meta["device"].startswith("cuda") else ""),
        "n_train_images": len(meta["train_ids"]), "n_val_images": len(meta["val_ids"]),
        "test_ids_overlapping_training_or_validation": len(leaked),
        "manifest_sha256": hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest(),
        "test_source_ids_sha256": hashlib.sha256("\n".join(sorted(src_ids)).encode()).hexdigest(),
    })
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "manifest_subset.json").write_text(json.dumps({"splits": manifest["splits"], "samples": manifest["samples"],
        "rejections": manifest["rejections"], "duplicate_drops": manifest["duplicate_drops"], "alignment_report": manifest["alignment_report"],
        "unmapped_room_names": manifest["unmapped_room_names"], "official_list_sha256": manifest["official_list_sha256"]}, indent=2), encoding="utf-8")
    pc = summary["per_class"]
    print(f"\nmean IoU {summary['mean_iou']:.4f}  mean Dice/F1 {summary['mean_dice_f1']:.4f}  pixel acc {summary['pixel_accuracy']:.4f}")
    for c in cfg["class_names"]:
        f = lambda v: "  n/a " if v is None else f"{v:6.3f}"  # noqa: E731
        print(f"  {c:11s} IoU {f(pc[c]['iou'])}  Dice {f(pc[c]['dice_f1'])}  P {f(pc[c]['precision'])}  R {f(pc[c]['recall'])}")
    print(f"JSON validity {summary['json_validity']['valid']}/{summary['json_validity']['total']}; results in {out_dir}")


if __name__ == "__main__":
    main()
