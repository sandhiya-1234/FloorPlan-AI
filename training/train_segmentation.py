"""
Fine-tune a U-Net with a pretrained ResNet-18 encoder on the prepared CubiCasa5K subset.

Real gradient-based training: ImageNet-pretrained encoder + randomly initialised decoder, both updated
with AdamW on cross-entropy + Dice loss. Run on a CUDA GPU (Colab T4).

    python training/train_segmentation.py --config config/segmentation.yaml                 # full run
    python training/train_segmentation.py --overfit 8 --epochs 30 --run-name overfit8       # pipeline sanity

Safety gates (not options):
  * data_dir/manifest.json must exist (created by training.prepare_dataset prepare)
  * data_dir/overlays_confirmed.txt must exist: a person has looked at the overlays and they line up
  * the saved checkpoint records the ids of every training/validation image so evaluation can prove the
    test images were never seen

An --overfit run trains and validates on the SAME few images. It only shows that the pipeline can learn;
its output is labelled overfit_sanity=true and must never be reported as a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.seg_common import (  # noqa: E402
    IGNORE_INDEX, class_weights_from_counts, letterbox_image, letterbox_mask, load_config, metrics_from_confusion,
)
from training.seg_model import UNetResNet18, normalise  # noqa: E402


# ------------------------------------------------------------------------------------- data

class SegDataset(Dataset):
    """Images and masks prepared by training.prepare_dataset, letterboxed to a square canvas."""

    def __init__(self, data_dir: Path, split: str, names: List[str], size: int, augment: bool = False,
                 flips: bool = True, rot90: bool = True):
        self.dir, self.split, self.names, self.size = Path(data_dir), split, list(names), size
        self.augment, self.flips, self.rot90 = augment, flips, rot90

    def __len__(self) -> int:
        return len(self.names)

    def load(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        n = self.names[i]
        img = cv2.imread(str(self.dir / "images" / self.split / f"{n}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.dir / "masks" / self.split / f"{n}.png"), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None or img.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"bad sample {self.split}/{n}: image/mask missing or size mismatch")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = letterbox_image(img, self.size)
        return img, letterbox_mask(mask, self.size)

    def __getitem__(self, i: int):
        img, mask = self.load(i)
        if self.augment:
            if self.flips and random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if self.flips and random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            if self.rot90:
                k = random.randint(0, 3)
                img, mask = np.rot90(img, k), np.rot90(mask, k)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(np.ascontiguousarray(mask)).long()


# ------------------------------------------------------------------------------------- loss

def dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore: int = IGNORE_INDEX, eps: float = 1.0) -> torch.Tensor:
    """Soft multi-class Dice over the batch, ignoring pixels labelled `ignore`. Mean over all classes."""
    n_classes = logits.shape[1]
    valid = (target != ignore).unsqueeze(1).float()
    t = target.clone()
    t[target == ignore] = 0
    onehot = F.one_hot(t, n_classes).permute(0, 3, 1, 2).float() * valid
    probs = logits.softmax(1) * valid
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


class SegLoss:
    def __init__(self, ce_w: float, dice_w: float, class_weights: Optional[List[float]], device: torch.device):
        self.ce_w, self.dice_w = ce_w, dice_w
        self.weights = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None

    def __call__(self, logits: torch.Tensor, target: torch.Tensor):
        ce = F.cross_entropy(logits, target, weight=self.weights, ignore_index=IGNORE_INDEX)
        dc = dice_loss(logits, target)
        return self.ce_w * ce + self.dice_w * dc, ce.detach(), dc.detach()


# --------------------------------------------------------------------------------- utilities

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(_):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def run_validation(model, loader, criterion, device, n_classes: int, use_amp: bool) -> Tuple[float, np.ndarray]:
    model.eval()
    cm = torch.zeros(n_classes * n_classes, dtype=torch.long, device=device)
    loss_sum, n = 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(normalise(imgs))
        loss, _, _ = criterion(logits.float(), masks)
        loss_sum += float(loss) * imgs.size(0)
        n += imgs.size(0)
        pred, valid = logits.argmax(1), masks != IGNORE_INDEX
        cm += torch.bincount(masks[valid] * n_classes + pred[valid], minlength=n_classes * n_classes)
    return loss_sum / max(n, 1), cm.view(n_classes, n_classes).cpu().numpy()


def _atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------- train

def train(cfg: Dict[str, Any], data_dir: Path, run_dir: Path, overfit: int = 0, epochs: Optional[int] = None,
          pretrained: Optional[bool] = None) -> Dict[str, Any]:
    tc, mc = cfg["train"], cfg["model"]
    n_cls, size, names = cfg["n_classes"], cfg["image"]["size"], cfg["class_names"]
    epochs = epochs or tc["epochs"]
    pretrained = mc["pretrained_encoder"] if pretrained is None else pretrained
    seed_everything(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found; training on CPU is only suitable for a few smoke-test steps.")
    use_amp = bool(tc["amp"]) and device.type == "cuda"

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_names, val_names = manifest["splits"]["train"], manifest["splits"]["val"]
    if overfit:
        train_names = train_names[:overfit]
        val_names = train_names                      # deliberately the same images: sanity run only
    if not train_names or not val_names:
        raise RuntimeError("empty train or validation split; run prepare_dataset first")
    assert not set(train_names) & set(manifest["splits"]["test"]), "test image found in the training split"
    if not overfit:
        assert not set(val_names) & set(manifest["splits"]["test"]), "test image found in the validation split"

    cw = class_weights_from_counts(manifest["class_pixels"]["train"], tc["loss"]["ce_class_weights"])
    aug = tc["augment"]
    train_set = SegDataset(data_dir, "train", train_names, size, True, aug["flips"], aug["rot90"])
    val_set = SegDataset(data_dir, "val" if not overfit else "train", val_names, size, False)
    nw = tc["num_workers"]
    train_loader = DataLoader(train_set, batch_size=tc["batch_size"], shuffle=True, num_workers=nw, pin_memory=device.type == "cuda",
                              drop_last=len(train_set) >= tc["batch_size"], worker_init_fn=_worker_init, persistent_workers=nw > 0)
    val_loader = DataLoader(val_set, batch_size=tc["batch_size"], shuffle=False, num_workers=nw)

    model = UNetResNet18(n_cls, pretrained=pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = epochs * len(train_loader)
    warm = min(100, max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / max(total, 1))))
    scaler = make_scaler(use_amp)
    criterion = SegLoss(tc["loss"]["ce_weight"], tc["loss"]["dice_weight"], cw, device)

    run_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "overfit_sanity": bool(overfit),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "torch": torch.__version__,
        "amp_fp16": use_amp, "pretrained_encoder": bool(pretrained), "epochs": epochs, "batch_size": tc["batch_size"],
        "n_train_images": len(train_set), "n_val_images": len(val_set), "steps_per_epoch": len(train_loader),
        "n_parameters": n_params, "n_trainable_parameters": n_train, "ce_class_weights": cw,
        "config": {k: cfg[k] for k in ("experiment", "classes", "image", "model", "train", "subset")},
    }
    _atomic_json(run_dir / "config.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "config"}, indent=2))

    history: List[Dict[str, Any]] = []
    best, best_epoch, steps, bad_steps, t_start = -1.0, 0, 0, 0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, acc = time.time(), np.zeros(3)
        n_batches = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(normalise(imgs))
            loss, ce, dc = criterion(logits.float(), masks)
            if not torch.isfinite(loss):
                bad_steps += 1
                if bad_steps > 5:
                    raise RuntimeError("loss is not finite for more than 5 steps; stopping instead of continuing")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            steps += 1
            n_batches += 1
            acc += [float(loss), float(ce), float(dc)]
        val_loss, cm = run_validation(model, val_loader, criterion, device, n_cls, use_amp)
        m = metrics_from_confusion(cm, names)
        score = m["mean_iou"] if m["mean_iou"] is not None else -1.0
        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": acc[0] / max(n_batches, 1),
               "train_ce": acc[1] / max(n_batches, 1), "train_dice_loss": acc[2] / max(n_batches, 1),
               "val_loss": val_loss, "val_mean_iou": m["mean_iou"], "val_mean_dice_f1": m["mean_dice_f1"],
               "val_pixel_accuracy": m["pixel_accuracy"], "val_iou_per_class": {k: v["iou"] for k, v in m["per_class"].items()},
               "epoch_seconds": round(time.time() - t0, 1), "optimizer_steps_total": steps}
        history.append(rec)
        _atomic_json(run_dir / "history.json", history)
        print(f"epoch {epoch:3d}/{epochs}  train {rec['train_loss']:.4f}  val {val_loss:.4f}  val mIoU {score:.4f}  "
              f"({rec['epoch_seconds']}s)  IoU " + " ".join(f"{k[:3]}={'-' if v is None else f'{v:.2f}'}" for k, v in rec["val_iou_per_class"].items()))
        if score > best:
            best, best_epoch = score, epoch
            ckpt = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": epoch,
                    "val_mean_iou": m["mean_iou"], "class_names": names, "image_size": size, "n_classes": n_cls,
                    "encoder": mc["encoder"], "pretrained_encoder": bool(pretrained), "overfit_sanity": bool(overfit),
                    "train_ids": list(train_names), "val_ids": list(val_names), "created_utc": info["created_utc"]}
            tmp = run_dir / "best.pt.tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, run_dir / "best.pt")

    summary = {"best_epoch": best_epoch, "best_val_mean_iou": best, "optimizer_steps": steps, "total_seconds": round(time.time() - t_start, 1),
               "first_epoch_train_loss": history[0]["train_loss"], "last_epoch_train_loss": history[-1]["train_loss"],
               "overfit_sanity": bool(overfit), "non_finite_loss_steps": bad_steps,
               "checkpoint": "best.pt", "checkpoint_sha256": _sha256(run_dir / "best.pt"),
               "checkpoint_bytes": (run_dir / "best.pt").stat().st_size, **{k: info[k] for k in ("device", "torch", "amp_fp16", "pretrained_encoder", "n_parameters")}}
    _atomic_json(run_dir / "train_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/segmentation.yaml")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--run-name", default="unet_resnet18")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--overfit", type=int, default=0, help="train+validate on N training images only (pipeline sanity check)")
    args = p.parse_args()
    cfg = load_config(Path(args.config))
    data_dir = Path(args.data_dir or cfg["paths"]["data_dir"])
    if not (data_dir / "manifest.json").is_file():
        raise SystemExit(f"{data_dir/'manifest.json'} not found: run `python -m training.prepare_dataset prepare` first")
    if not (data_dir / "overlays_confirmed.txt").is_file():
        raise SystemExit(f"{data_dir/'overlays_confirmed.txt'} not found: inspect the overlays in {data_dir/'overlays'} and confirm they "
                         "line up with the images (the notebook has a cell for this). Training on unverified masks is not allowed.")
    train(cfg, data_dir, Path(args.runs_dir or cfg["paths"]["runs_dir"]) / args.run_name, overfit=args.overfit, epochs=args.epochs)


if __name__ == "__main__":
    main()
