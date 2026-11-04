"""
Torch-side mechanics tests: model shapes, losses, one real optimisation loop, checkpoint round-trip, inference.

Skipped where torch/torchvision are not installed (the development sandbox has neither). They run on Colab
(see COLAB.md / the notebook). They use tiny random-initialised models and fake data: they prove the code
path works, NOT that the model is any good. No metric from these tests may be reported.
"""

import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs torch (Colab); not installed in the development sandbox")
pytest.importorskip("torchvision", reason="needs torchvision (Colab)")

from evaluation.evaluate_segmentation import evaluate_split  # noqa: E402
from inference.predict_segmentation import load_model, predict_file, predict_mask  # noqa: E402
from training.seg_common import IGNORE_INDEX, load_config  # noqa: E402
from training.seg_model import UNetResNet18, normalise  # noqa: E402
from training.train_segmentation import SegLoss, dice_loss, train  # noqa: E402

CFG = load_config()


def test_forward_shape_and_parameter_count():
    m = UNetResNet18(5, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 3, 64, 96))
    assert out.shape == (2, 5, 64, 96)
    assert sum(p.numel() for p in m.parameters()) > 10_000_000          # ResNet-18 encoder alone is 11.2 M


def test_dice_loss_values_and_ignore_handling():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, 4:] = 1
    perfect = torch.zeros(1, 5, 8, 8)
    perfect[0, 0, :, :4] = 20.0
    perfect[0, 1, :, 4:] = 20.0
    assert dice_loss(perfect, target).item() < 0.05                      # classes 2-4 absent in both -> smoothed Dice = 1
    wrong = torch.roll(perfect, 4, dims=3)
    assert dice_loss(wrong, target).item() > dice_loss(perfect, target).item()
    all_ignored = torch.full((1, 8, 8), IGNORE_INDEX, dtype=torch.long)
    assert torch.isfinite(dice_loss(perfect, all_ignored))


def test_a_real_optimiser_loop_reduces_loss_and_changes_weights():
    torch.manual_seed(0)
    m = UNetResNet18(5, pretrained=False)
    y = torch.zeros(2, 64, 64, dtype=torch.long)
    y[:, :, 32:] = 1                                                      # learnable target: left half class 0, right half class 1
    x = torch.where(y[:, None] == 1, 200, 40).expand(2, 3, 64, 64).to(torch.uint8)
    before = m.head[-1].weight.detach().clone()
    crit = SegLoss(1.0, 1.0, None, torch.device("cpu"))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(12):
        opt.zero_grad()
        loss, _, _ = crit(m(normalise(x)), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] and not torch.equal(before, m.head[-1].weight)


def _tiny_dataset(root, n_train=4, n_val=2, n_test=2, size=(96, 128)):
    rng = np.random.default_rng(0)
    manifest = {"splits": {"train": [], "val": [], "test": []}, "samples": {}, "class_pixels": {"train": [900, 400, 60, 20, 10]}}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            name = f"{split}__{i}"
            mask = np.zeros(size, np.uint8)
            x0 = int(rng.integers(5, 20))
            mask[x0:size[0] - 5, x0:size[1] - 5] = 1
            mask[x0:x0 + 4, x0:size[1] - 5] = 2
            mask[x0:x0 + 4, 30:40] = 4
            img = np.where(mask[..., None] == 2, 30, 235).astype(np.uint8).repeat(3, axis=2)
            cv2.imwrite(str(root / "images" / split / f"{name}.png"), img)
            cv2.imwrite(str(root / "masks" / split / f"{name}.png"), mask)
            manifest["splits"][split].append(name)
            manifest["samples"][name] = {"source_id": f"cat/{split}{i}"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_train_checkpoint_strict_load_predict_and_evaluate_roundtrip(tmp_path):
    cfg = load_config()
    cfg["image"]["size"] = 64
    cfg["train"].update(epochs=2, batch_size=2, num_workers=0, amp=False)
    data, run = tmp_path / "data", tmp_path / "run"
    manifest = _tiny_dataset(data)
    summary = train(cfg, data, run, pretrained=False)                      # real forward/backward/AdamW steps
    assert summary["optimizer_steps"] == 4 and not summary["overfit_sanity"] and summary["pretrained_encoder"] is False
    hist = json.loads((run / "history.json").read_text())
    assert [h["epoch"] for h in hist] == [1, 2] and (run / "best.pt").is_file() and (run / "config.json").is_file()

    model, meta = load_model(run / "best.pt", device="cpu")                # strict state-dict load
    assert set(meta["train_ids"]) == set(manifest["splits"]["train"]) and not set(meta["train_ids"]) & set(manifest["splits"]["test"])
    rgb = cv2.cvtColor(cv2.imread(str(data / "images" / "test" / "test__0.png")), cv2.COLOR_BGR2RGB)
    mask = predict_mask(model, rgb, meta["image_size"], amp=False)
    assert mask.shape == rgb.shape[:2] and mask.dtype == np.uint8 and mask.max() < 5   # original resolution preserved

    mask2, pred, timing = predict_file(model, meta, data / "images" / "test" / "test__0.png", cfg)
    assert np.array_equal(mask, mask2) and (pred.image_width, pred.image_height) == (128, 96) and timing["total_ms"] > 0

    names = manifest["splits"]["test"]
    samples = [(n, cv2.cvtColor(cv2.imread(str(data / "images" / "test" / f"{n}.png")), cv2.COLOR_BGR2RGB),
                cv2.imread(str(data / "masks" / "test" / f"{n}.png"), cv2.IMREAD_UNCHANGED)) for n in names]
    res = evaluate_split(samples, lambda im: (predict_mask(model, im, meta["image_size"], amp=False), 0.0), cfg, tmp_path / "res", n_visuals=2)
    assert res["n_test_images"] == 2 and res["json_validity"]["total"] == 2


def test_loading_a_wrong_or_missing_checkpoint_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pt")
    bad = tmp_path / "bad.pt"
    torch.save({"model_state": {}}, bad)
    with pytest.raises(ValueError, match="not a checkpoint"):
        load_model(bad)
