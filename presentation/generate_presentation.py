"""
PowerPoint generator for the FloorPlan-AI assessment (white background, python-pptx).

Integrity rule: every training or evaluation number on the slides is READ from saved result files; nothing is typed in.
  results/segmentation/summary.json          evaluation on the held-out test split (written by evaluation/evaluate_segmentation.py)
  results/segmentation/train_summary.json    training summary copied next to the results
  results/segmentation/history.json          per-epoch training history
  results/segmentation/per_image.csv         per-image scores (used to pick the example panels)
  results/segmentation/visuals/*.png         ground-truth vs prediction panels
When those files do not exist, the slides say that the model has not been trained / evaluated. Regenerate after a real run:
    python presentation/generate_presentation.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "segmentation"
CFG = yaml.safe_load((ROOT / "config" / "segmentation.yaml").read_text(encoding="utf-8"))

WHITE = RGBColor(255, 255, 255)
DARK = RGBColor(15, 23, 42)
MUTED = RGBColor(71, 85, 105)
BLUE = RGBColor(37, 99, 235)
SOFT = RGBColor(241, 245, 249)
RED = RGBColor(185, 28, 28)
GREEN = RGBColor(21, 128, 61)


def _json(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


TRAIN = _json(RES / "train_summary.json")
EVAL = _json(RES / "summary.json")
HIST = _json(RES / "history.json")
EVALUATED = bool(EVAL and EVAL.get("held_out") and EVAL.get("split") == "test")
TRAINED = bool(TRAIN)


# ------------------------------------------------------------------------------------------ helpers

def new_slide(prs: Presentation):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    return s


def header(slide, title: str, kicker: str = "FLOORPLAN-AI: U-NET / RESNET-18 SEGMENTATION") -> None:
    box = slide.shapes.add_textbox(Inches(0.7), Inches(0.4), Inches(12.0), Inches(1.0))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = kicker
    p.font.size, p.font.bold, p.font.color.rgb = Pt(10), True, BLUE
    t = tf.add_paragraph()
    t.text = title
    t.font.size, t.font.bold, t.font.color.rgb = Pt(24), True, DARK
    t.space_before = Pt(3)


def bullets(slide, items: Sequence[str], top: float = 1.6, left: float = 0.7, width: float = 12.0, height: float = 5.4,
            size: int = 15, color: RGBColor = DARK) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = it
        p.font.size, p.font.color.rgb = Pt(size), color
        p.space_after = Pt(9)


def table(slide, cols: Sequence[str], rows: Sequence[Sequence[str]], top: float = 1.6, left: float = 0.7, width: float = 12.0,
          col_w: Optional[Sequence[float]] = None, size: int = 12, row_h: float = 0.42) -> float:
    """White table with a blue header row; returns the bottom edge (inches)."""
    shape = slide.shapes.add_table(len(rows) + 1, len(cols), Inches(left), Inches(top), Inches(width), Inches(row_h * (len(rows) + 1)))
    tbl = shape.table
    if col_w:
        for j, w in enumerate(col_w):
            tbl.columns[j].width = Inches(w)
    for j, c in enumerate(cols):
        cell = tbl.cell(0, j)
        cell.text = c
        cell.fill.solid(); cell.fill.fore_color.rgb = BLUE
        for p in cell.text_frame.paragraphs:
            p.font.size, p.font.bold, p.font.color.rgb = Pt(size), True, WHITE
    for i, r in enumerate(rows, 1):
        for j, v in enumerate(r):
            cell = tbl.cell(i, j)
            cell.text = str(v)
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if i % 2 else SOFT
            for p in cell.text_frame.paragraphs:
                p.font.size, p.font.color.rgb = Pt(size), DARK
    return top + row_h * (len(rows) + 1)


def note(slide, text: str, top: float, color: RGBColor = MUTED, size: int = 12, height: float = 1.2) -> None:
    bullets(slide, [text], top=top, size=size, color=color, height=height)


def f3(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


# --------------------------------------------------------------------------------------------- deck

def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    names: List[str] = CFG["classes"]
    tc, mc = CFG["train"], CFG["model"]

    # 1 Title ----------------------------------------------------------------------------------------
    s = new_slide(prs)
    box = s.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.3), Inches(3.6))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Floor-plan understanding with a fine-tuned U-Net"
    p.font.size, p.font.bold, p.font.color.rgb = Pt(34), True, BLUE
    for txt, sz, col, before in [
        ("Walls, doors, windows and rooms from 2D floor plans, as structured JSON for 2D-to-3D processing", 18, DARK, 10),
        ("Pretrained ResNet-18 encoder + U-Net decoder, fine-tuned on real CubiCasa5K plans (Colab T4)", 15, MUTED, 8),
        (("Status: trained and evaluated on held-out test plans; every number comes from saved result files" if (TRAINED and EVALUATED)
          else "Status: code written and unit-tested; NOT trained and NOT evaluated yet. No result numbers appear in this deck."),
         14, GREEN if (TRAINED and EVALUATED) else RED, 26),
    ]:
        q = tf.add_paragraph()
        q.text = txt
        q.font.size, q.font.color.rgb = Pt(sz), col
        q.space_before = Pt(before)

    # 2 Task -----------------------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Task: from a raster floor plan to structured geometry")
    bullets(s, [
        "Input: a 2D floor-plan image (PNG/JPG).",
        "Output: JSON with rooms, walls, doors and windows, each with an id, a type and polygon coordinates in the original image's pixel space.",
        "Why it matters: 2D-to-3D and BIM workflows need explicit geometry, not pixels.",
        "Assessment requirement: genuinely fine-tune a pretrained model on this task and evaluate it on unseen plans.",
        "Integrity rules followed here: no fabricated metrics, weights, logs or predictions; no heuristics presented as AI; no synthetic data presented as real; no invented confidence values.",
    ])

    # 3 Status table (dynamic) -----------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Status at a glance: implemented vs actually run")
    manifest = (RES / "manifest_subset.json").is_file()
    rows = [
        ["Dataset: official CubiCasa5K subset, mask mapping", "Yes (prepare_dataset.py)",
         "Yes: subset used by the saved results" if manifest else "NO: not run on real data in the authoring sandbox",
         "Real SVG inspection and overlay check by a person" if not manifest else "see docs/dataset.md"],
        ["Fine-tuning U-Net + ResNet-18", "Yes (train_segmentation.py)",
         f"Yes: best epoch {TRAIN['best_epoch']}, {TRAIN['optimizer_steps']} optimiser steps" if TRAINED else "NO: not trained",
         "-" if TRAINED else "Colab run (Stage B)"],
        ["Inference: checkpoint to mask to polygon JSON", "Yes (predict_segmentation.py)",
         "Run on test images" if EVALUATED else "Unit-tested only; no model run", "-" if EVALUATED else "Needs a trained checkpoint"],
        ["Evaluation on held-out test plans", "Yes (evaluate_segmentation.py)",
         f"Yes: {EVAL['n_test_images']} test plans" if EVALUATED else "NO: not evaluated", "-" if EVALUATED else "Needs a trained checkpoint"],
        ["Weights outside Git, download link", "Policy in place", "Weights exist" if TRAINED else "No weights exist", "Upload + sha256" ],
        ["OpenCV baseline (not AI)", "Yes, kept separate", "Runs; no benchmark on real plans", "Not part of the learned result"],
        ["Qwen2.5-VL LoRA (original plan)", "Early draft only", "NEVER executed", "Abandoned for this submission"],
    ]
    table(s, ["Component", "Implemented", "Actually run", "Not completed"], rows, col_w=[3.6, 2.6, 3.4, 2.4], size=11, row_h=0.62)

    # 4 Model choice ---------------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Model choice: why a U-Net with a pretrained ResNet-18 encoder")
    bullets(s, [
        "Dense, exact supervision: CubiCasa5K polygons rasterise to per-pixel masks; a VLM would have to emit hundreds of coordinates as text.",
        "Measurable: per-class IoU, Dice/F1, precision, recall and a confusion matrix; no model-side JSON parsing failures.",
        "Fits the hardware and deadline: a small CNN trains on one Colab T4; a 3B VLM with 4-bit LoRA at high resolution is slow, and its draft trainer had never been run.",
        "No invented entities: polygons are extracted from predicted pixels, so every vertex has pixels behind it.",
        "Costs: instances that touch merge; walls are regions, not centre-lines; one Room class (no room types); no confidence scores.",
        "Pretrained part: ResNet-18 with ImageNet weights, fine-tuned end to end. The decoder starts from random weights.",
    ])

    # 5 Dataset provenance ---------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Dataset: official CubiCasa5K")
    src = CFG["source"]
    bullets(s, [
        f"Source: Zenodo record {src['zenodo_record']} (DOI {src['zenodo_doi']}), file {src['zenodo_file']}, 5.5 GB, md5 {src['zenodo_md5']}.",
        "5,000 annotated floor plans with SVG vector annotations (Kalervo et al., SCIA 2019). Official train / val / test lists are used as they are.",
        "Licence: CC BY-NC-SA 4.0 on Zenodo; the official repository says CC BY-NC 4.0. The stricter reading is followed: non-commercial, share-alike, and so are trained weights.",
        f"Subset: {CFG['subset']['n_train']} / {CFG['subset']['n_val']} / {CFG['subset']['n_test']} plans (train / val / test) drawn with seed {CFG['subset']['seed']} from the official lists. The 105 GB LMDB is not built.",
        "Checked so far: record metadata only. The real SVG structure and the image/mask overlays are checked in Colab before any training (a manual gate).",
    ])

    # 6 Mapping -------------------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Explicit source-to-target class mapping (config/segmentation.yaml)")
    table(s, ["Source in model.svg", "Target class", "Rule"], [
        ["<g id=\"Wall\"> polygon", "Wall (2)", "walls under 4 px extent dropped, as in the official loader"],
        ["<g id=\"Door\"> polygon", "Door (3)", "painted over walls"],
        ["<g id=\"Window\"> polygon", "Window (4)", "painted over walls"],
        ["<g class=\"Space <Name>\">, all 59 official names", "Room (1)", "every name listed explicitly"],
        ["Space Outdoor, Space Below150cm, Railing", "ignored (255)", "excluded from loss and metrics"],
        ["Space name not in the config", "sample rejected", "counted in manifest.json; never guessed"],
        ["everything else", "Background (0)", ""],
    ], col_w=[4.6, 2.2, 5.2], size=12)
    note(s, "Paint order: " + ", ".join(CFG["mapping"]["paint_order"]) + ". The mapping was written from the official loader code; the Colab 'inspect' step and a person's overlay check confirm it on the real files.", top=5.3)

    # 7 Splits & preprocessing ------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Splits, duplicate control and preprocessing")
    bullets(s, [
        "Splits: the official lists; nothing is re-split. Test is processed first, then val, then train.",
        f"Duplicates: a val or train candidate is dropped when it matches an earlier split by image sha256, mask hash or perceptual dHash (Hamming <= {CFG['subset']['dhash_max_hamming']}). Every drop is listed in manifest.json.",
        "Leakage guards: id and hash sets must be disjoint; evaluation aborts if a test id was used in training or validation.",
        f"Preprocessing: letterbox to {CFG['image']['size']}x{CFG['image']['size']} (aspect kept; padding white in the image and 255 in the mask), ImageNet normalisation; flips and 90-degree rotations during training.",
        "Limit: hashes catch re-exports of one plan, not two different files of the same building.",
    ])

    # 8 Training configuration -----------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Training configuration (from config/segmentation.yaml)")
    table(s, ["Item", "Value"], [
        ["Architecture", f"{mc['encoder']} encoder (ImageNet weights, fine-tuned) + 4-stage U-Net decoder, {mc['n_classes']} classes"],
        ["Input", f"{CFG['image']['size']}x{CFG['image']['size']} letterboxed"],
        ["Loss", f"cross-entropy ({tc['loss']['ce_class_weights']} class weights) + soft Dice; ignore label {CFG['ignore_index']}"],
        ["Optimiser", f"AdamW, lr {tc['lr']}, weight decay {tc['weight_decay']}, warmup + cosine decay, grad clip 5.0"],
        ["Batch / epochs", f"batch size {tc['batch_size']}, {tc['epochs']} epochs"],
        ["Precision", "fp16 autocast + gradient scaling on CUDA (T4 has no bf16)"],
        ["Model selection", f"best validation {tc['select_best_by']}; checkpoint + history.json + config.json saved"],
        ["Seed", str(CFG["experiment"]["seed"])],
    ], col_w=[2.8, 9.2], size=12)
    note(s, "These are untuned defaults. Gates: no training without manifest.json and a human overlay confirmation; --overfit runs are labelled and refused by evaluation.", top=5.6)

    # 9 Training outcome ----------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Training outcome")
    if TRAINED and HIST:
        last = HIST[-1]
        bullets(s, [
            f"Device: {TRAIN.get('device', 'n/a')}; epochs run: {len(HIST)}; best epoch {TRAIN['best_epoch']} with validation mean IoU {f3(TRAIN['best_val_mean_iou'])}.",
            f"Train loss: {f3(HIST[0]['train_loss'])} (first epoch) to {f3(last['train_loss'])} (last epoch); validation loss at the last epoch {f3(last.get('val_loss'))}.",
            f"Optimiser steps: {TRAIN['optimizer_steps']}; wall-clock {TRAIN['total_seconds']} s; checkpoint sha256 {str(TRAIN.get('checkpoint_sha256', ''))[:16]}...",
            "Source: results/segmentation/train_summary.json and history.json (copied from the Colab run).",
        ])
        curves = RES / "training_curves.png"
        if curves.is_file():
            s.shapes.add_picture(str(curves), Inches(2.2), Inches(3.5), height=Inches(3.6))
    else:
        bullets(s, [
            "No training run exists. There are no loss curves, no checkpoint and no best epoch to report.",
            "Why: the sandbox that produced this code had no GPU, no PyTorch and no access to the dataset host. Training is the Colab notebook, notebooks/colab_segmentation.ipynb.",
            "This slide fills itself from results/segmentation/train_summary.json and history.json once a real run has been copied into the repository.",
        ], color=RED)

    # 10 Inference & JSON ----------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Inference: checkpoint to mask to polygon JSON")
    bullets(s, [
        "image -> letterbox -> U-Net -> crop padding -> resize logits to the original size -> argmax -> class mask.",
        "Per class: connected regions -> Douglas-Peucker simplification -> polygon (+ holes) in ORIGINAL pixel coordinates.",
        "JSON: image, image_width, image_height, units, generated_by (architecture, checkpoint sha256, epoch), rooms / walls / doors / windows; each entity has id, polygon, holes, area_px.",
        "No confidence field: the network is not calibrated, and an invented score would be fabricated data.",
        f"Regions under {CFG['inference']['min_area_frac'] * 100:.2f}% of the image area are dropped (a visible, configurable rule).",
        "The checkpoint is loaded strictly (exact architecture, sha256 recorded); a missing or mismatched checkpoint is an error. There is no heuristic fallback.",
    ])

    # 11 Evaluation protocol --------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Evaluation protocol")
    bullets(s, [
        "Data: the held-out test split of real plans only; ids asserted disjoint from the checkpoint's training and validation ids.",
        "Pixel metrics at original resolution, ignore-labelled pixels excluded: per-class IoU, Dice/F1, precision, recall; confusion matrix; mean IoU / Dice; pixel accuracy.",
        "A class absent from both ground truth and prediction is reported as n/a, never as 0 or 1.",
        "JSON validity: serialise, re-parse and validate against the schema; failures are counted, never repaired.",
        "Inference time: model forward + resize + argmax, and mask-to-JSON, measured separately; two warm-up images excluded.",
        "A limitation note is attached when the test set has fewer than 100 plans.",
    ])

    # 12 Results -------------------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Results: held-out test plans")
    if EVALUATED:
        pc = EVAL["per_class"]
        bottom = table(s, ["Class", "IoU", "Dice / F1", "Precision", "Recall"],
                       [[c, f3(pc[c]["iou"]), f3(pc[c]["dice_f1"]), f3(pc[c]["precision"]), f3(pc[c]["recall"])] for c in EVAL["class_names"]],
                       width=8.0, col_w=[2.4, 1.4, 1.4, 1.4, 1.4], size=13, row_h=0.45)
        t = EVAL.get("inference_time_ms", {}) or {}
        mf = (t.get("model_forward_resize_argmax") or {}).get("mean")
        jm = (t.get("mask_to_json") or {}).get("mean")
        jv = EVAL["json_validity"]
        lines = [
            f"Test plans: {EVAL['n_test_images']}   |   mean IoU {f3(EVAL['mean_iou'])}   |   mean Dice/F1 {f3(EVAL['mean_dice_f1'])}   |   pixel accuracy {f3(EVAL['pixel_accuracy'])}",
            f"JSON validity: {jv['valid']}/{jv['total']}   |   mean model time {'n/a' if mf is None else f'{mf:.1f} ms'}, mask-to-JSON {'n/a' if jm is None else f'{jm:.1f} ms'} on {EVAL.get('device', 'n/a')}",
            f"Checkpoint sha256 {str(EVAL.get('checkpoint_sha256', ''))[:16]}..., epoch {EVAL.get('checkpoint_epoch')}; test ids overlapping training/validation: {EVAL.get('test_ids_overlapping_training_or_validation')}",
        ] + list(EVAL.get("limitations", []))
        bullets(s, lines, top=bottom + 0.3, size=12, color=MUTED, height=2.4)
    else:
        bullets(s, [
            "No evaluation results exist. The model has not been trained, so no metric can be reported.",
            "Earlier numbers (for example Macro F1 1.000) came from leaked or simulated evaluation and were withdrawn; see docs/audit/.",
            "This slide fills itself from results/segmentation/summary.json after a real evaluation on held-out plans.",
        ], color=RED)

    # 13 Confusion matrix ----------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Confusion matrix (row-normalised: ground truth by prediction)")
    if EVALUATED:
        cm = EVAL["confusion_matrix_counts"]
        rows = []
        for i, c in enumerate(EVAL["class_names"]):
            tot = sum(cm[i])
            rows.append([c] + [("n/a" if tot == 0 else f"{100 * v / tot:.1f}%") for v in cm[i]] + [f"{tot:,}"])
        table(s, ["GT \\ predicted"] + EVAL["class_names"] + ["GT pixels"], rows, size=13, row_h=0.5,
              col_w=[2.4] + [1.4] * len(EVAL["class_names"]) + [2.6])
        note(s, "Raw counts: results/segmentation/confusion_matrix.csv. Pixels labelled ignore are excluded.", top=5.2)
    else:
        bullets(s, ["Not available: no evaluation has been run."], color=RED)

    # 14 Example predictions -------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Example predictions: image | ground truth | prediction")
    shown = False
    per_csv = RES / "per_image.csv"
    if EVALUATED and per_csv.is_file():
        with open(per_csv, newline="", encoding="utf-8") as f:
            per = list(csv.DictReader(f))
        scored = [r for r in per if r.get("mean_iou") not in (None, "")]
        picks = []
        if per:
            picks.append(("First test image (list order)", per[0]["image"]))
        if scored:
            worst = min(scored, key=lambda r: float(r["mean_iou"]))
            if worst["image"] != picks[0][1]:
                picks.append((f"Lowest mean IoU in the test set ({float(worst['mean_iou']):.3f})", worst["image"]))
        y = 1.5
        for label, name in picks[:2]:
            img = RES / "visuals" / f"{name}.png"
            if img.is_file():
                note(s, label, top=y, color=DARK, size=11, height=0.3)
                s.shapes.add_picture(str(img), Inches(1.2), Inches(y + 0.3), width=Inches(10.9))
                y += 2.8
                shown = True
    if not shown:
        bullets(s, ["No prediction images exist: no model has been trained or run.",
                    "After a real run, this slide shows the first test image and the lowest-scoring test image, which avoids hand-picking flattering examples."], color=RED)

    # 15 Limitations ---------------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Limitations and what is not done")
    bullets(s, [
        "Not done: " + ("see results/segmentation for what was run." if (TRAINED and EVALUATED) else "training, evaluation, sample predictions and hosted weights (they need the Colab T4 run)."),
        "The class mapping comes from the official loader code until the real-file inspection and the overlay check have been done.",
        "Small subset and a small test split: results are indicative, with wide uncertainty.",
        "Hash-based duplicate control cannot prove that two different files do not show the same building.",
        "Semantic segmentation: touching instances merge; walls are regions, not centre-lines; no room types; no confidence values; no CAD snapping on the learned output.",
        "Non-commercial / share-alike licence applies to the data and therefore to the trained weights.",
        "The OpenCV baseline is kept only as a labelled comparison point; the Qwen2.5-VL plan was never executed.",
    ], size=14)

    # 16 Reproducibility ----------------------------------------------------------------------------
    s = new_slide(prs)
    header(s, "Reproducibility and deliverables")
    bullets(s, [
        "Repository: github.com/sandhiya-1234/FloorPlan-AI, branch repair/real-vlm-pipeline (main untouched).",
        "Colab: notebooks/colab_segmentation.ipynb (Stage A data and overlays, manual gate, Stage B training and evaluation).",
        "Code: training/prepare_dataset.py, train_segmentation.py; inference/predict_segmentation.py; evaluation/evaluate_segmentation.py; config/segmentation.yaml.",
        "Dependencies: requirements-seg.txt for the experiment; dashboard and Qwen requirements are separate files.",
        "Weights: kept outside Git. " + ("Location and sha256 are in the README." if TRAINED else "None exist yet; the link and sha256 will be added after a real run."),
        "Tests: python -m pytest tests -q -rs (skipped tests are reported as not run, not as passed).",
    ])

    out = ROOT / "presentation" / "Floorplan_AI_Assessment.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


if __name__ == "__main__":
    print(f"Generated {build()}  | trained={TRAINED} evaluated={EVALUATED}")
