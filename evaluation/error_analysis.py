"""
Error Analysis & Diagnostic Suite for FloorPlan-AI.

Performs granular per-sample and per-class error breakdowns:
- Analyzes True Positives, False Positives (over-detection), and False Negatives (missed elements)
- Quantifies pixel-level spatial coordinate error distributions
- Evaluates geometric postprocessing ablation impact
- Generates outputs/evaluation/error_analysis.png
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

# Set Matplotlib temp dir
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "C:/Temp")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_error_analysis_report(
    metrics_json_path: str = "outputs/evaluation/metrics.json",
    output_png_path: str = "outputs/evaluation/error_analysis.png"
) -> Dict[str, Any]:
    """
    Parses evaluation results and renders a 4-panel diagnostic chart.
    """
    m_path = Path(metrics_json_path)
    if not m_path.exists():
        logger.error(f"Metrics file {m_path} does not exist. Run evaluation/evaluate.py first.")
        return {}

    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_p = Path(output_png_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ft_model = data["fine_tuned_model"]["class_metrics"]
    base_model = data["base_model"]["class_metrics"]
    per_image = data.get("per_image_results", [])

    classes = ["Walls", "Doors", "Windows", "Rooms"]
    class_keys = ["walls", "doors", "windows", "rooms"]

    # Setup 4-panel visualization figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Per-Sample F1 Scores
    ax1 = axes[0, 0]
    sample_names = [item["image"] for item in per_image]
    if sample_names:
        x_indices = np.arange(len(sample_names))
        bar_width = 0.2

        for i, (c_label, c_key) in enumerate(zip(classes, class_keys)):
            scores = [
                item["fine_tuned_metrics"].get(c_key, {}).get("f1", 0.0)
                for item in per_image
            ]
            ax1.bar(x_indices + i * bar_width, scores, bar_width, label=c_label)

        ax1.set_title("A. Per-Sample F1-Score Breakdown (Unseen Test Set)", fontsize=11, fontweight="bold")
        ax1.set_xticks(x_indices + 1.5 * bar_width)
        ax1.set_xticklabels(sample_names, fontsize=9)
        ax1.set_ylabel("F1 Score", fontsize=10)
        ax1.set_ylim(0, 1.15)
        ax1.grid(axis="y", linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Detection Outcome Breakdown (TP, FP, FN) for Calibrated Model
    ax2 = axes[0, 1]
    tp_vals = [ft_model[k]["total_tp"] for k in class_keys]
    fp_vals = [ft_model[k]["total_fp"] for k in class_keys]
    fn_vals = [ft_model[k]["total_fn"] for k in class_keys]

    x = np.arange(len(classes))
    width = 0.25

    ax2.bar(x - width, tp_vals, width, label="True Positives (TP)", color="#10B981")
    ax2.bar(x, fp_vals, width, label="False Positives (FP)", color="#EF4444")
    ax2.bar(x + width, fn_vals, width, label="False Negatives (FN)", color="#F59E0B")

    ax2.set_title("B. Detection Outcomes: TP vs. FP vs. FN by Class", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=10)
    ax2.set_ylabel("Element Count", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    # Annotate counts
    for i in range(len(classes)):
        ax2.text(x[i] - width, tp_vals[i] + 0.3, str(tp_vals[i]), ha="center", fontsize=8, fontweight="bold")
        ax2.text(x[i], fp_vals[i] + 0.3, str(fp_vals[i]), ha="center", fontsize=8)
        ax2.text(x[i] + width, fn_vals[i] + 0.3, str(fn_vals[i]), ha="center", fontsize=8)

    # Panel 3: Spatial Coordinate Error Comparison (Pixels)
    ax3 = axes[1, 0]
    base_errs = [base_model[k].get("mean_coord_error_px", 0.0) for k in class_keys]
    ft_errs = [ft_model[k].get("mean_coord_error_px", 0.0) for k in class_keys]

    w_err = 0.35
    ax3.bar(x - w_err / 2, base_errs, w_err, label="Raw Baseline", color="#94A3B8")
    ax3.bar(x + w_err / 2, ft_errs, w_err, label="Calibrated Pipeline", color="#3B82F6")

    ax3.set_title("C. Spatial Coordinate Error: Mean Deviation (Pixels)", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, fontsize=10)
    ax3.set_ylabel("Error (px)", fontsize=10)
    ax3.grid(axis="y", linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=9)

    for i in range(len(classes)):
        ax3.text(x[i] - w_err / 2, base_errs[i] + 0.2, f"{base_errs[i]:.1f}", ha="center", fontsize=8)
        ax3.text(x[i] + w_err / 2, ft_errs[i] + 0.2, f"{ft_errs[i]:.1f}", ha="center", fontsize=8, fontweight="bold")

    # Panel 4: Postprocessing Ablation Impact (Delta F1)
    ax4 = axes[1, 1]
    f1_deltas = [
        round(ft_model[k]["f1"] - base_model[k]["f1"], 4)
        for k in class_keys
    ]
    colors = ["#10B981" if d >= 0 else "#EF4444" for d in f1_deltas]

    bars = ax4.bar(classes, f1_deltas, width=0.45, color=colors)
    ax4.axhline(0, color="#64748B", linewidth=0.8, linestyle="--")
    ax4.set_title("D. Geometric Postprocessing Gain (Delta F1)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Delta F1 (Calibrated - Baseline)", fontsize=10)
    ax4.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, d in zip(bars, f1_deltas):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax4.annotate(f"{d:+.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                     ha="center", va=va, fontsize=9, fontweight="bold")

    plt.suptitle("FloorPlan-AI: Rigorous Diagnostic Error Analysis & Ablation Breakdown", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved diagnostic error analysis visualization to: {out_p}")
    return {"status": "success", "output_path": str(out_p)}


def main():
    generate_error_analysis_report()


if __name__ == "__main__":
    main()
