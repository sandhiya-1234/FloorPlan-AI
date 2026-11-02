"""
Fine-Tuning Engine for FloorPlan-AI: Vision-Language Model Adaptation.

STATUS: GPU-only. Has never been executed on a GPU; a verified rewrite (scheduler, mixed precision,
resumable checkpoints, run manifest) is Phase 3. Do not cite any result from this file until then.

Orchestrates LoRA/QLoRA parameter-efficient fine-tuning on architectural floor plans.
Calculates ground-truth cross-entropy loss, evaluates validation loss, checkpoints models,
and saves the trained PEFT adapter for downstream inference.
"""

import os
import sys
import json
import time
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import FloorPlanTrainingConfig
from training.dataset import FloorPlanVLMDataset
from training.collator import FloorPlanDataCollator
from training.model_loader import check_cuda_environment, load_model_and_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_training(config: FloorPlanTrainingConfig):
    """
    Executes actual training process with ground-truth loss calculation,
    validation loss tracking, checkpointing, and adapter saving.
    """
    logger.info("=" * 60)
    logger.info("STARTING FLOORPLAN-AI VLM FINE-TUNING PIPELINE")
    logger.info("=" * 60)

    # 1. Audit hardware
    hw = check_cuda_environment()
    if not hw["has_cuda"]:
        raise RuntimeError(
            "Qwen2.5-VL LoRA training requires a CUDA GPU (use the Google Colab T4, see COLAB.md). "
            "There is deliberately no CPU fallback: a toy model must never be saved as the project's adapter."
        )
    device = torch.device("cuda")
    logger.info(f"Target Execution Device: {device}")

    # Create artifact directories
    checkpoints_dir = Path(config.training.output_dir)
    adapters_dir = Path(config.training.adapter_output_dir)
    logs_dir = Path(config.training.logs_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 2. Qwen2.5-VL-3B-Instruct + LoRA (CUDA GPU only)
    logger.info(f"Base model: {config.model.base_model_name}")
    model, processor = load_model_and_processor(config.model, config.lora, is_training=True)
    collator = FloorPlanDataCollator(processor)

    # 3. Load datasets
    train_dataset = FloorPlanVLMDataset(config.train_dir, processor=processor)
    val_dataset = FloorPlanVLMDataset(config.val_dir, processor=processor)

    if len(train_dataset) == 0:
        logger.error(f"No training samples found in {config.train_dir}! Please run preprocessing first.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator
    )

    # 4. Optimizer and Learning Rate Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    total_steps = len(train_loader) * config.training.num_train_epochs
    warmup_steps = max(1, int(total_steps * config.training.warmup_ratio))

    logger.info(f"Training Plan: Epochs={config.training.num_train_epochs}, Total Steps={total_steps}, Warmup={warmup_steps}")

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "step_losses": []
    }

    best_val_loss = float("inf")
    start_time = time.time()
    global_step = 0

    # 5. Training Loop
    for epoch in range(1, config.training.num_train_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        logger.info(f"--- Epoch {epoch}/{config.training.num_train_epochs} ---")
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            # Move to device
            batch_inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Forward pass
            outputs = model(**batch_inputs)
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss

            if loss is None or torch.isnan(loss):
                logger.warning(f"Warning: NaN loss encountered at step {global_step}")
                continue

            # Scale loss for gradient accumulation
            accum_loss = loss / config.training.gradient_accumulation_steps
            accum_loss.backward()

            if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_val = float(loss.item())
            epoch_loss += loss_val
            step_count += 1
            history["step_losses"].append({"step": global_step, "loss": round(loss_val, 4)})

            if (batch_idx + 1) % config.training.logging_steps == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(f"Step {global_step} (Epoch {epoch} [{batch_idx+1}/{len(train_loader)}]): Train Loss = {loss_val:.4f}")

        avg_train_loss = epoch_loss / max(1, step_count)

        # 6. Validation loop
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_batch in val_loader:
                v_inputs = {k: v.to(device) for k, v in val_batch.items() if isinstance(v, torch.Tensor)}
                v_outputs = model(**v_inputs)
                v_loss = v_outputs.get("loss") if isinstance(v_outputs, dict) else v_outputs.loss
                if v_loss is not None and not torch.isnan(v_loss):
                    val_loss_total += float(v_loss.item())
                    val_steps += 1

        if val_steps == 0:
            # Never invent a validation loss: best-adapter selection depends on it.
            raise RuntimeError(
                f"Validation produced no valid loss in epoch {epoch} (empty validation set or all-NaN loss). "
                "Fix the data/config instead of continuing with a made-up number."
            )
        avg_val_loss = val_loss_total / val_steps

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))

        logger.info(f"Epoch {epoch} Result: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Checkpoint per epoch
        epoch_checkpoint_path = checkpoints_dir / f"checkpoint-epoch-{epoch}"
        epoch_checkpoint_path.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(epoch_checkpoint_path))

        # Save Best Adapter
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving best adapter...")
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(str(adapters_dir))
            else:
                torch.save(model.state_dict(), adapters_dir / "adapter_model.bin")

    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.1f} seconds. Best Val Loss = {best_val_loss:.4f}")

    # 7. Save Training Logs and Loss Curve
    history["total_training_time_sec"] = round(elapsed_time, 2)
    history["best_val_loss"] = round(best_val_loss, 4)
    history["final_train_loss"] = round(history["train_loss"][-1], 4)

    with open(logs_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Also export to CSV for easy inspection
    csv_path = logs_dir / "training_history.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for ep, tl, vl in zip(history["epochs"], history["train_loss"], history["val_loss"]):
            f.write(f"{ep},{tl},{vl}\n")
    logger.info(f"Training history CSV saved to {csv_path}")

    # Plot Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(history["epochs"], history["train_loss"], marker="o", label="Training Loss", color="#1E40AF", lw=2)
    plt.plot(history["epochs"], history["val_loss"], marker="s", label="Validation Loss", color="#DC2626", lw=2, linestyle="--")
    plt.title("FloorPlan-AI: VLM Fine-Tuning Loss Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Cross-Entropy Loss", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    loss_plot_path = logs_dir / "training_loss.png"
    plt.savefig(loss_plot_path, dpi=300)
    # Also save to outputs/evaluation for evaluation report
    eval_loss_plot = Path("outputs/evaluation/training_loss.png")
    eval_loss_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(eval_loss_plot, dpi=300)
    plt.close()

    logger.info(f"Loss curves saved to {loss_plot_path} and {eval_loss_plot}")
    logger.info("LoRA Adapter weights and training artifacts successfully saved!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLM on architectural floor plans.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    cfg = FloorPlanTrainingConfig.from_yaml(args.config)
    if args.epochs:
        cfg.training.num_train_epochs = args.epochs
    if args.lr:
        cfg.training.learning_rate = args.lr

    run_training(cfg)


if __name__ == "__main__":
    main()
