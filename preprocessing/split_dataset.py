"""
Dataset Splitting Script for FloorPlan-AI.

Splits processed floor-plan image and JSON pairs into train, validation, and test
splits by image ID (preventing data leakage) using a deterministic random seed.
"""

import os
import sys
import shutil
import random
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_dataset(
    processed_dir: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Deterministically partitions floor plan pairs into train, validation, and test splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    random.seed(seed)

    # Find all sample IDs (files named <id>.png and <id>.json)
    png_files = sorted(list(processed_dir.glob("*.png")))
    sample_ids = []
    for p in png_files:
        json_file = p.with_suffix(".json")
        if json_file.exists():
            sample_ids.append(p.stem)

    logger.info(f"Found {len(sample_ids)} complete image-JSON sample pairs in {processed_dir}")

    if not sample_ids:
        logger.warning(f"No processed pairs found in {processed_dir}. Run convert_annotations.py first.")
        return

    # Shuffle deterministically
    random.shuffle(sample_ids)

    n_total = len(sample_ids)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train:n_train + n_val]
    test_ids = sample_ids[n_train + n_val:]

    # Ensure at least one test and val sample if n_total >= 3
    if n_total >= 3:
        if not val_ids and train_ids:
            val_ids.append(train_ids.pop())
        if not test_ids and train_ids:
            test_ids.append(train_ids.pop())

    logger.info(f"Partition counts: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Clear/create destination directories
    for split_dir, split_ids in [(train_dir, train_ids), (val_dir, val_ids), (test_dir, test_ids)]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for s_id in split_ids:
            src_png = processed_dir / f"{s_id}.png"
            src_json = processed_dir / f"{s_id}.json"
            dst_png = split_dir / f"{s_id}.png"
            dst_json = split_dir / f"{s_id}.json"

            shutil.copy2(src_png, dst_png)
            shutil.copy2(src_json, dst_json)

    logger.info("Dataset split successfully completed without data leakage.")


def main():
    parser = argparse.ArgumentParser(description="Split processed floor plan dataset.")
    parser.add_argument("--processed-dir", type=str, default="dataset/processed", help="Processed directory")
    parser.add_argument("--train-dir", type=str, default="dataset/train", help="Train directory")
    parser.add_argument("--val-dir", type=str, default="dataset/validation", help="Validation directory")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Test directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split_dataset(
        Path(args.processed_dir),
        Path(args.train_dir),
        Path(args.val_dir),
        Path(args.test_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
