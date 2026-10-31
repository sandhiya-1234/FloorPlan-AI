"""
Batch Architectural Floor Plan Predictor for FloorPlan-AI.

Runs high-throughput batch prediction over image directories, applying the fine-tuned
adapter model, postprocessing geometry, and saving structured JSON and visualizations.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.predict import predict_floorplan

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_batch_inference(
    input_dir: str,
    output_dir: str = "outputs/predictions",
    adapter_path: str = "artifacts/adapters/final_lora_adapter",
    save_vis: bool = True
) -> Dict[str, Any]:
    """Runs prediction across all floor plan images in input directory."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(in_path.glob(ext)))
    image_files = sorted(image_files)

    logger.info(f"Discovered {len(image_files)} floor plan images in {in_path}")

    if not image_files:
        logger.warning(f"No images found in {in_path}!")
        return {"total_images": 0, "successful": 0}

    results = []
    start_total = time.time()

    for idx, img_f in enumerate(image_files, 1):
        target_json = out_path / f"{img_f.stem}.json"
        t0 = time.time()
        try:
            schema = predict_floorplan(
                image_path=str(img_f),
                adapter_path=adapter_path,
                output_json_path=str(target_json),
                save_vis=save_vis
            )
            lat = time.time() - t0
            results.append({
                "image": img_f.name,
                "status": "success",
                "latency_sec": round(lat, 3),
                "walls_detected": len(schema.walls),
                "doors_detected": len(schema.doors),
                "windows_detected": len(schema.windows),
                "rooms_detected": len(schema.rooms)
            })
            logger.info(f"[{idx}/{len(image_files)}] {img_f.name} processed in {lat:.3f}s")
        except Exception as e:
            logger.error(f"Error processing {img_f.name}: {e}")
            results.append({
                "image": img_f.name,
                "status": "failed",
                "error": str(e)
            })

    total_time = time.time() - start_total
    successful_count = sum(1 for r in results if r["status"] == "success")
    avg_latency = total_time / len(image_files) if image_files else 0.0

    summary = {
        "input_directory": str(in_path),
        "output_directory": str(out_path),
        "total_images": len(image_files),
        "successful_predictions": successful_count,
        "failed_predictions": len(image_files) - successful_count,
        "total_time_seconds": round(total_time, 2),
        "average_latency_seconds": round(avg_latency, 3),
        "throughput_fps": round(len(image_files) / max(0.001, total_time), 2),
        "predictions": results
    }

    summary_file = out_path / "batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Batch inference complete! Summary saved to {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch inference on floor plans.")
    parser.add_argument("--input-dir", type=str, default="dataset/test", help="Directory with floor plan images")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Directory for JSON predictions")
    parser.add_argument("--adapter", type=str, default="artifacts/adapters/final_lora_adapter", help="Adapter path")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualizations")
    args = parser.parse_args()

    run_batch_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        adapter_path=args.adapter,
        save_vis=not args.no_vis
    )


if __name__ == "__main__":
    main()
