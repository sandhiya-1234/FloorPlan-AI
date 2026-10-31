"""
FloorPlan-AI: Streamlit Application.

Interactive visual dashboard for fine-tuned Vision-Language Model 2D floor-plan detection.
Supports single-image upload, interactive visualization, Pydantic JSON inspection,
and model benchmark exploration.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas.floorplan_schema import FloorPlanSchema
from inference.predict import predict_floorplan
from preprocessing.visualize_annotations import draw_floorplan_annotations

# Set page configuration
st.set_page_config(
    page_title="FloorPlan-AI | Architectural Detection",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal, clean styling
st.markdown("""
<style>
    .main { background-color: #FAFAFA; font-family: 'Inter', -apple-system, sans-serif; }
    h1, h2, h3 { color: #0F172A; font-weight: 700; }
    .stMetric { background-color: #FFFFFF; padding: 16px; border-radius: 8px; border: 1px solid #E2E8F0; }
    .element-card { background: white; padding: 1.2rem; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 1rem; }
    .badge-wall { background-color: #1E293B; color: white; padding: 2px 8px; border-radius: 4px; }
    .badge-door { background-color: #F59E0B; color: white; padding: 2px 8px; border-radius: 4px; }
    .badge-window { background-color: #06B6D4; color: white; padding: 2px 8px; border-radius: 4px; }
    .badge-room { background-color: #3B82F6; color: white; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


def load_metrics_data():
    metrics_path = Path("outputs/evaluation/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    st.title("📐 FloorPlan-AI: 2D Architectural Element Detection")
    st.markdown(
        "**Fine-Tuned Vision-Language Model (Qwen2.5-VL-3B with LoRA)** for automated parsing of "
        "architectural walls, doors, windows, and rooms into CAD-ready structured JSON."
    )

    # Sidebar Navigation & Settings
    st.sidebar.header("Navigation & Model")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["Floor Plan Detector", "Evaluation Benchmark", "Dataset Explorer", "System Architecture"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    adapter_path = Path("artifacts/adapters/final_lora_adapter")
    if adapter_path.exists():
        st.sidebar.success("LoRA Adapter: Ready (Loaded)")
    else:
        st.sidebar.warning("LoRA Adapter: Missing (Run train.py)")

    st.sidebar.info("Model: **Qwen2.5-VL-3B-Instruct**\nTuning: **LoRA (r=16, alpha=32)**\nBackend: **PyTorch + Peft**")

    # Mode 1: Detector
    if app_mode == "Floor Plan Detector":
        st.subheader("1. Select or Upload Floor Plan Image")

        col_select, col_upload = st.columns([1, 1])

        sample_choice = col_select.selectbox(
            "Select Sample Benchmark Image:",
            ["samples/sample1.png", "samples/sample2.png", "samples/sample3.png"]
        )

        uploaded_file = col_upload.file_uploader(
            "Or Upload Floor Plan (PNG / JPG):",
            type=["png", "jpg", "jpeg"]
        )

        # Determine target image
        temp_img_path = Path("samples/temp_input.png")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image.save(temp_img_path)
            active_image_path = str(temp_img_path)
        else:
            active_image_path = sample_choice

        # Display raw image
        raw_img = cv2.imread(active_image_path)
        if raw_img is not None:
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### Input Floor Plan")
                st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            run_btn = st.button("🚀 Run Fine-Tuned Model Inference", type="primary")

            if run_btn or "last_schema" in st.session_state:
                with st.spinner("Executing LoRA adapter inference and geometric postprocessing..."):
                    schema = predict_floorplan(
                        image_path=active_image_path,
                        adapter_path="artifacts/adapters/final_lora_adapter",
                        save_vis=False
                    )
                    st.session_state["last_schema"] = schema
                    st.session_state["last_img_path"] = active_image_path

                with col_right:
                    st.markdown("#### Detected Architectural Elements")
                    annotated = draw_floorplan_annotations(raw_img.copy(), schema)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.markdown("---")
                st.subheader("Detected Architectural Counts")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Walls Detected", len(schema.walls))
                m2.metric("Doors Detected", len(schema.doors))
                m3.metric("Windows Detected", len(schema.windows))
                m4.metric("Rooms Detected", len(schema.rooms))

                # Tabs for detailed results
                tab_json, tab_rooms, tab_walls = st.tabs(["Structured JSON Output", "Rooms Detail", "Walls & Openings"])

                with tab_json:
                    json_str = json.dumps(schema.to_compact_dict(), indent=2)
                    st.download_button(
                        label="📥 Download Structured Floor Plan JSON",
                        data=json_str,
                        file_name=f"{Path(active_image_path).stem}_detected.json",
                        mime="application/json"
                    )
                    st.json(schema.to_compact_dict())

                with tab_rooms:
                    room_data = [
                        {
                            "Room ID": r.id,
                            "Classification": r.type.upper(),
                            "Vertices": len(r.polygon),
                            "Area (sq px)": f"{r.area_pixels:,.0f}" if r.area_pixels else "N/A",
                            "Confidence": f"{r.confidence:.2f}"
                        }
                        for r in schema.rooms
                    ]
                    st.dataframe(pd.DataFrame(room_data), use_container_width=True)

                with tab_walls:
                    w_data = [
                        {
                            "Wall ID": w.id,
                            "Start [X, Y]": f"[{w.start[0]:.1f}, {w.start[1]:.1f}]",
                            "End [X, Y]": f"[{w.end[0]:.1f}, {w.end[1]:.1f}]",
                            "Thickness (px)": w.thickness,
                            "Confidence": f"{w.confidence:.2f}"
                        }
                        for w in schema.walls
                    ]
                    st.dataframe(pd.DataFrame(w_data), use_container_width=True)

    # Mode 2: Evaluation Benchmark
    elif app_mode == "Evaluation Benchmark":
        st.subheader("Model Evaluation & Zero-Shot Benchmark Comparison")
        st.markdown(
            "Evaluation conducted on an **unseen held-out test split** (partitioned strictly by image ID to avoid data leakage)."
        )

        metrics = load_metrics_data()
        if metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Fine-Tuned Model (LoRA)")
                ft = metrics["fine_tuned_model"]
                st.metric("Macro F1-Score", f"{ft['macro_f1']:.4f}")
                st.metric("JSON Validity Rate", f"{ft['json_validity_rate']*100:.0f}%")
                st.metric("Pydantic Schema Validation", f"{ft['schema_validation_rate']*100:.0f}%")

            with col2:
                st.markdown("#### Pretrained Base Model (Zero-Shot)")
                base = metrics["base_model"]
                st.metric("Macro F1-Score", f"{base['macro_f1']:.4f}")
                st.metric("JSON Validity Rate", f"{base['json_validity_rate']*100:.0f}%")
                st.metric("Pydantic Schema Validation", f"{base['schema_validation_rate']*100:.0f}%")

            st.markdown("---")
            st.markdown("#### Class-wise Performance Comparison")
            comp_img_path = Path("outputs/evaluation/comparison.png")
            if comp_img_path.exists():
                st.image(str(comp_img_path), use_container_width=True)

            csv_path = Path("outputs/evaluation/classification_report.csv")
            if csv_path.exists():
                st.markdown("#### Detailed Classification Report Table")
                df_rep = pd.read_csv(csv_path)
                st.dataframe(df_rep, use_container_width=True)

            err_img_path = Path("outputs/evaluation/error_analysis.png")
            if err_img_path.exists():
                st.markdown("#### Diagnostic Error Analysis & Ablation Breakdown")
                st.image(str(err_img_path), use_container_width=True)

            loss_img_path = Path("outputs/evaluation/training_loss.png")
            if loss_img_path.exists():
                st.markdown("#### Training Loss Convergence (LoRA Adaptation)")
                st.image(str(loss_img_path), use_container_width=True)
        else:
            st.info("Run `python evaluation/evaluate.py` to generate complete evaluation metrics.")

    # Mode 3: Dataset Explorer
    elif app_mode == "Dataset Explorer":
        st.subheader("CubiCasa5K Architectural Dataset Exploration")
        st.markdown(
            "CubiCasa5K provides 5,000 architectural floor plans with dense vector annotations "
            "(spaces, walls, doors, windows). Our pipeline converts raw SVG primitives into normalized coordinates."
        )

        processed_dir = Path("dataset/processed")
        if processed_dir.exists():
            sample_pngs = sorted(list(processed_dir.glob("*.png")))
            st.write(f"Total Processed Floor Plans: **{len(sample_pngs)}**")

            if sample_pngs:
                pick = st.selectbox("Select Processed Sample:", [p.name for p in sample_pngs])
                chosen_png = processed_dir / pick
                chosen_json = chosen_png.with_suffix(".json")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(str(chosen_png), caption=f"Processed Image ({pick})", use_container_width=True)
                with c2:
                    if chosen_json.exists():
                        with open(chosen_json, "r") as f:
                            gt_d = json.load(f)
                        st.markdown(f"**Elements in {pick}:**")
                        st.write(f"- Walls: {len(gt_d.get('walls', []))}")
                        st.write(f"- Doors: {len(gt_d.get('doors', []))}")
                        st.write(f"- Windows: {len(gt_d.get('windows', []))}")
                        st.write(f"- Rooms: {len(gt_d.get('rooms', []))}")
                        st.json(gt_d)

    # Mode 4: Architecture
    elif app_mode == "System Architecture":
        st.subheader("FloorPlan-AI System Architecture")
        st.markdown("""
```
Floor Plan Image (PNG/JPG)
       │
       ▼
[Image Preprocessing & Resolution Normalization]
  • Contrast enhancement (CLAHE)
  • Aspect ratio preservation
       │
       ▼
[Fine-Tuned Vision-Language Model]
  • Base: Qwen2.5-VL-3B-Instruct
  • LoRA: r=16, alpha=32, target_modules=[q, k, v, o, gate, up, down]
       │
       ▼
[Architectural Element Identification]
  • Multimodal conversational prompting
  • Next-token autoregressive generation
       │
       ▼
[Geometric Postprocessing Engine]
  • Orthogonal wall angle snapping
  • Topology validation (Shapely)
  • Non-Maximum Suppression (IoU) duplicate suppression
  • Door / Window centerline wall alignment
       │
       ▼
[Pydantic JSON Validation]
  • Boundary enforcement
  • Unique ID assignment
       │
       ▼
Structured Floor Plan JSON (CAD/BIM Ready)
```
        """)


if __name__ == "__main__":
    main()
