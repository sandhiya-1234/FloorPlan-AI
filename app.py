"""
FloorPlan-AI: Streamlit dashboard.

This app does NOT run Qwen2.5-VL itself. It shows:
  * PRECOMPUTED example predictions (clearly labelled, produced offline on Colab), and
  * verified evaluation results read from outputs/evaluation/summary.json.
A "live inference" button is offered only when a verified LoRA adapter exists on disk, and any
failure is shown as an error, never replaced by a fake result.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference.adapter_check import AdapterError, inspect_adapter
from inference.predict import DEFAULT_ADAPTER_DIR, DEFAULT_BASE_MODEL, predict_floorplan
from inference.records import load_record
from preprocessing.visualize_annotations import draw_floorplan_annotations
from schemas.floorplan_schema import FloorPlanSchema

EXAMPLES_DIR = Path("outputs/predictions/examples")
SUMMARY_PATH = Path("outputs/evaluation/summary.json")

st.set_page_config(page_title="FloorPlan-AI", page_icon="📐", layout="wide", initial_sidebar_state="expanded")


def fmt_conf(value) -> str:
    return "not reported" if value is None else f"{value:.2f}"


def adapter_status():
    """(ok, info_or_reason). Verifies files on disk only; it does not load any model."""
    try:
        return True, inspect_adapter(DEFAULT_ADAPTER_DIR, expected_base_model=DEFAULT_BASE_MODEL)
    except AdapterError as e:
        return False, str(e)


def show_prediction(image_bgr: np.ndarray, schema: FloorPlanSchema) -> None:
    left, right = st.columns(2)
    left.markdown("#### Input floor plan")
    left.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    right.markdown("#### Predicted elements")
    annotated = draw_floorplan_annotations(image_bgr.copy(), schema)
    right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Walls", len(schema.walls))
    c2.metric("Doors", len(schema.doors))
    c3.metric("Windows", len(schema.windows))
    c4.metric("Rooms", len(schema.rooms))

    tab_json, tab_rooms, tab_walls = st.tabs(["JSON", "Rooms", "Walls"])
    with tab_json:
        payload = schema.to_compact_dict()
        st.download_button("Download JSON", json.dumps(payload, indent=2), "floorplan_prediction.json", "application/json")
        st.json(payload)
    with tab_rooms:
        st.dataframe(pd.DataFrame([
            {"id": r.id, "type": r.type, "vertices": len(r.polygon), "area_px": r.area_pixels,
             "confidence": fmt_conf(r.confidence)} for r in schema.rooms]), use_container_width=True)
    with tab_walls:
        st.dataframe(pd.DataFrame([
            {"id": w.id, "start": w.start, "end": w.end, "thickness_px": w.thickness,
             "confidence": fmt_conf(w.confidence)} for w in schema.walls]), use_container_width=True)


def page_detector(adapter_ok: bool) -> None:
    st.subheader("Floor plan detector")

    st.markdown("##### Precomputed examples")
    st.caption("Produced offline by the evaluated system; **not** computed live by this app.")
    records = sorted(EXAMPLES_DIR.glob("*.pred.json")) if EXAMPLES_DIR.exists() else []
    if not records:
        st.info("No precomputed examples have been published yet (outputs/predictions/examples/).")
    else:
        pick = st.selectbox("Example", [r.name.replace(".pred.json", "") for r in records])
        rec = load_record(EXAMPLES_DIR / f"{pick}.pred.json")
        img_path = EXAMPLES_DIR / f"{pick}.png"
        st.markdown(f"**System:** `{rec['system']}`  |  **Status:** {rec['status']}")
        with st.expander("Run metadata"):
            st.json(rec.get("meta", {}))
        if rec["status"] == "failed":
            st.error(f"This example failed at stage '{rec['failure_stage']}': {rec['error']}")
        elif img_path.exists():
            show_prediction(cv2.imread(str(img_path)), FloorPlanSchema(**rec["prediction"]))
        else:
            st.warning(f"Image {img_path.name} is missing next to the record.")

    st.markdown("---")
    st.markdown("##### Live inference")
    uploaded = st.file_uploader("Upload a floor plan (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if not adapter_ok:
        st.warning("Live inference is disabled: no verified LoRA adapter (see sidebar). "
                   "Free Streamlit hosting also has no GPU; live inference needs a GPU backend.")
        return
    if uploaded is not None and st.button("Run live inference"):
        tmp = Path("outputs/predictions/_upload.png")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        Image.open(uploaded).convert("RGB").save(tmp)
        try:
            schema = predict_floorplan(str(tmp), save_vis=False)
        except Exception as e:  # noqa: BLE001 - shown to the user, never replaced by fake output
            st.error(f"Inference failed: {type(e).__name__}: {e}")
        else:
            st.session_state["live"] = (cv2.imread(str(tmp)), schema)
    if "live" in st.session_state:
        st.caption("LIVE result")
        show_prediction(*st.session_state["live"])


def page_evaluation() -> None:
    st.subheader("Evaluation (verified results only)")
    if not SUMMARY_PATH.exists():
        st.info("No verified evaluation results yet. Earlier numbers were invalid and have been withdrawn "
                "(see docs/audit/). Results appear here after the three systems are evaluated on unseen plans.")
        return
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    st.write(f"Unseen test floor plans: **{summary['n_test_images']}**")
    for note in summary.get("limitations", []):
        st.warning(note)
    st.dataframe(pd.DataFrame([{
        "system": s["label"], "failed": f"{s['n_failed']}/{summary['n_test_images']}",
        "JSON valid": s["json_validity_rate"], "wall F1": s["f1"]["walls"], "door F1": s["f1"]["doors"],
        "window F1": s["f1"]["windows"], "room F1": s["f1"]["rooms"], "macro F1": s["macro_f1"],
        "room type acc.": s.get("room_type_accuracy")} for s in summary["systems"]]), use_container_width=True)


def page_dataset() -> None:
    st.subheader("Dataset explorer")
    manifest = Path("dataset/processed/DATASET_MANIFEST.json")
    if not manifest.exists():
        st.warning("No verified dataset manifest found: the contents of dataset/ are unverified "
                   "and must not be treated as CubiCasa5K.")
        return
    st.json(json.loads(manifest.read_text(encoding="utf-8")))


def main() -> None:
    st.title("📐 FloorPlan-AI")
    st.markdown("Qwen2.5-VL-3B-Instruct + LoRA for walls, doors, windows and rooms from 2D floor plans. "
                "Adapter status and the origin of every result are shown explicitly.")

    mode = st.sidebar.radio("Page", ["Detector", "Evaluation", "Dataset"])
    ok, info = adapter_status()
    st.sidebar.subheader("Adapter status")
    if ok:
        st.sidebar.success(f"LoRA adapter files verified: r={info['r']}, {info['n_lora_params']:,} params, "
                           f"sha256 {info['weights_sha256'][:12]}…")
        st.sidebar.caption("Verified on disk. This app does not load the model by itself.")
    else:
        st.sidebar.error("No verified LoRA adapter")
        with st.sidebar.expander("Reason"):
            st.write(info)
    st.sidebar.caption(f"Base model: {DEFAULT_BASE_MODEL}")

    {"Detector": lambda: page_detector(ok), "Evaluation": page_evaluation, "Dataset": page_dataset}[mode]()


if __name__ == "__main__":
    main()
