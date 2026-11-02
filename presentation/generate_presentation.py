"""
PowerPoint Presentation Generator for FloorPlan-AI Assessment.

Generates a clean, professional, academic-style presentation using python-pptx:
- 14 distinct slides adhering strictly to assessment requirements
- Minimalist typography (Dark Slate text on crisp White background with Blue accent)
- Every number on the results slide is read from outputs/evaluation/summary.json;
  nothing is hard-coded. With no such file the slide says that no verified results exist.
"""

import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Styling Constants
COLOR_BG = RGBColor(255, 255, 255)
COLOR_TEXT_DARK = RGBColor(15, 23, 42)       # #0F172A Dark Slate
COLOR_TEXT_MUTED = RGBColor(71, 85, 105)     # #475569 Slate
COLOR_ACCENT_BLUE = RGBColor(37, 99, 235)    # #2563EB Primary Blue
COLOR_ACCENT_BG = RGBColor(241, 245, 249)    # #F1F5F9 Soft Grey/Blue


def create_blank_slide(prs):
    """Creates a clean blank slide with white background."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG
    return slide


def add_slide_header(slide, title_text: str, category_text: str = "TECHNICAL ASSESSMENT: FLOORPLAN-AI"):
    """Adds standardized academic header to slide."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.5), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_ACCENT_BLUE

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_DARK
    p_title.space_before = Pt(4)


def add_bullet_points(slide, points, left=Inches(0.8), top=Inches(1.8), width=Inches(11.5), height=Inches(5.0)):
    """Renders structured bullet points with clean spacing."""
    tx_box = slide.shapes.add_textbox(left, top, width, height)
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    for idx, pt in enumerate(points):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(12)
        p.level = 0


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ----------------------------------------------------
    # Slide 1: Title
    # ----------------------------------------------------
    s1 = create_blank_slide(prs)
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.0))
    tf = t_box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "AI-POWERED 2D FLOOR PLAN DETECTION"
    p0.font.size = Pt(32)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_ACCENT_BLUE

    p1 = tf.add_paragraph()
    p1.text = "Fine-tuning Qwen2.5-VL-3B-Instruct with LoRA to parse floor plans into structured JSON"
    p1.font.size = Pt(18)
    p1.font.color.rgb = COLOR_TEXT_DARK
    p1.space_before = Pt(10)

    p2 = tf.add_paragraph()
    p2.text = "Technical assessment | Status: pipeline under verification; see the results slide for what has actually been measured"
    p2.font.size = Pt(13)
    p2.font.color.rgb = COLOR_TEXT_MUTED
    p2.space_before = Pt(24)

    # ----------------------------------------------------
    # Slide 2: Problem Statement
    # ----------------------------------------------------
    s2 = create_blank_slide(prs)
    add_slide_header(s2, "Problem Statement: Converting Raster Floor Plans into Structured CAD")
    add_bullet_points(s2, [
        "Architectural floor plans are predominantly stored as static raster images (PNG/JPG) or scanned blueprints.",
        "Downstream 3D building generation and BIM workflows require precise geometric primitives: wall centerlines, door/window openings, and closed room polygons.",
        "Core Technical Challenges:\n  • Discontinuous wall segments and varied drafting styles across blueprints\n  • Scale and rotation variance in door swing symbols and architectural window annotations\n  • Semantic ambiguity between room spaces (e.g. living room vs. dining room boundaries)\n  • Strict requirement for syntactically and geometrically valid structured JSON outputs.",
        "Objective: Formulate an AI detection system that extracts exact spatial coordinates, element classifications, and topologically sound geometries suitable for direct 2D-to-3D CAD reconstruction."
    ])

    # ----------------------------------------------------
    # Slide 3: Project Objectives
    # ----------------------------------------------------
    s3 = create_blank_slide(prs)
    add_slide_header(s3, "Project Objectives & Key Engineering Deliverables")
    add_bullet_points(s3, [
        "1. Vision-Language Fine-Tuning: Adapt a pretrained multimodal VLM (Qwen2.5-VL-3B-Instruct) using parameter-efficient LoRA/QLoRA on architectural datasets.",
        "2. Ground-Truth Data Pipeline: Build an automated ingestion engine for CubiCasa5K vector SVG annotations, converting spaces and structural lines into normalized coordinates.",
        "3. Strict Pydantic Schema: Enforce coordinate boundary checks, unique entity identifiers, non-zero wall lengths, and polygon closure.",
        "4. Geometric Postprocessing Engine: Formulate algorithmic refinement for orthogonal wall angle snapping, polygon topology validation (Shapely), and door/window wall attachment.",
        "5. Rigorous Empirical Evaluation: Benchmark fine-tuned performance against zero-shot base models on an unseen held-out test split, reporting precision, recall, F1, and IoU.",
        "6. Interactive Application: Provide an intuitive Streamlit dashboard for real-time inference and JSON export."
    ])

    # ----------------------------------------------------
    # Slide 4: Proposed System Architecture
    # ----------------------------------------------------
    s4 = create_blank_slide(prs)
    add_slide_header(s4, "Planned System Architecture: End-to-End Multimodal Pipeline (design, see README for implementation status)")
    add_bullet_points(s4, [
        "1. Image Ingestion & Preprocessing:\n   • Contrast-limited adaptive histogram equalization (CLAHE) for drafting line enhancement\n   • Aspect-ratio-preserving letterbox onto a 1008x1008 canvas (coordinates mapped back to the original image)",
        "2. Multimodal VLM Representation:\n   • Dynamic resolution Vision Transformer (ViT) patch extraction without spatial distortion\n   • Conversational instruction prompt guiding structured architectural decomposition",
        "3. Parameter-Efficient LoRA Adapter:\n   • Low-rank matrices (r=16, alpha=32) attached to query, key, value, and MLP projection layers\n   • Masked target loss computing cross-entropy strictly on assistant JSON tokens",
        "4. Geometric Postprocessing & JSON Sanitization:\n   • Non-Maximum Suppression (IoU >= 0.75) for duplicate entity elimination\n   • Orthogonal snapping aligning near-perpendicular walls to exact CAD axes\n   • Pydantic schema validation ensuring watertight geometric integrity."
    ])

    # ----------------------------------------------------
    # Slide 5: Selected Pretrained Model
    # ----------------------------------------------------
    s5 = create_blank_slide(prs)
    add_slide_header(s5, "Selected Pretrained Model: Qwen2.5-VL-3B-Instruct")
    add_bullet_points(s5, [
        "Model Selection Justification:\n  • Native dynamic resolution Vision Transformer avoids pixel distortion on non-square floor plans.\n  • Multimodal 3D-RoPE positional embeddings provide strong spatial-coordinate grounding.\n  • 3B parameters fit LoRA/QLoRA fine-tuning on a free Colab T4.",
        "Architectural Capabilities:\n  • Instruction-following chat model with image input and absolute-coordinate grounding (per the model card); how well it parses floor plans is what the evaluation measures.",
        "Compute & Memory Profile:\n  • Full FP16: ~7 GB weights (~14-16 GB training VRAM).\n  • 4-Bit QLoRA: ~2.2 GB VRAM for model weights, which is what makes parameter-efficient fine-tuning feasible on a Colab T4 (not full fine-tuning)."
    ])

    # ----------------------------------------------------
    # Slide 6: Dataset and Annotation Format
    # ----------------------------------------------------
    s6 = create_blank_slide(prs)
    add_slide_header(s6, "Dataset: CubiCasa5K (target) and current status")
    add_bullet_points(s6, [
        "Dataset Overview:\n  • Target source: CubiCasa5K (5,000 floor plans, CC BY-NC-SA 4.0, vector SVG annotations).\n  • Status: real data not yet ingested. An earlier set of 15 synthetic fixtures was found to be unusable (identical images, mismatched labels) and has been discarded; see docs/audit/.",
        "Category Extraction & Mapping:\n  • Walls: SVG <line> and <polyline> elements mapped to [start_x, start_y] -> [end_x, end_y] segments.\n  • Doors: Single and double door symbols mapped to bounding boxes, positions, and orientations.\n  • Windows: Perimeter opening symbols mapped to bounding boxes and midpoints.\n  • Rooms: Dense <polygon> space elements mapped to room types (Bedroom, Bathroom, Kitchen, Living Room, etc.).",
        "Leakage Prevention:\n  • Splits are by floor-plan ID, then checked for exact-duplicate images, perceptual near-duplicates and annotation similarity across splits."
    ])

    # ----------------------------------------------------
    # Slide 7: Image Preprocessing
    # ----------------------------------------------------
    s7 = create_blank_slide(prs)
    add_slide_header(s7, "Image Preprocessing & Coordinate Transformation")
    add_bullet_points(s7, [
        "1. Color & Resolution Normalization:\n   • Letterbox: one uniform scale (long side to 1008 px) plus white padding; no distortion.\n   • Scale and padding offsets are stored so predictions map back to original-image coordinates.",
        "2. Contrast & Edge Enhancement:\n   • Contrast Limited Adaptive Histogram Equalization (CLAHE) applied to the luminance channel.\n   • Sharpens low-contrast blueprint drafting marks without introducing noise.",
        "3. Coordinate Mapping Integrity:\n   • All ground-truth annotations are scaled synchronously with the raster canvas.\n   • Aspect ratio is preserved (letterbox), so room angles are not warped."
    ])

    # ----------------------------------------------------
    # Slide 8: Fine-Tuning Approach
    # ----------------------------------------------------
    s8 = create_blank_slide(prs)
    add_slide_header(s8, "Fine-Tuning Approach: Instruction-Tuned Multimodal Adaptation")
    add_bullet_points(s8, [
        "1. Conversational Template:\n   • User: <image> + 'Analyze this 2D architectural floor plan image and detect all architectural elements...'\n   • Assistant: Compact serialized JSON matching FloorPlanSchema.",
        "2. Label Masking Strategy:\n   • All image tokens and user instruction prompt tokens are masked with label = -100.\n   • Model gradients are computed exclusively on target JSON architectural syntax and coordinates.",
        "3. Loss Formulation:\n   • Standard autoregressive token Cross-Entropy loss over the discrete coordinate and classification vocabulary.",
        "4. Validation & Checkpointing:\n   • Checkpoints saved per epoch; best adapter checkpoint retained based on minimum validation loss."
    ])

    # ----------------------------------------------------
    # Slide 9: LoRA/QLoRA Hyperparameters
    # ----------------------------------------------------
    s9 = create_blank_slide(prs)
    add_slide_header(s9, "LoRA / QLoRA Hyperparameter Configuration (planned; not yet executed)")
    add_bullet_points(s9, [
        "Fine-Tuning Hyperparameters:\n  • LoRA Rank (r): 16\n  • LoRA Alpha: 32 (Scaling ratio = 2.0)\n  • LoRA Dropout: 0.05\n  • Target Modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]\n  • Trainable parameters: measured and logged at training time (not assumed).",
        "Optimization Parameters:\n  • Optimizer: AdamW (weight_decay = 0.01)\n  • Learning Rate: 2e-4 with Linear Warmup (warmup_ratio = 0.03)\n  • Epochs: 3 | Batch Size: 1 per device | Gradient Accumulation Steps: 8\n  • Precision: FP16 mixed precision on T4 (no BF16) with gradient checkpointing."
    ])

    # ----------------------------------------------------
    # Slide 10: JSON Output Structure
    # ----------------------------------------------------
    s10 = create_blank_slide(prs)
    add_slide_header(s10, "Standardized JSON Schema (Pydantic Validated)")
    add_bullet_points(s10, [
        "JSON Specification Attributes:\n  • image, image_width, image_height, units ('pixels')\n  • rooms: id, type (e.g. 'bedroom'), polygon [[x1, y1], ...], area_pixels, confidence\n  • walls: id, start [x, y], end [x, y], thickness, confidence\n  • doors: id, bbox [xmin, ymin, xmax, ymax], position [x, y], orientation, confidence\n  • windows: id, bbox [xmin, ymin, xmax, ymax], position [x, y], orientation, confidence",
        "Automated Quality Constraints:\n  • In-bounds coordinate clamping: 0 <= coord <= image_dimension\n  • Closed non-degenerate polygons with automated Shoelace formula area calculation\n  • Non-zero wall segment lengths and unique identifier guarantees."
    ])

    # ----------------------------------------------------
    # Slide 11: Evaluation Metrics
    # ----------------------------------------------------
    s11 = create_blank_slide(prs)
    add_slide_header(s11, "Evaluation Methodology: Multi-Faceted Metric Suite")
    add_bullet_points(s11, [
        "1. Architectural Object Detection Metrics:\n   • Precision, Recall, and F1-Score calculated via Hungarian (optimal) bipartite matching.\n   • True Positives defined by spatial intersection thresholds.",
        "2. Geometric Alignment Metrics:\n   • Doors & Windows: Bounding Box Intersection over Union (IoU >= 0.5).\n   • Rooms: Polygon IoU computed using exact geometric polygon intersections (Shapely).\n   • Walls: Mean endpoint distance error (tolerance <= 25 px).",
        "3. JSON Integrity Metrics:\n   • JSON Syntax Validity Rate (% parseable without error).\n   • Pydantic Schema Validation Compliance Rate (% passing strict spatial constraints)."
    ])

    # ----------------------------------------------------
    # Slide 12: Experimental Results (read from outputs/evaluation/summary.json only)
    # ----------------------------------------------------
    s12 = create_blank_slide(prs)
    add_slide_header(s12, "Experimental Results: Held-Out Test Split")
    summary_path = Path("outputs/evaluation/summary.json")
    if not summary_path.exists():
        add_bullet_points(s12, [
            "No verified evaluation results exist yet.",
            "Earlier reported numbers (Macro F1 1.000, etc.) were invalid and have been withdrawn: they came from leaked or simulated evaluation. Details: docs/audit/.",
            "Results will appear here once the zero-shot, fine-tuned and OpenCV-baseline systems have been evaluated on unseen CubiCasa5K floor plans.",
        ], top=Inches(1.8))
    else:
        import json
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        systems = summary["systems"]
        cols = ["System", "Failed", "JSON valid", "Wall F1", "Door F1", "Window F1", "Room F1", "Macro F1"]
        rows, cols_n = len(systems) + 1, len(cols)
        table = s12.shapes.add_table(rows, cols_n, Inches(0.8), Inches(1.9), Inches(11.7), Inches(0.5) * rows).table
        for j, name in enumerate(cols):
            table.cell(0, j).text = name
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        for i, sysm in enumerate(systems, 1):
            vals = [sysm["label"], f'{sysm["n_failed"]}/{summary["n_test_images"]}', fmt(sysm["json_validity_rate"]),
                    fmt(sysm["f1"]["walls"]), fmt(sysm["f1"]["doors"]), fmt(sysm["f1"]["windows"]),
                    fmt(sysm["f1"]["rooms"]), fmt(sysm["macro_f1"])]
            for j, v in enumerate(vals):
                table.cell(i, j).text = v
        note = s12.shapes.add_textbox(Inches(0.8), Inches(1.9) + Inches(0.5) * rows + Inches(0.3), Inches(11.7), Inches(2.0))
        ntf = note.text_frame
        ntf.word_wrap = True
        ntf.text = f'Test set: {summary["n_test_images"]} unseen floor plans. ' + " ".join(summary.get("limitations", []))
        ntf.paragraphs[0].font.size = Pt(12)

    # ----------------------------------------------------
    # Slide 13: Challenges and Limitations
    # ----------------------------------------------------
    s13 = create_blank_slide(prs)
    add_slide_header(s13, "Technical Challenges, Edge Cases & Practical Limitations")
    add_bullet_points(s13, [
        "1. Token Context Length in Dense Multi-Room Plans:\n   • Large complexes with 50+ rooms generate >4,000 output tokens, requiring token sequence budgeting.",
        "2. Curved and Non-Manhattan Wall Geometries:\n   • Current orthogonal postprocessing assumes predominantly 90-degree drafting angles; circular walls require spline approximation.",
        "3. Blueprint Scan Artifacts & Text Occlusion:\n   • Text dimension callouts intersecting wall lines occasionally disrupt edge continuity.",
        "4. CAD Precision vs. Neural Approximation:\n   • VLM coordinates are approximate; the achieved pixel error is reported on the results slide, and postprocessing is logged so it cannot hide model errors."
    ])

    # ----------------------------------------------------
    # Slide 14: Conclusion and Future Scope
    # ----------------------------------------------------
    s14 = create_blank_slide(prs)
    add_slide_header(s14, "Conclusion & Future Research Directions")
    add_bullet_points(s14, [
        "Status:\n  \u2022 The pipeline code (schema, preprocessing, LoRA trainer, inference, evaluation) is implemented; what has actually been executed and measured is stated in the README status table and the results slide.\n  \u2022 Fine-tuning is claimed only if a verified LoRA adapter exists and the evaluation above was run with it.",
        "Next steps:\n  \u2022 Train on a larger real CubiCasa5K subset on Colab and evaluate on a larger unseen test split.\n  \u2022 Compare against zero-shot Qwen2.5-VL and the OpenCV baseline with identical metrics.\n  \u2022 Future work: 2D-to-3D IFC/BIM export, multi-storey alignment, tiled inference for very large plans."
    ])

    out_pptx = Path("presentation/Floorplan_AI_Assessment.pptx")
    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_pptx))
    print(f"Successfully generated PowerPoint presentation at: {out_pptx}")


if __name__ == "__main__":
    build_presentation()
