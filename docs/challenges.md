# Technical Challenges, Edge Cases & Practical Limitations

## 1. Architectural Floor-Plan Domain Challenges

1. **Token Length vs. Plan Complexity**:
   Large commercial or multi-family floor plans contain dozens of rooms and hundreds of wall segments. Generating a complete JSON output can exceed 4,096 tokens, risking sequence truncation.
   *Planned mitigation (not implemented)*: a compact output format without ids, areas or confidence values.

2. **Non-Manhattan and Curved Wall Geometries**:
   Traditional drafting systems often employ orthogonal (Manhattan) walls at $0^\circ$ and $90^\circ$. However, contemporary architecture features slanted and circular curved walls.
   *Current behaviour*: the orthogonal snapping step applies a small angular tolerance (4 degrees by default). Walls with larger slant angles are kept as predicted.

3. **Blueprint Text and Annotation Occlusion**:
   Scanned blueprints frequently include dimension lines, room labels, electrical symbols, and area callouts that intersect wall lines.
   *Status*: untested. No preprocessing in the current pipeline removes text, and how the model copes with it is unmeasured.

## 2. Model & Compute Limitations

1. **Zero-shot base-model behaviour (hypotheses to measure, not findings)**:
   A general-purpose VLM may add conversational text around the JSON, break the schema, or place elements at wrong
   coordinates. How often this happens for Qwen2.5-VL-3B on floor plans has **not been measured**. The evaluation plan
   includes a zero-shot run so that JSON-validity and schema-validity rates are reported from data. Whether LoRA
   fine-tuning improves them is likewise unknown until a trained adapter is evaluated on unseen plans.

2. **CAD tolerance vs. neural precision**:
   Model coordinates are approximate, while CAD use needs consistent topology. The geometric refinement step (wall snapping,
   portal snapping, polygon repair) targets this. Its effect on real model output has not been measured; raw and refined
   outputs are both kept so the effect can be evaluated.
