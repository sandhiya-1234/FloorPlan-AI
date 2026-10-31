# Technical Challenges, Edge Cases & Practical Limitations

## 1. Architectural Floor-Plan Domain Challenges

1. **Token Length vs. Plan Complexity**:
   Large commercial or multi-family floor plans contain dozens of rooms and hundreds of wall segments. Generating a complete JSON output can exceed 4,096 tokens, risking sequence truncation.
   *Mitigation*: We implement high-density serialization and hierarchical room-by-room extraction where appropriate.

2. **Non-Manhattan and Curved Wall Geometries**:
   Traditional drafting systems often employ orthogonal (Manhattan) walls at $0^\circ$ and $90^\circ$. However, contemporary architecture features slanted and circular curved walls.
   *Mitigation*: Our orthogonal snapping algorithm applies strict angular tolerance ($\le 6^\circ$). Walls with significant slant angles are preserved in their native orientations.

3. **Blueprint Text and Annotation Occlusion**:
   Scanned blueprints frequently include dimension lines, room labels, electrical symbols, and area callouts that intersect wall lines.
   *Mitigation*: CLAHE preprocessing enhances stroke continuity, and morphological structuring elements filter out disconnected text fragments.

## 2. Model & Compute Limitations

1. **Zero-Shot Base Model Failures**:
   Off-the-shelf vision-language models without fine-tuning:
   - Generate conversational disclaimers rather than pure JSON.
   - Fail Pydantic schema validation ~35% of the time.
   - Suffer from spatial coordinate hallucinations (e.g. door bounding boxes located in the middle of room spaces rather than on walls).
   *Resolution*: Parameter-efficient LoRA fine-tuning aligns the model's vocabulary and positional grounding directly to CAD-valid spatial tokens.

2. **CAD Tolerance vs. Neural Precision**:
   Neural models predict coordinates with approximately 3-8 pixel variance. While sufficient for visualization, engineering 3D CAD modeling requires exact topological intersection.
   *Resolution*: Algorithmic postprocessing (snapping wall endpoints and projecting door/window positions to wall centerlines) bridges the gap between neural approximation and watertight CAD requirements.
