# Technical Methodology: AI-Powered 2D Floor Plan Detection

## 1. System Overview & Problem Formulation

Architectural 2D floor plans are dense graphical blueprints encoding structural partitions (walls), circulation portals (doors), fenestrations (windows), and functional spaces (rooms). Automated extraction of these entities is critical for automated Building Information Modeling (BIM), 2D-to-3D CAD reconstruction, structural simulation, and indoor navigation.

> **Status:** this page describes the intended design. Only the preprocessing (letterbox), strict parsing and the
> geometric refinement below are implemented and unit-tested; the model stage is not trained or run yet (see README).

The system is formulated as a hybrid vision-language detection and geometric refinement pipeline:

$$\mathcal{I} \in \mathbb{R}^{H \times W \times 3} \xrightarrow{\text{letterbox 1008}} \tilde{\mathcal{I}} \xrightarrow{\text{VLM + LoRA}} \mathcal{S}_{\text{tokens}} \xrightarrow{\text{JSON Parse}} \mathcal{G}_{\text{raw}} \xrightarrow{\text{Geometric Snapping}} \mathcal{G}_{\text{refined}}$$

## 2. Vision-Language Model Architecture: Qwen2.5-VL-3B-Instruct

### 2.1 Model Architecture
- **Vision Encoder**: Dynamic resolution Vision Transformer (ViT). Floor plans are partitioned into dynamic 2D patch grids rather than forcibly downsampled or padded to square proportions. This prevents architectural scale distortion.
- **Multimodal Alignment**: Spatial tokens are fused into the language embedding space using 3D-RoPE (Rotary Position Embeddings), allowing simultaneous awareness of 2D image coordinates and 1D sequential token order.
- **Language Decoder**: 3B parameter dense transformer operating with SwiGLU activations, RMSNorm, and grouped-query attention.

### 2.2 Parameter-Efficient Fine-Tuning (PEFT with LoRA / QLoRA)
Fine-tuning the entire 3B parameter model is computationally prohibitive and prone to catastrophic forgetting on domain-specific formats. We employ Low-Rank Adaptation (LoRA):

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

where $W_0 \in \mathbb{R}^{d \times k}$ are frozen base model weights, $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$ are trainable low-rank decomposition matrices ($r=16, \alpha=32$).

Target modules include all linear projections in the multi-head self-attention and MLP feedforward blocks:
`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.

### 2.3 Loss Masking Strategy
Training loss is calculated autoregressively over next-token prediction, but **all prompt tokens (user instructions and vision patch tokens) are masked with ignore index `-100`**:

$$\mathcal{L} = -\sum_{t = T_{\text{prompt}} + 1}^{T_{\text{total}}} \log P(y_t \mid y_{<t}, \tilde{\mathcal{I}})$$

The model receives gradients exclusively on the architectural coordinates, entity tags, and JSON syntax.

## 3. Geometric Postprocessing Engine

Raw vision model outputs approximate spatial positions. To generate CAD-viable geometry, algorithmic postprocessing is applied:

1. **Orthogonal Wall Snapping**:
   Blueprints predominantly follow orthogonal axes. For each detected wall segment with angle $\theta = \arctan2(|\Delta y|, |\Delta x|)$:
   - If $\theta \le \tau_{\text{angle}}$ (default $4.0^\circ$ in `postprocess_floorplan_json`), the segment is snapped to a horizontal line with $y = \frac{y_1 + y_2}{2}$.
   - If $|90^\circ - \theta| \le \tau_{\text{angle}}$, the segment is snapped to a vertical line with $x = \frac{x_1 + x_2}{2}$.

2. **Non-Maximum Suppression (Duplicate Filtering)**:
   For overlapping bounding boxes (doors and windows), a box with $\text{IoU} > 0.75$ against an already kept box is dropped.
   The first box in the model's output order is kept; no confidence is used because the model emits none.

3. **Topology Repair & Area Computation**:
   Room polygons are evaluated with Shapely. Self-intersecting rings are resolved via zero-buffer dilation (`poly.buffer(0)`). Verified polygon area is computed via the Shoelace formula:
   $$A = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$$

4. **Portal-to-Wall Snap**:
   Door and window midpoints are projected onto the nearest wall segment whose orientation matches the portal, within $\tau_{\text{dist}} = 15\text{ px}$.
   Raw and refined outputs are both stored in the prediction record, so refinement never hides model errors.
