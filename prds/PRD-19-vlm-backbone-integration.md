# PRD-19: VLM Backbone Integration via mlx-vlm

> **Status:** TODO
> **Priority:** P0 — Enables real-world VLA policy usage
> **Dependencies:** PRD-16 (Pi0), PRD-17 (SmolVLA)
> **Estimated LOC:** ~600

---

## Objective

Create a universal VLM (Vision-Language Model) backbone loader that wraps mlx-vlm to provide any VLM as a feature extractor for VLA policies. Users should be able to:

```python
from lerobot_mlx.model.vlm_backbone import VLMBackbone

# Load any VLM as a backbone
vlm = VLMBackbone("google/paligemma-3b-pt-224")
vlm = VLMBackbone("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
vlm = VLMBackbone("mlx-community/Qwen2-VL-2B-Instruct-4bit")

# Extract features from images + text
features = vlm.encode(images=pixel_values, text="pick up the cube")
# features.shape = (B, seq_len, hidden_size)

# Use in Pi0/SmolVLA policy
policy = Pi0Policy(config, vlm_backbone=vlm)
```

---

## Architecture

```
VLMBackbone (universal interface)
  ├── load(model_id) → downloads via mlx-vlm
  ├── encode(images, text) → feature embeddings
  ├── get_hidden_size() → int
  ├── get_vision_features(images) → vision-only features
  ├── freeze() / unfreeze() → for training
  └── supports: PaliGemma, SmolVLM, Qwen2-VL, LLaVA, etc.
          ↓ delegates to
      mlx-vlm (repositories/mlx-vlm)
        ├── mlx_vlm.load(model_id) → (model, processor)
        ├── model.vision_tower → vision features
        └── model.language_model → text features
```

---

## Deliverables

### 1. `src/lerobot_mlx/model/vlm_backbone.py`

Universal VLM wrapper:
- `VLMBackbone(model_id, device=None, quantize=None)` — loads any mlx-vlm model
- `encode(images, text)` — returns combined vision+language features
- `get_vision_features(images)` — vision-only encoding
- `get_text_features(text)` — text-only encoding
- `hidden_size` property — for policy layer sizing
- `freeze()` / `unfreeze()` — freeze/unfreeze VLM weights
- `from_pretrained(model_id)` — class method for loading

### 2. `src/lerobot_mlx/model/image_processor.py`

Image preprocessing that bridges PIL/numpy/mlx:
- Resize, normalize, pad to model's expected format
- Support multiple images (multi-view cameras)
- Handle NCHW/NHWC conversion

### 3. Integration with Pi0 and SmolVLA

Update Pi0Policy and SmolVLAPolicy to accept optional VLM backbone:
- If VLM provided: use real vision+language features for conditioning
- If VLM not provided: standalone mode (action expert only, for testing)

### 4. Tests

- test_vlm_backbone_creation (mock, no actual model download)
- test_vlm_backbone_encode_shape
- test_vlm_backbone_freeze_unfreeze
- test_vlm_backbone_hidden_size
- test_vlm_backbone_image_processing
- test_pi0_with_vlm_backbone (integration)
- test_smolvla_with_vlm_backbone (integration)

---

## Acceptance Criteria

1. `VLMBackbone("google/paligemma-3b-pt-224")` loads without error (if model cached)
2. `encode()` returns features with correct hidden_size dimension
3. Pi0Policy works with VLM backbone attached
4. SmolVLAPolicy works with VLM backbone attached
5. Supports at least: PaliGemma, SmolVLM, Qwen2-VL
6. 15+ tests passing
