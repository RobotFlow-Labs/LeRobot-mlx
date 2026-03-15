# PRD-17: SmolVLA & Remaining VLA Policies

> **Status:** TODO
> **Priority:** P3 — Stretch goals
> **Dependencies:** PRD-16 (Pi0 establishes VLM patterns)
> **Estimated LOC:** ~1000
> **Phase:** 6 (VLA Foundation Models)

---

## Objective

Port remaining Vision-Language-Action (VLA) policies after Pi0 establishes the pattern.

---

## Policies Covered

### SmolVLA (P3, ~1761 LOC)
- Small VLA using SmolVLM backbone
- Minimal custom nn, delegates to transformers
- **Strategy**: Use mlx-vlm's SmolVLM if available, otherwise port minimal VLM

### GR00T N1.5 (P3, ~4029 LOC)
- NVIDIA Eagle2-5-VL multimodal encoder + flow matching DiT action head
- **Very large** — may need significant mlx-vlm extensions
- **Strategy**: Defer unless specific demand

### WallX (P3, ~6083 LOC)
- Qwen2.5-VL + LoRA + flow matching + torchdiffeq ODE solver
- Most complex policy in LeRobot
- **Strategy**: Defer, requires Qwen VLM support in mlx-vlm

### XVLA (P3, ~5580 LOC)
- Florence2 + CLIP + 20 unique nn modules
- **Strategy**: Defer, requires Florence2 port

---

## Common VLA Porting Pattern

All VLA policies follow the same structure:
1. **VLM backbone** → produces multimodal features
2. **Action head** → converts features to robot actions
3. **Training** → fine-tune action head, optionally LoRA the VLM

The compat layer + mlx-vlm handles the VLM. We only need to port the action heads.

---

## Acceptance Criteria

1. At least SmolVLA forward pass works
2. Action heads produce correct output shapes
3. LoRA fine-tuning works (if applicable)
4. 5+ tests per policy
