# PRD-16: Pi0 VLA Foundation Model Policy

> **Status:** TODO
> **Priority:** P3 — Stretch goal, requires VLM infrastructure
> **Dependencies:** PRD-02, PRD-03, PRD-04, mlx-vlm repo
> **Estimated LOC:** ~800
> **Phase:** 6 (VLA Foundation Models)

---

## Objective

Port Pi0 — Physical Intelligence foundation model based on PaliGemma VLM with flow matching action head. This is the most technically challenging policy due to its dependency on the HuggingFace `transformers` ecosystem.

---

## Architecture

```
Pi0:
  ├── PaliGemma VLM (Vision-Language Model)
  │   ├── SigLIP Vision Encoder → image tokens
  │   ├── Gemma Language Model → text understanding
  │   └── Cross-modal attention
  ├── Flow Matching Action Head
  │   ├── Conditional flow: (observation, noise) → action
  │   └── ODE integration at inference
  └── Action tokenization (optional, Pi0-FAST)
```

## Strategy: Use mlx-vlm

Instead of porting the full `transformers` PaliGemma, leverage the `mlx-vlm` repository (already cloned) which has MLX-native VLM implementations:

```
repositories/mlx-vlm/
  ├── mlx_vlm/models/
  │   ├── paligemma/ → Direct replacement for PaliGemma
  │   ├── gemma/ → Language backbone
  │   └── ...
```

## Porting Approach

1. **VLM backbone**: Use mlx-vlm's PaliGemma directly (already MLX-native)
2. **Action head**: Port the flow matching head via compat layer
3. **Adapter layer**: Bridge between mlx-vlm VLM output and our action head
4. **Weight conversion**: Convert Pi0 HF Hub weights → mlx-vlm format + action head

## Blockers

- mlx-vlm PaliGemma compatibility with LeRobot's expected interface
- Flow matching ODE solver (may need scipy.integrate as fallback)
- Pi0 weights are large (multi-GB) — need efficient loading

---

## Acceptance Criteria

1. PaliGemma VLM loads and produces image+text features
2. Flow matching action head denoises actions
3. End-to-end: image + instruction → action sequence
4. Weight loading from HF Hub Pi0 checkpoints
5. 10+ tests passing

---

## Also Covers (sub-PRDs if needed)

- **Pi0.5** — extended Pi0, same architecture with modifications
- **Pi0-FAST** — Pi0 with tokenized actions (action vocabulary)
