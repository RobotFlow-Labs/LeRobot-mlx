# LeRobot-MLX — PRD Index

## Build Dependency Graph

```
PRD-00 (Architecture)
  │
  ├── PRD-01 (Dev Environment)
  │     │
  │     ├── PRD-02 (Compat Core: tensor_ops, nn_modules, nn_layers)
  │     │     │
  │     │     ├── PRD-03 (Compat: functional, optim, distributions)
  │     │     │     │
  │     │     │     ├── PRD-04 (Compat: vision, einops, diffusers)
  │     │     │     │     │
  │     │     │     │     ├── PRD-05 (ACT Policy)──────────┐
  │     │     │     │     ├── PRD-06 (Diffusion Policy)────┤
  │     │     │     │     │                                 │
  │     │     │     │     │     PRD-07 (Training Loop) ◄────┘
  │     │     │     │     │
  │     │     │     │     ├── PRD-09 (TD-MPC)──────┐
  │     │     │     │     ├── PRD-10 (VQ-BeT)──────┤  ← PARALLEL
  │     │     │     │     ├── PRD-11 (SAC)─────────┤
  │     │     │     │     └── PRD-15 (SARM)────────┘
  │     │     │     │
  │     │     │     └── PRD-08 (Datasets + Processor)
  │     │     │
  │     │     ├── PRD-12 (Weight Conversion)
  │     │     └── PRD-13 (Policy Factory)
  │     │
  │     └── PRD-14 (CLI Scripts)
  │
  ├── PRD-16 (Pi0 VLA) ← requires mlx-vlm
  ├── PRD-17 (SmolVLA + other VLAs)
  └── PRD-18 (Benchmarks + Docs)
```

## Phase Execution Plan

### Phase 1: Foundation (PRD-00 through PRD-04)
All compat layer work. **Sequential** — each builds on the previous.
- PRD-00: Architecture document (no code)
- PRD-01: pyproject.toml, scaffold, smoke tests
- PRD-02: tensor_ops, nn_modules, nn_layers (~800 LOC)
- PRD-03: functional, optim, distributions (~600 LOC)
- PRD-04: vision ResNet, einops, diffusers schedulers (~700 LOC)

**Gate:** All compat tests pass. `from lerobot_mlx.compat import nn, F` works.

### Phase 2: First Policies + Training (PRD-05 through PRD-07)
- PRD-05: ACT Policy (~400 LOC) ← can **parallel** with PRD-06
- PRD-06: Diffusion Policy (~500 LOC) ← can **parallel** with PRD-05
- PRD-07: Training Loop (~400 LOC) ← needs one policy

**Gate:** ACT and Diffusion forward pass works. Training loss decreases.

### Phase 3: Data Pipeline (PRD-08)
- PRD-08: Datasets + Processor (~500 LOC)

**Gate:** Real dataset loads from HF Hub, returns mx.array batches.

### Phase 4: Remaining Policies + Infrastructure (PRD-09 through PRD-14)
All **parallel** — independent of each other:
- PRD-09: TD-MPC (~450 LOC)
- PRD-10: VQ-BeT (~600 LOC)
- PRD-11: SAC (~400 LOC)
- PRD-12: Weight Conversion (~300 LOC)
- PRD-13: Policy Factory (~200 LOC)
- PRD-14: CLI Scripts (~300 LOC)

**Gate:** 5 policies ported. `lerobot-mlx-train` and `lerobot-mlx-eval` work.

### Phase 5: Extended Policies (PRD-15)
- PRD-15: SARM (~600 LOC)

### Phase 6: VLA Foundation Models (PRD-16, PRD-17)
- PRD-16: Pi0 (~800 LOC) — requires mlx-vlm integration
- PRD-17: SmolVLA + others (~1000 LOC)

### Phase 7: Polish (PRD-18)
- PRD-18: Benchmarks + Documentation (~400 LOC)

---

## Total Estimated LOC: ~7,750

## Key Architecture Decision: Upstream Tracking

Every PRD follows the **Three-Layer Cake** (PRD-00):
1. Policy code mirrors upstream structure exactly
2. Compat layer absorbs all torch→mlx translation
3. MLX core handles GPU execution

When LeRobot v0.6 ships, the update protocol is:
1. `git diff v0.5.1..v0.6.0` on upstream
2. Copy config changes verbatim
3. Apply structural changes to our mirror
4. Add new compat entries if new torch types used
5. Create new PRDs for new policies

## Reference Repositories (in /repositories/)

| Repo | Purpose |
|------|---------|
| `lerobot-upstream` | Source of truth — upstream LeRobot v0.5.1 |
| `mlx` | MLX core — API reference |
| `mlx-examples` | MLX model examples (ResNet, BERT, Llama, Stable Diffusion) |
| `mlx-lm` | MLX language model loading — patterns for weight conversion |
| `mlx-data` | MLX data loading utilities |
| `mlx-vlm` | MLX vision-language models — needed for Pi0/SmolVLA |
| `mlx-image` | MLX image processing — vision transforms |
| `mlx-swift` | MLX Swift bindings (reference only) |
| `pointelligence-mlx-ref` | Our prior port — proven patterns |
