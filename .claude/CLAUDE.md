# LeRobot-MLX — Claude Code Project Config

## Project Overview
Port of HuggingFace LeRobot (v0.5.1) from PyTorch to Apple MLX for native Apple Silicon robotics policy training and inference.

## Key Architecture: Three-Layer Cake
1. **Policy Mirror** (`src/lerobot_mlx/policies/`) — structurally identical to upstream, same class/method names
2. **Compat Shim** (`src/lerobot_mlx/compat/`) — torch.* → mx.* translation layer (THE foundation)
3. **MLX Core** — Apple's mlx package for Metal GPU execution

Other directories:
- `src/lerobot_mlx/training/` — MLX-native training loop (replaces accelerate)
- `src/lerobot_mlx/datasets/` — Dataset loading returning mx.array
- `src/lerobot_mlx/model/` — Shared model components (vision backbones)
- `src/lerobot_mlx/processor/` — Pre/post processing pipeline
- `src/lerobot_mlx/scripts/` — CLI entry points (train, eval)
- `repositories/` — Reference repos (gitignored, read-only)
- `prds/` — 19 PRDs defining the complete build plan (see prds/README.md)
- `PROMPT.md` — Full project specification

## Critical Design Rules
1. **NEVER modify upstream files** — create parallel MLX implementations
2. **Mirror upstream structure exactly** — same class names, method signatures, configs
3. **Use the compat layer** — don't rewrite torch calls inline, route through compat/
4. **PRD-driven** — one PRD per component in prds/ before coding
5. **Cross-framework tests** — compare MLX output vs PyTorch reference
6. **Config files copy verbatim** — `configuration_*.py` are pure dataclasses, no torch
7. **Channel format** — upstream uses NCHW (torch), MLX uses NHWC; handle in compat/vision.py

## Compat Layer Modules
| Module | Replaces | Key Functions |
|--------|----------|---------------|
| `compat/tensor_ops.py` | `torch.*` | tensor, zeros, cat, stack, where, einsum |
| `compat/nn_modules.py` | `torch.nn.Module` | Module base with .to()/.train()/.eval() no-ops |
| `compat/nn_layers.py` | `torch.nn.*` | All 33 nn types: Linear, Conv2d, MultiheadAttention, etc. |
| `compat/functional.py` | `torch.nn.functional` | mse_loss, pad, scaled_dot_product_attention, etc. |
| `compat/optim.py` | `torch.optim` | Adam, AdamW, LR schedulers, gradient clipping |
| `compat/distributions.py` | `torch.distributions` | Normal, Beta, kl_divergence |
| `compat/einops_mlx.py` | `einops` | rearrange, repeat for common LeRobot patterns |
| `compat/vision.py` | `torchvision` | ResNet18/34, image transforms, NCHW↔NHWC |
| `compat/diffusers_mlx.py` | `diffusers` | DDPMScheduler, DDIMScheduler |

## Build Order (19 PRDs)
**Phase 1 (Foundation):** PRD-00→01→02→03→04 (sequential, compat layer)
**Phase 2 (First Policies):** PRD-05+06 (parallel ACT+Diffusion), then PRD-07 (training)
**Phase 3 (Data):** PRD-08 (datasets+processor)
**Phase 4 (Expand):** PRD-09→14 (parallel: TD-MPC, VQ-BeT, SAC, weights, factory, CLI)
**Phase 5 (Extended):** PRD-15 (SARM)
**Phase 6 (VLA):** PRD-16→17 (Pi0, SmolVLA — needs mlx-vlm)
**Phase 7 (Polish):** PRD-18 (benchmarks+docs)

## Reference Repositories (in /repositories/)
- `lerobot-upstream` — LeRobot v0.5.1 source of truth
- `mlx` — MLX core API reference
- `mlx-examples` — Model examples (ResNet, BERT, Stable Diffusion)
- `mlx-lm` — Language model patterns, weight loading
- `mlx-vlm` — Vision-language models (for Pi0/SmolVLA)
- `mlx-image` — Image processing
- `mlx-data` — Data loading utilities
- `pointelligence-mlx-ref` — Our proven port patterns (344 tests)

## Dev Commands
```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
```

## Upstream Sync Protocol
When LeRobot v0.6+ ships:
1. `cd repositories/lerobot-upstream && git fetch && git checkout v0.6.0`
2. `git diff v0.5.1..v0.6.0 -- src/lerobot/policies/` → classify changes
3. Config changes → copy verbatim. New nn types → add to compat/. Structural changes → mirror.
4. New policies → create new PRD, port using existing compat/
5. Update `UPSTREAM_VERSION.md`

## MLX Gotchas
- `mlx.__version__` doesn't exist → use `importlib.metadata.version("mlx")`
- No int64 on GPU → cast to int32 for scatter ops
- `mx.eval()` is MANDATORY after optimizer step → prevents lazy graph buildup
- No boolean indexing → use argsort with sentinel values
- Conv weights: PyTorch OIHW → MLX OHWI (transpose in weight conversion)
- `.at[idx].add(val)` returns NEW array (functional, no mutation)

# currentDate
Today's date is 2026-03-15.
