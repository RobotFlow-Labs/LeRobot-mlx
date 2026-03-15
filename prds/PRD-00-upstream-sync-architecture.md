# PRD-00: Upstream Sync Architecture

> **Status:** DESIGN
> **Priority:** P0 — This PRD defines the porting philosophy that ALL other PRDs follow.
> **Dependencies:** None
> **Estimated LOC:** 0 (architecture document only)

---

## Problem

LeRobot is under active development (v0.5.1 → v0.6 → v0.7+). Every design decision must answer: **"When upstream ships a new version, how hard is the merge?"**

Bad answer: Fork upstream, modify files in-place → merge conflicts on every update.
Good answer: Thin adapter layer that keeps our code structurally identical to upstream.

---

## Architecture: The Three-Layer Cake

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: Policy Mirror (src/lerobot_mlx/policies/) │
│  ─ Structurally identical to upstream                │
│  ─ Same class names, method signatures, configs      │
│  ─ Imports from compat/ instead of torch             │
├─────────────────────────────────────────────────────┤
│  Layer 2: Compatibility Shim (src/lerobot_mlx/compat/)│
│  ─ torch.* → mx.* translation                       │
│  ─ torch.nn.* → mlx.nn.* adapters                   │
│  ─ torch.nn.functional.* → mlx equivalents           │
│  ─ torch.distributions → custom implementations      │
│  ─ einops → reshape/transpose utilities              │
│  ─ torchvision → mlx-image / custom vision           │
│  ─ diffusers → custom schedulers                     │
├─────────────────────────────────────────────────────┤
│  Layer 1: MLX Core (Apple's mlx package)             │
│  ─ mx.array, mx.nn.Module, mx.optimizers             │
│  ─ Metal GPU backend, unified memory                 │
│  ─ Lazy evaluation, mx.eval() materialization        │
└─────────────────────────────────────────────────────┘
```

---

## The Upstream Sync Protocol

### When a new LeRobot version ships:

```bash
# 1. Pull new upstream
cd repositories/lerobot-upstream
git fetch origin
git checkout v0.6.0

# 2. Generate diff for each ported component
git diff v0.5.1..v0.6.0 -- src/lerobot/policies/act/ > /tmp/act-diff.patch
git diff v0.5.1..v0.6.0 -- src/lerobot/policies/diffusion/ > /tmp/diffusion-diff.patch
# ... repeat per policy

# 3. Classify each change:
#    A) Config changes (configuration_*.py) → COPY VERBATIM (pure dataclasses)
#    B) New layers/modules → Add to compat/ if new torch type, then use in mirror
#    C) Structural changes → Apply same change to our mirror file
#    D) New policies → Create new PRD, port using existing compat/
#    E) New dependencies → Check if MLX equivalent exists, add to compat/

# 4. Update version tracking
echo "v0.6.0" > UPSTREAM_VERSION.md
```

### Why this works:

| Change Type | Upstream | Our Mirror | Merge Effort |
|-------------|----------|------------|--------------|
| New config field | `configuration_act.py` | Copy file verbatim | Zero |
| New nn layer used | `modeling_act.py` adds `nn.GRU` | Add GRU to `compat/nn_layers.py`, update mirror | Low |
| Structural refactor | Method renamed/split | Apply same rename/split to mirror | Low |
| New policy added | `policies/newpolicy/` | New PRD, port using existing compat/ | Medium |
| New external dep | `import new_lib` | Evaluate: add to compat/ or skip | Varies |
| Bug fix | Fix in computation | Apply same fix (logic is identical) | Zero |

---

## File Classification Rules

For every upstream file, decide its treatment:

### COPY VERBATIM (no torch dependency)
- `configuration_*.py` — Pure Python dataclasses
- `configs/*.yaml` — Config files
- Hardware modules: `cameras/`, `motors/`, `robots/`, `teleoperators/`
- `transport/`, `envs/` (minimal torch)

### PORT via COMPAT LAYER (has torch, need MLX)
- `modeling_*.py` — Neural network definitions
- `processor_*.py` — Pre/post processing
- `factory.py`, `pretrained.py`, `utils.py` — Policy registry
- `datasets/` — Data loading (torch.Tensor → mx.array)
- `training/` — Training loop (accelerate → mlx.nn.value_and_grad)
- `optim/` — Optimizer configuration
- `model/` — Shared model components (vision backbones)
- `scripts/` — CLI entry points

### SKIP (not needed for MLX)
- `async_inference/` — CUDA-specific async patterns
- CUDA/GPU-specific code paths
- `accelerate`-specific distributed code

---

## Import Translation Table

Every `modeling_*.py` file in upstream starts with torch imports. Our mirror replaces them:

```python
# UPSTREAM (modeling_act.py)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops
from torchvision.models import resnet18

# OUR MIRROR (modeling_act.py)
import mlx.core as mx
from lerobot_mlx.compat import nn, F, Tensor
from lerobot_mlx.compat.einops_mlx import rearrange, repeat
from lerobot_mlx.compat.vision import resnet18
```

The rest of the file stays **structurally identical**. Class names, method names, variable names, comments — all the same. Only the import header changes.

---

## Directory Mirror Rules

```
upstream: src/lerobot/policies/act/modeling_act.py
ours:     src/lerobot_mlx/policies/act/modeling_act.py

upstream: src/lerobot/policies/act/configuration_act.py
ours:     src/lerobot_mlx/policies/act/configuration_act.py  ← COPY VERBATIM

upstream: src/lerobot/model/backbone.py
ours:     src/lerobot_mlx/model/backbone.py

upstream: src/lerobot/datasets/lerobot_dataset.py
ours:     src/lerobot_mlx/datasets/lerobot_dataset.py
```

---

## Compat Layer Contract

The `compat/` module MUST provide these namespaces:

| Compat Module | What It Replaces | Coverage |
|---------------|-----------------|----------|
| `compat.tensor_ops` | `torch.tensor`, `torch.zeros`, `torch.cat`, `torch.stack`, etc. | All tensor creation + manipulation |
| `compat.nn` | `torch.nn.Module`, `nn.Linear`, `nn.Conv2d`, etc. | All 33 nn modules used by upstream |
| `compat.F` | `torch.nn.functional.*` | All functional ops (mse_loss, pad, softmax, etc.) |
| `compat.distributions` | `torch.distributions.*` | Normal, Beta, kl_divergence |
| `compat.optim` | `torch.optim.*` | Adam, AdamW, SGD + LR schedulers |
| `compat.einops_mlx` | `einops.rearrange`, `einops.repeat` | Tensor rearrangement |
| `compat.vision` | `torchvision.models`, `torchvision.transforms` | ResNet, image transforms |
| `compat.diffusers_mlx` | `diffusers.DDPMScheduler`, `diffusers.DDIMScheduler` | Noise schedulers |

---

## Success Metric

**The Diff Test:** When upstream v0.6.0 ships, run:
```bash
diff -r src/lerobot_mlx/policies/ repositories/lerobot-upstream/src/lerobot/policies/ \
  --exclude='__pycache__' --exclude='configuration_*'
```

The diff should show ONLY:
1. Import header changes (torch → compat)
2. `.to(device)` removals (no-ops in MLX)
3. `torch.no_grad()` removals (not needed in MLX)
4. Minimal API differences (e.g., `MultiheadAttention` → `MultiHeadAttention`)

If the diff shows large structural divergences, the compat layer has failed.

---

## Anti-Patterns (NEVER DO THESE)

1. **NEVER modify upstream files** — clone is read-only reference
2. **NEVER rewrite policy logic** — only swap framework calls
3. **NEVER add MLX-specific features to policy code** — those go in compat/ or training/
4. **NEVER inline torch→mlx translations** — always route through compat/
5. **NEVER skip config files** — copy them verbatim, they're the API contract
6. **NEVER change class/method names** — upstream compatibility is sacred
