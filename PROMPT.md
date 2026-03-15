# LeRobot-MLX — Master Build Prompt

> Copy this entire file and paste it as a prompt when working in the `/Users/ilessio/Development/AIFLOWLABS/R&D/LeRobot-mlx` directory.

---

## Mission

Port **LeRobot v0.5.1** (HuggingFace's open-source robotics framework) from **PyTorch** to **Apple MLX**, creating the first native Apple Silicon robotics policy training and inference framework.

**Built by [AIFLOW LABS](https://aiflowlabs.io) / [RobotFlow Labs](https://robotflowlabs.com)**

---

## Context & Prior Art

We already successfully ported **PointCNN++ (CVPR 2026)** from PyTorch+Triton+CUDA to MLX in `pointelligence-mlx`. That port involved 5 custom CUDA/Triton kernels rewritten as MLX ops with `@mx.custom_function` VJPs. 344 tests, all passing. The proven approach:

1. PRD-driven development (one PRD per component)
2. Adapter/bridge pattern (keep upstream structure, replace torch ops with mlx ops)
3. Cross-framework tests (compare MLX output vs PyTorch reference)
4. Bottom-up build order (primitives → layers → models → training)

**Key difference**: LeRobot has **zero custom CUDA kernels**. It's pure PyTorch. The port is about replacing `torch.*` / `torch.nn.*` / `torchvision.*` calls with `mlx.*` / `mlx.nn.*` equivalents — NOT kernel engineering.

---

## Reference Repositories

Clone these into `repositories/` (gitignored, never pushed):

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/LeRobot-mlx

# Upstream LeRobot (the source of truth)
mkdir -p repositories
git clone --depth 1 https://github.com/huggingface/lerobot.git repositories/lerobot-upstream

# Our pointelligence-mlx (reference for how we port)
git clone --depth 1 https://github.com/RobotFlow-Labs/pointelligence-mlx.git repositories/pointelligence-mlx-ref
```

The upstream is at: `repositories/lerobot-upstream/`
Our target repo is: `https://github.com/RobotFlow-Labs/LeRobot-mlx.git` (this repo)

---

## Upstream LeRobot Architecture (v0.5.1)

```
src/lerobot/
  policies/              ← THE CORE: what we port (neural networks)
    act/                 #  ACT: Action Chunking Transformer (2 files)
    diffusion/           #  Diffusion Policy: DDPM-based visuomotor (2 files)
    tdmpc/               #  TD-MPC: temporal difference MPC (2 files)
    vqbet/               #  VQ-BeT: Vector-Quantized Behavior Transformer
    sac/                 #  SAC: Soft Actor-Critic (RL policy)
    pi0/                 #  Pi0: Physical Intelligence foundation model
    pi05/                #  Pi0.5: upgraded Pi0
    pi0_fast/            #  Pi0-FAST: tokenized action Pi0
    groot/               #  GR00T N1.5: NVIDIA foundation model
    smolvla/             #  SmolVLA: small VLA model
    wall_x/              #  WallX: Qwen2.5-VL based
    xvla/                #  XVLA: Florence2-based
    sarm/                #  SARM: reward-based
    rtc/                 #  RTC: real-time control
    factory.py           #  Policy registry
    pretrained.py        #  HuggingFace Hub loading
    utils.py             #  Shared utilities

  datasets/              ← Data loading (Parquet + video)
  processor/             ← Pre/post processing pipeline
  model/                 ← Shared model components
  configs/               ← Draccus config system
  optim/                 ← Optimizer configs
  training/              ← Training loop (uses accelerate)
  scripts/               ← CLI entry points

  # Hardware (keep as-is, no torch dependency):
  cameras/               ← Camera drivers (OpenCV)
  motors/                ← Motor drivers (Dynamixel, Feetech, etc.)
  robots/                ← Robot abstractions
  teleoperators/         ← Teleoperation
  envs/                  ← Gym environments
  transport/             ← gRPC transport
  async_inference/       ← Async inference server
```

### Policies by Complexity (port in this order)

| Priority | Policy | Architecture | Complexity | Why First |
|----------|--------|-------------|------------|-----------|
| P0 | **ACT** | Transformer encoder-decoder + CVAE | Medium | Most-used policy for manipulation |
| P0 | **Diffusion** | UNet/Transformer + DDPM scheduler | Medium | Second most-used, iconic paper |
| P1 | **TD-MPC** | World model + actor-critic | Medium | Good RL coverage |
| P1 | **VQ-BeT** | VQ-VAE + Transformer | Medium | Discrete action space |
| P1 | **SAC** | Twin Q + actor | Low | Simple RL baseline |
| P2 | **Pi0** | PaliGemma VLM + flow matching | High | Foundation model, needs transformers |
| P2 | **Pi0.5** | Extended Pi0 | High | Depends on Pi0 |
| P2 | **Pi0-FAST** | Pi0 + tokenized actions | High | Depends on Pi0 |
| P3 | **GR00T** | Eagle2 VLM + DiT action head | Very High | Huge model, flash-attn |
| P3 | **SmolVLA** | SmolVLM + action expert | High | Depends on transformers |
| P3 | **WallX** | Qwen2.5-VL + diffusion | Very High | Massive VLM |
| P3 | **XVLA** | Florence2 + soft transformer | High | Custom Florence2 |

---

## Architecture Design: Thin Adapter Layer

### Critical Design Principle: TRACK UPSTREAM

The #1 requirement is that when LeRobot v0.6, v0.7, etc. ship, we can **diff upstream and apply changes easily**. This means:

1. **DO NOT fork and modify upstream files**. Instead, create parallel MLX implementations.
2. **Mirror the upstream directory structure exactly** in our `src/lerobot_mlx/` package.
3. **Keep the same class names, method signatures, and config structures** as upstream.
4. **Use a thin torch→mlx compatibility layer** so policy code stays as close to upstream as possible.

### Project Structure

```
LeRobot-mlx/
├── .gitignore
├── pyproject.toml                    # Our package: lerobot-mlx
├── PROMPT.md                         # This file
├── UPSTREAM_VERSION.md               # Tracks which upstream commit we're synced to
├── repositories/                     # (gitignored) reference repos
│   └── lerobot-upstream/
│
├── src/
│   └── lerobot_mlx/
│       ├── __init__.py
│       ├── _version.py               # "0.1.0"
│       │
│       ├── compat/                   # ★ THE KEY: torch→mlx compatibility shim
│       │   ├── __init__.py
│       │   ├── tensor_ops.py         # torch.tensor → mx.array, .to() → noop, etc.
│       │   ├── nn_modules.py         # torch.nn.Module → mlx.nn.Module adapter
│       │   ├── nn_layers.py          # Linear, Conv2d, LayerNorm, etc. mapped
│       │   ├── functional.py         # F.relu, F.softmax, F.cross_entropy → mlx
│       │   ├── optim.py              # Adam, AdamW, SGD → mlx.optimizers
│       │   ├── distributions.py      # torch.distributions → mlx implementations
│       │   ├── einops_mlx.py         # einops rearrange/repeat for mlx
│       │   └── vision.py             # torchvision transforms → mlx/numpy
│       │
│       ├── policies/                 # Mirror of upstream policies, using compat layer
│       │   ├── __init__.py
│       │   ├── act/
│       │   │   ├── configuration_act.py    # Copy from upstream (pure dataclass)
│       │   │   ├── modeling_act.py         # Ported: torch→mlx via compat
│       │   │   └── processor_act.py        # Ported processor
│       │   ├── diffusion/
│       │   │   ├── configuration_diffusion.py
│       │   │   ├── modeling_diffusion.py
│       │   │   └── processor_diffusion.py
│       │   ├── tdmpc/
│       │   ├── vqbet/
│       │   ├── sac/
│       │   ├── factory.py
│       │   ├── pretrained.py          # Load weights from HF Hub → mlx
│       │   └── utils.py
│       │
│       ├── model/                    # Shared model components (vision encoders, etc.)
│       │   └── ...
│       │
│       ├── datasets/                 # Dataset loading (keep upstream, just swap tensors)
│       │   ├── lerobot_dataset.py    # Minimal port: return mx.array instead of torch
│       │   └── ...
│       │
│       ├── processor/                # Processing pipeline
│       │   └── ...
│       │
│       ├── training/                 # MLX training loop (replaces accelerate)
│       │   ├── trainer.py            # mlx.nn.value_and_grad based training
│       │   └── data.py
│       │
│       ├── configs/                  # Config system (keep draccus, swap device logic)
│       │   └── ...
│       │
│       └── scripts/                  # CLI entry points
│           ├── train.py              # lerobot-mlx-train
│           └── eval.py               # lerobot-mlx-eval
│
├── tests/
│   ├── test_smoke.py                 # Package imports, MLX env
│   ├── test_compat.py                # torch→mlx shim correctness
│   ├── test_act.py                   # ACT policy forward/backward
│   ├── test_diffusion.py             # Diffusion policy forward/backward
│   ├── test_training.py              # End-to-end training convergence
│   └── conftest.py                   # Shared fixtures
│
└── prds/                             # PRD per component (our proven process)
    ├── PRD-01-compat-layer.md
    ├── PRD-02-act-policy.md
    ├── PRD-03-diffusion-policy.md
    └── ...
```

### The Compatibility Layer (`compat/`) — Design Details

This is the **key architectural decision** that makes upstream tracking easy. Instead of rewriting every policy from scratch, we create a shim that makes MLX look like PyTorch:

```python
# src/lerobot_mlx/compat/tensor_ops.py

import mlx.core as mx

def tensor(data, dtype=None, device=None):
    """Drop-in for torch.tensor(). Ignores device param."""
    return mx.array(data, dtype=_map_dtype(dtype))

def zeros(*shape, dtype=None, device=None):
    return mx.zeros(shape, dtype=_map_dtype(dtype))

def cat(tensors, dim=0):
    return mx.concatenate(tensors, axis=dim)

def stack(tensors, dim=0):
    return mx.stack(tensors, axis=dim)

# etc.
```

```python
# src/lerobot_mlx/compat/nn_modules.py

import mlx.nn as nn

class Module(nn.Module):
    """Adapter that adds torch.nn.Module-like methods to mlx.nn.Module."""

    def to(self, device=None, dtype=None):
        """No-op for device (MLX unified memory). Handles dtype."""
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)
```

**Why this works for upstream tracking:**
- When upstream changes `modeling_act.py`, we diff our version against theirs
- 90% of changes are structural (new layers, config fields) — these transfer directly
- Only torch-specific calls need remapping, and the compat layer handles most automatically
- Config dataclasses (`configuration_*.py`) copy verbatim — they're pure Python

### Weight Loading Strategy

```python
# Load PyTorch weights from HF Hub → convert to MLX
def load_pretrained_weights(repo_id: str, policy_class):
    """Download safetensors from HF Hub, convert torch→mlx."""
    # 1. Download .safetensors via huggingface_hub
    # 2. Load with mlx.core.load() (safetensors natively supported)
    # 3. Rename keys if needed (torch naming → mlx naming)
    # 4. model.load_weights(weights)
```

---

## Build Order (PRD sequence)

### Phase 1: Foundation (Week 1)
- **PRD-01: Dev Environment** — pyproject.toml, uv setup, test harness
- **PRD-02: Compatibility Layer** — tensor_ops, nn_modules, nn_layers, functional, optim
- **PRD-03: Compat Tests** — verify every shim against torch reference output

### Phase 2: First Policy (Week 2)
- **PRD-04: ACT Policy** — port modeling_act.py using compat layer
- **PRD-05: ACT Tests** — forward pass, backward, weight loading from HF Hub
- **PRD-06: Training Loop** — mlx.nn.value_and_grad training, synthetic data

### Phase 3: Second Policy (Week 3)
- **PRD-07: Diffusion Policy** — port modeling_diffusion.py (UNet + DDPM)
- **PRD-08: Diffusion Tests** — forward, denoising loop, training convergence

### Phase 4: Data Pipeline (Week 4)
- **PRD-09: Dataset Loading** — LeRobotDataset → mx.array output
- **PRD-10: Processor Pipeline** — normalize, delta actions, device placement

### Phase 5: Remaining Policies (Weeks 5-6)
- **PRD-11: TD-MPC**
- **PRD-12: VQ-BeT**
- **PRD-13: SAC**
- **PRD-14: Weight Conversion** — batch convert any HF Hub checkpoint

### Phase 6: VLA Foundation Models (Weeks 7+, stretch)
- **PRD-15: Pi0** (requires mlx-vlm or custom VLM port)
- **PRD-16: SmolVLA**

---

## Upstream Sync Protocol

When a new LeRobot version ships:

```bash
# 1. Update the reference
cd repositories/lerobot-upstream
git fetch origin
git diff v0.5.1..v0.6.0 -- src/lerobot/policies/ > /tmp/upstream-policies-diff.patch
git diff v0.5.1..v0.6.0 -- src/lerobot/model/ > /tmp/upstream-model-diff.patch

# 2. Review what changed
# - New policies? → Create new PRD, port using compat layer
# - Changed policy code? → Apply same changes to our mirror
# - New config fields? → Copy config file verbatim
# - New dependencies? → Check if MLX equivalent exists

# 3. Update UPSTREAM_VERSION.md
echo "Synced to: huggingface/lerobot@<commit-hash> (v0.6.0)" > UPSTREAM_VERSION.md
```

---

## Key Technical Decisions

### torch→mlx Mapping Reference

| PyTorch | MLX |
|---------|-----|
| `torch.tensor()` | `mx.array()` |
| `torch.nn.Module` | `mlx.nn.Module` |
| `torch.nn.Linear` | `mlx.nn.Linear` |
| `torch.nn.Conv2d` | `mlx.nn.Conv2d` |
| `torch.nn.LayerNorm` | `mlx.nn.LayerNorm` |
| `torch.nn.MultiheadAttention` | `mlx.nn.MultiHeadAttention` |
| `torch.nn.TransformerEncoder` | Custom (compose from mlx.nn) |
| `torch.nn.functional.relu` | `mlx.nn.relu` |
| `torch.nn.functional.gelu` | `mlx.nn.gelu` |
| `torch.nn.functional.softmax` | `mx.softmax` |
| `torch.nn.functional.cross_entropy` | `mlx.nn.losses.cross_entropy` |
| `torch.optim.Adam` | `mlx.optimizers.Adam` |
| `torch.optim.AdamW` | `mlx.optimizers.AdamW` |
| `torch.distributions.Normal` | Custom implementation |
| `torch.distributions.kl_divergence` | Custom implementation |
| `einops.rearrange` | Custom or `mx.reshape` + `mx.transpose` |
| `torchvision.transforms` | OpenCV / numpy / custom |
| `torch.no_grad()` | Not needed (MLX is lazy) |
| `tensor.to(device)` | No-op (unified memory) |
| `tensor.detach()` | `mx.stop_gradient()` |
| `tensor.requires_grad` | Implicit in `value_and_grad` |
| `model.parameters()` | `model.parameters()` (same API!) |
| `accelerate.Accelerator` | Not needed (single-device MLX) |

### What We DON'T Port

These modules work fine as-is or are hardware-specific (no torch dependency):

- `cameras/` — OpenCV, no torch
- `motors/` — Serial protocols, no torch
- `robots/` — Orchestration, minimal torch
- `teleoperators/` — Input devices, no torch
- `envs/` — Gymnasium, render-only
- `transport/` — gRPC, no torch

---

## Dev Commands

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/LeRobot-mlx

# Setup
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific policy test
pytest tests/test_act.py -v

# Cross-framework comparison (needs torch installed)
pytest tests/ -v -m "requires_torch"
```

---

## Requirements

| Requirement | Version |
|-------------|---------|
| macOS | Apple Silicon (M1/M2/M3/M4) |
| Python | >= 3.12 |
| MLX | >= 0.31.0 |
| NumPy | >= 2.0.0 |
| SciPy | >= 1.14.0 |
| huggingface-hub | >= 1.0.0 |
| safetensors | >= 0.4.0 |
| draccus | == 0.10.0 |

**Dev extras**: pytest, torch (for cross-framework validation)

---

## Success Criteria

### Phase 1 (MVP)
- [ ] ACT policy: forward pass matches PyTorch within atol=1e-3
- [ ] ACT policy: training converges on synthetic data
- [ ] ACT policy: load pretrained weights from HF Hub
- [ ] Diffusion policy: forward pass + denoising loop works
- [ ] 100+ tests, all passing

### Phase 2 (Usable)
- [ ] 5 policies ported (ACT, Diffusion, TD-MPC, VQ-BeT, SAC)
- [ ] Real dataset loading (LeRobotDataset → mx.array)
- [ ] CLI: `lerobot-mlx-train` and `lerobot-mlx-eval`
- [ ] Weight conversion from any HF Hub checkpoint

### Phase 3 (Complete)
- [ ] VLA policies (Pi0, SmolVLA)
- [ ] On-device inference benchmarks vs PyTorch MPS
- [ ] Documentation and examples

---

## Start Building

Begin with **PRD-01: Dev Environment**. Create `pyproject.toml`, set up the package structure, write smoke tests that verify MLX is available and the package imports cleanly. Then move to **PRD-02: Compatibility Layer** — this is the foundation everything else builds on.

Use the `/port-to-mlx` skill when porting individual policy files. Use `/write-tests` after each component. Follow the PRD-driven process from pointelligence-mlx.
