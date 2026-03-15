# Getting Started with LeRobot-MLX

LeRobot-MLX brings state-of-the-art robotics policies to Apple Silicon, running natively on the Metal GPU via MLX. No CUDA required.

## What You'll Learn

1. Install LeRobot-MLX on your Mac
2. Verify your setup works
3. Run your first robotics policy (5 lines of code)
4. Train a policy on synthetic data
5. Compare all 8 policies
6. Understand the architecture

## Prerequisites

- **Apple Silicon Mac** (M1, M2, M3, or M4 -- any variant)
- **Python 3.12+**
- **8 GB RAM** minimum (16 GB recommended for larger models)

## Step 1: Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/AIFLOW-LABS/LeRobot-mlx.git
cd LeRobot-mlx

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

To run cross-framework validation tests against PyTorch (optional):

```bash
pip install -e ".[torch]"
```

## Step 2: Verify Installation

Run the system info command to confirm everything is set up:

```bash
lerobot-mlx-info
```

You should see output like:

```
LeRobot-MLX v0.1.0
========================================
Platform:  macOS 15.3 (arm64)
Python:    3.12.12
MLX:       0.31.1
Metal GPU: Available

Policies (8 available):
  act          Action Chunking Transformer
  diffusion    Diffusion Policy (DDPM/DDIM)
  sac          Soft Actor-Critic
  tdmpc        Temporal Difference MPC
  vqbet        Vector Quantized Behavior Transformer
  sarm         State-Action Reward Model
  pi0          Pi0 Flow Matching (VLA)
  smolvla      SmolVLA Flow Matching (VLA)

VLM Backend: Not installed (pip install mlx-vlm)
Memory:    64 GB unified memory
```

Run the test suite to verify everything passes:

```bash
python -m pytest tests/ -x -q
```

## Step 3: Your First Policy (5 Lines of Code)

Create a file called `hello_policy.py`:

```python
import mlx.core as mx
from lerobot_mlx.policies.factory import make_policy

# Create an ACT policy with default config
policy = make_policy("act")

# Create a synthetic observation
batch = {
    "observation.state": mx.random.normal((1, 14)),
    "action": mx.random.normal((1, 100, 14)),
}

# Run a forward pass
output = policy(batch)
mx.eval(output["action"])

print(f"Action shape: {output['action'].shape}")
print(f"Loss: {output['loss'].item():.4f}")
```

Run it:

```bash
python hello_policy.py
```

## Step 4: Train a Policy

Train an ACT policy on synthetic data for a quick test:

```bash
lerobot-mlx-train \
  --policy-type act \
  --batch-size 32 \
  --training-steps 100 \
  --log-interval 10 \
  --output-dir outputs/act_test
```

This will:
- Create a synthetic dataset
- Train for 100 steps
- Log loss every 10 steps
- Save the final checkpoint to `outputs/act_test/`

For real data from HuggingFace Hub:

```bash
lerobot-mlx-train \
  --policy-type act \
  --dataset-repo-id lerobot/pusht \
  --batch-size 16 \
  --training-steps 10000 \
  --output-dir outputs/act_pusht
```

## Step 5: Compare All Policies

Run a quick benchmark across policies:

```bash
# Benchmark a single policy
lerobot-mlx-benchmark --policy act --batch-size 1

# Try different policies
lerobot-mlx-benchmark --policy diffusion --batch-size 4
lerobot-mlx-benchmark --policy tdmpc --batch-size 8
```

The benchmark reports latency (ms per forward pass) and throughput (samples/sec), so you can compare policies on your hardware.

## Step 6: Understanding the Architecture

LeRobot-MLX mirrors the original LeRobot structure but replaces PyTorch with MLX:

```
src/lerobot_mlx/
  policies/           # All 8 policy implementations
    act/              # Action Chunking Transformer
    diffusion/        # Diffusion Policy
    sac/              # Soft Actor-Critic
    tdmpc/            # Temporal Difference MPC
    vqbet/            # VQ-BeT
    sarm/             # State-Action Reward Model
    pi0/              # Pi0 (VLA)
    smolvla/          # SmolVLA (VLA)
    factory.py        # make_policy() -- creates any policy by name
    pretrained.py     # Load pretrained weights from HuggingFace
  models/             # Shared model components (transformers, diffusion, etc.)
  training/           # Training loop, optimizer, scheduler
  datasets/           # Dataset loading and synthetic data generation
  scripts/            # CLI entry points (train, eval, info, convert, benchmark)
```

Key concepts:

- **Policies** are MLX `nn.Module` subclasses. Each takes a batch dict and returns `{"action": ..., "loss": ...}`.
- **Factory** (`make_policy`) creates any policy by name string, with sensible defaults.
- **Pretrained** loading auto-converts PyTorch safetensors weights to MLX format.
- **Lazy evaluation**: MLX uses lazy evaluation. Call `mx.eval()` to materialize results.

## Step 7: Convert Pretrained Weights

Convert a PyTorch checkpoint from HuggingFace to MLX format:

```bash
lerobot-mlx-convert \
  --repo-id lerobot/act_aloha_sim_transfer_cube_human \
  --output-dir converted/act_aloha
```

## Next Steps

- **Load pretrained weights** from HuggingFace Hub with `load_pretrained()`
- **Integrate a VLM backbone** for vision-language policies (pi0, smolvla) via `mlx-vlm`
- **Run benchmarks** on your hardware to find the best policy for your use case
- **Contribute** -- see the PRDs in `prds/` for the full development roadmap

## Troubleshooting

### "No module named mlx"

MLX only runs on Apple Silicon. Make sure you are on an M1/M2/M3/M4 Mac:

```bash
python -c "import platform; print(platform.machine())"
# Should print: arm64
```

Install MLX:

```bash
pip install mlx
```

### "Metal is not available"

This usually means you are running on an Intel Mac or in a virtualized environment. LeRobot-MLX requires Apple Silicon with Metal support.

### Import errors after installation

Make sure you installed in editable mode from the project root:

```bash
pip install -e ".[dev]"
```

### Out of memory during training

Reduce the batch size:

```bash
lerobot-mlx-train --policy-type act --batch-size 8
```

For VLA models (pi0, smolvla), 16 GB RAM is recommended. Use batch size 1 if memory is tight.

### Tests failing

Run the full test suite with verbose output:

```bash
python -m pytest tests/ -v --tb=short
```

If only cross-framework tests fail, install PyTorch:

```bash
pip install -e ".[torch]"
```
