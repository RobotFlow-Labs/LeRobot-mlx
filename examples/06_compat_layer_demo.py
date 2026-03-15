#!/usr/bin/env python
"""The compat layer: write PyTorch-style code that runs on Apple Silicon via MLX.

LeRobot-MLX includes a compatibility layer that maps PyTorch APIs to MLX,
allowing policy code to be written in familiar PyTorch style while running
natively on Apple Silicon with MLX.

This example demonstrates:
  1. Tensor creation (torch-like API -> MLX arrays)
  2. nn.Module subclassing
  3. Loss computation
  4. Gradient computation via mlx.nn.value_and_grad
  5. Optimizer step
"""

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Import the compat layer — these provide PyTorch-like APIs
from lerobot_mlx.compat.tensor_ops import (
    zeros, ones, randn, cat, stack, Tensor, float32,
)
from lerobot_mlx.compat.nn_modules import Module
from lerobot_mlx.compat.nn_layers import Linear, LayerNorm, Dropout, Sequential, ReLU

# =========================================================================
# 1. Tensor creation — torch-style API, MLX backend
# =========================================================================
print("=" * 60)
print("1. TENSOR CREATION")
print("=" * 60)

# Create tensors just like PyTorch
z = zeros(3, 4)
o = ones(2, 3)
r = mx.random.normal((2, 4))  # randn equivalent

print(f"  zeros(3, 4)    shape={z.shape}, dtype={z.dtype}")
print(f"  ones(2, 3)     shape={o.shape}, dtype={o.dtype}")
print(f"  random(2, 4)   shape={r.shape}, dtype={r.dtype}")

# Concatenation and stacking
a = mx.random.normal((2, 3))
b = mx.random.normal((2, 3))
c = cat([a, b], dim=0)  # torch.cat -> mx.concatenate
s = stack([a, b], dim=0)  # torch.stack -> mx.stack
print(f"  cat([2x3, 2x3], dim=0)   -> {c.shape}")
print(f"  stack([2x3, 2x3], dim=0) -> {s.shape}")

# =========================================================================
# 2. nn.Module subclassing — identical to PyTorch patterns
# =========================================================================
print(f"\n{'=' * 60}")
print("2. NEURAL NETWORK MODULES")
print("=" * 60)


class SimplePolicy(Module):
    """A simple 2-layer MLP policy, written in PyTorch style."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = Sequential(
            Linear(obs_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, action_dim),
        )
        self.layer_norm = LayerNorm(action_dim)

    def __call__(self, x):
        return self.layer_norm(self.net(x))

    def compute_loss(self, batch):
        """MSE loss between predicted and target actions."""
        obs = batch["obs"]
        target = batch["target"]
        pred = self(obs)
        return mx.mean((pred - target) ** 2)


# Create and inspect the model
model = SimplePolicy(obs_dim=10, action_dim=4, hidden_dim=64)
n_params = model.num_parameters()
print(f"  SimplePolicy: {n_params:,} parameters")

# Train / eval mode (no-ops in MLX, but API-compatible)
model.train()
print(f"  model.train() -> training={model.training}")
model.eval()
print(f"  model.eval()  -> training={model.training}")

# .to() is a no-op (MLX uses unified memory)
model = model.to("cpu")
print("  model.to('cpu') -> no-op (unified memory)")

# =========================================================================
# 3. Forward pass
# =========================================================================
print(f"\n{'=' * 60}")
print("3. FORWARD PASS")
print("=" * 60)

obs = mx.random.normal((8, 10))  # batch of 8 observations
actions = model(obs)
mx.eval(actions)
print(f"  Input:  obs.shape = {obs.shape}")
print(f"  Output: actions.shape = {actions.shape}")
print(f"  Output sample: [{actions[0, 0].item():.4f}, {actions[0, 1].item():.4f}, ...]")

# =========================================================================
# 4. Loss computation and gradients
# =========================================================================
print(f"\n{'=' * 60}")
print("4. LOSS AND GRADIENTS")
print("=" * 60)

batch = {
    "obs": mx.random.normal((16, 10)),
    "target": mx.random.normal((16, 4)),
}

# Compute loss
model.train()
loss = model.compute_loss(batch)
mx.eval(loss)
print(f"  MSE loss: {loss.item():.4f}")

# Compute gradients using MLX's value_and_grad
loss_and_grad_fn = _nn.value_and_grad(model, lambda m, b: m.compute_loss(b))
loss_val, grads = loss_and_grad_fn(model, batch)
mx.eval(loss_val, grads)

flat_grads = tree_flatten(grads)
grad_norms = [(name, mx.sqrt(mx.sum(g ** 2)).item()) for name, g in flat_grads]
print(f"  Loss: {loss_val.item():.4f}")
print(f"  Gradient tensors: {len(grad_norms)}")
print(f"  Sample grad norms:")
for name, norm in grad_norms[:3]:
    print(f"    {name}: {norm:.6f}")

# =========================================================================
# 5. Optimizer step
# =========================================================================
print(f"\n{'=' * 60}")
print("5. OPTIMIZER STEP")
print("=" * 60)

optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)

# Training loop: 10 steps
losses = []
for step in range(10):
    loss_val, grads = loss_and_grad_fn(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss_val)
    losses.append(loss_val.item())

print(f"  Step  0 loss: {losses[0]:.4f}")
print(f"  Step  9 loss: {losses[-1]:.4f}")
print(f"  Loss decreased: {losses[-1] < losses[0]}")

# =========================================================================
# 6. Summary
# =========================================================================
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print("  The compat layer lets you write PyTorch-style code that runs")
print("  natively on Apple Silicon via MLX. Key mappings:")
print("    torch.zeros   -> compat.tensor_ops.zeros")
print("    torch.cat     -> compat.tensor_ops.cat")
print("    nn.Module     -> compat.nn_modules.Module")
print("    nn.Linear     -> compat.nn_layers.Linear (= mlx.nn.Linear)")
print("    nn.LayerNorm  -> compat.nn_layers.LayerNorm")
print("  All computations use MLX's lazy evaluation + Apple GPU.")
print("\nCompat layer demo complete!")
