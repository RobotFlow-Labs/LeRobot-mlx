#!/usr/bin/env python
"""Train an ACT policy on synthetic data using MLX.

Demonstrates the full training pipeline:
  1. Configure an ACT policy
  2. Create a SyntheticDataset and SimpleDataLoader
  3. Run a training loop for 50 steps using mlx.nn.value_and_grad
  4. Print the loss curve
  5. Save a checkpoint
"""

import tempfile
import time

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from lerobot_mlx.policies.act.configuration_act import (
    ACTConfig, PolicyFeature, FeatureType, ACTION, OBS_STATE, OBS_ENV_STATE,
)
from lerobot_mlx.policies.act.modeling_act import ACTPolicy
from lerobot_mlx.datasets.lerobot_dataset import SyntheticDataset
from lerobot_mlx.datasets.dataloader import SimpleDataLoader

# ---------------------------------------------------------------------------
# 1. Policy configuration (small model for fast training)
# ---------------------------------------------------------------------------
STATE_DIM = 14
ACTION_DIM = 14
CHUNK_SIZE = 10
BATCH_SIZE = 8
NUM_STEPS = 50

config = ACTConfig(
    chunk_size=CHUNK_SIZE,
    n_action_steps=CHUNK_SIZE,
    input_features={
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(STATE_DIM,)),
    },
    output_features={
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    },
    dim_model=64,
    n_heads=4,
    dim_feedforward=128,
    n_encoder_layers=2,
    n_decoder_layers=1,
    n_vae_encoder_layers=2,
    latent_dim=8,
    use_vae=True,
    dropout=0.0,
    pretrained_backbone_weights=None,
)

policy = ACTPolicy(config)
print(f"ACT policy created  -- {policy.num_parameters():,} parameters")

# ---------------------------------------------------------------------------
# 2. Synthetic dataset and dataloader
# ---------------------------------------------------------------------------
dataset = SyntheticDataset(
    num_samples=200,
    obs_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    chunk_size=CHUNK_SIZE,
    image_shape=None,   # state-only, no images
    seed=42,
)
dataloader = SimpleDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, seed=42)

print(f"Dataset            -- {len(dataset)} samples, batch_size={BATCH_SIZE}")
print(f"Batches per epoch  -- {len(dataloader)}")

# ---------------------------------------------------------------------------
# 3. Training loop setup
# ---------------------------------------------------------------------------
# ACTPolicy.forward(batch) returns (loss, loss_dict).
# We use mlx.nn.value_and_grad to compute gradients of the loss.

def compute_loss(model, batch):
    """Wrapper for mlx.nn.value_and_grad: returns scalar loss."""
    loss, _loss_dict = model.forward(batch)
    return loss

loss_and_grad_fn = _nn.value_and_grad(policy, compute_loss)
optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=1e-4)

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
print(f"\nTraining for {NUM_STEPS} steps...")
policy.train()
losses = []
t0 = time.time()
step = 0

while step < NUM_STEPS:
    for batch in dataloader:
        if step >= NUM_STEPS:
            break

        # Add fields ACTPolicy expects beyond what SyntheticDataset provides
        bs = batch["observation.state"].shape[0]
        batch[OBS_ENV_STATE] = mx.random.normal((bs, STATE_DIM))
        batch["action_is_pad"] = mx.zeros((bs, CHUNK_SIZE), dtype=mx.bool_)

        # Forward + backward
        loss, grads = loss_and_grad_fn(policy, batch)

        # Gradient clipping
        flat_grads = tree_flatten(grads)
        grad_arrays = [g for _, g in flat_grads if g is not None]
        total_norm = mx.sqrt(mx.sum(mx.stack([mx.sum(g ** 2) for g in grad_arrays])))
        clip_coef = min(1.0, 1.0 / (total_norm.item() + 1e-6))
        if clip_coef < 1.0:
            from mlx.utils import tree_unflatten
            clipped = [(k, g * clip_coef) for k, g in flat_grads]
            grads = tree_unflatten(clipped)

        # Update
        optimizer.update(policy, grads)
        mx.eval(policy.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 10 == 0:
            print(f"  step {step:3d}: loss={loss_val:.4f}")
        step += 1

elapsed = time.time() - t0

# ---------------------------------------------------------------------------
# 5. Print loss curve summary
# ---------------------------------------------------------------------------
print(f"\nTraining complete in {elapsed:.1f}s ({len(losses)} steps)")
print(f"  First loss  : {losses[0]:.4f}")
print(f"  Last loss   : {losses[-1]:.4f}")
print(f"  Min loss    : {min(losses):.4f}")
print(f"  Steps/sec   : {len(losses) / elapsed:.1f}")

# Simple ASCII loss curve (10 points)
n_points = min(10, len(losses))
step_size = max(1, len(losses) // n_points)
sampled = [losses[i * step_size] for i in range(n_points)]
max_loss = max(sampled)
bar_width = 30
print("\nLoss curve:")
for i, l in enumerate(sampled):
    bar = "#" * int(bar_width * l / max_loss) if max_loss > 0 else ""
    print(f"  step {i * step_size:3d} | {bar:<{bar_width}} | {l:.4f}")

# ---------------------------------------------------------------------------
# 6. Save checkpoint
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    import pathlib
    ckpt_path = pathlib.Path(tmpdir) / f"act_step_{NUM_STEPS}.npz"
    weights = dict(tree_flatten(policy.parameters()))
    mx.savez(str(ckpt_path), **weights)
    print(f"\nCheckpoint saved to: {ckpt_path}")

print("\nACT training example complete!")
