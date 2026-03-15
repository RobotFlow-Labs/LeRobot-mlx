#!/usr/bin/env python
"""Train a Diffusion policy on synthetic data using MLX.

Demonstrates:
  1. Configure a Diffusion Policy (noise-conditioned UNet)
  2. Create synthetic batches matching the expected observation format
  3. Train for 30 steps using a manual training loop
  4. Report the loss curve
  5. Save a checkpoint
"""

import pathlib
import tempfile
import time

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from lerobot_mlx.policies.diffusion.configuration_diffusion import (
    DiffusionConfig, PolicyFeature,
)
from lerobot_mlx.policies.diffusion.modeling_diffusion import DiffusionPolicy

# ---------------------------------------------------------------------------
# 1. Policy configuration (small model, fast training)
# ---------------------------------------------------------------------------
STATE_DIM = 2
ACTION_DIM = 2
N_OBS_STEPS = 2
HORIZON = 16
N_ACTION_STEPS = 8
BATCH_SIZE = 8
NUM_STEPS = 30

config = DiffusionConfig(
    n_obs_steps=N_OBS_STEPS,
    horizon=HORIZON,
    n_action_steps=N_ACTION_STEPS,
    input_features={
        "observation.state": PolicyFeature(type="STATE", shape=(STATE_DIM,)),
        "observation.environment_state": PolicyFeature(type="ENV_STATE", shape=(STATE_DIM,)),
    },
    output_features={
        "action": PolicyFeature(type="ACTION", shape=(ACTION_DIM,)),
    },
    down_dims=(64, 128, 256),
    num_train_timesteps=10,
    diffusion_step_embed_dim=32,
    n_groups=4,
    kernel_size=3,
    use_group_norm=True,
    use_film_scale_modulation=True,
    spatial_softmax_num_keypoints=8,
)

policy = DiffusionPolicy(config)
print(f"Diffusion policy created  -- {policy.num_parameters():,} parameters")

# ---------------------------------------------------------------------------
# 2. Synthetic data generator
# ---------------------------------------------------------------------------
def make_batch():
    """Generate a single random batch in the format DiffusionPolicy expects."""
    return {
        "observation.state": mx.random.normal(
            (BATCH_SIZE, N_OBS_STEPS, STATE_DIM)
        ),
        "observation.environment_state": mx.random.normal(
            (BATCH_SIZE, N_OBS_STEPS, STATE_DIM)
        ),
        "action": mx.random.normal(
            (BATCH_SIZE, HORIZON, ACTION_DIM)
        ),
        "action_is_pad": mx.zeros(
            (BATCH_SIZE, HORIZON), dtype=mx.bool_
        ),
    }

# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
# We train the inner DiffusionModel directly and use a flat parameter update
# to avoid tree structure mismatches between gradient and parameter trees.

diffusion_model = policy.diffusion

def compute_loss(model, batch):
    """Compute noise prediction loss."""
    return model.compute_loss(batch)

loss_and_grad_fn = _nn.value_and_grad(diffusion_model, compute_loss)

# Use a simple SGD-style update with flat parameter lists to avoid
# tree_map KeyError issues that can occur with wrapped Conv1d modules.
lr = 1e-4

print(f"\nTraining for {NUM_STEPS} steps...")
diffusion_model.train()
losses = []
t0 = time.time()

for step in range(NUM_STEPS):
    batch = make_batch()

    loss, grads = loss_and_grad_fn(diffusion_model, batch)

    # Flatten gradients and parameters for manual update
    flat_grads = tree_flatten(grads)
    flat_params = dict(tree_flatten(diffusion_model.trainable_parameters()))

    # Apply gradient update only for matching keys
    updates = []
    for key, grad in flat_grads:
        if key in flat_params and grad is not None:
            updates.append((key, flat_params[key] - lr * grad))

    if updates:
        diffusion_model.load_weights(updates)

    mx.eval(diffusion_model.parameters(), loss)

    loss_val = loss.item()
    losses.append(loss_val)

    if step % 10 == 0:
        print(f"  step {step:3d}: loss={loss_val:.4f}")

elapsed = time.time() - t0

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
print(f"\nTraining complete in {elapsed:.1f}s ({len(losses)} steps)")
print(f"  First loss  : {losses[0]:.4f}")
print(f"  Last loss   : {losses[-1]:.4f}")
print(f"  Min loss    : {min(losses):.4f}")
print(f"  Steps/sec   : {len(losses) / elapsed:.1f}")

# Simple ASCII loss curve
n_points = min(10, len(losses))
step_size = max(1, len(losses) // n_points)
sampled = [losses[i * step_size] for i in range(n_points)]
max_loss = max(sampled) if sampled else 1.0
bar_width = 30
print("\nLoss curve:")
for i, l in enumerate(sampled):
    bar = "#" * int(bar_width * l / max_loss) if max_loss > 0 else ""
    print(f"  step {i * step_size:3d} | {bar:<{bar_width}} | {l:.4f}")

# ---------------------------------------------------------------------------
# 5. Checkpoint
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = pathlib.Path(tmpdir) / f"diffusion_step_{NUM_STEPS}.npz"
    weights = dict(tree_flatten(policy.parameters()))
    mx.savez(str(ckpt_path), **weights)
    print(f"\nCheckpoint saved to: {ckpt_path}")

print("\nDiffusion training example complete!")
