#!/usr/bin/env python
"""LeRobot-MLX Quick Start: Create and run an ACT policy in 10 lines.

Demonstrates the minimal code needed to instantiate an ACT (Action Chunking
Transformer) policy, run a forward pass on synthetic data, and inspect the
output. No external data or pretrained weights required.
"""

import mlx.core as mx

from lerobot_mlx.policies.act.configuration_act import (
    ACTConfig, PolicyFeature, FeatureType, ACTION, OBS_STATE, OBS_ENV_STATE,
)
from lerobot_mlx.policies.act.modeling_act import ACTPolicy

# ---------------------------------------------------------------------------
# 1. Configure a small ACT policy for a 14-DOF robot arm
# ---------------------------------------------------------------------------
STATE_DIM = 14
ACTION_DIM = 14
CHUNK_SIZE = 10  # predict 10 future action steps at once

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
    # Small model for fast demo
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

# ---------------------------------------------------------------------------
# 2. Create the policy
# ---------------------------------------------------------------------------
policy = ACTPolicy(config)
print(f"Policy type      : {policy.name}")
print(f"Parameter count  : {policy.num_parameters():,}")

# ---------------------------------------------------------------------------
# 3. Create a fake observation batch (batch_size=2)
# ---------------------------------------------------------------------------
batch_size = 2
batch = {
    OBS_STATE: mx.random.normal((batch_size, STATE_DIM)),
    OBS_ENV_STATE: mx.random.normal((batch_size, STATE_DIM)),
}

# ---------------------------------------------------------------------------
# 4. Run inference (eval mode) — predict action chunks
# ---------------------------------------------------------------------------
policy.eval()
actions, (mu, log_sigma_x2) = policy.model(batch)
mx.eval(actions)

print(f"\nInference output shape: {actions.shape}")
print(f"  -> (batch={actions.shape[0]}, chunk={actions.shape[1]}, action_dim={actions.shape[2]})")

# ---------------------------------------------------------------------------
# 5. Run training forward pass — compute loss
# ---------------------------------------------------------------------------
policy.train()
train_batch = dict(batch)
train_batch[ACTION] = mx.random.normal((batch_size, CHUNK_SIZE, ACTION_DIM))
train_batch["action_is_pad"] = mx.zeros((batch_size, CHUNK_SIZE), dtype=mx.bool_)

loss, loss_dict = policy.forward(train_batch)
mx.eval(loss)

print(f"\nTraining loss    : {loss.item():.4f}")
print(f"  L1 loss        : {loss_dict['l1_loss']:.4f}")
if "kld_loss" in loss_dict:
    print(f"  KL divergence  : {loss_dict['kld_loss']:.4f}")

print("\nQuick start complete!")
