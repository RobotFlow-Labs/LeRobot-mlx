#!/usr/bin/env python
"""Use the policy factory to create any supported policy by name.

Demonstrates the factory pattern that lets you create policies without
importing each one individually. Shows available_policies(), make_policy(),
and how to get config/policy classes programmatically.
"""

import mlx.core as mx

from lerobot_mlx.policies.factory import (
    available_policies,
    make_policy,
    get_policy_class,
    get_config_class,
)

# ---------------------------------------------------------------------------
# 1. List all registered policies
# ---------------------------------------------------------------------------
policies = available_policies()
print("Registered policies:")
for name in policies:
    print(f"  - {name}")

# ---------------------------------------------------------------------------
# 2. Create each registered policy using the factory
# ---------------------------------------------------------------------------
print("\nCreating policies via factory:")

# ACT policy — needs features configured to pass validation
from lerobot_mlx.policies.act.configuration_act import (
    ACTConfig, PolicyFeature as ACTPolicyFeature,
    FeatureType as ACTFeatureType, ACTION, OBS_STATE, OBS_ENV_STATE,
)

act_config = ACTConfig(
    chunk_size=10,
    n_action_steps=10,
    input_features={
        OBS_STATE: ACTPolicyFeature(type=ACTFeatureType.STATE, shape=(14,)),
        OBS_ENV_STATE: ACTPolicyFeature(type=ACTFeatureType.ENV, shape=(14,)),
    },
    output_features={
        ACTION: ACTPolicyFeature(type=ACTFeatureType.ACTION, shape=(14,)),
    },
    dim_model=64, n_heads=4, dim_feedforward=128,
    n_encoder_layers=2, n_decoder_layers=1,
    n_vae_encoder_layers=2, latent_dim=8,
    pretrained_backbone_weights=None,
)
act_policy = make_policy(act_config)
print(f"  ACT       : {act_policy.num_parameters():>10,} params")

# Diffusion policy
from lerobot_mlx.policies.diffusion.configuration_diffusion import (
    DiffusionConfig, PolicyFeature as DiffPolicyFeature,
)

diff_config = DiffusionConfig(
    n_obs_steps=2, horizon=16, n_action_steps=8,
    input_features={
        "observation.state": DiffPolicyFeature(type="STATE", shape=(2,)),
        "observation.environment_state": DiffPolicyFeature(type="ENV_STATE", shape=(2,)),
    },
    output_features={
        "action": DiffPolicyFeature(type="ACTION", shape=(2,)),
    },
    down_dims=(64, 128, 256),
    num_train_timesteps=10,
    diffusion_step_embed_dim=32,
    n_groups=4, kernel_size=3,
    spatial_softmax_num_keypoints=8,
)
diff_policy = make_policy(diff_config)
print(f"  Diffusion : {diff_policy.num_parameters():>10,} params")

# SAC policy
from lerobot_mlx.policies.sac.configuration_sac import (
    SACConfig, FeatureShape,
    ACTION as SAC_ACTION, OBS_STATE as SAC_OBS_STATE,
)

sac_config = SACConfig(
    input_features={SAC_OBS_STATE: FeatureShape(shape=(10,))},
    output_features={SAC_ACTION: FeatureShape(shape=(4,))},
    latent_dim=64,
    num_critics=2,
)
sac_policy = make_policy(sac_config)
print(f"  SAC       : {sac_policy.num_parameters():>10,} params")

# ---------------------------------------------------------------------------
# 3. Show how to get classes programmatically
# ---------------------------------------------------------------------------
print("\nProgrammatic class access:")
for name in policies:
    policy_cls = get_policy_class(name)
    config_cls = get_config_class(name)
    print(f"  {name:12s} -> Policy: {policy_cls.__name__}, Config: {config_cls.__name__}")

print("\nFactory example complete!")
