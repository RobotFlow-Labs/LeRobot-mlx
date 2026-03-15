#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TD-MPC policy configuration — MLX port.

Mirrors upstream lerobot.policies.tdmpc.configuration_tdmpc.TDMPCConfig.
Decoupled from upstream PreTrainedConfig / draccus to avoid torch dependencies.
Uses simple dataclass with equivalent fields and validation.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Lightweight types (mirrors upstream lerobot.configs.types)
# ---------------------------------------------------------------------------

class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


class PolicyFeature:
    """Simple feature descriptor: type + shape."""
    def __init__(self, type: FeatureType, shape: tuple[int, ...]):
        self.type = type
        self.shape = shape


# Observation key constants
OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
OBS_IMAGE = "observation.image"
OBS_PREFIX = "observation."
OBS_STR = "observation"
ACTION = "action"
REWARD = "next.reward"


@dataclass
class TDMPCConfig:
    """Configuration class for TDMPCPolicy.

    Defaults are configured for training with xarm_lift_medium_replay providing
    proprioceptive and single camera observations.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    n_action_repeats: int = 2
    horizon: int = 5
    n_action_steps: int = 1

    # Feature specifications — to be set by the user
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture / modeling.
    image_encoder_hidden_dim: int = 32
    state_encoder_hidden_dim: int = 256
    latent_dim: int = 50
    q_ensemble_size: int = 5
    mlp_dim: int = 512
    # Reinforcement learning.
    discount: float = 0.9

    # Inference.
    use_mpc: bool = True
    cem_iterations: int = 6
    max_std: float = 2.0
    min_std: float = 0.05
    n_gaussian_samples: int = 512
    n_pi_samples: int = 51
    uncertainty_regularizer_coeff: float = 1.0
    n_elites: int = 50
    elite_weighting_temperature: float = 0.5
    gaussian_mean_momentum: float = 0.1

    # Training and loss computation.
    max_random_shift_ratio: float = 0.0476
    reward_coeff: float = 0.5
    expectile_weight: float = 0.9
    value_coeff: float = 0.1
    consistency_coeff: float = 20.0
    advantage_scaling: float = 3.0
    pi_coeff: float = 0.5
    temporal_decay_coeff: float = 0.5
    # Target model.
    target_model_momentum: float = 0.995

    # Training presets
    optimizer_lr: float = 3e-4

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if self.n_gaussian_samples <= 0:
            raise ValueError(
                f"The number of gaussian samples for CEM should be non-zero. Got `{self.n_gaussian_samples=}`"
            )
        if self.normalization_mapping.get("ACTION") is not None and \
           self.normalization_mapping["ACTION"] is not NormalizationMode.MIN_MAX:
            raise ValueError(
                "TD-MPC assumes the action space dimensions to all be in [-1, 1]. "
                "Therefore it is strongly advised that you stick with the default."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.n_action_steps > 1:
            if self.n_action_repeats != 1:
                raise ValueError(
                    "If `n_action_steps > 1`, `n_action_repeats` must be left to its default value of 1."
                )
            if not self.use_mpc:
                raise ValueError("If `n_action_steps > 1`, `use_mpc` must be set to `True`.")
            if self.n_action_steps > self.horizon:
                raise ValueError("`n_action_steps` must be less than or equal to `horizon`.")

    # ---- Feature helper properties (mirroring upstream PreTrainedConfig) ----

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """Return dict of features whose type is VISUAL."""
        return {k: v for k, v in self.input_features.items()
                if v.type == FeatureType.VISUAL}

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        """Return the robot state feature if present."""
        for k, v in self.input_features.items():
            if v.type == FeatureType.STATE:
                return v
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        """Return the environment state feature if present."""
        for k, v in self.input_features.items():
            if v.type == FeatureType.ENV:
                return v
        return None

    @property
    def action_feature(self) -> PolicyFeature | None:
        """Return the action feature."""
        for k, v in self.output_features.items():
            if v.type == FeatureType.ACTION:
                return v
        return None

    def validate_features(self) -> None:
        """Validate feature configuration."""
        if len(self.image_features) > 1:
            raise ValueError(
                f"TDMPCConfig handles at most one image for now. "
                f"Got image keys {list(self.image_features.keys())}."
            )
        if len(self.image_features) > 0:
            image_ft = next(iter(self.image_features.values()))
            if image_ft.shape[-2] != image_ft.shape[-1]:
                raise ValueError(
                    f"Only square images are handled now. Got image shape {image_ft.shape}."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(self.horizon + 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> list:
        return list(range(self.horizon))
