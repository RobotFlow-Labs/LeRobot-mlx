# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

"""SAC configuration — COPIED VERBATIM from upstream (pure dataclasses, no torch dependency).

Only change: removed imports that reference upstream-specific modules
(PreTrainedConfig, NormalizationMode, MultiAdamConfig, constants).
These are replaced with minimal local stubs/equivalents.
"""

from dataclasses import dataclass, field


# --- Minimal stubs for upstream imports ---

# Upstream constants
ACTION = "action"
OBS_IMAGE = "observation.image"
OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"


def is_image_feature(key: str) -> bool:
    """Check if a feature key represents an image feature."""
    return key.startswith(OBS_IMAGE)


@dataclass
class FeatureShape:
    """Minimal feature shape descriptor."""
    shape: tuple

    def __init__(self, shape: tuple):
        self.shape = shape


@dataclass
class ConcurrencyConfig:
    actor: str = "threads"
    learner: str = "threads"


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@dataclass
class CriticNetworkConfig:
    hidden_dims: list = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    final_activation: str | None = None


@dataclass
class ActorNetworkConfig:
    hidden_dims: list = field(default_factory=lambda: [256, 256])
    activate_final: bool = True


@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    std_min: float = 1e-5
    std_max: float = 10.0
    init_final: float = 0.05


@dataclass
class SACConfig:
    """Soft Actor-Critic (SAC) configuration.

    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework.
    """

    # Features
    input_features: dict = field(default_factory=lambda: {
        OBS_STATE: FeatureShape(shape=(10,)),
    })
    output_features: dict = field(default_factory=lambda: {
        ACTION: FeatureShape(shape=(4,)),
    })

    # Architecture
    device: str = "cpu"
    storage_device: str = "cpu"
    vision_encoder_name: str | None = None
    freeze_vision_encoder: bool = True
    image_encoder_hidden_dim: int = 32
    shared_encoder: bool = True
    num_discrete_actions: int | None = None
    image_embedding_pooling_dim: int = 8

    # Training parameters
    online_steps: int = 1000000
    online_buffer_capacity: int = 100000
    offline_buffer_capacity: int = 100000
    async_prefetch: bool = False
    online_step_before_learning: int = 100
    policy_update_freq: int = 1

    # SAC algorithm parameters
    discount: float = 0.99
    temperature_init: float = 1.0
    num_critics: int = 2
    num_subsample_critics: int | None = None
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    temperature_lr: float = 3e-4
    critic_target_update_weight: float = 0.005
    utd_ratio: int = 1
    state_encoder_hidden_dim: int = 256
    latent_dim: int = 256
    target_entropy: float | None = None
    use_backup_entropy: bool = True
    grad_clip_norm: float = 40.0

    # Network configuration
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Optimizations
    use_torch_compile: bool = False  # Not applicable in MLX

    def validate_features(self) -> None:
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation "
                "(key starting with 'observation.image') in the input features"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list:
        return [key for key in self.input_features if is_image_feature(key)]
