#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
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
#
# MLX port: configuration copied from upstream with minimal changes.
# Removed dependencies on upstream PreTrainedConfig, NormalizationMode, etc.
# This is a standalone dataclass config matching the upstream API.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PolicyFeature:
    """Describes a feature (input or output) of the policy."""
    type: str  # "STATE", "ACTION", "VISUAL", "ENV_STATE"
    shape: Tuple[int, ...]


@dataclass
class DiffusionConfig:
    """Configuration class for DiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive
    and single camera observations.

    This is a standalone MLX config that mirrors the upstream DiffusionConfig
    interface without depending on upstream PreTrainedConfig.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: Optional[Tuple[int, int]] = None
    crop_ratio: float = 1.0
    crop_shape: Optional[Tuple[int, int]] = None
    crop_is_random: bool = True
    pretrained_backbone_weights: Optional[str] = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # Unet.
    down_dims: Tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: Optional[int] = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Feature definitions (set by user or test harness)
    input_features: Dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: Dict[str, PolicyFeature] = field(default_factory=dict)

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: Tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    drop_n_last_frames: int = 7

    def __post_init__(self):
        """Input validation."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got horizon={self.horizon} and down_dims={self.down_dims}"
            )

    # -------------------------------------------------------------------------
    # Feature access helpers (mirror upstream PreTrainedConfig interface)
    # -------------------------------------------------------------------------

    @property
    def image_features(self) -> Dict[str, PolicyFeature]:
        """Return dict of image input features."""
        return {k: v for k, v in self.input_features.items() if v.type == "VISUAL"}

    @property
    def robot_state_feature(self) -> Optional[PolicyFeature]:
        """Return the robot state feature."""
        for k, v in self.input_features.items():
            if v.type == "STATE":
                return v
        return None

    @property
    def env_state_feature(self) -> Optional[PolicyFeature]:
        """Return the environment state feature, if any."""
        for k, v in self.input_features.items():
            if v.type == "ENV_STATE":
                return v
        return None

    @property
    def action_feature(self) -> Optional[PolicyFeature]:
        """Return the action output feature."""
        for k, v in self.output_features.items():
            if v.type == "ACTION":
                return v
        return None

    def validate_features(self) -> None:
        """Validate that required features are present."""
        if self.robot_state_feature is None:
            raise ValueError("You must provide a STATE input feature (observation.state).")
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        if self.action_feature is None:
            raise ValueError("You must provide an ACTION output feature.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
