#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
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
"""VQ-BeT policy configuration -- MLX port.

Mirrors upstream lerobot.policies.vqbet.configuration_vqbet.VQBeTConfig.
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


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


# ---------------------------------------------------------------------------
# String constants (mirrors upstream lerobot.utils.constants)
# ---------------------------------------------------------------------------

OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
OBS_IMAGES = "observation.images"
ACTION = "action"


# ---------------------------------------------------------------------------
# VQBeTConfig
# ---------------------------------------------------------------------------

@dataclass
class VQBeTConfig:
    """Configuration class for VQ-BeT.

    Defaults are configured for training with PushT providing proprioceptive
    and single camera observations.

    Mirrors upstream VQBeTConfig field-for-field.
    """

    # Input / output structure
    n_obs_steps: int = 5
    n_action_pred_token: int = 3
    action_chunk_size: int = 5

    # Features -- dict of name -> PolicyFeature
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture -- Vision backbone
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32

    # VQ-VAE
    n_vqvae_training_steps: int = 20000
    vqvae_n_embed: int = 16
    vqvae_embedding_dim: int = 256
    vqvae_enc_hidden_dim: int = 128

    # VQ-BeT (GPT)
    gpt_block_size: int = 500
    gpt_input_dim: int = 512
    gpt_output_dim: int = 512
    gpt_n_layer: int = 8
    gpt_n_head: int = 8
    gpt_hidden_dim: int = 512
    dropout: float = 0.1

    # Loss weights
    offset_loss_weight: float = 10000.0
    primary_code_loss_weight: float = 5.0
    secondary_code_loss_weight: float = 0.5
    bet_softmax_temperature: float = 0.1
    sequentially_select: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_vqvae_lr: float = 1e-3
    optimizer_vqvae_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        """Input validation (mirrors upstream)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

    # -----------------------------------------------------------------------
    # Feature property helpers (mirrors upstream PreTrainedConfig)
    # -----------------------------------------------------------------------

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def validate_features(self) -> None:
        """Validate required features (mirrors upstream)."""
        if not len(self.image_features) == 1:
            raise ValueError("You must provide only one image among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, self.n_action_pred_token + self.action_chunk_size - 1))

    @property
    def reward_delta_indices(self) -> None:
        return None
