#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
"""Pi0 policy configuration -- MLX port.

Mirrors upstream lerobot.policies.pi0.configuration_pi0.PI0Config.
Decoupled from upstream PreTrainedConfig / draccus to avoid torch dependencies.
Uses simple dataclass with equivalent fields and validation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Gemma variant configs (mirrors upstream get_gemma_config)
# ---------------------------------------------------------------------------

@dataclass
class GemmaVariantConfig:
    """Config for a Gemma model variant."""
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


GEMMA_VARIANTS = {
    "gemma_300m": GemmaVariantConfig(
        width=1024, depth=18, mlp_dim=4096,
        num_heads=8, num_kv_heads=1, head_dim=256,
    ),
    "gemma_2b": GemmaVariantConfig(
        width=2048, depth=18, mlp_dim=16_384,
        num_heads=8, num_kv_heads=1, head_dim=256,
    ),
}


def get_gemma_config(variant: str) -> GemmaVariantConfig:
    """Returns config for specified gemma variant."""
    if variant not in GEMMA_VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant}. Available: {list(GEMMA_VARIANTS.keys())}"
        )
    return GEMMA_VARIANTS[variant]


# ---------------------------------------------------------------------------
# Pi0 Config
# ---------------------------------------------------------------------------

@dataclass
class Pi0Config:
    """Configuration for the Pi0 VLA policy.

    Mirrors upstream PI0Config fields. The Gemma expert dimensions are derived
    from `action_expert_variant` rather than being set manually.
    """

    # VLM backbone variant
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # Observation / action dims
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Shorter state/action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters (mirrors upstream / openpi)
    num_inference_steps: int = 10
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Image resolution (PaliGemma expects square)
    image_resolution: Tuple[int, int] = (224, 224)

    # Training settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Optimizer settings (defaults from upstream)
    optimizer_lr: float = 2.5e-5
    optimizer_betas: Tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Observation/action feature keys
    input_features: Optional[Dict[str, Any]] = None
    output_features: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater "
                f"than chunk_size ({self.chunk_size})"
            )

    # --- Derived properties from the expert variant ---

    @property
    def expert_config(self) -> GemmaVariantConfig:
        return get_gemma_config(self.action_expert_variant)

    @property
    def expert_hidden_size(self) -> int:
        return self.expert_config.width

    @property
    def expert_intermediate_size(self) -> int:
        return self.expert_config.mlp_dim

    @property
    def expert_num_heads(self) -> int:
        return self.expert_config.num_heads

    @property
    def expert_num_kv_heads(self) -> int:
        return self.expert_config.num_kv_heads

    @property
    def expert_head_dim(self) -> int:
        return self.expert_config.head_dim

    @property
    def expert_num_layers(self) -> int:
        return self.expert_config.depth
