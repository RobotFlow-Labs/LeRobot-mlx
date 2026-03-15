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
"""Pi0-FAST policy configuration -- MLX port.

Pi0-FAST uses tokenized action generation instead of continuous flow matching.
Actions are discretized into tokens and generated autoregressively.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from lerobot_mlx.policies.pi0.configuration_pi0 import (
    Pi0Config,
    GemmaVariantConfig,
    get_gemma_config,
)


@dataclass
class Pi0FastConfig:
    """Configuration for the Pi0-FAST tokenized action policy.

    Instead of flow matching (Pi0), Pi0-FAST discretizes actions into tokens
    and generates them autoregressively via the Gemma expert.
    """

    # VLM backbone variant (shared with Pi0)
    paligemma_variant: str = "google/paligemma-3b-pt-224"
    action_expert_variant: str = "gemma_300m"

    # Action dimensions
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image resolution
    image_resolution: Tuple[int, int] = (224, 224)

    # Pi0-FAST specific: token-based actions
    max_action_tokens: int = 256
    action_vocab_size: int = 256  # Discrete action vocabulary
    temperature: float = 0.0  # Greedy decoding by default
    max_decoding_steps: int = 256
    use_kv_cache: bool = True

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
