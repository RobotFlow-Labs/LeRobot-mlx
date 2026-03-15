#!/usr/bin/env python

# Copyright 2025 Qianzhong Chen, Justin Yu, Mac Schwager, Pieter Abbeel, Yide Shentu, Philipp Wu
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

"""SARM policy configuration -- MLX port.

Mirrors upstream lerobot.policies.sarm.configuration_sarm.SARMConfig.
Decoupled from upstream PreTrainedConfig / draccus to avoid torch dependencies.
Uses simple dataclass with equivalent fields and validation.
"""
from dataclasses import dataclass, field


# Constants matching upstream
OBS_IMAGES = "observation.images"
OBS_STATE = "observation.state"


@dataclass
class SARMConfig:
    """Configuration class for SARM (Stage-Aware Reward Modeling).

    Supports three annotation modes:

    1. single_stage (default): No annotations needed. Uses the episode's task description
       as a single stage covering the entire episode.

    2. dense_only: Uses dense (fine-grained) annotations from VLM, with an auto-generated
       single sparse "task" stage covering the full episode.

    3. dual: Full dual-head mode with both sparse (high-level) and dense (fine-grained)
       annotations from VLM.
    """

    annotation_mode: str = "single_stage"
    n_obs_steps: int = 8
    frame_gap: int = 30
    max_rewind_steps: int = 4

    # Architecture params
    image_dim: int = 512
    text_dim: int = 512
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 8
    max_state_dim: int = 32
    drop_n_last_frames: int = 1
    batch_size: int = 64
    clip_batch_size: int = 64
    dropout: float = 0.1
    stage_loss_weight: float = 1.0

    rewind_probability: float = 0.8
    language_perturbation_probability: float = 0.2

    # Sparse annotations (high-level stages)
    num_sparse_stages: int = 1
    sparse_subtask_names: list | None = None
    sparse_temporal_proportions: list | None = None

    # Dense annotations (fine-grained stages)
    num_dense_stages: int | None = None
    dense_subtask_names: list | None = None
    dense_temporal_proportions: list | None = None

    pretrained_model_path: str | None = None
    device: str | None = None
    image_key: str = OBS_IMAGES + ".top"
    state_key: str = OBS_STATE

    # Populated by the processor
    input_features: dict = field(default_factory=dict)
    output_features: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.annotation_mode not in ["single_stage", "dense_only", "dual"]:
            raise ValueError(
                f"annotation_mode must be 'single_stage', 'dense_only', or 'dual', got {self.annotation_mode}"
            )

        if self.annotation_mode == "single_stage":
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]
            self.num_dense_stages = None
            self.dense_subtask_names = None
            self.dense_temporal_proportions = None

        elif self.annotation_mode == "dense_only":
            self.num_sparse_stages = 1
            self.sparse_subtask_names = ["task"]
            self.sparse_temporal_proportions = [1.0]

        if self.max_rewind_steps >= self.n_obs_steps:
            raise ValueError(
                f"max_rewind_steps ({self.max_rewind_steps}) must be less than n_obs_steps ({self.n_obs_steps})"
            )
        if self.num_sparse_stages < 1:
            raise ValueError(f"num_sparse_stages must be at least 1, got {self.num_sparse_stages}")
        if (
            self.annotation_mode in ["dense_only", "dual"]
            and self.num_dense_stages is not None
            and self.num_dense_stages < 2
        ):
            raise ValueError(f"num_dense_stages must be at least 2, got {self.num_dense_stages}")

    def validate_features(self) -> None:
        pass

    @property
    def uses_dual_heads(self) -> bool:
        """Whether the model uses dual heads (dense_only or dual annotation modes)."""
        return self.annotation_mode in ["dense_only", "dual"]

    @property
    def num_frames(self) -> int:
        """Total number of frames in sequence."""
        return 1 + self.n_obs_steps + self.max_rewind_steps

    @property
    def max_length(self) -> int:
        return self.num_frames

    @property
    def observation_delta_indices(self) -> list[int]:
        """Bidirectional frame sampling centered on target frame."""
        half_steps = self.n_obs_steps // 2

        past_deltas = [-self.frame_gap * i for i in range(half_steps, 0, -1)]
        future_deltas = [self.frame_gap * i for i in range(1, half_steps + 1)]
        obs_deltas = past_deltas + [0] + future_deltas

        rewind_deltas = [-self.frame_gap * (i + 1) for i in range(self.max_rewind_steps)]

        return obs_deltas + rewind_deltas

    @property
    def action_delta_indices(self) -> None:
        """SARM is a reward model, not an action policy."""
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
