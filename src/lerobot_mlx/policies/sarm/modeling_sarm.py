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

"""SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation -- MLX port.

Paper: https://arxiv.org/abs/2509.25358

- StageTransformer: Predicts stage classification (sparse/dense)
- SubtaskTransformer: Predicts within-stage progress (tau) conditioned on stage
"""

import logging
import random

import numpy as np

import mlx.core as mx
import mlx.nn as _nn

from lerobot_mlx.compat import F, Tensor
from lerobot_mlx.compat.nn_modules import Module
from lerobot_mlx.compat.nn_layers import (
    Linear,
    LayerNorm,
    ModuleDict,
    ModuleList,
    Parameter,
    Sequential,
    ReLU,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from lerobot_mlx.compat.tensor_ops import (
    zeros,
    zeros_like,
    ones,
    cat,
    arange,
    eye,
    full,
    clamp,
    float32,
)

from lerobot_mlx.policies.sarm.configuration_sarm import SARMConfig


# ---------------------------------------------------------------------------
# Utility: normalize_stage_tau  (pure MLX, mirrors upstream sarm_utils)
# ---------------------------------------------------------------------------

def _temporal_proportions_to_breakpoints(
    temporal_proportions: list[float] | None,
    subtask_names: list[str] | None = None,
) -> list[float] | None:
    """Convert temporal proportions to cumulative breakpoints."""
    if temporal_proportions is None:
        return None
    proportions = list(temporal_proportions)
    total = sum(proportions)
    if total > 0 and abs(total - 1.0) > 1e-6:
        proportions = [p / total for p in proportions]
    breakpoints = [0.0]
    cumsum = 0.0
    for prop in proportions:
        cumsum += prop
        breakpoints.append(cumsum)
    breakpoints[-1] = 1.0
    return breakpoints


def normalize_stage_tau(
    x: mx.array,
    num_stages: int | None = None,
    breakpoints: list[float] | None = None,
    temporal_proportions: list[float] | None = None,
    subtask_names: list[str] | None = None,
) -> mx.array:
    """Normalize stage+tau reward to [0, 1] with custom breakpoints.

    Priority: breakpoints > temporal_proportions > linear fallback.
    """
    if breakpoints is not None:
        num_stages = len(breakpoints) - 1
    elif temporal_proportions is not None:
        breakpoints = _temporal_proportions_to_breakpoints(temporal_proportions, subtask_names)
        num_stages = len(breakpoints) - 1
    elif num_stages is not None:
        breakpoints = [i / num_stages for i in range(num_stages + 1)]
    else:
        raise ValueError("Either num_stages, breakpoints, or temporal_proportions must be provided")

    result = mx.zeros_like(x)
    for i in range(num_stages):
        mask = (x >= i) & (x < i + 1)
        tau_in_stage = x - i
        stage_result = breakpoints[i] + tau_in_stage * (breakpoints[i + 1] - breakpoints[i])
        result = mx.where(mask, stage_result, result)
    result = mx.where(x >= num_stages, 1.0, result)
    return mx.clip(result, 0.0, 1.0)


def pad_state_to_max_dim(state: mx.array, max_state_dim: int) -> mx.array:
    """Pad the state tensor's last dimension to max_state_dim with zeros."""
    current_dim = state.shape[-1]
    if current_dim >= max_state_dim:
        return state[..., :max_state_dim]
    # Pad with zeros on the right
    pad_width = [(0, 0)] * (state.ndim - 1) + [(0, max_state_dim - current_dim)]
    return mx.pad(state, pad_width)


# ---------------------------------------------------------------------------
# StageTransformer
# ---------------------------------------------------------------------------

class StageTransformer(Module):
    """Stage classification transformer for SARM.

    Predicts which stage/subtask the current frame belongs to.
    Supports both sparse (high-level) and dense (fine-grained) annotation schemes.

    Input streams: [vis_proj, lang_proj, state_proj] concatenated -> (B, N+2, T, D)
    Output: stage logits (B, T, num_classes)
    """

    def __init__(
        self,
        d_model: int = 512,
        vis_emb_dim: int = 512,
        text_emb_dim: int = 512,
        state_dim: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_cameras: int = 1,
        num_classes_sparse: int = 4,
        num_classes_dense: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        # Projections
        self.lang_proj = Linear(text_emb_dim, d_model)
        self.visual_proj = Linear(vis_emb_dim, d_model)
        self.state_proj = Linear(state_dim, d_model)

        # Encoder
        enc_layer = TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = TransformerEncoder(enc_layer, n_layers)

        # Positional bias on first visual frame
        self.first_pos = mx.zeros((1, d_model))

        # Shared fusion MLP
        fused_in = d_model * (num_cameras + 2)
        self.fusion_ln = LayerNorm(fused_in)
        self.fusion_linear = Linear(fused_in, d_model)

        # Scheme-specific heads
        self.heads = ModuleDict(
            {
                "sparse": Linear(d_model, num_classes_sparse),
                "dense": Linear(d_model, num_classes_dense),
            }
        )

    def _prep_lang(self, lang_emb: mx.array, B: int, T: int, D: int) -> mx.array:
        """Prepare language embeddings for fusion.

        Accepts lang_emb of shape:
          - (B, text_emb_dim) -> broadcast across time
          - (B, T, text_emb_dim) -> per-timestep

        Returns: (B, 1, T, D)
        """
        if lang_emb.ndim == 3:
            # (B, T, E) -> (B, T, D) -> (B, 1, T, D)
            lang_proj = mx.expand_dims(self.lang_proj(lang_emb), axis=1)
        else:
            # (B, E) -> (B, D) -> (B, 1, 1, D) -> broadcast to (B, 1, T, D)
            proj = self.lang_proj(lang_emb)  # (B, D)
            proj = mx.expand_dims(mx.expand_dims(proj, axis=1), axis=1)  # (B, 1, 1, D)
            lang_proj = mx.broadcast_to(proj, (B, 1, T, D))
        return lang_proj

    def __call__(
        self,
        img_seq: mx.array,      # (B, N, T, vis_emb_dim)
        lang_emb: mx.array,     # (B, E) or (B, T, E)
        state: mx.array,        # (B, T, state_dim)
        lengths: mx.array,      # (B,) - valid sequence lengths
        scheme: str = "sparse",  # "sparse" or "dense"
    ) -> mx.array:
        """Forward pass for stage classification.

        Returns:
            Stage logits (B, T, num_classes)
        """
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."

        B, N, T, _ = img_seq.shape
        D = self.d_model

        # Project inputs
        vis_proj = self.visual_proj(img_seq)                     # (B, N, T, D)
        state_proj = mx.expand_dims(self.state_proj(state), axis=1)  # (B, 1, T, D)
        lang_proj = self._prep_lang(lang_emb, B, T, D)          # (B, 1, T, D)

        # Concatenate streams: (B, N+2, T, D)
        x = mx.concatenate([vis_proj, lang_proj, state_proj], axis=1)

        # Add positional bias to first visual frame
        # x[:, :N, 0, :] += self.first_pos
        first_slice = x[:, :N, 0:1, :] + self.first_pos
        x = mx.concatenate([
            mx.concatenate([first_slice, x[:, :N, 1:, :]], axis=2),
            x[:, N:, :, :]
        ], axis=1)

        # Flatten to tokens for Transformer: (B, (N+2)*T, D)
        x_tokens = x.reshape(B, (N + 2) * T, D)
        L = x_tokens.shape[1]

        # Create causal mask for transformer
        causal_mask = mx.triu(mx.ones((L, L)), k=1).astype(mx.bool_)

        # Encode (compat TransformerEncoder takes mask, src_key_padding_mask)
        h = self.transformer(x_tokens, mask=causal_mask)

        # Reshape and fuse
        h = h.reshape(B, N + 2, T, D)
        h = mx.transpose(h, axes=(0, 2, 1, 3))  # (B, T, N+2, D)
        h = h.reshape(B, T, (N + 2) * D)
        fused = F.relu(self.fusion_linear(self.fusion_ln(h)))  # (B, T, D)

        # Scheme-specific logits
        logits = self.heads[scheme](fused)  # (B, T, num_classes)
        return logits


# ---------------------------------------------------------------------------
# SubtaskTransformer
# ---------------------------------------------------------------------------

class SubtaskTransformer(Module):
    """Subtask progress regression transformer for SARM.

    Predicts within-stage normalized progress (tau) conditioned on stage prior.

    Input streams: [vis_proj, lang_proj, state_proj, stage_emb] -> (B, N+3, T, D)
    Output: tau predictions (B, T) in [0, 1]
    """

    def __init__(
        self,
        d_model: int = 512,
        vis_emb_dim: int = 512,
        text_emb_dim: int = 512,
        state_dim: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_cameras: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras

        # Projections
        self.lang_proj = Linear(text_emb_dim, d_model)
        self.visual_proj = Linear(vis_emb_dim, d_model)
        self.state_proj = Linear(state_dim, d_model)

        # Encoder
        enc = TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.transformer = TransformerEncoder(enc, n_layers)

        # Learned bias on first visual frame
        self.first_pos = mx.zeros((1, d_model))

        # Shared fusion backbone
        fused_in = d_model * (num_cameras + 3)
        self.fusion_ln = LayerNorm(fused_in)
        self.fusion_linear = Linear(fused_in, d_model)

        # Scheme-specific regression heads
        self.heads = ModuleDict(
            {
                "sparse": Linear(d_model, 1),
                "dense": Linear(d_model, 1),
            }
        )

    def _prep_lang(self, lang_emb: mx.array, B: int, T: int, D: int) -> mx.array:
        """Prepare language embeddings for fusion."""
        if lang_emb.ndim == 3:
            return mx.expand_dims(self.lang_proj(lang_emb), axis=1)
        else:
            proj = self.lang_proj(lang_emb)
            proj = mx.expand_dims(mx.expand_dims(proj, axis=1), axis=1)
            return mx.broadcast_to(proj, (B, 1, T, D))

    def _stage_to_dmodel(self, stage_prior: mx.array) -> mx.array:
        """Deterministic projection of one-hot stage to d_model by pad/truncate.

        Args:
            stage_prior: One-hot stage embedding (B, 1, T, C)

        Returns:
            Projected stage embedding (B, 1, T, d_model)
        """
        B, one, T, C = stage_prior.shape
        D = self.d_model
        if D == C:
            return stage_prior
        elif D > C:
            pad_width = [(0, 0), (0, 0), (0, 0), (0, D - C)]
            return mx.pad(stage_prior, pad_width)
        else:
            return stage_prior[..., :D]

    def __call__(
        self,
        img_seq: mx.array,       # (B, N, T, vis_emb_dim)
        lang_emb: mx.array,      # (B, E) or (B, T, E)
        state: mx.array,         # (B, T, state_dim)
        lengths: mx.array,       # (B,) - valid sequence lengths
        stage_prior: mx.array,   # (B, 1, T, C) one-hot from gen_stage_emb
        scheme: str = "sparse",  # "sparse" or "dense"
    ) -> mx.array:
        """Forward pass for subtask progress regression.

        Returns:
            Tau predictions (B, T) in [0, 1] via sigmoid
        """
        assert scheme in self.heads, f"Unknown scheme '{scheme}'. Use one of {list(self.heads.keys())}."

        B, N, T, _ = img_seq.shape
        D = self.d_model

        # Project inputs
        vis_proj = self.visual_proj(img_seq)                          # (B, N, T, D)
        state_proj = mx.expand_dims(self.state_proj(state), axis=1)   # (B, 1, T, D)
        lang_proj = self._prep_lang(lang_emb, B, T, D)               # (B, 1, T, D)
        stage_emb = self._stage_to_dmodel(stage_prior)                # (B, 1, T, D)

        # Concatenate all streams: (B, N+3, T, D)
        x = mx.concatenate([vis_proj, lang_proj, state_proj, stage_emb], axis=1)

        # Add positional bias to first visual frame
        first_slice = x[:, :N, 0:1, :] + self.first_pos
        x = mx.concatenate([
            mx.concatenate([first_slice, x[:, :N, 1:, :]], axis=2),
            x[:, N:, :, :]
        ], axis=1)

        # Flatten to tokens
        x_tokens = x.reshape(B, (N + 3) * T, D)
        L = x_tokens.shape[1]

        # Create causal mask
        causal_mask = mx.triu(mx.ones((L, L)), k=1).astype(mx.bool_)

        # Encode
        h = self.transformer(x_tokens, mask=causal_mask)

        # Reshape and fuse
        h = h.reshape(B, N + 3, T, D)
        h_flat = mx.transpose(h, axes=(0, 2, 1, 3)).reshape(B, T, (N + 3) * D)
        fused = F.relu(self.fusion_linear(self.fusion_ln(h_flat)))  # (B, T, D)

        # Scheme-specific regression head -> sigmoid
        r = mx.sigmoid(self.heads[scheme](fused)).squeeze(axis=-1)  # (B, T)
        return r


# ---------------------------------------------------------------------------
# gen_stage_emb
# ---------------------------------------------------------------------------

def gen_stage_emb(num_classes: int, targets: mx.array) -> mx.array:
    """Generate one-hot stage embeddings from targets.

    Args:
        num_classes: Number of stage classes
        targets: Target values (B, T) where integer part is stage index

    Returns:
        One-hot stage embedding (B, 1, T, num_classes)
    """
    idx = mx.clip(mx.floor(targets).astype(mx.int32), 0, num_classes - 1)  # (B, T)
    stage_onehot = mx.eye(num_classes)[idx]  # (B, T, C)
    stage_onehot = mx.expand_dims(stage_onehot, axis=1)  # (B, 1, T, C)
    return stage_onehot


# ---------------------------------------------------------------------------
# SARMRewardModel
# ---------------------------------------------------------------------------

class SARMRewardModel(Module):
    """SARM Reward Model for stage-aware task completion rewards.

    Uses two separate transformer models:
    - StageTransformer: Classifies which stage/subtask
    - SubtaskTransformer: Predicts within-stage progress (tau)

    Training uses 75%/25% GT/predicted stage conditioning (teacher forcing).
    """

    name = "sarm"
    config_class = SARMConfig

    def __init__(self, config: SARMConfig, dataset_stats: dict | None = None):
        super().__init__()
        config.validate_features()
        self.config = config

        # Create two separate models
        self.stage_model = StageTransformer(
            d_model=config.hidden_dim,
            vis_emb_dim=config.image_dim,
            text_emb_dim=config.text_dim,
            state_dim=config.max_state_dim,
            n_layers=config.num_layers,
            n_heads=config.num_heads,
            dropout=config.dropout,
            num_cameras=1,
            num_classes_sparse=config.num_sparse_stages,
            num_classes_dense=config.num_dense_stages or config.num_sparse_stages,
        )

        self.subtask_model = SubtaskTransformer(
            d_model=config.hidden_dim,
            vis_emb_dim=config.image_dim,
            text_emb_dim=config.text_dim,
            state_dim=config.max_state_dim,
            n_layers=config.num_layers,
            n_heads=config.num_heads,
            dropout=config.dropout,
            num_cameras=1,
        )

        # GT/predicted stage ratio for teacher forcing
        self.gt_stage_ratio = 0.75
        self._training_mode = True

    def calculate_rewards(
        self,
        text_embeddings: mx.array | np.ndarray,
        video_embeddings: mx.array | np.ndarray,
        state_features: mx.array | np.ndarray | None = None,
        lengths: mx.array | np.ndarray | None = None,
        return_all_frames: bool = False,
        return_stages: bool = False,
        return_confidence: bool = False,
        head_mode: str | None = "sparse",
        frame_index: int | None = None,
    ) -> np.ndarray | tuple:
        """Calculate rewards for given text, video, and state representations."""
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = mx.array(text_embeddings)
        if isinstance(video_embeddings, np.ndarray):
            video_embeddings = mx.array(video_embeddings)
        if state_features is not None and isinstance(state_features, np.ndarray):
            state_features = mx.array(state_features)

        # Handle single sample case
        single_sample = False
        if text_embeddings.ndim == 1:
            text_embeddings = mx.expand_dims(text_embeddings, axis=0)
            video_embeddings = mx.expand_dims(video_embeddings, axis=0)
            if state_features is not None:
                state_features = mx.expand_dims(state_features, axis=0)
            single_sample = True

        batch_size = video_embeddings.shape[0]
        seq_len = video_embeddings.shape[1]
        scheme = head_mode

        # Default lengths if not provided
        if lengths is None:
            lengths = mx.full((batch_size,), seq_len, dtype=mx.int32)
        elif isinstance(lengths, np.ndarray):
            lengths = mx.array(lengths)

        # Reshape video to (B, 1, T, D)
        img_seq = mx.expand_dims(video_embeddings, axis=1)
        lang_emb = text_embeddings
        state = (
            state_features
            if state_features is not None
            else mx.zeros((batch_size, seq_len, self.config.max_state_dim))
        )

        # Pad state to max_state_dim
        state = pad_state_to_max_dim(state, self.config.max_state_dim)

        # Get num_classes for this scheme
        num_classes = self.config.num_sparse_stages if scheme == "sparse" else self.config.num_dense_stages

        # Run stage model
        stage_logits = self.stage_model(img_seq, lang_emb, state, lengths, scheme=scheme)
        stage_probs = mx.softmax(stage_logits, axis=-1)
        stage_idx = mx.argmax(stage_probs, axis=-1)  # (B, T)

        # Create one-hot stage prior
        stage_onehot = mx.eye(num_classes)[stage_idx]  # (B, T, C)
        stage_emb = mx.expand_dims(stage_onehot, axis=1)  # (B, 1, T, C)

        # Run subtask model
        tau_pred = self.subtask_model(img_seq, lang_emb, state, lengths, stage_emb, scheme=scheme)

        # Compute final reward: stage + tau
        raw_reward = stage_idx.astype(mx.float32) + tau_pred  # (B, T)

        # Normalize to [0, 1]
        if scheme == "sparse":
            normalized_reward = normalize_stage_tau(
                raw_reward,
                num_stages=num_classes,
                temporal_proportions=self.config.sparse_temporal_proportions,
                subtask_names=self.config.sparse_subtask_names,
            )
        else:
            normalized_reward = normalize_stage_tau(
                raw_reward,
                num_stages=num_classes,
                temporal_proportions=self.config.dense_temporal_proportions,
                subtask_names=self.config.dense_subtask_names,
            )

        # Default frame index
        if frame_index is None:
            frame_index = self.config.n_obs_steps

        # Prepare outputs
        if return_all_frames:
            rewards = np.array(normalized_reward)
        else:
            rewards = np.array(normalized_reward[:, frame_index])

        if single_sample:
            rewards = rewards[0]

        outputs = [rewards]
        if return_stages:
            probs = np.array(stage_probs)
            if single_sample:
                probs = probs[0]
            outputs.append(probs)
        if return_confidence:
            stage_conf_vals = mx.max(stage_probs, axis=-1)
            conf = np.array(stage_conf_vals)
            if single_sample:
                conf = conf[0]
            outputs.append(conf)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def reset(self):
        """Required by PreTrainedPolicy but not used for reward models."""
        pass

    def predict_action_chunk(self, batch: dict) -> mx.array:
        """Required by PreTrainedPolicy but not used for reward models."""
        raise NotImplementedError("SARM model does not predict action chunks")

    def select_action(self, batch: dict) -> mx.array:
        """Required by PreTrainedPolicy but not used for SARM."""
        raise NotImplementedError("SARM model does not select actions")

    def _train_step(
        self,
        img_emb: mx.array,     # (B, N, T, D)
        lang_emb: mx.array,    # (B, E) or (B, T, E)
        state: mx.array,       # (B, T, state_dim)
        lengths: mx.array,     # (B,)
        targets: mx.array,     # (B, T) - format: stage.tau
        scheme: str,
    ) -> dict[str, mx.array]:
        """Single training step for one annotation scheme.

        Implements 75%/25% GT/predicted stage conditioning.
        """
        num_classes = self.config.num_sparse_stages if scheme == "sparse" else self.config.num_dense_stages

        # Ground truth
        gt_stage = mx.clip(mx.floor(targets).astype(mx.int32), 0, num_classes - 1)
        gt_tau = targets - mx.floor(targets)

        # Run stage model
        stage_pred = self.stage_model(img_emb, lang_emb, state, lengths, scheme=scheme)

        # 75%/25% GT/predicted stage conditioning
        if random.random() < self.gt_stage_ratio:
            stage_emb = gen_stage_emb(num_classes, targets)
        else:
            stage_idx = mx.argmax(stage_pred, axis=-1)
            stage_onehot = mx.eye(num_classes)[stage_idx]
            stage_emb = mx.expand_dims(stage_onehot, axis=1)

        # Run subtask model with stage prior
        tau_pred = self.subtask_model(img_emb, lang_emb, state, lengths, stage_emb, scheme=scheme)

        # Compute losses
        stage_loss = F.cross_entropy(
            stage_pred.reshape(-1, num_classes),
            gt_stage.reshape(-1),
            reduction="mean",
        )
        subtask_loss = F.mse_loss(tau_pred, gt_tau, reduction="mean")

        return {
            "stage_loss": stage_loss,
            "subtask_loss": subtask_loss,
            "total_loss": stage_loss + subtask_loss,
        }

    def __call__(self, batch: dict):
        """Forward pass for SARM reward model training.

        Args:
            batch: Dictionary with observation data containing:
                - 'video_features': (B, T, 512)
                - 'text_features': (B, 512) or (B, T, 512)
                - 'state_features': (B, T, state_dim)
                - 'lengths': (B,)
                - 'sparse_targets': (B, T)
                - 'dense_targets': (B, T) optional

        Returns:
            Tuple of (total_loss, output_dict with loss components)
        """
        observation = batch.get("observation", batch)

        # Extract features
        video_features = observation["video_features"]
        text_features = observation["text_features"]
        state_features = observation.get("state_features")

        batch_size = video_features.shape[0]
        seq_len = video_features.shape[1]

        # Get lengths
        lengths = observation.get("lengths")
        if lengths is None:
            lengths = mx.full((batch_size,), seq_len, dtype=mx.int32)

        # Reshape video to (B, 1, T, D)
        img_emb = mx.expand_dims(video_features, axis=1)

        # Pad state
        if state_features is None:
            state_features = mx.zeros((batch_size, seq_len, self.config.max_state_dim))
        else:
            state_features = pad_state_to_max_dim(state_features, self.config.max_state_dim)

        output_dict = {}
        total_loss = mx.array(0.0)

        # Sparse training (always)
        sparse_targets = observation.get("sparse_targets")
        if sparse_targets is None:
            sparse_targets = observation.get("targets")
        if sparse_targets is None:
            raise ValueError("sparse_targets (or targets) is required for SARM training")

        sparse_result = self._train_step(
            img_emb, text_features, state_features, lengths, sparse_targets, scheme="sparse"
        )
        output_dict["sparse_stage_loss"] = sparse_result["stage_loss"].item()
        output_dict["sparse_subtask_loss"] = sparse_result["subtask_loss"].item()
        total_loss = total_loss + sparse_result["total_loss"]

        # Dense training (if dual mode)
        if self.config.uses_dual_heads:
            dense_targets = observation.get("dense_targets")
            if dense_targets is not None:
                dense_result = self._train_step(
                    img_emb, text_features, state_features, lengths, dense_targets, scheme="dense"
                )
                output_dict["dense_stage_loss"] = dense_result["stage_loss"].item()
                output_dict["dense_subtask_loss"] = dense_result["subtask_loss"].item()
                total_loss = total_loss + dense_result["total_loss"]

        output_dict["total_loss"] = total_loss.item()
        return total_loss, output_dict


# ---------------------------------------------------------------------------
# compute_stage_loss (standalone utility)
# ---------------------------------------------------------------------------

def compute_stage_loss(stage_logits: mx.array, target_stages: mx.array) -> mx.array:
    """Compute cross-entropy loss for stage classification."""
    _, _, num_stages = stage_logits.shape
    stage_logits_flat = stage_logits.reshape(-1, num_stages)
    target_stages_flat = mx.clip(target_stages.reshape(-1).astype(mx.int32), 0, num_stages - 1)
    return F.cross_entropy(stage_logits_flat, target_stages_flat)
