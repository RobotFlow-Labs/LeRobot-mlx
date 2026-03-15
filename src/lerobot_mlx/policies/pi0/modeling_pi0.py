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
"""Pi0 VLA (Vision-Language-Action) policy -- MLX implementation.

Ports the flow matching action head, Gemma action expert, and policy wrapper
from PyTorch to Apple MLX. The VLM backbone (PaliGemma) is optional and can
be loaded separately from mlx-vlm.

Architecture:
    Pi0Policy -> Pi0FlowMatching -> GemmaExpert
                                 -> projection layers (state, action, time)

Flow matching:
    Training: x_t = t * noise + (1-t) * actions, target u_t = noise - actions
    Inference: Euler integration from noise (t=1) to clean actions (t=0)
"""

import math
from collections import deque

import mlx.core as mx
import mlx.nn as _nn

from lerobot_mlx.policies.pi0.configuration_pi0 import Pi0Config


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding (mirrors upstream create_sinusoidal_pos_embedding)
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding(
    time: mx.array,
    dimension: int,
    min_period: float,
    max_period: float,
) -> mx.array:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    Args:
        time: (batch_size,) array of timestep values.
        dimension: Embedding dimension (must be even).
        min_period: Minimum period for sinusoidal encoding.
        max_period: Maximum period for sinusoidal encoding.

    Returns:
        (batch_size, dimension) positional embeddings.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    half_dim = dimension // 2
    # fraction in [0, 1]
    fraction = mx.linspace(0.0, 1.0, half_dim)
    # period grows exponentially from min_period to max_period
    period = min_period * (max_period / min_period) ** fraction
    # scaling factor
    scaling_factor = (1.0 / period) * 2.0 * math.pi
    # outer product: (B, half_dim)
    sin_input = time[:, None] * scaling_factor[None, :]
    return mx.concatenate([mx.sin(sin_input), mx.cos(sin_input)], axis=1)


# ---------------------------------------------------------------------------
# Gemma Expert -- small transformer for action decoding
# ---------------------------------------------------------------------------

class _ExpertLayer(_nn.Module):
    """Single transformer layer for the Gemma action expert.

    Uses pre-norm, GQA attention, and GeGLU MLP -- matching the Gemma architecture.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Attention projections
        self.q_proj = _nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = _nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = _nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = _nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # GeGLU MLP
        self.gate_proj = _nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = _nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = _nn.Linear(intermediate_size, hidden_size, bias=False)

        # Norms
        self.input_layernorm = _nn.RMSNorm(hidden_size)
        self.post_attention_layernorm = _nn.RMSNorm(hidden_size)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape

        # Pre-norm self attention
        h = self.input_layernorm(x)

        # QKV projections
        q = self.q_proj(h).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # GQA: repeat k,v if num_kv_heads < num_heads
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        attn_out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        attn_out = self.o_proj(attn_out)

        x = x + attn_out

        # Pre-norm GeGLU MLP
        h = self.post_attention_layernorm(x)
        gate = _nn.silu(self.gate_proj(h))
        up = self.up_proj(h)
        h = self.down_proj(gate * up)

        return x + h


class GemmaExpert(_nn.Module):
    """Small Gemma-style transformer expert for action decoding.

    This is a standalone transformer that processes action tokens.
    Architecture mirrors the Gemma model used in Pi0's action expert.
    """

    def __init__(self, config: Pi0Config):
        super().__init__()
        hidden = config.expert_hidden_size
        n_heads = config.expert_num_heads
        n_kv_heads = config.expert_num_kv_heads
        head_dim = config.expert_head_dim
        n_layers = config.expert_num_layers
        intermediate = config.expert_intermediate_size

        self.layers = [
            _ExpertLayer(hidden, n_heads, n_kv_heads, head_dim, intermediate)
            for _ in range(n_layers)
        ]
        self.norm = _nn.RMSNorm(hidden)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Pi0 Flow Matching model
# ---------------------------------------------------------------------------

class Pi0FlowMatching(_nn.Module):
    """Flow matching model with VLM conditioning.

    Implements the core Pi0 training (flow matching loss) and inference
    (Euler integration denoising) loops. The VLM backbone is optional;
    without it, the model operates on state + action tokens only.

    Training flow:
        1. Sample noise and time t ~ Beta(alpha, beta)
        2. Create noisy actions: x_t = t * noise + (1-t) * actions
        3. Target velocity: u_t = noise - actions
        4. Predict velocity v_t from (x_t, t, state) via expert
        5. Loss = MSE(v_t, u_t)

    Inference flow:
        1. Start from pure noise at t=1
        2. Euler integrate: x_{t+dt} = x_t + dt * v_t  (dt < 0, going t=1->0)
        3. Final x_0 is the predicted action sequence
    """

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        hidden = config.expert_hidden_size

        # Action head projections (these are what we port from upstream)
        self.state_proj = _nn.Linear(config.max_state_dim, hidden)
        self.action_in_proj = _nn.Linear(config.max_action_dim, hidden)
        self.action_out_proj = _nn.Linear(hidden, config.max_action_dim)
        self.action_time_mlp_in = _nn.Linear(2 * hidden, hidden)
        self.action_time_mlp_out = _nn.Linear(hidden, hidden)

        # Gemma expert (action decoder transformer)
        self.expert = GemmaExpert(config)

        # VLM backbone -- optional, loaded separately
        self._vlm = None
        self._vlm_loaded = False

    def encode_time(self, timestep: mx.array) -> mx.array:
        """Sinusoidal time encoding for flow matching.

        Args:
            timestep: (batch_size,) array of timestep values in [0, 1].

        Returns:
            (batch_size, expert_hidden_size) time embeddings.
        """
        return create_sinusoidal_pos_embedding(
            timestep,
            self.config.expert_hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
        )

    def embed_suffix(
        self,
        state: mx.array,
        noisy_actions: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        """Embed state + noisy actions + timestep into expert input tokens.

        This mirrors upstream embed_suffix: state is projected to a single
        token, noisy actions are projected and fused with time via MLP,
        then all are concatenated.

        Args:
            state: (B, state_dim) or (B, 1, state_dim)
            noisy_actions: (B, chunk_size, action_dim)
            timestep: (B,) timestep values

        Returns:
            (B, 1 + chunk_size, hidden) suffix embeddings
        """
        # State projection
        if state.ndim == 3:
            state = state.squeeze(1)
        state_emb = self.state_proj(state)[:, None, :]  # (B, 1, H)

        # Time embedding
        time_emb = self.encode_time(timestep)  # (B, H)
        time_emb = time_emb.astype(noisy_actions.dtype)

        # Action projection
        action_emb = self.action_in_proj(noisy_actions)  # (B, T, H)

        # Fuse action + time via MLP
        time_expanded = mx.broadcast_to(
            time_emb[:, None, :], action_emb.shape
        )
        action_time = mx.concatenate([action_emb, time_expanded], axis=-1)  # (B, T, 2H)
        action_time_emb = self.action_time_mlp_out(
            _nn.silu(self.action_time_mlp_in(action_time))
        )  # (B, T, H)

        # Concatenate: [state_token, action_time_tokens]
        suffix = mx.concatenate([state_emb, action_time_emb], axis=1)  # (B, 1+T, H)
        return suffix

    def __call__(
        self,
        batch: dict,
        noise: mx.array | None = None,
        time: mx.array | None = None,
    ) -> dict:
        """Training forward: compute flow matching loss.

        Args:
            batch: Dictionary with keys:
                - 'observation.state': (B, state_dim) or (B, 1, state_dim)
                - 'action': (B, chunk_size, action_dim)
            noise: Optional pre-sampled noise, shape = actions.shape
            time: Optional pre-sampled time, shape = (B,)

        Returns:
            Dict with 'loss' and 'predicted_velocity'.
        """
        state = batch.get("observation.state")
        actions = batch.get("action")
        B = actions.shape[0]

        # Sample noise
        if noise is None:
            noise = mx.random.normal(actions.shape)

        # Sample time ~ uniform (Beta distribution approximation for simplicity)
        if time is None:
            time = mx.random.uniform(shape=(B,))
            time = time * self.config.time_sampling_scale + self.config.time_sampling_offset

        # Noisy actions: x_t = t * noise + (1-t) * actions (upstream convention)
        t = time.reshape(B, 1, 1)
        x_t = t * noise + (1 - t) * actions

        # Target velocity: u_t = noise - actions (upstream convention)
        target = noise - actions

        # Build suffix embeddings
        suffix = self.embed_suffix(state, x_t, time)  # (B, 1+T, H)

        # Create causal mask for suffix tokens
        # State + action tokens: state can attend to itself,
        # action tokens attend causally to state and previous action tokens
        seq_len = suffix.shape[1]
        causal_mask = _nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        # Allow all tokens to attend to the first token (state)
        causal_mask = causal_mask.at[:, 0].add(-causal_mask[:, 0])

        # Run through expert
        expert_out = self.expert(suffix, mask=causal_mask)  # (B, 1+T, H)

        # Take only the action tokens (skip state token)
        action_out = expert_out[:, 1:, :]  # (B, T, H)

        # Project to action space
        predicted_velocity = self.action_out_proj(action_out)  # (B, T, A)

        # Flow matching loss: MSE(v_t, u_t)
        loss = mx.mean((predicted_velocity - target) ** 2)

        return {"loss": loss, "predicted_velocity": predicted_velocity}

    def generate_actions(
        self,
        batch: dict,
        num_steps: int | None = None,
    ) -> mx.array:
        """Inference: generate actions via flow matching denoising.

        Euler integration from t=1 (noise) to t=0 (clean actions):
            x_1 = noise
            for step in range(num_steps):
                t = 1 - step/num_steps
                v_t = model(x_t, t, state)
                x_{t+dt} = x_t + dt * v_t   (dt = -1/num_steps)

        Args:
            batch: Dictionary with 'observation.state'.
            num_steps: Number of denoising steps (default: config.num_inference_steps).

        Returns:
            (B, chunk_size, action_dim) predicted actions.
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        state = batch.get("observation.state")
        if state is not None:
            B = state.shape[0]
        else:
            B = 1

        shape = (B, self.config.chunk_size, self.config.max_action_dim)

        # Start from pure noise at t=1
        x_t = mx.random.normal(shape)

        dt = -1.0 / num_steps

        for step in range(num_steps):
            t_val = 1.0 + step * dt  # goes from 1.0 towards 0.0
            t = mx.array([t_val] * B)

            # Build suffix and run expert
            suffix = self.embed_suffix(state, x_t, t)

            seq_len = suffix.shape[1]
            causal_mask = _nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            causal_mask = causal_mask.at[:, 0].add(-causal_mask[:, 0])

            expert_out = self.expert(suffix, mask=causal_mask)
            action_out = expert_out[:, 1:, :]
            velocity = self.action_out_proj(action_out)

            # Euler step (dt is negative, moving from t=1 to t=0)
            x_t = x_t + velocity * dt
            mx.eval(x_t)  # Prevent lazy graph buildup

        return x_t


# ---------------------------------------------------------------------------
# Pi0 Policy (top-level wrapper)
# ---------------------------------------------------------------------------

class Pi0Policy(_nn.Module):
    """Top-level Pi0 policy -- wraps VLM + action head.

    Provides select_action (inference with action queue) and forward (training).
    """

    config_class = Pi0Config
    name = "pi0"

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config
        self.model = Pi0FlowMatching(config)
        self._action_queue: deque = deque(maxlen=config.n_action_steps)

    def reset(self):
        """Reset the action queue (call between episodes)."""
        self._action_queue.clear()

    def select_action(self, batch: dict) -> mx.array:
        """Generate actions via flow matching denoising, with action queue.

        On first call (or after queue depletes), generates a full chunk
        of actions and caches them. Subsequent calls pop from the queue.

        Args:
            batch: Dictionary with observation data.

        Returns:
            Single action array of shape (action_dim,) or (B, action_dim).
        """
        if len(self._action_queue) == 0:
            actions = self.model.generate_actions(batch)
            # actions: (B, chunk_size, action_dim)
            # Queue individual timestep actions
            n_steps = min(self.config.n_action_steps, actions.shape[1])
            for i in range(n_steps):
                self._action_queue.append(actions[:, i, :])

        return self._action_queue.popleft()

    def __call__(self, batch: dict) -> dict:
        """Training forward: compute flow matching loss.

        Args:
            batch: Dictionary with 'observation.state' and 'action'.

        Returns:
            Dict with 'loss' key.
        """
        return self.model(batch)
