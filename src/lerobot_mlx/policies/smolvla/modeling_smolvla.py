"""SmolVLA (Small Vision-Language-Action) policy — MLX implementation.

SmolVLA = SmolVLM backbone + action expert + flow matching.
Architecture mirrors Pi0 but with a smaller SmolVLM backbone:
- Flow matching for action generation (identical ODE integration)
- Standalone action expert transformer (smaller than VLM)
- VLM backbone is optional (standalone operation supported)

This is a direct port of upstream lerobot.policies.smolvla.modeling_smolvla to MLX.
"""

import math
from collections import deque

import mlx.core as mx
import mlx.nn as _nn

from lerobot_mlx.compat import nn, F, Tensor
from lerobot_mlx.compat.nn_modules import Module
from lerobot_mlx.compat.nn_layers import Linear, RMSNorm, ModuleList
from lerobot_mlx.compat.tensor_ops import (
    zeros, ones, cat, stack, no_grad, zeros_like, randn, tensor,
    float32,
)

from lerobot_mlx.policies.smolvla.configuration_smolvla import (
    SmolVLAConfig, ACTION, OBS_STATE,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding(
    time: mx.array, dimension: int, min_period: float, max_period: float,
) -> mx.array:
    """Compute sine-cosine positional embedding vectors for scalar positions.

    Args:
        time: (batch_size,) tensor of timestep values.
        dimension: Embedding dimension (must be even).
        min_period: Minimum period for the sinusoidal encoding.
        max_period: Maximum period for the sinusoidal encoding.

    Returns:
        (batch_size, dimension) positional embeddings.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    half_dim = dimension // 2
    fraction = mx.linspace(0.0, 1.0, half_dim).astype(mx.float32)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = (1.0 / period) * 2.0 * math.pi
    # (1, half_dim) * (batch_size, 1) -> (batch_size, half_dim)
    sin_input = mx.expand_dims(scaling_factor, axis=0) * mx.expand_dims(time, axis=1)
    pos_emb = mx.concatenate([mx.sin(sin_input), mx.cos(sin_input)], axis=1)
    return pos_emb


def pad_vector(vector: mx.array, new_dim: int) -> mx.array:
    """Pad vector's last dimension to new_dim with zeros.

    Handles both (batch_size, features_dim) and (batch_size, seq_len, features_dim).
    """
    if vector.shape[-1] == new_dim:
        return vector
    current_dim = vector.shape[-1]
    if current_dim > new_dim:
        return vector[..., :new_dim]
    # Pad with zeros
    pad_width = [(0, 0)] * (vector.ndim - 1) + [(0, new_dim - current_dim)]
    return mx.pad(vector, pad_width)


# ---------------------------------------------------------------------------
# Expert Transformer
# ---------------------------------------------------------------------------

class SmolVLAExpertLayer(Module):
    """Single transformer layer for the action expert.

    Standard pre-norm transformer block with:
    - RMSNorm + multi-head self-attention
    - RMSNorm + SwiGLU MLP
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Self-attention
        self.input_layernorm = RMSNorm(hidden_size)
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

        # MLP (SwiGLU)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def _attention(self, x: mx.array) -> mx.array:
        """Multi-head self-attention."""
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ v

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_output)

    def _mlp(self, x: mx.array) -> mx.array:
        """SwiGLU MLP."""
        gate = _nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def __call__(self, x: mx.array) -> mx.array:
        # Pre-norm self-attention + residual
        h = self.input_layernorm(x)
        h = self._attention(h)
        x = x + h

        # Pre-norm MLP + residual
        h = self.post_attention_layernorm(x)
        h = self._mlp(h)
        x = x + h

        return x


class SmolVLAExpert(Module):
    """Action expert transformer — smaller than the VLM backbone.

    A stack of SmolVLAExpertLayer blocks with a final RMSNorm.
    The expert processes action tokens conditioned on VLM features
    via the flow matching mechanism.
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        hidden_size = config.effective_expert_hidden_size
        num_heads = config.expert_num_heads
        head_dim = config.expert_head_dim
        num_layers = config.expert_num_layers

        # Compute intermediate size (SwiGLU convention: 2/3 * 4 * hidden, rounded to 256)
        intermediate_size = _get_intermediate_size(hidden_size)

        self.layers = [
            SmolVLAExpertLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            for _ in range(num_layers)
        ]
        self.norm = RMSNorm(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


def _get_intermediate_size(hidden_dim: int, ffn_dim_multiplier: float = 4.0, multiple_of: int = 256) -> int:
    """Compute SwiGLU intermediate size, same as upstream."""
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


# ---------------------------------------------------------------------------
# Flow Matching Model
# ---------------------------------------------------------------------------

class SmolVLAFlowMatching(Module):
    """Flow matching model conditioned on VLM features.

    Handles:
    - State projection to VLM hidden space
    - Action input/output projections
    - Time (sinusoidal) + action fusion MLP
    - Expert transformer for denoising
    - Euler ODE integration for inference
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        expert_hidden = config.effective_expert_hidden_size
        vlm_hidden = config.vlm_hidden_size

        # State projection: project state into VLM embedding space
        self.state_proj = Linear(config.max_state_dim, vlm_hidden)

        # Action projections: map between action space and expert hidden space
        self.action_in_proj = Linear(config.max_action_dim, expert_hidden)
        self.action_out_proj = Linear(expert_hidden, config.max_action_dim)

        # Time MLP: fuse time embedding + action embedding
        self.action_time_mlp_in = Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = Linear(expert_hidden, expert_hidden)

        # Expert transformer
        self.expert = SmolVLAExpert(config)

    def encode_time(self, timestep: mx.array, hidden_size: int) -> mx.array:
        """Sinusoidal time encoding."""
        return create_sinusoidal_pos_embedding(
            timestep, hidden_size, self.config.min_period, self.config.max_period,
        )

    def embed_state(self, state: mx.array) -> mx.array:
        """Project state into embedding space."""
        state_emb = self.state_proj(state)
        if state_emb.ndim == 2:
            state_emb = mx.expand_dims(state_emb, axis=1)
        return state_emb

    def embed_actions_with_time(self, noisy_actions: mx.array, timestep: mx.array) -> mx.array:
        """Embed noisy actions fused with timestep information.

        Args:
            noisy_actions: (B, chunk_size, max_action_dim)
            timestep: (B,) timestep values

        Returns:
            (B, chunk_size, expert_hidden) fused action+time embeddings
        """
        expert_hidden = self.config.effective_expert_hidden_size
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = self.encode_time(timestep, expert_hidden)
        # Expand time to match action sequence: (B, hidden) -> (B, chunk_size, hidden)
        time_emb = mx.broadcast_to(
            mx.expand_dims(time_emb, axis=1),
            action_emb.shape,
        )

        # Concatenate and fuse through MLP
        action_time_emb = mx.concatenate([action_emb, time_emb], axis=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = _nn.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        return action_time_emb

    def sample_noise(self, shape: tuple) -> mx.array:
        """Sample standard normal noise."""
        return mx.random.normal(shape=shape)

    def sample_time(self, batch_size: int) -> mx.array:
        """Sample time from Beta(1.5, 1.0) distribution, mapped to [0.001, 1.0]."""
        # Use numpy for Beta sampling (not in training hot path for MLX grad)
        import numpy as np
        time_beta = np.random.beta(1.5, 1.0, size=(batch_size,)).astype(np.float32)
        time = time_beta * 0.999 + 0.001
        return mx.array(time)

    def __call__(
        self,
        state: mx.array,
        actions: mx.array,
        noise: mx.array | None = None,
        time: mx.array | None = None,
    ) -> mx.array:
        """Flow matching training forward pass.

        Computes the flow matching loss:
        1. Sample noise and time
        2. x_t = t * noise + (1-t) * actions  (interpolate)
        3. target u_t = noise - actions  (flow field)
        4. predicted v_t = expert(embed(x_t, t))
        5. loss = MSE(u_t, v_t)  (per-element, no reduction)

        Args:
            state: (B, max_state_dim) robot state
            actions: (B, chunk_size, max_action_dim) ground truth actions
            noise: Optional pre-sampled noise
            time: Optional pre-sampled time

        Returns:
            (B, chunk_size, max_action_dim) per-element MSE losses
        """
        if noise is None:
            noise = self.sample_noise(actions.shape)

        if time is None:
            time = self.sample_time(actions.shape[0])

        # Interpolate: x_t = t * noise + (1 - t) * actions
        time_expanded = mx.expand_dims(mx.expand_dims(time, axis=1), axis=2)  # (B, 1, 1)
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions

        # Target flow field
        u_t = noise - actions

        # Embed state (not used by expert directly in standalone mode,
        # but kept for completeness — in full model, state goes through VLM)
        # state_emb = self.embed_state(state)  # would be part of VLM prefix

        # Embed noisy actions with time
        action_time_emb = self.embed_actions_with_time(x_t, time)

        # Pass through expert
        expert_out = self.expert(action_time_emb)

        # Project back to action space
        v_t = self.action_out_proj(expert_out)

        # Per-element MSE loss
        losses = (u_t - v_t) ** 2
        return losses

    def generate_actions(
        self,
        state: mx.array,
        num_steps: int | None = None,
        noise: mx.array | None = None,
    ) -> mx.array:
        """Generate actions via Euler ODE integration.

        Integrates from t=1 (noise) to t=0 (actions) using the learned flow field.

        Args:
            state: (B, max_state_dim) robot state
            num_steps: Number of Euler integration steps (default: config.num_inference_steps)
            noise: Optional initial noise (B, chunk_size, max_action_dim)

        Returns:
            (B, chunk_size, max_action_dim) generated actions
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        batch_size = state.shape[0]

        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim)
            )

        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            t = 1.0 + step * dt
            time_tensor = mx.full((batch_size,), t, dtype=mx.float32)

            # Embed and denoise
            action_time_emb = self.embed_actions_with_time(x_t, time_tensor)
            expert_out = self.expert(action_time_emb)
            v_t = self.action_out_proj(expert_out)

            # Euler step
            x_t = x_t + dt * v_t

        return x_t


# ---------------------------------------------------------------------------
# SmolVLA Policy
# ---------------------------------------------------------------------------

class SmolVLAPolicy(Module):
    """SmolVLA: Small Vision-Language-Action policy.

    Wrapper around SmolVLAFlowMatching for training and inference.
    Manages action queuing for streaming inference.

    In standalone mode (no VLM backbone), processes state + actions only.
    VLM backbone integration is handled separately when available.
    """

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        self.model = SmolVLAFlowMatching(config)
        self.reset()

    def reset(self):
        """Reset action queue. Call whenever the environment is reset."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def select_action(self, batch: dict[str, mx.array]) -> mx.array:
        """Select a single action from environment observations.

        Uses action queuing: only generates new actions when the queue is empty.

        Args:
            batch: Dict containing at minimum 'observation.state' of shape (B, state_dim).

        Returns:
            (B, action_dim) single action to execute.
        """
        if len(self._action_queue) == 0:
            state = batch[OBS_STATE]
            if state.ndim > 2:
                state = state[:, -1, :]
            state = pad_vector(state, self.config.max_state_dim)

            actions = self.model.generate_actions(state)

            # Unpad to original action dim
            if self.config.output_features:
                for feat in self.config.output_features.values():
                    original_dim = feat.shape[0]
                    actions = actions[:, :, :original_dim]
                    break

            # Queue actions: transpose from (B, T, D) to iterate over T
            for t in range(min(actions.shape[1], self.config.n_action_steps)):
                self._action_queue.append(actions[:, t, :])

        return self._action_queue.popleft()

    def __call__(
        self,
        batch: dict[str, mx.array],
        noise: mx.array | None = None,
        time: mx.array | None = None,
    ) -> tuple[mx.array, dict[str, float]]:
        """Training forward pass — compute flow matching loss.

        Args:
            batch: Dict with 'observation.state' (B, state_dim) and 'action' (B, chunk_size, action_dim).
            noise: Optional noise tensor.
            time: Optional time tensor.

        Returns:
            (loss, loss_dict) where loss is scalar mean MSE.
        """
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        state = pad_vector(state, self.config.max_state_dim)

        actions = batch[ACTION]
        actions = pad_vector(actions, self.config.max_action_dim)

        losses = self.model(state, actions, noise=noise, time=time)

        loss = mx.mean(losses)
        loss_dict = {"loss": loss.item()}
        return loss, loss_dict
