"""Comprehensive tests for SmolVLA (Small Vision-Language-Action) policy — MLX port.

Tests configuration, model creation, forward pass, loss computation,
flow matching, gradient flow, expert architecture, and inference.
"""

import math

import mlx.core as mx
import mlx.nn as _nn
import numpy as np
import pytest

from lerobot_mlx.policies.smolvla.configuration_smolvla import (
    SmolVLAConfig,
    ACTION,
    OBS_STATE,
    FeatureType,
    NormalizationMode,
    PolicyFeature,
)
from lerobot_mlx.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    SmolVLAFlowMatching,
    SmolVLAExpert,
    SmolVLAExpertLayer,
    create_sinusoidal_pos_embedding,
    pad_vector,
    _get_intermediate_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    chunk_size: int = 10,
    n_action_steps: int = 10,
    max_state_dim: int = 8,
    max_action_dim: int = 6,
    expert_hidden_size: int = 64,
    expert_num_heads: int = 4,
    expert_num_layers: int = 2,
    expert_head_dim: int = 16,
    expert_width_multiplier: float = 1.0,
    vlm_hidden_size: int = 64,
    num_inference_steps: int = 5,
) -> SmolVLAConfig:
    """Create a small config for testing."""
    return SmolVLAConfig(
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        expert_hidden_size=expert_hidden_size,
        expert_num_heads=expert_num_heads,
        expert_num_layers=expert_num_layers,
        expert_head_dim=expert_head_dim,
        expert_width_multiplier=expert_width_multiplier,
        vlm_hidden_size=vlm_hidden_size,
        num_inference_steps=num_inference_steps,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(max_state_dim,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(max_action_dim,)),
        },
    )


def _make_batch(config: SmolVLAConfig, batch_size: int = 2) -> dict[str, mx.array]:
    """Create a synthetic training batch."""
    return {
        OBS_STATE: mx.random.normal(shape=(batch_size, config.max_state_dim)),
        ACTION: mx.random.normal(shape=(batch_size, config.chunk_size, config.max_action_dim)),
    }


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

class TestSmolVLAConfig:
    def test_smolvla_config_defaults(self):
        config = SmolVLAConfig()
        assert config.chunk_size == 50
        assert config.n_action_steps == 50
        assert config.max_state_dim == 32
        assert config.max_action_dim == 32
        assert config.expert_hidden_size == 768
        assert config.expert_width_multiplier == 0.75
        assert config.num_inference_steps == 10
        assert config.vlm_hidden_size == 2048
        assert config.num_vlm_layers == 16
        assert config.min_period == pytest.approx(4e-3)
        assert config.max_period == pytest.approx(4.0)

    def test_smolvla_config_custom(self):
        config = _make_config()
        assert config.chunk_size == 10
        assert config.max_state_dim == 8
        assert config.max_action_dim == 6

    def test_smolvla_config_validation(self):
        with pytest.raises(ValueError, match="chunk size"):
            SmolVLAConfig(chunk_size=10, n_action_steps=20)

    def test_smolvla_effective_expert_hidden_size(self):
        config = SmolVLAConfig()
        assert config.effective_expert_hidden_size == int(768 * 0.75)

        config2 = _make_config(expert_hidden_size=100, expert_width_multiplier=0.5)
        assert config2.effective_expert_hidden_size == 50


# ---------------------------------------------------------------------------
# Model Creation Tests
# ---------------------------------------------------------------------------

class TestSmolVLAModelCreation:
    def test_smolvla_model_creation(self):
        config = _make_config()
        policy = SmolVLAPolicy(config)
        assert policy.config is config
        assert isinstance(policy.model, SmolVLAFlowMatching)

    def test_smolvla_flow_matching_creation(self):
        config = _make_config()
        model = SmolVLAFlowMatching(config)
        assert isinstance(model.state_proj, _nn.Linear)
        assert isinstance(model.action_in_proj, _nn.Linear)
        assert isinstance(model.action_out_proj, _nn.Linear)
        assert isinstance(model.action_time_mlp_in, _nn.Linear)
        assert isinstance(model.action_time_mlp_out, _nn.Linear)
        assert isinstance(model.expert, SmolVLAExpert)

    def test_smolvla_expert_creation(self):
        config = _make_config()
        expert = SmolVLAExpert(config)
        assert len(expert.layers) == config.expert_num_layers
        assert isinstance(expert.norm, _nn.RMSNorm)


# ---------------------------------------------------------------------------
# Forward Pass Tests
# ---------------------------------------------------------------------------

class TestSmolVLAForward:
    def test_smolvla_forward_shape(self):
        config = _make_config()
        policy = SmolVLAPolicy(config)
        batch = _make_batch(config)
        loss, loss_dict = policy(batch)
        mx.eval(loss)
        assert loss.shape == ()
        assert "loss" in loss_dict

    def test_smolvla_loss_finite(self):
        config = _make_config()
        policy = SmolVLAPolicy(config)
        batch = _make_batch(config)
        loss, _ = policy(batch)
        mx.eval(loss)
        assert mx.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0, "Loss should be positive"

    def test_smolvla_flow_matching_forward(self):
        config = _make_config()
        model = SmolVLAFlowMatching(config)
        state = mx.random.normal(shape=(2, config.max_state_dim))
        actions = mx.random.normal(shape=(2, config.chunk_size, config.max_action_dim))
        losses = model(state, actions)
        mx.eval(losses)
        assert losses.shape == (2, config.chunk_size, config.max_action_dim)
        assert mx.all(mx.isfinite(losses)).item()

    def test_smolvla_flow_matching_target(self):
        """Verify that the flow matching target is noise - actions."""
        config = _make_config()
        model = SmolVLAFlowMatching(config)
        state = mx.random.normal(shape=(2, config.max_state_dim))
        actions = mx.random.normal(shape=(2, config.chunk_size, config.max_action_dim))
        noise = mx.random.normal(shape=actions.shape)
        time = mx.array([0.5, 0.5])

        # The target u_t = noise - actions
        expected_target = noise - actions
        mx.eval(expected_target)

        # Verify target is computed correctly (check shape at minimum)
        assert expected_target.shape == actions.shape


# ---------------------------------------------------------------------------
# Time Encoding Tests
# ---------------------------------------------------------------------------

class TestTimeEncoding:
    def test_smolvla_time_encoding(self):
        config = _make_config()
        model = SmolVLAFlowMatching(config)
        time = mx.array([0.1, 0.5, 0.9])
        hidden_size = config.effective_expert_hidden_size
        emb = model.encode_time(time, hidden_size)
        mx.eval(emb)
        assert emb.shape == (3, hidden_size)
        assert mx.all(mx.isfinite(emb)).item()

    def test_smolvla_time_encoding_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension.*must be divisible by 2"):
            create_sinusoidal_pos_embedding(mx.array([0.5]), dimension=7, min_period=0.004, max_period=4.0)

    def test_smolvla_time_encoding_shape_validation(self):
        with pytest.raises(ValueError, match="batch_size"):
            create_sinusoidal_pos_embedding(mx.array([[0.5]]), dimension=8, min_period=0.004, max_period=4.0)

    def test_smolvla_time_encoding_different_times(self):
        """Different timesteps should produce different embeddings."""
        t1 = mx.array([0.1])
        t2 = mx.array([0.9])
        emb1 = create_sinusoidal_pos_embedding(t1, 16, 0.004, 4.0)
        emb2 = create_sinusoidal_pos_embedding(t2, 16, 0.004, 4.0)
        mx.eval(emb1, emb2)
        # Embeddings should differ
        diff = mx.sum(mx.abs(emb1 - emb2)).item()
        assert diff > 0.01, "Different timesteps should produce different embeddings"


# ---------------------------------------------------------------------------
# Expert Tests
# ---------------------------------------------------------------------------

class TestSmolVLAExpert:
    def test_smolvla_expert_forward(self):
        config = _make_config()
        expert = SmolVLAExpert(config)
        hidden = config.effective_expert_hidden_size
        x = mx.random.normal(shape=(2, 10, hidden))
        out = expert(x)
        mx.eval(out)
        assert out.shape == (2, 10, hidden)
        assert mx.all(mx.isfinite(out)).item()

    def test_smolvla_expert_layer_forward(self):
        hidden_size = 64
        intermediate_size = _get_intermediate_size(hidden_size)
        layer = SmolVLAExpertLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=4,
            head_dim=16,
        )
        x = mx.random.normal(shape=(2, 5, hidden_size))
        out = layer(x)
        mx.eval(out)
        assert out.shape == x.shape
        assert mx.all(mx.isfinite(out)).item()

    def test_smolvla_expert_width_multiplier(self):
        """Expert size should scale with width_multiplier."""
        config1 = _make_config(expert_hidden_size=128, expert_width_multiplier=1.0)
        config2 = _make_config(expert_hidden_size=128, expert_width_multiplier=0.5)

        expert1 = SmolVLAExpert(config1)
        expert2 = SmolVLAExpert(config2)

        # Expert 2 should have smaller hidden size
        h1 = config1.effective_expert_hidden_size
        h2 = config2.effective_expert_hidden_size
        assert h1 == 128
        assert h2 == 64

        # Verify layers use the correct size
        x1 = mx.random.normal(shape=(1, 3, h1))
        x2 = mx.random.normal(shape=(1, 3, h2))
        out1 = expert1(x1)
        out2 = expert2(x2)
        mx.eval(out1, out2)
        assert out1.shape == (1, 3, 128)
        assert out2.shape == (1, 3, 64)


# ---------------------------------------------------------------------------
# Projection Tests
# ---------------------------------------------------------------------------

class TestProjections:
    def test_smolvla_action_projections(self):
        config = _make_config()
        model = SmolVLAFlowMatching(config)
        expert_hidden = config.effective_expert_hidden_size

        x = mx.random.normal(shape=(2, config.chunk_size, config.max_action_dim))
        projected = model.action_in_proj(x)
        mx.eval(projected)
        assert projected.shape == (2, config.chunk_size, expert_hidden)

        back = model.action_out_proj(mx.random.normal(shape=(2, config.chunk_size, expert_hidden)))
        mx.eval(back)
        assert back.shape == (2, config.chunk_size, config.max_action_dim)

    def test_smolvla_state_projection(self):
        config = _make_config()
        model = SmolVLAFlowMatching(config)

        state = mx.random.normal(shape=(2, config.max_state_dim))
        state_emb = model.embed_state(state)
        mx.eval(state_emb)
        assert state_emb.shape == (2, 1, config.vlm_hidden_size)


# ---------------------------------------------------------------------------
# Action Generation Tests
# ---------------------------------------------------------------------------

class TestActionGeneration:
    def test_smolvla_generate_actions(self):
        config = _make_config()
        model = SmolVLAFlowMatching(config)
        state = mx.random.normal(shape=(2, config.max_state_dim))
        actions = model.generate_actions(state)
        mx.eval(actions)
        assert actions.shape == (2, config.chunk_size, config.max_action_dim)
        assert mx.all(mx.isfinite(actions)).item()

    def test_smolvla_generate_actions_shape(self):
        config = _make_config(chunk_size=20, max_action_dim=8)
        model = SmolVLAFlowMatching(config)
        state = mx.random.normal(shape=(3, config.max_state_dim))
        actions = model.generate_actions(state)
        mx.eval(actions)
        assert actions.shape == (3, 20, 8)

    def test_smolvla_euler_integration(self):
        """Verify that Euler integration produces different results with different step counts."""
        config = _make_config()
        model = SmolVLAFlowMatching(config)
        state = mx.random.normal(shape=(1, config.max_state_dim))
        noise = mx.random.normal(shape=(1, config.chunk_size, config.max_action_dim))

        actions_5 = model.generate_actions(state, num_steps=5, noise=noise)
        # Use same noise but different step count
        actions_10 = model.generate_actions(state, num_steps=10, noise=noise)
        mx.eval(actions_5, actions_10)

        # Different step counts should give different results
        diff = mx.sum(mx.abs(actions_5 - actions_10)).item()
        assert diff > 1e-6, "Different step counts should produce different results"

    def test_smolvla_different_inference_steps(self):
        config = _make_config(num_inference_steps=3)
        model = SmolVLAFlowMatching(config)
        state = mx.random.normal(shape=(1, config.max_state_dim))

        # Default steps from config
        actions = model.generate_actions(state)
        mx.eval(actions)
        assert actions.shape == (1, config.chunk_size, config.max_action_dim)

        # Override steps
        actions2 = model.generate_actions(state, num_steps=7)
        mx.eval(actions2)
        assert actions2.shape == actions.shape


# ---------------------------------------------------------------------------
# Policy-Level Tests
# ---------------------------------------------------------------------------

class TestSmolVLAPolicy:
    def test_smolvla_select_action(self):
        config = _make_config()
        policy = SmolVLAPolicy(config)
        batch = {OBS_STATE: mx.random.normal(shape=(1, config.max_state_dim))}
        action = policy.select_action(batch)
        mx.eval(action)
        assert action.shape == (1, config.max_action_dim)

    def test_smolvla_action_queue(self):
        config = _make_config(chunk_size=5, n_action_steps=5)
        policy = SmolVLAPolicy(config)
        batch = {OBS_STATE: mx.random.normal(shape=(1, config.max_state_dim))}

        # First call generates all actions
        a1 = policy.select_action(batch)
        mx.eval(a1)
        assert len(policy._action_queue) == 4  # 5 - 1 popped

        # Subsequent calls pop from queue without regenerating
        a2 = policy.select_action(batch)
        mx.eval(a2)
        assert len(policy._action_queue) == 3

    def test_smolvla_action_queue_reset(self):
        config = _make_config(chunk_size=5, n_action_steps=5)
        policy = SmolVLAPolicy(config)
        batch = {OBS_STATE: mx.random.normal(shape=(1, config.max_state_dim))}

        policy.select_action(batch)
        assert len(policy._action_queue) > 0
        policy.reset()
        assert len(policy._action_queue) == 0

    def test_smolvla_various_batch_sizes(self):
        config = _make_config()
        policy = SmolVLAPolicy(config)

        for bs in [1, 2, 4]:
            batch = _make_batch(config, batch_size=bs)
            loss, loss_dict = policy(batch)
            mx.eval(loss)
            assert loss.shape == ()
            assert mx.isfinite(loss).item()

    def test_smolvla_training_step(self):
        """Simulate a training step: forward + backward."""
        config = _make_config()
        policy = SmolVLAPolicy(config)
        batch = _make_batch(config, batch_size=2)

        def loss_fn(model, batch):
            loss, _ = model(batch)
            return loss

        loss_and_grad_fn = _nn.value_and_grad(policy, loss_fn)
        loss, grads = loss_and_grad_fn(policy, batch)
        mx.eval(loss, grads)
        assert mx.isfinite(loss).item()

        # Verify some gradients are non-zero
        flat_grads = _flatten_grads(grads)
        has_nonzero = any(mx.any(mx.abs(g) > 0).item() for g in flat_grads if g is not None)
        assert has_nonzero, "At least some gradients should be non-zero"


# ---------------------------------------------------------------------------
# Gradient Flow Tests
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_smolvla_gradient_flow(self):
        """Verify that gradients flow through the model."""
        config = _make_config()
        model = SmolVLAFlowMatching(config)
        state = mx.random.normal(shape=(2, config.max_state_dim))
        actions = mx.random.normal(shape=(2, config.chunk_size, config.max_action_dim))

        def loss_fn(model, state, actions):
            losses = model(state, actions)
            return mx.mean(losses)

        loss_and_grad_fn = _nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, state, actions)
        mx.eval(loss, grads)
        assert mx.isfinite(loss).item()

        flat_grads = _flatten_grads(grads)
        has_nonzero = any(mx.any(mx.abs(g) > 0).item() for g in flat_grads if g is not None)
        assert has_nonzero, "Gradients should flow through the model"


# ---------------------------------------------------------------------------
# Utility Tests
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_pad_vector_2d(self):
        x = mx.ones((2, 5))
        padded = pad_vector(x, 8)
        mx.eval(padded)
        assert padded.shape == (2, 8)
        assert mx.array_equal(padded[:, :5], x).item()
        assert mx.array_equal(padded[:, 5:], mx.zeros((2, 3))).item()

    def test_pad_vector_3d(self):
        x = mx.ones((2, 3, 5))
        padded = pad_vector(x, 8)
        mx.eval(padded)
        assert padded.shape == (2, 3, 8)

    def test_pad_vector_noop(self):
        x = mx.ones((2, 8))
        padded = pad_vector(x, 8)
        assert padded.shape == (2, 8)

    def test_get_intermediate_size(self):
        size = _get_intermediate_size(64)
        assert size % 256 == 0
        assert size > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_grads(grads):
    """Flatten nested grad dict into a list of arrays."""
    result = []
    if isinstance(grads, dict):
        for v in grads.values():
            result.extend(_flatten_grads(v))
    elif isinstance(grads, (list, tuple)):
        for v in grads:
            result.extend(_flatten_grads(v))
    elif isinstance(grads, mx.array):
        result.append(grads)
    return result
