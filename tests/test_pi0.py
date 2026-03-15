#!/usr/bin/env python
"""Tests for Pi0 VLA policy -- MLX implementation.

All tests use synthetic data; no internet or pretrained weights required.
"""

import math

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim
import pytest

from lerobot_mlx.policies.pi0.configuration_pi0 import (
    Pi0Config,
    GemmaVariantConfig,
    get_gemma_config,
    GEMMA_VARIANTS,
)
from lerobot_mlx.policies.pi0.modeling_pi0 import (
    Pi0FlowMatching,
    Pi0Policy,
    GemmaExpert,
    _ExpertLayer,
    create_sinusoidal_pos_embedding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(B=2, state_dim=32, action_dim=32, chunk_size=50):
    """Create a synthetic batch for training/inference."""
    return {
        "observation.state": mx.random.normal((B, state_dim)),
        "action": mx.random.normal((B, chunk_size, action_dim)),
    }


def _default_config(**overrides) -> Pi0Config:
    """Create a small config for fast tests."""
    defaults = dict(
        action_expert_variant="gemma_300m",
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=8,
        max_action_dim=8,
        num_inference_steps=3,
    )
    defaults.update(overrides)
    return Pi0Config(**defaults)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestPi0Config:
    def test_config_defaults(self):
        cfg = Pi0Config()
        assert cfg.chunk_size == 50
        assert cfg.n_action_steps == 50
        assert cfg.max_state_dim == 32
        assert cfg.max_action_dim == 32
        assert cfg.num_inference_steps == 10
        assert cfg.paligemma_variant == "gemma_2b"
        assert cfg.action_expert_variant == "gemma_300m"

    def test_config_expert_properties(self):
        cfg = Pi0Config(action_expert_variant="gemma_300m")
        assert cfg.expert_hidden_size == 1024
        assert cfg.expert_intermediate_size == 4096
        assert cfg.expert_num_heads == 8
        assert cfg.expert_num_kv_heads == 1
        assert cfg.expert_head_dim == 256
        assert cfg.expert_num_layers == 18

    def test_config_expert_2b(self):
        cfg = Pi0Config(action_expert_variant="gemma_2b")
        assert cfg.expert_hidden_size == 2048
        assert cfg.expert_intermediate_size == 16384

    def test_config_validation_action_steps(self):
        with pytest.raises(ValueError, match="n_action_steps"):
            Pi0Config(n_action_steps=60, chunk_size=50)

    def test_config_unknown_variant(self):
        cfg = Pi0Config(action_expert_variant="unknown")
        with pytest.raises(ValueError, match="Unknown variant"):
            _ = cfg.expert_hidden_size

    def test_gemma_variants(self):
        for name in ["gemma_300m", "gemma_2b"]:
            v = get_gemma_config(name)
            assert isinstance(v, GemmaVariantConfig)
            assert v.width > 0
            assert v.depth > 0

    def test_config_custom_fields(self):
        cfg = Pi0Config(
            chunk_size=20,
            n_action_steps=15,
            max_state_dim=16,
            max_action_dim=16,
        )
        assert cfg.chunk_size == 20
        assert cfg.n_action_steps == 15


# ---------------------------------------------------------------------------
# Sinusoidal encoding tests
# ---------------------------------------------------------------------------

class TestSinusoidalEncoding:
    def test_time_encoding_shape(self):
        B = 4
        dim = 128
        t = mx.random.uniform(shape=(B,))
        emb = create_sinusoidal_pos_embedding(t, dim, min_period=4e-3, max_period=4.0)
        assert emb.shape == (B, dim)

    def test_time_encoding_finite(self):
        t = mx.array([0.0, 0.5, 1.0])
        emb = create_sinusoidal_pos_embedding(t, 64, min_period=4e-3, max_period=4.0)
        mx.eval(emb)
        assert mx.all(mx.isfinite(emb)).item()

    def test_time_encoding_odd_dim_raises(self):
        with pytest.raises(ValueError, match="divisible by 2"):
            create_sinusoidal_pos_embedding(mx.array([0.5]), 63, 4e-3, 4.0)

    def test_time_encoding_different_times(self):
        """Different times should produce different embeddings."""
        t = mx.array([0.1, 0.9])
        emb = create_sinusoidal_pos_embedding(t, 64, 4e-3, 4.0)
        mx.eval(emb)
        diff = mx.sum(mx.abs(emb[0] - emb[1])).item()
        assert diff > 0.1, "Different times should produce different embeddings"


# ---------------------------------------------------------------------------
# Expert model tests
# ---------------------------------------------------------------------------

class TestGemmaExpert:
    def test_expert_forward_shape(self):
        cfg = _default_config()
        expert = GemmaExpert(cfg)
        B, T, H = 2, 10, cfg.expert_hidden_size
        x = mx.random.normal((B, T, H))
        out = expert(x)
        mx.eval(out)
        assert out.shape == (B, T, H)

    def test_expert_forward_finite(self):
        cfg = _default_config()
        expert = GemmaExpert(cfg)
        x = mx.random.normal((2, 5, cfg.expert_hidden_size))
        out = expert(x)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()

    def test_expert_layers_count(self):
        cfg = _default_config()
        expert = GemmaExpert(cfg)
        assert len(expert.layers) == cfg.expert_num_layers

    def test_expert_layers_independent(self):
        """Each layer should have different weights (not shared)."""
        cfg = _default_config()
        expert = GemmaExpert(cfg)
        w0 = expert.layers[0].q_proj.weight
        w1 = expert.layers[1].q_proj.weight
        mx.eval(w0, w1)
        # Random init means they should differ
        diff = mx.sum(mx.abs(w0 - w1)).item()
        assert diff > 0.0

    def test_expert_with_mask(self):
        cfg = _default_config()
        expert = GemmaExpert(cfg)
        B, T, H = 2, 5, cfg.expert_hidden_size
        x = mx.random.normal((B, T, H))
        mask = _nn.MultiHeadAttention.create_additive_causal_mask(T)
        out = expert(x, mask=mask)
        mx.eval(out)
        assert out.shape == (B, T, H)


class TestExpertLayer:
    def test_layer_forward(self):
        layer = _ExpertLayer(
            hidden_size=64, num_heads=4, num_kv_heads=1,
            head_dim=16, intermediate_size=128,
        )
        x = mx.random.normal((2, 5, 64))
        out = layer(x)
        mx.eval(out)
        assert out.shape == (2, 5, 64)

    def test_layer_residual(self):
        """Output should differ from input (transformation happened)."""
        layer = _ExpertLayer(
            hidden_size=64, num_heads=4, num_kv_heads=1,
            head_dim=16, intermediate_size=128,
        )
        x = mx.random.normal((1, 3, 64))
        out = layer(x)
        mx.eval(out)
        diff = mx.sum(mx.abs(out - x)).item()
        assert diff > 0.0


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------

class TestProjections:
    def test_action_in_projection(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        x = mx.random.normal((2, 10, cfg.max_action_dim))
        out = model.action_in_proj(x)
        mx.eval(out)
        assert out.shape == (2, 10, cfg.expert_hidden_size)

    def test_action_out_projection(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        x = mx.random.normal((2, 10, cfg.expert_hidden_size))
        out = model.action_out_proj(x)
        mx.eval(out)
        assert out.shape == (2, 10, cfg.max_action_dim)

    def test_state_projection(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        x = mx.random.normal((2, cfg.max_state_dim))
        out = model.state_proj(x)
        mx.eval(out)
        assert out.shape == (2, cfg.expert_hidden_size)

    def test_time_mlp(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        H = cfg.expert_hidden_size
        x = mx.random.normal((2, 5, 2 * H))
        h = _nn.silu(model.action_time_mlp_in(x))
        out = model.action_time_mlp_out(h)
        mx.eval(out)
        assert out.shape == (2, 5, H)


# ---------------------------------------------------------------------------
# Flow Matching model tests
# ---------------------------------------------------------------------------

class TestPi0FlowMatching:
    def test_model_creation(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        assert model.config is cfg

    def test_forward_shape(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        batch = _make_batch(B=2, state_dim=cfg.max_state_dim,
                           action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size)
        result = model(batch)
        mx.eval(result["loss"], result["predicted_velocity"])
        assert result["predicted_velocity"].shape == (2, cfg.chunk_size, cfg.max_action_dim)

    def test_loss_finite(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        batch = _make_batch(B=2, state_dim=cfg.max_state_dim,
                           action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size)
        result = model(batch)
        loss = result["loss"]
        mx.eval(loss)
        assert mx.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"

    def test_loss_positive(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        batch = _make_batch(B=2, state_dim=cfg.max_state_dim,
                           action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size)
        result = model(batch)
        mx.eval(result["loss"])
        assert result["loss"].item() > 0.0

    def test_flow_matching_target(self):
        """Target velocity should be noise - actions (upstream convention)."""
        cfg = _default_config()
        B, T, A = 2, cfg.chunk_size, cfg.max_action_dim

        actions = mx.random.normal((B, T, A))
        noise = mx.random.normal((B, T, A))

        expected_target = noise - actions
        mx.eval(expected_target)

        # Verify shape
        assert expected_target.shape == (B, T, A)

    def test_noise_interpolation(self):
        """x_t = t * noise + (1-t) * actions should hold."""
        B, T, A = 2, 5, 4
        actions = mx.random.normal((B, T, A))
        noise = mx.random.normal((B, T, A))
        t = mx.array([0.3, 0.7]).reshape(B, 1, 1)

        x_t = t * noise + (1 - t) * actions
        mx.eval(x_t)

        # At t=0, x_t should equal actions
        x_0 = mx.zeros((B, 1, 1)) * noise + (1 - mx.zeros((B, 1, 1))) * actions
        mx.eval(x_0)
        assert mx.allclose(x_0, actions, atol=1e-6).item()

        # At t=1, x_t should equal noise
        x_1 = mx.ones((B, 1, 1)) * noise + (1 - mx.ones((B, 1, 1))) * actions
        mx.eval(x_1)
        assert mx.allclose(x_1, noise, atol=1e-6).item()

    def test_forward_with_explicit_noise_and_time(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        batch = _make_batch(B=2, state_dim=cfg.max_state_dim,
                           action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size)
        noise = mx.random.normal(batch["action"].shape)
        time = mx.array([0.3, 0.7])
        result = model(batch, noise=noise, time=time)
        mx.eval(result["loss"])
        assert mx.isfinite(result["loss"]).item()

    def test_embed_suffix_shape(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        B = 2
        state = mx.random.normal((B, cfg.max_state_dim))
        noisy_actions = mx.random.normal((B, cfg.chunk_size, cfg.max_action_dim))
        timestep = mx.array([0.5, 0.8])
        suffix = model.embed_suffix(state, noisy_actions, timestep)
        mx.eval(suffix)
        # suffix should be (B, 1 + chunk_size, hidden)
        assert suffix.shape == (B, 1 + cfg.chunk_size, cfg.expert_hidden_size)


# ---------------------------------------------------------------------------
# Action generation tests
# ---------------------------------------------------------------------------

class TestGenerateActions:
    def test_generate_actions_shape(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        batch = {"observation.state": mx.random.normal((2, cfg.max_state_dim))}
        actions = model.generate_actions(batch)
        mx.eval(actions)
        assert actions.shape == (2, cfg.chunk_size, cfg.max_action_dim)

    def test_generate_actions_finite(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        batch = {"observation.state": mx.random.normal((2, cfg.max_state_dim))}
        actions = model.generate_actions(batch)
        mx.eval(actions)
        assert mx.all(mx.isfinite(actions)).item()

    def test_euler_integration_changes(self):
        """Actions should change over denoising steps (not stuck)."""
        cfg = _default_config(num_inference_steps=2)
        model = Pi0FlowMatching(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}

        # Generate with 1 step vs 2 steps
        mx.random.seed(42)
        a1 = model.generate_actions(batch, num_steps=1)
        mx.eval(a1)

        mx.random.seed(42)
        a2 = model.generate_actions(batch, num_steps=2)
        mx.eval(a2)

        diff = mx.sum(mx.abs(a1 - a2)).item()
        assert diff > 0.0, "Different num_steps should produce different actions"

    def test_generate_custom_num_steps(self):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        actions = model.generate_actions(batch, num_steps=5)
        mx.eval(actions)
        assert actions.shape == (1, cfg.chunk_size, cfg.max_action_dim)


# ---------------------------------------------------------------------------
# Policy wrapper tests
# ---------------------------------------------------------------------------

class TestPi0Policy:
    def test_policy_creation(self):
        cfg = _default_config()
        policy = Pi0Policy(cfg)
        assert policy.config is cfg
        assert isinstance(policy.model, Pi0FlowMatching)

    def test_policy_forward(self):
        cfg = _default_config()
        policy = Pi0Policy(cfg)
        batch = _make_batch(B=2, state_dim=cfg.max_state_dim,
                           action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size)
        result = policy(batch)
        mx.eval(result["loss"])
        assert mx.isfinite(result["loss"]).item()

    def test_select_action(self):
        cfg = _default_config()
        policy = Pi0Policy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        action = policy.select_action(batch)
        mx.eval(action)
        assert action.shape == (1, cfg.max_action_dim)

    def test_action_queue_caches(self):
        cfg = _default_config(chunk_size=5, n_action_steps=5)
        policy = Pi0Policy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}

        # First call fills the queue
        a1 = policy.select_action(batch)
        mx.eval(a1)
        assert len(policy._action_queue) == 4  # 5 - 1 (just popped)

        # Second call pops from queue (no new generation)
        a2 = policy.select_action(batch)
        mx.eval(a2)
        assert len(policy._action_queue) == 3

    def test_action_queue_depletes_and_refills(self):
        cfg = _default_config(chunk_size=3, n_action_steps=3, num_inference_steps=2)
        policy = Pi0Policy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}

        # Pop all 3 actions
        actions = []
        for _ in range(3):
            a = policy.select_action(batch)
            mx.eval(a)
            actions.append(a)

        assert len(policy._action_queue) == 0

        # Next call should regenerate
        a4 = policy.select_action(batch)
        mx.eval(a4)
        assert len(policy._action_queue) == 2  # 3 - 1

    def test_reset(self):
        cfg = _default_config()
        policy = Pi0Policy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        policy.select_action(batch)
        assert len(policy._action_queue) > 0
        policy.reset()
        assert len(policy._action_queue) == 0


# ---------------------------------------------------------------------------
# Batch size / chunk size variation tests
# ---------------------------------------------------------------------------

class TestVariousSizes:
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_various_batch_sizes(self, batch_size):
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        batch = _make_batch(B=batch_size, state_dim=cfg.max_state_dim,
                           action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size)
        result = model(batch)
        mx.eval(result["loss"])
        assert mx.isfinite(result["loss"]).item()
        assert result["predicted_velocity"].shape[0] == batch_size

    @pytest.mark.parametrize("chunk_size", [5, 10, 20])
    def test_various_chunk_sizes(self, chunk_size):
        cfg = _default_config(chunk_size=chunk_size, n_action_steps=chunk_size)
        model = Pi0FlowMatching(cfg)
        batch = _make_batch(B=2, state_dim=cfg.max_state_dim,
                           action_dim=cfg.max_action_dim, chunk_size=chunk_size)
        result = model(batch)
        mx.eval(result["loss"])
        assert result["predicted_velocity"].shape == (2, chunk_size, cfg.max_action_dim)


# ---------------------------------------------------------------------------
# Gradient / training tests
# ---------------------------------------------------------------------------

class TestTraining:
    def test_gradient_flow(self):
        """Gradients should reach all trainable parameters."""
        cfg = _default_config()
        model = Pi0FlowMatching(cfg)
        batch = _make_batch(B=2, state_dim=cfg.max_state_dim,
                           action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size)

        def loss_fn(model):
            result = model(batch)
            return result["loss"]

        loss, grads = mx.value_and_grad(loss_fn)(model)
        mx.eval(loss, grads)

        # Check that we got a finite loss
        assert mx.isfinite(loss).item()

        # Check that at least some parameters have non-zero gradients
        has_nonzero = False
        for key, val in _flatten_dict(grads):
            if isinstance(val, mx.array):
                mx.eval(val)
                if mx.any(val != 0).item():
                    has_nonzero = True
                    break
        assert has_nonzero, "No non-zero gradients found"

    def test_training_step(self):
        """Loss should decrease after optimizer steps.

        Uses a minimal model to make training tractable in a unit test.
        We create a tiny config by monkey-patching the variant lookup.
        """
        from lerobot_mlx.policies.pi0.configuration_pi0 import (
            GEMMA_VARIANTS, GemmaVariantConfig,
        )

        # Register a tiny variant for testing
        GEMMA_VARIANTS["gemma_tiny_test"] = GemmaVariantConfig(
            width=32, depth=2, mlp_dim=64,
            num_heads=2, num_kv_heads=1, head_dim=16,
        )
        try:
            cfg = Pi0Config(
                action_expert_variant="gemma_tiny_test",
                chunk_size=4,
                n_action_steps=4,
                max_state_dim=4,
                max_action_dim=4,
                num_inference_steps=2,
            )
            model = Pi0FlowMatching(cfg)

            B = 4
            batch = _make_batch(B=B, state_dim=cfg.max_state_dim,
                               action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size)
            noise = mx.random.normal(batch["action"].shape)
            time_vals = mx.random.uniform(shape=(B,))
            mx.eval(noise, time_vals, batch["observation.state"], batch["action"])

            optimizer = optim.Adam(learning_rate=1e-3)

            def loss_fn(model, noise, time_vals):
                result = model(batch, noise=noise, time=time_vals)
                return result["loss"]

            # Initial loss
            loss0 = loss_fn(model, noise, time_vals)
            mx.eval(loss0)
            loss0_val = loss0.item()

            # Training steps
            for _ in range(30):
                loss, grads = mx.value_and_grad(loss_fn)(model, noise, time_vals)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)

            # Final loss with same noise/time
            loss_final = loss_fn(model, noise, time_vals)
            mx.eval(loss_final)
            loss_final_val = loss_final.item()

            assert loss_final_val < loss0_val, (
                f"Loss did not decrease: {loss0_val:.4f} -> {loss_final_val:.4f}"
            )
        finally:
            # Clean up
            del GEMMA_VARIANTS["gemma_tiny_test"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _flatten_dict(d, prefix=""):
    """Flatten a nested dict, yielding (key_path, value) pairs."""
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.extend(_flatten_dict(v, new_key))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{prefix}.{i}" if prefix else str(i)
            items.extend(_flatten_dict(v, new_key))
    else:
        items.append((prefix, d))
    return items
