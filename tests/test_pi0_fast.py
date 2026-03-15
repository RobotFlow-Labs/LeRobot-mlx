#!/usr/bin/env python
"""Tests for Pi0-FAST policy -- MLX implementation.

Pi0-FAST uses tokenized autoregressive action generation instead of
continuous flow matching. All tests use synthetic data.
"""

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim
import pytest

from lerobot_mlx.policies.pi0.configuration_pi0 import (
    GEMMA_VARIANTS,
    GemmaVariantConfig,
)
from lerobot_mlx.policies.pi0_fast.configuration_pi0_fast import Pi0FastConfig
from lerobot_mlx.policies.pi0_fast.modeling_pi0_fast import (
    Pi0FastModel,
    Pi0FastPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(B=2, state_dim=8, action_dim=8, chunk_size=10):
    return {
        "observation.state": mx.random.normal((B, state_dim)),
        "action": mx.random.normal((B, chunk_size, action_dim)),
    }


def _default_config(**overrides) -> Pi0FastConfig:
    defaults = dict(
        action_expert_variant="gemma_300m",
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=8,
        max_action_dim=8,
        action_vocab_size=32,
        max_decoding_steps=10,
        temperature=0.0,
    )
    defaults.update(overrides)
    return Pi0FastConfig(**defaults)


# Use a tiny variant for fast tests
_TINY_REGISTERED = False


def _register_tiny():
    global _TINY_REGISTERED
    if not _TINY_REGISTERED:
        GEMMA_VARIANTS["gemma_tiny_fast_test"] = GemmaVariantConfig(
            width=32, depth=2, mlp_dim=64,
            num_heads=2, num_kv_heads=1, head_dim=16,
        )
        _TINY_REGISTERED = True


def _tiny_config(**overrides) -> Pi0FastConfig:
    _register_tiny()
    defaults = dict(
        action_expert_variant="gemma_tiny_fast_test",
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=4,
        max_action_dim=4,
        action_vocab_size=16,
        max_decoding_steps=4,
        temperature=0.0,
    )
    defaults.update(overrides)
    return Pi0FastConfig(**defaults)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestPi0FastConfig:
    def test_config_defaults(self):
        cfg = Pi0FastConfig()
        assert cfg.chunk_size == 50
        assert cfg.n_action_steps == 50
        assert cfg.action_vocab_size == 256
        assert cfg.temperature == 0.0
        assert cfg.max_decoding_steps == 256
        assert cfg.use_kv_cache is True

    def test_config_expert_properties(self):
        cfg = Pi0FastConfig(action_expert_variant="gemma_300m")
        assert cfg.expert_hidden_size == 1024
        assert cfg.expert_intermediate_size == 4096
        assert cfg.expert_num_heads == 8
        assert cfg.expert_num_layers == 18

    def test_config_validation(self):
        with pytest.raises(ValueError, match="n_action_steps"):
            Pi0FastConfig(n_action_steps=60, chunk_size=50)

    def test_config_custom_fields(self):
        cfg = Pi0FastConfig(
            chunk_size=20,
            n_action_steps=15,
            action_vocab_size=512,
            temperature=0.5,
        )
        assert cfg.chunk_size == 20
        assert cfg.action_vocab_size == 512
        assert cfg.temperature == 0.5


# ---------------------------------------------------------------------------
# Model creation tests
# ---------------------------------------------------------------------------

class TestPi0FastModel:
    def test_model_creation(self):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        assert model.config is cfg

    def test_model_has_expected_layers(self):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        assert hasattr(model, "state_proj")
        assert hasattr(model, "action_token_embed")
        assert hasattr(model, "expert")
        assert hasattr(model, "token_head")
        assert hasattr(model, "action_detokenizer")


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------

class TestForward:
    def test_forward_shape(self):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        batch = _make_batch(
            B=2, state_dim=cfg.max_state_dim,
            action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size,
        )
        result = model(batch)
        mx.eval(result["loss"], result["logits"])
        # logits: (B, chunk_size, vocab_size)
        assert result["logits"].shape == (2, cfg.chunk_size, cfg.action_vocab_size)

    def test_forward_loss_finite(self):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        batch = _make_batch(
            B=2, state_dim=cfg.max_state_dim,
            action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size,
        )
        result = model(batch)
        mx.eval(result["loss"])
        assert mx.isfinite(result["loss"]).item()

    def test_forward_loss_positive(self):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        batch = _make_batch(
            B=2, state_dim=cfg.max_state_dim,
            action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size,
        )
        result = model(batch)
        mx.eval(result["loss"])
        assert result["loss"].item() > 0.0

    def test_gradient_flow(self):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        batch = _make_batch(
            B=2, state_dim=cfg.max_state_dim,
            action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size,
        )

        def loss_fn(model):
            return model(batch)["loss"]

        loss, grads = mx.value_and_grad(loss_fn)(model)
        mx.eval(loss, grads)
        assert mx.isfinite(loss).item()

        # Check at least some gradients are non-zero
        has_nonzero = False
        for key, val in _flatten_dict(grads):
            if isinstance(val, mx.array):
                mx.eval(val)
                if mx.any(val != 0).item():
                    has_nonzero = True
                    break
        assert has_nonzero, "No non-zero gradients found"


# ---------------------------------------------------------------------------
# Action generation tests
# ---------------------------------------------------------------------------

class TestGenerateActions:
    def test_generate_actions_shape(self):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        batch = {"observation.state": mx.random.normal((2, cfg.max_state_dim))}
        actions = model.generate_actions(batch)
        mx.eval(actions)
        assert actions.shape == (2, cfg.chunk_size, cfg.max_action_dim)

    def test_generate_actions_finite(self):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        batch = {"observation.state": mx.random.normal((2, cfg.max_state_dim))}
        actions = model.generate_actions(batch)
        mx.eval(actions)
        assert mx.all(mx.isfinite(actions)).item()

    def test_generate_greedy_deterministic(self):
        """temperature=0 (greedy) should produce deterministic results."""
        cfg = _tiny_config(temperature=0.0)
        model = Pi0FastModel(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        mx.eval(batch["observation.state"])

        # Fix model parameters
        mx.eval(model.parameters())

        a1 = model.generate_actions(batch, temperature=0.0)
        mx.eval(a1)
        a2 = model.generate_actions(batch, temperature=0.0)
        mx.eval(a2)
        assert mx.allclose(a1, a2, atol=1e-5).item(), "Greedy decoding should be deterministic"

    def test_generate_with_temperature(self):
        """Non-zero temperature should allow sampling."""
        cfg = _tiny_config(temperature=1.0)
        model = Pi0FastModel(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        mx.eval(batch["observation.state"], model.parameters())

        # With temperature, results may differ due to sampling
        actions = model.generate_actions(batch, temperature=1.0)
        mx.eval(actions)
        assert actions.shape == (1, cfg.chunk_size, cfg.max_action_dim)
        assert mx.all(mx.isfinite(actions)).item()


# ---------------------------------------------------------------------------
# Tokenization tests
# ---------------------------------------------------------------------------

class TestTokenization:
    def test_tokenize_actions_range(self):
        """Tokens should be in [0, vocab_size-1]."""
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        actions = mx.random.uniform(low=-1.0, high=1.0, shape=(2, 4, cfg.max_action_dim))
        tokens = model._tokenize_actions(actions)
        mx.eval(tokens)
        assert tokens.shape == (2, 4)
        assert mx.all(tokens >= 0).item()
        assert mx.all(tokens < cfg.action_vocab_size).item()

    def test_tokenize_actions_boundary(self):
        """Actions at -1 and 1 should map to 0 and vocab_size-1."""
        cfg = _tiny_config(action_vocab_size=16)
        model = Pi0FastModel(cfg)
        actions = mx.array([[[-1.0, 0.0], [1.0, 0.0]]])  # (1, 2, 2)
        tokens = model._tokenize_actions(actions)
        mx.eval(tokens)
        assert tokens[0, 0].item() == 0
        assert tokens[0, 1].item() == 15  # vocab_size - 1

    def test_action_detokenizer(self):
        """Detokenizer should map one-hot vectors to continuous actions."""
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        B, T, V = 2, 4, cfg.action_vocab_size
        one_hot = mx.zeros((B, T, V))
        # Set arbitrary tokens
        one_hot = one_hot.at[:, 0, 0].add(1.0)
        one_hot = one_hot.at[:, 1, 5].add(1.0)
        actions = model.action_detokenizer(one_hot)
        mx.eval(actions)
        assert actions.shape == (B, T, cfg.max_action_dim)
        assert mx.all(mx.isfinite(actions)).item()


# ---------------------------------------------------------------------------
# Policy wrapper tests
# ---------------------------------------------------------------------------

class TestPi0FastPolicy:
    def test_policy_creation(self):
        cfg = _tiny_config()
        policy = Pi0FastPolicy(cfg)
        assert policy.config is cfg
        assert isinstance(policy.model, Pi0FastModel)

    def test_policy_name(self):
        cfg = _tiny_config()
        policy = Pi0FastPolicy(cfg)
        assert policy.name == "pi0_fast"

    def test_select_action(self):
        cfg = _tiny_config()
        policy = Pi0FastPolicy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        action = policy.select_action(batch)
        mx.eval(action)
        assert action.shape == (1, cfg.max_action_dim)

    def test_action_queue_caches(self):
        cfg = _tiny_config(chunk_size=4, n_action_steps=4)
        policy = Pi0FastPolicy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        a1 = policy.select_action(batch)
        mx.eval(a1)
        assert len(policy._action_queue) == 3  # 4 - 1

    def test_reset(self):
        cfg = _tiny_config()
        policy = Pi0FastPolicy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        policy.select_action(batch)
        assert len(policy._action_queue) > 0
        policy.reset()
        assert len(policy._action_queue) == 0

    def test_forward(self):
        cfg = _tiny_config()
        policy = Pi0FastPolicy(cfg)
        batch = _make_batch(
            B=2, state_dim=cfg.max_state_dim,
            action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size,
        )
        result = policy(batch)
        mx.eval(result["loss"])
        assert mx.isfinite(result["loss"]).item()

    def test_default_config_creation(self):
        policy = Pi0FastPolicy()
        assert isinstance(policy.config, Pi0FastConfig)


# ---------------------------------------------------------------------------
# Batch size variation tests
# ---------------------------------------------------------------------------

class TestVariousBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_various_batch_sizes(self, batch_size):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        batch = _make_batch(
            B=batch_size, state_dim=cfg.max_state_dim,
            action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size,
        )
        result = model(batch)
        mx.eval(result["loss"])
        assert mx.isfinite(result["loss"]).item()
        assert result["logits"].shape[0] == batch_size

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_generate_various_batch_sizes(self, batch_size):
        cfg = _tiny_config()
        model = Pi0FastModel(cfg)
        batch = {"observation.state": mx.random.normal((batch_size, cfg.max_state_dim))}
        actions = model.generate_actions(batch)
        mx.eval(actions)
        assert actions.shape == (batch_size, cfg.chunk_size, cfg.max_action_dim)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _flatten_dict(d, prefix=""):
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
