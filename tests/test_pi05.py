#!/usr/bin/env python
"""Tests for Pi0.5 policy -- MLX implementation.

Pi0.5 is identical to Pi0 with QUANTILES normalization.
All tests use synthetic data; no internet or pretrained weights required.
"""

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim
import pytest

from lerobot_mlx.policies.pi0.configuration_pi0 import Pi0Config
from lerobot_mlx.policies.pi0.modeling_pi0 import Pi0Policy, Pi0FlowMatching
from lerobot_mlx.policies.pi05.configuration_pi05 import Pi05Config
from lerobot_mlx.policies.pi05.modeling_pi05 import Pi05Policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(B=2, state_dim=8, action_dim=8, chunk_size=10):
    return {
        "observation.state": mx.random.normal((B, state_dim)),
        "action": mx.random.normal((B, chunk_size, action_dim)),
    }


def _default_config(**overrides) -> Pi05Config:
    defaults = dict(
        action_expert_variant="gemma_300m",
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=8,
        max_action_dim=8,
        num_inference_steps=3,
    )
    defaults.update(overrides)
    return Pi05Config(**defaults)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestPi05Config:
    def test_config_defaults(self):
        cfg = Pi05Config()
        assert cfg.normalization_mode == "quantiles"
        assert cfg.tokenizer_max_length == 200

    def test_config_inherits_pi0(self):
        cfg = Pi05Config()
        assert isinstance(cfg, Pi0Config)
        assert cfg.chunk_size == 50
        assert cfg.n_action_steps == 50
        assert cfg.max_state_dim == 32
        assert cfg.max_action_dim == 32

    def test_config_differs_from_pi0(self):
        pi0_cfg = Pi0Config()
        pi05_cfg = Pi05Config()
        # Pi0 does not have normalization_mode by default
        assert not hasattr(pi0_cfg, "normalization_mode") or \
               getattr(pi0_cfg, "normalization_mode", None) != "quantiles"
        assert pi05_cfg.normalization_mode == "quantiles"

    def test_config_expert_properties(self):
        cfg = Pi05Config(action_expert_variant="gemma_300m")
        assert cfg.expert_hidden_size == 1024
        assert cfg.expert_intermediate_size == 4096
        assert cfg.expert_num_heads == 8
        assert cfg.expert_num_layers == 18

    def test_config_validation(self):
        with pytest.raises(ValueError, match="n_action_steps"):
            Pi05Config(n_action_steps=60, chunk_size=50)

    def test_config_custom_fields(self):
        cfg = Pi05Config(
            chunk_size=20,
            n_action_steps=15,
            normalization_mode="quantiles",
            tokenizer_max_length=300,
        )
        assert cfg.chunk_size == 20
        assert cfg.tokenizer_max_length == 300


# ---------------------------------------------------------------------------
# Model tests -- Pi0.5 inherits Pi0 model
# ---------------------------------------------------------------------------

class TestPi05Policy:
    def test_policy_creation(self):
        cfg = _default_config()
        policy = Pi05Policy(cfg)
        assert policy.config is cfg
        assert isinstance(policy.model, Pi0FlowMatching)

    def test_policy_inherits_pi0(self):
        cfg = _default_config()
        policy = Pi05Policy(cfg)
        assert isinstance(policy, Pi0Policy)

    def test_policy_name(self):
        cfg = _default_config()
        policy = Pi05Policy(cfg)
        assert policy.name == "pi05"

    def test_normalization_mode(self):
        cfg = _default_config()
        policy = Pi05Policy(cfg)
        assert policy._normalization_mode == "quantiles"

    def test_forward(self):
        cfg = _default_config()
        policy = Pi05Policy(cfg)
        batch = _make_batch(
            B=2, state_dim=cfg.max_state_dim,
            action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size,
        )
        result = policy(batch)
        mx.eval(result["loss"])
        assert mx.isfinite(result["loss"]).item()

    def test_forward_shape(self):
        cfg = _default_config()
        policy = Pi05Policy(cfg)
        batch = _make_batch(
            B=2, state_dim=cfg.max_state_dim,
            action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size,
        )
        result = policy(batch)
        mx.eval(result["predicted_velocity"])
        assert result["predicted_velocity"].shape == (2, cfg.chunk_size, cfg.max_action_dim)

    def test_select_action(self):
        cfg = _default_config()
        policy = Pi05Policy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        action = policy.select_action(batch)
        mx.eval(action)
        assert action.shape == (1, cfg.max_action_dim)

    def test_action_queue(self):
        cfg = _default_config(chunk_size=5, n_action_steps=5)
        policy = Pi05Policy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        a1 = policy.select_action(batch)
        mx.eval(a1)
        assert len(policy._action_queue) == 4

    def test_reset(self):
        cfg = _default_config()
        policy = Pi05Policy(cfg)
        batch = {"observation.state": mx.random.normal((1, cfg.max_state_dim))}
        policy.select_action(batch)
        assert len(policy._action_queue) > 0
        policy.reset()
        assert len(policy._action_queue) == 0

    def test_gradient_flow(self):
        cfg = _default_config()
        model = Pi05Policy(cfg)
        batch = _make_batch(
            B=2, state_dim=cfg.max_state_dim,
            action_dim=cfg.max_action_dim, chunk_size=cfg.chunk_size,
        )

        def loss_fn(model):
            return model(batch)["loss"]

        loss, grads = mx.value_and_grad(loss_fn)(model)
        mx.eval(loss, grads)
        assert mx.isfinite(loss).item()

    def test_default_config_creation(self):
        """Pi05Policy with no config should use Pi05Config defaults."""
        policy = Pi05Policy()
        assert isinstance(policy.config, Pi05Config)
        assert policy._normalization_mode == "quantiles"
