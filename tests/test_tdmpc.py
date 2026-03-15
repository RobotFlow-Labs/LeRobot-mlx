#!/usr/bin/env python
"""Tests for TD-MPC policy — MLX port.

20+ comprehensive tests covering config, model creation, forward passes,
loss computation, planning, gradient flow, and various batch sizes.
"""

import numpy as np
import pytest
import mlx.core as mx

from lerobot_mlx.policies.tdmpc.configuration_tdmpc import (
    TDMPCConfig, PolicyFeature, FeatureType, NormalizationMode,
    ACTION, OBS_STATE, OBS_ENV_STATE, OBS_IMAGE, REWARD, OBS_STR,
)
from lerobot_mlx.policies.tdmpc.modeling_tdmpc import (
    TDMPCPolicy, TDMPCTOLD, TDMPCObservationEncoder,
    update_ema_parameters, flatten_forward_unflatten,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    state_dim: int = 4,
    action_dim: int = 2,
    latent_dim: int = 16,
    mlp_dim: int = 32,
    q_ensemble_size: int = 3,
    horizon: int = 3,
    n_gaussian_samples: int = 8,
    n_pi_samples: int = 4,
    n_elites: int = 4,
    image: bool = False,
    image_shape: tuple = (3, 64, 64),
    env_state_dim: int = 0,
    **kwargs,
) -> TDMPCConfig:
    """Create a small TDMPCConfig suitable for testing."""
    input_features = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (state_dim,)),
    }
    if image:
        input_features["observation.images.top"] = PolicyFeature(FeatureType.VISUAL, image_shape)
    if env_state_dim > 0:
        input_features[OBS_ENV_STATE] = PolicyFeature(FeatureType.ENV, (env_state_dim,))

    output_features = {
        ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,)),
    }

    return TDMPCConfig(
        input_features=input_features,
        output_features=output_features,
        latent_dim=latent_dim,
        mlp_dim=mlp_dim,
        q_ensemble_size=q_ensemble_size,
        horizon=horizon,
        n_gaussian_samples=n_gaussian_samples,
        n_pi_samples=n_pi_samples,
        n_elites=n_elites,
        **kwargs,
    )


def _make_told(config: TDMPCConfig | None = None) -> TDMPCTOLD:
    if config is None:
        config = _make_config()
    return TDMPCTOLD(config)


# ===========================================================================
# TESTS
# ===========================================================================


class TestTDMPCConfig:
    def test_tdmpc_config_defaults(self):
        """Config should instantiate with valid defaults."""
        config = _make_config()
        assert config.latent_dim == 16
        assert config.horizon == 3
        assert config.discount == 0.9
        assert config.n_obs_steps == 1
        assert config.use_mpc is True
        assert config.n_action_repeats == 2

    def test_tdmpc_config_validation_gaussian_samples(self):
        """Should raise when n_gaussian_samples <= 0."""
        with pytest.raises(ValueError, match="gaussian samples"):
            _make_config(n_gaussian_samples=0)

    def test_tdmpc_config_validation_n_obs_steps(self):
        """Should raise when n_obs_steps != 1."""
        with pytest.raises(ValueError, match="observation steps"):
            _make_config(n_obs_steps=2)

    def test_tdmpc_config_feature_properties(self):
        """Feature helper properties should work correctly."""
        config = _make_config(env_state_dim=3)
        assert config.robot_state_feature is not None
        assert config.robot_state_feature.shape == (4,)
        assert config.env_state_feature is not None
        assert config.env_state_feature.shape == (3,)
        assert config.action_feature is not None
        assert config.action_feature.shape == (2,)


class TestTDMPCModelCreation:
    def test_tdmpc_model_creation(self):
        """TOLD model should instantiate without errors."""
        config = _make_config()
        told = TDMPCTOLD(config)
        assert told.dynamics is not None
        assert told.reward_net is not None
        assert told.pi_net is not None
        assert len(told.q_nets) == config.q_ensemble_size
        assert told.v_net is not None

    def test_tdmpc_policy_creation(self):
        """Full TDMPCPolicy should instantiate without errors."""
        config = _make_config()
        policy = TDMPCPolicy(config)
        assert policy.model is not None
        assert policy.model_target is not None

    def test_tdmpc_model_with_image_encoder(self):
        """TOLD with image encoder should instantiate."""
        config = _make_config(image=True)
        told = TDMPCTOLD(config)
        assert hasattr(told.encoder, 'image_enc_layers')
        assert hasattr(told.encoder, 'image_fc')


class TestTDMPCEncoderForward:
    def test_tdmpc_encoder_forward_state_only(self):
        """Encoder with state-only input should produce correct latent shape."""
        config = _make_config()
        told = _make_told(config)
        batch_size = 4
        obs = {OBS_STATE: mx.random.normal(shape=(batch_size, 4))}
        z = told.encode(obs)
        mx.eval(z)
        assert z.shape == (batch_size, config.latent_dim)

    def test_tdmpc_encoder_forward_with_env_state(self):
        """Encoder with state + env_state should produce correct latent shape."""
        config = _make_config(env_state_dim=3)
        told = TDMPCTOLD(config)
        batch_size = 2
        obs = {
            OBS_STATE: mx.random.normal(shape=(batch_size, 4)),
            OBS_ENV_STATE: mx.random.normal(shape=(batch_size, 3)),
        }
        z = told.encode(obs)
        mx.eval(z)
        assert z.shape == (batch_size, config.latent_dim)

    def test_tdmpc_encoder_forward_with_image(self):
        """Encoder with image + state should produce correct latent shape."""
        config = _make_config(image=True)
        told = TDMPCTOLD(config)
        batch_size = 2
        obs = {
            OBS_STATE: mx.random.normal(shape=(batch_size, 4)),
            "observation.images.top": mx.random.normal(shape=(batch_size, 3, 64, 64)),
        }
        z = told.encode(obs)
        mx.eval(z)
        assert z.shape == (batch_size, config.latent_dim)


class TestTDMPCDynamics:
    def test_tdmpc_dynamics_forward(self):
        """Dynamics model should predict next latent with correct shape."""
        config = _make_config()
        told = _make_told(config)
        batch_size = 4
        z = mx.random.normal(shape=(batch_size, config.latent_dim))
        a = mx.random.normal(shape=(batch_size, 2))
        z_next = told.latent_dynamics(z, a)
        mx.eval(z_next)
        assert z_next.shape == (batch_size, config.latent_dim)
        # Sigmoid output: should be in [0, 1]
        assert float(mx.min(z_next)) >= 0.0
        assert float(mx.max(z_next)) <= 1.0

    def test_tdmpc_dynamics_and_reward(self):
        """Dynamics + reward prediction should produce correct shapes."""
        config = _make_config()
        told = _make_told(config)
        batch_size = 4
        z = mx.random.normal(shape=(batch_size, config.latent_dim))
        a = mx.random.normal(shape=(batch_size, 2))
        z_next, reward = told.latent_dynamics_and_reward(z, a)
        mx.eval(z_next)
        mx.eval(reward)
        assert z_next.shape == (batch_size, config.latent_dim)
        assert reward.shape == (batch_size,)


class TestTDMPCReward:
    def test_tdmpc_reward_prediction(self):
        """Reward prediction should have scalar output per batch element."""
        config = _make_config()
        told = _make_told(config)
        batch_size = 4
        z = mx.random.normal(shape=(batch_size, config.latent_dim))
        a = mx.random.normal(shape=(batch_size, 2))
        _, reward = told.latent_dynamics_and_reward(z, a)
        mx.eval(reward)
        assert reward.shape == (batch_size,)


class TestTDMPCValue:
    def test_tdmpc_value_prediction(self):
        """V(z) should produce scalar per batch element."""
        config = _make_config()
        told = _make_told(config)
        batch_size = 4
        z = mx.random.normal(shape=(batch_size, config.latent_dim))
        v = told.V(z)
        mx.eval(v)
        assert v.shape == (batch_size,)

    def test_tdmpc_q_ensemble_prediction(self):
        """Qs(z, a) should produce (ensemble, batch) tensor."""
        config = _make_config()
        told = _make_told(config)
        batch_size = 4
        z = mx.random.normal(shape=(batch_size, config.latent_dim))
        a = mx.random.normal(shape=(batch_size, 2))
        qs = told.Qs(z, a)
        mx.eval(qs)
        assert qs.shape == (config.q_ensemble_size, batch_size)

    def test_tdmpc_q_return_min(self):
        """Qs with return_min should produce (batch,) tensor."""
        config = _make_config()
        told = _make_told(config)
        batch_size = 4
        z = mx.random.normal(shape=(batch_size, config.latent_dim))
        a = mx.random.normal(shape=(batch_size, 2))
        q_min = told.Qs(z, a, return_min=True)
        mx.eval(q_min)
        assert q_min.shape == (batch_size,)


class TestTDMPCPolicy_Forward:
    def test_tdmpc_policy_forward(self):
        """Policy pi(z) should produce actions with correct shape."""
        config = _make_config()
        told = _make_told(config)
        batch_size = 4
        z = mx.random.normal(shape=(batch_size, config.latent_dim))
        action = told.pi(z)
        mx.eval(action)
        assert action.shape == (batch_size, 2)
        # tanh output: should be in [-1, 1]
        assert float(mx.max(mx.abs(action))) <= 1.0 + 1e-6

    def test_tdmpc_policy_forward_with_noise(self):
        """Policy with std > 0 should add noise to actions."""
        config = _make_config()
        told = _make_told(config)
        z = mx.random.normal(shape=(2, config.latent_dim))
        action_no_noise = told.pi(z, std=0.0)
        action_with_noise = told.pi(z, std=0.5)
        mx.eval(action_no_noise)
        mx.eval(action_with_noise)
        # With noise, actions can exceed [-1, 1]
        assert action_with_noise.shape == (2, 2)


class TestTDMPCForwardShape:
    def test_tdmpc_forward_shape(self):
        """Full forward pass should produce loss and info dict."""
        config = _make_config(horizon=3)
        policy = TDMPCPolicy(config)
        batch_size = 4
        horizon = config.horizon

        batch = {
            OBS_STATE: mx.random.normal(shape=(batch_size, horizon + 1, 4)),
            ACTION: mx.random.normal(shape=(batch_size, horizon, 2)),
            REWARD: mx.random.normal(shape=(batch_size, horizon)),
            "index": mx.zeros((batch_size,)),
        }

        loss, info = policy.forward(batch)
        mx.eval(loss)
        assert loss.shape == ()
        assert "consistency_loss" in info
        assert "reward_loss" in info
        assert "Q_value_loss" in info
        assert "V_value_loss" in info
        assert "pi_loss" in info
        assert "sum_loss" in info
        assert "Q" in info
        assert "V" in info


class TestTDMPCLoss:
    def test_tdmpc_loss_computation(self):
        """Loss should be a finite scalar."""
        config = _make_config(horizon=2)
        policy = TDMPCPolicy(config)
        batch_size = 2

        batch = {
            OBS_STATE: mx.random.normal(shape=(batch_size, 3, 4)),
            ACTION: mx.random.normal(shape=(batch_size, 2, 2)),
            REWARD: mx.random.normal(shape=(batch_size, 2)),
            "index": mx.zeros((batch_size,)),
        }

        loss, info = policy.forward(batch)
        mx.eval(loss)
        loss_val = float(loss)
        assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"


class TestTDMPCGradient:
    def test_tdmpc_gradient_flow(self):
        """Gradients should flow through the model."""
        config = _make_config(horizon=2, mlp_dim=16, latent_dim=8)
        policy = TDMPCPolicy(config)
        batch_size = 2

        batch = {
            OBS_STATE: mx.random.normal(shape=(batch_size, 3, 4)),
            ACTION: mx.random.normal(shape=(batch_size, 2, 2)),
            REWARD: mx.random.normal(shape=(batch_size, 2)),
            "index": mx.zeros((batch_size,)),
        }

        def loss_fn(model):
            # Temporarily replace policy.model
            old_model = policy.model
            policy.model = model
            loss, _ = policy.forward(dict(batch))
            policy.model = old_model
            return loss

        loss, grads = mx.value_and_grad(loss_fn)(policy.model)
        mx.eval(loss)
        mx.eval(grads)

        # Check that at least some gradients are non-zero
        from mlx.utils import tree_flatten
        flat_grads = tree_flatten(grads)
        has_nonzero = False
        for name, g in flat_grads:
            if isinstance(g, mx.array) and g.size > 0:
                if float(mx.max(mx.abs(g))) > 0:
                    has_nonzero = True
                    break
        assert has_nonzero, "All gradients are zero!"


class TestTDMPCSelectAction:
    def test_tdmpc_select_action_no_mpc(self):
        """select_action without MPC should work."""
        config = _make_config(use_mpc=False, n_action_repeats=1, n_action_steps=1)
        policy = TDMPCPolicy(config)
        policy.reset()

        obs = {
            OBS_STATE: mx.random.normal(shape=(1, 4)),
        }
        action = policy.select_action(obs)
        mx.eval(action)
        assert action.shape == (1, 2)

    def test_tdmpc_select_action_with_mpc(self):
        """select_action with MPC (planning) should produce valid actions."""
        config = _make_config(
            use_mpc=True,
            n_gaussian_samples=8,
            n_pi_samples=4,
            n_elites=4,
            cem_iterations=2,
            horizon=2,
            n_action_repeats=1,
            n_action_steps=1,
        )
        policy = TDMPCPolicy(config)
        policy.reset()

        obs = {
            OBS_STATE: mx.random.normal(shape=(1, 4)),
        }
        action = policy.select_action(obs)
        mx.eval(action)
        assert action.shape == (1, 2)
        # Actions should be clipped to [-1, 1]
        assert float(mx.max(mx.abs(action))) <= 1.0 + 1e-6


class TestTDMPCPlanning:
    def test_tdmpc_planning_loop(self):
        """MPPI/CEM planning should produce valid action sequences."""
        config = _make_config(
            n_gaussian_samples=8,
            n_pi_samples=4,
            n_elites=4,
            cem_iterations=2,
            horizon=3,
        )
        policy = TDMPCPolicy(config)
        batch_size = 2
        z = mx.random.normal(shape=(batch_size, config.latent_dim))

        actions = policy.plan(z)
        mx.eval(actions)
        assert actions.shape == (config.horizon, batch_size, 2)

    def test_tdmpc_planning_warm_start(self):
        """Planning should warm-start from previous mean."""
        config = _make_config(
            n_gaussian_samples=8,
            n_pi_samples=4,
            n_elites=4,
            cem_iterations=2,
            horizon=3,
        )
        policy = TDMPCPolicy(config)
        z = mx.random.normal(shape=(1, config.latent_dim))

        # First plan
        actions1 = policy.plan(z)
        mx.eval(actions1)
        assert policy._prev_mean is not None

        # Second plan should use warm start
        z2 = mx.random.normal(shape=(1, config.latent_dim))
        actions2 = policy.plan(z2)
        mx.eval(actions2)
        assert actions2.shape == (config.horizon, 1, 2)


class TestTDMPCBatchSizes:
    def test_tdmpc_various_batch_sizes(self):
        """Model should handle various batch sizes."""
        config = _make_config()
        told = _make_told(config)

        for batch_size in [1, 2, 4, 8]:
            z = mx.random.normal(shape=(batch_size, config.latent_dim))
            a = mx.random.normal(shape=(batch_size, 2))

            # Encoder
            obs = {OBS_STATE: mx.random.normal(shape=(batch_size, 4))}
            z_enc = told.encode(obs)
            mx.eval(z_enc)
            assert z_enc.shape == (batch_size, config.latent_dim)

            # Dynamics
            z_next = told.latent_dynamics(z, a)
            mx.eval(z_next)
            assert z_next.shape == (batch_size, config.latent_dim)

            # Reward
            _, reward = told.latent_dynamics_and_reward(z, a)
            mx.eval(reward)
            assert reward.shape == (batch_size,)

            # Value
            v = told.V(z)
            mx.eval(v)
            assert v.shape == (batch_size,)

            # Q
            qs = told.Qs(z, a)
            mx.eval(qs)
            assert qs.shape == (config.q_ensemble_size, batch_size)

            # Policy
            action = told.pi(z)
            mx.eval(action)
            assert action.shape == (batch_size, 2)


class TestTDMPCEMA:
    def test_ema_update(self):
        """EMA update should move target params toward source params."""
        config = _make_config()
        policy = TDMPCPolicy(config)

        # Get initial target params
        from mlx.utils import tree_flatten
        target_before = dict(tree_flatten(policy.model_target.parameters()))

        # Modify model params (simulate training step)
        model_params = tree_flatten(policy.model.parameters())
        modified = [(k, v + 1.0) for k, v in model_params]
        policy.model.load_weights(modified)

        # Run EMA update
        policy.update()

        # Target should have moved
        target_after = dict(tree_flatten(policy.model_target.parameters()))
        moved = False
        for key in target_before:
            if key in target_after:
                diff = float(mx.max(mx.abs(target_after[key] - target_before[key])))
                if diff > 1e-6:
                    moved = True
                    break
        assert moved, "EMA update did not change target parameters"


class TestTDMPCEstimateValue:
    def test_estimate_value_shape(self):
        """estimate_value should produce (n_samples, batch) tensor."""
        config = _make_config(
            n_gaussian_samples=4,
            n_pi_samples=2,
            n_elites=3,
            horizon=2,
        )
        policy = TDMPCPolicy(config)
        n_samples = 6
        batch_size = 2
        z = mx.random.normal(shape=(n_samples, batch_size, config.latent_dim))
        actions = mx.random.normal(shape=(config.horizon, n_samples, batch_size, 2))
        value = policy.estimate_value(z, actions)
        mx.eval(value)
        assert value.shape == (n_samples, batch_size)


class TestFlattenForwardUnflatten:
    def test_4d_passthrough(self):
        """4D tensor should pass through unchanged."""
        x = mx.random.normal(shape=(2, 3, 8, 8))
        result = flatten_forward_unflatten(lambda t: t.reshape(t.shape[0], -1), x)
        mx.eval(result)
        assert result.shape == (2, 3 * 8 * 8)

    def test_5d_flatten_unflatten(self):
        """5D tensor should be flattened to 4D, processed, and unflattened."""
        x = mx.random.normal(shape=(3, 4, 3, 8, 8))
        result = flatten_forward_unflatten(lambda t: t.reshape(t.shape[0], -1), x)
        mx.eval(result)
        assert result.shape == (3, 4, 3 * 8 * 8)


class TestTDMPCReset:
    def test_reset_clears_queues(self):
        """Reset should clear all queues and prev_mean."""
        config = _make_config()
        policy = TDMPCPolicy(config)
        policy._prev_mean = mx.zeros((3, 1, 2))
        policy.reset()
        assert policy._prev_mean is None
        assert len(policy._queues[ACTION]) == 0
        assert len(policy._queues[OBS_STATE]) == 0
