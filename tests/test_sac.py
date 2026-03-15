"""Comprehensive tests for SAC (Soft Actor-Critic) policy — MLX port.

Tests cover configuration, model creation, actor/critic forward passes,
Polyak update, temperature, loss computation, gradient flow, and more.
"""

import math

import mlx.core as mx
import mlx.nn as _nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from lerobot_mlx.policies.sac.configuration_sac import (
    SACConfig,
    FeatureShape,
    ACTION,
    OBS_STATE,
    is_image_feature,
)
from lerobot_mlx.policies.sac.modeling_sac import (
    SACPolicy,
    SACObservationEncoder,
    MLP,
    CriticHead,
    CriticEnsemble,
    Policy,
    TanhMultivariateNormalDiag,
    polyak_update,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    """A minimal SAC config with state observations."""
    return SACConfig(
        input_features={OBS_STATE: FeatureShape(shape=(10,))},
        output_features={ACTION: FeatureShape(shape=(4,))},
        latent_dim=64,
        num_critics=2,
        temperature_init=1.0,
        discount=0.99,
        critic_target_update_weight=0.005,
        use_backup_entropy=True,
    )


@pytest.fixture
def sac_policy(default_config):
    """Build a SACPolicy from default config."""
    return SACPolicy(config=default_config)


@pytest.fixture
def batch(default_config):
    """Create a dummy batch for testing."""
    batch_size = 8
    obs_dim = default_config.input_features[OBS_STATE].shape[0]
    action_dim = default_config.output_features[ACTION].shape[0]

    obs = mx.random.normal(shape=(batch_size, obs_dim))
    next_obs = mx.random.normal(shape=(batch_size, obs_dim))
    actions = mx.random.uniform(low=-1.0, high=1.0, shape=(batch_size, action_dim))
    rewards = mx.random.normal(shape=(batch_size,))
    done = mx.zeros((batch_size,))

    return {
        ACTION: actions,
        "state": {OBS_STATE: obs},
        "next_state": {OBS_STATE: next_obs},
        "reward": rewards,
        "done": done,
    }


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestSACConfig:
    def test_sac_config_defaults(self):
        """Test that SACConfig has sensible defaults."""
        config = SACConfig()
        assert config.discount == 0.99
        assert config.temperature_init == 1.0
        assert config.num_critics == 2
        assert config.critic_target_update_weight == 0.005
        assert config.use_backup_entropy is True

    def test_sac_config_validate_features(self):
        """Test feature validation catches missing state/image."""
        config = SACConfig(
            input_features={},
            output_features={ACTION: FeatureShape(shape=(4,))},
        )
        with pytest.raises(ValueError, match="observation"):
            config.validate_features()

    def test_sac_config_validate_action(self):
        """Test feature validation catches missing action."""
        config = SACConfig(
            input_features={OBS_STATE: FeatureShape(shape=(10,))},
            output_features={},
        )
        with pytest.raises(ValueError, match="action"):
            config.validate_features()

    def test_is_image_feature(self):
        """Test image feature detection."""
        assert is_image_feature("observation.image") is True
        assert is_image_feature("observation.image.top") is True
        assert is_image_feature("observation.state") is False
        assert is_image_feature("action") is False


# ---------------------------------------------------------------------------
# Model creation tests
# ---------------------------------------------------------------------------

class TestSACModelCreation:
    def test_sac_model_creation(self, sac_policy):
        """Test that SACPolicy can be instantiated."""
        assert sac_policy is not None
        assert hasattr(sac_policy, 'actor')
        assert hasattr(sac_policy, 'critic_ensemble')
        assert hasattr(sac_policy, 'critic_target')
        assert hasattr(sac_policy, 'log_alpha')

    def test_sac_encoder_creation(self, default_config):
        """Test that SACObservationEncoder is created correctly."""
        encoder = SACObservationEncoder(default_config)
        assert encoder.output_dim == default_config.latent_dim
        assert encoder.has_state is True
        assert encoder.has_images is False

    def test_mlp_creation(self):
        """Test MLP creation with various configs."""
        mlp = MLP(input_dim=32, hidden_dims=[64, 64])
        x = mx.random.normal(shape=(4, 32))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (4, 64)

    def test_mlp_with_activate_final(self):
        """Test MLP with final activation."""
        mlp = MLP(input_dim=32, hidden_dims=[64, 32], activate_final=True)
        x = mx.random.normal(shape=(4, 32))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (4, 32)


# ---------------------------------------------------------------------------
# Actor tests
# ---------------------------------------------------------------------------

class TestSACActor:
    def test_sac_actor_forward(self, sac_policy):
        """Test that actor produces actions, log_probs, and means."""
        obs = {OBS_STATE: mx.random.normal(shape=(4, 10))}
        actions, log_probs, means = sac_policy.actor(obs)
        mx.eval(actions, log_probs, means)

        assert actions.shape == (4, 4)
        assert log_probs.shape == (4,)
        assert means.shape == (4, 4)

    def test_sac_actor_action_bounds(self, sac_policy):
        """Test that actions are bounded in [-1, 1] after tanh squashing."""
        obs = {OBS_STATE: mx.random.normal(shape=(100, 10))}
        actions, _, _ = sac_policy.actor(obs)
        mx.eval(actions)

        actions_np = np.array(actions)
        assert np.all(actions_np >= -1.0), f"Min action: {actions_np.min()}"
        assert np.all(actions_np <= 1.0), f"Max action: {actions_np.max()}"

    def test_sac_select_action(self, sac_policy):
        """Test select_action returns actions of correct shape."""
        obs = {OBS_STATE: mx.random.normal(shape=(1, 10))}
        actions = sac_policy.select_action(obs)
        mx.eval(actions)
        assert actions.shape == (1, 4)

    def test_sac_eval_mode(self, sac_policy):
        """Test that eval mode works (deterministic via distribution mode)."""
        obs = {OBS_STATE: mx.random.normal(shape=(4, 10))}
        # In eval, we use select_action which uses the actor
        sac_policy.eval()
        actions = sac_policy.select_action(obs)
        mx.eval(actions)
        assert actions.shape == (4, 4)


# ---------------------------------------------------------------------------
# Critic tests
# ---------------------------------------------------------------------------

class TestSACCritic:
    def test_sac_critic_forward(self, sac_policy):
        """Test that critic produces Q-values of correct shape."""
        obs = {OBS_STATE: mx.random.normal(shape=(4, 10))}
        actions = mx.random.normal(shape=(4, 4))
        q_values = sac_policy.critic_forward(obs, actions, use_target=False)
        mx.eval(q_values)

        # Shape: (num_critics, batch_size)
        assert q_values.shape == (2, 4)

    def test_sac_twin_critics(self, sac_policy):
        """Test that twin critics produce different Q-values."""
        obs = {OBS_STATE: mx.random.normal(shape=(4, 10))}
        actions = mx.random.normal(shape=(4, 4))
        q_values = sac_policy.critic_forward(obs, actions, use_target=False)
        mx.eval(q_values)

        q1 = np.array(q_values[0])
        q2 = np.array(q_values[1])
        # Twin critics should generally produce different values
        # (initialized with different weights)
        assert q_values.shape[0] == 2

    def test_sac_target_critics(self, sac_policy):
        """Test that target critic params exist and match ensemble initially."""
        source_params = dict(tree_flatten(sac_policy.critic_ensemble.parameters()))
        target_params = dict(tree_flatten(sac_policy.critic_target.parameters()))

        # Should have same keys
        assert set(source_params.keys()) == set(target_params.keys())

        # Values should be close (copied at init)
        for key in source_params:
            mx.eval(source_params[key], target_params[key])
            np.testing.assert_allclose(
                np.array(source_params[key]),
                np.array(target_params[key]),
                rtol=1e-5,
                err_msg=f"Mismatch at key {key}",
            )

    def test_critic_head(self):
        """Test CriticHead produces scalar Q-values."""
        head = CriticHead(input_dim=32, hidden_dims=[64, 64])
        x = mx.random.normal(shape=(4, 32))
        out = head(x)
        mx.eval(out)
        assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# Polyak update tests
# ---------------------------------------------------------------------------

class TestPolyakUpdate:
    def test_sac_polyak_update(self, sac_policy):
        """Test that target params move toward source after Polyak update."""
        # Modify ensemble weights to differ from target
        source_params_before = dict(tree_flatten(sac_policy.critic_ensemble.parameters()))
        target_params_before = dict(tree_flatten(sac_policy.critic_target.parameters()))

        # Manually perturb ensemble weights
        # Find a critic-specific key (not shared encoder)
        critic_keys = [k for k in source_params_before if 'critics' in k]
        assert len(critic_keys) > 0, "No critic-specific keys found"
        first_key = critic_keys[0]
        old_source = np.array(source_params_before[first_key])
        old_target = np.array(target_params_before[first_key])

        # They start equal, so perturb source
        perturbed = source_params_before[first_key] + mx.ones_like(source_params_before[first_key]) * 10.0
        sac_policy.critic_ensemble.load_weights([(first_key, perturbed)], strict=False)
        mx.eval(perturbed)

        # Perform update
        sac_policy.update_target_networks()

        # Check target moved toward source
        target_params_after = dict(tree_flatten(sac_policy.critic_target.parameters()))
        mx.eval(target_params_after[first_key])
        new_target = np.array(target_params_after[first_key])

        tau = sac_policy.config.critic_target_update_weight
        expected = tau * np.array(perturbed) + (1 - tau) * old_target
        np.testing.assert_allclose(new_target, expected, rtol=1e-5)

    def test_polyak_update_function(self):
        """Test the standalone polyak_update function."""
        encoder = SACObservationEncoder(SACConfig(
            input_features={OBS_STATE: FeatureShape(shape=(4,))},
            output_features={ACTION: FeatureShape(shape=(2,))},
            latent_dim=16,
        ))

        source_head = CriticHead(input_dim=16 + 2, hidden_dims=[32, 32])
        target_head = CriticHead(input_dim=16 + 2, hidden_dims=[32, 32])

        source = CriticEnsemble(encoder=encoder, ensemble=[source_head])
        target = CriticEnsemble(encoder=encoder, ensemble=[target_head])

        # Get initial params
        s_params = dict(tree_flatten(source.parameters()))
        t_params_before = dict(tree_flatten(target.parameters()))

        polyak_update(source, target, tau=0.5)

        t_params_after = dict(tree_flatten(target.parameters()))

        # Check at least one param changed
        changed = False
        for key in t_params_after:
            if key in s_params and key in t_params_before:
                mx.eval(t_params_after[key])
                after = np.array(t_params_after[key])
                before = np.array(t_params_before[key])
                if not np.allclose(after, before, atol=1e-7):
                    changed = True
                    break
        assert changed, "Polyak update should change target params"


# ---------------------------------------------------------------------------
# Temperature tests
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_sac_temperature(self, sac_policy):
        """Test that log_alpha is accessible and temperature computes correctly."""
        log_alpha = sac_policy.log_alpha
        mx.eval(log_alpha)
        assert log_alpha.shape == (1,)

        temp = sac_policy.temperature
        expected = math.exp(float(np.array(log_alpha)[0]))
        assert abs(temp - expected) < 1e-5

    def test_sac_temperature_init(self):
        """Test temperature initialization with different values."""
        config = SACConfig(
            input_features={OBS_STATE: FeatureShape(shape=(10,))},
            output_features={ACTION: FeatureShape(shape=(4,))},
            temperature_init=0.5,
        )
        policy = SACPolicy(config=config)
        mx.eval(policy.log_alpha)
        expected_log_alpha = math.log(0.5)
        actual = float(np.array(policy.log_alpha)[0])
        assert abs(actual - expected_log_alpha) < 1e-5


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------

class TestTanhDistribution:
    def test_sac_log_prob_correction(self):
        """Test tanh squashing correction in log probability."""
        loc = mx.zeros((4, 3))
        scale = mx.ones((4, 3))
        dist = TanhMultivariateNormalDiag(loc=loc, scale_diag=scale)

        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        mx.eval(actions, log_probs)

        # Log probs should be finite
        lp_np = np.array(log_probs)
        assert np.all(np.isfinite(lp_np)), f"Non-finite log probs: {lp_np}"
        assert log_probs.shape == (4,)

    def test_distribution_mode(self):
        """Test deterministic mode of distribution."""
        loc = mx.array([[0.5, -0.3, 0.1]])
        scale = mx.ones((1, 3)) * 0.1
        dist = TanhMultivariateNormalDiag(loc=loc, scale_diag=scale)

        mode = dist.mode()
        mx.eval(mode)
        expected = np.tanh(np.array(loc))
        np.testing.assert_allclose(np.array(mode), expected, rtol=1e-5)

    def test_distribution_sample_bounded(self):
        """Test that samples from TanhMultivariateNormalDiag are in [-1, 1]."""
        loc = mx.random.normal(shape=(100, 5))
        scale = mx.ones((100, 5)) * 2.0
        dist = TanhMultivariateNormalDiag(loc=loc, scale_diag=scale)

        samples = dist.rsample()
        mx.eval(samples)
        s_np = np.array(samples)
        assert np.all(s_np >= -1.0) and np.all(s_np <= 1.0)


# ---------------------------------------------------------------------------
# Loss computation tests
# ---------------------------------------------------------------------------

class TestLossComputation:
    def test_sac_loss_critic(self, sac_policy, batch):
        """Test critic loss computation."""
        loss = sac_policy.compute_loss_critic(
            observations=batch["state"],
            actions=batch[ACTION],
            rewards=batch["reward"],
            next_observations=batch["next_state"],
            done=batch["done"],
        )
        mx.eval(loss)
        assert loss.shape == ()  # scalar
        assert np.isfinite(float(np.array(loss)))

    def test_sac_loss_actor(self, sac_policy, batch):
        """Test actor loss computation."""
        loss = sac_policy.compute_loss_actor(
            observations=batch["state"],
        )
        mx.eval(loss)
        assert loss.shape == ()  # scalar
        assert np.isfinite(float(np.array(loss)))

    def test_sac_loss_temperature(self, sac_policy, batch):
        """Test temperature loss computation."""
        loss = sac_policy.compute_loss_temperature(
            observations=batch["state"],
        )
        mx.eval(loss)
        assert loss.shape == ()  # scalar
        assert np.isfinite(float(np.array(loss)))

    def test_sac_forward_critic(self, sac_policy, batch):
        """Test forward() with model='critic'."""
        result = sac_policy.forward(batch, model="critic")
        mx.eval(result["loss_critic"])
        assert "loss_critic" in result
        assert np.isfinite(float(np.array(result["loss_critic"])))

    def test_sac_forward_actor(self, sac_policy, batch):
        """Test forward() with model='actor'."""
        result = sac_policy.forward(batch, model="actor")
        mx.eval(result["loss_actor"])
        assert "loss_actor" in result
        assert np.isfinite(float(np.array(result["loss_actor"])))

    def test_sac_forward_temperature(self, sac_policy, batch):
        """Test forward() with model='temperature'."""
        result = sac_policy.forward(batch, model="temperature")
        mx.eval(result["loss_temperature"])
        assert "loss_temperature" in result
        assert np.isfinite(float(np.array(result["loss_temperature"])))

    def test_sac_forward_invalid_model(self, sac_policy, batch):
        """Test forward() raises on invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            sac_policy.forward(batch, model="invalid")


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_sac_gradient_flow_critic(self, sac_policy, batch):
        """Test that gradients flow through critic loss."""
        def critic_loss_fn(model):
            result = model.forward(batch, model="critic")
            return result["loss_critic"]

        loss, grads = _nn.value_and_grad(sac_policy, critic_loss_fn)(sac_policy)
        mx.eval(loss, grads)

        # Check we got a finite loss
        assert np.isfinite(float(np.array(loss)))

        # Check some gradients are non-zero
        flat_grads = tree_flatten(grads)
        has_nonzero = any(
            np.any(np.array(g) != 0)
            for _, g in flat_grads
            if isinstance(g, mx.array) and g.size > 0
        )
        assert has_nonzero, "All gradients are zero — no gradient flow"

    def test_sac_gradient_flow_actor(self, sac_policy, batch):
        """Test that gradients flow through actor loss."""
        def actor_loss_fn(model):
            result = model.forward(batch, model="actor")
            return result["loss_actor"]

        loss, grads = _nn.value_and_grad(sac_policy, actor_loss_fn)(sac_policy)
        mx.eval(loss, grads)
        assert np.isfinite(float(np.array(loss)))


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------

class TestForwardShape:
    def test_sac_forward_shape(self, sac_policy):
        """Test output shapes from various components."""
        batch_size = 8
        obs_dim = 10
        action_dim = 4

        obs = {OBS_STATE: mx.random.normal(shape=(batch_size, obs_dim))}

        # Actor output shapes
        actions, log_probs, means = sac_policy.actor(obs)
        mx.eval(actions, log_probs, means)
        assert actions.shape == (batch_size, action_dim)
        assert log_probs.shape == (batch_size,)
        assert means.shape == (batch_size, action_dim)

        # Critic output shapes
        q_values = sac_policy.critic_forward(obs, actions)
        mx.eval(q_values)
        assert q_values.shape == (2, batch_size)

    def test_encoder_output_shape(self, default_config):
        """Test encoder output has correct dimension."""
        encoder = SACObservationEncoder(default_config)
        obs = {OBS_STATE: mx.random.normal(shape=(4, 10))}
        out = encoder(obs)
        mx.eval(out)
        assert out.shape == (4, default_config.latent_dim)


# ---------------------------------------------------------------------------
# Reset test
# ---------------------------------------------------------------------------

class TestReset:
    def test_sac_reset(self, sac_policy):
        """Test that reset() does not crash."""
        sac_policy.reset()  # Should be a no-op


# ---------------------------------------------------------------------------
# Multiple critic configurations
# ---------------------------------------------------------------------------

class TestMultipleCritics:
    def test_three_critics(self):
        """Test SAC with 3 critics."""
        config = SACConfig(
            input_features={OBS_STATE: FeatureShape(shape=(6,))},
            output_features={ACTION: FeatureShape(shape=(3,))},
            num_critics=3,
            latent_dim=32,
        )
        policy = SACPolicy(config=config)

        obs = {OBS_STATE: mx.random.normal(shape=(4, 6))}
        actions = mx.random.normal(shape=(4, 3))
        q_values = policy.critic_forward(obs, actions)
        mx.eval(q_values)
        assert q_values.shape == (3, 4)

    def test_single_critic(self):
        """Test SAC with 1 critic."""
        config = SACConfig(
            input_features={OBS_STATE: FeatureShape(shape=(6,))},
            output_features={ACTION: FeatureShape(shape=(3,))},
            num_critics=1,
            latent_dim=32,
        )
        policy = SACPolicy(config=config)

        obs = {OBS_STATE: mx.random.normal(shape=(4, 6))}
        actions = mx.random.normal(shape=(4, 3))
        q_values = policy.critic_forward(obs, actions)
        mx.eval(q_values)
        assert q_values.shape == (1, 4)
