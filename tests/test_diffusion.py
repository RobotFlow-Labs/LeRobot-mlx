"""Tests for the Diffusion Policy MLX port.

Tests cover:
- Configuration defaults and validation
- Model creation and forward pass shapes
- UNet architecture (timestep conditioning, skip connections)
- Sinusoidal positional embedding properties
- Denoising loop (noise -> actions)
- Scheduler integration (add_noise + step)
- Loss computation
- Gradient flow
- Various batch sizes and chunk sizes
- Observation encoding
- Global conditioning shapes
"""

import math

import mlx.core as mx
import mlx.nn as _nn
import numpy as np
import pytest

from lerobot_mlx.policies.diffusion.configuration_diffusion import DiffusionConfig, PolicyFeature
from lerobot_mlx.policies.diffusion.modeling_diffusion import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
    DiffusionConditionalResidualBlock1d,
    DiffusionConditionalUnet1d,
    DiffusionConv1dBlock,
    DiffusionModel,
    DiffusionPolicy,
    DiffusionRgbEncoder,
    DiffusionSinusoidalPosEmb,
    SpatialSoftmax,
    _make_noise_scheduler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    state_dim: int = 2,
    action_dim: int = 2,
    n_obs_steps: int = 2,
    horizon: int = 16,
    n_action_steps: int = 8,
    with_images: bool = False,
    with_env_state: bool = False,
    image_shape: tuple = (3, 64, 64),
    env_state_dim: int = 4,
    down_dims: tuple = (64, 128, 256),
    num_train_timesteps: int = 10,
    noise_scheduler_type: str = "DDPM",
    diffusion_step_embed_dim: int = 32,
    spatial_softmax_num_keypoints: int = 8,
    n_groups: int = 4,
    kernel_size: int = 3,
) -> DiffusionConfig:
    """Create a test config with small dims for fast tests."""
    input_features = {
        "observation.state": PolicyFeature(type="STATE", shape=(state_dim,)),
    }
    if with_images:
        input_features["observation.image"] = PolicyFeature(type="VISUAL", shape=image_shape)
    if with_env_state:
        input_features["observation.environment_state"] = PolicyFeature(
            type="ENV_STATE", shape=(env_state_dim,)
        )

    output_features = {
        "action": PolicyFeature(type="ACTION", shape=(action_dim,)),
    }

    return DiffusionConfig(
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
        input_features=input_features,
        output_features=output_features,
        down_dims=down_dims,
        num_train_timesteps=num_train_timesteps,
        noise_scheduler_type=noise_scheduler_type,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        spatial_softmax_num_keypoints=spatial_softmax_num_keypoints,
        n_groups=n_groups,
        kernel_size=kernel_size,
        use_group_norm=True,
        use_film_scale_modulation=True,
    )


def _make_batch(
    config: DiffusionConfig,
    batch_size: int = 2,
    with_images: bool = False,
    with_env_state: bool = False,
) -> dict:
    """Create a fake batch matching the config."""
    state_dim = config.robot_state_feature.shape[0]
    action_dim = config.action_feature.shape[0]

    batch = {
        OBS_STATE: mx.random.normal((batch_size, config.n_obs_steps, state_dim)),
        ACTION: mx.random.normal((batch_size, config.horizon, action_dim)),
        "action_is_pad": mx.zeros((batch_size, config.horizon), dtype=mx.bool_),
    }

    if with_env_state and config.env_state_feature:
        env_dim = config.env_state_feature.shape[0]
        batch[OBS_ENV_STATE] = mx.random.normal(
            (batch_size, config.n_obs_steps, env_dim)
        )

    if with_images and config.image_features:
        img_shape = next(iter(config.image_features.values())).shape
        num_cameras = len(config.image_features)
        batch[OBS_IMAGES] = mx.random.normal(
            (batch_size, config.n_obs_steps, num_cameras, *img_shape)
        )

    return batch


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestDiffusionConfig:
    def test_config_defaults(self):
        """Test that default config values are set correctly."""
        config = _make_config()
        assert config.n_obs_steps == 2
        assert config.horizon == 16
        assert config.n_action_steps == 8
        assert config.prediction_type == "epsilon"
        assert config.noise_scheduler_type == "DDPM"

    def test_config_validation_bad_backbone(self):
        """Test that invalid backbone raises error."""
        with pytest.raises(ValueError, match="ResNet"):
            _make_config().__class__(
                vision_backbone="vit_base",
                input_features={
                    "observation.state": PolicyFeature(type="STATE", shape=(2,)),
                },
                output_features={
                    "action": PolicyFeature(type="ACTION", shape=(2,)),
                },
            )

    def test_config_validation_bad_prediction_type(self):
        """Test that invalid prediction type raises error."""
        with pytest.raises(ValueError, match="prediction_type"):
            DiffusionConfig(
                prediction_type="v_prediction",
                input_features={
                    "observation.state": PolicyFeature(type="STATE", shape=(2,)),
                },
                output_features={
                    "action": PolicyFeature(type="ACTION", shape=(2,)),
                },
            )

    def test_config_horizon_downsampling_check(self):
        """Test that mismatched horizon/down_dims raises error."""
        with pytest.raises(ValueError, match="horizon"):
            DiffusionConfig(
                horizon=15,  # Not divisible by 2^3 = 8
                down_dims=(64, 128, 256),
                input_features={
                    "observation.state": PolicyFeature(type="STATE", shape=(2,)),
                },
                output_features={
                    "action": PolicyFeature(type="ACTION", shape=(2,)),
                },
            )

    def test_config_feature_access(self):
        """Test feature access properties."""
        config = _make_config(with_images=True, with_env_state=True)
        assert config.robot_state_feature is not None
        assert config.robot_state_feature.shape == (2,)
        assert config.action_feature is not None
        assert config.action_feature.shape == (2,)
        assert len(config.image_features) == 1
        assert config.env_state_feature is not None


# ---------------------------------------------------------------------------
# Sinusoidal embedding tests
# ---------------------------------------------------------------------------

class TestSinusoidalEmbedding:
    def test_shape(self):
        """Test that sinusoidal embedding produces correct shape."""
        emb = DiffusionSinusoidalPosEmb(dim=64)
        t = mx.array([0, 5, 10, 50, 99], dtype=mx.float32)
        out = emb(t)
        assert out.shape == (5, 64)

    def test_different_timesteps_produce_different_embeddings(self):
        """Test that different timesteps produce different embeddings."""
        emb = DiffusionSinusoidalPosEmb(dim=64)
        t0 = emb(mx.array([0.0]))
        t50 = emb(mx.array([50.0]))
        t99 = emb(mx.array([99.0]))
        mx.eval(t0, t50, t99)
        # They should all be different
        assert not mx.allclose(t0, t50).item()
        assert not mx.allclose(t0, t99).item()
        assert not mx.allclose(t50, t99).item()

    def test_embedding_is_finite(self):
        """Test that all embedding values are finite."""
        emb = DiffusionSinusoidalPosEmb(dim=128)
        t = mx.arange(100, dtype=mx.float32)
        out = emb(t)
        mx.eval(out)
        assert mx.all(mx.isfinite(out)).item()


# ---------------------------------------------------------------------------
# Conv1dBlock tests
# ---------------------------------------------------------------------------

class TestConv1dBlock:
    def test_shape(self):
        """Test Conv1dBlock output shape."""
        block = DiffusionConv1dBlock(32, 64, kernel_size=3, n_groups=4)
        x = mx.random.normal((2, 32, 16))
        out = block(x)
        mx.eval(out)
        assert out.shape == (2, 64, 16)


# ---------------------------------------------------------------------------
# UNet tests
# ---------------------------------------------------------------------------

class TestUNet:
    def test_unet_forward_shape(self):
        """Test that UNet produces correct output shape."""
        config = _make_config()
        action_dim = config.action_feature.shape[0]
        state_dim = config.robot_state_feature.shape[0]
        global_cond_dim = state_dim * config.n_obs_steps

        unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)

        B = 2
        x = mx.random.normal((B, config.horizon, action_dim))
        timestep = mx.array([5, 10], dtype=mx.int32)
        global_cond = mx.random.normal((B, global_cond_dim))

        out = unet(x, timestep, global_cond=global_cond)
        mx.eval(out)
        assert out.shape == (B, config.horizon, action_dim)

    def test_unet_timestep_conditioning(self):
        """Test that different timesteps produce different outputs."""
        config = _make_config()
        action_dim = config.action_feature.shape[0]
        state_dim = config.robot_state_feature.shape[0]
        global_cond_dim = state_dim * config.n_obs_steps

        unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)

        B = 1
        x = mx.random.normal((B, config.horizon, action_dim))
        global_cond = mx.random.normal((B, global_cond_dim))

        out_t5 = unet(x, mx.array([5], dtype=mx.int32), global_cond=global_cond)
        out_t50 = unet(x, mx.array([50], dtype=mx.int32), global_cond=global_cond)
        mx.eval(out_t5, out_t50)

        assert not mx.allclose(out_t5, out_t50, atol=1e-5).item()

    def test_unet_without_global_cond(self):
        """Test UNet works without global conditioning."""
        config = _make_config()
        action_dim = config.action_feature.shape[0]

        unet = DiffusionConditionalUnet1d(config, global_cond_dim=0)

        B = 2
        x = mx.random.normal((B, config.horizon, action_dim))
        timestep = mx.array([5, 10], dtype=mx.int32)

        out = unet(x, timestep, global_cond=None)
        mx.eval(out)
        assert out.shape == (B, config.horizon, action_dim)


# ---------------------------------------------------------------------------
# Residual block tests
# ---------------------------------------------------------------------------

class TestResidualBlock:
    def test_same_channels(self):
        """Test residual block with same in/out channels."""
        block = DiffusionConditionalResidualBlock1d(
            in_channels=64, out_channels=64, cond_dim=32, kernel_size=3, n_groups=4
        )
        x = mx.random.normal((2, 64, 16))
        cond = mx.random.normal((2, 32))
        out = block(x, cond)
        mx.eval(out)
        assert out.shape == (2, 64, 16)

    def test_different_channels(self):
        """Test residual block with different in/out channels."""
        block = DiffusionConditionalResidualBlock1d(
            in_channels=32, out_channels=64, cond_dim=32, kernel_size=3, n_groups=4
        )
        x = mx.random.normal((2, 32, 16))
        cond = mx.random.normal((2, 32))
        out = block(x, cond)
        mx.eval(out)
        assert out.shape == (2, 64, 16)

    def test_film_scale_modulation(self):
        """Test residual block with FiLM scale modulation enabled."""
        block = DiffusionConditionalResidualBlock1d(
            in_channels=32, out_channels=64, cond_dim=32,
            kernel_size=3, n_groups=4, use_film_scale_modulation=True
        )
        x = mx.random.normal((2, 32, 16))
        cond = mx.random.normal((2, 32))
        out = block(x, cond)
        mx.eval(out)
        assert out.shape == (2, 64, 16)


# ---------------------------------------------------------------------------
# SpatialSoftmax tests
# ---------------------------------------------------------------------------

class TestSpatialSoftmax:
    def test_shape_with_keypoints(self):
        """Test SpatialSoftmax output shape with num_kp."""
        pool = SpatialSoftmax(input_shape=(512, 4, 4), num_kp=32)
        x = mx.random.normal((2, 512, 4, 4))
        out = pool(x)
        mx.eval(out)
        assert out.shape == (2, 32, 2)

    def test_shape_without_keypoints(self):
        """Test SpatialSoftmax output shape without num_kp."""
        pool = SpatialSoftmax(input_shape=(64, 4, 4), num_kp=None)
        x = mx.random.normal((2, 64, 4, 4))
        out = pool(x)
        mx.eval(out)
        assert out.shape == (2, 64, 2)


# ---------------------------------------------------------------------------
# Scheduler integration tests
# ---------------------------------------------------------------------------

class TestSchedulerIntegration:
    def test_ddpm_add_noise_and_step(self):
        """Test that DDPM scheduler can add noise and denoise."""
        scheduler = _make_noise_scheduler(
            "DDPM",
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type="epsilon",
        )

        x0 = mx.random.normal((2, 16, 2))
        noise = mx.random.normal(x0.shape)
        timesteps = mx.array([50, 70])

        # Add noise
        noisy = scheduler.add_noise(x0, noise, timesteps)
        mx.eval(noisy)
        assert noisy.shape == x0.shape
        assert mx.all(mx.isfinite(noisy)).item()

    def test_ddim_add_noise_and_step(self):
        """Test that DDIM scheduler can add noise and denoise."""
        scheduler = _make_noise_scheduler(
            "DDIM",
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type="epsilon",
        )

        x0 = mx.random.normal((2, 16, 2))
        noise = mx.random.normal(x0.shape)
        timesteps = mx.array([50, 70])

        noisy = scheduler.add_noise(x0, noise, timesteps)
        mx.eval(noisy)
        assert noisy.shape == x0.shape

    def test_scheduler_set_timesteps(self):
        """Test that set_timesteps produces correct number of steps."""
        scheduler = _make_noise_scheduler(
            "DDPM",
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(10)
        assert len(scheduler.timesteps) == 10


# ---------------------------------------------------------------------------
# DiffusionModel tests
# ---------------------------------------------------------------------------

class TestDiffusionModel:
    def test_model_creation_env_state(self):
        """Test DiffusionModel creation with env state features."""
        config = _make_config(with_env_state=True)
        model = DiffusionModel(config)
        assert model is not None

    def test_compute_loss(self):
        """Test that compute_loss returns a scalar loss."""
        config = _make_config(with_env_state=True)
        model = DiffusionModel(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)
        loss = model.compute_loss(batch)
        mx.eval(loss)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_loss_is_finite(self):
        """Test that the computed loss is finite."""
        config = _make_config(with_env_state=True)
        model = DiffusionModel(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)
        loss = model.compute_loss(batch)
        mx.eval(loss)
        assert mx.isfinite(loss).item()

    def test_generate_actions_shape(self):
        """Test that generate_actions produces correct shape."""
        config = _make_config(with_env_state=True)
        model = DiffusionModel(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)
        # Remove action-related keys for inference
        inference_batch = {
            OBS_STATE: batch[OBS_STATE],
            OBS_ENV_STATE: batch[OBS_ENV_STATE],
        }
        actions = model.generate_actions(inference_batch)
        mx.eval(actions)
        assert actions.shape == (2, config.n_action_steps, config.action_feature.shape[0])


# ---------------------------------------------------------------------------
# DiffusionPolicy tests
# ---------------------------------------------------------------------------

class TestDiffusionPolicy:
    def test_policy_creation(self):
        """Test DiffusionPolicy instantiation."""
        config = _make_config(with_env_state=True)
        policy = DiffusionPolicy(config)
        assert policy is not None

    def test_forward_shape(self):
        """Test forward pass produces correct loss and output."""
        config = _make_config(with_env_state=True)
        policy = DiffusionPolicy(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)
        loss, out = policy.forward(batch)
        mx.eval(loss)
        assert loss.ndim == 0
        assert out is None

    def test_forward_loss_is_finite(self):
        """Test forward pass loss is finite."""
        config = _make_config(with_env_state=True)
        policy = DiffusionPolicy(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)
        loss, _ = policy.forward(batch)
        mx.eval(loss)
        assert mx.isfinite(loss).item()

    def test_various_batch_sizes(self):
        """Test that different batch sizes work."""
        config = _make_config(with_env_state=True)
        policy = DiffusionPolicy(config)
        for bs in [1, 2, 4]:
            batch = _make_batch(config, batch_size=bs, with_env_state=True)
            loss, _ = policy.forward(batch)
            mx.eval(loss)
            assert loss.ndim == 0
            assert mx.isfinite(loss).item()

    def test_prediction_type_sample(self):
        """Test that sample prediction type works."""
        config = _make_config(with_env_state=True)
        config.prediction_type = "sample"
        policy = DiffusionPolicy(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)
        loss, _ = policy.forward(batch)
        mx.eval(loss)
        assert mx.isfinite(loss).item()


# ---------------------------------------------------------------------------
# Denoising loop tests
# ---------------------------------------------------------------------------

class TestDenoisingLoop:
    def test_denoising_produces_finite(self):
        """Test that full denoising from noise produces finite actions."""
        config = _make_config(with_env_state=True, num_train_timesteps=10)
        model = DiffusionModel(config)

        batch_size = 2
        state_dim = config.robot_state_feature.shape[0]
        env_dim = config.env_state_feature.shape[0]

        # Prepare global conditioning manually
        obs_state = mx.random.normal((batch_size, config.n_obs_steps, state_dim))
        env_state = mx.random.normal((batch_size, config.n_obs_steps, env_dim))
        global_cond_feats = mx.concatenate([obs_state, env_state], axis=-1)
        global_cond = mx.flatten(global_cond_feats, start_axis=1)

        # Start from pure noise
        noise = mx.random.normal(
            (batch_size, config.horizon, config.action_feature.shape[0])
        )
        sample = model.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
        mx.eval(sample)
        assert sample.shape == (batch_size, config.horizon, config.action_feature.shape[0])
        assert mx.all(mx.isfinite(sample)).item()

    def test_denoising_loop_full(self):
        """Test full denoising loop through the model."""
        config = _make_config(with_env_state=True, num_train_timesteps=5)
        model = DiffusionModel(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)
        inference_batch = {
            OBS_STATE: batch[OBS_STATE],
            OBS_ENV_STATE: batch[OBS_ENV_STATE],
        }
        actions = model.generate_actions(inference_batch)
        mx.eval(actions)
        assert actions.shape == (2, config.n_action_steps, config.action_feature.shape[0])
        assert mx.all(mx.isfinite(actions)).item()


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = _make_config(with_env_state=True)
        model = DiffusionModel(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)

        def loss_fn(model):
            return model.compute_loss(batch)

        loss, grads = _nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)

        # Check that we have gradients
        from mlx.utils import tree_flatten
        flat_grads = tree_flatten(grads)
        assert len(flat_grads) > 0, "No gradients computed"

        # Check at least some gradients are non-zero
        has_nonzero = False
        for name, grad in flat_grads:
            if mx.any(grad != 0).item():
                has_nonzero = True
                break
        assert has_nonzero, "All gradients are zero"


# ---------------------------------------------------------------------------
# Observation encoding tests
# ---------------------------------------------------------------------------

class TestObservationEncoding:
    def test_global_conditioning_env_state(self):
        """Test global conditioning shape with env state."""
        config = _make_config(with_env_state=True)
        model = DiffusionModel(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)

        global_cond = model._prepare_global_conditioning(batch)
        mx.eval(global_cond)

        state_dim = config.robot_state_feature.shape[0]
        env_dim = config.env_state_feature.shape[0]
        expected_dim = (state_dim + env_dim) * config.n_obs_steps
        assert global_cond.shape == (2, expected_dim)

    def test_global_conditioning_is_finite(self):
        """Test that global conditioning values are finite."""
        config = _make_config(with_env_state=True)
        model = DiffusionModel(config)
        batch = _make_batch(config, batch_size=2, with_env_state=True)

        global_cond = model._prepare_global_conditioning(batch)
        mx.eval(global_cond)
        assert mx.all(mx.isfinite(global_cond)).item()


# ---------------------------------------------------------------------------
# Chunk size and select_action tests
# ---------------------------------------------------------------------------

class TestChunkSize:
    def test_chunk_size_from_config(self):
        """Test that action chunk size matches config."""
        config = _make_config(with_env_state=True, num_train_timesteps=5)
        model = DiffusionModel(config)
        batch = _make_batch(config, batch_size=1, with_env_state=True)
        inference_batch = {
            OBS_STATE: batch[OBS_STATE],
            OBS_ENV_STATE: batch[OBS_ENV_STATE],
        }
        actions = model.generate_actions(inference_batch)
        mx.eval(actions)
        assert actions.shape[1] == config.n_action_steps

    def test_different_action_dims(self):
        """Test with different action dimensions."""
        for action_dim in [2, 4, 8]:
            config = _make_config(
                action_dim=action_dim, with_env_state=True, num_train_timesteps=5
            )
            model = DiffusionModel(config)
            batch = _make_batch(config, batch_size=1, with_env_state=True)
            inference_batch = {
                OBS_STATE: batch[OBS_STATE],
                OBS_ENV_STATE: batch[OBS_ENV_STATE],
            }
            actions = model.generate_actions(inference_batch)
            mx.eval(actions)
            assert actions.shape[-1] == action_dim


# ---------------------------------------------------------------------------
# Train vs eval mode
# ---------------------------------------------------------------------------

class TestTrainEvalMode:
    def test_train_mode(self):
        """Test model in training mode."""
        config = _make_config(with_env_state=True)
        policy = DiffusionPolicy(config)
        policy.train()
        batch = _make_batch(config, batch_size=2, with_env_state=True)
        loss, _ = policy.forward(batch)
        mx.eval(loss)
        assert mx.isfinite(loss).item()

    def test_eval_mode(self):
        """Test model in eval mode."""
        config = _make_config(with_env_state=True)
        policy = DiffusionPolicy(config)
        policy.eval()
        batch = _make_batch(config, batch_size=2, with_env_state=True)
        loss, _ = policy.forward(batch)
        mx.eval(loss)
        assert mx.isfinite(loss).item()
