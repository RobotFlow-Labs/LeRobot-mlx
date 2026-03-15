"""Comprehensive tests for ACT (Action Chunking Transformer) policy — MLX port.

Tests configuration, model creation, forward pass, loss computation,
VAE behavior, gradient flow, and various batch sizes.
"""

import math

import mlx.core as mx
import mlx.nn as _nn
import numpy as np
import pytest

from lerobot_mlx.policies.act.configuration_act import (
    ACTConfig,
    ACTION,
    FeatureType,
    NormalizationMode,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
    PolicyFeature,
)
from lerobot_mlx.policies.act.modeling_act import (
    ACT,
    ACTDecoder,
    ACTDecoderLayer,
    ACTEncoder,
    ACTEncoderLayer,
    ACTPolicy,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
    _get_activation_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_only_config(
    state_dim: int = 14,
    action_dim: int = 14,
    chunk_size: int = 10,
    dim_model: int = 64,
    n_heads: int = 4,
    dim_feedforward: int = 128,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 1,
    n_vae_encoder_layers: int = 2,
    latent_dim: int = 8,
    use_vae: bool = True,
    dropout: float = 0.0,
) -> ACTConfig:
    """Create a minimal config with state-only observations (no images)."""
    return ACTConfig(
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(state_dim,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        dim_model=dim_model,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_vae_encoder_layers=n_vae_encoder_layers,
        latent_dim=latent_dim,
        use_vae=use_vae,
        dropout=dropout,
        pretrained_backbone_weights=None,
    )


def _make_image_config(
    state_dim: int = 14,
    action_dim: int = 14,
    chunk_size: int = 10,
    dim_model: int = 64,
    n_heads: int = 4,
    dim_feedforward: int = 128,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 1,
    n_vae_encoder_layers: int = 2,
    latent_dim: int = 8,
    use_vae: bool = True,
    dropout: float = 0.0,
    image_size: tuple[int, int] = (64, 64),
) -> ACTConfig:
    """Create a config with image observations."""
    return ACTConfig(
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.images.top": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, *image_size)
            ),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        dim_model=dim_model,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_vae_encoder_layers=n_vae_encoder_layers,
        latent_dim=latent_dim,
        use_vae=use_vae,
        dropout=dropout,
        pretrained_backbone_weights=None,
    )


def _make_batch(config: ACTConfig, batch_size: int = 2, training: bool = True) -> dict[str, mx.array]:
    """Create a random batch matching the config's feature specifications."""
    batch = {}

    if config.robot_state_feature:
        state_dim = config.robot_state_feature.shape[0]
        batch[OBS_STATE] = mx.random.normal((batch_size, state_dim))

    if config.env_state_feature:
        env_dim = config.env_state_feature.shape[0]
        batch[OBS_ENV_STATE] = mx.random.normal((batch_size, env_dim))

    if config.image_features:
        for key, ft in config.image_features.items():
            c, h, w = ft.shape
            batch[key] = mx.random.normal((batch_size, c, h, w))

    if training and config.action_feature:
        action_dim = config.action_feature.shape[0]
        batch[ACTION] = mx.random.normal((batch_size, config.chunk_size, action_dim))
        batch["action_is_pad"] = mx.zeros((batch_size, config.chunk_size), dtype=mx.bool_)

    return batch


# ===========================================================================
# Config Tests
# ===========================================================================

class TestACTConfig:
    def test_config_defaults(self):
        """Config creates with default values."""
        config = ACTConfig(
            input_features={
                OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(14,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
            },
        )
        assert config.chunk_size == 100
        assert config.dim_model == 512
        assert config.n_heads == 8
        assert config.use_vae is True
        assert config.latent_dim == 32
        assert config.kl_weight == 10.0

    def test_config_all_fields_accessible(self):
        """All config fields are accessible."""
        config = _make_state_only_config()
        assert config.chunk_size == 10
        assert config.dim_model == 64
        assert config.n_heads == 4
        assert config.dim_feedforward == 128
        assert config.n_encoder_layers == 2
        assert config.n_decoder_layers == 1
        assert config.latent_dim == 8
        assert config.dropout == 0.0
        assert config.feedforward_activation == "relu"
        assert config.pre_norm is False
        assert config.vision_backbone == "resnet18"

    def test_config_validation_backbone(self):
        """Config rejects non-resnet backbone."""
        with pytest.raises(ValueError, match="ResNet"):
            ACTConfig(
                vision_backbone="vgg16",
                input_features={OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(14,))},
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
            )

    def test_config_validation_action_steps(self):
        """Config rejects n_action_steps > chunk_size."""
        with pytest.raises(ValueError, match="chunk size"):
            ACTConfig(
                chunk_size=10,
                n_action_steps=20,
                input_features={OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(14,))},
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
            )

    def test_config_validation_obs_steps(self):
        """Config rejects n_obs_steps != 1."""
        with pytest.raises(ValueError, match="Multiple observation"):
            ACTConfig(
                n_obs_steps=2,
                input_features={OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(14,))},
                output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
            )

    def test_config_feature_properties(self):
        """Feature property accessors work correctly."""
        config = _make_state_only_config()
        assert config.robot_state_feature is not None
        assert config.robot_state_feature.shape == (14,)
        assert config.env_state_feature is not None
        assert config.action_feature is not None
        assert config.action_feature.shape == (14,)

    def test_config_image_features(self):
        """Image feature detection works."""
        config = _make_image_config()
        assert len(config.image_features) == 1
        assert "observation.images.top" in config.image_features

    def test_config_normalization_mapping(self):
        """Default normalization mapping is present."""
        config = _make_state_only_config()
        assert "STATE" in config.normalization_mapping
        assert config.normalization_mapping["STATE"] == NormalizationMode.MEAN_STD


# ===========================================================================
# Model Creation Tests
# ===========================================================================

class TestACTModelCreation:
    def test_model_creation_state_only(self):
        """ACTPolicy with state-only config doesn't crash."""
        config = _make_state_only_config()
        model = ACTPolicy(config)
        assert model is not None

    def test_model_creation_with_images(self):
        """ACTPolicy with image config doesn't crash."""
        config = _make_image_config()
        model = ACTPolicy(config)
        assert model is not None

    def test_model_creation_no_vae(self):
        """ACTPolicy without VAE creates successfully."""
        config = _make_state_only_config(use_vae=False)
        model = ACTPolicy(config)
        assert model is not None

    def test_parameter_count(self):
        """Model has a nonzero number of parameters."""
        config = _make_state_only_config()
        model = ACTPolicy(config)
        n_params = model.num_parameters()
        assert n_params > 0
        # Sanity: a small model with dim_model=64 should have at least a few thousand params
        assert n_params > 1000


# ===========================================================================
# Forward Pass Tests
# ===========================================================================

class TestACTForward:
    def test_forward_state_only(self):
        """Forward pass with state-only observations produces correct shape."""
        config = _make_state_only_config(chunk_size=10, use_vae=True)
        model = ACTPolicy(config)
        model.train()
        batch = _make_batch(config, batch_size=2, training=True)
        loss, loss_dict = model.forward(batch)
        mx.eval(loss)
        assert loss.ndim == 0  # scalar
        assert np.isfinite(loss.item())

    def test_forward_with_images(self):
        """Forward pass with image observations produces correct shape."""
        config = _make_image_config(chunk_size=5, image_size=(64, 64))
        model = ACTPolicy(config)
        model.train()
        batch = _make_batch(config, batch_size=2, training=True)
        loss, loss_dict = model.forward(batch)
        mx.eval(loss)
        assert loss.ndim == 0
        assert np.isfinite(loss.item())

    def test_forward_shape_inference(self):
        """In eval mode, model produces action chunks of the right shape."""
        config = _make_state_only_config(chunk_size=10, use_vae=True)
        model = ACTPolicy(config)
        model.eval()
        batch = _make_batch(config, batch_size=2, training=False)
        actions, (mu, log_sigma_x2) = model.model(batch)
        mx.eval(actions)
        assert actions.shape == (2, 10, 14)
        # In eval mode with no actions, mu and log_sigma_x2 should be None
        assert mu is None
        assert log_sigma_x2 is None

    def test_forward_no_vae(self):
        """Forward pass without VAE works."""
        config = _make_state_only_config(use_vae=False)
        model = ACTPolicy(config)
        model.train()
        batch = _make_batch(config, batch_size=2, training=True)
        loss, loss_dict = model.forward(batch)
        mx.eval(loss)
        assert np.isfinite(loss.item())
        assert "l1_loss" in loss_dict
        assert "kld_loss" not in loss_dict

    def test_chunk_size_respected(self):
        """Model produces the correct number of action steps."""
        for cs in [5, 10, 20]:
            config = _make_state_only_config(chunk_size=cs)
            model = ACTPolicy(config)
            model.eval()
            batch = _make_batch(config, batch_size=1, training=False)
            actions, _ = model.model(batch)
            mx.eval(actions)
            assert actions.shape[1] == cs


# ===========================================================================
# Loss Computation Tests
# ===========================================================================

class TestACTLoss:
    def test_loss_is_finite_scalar(self):
        """compute_loss returns a finite scalar."""
        config = _make_state_only_config()
        model = ACTPolicy(config)
        model.train()
        batch = _make_batch(config, batch_size=4, training=True)
        loss, loss_dict = model.forward(batch)
        mx.eval(loss)
        assert loss.ndim == 0
        assert np.isfinite(loss.item())

    def test_loss_components(self):
        """Loss dict has l1_loss and kld_loss when using VAE."""
        config = _make_state_only_config(use_vae=True)
        model = ACTPolicy(config)
        model.train()
        batch = _make_batch(config, batch_size=2, training=True)
        loss, loss_dict = model.forward(batch)
        mx.eval(loss)
        assert "l1_loss" in loss_dict
        assert "kld_loss" in loss_dict
        assert np.isfinite(loss_dict["l1_loss"])
        assert np.isfinite(loss_dict["kld_loss"])

    def test_kl_nonnegative(self):
        """KL divergence is non-negative."""
        config = _make_state_only_config(use_vae=True)
        model = ACTPolicy(config)
        model.train()
        batch = _make_batch(config, batch_size=4, training=True)
        loss, loss_dict = model.forward(batch)
        mx.eval(loss)
        # KL divergence should be >= 0 (up to numerical precision)
        assert loss_dict["kld_loss"] >= -1e-6

    def test_l1_loss_nonnegative(self):
        """L1 loss is non-negative."""
        config = _make_state_only_config()
        model = ACTPolicy(config)
        model.train()
        batch = _make_batch(config, batch_size=2, training=True)
        loss, loss_dict = model.forward(batch)
        mx.eval(loss)
        assert loss_dict["l1_loss"] >= 0


# ===========================================================================
# VAE Behavior Tests
# ===========================================================================

class TestACTVAE:
    def test_eval_uses_prior(self):
        """In eval mode, z is sampled from prior N(0, I) — mu and log_sigma_x2 are None."""
        config = _make_state_only_config(use_vae=True)
        model = ACTPolicy(config)
        model.eval()
        batch = _make_batch(config, batch_size=2, training=False)
        actions, (mu, log_sigma_x2) = model.model(batch)
        mx.eval(actions)
        assert mu is None
        assert log_sigma_x2 is None

    def test_train_uses_posterior(self):
        """In train mode, z is sampled from posterior — mu and log_sigma_x2 are returned."""
        config = _make_state_only_config(use_vae=True)
        model = ACTPolicy(config)
        model.train()
        batch = _make_batch(config, batch_size=2, training=True)
        actions, (mu, log_sigma_x2) = model.model(batch)
        mx.eval(actions, mu, log_sigma_x2)
        assert mu is not None
        assert log_sigma_x2 is not None
        assert mu.shape == (2, config.latent_dim)
        assert log_sigma_x2.shape == (2, config.latent_dim)


# ===========================================================================
# Select Action Tests
# ===========================================================================

class TestACTSelectAction:
    def test_select_action_returns_single_action(self):
        """select_action returns a single action (batch, action_dim)."""
        config = _make_state_only_config(chunk_size=10)
        model = ACTPolicy(config)
        model.eval()
        batch = _make_batch(config, batch_size=1, training=False)
        action = model.select_action(batch)
        mx.eval(action)
        assert action.shape == (1, 14)

    def test_select_action_queue(self):
        """select_action uses the action queue for multiple calls."""
        config = _make_state_only_config(chunk_size=10)
        model = ACTPolicy(config)
        model.eval()
        batch = _make_batch(config, batch_size=1, training=False)

        # First call fills the queue
        action1 = model.select_action(batch)
        mx.eval(action1)
        assert action1.shape == (1, 14)

        # Subsequent calls pop from the queue
        action2 = model.select_action(batch)
        mx.eval(action2)
        assert action2.shape == (1, 14)


# ===========================================================================
# Architecture Component Tests
# ===========================================================================

class TestACTComponents:
    def test_encoder_layer(self):
        """ACTEncoderLayer processes input correctly."""
        config = _make_state_only_config()
        layer = ACTEncoderLayer(config)
        x = mx.random.normal((2, 5, config.dim_model))
        out = layer(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_decoder_layer(self):
        """ACTDecoderLayer processes input correctly."""
        config = _make_state_only_config()
        layer = ACTDecoderLayer(config)
        x = mx.random.normal((2, 10, config.dim_model))
        encoder_out = mx.random.normal((2, 5, config.dim_model))
        out = layer(x, encoder_out)
        mx.eval(out)
        assert out.shape == x.shape

    def test_encoder(self):
        """ACTEncoder stack processes input correctly."""
        config = _make_state_only_config(n_encoder_layers=3)
        encoder = ACTEncoder(config)
        x = mx.random.normal((2, 5, config.dim_model))
        out = encoder(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_decoder(self):
        """ACTDecoder stack processes input correctly."""
        config = _make_state_only_config(n_decoder_layers=2)
        decoder = ACTDecoder(config)
        x = mx.random.normal((2, 10, config.dim_model))
        encoder_out = mx.random.normal((2, 5, config.dim_model))
        out = decoder(x, encoder_out)
        mx.eval(out)
        assert out.shape == x.shape

    def test_sinusoidal_pos_embedding_1d(self):
        """1D sinusoidal position embedding has correct shape and values."""
        emb = create_sinusoidal_pos_embedding(10, 64)
        mx.eval(emb)
        assert emb.shape == (10, 64)
        # First position should have sin(0)=0 for even dims
        assert abs(emb[0, 0].item()) < 1e-5

    def test_sinusoidal_pos_embedding_2d(self):
        """2D sinusoidal position embedding has correct shape."""
        embed = ACTSinusoidalPositionEmbedding2d(32)
        x = mx.random.normal((2, 64, 8, 8))  # NCHW
        out = embed(x)
        mx.eval(out)
        assert out.shape == (1, 64, 8, 8)  # (1, C, H, W)

    def test_resnet_backbone(self):
        """Vision backbone extracts features with correct output channels."""
        from lerobot_mlx.policies.act.modeling_act import _ACTBackbone
        config = _make_image_config(image_size=(64, 64))
        backbone = _ACTBackbone(config)
        img = mx.random.normal((2, 3, 64, 64))
        features = backbone(img)
        mx.eval(features)
        # ResNet18 layer4 outputs 512 channels
        assert features.shape[0] == 2
        assert features.shape[1] == 512
        # Spatial dims should be reduced
        assert features.shape[2] < 64
        assert features.shape[3] < 64


# ===========================================================================
# Batch Size Tests
# ===========================================================================

class TestACTBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_various_batch_sizes(self, batch_size):
        """Model works with different batch sizes."""
        config = _make_state_only_config(chunk_size=5)
        model = ACTPolicy(config)
        model.eval()
        batch = _make_batch(config, batch_size=batch_size, training=False)
        actions, _ = model.model(batch)
        mx.eval(actions)
        assert actions.shape == (batch_size, 5, 14)


# ===========================================================================
# Gradient Flow Tests
# ===========================================================================

class TestACTGradients:
    def test_gradient_flow(self):
        """Gradients flow through the model."""
        config = _make_state_only_config(chunk_size=5, use_vae=True)
        model = ACTPolicy(config)
        model.train()
        batch = _make_batch(config, batch_size=2, training=True)

        def loss_fn(model):
            loss, _ = model.forward(batch)
            return loss

        loss_and_grad = _nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model)
        mx.eval(loss, grads)

        assert np.isfinite(loss.item())
        # Check that at least some gradients are non-zero
        from mlx.utils import tree_flatten
        flat_grads = tree_flatten(grads)
        has_nonzero = any(mx.any(mx.abs(g) > 0).item() for _, g in flat_grads if isinstance(g, mx.array))
        assert has_nonzero, "All gradients are zero — no gradient flow"


# ===========================================================================
# Utility Tests
# ===========================================================================

class TestACTUtilities:
    def test_get_activation_fn_relu(self):
        """Activation function factory returns relu."""
        fn = _get_activation_fn("relu")
        x = mx.array([-1.0, 0.0, 1.0])
        out = fn(x)
        mx.eval(out)
        np.testing.assert_allclose(np.array(out), [0.0, 0.0, 1.0])

    def test_get_activation_fn_gelu(self):
        """Activation function factory returns gelu."""
        fn = _get_activation_fn("gelu")
        x = mx.array([1.0])
        out = fn(x)
        mx.eval(out)
        assert out.item() > 0.8  # gelu(1) ~ 0.841

    def test_get_activation_fn_invalid(self):
        """Activation function factory raises on invalid name."""
        with pytest.raises(RuntimeError):
            _get_activation_fn("invalid_activation")

    def test_temporal_ensembler(self):
        """Temporal ensembler produces correct output shape."""
        ensembler = ACTTemporalEnsembler(0.01, chunk_size=5)
        actions = mx.random.normal((1, 5, 3))
        action = ensembler.update(actions)
        mx.eval(action)
        assert action.shape == (1, 3)

        # Second call
        actions2 = mx.random.normal((1, 5, 3))
        action2 = ensembler.update(actions2)
        mx.eval(action2)
        assert action2.shape == (1, 3)

    def test_reset_clears_queue(self):
        """Policy reset clears the action queue."""
        config = _make_state_only_config(chunk_size=10)
        model = ACTPolicy(config)
        model.eval()
        batch = _make_batch(config, batch_size=1, training=False)

        # Fill the queue
        model.select_action(batch)
        assert len(model._action_queue) > 0

        # Reset clears it
        model.reset()
        assert len(model._action_queue) == 0
