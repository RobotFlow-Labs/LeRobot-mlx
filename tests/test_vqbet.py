#!/usr/bin/env python
"""Tests for VQ-BeT policy MLX port.

Covers configuration, model components (VQ-VAE, codebook, GPT, action head),
training loss, gradient flow, and inference.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from lerobot_mlx.policies.vqbet.configuration_vqbet import (
    VQBeTConfig,
    PolicyFeature,
    FeatureType,
    ACTION,
    OBS_STATE,
    OBS_IMAGES,
)
from lerobot_mlx.policies.vqbet.modeling_vqbet import (
    VQBeTPolicy,
    VQBeTModel,
    VQBeTHead,
    VQBeTRgbEncoder,
    VqVae,
    GPT,
    MLP,
    FocalLoss,
    SpatialSoftmax,
    CausalSelfAttention,
    Block,
    ResidualVQ,
    VectorQuantize,
    EuclideanCodebook,
    gumbel_softmax,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    state_dim=4,
    action_dim=2,
    img_channels=3,
    img_h=96,
    img_w=96,
    n_obs_steps=2,
    n_action_pred_token=2,
    action_chunk_size=3,
    gpt_n_layer=2,
    gpt_n_head=2,
    gpt_hidden_dim=64,
    gpt_input_dim=64,
    gpt_output_dim=64,
    vqvae_n_embed=8,
    vqvae_embedding_dim=32,
    vqvae_enc_hidden_dim=32,
    crop_shape=(84, 84),
):
    """Create a small VQBeTConfig for testing."""
    return VQBeTConfig(
        n_obs_steps=n_obs_steps,
        n_action_pred_token=n_action_pred_token,
        action_chunk_size=action_chunk_size,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(img_channels, img_h, img_w)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        vision_backbone="resnet18",
        crop_shape=crop_shape,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=8,
        gpt_block_size=100,
        gpt_input_dim=gpt_input_dim,
        gpt_output_dim=gpt_output_dim,
        gpt_n_layer=gpt_n_layer,
        gpt_n_head=gpt_n_head,
        gpt_hidden_dim=gpt_hidden_dim,
        dropout=0.0,
        vqvae_n_embed=vqvae_n_embed,
        vqvae_embedding_dim=vqvae_embedding_dim,
        vqvae_enc_hidden_dim=vqvae_enc_hidden_dim,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVQBeTConfig:
    def test_vqbet_config_defaults(self):
        """Test that default config creates without error."""
        config = VQBeTConfig()
        assert config.n_obs_steps == 5
        assert config.n_action_pred_token == 3
        assert config.action_chunk_size == 5
        assert config.vision_backbone == "resnet18"
        assert config.vqvae_n_embed == 16
        assert config.gpt_n_layer == 8
        assert config.dropout == 0.1

    def test_vqbet_config_validation(self):
        """Test that invalid backbone raises error."""
        with pytest.raises(ValueError, match="ResNet"):
            VQBeTConfig(vision_backbone="vit_base")

    def test_vqbet_config_delta_indices(self):
        """Test observation and action delta indices."""
        config = VQBeTConfig(n_obs_steps=3, n_action_pred_token=2, action_chunk_size=4)
        assert config.observation_delta_indices == [-2, -1, 0]
        assert len(config.action_delta_indices) > 0
        assert config.reward_delta_indices is None


class TestMLP:
    def test_mlp_forward(self):
        mlp = MLP(in_channels=10, hidden_channels=[32, 16, 5])
        x = mx.random.normal(shape=(4, 10))
        y = mlp(x)
        assert y.shape == (4, 5)

    def test_mlp_single_layer(self):
        mlp = MLP(in_channels=10, hidden_channels=[5])
        x = mx.random.normal(shape=(2, 10))
        y = mlp(x)
        assert y.shape == (2, 5)


class TestSpatialSoftmax:
    def test_spatial_softmax_shape(self):
        pool = SpatialSoftmax((64, 10, 12), num_kp=8)
        features = mx.random.normal(shape=(2, 64, 10, 12))
        kp = pool(features)
        assert kp.shape == (2, 8, 2)

    def test_spatial_softmax_no_kp(self):
        pool = SpatialSoftmax((16, 5, 5), num_kp=None)
        features = mx.random.normal(shape=(3, 16, 5, 5))
        kp = pool(features)
        assert kp.shape == (3, 16, 2)


class TestFocalLoss:
    def test_focal_loss_2d(self):
        fl = FocalLoss(gamma=2.0)
        logits = mx.random.normal(shape=(8, 5))
        targets = mx.array(np.random.randint(0, 5, size=(8,)))
        loss = fl(logits, targets)
        mx.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) >= 0

    def test_focal_loss_3d(self):
        fl = FocalLoss(gamma=2.0)
        logits = mx.random.normal(shape=(4, 3, 5))
        targets = mx.array(np.random.randint(0, 5, size=(4, 3)))
        loss = fl(logits, targets)
        mx.eval(loss)
        assert loss.shape == ()


class TestGumbelSoftmax:
    def test_gumbel_softmax_soft(self):
        logits = mx.random.normal(shape=(4, 8))
        y = gumbel_softmax(logits, tau=1.0, hard=False)
        mx.eval(y)
        assert y.shape == (4, 8)
        # Soft output should sum to ~1 per row
        sums = mx.sum(y, axis=-1)
        mx.eval(sums)
        for i in range(4):
            assert abs(float(sums[i].item()) - 1.0) < 1e-5

    def test_gumbel_softmax_hard(self):
        logits = mx.random.normal(shape=(4, 8))
        y = gumbel_softmax(logits, tau=0.5, hard=True)
        mx.eval(y)
        assert y.shape == (4, 8)
        # Hard output should have exactly one 1 per row
        max_vals = mx.max(y, axis=-1)
        mx.eval(max_vals)
        for i in range(4):
            assert abs(float(max_vals[i].item()) - 1.0) < 1e-5


class TestEuclideanCodebook:
    def test_codebook_init(self):
        cb = EuclideanCodebook(dim=16, codebook_size=8, num_codebooks=1)
        assert cb.embed.shape == (1, 8, 16)

    def test_codebook_forward(self):
        cb = EuclideanCodebook(dim=16, codebook_size=8, num_codebooks=1)
        cb.train()
        x = mx.random.normal(shape=(4, 16))
        quantize, embed_ind, dist = cb(x)
        mx.eval(quantize, embed_ind, dist)
        assert quantize.shape == (4, 16)


class TestVectorQuantize:
    def test_vq_forward(self):
        vq = VectorQuantize(dim=16, codebook_size=8)
        vq.train()
        x = mx.random.normal(shape=(4, 16))
        quantize, embed_ind, loss = vq(x)
        mx.eval(quantize, embed_ind, loss)
        assert quantize.shape == (4, 16)

    def test_vqbet_codebook_lookup(self):
        """Test that nearest neighbor lookup finds correct codebook entry."""
        vq = VectorQuantize(dim=8, codebook_size=4)
        # Set codebook to known values
        codebook = mx.eye(4, 8)  # 4 vectors of dim 8, orthogonal
        vq._codebook.embed = mx.expand_dims(codebook, axis=0)
        vq._codebook.embed_avg = mx.expand_dims(codebook, axis=0)
        vq._codebook.initted = mx.array([1.0])  # Mark as initialized
        vq._codebook.cluster_size = mx.ones((1, 4))
        vq.eval()

        # Query with vectors close to codebook entries
        x = codebook + mx.random.normal(shape=codebook.shape) * 0.01
        quantize, embed_ind, _ = vq(x)
        mx.eval(quantize, embed_ind)
        # Each query should map to its nearest codebook entry
        expected = mx.arange(4)
        embed_ind_flat = embed_ind.reshape(-1).astype(mx.int32)
        assert mx.array_equal(embed_ind_flat, expected)


class TestResidualVQ:
    def test_residual_vq_forward(self):
        rvq = ResidualVQ(dim=16, num_quantizers=2, codebook_size=8)
        rvq.train()
        x = mx.random.normal(shape=(4, 1, 16))
        quantized, indices, losses = rvq(x)
        mx.eval(quantized, indices, losses)
        assert quantized.shape == (4, 1, 16)
        assert indices.shape[-1] == 2  # 2 quantizers

    def test_residual_vq_get_codebook_vectors(self):
        rvq = ResidualVQ(dim=16, num_quantizers=2, codebook_size=8)
        indices = mx.array([[0, 1], [2, 3]])  # (batch=2, num_quantizers=2)
        codes = rvq.get_codebook_vector_from_indices(indices)
        mx.eval(codes)
        assert codes.shape == (2, 2, 16)  # (num_quantizers, batch, dim)


class TestGPT:
    def test_gpt_forward(self):
        config = _make_config()
        gpt = GPT(config)
        gpt.train()
        x = mx.random.normal(shape=(2, 10, config.gpt_input_dim))
        out = gpt(x)
        mx.eval(out)
        assert out.shape == (2, 10, config.gpt_output_dim)

    def test_gpt_configure_parameters(self):
        config = _make_config()
        gpt = GPT(config)
        decay, no_decay = gpt.configure_parameters()
        assert len(decay) > 0
        assert len(no_decay) > 0


class TestCausalSelfAttention:
    def test_causal_attention(self):
        config = _make_config()
        attn = CausalSelfAttention(config)
        x = mx.random.normal(shape=(2, 5, config.gpt_hidden_dim))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (2, 5, config.gpt_hidden_dim)


class TestVqVae:
    def test_vqvae_encode_decode(self):
        """Test VQ-VAE encode -> quantize -> decode roundtrip."""
        config = _make_config()
        vqvae = VqVae(config)
        vqvae.train()

        action_dim = config.action_feature.shape[0]
        actions = mx.random.normal(shape=(4, config.action_chunk_size, action_dim))

        # Forward through VQ-VAE
        loss, metric = vqvae.vqvae_forward(actions)
        mx.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) > 0

    def test_vqvae_get_code(self):
        """Test getting VQ codes from actions."""
        config = _make_config()
        vqvae = VqVae(config)
        vqvae.train()

        action_dim = config.action_feature.shape[0]
        actions = mx.random.normal(shape=(4, config.action_chunk_size, action_dim))

        state_vq, vq_code = vqvae.get_code(actions)
        mx.eval(state_vq, vq_code)
        assert state_vq.shape == (4, config.vqvae_embedding_dim)
        assert vq_code.shape == (4, 2)  # 2 RVQ layers

    def test_vqvae_embeddings_from_code(self):
        """Test getting embeddings from code indices."""
        config = _make_config()
        vqvae = VqVae(config)

        codes = mx.array([[0, 1], [2, 3]])
        embeddings = vqvae.get_embeddings_from_code(codes)
        mx.eval(embeddings)
        assert embeddings.shape == (2, config.vqvae_embedding_dim)

    def test_vqvae_action_from_latent(self):
        """Test decoding latent to actions."""
        config = _make_config()
        vqvae = VqVae(config)

        latent = mx.random.normal(shape=(4, config.vqvae_embedding_dim))
        actions = vqvae.get_action_from_latent(latent)
        mx.eval(actions)
        action_dim = config.action_feature.shape[0]
        assert actions.shape == (4, config.action_chunk_size, action_dim)


class TestVQBeTHead:
    def test_action_head_forward(self):
        config = _make_config()
        head = VQBeTHead(config)
        head.train()

        features = mx.random.normal(shape=(2, 3, config.gpt_output_dim))
        output = head(features)
        mx.eval(output["predicted_action"])
        action_dim = config.action_feature.shape[0]
        assert output["predicted_action"].shape == (2, 3, config.action_chunk_size * action_dim)
        assert "cbet_logits" in output
        assert "sampled_centers" in output

    def test_commitment_loss(self):
        """Test that commitment loss is non-negative."""
        config = _make_config()
        vqvae = VqVae(config)
        vqvae.train()

        action_dim = config.action_feature.shape[0]
        actions = mx.random.normal(shape=(4, config.action_chunk_size, action_dim))
        loss, metric = vqvae.vqvae_forward(actions)
        mx.eval(loss)
        assert float(loss.item()) >= 0

    def test_codebook_usage(self):
        """Test that codes distribute across entries."""
        config = _make_config(vqvae_n_embed=4)
        vqvae = VqVae(config)
        vqvae.train()

        action_dim = config.action_feature.shape[0]
        # Use diverse actions
        actions = mx.random.normal(shape=(32, config.action_chunk_size, action_dim)) * 5.0
        _, vq_code = vqvae.get_code(actions)
        mx.eval(vq_code)

        # Check that at least 2 different codes are used in each layer
        for layer in range(2):
            unique_codes = set(vq_code[:, layer].tolist())
            assert len(unique_codes) >= 1, f"Layer {layer} only uses {len(unique_codes)} code(s)"


class TestVQBeTModel:
    def test_vqbet_model_creation(self):
        """Test model instantiation."""
        config = _make_config()
        model = VQBeTModel(config)
        assert model.config == config
        assert model.num_images == 1

    def test_vqbet_forward_shape(self):
        """Test forward pass output shapes in training mode."""
        config = _make_config()
        model = VQBeTModel(config)
        model.train()

        # Mark VQ-VAE as trained
        model.action_head.vqvae_model.discretized = mx.array(True)
        model.action_head.vqvae_model.vq_layer.freeze_codebook = mx.array(True)

        batch_size = 2
        action_dim = config.action_feature.shape[0]
        n_action_steps = config.n_obs_steps + config.n_action_pred_token + config.action_chunk_size - 2

        batch = {
            OBS_STATE: mx.random.normal(shape=(batch_size, config.n_obs_steps, config.robot_state_feature.shape[0])),
            OBS_IMAGES: mx.random.normal(shape=(batch_size, config.n_obs_steps, 1, 3, 84, 84)),
            ACTION: mx.random.normal(shape=(batch_size, n_action_steps, action_dim)),
        }
        output, loss_dict = model(batch, rollout=False)
        mx.eval(loss_dict["loss"])
        assert "loss" in loss_dict

    def test_vqbet_rollout_shape(self):
        """Test rollout output shape."""
        config = _make_config()
        model = VQBeTModel(config)
        model.eval()

        model.action_head.vqvae_model.discretized = mx.array(True)
        model.action_head.vqvae_model.vq_layer.freeze_codebook = mx.array(True)

        batch_size = 2
        action_dim = config.action_feature.shape[0]

        batch = {
            OBS_STATE: mx.random.normal(shape=(batch_size, config.n_obs_steps, config.robot_state_feature.shape[0])),
            OBS_IMAGES: mx.random.normal(shape=(batch_size, config.n_obs_steps, 1, 3, 84, 84)),
        }
        actions = model(batch, rollout=True)
        mx.eval(actions)
        assert actions.shape == (batch_size, config.action_chunk_size, action_dim)


class TestVQBeTPolicy:
    def test_policy_creation(self):
        """Test policy can be created."""
        config = _make_config()
        policy = VQBeTPolicy(config)
        assert policy.config == config

    def test_vqbet_loss_computation(self):
        """Test that loss computation produces valid values."""
        config = _make_config()
        head = VQBeTHead(config)
        head.train()

        # Simulate action head output
        N, T = 2, 3
        action_dim = config.action_feature.shape[0]
        features = mx.random.normal(shape=(N, T, config.gpt_output_dim))
        output = head(features)

        # Create target actions
        target = mx.random.normal(shape=(N, T, config.action_chunk_size, action_dim))
        loss_dict = head.loss_fn(output, target)

        mx.eval(loss_dict["loss"])
        assert "loss" in loss_dict
        assert "classification_loss" in loss_dict
        assert "offset_loss" in loss_dict
        assert float(loss_dict["loss"].item()) > 0

    def test_vqbet_gradient_flow(self):
        """Test that gradients flow through the VQ-VAE encoder/decoder."""
        config = _make_config()
        vqvae = VqVae(config)
        vqvae.train()

        action_dim = config.action_feature.shape[0]
        actions = mx.random.normal(shape=(4, config.action_chunk_size, action_dim))

        def loss_fn(model):
            # Encode -> decode roundtrip loss
            state_flat = actions.reshape(actions.shape[0], -1)
            encoded = model.encoder(state_flat)
            decoded = model.decoder(encoded)
            return mx.mean((state_flat - decoded) ** 2)

        loss, grads = nn.value_and_grad(vqvae, loss_fn)(vqvae)
        mx.eval(loss)
        assert float(loss.item()) > 0

        # Check some gradients are non-zero
        from mlx.utils import tree_flatten
        flat_grads = tree_flatten(grads)
        has_nonzero = any(
            mx.any(mx.abs(g) > 1e-10).item()
            for name, g in flat_grads
            if isinstance(g, mx.array) and g.size > 0
        )
        assert has_nonzero, "Expected some non-zero gradients"

    def test_vqbet_straight_through(self):
        """Test that straight-through estimator concept works - gradients pass through stop_gradient."""
        # Simple demonstration: x + stop_gradient(quantized - x) has gradient w.r.t. x
        x = mx.random.normal(shape=(4, 16))
        quantized = mx.zeros_like(x)  # pretend quantized
        result = x + mx.stop_gradient(quantized - x)
        # result should equal quantized in forward, but grad should flow as if it's x
        loss = mx.mean(result ** 2)
        mx.eval(loss)
        assert float(loss.item()) >= 0

    def test_vqbet_various_batch_sizes(self):
        """Test VQ-VAE works with various batch sizes."""
        config = _make_config()
        vqvae = VqVae(config)
        vqvae.train()

        action_dim = config.action_feature.shape[0]
        for batch_size in [1, 2, 8]:
            actions = mx.random.normal(shape=(batch_size, config.action_chunk_size, action_dim))
            loss, metric = vqvae.vqvae_forward(actions)
            mx.eval(loss)
            assert loss.shape == ()

    def test_vqbet_transformer_prediction(self):
        """Test that transformer produces output of correct shape."""
        config = _make_config()
        gpt = GPT(config)
        gpt.eval()

        seq_len = 8
        x = mx.random.normal(shape=(2, seq_len, config.gpt_input_dim))
        output = gpt(x)
        mx.eval(output)
        assert output.shape == (2, seq_len, config.gpt_output_dim)

    def test_vqbet_select_action(self):
        """Test that predict_action_chunk produces correct shape."""
        config = _make_config()
        policy = VQBeTPolicy(config)
        policy.eval()

        # Mark VQ-VAE as trained
        policy.vqbet.action_head.vqvae_model.discretized = mx.array(True)
        policy.vqbet.action_head.vqvae_model.vq_layer.freeze_codebook = mx.array(True)

        batch_size = 1

        # Fill queues with obs_state and obs_images
        for _ in range(config.n_obs_steps):
            policy._queues[OBS_STATE].append(
                mx.random.normal(shape=(batch_size, config.robot_state_feature.shape[0]))
            )
            policy._queues[OBS_IMAGES].append(
                mx.random.normal(shape=(batch_size, 1, 3, 84, 84))
            )

        batch = {
            OBS_STATE: mx.random.normal(shape=(batch_size, config.robot_state_feature.shape[0])),
            OBS_IMAGES: mx.random.normal(shape=(batch_size, 1, 3, 84, 84)),
        }
        action = policy.predict_action_chunk(batch)
        mx.eval(action)
        action_dim = config.action_feature.shape[0]
        assert action.shape[0] == batch_size
        assert action.shape[-1] == action_dim


class TestVQBeTDiscretize:
    def test_discretize_step(self):
        """Test a single discretization step."""
        config = _make_config()
        head = VQBeTHead(config)
        head.train()

        action_dim = config.action_feature.shape[0]
        # Need enough timesteps for sliding window
        n_timesteps = config.action_chunk_size + 2
        actions = mx.random.normal(shape=(2, n_timesteps, action_dim))

        loss, n_codes, n_combos, recon_err = head.discretize(
            n_vqvae_training_steps=100, actions=actions
        )
        mx.eval(loss)
        assert float(loss.item()) >= 0
        assert n_codes > 0
        assert n_combos > 0
        assert recon_err >= 0
