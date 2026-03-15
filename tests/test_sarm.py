"""Tests for SARM policy -- MLX port."""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from lerobot_mlx.policies.sarm.configuration_sarm import SARMConfig
from lerobot_mlx.policies.sarm.modeling_sarm import (
    SARMRewardModel,
    StageTransformer,
    SubtaskTransformer,
    gen_stage_emb,
    compute_stage_loss,
    normalize_stage_tau,
    pad_state_to_max_dim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> SARMConfig:
    """Create a small SARM config suitable for testing."""
    defaults = dict(
        annotation_mode="single_stage",
        n_obs_steps=8,
        max_rewind_steps=4,
        image_dim=32,
        text_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        max_state_dim=8,
        dropout=0.0,
    )
    defaults.update(overrides)
    return SARMConfig(**defaults)


def _make_batch(config: SARMConfig, batch_size: int = 2):
    """Create a synthetic batch for testing."""
    T = config.num_frames  # 1 + n_obs_steps + max_rewind_steps = 13
    return {
        "video_features": mx.random.normal((batch_size, T, config.image_dim)),
        "text_features": mx.random.normal((batch_size, config.text_dim)),
        "state_features": mx.random.normal((batch_size, T, config.max_state_dim)),
        "lengths": mx.full((batch_size,), T, dtype=mx.int32),
        "sparse_targets": mx.random.uniform(shape=(batch_size, T)) * 0.99,  # stage 0, tau in [0, 1)
    }


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestSARMConfig:
    def test_sarm_config_defaults(self):
        config = SARMConfig()
        assert config.annotation_mode == "single_stage"
        assert config.n_obs_steps == 8
        assert config.max_rewind_steps == 4
        assert config.num_sparse_stages == 1
        assert config.sparse_subtask_names == ["task"]
        assert config.sparse_temporal_proportions == [1.0]
        assert config.num_dense_stages is None
        assert config.uses_dual_heads is False
        assert config.num_frames == 13  # 1 + 8 + 4

    def test_sarm_config_invalid_annotation_mode(self):
        with pytest.raises(ValueError, match="annotation_mode"):
            SARMConfig(annotation_mode="invalid")

    def test_sarm_config_invalid_rewind(self):
        with pytest.raises(ValueError, match="max_rewind_steps"):
            SARMConfig(max_rewind_steps=10, n_obs_steps=8)

    def test_sarm_config_dense_only(self):
        config = SARMConfig(annotation_mode="dense_only", num_dense_stages=4)
        assert config.uses_dual_heads is True
        assert config.num_sparse_stages == 1
        assert config.sparse_subtask_names == ["task"]

    def test_sarm_config_observation_delta_indices(self):
        config = SARMConfig()
        deltas = config.observation_delta_indices
        # n_obs_steps=8 -> half=4 -> 4 past + 1 current + 4 future + 4 rewind = 13
        assert len(deltas) == config.num_frames
        assert deltas[4] == 0  # center frame

    def test_sarm_config_action_delta_indices(self):
        config = SARMConfig()
        assert config.action_delta_indices is None
        assert config.reward_delta_indices is None


# ---------------------------------------------------------------------------
# Model creation tests
# ---------------------------------------------------------------------------

class TestSARMModelCreation:
    def test_sarm_model_creation(self):
        config = _make_config()
        model = SARMRewardModel(config)
        assert model.config is config
        assert isinstance(model.stage_model, StageTransformer)
        assert isinstance(model.subtask_model, SubtaskTransformer)

    def test_sarm_model_creation_dual_heads(self):
        config = _make_config(annotation_mode="dense_only", num_dense_stages=4)
        model = SARMRewardModel(config)
        assert config.uses_dual_heads is True


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------

class TestSARMForwardShape:
    def test_sarm_forward_shape(self):
        config = _make_config()
        model = SARMRewardModel(config)
        batch = _make_batch(config, batch_size=2)
        total_loss, output_dict = model(batch)
        assert total_loss.shape == ()
        assert "sparse_stage_loss" in output_dict
        assert "sparse_subtask_loss" in output_dict
        assert "total_loss" in output_dict
        mx.eval(total_loss)
        assert float(total_loss) > 0

    def test_sarm_forward_shape_batch4(self):
        config = _make_config()
        model = SARMRewardModel(config)
        batch = _make_batch(config, batch_size=4)
        total_loss, output_dict = model(batch)
        mx.eval(total_loss)
        assert float(total_loss) > 0


# ---------------------------------------------------------------------------
# Loss computation tests
# ---------------------------------------------------------------------------

class TestSARMLoss:
    def test_sarm_loss_computation(self):
        config = _make_config()
        model = SARMRewardModel(config)
        batch = _make_batch(config, batch_size=2)
        total_loss, output_dict = model(batch)
        mx.eval(total_loss)

        assert output_dict["sparse_stage_loss"] >= 0
        assert output_dict["sparse_subtask_loss"] >= 0
        assert output_dict["total_loss"] >= 0

    def test_sarm_loss_no_state_features(self):
        """Model should work without state_features (defaults to zeros)."""
        config = _make_config()
        model = SARMRewardModel(config)
        T = config.num_frames
        batch = {
            "video_features": mx.random.normal((2, T, config.image_dim)),
            "text_features": mx.random.normal((2, config.text_dim)),
            "sparse_targets": mx.random.uniform(shape=(2, T)) * 0.99,
        }
        total_loss, output_dict = model(batch)
        mx.eval(total_loss)
        assert float(total_loss) > 0


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

class TestSARMGradientFlow:
    def test_sarm_gradient_flow(self):
        config = _make_config()
        model = SARMRewardModel(config)
        batch = _make_batch(config, batch_size=2)

        def loss_fn(model, batch):
            total_loss, _ = model(batch)
            return total_loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss_val, grads = loss_and_grad_fn(model, batch)
        mx.eval(loss_val, grads)

        # Check that gradients exist and are not all zero
        import mlx.utils
        flat_grads = mlx.utils.tree_flatten(grads)
        has_nonzero = False
        for name, g in flat_grads:
            if isinstance(g, mx.array) and g.size > 0:
                mx.eval(g)
                if mx.any(g != 0).item():
                    has_nonzero = True
                    break
        assert has_nonzero, "Expected at least some non-zero gradients"


# ---------------------------------------------------------------------------
# select_action test
# ---------------------------------------------------------------------------

class TestSARMSelectAction:
    def test_sarm_select_action(self):
        config = _make_config()
        model = SARMRewardModel(config)
        with pytest.raises(NotImplementedError, match="does not select actions"):
            model.select_action({})

    def test_sarm_predict_action_chunk(self):
        config = _make_config()
        model = SARMRewardModel(config)
        with pytest.raises(NotImplementedError, match="does not predict action chunks"):
            model.predict_action_chunk({})


# ---------------------------------------------------------------------------
# Various batch sizes
# ---------------------------------------------------------------------------

class TestSARMVariousBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_sarm_various_batch_sizes(self, batch_size):
        config = _make_config()
        model = SARMRewardModel(config)
        batch = _make_batch(config, batch_size=batch_size)
        total_loss, output_dict = model(batch)
        mx.eval(total_loss)
        assert float(total_loss) > 0


# ---------------------------------------------------------------------------
# Transformer layer tests
# ---------------------------------------------------------------------------

class TestSARMTransformerLayers:
    def test_sarm_transformer_layers(self):
        """Encoder produces correct output shapes."""
        config = _make_config()
        model = SARMRewardModel(config)

        B, N, T = 2, 1, config.num_frames
        img_seq = mx.random.normal((B, N, T, config.image_dim))
        lang_emb = mx.random.normal((B, config.text_dim))
        state = mx.random.normal((B, T, config.max_state_dim))
        lengths = mx.full((B,), T, dtype=mx.int32)

        # Stage model forward
        stage_logits = model.stage_model(img_seq, lang_emb, state, lengths, scheme="sparse")
        mx.eval(stage_logits)
        assert stage_logits.shape == (B, T, config.num_sparse_stages)

    def test_stage_transformer_dense(self):
        """Stage transformer with dense scheme."""
        config = _make_config(annotation_mode="dense_only", num_dense_stages=4)
        model = SARMRewardModel(config)

        B, N, T = 2, 1, config.num_frames
        img_seq = mx.random.normal((B, N, T, config.image_dim))
        lang_emb = mx.random.normal((B, config.text_dim))
        state = mx.random.normal((B, T, config.max_state_dim))
        lengths = mx.full((B,), T, dtype=mx.int32)

        logits = model.stage_model(img_seq, lang_emb, state, lengths, scheme="dense")
        mx.eval(logits)
        assert logits.shape == (B, T, config.num_dense_stages)


# ---------------------------------------------------------------------------
# State prediction tests
# ---------------------------------------------------------------------------

class TestSARMStatePrediction:
    def test_sarm_state_prediction(self):
        """State features are processed correctly through the model."""
        config = _make_config()
        model = SARMRewardModel(config)

        B, T = 2, config.num_frames
        batch_no_state = {
            "video_features": mx.random.normal((B, T, config.image_dim)),
            "text_features": mx.random.normal((B, config.text_dim)),
            "sparse_targets": mx.random.uniform(shape=(B, T)) * 0.99,
        }
        batch_with_state = {
            **batch_no_state,
            "state_features": mx.random.normal((B, T, config.max_state_dim)),
        }

        loss_no_state, _ = model(batch_no_state)
        loss_with_state, _ = model(batch_with_state)
        mx.eval(loss_no_state, loss_with_state)

        # Both should produce valid losses (different values expected)
        assert float(loss_no_state) > 0
        assert float(loss_with_state) > 0

    def test_pad_state_to_max_dim(self):
        state = mx.ones((2, 5, 4))
        padded = pad_state_to_max_dim(state, 8)
        assert padded.shape == (2, 5, 8)
        mx.eval(padded)
        # First 4 dims should be 1, rest 0
        assert np.allclose(np.array(padded[0, 0, :4]), 1.0)
        assert np.allclose(np.array(padded[0, 0, 4:]), 0.0)

    def test_pad_state_truncate(self):
        state = mx.ones((2, 5, 16))
        padded = pad_state_to_max_dim(state, 8)
        assert padded.shape == (2, 5, 8)


# ---------------------------------------------------------------------------
# Reward prediction tests
# ---------------------------------------------------------------------------

class TestSARMRewardPrediction:
    def test_sarm_reward_prediction(self):
        """calculate_rewards returns valid normalized values."""
        config = _make_config()
        model = SARMRewardModel(config)

        B, T = 2, config.num_frames
        text_emb = np.random.randn(B, config.text_dim).astype(np.float32)
        video_emb = np.random.randn(B, T, config.image_dim).astype(np.float32)
        state = np.random.randn(B, T, config.max_state_dim).astype(np.float32)

        rewards = model.calculate_rewards(
            text_embeddings=text_emb,
            video_embeddings=video_emb,
            state_features=state,
            return_all_frames=True,
        )
        assert rewards.shape == (B, T)
        assert np.all(rewards >= 0.0) and np.all(rewards <= 1.0)

    def test_sarm_reward_prediction_single_sample(self):
        """calculate_rewards works with single sample (no batch dim)."""
        config = _make_config()
        model = SARMRewardModel(config)

        T = config.num_frames
        text_emb = np.random.randn(config.text_dim).astype(np.float32)
        video_emb = np.random.randn(T, config.image_dim).astype(np.float32)

        rewards = model.calculate_rewards(
            text_embeddings=text_emb,
            video_embeddings=video_emb,
            return_all_frames=True,
        )
        assert rewards.shape == (T,)

    def test_sarm_reward_with_stages(self):
        """calculate_rewards with return_stages=True."""
        config = _make_config()
        model = SARMRewardModel(config)

        B, T = 2, config.num_frames
        text_emb = np.random.randn(B, config.text_dim).astype(np.float32)
        video_emb = np.random.randn(B, T, config.image_dim).astype(np.float32)

        rewards, stages = model.calculate_rewards(
            text_embeddings=text_emb,
            video_embeddings=video_emb,
            return_all_frames=True,
            return_stages=True,
        )
        assert rewards.shape == (B, T)
        assert stages.shape == (B, T, config.num_sparse_stages)


# ---------------------------------------------------------------------------
# gen_stage_emb tests
# ---------------------------------------------------------------------------

class TestGenStageEmb:
    def test_gen_stage_emb_basic(self):
        targets = mx.array([[0.3, 1.7, 2.1]])  # (1, 3)
        emb = gen_stage_emb(4, targets)
        mx.eval(emb)
        assert emb.shape == (1, 1, 3, 4)
        # Stage 0, 1, 2
        np_emb = np.array(emb[0, 0])
        assert np_emb[0, 0] == 1.0  # stage 0
        assert np_emb[1, 1] == 1.0  # stage 1
        assert np_emb[2, 2] == 1.0  # stage 2

    def test_gen_stage_emb_clamp(self):
        """Values >= num_classes should be clamped."""
        targets = mx.array([[5.0, -1.0]])
        emb = gen_stage_emb(3, targets)
        mx.eval(emb)
        # stage 5 clamped to 2, stage -1 clamped to 0
        np_emb = np.array(emb[0, 0])
        assert np_emb[0, 2] == 1.0
        assert np_emb[1, 0] == 1.0


# ---------------------------------------------------------------------------
# compute_stage_loss tests
# ---------------------------------------------------------------------------

class TestComputeStageLoss:
    def test_compute_stage_loss_basic(self):
        logits = mx.random.normal((2, 5, 3))
        targets = mx.array([[0, 1, 2, 0, 1], [2, 2, 1, 0, 0]], dtype=mx.float32)
        loss = compute_stage_loss(logits, targets)
        mx.eval(loss)
        assert loss.shape == ()
        assert float(loss) > 0


# ---------------------------------------------------------------------------
# normalize_stage_tau tests
# ---------------------------------------------------------------------------

class TestNormalizeStageTau:
    def test_normalize_single_stage(self):
        x = mx.array([0.0, 0.5, 0.99])
        result = normalize_stage_tau(x, num_stages=1)
        mx.eval(result)
        np_result = np.array(result)
        assert np.allclose(np_result, [0.0, 0.5, 0.99], atol=1e-3)

    def test_normalize_multi_stage(self):
        x = mx.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = normalize_stage_tau(x, num_stages=3)
        mx.eval(result)
        np_result = np.array(result)
        # Linear breakpoints: [0, 1/3, 2/3, 1]
        expected = [0.0, 0.5 / 3, 1 / 3, 1 / 3 + 0.5 / 3, 2 / 3]
        assert np.allclose(np_result, expected, atol=1e-4)

    def test_normalize_clamp(self):
        x = mx.array([3.5])
        result = normalize_stage_tau(x, num_stages=3)
        mx.eval(result)
        assert float(result[0]) == 1.0
