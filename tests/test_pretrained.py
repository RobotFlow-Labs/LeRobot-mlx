"""Tests for weight conversion pipeline (PRD-12).

Tests cover:
- PyTorch -> MLX weight conversion
- Conv2d/Conv1d weight transposition
- Optimizer state / batch norm tracking key skipping
- Policy type detection from config dicts
- Config creation from dicts
- Heuristics for conv weight detection
"""

import dataclasses

import mlx.core as mx
import numpy as np
import pytest

from lerobot_mlx.policies.pretrained import (
    _create_config,
    _detect_policy_type,
    _is_conv1d_weight,
    _is_conv2d_weight,
    _remap_weight_key,
    convert_torch_weights_to_mlx,
)


# ---------------------------------------------------------------------------
# convert_torch_weights_to_mlx
# ---------------------------------------------------------------------------


class TestConvertWeights:
    def test_convert_torch_weights_basic(self):
        """Simple linear weights pass through unchanged."""
        weights = {
            "layer.weight": np.random.randn(10, 5).astype(np.float32),
            "layer.bias": np.random.randn(10).astype(np.float32),
        }
        result = convert_torch_weights_to_mlx(weights)
        assert "layer.weight" in result
        assert "layer.bias" in result
        assert isinstance(result["layer.weight"], mx.array)
        np.testing.assert_array_almost_equal(
            np.array(result["layer.weight"]), weights["layer.weight"]
        )

    def test_convert_conv2d_transpose(self):
        """Conv2d weights transposed from OIHW -> OHWI."""
        weights = {
            "conv1.weight": np.random.randn(16, 3, 3, 3).astype(np.float32),
        }
        result = convert_torch_weights_to_mlx(weights)
        arr = np.array(result["conv1.weight"])
        # Original: (16, 3, 3, 3) -> transposed: (16, 3, 3, 3)
        # O=16, I=3, H=3, W=3 -> O=16, H=3, W=3, I=3
        assert arr.shape == (16, 3, 3, 3)
        # Verify the transposition happened correctly
        expected = np.transpose(weights["conv1.weight"], (0, 2, 3, 1))
        np.testing.assert_array_almost_equal(arr, expected)

    def test_convert_conv2d_non_square(self):
        """Conv2d with non-square kernel transposes correctly."""
        weights = {
            "conv_layer.weight": np.random.randn(32, 16, 5, 7).astype(np.float32),
        }
        result = convert_torch_weights_to_mlx(weights)
        arr = np.array(result["conv_layer.weight"])
        # OIHW (32, 16, 5, 7) -> OHWI (32, 5, 7, 16)
        assert arr.shape == (32, 5, 7, 16)
        expected = np.transpose(weights["conv_layer.weight"], (0, 2, 3, 1))
        np.testing.assert_array_almost_equal(arr, expected)

    def test_convert_conv1d_transpose(self):
        """Conv1d weights transposed from OIL -> OLI."""
        weights = {
            "temporal_conv.weight": np.random.randn(8, 4, 5).astype(np.float32),
        }
        result = convert_torch_weights_to_mlx(weights)
        arr = np.array(result["temporal_conv.weight"])
        # OIL (8, 4, 5) -> OLI (8, 5, 4)
        assert arr.shape == (8, 5, 4)
        expected = np.transpose(weights["temporal_conv.weight"], (0, 2, 1))
        np.testing.assert_array_almost_equal(arr, expected)

    def test_convert_skips_optimizer_state(self):
        """Optimizer state keys are excluded."""
        weights = {
            "layer.weight": np.ones((3, 3), dtype=np.float32),
            "optimizer.state.step": np.array([100], dtype=np.float32),
            "optimizer.param_groups": np.array([0.01], dtype=np.float32),
        }
        result = convert_torch_weights_to_mlx(weights)
        assert "layer.weight" in result
        assert "optimizer.state.step" not in result
        assert "optimizer.param_groups" not in result

    def test_convert_skips_num_batches_tracked(self):
        """BatchNorm num_batches_tracked is excluded."""
        weights = {
            "bn.weight": np.ones(16, dtype=np.float32),
            "bn.bias": np.zeros(16, dtype=np.float32),
            "bn.num_batches_tracked": np.array([500], dtype=np.int64),
        }
        result = convert_torch_weights_to_mlx(weights)
        assert "bn.weight" in result
        assert "bn.bias" in result
        assert "bn.num_batches_tracked" not in result

    def test_convert_skips_scheduler_state(self):
        """Scheduler state keys are excluded."""
        weights = {
            "layer.weight": np.ones((2, 2), dtype=np.float32),
            "scheduler.last_epoch": np.array([10], dtype=np.float32),
        }
        result = convert_torch_weights_to_mlx(weights)
        assert "layer.weight" in result
        assert "scheduler.last_epoch" not in result

    def test_convert_preserves_dtype(self):
        """Output arrays preserve float32 dtype."""
        weights = {
            "layer.weight": np.random.randn(4, 4).astype(np.float32),
        }
        result = convert_torch_weights_to_mlx(weights)
        assert result["layer.weight"].dtype == mx.float32

    def test_convert_empty_dict(self):
        """Empty weight dict returns empty result."""
        result = convert_torch_weights_to_mlx({})
        assert result == {}

    def test_convert_downsample_as_conv(self):
        """downsample.0.weight treated as conv2d and transposed."""
        weights = {
            "layer1.0.downsample.0.weight": np.random.randn(64, 32, 1, 1).astype(np.float32),
        }
        result = convert_torch_weights_to_mlx(weights)
        arr = np.array(result["layer1.0.downsample.0.weight"])
        # OIHW (64, 32, 1, 1) -> OHWI (64, 1, 1, 32)
        assert arr.shape == (64, 1, 1, 32)

    def test_convert_bias_not_transposed(self):
        """Bias vectors are not transposed even with 'conv' in key."""
        weights = {
            "conv1.bias": np.random.randn(16).astype(np.float32),
        }
        result = convert_torch_weights_to_mlx(weights)
        arr = np.array(result["conv1.bias"])
        assert arr.shape == (16,)


# ---------------------------------------------------------------------------
# _detect_policy_type
# ---------------------------------------------------------------------------


class TestDetectPolicyType:
    def test_detect_from_target_act(self):
        config = {"_target_": "lerobot.policies.act.modeling_act.ACTPolicy"}
        assert _detect_policy_type(config) == "act"

    def test_detect_from_target_diffusion(self):
        config = {"_target_": "lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy"}
        assert _detect_policy_type(config) == "diffusion"

    def test_detect_from_target_sac(self):
        config = {"_target_": "lerobot.policies.sac.modeling_sac.SACPolicy"}
        assert _detect_policy_type(config) == "sac"

    def test_detect_from_model_type(self):
        config = {"model_type": "act"}
        assert _detect_policy_type(config) == "act"

    def test_detect_from_policy_dict(self):
        config = {"policy": {"type": "diffusion"}}
        assert _detect_policy_type(config) == "diffusion"

    def test_detect_unknown_raises(self):
        config = {"some_random_key": 42}
        with pytest.raises(ValueError, match="Cannot detect policy type"):
            _detect_policy_type(config)


# ---------------------------------------------------------------------------
# _create_config
# ---------------------------------------------------------------------------


class TestCreateConfig:
    def test_create_config_from_dict(self):
        """Create ACTConfig from a dict with matching fields."""
        from lerobot_mlx.policies.act.configuration_act import ACTConfig

        config_dict = {"dim_model": 256, "n_heads": 4, "chunk_size": 50, "n_action_steps": 50}
        config = _create_config(ACTConfig, config_dict)
        assert config.dim_model == 256
        assert config.n_heads == 4
        assert config.chunk_size == 50

    def test_create_config_filters_unknown_fields(self):
        """Unknown fields are silently ignored."""
        from lerobot_mlx.policies.act.configuration_act import ACTConfig

        config_dict = {"dim_model": 128, "unknown_field": "should_be_ignored"}
        config = _create_config(ACTConfig, config_dict)
        assert config.dim_model == 128
        assert not hasattr(config, "unknown_field")

    def test_create_config_empty_dict_uses_defaults(self):
        """Empty dict produces config with all defaults."""
        from lerobot_mlx.policies.act.configuration_act import ACTConfig

        config = _create_config(ACTConfig, {})
        assert config.dim_model == 512  # default
        assert config.n_heads == 8  # default


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


class TestConvHeuristics:
    def test_is_conv2d_weight_positive(self):
        assert _is_conv2d_weight("encoder.conv1.weight") is True
        assert _is_conv2d_weight("backbone.layer1.0.conv2.weight") is True
        assert _is_conv2d_weight("layer1.0.downsample.0.weight") is True

    def test_is_conv2d_weight_negative(self):
        assert _is_conv2d_weight("encoder.conv1.bias") is False
        assert _is_conv2d_weight("linear.weight") is False

    def test_is_conv1d_weight_positive(self):
        assert _is_conv1d_weight("temporal_conv.weight") is True
        assert _is_conv1d_weight("model.conv1d_layer.weight") is True

    def test_is_conv1d_weight_negative(self):
        assert _is_conv1d_weight("temporal_conv.bias") is False
        assert _is_conv1d_weight("linear.weight") is False
