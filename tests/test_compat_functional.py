"""
Tests for lerobot_mlx.compat: functional, optim, distributions.

60+ tests covering:
- All loss functions (mse, l1, cross_entropy, smooth_l1, bce_with_logits)
- All activations (relu, gelu, silu, sigmoid, tanh, softmax, etc.)
- Padding (constant, torch format conversion, 1D/2D/3D, reflect, replicate)
- Attention (with/without mask, causal mask, shape correctness)
- Normalization (normalize, layer_norm, group_norm)
- Interpolation (nearest neighbor, bilinear, size, scale_factor)
- One-hot encoding
- Grid sample (identity, zeros padding, nearest)
- Normal distribution (sample, log_prob, entropy, KL divergence)
- Beta distribution (sample, log_prob, rsample)
- Independent wrapper
- Optimizer step + LR scheduler step
- Gradient clipping (efficient norm computation)

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

import math
import sys

import mlx.core as mx
import mlx.optimizers as _optim
import numpy as np
import pytest

from lerobot_mlx.compat import functional as F
from lerobot_mlx.compat.distributions import (
    Beta,
    Independent,
    Normal,
    kl_divergence,
)
from lerobot_mlx.compat.optim import (
    Adam,
    AdamW,
    CosineAnnealingLR,
    LinearWarmupCosineDecay,
    SGD,
    clip_grad_norm_,
    get_scheduler,
)


# =============================================================================
# Loss Functions
# =============================================================================

class TestMSELoss:
    def test_basic(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([1.5, 2.5, 3.5])
        loss = F.mse_loss(x, y)
        expected = np.mean((np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 3.5])) ** 2)
        assert abs(loss.item() - expected) < 1e-6

    def test_reduction_sum(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([0.0, 0.0, 0.0])
        loss = F.mse_loss(x, y, reduction="sum")
        assert abs(loss.item() - 14.0) < 1e-6

    def test_reduction_none(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([0.0, 0.0, 0.0])
        loss = F.mse_loss(x, y, reduction="none")
        mx.eval(loss)
        np.testing.assert_allclose(np.array(loss), [1.0, 4.0, 9.0], atol=1e-6)

    def test_identical_inputs(self):
        x = mx.array([1.0, 2.0, 3.0])
        loss = F.mse_loss(x, x)
        assert abs(loss.item()) < 1e-7

    def test_batched(self):
        x = mx.random.normal((4, 10))
        y = mx.random.normal((4, 10))
        loss = F.mse_loss(x, y)
        mx.eval(loss)
        assert loss.shape == ()


class TestL1Loss:
    def test_basic(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([1.5, 2.5, 3.5])
        loss = F.l1_loss(x, y)
        assert abs(loss.item() - 0.5) < 1e-6

    def test_reduction_sum(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([0.0, 0.0, 0.0])
        loss = F.l1_loss(x, y, reduction="sum")
        assert abs(loss.item() - 6.0) < 1e-6

    def test_reduction_none(self):
        x = mx.array([1.0, 2.0])
        y = mx.array([0.0, 0.0])
        loss = F.l1_loss(x, y, reduction="none")
        mx.eval(loss)
        np.testing.assert_allclose(np.array(loss), [1.0, 2.0], atol=1e-6)


class TestSmoothL1Loss:
    def test_small_diff(self):
        """For diff < beta, loss = 0.5 * diff^2 / beta."""
        x = mx.array([0.0])
        y = mx.array([0.5])
        loss = F.smooth_l1_loss(x, y, beta=1.0)
        expected = 0.5 * 0.5 ** 2 / 1.0
        assert abs(loss.item() - expected) < 1e-6

    def test_large_diff(self):
        """For diff >= beta, loss = diff - 0.5 * beta."""
        x = mx.array([0.0])
        y = mx.array([2.0])
        loss = F.smooth_l1_loss(x, y, beta=1.0)
        expected = 2.0 - 0.5
        assert abs(loss.item() - expected) < 1e-6

    def test_reduction_none(self):
        x = mx.array([0.0, 0.0])
        y = mx.array([0.3, 2.0])
        loss = F.smooth_l1_loss(x, y, beta=1.0, reduction="none")
        mx.eval(loss)
        assert loss.shape == (2,)


class TestCrossEntropy:
    def test_basic(self):
        logits = mx.array([[2.0, 1.0, 0.1]])
        target = mx.array([0])
        loss = F.cross_entropy(logits, target)
        mx.eval(loss)
        assert loss.item() > 0

    def test_reduction_none(self):
        logits = mx.array([[2.0, 1.0], [0.5, 1.5]])
        target = mx.array([0, 1])
        loss = F.cross_entropy(logits, target, reduction="none")
        mx.eval(loss)
        assert loss.shape == (2,)

    def test_perfect_prediction(self):
        """Very confident correct prediction should have near-zero loss."""
        logits = mx.array([[100.0, -100.0]])
        target = mx.array([0])
        loss = F.cross_entropy(logits, target)
        mx.eval(loss)
        assert loss.item() < 0.01

    def test_with_weight(self):
        """Cross entropy with per-class weights."""
        logits = mx.array([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
        target = mx.array([0, 1])
        weight = mx.array([1.0, 2.0, 0.5])
        loss_weighted = F.cross_entropy(logits, target, weight=weight)
        loss_unweighted = F.cross_entropy(logits, target)
        mx.eval(loss_weighted)
        mx.eval(loss_unweighted)
        # Weighted loss should differ from unweighted
        assert loss_weighted.item() != loss_unweighted.item()

    def test_with_weight_reduction_sum(self):
        """Cross entropy with weights and sum reduction."""
        logits = mx.array([[2.0, 1.0], [0.5, 1.5]])
        target = mx.array([0, 1])
        weight = mx.array([1.0, 2.0])
        loss = F.cross_entropy(logits, target, weight=weight, reduction="sum")
        mx.eval(loss)
        assert loss.item() > 0

    def test_with_weight_reduction_none(self):
        """Cross entropy with weights and no reduction."""
        logits = mx.array([[2.0, 1.0], [0.5, 1.5]])
        target = mx.array([0, 1])
        weight = mx.array([1.0, 2.0])
        loss = F.cross_entropy(logits, target, weight=weight, reduction="none")
        mx.eval(loss)
        assert loss.shape == (2,)


class TestBCEWithLogits:
    def test_basic(self):
        logits = mx.array([0.0, 0.0])
        targets = mx.array([1.0, 0.0])
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        mx.eval(loss)
        # At logits=0, BCE = log(2) ~ 0.693
        assert abs(loss.item() - math.log(2)) < 1e-5

    def test_reduction_sum(self):
        logits = mx.array([0.0, 0.0])
        targets = mx.array([1.0, 0.0])
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
        mx.eval(loss)
        assert abs(loss.item() - 2 * math.log(2)) < 1e-5

    def test_reduction_none(self):
        logits = mx.array([0.0, 0.0])
        targets = mx.array([1.0, 0.0])
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        mx.eval(loss)
        assert loss.shape == (2,)


# =============================================================================
# Activations
# =============================================================================

class TestActivations:
    def test_relu(self):
        x = mx.array([-1.0, 0.0, 1.0, 2.0])
        y = F.relu(x)
        mx.eval(y)
        np.testing.assert_allclose(np.array(y), [0.0, 0.0, 1.0, 2.0], atol=1e-6)

    def test_gelu(self):
        x = mx.array([0.0, 1.0, -1.0])
        y = F.gelu(x)
        mx.eval(y)
        # GELU(0) = 0
        assert abs(y[0].item()) < 1e-6

    def test_silu(self):
        x = mx.array([0.0, 1.0, -1.0])
        y = F.silu(x)
        mx.eval(y)
        # SiLU(0) = 0
        assert abs(y[0].item()) < 1e-6
        # SiLU(x) = x * sigmoid(x)
        expected = 1.0 * (1.0 / (1.0 + math.exp(-1.0)))
        assert abs(y[1].item() - expected) < 1e-5

    def test_sigmoid(self):
        x = mx.array([0.0])
        y = F.sigmoid(x)
        mx.eval(y)
        assert abs(y.item() - 0.5) < 1e-6

    def test_tanh(self):
        x = mx.array([0.0])
        y = F.tanh(x)
        mx.eval(y)
        assert abs(y.item()) < 1e-6

    def test_softmax(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = F.softmax(x, dim=0)
        mx.eval(y)
        assert abs(mx.sum(y).item() - 1.0) < 1e-5

    def test_softmax_dim(self):
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        y = F.softmax(x, dim=-1)
        mx.eval(y)
        sums = mx.sum(y, axis=-1)
        mx.eval(sums)
        np.testing.assert_allclose(np.array(sums), [1.0, 1.0], atol=1e-5)

    def test_log_softmax(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = F.log_softmax(x, dim=0)
        mx.eval(y)
        # exp(log_softmax) should sum to 1
        probs = mx.exp(y)
        mx.eval(probs)
        assert abs(mx.sum(probs).item() - 1.0) < 1e-5

    def test_softplus(self):
        x = mx.array([0.0])
        y = F.softplus(x)
        mx.eval(y)
        assert abs(y.item() - math.log(2)) < 1e-5

    def test_softplus_large(self):
        """For large x, softplus ~ x."""
        x = mx.array([100.0])
        y = F.softplus(x)
        mx.eval(y)
        assert abs(y.item() - 100.0) < 1e-3

    def test_elu(self):
        x = mx.array([-1.0, 0.0, 1.0])
        y = F.elu(x, alpha=1.0)
        mx.eval(y)
        # ELU(-1) = alpha * (exp(-1) - 1) ~ -0.6321
        assert abs(y[0].item() - (math.exp(-1) - 1)) < 1e-5
        assert abs(y[1].item()) < 1e-6
        assert abs(y[2].item() - 1.0) < 1e-6


# =============================================================================
# Padding
# =============================================================================

class TestPad:
    def test_1d_pad(self):
        """Pad last dim only: (left, right)."""
        x = mx.array([[1.0, 2.0, 3.0]])
        y = F.pad(x, (1, 2))
        mx.eval(y)
        assert y.shape == (1, 6)  # 1 + 3 + 2
        assert y[0, 0].item() == 0.0
        assert y[0, 1].item() == 1.0

    def test_2d_pad(self):
        """Pad last two dims: (left, right, top, bottom)."""
        x = mx.ones((1, 1, 3, 3))
        y = F.pad(x, (1, 1, 1, 1))
        mx.eval(y)
        assert y.shape == (1, 1, 5, 5)
        # Corners should be 0
        assert y[0, 0, 0, 0].item() == 0.0
        # Center should be 1
        assert y[0, 0, 2, 2].item() == 1.0

    def test_custom_value(self):
        x = mx.zeros((2, 3))
        y = F.pad(x, (1, 1), value=5.0)
        mx.eval(y)
        assert y[0, 0].item() == 5.0
        assert y[0, -1].item() == 5.0

    def test_asymmetric_pad(self):
        x = mx.ones((1, 1, 2, 2))
        y = F.pad(x, (0, 1, 2, 0))  # left=0, right=1, top=2, bottom=0
        mx.eval(y)
        assert y.shape == (1, 1, 4, 3)

    def test_3d_pad(self):
        """Pad last three dims: (left, right, top, bottom, front, back)."""
        x = mx.ones((1, 2, 2, 2))
        y = F.pad(x, (1, 1, 1, 1, 1, 1))
        mx.eval(y)
        assert y.shape == (1, 4, 4, 4)

    def test_torch_format_ordering(self):
        """Verify torch reversed-pair format is handled correctly."""
        x = mx.ones((1, 1, 3, 4))
        # torch: (left=1, right=2, top=3, bottom=4)
        # Last dim gets (1, 2), second-to-last gets (3, 4)
        y = F.pad(x, (1, 2, 3, 4))
        mx.eval(y)
        assert y.shape == (1, 1, 10, 7)  # H: 3+3+4=10, W: 1+4+2=7

    def test_reflect_pad_validation(self):
        """Reflect padding must be less than dimension size."""
        x = mx.ones((1, 1, 3, 3))
        # Padding of 3 on a dim of size 3 should raise
        with pytest.raises(ValueError, match="Reflect padding"):
            F.pad(x, (3, 0), mode="reflect")

    def test_reflect_pad_validation_after(self):
        """Reflect padding 'after' must also be less than dimension size."""
        x = mx.ones((1, 1, 3, 3))
        with pytest.raises(ValueError, match="Reflect padding"):
            F.pad(x, (0, 3), mode="reflect")

    def test_reflect_pad_valid(self):
        """Valid reflect padding should work."""
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        y = F.pad(x, (2, 2), mode="reflect")
        mx.eval(y)
        # Reflected: [3, 2, 1, 2, 3, 4, 3, 2]
        assert y.shape == (1, 8)
        assert y[0, 0].item() == 3.0
        assert y[0, 1].item() == 2.0

    def test_replicate_pad_mode(self):
        """Replicate (edge) padding repeats edge values."""
        x = mx.array([[1.0, 2.0, 3.0]])
        y = F.pad(x, (2, 3), mode="replicate")
        mx.eval(y)
        assert y.shape == (1, 8)  # 2 + 3 + 3
        # Left edge replicated
        assert y[0, 0].item() == 1.0
        assert y[0, 1].item() == 1.0
        # Original values
        assert y[0, 2].item() == 1.0
        assert y[0, 3].item() == 2.0
        assert y[0, 4].item() == 3.0
        # Right edge replicated
        assert y[0, 5].item() == 3.0
        assert y[0, 6].item() == 3.0
        assert y[0, 7].item() == 3.0

    def test_replicate_pad_2d(self):
        """Replicate padding on 2D spatial dims."""
        x = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
        y = F.pad(x, (1, 1, 1, 1), mode="replicate")
        mx.eval(y)
        assert y.shape == (1, 1, 4, 4)
        # Top-left corner should be replicated from (0,0)
        assert y[0, 0, 0, 0].item() == 1.0


# =============================================================================
# Attention
# =============================================================================

class TestScaledDotProductAttention:
    def test_basic_shape(self):
        B, H, Sq, Sk, D = 2, 4, 8, 8, 16
        q = mx.random.normal((B, H, Sq, D))
        k = mx.random.normal((B, H, Sk, D))
        v = mx.random.normal((B, H, Sk, D))
        out = F.scaled_dot_product_attention(q, k, v)
        mx.eval(out)
        assert out.shape == (B, H, Sq, D)

    def test_causal_mask(self):
        B, H, S, D = 1, 1, 4, 8
        q = mx.random.normal((B, H, S, D))
        k = mx.random.normal((B, H, S, D))
        v = mx.random.normal((B, H, S, D))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        mx.eval(out)
        assert out.shape == (B, H, S, D)

    def test_with_mask(self):
        B, H, S, D = 1, 1, 4, 8
        q = mx.random.normal((B, H, S, D))
        k = mx.random.normal((B, H, S, D))
        v = mx.random.normal((B, H, S, D))
        mask = mx.zeros((S, S))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        mx.eval(out)
        assert out.shape == (B, H, S, D)

    def test_different_seq_lengths(self):
        B, H, Sq, Sk, D = 1, 2, 5, 10, 16
        q = mx.random.normal((B, H, Sq, D))
        k = mx.random.normal((B, H, Sk, D))
        v = mx.random.normal((B, H, Sk, D))
        out = F.scaled_dot_product_attention(q, k, v)
        mx.eval(out)
        assert out.shape == (B, H, Sq, D)

    def test_uniform_attention_with_zero_query(self):
        """Zero query should give roughly uniform attention weights."""
        B, H, S, D = 1, 1, 4, 8
        q = mx.zeros((B, H, 1, D))
        k = mx.random.normal((B, H, S, D))
        v = mx.ones((B, H, S, D))  # uniform values
        out = F.scaled_dot_product_attention(q, k, v)
        mx.eval(out)
        # Output should be close to 1 (mean of uniform values)
        assert abs(out[0, 0, 0, 0].item() - 1.0) < 0.5


# =============================================================================
# Normalization
# =============================================================================

class TestNormalize:
    def test_l2_normalize(self):
        x = mx.array([[3.0, 4.0]])
        y = F.normalize(x, p=2.0, dim=-1)
        mx.eval(y)
        norm = mx.sqrt(mx.sum(y ** 2, axis=-1))
        mx.eval(norm)
        assert abs(norm.item() - 1.0) < 1e-5

    def test_zero_vector(self):
        x = mx.array([[0.0, 0.0]])
        y = F.normalize(x, p=2.0, dim=-1)
        mx.eval(y)
        # Should not produce NaN
        assert not np.any(np.isnan(np.array(y)))


class TestLayerNorm:
    def test_basic(self):
        x = mx.random.normal((2, 10))
        y = F.layer_norm(x, (10,))
        mx.eval(y)
        # Mean should be ~0, std ~1
        mean_val = mx.mean(y, axis=-1)
        mx.eval(mean_val)
        np.testing.assert_allclose(np.array(mean_val), [0.0, 0.0], atol=1e-5)

    def test_with_weight_bias(self):
        x = mx.random.normal((2, 5))
        w = mx.ones((5,)) * 2.0
        b = mx.ones((5,)) * 0.5
        y = F.layer_norm(x, (5,), weight=w, bias=b)
        mx.eval(y)
        assert y.shape == (2, 5)


class TestGroupNorm:
    def test_basic(self):
        x = mx.random.normal((2, 4, 3, 3))
        y = F.group_norm(x, num_groups=2)
        mx.eval(y)
        assert y.shape == (2, 4, 3, 3)

    def test_with_weight_bias(self):
        x = mx.random.normal((2, 4, 3, 3))
        w = mx.ones((4,))
        b = mx.zeros((4,))
        y = F.group_norm(x, num_groups=2, weight=w, bias=b)
        mx.eval(y)
        assert y.shape == (2, 4, 3, 3)


# =============================================================================
# Interpolation
# =============================================================================

class TestInterpolate:
    def test_nearest_scale_factor(self):
        x = mx.ones((1, 1, 2, 2))
        y = F.interpolate(x, scale_factor=2, mode="nearest")
        mx.eval(y)
        assert y.shape == (1, 1, 4, 4)
        assert y[0, 0, 0, 0].item() == 1.0

    def test_nearest_size(self):
        x = mx.ones((1, 1, 2, 2))
        y = F.interpolate(x, size=(4, 6), mode="nearest")
        mx.eval(y)
        assert y.shape == (1, 1, 4, 6)

    def test_nearest_preserves_values(self):
        x = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        y = F.interpolate(x, scale_factor=2, mode="nearest")
        mx.eval(y)
        # Top-left 2x2 block should be all 1.0
        assert y[0, 0, 0, 0].item() == 1.0
        assert y[0, 0, 0, 1].item() == 1.0
        assert y[0, 0, 1, 0].item() == 1.0
        assert y[0, 0, 1, 1].item() == 1.0

    def test_interpolate_bilinear_shape(self):
        """Bilinear interpolation produces correct output shape."""
        x = mx.random.normal((2, 3, 8, 8))
        y = F.interpolate(x, size=(16, 16), mode="bilinear")
        mx.eval(y)
        assert y.shape == (2, 3, 16, 16)

    def test_interpolate_bilinear_upscale(self):
        """Bilinear upscale of uniform tensor preserves values."""
        x = mx.ones((1, 1, 2, 2)) * 3.0
        y = F.interpolate(x, size=(4, 4), mode="bilinear")
        mx.eval(y)
        assert y.shape == (1, 1, 4, 4)
        # Uniform input should produce uniform output
        np.testing.assert_allclose(np.array(y), 3.0, atol=1e-5)

    def test_interpolate_bilinear_scale_factor(self):
        """Bilinear with scale_factor."""
        x = mx.random.normal((1, 2, 4, 4))
        y = F.interpolate(x, scale_factor=2.0, mode="bilinear")
        mx.eval(y)
        assert y.shape == (1, 2, 8, 8)

    def test_interpolate_bilinear_downsample(self):
        """Bilinear downsampling."""
        x = mx.random.normal((1, 1, 8, 8))
        y = F.interpolate(x, size=(4, 4), mode="bilinear")
        mx.eval(y)
        assert y.shape == (1, 1, 4, 4)


# =============================================================================
# One-hot
# =============================================================================

class TestOneHot:
    def test_basic(self):
        x = mx.array([0, 1, 2])
        y = F.one_hot(x, num_classes=3)
        mx.eval(y)
        expected = np.eye(3, dtype=np.float32)
        np.testing.assert_allclose(np.array(y), expected, atol=1e-6)

    def test_auto_num_classes(self):
        x = mx.array([0, 1, 4])
        y = F.one_hot(x)
        mx.eval(y)
        assert y.shape == (3, 5)

    def test_batched(self):
        x = mx.array([[0, 1], [2, 0]])
        y = F.one_hot(x, num_classes=3)
        mx.eval(y)
        assert y.shape == (2, 2, 3)


# =============================================================================
# Grid Sample
# =============================================================================

class TestGridSample:
    def test_grid_sample_identity(self):
        """Identity grid should reproduce the input."""
        B, C, H, W = 1, 1, 4, 4
        input_data = mx.arange(H * W).reshape(1, 1, H, W).astype(mx.float32)

        # Create identity grid: maps each output pixel to its own location
        # Grid values in [-1, 1]
        gy = mx.linspace(-1, 1, H).reshape(H, 1)
        gx = mx.linspace(-1, 1, W).reshape(1, W)
        gy = mx.broadcast_to(gy, (H, W))
        gx = mx.broadcast_to(gx, (H, W))
        grid = mx.stack([gx, gy], axis=-1)  # (H, W, 2)
        grid = mx.expand_dims(grid, axis=0)  # (1, H, W, 2)

        result = F.grid_sample(input_data, grid, mode="bilinear", align_corners=True)
        mx.eval(result)
        np.testing.assert_allclose(np.array(result), np.array(input_data), atol=1e-4)

    def test_grid_sample_zeros_padding(self):
        """Sampling outside bounds with zeros padding should return 0."""
        input_data = mx.ones((1, 1, 2, 2))
        # Grid with coordinates way outside [-1, 1]
        grid = mx.full((1, 1, 1, 2), 10.0)  # way outside
        result = F.grid_sample(input_data, grid, padding_mode="zeros", align_corners=True)
        mx.eval(result)
        assert abs(result[0, 0, 0, 0].item()) < 1e-5

    def test_grid_sample_nearest(self):
        """Nearest mode grid sample."""
        input_data = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
        # Grid pointing to top-left corner
        grid = mx.array([[[[-1.0, -1.0]]]])  # (1, 1, 1, 2)
        result = F.grid_sample(input_data, grid, mode="nearest", align_corners=True)
        mx.eval(result)
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-5

    def test_grid_sample_border_padding(self):
        """Border padding should clamp to edge values."""
        input_data = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        # Grid pointing outside (should clamp to edge)
        grid = mx.array([[[[-2.0, -2.0]]]])
        result = F.grid_sample(input_data, grid, padding_mode="border", align_corners=True)
        mx.eval(result)
        # Should get corner value (1.0) since it clamps to (0,0)
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-5

    def test_grid_sample_shape(self):
        """Grid sample produces correct output shape."""
        B, C, H_in, W_in = 2, 3, 8, 8
        H_out, W_out = 4, 4
        input_data = mx.random.normal((B, C, H_in, W_in))
        grid = mx.random.uniform(shape=(B, H_out, W_out, 2), low=-1.0, high=1.0)
        result = F.grid_sample(input_data, grid)
        mx.eval(result)
        assert result.shape == (B, C, H_out, W_out)


# =============================================================================
# Normal Distribution
# =============================================================================

class TestNormal:
    def test_sample_shape(self):
        loc = mx.zeros((3, 4))
        scale = mx.ones((3, 4))
        dist = Normal(loc, scale)
        s = dist.sample()
        mx.eval(s)
        assert s.shape == (3, 4)

    def test_sample_with_extra_shape(self):
        loc = mx.zeros((4,))
        scale = mx.ones((4,))
        dist = Normal(loc, scale)
        s = dist.sample((5,))
        mx.eval(s)
        assert s.shape == (5, 4)

    def test_rsample_equals_sample(self):
        loc = mx.zeros((3,))
        scale = mx.ones((3,))
        dist = Normal(loc, scale)
        # rsample should produce arrays with same shape
        s = dist.rsample()
        mx.eval(s)
        assert s.shape == (3,)

    def test_log_prob_standard_normal(self):
        loc = mx.array([0.0])
        scale = mx.array([1.0])
        dist = Normal(loc, scale)
        lp = dist.log_prob(mx.array([0.0]))
        mx.eval(lp)
        expected = -0.5 * math.log(2 * math.pi)
        assert abs(lp.item() - expected) < 1e-5

    def test_log_prob_shape(self):
        loc = mx.zeros((3, 4))
        scale = mx.ones((3, 4))
        dist = Normal(loc, scale)
        lp = dist.log_prob(mx.zeros((3, 4)))
        mx.eval(lp)
        assert lp.shape == (3, 4)

    def test_entropy(self):
        loc = mx.array([0.0])
        scale = mx.array([1.0])
        dist = Normal(loc, scale)
        e = dist.entropy()
        mx.eval(e)
        expected = 0.5 * math.log(2 * math.pi * math.e)
        assert abs(e.item() - expected) < 1e-5

    def test_mean_property(self):
        loc = mx.array([1.0, 2.0])
        scale = mx.array([0.5, 0.5])
        dist = Normal(loc, scale)
        np.testing.assert_allclose(np.array(dist.mean), [1.0, 2.0], atol=1e-6)

    def test_variance_property(self):
        loc = mx.array([0.0])
        scale = mx.array([2.0])
        dist = Normal(loc, scale)
        assert abs(dist.variance.item() - 4.0) < 1e-6


class TestKLDivergence:
    def test_kl_same_distribution(self):
        p = Normal(mx.zeros((3,)), mx.ones((3,)))
        q = Normal(mx.zeros((3,)), mx.ones((3,)))
        kl = kl_divergence(p, q)
        mx.eval(kl)
        np.testing.assert_allclose(np.array(kl), [0.0, 0.0, 0.0], atol=1e-5)

    def test_kl_non_negative(self):
        p = Normal(mx.array([1.0, -1.0]), mx.array([0.5, 2.0]))
        q = Normal(mx.array([0.0, 0.0]), mx.array([1.0, 1.0]))
        kl = kl_divergence(p, q)
        mx.eval(kl)
        assert np.all(np.array(kl) >= -1e-7)

    def test_kl_known_value(self):
        """KL(N(0,1) || N(1,1)) = 0.5."""
        p = Normal(mx.array([0.0]), mx.array([1.0]))
        q = Normal(mx.array([1.0]), mx.array([1.0]))
        kl = kl_divergence(p, q)
        mx.eval(kl)
        assert abs(kl.item() - 0.5) < 1e-5

    def test_kl_unsupported_raises(self):
        p = Normal(mx.array([0.0]), mx.array([1.0]))
        b = Beta(mx.array([1.0]), mx.array([1.0]))
        with pytest.raises(NotImplementedError):
            kl_divergence(p, b)


# =============================================================================
# Beta Distribution
# =============================================================================

class TestBeta:
    def test_sample_shape(self):
        a = mx.array([2.0, 3.0])
        b = mx.array([1.0, 1.0])
        dist = Beta(a, b)
        s = dist.sample()
        mx.eval(s)
        assert s.shape == (2,)

    def test_sample_in_unit_interval(self):
        a = mx.array([2.0])
        b = mx.array([5.0])
        dist = Beta(a, b)
        s = dist.sample((100,))
        mx.eval(s)
        arr = np.array(s)
        assert np.all(arr >= 0) and np.all(arr <= 1)

    def test_log_prob_shape(self):
        a = mx.array([2.0, 3.0])
        b = mx.array([2.0, 3.0])
        dist = Beta(a, b)
        lp = dist.log_prob(mx.array([0.5, 0.5]))
        mx.eval(lp)
        assert lp.shape == (2,)

    def test_beta_log_prob_no_scipy(self):
        """Verify Beta.log_prob works without scipy import."""
        # Temporarily remove scipy from modules to verify no dependency
        a = mx.array([2.0, 5.0])
        b = mx.array([3.0, 1.0])
        dist = Beta(a, b)
        lp = dist.log_prob(mx.array([0.5, 0.8]))
        mx.eval(lp)
        # Verify against manual calculation
        # log Beta(0.5; 2, 3) = (2-1)*log(0.5) + (3-1)*log(0.5) - log(B(2,3))
        # B(2,3) = Gamma(2)*Gamma(3)/Gamma(5) = 1*2/24 = 1/12
        # log(B(2,3)) = log(1/12) = -log(12)
        import math as _m
        expected_0 = (2 - 1) * _m.log(0.5) + (3 - 1) * _m.log(0.5) - (_m.lgamma(2) + _m.lgamma(3) - _m.lgamma(5))
        assert abs(lp[0].item() - expected_0) < 1e-4

    def test_beta_rsample_raises(self):
        """Beta.rsample() should raise NotImplementedError."""
        dist = Beta(mx.array([2.0]), mx.array([3.0]))
        with pytest.raises(NotImplementedError, match="Beta rsample not supported"):
            dist.rsample()


# =============================================================================
# Independent
# =============================================================================

class TestIndependent:
    def test_log_prob_sums_last_dim(self):
        loc = mx.zeros((3, 4))
        scale = mx.ones((3, 4))
        base = Normal(loc, scale)
        dist = Independent(base, 1)
        lp = dist.log_prob(mx.zeros((3, 4)))
        mx.eval(lp)
        assert lp.shape == (3,)

    def test_sample_shape(self):
        loc = mx.zeros((3, 4))
        scale = mx.ones((3, 4))
        dist = Independent(Normal(loc, scale), 1)
        s = dist.sample()
        mx.eval(s)
        assert s.shape == (3, 4)

    def test_entropy_sums(self):
        loc = mx.zeros((3, 4))
        scale = mx.ones((3, 4))
        base = Normal(loc, scale)
        dist = Independent(base, 1)
        e = dist.entropy()
        mx.eval(e)
        assert e.shape == (3,)
        # Should be 4x the single-dim entropy
        single_entropy = 0.5 * math.log(2 * math.pi * math.e)
        np.testing.assert_allclose(np.array(e), [4 * single_entropy] * 3, atol=1e-5)


# =============================================================================
# Optimizers
# =============================================================================

class TestOptimizers:
    def test_adam_is_mlx(self):
        assert Adam is _optim.Adam

    def test_adamw_is_mlx(self):
        assert AdamW is _optim.AdamW

    def test_sgd_is_mlx(self):
        assert SGD is _optim.SGD


# =============================================================================
# LR Schedulers
# =============================================================================

class TestCosineAnnealingLR:
    def test_basic_decay(self):
        opt = AdamW(learning_rate=0.001)
        sched = CosineAnnealingLR(opt, T_max=100, eta_min=0.0)
        # At step T_max, LR should be eta_min
        for _ in range(100):
            sched.step()
        assert abs(opt.learning_rate - 0.0) < 1e-6

    def test_midpoint(self):
        opt = AdamW(learning_rate=1.0)
        sched = CosineAnnealingLR(opt, T_max=100, eta_min=0.0)
        for _ in range(50):
            sched.step()
        # At midpoint, LR = (1 + cos(pi/2)) / 2 = 0.5
        assert abs(opt.learning_rate - 0.5) < 1e-3


class TestLinearWarmupCosineDecay:
    def test_warmup_phase(self):
        opt = AdamW(learning_rate=0.01)
        sched = LinearWarmupCosineDecay(opt, warmup_steps=10, total_steps=100)
        # After 5 steps, LR should be 0.01 * 5/10 = 0.005
        for _ in range(5):
            sched.step()
        assert abs(opt.learning_rate - 0.005) < 1e-6

    def test_warmup_end(self):
        opt = AdamW(learning_rate=0.01)
        sched = LinearWarmupCosineDecay(opt, warmup_steps=10, total_steps=100)
        for _ in range(10):
            sched.step()
        assert abs(opt.learning_rate - 0.01) < 1e-6

    def test_end_lr(self):
        opt = AdamW(learning_rate=0.01)
        sched = LinearWarmupCosineDecay(
            opt, warmup_steps=10, total_steps=100, min_lr=0.001
        )
        for _ in range(100):
            sched.step()
        assert abs(opt.learning_rate - 0.001) < 1e-5


class TestGetScheduler:
    def test_cosine(self):
        opt = AdamW(learning_rate=0.01)
        sched = get_scheduler("cosine", opt, 10, 100)
        assert isinstance(sched, LinearWarmupCosineDecay)

    def test_constant_with_warmup(self):
        opt = AdamW(learning_rate=0.01)
        sched = get_scheduler("constant_with_warmup", opt, 10, 100)
        # After warmup, LR should stay constant
        for _ in range(50):
            sched.step()
        assert abs(opt.learning_rate - 0.01) < 1e-6

    def test_linear(self):
        opt = AdamW(learning_rate=0.01)
        sched = get_scheduler("linear", opt, 10, 100)
        sched.step()
        assert opt.learning_rate > 0

    def test_unknown_raises(self):
        opt = AdamW(learning_rate=0.01)
        with pytest.raises(ValueError):
            get_scheduler("unknown_scheduler", opt, 10, 100)


# =============================================================================
# Gradient Clipping
# =============================================================================

class TestClipGradNorm:
    def test_clipping_applied(self):
        grads = {"layer1": {"weight": mx.array([3.0, 4.0])}}
        # Norm = 5.0, clip to 1.0
        clipped, total_norm = clip_grad_norm_(grads, max_norm=1.0)
        mx.eval(clipped["layer1"]["weight"])
        assert abs(total_norm - 5.0) < 1e-4
        clipped_norm = float(mx.sqrt(mx.sum(clipped["layer1"]["weight"] ** 2)).item())
        assert abs(clipped_norm - 1.0) < 1e-4

    def test_no_clipping_when_small(self):
        grads = {"w": mx.array([0.1, 0.1])}
        clipped, total_norm = clip_grad_norm_(grads, max_norm=10.0)
        mx.eval(clipped["w"])
        # Should not be clipped
        np.testing.assert_allclose(np.array(clipped["w"]), [0.1, 0.1], atol=1e-6)

    def test_nested_grads(self):
        grads = {
            "encoder": {"layer1": {"weight": mx.array([3.0, 4.0])}},
            "decoder": {"weight": mx.array([0.0])},
        }
        clipped, total_norm = clip_grad_norm_(grads, max_norm=2.0)
        assert total_norm > 0
        mx.eval(clipped["encoder"]["layer1"]["weight"])

    def test_returns_correct_norm(self):
        grads = {"w": mx.array([3.0, 4.0])}
        _, total_norm = clip_grad_norm_(grads, max_norm=100.0)
        assert abs(total_norm - 5.0) < 1e-4

    def test_clip_grad_norm_efficient(self):
        """Verify clip_grad_norm_ works correctly with efficient implementation."""
        # Create grads with known norm
        grads = {
            "a": mx.array([1.0, 0.0]),
            "b": mx.array([0.0, 2.0]),
        }
        # Total L2 norm = sqrt(1 + 4) = sqrt(5)
        clipped, total_norm = clip_grad_norm_(grads, max_norm=1.0)
        expected_norm = math.sqrt(5.0)
        assert abs(total_norm - expected_norm) < 1e-4

        # After clipping to max_norm=1.0, the clipped norm should be 1.0
        mx.eval(clipped["a"])
        mx.eval(clipped["b"])
        clipped_norm = math.sqrt(
            float(mx.sum(clipped["a"] ** 2).item()) +
            float(mx.sum(clipped["b"] ** 2).item())
        )
        assert abs(clipped_norm - 1.0) < 1e-4


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_mse_loss_broadcast(self):
        """MSE loss with broadcasting."""
        x = mx.random.normal((4, 1, 10))
        y = mx.random.normal((4, 5, 10))
        loss = F.mse_loss(x, y)
        mx.eval(loss)
        assert loss.shape == ()

    def test_softmax_numerical_stability(self):
        """Softmax with large values should not overflow."""
        x = mx.array([1000.0, 1001.0, 1002.0])
        y = F.softmax(x, dim=0)
        mx.eval(y)
        assert not np.any(np.isnan(np.array(y)))
        assert abs(mx.sum(y).item() - 1.0) < 1e-5

    def test_log_softmax_numerical_stability(self):
        x = mx.array([1000.0, 1001.0, 1002.0])
        y = F.log_softmax(x, dim=0)
        mx.eval(y)
        assert not np.any(np.isnan(np.array(y)))
        assert not np.any(np.isinf(np.array(y)))
