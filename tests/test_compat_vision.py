# Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
# Tests for PRD-04: vision backbone, einops, and diffusers schedulers

import math

import mlx.core as mx
import numpy as np
import pytest


# ===================================================================
# Vision (ResNet) tests
# ===================================================================

class TestChannelTranspose:
    """Test NCHW <-> NHWC channel format conversions."""

    def test_channel_first_to_last(self):
        from lerobot_mlx.compat.vision import _channel_first_to_last
        x = mx.random.normal((2, 3, 8, 8))  # NCHW
        y = _channel_first_to_last(x)
        assert y.shape == (2, 8, 8, 3)

    def test_channel_last_to_first(self):
        from lerobot_mlx.compat.vision import _channel_last_to_first
        x = mx.random.normal((2, 8, 8, 3))  # NHWC
        y = _channel_last_to_first(x)
        assert y.shape == (2, 3, 8, 8)

    def test_channel_roundtrip(self):
        from lerobot_mlx.compat.vision import _channel_first_to_last, _channel_last_to_first
        x = mx.random.normal((2, 3, 16, 16))
        y = _channel_last_to_first(_channel_first_to_last(x))
        np.testing.assert_allclose(np.array(x), np.array(y), atol=1e-6)

    def test_channel_transpose_values(self):
        from lerobot_mlx.compat.vision import _channel_first_to_last
        x = mx.arange(24).reshape(1, 3, 2, 4)  # NCHW
        y = _channel_first_to_last(x)
        # y[0, h, w, c] == x[0, c, h, w]
        assert y[0, 0, 0, 0].item() == x[0, 0, 0, 0].item()
        assert y[0, 0, 0, 1].item() == x[0, 1, 0, 0].item()
        assert y[0, 1, 2, 2].item() == x[0, 2, 1, 2].item()


class TestMaxPool2d:
    """Test manual max pooling implementation."""

    def test_output_shape_basic(self):
        from lerobot_mlx.compat.vision import _max_pool_2d
        x = mx.random.normal((1, 8, 8, 3))  # NHWC
        y = _max_pool_2d(x, kernel_size=3, stride=2, padding=1)
        assert y.shape == (1, 4, 4, 3)

    def test_output_shape_no_padding(self):
        from lerobot_mlx.compat.vision import _max_pool_2d
        x = mx.random.normal((2, 7, 7, 16))
        y = _max_pool_2d(x, kernel_size=3, stride=2, padding=0)
        oH = (7 - 3) // 2 + 1  # 3
        oW = (7 - 3) // 2 + 1  # 3
        assert y.shape == (2, oH, oW, 16)

    def test_maxpool_values(self):
        from lerobot_mlx.compat.vision import _max_pool_2d
        # 1x4x4x1 with known values, kernel=2, stride=2, no padding
        data = mx.array([[[[1], [2], [3], [4]],
                          [[5], [6], [7], [8]],
                          [[9], [10], [11], [12]],
                          [[13], [14], [15], [16]]]], dtype=mx.float32)
        y = _max_pool_2d(data, kernel_size=2, stride=2, padding=0)
        assert y.shape == (1, 2, 2, 1)
        expected = np.array([[[[6], [8]], [[14], [16]]]])
        np.testing.assert_allclose(np.array(y), expected)

    def test_maxpool_with_padding(self):
        from lerobot_mlx.compat.vision import _max_pool_2d
        x = mx.ones((1, 4, 4, 1))
        y = _max_pool_2d(x, kernel_size=3, stride=1, padding=1)
        assert y.shape == (1, 4, 4, 1)
        # All 1s input -> all 1s output (padding is -inf, max of 1s is 1)
        np.testing.assert_allclose(np.array(y), 1.0)

    def test_maxpool_batch(self):
        from lerobot_mlx.compat.vision import _max_pool_2d
        x = mx.random.normal((4, 16, 16, 8))
        y = _max_pool_2d(x, kernel_size=3, stride=2, padding=1)
        assert y.shape == (4, 8, 8, 8)

    def test_maxpool_112_to_56(self):
        """Test the typical ResNet stem pooling: 112x112 -> 56x56."""
        from lerobot_mlx.compat.vision import _max_pool_2d
        x = mx.random.normal((1, 112, 112, 64))
        y = _max_pool_2d(x, kernel_size=3, stride=2, padding=1)
        assert y.shape == (1, 56, 56, 64)

    def test_maxpool_uses_native_or_fallback(self):
        """_max_pool_2d should work regardless of nn.MaxPool2d availability."""
        from lerobot_mlx.compat.vision import _max_pool_2d
        x = mx.random.normal((1, 4, 4, 2))
        y = _max_pool_2d(x, kernel_size=2, stride=2, padding=0)
        assert y.shape == (1, 2, 2, 2)
        mx.eval(y)
        assert np.all(np.isfinite(np.array(y)))


class TestBasicBlock:
    """Test the ResNet basic residual block."""

    def test_basic_block_same_dims(self):
        from lerobot_mlx.compat.vision import _BasicBlock
        block = _BasicBlock(64, 64)
        x = mx.random.normal((1, 8, 8, 64))  # NHWC
        y = block(x)
        assert y.shape == (1, 8, 8, 64)

    def test_basic_block_with_downsample(self):
        from lerobot_mlx.compat.vision import _BasicBlock, _Downsample
        ds = _Downsample(64, 128, stride=2)
        block = _BasicBlock(64, 128, stride=2, downsample=ds)
        x = mx.random.normal((1, 8, 8, 64))
        y = block(x)
        assert y.shape == (1, 4, 4, 128)

    def test_basic_block_produces_finite(self):
        from lerobot_mlx.compat.vision import _BasicBlock
        block = _BasicBlock(32, 32)
        x = mx.random.normal((2, 4, 4, 32))
        y = block(x)
        mx.eval(y)
        assert np.all(np.isfinite(np.array(y)))


class TestResNet:
    """Test full ResNet model."""

    def test_resnet18_output_shape(self):
        from lerobot_mlx.compat.vision import resnet18
        model = resnet18(pretrained=False)
        x = mx.random.normal((2, 3, 224, 224))  # NCHW
        y = model(x)
        mx.eval(y)
        assert y.shape == (2, 1000)

    def test_resnet18_custom_num_classes(self):
        from lerobot_mlx.compat.vision import resnet18
        model = resnet18(pretrained=False, num_classes=512)
        x = mx.random.normal((1, 3, 224, 224))
        y = model(x)
        mx.eval(y)
        assert y.shape == (1, 512)

    def test_resnet34_output_shape(self):
        from lerobot_mlx.compat.vision import resnet34
        model = resnet34(pretrained=False)
        x = mx.random.normal((1, 3, 224, 224))
        y = model(x)
        mx.eval(y)
        assert y.shape == (1, 1000)

    def test_resnet18_forward_features(self):
        from lerobot_mlx.compat.vision import resnet18
        model = resnet18(pretrained=False)
        x = mx.random.normal((1, 3, 224, 224))
        final, features = model.forward_features(x)
        mx.eval(final)
        # After stem (conv7x7 s2 + maxpool s2): 224 -> 112 -> 56
        # layer1: 56x56, 64 channels
        assert features["layer1"].shape == (1, 56, 56, 64)
        # layer2: 28x28, 128 channels
        assert features["layer2"].shape == (1, 28, 28, 128)
        # layer3: 14x14, 256 channels
        assert features["layer3"].shape == (1, 14, 14, 256)
        # layer4: 7x7, 512 channels
        assert features["layer4"].shape == (1, 7, 7, 512)

    def test_resnet18_produces_finite(self):
        from lerobot_mlx.compat.vision import resnet18
        model = resnet18(pretrained=False)
        x = mx.random.normal((1, 3, 224, 224))
        y = model(x)
        mx.eval(y)
        assert np.all(np.isfinite(np.array(y)))

    @pytest.mark.slow
    def test_resnet18_pretrained_weights(self):
        """Test pretrained weight loading from HF Hub (requires internet)."""
        try:
            from huggingface_hub import hf_hub_download
            import safetensors
        except ImportError:
            pytest.skip("huggingface-hub and safetensors required for pretrained weight test")

        from lerobot_mlx.compat.vision import resnet18
        try:
            model = resnet18(pretrained=True)
        except RuntimeError:
            pytest.skip("Could not download pretrained weights (no internet?)")

        x = mx.random.normal((1, 3, 224, 224))
        y = model(x)
        mx.eval(y)
        assert y.shape == (1, 1000)
        assert np.all(np.isfinite(np.array(y)))

    def test_resnet_pretrained_invalid_name(self):
        """Unknown model name raises ValueError."""
        from lerobot_mlx.compat.vision import _load_pretrained_resnet, resnet18
        model = resnet18(pretrained=False)
        with pytest.raises(ValueError, match="No pretrained weights"):
            _load_pretrained_resnet(model, "resnet999")

    def test_resnet18_smaller_input(self):
        """ResNet should handle non-224 input sizes (used by some policies)."""
        from lerobot_mlx.compat.vision import resnet18
        model = resnet18(pretrained=False)
        x = mx.random.normal((1, 3, 128, 128))
        y = model(x)
        mx.eval(y)
        assert y.shape == (1, 1000)


# ===================================================================
# Einops tests
# ===================================================================

class TestRearrangeFlatten:
    """Test 'b c h w -> b (c h w)' pattern."""

    def test_flatten_shape(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.random.normal((2, 3, 4, 5))
        y = rearrange(x, "b c h w -> b (c h w)")
        assert y.shape == (2, 60)

    def test_flatten_values(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.arange(24).reshape(1, 2, 3, 4)
        y = rearrange(x, "b c h w -> b (c h w)")
        assert y.shape == (1, 24)
        np.testing.assert_allclose(np.array(y).flatten(), np.arange(24))


class TestRearrangeUnflatten:
    """Test 'b (h w) c -> b h w c' pattern."""

    def test_unflatten_shape(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.random.normal((2, 16, 8))
        y = rearrange(x, "b (h w) c -> b h w c", h=4, w=4)
        assert y.shape == (2, 4, 4, 8)

    def test_unflatten_values(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.arange(32).reshape(1, 8, 4)
        y = rearrange(x, "b (h w) c -> b h w c", h=2, w=4)
        assert y.shape == (1, 2, 4, 4)
        # y[0, 0, 0, :] should be x[0, 0, :]
        np.testing.assert_allclose(np.array(y[0, 0, 0]), np.array(x[0, 0]))


class TestRearrangeTranspose:
    """Test 'b c h w -> b h w c' pattern."""

    def test_transpose_shape(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.random.normal((2, 3, 8, 8))
        y = rearrange(x, "b c h w -> b h w c")
        assert y.shape == (2, 8, 8, 3)

    def test_transpose_values(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.arange(48).reshape(1, 3, 4, 4)
        y = rearrange(x, "b c h w -> b h w c")
        # y[0, h, w, c] == x[0, c, h, w]
        assert y[0, 0, 0, 0].item() == x[0, 0, 0, 0].item()
        assert y[0, 0, 0, 1].item() == x[0, 1, 0, 0].item()
        assert y[0, 2, 3, 2].item() == x[0, 2, 2, 3].item()


class TestRearrangeMergeBatch:
    """Test 'b t c -> (b t) c' pattern."""

    def test_merge_shape(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.random.normal((4, 10, 64))
        y = rearrange(x, "b t c -> (b t) c")
        assert y.shape == (40, 64)

    def test_merge_values(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.arange(24).reshape(2, 3, 4)
        y = rearrange(x, "b t c -> (b t) c")
        assert y.shape == (6, 4)
        np.testing.assert_allclose(np.array(y[0]), np.array(x[0, 0]))
        np.testing.assert_allclose(np.array(y[3]), np.array(x[1, 0]))


class TestRearrangeSplitBatch:
    """Test '(b t) c -> b t c' pattern."""

    def test_split_shape(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.random.normal((40, 64))
        y = rearrange(x, "(b t) c -> b t c", b=4, t=10)
        assert y.shape == (4, 10, 64)

    def test_split_values(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.arange(24).reshape(6, 4)
        y = rearrange(x, "(b t) c -> b t c", b=2, t=3)
        assert y.shape == (2, 3, 4)
        np.testing.assert_allclose(np.array(y[0, 0]), np.array(x[0]))
        np.testing.assert_allclose(np.array(y[1, 0]), np.array(x[3]))

    def test_split_infer_one_dim(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.random.normal((40, 64))
        y = rearrange(x, "(b t) c -> b t c", b=4)
        assert y.shape == (4, 10, 64)


class TestRepeatBroadcast:
    """Test '1 c -> b c' pattern."""

    def test_broadcast_shape(self):
        from lerobot_mlx.compat.einops_mlx import repeat
        x = mx.random.normal((1, 64))
        y = repeat(x, "1 c -> b c", b=8)
        assert y.shape == (8, 64)

    def test_broadcast_values(self):
        from lerobot_mlx.compat.einops_mlx import repeat
        x = mx.array([[1.0, 2.0, 3.0]])  # (1, 3)
        y = repeat(x, "1 c -> b c", b=4)
        assert y.shape == (4, 3)
        for i in range(4):
            np.testing.assert_allclose(np.array(y[i]), [1.0, 2.0, 3.0])


class TestRepeatNewAxis:
    """Test inserting a new dimension with repeat."""

    def test_repeat_insert_axis(self):
        from lerobot_mlx.compat.einops_mlx import repeat
        x = mx.random.normal((2, 64))
        y = repeat(x, "b c -> b t c", t=10)
        assert y.shape == (2, 10, 64)

    def test_repeat_insert_axis_values(self):
        from lerobot_mlx.compat.einops_mlx import repeat
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        y = repeat(x, "b c -> b t c", t=3)
        assert y.shape == (2, 3, 2)
        for t in range(3):
            np.testing.assert_allclose(np.array(y[0, t]), [1.0, 2.0])
            np.testing.assert_allclose(np.array(y[1, t]), [3.0, 4.0])


class TestEinopsEdgeCases:
    """Edge cases for einops."""

    def test_rearrange_no_change(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.random.normal((2, 3, 4))
        y = rearrange(x, "b c h -> b c h")
        assert y.shape == x.shape
        np.testing.assert_allclose(np.array(y), np.array(x))

    def test_rearrange_invalid_pattern(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.random.normal((2, 3))
        with pytest.raises(ValueError):
            rearrange(x, "b c")  # no '->'

    def test_rearrange_partial_flatten(self):
        from lerobot_mlx.compat.einops_mlx import rearrange
        x = mx.random.normal((2, 3, 4, 5))
        y = rearrange(x, "b c h w -> b c (h w)")
        assert y.shape == (2, 3, 20)


# ===================================================================
# Diffusers scheduler tests
# ===================================================================

class TestDDPMScheduler:
    """Test DDPM scheduler."""

    def test_init_default(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler()
        assert scheduler.num_train_timesteps == 1000
        assert scheduler.betas.shape == (1000,)
        assert scheduler.alphas.shape == (1000,)
        assert scheduler.alphas_cumprod.shape == (1000,)

    def test_init_linear(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler(beta_schedule="linear")
        assert scheduler.betas.shape == (1000,)

    def test_init_invalid_schedule(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        with pytest.raises(ValueError):
            DDPMScheduler(beta_schedule="unknown")

    def test_cosine_schedule_betas_in_range(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2")
        betas = np.array(scheduler.betas)
        assert np.all(betas > 0)
        assert np.all(betas < 1)

    def test_cosine_schedule_alphas_cumprod_decreasing(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2")
        ac = np.array(scheduler.alphas_cumprod)
        # alphas_cumprod should be monotonically decreasing
        assert np.all(np.diff(ac) <= 0)

    def test_linear_schedule_betas_increasing(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler(beta_schedule="linear")
        betas = np.array(scheduler.betas)
        assert np.all(np.diff(betas) >= 0)

    def test_add_noise_shape(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler()
        x = mx.random.normal((4, 32))
        noise = mx.random.normal((4, 32))
        timesteps = mx.array([100, 200, 500, 900])
        noisy = scheduler.add_noise(x, noise, timesteps)
        assert noisy.shape == (4, 32)

    def test_add_noise_more_noise_at_higher_t(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler()
        x = mx.zeros((1, 64))
        noise = mx.ones((1, 64))
        low_t = mx.array([10])
        high_t = mx.array([900])
        noisy_low = scheduler.add_noise(x, noise, low_t)
        noisy_high = scheduler.add_noise(x, noise, high_t)
        mx.eval(noisy_low, noisy_high)
        # Higher t => more noise contribution
        assert np.linalg.norm(np.array(noisy_high)) > np.linalg.norm(np.array(noisy_low))

    def test_add_noise_at_t0_nearly_clean(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler()
        x = mx.ones((1, 32))
        noise = mx.random.normal((1, 32))
        noisy = scheduler.add_noise(x, noise, mx.array([0]))
        mx.eval(noisy)
        # At t=0, alpha_bar ~ 1, so noisy ~ x
        np.testing.assert_allclose(np.array(noisy), np.array(x), atol=0.05)

    def test_step_produces_finite(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        mx.random.seed(42)
        scheduler = DDPMScheduler()
        sample = mx.random.normal((1, 32))
        model_output = mx.random.normal((1, 32))
        output = scheduler.step(model_output, 500, sample)
        mx.eval(output.prev_sample, output.pred_original_sample)
        assert np.all(np.isfinite(np.array(output.prev_sample)))
        assert np.all(np.isfinite(np.array(output.pred_original_sample)))

    def test_step_at_t0_no_noise(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler()
        sample = mx.random.normal((1, 32))
        model_output = mx.random.normal((1, 32))
        # At t=0, step should not add noise
        output1 = scheduler.step(model_output, 0, sample)
        output2 = scheduler.step(model_output, 0, sample)
        mx.eval(output1.prev_sample, output2.prev_sample)
        np.testing.assert_allclose(
            np.array(output1.prev_sample), np.array(output2.prev_sample)
        )

    def test_step_output_type(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler, SchedulerOutput
        scheduler = DDPMScheduler()
        output = scheduler.step(mx.zeros((1, 8)), 100, mx.zeros((1, 8)))
        assert isinstance(output, SchedulerOutput)
        assert hasattr(output, "prev_sample")
        assert hasattr(output, "pred_original_sample")

    def test_set_timesteps(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(50)
        assert len(scheduler.timesteps) == 50
        # Should be descending
        ts = np.array(scheduler.timesteps)
        assert np.all(np.diff(ts) < 0)

    def test_set_timesteps_100(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(100)
        assert len(scheduler.timesteps) == 100

    def test_prediction_type_sample(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        scheduler = DDPMScheduler(prediction_type="sample", clip_sample=False)
        sample = mx.random.normal((1, 16))
        model_output = mx.random.normal((1, 16))
        output = scheduler.step(model_output, 0, sample)
        mx.eval(output.pred_original_sample)
        # When prediction_type='sample', pred_original == model_output
        np.testing.assert_allclose(
            np.array(output.pred_original_sample),
            np.array(model_output),
            atol=1e-6,
        )


class TestDDIMScheduler:
    """Test DDIM scheduler."""

    def test_init_default(self):
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler()
        assert scheduler.num_train_timesteps == 1000

    def test_deterministic_eta0(self):
        """DDIM with eta=0 should be fully deterministic."""
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler()
        sample = mx.array(np.random.randn(1, 32).astype(np.float32))
        model_output = mx.array(np.random.randn(1, 32).astype(np.float32))

        out1 = scheduler.step(model_output, 500, sample, eta=0.0)
        out2 = scheduler.step(model_output, 500, sample, eta=0.0)
        mx.eval(out1.prev_sample, out2.prev_sample)
        np.testing.assert_allclose(
            np.array(out1.prev_sample), np.array(out2.prev_sample), atol=1e-6
        )

    def test_ddim_add_noise_shape(self):
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler()
        x = mx.random.normal((4, 32))
        noise = mx.random.normal((4, 32))
        noisy = scheduler.add_noise(x, noise, mx.array([100, 200, 500, 900]))
        assert noisy.shape == (4, 32)

    def test_ddim_step_finite(self):
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler()
        output = scheduler.step(
            mx.random.normal((1, 16)), 500, mx.random.normal((1, 16)), eta=0.0
        )
        mx.eval(output.prev_sample)
        assert np.all(np.isfinite(np.array(output.prev_sample)))

    def test_ddim_set_timesteps(self):
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler()
        scheduler.set_timesteps(20)
        assert len(scheduler.timesteps) == 20

    def test_ddim_multidim_samples(self):
        """Test with multi-dimensional samples (like action sequences)."""
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler()
        sample = mx.random.normal((2, 16, 7))  # batch=2, horizon=16, action_dim=7
        model_output = mx.random.normal((2, 16, 7))
        output = scheduler.step(model_output, 500, sample, eta=0.0)
        mx.eval(output.prev_sample)
        assert output.prev_sample.shape == (2, 16, 7)

    def test_ddim_stochastic_eta_positive(self):
        """DDIM with eta > 0 should add noise (non-deterministic)."""
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler(clip_sample=False)
        sample = mx.array(np.random.randn(1, 32).astype(np.float32))
        model_output = mx.array(np.random.randn(1, 32).astype(np.float32))

        # Two calls with eta=1.0 at t>0 should differ (due to random noise)
        mx.random.seed(0)
        out1 = scheduler.step(model_output, 500, sample, eta=1.0)
        mx.eval(out1.prev_sample)
        mx.random.seed(1)
        out2 = scheduler.step(model_output, 500, sample, eta=1.0)
        mx.eval(out2.prev_sample)
        # They should NOT be equal
        assert not np.allclose(np.array(out1.prev_sample), np.array(out2.prev_sample))

    def test_ddim_clip_sample(self):
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler(clip_sample=True, clip_sample_range=1.0)
        # Use large model output to force clipping
        sample = mx.ones((1, 8)) * 10.0
        model_output = mx.ones((1, 8)) * 10.0
        output = scheduler.step(model_output, 500, sample, eta=0.0)
        mx.eval(output.pred_original_sample)
        pred = np.array(output.pred_original_sample)
        assert np.all(pred >= -1.0 - 1e-6)
        assert np.all(pred <= 1.0 + 1e-6)


class TestSchedulerOutput:
    """Test the SchedulerOutput dataclass."""

    def test_scheduler_output_fields(self):
        from lerobot_mlx.compat.diffusers_mlx import SchedulerOutput
        out = SchedulerOutput(
            prev_sample=mx.zeros((1, 8)),
            pred_original_sample=mx.ones((1, 8)),
        )
        assert out.prev_sample.shape == (1, 8)
        assert out.pred_original_sample.shape == (1, 8)


class TestVPrediction:
    """Test v_prediction support in both schedulers."""

    def test_ddpm_v_prediction_step_finite(self):
        from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler
        mx.random.seed(42)
        scheduler = DDPMScheduler(prediction_type="v_prediction", clip_sample=False)
        sample = mx.random.normal((1, 32))
        model_output = mx.random.normal((1, 32))
        output = scheduler.step(model_output, 500, sample)
        mx.eval(output.prev_sample, output.pred_original_sample)
        assert np.all(np.isfinite(np.array(output.prev_sample)))
        assert np.all(np.isfinite(np.array(output.pred_original_sample)))

    def test_ddim_v_prediction_step_finite(self):
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler(prediction_type="v_prediction", clip_sample=False)
        sample = mx.random.normal((1, 32))
        model_output = mx.random.normal((1, 32))
        output = scheduler.step(model_output, 500, sample, eta=0.0)
        mx.eval(output.prev_sample, output.pred_original_sample)
        assert np.all(np.isfinite(np.array(output.prev_sample)))
        assert np.all(np.isfinite(np.array(output.pred_original_sample)))

    def test_ddim_v_prediction_deterministic(self):
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler(prediction_type="v_prediction", clip_sample=False)
        sample = mx.array(np.random.randn(1, 16).astype(np.float32))
        model_output = mx.array(np.random.randn(1, 16).astype(np.float32))
        out1 = scheduler.step(model_output, 500, sample, eta=0.0)
        out2 = scheduler.step(model_output, 500, sample, eta=0.0)
        mx.eval(out1.prev_sample, out2.prev_sample)
        np.testing.assert_allclose(
            np.array(out1.prev_sample), np.array(out2.prev_sample), atol=1e-6
        )


# ===================================================================
# Cross-module integration tests
# ===================================================================

class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_resnet_with_einops_transpose(self):
        """Verify einops transpose matches vision.py channel transpose."""
        from lerobot_mlx.compat.einops_mlx import rearrange
        from lerobot_mlx.compat.vision import _channel_first_to_last
        x = mx.random.normal((2, 3, 8, 8))
        y_einops = rearrange(x, "b c h w -> b h w c")
        y_vision = _channel_first_to_last(x)
        mx.eval(y_einops, y_vision)
        np.testing.assert_allclose(np.array(y_einops), np.array(y_vision), atol=1e-6)

    def test_scheduler_full_denoise_loop(self):
        """Run a short denoising loop to verify stability."""
        from lerobot_mlx.compat.diffusers_mlx import DDIMScheduler
        scheduler = DDIMScheduler(num_train_timesteps=100)
        scheduler.set_timesteps(10)

        # Start from pure noise
        x = mx.random.normal((1, 16))

        for t in np.array(scheduler.timesteps):
            # Fake model output (zero prediction = identity)
            model_output = mx.zeros_like(x)
            output = scheduler.step(model_output, int(t), x, eta=0.0)
            x = output.prev_sample
            mx.eval(x)

        assert np.all(np.isfinite(np.array(x)))
