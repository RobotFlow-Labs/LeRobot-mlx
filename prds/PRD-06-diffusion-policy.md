# PRD-06: Diffusion Policy

> **Status:** TODO
> **Priority:** P0 — Second most-used policy, iconic paper
> **Dependencies:** PRD-02, PRD-03, PRD-04 (especially DDPMScheduler)
> **Estimated LOC:** ~500 (mirror of upstream's ~1138 LOC)
> **Phase:** 2 (First Policy)

---

## Objective

Port the Diffusion Policy — DDPM-based visuomotor policy that generates action sequences via iterative denoising. Uses UNet/Transformer architecture with noise schedulers from diffusers.

---

## Upstream Files to Mirror

| Upstream File | Treatment |
|---------------|-----------|
| `policies/diffusion/configuration_diffusion.py` | **COPY VERBATIM** |
| `policies/diffusion/modeling_diffusion.py` | **PORT** |

---

## Architecture

```
Diffusion Policy:
  Training:
    1. Sample action sequence from dataset
    2. Add noise at random timestep t → noisy_actions
    3. Predict noise from: UNet(noisy_actions, t, observations)
    4. Loss = MSE(predicted_noise, actual_noise)

  Inference:
    1. Start with pure noise x_T ~ N(0, I)
    2. For t = T, T-1, ..., 0:
       predicted_noise = UNet(x_t, t, observations)
       x_{t-1} = scheduler.step(predicted_noise, t, x_t)
    3. Return x_0 as action sequence
```

## Key Dependencies

| Component | Upstream | Our Replacement |
|-----------|----------|----------------|
| UNet backbone | Custom 1D UNet (temporal) | Port via compat/ |
| Noise scheduler | `diffusers.DDPMScheduler` | `compat/diffusers_mlx.DDPMScheduler` (PRD-04) |
| Vision backbone | ResNet18 (torchvision) | `compat/vision.resnet18` (PRD-04) |
| Sinusoidal embeddings | Custom torch impl | MLX reimplementation |
| GroupNorm | `torch.nn.GroupNorm` | `mlx.nn.GroupNorm` |
| Conv1d blocks | `torch.nn.Conv1d` | `mlx.nn.Conv1d` |
| einops | `rearrange` patterns | `compat/einops_mlx` |

## Specific Porting Challenges

1. **Temporal 1D UNet**: Uses Conv1d + GroupNorm + SiLU blocks — all available in MLX
2. **Sinusoidal timestep embedding**: Pure math, easy port
3. **Denoising loop**: Iterative process calling scheduler.step() — straightforward
4. **Conditional generation**: Cross-attention between UNet and observation features
5. **EMA (Exponential Moving Average)**: `torch.optim.swa_utils` → custom MLX impl

## Test Plan

```python
class TestDiffusionForward:
    def test_noise_prediction_shape(self):
        """UNet predicts noise with same shape as input."""

    def test_denoising_loop(self):
        """Full denoising from pure noise produces finite actions."""

    def test_loss_decreases(self):
        """Training loss decreases over 10 steps on synthetic data."""

class TestUNet:
    def test_unet_forward(self):
        """1D temporal UNet produces correct output shape."""

    def test_timestep_embedding(self):
        """Sinusoidal embedding produces correct dims."""

class TestSchedulerIntegration:
    def test_add_noise_roundtrip(self):
        """add_noise → denoise loop approximately recovers original."""
```

---

## Acceptance Criteria

1. UNet forward pass: correct shape for `(B, T, action_dim)` input
2. Full denoising loop: pure noise → finite action sequence
3. Training loss decreases on synthetic data (10 steps)
4. Scheduler integration: noise addition + removal works
5. EMA weight tracking works
6. 20+ tests passing
