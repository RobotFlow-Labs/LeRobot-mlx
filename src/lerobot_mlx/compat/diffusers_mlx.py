# Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
# LeRobot-MLX: DDPM/DDIM noise schedulers (diffusers replacement)
#
# Pure-math implementations of the diffusion schedulers used by
# LeRobot's Diffusion Policy. Compatible with the diffusers API.

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np


__all__ = ["DDPMScheduler", "DDIMScheduler", "SchedulerOutput"]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class SchedulerOutput:
    """Output of a scheduler step."""
    prev_sample: mx.array
    pred_original_sample: mx.array


# ---------------------------------------------------------------------------
# Beta schedule helpers
# ---------------------------------------------------------------------------

def _cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> np.ndarray:
    """Squared-cosine beta schedule (squaredcos_cap_v2).

    From "Improved Denoising Diffusion Probabilistic Models"
    (Nichol & Dhariwal, 2021).
    """
    steps = num_timesteps + 1
    t = np.linspace(0, num_timesteps, steps, dtype=np.float64) / num_timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0, 0.999).astype(np.float32)


def _linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> np.ndarray:
    """Linear beta schedule."""
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)


# ---------------------------------------------------------------------------
# DDPM Scheduler
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Models scheduler.

    Matches the ``diffusers.DDPMScheduler`` API subset used by
    LeRobot's diffusion policy.

    Args:
        num_train_timesteps: Number of diffusion steps during training.
        beta_start: Starting beta value (for linear schedule).
        beta_end: Ending beta value (for linear schedule).
        beta_schedule: Schedule type ('linear' or 'squaredcos_cap_v2').
        clip_sample: Whether to clip predicted x_0 to [-1, 1].
        prediction_type: Model prediction type ('epsilon' or 'sample').
        clip_sample_range: Range for clipping (default 1.0).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        clip_sample_range: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # Compute beta schedule
        if beta_schedule == "squaredcos_cap_v2":
            betas_np = _cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == "linear":
            betas_np = _linear_beta_schedule(num_train_timesteps, beta_start, beta_end)
        else:
            raise ValueError(
                f"Unknown beta schedule '{beta_schedule}'. "
                "Use 'linear' or 'squaredcos_cap_v2'."
            )

        self.betas = mx.array(betas_np)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)

        # Pre-compute commonly used quantities
        self.sqrt_alphas_cumprod = mx.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - self.alphas_cumprod)

        # Default timesteps (overridden by set_timesteps)
        self.timesteps = mx.arange(num_train_timesteps - 1, -1, -1)

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.

        Args:
            original_samples: Clean samples x_0, shape (B, ...).
            noise: Gaussian noise, same shape as original_samples.
            timesteps: Integer timestep indices, shape (B,) or scalar.

        Returns:
            Noisy samples x_t.
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Expand dims to match sample shape for broadcasting
        while sqrt_alpha_prod.ndim < original_samples.ndim:
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = mx.expand_dims(sqrt_one_minus_alpha_prod, axis=-1)

        noisy = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
    ) -> SchedulerOutput:
        """Reverse diffusion step: p(x_{t-1} | x_t).

        Args:
            model_output: Model prediction (noise or sample).
            timestep: Current integer timestep t.
            sample: Current noisy sample x_t.

        Returns:
            SchedulerOutput with prev_sample and pred_original_sample.
        """
        t = int(timestep)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else mx.array(1.0)
        beta_prod_t = 1.0 - alpha_prod_t

        # Predict x_0
        if self.prediction_type == "epsilon":
            pred_original = (
                sample - mx.sqrt(beta_prod_t) * model_output
            ) / mx.sqrt(alpha_prod_t)
        elif self.prediction_type == "sample":
            pred_original = model_output
        elif self.prediction_type == "v_prediction":
            # v = sqrt(alpha_prod) * noise - sqrt(1 - alpha_prod) * sample
            # => x_0 = sqrt(alpha_prod) * sample - sqrt(1 - alpha_prod) * v
            pred_original = (
                mx.sqrt(alpha_prod_t) * sample - mx.sqrt(beta_prod_t) * model_output
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        if self.clip_sample:
            pred_original = mx.clip(
                pred_original, -self.clip_sample_range, self.clip_sample_range
            )

        # At t=0, return the predicted original directly (no blending needed)
        if t == 0:
            return SchedulerOutput(
                prev_sample=pred_original,
                pred_original_sample=pred_original,
            )

        # Compute coefficients for x_{t-1} (safe: t > 0, so beta_prod_t > 0)
        pred_original_coeff = (
            mx.sqrt(alpha_prod_t_prev) * self.betas[t] / beta_prod_t
        )
        current_coeff = (
            mx.sqrt(self.alphas[t]) * (1.0 - alpha_prod_t_prev) / beta_prod_t
        )
        pred_prev = pred_original_coeff * pred_original + current_coeff * sample

        # Add noise for t > 0
        noise = mx.random.normal(sample.shape)
        variance = (
            self.betas[t] * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
        )
        pred_prev = pred_prev + mx.sqrt(variance) * noise

        return SchedulerOutput(
            prev_sample=pred_prev,
            pred_original_sample=pred_original,
        )

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set the discrete timesteps for inference.

        Spaces timesteps evenly across the training range.

        Args:
            num_inference_steps: Number of denoising steps.
        """
        # Match diffusers: start from highest timestep and descend evenly
        import numpy as _np
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = _np.arange(0, self.num_train_timesteps)[::-1][::step_ratio].tolist()
        self.timesteps = mx.array(timesteps)
        self.num_inference_steps = num_inference_steps


# ---------------------------------------------------------------------------
# DDIM Scheduler
# ---------------------------------------------------------------------------

class DDIMScheduler:
    """Denoising Diffusion Implicit Models scheduler.

    Deterministic when eta=0 (default). Matches the ``diffusers.DDIMScheduler``
    API subset used by LeRobot's diffusion policy.

    Args:
        num_train_timesteps: Number of diffusion steps during training.
        beta_start: Starting beta value (for linear schedule).
        beta_end: Ending beta value (for linear schedule).
        beta_schedule: Schedule type ('linear' or 'squaredcos_cap_v2').
        clip_sample: Whether to clip predicted x_0 to [-1, 1].
        prediction_type: Model prediction type ('epsilon' or 'sample').
        clip_sample_range: Range for clipping (default 1.0).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        clip_sample_range: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # Compute beta schedule (reuse the same functions)
        if beta_schedule == "squaredcos_cap_v2":
            betas_np = _cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == "linear":
            betas_np = _linear_beta_schedule(num_train_timesteps, beta_start, beta_end)
        else:
            raise ValueError(
                f"Unknown beta schedule '{beta_schedule}'. "
                "Use 'linear' or 'squaredcos_cap_v2'."
            )

        self.betas = mx.array(betas_np)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)

        # Pre-compute
        self.sqrt_alphas_cumprod = mx.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - self.alphas_cumprod)

        # Default timesteps
        self.timesteps = mx.arange(num_train_timesteps - 1, -1, -1)

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """Forward diffusion: same formula as DDPM.

        Args:
            original_samples: Clean samples x_0.
            noise: Gaussian noise.
            timesteps: Integer timestep indices.

        Returns:
            Noisy samples x_t.
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while sqrt_alpha_prod.ndim < original_samples.ndim:
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = mx.expand_dims(sqrt_one_minus_alpha_prod, axis=-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        eta: float = 0.0,
    ) -> SchedulerOutput:
        """DDIM step: deterministic when eta=0, stochastic otherwise.

        Args:
            model_output: Model prediction.
            timestep: Current integer timestep t.
            sample: Current noisy sample x_t.
            eta: Noise injection factor (0 = deterministic DDIM).

        Returns:
            SchedulerOutput with prev_sample and pred_original_sample.
        """
        t = int(timestep)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else mx.array(1.0)

        # Predict x_0
        if self.prediction_type == "epsilon":
            pred_original = (
                sample - mx.sqrt(1.0 - alpha_prod_t) * model_output
            ) / mx.sqrt(alpha_prod_t)
        elif self.prediction_type == "sample":
            pred_original = model_output
        elif self.prediction_type == "v_prediction":
            # v = sqrt(alpha_prod) * noise - sqrt(1 - alpha_prod) * sample
            # => x_0 = sqrt(alpha_prod) * sample - sqrt(1 - alpha_prod) * v
            pred_original = (
                mx.sqrt(alpha_prod_t) * sample - mx.sqrt(1.0 - alpha_prod_t) * model_output
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        if self.clip_sample:
            pred_original = mx.clip(
                pred_original, -self.clip_sample_range, self.clip_sample_range
            )

        # Compute "predicted epsilon" from x_0 prediction
        pred_epsilon = (
            sample - mx.sqrt(alpha_prod_t) * pred_original
        ) / mx.sqrt(1.0 - alpha_prod_t)

        # DDIM formula
        if eta > 0.0 and t > 0:
            # Stochastic DDIM (eta > 0)
            variance = (
                (1.0 - alpha_prod_t_prev)
                / (1.0 - alpha_prod_t)
                * (1.0 - alpha_prod_t / alpha_prod_t_prev)
            )
            sigma = eta * mx.sqrt(variance)
            noise = mx.random.normal(sample.shape)
            pred_prev = (
                mx.sqrt(alpha_prod_t_prev) * pred_original
                + mx.sqrt(mx.maximum(1.0 - alpha_prod_t_prev - sigma**2, mx.array(0.0))) * pred_epsilon
                + sigma * noise
            )
        else:
            # Deterministic DDIM (eta = 0)
            pred_prev = (
                mx.sqrt(alpha_prod_t_prev) * pred_original
                + mx.sqrt(1.0 - alpha_prod_t_prev) * pred_epsilon
            )

        return SchedulerOutput(
            prev_sample=pred_prev,
            pred_original_sample=pred_original,
        )

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set the discrete timesteps for inference.

        Args:
            num_inference_steps: Number of denoising steps.
        """
        # Match diffusers: start from highest timestep and descend evenly
        import numpy as _np
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = _np.arange(0, self.num_train_timesteps)[::-1][::step_ratio].tolist()
        self.timesteps = mx.array(timesteps)
        self.num_inference_steps = num_inference_steps
