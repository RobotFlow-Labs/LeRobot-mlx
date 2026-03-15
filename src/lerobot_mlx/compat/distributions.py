"""
torch.distributions → MLX probability distributions.

Implementations of Normal, Beta, Independent, and kl_divergence
used by ACT (CVAE), SAC (stochastic policy), and other LeRobot policies.

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

import math  # used in Normal.log_prob, Normal.entropy

import mlx.core as mx
import numpy as np


def _lgamma_array(x):
    """Vectorized lgamma using math.lgamma (no scipy dependency).

    Works with any NumPy version (np.lgamma was removed in NumPy 2.0).
    """
    x = np.asarray(x, dtype=np.float64)
    return np.vectorize(math.lgamma)(x)


def _log_beta(a, b):
    """Log of the Beta function using lgamma (no scipy dependency).

    B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)
    log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)

    Args:
        a: numpy array of alpha values.
        b: numpy array of beta values.

    Returns:
        numpy array of log-beta values.
    """
    return _lgamma_array(a) + _lgamma_array(b) - _lgamma_array(a + b)


class Normal:
    """torch.distributions.Normal → MLX implementation.

    Univariate normal distribution parameterized by loc (mean) and scale (std dev).

    Used by:
    - ACT policy (CVAE latent space)
    - SAC policy (stochastic Gaussian policy)
    """

    def __init__(self, loc: mx.array, scale: mx.array):
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape: tuple = ()) -> mx.array:
        """Draw samples from the distribution.

        Args:
            sample_shape: Additional batch dimensions prepended to the output.

        Returns:
            Samples of shape (*sample_shape, *batch_shape).
        """
        shape = tuple(sample_shape) + self.loc.shape
        eps = mx.random.normal(shape)
        return self.loc + self.scale * eps

    def rsample(self, sample_shape: tuple = ()) -> mx.array:
        """Reparameterized sample (gradients flow through).

        In MLX, all operations are differentiable by default when used inside
        a value_and_grad context, so rsample is identical to sample.
        """
        return self.sample(sample_shape)

    def log_prob(self, value: mx.array) -> mx.array:
        """Log probability density at value.

        log p(x) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
        """
        var = self.scale ** 2
        log_scale = mx.log(self.scale)
        return (
            -0.5 * ((value - self.loc) ** 2 / var)
            - log_scale
            - 0.5 * math.log(2 * math.pi)
        )

    def entropy(self) -> mx.array:
        """Differential entropy of the normal distribution.

        H = 0.5 * log(2*pi*e*sigma^2) = 0.5 + 0.5*log(2*pi) + log(sigma)
        """
        return 0.5 + 0.5 * math.log(2 * math.pi) + mx.log(self.scale)

    @property
    def mean(self) -> mx.array:
        """Distribution mean."""
        return self.loc

    @property
    def variance(self) -> mx.array:
        """Distribution variance."""
        return self.scale ** 2

    @property
    def stddev(self) -> mx.array:
        """Distribution standard deviation."""
        return self.scale


class Beta:
    """torch.distributions.Beta → MLX implementation.

    Beta distribution parameterized by concentration1 (alpha) and concentration0 (beta).
    Uses numpy fallback for sampling (not in training hot path).

    Used by SAC for bounded action spaces.

    NOTE: sample() uses numpy and is NOT differentiable. For training, only use
    log_prob() and analytical properties (mean, variance).
    """

    def __init__(self, concentration1: mx.array, concentration0: mx.array):
        self.concentration1 = concentration1  # alpha
        self.concentration0 = concentration0  # beta

    @property
    def alpha(self) -> mx.array:
        return self.concentration1

    @property
    def beta_param(self) -> mx.array:
        return self.concentration0

    def sample(self, sample_shape: tuple = ()) -> mx.array:
        """Draw samples using numpy fallback (Beta via gamma ratio).

        NOT differentiable — use only for exploration/evaluation, not training loss.
        """
        a = np.array(self.concentration1)
        b = np.array(self.concentration0)
        shape = tuple(sample_shape) + a.shape
        samples = np.random.beta(a, b, size=shape).astype(np.float32)
        return mx.array(samples)

    def rsample(self, sample_shape: tuple = ()) -> mx.array:
        """Reparameterized sample — NOT supported for Beta in MLX.

        Raises:
            NotImplementedError: Always. Beta rsample requires implicit
                reparameterization gradients which are not available in MLX.
        """
        raise NotImplementedError(
            "Beta rsample not supported in MLX. "
            "Use sample() for non-differentiable sampling only."
        )

    def log_prob(self, value: mx.array) -> mx.array:
        """Log probability density at value.

        log Beta(x; a, b) = (a-1)*log(x) + (b-1)*log(1-x) - log(B(a,b))

        Uses math.lgamma (no scipy dependency).
        """
        a = np.array(self.concentration1)
        b = np.array(self.concentration0)
        x = np.array(value)

        # Clamp x to avoid log(0)
        x = np.clip(x, 1e-8, 1 - 1e-8)
        lp = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - _log_beta(a, b)
        return mx.array(lp.astype(np.float32))

    @property
    def mean(self) -> mx.array:
        """Distribution mean: alpha / (alpha + beta)."""
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self) -> mx.array:
        """Distribution variance."""
        a = self.concentration1
        b = self.concentration0
        total = a + b
        return (a * b) / (total ** 2 * (total + 1))


class Independent:
    """torch.distributions.Independent → MLX implementation.

    Reinterprets batch dimensions as event dimensions by summing
    log_prob over the rightmost reinterpreted_batch_ndims dimensions.

    Example:
        base = Normal(loc=mx.zeros((3, 4)), scale=mx.ones((3, 4)))
        # base.log_prob(x) has shape (3, 4)
        dist = Independent(base, 1)
        # dist.log_prob(x) has shape (3,) — sums over last dim
    """

    def __init__(self, base_distribution, reinterpreted_batch_ndims: int):
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def log_prob(self, value: mx.array) -> mx.array:
        """Log probability, summing over reinterpreted batch dims."""
        lp = self.base_dist.log_prob(value)
        for _ in range(self.reinterpreted_batch_ndims):
            lp = mx.sum(lp, axis=-1)
        return lp

    def sample(self, sample_shape: tuple = ()) -> mx.array:
        """Draw samples (shape unchanged from base distribution)."""
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape: tuple = ()) -> mx.array:
        """Reparameterized sample."""
        return self.base_dist.rsample(sample_shape)

    def entropy(self) -> mx.array:
        """Entropy, summing over reinterpreted batch dims."""
        e = self.base_dist.entropy()
        for _ in range(self.reinterpreted_batch_ndims):
            e = mx.sum(e, axis=-1)
        return e

    @property
    def mean(self) -> mx.array:
        """Distribution mean (same as base)."""
        return self.base_dist.mean

    @property
    def variance(self) -> mx.array:
        """Distribution variance (same as base)."""
        return self.base_dist.variance


def kl_divergence(p, q) -> mx.array:
    """Compute KL divergence KL(p || q).

    Currently supports:
    - Normal vs Normal: closed-form KL divergence
      KL = 0.5 * (var_ratio + t1 - 1 - log(var_ratio))
      where var_ratio = (p.scale / q.scale)^2
            t1 = ((p.loc - q.loc) / q.scale)^2

    Args:
        p: Distribution p.
        q: Distribution q.

    Returns:
        KL divergence (element-wise for batched distributions).
    """
    if isinstance(p, Normal) and isinstance(q, Normal):
        var_ratio = (p.scale / q.scale) ** 2
        t1 = ((p.loc - q.loc) / q.scale) ** 2
        return 0.5 * (var_ratio + t1 - 1 - mx.log(var_ratio))

    raise NotImplementedError(
        f"KL divergence not implemented for {type(p).__name__} and {type(q).__name__}. "
        f"Only Normal-Normal KL is currently supported."
    )
