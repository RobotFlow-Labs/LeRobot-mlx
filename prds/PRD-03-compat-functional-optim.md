# PRD-03: Compatibility Layer — Functional, Optim, Distributions

> **Status:** TODO
> **Priority:** P0 — Required by all policies
> **Dependencies:** PRD-02
> **Estimated LOC:** ~600
> **Phase:** 1 (Foundation)

---

## Objective

Complete the compat layer with `functional.py` (F.*), `optim.py` (optimizers + LR schedulers), and `distributions.py` (probability distributions). These are needed across ALL policies.

---

## Deliverables

### 1. `compat/functional.py` — torch.nn.functional.* Mapping

All functional ops found in the upstream scan (sorted by usage frequency):

```python
import mlx.core as mx
import mlx.nn as _nn

# === Loss Functions ===
def mse_loss(input, target, reduction='mean'):
    """F.mse_loss — 17 uses across upstream."""
    diff = (input - target) ** 2
    if reduction == 'mean': return mx.mean(diff)
    if reduction == 'sum': return mx.sum(diff)
    return diff

def l1_loss(input, target, reduction='mean'):
    diff = mx.abs(input - target)
    if reduction == 'mean': return mx.mean(diff)
    if reduction == 'sum': return mx.sum(diff)
    return diff

def smooth_l1_loss(input, target, beta=1.0, reduction='mean'):
    diff = mx.abs(input - target)
    loss = mx.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduction == 'mean': return mx.mean(loss)
    if reduction == 'sum': return mx.sum(loss)
    return loss

def cross_entropy(input, target, weight=None, reduction='mean', label_smoothing=0.0):
    return _nn.losses.cross_entropy(input, target, reduction=reduction,
                                      label_smoothing=label_smoothing)

def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean'):
    return _nn.losses.binary_cross_entropy(input, target, reduction=reduction)

# === Activation Functions ===
def relu(x, inplace=False): return _nn.relu(x)
def gelu(x, approximate='none'): return _nn.gelu(x)
def silu(x, inplace=False): return _nn.silu(x)
def sigmoid(x): return mx.sigmoid(x)
def tanh(x): return mx.tanh(x)
def softmax(x, dim=-1): return mx.softmax(x, axis=dim)
def log_softmax(x, dim=-1): return mx.log(mx.softmax(x, axis=dim))
def softplus(x, beta=1.0, threshold=20.0): return _nn.softplus(x)
def elu(x, alpha=1.0, inplace=False): return _nn.elu(x, alpha)

# === Padding ===
def pad(input, pad_widths, mode='constant', value=0):
    """F.pad — 14 uses. Convert torch pad format to numpy/mlx format.

    torch pad: (left, right, top, bottom, ...) — reversed pairs
    mlx pad: ((before_0, after_0), (before_1, after_1), ...) — per axis
    """
    # Convert torch format to mlx format
    n = len(pad_widths) // 2
    ndim = input.ndim
    mlx_pad = [(0, 0)] * (ndim - n)
    for i in range(n - 1, -1, -1):
        mlx_pad.append((pad_widths[2*i], pad_widths[2*i+1]))
    return mx.pad(input, mlx_pad, constant_values=value)

# === Attention ===
def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                  dropout_p=0.0, is_causal=False):
    """F.scaled_dot_product_attention — 6 uses."""
    d_k = query.shape[-1]
    scores = (query @ mx.transpose(key, axes=(*range(key.ndim-2), -1, -2))) / (d_k ** 0.5)
    if is_causal:
        seq_len = query.shape[-2]
        causal_mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
        scores = scores + causal_mask
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ value

# === Normalization ===
def normalize(input, p=2.0, dim=-1, eps=1e-12):
    """F.normalize — L2 normalization."""
    norm = mx.linalg.norm(input, axis=dim, keepdims=True)
    return input / mx.maximum(norm, eps)

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    axes = tuple(range(-len(normalized_shape), 0))
    mean = mx.mean(input, axis=axes, keepdims=True)
    var = mx.var(input, axis=axes, keepdims=True)
    x = (input - mean) / mx.sqrt(var + eps)
    if weight is not None: x = x * weight
    if bias is not None: x = x + bias
    return x

def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """F.group_norm for diffusion models."""
    N, C = input.shape[:2]
    spatial = input.shape[2:]
    x = input.reshape(N, num_groups, C // num_groups, *spatial)
    axes = tuple(range(2, x.ndim))
    mean = mx.mean(x, axis=axes, keepdims=True)
    var = mx.var(x, axis=axes, keepdims=True)
    x = (x - mean) / mx.sqrt(var + eps)
    x = x.reshape(N, C, *spatial)
    if weight is not None:
        shape = [1, C] + [1] * len(spatial)
        x = x * weight.reshape(shape) + bias.reshape(shape)
    return x

# === Interpolation ===
def interpolate(input, size=None, scale_factor=None, mode='nearest'):
    """F.interpolate — 5 uses. Nearest and bilinear."""
    if mode == 'nearest':
        # Use repeat for nearest neighbor upsampling
        if scale_factor is not None:
            sf = int(scale_factor)
            # (B, C, H, W) → repeat along H and W
            x = mx.repeat(input, sf, axis=-2)
            x = mx.repeat(x, sf, axis=-1)
            return x
        # TODO: handle size parameter
    # bilinear: implement via grid sampling or numpy
    ...

# === One-hot ===
def one_hot(tensor, num_classes=-1):
    """F.one_hot — 4 uses."""
    if num_classes < 0:
        num_classes = int(mx.max(tensor).item()) + 1
    return mx.eye(num_classes)[tensor]

# === Grid Sample (for spatial transformer / diffusion) ===
def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    """F.grid_sample — used in diffusion policy spatial transforms."""
    # Complex: implement bilinear interpolation via gather
    # Can use numpy fallback initially
    ...
```

### 2. `compat/optim.py` — Optimizers + LR Schedulers

```python
import mlx.optimizers as _optim
import math

# === Optimizers (thin wrappers) ===
Adam = _optim.Adam
AdamW = _optim.AdamW
SGD = _optim.SGD

# === Learning Rate Schedulers ===
# Upstream uses diffusers.optimization.get_scheduler and torch.optim.lr_scheduler

class CosineAnnealingLR:
    """Cosine annealing with warm restarts."""
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.learning_rate
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.step_count / self.T_max)) / 2
        self.optimizer.learning_rate = lr

class LinearWarmupCosineDecay:
    """Linear warmup followed by cosine decay — most common in LeRobot."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.learning_rate
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / \
                       max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress)) / 2
        self.optimizer.learning_rate = lr

def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    """Drop-in for diffusers.optimization.get_scheduler."""
    if name == "cosine":
        return LinearWarmupCosineDecay(optimizer, num_warmup_steps, num_training_steps)
    if name == "linear":
        return LinearWarmupCosineDecay(optimizer, num_warmup_steps, num_training_steps, min_lr=0)
    if name == "constant_with_warmup":
        return LinearWarmupCosineDecay(optimizer, num_warmup_steps, num_training_steps,
                                        min_lr=optimizer.learning_rate)
    raise ValueError(f"Unknown scheduler: {name}")

# === Gradient Clipping ===
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """Clip gradient norm. Returns total norm."""
    from mlx.utils import tree_flatten
    if isinstance(parameters, dict):
        grads = [v for _, v in tree_flatten(parameters)]
    else:
        grads = list(parameters)

    total_norm_sq = sum(mx.sum(g ** 2).item() for g in grads if g is not None)
    total_norm = total_norm_sq ** (1.0 / norm_type)

    clip_coef = max_norm / max(total_norm, 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            if g is not None:
                g *= clip_coef  # Note: MLX arrays are functional, need tree_map
    return total_norm
```

### 3. `compat/distributions.py` — Probability Distributions

Used by ACT (CVAE), SAC (stochastic policy), and others:

```python
import mlx.core as mx
import math

class Normal:
    """torch.distributions.Normal → MLX implementation."""
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape=()):
        shape = sample_shape + self.loc.shape
        eps = mx.random.normal(shape)
        return self.loc + self.scale * eps

    def rsample(self, sample_shape=()):
        """Reparameterized sample (same as sample in MLX, grads flow through)."""
        return self.sample(sample_shape)

    def log_prob(self, value):
        var = self.scale ** 2
        log_scale = mx.log(self.scale)
        return -0.5 * ((value - self.loc) ** 2 / var + math.log(2 * math.pi)) - log_scale

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + mx.log(self.scale)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2

class Beta:
    """torch.distributions.Beta — used by SAC for bounded actions."""
    def __init__(self, concentration1, concentration0):
        self.alpha = concentration1
        self.beta_param = concentration0

    def sample(self, sample_shape=()):
        # Beta via gamma: X ~ Gamma(alpha), Y ~ Gamma(beta), X/(X+Y) ~ Beta
        # MLX doesn't have gamma directly, use numpy fallback
        import numpy as np
        shape = sample_shape + self.alpha.shape
        a = np.array(self.alpha)
        b = np.array(self.beta_param)
        samples = np.random.beta(a, b, size=shape).astype(np.float32)
        return mx.array(samples)

    def log_prob(self, value):
        # log Beta(x; a, b) = (a-1)log(x) + (b-1)log(1-x) - log(B(a,b))
        from scipy.special import betaln
        import numpy as np
        a, b = np.array(self.alpha), np.array(self.beta_param)
        x = np.array(value)
        lp = (a - 1) * np.log(x + 1e-8) + (b - 1) * np.log(1 - x + 1e-8) - betaln(a, b)
        return mx.array(lp.astype(np.float32))

def kl_divergence(p, q):
    """torch.distributions.kl_divergence for Normal distributions."""
    if isinstance(p, Normal) and isinstance(q, Normal):
        var_ratio = (p.scale / q.scale) ** 2
        t1 = ((p.loc - q.loc) / q.scale) ** 2
        return 0.5 * (var_ratio + t1 - 1 - mx.log(var_ratio))
    raise NotImplementedError(f"KL divergence not implemented for {type(p)}, {type(q)}")

class Independent:
    """Reinterprets batch dims as event dims."""
    def __init__(self, base_distribution, reinterpreted_batch_ndims):
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def log_prob(self, value):
        lp = self.base_dist.log_prob(value)
        for _ in range(self.reinterpreted_batch_ndims):
            lp = mx.sum(lp, axis=-1)
        return lp

    def sample(self, sample_shape=()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=()):
        return self.base_dist.rsample(sample_shape)

    def entropy(self):
        e = self.base_dist.entropy()
        for _ in range(self.reinterpreted_batch_ndims):
            e = mx.sum(e, axis=-1)
        return e
```

---

## Acceptance Criteria

1. `from lerobot_mlx.compat.functional import mse_loss, pad, scaled_dot_product_attention` — works
2. `from lerobot_mlx.compat.optim import AdamW, get_scheduler` — works
3. `from lerobot_mlx.compat.distributions import Normal, kl_divergence` — works
4. 60+ tests covering:
   - All loss functions (mse, l1, cross_entropy, smooth_l1)
   - Padding (constant, all torch pad format variations)
   - Attention (with and without masks, causal)
   - Normal distribution (sample, log_prob, kl_divergence)
   - Optimizer step + LR scheduler step
   - Gradient clipping
5. Cross-framework tests: loss values match PyTorch within atol=1e-5

---

## Notes

- `grid_sample` is complex — can stub initially with numpy fallback, optimize later
- `interpolate` bilinear mode can use numpy initially
- Beta distribution uses numpy/scipy fallback — acceptable for sampling (not in training loop hot path)
- LR schedulers update `optimizer.learning_rate` directly (MLX optimizers have mutable lr)
- `clip_grad_norm_` needs `tree_map` for functional correctness in MLX
