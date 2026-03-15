"""
torch.optim → MLX optimizers, LR schedulers, and gradient clipping.

Drop-in replacements for torch.optim.*, torch.optim.lr_scheduler.*,
and diffusers.optimization.get_scheduler used across upstream LeRobot.

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

import math

import mlx.core as mx
import mlx.optimizers as _optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten


# =============================================================================
# Optimizers (thin re-exports from mlx.optimizers)
# =============================================================================

Adam = _optim.Adam
AdamW = _optim.AdamW
SGD = _optim.SGD


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

class CosineAnnealingLR:
    """torch.optim.lr_scheduler.CosineAnnealingLR equivalent.

    Cosine annealing from base_lr to eta_min over T_max steps.

    lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * step / T_max)) / 2
    """

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = float(optimizer.learning_rate)
        self.step_count = 0
        self._last_lr = self.base_lr

    def step(self):
        """Advance the scheduler by one step."""
        self.step_count += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.step_count / self.T_max)
        ) / 2
        self.optimizer.learning_rate = lr
        self._last_lr = lr

    def get_last_lr(self) -> float:
        """Return the last computed learning rate."""
        return self._last_lr


class LinearWarmupCosineDecay:
    """Linear warmup followed by cosine decay — most common schedule in LeRobot.

    During warmup (step <= warmup_steps):
        lr = base_lr * step / warmup_steps

    After warmup:
        lr = min_lr + (base_lr - min_lr) * (1 + cos(pi * progress)) / 2
        where progress = (step - warmup_steps) / (total_steps - warmup_steps)
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = float(optimizer.learning_rate)
        self.min_lr = min_lr
        self.step_count = 0
        self._last_lr = self.base_lr

    def step(self):
        """Advance the scheduler by one step."""
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            lr = self.min_lr + (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            ) / 2
        self.optimizer.learning_rate = lr
        self._last_lr = lr

    def get_last_lr(self) -> float:
        """Return the last computed learning rate."""
        return self._last_lr


class _ConstantWithWarmup:
    """Constant LR after linear warmup."""

    def __init__(self, optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = float(optimizer.learning_rate)
        self.step_count = 0
        self._last_lr = 0.0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            lr = self.base_lr
        self.optimizer.learning_rate = lr
        self._last_lr = lr

    def get_last_lr(self) -> float:
        return self._last_lr


class _LinearSchedule:
    """Linear decay from base_lr to 0 after warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = float(optimizer.learning_rate)
        self.step_count = 0
        self._last_lr = 0.0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            lr = self.base_lr * (1.0 - progress)
        self.optimizer.learning_rate = lr
        self._last_lr = lr

    def get_last_lr(self) -> float:
        return self._last_lr


def get_scheduler(
    name: str,
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Drop-in for diffusers.optimization.get_scheduler.

    Supports: "cosine", "linear", "constant_with_warmup".

    Args:
        name: Scheduler name.
        optimizer: MLX optimizer with .learning_rate attribute.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.

    Returns:
        LR scheduler with a .step() method.
    """
    if name == "cosine":
        return LinearWarmupCosineDecay(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif name == "linear":
        return _LinearSchedule(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif name == "constant_with_warmup":
        return _ConstantWithWarmup(optimizer, num_warmup_steps)
    else:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            f"Supported: 'cosine', 'linear', 'constant_with_warmup'."
        )


# =============================================================================
# Gradient Clipping
# =============================================================================

def clip_grad_norm_(
    grads,
    max_norm: float,
    norm_type: float = 2.0,
) -> tuple[dict | list, float]:
    """Clip gradient norm, analogous to torch.nn.utils.clip_grad_norm_.

    Unlike PyTorch which mutates tensors in-place, MLX arrays are functional.
    This function returns a NEW gradient tree with clipped values plus the
    total norm. Use the returned grads for the optimizer update.

    This implementation accumulates norms in MLX without per-tensor .item()
    calls, using a single synchronization point for efficiency.

    Args:
        grads: Gradient tree (dict, list, or nested structure from mlx.nn.value_and_grad).
        max_norm: Maximum gradient norm.
        norm_type: Type of norm (2.0 = L2 norm).

    Returns:
        (clipped_grads, total_norm) — new gradient tree and the original total norm.
    """
    # Flatten the gradient tree to get all leaf arrays
    flat_grads = tree_flatten(grads)

    # Compute total norm — accumulate in MLX, single .item() at the end
    if norm_type == float("inf"):
        # For inf norm, we need per-tensor max — stack them
        max_parts = [mx.max(mx.abs(g)) for _, g in flat_grads if g is not None]
        if max_parts:
            total_norm_val = mx.max(mx.stack(max_parts)).item()
        else:
            total_norm_val = 0.0
    else:
        # Accumulate norm^p parts as MLX arrays, single sync
        norm_sq_parts = [mx.sum(g ** norm_type) for _, g in flat_grads if g is not None]
        if norm_sq_parts:
            total_norm_sq = mx.sum(mx.stack(norm_sq_parts))
            total_norm_val = (total_norm_sq ** (1.0 / norm_type)).item()
        else:
            total_norm_val = 0.0

    clip_coef = max_norm / max(total_norm_val, 1e-6)
    clip_coef = min(clip_coef, 1.0)

    if clip_coef < 1.0:
        clipped = tree_map(lambda g: g * clip_coef, grads)
    else:
        clipped = grads

    return clipped, total_norm_val
