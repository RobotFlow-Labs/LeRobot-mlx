"""MLX-native training loop replacing accelerate.

This module provides a Trainer class that uses mlx.nn.value_and_grad for
forward + backward passes, with gradient clipping, LR scheduling, and
checkpoint management. Designed to be policy-agnostic: any model with a
compute_loss(batch) -> scalar method can be trained.

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

import logging

# Import from our compat layer (with fallback)
try:
    from lerobot_mlx.compat.optim import get_scheduler, clip_grad_norm_
except ImportError:
    # Fallback: inline implementations if compat not available yet
    get_scheduler = None
    clip_grad_norm_ = None

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    lr: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    lr_scheduler_type: str = "cosine"
    lr_warmup_steps: int = 500
    training_steps: int = 100_000
    batch_size: int = 32
    log_interval: int = 100
    save_interval: int = 10_000
    output_dir: str = "outputs"


# =============================================================================
# Gradient Clipping (functional)
# =============================================================================

def _clip_grads(grads: Any, max_norm: float) -> Any:
    """Clip gradient norms (functional, returns new grads).

    Accumulates norm on-device without per-tensor .item() calls to avoid
    synchronization overhead.

    Args:
        grads: Gradient tree from nn.value_and_grad.
        max_norm: Maximum gradient L2 norm.

    Returns:
        Clipped gradient tree (new object, original is untouched).
    """
    flat_grads = tree_flatten(grads)
    grad_arrays = [g for _, g in flat_grads if g is not None]
    if not grad_arrays:
        return grads
    norm_parts = [mx.sum(g ** 2) for g in grad_arrays]
    total_norm = mx.sqrt(mx.sum(mx.stack(norm_parts)))
    total_norm_val = total_norm.item()
    clip_coef = min(1.0, max_norm / (total_norm_val + 1e-6))
    if clip_coef < 1.0:
        clipped = [(k, g * clip_coef) for k, g in flat_grads]
        return tree_unflatten(clipped)
    return grads


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """MLX-native training loop replacing accelerate.

    Usage::

        config = TrainingConfig(lr=1e-4, training_steps=10000)
        trainer = Trainer(policy, config)
        metrics = trainer.train(dataloader, num_steps=10000)

    The policy must implement either:
      - ``compute_loss(batch) -> mx.array`` (preferred), or
      - ``forward(batch) -> dict`` where dict contains ``'loss'`` key.
    """

    def __init__(self, policy: nn.Module, config: TrainingConfig | None = None):
        self.policy = policy
        self.config = config or TrainingConfig()
        self.global_step = 0

        # ---- Optimizer ----
        self.optimizer = optim.AdamW(
            learning_rate=self.config.lr,
            betas=[self.config.adam_beta1, self.config.adam_beta2],
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )

        # ---- LR Scheduler ----
        self.lr_scheduler = self._create_scheduler()

        # ---- Loss + grad function ----
        # nn.value_and_grad takes the model as first arg to loss_fn,
        # returns (loss_value, grad_tree)
        self._loss_and_grad_fn = nn.value_and_grad(policy, self._compute_loss)

    def _create_scheduler(self):
        """Create LR scheduler from config."""
        if get_scheduler is not None:
            return get_scheduler(
                self.config.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.config.lr_warmup_steps,
                num_training_steps=self.config.training_steps,
            )
        # Fallback: no scheduler
        return None

    @staticmethod
    def _compute_loss(model: nn.Module, batch: dict) -> mx.array:
        """Compute training loss. Policy-agnostic.

        Tries model.compute_loss(batch) first, then model.forward(batch)['loss'].
        """
        if hasattr(model, "compute_loss"):
            return model.compute_loss(batch)
        output = model(batch)
        if isinstance(output, dict) and "loss" in output:
            return output["loss"]
        raise ValueError(
            "Policy must implement compute_loss(batch) -> scalar "
            "or forward(batch) -> {'loss': scalar, ...}"
        )

    def train_step(self, batch: dict) -> dict[str, float]:
        """Single training step: forward + backward + optimizer update.

        Args:
            batch: Dict of mx.array tensors (e.g. from SimpleDataLoader).

        Returns:
            Dict with 'loss' (float) and 'lr' (float).
        """
        # Forward + backward
        loss, grads = self._loss_and_grad_fn(self.policy, batch)

        # Gradient clipping
        if self.config.max_grad_norm is not None:
            grads = _clip_grads(grads, self.config.max_grad_norm)

        # Optimizer step
        self.optimizer.update(self.policy, grads)

        # LR scheduler step
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # CRITICAL: materialize computation — without this, MLX builds an
        # ever-growing lazy graph that eventually OOMs
        mx.eval(self.policy.parameters(), self.optimizer.state, loss)

        self.global_step += 1

        # Get current LR
        lr = float(self.optimizer.learning_rate)

        return {"loss": loss.item(), "lr": lr}

    def train(
        self,
        dataloader,
        num_steps: int | None = None,
    ) -> list[dict[str, float]]:
        """Full training loop with logging and checkpointing.

        Args:
            dataloader: Iterable yielding batched dicts of mx.array.
            num_steps: Number of steps to train. If None, uses config.training_steps.

        Returns:
            List of per-step metric dicts.
        """
        num_steps = num_steps or self.config.training_steps
        self.policy.train()
        metrics: list[dict[str, float]] = []
        step = 0
        t_start = time.time()

        while step < num_steps:
            for batch in dataloader:
                if step >= num_steps:
                    break

                step_metrics = self.train_step(batch)
                step_metrics["step"] = step
                metrics.append(step_metrics)

                if step % self.config.log_interval == 0:
                    elapsed = time.time() - t_start
                    steps_per_sec = (step + 1) / max(elapsed, 1e-6)
                    logger.info(
                        f"Step {step}: loss={step_metrics['loss']:.4f}, "
                        f"lr={step_metrics['lr']:.2e}, "
                        f"steps/s={steps_per_sec:.1f}"
                    )

                if (
                    self.config.save_interval > 0
                    and step > 0
                    and step % self.config.save_interval == 0
                ):
                    self.save_checkpoint(step)

                step += 1

        return metrics

    def save_checkpoint(self, step: int, path: str | Path | None = None) -> str:
        """Save model weights to disk.

        Args:
            step: Current training step (used in filename).
            path: Optional explicit path. If None, uses config.output_dir.

        Returns:
            Path to saved checkpoint file.
        """
        if path is None:
            ckpt_dir = Path(self.config.output_dir) / "checkpoints"
        else:
            ckpt_dir = Path(path)

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"step_{step}.npz"

        weights = dict(tree_flatten(self.policy.parameters()))
        mx.savez(str(ckpt_path), **weights)

        return str(ckpt_path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights from checkpoint.

        Args:
            path: Path to .npz or .safetensors checkpoint.
        """
        path = str(path)
        weights = mx.load(path)
        # mx.load returns dict for .npz, list of tuples for .safetensors
        if isinstance(weights, dict):
            self.policy.load_weights(list(weights.items()))
        else:
            self.policy.load_weights(weights)
