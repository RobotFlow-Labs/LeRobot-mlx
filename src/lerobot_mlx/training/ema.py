"""Exponential Moving Average of model parameters.

Used by Diffusion Policy for stable inference weights. The EMA model
maintains a shadow copy of all trainable parameters and blends them
with the live parameters using an exponential decay.

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains shadow weights that are a running average of training weights.
    Used at inference time for more stable predictions.

    Usage::

        ema = EMAModel(model, decay=0.999)
        for batch in dataloader:
            trainer.train_step(batch)
            ema.update(model)

        # For inference:
        ema.apply(model)        # load EMA weights
        output = model(input)
        ema.restore(model)      # restore training weights

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor. Higher = smoother (0.999 typical).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Deep copy all parameters as shadow weights
        self.shadow: dict[str, mx.array] = {
            k: mx.array(v) for k, v in tree_flatten(model.parameters())
        }
        # Store original weights for restore
        self._backup: dict[str, mx.array] | None = None

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model parameters.

        shadow = decay * shadow + (1 - decay) * param

        Args:
            model: Model with updated training weights.
        """
        for k, v in tree_flatten(model.parameters()):
            if k in self.shadow:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
        # Materialize the updated shadow weights
        mx.eval(*self.shadow.values())

    def apply(self, model: nn.Module) -> None:
        """Load EMA shadow weights into model (for inference).

        Call restore() afterwards to return to training weights.

        Args:
            model: Model to load EMA weights into.
        """
        # Backup current weights before overwriting
        self._backup = {
            k: mx.array(v) for k, v in tree_flatten(model.parameters())
        }
        model.load_weights(list(self.shadow.items()))

    def restore(self, model: nn.Module) -> None:
        """Restore original (non-EMA) weights to model.

        Must be called after apply() to return to training weights.

        Args:
            model: Model to restore weights to.

        Raises:
            RuntimeError: If apply() was not called first.
        """
        if self._backup is None:
            raise RuntimeError("restore() called without prior apply()")
        model.load_weights(list(self._backup.items()))
        self._backup = None

    def get_decay(self) -> float:
        """Return the current decay factor."""
        return self.decay

    def set_decay(self, decay: float) -> None:
        """Update the decay factor.

        Args:
            decay: New decay factor in [0, 1).
        """
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
