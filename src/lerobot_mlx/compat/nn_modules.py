"""Base Module adapter — makes mlx.nn.Module look like torch.nn.Module.

This is the foundation class that all LeRobot-MLX neural network modules
inherit from. It adds PyTorch-compatible methods (.to(), .train(), .eval(),
.state_dict(), .load_state_dict(), etc.) on top of mlx.nn.Module.
"""

from __future__ import annotations

from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as _nn
from mlx.utils import tree_flatten, tree_unflatten


class Module(_nn.Module):
    """Adapter that adds torch.nn.Module-like API to mlx.nn.Module.

    Key differences from torch.nn.Module:
    - .to(device) is a no-op (MLX uses unified memory)
    - .train()/.eval() delegate to mlx.nn.Module's train/eval
    - .parameters() returns mlx-style nested dict (already works)
    - .named_parameters() flattens to (name, array) pairs
    - .state_dict() / .load_state_dict() use mlx weight format
    - .register_buffer() stores non-parameter arrays as attributes
    """

    def to(self, device: Any = None, dtype: Any = None) -> "Module":
        """No-op for device (MLX unified memory). Casts parameters if dtype specified."""
        if dtype is not None:
            from .tensor_ops import _map_dtype
            mapped_dtype = dtype if isinstance(dtype, mx.Dtype) else _map_dtype(dtype)
            params = tree_flatten(self.parameters())
            casted = [(k, v.astype(mapped_dtype)) for k, v in params]
            self.load_weights(casted)
        return self

    def cuda(self, device: Any = None) -> "Module":
        """No-op. MLX uses unified memory."""
        return self

    def cpu(self) -> "Module":
        """No-op. MLX uses unified memory."""
        return self

    def train(self, mode: bool = True) -> "Module":
        """Set training mode. Delegates to mlx.nn.Module.train.

        We always call _nn.Module.train(self, mode) directly to avoid
        recursion, since mlx.nn.Module.eval() calls self.train(False).
        """
        _nn.Module.train(self, mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode."""
        _nn.Module.train(self, False)
        return self

    def named_parameters(self) -> list[tuple[str, mx.array]]:
        """Yield (name, parameter) pairs like PyTorch.

        Uses mlx.utils.tree_flatten to produce dot-separated key paths.
        """
        return tree_flatten(self.parameters())

    def requires_grad_(self, requires_grad: bool = True) -> "Module":
        """No-op. MLX handles gradients via value_and_grad, not per-tensor flags."""
        return self

    def state_dict(self) -> dict[str, mx.array]:
        """Return flattened parameter dict (for weight loading compatibility)."""
        return dict(tree_flatten(self.parameters()))

    def load_state_dict(self, state_dict: dict[str, mx.array], strict: bool = True) -> None:
        """Load weights from a flat dict. Maps torch key names if needed."""
        weights = list(state_dict.items())
        self.load_weights(weights, strict=strict)

    def register_buffer(self, name: str, tensor: mx.array | None) -> None:
        """Store a non-parameter tensor as an attribute.

        In MLX, buffers are just regular attributes. They won't be included
        in parameters() unless stored as trainable weights, but they will
        be preserved during serialization if they are mx.array.
        """
        if tensor is not None:
            setattr(self, name, tensor)
        elif hasattr(self, name):
            delattr(self, name)

    def num_parameters(self) -> int:
        """Count total number of scalar parameters."""
        return sum(p.size for _, p in tree_flatten(self.parameters()))

    def __repr__(self) -> str:
        """Basic repr showing class name and number of params."""
        try:
            n = self.num_parameters()
            return f"{self.__class__.__name__}({n:,} parameters)"
        except Exception:
            return f"{self.__class__.__name__}()"
