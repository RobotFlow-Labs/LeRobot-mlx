"""Compatibility layer — makes MLX look like PyTorch for policy code.

Usage in policy files:
    from lerobot_mlx.compat import nn, F, Tensor
    from lerobot_mlx.compat import tensor_ops as torch_ops

This module re-exports all compat submodules so that policy code
can use familiar PyTorch patterns with MLX as the backend.
"""

from lerobot_mlx.compat import (
    diffusers_mlx,
    distributions,
    einops_mlx,
    functional,
    nn_layers,
    nn_modules,
    optim,
    tensor_ops,
    vision,
)
from lerobot_mlx.compat import functional as F
from lerobot_mlx.compat import nn_layers as nn
from lerobot_mlx.compat.tensor_ops import Tensor

__all__ = [
    "nn",
    "nn_modules",
    "nn_layers",
    "tensor_ops",
    "functional",
    "F",
    "optim",
    "distributions",
    "einops_mlx",
    "vision",
    "diffusers_mlx",
    "Tensor",
]
