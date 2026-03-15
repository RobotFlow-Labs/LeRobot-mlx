"""Tensor creation and manipulation ops — torch API mapped to MLX.

This module provides drop-in replacements for torch tensor operations
using MLX as the backend. Device parameters are ignored (MLX uses
unified memory on Apple Silicon).
"""

from __future__ import annotations

import contextlib
from collections import namedtuple
from typing import Any, Sequence

import mlx.core as mx
import numpy as np


# =============================================================================
# Dtype Mapping
# =============================================================================

# Expose dtype constants matching torch namespace
float32 = mx.float32
float16 = mx.float16
bfloat16 = mx.bfloat16
int32 = mx.int32
int64 = mx.int32  # LIMITATION: MLX has no int64 GPU support; silently truncates values > 2^31-1 to int32
int16 = mx.int16
int8 = mx.int8
uint8 = mx.uint8
bool_ = mx.bool_

_DTYPE_MAP: dict[str | Any, mx.Dtype] = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "int32": mx.int32,
    "int64": mx.int32,  # LIMITATION: no int64 GPU support; values > 2^31-1 silently truncated
    "int16": mx.int16,
    "int8": mx.int8,
    "uint8": mx.uint8,
    "bool": mx.bool_,
}


def _map_dtype(dtype: Any | None) -> mx.Dtype | None:
    """Map a torch-style dtype (string, torch dtype, or mx.Dtype) to mx.Dtype.

    Raises ValueError for unknown dtype strings instead of silently defaulting.
    """
    if dtype is None:
        return None
    if isinstance(dtype, mx.Dtype):
        return dtype
    if isinstance(dtype, str):
        result = _DTYPE_MAP.get(dtype)
        if result is None:
            raise ValueError(f"Unknown dtype: {dtype}")
        return result
    # Handle torch dtype objects if torch is available (e.g. torch.float32)
    name = getattr(dtype, "name", None) or str(dtype).rsplit(".", 1)[-1]
    result = _DTYPE_MAP.get(name)
    if result is None:
        raise ValueError(f"Unknown dtype: {dtype}")
    return result


# =============================================================================
# Shape Utilities
# =============================================================================

def _normalize_shape(shape: tuple) -> tuple[int, ...]:
    """Normalize shape args: zeros(3, 4) or zeros((3, 4)) both work."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return shape


# =============================================================================
# Tensor Creation
# =============================================================================

def tensor(data: Any, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.tensor(). Device param ignored (unified memory)."""
    if isinstance(data, mx.array):
        return data.astype(_map_dtype(dtype)) if dtype is not None else data
    arr = np.asarray(data)
    mapped = _map_dtype(dtype)
    return mx.array(arr, dtype=mapped) if mapped is not None else mx.array(arr)


def zeros(*shape: Any, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.zeros()."""
    s = _normalize_shape(shape)
    mapped = _map_dtype(dtype)
    return mx.zeros(s, dtype=mapped) if mapped is not None else mx.zeros(s)


def ones(*shape: Any, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.ones()."""
    s = _normalize_shape(shape)
    mapped = _map_dtype(dtype)
    return mx.ones(s, dtype=mapped) if mapped is not None else mx.ones(s)


def zeros_like(x: mx.array, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.zeros_like()."""
    if dtype is None:
        return mx.zeros_like(x)
    return mx.zeros(x.shape, dtype=_map_dtype(dtype))


def ones_like(x: mx.array, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.ones_like()."""
    if dtype is None:
        return mx.ones_like(x)
    return mx.ones(x.shape, dtype=_map_dtype(dtype))


def full(shape: Sequence[int], fill_value: float, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.full()."""
    mapped = _map_dtype(dtype)
    return mx.full(shape, fill_value, dtype=mapped) if mapped is not None else mx.full(shape, fill_value)


def arange(*args: Any, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.arange()."""
    mapped = _map_dtype(dtype)
    if mapped is not None:
        return mx.arange(*args, dtype=mapped)
    return mx.arange(*args)


def linspace(start: float, end: float, steps: int, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.linspace()."""
    result = mx.linspace(start, end, steps)
    mapped = _map_dtype(dtype)
    if mapped is not None:
        result = result.astype(mapped)
    return result


def eye(n: int, m: int | None = None, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.eye()."""
    mapped = _map_dtype(dtype)
    result = mx.eye(n, m or n)
    if mapped is not None:
        result = result.astype(mapped)
    return result


def rand(*shape: Any, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.rand() — uniform [0, 1)."""
    s = _normalize_shape(shape)
    result = mx.random.uniform(shape=s)
    mapped = _map_dtype(dtype)
    if mapped is not None:
        result = result.astype(mapped)
    return result


def randn(*shape: Any, dtype: Any = None, device: Any = None) -> mx.array:
    """Drop-in for torch.randn() — standard normal."""
    s = _normalize_shape(shape)
    result = mx.random.normal(shape=s)
    mapped = _map_dtype(dtype)
    if mapped is not None:
        result = result.astype(mapped)
    return result


# =============================================================================
# Tensor Manipulation
# =============================================================================

def cat(tensors: Sequence[mx.array], dim: int = 0) -> mx.array:
    """Drop-in for torch.cat()."""
    return mx.concatenate(tensors, axis=dim)


def stack(tensors: Sequence[mx.array], dim: int = 0) -> mx.array:
    """Drop-in for torch.stack()."""
    return mx.stack(tensors, axis=dim)


def split(x: mx.array, split_size_or_sections: int | list[int], dim: int = 0) -> list[mx.array]:
    """Drop-in for torch.split().

    When split_size_or_sections is an int, splits into chunks of that size.
    When it's a list, splits at cumulative boundaries.
    """
    if isinstance(split_size_or_sections, int):
        size = split_size_or_sections
        total = x.shape[dim]
        indices = list(range(size, total, size))
        return mx.split(x, indices, axis=dim)
    else:
        # List of sizes -> cumulative indices
        import itertools
        indices = list(itertools.accumulate(split_size_or_sections[:-1]))
        return mx.split(x, indices, axis=dim)


def chunk(x: mx.array, chunks: int, dim: int = 0) -> list[mx.array]:
    """Drop-in for torch.chunk() — split into N roughly equal chunks."""
    total = x.shape[dim]
    chunk_size = (total + chunks - 1) // chunks
    indices = list(range(chunk_size, total, chunk_size))
    return mx.split(x, indices, axis=dim)


def unsqueeze(x: mx.array, dim: int) -> mx.array:
    """Drop-in for torch.unsqueeze()."""
    return mx.expand_dims(x, axis=dim)


def squeeze(x: mx.array, dim: int | None = None) -> mx.array:
    """Drop-in for torch.squeeze()."""
    if dim is None:
        return mx.squeeze(x)
    return mx.squeeze(x, axis=dim)


def flatten(x: mx.array, start_dim: int = 0, end_dim: int = -1) -> mx.array:
    """Drop-in for torch.flatten()."""
    return mx.flatten(x, start_axis=start_dim, end_axis=end_dim)


def clamp(x: mx.array, min: float | None = None, max: float | None = None) -> mx.array:
    """Drop-in for torch.clamp()."""
    return mx.clip(x, min, max)


def where(condition: mx.array, x: mx.array, y: mx.array) -> mx.array:
    """Drop-in for torch.where()."""
    return mx.where(condition, x, y)


def einsum(equation: str, *operands: mx.array) -> mx.array:
    """Drop-in for torch.einsum()."""
    return mx.einsum(equation, *operands)


def transpose(x: mx.array, dim0: int, dim1: int) -> mx.array:
    """Drop-in for torch.transpose() (swap two dims)."""
    axes = list(range(x.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return mx.transpose(x, axes=axes)


def permute(x: mx.array, dims: Sequence[int]) -> mx.array:
    """Drop-in for Tensor.permute()."""
    return mx.transpose(x, axes=dims)


def reshape(x: mx.array, shape: Sequence[int]) -> mx.array:
    """Drop-in for torch.reshape()."""
    return mx.reshape(x, shape)


def repeat_interleave(x: mx.array, repeats: int, dim: int | None = None) -> mx.array:
    """Drop-in for torch.repeat_interleave()."""
    return mx.repeat(x, repeats, axis=dim)


def abs(x: mx.array) -> mx.array:
    """Drop-in for torch.abs()."""
    return mx.abs(x)


def mean(x: mx.array, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> mx.array:
    """Drop-in for torch.mean()."""
    return mx.mean(x, axis=dim, keepdims=keepdim)


def sum(x: mx.array, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> mx.array:
    """Drop-in for torch.sum()."""
    return mx.sum(x, axis=dim, keepdims=keepdim)


MaxResult = namedtuple("MaxResult", ["values", "indices"])
MinResult = namedtuple("MinResult", ["values", "indices"])


def max(x: mx.array, dim: int | None = None, keepdim: bool = False):
    """Drop-in for torch.max().

    When dim is None, returns a scalar (the global max).
    When dim is specified, returns a MaxResult(values, indices) namedtuple,
    matching PyTorch behavior.
    """
    if dim is None:
        return mx.max(x)
    values = mx.max(x, axis=dim, keepdims=keepdim)
    indices = mx.argmax(x, axis=dim, keepdims=keepdim)
    return MaxResult(values, indices)


def min(x: mx.array, dim: int | None = None, keepdim: bool = False):
    """Drop-in for torch.min().

    When dim is None, returns a scalar (the global min).
    When dim is specified, returns a MinResult(values, indices) namedtuple,
    matching PyTorch behavior.
    """
    if dim is None:
        return mx.min(x)
    values = mx.min(x, axis=dim, keepdims=keepdim)
    indices = mx.argmin(x, axis=dim, keepdims=keepdim)
    return MinResult(values, indices)


def exp(x: mx.array) -> mx.array:
    """Drop-in for torch.exp()."""
    return mx.exp(x)


def log(x: mx.array) -> mx.array:
    """Drop-in for torch.log()."""
    return mx.log(x)


def sqrt(x: mx.array) -> mx.array:
    """Drop-in for torch.sqrt()."""
    return mx.sqrt(x)


def matmul(a: mx.array, b: mx.array) -> mx.array:
    """Drop-in for torch.matmul()."""
    return mx.matmul(a, b)


# =============================================================================
# Device Handling (no-ops for unified memory)
# =============================================================================

class _DeviceStub:
    """Absorbs .to(device), .cuda(), .cpu() calls. MLX uses unified memory."""

    def __init__(self, type_str: str = "mps"):
        self.type = type_str

    def __repr__(self) -> str:
        return f"device(type='{self.type}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _DeviceStub):
            return self.type == other.type
        if isinstance(other, str):
            return self.type == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.type)


device = _DeviceStub


def no_grad():
    """Context manager that does nothing (MLX has no grad tracking by default)."""
    return contextlib.nullcontext()


# =============================================================================
# Type alias
# =============================================================================

# Tensor is just mx.array in MLX
Tensor = mx.array
