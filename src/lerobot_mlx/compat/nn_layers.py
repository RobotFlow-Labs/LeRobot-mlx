"""NN layer mappings — all 33+ nn types used by upstream LeRobot.

Maps PyTorch nn modules to MLX equivalents. Direct mappings use mlx.nn
classes directly. Complex modules (MultiheadAttention, BatchNorm2d,
TransformerEncoder) are wrapped to provide torch-compatible APIs.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

import mlx.core as mx
import mlx.nn as _nn

from .nn_modules import Module

# =============================================================================
# Direct Mappings (API-compatible between torch and MLX)
# =============================================================================

Linear = _nn.Linear
LayerNorm = _nn.LayerNorm
Embedding = _nn.Embedding
Dropout = _nn.Dropout
RMSNorm = _nn.RMSNorm
GroupNorm = _nn.GroupNorm
class Conv1d(Module):
    """Conv1d with PyTorch NCL convention wrapping MLX NLC convention."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self._conv = _nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)

    def __call__(self, x):
        # x: (B, C, L) -> (B, L, C) for MLX
        x = mx.transpose(x, axes=(0, 2, 1))
        x = self._conv(x)
        # (B, L, C) -> (B, C, L) back to PyTorch convention
        return mx.transpose(x, axes=(0, 2, 1))


class ConvTranspose1d(Module):
    """ConvTranspose1d with PyTorch NCL convention wrapping MLX NLC convention.

    Uses MLX's native conv_transpose1d function.
    Input/output use PyTorch NCL (channels-first) convention.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # MLX conv_transpose1d weight shape: (C_out, K, C_in)
        import math
        fan_in = in_channels * kernel_size
        k = 1.0 / math.sqrt(fan_in)
        self.weight = mx.random.uniform(
            low=-k, high=k,
            shape=(out_channels, kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.random.uniform(low=-k, high=k, shape=(out_channels,))
        else:
            self.bias = None

    def __call__(self, x):
        # x: (B, C_in, L_in) -> (B, L_in, C_in) for MLX
        x = mx.transpose(x, axes=(0, 2, 1))
        result = mx.conv_transpose1d(
            x, self.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )
        if self.bias is not None:
            result = result + self.bias
        # (B, L_out, C_out) -> (B, C_out, L_out) back to PyTorch convention
        return mx.transpose(result, axes=(0, 2, 1))


class GroupNorm1d(Module):
    """GroupNorm wrapper for channels-first (B, C, L) format.

    MLX's GroupNorm expects channels-last (B, ..., C), so we transpose
    before and after the operation.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.gn = _nn.GroupNorm(num_groups, num_channels)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            # (B, C, L) -> (B, L, C)
            x = mx.transpose(x, axes=(0, 2, 1))
            x = self.gn(x)
            # (B, L, C) -> (B, C, L)
            return mx.transpose(x, axes=(0, 2, 1))
        return self.gn(x)


class Conv2d(Module):
    """Conv2d with PyTorch NCHW convention wrapping MLX NHWC convention."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self._conv = _nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)

    def __call__(self, x):
        # x: (B, C, H, W) -> (B, H, W, C) for MLX
        x = mx.transpose(x, axes=(0, 2, 3, 1))
        x = self._conv(x)
        # (B, H, W, C) -> (B, C, H, W) back to PyTorch convention
        return mx.transpose(x, axes=(0, 3, 1, 2))


# =============================================================================
# Activation Modules
# =============================================================================

class ReLU(Module):
    """Drop-in for torch.nn.ReLU."""

    def __call__(self, x: mx.array) -> mx.array:
        return _nn.relu(x)


class GELU(Module):
    """Drop-in for torch.nn.GELU."""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self._approximate = approximate

    def __call__(self, x: mx.array) -> mx.array:
        if self._approximate == "tanh":
            return _nn.gelu_approx(x)
        return _nn.gelu(x)


class SiLU(Module):
    """Drop-in for torch.nn.SiLU (Swish)."""

    def __call__(self, x: mx.array) -> mx.array:
        return _nn.silu(x)


class Mish(Module):
    """Drop-in for torch.nn.Mish."""

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.tanh(_nn.softplus(x))


class Tanh(Module):
    """Drop-in for torch.nn.Tanh."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(x)


class Sigmoid(Module):
    """Drop-in for torch.nn.Sigmoid."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sigmoid(x)


class ELU(Module):
    """Drop-in for torch.nn.ELU."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self._alpha = alpha

    def __call__(self, x: mx.array) -> mx.array:
        return _nn.elu(x, self._alpha)


class Softmax(Module):
    """Drop-in for torch.nn.Softmax."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self._dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        return mx.softmax(x, axis=self._dim)


class Identity(Module):
    """Drop-in for torch.nn.Identity."""

    def __call__(self, x: mx.array, *args: Any, **kwargs: Any) -> mx.array:
        return x


# =============================================================================
# Sequential & Containers
# =============================================================================

# MLX Sequential works identically to torch
Sequential = _nn.Sequential


class ModuleList(Module):
    """Drop-in for torch.nn.ModuleList.

    Stores child modules as indexed attributes so that mlx.nn.Module's
    parameter discovery finds them via attribute introspection.
    """

    def __init__(self, modules: Sequence[Any] | None = None):
        super().__init__()
        # Use object.__setattr__ to avoid mlx.nn.Module.__setattr__ -> __setitem__ recursion
        object.__setattr__(self, "_modules_list", [])
        if modules is not None:
            for m in modules:
                self.append(m)

    def append(self, module: Any) -> "ModuleList":
        """Append a module and register it as an indexed attribute."""
        idx = len(self._modules_list)
        # Use dict.__setitem__ to store the child in mlx's dict-based storage
        # (which is what parameters() walks), while avoiding the recursion from
        # mlx.nn.Module.__setattr__ -> self.__setitem__ -> __setattr__.
        dict.__setitem__(self, f"module_{idx}", module)
        self._modules_list.append(module)
        return self

    def extend(self, modules: Sequence[Any]) -> "ModuleList":
        """Extend with multiple modules."""
        for m in modules:
            self.append(m)
        return self

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(self._modules_list[idx])
        if isinstance(idx, str):
            # MLX Module.update() accesses children by attribute name (e.g. "module_0")
            if idx.startswith("module_") and idx[7:].isdigit():
                return self._modules_list[int(idx[7:])]
            return getattr(self, idx)
        return self._modules_list[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, str):
            if idx.startswith("module_") and idx[7:].isdigit():
                int_idx = int(idx[7:])
                if hasattr(self, '_modules_list') and int_idx < len(self._modules_list):
                    self._modules_list[int_idx] = value
            # Use dict.__setitem__ for mlx parameter discovery, avoiding recursion
            dict.__setitem__(self, idx, value)
            return
        if hasattr(self, '_modules_list') and idx < len(self._modules_list):
            self._modules_list[idx] = value
        dict.__setitem__(self, f"module_{idx}", value)

    def __len__(self) -> int:
        return len(self._modules_list)

    def __iter__(self):
        return iter(self._modules_list)

    def __repr__(self) -> str:
        lines = [f"  ({i}): {repr(m)}" for i, m in enumerate(self._modules_list)]
        return f"ModuleList(\n" + "\n".join(lines) + "\n)" if lines else "ModuleList()"


class ModuleDict(Module):
    """Drop-in for torch.nn.ModuleDict."""

    def __init__(self, modules: dict[str, Any] | None = None):
        super().__init__()
        # Use object.__setattr__ to avoid mlx.nn.Module.__setattr__ interception
        object.__setattr__(self, "_modules_dict", {})
        if modules is not None:
            for k, v in modules.items():
                self[k] = v

    def __setitem__(self, key: str, module: Any) -> None:
        # Use dict.__setitem__ to store in mlx's dict-based storage for parameter discovery,
        # while avoiding recursion from __setattr__ -> __setitem__ -> __setattr__.
        dict.__setitem__(self, key, module)
        self._modules_dict[key] = module

    def __getitem__(self, key: str) -> Any:
        return self._modules_dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self._modules_dict

    def __len__(self) -> int:
        return len(self._modules_dict)

    def __iter__(self):
        return iter(self._modules_dict)

    def keys(self):
        return self._modules_dict.keys()

    def values(self):
        return self._modules_dict.values()

    def items(self):
        return self._modules_dict.items()


# =============================================================================
# Parameter
# =============================================================================

class Parameter:
    """Drop-in for torch.nn.Parameter.

    In MLX, parameters are just mx.array attributes discovered automatically
    by mlx.nn.Module. This class simply returns an mx.array.
    """

    def __new__(cls, data: Any, requires_grad: bool = True) -> mx.array:
        if isinstance(data, mx.array):
            return data
        return mx.array(data)


# =============================================================================
# MultiheadAttention
# =============================================================================

class MultiheadAttention(Module):
    """Drop-in for torch.nn.MultiheadAttention.

    Wraps mlx.nn.MultiHeadAttention with a torch-compatible call signature:
        (query, key, value, attn_mask=None, need_weights=False) -> (output, weights)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.mha = _nn.MultiHeadAttention(
            dims=embed_dim,
            num_heads=num_heads,
            bias=bias,
            query_input_dims=embed_dim,
            key_input_dims=kdim or embed_dim,
            value_input_dims=vdim or embed_dim,
        )

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        attn_mask: mx.array | None = None,
        need_weights: bool = False,
        key_padding_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """Forward pass with torch-compatible signature.

        Args:
            query: (B, T, E) if batch_first else (T, B, E)
            key: (B, S, E) if batch_first else (S, B, E)
            value: (B, S, E) if batch_first else (S, B, E)
            attn_mask: optional attention mask
            need_weights: ignored (always returns None for weights)
            key_padding_mask: optional key padding mask

        Returns:
            Tuple of (output, attention_weights). Weights are always None.
        """
        if not self.batch_first:
            # (T, B, E) -> (B, T, E)
            query = mx.transpose(query, axes=(1, 0, 2))
            key = mx.transpose(key, axes=(1, 0, 2))
            value = mx.transpose(value, axes=(1, 0, 2))

        out = self.mha(query, key, value, mask=attn_mask)

        if not self.batch_first:
            # (B, T, E) -> (T, B, E)
            out = mx.transpose(out, axes=(1, 0, 2))

        return out, None


# =============================================================================
# BatchNorm2d (NCHW -> NHWC transpose for MLX)
# =============================================================================

class BatchNorm2d(Module):
    """Drop-in for torch.nn.BatchNorm2d.

    Handles NCHW (PyTorch) <-> NHWC (MLX) format conversion automatically.
    Input is expected in NCHW format (as PyTorch policies produce).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.bn = _nn.BatchNorm(num_features, eps=eps, momentum=momentum, affine=affine)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, H, W) NCHW -> (B, H, W, C) NHWC
        x = mx.transpose(x, axes=(0, 2, 3, 1))
        x = self.bn(x)
        # (B, H, W, C) NHWC -> (B, C, H, W) NCHW
        return mx.transpose(x, axes=(0, 3, 1, 2))


class BatchNorm1d(Module):
    """Drop-in for torch.nn.BatchNorm1d."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.bn = _nn.BatchNorm(num_features, eps=eps, momentum=momentum, affine=affine)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            # (B, C, L) -> (B, L, C)
            x = mx.transpose(x, axes=(0, 2, 1))
            x = self.bn(x)
            return mx.transpose(x, axes=(0, 2, 1))
        return self.bn(x)


# =============================================================================
# TransformerEncoder
# =============================================================================

class TransformerEncoderLayer(Module):
    """Drop-in for torch.nn.TransformerEncoderLayer.

    Implements a single transformer encoder layer with self-attention
    and feedforward network, supporting both pre-norm and post-norm.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable = "relu",
        batch_first: bool = True,
        norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = _nn.MultiHeadAttention(d_model, nhead)
        self.linear1 = _nn.Linear(d_model, dim_feedforward)
        self.linear2 = _nn.Linear(dim_feedforward, d_model)
        self.norm1 = _nn.LayerNorm(d_model)
        self.norm2 = _nn.LayerNorm(d_model)
        self.dropout = _nn.Dropout(dropout)
        self.norm_first = norm_first

        if callable(activation) and not isinstance(activation, str):
            self._act = activation
        elif activation == "gelu":
            self._act = _nn.gelu
        else:
            self._act = _nn.relu

    def __call__(
        self,
        src: mx.array,
        src_mask: mx.array | None = None,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask)
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src, src_mask))
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(self, x: mx.array, mask: mx.array | None) -> mx.array:
        return self.dropout(self.self_attn(x, x, x, mask=mask))

    def _ff_block(self, x: mx.array) -> mx.array:
        return self.dropout(self.linear2(self.dropout(self._act(self.linear1(x)))))


def _clone_encoder_layer(layer: "TransformerEncoderLayer") -> "TransformerEncoderLayer":
    """Create a new TransformerEncoderLayer with the same config but fresh weights.

    copy.deepcopy fails on mlx modules (unpicklable function refs), so we
    reconstruct the layer from its configuration and re-initialize weights.
    """
    # Extract config from the existing layer
    d_model = layer.linear2.weight.shape[0]
    # Get nhead from the MultiHeadAttention
    nhead = layer.self_attn.num_heads if hasattr(layer.self_attn, 'num_heads') else 1
    dim_feedforward = layer.linear1.weight.shape[0]
    norm_first = layer.norm_first

    # Reconstruct activation
    act = layer._act

    # Preserve the original dropout rate across all cloned layers
    dropout_p = layer.dropout.p if hasattr(layer.dropout, 'p') else 0.0

    new_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout_p,
        activation=act,
        norm_first=norm_first,
    )
    return new_layer


class TransformerEncoder(Module):
    """Drop-in for torch.nn.TransformerEncoder.

    Stacks N TransformerEncoderLayer instances with optional final LayerNorm.
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Any | None = None,
    ):
        super().__init__()
        # Match PyTorch behavior: create independent layer instances
        # so that each layer has its own weights.
        self.layers = ModuleList()
        self.layers.append(encoder_layer)
        for _ in range(num_layers - 1):
            self.layers.append(_clone_encoder_layer(encoder_layer))
        self.norm = norm

    def __call__(
        self,
        src: mx.array,
        mask: mx.array | None = None,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


# =============================================================================
# Conv3d (stub — full implementation in PRD-11 for TD-MPC)
# =============================================================================

class Conv3d(Module):
    """Stub for torch.nn.Conv3d.

    Conv3d is not natively available in MLX. This stub provides the
    constructor API so that TD-MPC code can import it. The actual
    forward pass will be implemented in PRD-11 using decomposed Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 1,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size

        # Initialize weight for parameter discovery
        import math
        k = 1.0 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
        self.weight = mx.random.uniform(
            low=-k,
            high=k,
            shape=(out_channels, in_channels, *kernel_size),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError(
            "Conv3d forward pass not yet implemented. See PRD-11. "
            "Input shape expected: (B, C_in, D, H, W)"
        )


# =============================================================================
# Flatten module
# =============================================================================

class Flatten(Module):
    """Drop-in for torch.nn.Flatten."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self._start_dim = start_dim
        self._end_dim = end_dim

    def __call__(self, x: mx.array) -> mx.array:
        return mx.flatten(x, start_axis=self._start_dim, end_axis=self._end_dim)


# =============================================================================
# Unflatten module
# =============================================================================

class Unflatten(Module):
    """Drop-in for torch.nn.Unflatten."""

    def __init__(self, dim: int, unflattened_size: tuple[int, ...]):
        super().__init__()
        self._dim = dim
        self._unflattened_size = unflattened_size

    def __call__(self, x: mx.array) -> mx.array:
        shape = list(x.shape)
        shape = shape[:self._dim] + list(self._unflattened_size) + shape[self._dim + 1:]
        return mx.reshape(x, shape)
