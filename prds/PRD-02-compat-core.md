# PRD-02: Compatibility Layer — Core (tensor_ops + nn_modules + nn_layers)

> **Status:** TODO
> **Priority:** P0 — THE foundation everything builds on
> **Dependencies:** PRD-01
> **Estimated LOC:** ~800
> **Phase:** 1 (Foundation)

---

## Objective

Implement the core compat layer that makes MLX "look like PyTorch" for tensor operations, nn.Module base class, and standard nn layers. After this PRD, policy code can `from lerobot_mlx.compat import nn` and use familiar PyTorch patterns.

---

## Deliverables

### 1. `compat/tensor_ops.py` — Tensor Creation & Manipulation

Map every tensor operation used across 143 upstream files:

```python
import mlx.core as mx
import numpy as np

# === Tensor Creation ===
def tensor(data, dtype=None, device=None):
    """torch.tensor() → mx.array(). Device param ignored (unified memory)."""
    if isinstance(data, mx.array):
        return data.astype(_map_dtype(dtype)) if dtype else data
    return mx.array(np.asarray(data), dtype=_map_dtype(dtype))

def zeros(*shape, dtype=None, device=None):
    shape = _normalize_shape(shape)
    return mx.zeros(shape, dtype=_map_dtype(dtype))

def ones(*shape, dtype=None, device=None):
    shape = _normalize_shape(shape)
    return mx.ones(shape, dtype=_map_dtype(dtype))

def zeros_like(x, dtype=None, device=None):
    return mx.zeros_like(x) if dtype is None else mx.zeros(x.shape, dtype=_map_dtype(dtype))

def ones_like(x, dtype=None, device=None):
    return mx.ones_like(x) if dtype is None else mx.ones(x.shape, dtype=_map_dtype(dtype))

def full(shape, fill_value, dtype=None, device=None):
    return mx.full(shape, fill_value, dtype=_map_dtype(dtype))

def arange(*args, dtype=None, device=None):
    return mx.arange(*args, dtype=_map_dtype(dtype))

def linspace(start, end, steps, dtype=None, device=None):
    return mx.linspace(start, end, steps, dtype=_map_dtype(dtype))

def eye(n, m=None, dtype=None, device=None):
    return mx.eye(n, m or n, dtype=_map_dtype(dtype))

# === Tensor Manipulation ===
def cat(tensors, dim=0):
    return mx.concatenate(tensors, axis=dim)

def stack(tensors, dim=0):
    return mx.stack(tensors, axis=dim)

def split(tensor, split_size_or_sections, dim=0):
    # Handle both int and list inputs
    ...

def chunk(tensor, chunks, dim=0):
    return mx.split(tensor, chunks, axis=dim)

def unsqueeze(x, dim):
    return mx.expand_dims(x, axis=dim)

def squeeze(x, dim=None):
    return mx.squeeze(x, axis=dim)

def flatten(x, start_dim=0, end_dim=-1):
    return mx.flatten(x, start_axis=start_dim, end_axis=end_dim)

def clamp(x, min=None, max=None):
    return mx.clip(x, min, max)

def where(condition, x, y):
    return mx.where(condition, x, y)

def einsum(equation, *operands):
    return mx.einsum(equation, *operands)

# === Dtype Mapping ===
def _map_dtype(dtype):
    """Map torch dtype strings/objects to mlx dtypes."""
    if dtype is None:
        return None
    DTYPE_MAP = {
        'float32': mx.float32, 'float16': mx.float16, 'bfloat16': mx.bfloat16,
        'int32': mx.int32, 'int64': mx.int32,  # MLX int64 limited on GPU
        'int8': mx.int8, 'bool': mx.bool_,
    }
    if isinstance(dtype, str):
        return DTYPE_MAP.get(dtype, mx.float32)
    return dtype  # Already an mx.Dtype

# === Device Handling (no-ops) ===
float32 = mx.float32
float16 = mx.float16
bfloat16 = mx.bfloat16
int32 = mx.int32
int64 = mx.int32  # Mapped to int32 for GPU compat
bool_ = mx.bool_

class _DeviceStub:
    """Absorbs .to(device), .cuda(), .cpu() calls."""
    def __init__(self, type_str="mps"):
        self.type = type_str

device = _DeviceStub

def no_grad():
    """Context manager that does nothing (MLX has no grad tracking by default)."""
    import contextlib
    return contextlib.nullcontext()
```

### 2. `compat/nn_modules.py` — Base Module Adapter

```python
import mlx.nn as _nn
import mlx.core as mx

class Module(_nn.Module):
    """Adapter that adds torch.nn.Module-like API to mlx.nn.Module."""

    def to(self, device=None, dtype=None):
        """No-op for device. Handles dtype conversion if specified."""
        return self

    def train(self, mode=True):
        if mode:
            _nn.Module.train(self)
        else:
            _nn.Module.eval(self)
        return self

    def eval(self):
        return self.train(False)

    def named_parameters(self):
        """Yield (name, parameter) pairs like PyTorch."""
        from mlx.utils import tree_flatten
        return tree_flatten(self.parameters())

    def requires_grad_(self, requires_grad=True):
        """No-op — MLX handles grads via value_and_grad."""
        return self

    def state_dict(self):
        """Return parameter dict (for weight loading compatibility)."""
        from mlx.utils import tree_flatten
        return dict(tree_flatten(self.parameters()))

    def load_state_dict(self, state_dict, strict=True):
        """Load weights from dict. Maps torch key names if needed."""
        from mlx.utils import tree_unflatten
        weights = list(state_dict.items())
        self.load_weights(weights)

    def register_buffer(self, name, tensor):
        """Store a non-parameter tensor as an attribute."""
        setattr(self, name, tensor)

    def num_parameters(self):
        """Count total parameters."""
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))
```

### 3. `compat/nn_layers.py` — All 33 nn Modules Used by Upstream

Map every nn module found in the upstream scan:

```python
import mlx.nn as _nn
import mlx.core as mx
from .nn_modules import Module

# === Direct Mappings (identical API) ===
Linear = _nn.Linear
LayerNorm = _nn.LayerNorm
Embedding = _nn.Embedding
Dropout = _nn.Dropout
RMSNorm = _nn.RMSNorm
GroupNorm = _nn.GroupNorm
Conv1d = _nn.Conv1d
Conv2d = _nn.Conv2d

# === Activation Modules ===
class ReLU(Module):
    def __call__(self, x): return _nn.relu(x)

class GELU(Module):
    def __call__(self, x): return _nn.gelu(x)

class SiLU(Module):
    def __call__(self, x): return _nn.silu(x)

class Mish(Module):
    def __call__(self, x): return x * mx.tanh(_nn.softplus(x))

class Tanh(Module):
    def __call__(self, x): return mx.tanh(x)

class Sigmoid(Module):
    def __call__(self, x): return mx.sigmoid(x)

class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self._alpha = alpha
    def __call__(self, x): return _nn.elu(x, self._alpha)

# === Sequential & Container ===
Sequential = _nn.Sequential

class ModuleList(Module):
    """torch.nn.ModuleList → list stored as indexed children."""
    def __init__(self, modules=None):
        super().__init__()
        self._modules_list = list(modules or [])
        for i, m in enumerate(self._modules_list):
            setattr(self, f"module_{i}", m)

    def __getitem__(self, idx): return self._modules_list[idx]
    def __len__(self): return len(self._modules_list)
    def __iter__(self): return iter(self._modules_list)
    def append(self, module):
        setattr(self, f"module_{len(self._modules_list)}", module)
        self._modules_list.append(module)

# === Parameter ===
class Parameter:
    """torch.nn.Parameter → just mx.array (MLX auto-discovers attrs)."""
    def __new__(cls, data, requires_grad=True):
        return mx.array(data) if not isinstance(data, mx.array) else data

# === Attention ===
class MultiheadAttention(Module):
    """Wrapper around mlx.nn.MultiHeadAttention with torch-compatible API."""
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 batch_first=True, kdim=None, vdim=None):
        super().__init__()
        self.mha = _nn.MultiHeadAttention(
            dims=embed_dim,
            num_heads=num_heads,
            bias=bias,
            query_input_dims=embed_dim,
            key_input_dims=kdim or embed_dim,
            value_input_dims=vdim or embed_dim,
        )
        self.batch_first = batch_first

    def __call__(self, query, key, value, attn_mask=None, need_weights=False):
        # torch returns (output, weights), mlx returns just output
        out = self.mha(query, key, value, mask=attn_mask)
        if need_weights:
            return out, None  # weights not easily extractable
        return out, None

# === Normalization ===
class BatchNorm2d(Module):
    """Minimal BatchNorm2d using running stats."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = _nn.BatchNorm(num_features, eps=eps, momentum=momentum)

    def __call__(self, x):
        # x: (B, C, H, W) → transpose to (B, H, W, C) for MLX
        x = mx.transpose(x, axes=(0, 2, 3, 1))
        x = self.bn(x)
        return mx.transpose(x, axes=(0, 3, 1, 2))

# === Conv3d (not in MLX, custom implementation) ===
class Conv3d(Module):
    """Conv3d via reshape to batch of Conv2d slices."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        # Implementation: decompose into spatial Conv2d + temporal reduction
        # Full implementation in PRD-02
        ...

# === TransformerEncoder (compose from MLX primitives) ===
# MLX has nn.TransformerEncoder — use it directly or wrap
TransformerEncoder = _nn.TransformerEncoder

class TransformerEncoderLayer(Module):
    """Single transformer encoder layer with torch-compatible API."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', batch_first=True, norm_first=False):
        super().__init__()
        self.self_attn = _nn.MultiHeadAttention(d_model, nhead)
        self.linear1 = _nn.Linear(d_model, dim_feedforward)
        self.linear2 = _nn.Linear(dim_feedforward, d_model)
        self.norm1 = _nn.LayerNorm(d_model)
        self.norm2 = _nn.LayerNorm(d_model)
        self.dropout = _nn.Dropout(dropout)
        self._act = _nn.relu if activation == 'relu' else _nn.gelu
        self.norm_first = norm_first

    def __call__(self, src, src_mask=None, src_key_padding_mask=None):
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask)
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src, src_mask))
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(self, x, mask):
        return self.dropout(self.self_attn(x, x, x, mask=mask))

    def _ff_block(self, x):
        return self.dropout(self.linear2(self.dropout(self._act(self.linear1(x)))))
```

---

## Acceptance Criteria

1. `from lerobot_mlx.compat import tensor_ops as torch_ops` — all creation functions work
2. `from lerobot_mlx.compat.nn_modules import Module` — subclassing works, `.to()` is no-op
3. `from lerobot_mlx.compat.nn_layers import Linear, Conv2d, MultiheadAttention` — all 33 types importable
4. 80+ tests in `test_compat_core.py` covering:
   - Every tensor creation function (zeros, ones, arange, etc.)
   - Module base class (to, train, eval, state_dict, load_state_dict)
   - Every nn layer (forward pass with random input)
   - Shape correctness for all operations
5. Cross-framework tests (requires_torch marker): compare output shapes and values vs PyTorch

---

## Test Strategy

```python
# test_compat_core.py

class TestTensorOps:
    def test_zeros_shape(self):
        from lerobot_mlx.compat.tensor_ops import zeros
        x = zeros(3, 4)
        assert x.shape == (3, 4)

    def test_cat(self):
        a = mx.ones((2, 3))
        b = mx.ones((2, 3))
        from lerobot_mlx.compat.tensor_ops import cat
        result = cat([a, b], dim=0)
        assert result.shape == (4, 3)

    @pytest.mark.requires_torch
    def test_tensor_ops_match_torch(self):
        """Cross-framework: verify outputs match PyTorch."""
        import torch
        from lerobot_mlx.compat import tensor_ops as T
        # Compare zeros, ones, arange, linspace, etc.

class TestModuleBase:
    def test_to_is_noop(self):
        from lerobot_mlx.compat.nn_modules import Module
        m = Module()
        assert m.to("cuda") is m
        assert m.to(device="mps") is m

class TestNNLayers:
    def test_linear_forward(self):
        from lerobot_mlx.compat.nn_layers import Linear
        layer = Linear(10, 5)
        x = mx.random.normal((2, 10))
        out = layer(x)
        assert out.shape == (2, 5)

    def test_multihead_attention(self):
        from lerobot_mlx.compat.nn_layers import MultiheadAttention
        mha = MultiheadAttention(embed_dim=64, num_heads=8)
        x = mx.random.normal((2, 10, 64))
        out, _ = mha(x, x, x)
        assert out.shape == (2, 10, 64)
```

---

## Notes

- `Conv3d` is used only by TD-MPC. Can be stubbed initially, full impl in PRD-11.
- `ModuleList` must properly register children so `model.parameters()` discovers them.
- `Parameter` is just `mx.array` — MLX auto-discovers all array attributes as parameters.
- The `MultiheadAttention` wrapper must handle both `batch_first=True` and `False`.
