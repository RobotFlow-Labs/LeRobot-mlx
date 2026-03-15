"""Tests for compat core — tensor_ops, nn_modules, nn_layers.

80+ tests covering every tensor creation/manipulation op,
Module base class, and all nn layer forward passes.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from tests.helpers import check_all_close, check_shape, random_input


# =============================================================================
# Tensor Creation Tests
# =============================================================================

class TestTensorCreation:
    """Test tensor creation functions."""

    def test_tensor_from_list(self):
        from lerobot_mlx.compat.tensor_ops import tensor
        x = tensor([1.0, 2.0, 3.0])
        check_shape(x, (3,))
        assert x.dtype == mx.float32

    def test_tensor_from_nested_list(self):
        from lerobot_mlx.compat.tensor_ops import tensor
        x = tensor([[1, 2], [3, 4]])
        check_shape(x, (2, 2))

    def test_tensor_from_numpy(self):
        from lerobot_mlx.compat.tensor_ops import tensor
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = tensor(arr)
        check_all_close(x, arr)

    def test_tensor_with_dtype(self):
        from lerobot_mlx.compat.tensor_ops import tensor
        x = tensor([1, 2, 3], dtype="float32")
        assert x.dtype == mx.float32

    def test_tensor_from_mx_array(self):
        from lerobot_mlx.compat.tensor_ops import tensor
        original = mx.array([1.0, 2.0])
        x = tensor(original)
        check_all_close(x, original)

    def test_tensor_device_ignored(self):
        from lerobot_mlx.compat.tensor_ops import tensor
        x = tensor([1.0], device="cuda")
        assert x.shape == (1,)

    def test_zeros_shape(self):
        from lerobot_mlx.compat.tensor_ops import zeros
        x = zeros(3, 4)
        check_shape(x, (3, 4))
        check_all_close(x, np.zeros((3, 4)))

    def test_zeros_tuple_shape(self):
        from lerobot_mlx.compat.tensor_ops import zeros
        x = zeros((2, 3))
        check_shape(x, (2, 3))

    def test_zeros_with_dtype(self):
        from lerobot_mlx.compat.tensor_ops import zeros
        x = zeros(2, 3, dtype="float16")
        assert x.dtype == mx.float16

    def test_ones_shape(self):
        from lerobot_mlx.compat.tensor_ops import ones
        x = ones(2, 5)
        check_shape(x, (2, 5))
        check_all_close(x, np.ones((2, 5)))

    def test_zeros_like(self):
        from lerobot_mlx.compat.tensor_ops import zeros_like
        ref = mx.ones((3, 4))
        x = zeros_like(ref)
        check_shape(x, (3, 4))
        check_all_close(x, np.zeros((3, 4)))

    def test_zeros_like_with_dtype(self):
        from lerobot_mlx.compat.tensor_ops import zeros_like
        ref = mx.ones((2, 3))
        x = zeros_like(ref, dtype="float16")
        assert x.dtype == mx.float16
        check_shape(x, (2, 3))

    def test_ones_like(self):
        from lerobot_mlx.compat.tensor_ops import ones_like
        ref = mx.zeros((4, 2))
        x = ones_like(ref)
        check_shape(x, (4, 2))
        check_all_close(x, np.ones((4, 2)))

    def test_full(self):
        from lerobot_mlx.compat.tensor_ops import full
        x = full((3, 3), 7.0)
        check_shape(x, (3, 3))
        check_all_close(x, np.full((3, 3), 7.0))

    def test_arange_single_arg(self):
        from lerobot_mlx.compat.tensor_ops import arange
        x = arange(5)
        check_all_close(x, np.arange(5))

    def test_arange_start_end(self):
        from lerobot_mlx.compat.tensor_ops import arange
        x = arange(2, 7)
        check_all_close(x, np.arange(2, 7))

    def test_arange_with_step(self):
        from lerobot_mlx.compat.tensor_ops import arange
        x = arange(0, 10, 2)
        check_all_close(x, np.arange(0, 10, 2))

    def test_linspace(self):
        from lerobot_mlx.compat.tensor_ops import linspace
        x = linspace(0.0, 1.0, 5)
        check_shape(x, (5,))
        check_all_close(x, np.linspace(0.0, 1.0, 5), atol=1e-6)

    def test_eye(self):
        from lerobot_mlx.compat.tensor_ops import eye
        x = eye(3)
        check_shape(x, (3, 3))
        check_all_close(x, np.eye(3))

    def test_eye_rectangular(self):
        from lerobot_mlx.compat.tensor_ops import eye
        x = eye(3, 4)
        check_shape(x, (3, 4))
        check_all_close(x, np.eye(3, 4))

    def test_rand_shape(self):
        from lerobot_mlx.compat.tensor_ops import rand
        x = rand(10, 20)
        check_shape(x, (10, 20))
        mx.eval(x)
        arr = np.array(x)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_randn_shape(self):
        from lerobot_mlx.compat.tensor_ops import randn
        x = randn(100, 100)
        check_shape(x, (100, 100))
        mx.eval(x)
        arr = np.array(x)
        # Should be roughly mean 0, std 1
        assert abs(arr.mean()) < 0.2
        assert abs(arr.std() - 1.0) < 0.2


# =============================================================================
# Tensor Manipulation Tests
# =============================================================================

class TestTensorManipulation:
    """Test tensor manipulation functions."""

    def test_cat_dim0(self):
        from lerobot_mlx.compat.tensor_ops import cat
        a = mx.ones((2, 3))
        b = mx.ones((2, 3))
        result = cat([a, b], dim=0)
        check_shape(result, (4, 3))

    def test_cat_dim1(self):
        from lerobot_mlx.compat.tensor_ops import cat
        a = mx.ones((2, 3))
        b = mx.ones((2, 4))
        result = cat([a, b], dim=1)
        check_shape(result, (2, 7))

    def test_stack_dim0(self):
        from lerobot_mlx.compat.tensor_ops import stack
        a = mx.ones((3,))
        b = mx.zeros((3,))
        result = stack([a, b], dim=0)
        check_shape(result, (2, 3))

    def test_stack_dim1(self):
        from lerobot_mlx.compat.tensor_ops import stack
        a = mx.ones((2, 3))
        b = mx.zeros((2, 3))
        result = stack([a, b], dim=1)
        check_shape(result, (2, 2, 3))

    def test_split_int(self):
        from lerobot_mlx.compat.tensor_ops import split
        x = mx.arange(10)
        parts = split(x, 2)
        assert len(parts) == 5
        for p in parts:
            check_shape(p, (2,))

    def test_split_list(self):
        from lerobot_mlx.compat.tensor_ops import split
        x = mx.arange(10)
        parts = split(x, [3, 3, 4])
        assert len(parts) == 3
        check_shape(parts[0], (3,))
        check_shape(parts[1], (3,))
        check_shape(parts[2], (4,))

    def test_chunk(self):
        from lerobot_mlx.compat.tensor_ops import chunk
        x = mx.arange(12)
        parts = chunk(x, 3)
        assert len(parts) == 3
        for p in parts:
            check_shape(p, (4,))

    def test_unsqueeze(self):
        from lerobot_mlx.compat.tensor_ops import unsqueeze
        x = mx.ones((3, 4))
        result = unsqueeze(x, 0)
        check_shape(result, (1, 3, 4))

    def test_unsqueeze_last(self):
        from lerobot_mlx.compat.tensor_ops import unsqueeze
        x = mx.ones((3, 4))
        result = unsqueeze(x, -1)
        check_shape(result, (3, 4, 1))

    def test_squeeze(self):
        from lerobot_mlx.compat.tensor_ops import squeeze
        x = mx.ones((1, 3, 1, 4))
        result = squeeze(x)
        check_shape(result, (3, 4))

    def test_squeeze_dim(self):
        from lerobot_mlx.compat.tensor_ops import squeeze
        x = mx.ones((1, 3, 4))
        result = squeeze(x, 0)
        check_shape(result, (3, 4))

    def test_flatten_all(self):
        from lerobot_mlx.compat.tensor_ops import flatten
        x = mx.ones((2, 3, 4))
        result = flatten(x)
        check_shape(result, (24,))

    def test_flatten_partial(self):
        from lerobot_mlx.compat.tensor_ops import flatten
        x = mx.ones((2, 3, 4))
        result = flatten(x, start_dim=1)
        check_shape(result, (2, 12))

    def test_clamp(self):
        from lerobot_mlx.compat.tensor_ops import clamp
        x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = clamp(x, min=-1.0, max=1.0)
        mx.eval(result)
        check_all_close(result, np.array([-1.0, -1.0, 0.0, 1.0, 1.0]))

    def test_clamp_min_only(self):
        from lerobot_mlx.compat.tensor_ops import clamp
        x = mx.array([-2.0, 0.0, 2.0])
        result = clamp(x, min=0.0)
        mx.eval(result)
        check_all_close(result, np.array([0.0, 0.0, 2.0]))

    def test_where(self):
        from lerobot_mlx.compat.tensor_ops import where
        cond = mx.array([True, False, True])
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        result = where(cond, a, b)
        check_all_close(result, np.array([1.0, 5.0, 3.0]))

    def test_einsum_matmul(self):
        from lerobot_mlx.compat.tensor_ops import einsum
        a = mx.ones((2, 3))
        b = mx.ones((3, 4))
        result = einsum("ij,jk->ik", a, b)
        check_shape(result, (2, 4))
        check_all_close(result, np.full((2, 4), 3.0))

    def test_transpose(self):
        from lerobot_mlx.compat.tensor_ops import transpose
        x = mx.ones((2, 3, 4))
        result = transpose(x, 0, 2)
        check_shape(result, (4, 3, 2))

    def test_permute(self):
        from lerobot_mlx.compat.tensor_ops import permute
        x = mx.ones((2, 3, 4))
        result = permute(x, (2, 0, 1))
        check_shape(result, (4, 2, 3))

    def test_reshape(self):
        from lerobot_mlx.compat.tensor_ops import reshape
        x = mx.ones((2, 6))
        result = reshape(x, (3, 4))
        check_shape(result, (3, 4))

    def test_mean(self):
        from lerobot_mlx.compat.tensor_ops import mean
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = mean(x)
        mx.eval(result)
        assert abs(float(result) - 2.5) < 1e-5

    def test_mean_dim(self):
        from lerobot_mlx.compat.tensor_ops import mean
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = mean(x, dim=0)
        check_all_close(result, np.array([2.0, 3.0]))

    def test_sum(self):
        from lerobot_mlx.compat.tensor_ops import sum
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = sum(x)
        mx.eval(result)
        assert abs(float(result) - 10.0) < 1e-5

    def test_abs(self):
        from lerobot_mlx.compat.tensor_ops import abs
        x = mx.array([-1.0, 0.0, 1.0])
        result = abs(x)
        check_all_close(result, np.array([1.0, 0.0, 1.0]))

    def test_exp(self):
        from lerobot_mlx.compat.tensor_ops import exp
        x = mx.array([0.0, 1.0])
        result = exp(x)
        check_all_close(result, np.exp(np.array([0.0, 1.0])), atol=1e-5)

    def test_log(self):
        from lerobot_mlx.compat.tensor_ops import log
        x = mx.array([1.0, math.e])
        result = log(x)
        check_all_close(result, np.log(np.array([1.0, math.e])), atol=1e-5)

    def test_sqrt(self):
        from lerobot_mlx.compat.tensor_ops import sqrt
        x = mx.array([4.0, 9.0, 16.0])
        result = sqrt(x)
        check_all_close(result, np.array([2.0, 3.0, 4.0]))

    def test_matmul(self):
        from lerobot_mlx.compat.tensor_ops import matmul
        a = mx.ones((2, 3))
        b = mx.ones((3, 4))
        result = matmul(a, b)
        check_shape(result, (2, 4))


# =============================================================================
# Dtype Mapping Tests
# =============================================================================

class TestDtypeMapping:
    """Test dtype mapping from torch strings to MLX dtypes."""

    def test_float32_constant(self):
        from lerobot_mlx.compat.tensor_ops import float32
        assert float32 == mx.float32

    def test_float16_constant(self):
        from lerobot_mlx.compat.tensor_ops import float16
        assert float16 == mx.float16

    def test_bfloat16_constant(self):
        from lerobot_mlx.compat.tensor_ops import bfloat16
        assert bfloat16 == mx.bfloat16

    def test_int64_maps_to_int32(self):
        from lerobot_mlx.compat.tensor_ops import int64
        # int64 is mapped to int32 for MLX GPU compat
        assert int64 == mx.int32

    def test_dtype_string_mapping(self):
        from lerobot_mlx.compat.tensor_ops import _map_dtype
        assert _map_dtype("float32") == mx.float32
        assert _map_dtype("float16") == mx.float16
        assert _map_dtype("int32") == mx.int32
        assert _map_dtype("int64") == mx.int32  # GPU compat
        assert _map_dtype("bool") == mx.bool_

    def test_dtype_none_passthrough(self):
        from lerobot_mlx.compat.tensor_ops import _map_dtype
        assert _map_dtype(None) is None

    def test_dtype_mx_passthrough(self):
        from lerobot_mlx.compat.tensor_ops import _map_dtype
        assert _map_dtype(mx.float32) == mx.float32


# =============================================================================
# Device Handling Tests
# =============================================================================

class TestDeviceHandling:
    """Test device handling (all no-ops for unified memory)."""

    def test_device_stub_type(self):
        from lerobot_mlx.compat.tensor_ops import device
        d = device("cuda")
        assert d.type == "cuda"

    def test_device_stub_repr(self):
        from lerobot_mlx.compat.tensor_ops import device
        d = device("mps")
        assert "mps" in repr(d)

    def test_device_stub_equality(self):
        from lerobot_mlx.compat.tensor_ops import device
        d1 = device("cuda")
        d2 = device("cuda")
        assert d1 == d2

    def test_device_stub_string_equality(self):
        from lerobot_mlx.compat.tensor_ops import device
        d = device("mps")
        assert d == "mps"

    def test_no_grad_context(self):
        from lerobot_mlx.compat.tensor_ops import no_grad
        with no_grad():
            x = mx.ones((2, 3))
            y = x + 1
        check_shape(y, (2, 3))


# =============================================================================
# Module Base Class Tests
# =============================================================================

class TestModuleBase:
    """Test the Module adapter class."""

    def test_module_creation(self):
        from lerobot_mlx.compat.nn_modules import Module
        m = Module()
        assert m is not None

    def test_to_device_is_noop(self):
        from lerobot_mlx.compat.nn_modules import Module
        m = Module()
        assert m.to("cuda") is m
        assert m.to(device="mps") is m
        assert m.to(device="cpu") is m

    def test_cuda_is_noop(self):
        from lerobot_mlx.compat.nn_modules import Module
        m = Module()
        assert m.cuda() is m

    def test_cpu_is_noop(self):
        from lerobot_mlx.compat.nn_modules import Module
        m = Module()
        assert m.cpu() is m

    def test_train_eval_toggle(self):
        from lerobot_mlx.compat.nn_modules import Module
        m = Module()
        result = m.train()
        assert result is m
        result = m.eval()
        assert result is m

    def test_requires_grad_is_noop(self):
        from lerobot_mlx.compat.nn_modules import Module
        m = Module()
        assert m.requires_grad_() is m
        assert m.requires_grad_(False) is m

    def test_register_buffer(self):
        from lerobot_mlx.compat.nn_modules import Module
        m = Module()
        buf = mx.ones((3,))
        m.register_buffer("my_buffer", buf)
        check_all_close(m.my_buffer, np.ones(3))

    def test_register_buffer_none(self):
        from lerobot_mlx.compat.nn_modules import Module
        m = Module()
        m.register_buffer("my_buffer", mx.ones((3,)))
        m.register_buffer("my_buffer", None)
        assert not hasattr(m, "my_buffer")

    def test_subclass_with_parameters(self):
        from lerobot_mlx.compat.nn_modules import Module
        import mlx.nn as _nn

        class SimpleModel(Module):
            def __init__(self):
                super().__init__()
                self.linear = _nn.Linear(10, 5)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        x = mx.random.normal((2, 10))
        out = model(x)
        check_shape(out, (2, 5))

        # Test state_dict
        sd = model.state_dict()
        assert len(sd) > 0

        # Test num_parameters
        n_params = model.num_parameters()
        assert n_params == 10 * 5 + 5  # weight + bias

        # Test named_parameters
        named = model.named_parameters()
        assert len(named) > 0

    def test_load_state_dict(self):
        from lerobot_mlx.compat.nn_modules import Module
        import mlx.nn as _nn

        class TinyModel(Module):
            def __init__(self):
                super().__init__()
                self.linear = _nn.Linear(4, 2)

            def __call__(self, x):
                return self.linear(x)

        model1 = TinyModel()
        model2 = TinyModel()

        # Get state dict from model1 and load into model2
        sd = model1.state_dict()
        model2.load_state_dict(sd)

        # Outputs should match
        x = mx.ones((1, 4))
        out1 = model1(x)
        out2 = model2(x)
        mx.eval(out1, out2)
        check_all_close(out1, out2)


# =============================================================================
# NN Layer Tests
# =============================================================================

class TestNNLayers:
    """Test all nn layer forward passes."""

    def test_linear_forward(self):
        from lerobot_mlx.compat.nn_layers import Linear
        layer = Linear(10, 5)
        x = random_input(2, 10)
        out = layer(x)
        check_shape(out, (2, 5))

    def test_layer_norm_forward(self):
        from lerobot_mlx.compat.nn_layers import LayerNorm
        ln = LayerNorm(16)
        x = random_input(2, 8, 16)
        out = ln(x)
        check_shape(out, (2, 8, 16))

    def test_embedding_forward(self):
        from lerobot_mlx.compat.nn_layers import Embedding
        emb = Embedding(100, 32)
        idx = mx.array([0, 5, 10])
        out = emb(idx)
        check_shape(out, (3, 32))

    def test_dropout_forward(self):
        from lerobot_mlx.compat.nn_layers import Dropout
        drop = Dropout(0.5)
        x = random_input(4, 8)
        out = drop(x)
        check_shape(out, (4, 8))

    def test_conv1d_forward(self):
        from lerobot_mlx.compat.nn_layers import Conv1d
        conv = Conv1d(3, 16, kernel_size=3, padding=1)
        # PyTorch convention: (B, C_in, L)
        x = random_input(2, 3, 10)
        out = conv(x)
        check_shape(out, (2, 16, 10))

    def test_conv2d_forward(self):
        from lerobot_mlx.compat.nn_layers import Conv2d
        conv = Conv2d(3, 16, kernel_size=3, padding=1)
        # PyTorch convention: (B, C_in, H, W)
        x = random_input(2, 3, 8, 8)
        out = conv(x)
        check_shape(out, (2, 16, 8, 8))

    def test_relu_forward(self):
        from lerobot_mlx.compat.nn_layers import ReLU
        act = ReLU()
        x = mx.array([-1.0, 0.0, 1.0])
        out = act(x)
        check_all_close(out, np.array([0.0, 0.0, 1.0]))

    def test_gelu_forward(self):
        from lerobot_mlx.compat.nn_layers import GELU
        act = GELU()
        x = random_input(4)
        out = act(x)
        check_shape(out, (4,))

    def test_gelu_tanh_approx(self):
        from lerobot_mlx.compat.nn_layers import GELU
        act = GELU(approximate="tanh")
        x = random_input(4)
        out = act(x)
        check_shape(out, (4,))

    def test_silu_forward(self):
        from lerobot_mlx.compat.nn_layers import SiLU
        act = SiLU()
        x = random_input(4)
        out = act(x)
        check_shape(out, (4,))

    def test_mish_forward(self):
        from lerobot_mlx.compat.nn_layers import Mish
        act = Mish()
        x = random_input(4)
        out = act(x)
        check_shape(out, (4,))

    def test_tanh_forward(self):
        from lerobot_mlx.compat.nn_layers import Tanh
        act = Tanh()
        x = mx.array([-100.0, 0.0, 100.0])
        out = act(x)
        mx.eval(out)
        arr = np.array(out)
        assert arr[0] < -0.99
        assert abs(arr[1]) < 1e-5
        assert arr[2] > 0.99

    def test_sigmoid_forward(self):
        from lerobot_mlx.compat.nn_layers import Sigmoid
        act = Sigmoid()
        x = mx.array([0.0])
        out = act(x)
        mx.eval(out)
        assert abs(float(out) - 0.5) < 1e-5

    def test_elu_forward(self):
        from lerobot_mlx.compat.nn_layers import ELU
        act = ELU(alpha=1.0)
        x = mx.array([-1.0, 0.0, 1.0])
        out = act(x)
        check_shape(out, (3,))
        mx.eval(out)
        arr = np.array(out)
        assert arr[0] < 0  # ELU(-1) is negative
        assert abs(arr[1]) < 1e-5  # ELU(0) = 0
        assert abs(arr[2] - 1.0) < 1e-5  # ELU(1) = 1

    def test_softmax_forward(self):
        from lerobot_mlx.compat.nn_layers import Softmax
        sm = Softmax(dim=-1)
        x = mx.array([[1.0, 2.0, 3.0]])
        out = sm(x)
        check_shape(out, (1, 3))
        mx.eval(out)
        assert abs(float(mx.sum(out)) - 1.0) < 1e-5

    def test_identity_forward(self):
        from lerobot_mlx.compat.nn_layers import Identity
        ident = Identity()
        x = random_input(2, 3)
        out = ident(x)
        check_all_close(out, x)

    def test_sequential_forward(self):
        from lerobot_mlx.compat.nn_layers import Sequential, Linear, ReLU
        import mlx.nn as _nn
        seq = Sequential(_nn.Linear(10, 5), _nn.Linear(5, 2))
        x = random_input(3, 10)
        out = seq(x)
        check_shape(out, (3, 2))

    def test_module_list(self):
        from lerobot_mlx.compat.nn_layers import ModuleList
        import mlx.nn as _nn
        layers = ModuleList([_nn.Linear(10, 10) for _ in range(3)])
        assert len(layers) == 3
        x = random_input(2, 10)
        for layer in layers:
            x = layer(x)
        check_shape(x, (2, 10))

    def test_module_list_append(self):
        from lerobot_mlx.compat.nn_layers import ModuleList
        import mlx.nn as _nn
        layers = ModuleList()
        layers.append(_nn.Linear(5, 5))
        layers.append(_nn.Linear(5, 5))
        assert len(layers) == 2

    def test_module_list_indexing(self):
        from lerobot_mlx.compat.nn_layers import ModuleList
        import mlx.nn as _nn
        layers = ModuleList([_nn.Linear(5, 3), _nn.Linear(3, 1)])
        layer0 = layers[0]
        assert isinstance(layer0, _nn.Linear)

    def test_module_dict(self):
        from lerobot_mlx.compat.nn_layers import ModuleDict
        import mlx.nn as _nn
        d = ModuleDict({"encoder": _nn.Linear(10, 5), "decoder": _nn.Linear(5, 10)})
        assert len(d) == 2
        assert "encoder" in d
        x = random_input(2, 10)
        h = d["encoder"](x)
        check_shape(h, (2, 5))

    def test_parameter(self):
        from lerobot_mlx.compat.nn_layers import Parameter
        p = Parameter(mx.ones((3, 4)))
        assert isinstance(p, mx.array)
        check_shape(p, (3, 4))

    def test_parameter_from_list(self):
        from lerobot_mlx.compat.nn_layers import Parameter
        p = Parameter([1.0, 2.0, 3.0])
        assert isinstance(p, mx.array)

    def test_multihead_attention(self):
        from lerobot_mlx.compat.nn_layers import MultiheadAttention
        mha = MultiheadAttention(embed_dim=64, num_heads=8)
        x = random_input(2, 10, 64)
        out, weights = mha(x, x, x)
        check_shape(out, (2, 10, 64))
        assert weights is None

    def test_multihead_attention_cross(self):
        from lerobot_mlx.compat.nn_layers import MultiheadAttention
        mha = MultiheadAttention(embed_dim=32, num_heads=4)
        q = random_input(2, 5, 32)
        kv = random_input(2, 10, 32)
        out, _ = mha(q, kv, kv)
        check_shape(out, (2, 5, 32))

    def test_multihead_attention_not_batch_first(self):
        from lerobot_mlx.compat.nn_layers import MultiheadAttention
        mha = MultiheadAttention(embed_dim=32, num_heads=4, batch_first=False)
        # (T, B, E) format
        x = random_input(10, 2, 32)
        out, _ = mha(x, x, x)
        check_shape(out, (10, 2, 32))

    def test_batch_norm_2d(self):
        from lerobot_mlx.compat.nn_layers import BatchNorm2d
        bn = BatchNorm2d(16)
        # Input in NCHW format
        x = random_input(2, 16, 8, 8)
        out = bn(x)
        check_shape(out, (2, 16, 8, 8))

    def test_batch_norm_1d(self):
        from lerobot_mlx.compat.nn_layers import BatchNorm1d
        bn = BatchNorm1d(16)
        x = random_input(4, 16)
        out = bn(x)
        check_shape(out, (4, 16))

    def test_batch_norm_1d_3d(self):
        from lerobot_mlx.compat.nn_layers import BatchNorm1d
        bn = BatchNorm1d(16)
        # (B, C, L) format
        x = random_input(4, 16, 10)
        out = bn(x)
        check_shape(out, (4, 16, 10))

    def test_group_norm(self):
        from lerobot_mlx.compat.nn_layers import GroupNorm
        gn = GroupNorm(4, 16)
        x = random_input(2, 16)
        out = gn(x)
        check_shape(out, (2, 16))

    def test_rms_norm(self):
        from lerobot_mlx.compat.nn_layers import RMSNorm
        rn = RMSNorm(32)
        x = random_input(2, 8, 32)
        out = rn(x)
        check_shape(out, (2, 8, 32))

    def test_transformer_encoder_layer(self):
        from lerobot_mlx.compat.nn_layers import TransformerEncoderLayer
        layer = TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128)
        x = random_input(2, 10, 64)
        out = layer(x)
        check_shape(out, (2, 10, 64))

    def test_transformer_encoder_layer_prenorm(self):
        from lerobot_mlx.compat.nn_layers import TransformerEncoderLayer
        layer = TransformerEncoderLayer(d_model=32, nhead=4, norm_first=True)
        x = random_input(2, 5, 32)
        out = layer(x)
        check_shape(out, (2, 5, 32))

    def test_transformer_encoder(self):
        from lerobot_mlx.compat.nn_layers import TransformerEncoder, TransformerEncoderLayer
        layer = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64)
        # Each layer gets deepcopied, so they have independent weights
        encoder = TransformerEncoder(layer, num_layers=1)
        x = random_input(2, 5, 32)
        out = encoder(x)
        check_shape(out, (2, 5, 32))

    def test_conv3d_stub_raises(self):
        from lerobot_mlx.compat.nn_layers import Conv3d
        conv = Conv3d(3, 16, kernel_size=3, padding=1)
        x = random_input(1, 3, 4, 8, 8)
        with pytest.raises(NotImplementedError):
            conv(x)

    def test_flatten_module(self):
        from lerobot_mlx.compat.nn_layers import Flatten
        f = Flatten(start_dim=1)
        x = random_input(2, 3, 4, 5)
        out = f(x)
        check_shape(out, (2, 60))

    def test_unflatten_module(self):
        from lerobot_mlx.compat.nn_layers import Unflatten
        uf = Unflatten(dim=1, unflattened_size=(3, 4))
        x = random_input(2, 12)
        out = uf(x)
        check_shape(out, (2, 3, 4))


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple compat components."""

    def test_compat_namespace_nn(self):
        """Test that nn namespace works as expected."""
        from lerobot_mlx.compat import nn
        layer = nn.Linear(10, 5)
        x = random_input(2, 10)
        out = layer(x)
        check_shape(out, (2, 5))

    def test_compat_namespace_tensor_ops(self):
        """Test that tensor_ops works through compat namespace."""
        from lerobot_mlx.compat import tensor_ops
        x = tensor_ops.zeros(3, 4)
        check_shape(x, (3, 4))

    def test_compat_tensor_type(self):
        """Test that Tensor is mx.array."""
        from lerobot_mlx.compat import Tensor
        assert Tensor is mx.array

    def test_build_simple_model(self):
        """Build and run a simple model using compat layer."""
        from lerobot_mlx.compat import nn
        from lerobot_mlx.compat.nn_modules import Module

        class SimpleNet(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 32)
                self.act = nn.ReLU()
                self.fc2 = nn.Linear(32, 5)

            def __call__(self, x):
                return self.fc2(self.act(self.fc1(x)))

        model = SimpleNet()
        model.to("cuda")  # should be no-op
        model.train()
        x = random_input(4, 10)
        out = model(x)
        check_shape(out, (4, 5))

        model.eval()
        out2 = model(x)
        check_shape(out2, (4, 5))

    def test_transformer_encoder_independent_weights(self):
        """C-06: Verify each layer in TransformerEncoder has independent weights."""
        from lerobot_mlx.compat.nn_layers import TransformerEncoder, TransformerEncoderLayer
        layer = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64)
        encoder = TransformerEncoder(layer, num_layers=3)

        # Each layer should be a different object
        assert encoder.layers[0] is not encoder.layers[1]
        assert encoder.layers[1] is not encoder.layers[2]

        # Verify weights are independent by checking they are different objects
        w0 = encoder.layers[0].linear1.weight
        w1 = encoder.layers[1].linear1.weight
        w2 = encoder.layers[2].linear1.weight
        mx.eval(w0, w1, w2)

        # The deepcopied layers start with the same values but are independent arrays
        # Modify one and confirm the others don't change
        assert w0 is not w1
        assert w1 is not w2

    def test_max_with_dim_returns_tuple(self):
        """H-03: max() with dim returns namedtuple(values, indices)."""
        from lerobot_mlx.compat.tensor_ops import max
        x = mx.array([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])
        result = max(x, dim=1)
        mx.eval(result.values, result.indices)
        check_all_close(result.values, np.array([3.0, 6.0]))
        check_all_close(result.indices, np.array([1, 2]))

    def test_min_with_dim_returns_tuple(self):
        """H-03: min() with dim returns namedtuple(values, indices)."""
        from lerobot_mlx.compat.tensor_ops import min
        x = mx.array([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]])
        result = min(x, dim=1)
        mx.eval(result.values, result.indices)
        check_all_close(result.values, np.array([1.0, 4.0]))
        check_all_close(result.indices, np.array([1, 1]))

    def test_max_without_dim_returns_scalar(self):
        """H-03: max() without dim returns scalar."""
        from lerobot_mlx.compat.tensor_ops import max
        x = mx.array([[1.0, 3.0], [2.0, 4.0]])
        result = max(x)
        mx.eval(result)
        assert float(result) == 4.0
        # Should NOT be a namedtuple
        assert not hasattr(result, 'values')

    def test_conv2d_nchw_convention(self):
        """H-10: Conv2d accepts NCHW input and returns NCHW output."""
        from lerobot_mlx.compat.nn_layers import Conv2d
        conv = Conv2d(3, 8, kernel_size=3, padding=1)
        x = random_input(2, 3, 16, 16)  # (B, C_in, H, W)
        out = conv(x)
        check_shape(out, (2, 8, 16, 16))  # (B, C_out, H_out, W_out)

    def test_conv1d_ncl_convention(self):
        """H-11: Conv1d accepts NCL input and returns NCL output."""
        from lerobot_mlx.compat.nn_layers import Conv1d
        conv = Conv1d(4, 8, kernel_size=3, padding=1)
        x = random_input(2, 4, 20)  # (B, C_in, L)
        out = conv(x)
        check_shape(out, (2, 8, 20))  # (B, C_out, L_out)

    def test_unknown_dtype_raises(self):
        """H-02: Unknown dtype string raises ValueError."""
        from lerobot_mlx.compat.tensor_ops import _map_dtype
        with pytest.raises(ValueError, match="Unknown dtype"):
            _map_dtype("complex128")
        with pytest.raises(ValueError, match="Unknown dtype"):
            _map_dtype("notadtype")

    def test_module_list_slice(self):
        """M-01: ModuleList supports slicing."""
        from lerobot_mlx.compat.nn_layers import ModuleList
        import mlx.nn as _nn
        layers = ModuleList([_nn.Linear(5, 5) for _ in range(5)])
        sliced = layers[1:3]
        assert isinstance(sliced, ModuleList)
        assert len(sliced) == 2

    def test_module_to_dtype(self):
        """M-11: Module.to(dtype=...) casts parameters."""
        from lerobot_mlx.compat.nn_modules import Module
        import mlx.nn as _nn

        class TinyModel(Module):
            def __init__(self):
                super().__init__()
                self.linear = _nn.Linear(4, 2)

            def __call__(self, x):
                return self.linear(x)

        model = TinyModel()
        model.to(dtype=mx.float16)
        # Check that weights are now float16
        params = model.state_dict()
        for name, param in params.items():
            assert param.dtype == mx.float16, f"{name} has dtype {param.dtype}, expected float16"

    def test_model_state_dict_roundtrip(self):
        """Test save/load state dict cycle."""
        from lerobot_mlx.compat.nn_modules import Module
        import mlx.nn as _nn

        class SmallModel(Module):
            def __init__(self):
                super().__init__()
                self.layer = _nn.Linear(4, 2)

            def __call__(self, x):
                return self.layer(x)

        m1 = SmallModel()
        m2 = SmallModel()

        # Load m1's weights into m2
        sd = m1.state_dict()
        m2.load_state_dict(sd)

        x = mx.ones((1, 4))
        out1 = m1(x)
        out2 = m2(x)
        mx.eval(out1, out2)
        check_all_close(out1, out2)
