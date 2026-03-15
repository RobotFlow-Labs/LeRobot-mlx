"""
torch.nn.functional → MLX functional operations.

Drop-in replacements for torch.nn.functional.* used across upstream LeRobot policies.
All functions accept and return mlx.core.array objects.

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

import math

import mlx.core as mx
import mlx.nn as _nn


# =============================================================================
# Loss Functions
# =============================================================================

def mse_loss(input: mx.array, target: mx.array, reduction: str = "mean") -> mx.array:
    """F.mse_loss — 17 uses upstream. Mean squared error loss."""
    diff = (input - target) ** 2
    if reduction == "mean":
        return mx.mean(diff)
    if reduction == "sum":
        return mx.sum(diff)
    return diff


def l1_loss(input: mx.array, target: mx.array, reduction: str = "mean") -> mx.array:
    """F.l1_loss — Mean absolute error loss."""
    diff = mx.abs(input - target)
    if reduction == "mean":
        return mx.mean(diff)
    if reduction == "sum":
        return mx.sum(diff)
    return diff


def smooth_l1_loss(
    input: mx.array,
    target: mx.array,
    beta: float = 1.0,
    reduction: str = "mean",
) -> mx.array:
    """F.smooth_l1_loss — Huber loss."""
    diff = mx.abs(input - target)
    loss = mx.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def cross_entropy(
    input: mx.array,
    target: mx.array,
    weight: mx.array | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> mx.array:
    """F.cross_entropy — wraps mlx.nn.losses.cross_entropy.

    Args:
        input: Logits of shape (N, C).
        target: Class indices of shape (N,) or probabilities of shape (N, C).
        weight: Per-class weights of shape (C,). Weights each sample's loss by
                the weight corresponding to its target class.
        reduction: 'mean', 'sum', or 'none'.
        label_smoothing: Amount of label smoothing [0, 1).
    """
    # MLX cross_entropy expects (logits, targets)
    loss = _nn.losses.cross_entropy(
        input, target, label_smoothing=label_smoothing, reduction="none"
    )
    if weight is not None:
        # weight: (num_classes,) -> select weight for each target
        w = weight[target]  # (N,) matching loss shape
        loss = loss * w
    if reduction == "mean":
        if weight is not None:
            return mx.sum(loss) / mx.sum(w)
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def binary_cross_entropy_with_logits(
    input: mx.array,
    target: mx.array,
    weight: mx.array | None = None,
    reduction: str = "mean",
) -> mx.array:
    """F.binary_cross_entropy_with_logits — BCE with sigmoid built in."""
    # Numerically stable: max(x, 0) - x*z + log(1 + exp(-|x|))
    relu_input = mx.maximum(input, 0.0)
    loss = relu_input - input * target + mx.log(1.0 + mx.exp(-mx.abs(input)))
    if weight is not None:
        loss = loss * weight
    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


# =============================================================================
# Activation Functions
# =============================================================================

def relu(x: mx.array, inplace: bool = False) -> mx.array:
    """F.relu activation."""
    return mx.maximum(x, 0.0)


def gelu(x: mx.array, approximate: str = "none") -> mx.array:
    """F.gelu activation."""
    if approximate == "tanh":
        return _nn.gelu_approx(x)
    return _nn.gelu(x)


def silu(x: mx.array, inplace: bool = False) -> mx.array:
    """F.silu (swish) activation."""
    return x * mx.sigmoid(x)


def sigmoid(x: mx.array) -> mx.array:
    """F.sigmoid activation."""
    return mx.sigmoid(x)


def tanh(x: mx.array) -> mx.array:
    """F.tanh activation."""
    return mx.tanh(x)


def softmax(x: mx.array, dim: int = -1) -> mx.array:
    """F.softmax — normalized exponential."""
    return mx.softmax(x, axis=dim)


def log_softmax(x: mx.array, dim: int = -1) -> mx.array:
    """F.log_softmax — log of softmax (numerically stable)."""
    # Numerically stable: x - log(sum(exp(x)))
    x_max = mx.stop_gradient(mx.max(x, axis=dim, keepdims=True))
    shifted = x - x_max
    log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=dim, keepdims=True))
    return shifted - log_sum_exp


def softplus(x: mx.array, beta: float = 1.0, threshold: float = 20.0) -> mx.array:
    """F.softplus — smooth approximation to ReLU."""
    # softplus(x) = (1/beta) * log(1 + exp(beta * x))
    # For large beta*x, use x directly for numerical stability
    scaled = beta * x
    return mx.where(
        scaled > threshold,
        x,
        (1.0 / beta) * mx.log(1.0 + mx.exp(scaled)),
    )


def elu(x: mx.array, alpha: float = 1.0, inplace: bool = False) -> mx.array:
    """F.elu activation."""
    return mx.where(x > 0, x, alpha * (mx.exp(x) - 1.0))


# =============================================================================
# Padding
# =============================================================================

def pad(
    input: mx.array,
    pad_widths: tuple | list,
    mode: str = "constant",
    value: float = 0,
) -> mx.array:
    """F.pad — 14 uses upstream.

    CRITICAL: torch pad format is (left, right, top, bottom, front, back, ...)
    as REVERSED pairs starting from the last dimension.
    MLX/numpy pad format is ((before_0, after_0), (before_1, after_1), ...)
    per axis starting from the first dimension.

    This function converts torch format to MLX format.

    Args:
        input: Input array.
        pad_widths: Padding in torch format: (left, right) for 1D,
                    (left, right, top, bottom) for 2D, etc.
        mode: 'constant', 'reflect', or 'replicate'.
        value: Fill value for constant padding.
    """
    n_pad_pairs = len(pad_widths) // 2
    ndim = input.ndim

    # Build MLX pad spec: leading dims get (0,0), then reversed torch pairs
    mlx_pad = [(0, 0)] * (ndim - n_pad_pairs)
    for i in range(n_pad_pairs - 1, -1, -1):
        mlx_pad.append((pad_widths[2 * i], pad_widths[2 * i + 1]))

    if mode == "constant":
        return mx.pad(input, mlx_pad, constant_values=value)
    elif mode == "reflect":
        # Validate reflect padding sizes
        for axis_idx in range(ndim):
            before, after = mlx_pad[axis_idx]
            dim_size = input.shape[axis_idx]
            if before >= dim_size or after >= dim_size:
                raise ValueError(
                    f"Reflect padding ({before}, {after}) must be less than "
                    f"dimension size ({dim_size}) for axis {axis_idx}"
                )
        # Reflect padding: implement manually
        result = input
        for axis in range(ndim):
            before, after = mlx_pad[axis]
            if before == 0 and after == 0:
                continue
            # Reflect pad along this axis
            slices_before = []
            slices_after = []
            if before > 0:
                # Take 'before' elements starting from index 1, reversed
                idx = list(range(1, before + 1))[::-1]
                # Clamp to valid range
                idx = [min(i, result.shape[axis] - 1) for i in idx]
                slices_before = [mx.take(result, mx.array(idx), axis=axis)]
            if after > 0:
                # Take 'after' elements ending at index -2, reversed
                n = result.shape[axis]
                idx = list(range(n - 2, n - 2 - after, -1))
                idx = [max(i, 0) for i in idx]
                slices_after = [mx.take(result, mx.array(idx), axis=axis)]
            parts = slices_before + [result] + slices_after
            result = mx.concatenate(parts, axis=axis)
        return result
    elif mode == "replicate":
        # Replicate (edge) padding: repeat edge values
        result = input
        for axis in range(ndim):
            before, after = mlx_pad[axis]
            if before == 0 and after == 0:
                continue
            parts = []
            if before > 0:
                # Take the first slice along this axis and repeat it
                first_slice = mx.take(result, mx.array([0]), axis=axis)
                # Broadcast to 'before' copies
                reps = [1] * result.ndim
                reps[axis] = before
                parts.append(mx.tile(first_slice, reps))
            parts.append(result)
            if after > 0:
                # Take the last slice along this axis and repeat it
                last_idx = result.shape[axis] - 1
                last_slice = mx.take(result, mx.array([last_idx]), axis=axis)
                reps = [1] * result.ndim
                reps[axis] = after
                parts.append(mx.tile(last_slice, reps))
            result = mx.concatenate(parts, axis=axis)
        return result
    else:
        raise NotImplementedError(
            f"Padding mode '{mode}' not implemented. Use 'constant', 'reflect', or 'replicate'."
        )


# =============================================================================
# Attention
# =============================================================================

def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    attn_mask: mx.array | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> mx.array:
    """F.scaled_dot_product_attention — 6 uses upstream.

    Args:
        query: (B, ..., Sq, D)
        key:   (B, ..., Sk, D)
        value: (B, ..., Sk, Dv)
        attn_mask: Additive mask (0 = attend, -inf = mask out).
        dropout_p: Dropout probability (ignored in MLX, no dropout in eval).
        is_causal: If True, apply causal (lower-triangular) mask.

    Returns:
        Output of shape (B, ..., Sq, Dv)
    """
    d_k = query.shape[-1]
    scale = d_k ** -0.5

    # (B, ..., Sq, D) @ (B, ..., D, Sk) -> (B, ..., Sq, Sk)
    # Transpose last two dims of key
    key_t_axes = list(range(key.ndim - 2)) + [key.ndim - 1, key.ndim - 2]
    scores = (query @ mx.transpose(key, axes=key_t_axes)) * scale

    if is_causal:
        seq_len_q = query.shape[-2]
        seq_len_k = key.shape[-2]
        causal_mask = mx.triu(
            mx.full((seq_len_q, seq_len_k), float("-inf")), k=1
        )
        scores = scores + causal_mask

    if attn_mask is not None:
        scores = scores + attn_mask

    weights = mx.softmax(scores, axis=-1)

    return weights @ value


# =============================================================================
# Normalization
# =============================================================================

def normalize(
    input: mx.array, p: float = 2.0, dim: int = -1, eps: float = 1e-12
) -> mx.array:
    """F.normalize — Lp normalization along a dimension."""
    if p == 2.0:
        norm = mx.sqrt(mx.sum(input ** 2, axis=dim, keepdims=True))
    else:
        norm = mx.sum(mx.abs(input) ** p, axis=dim, keepdims=True) ** (1.0 / p)
    return input / mx.maximum(norm, mx.array(eps))


def layer_norm(
    input: mx.array,
    normalized_shape: tuple | list,
    weight: mx.array | None = None,
    bias: mx.array | None = None,
    eps: float = 1e-5,
) -> mx.array:
    """F.layer_norm — layer normalization."""
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    axes = tuple(range(-len(normalized_shape), 0))
    mean = mx.mean(input, axis=axes, keepdims=True)
    var = mx.var(input, axis=axes, keepdims=True)
    x = (input - mean) / mx.sqrt(var + eps)
    if weight is not None:
        x = x * weight
    if bias is not None:
        x = x + bias
    return x


def group_norm(
    input: mx.array,
    num_groups: int,
    weight: mx.array | None = None,
    bias: mx.array | None = None,
    eps: float = 1e-5,
) -> mx.array:
    """F.group_norm — group normalization for diffusion models.

    Args:
        input: (N, C, *spatial) — channels-first format.
        num_groups: Number of groups to divide channels into.
        weight: Per-channel scale of shape (C,).
        bias: Per-channel shift of shape (C,).
        eps: Epsilon for numerical stability.
    """
    N, C = input.shape[:2]
    spatial = input.shape[2:]
    assert C % num_groups == 0, f"Channels {C} not divisible by num_groups {num_groups}"

    x = input.reshape(N, num_groups, C // num_groups, *spatial)
    axes = tuple(range(2, x.ndim))
    mean = mx.mean(x, axis=axes, keepdims=True)
    var = mx.var(x, axis=axes, keepdims=True)
    x = (x - mean) / mx.sqrt(var + eps)
    x = x.reshape(N, C, *spatial)

    if weight is not None:
        shape = [1, C] + [1] * len(spatial)
        x = x * weight.reshape(shape)
    if bias is not None:
        shape = [1, C] + [1] * len(spatial)
        x = x + bias.reshape(shape)
    return x


# =============================================================================
# Interpolation
# =============================================================================

def _interpolate_bilinear(input: mx.array, size: tuple) -> mx.array:
    """Bilinear interpolation using MLX ops (no numpy loop).

    Args:
        input: (B, C, H_in, W_in) tensor in NCHW format.
        size: Target (H_out, W_out).

    Returns:
        (B, C, H_out, W_out) interpolated tensor.
    """
    B, C, H_in, W_in = input.shape
    H_out, W_out = size

    # Convert to NHWC for spatial indexing
    x = mx.transpose(input, axes=(0, 2, 3, 1))  # (B, H_in, W_in, C)

    # Create coordinate grids (half-pixel offset for non-align-corners)
    h_scale = H_in / H_out
    w_scale = W_in / W_out

    iy = (mx.arange(H_out).astype(mx.float32) + 0.5) * h_scale - 0.5
    ix = (mx.arange(W_out).astype(mx.float32) + 0.5) * w_scale - 0.5

    iy0 = mx.floor(iy).astype(mx.int32)
    ix0 = mx.floor(ix).astype(mx.int32)
    iy1 = iy0 + 1
    ix1 = ix0 + 1

    wy1 = iy - iy0.astype(mx.float32)
    wx1 = ix - ix0.astype(mx.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    # Clamp indices
    iy0 = mx.clip(iy0, 0, H_in - 1)
    iy1 = mx.clip(iy1, 0, H_in - 1)
    ix0 = mx.clip(ix0, 0, W_in - 1)
    ix1 = mx.clip(ix1, 0, W_in - 1)

    # Gather the four corner values using sequential indexing
    # x[:, iy0, :, :] -> (B, H_out, W_in, C), then [:, :, ix0, :] -> (B, H_out, W_out, C)
    v00 = x[:, iy0][:, :, ix0]  # (B, H_out, W_out, C)
    v01 = x[:, iy1][:, :, ix0]
    v10 = x[:, iy0][:, :, ix1]
    v11 = x[:, iy1][:, :, ix1]

    # Bilinear weights: reshape for broadcasting
    # wy: (H_out,) -> (H_out, 1, 1) for broadcasting with (B, H_out, W_out, C)
    wy0 = wy0.reshape(-1, 1, 1)
    wy1 = wy1.reshape(-1, 1, 1)
    wx0 = wx0.reshape(1, -1, 1)
    wx1 = wx1.reshape(1, -1, 1)

    result = wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11)

    # Back to NCHW
    return mx.transpose(result, axes=(0, 3, 1, 2))


def interpolate(
    input: mx.array,
    size: tuple | int | None = None,
    scale_factor: float | tuple | None = None,
    mode: str = "nearest",
) -> mx.array:
    """F.interpolate — spatial upsampling/downsampling.

    Supports 'nearest' and 'bilinear' modes for (B, C, H, W) tensors.

    Args:
        input: (B, C, H, W) tensor.
        size: Target (H, W) size.
        scale_factor: Multiplicative scale factor.
        mode: Interpolation mode ('nearest' or 'bilinear').
    """
    if mode == "nearest":
        if scale_factor is not None:
            if isinstance(scale_factor, (int, float)):
                sf_h = sf_w = int(scale_factor)
            else:
                sf_h, sf_w = int(scale_factor[0]), int(scale_factor[1])
            # (B, C, H, W) → repeat along H and W
            x = mx.repeat(input, sf_h, axis=-2)
            x = mx.repeat(x, sf_w, axis=-1)
            return x
        elif size is not None:
            if isinstance(size, int):
                size = (size, size)
            h_in, w_in = input.shape[-2], input.shape[-1]
            h_out, w_out = size

            # Compute indices for nearest neighbor
            h_idx = mx.array([int(i * h_in / h_out) for i in range(h_out)])
            w_idx = mx.array([int(i * w_in / w_out) for i in range(w_out)])

            # Index along spatial dims
            x = mx.take(input, h_idx, axis=-2)
            x = mx.take(x, w_idx, axis=-1)
            return x
        else:
            raise ValueError("Either size or scale_factor must be specified")
    elif mode == "bilinear":
        if size is None and scale_factor is not None:
            if isinstance(scale_factor, (int, float)):
                sf_h = sf_w = scale_factor
            else:
                sf_h, sf_w = scale_factor
            h_out = int(input.shape[-2] * sf_h)
            w_out = int(input.shape[-1] * sf_w)
            size = (h_out, w_out)
        elif size is not None:
            if isinstance(size, int):
                size = (size, size)
        else:
            raise ValueError("Either size or scale_factor must be specified")

        return _interpolate_bilinear(input, size)
    else:
        raise NotImplementedError(f"Interpolation mode '{mode}' not implemented.")


# =============================================================================
# One-hot
# =============================================================================

def one_hot(tensor: mx.array, num_classes: int = -1) -> mx.array:
    """F.one_hot — 4 uses upstream.

    Args:
        tensor: Integer tensor of class indices.
        num_classes: Total number of classes. If -1, inferred from max value.

    Returns:
        One-hot encoded tensor with an extra trailing dimension.
    """
    if num_classes < 0:
        num_classes = int(mx.max(tensor).item()) + 1
    return mx.eye(num_classes, dtype=mx.float32)[tensor]


# =============================================================================
# Grid Sample
# =============================================================================

def grid_sample(
    input: mx.array,
    grid: mx.array,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> mx.array:
    """F.grid_sample — bilinear/nearest sampling from input at grid coordinates.

    Args:
        input: (B, C, H_in, W_in) input tensor.
        grid: (B, H_out, W_out, 2) sampling grid with values in [-1, 1].
        mode: 'bilinear' or 'nearest'.
        padding_mode: 'zeros' or 'border'.
        align_corners: if True, grid corners (-1, 1) correspond to corner pixels.

    Returns:
        (B, C, H_out, W_out) sampled tensor.
    """
    B, C, H_in, W_in = input.shape
    _, H_out, W_out, _ = grid.shape

    # Unnormalize grid from [-1, 1] to pixel coordinates
    if align_corners:
        ix = (grid[..., 0] + 1) / 2 * (W_in - 1)
        iy = (grid[..., 1] + 1) / 2 * (H_in - 1)
    else:
        ix = ((grid[..., 0] + 1) * W_in - 1) / 2
        iy = ((grid[..., 1] + 1) * H_in - 1) / 2

    if mode == "nearest":
        ix_nearest = mx.round(ix).astype(mx.int32)
        iy_nearest = mx.round(iy).astype(mx.int32)
        ix_nearest = mx.clip(ix_nearest, 0, W_in - 1)
        iy_nearest = mx.clip(iy_nearest, 0, H_in - 1)
        # Gather: for each (b, h, w), get input[b, :, iy, ix]
        # Use NHWC then transpose back
        x_nhwc = mx.transpose(input, axes=(0, 2, 3, 1))  # (B, H_in, W_in, C)
        # Index: need to gather per-batch
        # Flatten spatial for gathering
        result_parts = []
        for b_idx in range(B):
            gathered = x_nhwc[b_idx, iy_nearest[b_idx], ix_nearest[b_idx]]  # (H_out, W_out, C)
            result_parts.append(gathered)
        result = mx.stack(result_parts, axis=0)  # (B, H_out, W_out, C)
        return mx.transpose(result, axes=(0, 3, 1, 2))  # (B, C, H_out, W_out)

    elif mode == "bilinear":
        # Floor and ceil indices
        ix0 = mx.floor(ix).astype(mx.int32)
        iy0 = mx.floor(iy).astype(mx.int32)
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        # Interpolation weights
        wx1 = ix - ix0.astype(mx.float32)
        wy1 = iy - iy0.astype(mx.float32)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        def _clamp_and_gather(x_idx, y_idx):
            """Gather values from input at (y_idx, x_idx) with padding."""
            if padding_mode == "zeros":
                valid = (x_idx >= 0) & (x_idx < W_in) & (y_idx >= 0) & (y_idx < H_in)
                x_clamped = mx.clip(x_idx, 0, W_in - 1)
                y_clamped = mx.clip(y_idx, 0, H_in - 1)
            elif padding_mode == "border":
                x_clamped = mx.clip(x_idx, 0, W_in - 1)
                y_clamped = mx.clip(y_idx, 0, H_in - 1)
                valid = None
            else:
                x_clamped = mx.clip(x_idx, 0, W_in - 1)
                y_clamped = mx.clip(y_idx, 0, H_in - 1)
                valid = None

            # Gather per-batch using NHWC layout
            x_nhwc = mx.transpose(input, axes=(0, 2, 3, 1))  # (B, H_in, W_in, C)
            result_parts = []
            for b_idx in range(B):
                gathered = x_nhwc[b_idx, y_clamped[b_idx], x_clamped[b_idx]]  # (H_out, W_out, C)
                result_parts.append(gathered)
            gathered = mx.stack(result_parts, axis=0)  # (B, H_out, W_out, C)
            # Transpose to (B, C, H_out, W_out)
            gathered = mx.transpose(gathered, axes=(0, 3, 1, 2))

            if valid is not None:
                valid_expanded = mx.expand_dims(valid, axis=1)  # (B, 1, H_out, W_out)
                gathered = gathered * valid_expanded.astype(gathered.dtype)
            return gathered

        v00 = _clamp_and_gather(ix0, iy0)
        v01 = _clamp_and_gather(ix0, iy1)
        v10 = _clamp_and_gather(ix1, iy0)
        v11 = _clamp_and_gather(ix1, iy1)

        # Expand weights for channel dimension: (B, H_out, W_out) -> (B, 1, H_out, W_out)
        wx0 = mx.expand_dims(wx0, axis=1)
        wx1 = mx.expand_dims(wx1, axis=1)
        wy0 = mx.expand_dims(wy0, axis=1)
        wy1 = mx.expand_dims(wy1, axis=1)

        result = wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11)
        return result
    else:
        raise NotImplementedError(f"grid_sample mode '{mode}' not implemented.")
