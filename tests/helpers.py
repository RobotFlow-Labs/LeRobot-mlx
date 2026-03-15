"""Test helpers and utilities for LeRobot-MLX tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


# Tolerance constants for cross-framework comparison
ATOL_FP32 = 1e-5
ATOL_FP16 = 1e-2
ATOL_POLICY = 1e-3  # For end-to-end policy forward pass comparison
RTOL_DEFAULT = 1e-5


def check_all_close(
    actual: mx.array,
    expected: mx.array | np.ndarray,
    atol: float = ATOL_FP32,
    rtol: float = RTOL_DEFAULT,
    msg: str = "",
) -> None:
    """Assert that two arrays are element-wise close within tolerance.

    Works with both mx.array and numpy arrays. Converts to numpy for
    comparison using np.testing.assert_allclose.

    Args:
        actual: The array to check.
        expected: The reference array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        msg: Optional message to include on failure.

    Raises:
        AssertionError: If any element differs by more than tolerance.
    """
    actual_np = np.array(actual) if isinstance(actual, mx.array) else actual
    expected_np = np.array(expected) if isinstance(expected, mx.array) else expected

    np.testing.assert_allclose(
        actual_np,
        expected_np,
        atol=atol,
        rtol=rtol,
        err_msg=msg,
    )


def check_shape(actual: mx.array, expected_shape: tuple[int, ...], msg: str = "") -> None:
    """Assert that an array has the expected shape.

    Args:
        actual: The array to check.
        expected_shape: Expected shape tuple.
        msg: Optional message to include on failure.

    Raises:
        AssertionError: If shapes don't match.
    """
    assert actual.shape == expected_shape, (
        f"Shape mismatch: got {actual.shape}, expected {expected_shape}. {msg}"
    )


def random_input(*shape: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Create a random input array for testing.

    Args:
        *shape: Shape dimensions.
        dtype: Data type.

    Returns:
        Random array with values from standard normal distribution.
    """
    result = mx.random.normal(shape=shape)
    if dtype != mx.float32:
        result = result.astype(dtype)
    return result
