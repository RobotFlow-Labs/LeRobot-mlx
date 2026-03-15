"""Pre/post processing pipeline using mx.array.

Handles normalization, delta action computation, and image preprocessing
for LeRobot policies on MLX.

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np


class ProcessorMLX:
    """Pre/post processing pipeline for LeRobot-MLX policies.

    Provides normalization, delta action computation, and image preprocessing.
    All operations are pure functions on mx.array tensors.

    Args:
        stats: Optional normalization statistics. Dict mapping keys to
            sub-dicts with ``'mean'`` and ``'std'`` arrays/lists.
            Example: ``{"observation.state": {"mean": [0.1, ...], "std": [0.5, ...]}}``
    """

    def __init__(self, stats: dict[str, dict[str, Any]] | None = None, eps: float = 1e-8):
        self.stats = stats or {}
        self.eps = eps
        # Cache converted stats as mx.array
        self._stats_cache: dict[str, dict[str, mx.array]] = {}

    def _get_stats(self, key: str) -> dict[str, mx.array]:
        """Get cached mx.array stats for a key."""
        if key not in self._stats_cache:
            if key not in self.stats:
                raise KeyError(
                    f"No normalization stats for key '{key}'. "
                    f"Available keys: {list(self.stats.keys())}"
                )
            self._stats_cache[key] = {
                "mean": mx.array(self.stats[key]["mean"], dtype=mx.float32),
                "std": mx.array(self.stats[key]["std"], dtype=mx.float32),
            }
        return self._stats_cache[key]

    def normalize(self, tensor: mx.array, key: str) -> mx.array:
        """Normalize a tensor using stored statistics.

        result = (tensor - mean) / (std + eps)

        Args:
            tensor: Input tensor to normalize.
            key: Stats key (e.g. "observation.state").

        Returns:
            Normalized tensor.
        """
        s = self._get_stats(key)
        return (tensor - s["mean"]) / (s["std"] + self.eps)

    def unnormalize(self, tensor: mx.array, key: str) -> mx.array:
        """Reverse normalization.

        result = tensor * (std + eps) + mean

        Uses the same epsilon as normalize() for symmetry, ensuring
        that unnormalize(normalize(x)) == x.

        Args:
            tensor: Normalized tensor.
            key: Stats key (e.g. "observation.state").

        Returns:
            Unnormalized tensor.
        """
        s = self._get_stats(key)
        return tensor * (s["std"] + self.eps) + s["mean"]

    @staticmethod
    def compute_delta(actions: mx.array) -> mx.array:
        """Compute delta actions: a[t] - a[t-1].

        Used for delta action spaces where the policy predicts changes
        rather than absolute positions.

        Args:
            actions: (T, D) or (B, T, D) action sequence.

        Returns:
            Delta actions with one fewer time step: (T-1, D) or (B, T-1, D).
        """
        return actions[..., 1:, :] - actions[..., :-1, :] if actions.ndim >= 2 else actions[1:] - actions[:-1]

    @staticmethod
    def undo_delta(
        delta_actions: mx.array,
        initial_action: mx.array,
    ) -> mx.array:
        """Recover absolute actions from delta actions via cumulative sum.

        Args:
            delta_actions: (T, D) delta action sequence.
            initial_action: (D,) initial absolute action.

        Returns:
            (T+1, D) absolute action sequence starting with initial_action.
        """
        combined = mx.concatenate(
            [mx.expand_dims(initial_action, axis=0), delta_actions], axis=0
        )
        return mx.cumsum(combined, axis=0)

    @staticmethod
    def process_images(
        images: np.ndarray,
        target_size: tuple[int, int] = (224, 224),
    ) -> mx.array:
        """Resize and normalize images for policy input.

        Converts uint8 NHWC images to float32 NCHW normalized to [0, 1].

        Args:
            images: (B, H, W, C) numpy uint8 images.
            target_size: (height, width) target resolution.

        Returns:
            (B, C, H, W) mx.array float32 in [0, 1].
        """
        try:
            import cv2
        except ImportError:
            # Fallback: simple nearest-neighbor resize via numpy
            return ProcessorMLX._process_images_numpy(images, target_size)

        processed = []
        for img in images:
            img_np = np.asarray(img)
            resized = cv2.resize(img_np, (target_size[1], target_size[0]))
            processed.append(resized)

        # Stack and normalize
        stacked = np.stack(processed).astype(np.float32) / 255.0
        result = mx.array(stacked)
        # NHWC -> NCHW
        return mx.transpose(result, axes=(0, 3, 1, 2))

    @staticmethod
    def _process_images_numpy(
        images: np.ndarray,
        target_size: tuple[int, int],
    ) -> mx.array:
        """Fallback image processing without OpenCV."""
        from PIL import Image

        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR  # Pillow < 10

        processed = []
        for img in images:
            pil_img = Image.fromarray(np.asarray(img))
            resized = pil_img.resize(
                (target_size[1], target_size[0]), resample
            )
            processed.append(np.array(resized))

        stacked = np.stack(processed).astype(np.float32) / 255.0
        result = mx.array(stacked)
        # NHWC -> NCHW
        return mx.transpose(result, axes=(0, 3, 1, 2))
