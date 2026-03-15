"""Image preprocessing utilities for VLA policies.

Handles PIL -> numpy -> mx.array conversion, resize, normalize, pad,
multi-view camera support, and NCHW/NHWC layout management.

Usage:
    from lerobot_mlx.model.image_processor import ImageProcessor

    processor = ImageProcessor(size=(224, 224), normalize=True)
    pixel_values = processor(pil_images)  # -> mx.array (B, C, H, W)
"""

from typing import Optional, Sequence, Union

import mlx.core as mx
import numpy as np


# ImageNet defaults
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageProcessor:
    """Simple image preprocessor for VLA policies.

    Converts images from various sources (PIL, numpy, mx.array) to a
    normalized (B, C, H, W) mx.array ready for model consumption.

    Args:
        size: Target (height, width) for resizing. None means no resize.
        mean: Per-channel mean for normalization. None means no normalization.
        std: Per-channel std for normalization. None means no normalization.
        channel_first: If True, output is NCHW. If False, output is NHWC.
        rescale: If True, rescale uint8 [0,255] to float [0,1] before normalizing.
    """

    def __init__(
        self,
        size: Optional[tuple[int, int]] = None,
        mean: Optional[tuple[float, ...]] = IMAGENET_MEAN,
        std: Optional[tuple[float, ...]] = IMAGENET_STD,
        channel_first: bool = True,
        rescale: bool = True,
    ):
        self.size = size
        self.mean = mean
        self.std = std
        self.channel_first = channel_first
        self.rescale = rescale

    def __call__(
        self,
        images: Union[
            "PILImage",
            np.ndarray,
            mx.array,
            list,
        ],
    ) -> mx.array:
        """Process one or more images into a batch tensor.

        Args:
            images: A single image or list of images. Each image can be:
                - PIL Image
                - numpy array (H, W, C) or (H, W) or (C, H, W)
                - mx.array with same layouts
                - A list of any of the above

        Returns:
            mx.array of shape (B, C, H, W) if channel_first else (B, H, W, C)
        """
        arrays = self._to_numpy_list(images)
        processed = []
        for arr in arrays:
            arr = self._ensure_hwc(arr)
            if self.size is not None:
                arr = self._resize(arr, self.size)
            if self.rescale and arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            elif arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if self.mean is not None and self.std is not None:
                arr = self._normalize(arr, self.mean, self.std)
            if self.channel_first:
                arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
            processed.append(arr)

        batch = np.stack(processed, axis=0)
        return mx.array(batch)

    # ------------------------------------------------------------------
    # Multi-view support
    # ------------------------------------------------------------------

    def process_multi_view(
        self,
        camera_images: dict[str, Union["PILImage", np.ndarray, mx.array, list]],
    ) -> dict[str, mx.array]:
        """Process images from multiple camera views.

        Args:
            camera_images: Dict mapping camera name to image(s).
                Example: {"front": pil_img, "wrist": pil_img}

        Returns:
            Dict mapping camera name to processed (B, C, H, W) tensors.
        """
        return {name: self(imgs) for name, imgs in camera_images.items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy_list(images) -> list[np.ndarray]:
        """Convert various image inputs to a list of numpy arrays."""
        try:
            from PIL import Image as PILImage

            is_pil = isinstance(images, PILImage.Image)
        except ImportError:
            is_pil = False

        if is_pil:
            return [np.array(images)]

        if isinstance(images, mx.array):
            arr = np.array(images)
            if arr.ndim == 4:
                return [arr[i] for i in range(arr.shape[0])]
            return [arr]

        if isinstance(images, np.ndarray):
            if images.ndim == 4:
                return [images[i] for i in range(images.shape[0])]
            return [images]

        if isinstance(images, (list, tuple)):
            result = []
            for img in images:
                try:
                    from PIL import Image as PILImage

                    if isinstance(img, PILImage.Image):
                        result.append(np.array(img))
                        continue
                except ImportError:
                    pass
                if isinstance(img, mx.array):
                    result.append(np.array(img))
                elif isinstance(img, np.ndarray):
                    result.append(img)
                else:
                    result.append(np.asarray(img, dtype=np.float32))
            return result

        return [np.asarray(images, dtype=np.float32)]

    @staticmethod
    def _ensure_hwc(arr: np.ndarray) -> np.ndarray:
        """Ensure array is in (H, W, C) format."""
        if arr.ndim == 2:
            # Grayscale (H, W) -> (H, W, 1)
            return arr[:, :, np.newaxis]
        if arr.ndim == 3:
            # Check if CHW: C is typically 1, 3, or 4 and smaller than H and W
            if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                return np.transpose(arr, (1, 2, 0))
            return arr
        return arr

    @staticmethod
    def _resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Resize image array to (height, width) using bilinear interpolation.

        Uses numpy-based nearest-neighbor resize to avoid extra dependencies.
        For production use, PIL or cv2 would be preferred.
        """
        target_h, target_w = size
        h, w = arr.shape[:2]
        if h == target_h and w == target_w:
            return arr

        # Simple bilinear-ish resize via numpy
        # Use nearest neighbor for simplicity (avoids scipy/cv2 dependency)
        row_indices = np.linspace(0, h - 1, target_h).astype(int)
        col_indices = np.linspace(0, w - 1, target_w).astype(int)
        return arr[np.ix_(row_indices, col_indices)]

    @staticmethod
    def _normalize(
        arr: np.ndarray,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> np.ndarray:
        """Normalize with per-channel mean and std."""
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)

        # Handle channel count mismatch (e.g. grayscale with RGB stats)
        channels = arr.shape[-1] if arr.ndim == 3 else 1
        if len(mean_arr) > channels:
            mean_arr = mean_arr[:channels]
            std_arr = std_arr[:channels]

        return (arr - mean_arr) / std_arr

    @staticmethod
    def pad_to_square(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """Pad an (H, W, C) image to a square with fill_value."""
        h, w = arr.shape[:2]
        if h == w:
            return arr
        size = max(h, w)
        padded = np.full(
            (size, size) + arr.shape[2:], fill_value, dtype=arr.dtype
        )
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        padded[y_offset : y_offset + h, x_offset : x_offset + w] = arr
        return padded
