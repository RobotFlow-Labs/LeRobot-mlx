"""LeRobot-MLX: LeRobot ported to Apple MLX for native Apple Silicon robotics."""

import importlib.metadata
import platform
import warnings

from lerobot_mlx._version import __version__

# Environment validation
if platform.machine() != "arm64":
    warnings.warn(
        "LeRobot-MLX is designed for Apple Silicon (arm64). "
        f"Running on {platform.machine()} may not work correctly.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    _mlx_version = importlib.metadata.version("mlx")
except importlib.metadata.PackageNotFoundError:
    raise ImportError(
        "MLX is required but not installed. Install with: pip install mlx>=0.31.0"
    ) from None

__all__ = ["__version__"]
