"""Dataset loading for LeRobot-MLX.

Provides SyntheticDataset for testing training loops without real data,
and LeRobotDatasetMLX for loading real datasets from HuggingFace Hub.

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np


class SyntheticDataset:
    """Random synthetic data for testing training loops without real datasets.

    Generates data matching the LeRobot observation/action format:
    - ``observation.state``: (obs_dim,) proprioceptive state
    - ``observation.images.top``: (C, H, W) camera image
    - ``action``: (action_dim,) or (chunk_size, action_dim) action targets

    Args:
        num_samples: Number of samples in the dataset.
        obs_dim: Dimensionality of proprioceptive observations.
        action_dim: Dimensionality of actions.
        chunk_size: Number of action steps per sample (for action chunking).
            If 1, action shape is (action_dim,). If >1, shape is (chunk_size, action_dim).
        image_shape: (C, H, W) shape for synthetic camera images.
            Set to None to omit images.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        obs_dim: int = 14,
        action_dim: int = 14,
        chunk_size: int = 1,
        image_shape: tuple[int, ...] | None = (3, 480, 640),
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.image_shape = image_shape

        # Pre-generate all data as numpy for determinism
        rng = np.random.RandomState(seed)
        self._obs_state = rng.randn(num_samples, obs_dim).astype(np.float32)

        if chunk_size > 1:
            self._actions = rng.randn(num_samples, chunk_size, action_dim).astype(
                np.float32
            )
        else:
            self._actions = rng.randn(num_samples, action_dim).astype(np.float32)

        if image_shape is not None:
            # Small random images (kept as float32 for simplicity)
            # Use a smaller internal representation to save memory
            self._images = rng.rand(num_samples, *image_shape).astype(np.float32)
        else:
            self._images = None

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Return a single sample as a dict of mx.array.

        Returns:
            Dict with keys:
            - ``observation.state``: (obs_dim,) float32
            - ``observation.images.top``: (C, H, W) float32 (if images enabled)
            - ``action``: (action_dim,) or (chunk_size, action_dim) float32
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")

        sample = {
            "observation.state": mx.array(self._obs_state[idx]),
            "action": mx.array(self._actions[idx]),
        }

        if self._images is not None:
            sample["observation.images.top"] = mx.array(self._images[idx])

        return sample

    def __len__(self) -> int:
        return self.num_samples


class LeRobotDatasetMLX:
    """Loads a real LeRobot dataset from HuggingFace Hub, returns mx.array.

    This is a minimal wrapper that downloads Parquet files from the Hub
    and converts them to mx.array on access. For Phase 1, use SyntheticDataset
    instead (no Hub dependency needed).

    Args:
        repo_id: HuggingFace Hub repository ID (e.g. "lerobot/pusht").
        split: Dataset split ("train" or "test").
        episodes: Optional list of episode indices to load.
        **kwargs: Additional arguments passed to dataset loading.
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        episodes: list[int] | None = None,
        **kwargs: Any,
    ):
        self.repo_id = repo_id
        self.split = split
        self._data: list[dict[str, np.ndarray]] | None = None
        self._num_frames = 0

        self._load_data(episodes=episodes, **kwargs)

    def _load_data(self, episodes: list[int] | None = None, **kwargs: Any) -> None:
        """Download and load dataset from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise ImportError(
                "huggingface-hub is required for real dataset loading. "
                "Install with: pip install huggingface-hub"
            ) from None

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for dataset loading. "
                "Install with: pip install pandas"
            ) from None

        # List parquet files in the repo
        all_files = list_repo_files(self.repo_id, repo_type="dataset")
        parquet_files = [
            f for f in all_files
            if f.endswith(".parquet") and self.split in f
        ]

        if not parquet_files:
            # Try without split filter
            parquet_files = [f for f in all_files if f.endswith(".parquet")]

        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {self.repo_id} for split '{self.split}'"
            )

        # Download and read parquet files
        frames = []
        for pf in parquet_files:
            local_path = hf_hub_download(
                self.repo_id, pf, repo_type="dataset"
            )
            df = pd.read_parquet(local_path)
            frames.append(df)

        full_df = pd.concat(frames, ignore_index=True)

        # Filter episodes if specified
        if episodes is not None and "episode_index" in full_df.columns:
            full_df = full_df[full_df["episode_index"].isin(episodes)]

        # Convert to columnar numpy arrays (vectorized, avoids slow iterrows)
        self._column_data: dict[str, np.ndarray] = {}
        self._numeric_columns: list[str] = []

        for col in full_df.columns:
            try:
                # Try vectorized conversion for the whole column
                col_values = full_df[col].values
                if isinstance(col_values[0], (list, np.ndarray)):
                    # Column contains list/array values - stack them
                    self._column_data[col] = np.stack(
                        [np.asarray(v, dtype=np.float32) for v in col_values]
                    )
                    self._numeric_columns.append(col)
                elif np.issubdtype(type(col_values[0]), np.integer) or \
                     np.issubdtype(type(col_values[0]), np.floating) or \
                     isinstance(col_values[0], (int, float)):
                    self._column_data[col] = col_values.astype(np.float32)
                    self._numeric_columns.append(col)
                # else: skip non-numeric columns (e.g. file paths)
            except (ValueError, TypeError):
                # Skip columns that can't be converted
                continue

        self._num_frames = len(full_df)
        self._data = None  # Not used in vectorized path

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Return a single sample as a dict of mx.array.

        All numeric columns are converted to mx.array float32.
        """
        if not self._column_data:
            raise RuntimeError("Dataset not loaded. Call _load_data() first.")
        if idx < 0 or idx >= self._num_frames:
            raise IndexError(f"Index {idx} out of range [0, {self._num_frames})")

        return {col: mx.array(self._column_data[col][idx]) for col in self._numeric_columns}

    def __len__(self) -> int:
        return self._num_frames
