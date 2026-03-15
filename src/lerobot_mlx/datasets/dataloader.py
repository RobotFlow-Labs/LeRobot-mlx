"""Simple batching dataloader for MLX.

MLX's unified memory and lazy evaluation make a simple Python iterator
sufficient for training. No multiprocessing or prefetching needed.

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

from __future__ import annotations

import random
from typing import Any

import mlx.core as mx


def collate_fn(items: list[dict[str, mx.array]]) -> dict[str, mx.array]:
    """Stack a list of sample dicts into a batched dict of mx.arrays.

    Each sample is a dict mapping keys to mx.array tensors. The collated
    output stacks all tensors along a new batch dimension (axis=0).

    Args:
        items: List of sample dicts from dataset.__getitem__.

    Returns:
        Batched dict where each value has an extra leading batch dim.

    Example::

        items = [
            {"obs": mx.array([1, 2]), "action": mx.array([3])},
            {"obs": mx.array([4, 5]), "action": mx.array([6])},
        ]
        batch = collate_fn(items)
        # batch["obs"].shape == (2, 2)
        # batch["action"].shape == (2, 1)
    """
    if not items:
        return {}

    batch: dict[str, mx.array] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        batch[key] = mx.stack(values)
    return batch


class SimpleDataLoader:
    """Minimal dataloader that batches dataset items into mx.arrays.

    Iterates over the dataset, groups items into batches, and collates
    them using mx.stack. Supports shuffling and dropping the last
    incomplete batch.

    Args:
        dataset: Any object with ``__getitem__`` and ``__len__``.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle indices each epoch.
        drop_last: Whether to drop the last incomplete batch.
        seed: Random seed for shuffling (None = non-deterministic).

    Usage::

        dataset = SyntheticDataset(num_samples=100)
        loader = SimpleDataLoader(dataset, batch_size=16, shuffle=True)
        for batch in loader:
            loss = model(batch)
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = random.Random(seed)

    def __iter__(self):
        """Yield batched dicts of mx.array."""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]

            # Drop incomplete last batch if requested
            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            items = [self.dataset[i] for i in batch_indices]
            yield collate_fn(items)

    def __len__(self) -> int:
        """Number of complete batches per epoch."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
