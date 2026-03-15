"""Tests for dataset loading and processor pipeline.

Tests cover:
- SyntheticDataset shapes and types
- Collate function correctness
- ProcessorMLX normalize/unnormalize roundtrip
- ProcessorMLX delta/undo_delta roundtrip
- ProcessorMLX image processing shapes and dtype
- SimpleDataLoader integration

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

import mlx.core as mx
import numpy as np
import pytest

from lerobot_mlx.datasets.lerobot_dataset import SyntheticDataset
from lerobot_mlx.datasets.dataloader import SimpleDataLoader, collate_fn
from lerobot_mlx.processor.processor_mlx import ProcessorMLX


# =============================================================================
# SyntheticDataset Tests
# =============================================================================

class TestSyntheticDatasetShapes:
    def test_default_shapes(self):
        """Default dataset returns correct shapes."""
        ds = SyntheticDataset(num_samples=5)
        sample = ds[0]
        assert sample["observation.state"].shape == (14,)
        assert sample["action"].shape == (14,)
        assert sample["observation.images.top"].shape == (3, 480, 640)

    def test_custom_shapes(self):
        """Custom dims are respected."""
        ds = SyntheticDataset(
            num_samples=5,
            obs_dim=7,
            action_dim=3,
            chunk_size=20,
            image_shape=(1, 32, 32),
        )
        sample = ds[0]
        assert sample["observation.state"].shape == (7,)
        assert sample["action"].shape == (20, 3)
        assert sample["observation.images.top"].shape == (1, 32, 32)

    def test_all_types_are_mx_array(self):
        """Every value in the sample dict is an mx.array."""
        ds = SyntheticDataset(num_samples=3, image_shape=(3, 8, 8))
        sample = ds[0]
        for key, val in sample.items():
            assert isinstance(val, mx.array), f"{key} is {type(val)}, expected mx.array"

    def test_dtypes_are_float32(self):
        """All tensors are float32."""
        ds = SyntheticDataset(num_samples=3, image_shape=(3, 8, 8))
        sample = ds[0]
        for key, val in sample.items():
            assert val.dtype == mx.float32, f"{key} dtype is {val.dtype}, expected float32"

    def test_different_samples_differ(self):
        """Different indices return different data."""
        ds = SyntheticDataset(num_samples=10, image_shape=None)
        s0 = np.array(ds[0]["observation.state"])
        s1 = np.array(ds[1]["observation.state"])
        assert not np.allclose(s0, s1), "Samples 0 and 1 should differ"


# =============================================================================
# Collate Tests
# =============================================================================

class TestCollateFunction:
    def test_collate_fn_stacks_correctly(self):
        """Collated batch has correct shapes."""
        items = [
            {"a": mx.zeros((3,)), "b": mx.ones((2, 4))},
            {"a": mx.zeros((3,)), "b": mx.ones((2, 4))},
            {"a": mx.zeros((3,)), "b": mx.ones((2, 4))},
        ]
        batch = collate_fn(items)
        assert batch["a"].shape == (3, 3)
        assert batch["b"].shape == (3, 2, 4)

    def test_collate_preserves_values(self):
        """Values are correctly preserved in batch."""
        items = [
            {"x": mx.array([1.0, 2.0])},
            {"x": mx.array([3.0, 4.0])},
        ]
        batch = collate_fn(items)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(np.array(batch["x"]), expected)

    def test_collate_single_item(self):
        """Collating one item adds batch dimension."""
        items = [{"v": mx.array([5.0])}]
        batch = collate_fn(items)
        assert batch["v"].shape == (1, 1)


# =============================================================================
# ProcessorMLX Tests
# =============================================================================

class TestProcessorNormalization:
    @pytest.fixture
    def processor(self):
        stats = {
            "obs": {"mean": [1.0, 2.0, 3.0], "std": [0.5, 1.0, 2.0]},
            "action": {"mean": [0.0], "std": [1.0]},
        }
        return ProcessorMLX(stats=stats)

    def test_normalize(self, processor):
        """Normalization formula is correct."""
        x = mx.array([1.0, 2.0, 3.0])
        result = processor.normalize(x, "obs")
        expected = np.array([0.0, 0.0, 0.0])  # (x - mean) / std
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6)

    def test_unnormalize(self, processor):
        """Unnormalization formula is correct."""
        x = mx.array([0.0, 0.0, 0.0])
        result = processor.unnormalize(x, "obs")
        expected = np.array([1.0, 2.0, 3.0])  # x * std + mean
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6)

    def test_normalize_unnormalize_roundtrip(self, processor):
        """normalize(unnormalize(x)) ~ x."""
        x = mx.array([1.5, 2.5, 3.5])
        normalized = processor.normalize(x, "obs")
        recovered = processor.unnormalize(normalized, "obs")
        np.testing.assert_allclose(np.array(recovered), np.array(x), atol=1e-5)

    def test_missing_key_raises(self, processor):
        """Accessing missing stats key raises KeyError."""
        x = mx.array([1.0])
        with pytest.raises(KeyError, match="nonexistent"):
            processor.normalize(x, "nonexistent")


class TestProcessorDelta:
    def test_compute_delta(self):
        """Delta computation: a[t] - a[t-1]."""
        actions = mx.array([[1.0], [3.0], [6.0]])  # (3, 1)
        delta = ProcessorMLX.compute_delta(actions)
        expected = np.array([[2.0], [3.0]])  # (2, 1)
        np.testing.assert_allclose(np.array(delta), expected, atol=1e-6)

    def test_undo_delta(self):
        """Undo delta recovers original actions."""
        actions = mx.array([[1.0], [3.0], [6.0]])
        delta = ProcessorMLX.compute_delta(actions)
        initial = mx.array([1.0])
        recovered = ProcessorMLX.undo_delta(delta, initial)
        np.testing.assert_allclose(
            np.array(recovered), np.array(actions), atol=1e-5
        )

    def test_delta_undo_delta_roundtrip(self):
        """compute_delta + undo_delta is identity (given initial)."""
        rng = np.random.RandomState(42)
        actions = mx.array(rng.randn(10, 3).astype(np.float32))
        delta = ProcessorMLX.compute_delta(actions)
        initial = actions[0]
        recovered = ProcessorMLX.undo_delta(delta, initial)
        np.testing.assert_allclose(
            np.array(recovered), np.array(actions), atol=1e-4
        )


class TestProcessorImages:
    def test_process_images_shape(self):
        """Image processing produces correct output shape."""
        images = np.random.randint(0, 255, (4, 100, 150, 3), dtype=np.uint8)
        result = ProcessorMLX.process_images(images, target_size=(64, 64))

        assert result.shape == (4, 3, 64, 64)  # NCHW

    def test_process_images_dtype(self):
        """Output is float32."""
        images = np.random.randint(0, 255, (2, 50, 50, 3), dtype=np.uint8)
        result = ProcessorMLX.process_images(images, target_size=(32, 32))
        assert result.dtype == mx.float32

    def test_process_images_range(self):
        """Output values are in [0, 1]."""
        images = np.random.randint(0, 255, (2, 50, 50, 3), dtype=np.uint8)
        result = ProcessorMLX.process_images(images, target_size=(32, 32))
        result_np = np.array(result)
        assert result_np.min() >= 0.0
        assert result_np.max() <= 1.0

    def test_process_images_single(self):
        """Processing a single image works."""
        images = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)
        result = ProcessorMLX.process_images(images, target_size=(224, 224))
        assert result.shape == (1, 3, 224, 224)


# =============================================================================
# Integration: Dataset + DataLoader + Processor
# =============================================================================

class TestIntegration:
    def test_dataset_to_dataloader(self):
        """Full pipeline: dataset -> dataloader -> batch."""
        ds = SyntheticDataset(num_samples=32, obs_dim=7, action_dim=3, image_shape=None)
        loader = SimpleDataLoader(ds, batch_size=8, shuffle=False, drop_last=True)

        batch = next(iter(loader))
        assert batch["observation.state"].shape == (8, 7)
        assert batch["action"].shape == (8, 3)

    def test_processor_on_batch(self):
        """Processor normalizes a batched tensor."""
        stats = {"observation.state": {"mean": [0.0] * 7, "std": [1.0] * 7}}
        processor = ProcessorMLX(stats=stats)

        ds = SyntheticDataset(num_samples=16, obs_dim=7, action_dim=3, image_shape=None)
        loader = SimpleDataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
        batch = next(iter(loader))

        normalized = processor.normalize(batch["observation.state"], "observation.state")
        assert normalized.shape == (4, 7)

    def test_processor_empty_stats(self):
        """Processor with no stats works for non-stats operations."""
        processor = ProcessorMLX()
        actions = mx.array([[1.0, 2.0], [3.0, 4.0], [6.0, 7.0]])
        delta = processor.compute_delta(actions)
        assert delta.shape == (2, 2)


# =============================================================================
# Normalize/Unnormalize Epsilon Consistency
# =============================================================================

class TestProcessorEpsilonConsistency:
    def test_processor_normalize_epsilon_consistency(self):
        """normalize and unnormalize use the same epsilon for perfect roundtrip."""
        # Use a very small std to exercise the epsilon path
        stats = {"x": {"mean": [0.0, 0.0], "std": [1e-12, 1e-12]}}
        processor = ProcessorMLX(stats=stats)

        original = mx.array([5.0, 10.0])
        normalized = processor.normalize(original, "x")
        recovered = processor.unnormalize(normalized, "x")
        np.testing.assert_allclose(
            np.array(recovered), np.array(original), atol=1e-4,
            err_msg="Roundtrip failed: normalize/unnormalize epsilon asymmetry"
        )

    def test_processor_custom_eps(self):
        """Custom epsilon is respected."""
        stats = {"x": {"mean": [0.0], "std": [0.0]}}
        processor = ProcessorMLX(stats=stats, eps=1.0)
        x = mx.array([5.0])
        # normalize: (5 - 0) / (0 + 1) = 5
        result = processor.normalize(x, "x")
        np.testing.assert_allclose(np.array(result), [5.0], atol=1e-6)
        # unnormalize: 5 * (0 + 1) + 0 = 5
        recovered = processor.unnormalize(result, "x")
        np.testing.assert_allclose(np.array(recovered), [5.0], atol=1e-6)
