"""Tests for MLX training loop, EMA, synthetic dataset, and dataloader.

Tests cover:
- Trainer creation and configuration
- Single train step returning loss and lr
- Loss convergence over multiple steps
- Gradient clipping
- LR scheduler warmup and decay
- Checkpoint save/load roundtrip
- EMA update, apply, and restore
- Synthetic dataset shapes and types
- Dataloader batching and shuffling

Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
"""

import os
import sys
import tempfile

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from lerobot_mlx.training.trainer import Trainer, TrainingConfig, _clip_grads
from lerobot_mlx.training.ema import EMAModel
from lerobot_mlx.datasets.lerobot_dataset import SyntheticDataset
from lerobot_mlx.datasets.dataloader import SimpleDataLoader, collate_fn


# =============================================================================
# Dummy policy for testing
# =============================================================================

class _DummyPolicy(nn.Module):
    """Simple MLP policy for testing the training loop."""

    def __init__(self, obs_dim: int = 14, action_dim: int = 14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Linear(64, action_dim),
        )

    def __call__(self, batch):
        pred = self.net(batch["observation.state"])
        loss = mx.mean((pred - batch["action"]) ** 2)
        return {"action": pred, "loss": loss}

    forward = __call__

    def compute_loss(self, batch):
        return self(batch)["loss"]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dummy_policy():
    return _DummyPolicy(obs_dim=14, action_dim=14)


@pytest.fixture
def training_config():
    return TrainingConfig(
        lr=1e-3,
        training_steps=100,
        lr_warmup_steps=10,
        batch_size=8,
        log_interval=1000,  # suppress logging during tests
        save_interval=0,  # disable auto-save
    )


@pytest.fixture
def synthetic_dataset():
    return SyntheticDataset(
        num_samples=64,
        obs_dim=14,
        action_dim=14,
        chunk_size=1,
        image_shape=None,  # no images for training tests
        seed=42,
    )


@pytest.fixture
def dataloader(synthetic_dataset):
    return SimpleDataLoader(
        synthetic_dataset, batch_size=8, shuffle=True, seed=123
    )


@pytest.fixture
def single_batch(synthetic_dataset):
    """A single batch for quick tests."""
    items = [synthetic_dataset[i] for i in range(8)]
    return collate_fn(items)


# =============================================================================
# Trainer Tests
# =============================================================================

class TestTrainer:
    def test_trainer_creation(self, dummy_policy, training_config):
        """Trainer initializes without error."""
        trainer = Trainer(dummy_policy, training_config)
        assert trainer.policy is dummy_policy
        assert trainer.config is training_config
        assert trainer.optimizer is not None
        assert trainer.lr_scheduler is not None
        assert trainer.global_step == 0

    def test_trainer_default_config(self, dummy_policy):
        """Trainer works with default config."""
        trainer = Trainer(dummy_policy)
        assert trainer.config.lr == 1e-4
        assert trainer.config.training_steps == 100_000

    def test_train_step_returns_loss(self, dummy_policy, training_config, single_batch):
        """Single train step returns dict with loss and lr."""
        trainer = Trainer(dummy_policy, training_config)
        result = trainer.train_step(single_batch)

        assert isinstance(result, dict)
        assert "loss" in result
        assert "lr" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["lr"], float)
        assert result["loss"] > 0  # MSE loss should be positive
        assert result["lr"] > 0

    def test_train_step_increments_global_step(self, dummy_policy, training_config, single_batch):
        """Global step increments after each train step."""
        trainer = Trainer(dummy_policy, training_config)
        assert trainer.global_step == 0
        trainer.train_step(single_batch)
        assert trainer.global_step == 1
        trainer.train_step(single_batch)
        assert trainer.global_step == 2

    def test_loss_decreases(self, dummy_policy, training_config, single_batch):
        """Loss decreases over 50 steps on the same batch (overfitting test)."""
        config = TrainingConfig(
            lr=1e-3,
            training_steps=50,
            lr_warmup_steps=5,
            log_interval=1000,
            save_interval=0,
            max_grad_norm=None,  # no clipping for clean convergence
        )
        trainer = Trainer(dummy_policy, config)

        losses = []
        for _ in range(50):
            result = trainer.train_step(single_batch)
            losses.append(result["loss"])

        # First loss should be larger than last loss
        assert losses[0] > losses[-1], (
            f"Loss did not decrease: first={losses[0]:.6f}, last={losses[-1]:.6f}"
        )
        # Loss should decrease significantly
        assert losses[-1] < losses[0] * 0.5, (
            f"Loss did not decrease enough: {losses[0]:.6f} -> {losses[-1]:.6f}"
        )

    def test_gradient_clipping(self):
        """Gradient clipping bounds the gradient norm."""
        # Create large gradients
        grads = {"layers": [{"weight": mx.ones((10, 10)) * 100.0}]}
        max_norm = 1.0

        clipped = _clip_grads(grads, max_norm)
        mx.eval(clipped)

        # Compute norm of clipped grads
        from mlx.utils import tree_flatten
        flat = tree_flatten(clipped)
        total_norm = sum(mx.sum(g ** 2).item() for _, g in flat) ** 0.5

        assert total_norm <= max_norm + 1e-3, (
            f"Clipped norm {total_norm} exceeds max_norm {max_norm}"
        )

    def test_gradient_clipping_noop_small_grads(self):
        """Small gradients pass through unchanged."""
        grads = {"weight": mx.ones((3, 3)) * 0.01}
        max_norm = 10.0

        clipped = _clip_grads(grads, max_norm)
        mx.eval(clipped)

        # Should be identical (no clipping needed)
        np.testing.assert_allclose(
            np.array(clipped["weight"]),
            np.array(grads["weight"]),
            atol=1e-6,
        )

    def test_lr_scheduler_warmup(self, dummy_policy, single_batch):
        """LR increases during warmup period."""
        config = TrainingConfig(
            lr=1e-3,
            lr_warmup_steps=10,
            training_steps=100,
            log_interval=1000,
            save_interval=0,
        )
        trainer = Trainer(dummy_policy, config)

        lrs = []
        for _ in range(10):
            result = trainer.train_step(single_batch)
            lrs.append(result["lr"])

        # LR should be increasing during warmup
        assert lrs[-1] > lrs[0], (
            f"LR did not increase during warmup: {lrs[0]:.6e} -> {lrs[-1]:.6e}"
        )

    def test_lr_scheduler_decay(self, dummy_policy, single_batch):
        """LR decreases after warmup (cosine decay)."""
        config = TrainingConfig(
            lr=1e-3,
            lr_warmup_steps=5,
            training_steps=100,
            log_interval=1000,
            save_interval=0,
        )
        trainer = Trainer(dummy_policy, config)

        # Run through warmup
        for _ in range(5):
            trainer.train_step(single_batch)

        lr_after_warmup = trainer.train_step(single_batch)["lr"]

        # Run more steps into decay
        for _ in range(30):
            trainer.train_step(single_batch)

        lr_later = trainer.train_step(single_batch)["lr"]

        assert lr_later < lr_after_warmup, (
            f"LR did not decay: {lr_after_warmup:.6e} -> {lr_later:.6e}"
        )

    def test_checkpoint_save_load(self, dummy_policy, training_config, single_batch):
        """Save weights, load into new model, weights match."""
        trainer = Trainer(dummy_policy, training_config)

        # Train a few steps to change weights from init
        for _ in range(5):
            trainer.train_step(single_batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            ckpt_path = trainer.save_checkpoint(step=5, path=tmpdir)
            assert os.path.exists(ckpt_path)

            # Create fresh model and load
            new_policy = _DummyPolicy(obs_dim=14, action_dim=14)
            new_trainer = Trainer(new_policy, training_config)
            new_trainer.load_checkpoint(ckpt_path)

            # Compare weights
            from mlx.utils import tree_flatten
            orig_params = dict(tree_flatten(dummy_policy.parameters()))
            loaded_params = dict(tree_flatten(new_policy.parameters()))

            for key in orig_params:
                np.testing.assert_allclose(
                    np.array(orig_params[key]),
                    np.array(loaded_params[key]),
                    atol=1e-6,
                    err_msg=f"Weight mismatch at {key}",
                )

    def test_train_loop(self, dummy_policy, training_config, dataloader):
        """Full train loop runs without error."""
        trainer = Trainer(dummy_policy, training_config)
        metrics = trainer.train(dataloader, num_steps=5)

        assert len(metrics) == 5
        for m in metrics:
            assert "loss" in m
            assert "lr" in m
            assert "step" in m


# =============================================================================
# EMA Tests
# =============================================================================

class TestEMA:
    def test_ema_creation(self, dummy_policy):
        """EMA initializes with a copy of model parameters."""
        ema = EMAModel(dummy_policy, decay=0.999)
        assert ema.decay == 0.999
        assert len(ema.shadow) > 0

    def test_ema_update(self, dummy_policy, training_config, single_batch):
        """Shadow weights differ from model after training + EMA update."""
        trainer = Trainer(dummy_policy, training_config)
        ema = EMAModel(dummy_policy, decay=0.99)

        # Save initial shadow
        from mlx.utils import tree_flatten
        initial_shadow = {k: np.array(v) for k, v in ema.shadow.items()}

        # Train and update EMA
        for _ in range(5):
            trainer.train_step(single_batch)
            ema.update(dummy_policy)

        # Shadow should have changed
        changed = False
        for k in initial_shadow:
            if not np.allclose(initial_shadow[k], np.array(ema.shadow[k]), atol=1e-6):
                changed = True
                break
        assert changed, "EMA shadow weights did not change after update"

        # Shadow should differ from model params (due to smoothing)
        model_params = dict(tree_flatten(dummy_policy.parameters()))
        differs = False
        for k in ema.shadow:
            if k in model_params:
                if not np.allclose(
                    np.array(ema.shadow[k]),
                    np.array(model_params[k]),
                    atol=1e-6,
                ):
                    differs = True
                    break
        assert differs, "EMA shadow should differ from model params"

    def test_ema_apply_restore(self, dummy_policy):
        """Apply puts EMA weights in model, restore puts original back."""
        from mlx.utils import tree_flatten

        ema = EMAModel(dummy_policy, decay=0.5)

        # Manually set different shadow weights
        for k in ema.shadow:
            ema.shadow[k] = ema.shadow[k] + 1.0
        mx.eval(*ema.shadow.values())

        # Save original weights
        orig_params = {k: np.array(v) for k, v in tree_flatten(dummy_policy.parameters())}

        # Apply EMA weights
        ema.apply(dummy_policy)
        applied_params = {k: np.array(v) for k, v in tree_flatten(dummy_policy.parameters())}

        # Weights should be different from original
        for k in orig_params:
            assert not np.allclose(orig_params[k], applied_params[k], atol=0.1), (
                f"EMA apply did not change weights at {k}"
            )

        # Restore original weights
        ema.restore(dummy_policy)
        restored_params = {k: np.array(v) for k, v in tree_flatten(dummy_policy.parameters())}

        # Should match original
        for k in orig_params:
            np.testing.assert_allclose(
                orig_params[k], restored_params[k], atol=1e-6,
                err_msg=f"Restore failed at {k}",
            )

    def test_ema_restore_without_apply_raises(self, dummy_policy):
        """Restore without prior apply raises RuntimeError."""
        ema = EMAModel(dummy_policy)
        with pytest.raises(RuntimeError, match="restore.*apply"):
            ema.restore(dummy_policy)

    def test_ema_decay_range(self, dummy_policy):
        """Invalid decay raises ValueError."""
        with pytest.raises(ValueError):
            ema = EMAModel(dummy_policy)
            ema.set_decay(1.5)


# =============================================================================
# Synthetic Dataset Tests
# =============================================================================

class TestSyntheticDataset:
    def test_synthetic_dataset_length(self):
        ds = SyntheticDataset(num_samples=100)
        assert len(ds) == 100

    def test_synthetic_dataset_shapes(self):
        """Returned tensors have correct shapes."""
        ds = SyntheticDataset(
            num_samples=10,
            obs_dim=7,
            action_dim=5,
            chunk_size=1,
            image_shape=(3, 64, 64),
        )
        sample = ds[0]

        assert sample["observation.state"].shape == (7,)
        assert sample["action"].shape == (5,)
        assert sample["observation.images.top"].shape == (3, 64, 64)

    def test_synthetic_dataset_chunked_actions(self):
        """Chunked actions have (chunk_size, action_dim) shape."""
        ds = SyntheticDataset(
            num_samples=10, obs_dim=7, action_dim=5, chunk_size=10, image_shape=None
        )
        sample = ds[0]
        assert sample["action"].shape == (10, 5)

    def test_synthetic_dataset_types(self):
        """All values are mx.array."""
        ds = SyntheticDataset(num_samples=5, image_shape=None)
        sample = ds[0]
        for key, val in sample.items():
            assert isinstance(val, mx.array), f"Expected mx.array for {key}, got {type(val)}"

    def test_synthetic_dataset_no_images(self):
        """Dataset without images omits image key."""
        ds = SyntheticDataset(num_samples=5, image_shape=None)
        sample = ds[0]
        assert "observation.images.top" not in sample

    def test_synthetic_dataset_index_error(self):
        """Out of range index raises IndexError."""
        ds = SyntheticDataset(num_samples=5)
        with pytest.raises(IndexError):
            ds[10]

    def test_synthetic_dataset_reproducibility(self):
        """Same seed produces same data."""
        ds1 = SyntheticDataset(num_samples=5, seed=99, image_shape=None)
        ds2 = SyntheticDataset(num_samples=5, seed=99, image_shape=None)
        np.testing.assert_array_equal(
            np.array(ds1[0]["observation.state"]),
            np.array(ds2[0]["observation.state"]),
        )


# =============================================================================
# DataLoader Tests
# =============================================================================

class TestDataLoader:
    def test_dataloader_batching(self, synthetic_dataset):
        """DataLoader produces batched mx.arrays with correct shapes."""
        loader = SimpleDataLoader(
            synthetic_dataset, batch_size=8, shuffle=False, drop_last=True
        )
        batch = next(iter(loader))

        assert batch["observation.state"].shape == (8, 14)
        assert batch["action"].shape == (8, 14)

    def test_dataloader_length(self, synthetic_dataset):
        """DataLoader length reflects number of complete batches."""
        loader = SimpleDataLoader(
            synthetic_dataset, batch_size=8, shuffle=False, drop_last=True
        )
        assert len(loader) == 64 // 8  # 8 batches

    def test_dataloader_drop_last(self):
        """drop_last=True skips incomplete final batch."""
        ds = SyntheticDataset(num_samples=10, image_shape=None)
        loader = SimpleDataLoader(ds, batch_size=3, shuffle=False, drop_last=True)
        batches = list(loader)
        assert len(batches) == 3  # 10 // 3 = 3, remainder 1 dropped

    def test_dataloader_keep_last(self):
        """drop_last=False keeps incomplete final batch."""
        ds = SyntheticDataset(num_samples=10, image_shape=None)
        loader = SimpleDataLoader(ds, batch_size=3, shuffle=False, drop_last=False)
        batches = list(loader)
        assert len(batches) == 4  # ceil(10/3) = 4
        assert batches[-1]["observation.state"].shape[0] == 1  # 10 - 9 = 1

    def test_dataloader_shuffle(self):
        """Different seeds produce different batch orders."""
        ds = SyntheticDataset(num_samples=32, image_shape=None, seed=42)

        loader1 = SimpleDataLoader(ds, batch_size=8, shuffle=True, seed=1)
        loader2 = SimpleDataLoader(ds, batch_size=8, shuffle=True, seed=2)

        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # Very unlikely to get same batch with different seeds
        obs1 = np.array(batch1["observation.state"])
        obs2 = np.array(batch2["observation.state"])
        assert not np.allclose(obs1, obs2), "Shuffling did not change batch order"


# =============================================================================
# Collate Function Tests
# =============================================================================

class TestCollate:
    def test_collate_fn_basic(self):
        """Collate stacks samples correctly."""
        items = [
            {"obs": mx.array([1.0, 2.0]), "act": mx.array([3.0])},
            {"obs": mx.array([4.0, 5.0]), "act": mx.array([6.0])},
        ]
        batch = collate_fn(items)

        assert batch["obs"].shape == (2, 2)
        assert batch["act"].shape == (2, 1)
        np.testing.assert_allclose(np.array(batch["obs"][0]), [1.0, 2.0])
        np.testing.assert_allclose(np.array(batch["obs"][1]), [4.0, 5.0])

    def test_collate_fn_empty(self):
        """Collating empty list returns empty dict."""
        assert collate_fn([]) == {}


# =============================================================================
# CLI Script Tests
# =============================================================================

class TestCLIScripts:
    def test_train_script_help(self):
        """Train script --help exits with code 0."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "lerobot_mlx.scripts.train", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "LeRobot-MLX Training" in result.stdout

    def test_eval_script_help(self):
        """Eval script --help exits with code 0."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "lerobot_mlx.scripts.eval", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "LeRobot-MLX Evaluation" in result.stdout


# =============================================================================
# Efficient Gradient Clipping Tests
# =============================================================================

class TestClipGradsEfficiency:
    def test_clip_grads_efficient_no_item_loop(self):
        """Verify _clip_grads uses stacked norm computation, not per-tensor .item()."""
        import inspect
        source = inspect.getsource(_clip_grads)
        # Should NOT have .item() inside a generator/comprehension that sums
        assert "sum(mx.sum" not in source and "sum(g" not in source.replace("mx.sum", ""), (
            "_clip_grads still uses per-tensor .item() loop"
        )
        # Should use mx.stack for accumulation
        assert "mx.stack" in source, "_clip_grads should use mx.stack for on-device norm"

    def test_clip_grads_correctness(self):
        """Clipped grads have norm <= max_norm."""
        grads = {"a": mx.ones((50, 50)) * 10.0, "b": mx.ones((30,)) * 20.0}
        max_norm = 1.0
        clipped = _clip_grads(grads, max_norm)
        mx.eval(clipped)

        flat = tree_flatten(clipped)
        total_norm = sum(mx.sum(g ** 2).item() for _, g in flat) ** 0.5
        assert total_norm <= max_norm + 1e-3

    def test_clip_grads_noop_when_small(self):
        """Small grads pass through unchanged."""
        grads = {"w": mx.ones((3, 3)) * 0.001}
        clipped = _clip_grads(grads, max_norm=100.0)
        mx.eval(clipped)
        np.testing.assert_allclose(np.array(clipped["w"]), np.array(grads["w"]), atol=1e-8)
