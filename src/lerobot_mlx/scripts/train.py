"""LeRobot-MLX Training CLI."""
import argparse
import json
import logging
import os
import sys
import time

import mlx.core as mx

from lerobot_mlx.training.trainer import Trainer, TrainingConfig
from lerobot_mlx.datasets.lerobot_dataset import SyntheticDataset
from lerobot_mlx.datasets.dataloader import SimpleDataLoader

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LeRobot-MLX Training")
    parser.add_argument("--policy-type", type=str, default="act", help="Policy type")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-steps", type=int, default=100000)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--dataset-repo-id", type=str, default=None,
                       help="HuggingFace dataset repo ID. If None, uses synthetic data.")
    parser.add_argument("--obs-dim", type=int, default=14)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Entry point for lerobot-mlx-train CLI."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    mx.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Config saved to {config_path}")

    # Create dataset
    if args.dataset_repo_id:
        from lerobot_mlx.datasets.lerobot_dataset import LeRobotDatasetMLX
        dataset = LeRobotDatasetMLX(args.dataset_repo_id)
        logger.info(f"Loaded dataset: {args.dataset_repo_id} ({len(dataset)} samples)")
    else:
        dataset = SyntheticDataset(
            num_samples=max(1000, args.batch_size * 100),
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            chunk_size=args.chunk_size,
        )
        logger.info(f"Using synthetic dataset ({len(dataset)} samples)")

    dataloader = SimpleDataLoader(dataset, batch_size=args.batch_size)

    # Create policy (for now, use a simple MLP until policy PRDs are done)
    try:
        from lerobot_mlx.policies.factory import make_policy
        policy = make_policy(args.policy_type)
    except (ImportError, Exception):
        # Fallback: simple MLP for testing the training loop
        import mlx.nn as nn

        class _SimpleMLP(nn.Module):
            def __init__(self, obs_dim, action_dim, chunk_size):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, 256),
                    nn.Linear(256, 256),
                    nn.Linear(256, action_dim * chunk_size),
                )
                self.action_dim = action_dim
                self.chunk_size = chunk_size

            def __call__(self, batch):
                x = batch['observation.state']
                pred = self.net(x).reshape(-1, self.chunk_size, self.action_dim)
                target = batch['action']
                loss = mx.mean((pred - target) ** 2)
                return {'action': pred, 'loss': loss}

            def compute_loss(self, batch):
                return self(batch)['loss']

        policy = _SimpleMLP(args.obs_dim, args.action_dim, args.chunk_size)
        logger.info("Using fallback MLP policy (policy factories not yet available)")

    # Create training config
    training_config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        max_grad_norm=args.max_grad_norm,
        lr_warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
    )

    # Train
    trainer = Trainer(policy, training_config)
    logger.info(f"Starting training for {args.training_steps} steps...")
    start_time = time.time()
    metrics = trainer.train(dataloader, num_steps=args.training_steps)
    elapsed = time.time() - start_time

    # Save final checkpoint
    trainer.save_checkpoint(args.training_steps)

    logger.info(f"Training complete in {elapsed:.1f}s ({args.training_steps / elapsed:.1f} steps/sec)")
    if metrics:
        logger.info(f"Final loss: {metrics[-1]['loss']:.6f}")


if __name__ == "__main__":
    main()
