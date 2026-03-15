"""LeRobot-MLX Evaluation CLI."""
import argparse
import json
import logging
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from lerobot_mlx.datasets.lerobot_dataset import SyntheticDataset
from lerobot_mlx.datasets.dataloader import SimpleDataLoader

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LeRobot-MLX Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (.npz or .safetensors)")
    parser.add_argument("--policy-type", type=str, default="act", help="Policy type")
    parser.add_argument("--dataset-repo-id", type=str, default=None,
                       help="HuggingFace dataset repo ID. If None, uses synthetic data.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--obs-dim", type=int, default=14)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Entry point for lerobot-mlx-eval CLI."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    mx.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Create or load dataset
    if args.dataset_repo_id:
        from lerobot_mlx.datasets.lerobot_dataset import LeRobotDatasetMLX
        dataset = LeRobotDatasetMLX(args.dataset_repo_id)
        logger.info(f"Loaded dataset: {args.dataset_repo_id} ({len(dataset)} samples)")
    else:
        dataset = SyntheticDataset(
            num_samples=max(100, args.batch_size * args.num_episodes),
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            chunk_size=args.chunk_size,
        )
        logger.info(f"Using synthetic dataset ({len(dataset)} samples)")

    dataloader = SimpleDataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Create policy
    try:
        from lerobot_mlx.policies.factory import make_policy
        policy = make_policy(args.policy_type)
    except (ImportError, Exception):
        # Fallback: simple MLP for testing the eval loop
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

    # Load checkpoint if provided
    if args.checkpoint:
        weights = mx.load(args.checkpoint)
        if isinstance(weights, dict):
            policy.load_weights(list(weights.items()))
        else:
            policy.load_weights(weights)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    policy.eval()
    all_losses = []
    all_predictions = []
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_episodes:
            break

        output = policy(batch)
        mx.eval(output['loss'], output['action'])

        loss_val = output['loss'].item()
        all_losses.append(loss_val)
        all_predictions.append(np.array(output['action']))

        logger.info(f"Batch {batch_idx}: loss={loss_val:.6f}")

    elapsed = time.time() - start_time

    # Compute summary metrics
    metrics = {
        "mean_loss": float(np.mean(all_losses)) if all_losses else 0.0,
        "std_loss": float(np.std(all_losses)) if all_losses else 0.0,
        "min_loss": float(np.min(all_losses)) if all_losses else 0.0,
        "max_loss": float(np.max(all_losses)) if all_losses else 0.0,
        "num_batches": len(all_losses),
        "elapsed_seconds": elapsed,
    }

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation complete in {elapsed:.1f}s")
    logger.info(f"Mean loss: {metrics['mean_loss']:.6f} +/- {metrics['std_loss']:.6f}")
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
