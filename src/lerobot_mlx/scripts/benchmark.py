"""CLI entry point for LeRobot-MLX benchmarks.

Usage:
    lerobot-mlx-benchmark                    # Run all benchmarks
    lerobot-mlx-benchmark --policy ACT       # Single policy
    lerobot-mlx-benchmark --batch-size 16    # Custom batch size
    lerobot-mlx-benchmark --inference-only   # Only inference benchmarks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    """Entry point for lerobot-mlx-benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="lerobot-mlx-benchmark",
        description="LeRobot-MLX Performance Benchmark Suite",
    )
    parser.add_argument(
        "--policy",
        type=str,
        nargs="*",
        default=None,
        help="Policy names to benchmark (e.g. ACT Diffusion). Default: all",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="*",
        default=None,
        help="Inference batch sizes (e.g. 1 4 16). Default: [1, 4]",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=20,
        help="Number of training steps to measure. Default: 20",
    )
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=4,
        help="Training batch size. Default: 4",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only run inference benchmarks",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Only run training benchmarks",
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Only run memory benchmarks",
    )
    args = parser.parse_args()

    # Add project root to path so benchmarks module is importable
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if args.inference_only:
        from benchmarks.bench_inference import run_inference_benchmarks, print_inference_results
        results = run_inference_benchmarks(policies=args.policy, batch_sizes=args.batch_size)
        print(print_inference_results(results))
    elif args.training_only:
        from benchmarks.bench_training import run_training_benchmarks, print_training_results
        results = run_training_benchmarks(
            policies=args.policy,
            num_steps=args.training_steps,
            batch_size=args.training_batch_size,
        )
        print(print_training_results(results))
    elif args.memory_only:
        from benchmarks.bench_memory import run_memory_benchmarks, print_memory_results
        results = run_memory_benchmarks(policies=args.policy)
        print(print_memory_results(results))
    else:
        from benchmarks.run_all import main as run_all_main
        run_all_main(
            policies=args.policy,
            batch_sizes=args.batch_size,
            training_steps=args.training_steps,
            training_batch_size=args.training_batch_size,
        )


if __name__ == "__main__":
    main()
