"""Run all benchmarks and produce a summary report.

Executes inference, training, and memory benchmarks for all LeRobot-MLX
policies, then outputs formatted tables and saves results as JSON and
Markdown.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so ``benchmarks`` is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.bench_inference import run_inference_benchmarks, print_inference_results
from benchmarks.bench_training import run_training_benchmarks, print_training_results
from benchmarks.bench_memory import run_memory_benchmarks, print_memory_results


def _system_info() -> dict:
    """Gather system information."""
    import mlx.core as mx
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "timestamp": datetime.now().isoformat(),
    }


def main(
    policies: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    training_steps: int = 20,
    training_batch_size: int = 4,
):
    """Run the full benchmark suite."""
    info = _system_info()
    chip = info["processor"]

    banner = (
        f"\n{'=' * 64}\n"
        f"  LeRobot-MLX Benchmark Suite\n"
        f"  {chip} | Python {info['python']} | MLX {info['mlx_version']}\n"
        f"  {info['timestamp']}\n"
        f"{'=' * 64}\n"
    )
    print(banner)

    # --- Inference -----------------------------------------------------------
    print("\n[1/3] Inference Latency Benchmarks")
    print("-" * 40)
    t0 = time.perf_counter()
    inf_results = run_inference_benchmarks(policies=policies, batch_sizes=batch_sizes)
    inf_elapsed = time.perf_counter() - t0
    inf_table = print_inference_results(inf_results)
    print(f"\n{inf_table}")
    print(f"\n  (completed in {inf_elapsed:.1f}s)")

    # --- Training ------------------------------------------------------------
    print("\n[2/3] Training Throughput Benchmarks")
    print("-" * 40)
    t0 = time.perf_counter()
    train_results = run_training_benchmarks(
        policies=policies,
        num_steps=training_steps,
        batch_size=training_batch_size,
    )
    train_elapsed = time.perf_counter() - t0
    train_table = print_training_results(train_results)
    print(f"\n{train_table}")
    print(f"\n  (completed in {train_elapsed:.1f}s)")

    # --- Memory --------------------------------------------------------------
    print("\n[3/3] Memory Profiling Benchmarks")
    print("-" * 40)
    t0 = time.perf_counter()
    mem_results = run_memory_benchmarks(policies=policies)
    mem_elapsed = time.perf_counter() - t0
    mem_table = print_memory_results(mem_results)
    print(f"\n{mem_table}")
    print(f"\n  (completed in {mem_elapsed:.1f}s)")

    # --- Save results --------------------------------------------------------
    results_dir = Path(__file__).resolve().parent
    all_data = {
        "system": info,
        "inference": inf_results,
        "training": train_results,
        "memory": mem_results,
    }

    json_path = results_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    md_path = results_dir / "RESULTS.md"
    with open(md_path, "w") as f:
        f.write(f"# LeRobot-MLX Benchmark Results\n\n")
        f.write(f"**System**: {chip} | Python {info['python']} | MLX {info['mlx_version']}\n")
        f.write(f"**Date**: {info['timestamp']}\n\n")

        f.write("## Inference Latency\n\n```\n")
        f.write(inf_table)
        f.write("\n```\n\n")

        f.write("## Training Throughput\n\n```\n")
        f.write(train_table)
        f.write("\n```\n\n")

        f.write("## Memory Profile\n\n```\n")
        f.write(mem_table)
        f.write("\n```\n")
    print(f"Markdown saved to {md_path}")

    print(f"\nDone. Total benchmark time: {inf_elapsed + train_elapsed + mem_elapsed:.1f}s")

    return all_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LeRobot-MLX Benchmark Suite")
    parser.add_argument("--policy", type=str, nargs="*", default=None, help="Policy names to benchmark")
    parser.add_argument("--batch-size", type=int, nargs="*", default=None, help="Batch sizes for inference")
    parser.add_argument("--training-steps", type=int, default=20, help="Training steps to measure")
    parser.add_argument("--training-batch-size", type=int, default=4, help="Training batch size")
    args = parser.parse_args()

    main(
        policies=args.policy,
        batch_sizes=args.batch_size,
        training_steps=args.training_steps,
        training_batch_size=args.training_batch_size,
    )
