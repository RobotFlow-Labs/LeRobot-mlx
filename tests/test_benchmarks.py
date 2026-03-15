"""Tests for the benchmark suite.

Verifies that benchmarks run without errors and produce correctly
formatted results. Uses ACT only for speed.
"""

import sys
from pathlib import Path

import pytest

# Ensure benchmarks package is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class TestBenchInference:
    def test_bench_inference_runs(self):
        """Benchmark suite runs without errors for ACT."""
        from benchmarks.bench_inference import run_inference_benchmarks

        results = run_inference_benchmarks(
            policies=["ACT"],
            batch_sizes=[1],
            warmup=2,
            iterations=3,
        )
        assert len(results) == 1
        assert results[0]["policy"] == "ACT"
        assert results[0]["median_ms"] > 0

    def test_bench_results_format(self):
        """Results have expected keys."""
        from benchmarks.bench_inference import run_inference_benchmarks

        results = run_inference_benchmarks(
            policies=["ACT"],
            batch_sizes=[1],
            warmup=1,
            iterations=2,
        )
        r = results[0]
        expected_keys = {
            "policy", "batch_size", "create_ms", "median_ms",
            "mean_ms", "min_ms", "max_ms", "throughput",
            "peak_mb", "params",
        }
        assert expected_keys.issubset(set(r.keys())), f"Missing keys: {expected_keys - set(r.keys())}"
        assert r["params"] > 0
        assert r["peak_mb"] >= 0

    def test_bench_multiple_batch_sizes(self):
        """Benchmarks work with multiple batch sizes."""
        from benchmarks.bench_inference import run_inference_benchmarks

        results = run_inference_benchmarks(
            policies=["ACT"],
            batch_sizes=[1, 2],
            warmup=1,
            iterations=2,
        )
        assert len(results) == 2
        assert results[0]["batch_size"] == 1
        assert results[1]["batch_size"] == 2


class TestBenchTraining:
    def test_bench_training_runs(self):
        """Training benchmark runs without errors for ACT."""
        from benchmarks.bench_training import run_training_benchmarks

        results = run_training_benchmarks(
            policies=["ACT"],
            num_steps=3,
            batch_size=2,
        )
        assert len(results) == 1
        assert results[0]["policy"] == "ACT"
        assert results[0]["steps_per_sec"] > 0

    def test_bench_training_results_format(self):
        """Training results have expected keys."""
        from benchmarks.bench_training import run_training_benchmarks

        results = run_training_benchmarks(
            policies=["ACT"],
            num_steps=2,
            batch_size=2,
        )
        r = results[0]
        expected_keys = {
            "policy", "batch_size", "num_steps", "elapsed_sec",
            "steps_per_sec", "peak_mb", "final_loss", "params",
        }
        assert expected_keys.issubset(set(r.keys()))


class TestBenchMemory:
    def test_bench_memory_runs(self):
        """Memory benchmark runs without errors for ACT."""
        from benchmarks.bench_memory import run_memory_benchmarks

        results = run_memory_benchmarks(
            policies=["ACT"],
            batch_size=1,
        )
        assert len(results) == 1
        assert results[0]["policy"] == "ACT"
        assert results[0]["param_mb"] > 0

    def test_bench_memory_results_format(self):
        """Memory results have expected keys."""
        from benchmarks.bench_memory import run_memory_benchmarks

        results = run_memory_benchmarks(
            policies=["ACT"],
            batch_size=1,
        )
        r = results[0]
        expected_keys = {
            "policy", "batch_size", "params", "param_mb",
            "forward_peak_mb", "training_peak_mb",
        }
        assert expected_keys.issubset(set(r.keys()))
