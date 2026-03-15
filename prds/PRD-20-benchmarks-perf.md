# PRD-20: Performance Benchmarks

> **Status:** TODO
> **Priority:** P1 — Proves the value of MLX port
> **Dependencies:** All policies ported
> **Estimated LOC:** ~400

---

## Objective

Create a comprehensive benchmark suite comparing LeRobot-MLX vs PyTorch (CPU/MPS) on Apple Silicon. Results should be impressive enough for investor presentations.

---

## Deliverables

### 1. `benchmarks/bench_inference.py`

Per-policy inference benchmarks:
- Warm up (10 iterations)
- Time 100 iterations
- Report: median latency (ms), throughput (actions/sec), peak memory (MB)
- Batch sizes: 1, 4, 16
- All 8 policies

### 2. `benchmarks/bench_training.py`

Training throughput:
- Steps/second for 100 training steps
- Peak memory during training
- Gradient computation time

### 3. `benchmarks/bench_memory.py`

Memory profiling:
- Model parameter memory per policy
- Peak memory during forward pass
- Peak memory during training
- Compare float32 vs float16

### 4. `benchmarks/run_all.py`

Master benchmark runner:
- Runs all benchmarks
- Produces summary table
- Exports results as JSON + markdown

### 5. Tests

- test_bench_inference_runs
- test_bench_training_runs
- test_bench_results_format

---

## Output Format

```
LeRobot-MLX Benchmark Results (Apple M3 Max, 64GB)
====================================================

Inference Latency (ms, batch_size=1):
  ACT:        2.3ms   (434 actions/sec)
  Diffusion:  15.7ms  (63 actions/sec)
  SAC:        0.8ms   (1250 actions/sec)
  TD-MPC:     3.1ms   (322 actions/sec)
  VQ-BeT:     4.2ms   (238 actions/sec)
  Pi0:        8.5ms   (117 actions/sec)

Training (steps/sec, batch_size=32):
  ACT:        45.2 steps/sec
  Diffusion:  28.1 steps/sec

Memory (MB):
  ACT:        82MB params, 340MB peak
  Diffusion:  156MB params, 520MB peak
```
