# PRD-18: Benchmarks, Performance & Documentation

> **Status:** TODO
> **Priority:** P2 — Final polish
> **Dependencies:** All previous PRDs
> **Estimated LOC:** ~400
> **Phase:** 7 (Polish)

---

## Objective

Create benchmarks comparing MLX vs PyTorch MPS performance, write documentation, and prepare for public release.

---

## Benchmarks

### 1. Inference Speed

```python
# benchmarks/bench_inference.py
# Compare: MLX vs PyTorch (CPU) vs PyTorch (MPS) on Apple Silicon

policies = ["act", "diffusion", "tdmpc"]
batch_sizes = [1, 4, 16]

for policy in policies:
    for bs in batch_sizes:
        # Warm up
        for _ in range(10): model(batch)
        # Time 100 iterations
        times = []
        for _ in range(100):
            start = time.perf_counter()
            output = model(batch)
            mx.eval(output)  # Force computation
            times.append(time.perf_counter() - start)
        print(f"{policy} bs={bs}: {np.median(times)*1000:.1f}ms")
```

### 2. Training Throughput

```python
# benchmarks/bench_training.py
# Measure: steps/second, memory usage

for policy in policies:
    trainer = Trainer(model, config)
    start = time.time()
    for step in range(100):
        trainer.train_step(batch)
    elapsed = time.time() - start
    print(f"{policy}: {100/elapsed:.1f} steps/sec")
    print(f"  Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")
```

### 3. Memory Usage

```python
# benchmarks/bench_memory.py
# Compare model sizes and peak memory during training
```

---

## Documentation

### README.md
- Installation guide
- Quick start (train + eval)
- Supported policies table
- Performance benchmarks
- Upstream sync instructions

### Examples
- `examples/train_act_pusht.py` — Train ACT on PushT dataset
- `examples/eval_pretrained.py` — Evaluate a pretrained checkpoint
- `examples/convert_weights.py` — Convert PyTorch weights to MLX

---

## Acceptance Criteria

1. Benchmark suite runs without errors
2. MLX inference is ≥ 2x faster than PyTorch MPS for ACT/Diffusion
3. README covers installation, usage, and all supported policies
4. Examples run end-to-end
5. All benchmarks report median + std timing
