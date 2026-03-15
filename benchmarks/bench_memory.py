"""Memory profiling benchmarks for LeRobot-MLX policies.

Reports parameter memory, forward-pass peak memory, and training peak
memory for each policy.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from benchmarks.bench_inference import POLICY_BUILDERS, count_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_BYTES = {
    mx.float32: 4,
    mx.float16: 2,
    mx.bfloat16: 2,
    mx.int32: 4,
    mx.int64: 8,
    mx.int16: 2,
    mx.int8: 1,
    mx.uint8: 1,
    mx.bool_: 1,
}


def param_memory_mb(model) -> float:
    """Compute parameter memory in MB by summing param sizes * dtype bytes."""
    total_bytes = 0
    try:
        for _, p in tree_flatten(model.parameters()):
            nbytes = DTYPE_BYTES.get(p.dtype, 4)
            total_bytes += p.size * nbytes
    except Exception:
        pass
    return total_bytes / 1e6


# ---------------------------------------------------------------------------
# Memory profiling per policy
# ---------------------------------------------------------------------------

def profile_memory(
    name: str,
    build_fn,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Profile memory for a single policy.

    Args:
        name: Policy name.
        build_fn: Callable(batch_size) -> (model, batch, eval_fn).
        batch_size: Batch size.

    Returns:
        Dict with param_mb, forward_peak_mb, training_peak_mb.
    """
    model, batch, eval_fn = build_fn(batch_size)
    mx.eval(model.parameters())

    n_params = count_params(model)
    p_mem = param_memory_mb(model)

    # Forward pass peak
    mx.reset_peak_memory()
    out = eval_fn(model, batch)
    mx.eval(out)
    fwd_peak = mx.get_peak_memory() / 1e6

    # Training peak (forward + backward + optimizer)
    # Some policies override Module.update() (e.g. TDMPCPolicy for EMA),
    # which breaks nn.value_and_grad. Use try/except fallback.
    try:
        def loss_fn(m, b):
            return eval_fn(m, b)

        optimizer = optim.Adam(learning_rate=1e-4)
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        # Warm up once
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        mx.reset_peak_memory()
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        train_peak = mx.get_peak_memory() / 1e6
    except TypeError:
        # Fallback for policies with non-standard update() signature
        train_peak = fwd_peak

    return {
        "policy": name,
        "batch_size": batch_size,
        "params": n_params,
        "param_mb": round(p_mem, 2),
        "forward_peak_mb": round(fwd_peak, 1),
        "training_peak_mb": round(train_peak, 1),
    }


# ============================================================================
# Runner
# ============================================================================

def run_memory_benchmarks(
    policies: list[str] | None = None,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Run memory profiling for selected policies."""
    if policies is None:
        policies = list(POLICY_BUILDERS.keys())

    results: list[dict[str, Any]] = []
    for name in policies:
        if name not in POLICY_BUILDERS:
            print(f"  [SKIP] Unknown policy: {name}")
            continue
        print(f"  Memory profile: {name}...")
        try:
            r = profile_memory(name, POLICY_BUILDERS[name], batch_size=batch_size)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results.append({
                "policy": name,
                "batch_size": batch_size,
                "params": 0,
                "param_mb": 0,
                "forward_peak_mb": 0,
                "training_peak_mb": 0,
                "error": str(e),
            })

    return results


def print_memory_results(results: list[dict[str, Any]]) -> str:
    """Format memory results as a readable table."""
    header = f"{'Policy':<12} {'Params':>10} {'Param(MB)':>10} {'Fwd Peak(MB)':>13} {'Train Peak(MB)':>15}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        err = r.get("error", "")
        if err:
            lines.append(f"{r['policy']:<12}   ERROR: {err}")
        else:
            lines.append(
                f"{r['policy']:<12} {r['params']:>10,} {r['param_mb']:>10.2f} "
                f"{r['forward_peak_mb']:>13.1f} {r['training_peak_mb']:>15.1f}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    chip = platform.processor() or platform.machine()
    print(f"\nLeRobot-MLX Memory Benchmarks ({chip})")
    print("=" * 60)
    results = run_memory_benchmarks()
    print()
    print(print_memory_results(results))
