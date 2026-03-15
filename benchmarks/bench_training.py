"""Training throughput benchmarks for LeRobot-MLX policies.

Measures steps/sec and peak memory for training loops using manual
forward + backward passes with mlx.nn.value_and_grad.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from benchmarks.bench_inference import count_params


# ---------------------------------------------------------------------------
# Generic training benchmark
# ---------------------------------------------------------------------------

def bench_training(
    name: str,
    build_fn,
    num_steps: int = 20,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Benchmark training throughput for a single policy.

    Args:
        name: Policy name.
        build_fn: Callable(batch_size) -> (model, batch, loss_fn).
            ``loss_fn(model, batch)`` -> scalar loss.
        num_steps: Number of training steps to measure.
        batch_size: Batch size.

    Returns:
        Result dict with steps_per_sec, peak_mb, etc.
    """
    model, batch, loss_fn = build_fn(batch_size)

    optimizer = optim.Adam(learning_rate=1e-4)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Warm up (3 steps)
    for _ in range(3):
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    # Measure
    mx.reset_peak_memory()
    t_start = time.perf_counter()
    for _ in range(num_steps):
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
    elapsed = time.perf_counter() - t_start

    peak_mb = mx.get_peak_memory() / 1e6
    steps_per_sec = num_steps / elapsed

    return {
        "policy": name,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "elapsed_sec": round(elapsed, 3),
        "steps_per_sec": round(steps_per_sec, 2),
        "peak_mb": round(peak_mb, 1),
        "final_loss": round(float(loss.item()), 6),
        "params": count_params(model),
    }


# ============================================================================
# Policy-specific training builders
# ============================================================================

def _train_build_act(batch_size: int):
    """ACT training builder."""
    from lerobot_mlx.policies.act.configuration_act import (
        ACTConfig, ACTION, FeatureType, OBS_ENV_STATE, OBS_STATE, PolicyFeature,
    )
    from lerobot_mlx.policies.act.modeling_act import ACTPolicy
    from benchmarks.bench_inference import _extract_loss

    state_dim, action_dim, chunk_size = 14, 14, 10
    config = ACTConfig(
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(state_dim,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        dim_model=64, n_heads=4, dim_feedforward=128,
        n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=2,
        latent_dim=8, use_vae=True, dropout=0.0,
        pretrained_backbone_weights=None,
    )
    model = ACTPolicy(config)
    model.train()

    batch = {
        OBS_STATE: mx.random.normal(shape=(batch_size, state_dim)),
        OBS_ENV_STATE: mx.random.normal(shape=(batch_size, state_dim)),
        ACTION: mx.random.normal(shape=(batch_size, chunk_size, action_dim)),
    }

    def loss_fn(m, b):
        out = m.forward(b)
        return _extract_loss(out)

    return model, batch, loss_fn


def _train_build_diffusion(batch_size: int):
    """Diffusion training builder."""
    from lerobot_mlx.policies.diffusion.configuration_diffusion import (
        DiffusionConfig, PolicyFeature,
    )
    from lerobot_mlx.policies.diffusion.modeling_diffusion import (
        ACTION, OBS_ENV_STATE, OBS_STATE, DiffusionPolicy,
    )
    from benchmarks.bench_inference import _extract_loss

    state_dim, action_dim, env_state_dim = 2, 2, 4
    n_obs_steps, horizon, n_action_steps = 2, 16, 8
    config = DiffusionConfig(
        n_obs_steps=n_obs_steps, horizon=horizon, n_action_steps=n_action_steps,
        input_features={
            "observation.state": PolicyFeature(type="STATE", shape=(state_dim,)),
            "observation.environment_state": PolicyFeature(type="ENV_STATE", shape=(env_state_dim,)),
        },
        output_features={"action": PolicyFeature(type="ACTION", shape=(action_dim,))},
        down_dims=(64, 128, 256), num_train_timesteps=10,
        noise_scheduler_type="DDPM", diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8, n_groups=4, kernel_size=3,
        use_group_norm=True, use_film_scale_modulation=True,
    )
    model = DiffusionPolicy(config)
    model.train()

    batch = {
        OBS_STATE: mx.random.normal((batch_size, n_obs_steps, state_dim)),
        OBS_ENV_STATE: mx.random.normal((batch_size, n_obs_steps, env_state_dim)),
        ACTION: mx.random.normal((batch_size, horizon, action_dim)),
    }

    def loss_fn(m, b):
        out = m(b)
        return _extract_loss(out)

    return model, batch, loss_fn


TRAINING_BUILDERS = {
    "ACT": _train_build_act,
    "Diffusion": _train_build_diffusion,
}


# ============================================================================
# Runner
# ============================================================================

def run_training_benchmarks(
    policies: list[str] | None = None,
    num_steps: int = 20,
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Run training benchmarks for selected policies."""
    if policies is None:
        policies = list(TRAINING_BUILDERS.keys())

    results: list[dict[str, Any]] = []
    for name in policies:
        if name not in TRAINING_BUILDERS:
            print(f"  [SKIP] No training builder for: {name}")
            continue
        print(f"  Training benchmark: {name}...")
        try:
            r = bench_training(name, TRAINING_BUILDERS[name], num_steps=num_steps, batch_size=batch_size)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results.append({
                "policy": name,
                "batch_size": batch_size,
                "num_steps": num_steps,
                "elapsed_sec": 0,
                "steps_per_sec": 0,
                "peak_mb": 0,
                "final_loss": 0,
                "params": 0,
                "error": str(e),
            })

    return results


def print_training_results(results: list[dict[str, Any]]) -> str:
    """Format training results as a readable table."""
    header = f"{'Policy':<12} {'BS':>3} {'Steps/sec':>10} {'Peak(MB)':>9} {'Loss':>10} {'Params':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        err = r.get("error", "")
        if err:
            lines.append(f"{r['policy']:<12} {r['batch_size']:>3}   ERROR: {err}")
        else:
            lines.append(
                f"{r['policy']:<12} {r['batch_size']:>3} "
                f"{r['steps_per_sec']:>10.2f} {r['peak_mb']:>9.1f} "
                f"{r['final_loss']:>10.6f} {r['params']:>10,}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    chip = platform.processor() or platform.machine()
    print(f"\nLeRobot-MLX Training Benchmarks ({chip})")
    print("=" * 60)
    results = run_training_benchmarks()
    print()
    print(print_training_results(results))
