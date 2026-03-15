"""Inference latency benchmarks for all LeRobot-MLX policies.

Measures model creation time, forward-pass latency (median over 50 iterations
after 5 warm-up passes), peak memory, and parameter count for each policy.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten


# ---------------------------------------------------------------------------
# Generic benchmark harness
# ---------------------------------------------------------------------------

def count_params(model) -> int:
    """Count total parameters in an MLX nn.Module."""
    try:
        return sum(p.size for _, p in tree_flatten(model.parameters()))
    except Exception:
        return 0


def bench_policy(
    name: str,
    build_fn,
    batch_sizes: list[int] | None = None,
    warmup: int = 5,
    iterations: int = 50,
) -> list[dict[str, Any]]:
    """Benchmark a single policy.

    Args:
        name: Human-readable policy name.
        build_fn: Callable(batch_size) -> (model, batch, eval_fn).
            ``eval_fn(model, batch)`` runs a forward pass and returns a value
            suitable for ``mx.eval()``.
        batch_sizes: List of batch sizes to test.
        warmup: Number of warm-up iterations (not timed).
        iterations: Number of timed iterations.

    Returns:
        List of result dicts, one per batch size.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4]

    results: list[dict[str, Any]] = []

    for bs in batch_sizes:
        # Build ---------------------------------------------------------------
        t0 = time.perf_counter()
        model, batch, eval_fn = build_fn(bs)
        mx.eval(model.parameters())
        create_ms = (time.perf_counter() - t0) * 1000

        n_params = count_params(model)

        # Warm-up --------------------------------------------------------------
        for _ in range(warmup):
            out = eval_fn(model, batch)
            mx.eval(out)

        # Measure --------------------------------------------------------------
        mx.reset_peak_memory()
        times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            out = eval_fn(model, batch)
            mx.eval(out)
            times.append(time.perf_counter() - start)

        median_ms = sorted(times)[len(times) // 2] * 1000
        mean_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        peak_mb = mx.get_peak_memory() / 1e6

        results.append({
            "policy": name,
            "batch_size": bs,
            "create_ms": round(create_ms, 2),
            "median_ms": round(median_ms, 3),
            "mean_ms": round(mean_ms, 3),
            "min_ms": round(min_ms, 3),
            "max_ms": round(max_ms, 3),
            "throughput": round(1000 / median_ms, 1) if median_ms > 0 else 0,
            "peak_mb": round(peak_mb, 1),
            "params": n_params,
        })

    return results


# ============================================================================
# Policy-specific builders
# ============================================================================

def _extract_loss(out):
    """Extract a scalar loss from various policy output formats."""
    if isinstance(out, dict):
        # Pi0 returns {"loss": ...}
        for key in ("loss", "loss_critic", "loss_actor"):
            if key in out:
                v = out[key]
                return v if isinstance(v, mx.array) else mx.array(float(v))
        # Fallback: first value
        for v in out.values():
            if isinstance(v, mx.array):
                return v
        return mx.array(0.0)
    if isinstance(out, tuple):
        # ACT, SmolVLA, SARM, VQ-BeT return (loss, info_dict)
        loss = out[0]
        return loss if isinstance(loss, mx.array) else mx.array(float(loss))
    if isinstance(out, mx.array):
        return out
    return mx.array(0.0)


def _build_act(batch_size: int):
    """ACT policy builder."""
    from lerobot_mlx.policies.act.configuration_act import (
        ACTConfig, ACTION, FeatureType, OBS_ENV_STATE, OBS_STATE, PolicyFeature,
    )
    from lerobot_mlx.policies.act.modeling_act import ACTPolicy

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
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        latent_dim=8,
        use_vae=True,
        dropout=0.0,
        pretrained_backbone_weights=None,
    )
    model = ACTPolicy(config)
    model.train()

    batch = {
        OBS_STATE: mx.random.normal(shape=(batch_size, state_dim)),
        OBS_ENV_STATE: mx.random.normal(shape=(batch_size, state_dim)),
        ACTION: mx.random.normal(shape=(batch_size, chunk_size, action_dim)),
    }

    def eval_fn(m, b):
        out = m.forward(b)
        return _extract_loss(out)

    return model, batch, eval_fn


def _build_diffusion(batch_size: int):
    """Diffusion policy builder."""
    from lerobot_mlx.policies.diffusion.configuration_diffusion import (
        DiffusionConfig, PolicyFeature,
    )
    from lerobot_mlx.policies.diffusion.modeling_diffusion import (
        ACTION, OBS_ENV_STATE, OBS_STATE, DiffusionPolicy,
    )

    state_dim, action_dim, env_state_dim = 2, 2, 4
    n_obs_steps, horizon, n_action_steps = 2, 16, 8

    config = DiffusionConfig(
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
        input_features={
            "observation.state": PolicyFeature(type="STATE", shape=(state_dim,)),
            "observation.environment_state": PolicyFeature(type="ENV_STATE", shape=(env_state_dim,)),
        },
        output_features={
            "action": PolicyFeature(type="ACTION", shape=(action_dim,)),
        },
        down_dims=(64, 128, 256),
        num_train_timesteps=10,
        noise_scheduler_type="DDPM",
        diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8,
        n_groups=4,
        kernel_size=3,
        use_group_norm=True,
        use_film_scale_modulation=True,
    )
    model = DiffusionPolicy(config)
    model.train()

    batch = {
        OBS_STATE: mx.random.normal((batch_size, n_obs_steps, state_dim)),
        OBS_ENV_STATE: mx.random.normal((batch_size, n_obs_steps, env_state_dim)),
        ACTION: mx.random.normal((batch_size, horizon, action_dim)),
    }

    def eval_fn(m, b):
        out = m(b)
        return _extract_loss(out)

    return model, batch, eval_fn


def _build_sac(batch_size: int):
    """SAC policy builder."""
    from lerobot_mlx.policies.sac.configuration_sac import (
        SACConfig, FeatureShape, ACTION, OBS_STATE,
    )
    from lerobot_mlx.policies.sac.modeling_sac import SACPolicy

    obs_dim, action_dim = 10, 4
    config = SACConfig(
        input_features={OBS_STATE: FeatureShape(shape=(obs_dim,))},
        output_features={ACTION: FeatureShape(shape=(action_dim,))},
        latent_dim=64,
        num_critics=2,
        temperature_init=1.0,
        discount=0.99,
        critic_target_update_weight=0.005,
        use_backup_entropy=True,
    )
    model = SACPolicy(config=config)
    model.train()

    obs = mx.random.normal(shape=(batch_size, obs_dim))
    next_obs = mx.random.normal(shape=(batch_size, obs_dim))
    actions = mx.random.uniform(low=-1.0, high=1.0, shape=(batch_size, action_dim))
    rewards = mx.random.normal(shape=(batch_size,))
    done = mx.zeros((batch_size,))

    batch = {
        ACTION: actions,
        "state": {OBS_STATE: obs},
        "next_state": {OBS_STATE: next_obs},
        "reward": rewards,
        "done": done,
    }

    def eval_fn(m, b):
        out = m.forward(b, model="critic")
        return _extract_loss(out)

    return model, batch, eval_fn


def _build_tdmpc(batch_size: int):
    """TD-MPC policy builder."""
    from lerobot_mlx.policies.tdmpc.configuration_tdmpc import (
        TDMPCConfig, PolicyFeature, FeatureType, ACTION, OBS_STATE, REWARD,
    )
    from lerobot_mlx.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

    state_dim, action_dim = 4, 2
    horizon = 3
    config = TDMPCConfig(
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (state_dim,)),
        },
        output_features={
            ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,)),
        },
        latent_dim=16,
        mlp_dim=32,
        q_ensemble_size=3,
        horizon=horizon,
        n_gaussian_samples=8,
        n_pi_samples=4,
        n_elites=4,
    )
    model = TDMPCPolicy(config)
    model.train()

    # TD-MPC forward expects: obs (B, horizon+1, state_dim), action (B, horizon, action_dim),
    # reward (B, horizon), transposes to (T, B, ...)
    batch = {
        OBS_STATE: mx.random.normal(shape=(batch_size, horizon + 1, state_dim)),
        ACTION: mx.random.normal(shape=(batch_size, horizon, action_dim)),
        REWARD: mx.random.normal(shape=(batch_size, horizon)),
    }

    def eval_fn(m, b):
        out = m.forward(b)
        return _extract_loss(out)

    return model, batch, eval_fn


def _build_pi0(batch_size: int):
    """Pi0 policy builder."""
    from lerobot_mlx.policies.pi0.configuration_pi0 import Pi0Config
    from lerobot_mlx.policies.pi0.modeling_pi0 import Pi0Policy

    config = Pi0Config(
        action_expert_variant="gemma_300m",
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=8,
        max_action_dim=8,
        num_inference_steps=3,
    )
    model = Pi0Policy(config)
    model.train()

    batch = {
        "observation.state": mx.random.normal((batch_size, 8)),
        "action": mx.random.normal((batch_size, 10, 8)),
    }

    def eval_fn(m, b):
        out = m(b)
        return _extract_loss(out)

    return model, batch, eval_fn


def _build_sarm(batch_size: int):
    """SARM reward model builder."""
    from lerobot_mlx.policies.sarm.configuration_sarm import SARMConfig
    from lerobot_mlx.policies.sarm.modeling_sarm import SARMRewardModel

    config = SARMConfig(
        annotation_mode="single_stage",
        n_obs_steps=8,
        max_rewind_steps=4,
        image_dim=32,
        text_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        max_state_dim=8,
        dropout=0.0,
    )
    model = SARMRewardModel(config)
    model.train()

    T = config.num_frames
    batch = {
        "video_features": mx.random.normal((batch_size, T, config.image_dim)),
        "text_features": mx.random.normal((batch_size, config.text_dim)),
        "state_features": mx.random.normal((batch_size, T, config.max_state_dim)),
        "lengths": mx.full((batch_size,), T, dtype=mx.int32),
        "sparse_targets": mx.random.uniform(shape=(batch_size, T)) * 0.99,
    }

    def eval_fn(m, b):
        out = m(b)
        return _extract_loss(out)

    return model, batch, eval_fn


def _build_smolvla(batch_size: int):
    """SmolVLA policy builder."""
    from lerobot_mlx.policies.smolvla.configuration_smolvla import (
        SmolVLAConfig, ACTION, OBS_STATE, FeatureType, PolicyFeature,
    )
    from lerobot_mlx.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig(
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=8,
        max_action_dim=6,
        expert_hidden_size=64,
        expert_num_heads=4,
        expert_num_layers=2,
        expert_head_dim=16,
        expert_width_multiplier=1.0,
        vlm_hidden_size=64,
        num_inference_steps=5,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )
    model = SmolVLAPolicy(config)
    model.train()

    batch = {
        OBS_STATE: mx.random.normal(shape=(batch_size, 8)),
        ACTION: mx.random.normal(shape=(batch_size, 10, 6)),
    }

    def eval_fn(m, b):
        out = m(b)
        return _extract_loss(out)

    return model, batch, eval_fn


def _build_vqbet(batch_size: int):
    """VQ-BeT policy builder (with synthetic images)."""
    from lerobot_mlx.policies.vqbet.configuration_vqbet import (
        VQBeTConfig, PolicyFeature, FeatureType, ACTION, OBS_STATE,
    )
    from lerobot_mlx.policies.vqbet.modeling_vqbet import VQBeTPolicy

    state_dim, action_dim = 4, 2
    img_c, img_h, img_w = 3, 96, 96
    n_obs_steps = 2
    n_action_pred_token = 2
    action_chunk_size = 3

    config = VQBeTConfig(
        n_obs_steps=n_obs_steps,
        n_action_pred_token=n_action_pred_token,
        action_chunk_size=action_chunk_size,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(img_c, img_h, img_w)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        vision_backbone="resnet18",
        crop_shape=(84, 84),
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=8,
        gpt_block_size=100,
        gpt_input_dim=64,
        gpt_output_dim=64,
        gpt_n_layer=2,
        gpt_n_head=2,
        gpt_hidden_dim=64,
        dropout=0.0,
        vqvae_n_embed=8,
        vqvae_embedding_dim=32,
        vqvae_enc_hidden_dim=32,
    )
    model = VQBeTPolicy(config)
    model.train()

    # VQ-BeT needs: OBS_STATE (B, T, state), observation.image (B, T, 1, C, H, W), ACTION (B, T_act, action)
    total_action_steps = n_obs_steps + n_action_pred_token - 1 + action_chunk_size - 1
    batch = {
        OBS_STATE: mx.random.normal(shape=(batch_size, n_obs_steps, state_dim)),
        "observation.image": mx.random.normal(shape=(batch_size, n_obs_steps, 1, img_c, img_h, img_w)),
        ACTION: mx.random.normal(shape=(batch_size, total_action_steps, action_dim)),
    }

    def eval_fn(m, b):
        out = m(b)
        return _extract_loss(out)

    return model, batch, eval_fn


# ============================================================================
# Registry of all policies
# ============================================================================

POLICY_BUILDERS: dict[str, Any] = {
    "ACT": _build_act,
    "Diffusion": _build_diffusion,
    "SAC": _build_sac,
    "TD-MPC": _build_tdmpc,
    "Pi0": _build_pi0,
    "SARM": _build_sarm,
    "SmolVLA": _build_smolvla,
    "VQ-BeT": _build_vqbet,
}


# ============================================================================
# Runner
# ============================================================================

def run_inference_benchmarks(
    policies: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    warmup: int = 5,
    iterations: int = 50,
) -> list[dict[str, Any]]:
    """Run inference benchmarks for selected policies.

    Args:
        policies: List of policy names (None = all).
        batch_sizes: List of batch sizes to test.
        warmup: Number of warm-up iterations.
        iterations: Number of timed iterations.

    Returns:
        List of result dicts.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4]
    if policies is None:
        policies = list(POLICY_BUILDERS.keys())

    all_results: list[dict[str, Any]] = []
    for name in policies:
        if name not in POLICY_BUILDERS:
            print(f"  [SKIP] Unknown policy: {name}")
            continue
        print(f"  Benchmarking {name}...")
        try:
            results = bench_policy(
                name=name,
                build_fn=POLICY_BUILDERS[name],
                batch_sizes=batch_sizes,
                warmup=warmup,
                iterations=iterations,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            for bs in batch_sizes:
                all_results.append({
                    "policy": name,
                    "batch_size": bs,
                    "create_ms": 0,
                    "median_ms": 0,
                    "mean_ms": 0,
                    "min_ms": 0,
                    "max_ms": 0,
                    "throughput": 0,
                    "peak_mb": 0,
                    "params": 0,
                    "error": str(e),
                })

    return all_results


def print_inference_results(results: list[dict[str, Any]]) -> str:
    """Format results as a readable table. Returns the table string."""
    header = f"{'Policy':<12} {'BS':>3} {'Median(ms)':>11} {'Mean(ms)':>10} {'Thru(act/s)':>12} {'Peak(MB)':>9} {'Params':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        err = r.get("error", "")
        if err:
            lines.append(f"{r['policy']:<12} {r['batch_size']:>3}   ERROR: {err}")
        else:
            lines.append(
                f"{r['policy']:<12} {r['batch_size']:>3} "
                f"{r['median_ms']:>11.3f} {r['mean_ms']:>10.3f} "
                f"{r['throughput']:>12.1f} {r['peak_mb']:>9.1f} "
                f"{r['params']:>10,}"
            )
    table = "\n".join(lines)
    return table


if __name__ == "__main__":
    import sys
    chip = platform.processor() or platform.machine()
    print(f"\nLeRobot-MLX Inference Benchmarks ({chip})")
    print("=" * 60)

    pols = sys.argv[1:] if len(sys.argv) > 1 else None
    results = run_inference_benchmarks(policies=pols)
    print()
    print(print_inference_results(results))
