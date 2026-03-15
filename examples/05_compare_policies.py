#!/usr/bin/env python
"""Benchmark all ported policies: parameter count, forward pass time, memory usage.

Creates each policy with a small configuration, runs a forward pass, and
reports timing and parameter counts in a summary table.
"""

import time

import mlx.core as mx

# ---------------------------------------------------------------------------
# Helper: time a forward pass
# ---------------------------------------------------------------------------
def time_forward(fn, *args, n_warmup=2, n_runs=5, **kwargs):
    """Time a function, returning median time in milliseconds."""
    # Warmup
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        if isinstance(result, tuple):
            mx.eval(*[r for r in result if r is not None and isinstance(r, mx.array)])
        elif isinstance(result, mx.array):
            mx.eval(result)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if isinstance(result, tuple):
            mx.eval(*[r for r in result if r is not None and isinstance(r, mx.array)])
        elif isinstance(result, mx.array):
            mx.eval(result)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Policy definitions: (name, create_fn, forward_fn)
# ---------------------------------------------------------------------------
results = []

# --- ACT ---
def bench_act():
    from lerobot_mlx.policies.act.configuration_act import (
        ACTConfig, PolicyFeature, FeatureType, ACTION, OBS_STATE, OBS_ENV_STATE,
    )
    from lerobot_mlx.policies.act.modeling_act import ACTPolicy

    config = ACTConfig(
        chunk_size=10, n_action_steps=10,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(14,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        },
        dim_model=64, n_heads=4, dim_feedforward=128,
        n_encoder_layers=2, n_decoder_layers=1,
        n_vae_encoder_layers=2, latent_dim=8,
        pretrained_backbone_weights=None, dropout=0.0,
    )
    policy = ACTPolicy(config)
    policy.eval()
    batch = {
        OBS_STATE: mx.random.normal((1, 14)),
        OBS_ENV_STATE: mx.random.normal((1, 14)),
    }
    def fwd():
        return policy.model(batch)
    t = time_forward(fwd)
    return "ACT", policy.num_parameters(), t

# --- Diffusion ---
def bench_diffusion():
    from lerobot_mlx.policies.diffusion.configuration_diffusion import (
        DiffusionConfig, PolicyFeature,
    )
    from lerobot_mlx.policies.diffusion.modeling_diffusion import DiffusionPolicy

    config = DiffusionConfig(
        n_obs_steps=2, horizon=16, n_action_steps=8,
        input_features={
            "observation.state": PolicyFeature(type="STATE", shape=(2,)),
            "observation.environment_state": PolicyFeature(type="ENV_STATE", shape=(2,)),
        },
        output_features={
            "action": PolicyFeature(type="ACTION", shape=(2,)),
        },
        down_dims=(64, 128, 256),
        num_train_timesteps=10, diffusion_step_embed_dim=32,
        n_groups=4, kernel_size=3,
        spatial_softmax_num_keypoints=8,
    )
    policy = DiffusionPolicy(config)
    policy.eval()
    batch = {
        "observation.state": mx.random.normal((1, 2, 2)),
        "observation.environment_state": mx.random.normal((1, 2, 2)),
    }
    def fwd():
        return policy.diffusion.generate_actions(batch)
    t = time_forward(fwd)
    return "Diffusion", policy.num_parameters(), t

# --- SAC ---
def bench_sac():
    from lerobot_mlx.policies.sac.configuration_sac import (
        SACConfig, FeatureShape, ACTION, OBS_STATE,
    )
    from lerobot_mlx.policies.sac.modeling_sac import SACPolicy

    config = SACConfig(
        input_features={OBS_STATE: FeatureShape(shape=(10,))},
        output_features={ACTION: FeatureShape(shape=(4,))},
        latent_dim=64, num_critics=2,
    )
    policy = SACPolicy(config)
    policy.eval()
    batch = {OBS_STATE: mx.random.normal((1, 10))}
    def fwd():
        return policy.select_action(batch)
    t = time_forward(fwd)
    return "SAC", policy.num_parameters(), t

# --- TD-MPC ---
def bench_tdmpc():
    from lerobot_mlx.policies.tdmpc.configuration_tdmpc import (
        TDMPCConfig, PolicyFeature, FeatureType, ACTION, OBS_STATE,
    )
    from lerobot_mlx.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

    config = TDMPCConfig(
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (4,)),
        },
        output_features={
            ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
        },
        latent_dim=16, mlp_dim=32, q_ensemble_size=3,
        horizon=3, n_gaussian_samples=8, n_pi_samples=4, n_elites=4,
    )
    policy = TDMPCPolicy(config)
    policy.eval()

    # TD-MPC encode + policy forward
    obs = {OBS_STATE: mx.random.normal((1, 4))}
    def fwd():
        z = policy.model.encode(obs)
        return policy.model.pi(z, config.max_random_shift_ratio)
    t = time_forward(fwd)
    return "TD-MPC", policy.num_parameters(), t

# --- VQ-BeT ---
def bench_vqbet():
    from lerobot_mlx.policies.vqbet.configuration_vqbet import (
        VQBeTConfig, PolicyFeature, FeatureType, ACTION, OBS_STATE,
    )
    from lerobot_mlx.policies.vqbet.modeling_vqbet import VQBeTPolicy

    config = VQBeTConfig(
        n_obs_steps=2, n_action_pred_token=2, action_chunk_size=3,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        crop_shape=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=8,
        gpt_block_size=100,
        gpt_input_dim=64, gpt_output_dim=64,
        gpt_n_layer=2, gpt_n_head=2, gpt_hidden_dim=64,
        dropout=0.0,
        vqvae_n_embed=8, vqvae_embedding_dim=32, vqvae_enc_hidden_dim=32,
    )
    policy = VQBeTPolicy(config)
    policy.eval()

    batch = {
        OBS_STATE: mx.random.normal((1, 2, 4)),
        "observation.images": mx.random.normal((1, 2, 1, 3, 64, 64)),
    }
    def fwd():
        return policy.vqbet(batch, rollout=True)
    t = time_forward(fwd)
    return "VQ-BeT", policy.num_parameters(), t

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
print("Benchmarking all policies...")
print("=" * 65)
print(f"{'Policy':<12} {'Parameters':>12} {'Forward (ms)':>14} {'Status':<10}")
print("-" * 65)

benchmarks = [bench_act, bench_diffusion, bench_sac, bench_tdmpc, bench_vqbet]

for bench_fn in benchmarks:
    name = bench_fn.__name__.replace("bench_", "").upper()
    try:
        name, params, t_ms = bench_fn()
        results.append((name, params, t_ms))
        print(f"{name:<12} {params:>12,} {t_ms:>12.1f}ms {'OK':<10}")
    except Exception as e:
        print(f"{name:<12} {'--':>12} {'--':>14} FAIL: {e}")

print("=" * 65)

# Summary
if results:
    total_params = sum(p for _, p, _ in results)
    print(f"\nTotal parameters across {len(results)} policies: {total_params:,}")
    fastest = min(results, key=lambda x: x[2])
    print(f"Fastest inference: {fastest[0]} ({fastest[2]:.1f}ms)")
    smallest = min(results, key=lambda x: x[1])
    print(f"Smallest model: {smallest[0]} ({smallest[1]:,} params)")

print("\nPolicy comparison complete!")
