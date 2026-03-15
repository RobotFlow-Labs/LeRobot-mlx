"""LeRobot-MLX System Info CLI.

Prints system information, available policies, and MLX status.
"""
import platform
import sys
from importlib.metadata import version, PackageNotFoundError


def main():
    """Entry point for lerobot-mlx-info CLI."""
    # Version
    try:
        pkg_version = version("lerobot-mlx")
    except PackageNotFoundError:
        pkg_version = "0.1.0 (dev)"

    print(f"LeRobot-MLX v{pkg_version}")
    print("=" * 40)

    # Platform
    machine = platform.machine()
    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        print(f"Platform:  macOS {mac_ver} ({machine})")
    else:
        print(f"Platform:  {platform.system()} {platform.release()} ({machine})")
    print(f"Python:    {sys.version.split()[0]}")

    # MLX
    try:
        mlx_ver = version("mlx")
        print(f"MLX:       {mlx_ver}")
        try:
            import mlx.core as mx
            metal = "Available" if mx.metal.is_available() else "Not available"
            print(f"Metal GPU: {metal}")
        except Exception:
            print("Metal GPU: Unknown")
    except PackageNotFoundError:
        print("MLX:       Not installed")
        print("           (pip install mlx)")

    # Policies
    policies = {
        "act": "Action Chunking Transformer",
        "diffusion": "Diffusion Policy (DDPM/DDIM)",
        "sac": "Soft Actor-Critic",
        "tdmpc": "Temporal Difference MPC",
        "vqbet": "Vector Quantized Behavior Transformer",
        "sarm": "State-Action Reward Model",
        "pi0": "Pi0 Flow Matching (VLA)",
        "smolvla": "SmolVLA Flow Matching (VLA)",
    }
    print(f"\nPolicies ({len(policies)} available):")
    for name, desc in policies.items():
        print(f"  {name:<12} {desc}")

    # VLM Backend
    try:
        vlm_ver = version("mlx-vlm")
        print(f"\nVLM Backend: mlx-vlm {vlm_ver}")
    except PackageNotFoundError:
        print(f"\nVLM Backend: Not installed (pip install mlx-vlm)")

    # Memory
    try:
        import mlx.core as mx
        # Use non-deprecated API if available (MLX >= 0.31)
        get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        active = get_active() / 1e9
        peak = get_peak() / 1e9
        print(f"MLX Memory: {active:.1f} GB active, {peak:.1f} GB peak")
    except Exception:
        pass

    # System memory via sysctl on macOS
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            mem_gb = int(result.stdout.strip()) / (1024 ** 3)
            print(f"Memory:    {mem_gb:.0f} GB unified memory")
    except Exception:
        pass


if __name__ == "__main__":
    main()
