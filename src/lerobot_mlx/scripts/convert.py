"""Convert PyTorch LeRobot checkpoints to MLX format.

Usage:
    lerobot-mlx-convert --repo-id lerobot/act_aloha_sim --output-dir converted/
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch LeRobot weights to MLX format"
    )
    parser.add_argument(
        "--repo-id", type=str, required=True,
        help="HuggingFace model repo ID (e.g., lerobot/act_aloha_sim)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="converted",
        help="Output directory for converted weights (default: converted/)"
    )
    parser.add_argument(
        "--policy-type", type=str, default=None,
        help="Policy type (auto-detected from config if not specified)"
    )
    parser.add_argument(
        "--revision", type=str, default="main",
        help="Git revision (branch, tag, or commit hash)"
    )
    return parser.parse_args()


def main():
    """Entry point for lerobot-mlx-convert CLI."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        import mlx.core as mx
    except ImportError:
        logger.error("MLX is not installed. Install with: pip install mlx")
        raise SystemExit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface-hub is not installed. Install with: pip install huggingface-hub")
        raise SystemExit(1)

    try:
        import safetensors.numpy as sf_np
    except ImportError:
        logger.error("safetensors is not installed. Install with: pip install safetensors")
        raise SystemExit(1)

    from lerobot_mlx.policies.pretrained import convert_torch_weights_to_mlx

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download weights and config
    logger.info(f"Downloading weights from {args.repo_id} (revision: {args.revision})...")
    try:
        weights_path = hf_hub_download(
            args.repo_id, "model.safetensors", revision=args.revision
        )
    except Exception as e:
        logger.error(f"Failed to download model.safetensors: {e}")
        raise SystemExit(1)

    try:
        config_path = hf_hub_download(
            args.repo_id, "config.json", revision=args.revision
        )
    except Exception as e:
        logger.error(f"Failed to download config.json: {e}")
        raise SystemExit(1)

    # Detect policy type from config if not specified
    policy_type = args.policy_type
    if policy_type is None:
        with open(config_path) as f:
            config_dict = json.load(f)
        # Try common config keys
        policy_type = config_dict.get("policy_type", config_dict.get("type", "auto"))
        logger.info(f"Auto-detected policy type: {policy_type}")

    # Load and convert weights
    logger.info("Loading PyTorch weights...")
    torch_weights = sf_np.load_file(weights_path)
    logger.info(f"  Source keys: {len(torch_weights)}")

    logger.info("Converting weights to MLX format...")
    mlx_weights = convert_torch_weights_to_mlx(torch_weights, policy_type)

    # Save as .safetensors (MLX native format)
    output_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(output_path), mlx_weights)

    # Copy config
    shutil.copy2(config_path, output_dir / "config.json")

    # Summary
    total_params = sum(v.size for v in mlx_weights.values())
    logger.info(f"Conversion complete!")
    logger.info(f"  Output:     {output_path}")
    logger.info(f"  Keys:       {len(mlx_weights)}")
    logger.info(f"  Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
