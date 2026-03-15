"""Load pretrained LeRobot policies from HuggingFace Hub, converting PyTorch weights to MLX."""

import json
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def load_pretrained(repo_id: str, policy_class=None, revision: str = "main"):
    """Load a pretrained policy from HuggingFace Hub.

    Downloads the model config and weights, auto-detects the policy type,
    converts PyTorch weights to MLX format, and returns the ready-to-use model.

    Args:
        repo_id: HuggingFace Hub model ID (e.g., "lerobot/act_aloha_sim_transfer_cube_human")
        policy_class: Optional explicit policy class. Auto-detected from config if None.
        revision: Git revision (branch, tag, or commit hash).

    Returns:
        Initialized policy with pretrained weights loaded.
    """
    from huggingface_hub import hf_hub_download

    # Download config
    config_path = hf_hub_download(repo_id, "config.json", revision=revision)
    with open(config_path) as f:
        config_dict = json.load(f)

    # Auto-detect policy type
    policy_type = _detect_policy_type(config_dict)
    logger.info(f"Detected policy type: {policy_type}")

    # Get policy class and config class
    if policy_class is None:
        policy_class, config_class = _get_policy_classes(policy_type)
    else:
        _, config_class = _get_policy_classes(policy_type)

    # Create config from dict
    config = _create_config(config_class, config_dict)

    # Create model
    model = policy_class(config)

    # Download and convert weights
    try:
        weights_path = hf_hub_download(repo_id, "model.safetensors", revision=revision)
    except Exception:
        weights_path = hf_hub_download(repo_id, "pytorch_model.bin", revision=revision)

    _load_and_convert_weights(model, weights_path, policy_type)

    logger.info(f"Loaded pretrained {policy_type} from {repo_id}")
    return model


def _detect_policy_type(config_dict):
    """Detect policy type from config dict."""
    # Check various fields that indicate policy type
    if "policy" in config_dict and isinstance(config_dict["policy"], dict) and "type" in config_dict["policy"]:
        return config_dict["policy"]["type"]
    # Check _target_ (draccus-style)
    if "_target_" in config_dict:
        target = config_dict["_target_"]
        for name in ["act", "diffusion", "tdmpc", "vqbet", "sac"]:
            if name in target.lower():
                return name
    # Check model_type
    if "model_type" in config_dict:
        return config_dict["model_type"]
    raise ValueError(f"Cannot detect policy type from config: {list(config_dict.keys())}")


def _get_policy_classes(policy_type):
    """Get (PolicyClass, ConfigClass) for a policy type."""
    REGISTRY = {
        "act": ("lerobot_mlx.policies.act", "ACTPolicy", "ACTConfig"),
        "diffusion": ("lerobot_mlx.policies.diffusion", "DiffusionPolicy", "DiffusionConfig"),
        "sac": ("lerobot_mlx.policies.sac", "SACPolicy", "SACConfig"),
    }
    if policy_type not in REGISTRY:
        raise ValueError(f"Unknown policy type: {policy_type}. Available: {list(REGISTRY.keys())}")

    module_path, policy_name, config_name = REGISTRY[policy_type]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, policy_name), getattr(mod, config_name)


def _create_config(config_class, config_dict):
    """Create a config instance from a dict, handling nested structures."""
    import dataclasses
    if dataclasses.is_dataclass(config_class):
        field_names = {f.name for f in dataclasses.fields(config_class)}
        filtered = {k: v for k, v in config_dict.items() if k in field_names}
        return config_class(**filtered)
    return config_class()


def _load_and_convert_weights(model, weights_path, policy_type):
    """Load PyTorch weights and convert to MLX format."""
    if weights_path.endswith(".safetensors"):
        import safetensors.numpy as sf_np
        torch_weights = sf_np.load_file(weights_path)
    else:
        # .bin format -- need torch to load
        raise NotImplementedError("Only safetensors format supported. PyTorch .bin requires torch.")

    # Convert weight names and formats
    mlx_weights = convert_torch_weights_to_mlx(torch_weights, policy_type)

    # Load into model
    model.load_weights(list(mlx_weights.items()))


def convert_torch_weights_to_mlx(torch_weights, policy_type="auto"):
    """Convert a dict of PyTorch weights to MLX format.

    Handles:
    - Conv2d weight transposition: OIHW -> OHWI
    - Conv1d weight transposition: OIL -> OLI
    - Key name remapping for torch->mlx module structure differences
    - Skipping optimizer state and batch norm tracking keys
    """
    mlx_weights = {}

    for key, value in torch_weights.items():
        mlx_key = _remap_weight_key(key, policy_type)
        if mlx_key is None:
            continue

        # Transpose conv weights
        if value.ndim == 4 and _is_conv2d_weight(key):
            # PyTorch Conv2d: (O, I, H, W) -> MLX: (O, H, W, I)
            value = np.transpose(value, (0, 2, 3, 1))
        elif value.ndim == 3 and _is_conv1d_weight(key):
            # PyTorch Conv1d: (O, I, L) -> MLX: (O, L, I)
            value = np.transpose(value, (0, 2, 1))

        mlx_weights[mlx_key] = mx.array(value)

    return mlx_weights


def _remap_weight_key(key, policy_type):
    """Remap PyTorch weight key to MLX key.

    Most keys map 1:1. Some need adjustment for our wrapper layers
    (Conv2d stores weights in ._conv, etc.)
    """
    # Skip optimizer state and other non-weight keys
    if "optimizer" in key or "scheduler" in key or "num_batches_tracked" in key:
        return None

    return key


def _is_conv2d_weight(key):
    """Heuristic: is this a Conv2d weight key?"""
    return "weight" in key and ("conv" in key.lower() or "downsample.0" in key)


def _is_conv1d_weight(key):
    """Heuristic: is this a Conv1d weight key?"""
    return "weight" in key and ("conv1d" in key.lower() or "temporal" in key.lower())
