"""Policy factory -- maps config types to policy classes."""

import importlib
import logging
from typing import Optional, Type

logger = logging.getLogger(__name__)

# Lazy registry: maps policy type name -> (module_path, class_name)
_POLICY_REGISTRY = {
    "act": ("lerobot_mlx.policies.act", "ACTPolicy"),
    "diffusion": ("lerobot_mlx.policies.diffusion", "DiffusionPolicy"),
    "sac": ("lerobot_mlx.policies.sac", "SACPolicy"),
    "tdmpc": ("lerobot_mlx.policies.tdmpc", "TDMPCPolicy"),
    "vqbet": ("lerobot_mlx.policies.vqbet", "VQBeTPolicy"),
    "sarm": ("lerobot_mlx.policies.sarm", "SARMRewardModel"),
    "pi0": ("lerobot_mlx.policies.pi0", "Pi0Policy"),
    "pi05": ("lerobot_mlx.policies.pi05", "Pi05Policy"),
    "pi0_fast": ("lerobot_mlx.policies.pi0_fast", "Pi0FastPolicy"),
    "smolvla": ("lerobot_mlx.policies.smolvla", "SmolVLAPolicy"),
}

_CONFIG_REGISTRY = {
    "act": ("lerobot_mlx.policies.act", "ACTConfig"),
    "diffusion": ("lerobot_mlx.policies.diffusion", "DiffusionConfig"),
    "sac": ("lerobot_mlx.policies.sac", "SACConfig"),
    "tdmpc": ("lerobot_mlx.policies.tdmpc", "TDMPCConfig"),
    "vqbet": ("lerobot_mlx.policies.vqbet", "VQBeTConfig"),
    "sarm": ("lerobot_mlx.policies.sarm", "SARMConfig"),
    "pi0": ("lerobot_mlx.policies.pi0", "Pi0Config"),
    "pi05": ("lerobot_mlx.policies.pi05", "Pi05Config"),
    "pi0_fast": ("lerobot_mlx.policies.pi0_fast", "Pi0FastConfig"),
    "smolvla": ("lerobot_mlx.policies.smolvla", "SmolVLAConfig"),
}


def make_policy(config, **kwargs):
    """Create a policy instance from config.

    Args:
        config: Policy config (dataclass with type info) or string policy type name.
        **kwargs: Additional arguments passed to policy constructor.

    Returns:
        Instantiated policy.
    """
    if isinstance(config, str):
        policy_type = config
        # Create default config
        config_cls = get_config_class(policy_type)
        config = config_cls()
    else:
        policy_type = _detect_type(config)

    policy_cls = get_policy_class(policy_type)
    return policy_cls(config, **kwargs)


def get_policy_class(policy_type: str):
    """Get policy class by type name (lazy import)."""
    if policy_type not in _POLICY_REGISTRY:
        available = list(_POLICY_REGISTRY.keys())
        raise ValueError(f"Unknown policy type: {policy_type!r}. Available: {available}")

    module_path, class_name = _POLICY_REGISTRY[policy_type]
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {class_name} from {module_path}: {e}") from e


def get_config_class(policy_type: str):
    """Get config class by type name (lazy import)."""
    if policy_type not in _CONFIG_REGISTRY:
        available = list(_CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown policy type: {policy_type!r}. Available: {available}")

    module_path, class_name = _CONFIG_REGISTRY[policy_type]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def register_policy(name: str, module_path: str, policy_class_name: str,
                    config_class_name: Optional[str] = None):
    """Register a new policy type.

    Args:
        name: Short name for the policy type (e.g., "tdmpc").
        module_path: Python module path (e.g., "lerobot_mlx.policies.tdmpc").
        policy_class_name: Name of the policy class in the module.
        config_class_name: Name of the config class in the module (optional).
    """
    _POLICY_REGISTRY[name] = (module_path, policy_class_name)
    if config_class_name:
        _CONFIG_REGISTRY[name] = (module_path, config_class_name)


def available_policies():
    """List all registered policy types."""
    return list(_POLICY_REGISTRY.keys())


def _detect_type(config):
    """Detect policy type from config object.

    Matches registry keys against the lowercase class name. Longer keys
    are checked first so that e.g. ``pi0_fast`` is matched before ``pi0``.
    """
    cls_name = type(config).__name__.lower()
    # Sort keys by length descending so more specific names match first
    # (e.g. "pi0_fast" before "pi0", "pi05" before "pi0")
    for name in sorted(_POLICY_REGISTRY, key=len, reverse=True):
        # Normalise: strip underscores for comparison (pi0_fast -> pi0fast)
        normalised_name = name.replace("_", "")
        if normalised_name in cls_name:
            return name
    if hasattr(config, "type"):
        return config.type
    if hasattr(config, "policy_type"):
        return config.policy_type
    raise ValueError(f"Cannot detect policy type from config: {type(config).__name__}")
