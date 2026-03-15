"""SmolVLA policy configuration — MLX port.

Mirrors upstream lerobot.policies.smolvla.configuration_smolvla.SmolVLAConfig.
Decoupled from upstream PreTrainedConfig / draccus to avoid torch dependencies.
Uses simple dataclass with equivalent fields and validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Lightweight types (mirrors upstream lerobot.configs.types)
# ---------------------------------------------------------------------------

class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


# ---------------------------------------------------------------------------
# String constants (mirrors upstream lerobot.utils.constants)
# ---------------------------------------------------------------------------

OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
ACTION = "action"


# ---------------------------------------------------------------------------
# SmolVLAConfig
# ---------------------------------------------------------------------------

@dataclass
class SmolVLAConfig:
    """Configuration class for the SmolVLA policy.

    SmolVLA = SmolVLM backbone + action expert + flow matching.

    Mirrors upstream SmolVLAConfig fields relevant to the MLX port.
    VLM backbone loading is optional — standalone operation supported.
    """

    # VLM backbone
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    vlm_hidden_size: int = 2048
    num_vlm_layers: int = 16

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # State/action padding
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Expert architecture
    expert_hidden_size: int = 768
    expert_intermediate_size: int = 3072
    expert_num_heads: int = 12
    expert_num_layers: int = 12
    expert_head_dim: int = 64
    expert_width_multiplier: float = 0.75

    # Decoding / flow matching
    num_inference_steps: int = 10

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Training
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True

    # Time encoding
    min_period: float = 4e-3
    max_period: float = 4.0

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Features
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    def __post_init__(self):
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

    @property
    def effective_expert_hidden_size(self) -> int:
        """The actual hidden size of the expert after applying width_multiplier."""
        return int(self.expert_hidden_size * self.expert_width_multiplier)
