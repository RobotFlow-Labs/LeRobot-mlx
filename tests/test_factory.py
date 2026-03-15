"""Tests for policy factory & registry (PRD-13).

Tests cover:
- make_policy from string and config
- get_policy_class / get_config_class for all registered types
- Error handling for unknown types
- Custom registration
- Lazy import behavior
- Policy type detection from config objects
"""

import sys

import pytest

from lerobot_mlx.policies.factory import (
    _CONFIG_REGISTRY,
    _POLICY_REGISTRY,
    _detect_type,
    available_policies,
    get_config_class,
    get_policy_class,
    make_policy,
    register_policy,
)


# ---------------------------------------------------------------------------
# make_policy
# ---------------------------------------------------------------------------


def _make_act_config():
    """Create a valid ACT config with required features."""
    from lerobot_mlx.policies.act.configuration_act import (
        ACTConfig, FeatureType, PolicyFeature,
    )
    return ACTConfig(
        chunk_size=10,
        n_action_steps=10,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        latent_dim=8,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(14,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        },
    )


def _make_diffusion_config():
    """Create a valid Diffusion config with required features."""
    from lerobot_mlx.policies.diffusion.configuration_diffusion import (
        DiffusionConfig, PolicyFeature,
    )
    return DiffusionConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        down_dims=(64, 128, 256),
        num_train_timesteps=10,
        diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8,
        n_groups=4,
        kernel_size=3,
        use_group_norm=True,
        use_film_scale_modulation=True,
        input_features={
            "observation.state": PolicyFeature(type="STATE", shape=(2,)),
            "observation.environment_state": PolicyFeature(type="ENV_STATE", shape=(4,)),
        },
        output_features={
            "action": PolicyFeature(type="ACTION", shape=(2,)),
        },
    )


def _make_sac_config():
    """Create a valid SAC config with required features."""
    from lerobot_mlx.policies.sac.configuration_sac import (
        SACConfig, FeatureShape,
    )
    return SACConfig(
        input_features={"observation.state": FeatureShape(shape=(10,))},
        output_features={"action": FeatureShape(shape=(4,))},
        latent_dim=64,
        num_critics=2,
        temperature_init=1.0,
        discount=0.99,
        critic_target_update_weight=0.005,
        use_backup_entropy=True,
    )


class TestMakePolicy:
    def test_make_policy_from_config_act(self):
        """make_policy(act_config) returns an ACTPolicy instance."""
        from lerobot_mlx.policies.act import ACTPolicy

        config = _make_act_config()
        policy = make_policy(config)
        assert isinstance(policy, ACTPolicy)

    def test_make_policy_from_config_diffusion(self):
        """make_policy(diffusion_config) returns a DiffusionPolicy."""
        from lerobot_mlx.policies.diffusion import DiffusionPolicy

        config = _make_diffusion_config()
        policy = make_policy(config)
        assert isinstance(policy, DiffusionPolicy)

    def test_make_policy_from_config_sac(self):
        """make_policy(sac_config) returns a SACPolicy."""
        from lerobot_mlx.policies.sac import SACPolicy

        config = _make_sac_config()
        policy = make_policy(config)
        assert isinstance(policy, SACPolicy)

    def test_make_policy_unknown_raises(self):
        """Unknown policy type string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown policy type"):
            make_policy("nonexistent_policy")

    def test_make_policy_returns_correct_class_for_act(self):
        """Verify that make_policy resolves the right class for ACT config."""
        from lerobot_mlx.policies.act import ACTPolicy

        cls = get_policy_class("act")
        assert cls is ACTPolicy


# ---------------------------------------------------------------------------
# get_policy_class
# ---------------------------------------------------------------------------


class TestGetPolicyClass:
    def test_get_policy_class_act(self):
        from lerobot_mlx.policies.act import ACTPolicy

        cls = get_policy_class("act")
        assert cls is ACTPolicy

    def test_get_policy_class_diffusion(self):
        from lerobot_mlx.policies.diffusion import DiffusionPolicy

        cls = get_policy_class("diffusion")
        assert cls is DiffusionPolicy

    def test_get_policy_class_sac(self):
        from lerobot_mlx.policies.sac import SACPolicy

        cls = get_policy_class("sac")
        assert cls is SACPolicy

    def test_get_policy_class_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown policy type"):
            get_policy_class("imaginary")


# ---------------------------------------------------------------------------
# get_config_class
# ---------------------------------------------------------------------------


class TestGetConfigClass:
    def test_get_config_class_act(self):
        from lerobot_mlx.policies.act import ACTConfig

        cls = get_config_class("act")
        assert cls is ACTConfig

    def test_get_config_class_diffusion(self):
        from lerobot_mlx.policies.diffusion import DiffusionConfig

        cls = get_config_class("diffusion")
        assert cls is DiffusionConfig

    def test_get_config_class_sac(self):
        from lerobot_mlx.policies.sac import SACConfig

        cls = get_config_class("sac")
        assert cls is SACConfig

    def test_get_config_class_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown policy type"):
            get_config_class("imaginary")


# ---------------------------------------------------------------------------
# available_policies
# ---------------------------------------------------------------------------


class TestAvailablePolicies:
    def test_available_policies_returns_list(self):
        result = available_policies()
        assert isinstance(result, list)
        assert "act" in result
        assert "diffusion" in result
        assert "sac" in result

    def test_available_policies_length(self):
        result = available_policies()
        assert len(result) >= 3


# ---------------------------------------------------------------------------
# register_policy
# ---------------------------------------------------------------------------


class TestRegisterPolicy:
    def test_register_policy_adds_to_registry(self):
        register_policy("test_policy", "some.module", "TestPolicy", "TestConfig")
        assert "test_policy" in _POLICY_REGISTRY
        assert _POLICY_REGISTRY["test_policy"] == ("some.module", "TestPolicy")
        assert "test_policy" in _CONFIG_REGISTRY
        assert _CONFIG_REGISTRY["test_policy"] == ("some.module", "TestConfig")
        # Cleanup
        del _POLICY_REGISTRY["test_policy"]
        del _CONFIG_REGISTRY["test_policy"]

    def test_register_policy_without_config(self):
        register_policy("test_policy2", "some.module", "TestPolicy2")
        assert "test_policy2" in _POLICY_REGISTRY
        assert "test_policy2" not in _CONFIG_REGISTRY
        # Cleanup
        del _POLICY_REGISTRY["test_policy2"]

    def test_register_policy_overrides_existing(self):
        """Re-registering a policy type replaces the old entry."""
        register_policy("act", "custom.module", "CustomACT", "CustomACTConfig")
        assert _POLICY_REGISTRY["act"] == ("custom.module", "CustomACT")
        # Restore original
        _POLICY_REGISTRY["act"] = ("lerobot_mlx.policies.act", "ACTPolicy")
        _CONFIG_REGISTRY["act"] = ("lerobot_mlx.policies.act", "ACTConfig")


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_lazy_import_no_policy_modules_loaded_at_import(self):
        """Importing factory does not eagerly load policy modules."""
        # The factory module uses string-based lazy registry,
        # so importing factory.py should not import the heavy modeling modules.
        # We verify by checking that the registry stores strings, not classes.
        for key, value in _POLICY_REGISTRY.items():
            assert isinstance(value, tuple), f"Registry entry for {key} should be a (module, class) tuple"
            assert isinstance(value[0], str), f"Module path for {key} should be a string"
            assert isinstance(value[1], str), f"Class name for {key} should be a string"


# ---------------------------------------------------------------------------
# _detect_type
# ---------------------------------------------------------------------------


class TestDetectType:
    def test_detect_type_from_config_classname(self):
        from lerobot_mlx.policies.act import ACTConfig

        config = ACTConfig()
        assert _detect_type(config) == "act"

    def test_detect_type_diffusion(self):
        from lerobot_mlx.policies.diffusion import DiffusionConfig

        config = DiffusionConfig()
        assert _detect_type(config) == "diffusion"

    def test_detect_type_sac(self):
        from lerobot_mlx.policies.sac import SACConfig

        config = SACConfig()
        assert _detect_type(config) == "sac"

    def test_detect_type_unknown_raises(self):
        class UnknownConfig:
            pass

        with pytest.raises(ValueError, match="Cannot detect policy type"):
            _detect_type(UnknownConfig())
