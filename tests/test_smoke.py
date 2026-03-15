"""Smoke tests — verify package imports, MLX environment, and scaffold structure."""

import importlib


class TestPackageImports:
    """Verify that the package and all submodules import cleanly."""

    def test_package_root(self):
        import lerobot_mlx
        assert hasattr(lerobot_mlx, "__version__")

    def test_version(self):
        from lerobot_mlx._version import __version__
        assert __version__ == "0.1.0"

    def test_version_from_package(self):
        import lerobot_mlx
        assert lerobot_mlx.__version__ == "0.1.0"

    def test_compat_module(self):
        from lerobot_mlx import compat
        assert hasattr(compat, "nn")
        assert hasattr(compat, "tensor_ops")
        assert hasattr(compat, "Tensor")

    def test_compat_tensor_ops(self):
        from lerobot_mlx.compat import tensor_ops
        assert hasattr(tensor_ops, "zeros")
        assert hasattr(tensor_ops, "ones")
        assert hasattr(tensor_ops, "cat")
        assert hasattr(tensor_ops, "stack")

    def test_compat_nn_modules(self):
        from lerobot_mlx.compat import nn_modules
        assert hasattr(nn_modules, "Module")

    def test_compat_nn_layers(self):
        from lerobot_mlx.compat import nn_layers
        assert hasattr(nn_layers, "Linear")
        assert hasattr(nn_layers, "Conv2d")
        assert hasattr(nn_layers, "MultiheadAttention")

    def test_compat_functional(self):
        from lerobot_mlx.compat import functional
        assert hasattr(functional, "relu")

    def test_compat_optim(self):
        from lerobot_mlx.compat import optim
        assert hasattr(optim, "Adam")

    def test_compat_distributions(self):
        from lerobot_mlx.compat import distributions
        assert hasattr(distributions, "Normal")

    def test_compat_einops_mlx(self):
        from lerobot_mlx.compat import einops_mlx
        assert hasattr(einops_mlx, "rearrange")

    def test_compat_vision(self):
        from lerobot_mlx.compat import vision
        assert hasattr(vision, "ResNet")

    def test_compat_diffusers_mlx(self):
        from lerobot_mlx.compat import diffusers_mlx
        assert diffusers_mlx is not None

    def test_policies_module(self):
        from lerobot_mlx import policies
        assert policies is not None

    def test_model_module(self):
        from lerobot_mlx import model
        assert model is not None

    def test_datasets_module(self):
        from lerobot_mlx import datasets
        assert datasets is not None

    def test_processor_module(self):
        from lerobot_mlx import processor
        assert processor is not None

    def test_training_module(self):
        from lerobot_mlx import training
        assert training is not None

    def test_configs_module(self):
        from lerobot_mlx import configs
        assert configs is not None

    def test_scripts_module(self):
        from lerobot_mlx import scripts
        assert scripts is not None


class TestMLXEnvironment:
    """Verify MLX is available and functional."""

    def test_mlx_importable(self):
        import mlx.core as mx
        assert hasattr(mx, "array")

    def test_mlx_array_creation(self):
        import mlx.core as mx
        x = mx.ones((2, 3))
        assert x.shape == (2, 3)

    def test_metal_available(self):
        import mlx.core as mx
        assert mx.metal.is_available()

    def test_mlx_version_accessible(self):
        import importlib.metadata
        version = importlib.metadata.version("mlx")
        assert version is not None
        parts = version.split(".")
        assert len(parts) >= 2


class TestScaffoldStructure:
    """Verify all expected modules exist and are importable."""

    EXPECTED_MODULES = [
        "lerobot_mlx",
        "lerobot_mlx._version",
        "lerobot_mlx.compat",
        "lerobot_mlx.compat.tensor_ops",
        "lerobot_mlx.compat.nn_modules",
        "lerobot_mlx.compat.nn_layers",
        "lerobot_mlx.compat.functional",
        "lerobot_mlx.compat.optim",
        "lerobot_mlx.compat.distributions",
        "lerobot_mlx.compat.einops_mlx",
        "lerobot_mlx.compat.vision",
        "lerobot_mlx.compat.diffusers_mlx",
        "lerobot_mlx.policies",
        "lerobot_mlx.model",
        "lerobot_mlx.datasets",
        "lerobot_mlx.processor",
        "lerobot_mlx.training",
        "lerobot_mlx.configs",
        "lerobot_mlx.scripts",
        "lerobot_mlx.scripts.train",
        "lerobot_mlx.scripts.eval",
    ]

    def test_all_modules_importable(self):
        for module_name in self.EXPECTED_MODULES:
            mod = importlib.import_module(module_name)
            assert mod is not None, f"Failed to import {module_name}"
