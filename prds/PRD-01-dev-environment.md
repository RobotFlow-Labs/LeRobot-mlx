# PRD-01: Dev Environment & Project Scaffold

> **Status:** TODO
> **Priority:** P0 — Foundation, everything depends on this
> **Dependencies:** None
> **Estimated LOC:** ~200 (config + scaffold + smoke tests)
> **Phase:** 1 (Foundation)

---

## Objective

Set up the `lerobot-mlx` Python package with proper structure, dependencies, test harness, and CI-ready configuration. After this PRD, `uv pip install -e ".[dev]"` works and `pytest tests/test_smoke.py` passes.

---

## Deliverables

### 1. `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lerobot-mlx"
version = "0.1.0"
description = "LeRobot ported to Apple MLX for native Apple Silicon robotics"
requires-python = ">=3.12"
license = { text = "Apache-2.0" }
authors = [
    { name = "AIFLOW LABS", email = "ilessio@aiflowlabs.io" },
]

dependencies = [
    "mlx>=0.31.0",
    "numpy>=2.0.0",
    "scipy>=1.14.0",
    "huggingface-hub>=1.0.0",
    "safetensors>=0.4.0",
    "draccus==0.10.0",
    "pillow>=10.0.0",
    "opencv-python>=4.8.0",
    "jsonlines>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
    "ruff>=0.4.0",
]
# Cross-framework validation (optional, only for comparison tests)
torch = [
    "torch>=2.6.0",
    "torchvision>=0.21.0",
    "einops>=0.8.0",
]

[project.scripts]
lerobot-mlx-train = "lerobot_mlx.scripts.train:main"
lerobot-mlx-eval = "lerobot_mlx.scripts.eval:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "requires_torch: marks tests that need PyTorch for cross-framework validation",
]

[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

### 2. Package Structure (empty scaffolds)

```
src/lerobot_mlx/
├── __init__.py              # Version + environment check
├── _version.py              # __version__ = "0.1.0"
├── compat/
│   ├── __init__.py          # Re-export all compat modules
│   ├── tensor_ops.py        # Stub
│   ├── nn_modules.py        # Stub
│   ├── nn_layers.py         # Stub
│   ├── functional.py        # Stub
│   ├── optim.py             # Stub
│   ├── distributions.py     # Stub
│   ├── einops_mlx.py        # Stub
│   ├── vision.py            # Stub
│   └── diffusers_mlx.py     # Stub
├── policies/
│   └── __init__.py
├── model/
│   └── __init__.py
├── datasets/
│   └── __init__.py
├── processor/
│   └── __init__.py
├── training/
│   └── __init__.py
├── configs/
│   └── __init__.py
└── scripts/
    ├── __init__.py
    ├── train.py             # Stub CLI
    └── eval.py              # Stub CLI
```

### 3. Test Harness

```
tests/
├── conftest.py              # Shared fixtures, platform checks
├── test_smoke.py            # Package import, MLX env, scaffold verification
└── helpers.py               # check_all_close, tolerance constants
```

### 4. `UPSTREAM_VERSION.md`

```markdown
# Upstream Sync Status

## Current Sync Target
- **Repository:** https://github.com/huggingface/lerobot
- **Version:** v0.5.1
- **Commit:** (fill from repositories/lerobot-upstream HEAD)
- **Date:** 2026-03-15

## Ported Components
- [ ] compat/ layer
- [ ] ACT policy
- [ ] Diffusion policy
- [ ] TD-MPC policy
- [ ] VQ-BeT policy
- [ ] SAC policy
- [ ] Datasets
- [ ] Training loop
- [ ] Processor pipeline

## Sync History
| Date | From | To | Notes |
|------|------|----|-------|
| 2026-03-15 | — | v0.5.1 | Initial port target |
```

---

## Acceptance Criteria

1. `uv venv .venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[dev]"` succeeds
2. `python -c "import lerobot_mlx; print(lerobot_mlx.__version__)"` prints `0.1.0`
3. `python -c "import mlx.core as mx; print(mx.metal.is_available())"` prints `True`
4. `pytest tests/test_smoke.py -v` — all tests pass
5. Every `__init__.py` in `compat/` is importable without error
6. `ruff check src/` passes clean
7. `UPSTREAM_VERSION.md` exists with correct upstream commit hash

---

## Smoke Tests (test_smoke.py)

```python
def test_package_imports():
    """Package and all submodules import cleanly."""
    import lerobot_mlx
    from lerobot_mlx import compat
    from lerobot_mlx.compat import tensor_ops, nn_modules, nn_layers
    from lerobot_mlx.compat import functional, optim, distributions
    from lerobot_mlx.compat import einops_mlx, vision, diffusers_mlx

def test_version():
    from lerobot_mlx._version import __version__
    assert __version__ == "0.1.0"

def test_mlx_available():
    import mlx.core as mx
    assert hasattr(mx, 'array')
    x = mx.ones((2, 3))
    assert x.shape == (2, 3)

def test_metal_available():
    import mlx.core as mx
    # On Apple Silicon this should be True
    assert mx.metal.is_available()

def test_compat_stubs_importable():
    """All compat modules exist and import without error."""
    from lerobot_mlx.compat import tensor_ops
    from lerobot_mlx.compat import nn_modules
    # ... etc
```

---

## conftest.py Pattern

```python
import platform
import pytest

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: slow test")
    config.addinivalue_line("markers", "requires_torch: needs PyTorch")

@pytest.fixture(autouse=True)
def _check_platform():
    if platform.machine() != "arm64":
        pytest.skip("Requires Apple Silicon (arm64)")

@pytest.fixture
def mx():
    import mlx.core as mx
    return mx
```

---

## Notes

- **No compilation step** — pure Python + MLX Metal backend
- All compat/ modules start as stubs (empty or minimal). PRD-02 fills them.
- The `torch` extra is OPTIONAL — only needed for cross-framework validation tests
- Use `uv` exclusively, never `pip` directly
