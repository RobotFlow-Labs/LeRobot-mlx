"""Shared test fixtures and configuration for LeRobot-MLX tests."""

import platform

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "requires_torch: marks tests that need PyTorch for cross-framework validation")


@pytest.fixture(autouse=True)
def _check_platform():
    """Skip tests on non-Apple Silicon platforms."""
    if platform.machine() != "arm64":
        pytest.skip("Requires Apple Silicon (arm64)")


@pytest.fixture
def mx():
    """Provide mlx.core module as fixture."""
    import mlx.core as mx
    return mx
