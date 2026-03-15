"""Test that all examples run without errors."""

import subprocess
import pytest

EXAMPLES = [
    "examples/01_quickstart.py",
    "examples/02_train_act_synthetic.py",
    "examples/03_train_diffusion_synthetic.py",
    "examples/04_policy_factory.py",
    "examples/05_compare_policies.py",
    "examples/06_compat_layer_demo.py",
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_runs(example):
    result = subprocess.run(
        [".venv/bin/python", example],
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/Users/ilessio/Development/AIFLOWLABS/R&D/LeRobot-mlx",
    )
    assert result.returncode == 0, f"{example} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
