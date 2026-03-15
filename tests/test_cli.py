"""Tests for LeRobot-MLX CLI tools."""
import subprocess
import pytest

PROJECT_DIR = "/Users/ilessio/Development/AIFLOWLABS/R&D/LeRobot-mlx"
PYTHON = f"{PROJECT_DIR}/.venv/bin/python"


def _run_cli(cmd, timeout=30):
    """Run a CLI command and return the result."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_DIR,
    )
    return result


class TestInfoCLI:
    """Tests for lerobot-mlx-info."""

    def test_info_runs(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.info"])
        assert r.returncode == 0
        assert "LeRobot-MLX" in r.stdout

    def test_info_shows_policies(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.info"])
        assert r.returncode == 0
        assert "act" in r.stdout
        assert "diffusion" in r.stdout
        assert "pi0" in r.stdout
        assert "smolvla" in r.stdout

    def test_info_shows_mlx(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.info"])
        assert r.returncode == 0
        assert "MLX" in r.stdout

    def test_info_shows_platform(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.info"])
        assert r.returncode == 0
        assert "Platform" in r.stdout
        assert "Python" in r.stdout

    def test_info_shows_memory(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.info"])
        assert r.returncode == 0
        # Should show either MLX Memory or system memory
        assert "Memory" in r.stdout or "memory" in r.stdout

    def test_info_shows_policy_count(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.info"])
        assert r.returncode == 0
        assert "8 available" in r.stdout

    def test_info_shows_vlm_backend(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.info"])
        assert r.returncode == 0
        assert "VLM Backend" in r.stdout


class TestConvertCLI:
    """Tests for lerobot-mlx-convert."""

    def test_convert_help(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.convert", "--help"])
        assert r.returncode == 0
        assert "repo-id" in r.stdout

    def test_convert_requires_repo_id(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.convert"])
        assert r.returncode != 0
        assert "repo-id" in r.stderr


class TestBenchmarkCLI:
    """Tests for lerobot-mlx-benchmark."""

    def test_benchmark_help(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.benchmark", "--help"])
        assert r.returncode == 0
        assert "policy" in r.stdout
        assert "batch-size" in r.stdout


class TestTrainCLI:
    """Tests for lerobot-mlx-train."""

    def test_train_help(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.train", "--help"])
        assert r.returncode == 0
        assert "policy-type" in r.stdout


class TestEvalCLI:
    """Tests for lerobot-mlx-eval."""

    def test_eval_help(self):
        r = _run_cli([PYTHON, "-m", "lerobot_mlx.scripts.eval", "--help"])
        assert r.returncode == 0
        assert "checkpoint" in r.stdout
