# LeRobot-MLX

Port of HuggingFace LeRobot (v0.5.1) from PyTorch to Apple MLX for native Apple Silicon robotics ML.

See `.claude/CLAUDE.md` for detailed architecture, design rules, and build order.

## Quick Reference

```bash
# Setup
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

# Test
pytest tests/ -v

# Lint
ruff check src/
ruff format src/
```

## Conventions
- Python 3.12+, `uv` as package manager
- Use `rg` (ripgrep) instead of `grep`
- Never modify `repositories/lerobot-upstream/` — read-only reference
- Mirror upstream structure in `src/lerobot_mlx/`
- Route all torch→mlx through `compat/` layer

# currentDate
Today's date is 2026-03-15.
