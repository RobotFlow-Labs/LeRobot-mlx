# PRD-21: CLI & User Experience Polish

> **Status:** TODO
> **Priority:** P1 — Makes the package user-friendly
> **Dependencies:** All policies + factory + pretrained
> **Estimated LOC:** ~500

---

## Objective

Make LeRobot-MLX a joy to use. Full CLI, helpful error messages, rich terminal output, getting-started guide.

---

## Deliverables

### 1. `src/lerobot_mlx/scripts/train.py` — Full CLI (upgrade from stub)

```bash
lerobot-mlx-train \
  --policy act \
  --dataset synthetic \
  --batch-size 32 \
  --steps 10000 \
  --lr 1e-4 \
  --output-dir outputs/act_run1

# With real data:
lerobot-mlx-train \
  --policy diffusion \
  --dataset lerobot/pusht \
  --batch-size 16 \
  --steps 50000
```

Features:
- Rich progress bar (tqdm or custom)
- Live loss display
- Checkpoint auto-save
- Config save/load (JSON)
- Resume from checkpoint

### 2. `src/lerobot_mlx/scripts/eval.py` — Full evaluation CLI

```bash
lerobot-mlx-eval \
  --checkpoint outputs/act_run1/checkpoint_10000.npz \
  --dataset lerobot/pusht \
  --episodes 50
```

### 3. `src/lerobot_mlx/scripts/info.py` — System info + package status

```bash
lerobot-mlx-info
```

Output:
```
LeRobot-MLX v0.1.0
Platform: macOS 15.3 (Apple M3 Max)
MLX: 0.31.1, Metal: available
Python: 3.12.12

Available Policies:
  act       ✓ (41 tests)
  diffusion ✓ (38 tests)
  sac       ✓ (37 tests)
  tdmpc     ✓ (31 tests)
  vqbet     ✓ (38 tests)
  sarm      ✓ (33 tests)
  pi0       ✓ (48 tests)
  smolvla   ✓ (34 tests)

VLM Backends: mlx-vlm (45 models)
Memory: 64GB unified, 23.5GB available
```

### 4. `src/lerobot_mlx/scripts/convert.py` — Weight conversion CLI

```bash
lerobot-mlx-convert \
  --repo-id lerobot/act_aloha_sim \
  --output-dir converted/act_aloha
```

### 5. `src/lerobot_mlx/scripts/benchmark.py` — Quick benchmark CLI

```bash
lerobot-mlx-benchmark --policy act --batch-size 1
```

### 6. Better error messages everywhere

- Import errors: suggest `uv pip install -e ".[dev]"`
- Missing MLX: explain Apple Silicon requirement
- Wrong shapes: show expected vs actual
- Missing keys in weight loading: list what's missing

### 7. `GETTING_STARTED.md` — Friendly walkthrough

Step-by-step guide for new users:
1. Install
2. Run first example
3. Train your first policy
4. Load pretrained weights
5. Customize a policy

---

## Acceptance Criteria

1. All CLI commands work with `--help`
2. `lerobot-mlx-info` prints correct system info
3. `lerobot-mlx-train --policy act --steps 10` completes
4. `lerobot-mlx-benchmark --policy act` produces timing output
5. Error messages are helpful and actionable
6. GETTING_STARTED.md is complete and tested
7. 10+ tests for CLI tools
