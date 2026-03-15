# PRD-14: CLI Scripts (train + eval)

> **Status:** TODO
> **Priority:** P1 — User-facing entry points
> **Dependencies:** PRD-07 (training), PRD-08 (datasets), PRD-13 (factory)
> **Estimated LOC:** ~300
> **Phase:** 4 (Infrastructure)

---

## Objective

Create CLI entry points `lerobot-mlx-train` and `lerobot-mlx-eval` that mirror upstream's `lerobot-train` and `lerobot-eval` interfaces. Uses draccus for config parsing (same as upstream).

---

## CLI Interface

```bash
# Training
lerobot-mlx-train \
  --policy.type=act \
  --policy.chunk_size=100 \
  --training.lr=1e-4 \
  --training.steps=100000 \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/act_pusht

# Evaluation
lerobot-mlx-eval \
  --policy.type=act \
  --pretrained_path=outputs/act_pusht/checkpoints/best \
  --dataset.repo_id=lerobot/pusht \
  --eval_episodes=50

# Inference
lerobot-mlx-infer \
  --pretrained_path=lerobot/act_pusht \
  --robot=so100
```

---

## Deliverables

### `scripts/train.py`
```python
import draccus
from lerobot_mlx.policies.factory import make_policy
from lerobot_mlx.training.trainer import Trainer
from lerobot_mlx.datasets.lerobot_dataset import LeRobotDatasetMLX
from lerobot_mlx.datasets.dataloader import SimpleDataLoader

@draccus.wrap()
def main(config):
    # Create dataset
    dataset = LeRobotDatasetMLX(config.dataset.repo_id)
    dataloader = SimpleDataLoader(dataset, batch_size=config.training.batch_size)

    # Create policy
    policy = make_policy(config.policy)

    # Train
    trainer = Trainer(policy, config.training)
    trainer.train(dataloader, num_steps=config.training.steps)

    # Save
    trainer.save_checkpoint(config.training.steps)

if __name__ == "__main__":
    main()
```

### `scripts/eval.py`
```python
@draccus.wrap()
def main(config):
    # Load pretrained policy
    policy = load_pretrained(config.pretrained_path)
    policy.eval()

    # Load eval dataset
    dataset = LeRobotDatasetMLX(config.dataset.repo_id, split="test")

    # Run evaluation
    metrics = evaluate(policy, dataset, num_episodes=config.eval_episodes)
    print(f"Success rate: {metrics['success_rate']:.2%}")
```

---

## Acceptance Criteria

1. `lerobot-mlx-train --help` shows config options
2. `lerobot-mlx-train` with synthetic data runs training loop
3. `lerobot-mlx-eval` with pretrained checkpoint runs evaluation
4. Config system matches upstream (draccus-based)
5. Output directory structure matches upstream convention
6. 10+ tests passing
