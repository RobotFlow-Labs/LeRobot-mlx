# PRD-07: MLX Training Loop

> **Status:** TODO
> **Priority:** P0 — Required for any training
> **Dependencies:** PRD-02, PRD-03 (compat optim), PRD-05 or PRD-06 (a policy to train)
> **Estimated LOC:** ~400
> **Phase:** 2 (First Policy)

---

## Objective

Replace the upstream's `accelerate`-based training loop with MLX-native training using `mlx.nn.value_and_grad`. This is the core training infrastructure that all policies share.

---

## What Upstream Uses (that we replace)

| Upstream | Our Replacement | Why |
|----------|----------------|-----|
| `accelerate.Accelerator` | Not needed | MLX = single device, unified memory |
| `accelerate.prepare(model, optimizer, dataloader)` | Direct use | No wrapping needed |
| `accelerator.backward(loss)` | `nn.value_and_grad(model, loss_fn)` | MLX autograd |
| `accelerator.clip_grad_norm_()` | `compat.optim.clip_grad_norm_()` | Custom impl |
| `torch.cuda.amp` | Not needed | MLX handles precision natively |
| `accelerator.save_state()` | `mx.save()` / `safetensors` | Direct save |
| `torch.utils.data.DataLoader` | Python iterator / custom | Simple batching |

---

## Training Loop Architecture

```python
# src/lerobot_mlx/training/trainer.py

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from lerobot_mlx.compat.optim import get_scheduler, clip_grad_norm_
import time

class Trainer:
    """MLX-native training loop replacing accelerate."""

    def __init__(self, policy, config):
        self.policy = policy
        self.config = config

        # Optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )

        # LR Scheduler
        self.lr_scheduler = get_scheduler(
            config.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.training_steps,
        )

        # Create loss+grad function
        self._loss_and_grad_fn = nn.value_and_grad(policy, self._compute_loss)

    def _compute_loss(self, model, batch):
        """Compute training loss. Policy-agnostic — calls model.compute_loss()."""
        output = model.forward(batch)
        return output['loss']

    def train_step(self, batch):
        """Single training step: forward + backward + optimizer step."""
        # Forward + backward
        loss, grads = self._loss_and_grad_fn(self.policy, batch)

        # Gradient clipping
        if self.config.max_grad_norm is not None:
            grads = _clip_grads(grads, self.config.max_grad_norm)

        # Optimizer step
        self.optimizer.update(self.policy, grads)

        # LR scheduler step
        self.lr_scheduler.step()

        # CRITICAL: materialize computation
        mx.eval(self.policy.parameters(), self.optimizer.state, loss)

        return {'loss': loss.item(), 'lr': self.optimizer.learning_rate}

    def train(self, dataloader, num_steps):
        """Full training loop."""
        self.policy.train()
        step = 0
        metrics = []

        for batch in dataloader:
            if step >= num_steps:
                break

            step_metrics = self.train_step(batch)
            step_metrics['step'] = step
            metrics.append(step_metrics)

            if step % self.config.log_interval == 0:
                print(f"Step {step}: loss={step_metrics['loss']:.4f}, "
                      f"lr={step_metrics['lr']:.2e}")

            if step % self.config.save_interval == 0 and step > 0:
                self.save_checkpoint(step)

            step += 1

        return metrics

    def save_checkpoint(self, step):
        """Save model + optimizer state."""
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(self.policy.parameters()))
        mx.savez(f"checkpoints/step_{step}.npz", **weights)

    def load_checkpoint(self, path):
        """Load model weights from checkpoint."""
        weights = mx.load(path)
        self.policy.load_weights(list(weights.items()))

def _clip_grads(grads, max_norm):
    """Clip gradient norms (functional, returns new grads)."""
    from mlx.utils import tree_flatten, tree_unflatten
    flat_grads = tree_flatten(grads)
    total_norm_sq = sum(mx.sum(g ** 2).item() for _, g in flat_grads)
    total_norm = total_norm_sq ** 0.5
    clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
    if clip_coef < 1.0:
        clipped = [(k, g * clip_coef) for k, g in flat_grads]
        return tree_unflatten(clipped)
    return grads
```

---

## EMA (Exponential Moving Average)

Used by Diffusion Policy for stable inference:

```python
class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        from mlx.utils import tree_flatten, tree_unflatten
        self.decay = decay
        # Deep copy parameters
        self.shadow = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}

    def update(self, model):
        from mlx.utils import tree_flatten
        for k, v in tree_flatten(model.parameters()):
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model):
        """Load EMA weights into model."""
        model.load_weights(list(self.shadow.items()))
```

---

## Acceptance Criteria

1. `Trainer(policy, config).train_step(batch)` returns loss dict
2. Loss decreases over 50 steps on synthetic ACT data
3. Gradient clipping works (grads are bounded)
4. LR scheduler updates learning rate correctly
5. Checkpoint save/load roundtrip preserves weights
6. EMA tracking works (shadow weights diverge from model weights)
7. `mx.eval()` is called after every step (no lazy graph buildup)
8. 15+ tests passing
