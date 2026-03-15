# PRD-11: SAC Policy (Soft Actor-Critic)

> **Status:** TODO
> **Priority:** P1 — Simple RL baseline
> **Dependencies:** PRD-02, PRD-03 (especially distributions.Beta)
> **Estimated LOC:** ~400 (mirror of upstream's ~1872 LOC)
> **Phase:** 4 (Remaining Policies)

---

## Objective

Port SAC — Soft Actor-Critic, a standard model-free RL policy with twin Q-networks and entropy regularization.

---

## Upstream Files

| File | Treatment |
|------|-----------|
| `policies/sac/configuration_sac.py` | **COPY VERBATIM** |
| `policies/sac/modeling_sac.py` | **PORT** |

---

## Architecture

```
SAC:
  Actor (π):
    ├── MLP encoder
    └── Output: distribution parameters (mean, log_std or Beta params)

  Twin Critics (Q1, Q2):
    ├── MLP encoder
    └── Output: Q-value scalar

  Target Critics (Q1_target, Q2_target):
    └── Polyak-averaged copies of critics

  Training:
    1. Actor loss: maximize Q(s, a~π) + α * entropy
    2. Critic loss: MSE(Q(s,a), r + γ * min(Q_target(s',a'~π)))
    3. Temperature α: auto-tuned to target entropy
```

## Key Porting Points

| Pattern | Solution |
|---------|----------|
| `torch.distributions.Beta` | `compat.distributions.Beta` (numpy fallback for sampling) |
| Twin Q-networks | Two separate MLP instances |
| Polyak averaging | `target = τ * model + (1-τ) * target` via tree_map |
| Auto temperature | Learnable log_alpha parameter |
| `.detach()` for target | `mx.stop_gradient()` |

## Polyak Averaging in MLX

```python
def polyak_update(source, target, tau=0.005):
    """Soft update: target = tau * source + (1 - tau) * target."""
    from mlx.utils import tree_flatten, tree_unflatten
    source_params = dict(tree_flatten(source.parameters()))
    target_params = dict(tree_flatten(target.parameters()))
    updated = [(k, tau * source_params[k] + (1 - tau) * v)
               for k, v in target_params.items()]
    target.load_weights(updated)
```

---

## Acceptance Criteria

1. Actor produces valid action distributions
2. Twin critics output scalar Q-values
3. Polyak averaging: target params slowly track source
4. Temperature auto-tuning adjusts α
5. Training on simple environment (CartPole-like) shows learning
6. 15+ tests passing
