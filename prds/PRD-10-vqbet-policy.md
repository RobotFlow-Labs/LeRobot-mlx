# PRD-10: VQ-BeT Policy (Vector Quantized Behavior Transformer)

> **Status:** TODO
> **Priority:** P1 — Discrete action space coverage
> **Dependencies:** PRD-02, PRD-03, PRD-04
> **Estimated LOC:** ~600 (mirror of upstream's ~2638 LOC)
> **Phase:** 4 (Remaining Policies)

---

## Objective

Port VQ-BeT — Vector-Quantized Behavior Transformer that discretizes the action space using VQ-VAE codebooks and predicts actions via a transformer decoder.

---

## Upstream Files

| File | Treatment |
|------|-----------|
| `policies/vqbet/configuration_vqbet.py` | **COPY VERBATIM** |
| `policies/vqbet/modeling_vqbet.py` | **PORT** |
| Any VQ-VAE codebook files | **PORT** |

---

## Architecture

```
VQ-BeT:
  ├── Vision Backbone (ResNet18) → image features
  ├── VQ-VAE Encoder: continuous actions → discrete codes
  │   ├── Encoder MLP → latent
  │   ├── Vector Quantization → nearest codebook entry
  │   └── Commitment loss + codebook loss
  ├── Transformer Decoder: predict next code from history
  └── VQ-VAE Decoder: discrete codes → continuous actions
```

## Key Components to Port

| Component | torch Module | MLX Replacement |
|-----------|-------------|----------------|
| VQ codebook | Custom (torch.nn.Embedding + manual nearest-neighbor) | MLX Embedding + manual distances |
| Straight-through estimator | `x + (quantized - x).detach()` | `x + mx.stop_gradient(quantized - x)` |
| Transformer decoder | `nn.TransformerDecoder` | Custom from compat/ |
| ResNet backbone | torchvision.resnet18 | compat/vision.resnet18 |
| Gumbel-softmax | torch.nn.functional.gumbel_softmax | Custom MLX impl |

## Special: Gumbel-Softmax for VQ

```python
def gumbel_softmax(logits, tau=1.0, hard=False):
    """Gumbel-Softmax for differentiable discrete sampling."""
    gumbels = -mx.log(-mx.log(mx.random.uniform(logits.shape) + 1e-20) + 1e-20)
    y = mx.softmax((logits + gumbels) / tau, axis=-1)
    if hard:
        index = mx.argmax(y, axis=-1)
        y_hard = mx.zeros_like(y)
        y_hard = y_hard.at[mx.arange(y.shape[0]), index].add(1.0)
        y = mx.stop_gradient(y_hard - y) + y
    return y
```

---

## Acceptance Criteria

1. VQ-VAE encode/decode roundtrip (actions → codes → actions)
2. Codebook usage: codes distribute across entries (no collapse)
3. Transformer decoder predicts next code given history
4. Straight-through estimator: gradients flow through quantization
5. Training loss decreases
6. 15+ tests passing
