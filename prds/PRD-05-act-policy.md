# PRD-05: ACT Policy (Action Chunking Transformer)

> **Status:** TODO
> **Priority:** P0 — Most-used policy for manipulation tasks
> **Dependencies:** PRD-02, PRD-03, PRD-04 (compat layer complete)
> **Estimated LOC:** ~400 (mirror of upstream's ~1011 LOC)
> **Phase:** 2 (First Policy)

---

## Objective

Port the ACT (Action Chunking Transformer) policy — the #1 most used policy in LeRobot for manipulation tasks. Architecture: Transformer encoder-decoder + CVAE (Conditional Variational Autoencoder) with torchvision ResNet backbone.

---

## Upstream Files to Mirror

| Upstream File | Our File | Treatment |
|---------------|----------|-----------|
| `policies/act/configuration_act.py` | `policies/act/configuration_act.py` | **COPY VERBATIM** (pure dataclass) |
| `policies/act/modeling_act.py` | `policies/act/modeling_act.py` | **PORT** (replace imports, keep structure) |
| `policies/act/act_policy.py` (if exists) | `policies/act/act_policy.py` | **PORT** |

---

## Architecture Overview

```
ACT Policy:
  Input: observations (images + proprioception)
  ├── Vision Backbone (ResNet18) → image features
  ├── Encoder: TransformerEncoder
  │   └── Encodes: image features + proprioception + [optional] action sequence
  ├── CVAE Latent Space
  │   ├── Training: posterior encoder → z ~ N(μ, σ)
  │   └── Inference: prior z ~ N(0, I)
  ├── Decoder: TransformerDecoder
  │   └── Cross-attention: query tokens attend to encoder output
  └── Output: action_chunk (B, chunk_size, action_dim)
```

## Key torch→mlx Translations for ACT

| Upstream Pattern | Our Replacement |
|-----------------|----------------|
| `import torch` | `import mlx.core as mx` |
| `import torch.nn as nn` | `from lerobot_mlx.compat import nn` |
| `from torch import Tensor` | `from lerobot_mlx.compat import Tensor` |
| `import einops` | `from lerobot_mlx.compat.einops_mlx import rearrange, repeat` |
| `from torchvision.models import resnet18` | `from lerobot_mlx.compat.vision import resnet18` |
| `nn.TransformerEncoder(...)` | `nn.TransformerEncoder(...)` (compat) |
| `nn.TransformerEncoderLayer(...)` | `nn.TransformerEncoderLayer(...)` (compat) |
| `nn.MultiheadAttention(...)` | `nn.MultiheadAttention(...)` (compat) |
| `torch.distributions.Normal(...)` | `from lerobot_mlx.compat.distributions import Normal` |
| `torch.distributions.kl_divergence(...)` | `from lerobot_mlx.compat.distributions import kl_divergence` |
| `tensor.to(device)` | Removed (no-op) |
| `torch.no_grad()` | Removed (not needed) |
| `F.mse_loss(...)` | `from lerobot_mlx.compat.functional import mse_loss` |

## Porting Strategy

1. **Copy** `configuration_act.py` verbatim (it's a pure dataclass with no torch imports)
2. **Copy** `modeling_act.py` and replace the import header
3. **Replace** torch-specific patterns:
   - `torch.zeros(B, T, D)` → `tensor_ops.zeros(B, T, D)`
   - `x.permute(0, 2, 1)` → `mx.transpose(x, axes=(0, 2, 1))`
   - `x.view(B, -1)` → `x.reshape(B, -1)`
   - `x.detach()` → `mx.stop_gradient(x)`
4. **Verify** class names match upstream exactly: `ACTPolicy`, `ACT`, `ACTEncoder`, `ACTDecoder`

## Test Plan

```python
# tests/test_act.py

class TestACTConfig:
    def test_default_config(self):
        from lerobot_mlx.policies.act.configuration_act import ACTConfig
        config = ACTConfig()
        assert config.chunk_size > 0

class TestACTForward:
    def test_forward_shape(self):
        """Forward pass produces correct action chunk shape."""
        config = ACTConfig(
            input_shapes={"observation.images.top": [3, 480, 640],
                          "observation.state": [14]},
            output_shapes={"action": [14]},
            chunk_size=100,
        )
        model = ACTPolicy(config)
        batch = _make_act_batch(batch_size=2, config=config)
        output = model(batch)
        assert output["action"].shape == (2, 100, 14)

    def test_loss_computation(self):
        """Training loss is scalar and finite."""
        ...

    @pytest.mark.requires_torch
    def test_forward_matches_torch(self):
        """Cross-framework: MLX forward ≈ PyTorch forward."""
        # Load same weights, same input → compare outputs
        ...

class TestACTCVAE:
    def test_kl_divergence_positive(self):
        """KL divergence is non-negative."""
        ...

    def test_inference_uses_prior(self):
        """In eval mode, z is sampled from prior N(0,I)."""
        ...

class TestACTBackward:
    def test_gradient_flow(self):
        """Gradients flow through all parameters."""
        config = ACTConfig(...)
        model = ACTPolicy(config)
        loss_fn = lambda model, batch: model.compute_loss(batch)
        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(model, batch)
        # Verify at least some gradients are non-zero
```

---

## Acceptance Criteria

1. `from lerobot_mlx.policies.act import ACTPolicy` imports cleanly
2. Forward pass with random input produces correct output shape
3. Loss computation returns finite scalar
4. KL divergence term is non-negative
5. Gradients flow through encoder, decoder, and vision backbone
6. Config class is identical to upstream (copy verbatim)
7. Cross-framework test: forward pass within atol=1e-3 of PyTorch (requires_torch)
8. 20+ tests passing
