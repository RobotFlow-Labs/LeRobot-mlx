# PRD-12: Weight Conversion & HF Hub Integration

> **Status:** TODO
> **Priority:** P1 — Enables using pretrained models
> **Dependencies:** PRD-02, PRD-05 or PRD-06 (at least one policy)
> **Estimated LOC:** ~300
> **Phase:** 4 (Infrastructure)

---

## Objective

Build a universal weight conversion pipeline that loads ANY LeRobot pretrained checkpoint from HuggingFace Hub (PyTorch safetensors) and converts it to MLX format. This is critical for practical use — users want to use existing pretrained policies.

---

## Architecture

```
HuggingFace Hub
  ├── model.safetensors (PyTorch weights)
  ├── config.json (policy config)
  └── preprocessor_config.json

        ↓ convert_weights()

  ├── Load safetensors → dict of numpy arrays
  ├── Rename keys (torch naming → MLX naming)
  ├── Transpose conv weights (NCHW → NHWC for MLX Conv2d)
  ├── Handle batch_norm running stats
  └── Save as MLX-compatible weights

        ↓ load_pretrained()

  Policy with pretrained weights, ready for inference
```

---

## Key Name Mappings

```python
# Common PyTorch → MLX weight name mappings
WEIGHT_MAP = {
    # Attention
    'self_attn.in_proj_weight': split into q/k/v projections,
    'self_attn.in_proj_bias': split into q/k/v biases,
    'self_attn.out_proj.weight': 'self_attn.out_proj.weight',

    # Normalization
    'norm.weight': 'norm.weight',  # LayerNorm: same name
    'norm.bias': 'norm.bias',
    'bn.weight': 'bn.weight',     # BatchNorm
    'bn.bias': 'bn.bias',
    'bn.running_mean': 'bn.running_mean',
    'bn.running_var': 'bn.running_var',

    # Conv2d: need to transpose OIHW → OHWI for MLX
    '*.conv*.weight': transpose(0, 2, 3, 1),
}
```

## Conv Weight Transposition

```python
def transpose_conv_weights(key, value):
    """PyTorch Conv2d: (out_ch, in_ch, kH, kW) → MLX: (out_ch, kH, kW, in_ch)."""
    if 'conv' in key and value.ndim == 4:
        return np.transpose(value, (0, 2, 3, 1))
    if 'conv' in key and value.ndim == 3:  # Conv1d
        return np.transpose(value, (0, 2, 1))
    return value
```

---

## Deliverables

### 1. `scripts/convert_weights.py` — CLI Tool

```bash
# Convert any HF Hub checkpoint
lerobot-mlx-convert --repo-id lerobot/act_pusht --output-dir ./converted/

# Convert local checkpoint
lerobot-mlx-convert --checkpoint-path ./model.safetensors --policy-type act --output-dir ./converted/
```

### 2. `policies/pretrained.py` — Programmatic Loading

```python
def load_pretrained(repo_id: str, policy_class=None):
    """Load pretrained policy from HF Hub, auto-converting weights."""
    from huggingface_hub import hf_hub_download
    import safetensors.numpy as sf_np

    # Download files
    config_path = hf_hub_download(repo_id, "config.json")
    weights_path = hf_hub_download(repo_id, "model.safetensors")

    # Load config → determine policy class
    config = _load_config(config_path)
    if policy_class is None:
        policy_class = _get_policy_class(config)

    # Create model
    model = policy_class(config)

    # Load and convert weights
    torch_weights = sf_np.load_file(weights_path)  # Returns numpy arrays
    mlx_weights = _convert_weights(torch_weights, policy_class)
    model.load_weights(list(mlx_weights.items()))

    return model

def _convert_weights(torch_weights, policy_class):
    """Convert PyTorch weight dict to MLX format."""
    mlx_weights = {}
    for key, value in torch_weights.items():
        # Rename keys
        mlx_key = _rename_key(key)
        # Transpose conv weights
        mlx_value = transpose_conv_weights(mlx_key, value)
        # Convert to mx.array
        mlx_weights[mlx_key] = mx.array(mlx_value)
    return mlx_weights
```

---

## Acceptance Criteria

1. Load ACT checkpoint from HF Hub → run inference → get actions
2. Load Diffusion checkpoint from HF Hub → run denoising → get actions
3. Weight shapes match between PyTorch and MLX models
4. Forward pass with converted weights matches PyTorch output (atol=1e-3)
5. CLI tool works: `lerobot-mlx-convert --repo-id <id>`
6. 10+ tests passing
