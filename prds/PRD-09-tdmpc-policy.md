# PRD-09: TD-MPC Policy

> **Status:** TODO
> **Priority:** P1 — Good RL coverage
> **Dependencies:** PRD-02, PRD-03, PRD-04
> **Estimated LOC:** ~450 (mirror of upstream's ~1131 LOC)
> **Phase:** 4 (Remaining Policies)

---

## Objective

Port TD-MPC (Temporal Difference Model Predictive Control) — a model-based RL policy that learns a world model and plans via model predictive control.

---

## Upstream Files

| File | Treatment |
|------|-----------|
| `policies/tdmpc/configuration_tdmpc.py` | **COPY VERBATIM** |
| `policies/tdmpc/modeling_tdmpc.py` | **PORT** |

---

## Architecture

```
TD-MPC:
  World Model:
    ├── Encoder: observation → latent state
    ├── Dynamics: latent + action → next latent (Conv3d)
    ├── Reward: latent → scalar reward
    └── Value: latent → value estimate

  Planning (MPC):
    1. Encode current observation
    2. Sample N action sequences
    3. Roll out each through dynamics model
    4. Score by reward + terminal value
    5. Execute best action
```

## Special Porting Challenges

| Challenge | Solution |
|-----------|----------|
| `nn.Conv3d` | Custom implementation (PRD-02 stub → full here) |
| `einops.rearrange` | `compat/einops_mlx` |
| MPC planning loop | Pure math, works with mx.array |
| Random shooting | `mx.random.normal` for action sampling |

## Conv3d Implementation

TD-MPC is the only policy using Conv3d. Implement as temporal Conv1d + spatial Conv2d:

```python
class Conv3d(Module):
    """Conv3d via decomposed temporal-spatial convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kT, kH, kW = kernel_size
        # Decompose: temporal Conv1d + spatial Conv2d
        self.temporal = _nn.Conv1d(in_channels, in_channels, kT,
                                    stride=stride if isinstance(stride, int) else stride[0],
                                    padding=padding if isinstance(padding, int) else padding[0])
        self.spatial = _nn.Conv2d(in_channels, out_channels, (kH, kW),
                                   stride=stride if isinstance(stride, int) else stride[1],
                                   padding=padding if isinstance(padding, int) else padding[1],
                                   bias=bias)

    def __call__(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # Temporal: reshape to (B*H*W, C, T)
        x_t = x.transpose(0, 3, 4, 1, 2).reshape(B*H*W, C, T)
        x_t = self.temporal(x_t)
        T_out = x_t.shape[-1]
        # Spatial: reshape to (B*T_out, C, H, W)
        x_s = x_t.reshape(B, H, W, C, T_out).transpose(0, 4, 3, 1, 2).reshape(B*T_out, C, H, W)
        x_s = self.spatial(x_s)
        # Reshape back: (B, C_out, T_out, H_out, W_out)
        ...
        return x_s
```

---

## Acceptance Criteria

1. World model forward pass: correct latent, reward, value shapes
2. MPC planning loop runs without error
3. Training loss decreases on synthetic environment data
4. Conv3d produces correct output shapes
5. 15+ tests passing
