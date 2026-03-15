# PRD-04: Compatibility Layer — Vision Backbone + Einops + Diffusers Schedulers

> **Status:** TODO
> **Priority:** P0 — Required by ACT (ResNet), Diffusion (schedulers), VQ-BeT (ResNet)
> **Dependencies:** PRD-02, PRD-03
> **Estimated LOC:** ~700
> **Phase:** 1 (Foundation)

---

## Objective

Complete the final compat modules: torchvision ResNet backbone (used by ACT, Diffusion, VQ-BeT), einops rearrange/repeat (11 files), and diffusers DDPM/DDIM schedulers (7 files).

---

## Deliverables

### 1. `compat/vision.py` — torchvision Replacement

ACT and Diffusion policies use `torchvision.models.resnet18` as a vision backbone. We port this using MLX.

```python
import mlx.core as mx
import mlx.nn as _nn
from .nn_modules import Module

class _BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _nn.BatchNorm(planes)
        self.conv2 = _nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _nn.BatchNorm(planes)
        self.downsample = downsample

    def __call__(self, x):
        identity = x
        out = _nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = _nn.relu(out + identity)
        return out

class ResNet(Module):
    """Minimal ResNet matching torchvision API for feature extraction."""

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = _nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = _nn.BatchNorm(64)
        self.maxpool = None  # Implement as stride-2 conv or manual max pool
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # avgpool + fc for classification (may not be used in policy extraction)
        self.fc = _nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = _nn.Sequential(
                _nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                _nn.BatchNorm(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return _nn.Sequential(*layers)

    def __call__(self, x):
        # Input: (B, C, H, W) in torch convention
        # MLX Conv2d expects (B, H, W, C) — handle channel format
        x = _channel_first_to_last(x)
        x = _nn.relu(self.bn1(self.conv1(x)))
        x = _max_pool_2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = mx.mean(x, axis=(1, 2))  # Global average pooling
        x = self.fc(x)
        return x

    def forward_features(self, x):
        """Extract intermediate features (used by policies as backbone)."""
        x = _channel_first_to_last(x)
        x = _nn.relu(self.bn1(self.conv1(x)))
        x = _max_pool_2d(x, kernel_size=3, stride=2, padding=1)
        features = {}
        x = self.layer1(x); features['layer1'] = x
        x = self.layer2(x); features['layer2'] = x
        x = self.layer3(x); features['layer3'] = x
        x = self.layer4(x); features['layer4'] = x
        return x, features

def resnet18(pretrained=False, **kwargs):
    model = ResNet(_BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained_resnet(model, 'resnet18')
    return model

def resnet34(pretrained=False, **kwargs):
    model = ResNet(_BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained_resnet(model, 'resnet34')
    return model

def _load_pretrained_resnet(model, name):
    """Load torchvision pretrained weights, converting to MLX format."""
    from huggingface_hub import hf_hub_download
    import safetensors.numpy
    # Download from HF Hub (we'll host converted weights)
    # Convert torch weight names → MLX weight names
    # Handle NCHW → NHWC conv weight transposition
    ...

def _channel_first_to_last(x):
    """(B, C, H, W) → (B, H, W, C) for MLX Conv2d."""
    return mx.transpose(x, axes=(0, 2, 3, 1))

def _channel_last_to_first(x):
    """(B, H, W, C) → (B, C, H, W) for torch convention."""
    return mx.transpose(x, axes=(0, 3, 1, 2))

def _max_pool_2d(x, kernel_size, stride, padding=0):
    """Manual max pool 2D (MLX doesn't have built-in)."""
    B, H, W, C = x.shape
    if padding > 0:
        x = mx.pad(x, ((0,0), (padding,padding), (padding,padding), (0,0)),
                    constant_values=-float('inf'))
    oH = (H + 2*padding - kernel_size) // stride + 1
    oW = (W + 2*padding - kernel_size) // stride + 1
    # Extract patches and take max
    # Implementation via reshape + stride tricks
    ...
```

### 2. `compat/einops_mlx.py` — Tensor Rearrangement

```python
import mlx.core as mx
import re

def rearrange(tensor, pattern, **axes_lengths):
    """einops.rearrange → MLX reshape + transpose.

    Supports common patterns used in LeRobot:
    - 'b c h w -> b (c h w)'           # flatten
    - 'b (h w) c -> b h w c'           # unflatten
    - 'b c h w -> b h w c'             # transpose
    - 'b t c -> (b t) c'               # merge batch+time
    - '(b t) c -> b t c'               # split batch+time
    """
    lhs, rhs = pattern.split('->')
    lhs = lhs.strip()
    rhs = rhs.strip()

    lhs_tokens = _parse_pattern(lhs)
    rhs_tokens = _parse_pattern(rhs)

    return _execute_rearrange(tensor, lhs_tokens, rhs_tokens, axes_lengths)

def repeat(tensor, pattern, **axes_lengths):
    """einops.repeat → MLX expand_dims + broadcast + reshape.

    Common patterns:
    - 'b c -> b t c'  with t=10         # repeat along new axis
    - '1 c -> b c'    with b=32         # broadcast
    """
    lhs, rhs = pattern.split('->')
    lhs = lhs.strip()
    rhs = rhs.strip()

    return _execute_repeat(tensor, lhs, rhs, axes_lengths)

def reduce(tensor, pattern, reduction='mean', **axes_lengths):
    """einops.reduce → MLX reduction operations."""
    ...

def _parse_pattern(pattern):
    """Parse 'b (h w) c' into tokens with group info."""
    tokens = []
    i = 0
    while i < len(pattern):
        if pattern[i] == '(':
            j = pattern.index(')', i)
            group = pattern[i+1:j].split()
            tokens.append(('group', group))
            i = j + 1
        elif pattern[i].isalpha() or pattern[i] == '_':
            j = i
            while j < len(pattern) and (pattern[j].isalnum() or pattern[j] == '_'):
                j += 1
            tokens.append(('dim', pattern[i:j]))
            i = j
        else:
            i += 1
    return tokens

def _execute_rearrange(tensor, lhs_tokens, rhs_tokens, axes_lengths):
    """Execute the rearrangement via reshape + transpose."""
    # 1. Resolve axis sizes from tensor shape + axes_lengths
    # 2. If groups differ between lhs/rhs, reshape to split/merge
    # 3. If axis order differs, transpose
    # 4. If groups merged on rhs, reshape to merge
    # Implementation handles the 6 common patterns in LeRobot
    ...
```

### 3. `compat/diffusers_mlx.py` — DDPM/DDIM Noise Schedulers

Used by Diffusion Policy (7 files). Pure math, no torch dependency:

```python
import mlx.core as mx
import numpy as np
import math

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Models scheduler.

    Matches diffusers.DDPMScheduler API used by LeRobot's diffusion policy.
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001,
                 beta_end=0.02, beta_schedule='squaredcos_cap_v2',
                 clip_sample=True, prediction_type='epsilon'):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        # Compute beta schedule
        if beta_schedule == 'squaredcos_cap_v2':
            betas = self._cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == 'linear':
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        self.betas = mx.array(betas)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        t = np.linspace(0, timesteps, steps, dtype=np.float64) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999).astype(np.float32)

    def add_noise(self, original_samples, noise, timesteps):
        """Forward diffusion: add noise at given timesteps."""
        sqrt_alpha_prod = mx.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = mx.sqrt(1.0 - self.alphas_cumprod[timesteps])

        # Expand dims to match sample shape
        while sqrt_alpha_prod.ndim < original_samples.ndim:
            sqrt_alpha_prod = mx.expand_dims(sqrt_alpha_prod, axis=-1)
            sqrt_one_minus_alpha_prod = mx.expand_dims(sqrt_one_minus_alpha_prod, axis=-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def step(self, model_output, timestep, sample):
        """Reverse diffusion step: denoise."""
        t = timestep
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else mx.array(1.0)
        beta_prod_t = 1 - alpha_prod_t

        if self.prediction_type == 'epsilon':
            pred_original = (sample - mx.sqrt(beta_prod_t) * model_output) / mx.sqrt(alpha_prod_t)
        elif self.prediction_type == 'sample':
            pred_original = model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        if self.clip_sample:
            pred_original = mx.clip(pred_original, -1, 1)

        # Compute predicted previous sample
        pred_original_coeff = mx.sqrt(alpha_prod_t_prev) * self.betas[t] / beta_prod_t
        current_coeff = mx.sqrt(self.alphas[t]) * (1 - alpha_prod_t_prev) / beta_prod_t
        pred_prev = pred_original_coeff * pred_original + current_coeff * sample

        # Add noise (except at t=0)
        if t > 0:
            noise = mx.random.normal(sample.shape)
            variance = self.betas[t] * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
            pred_prev = pred_prev + mx.sqrt(variance) * noise

        return type('SchedulerOutput', (), {'prev_sample': pred_prev, 'pred_original_sample': pred_original})()

    def set_timesteps(self, num_inference_steps):
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = mx.array(list(range(0, self.num_train_timesteps, step_ratio))[::-1])

class DDIMScheduler:
    """Denoising Diffusion Implicit Models scheduler (deterministic)."""
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001,
                 beta_end=0.02, beta_schedule='squaredcos_cap_v2',
                 clip_sample=True, prediction_type='epsilon'):
        # Same initialization as DDPM
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        if beta_schedule == 'squaredcos_cap_v2':
            betas = DDPMScheduler._cosine_beta_schedule(None, num_train_timesteps)
        else:
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)

        self.betas = mx.array(betas)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas, axis=0)

    def add_noise(self, original_samples, noise, timesteps):
        """Same as DDPM."""
        return DDPMScheduler.add_noise(self, original_samples, noise, timesteps)

    def step(self, model_output, timestep, sample, eta=0.0):
        """DDIM deterministic step (eta=0 for fully deterministic)."""
        t = timestep
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else mx.array(1.0)

        if self.prediction_type == 'epsilon':
            pred_original = (sample - mx.sqrt(1 - alpha_prod_t) * model_output) / mx.sqrt(alpha_prod_t)
        else:
            pred_original = model_output

        if self.clip_sample:
            pred_original = mx.clip(pred_original, -1, 1)

        # DDIM formula
        pred_epsilon = (sample - mx.sqrt(alpha_prod_t) * pred_original) / mx.sqrt(1 - alpha_prod_t)
        pred_prev = mx.sqrt(alpha_prod_t_prev) * pred_original + \
                    mx.sqrt(1 - alpha_prod_t_prev) * pred_epsilon

        return type('SchedulerOutput', (), {'prev_sample': pred_prev, 'pred_original_sample': pred_original})()

    def set_timesteps(self, num_inference_steps):
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = mx.array(list(range(0, self.num_train_timesteps, step_ratio))[::-1])
```

---

## Acceptance Criteria

1. ResNet18 forward pass produces correct output shape: `(B, 1000)` for `(B, 3, 224, 224)` input
2. ResNet feature extraction returns intermediate feature maps with correct spatial dims
3. `rearrange('b c h w -> b (c h w)', ...)` works for all 6 common patterns in LeRobot
4. `DDPMScheduler.add_noise()` matches diffusers output within atol=1e-5
5. `DDPMScheduler.step()` denoises correctly (single step matches diffusers)
6. `DDIMScheduler.step()` is deterministic (same input → same output, no randomness)
7. 50+ tests covering all components

---

## Notes

- ResNet pretrained weight loading deferred to PRD-14 (weight conversion)
- Channel format: upstream uses NCHW (torch convention), MLX Conv2d uses NHWC — handle in vision.py
- Max pooling: MLX has no built-in, implement via reshape tricks or custom Metal kernel
- Einops: only implement the ~6 patterns actually used in LeRobot, not the full einops API
- Schedulers are pure math — no framework dependency, just need mx.array for GPU acceleration
