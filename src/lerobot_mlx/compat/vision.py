# Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
# LeRobot-MLX: torchvision ResNet replacement using MLX
#
# Implements ResNet-18/34 for use as vision backbone in ACT, Diffusion, and VQ-BeT policies.
# MLX Conv2d expects NHWC layout; inputs arrive as NCHW (torch convention).

import math
from typing import List, Optional, Tuple, Dict

import mlx.core as mx
import mlx.nn as nn


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "_BasicBlock",
    "_max_pool_2d",
    "_channel_first_to_last",
    "_channel_last_to_first",
]


# ---------------------------------------------------------------------------
# Channel format helpers
# ---------------------------------------------------------------------------

def _channel_first_to_last(x: mx.array) -> mx.array:
    """(B, C, H, W) -> (B, H, W, C) for MLX Conv2d."""
    return mx.transpose(x, axes=(0, 2, 3, 1))


def _channel_last_to_first(x: mx.array) -> mx.array:
    """(B, H, W, C) -> (B, C, H, W) for torch convention."""
    return mx.transpose(x, axes=(0, 3, 1, 2))


# ---------------------------------------------------------------------------
# Max Pooling (MLX has no built-in max_pool2d)
# ---------------------------------------------------------------------------

def _max_pool_2d(
    x: mx.array,
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
) -> mx.array:
    """Manual 2D max pooling for NHWC tensors.

    Uses a sliding-window approach: for each output position, slice the
    corresponding input window and take the element-wise max.

    Args:
        x: Input tensor of shape (B, H, W, C) in NHWC format.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window.
        padding: Zero-padding added to both sides of the input.

    Returns:
        Pooled tensor of shape (B, oH, oW, C).
    """
    B, H, W, C = x.shape

    if padding > 0:
        x = mx.pad(
            x,
            pad_width=[(0, 0), (padding, padding), (padding, padding), (0, 0)],
            constant_values=float("-inf"),
        )

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    oH = (H_padded - kernel_size) // stride + 1
    oW = (W_padded - kernel_size) // stride + 1

    # Collect patches via slicing and reduce with max
    # Start with -inf and iterate over the kernel window
    result = mx.full((B, oH, oW, C), float("-inf"))
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            patch = x[:, kh : kh + oH * stride : stride, kw : kw + oW * stride : stride, :]
            result = mx.maximum(result, patch)

    return result


# ---------------------------------------------------------------------------
# Kaiming uniform init helper
# ---------------------------------------------------------------------------

def _kaiming_uniform_init(
    shape: Tuple[int, ...],
    fan_in: int,
) -> mx.array:
    """Kaiming uniform initialization (matches PyTorch Conv2d default)."""
    bound = math.sqrt(6.0 / fan_in)
    return mx.random.uniform(-bound, bound, shape=shape)


# ---------------------------------------------------------------------------
# Downsample helper (Sequential-like container for conv + bn)
# ---------------------------------------------------------------------------

class _Downsample(nn.Module):
    """1x1 conv + batch norm downsample for skip connections."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.bn(self.conv(x))


# ---------------------------------------------------------------------------
# BasicBlock
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    """ResNet basic residual block (two 3x3 convolutions)."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(planes)
        self.downsample = downsample

    def __call__(self, x: mx.array) -> mx.array:
        identity = x

        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = nn.relu(out + identity)
        return out


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    """ResNet model matching torchvision API for feature extraction.

    Accepts input in NCHW format (torch convention), converts internally
    to NHWC for MLX Conv2d, and returns output in NCHW convention.
    """

    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 1000,
    ):
        super().__init__()
        self.inplanes = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = _Downsample(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Input: (B, C, H, W) NCHW. Output: (B, num_classes)."""
        # Convert NCHW -> NHWC for MLX Conv2d
        x = _channel_first_to_last(x)

        # Stem
        x = nn.relu(self.bn1(self.conv1(x)))
        x = _max_pool_2d(x, kernel_size=3, stride=2, padding=1)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling over spatial dims (H, W in NHWC)
        x = mx.mean(x, axis=(1, 2))

        # Classification head
        x = self.fc(x)
        return x

    def forward_features(self, x: mx.array) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Extract intermediate features (used by policies as backbone).

        Args:
            x: Input tensor (B, C, H, W) in NCHW format.

        Returns:
            Tuple of (final_features_NHWC, dict_of_layer_features_NHWC).
        """
        x = _channel_first_to_last(x)

        # Stem
        x = nn.relu(self.bn1(self.conv1(x)))
        x = _max_pool_2d(x, kernel_size=3, stride=2, padding=1)

        features: Dict[str, mx.array] = {}
        x = self.layer1(x)
        features["layer1"] = x
        x = self.layer2(x)
        features["layer2"] = x
        x = self.layer3(x)
        features["layer3"] = x
        x = self.layer4(x)
        features["layer4"] = x

        return x, features


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def resnet18(pretrained: bool = False, **kwargs) -> ResNet:
    """ResNet-18 (2-2-2-2 basic blocks)."""
    model = ResNet(_BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained_resnet(model, "resnet18")
    return model


def resnet34(pretrained: bool = False, **kwargs) -> ResNet:
    """ResNet-34 (3-4-6-3 basic blocks)."""
    model = ResNet(_BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained_resnet(model, "resnet34")
    return model


def _load_pretrained_resnet(model: ResNet, name: str) -> None:
    """Load pretrained weights from HuggingFace Hub, converting to MLX format.

    Downloads Microsoft's ResNet weights (safetensors format) from HF Hub,
    converts weight names and transposes conv weights from OIHW to OHWI.

    Args:
        model: ResNet model instance to load weights into.
        name: Model name ('resnet18' or 'resnet34').
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface-hub is required for pretrained weight loading. "
            "Install with: pip install huggingface-hub"
        ) from None

    HF_REPOS = {
        'resnet18': 'microsoft/resnet-18',
        'resnet34': 'microsoft/resnet-34',
    }

    repo_id = HF_REPOS.get(name)
    if repo_id is None:
        raise ValueError(f"No pretrained weights available for {name}")

    try:
        weights_path = hf_hub_download(repo_id, "model.safetensors")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download pretrained weights for {name} from {repo_id}. "
            f"Ensure you have internet access and huggingface-hub installed. "
            f"Error: {e}"
        ) from e

    try:
        import safetensors.numpy as sf_np
    except ImportError:
        raise ImportError(
            "safetensors is required for pretrained weight loading. "
            "Install with: pip install safetensors"
        ) from None

    torch_weights = sf_np.load_file(weights_path)
    mlx_weights = _convert_hf_resnet_weights(torch_weights)
    model.load_weights(list(mlx_weights.items()))


def _convert_hf_resnet_weights(hf_weights: dict) -> dict:
    """Convert HuggingFace Microsoft ResNet weights to our MLX ResNet format.

    Microsoft's HF ResNet uses keys like:
        embedder.embedder.convolution.weight  -> conv1.weight
        embedder.embedder.normalization.weight -> bn1.weight
        encoder.stages.0.layers.0.layer.0.convolution.weight -> layer1.layers.0.conv1.weight
        encoder.stages.0.layers.0.layer.0.normalization.weight -> layer1.layers.0.bn1.weight
        encoder.stages.0.layers.0.layer.1.convolution.weight -> layer1.layers.0.conv2.weight
        encoder.stages.0.layers.0.layer.1.normalization.weight -> layer1.layers.0.bn2.weight
        encoder.stages.0.layers.0.shortcut.convolution.weight -> layer1.layers.0.downsample.conv.weight
        encoder.stages.0.layers.0.shortcut.normalization.weight -> layer1.layers.0.downsample.bn.weight
        classifier.1.weight -> fc.weight
        classifier.1.bias -> fc.bias

    Conv weights are transposed from PyTorch OIHW to MLX OHWI format.
    """
    mlx_weights = {}

    for hf_key, value in hf_weights.items():
        mlx_key = _map_hf_resnet_key(hf_key)
        if mlx_key is None:
            continue

        # Transpose conv weights: PyTorch OIHW -> MLX OHWI
        if value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)

        mlx_weights[mlx_key] = mx.array(value)

    return mlx_weights


def _map_hf_resnet_key(hf_key: str) -> Optional[str]:
    """Map a HuggingFace Microsoft ResNet key to our MLX ResNet key.

    Returns None for keys we don't need (e.g. pooler, num_batches_tracked).
    """
    # Strip 'resnet.' prefix — Microsoft HF models use 'resnet.embedder...' format
    if hf_key.startswith('resnet.'):
        hf_key = hf_key[len('resnet.'):]

    # Skip batch norm tracking counters (not used in MLX)
    if 'num_batches_tracked' in hf_key:
        return None

    # Stem: embedder.embedder.convolution/normalization
    if hf_key.startswith('embedder.embedder.convolution.'):
        suffix = hf_key.split('embedder.embedder.convolution.')[-1]
        return f'conv1.{suffix}'
    if hf_key.startswith('embedder.embedder.normalization.'):
        suffix = hf_key.split('embedder.embedder.normalization.')[-1]
        return f'bn1.{suffix}'

    # Encoder stages: encoder.stages.{stage_idx}.layers.{block_idx}
    if hf_key.startswith('encoder.stages.'):
        parts = hf_key.split('.')
        # encoder.stages.{stage}.layers.{block}.layer.{conv_idx}.{type}.{param}
        # encoder.stages.{stage}.layers.{block}.shortcut.{type}.{param}
        stage_idx = int(parts[2])
        block_idx = int(parts[4])
        layer_name = f'layer{stage_idx + 1}'

        rest = '.'.join(parts[5:])  # e.g. "layer.0.convolution.weight"

        if rest.startswith('layer.'):
            sub_parts = rest.split('.')
            conv_idx = int(sub_parts[1])  # 0 or 1
            param_type = sub_parts[2]  # convolution or normalization
            param_name = sub_parts[3]  # weight, bias, etc.

            if param_type == 'convolution':
                return f'{layer_name}.layers.{block_idx}.conv{conv_idx + 1}.{param_name}'
            elif param_type == 'normalization':
                return f'{layer_name}.layers.{block_idx}.bn{conv_idx + 1}.{param_name}'
        elif rest.startswith('shortcut.'):
            sub_parts = rest.split('.')
            param_type = sub_parts[1]  # convolution or normalization
            param_name = sub_parts[2]  # weight, bias, etc.

            if param_type == 'convolution':
                return f'{layer_name}.layers.{block_idx}.downsample.conv.{param_name}'
            elif param_type == 'normalization':
                return f'{layer_name}.layers.{block_idx}.downsample.bn.{param_name}'

        return None

    # Classifier: classifier.1.weight / classifier.1.bias
    if hf_key.startswith('classifier.1.'):
        suffix = hf_key.split('classifier.1.')[-1]
        return f'fc.{suffix}'

    # Skip pooler and other keys we don't need
    return None
