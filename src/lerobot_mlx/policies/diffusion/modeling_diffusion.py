#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MLX port of Diffusion Policy: "Diffusion Policy: Visuomotor Policy Learning
# via Action Diffusion" (https://huggingface.co/papers/2303.04137)

import math
from collections import deque
from collections.abc import Callable
from typing import Optional

import mlx.core as mx
import mlx.nn as _nn
import numpy as np

from lerobot_mlx.compat import nn, F
from lerobot_mlx.compat.tensor_ops import Tensor
from lerobot_mlx.compat.einops_mlx import rearrange
from lerobot_mlx.compat.diffusers_mlx import DDPMScheduler, DDIMScheduler
from lerobot_mlx.compat.vision import ResNet, _BasicBlock, _channel_first_to_last, _channel_last_to_first, _max_pool_2d
from lerobot_mlx.compat.nn_modules import Module

from lerobot_mlx.policies.diffusion.configuration_diffusion import DiffusionConfig

# Constants matching upstream
OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
OBS_ENV_STATE = "observation.environment_state"
ACTION = "action"


class DiffusionPolicy(Module):
    """Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion".

    MLX port of the upstream PyTorch implementation.
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(self, config: DiffusionConfig, **kwargs):
        super().__init__()
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    def predict_action_chunk(self, batch: dict, noise: Optional[Tensor] = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {k: mx.stack(list(self._queues[k]), axis=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    def select_action(self, batch: dict, noise: Optional[Tensor] = None) -> Tensor:
        """Select a single action given environment observations."""
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = mx.stack(
                [batch[key] for key in self.config.image_features], axis=-4
            )
        self._queues = _populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            # actions: (B, n_action_steps, action_dim)
            # Transpose to (n_action_steps, B, action_dim) then extend
            actions_t = mx.transpose(actions, axes=(1, 0, 2))
            for i in range(actions_t.shape[0]):
                self._queues[ACTION].append(actions_t[i])

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict) -> tuple:
        """Run the batch through the model and compute the loss for training."""
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = mx.expand_dims(batch[key], axis=1)
            batch[OBS_IMAGES] = mx.stack(
                [batch[key] for key in self.config.image_features], axis=-4
            )
        loss = self.diffusion.compute_loss(batch)
        return loss, None

    def __call__(self, batch: dict) -> tuple:
        """Forward pass — mirrors PyTorch nn.Module.__call__."""
        return self.forward(batch)


def _populate_queues(queues: dict, batch: dict) -> dict:
    """Populate observation queues from a batch (mirrors upstream populate_queues)."""
    for key in batch:
        if key in queues:
            queues[key].append(batch[key])
    return queues


def _make_noise_scheduler(name: str, **kwargs) -> DDPMScheduler | DDIMScheduler:
    """Factory for noise scheduler instances."""
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        # Build observation encoders
        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the reverse diffusion process to generate samples."""
        action_dim = self.config.action_feature.shape[0]

        # Sample prior
        if noise is not None:
            sample = noise
        else:
            sample = mx.random.normal(
                shape=(batch_size, self.config.horizon, action_dim)
            )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            t_val = int(t.item()) if isinstance(t, mx.array) else int(t)
            # Create timestep tensor for batch
            timestep = mx.full((sample.shape[0],), t_val, dtype=mx.int32)

            # Predict model output
            model_output = self.unet(sample, timestep, global_cond=global_cond)

            # Compute previous sample: x_t -> x_{t-1}
            result = self.noise_scheduler.step(model_output, t_val, sample)
            sample = result.prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict) -> Tensor:
        """Encode image features and concatenate with state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        # Extract image features
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims, camera index first
                images_per_camera = rearrange(
                    batch[OBS_IMAGES], "b s n ... -> n (b s) ..."
                )
                img_features_list = mx.concatenate(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera)
                    ]
                )
                img_features = rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            else:
                # Combine batch, sequence, and camera dims
                img_features = self.rgb_encoder(
                    rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = rearrange(
                    img_features,
                    "(b s n) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim)
        combined = mx.concatenate(global_cond_feats, axis=-1)
        return mx.flatten(combined, start_axis=1)

    def generate_actions(self, batch: dict, noise: Optional[Tensor] = None) -> Tensor:
        """Generate actions from observations."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        # Extract n_action_steps worth of actions
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict) -> Tensor:
        """Compute training loss (noise prediction MSE)."""
        assert OBS_STATE in batch and ACTION in batch
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode observations
        global_cond = self._prepare_global_conditioning(batch)

        # Forward diffusion
        trajectory = batch[ACTION]
        # Sample noise
        eps = mx.random.normal(trajectory.shape)
        # Sample random timesteps
        timesteps = mx.random.randint(
            low=0,
            high=self.config.num_train_timesteps,
            shape=(trajectory.shape[0],),
        ).astype(mx.int32)
        # Add noise
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run denoising network
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute loss
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss for padding if configured
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"do_mask_loss_for_padding={self.config.do_mask_loss_for_padding}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * mx.expand_dims(in_episode_bound, axis=-1)

        return mx.mean(loss)


class SpatialSoftmax(Module):
    """Spatial Soft Argmax operation.

    Takes 2D feature maps (B, C, H, W) in NCHW format and returns
    keypoint coordinates (B, K, 2).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape: (C, H, W) input feature map shape.
            num_kp: number of keypoints. If None, uses C.
        """
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # Create position grid
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = pos_x.reshape(self._in_h * self._in_w, 1).astype(np.float32)
        pos_y = pos_y.reshape(self._in_h * self._in_w, 1).astype(np.float32)
        self.pos_grid = mx.array(np.concatenate([pos_x, pos_y], axis=1))

    def __call__(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W]
        features = mx.reshape(features, (-1, self._in_h * self._in_w))
        # 2D softmax normalization
        attention = mx.softmax(features, axis=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2]
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = mx.reshape(expected_xy, (-1, self._out_c, 2))

        return feature_keypoints


class DiffusionRgbEncoder(Module):
    """Encodes an RGB image into a 1D feature vector.

    Uses a ResNet backbone with optional preprocessing (resize, crop)
    and SpatialSoftmax pooling.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing flags
        self.do_resize = config.resize_shape is not None
        self.resize_shape = config.resize_shape

        crop_shape = config.crop_shape
        self.do_crop = crop_shape is not None
        self.crop_shape = crop_shape

        # Set up backbone (ResNet18/34)
        # Build the backbone and extract feature layers (remove avgpool and fc)
        if config.vision_backbone == "resnet18":
            backbone_model = ResNet(_BasicBlock, [2, 2, 2, 2])
        elif config.vision_backbone == "resnet34":
            backbone_model = ResNet(_BasicBlock, [3, 4, 6, 3])
        else:
            raise ValueError(f"Unsupported vision backbone: {config.vision_backbone}")

        # Store backbone components for feature extraction
        # The ResNet operates internally in NHWC, but we work in NCHW convention
        self.backbone = backbone_model

        # Determine feature map shape with a dry run
        if config.crop_shape is not None:
            dummy_h, dummy_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_h, dummy_w = config.resize_shape
        else:
            images_shape = next(iter(config.image_features.values())).shape
            dummy_h, dummy_w = images_shape[1], images_shape[2]

        images_shape = next(iter(config.image_features.values())).shape
        dummy_c = images_shape[0]

        # Run a dummy forward pass through the backbone to get feature map shape
        dummy = mx.zeros((1, dummy_c, dummy_h, dummy_w))
        feature_map = self._backbone_features(dummy)
        feature_map_shape = feature_map.shape[1:]  # (C, H, W)

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def _backbone_features(self, x: Tensor) -> Tensor:
        """Extract feature maps from backbone (up to layer4, no avgpool/fc).

        Input: (B, C, H, W) NCHW
        Output: (B, C_out, H_out, W_out) NCHW
        """
        # Convert NCHW -> NHWC for ResNet internals
        x = _channel_first_to_last(x)

        # Stem
        x = _nn.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = _max_pool_2d(x, kernel_size=3, stride=2, padding=1)

        # Residual stages
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Convert NHWC -> NCHW
        x = _channel_last_to_first(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocessing: resize and crop are simplified for MLX
        # (Full torchvision transforms are not available, so we skip resize/crop
        # in the MLX port — images should be pre-processed to the correct size)

        # Extract backbone features
        x = self._backbone_features(x)

        # Pool and flatten
        x = self.pool(x)
        x = mx.flatten(x, start_axis=1)

        # Final linear layer with non-linearity
        x = self.relu(self.out(x))
        return x


class DiffusionSinusoidalPosEmb(Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim) * -emb)
        emb = mx.expand_dims(x, axis=-1) * mx.expand_dims(emb, axis=0)
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return emb


class DiffusionConv1dBlock(Module):
    """Conv1d --> GroupNorm --> Mish

    All operations use channels-first (B, C, L) convention internally.
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.group_norm = nn.GroupNorm1d(n_groups, out_channels)
        self.mish = nn.Mish()

    def __call__(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.mish(x)
        return x


class DiffusionConditionalUnet1d(Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original
    diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        # Encoder for the diffusion timestep
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block
        action_dim = config.action_feature.shape[0]
        in_out = [(action_dim, config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:])
        )

        # Common kwargs for residual blocks
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }

        # Unet encoder
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                    DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                    # Downsample as long as it is not the last block
                    nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        # Middle processing
        self.mid_modules = nn.ModuleList([
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
            ),
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
            ),
        ])

        # Unet decoder
        self.up_modules = nn.ModuleList()
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    # dim_in * 2 because it takes the encoder's skip connection
                    DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                    DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                    # Upsample as long as it is not the last block
                    nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], action_dim, 1),
        )

    def __call__(self, x: Tensor, timestep: Tensor, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of diffusion timesteps.
            global_cond: (B, global_cond_dim) conditioning features.
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions: (B, T, D) -> (B, D, T)
        x = rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep.astype(mx.float32))

        # Concatenate global conditioning
        if global_cond is not None:
            global_feature = mx.concatenate([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder with skip connections
        encoder_skip_features = []
        for module in self.down_modules:
            resnet = module[0]
            resnet2 = module[1]
            downsample = module[2]
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder with skip connections
        for module in self.up_modules:
            resnet = module[0]
            resnet2 = module[1]
            upsample = module[2]
            x = mx.concatenate([x, encoder_skip_features.pop()], axis=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B, D, T) -> (B, T, D)
        x = rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(Module):
    """ResNet style 1D convolutional block with FiLM modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # Residual convolution for dimension matching
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def __call__(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding: (B, cond_channels) -> (B, cond_channels, 1)
        cond_embed = mx.expand_dims(self.cond_encoder(cond), axis=-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, :self.out_channels]
            bias = cond_embed[:, self.out_channels:]
            out = scale * out + bias
        else:
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out
