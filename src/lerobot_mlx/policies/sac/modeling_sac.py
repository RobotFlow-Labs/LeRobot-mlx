#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# Copyright 2025 AIFLOW LABS / RobotFlow Labs (MLX port).
# All rights reserved.
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

"""SAC (Soft Actor-Critic) policy — MLX port.

Mirrors upstream lerobot/policies/sac/modeling_sac.py.
Only imports are changed: torch -> MLX via compat layer.
"""

import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import mlx.core as mx
import mlx.nn as _nn
import numpy as np

from lerobot_mlx.compat import nn, F
from lerobot_mlx.compat.nn_modules import Module
from lerobot_mlx.compat.nn_layers import (
    Linear,
    LayerNorm,
    Dropout,
    Sequential,
    ModuleList,
    ModuleDict,
    SiLU,
    ReLU,
    Tanh,
    Parameter,
    Conv2d,
)
from lerobot_mlx.compat.tensor_ops import (
    Tensor,
    cat,
    stack,
    zeros,
    no_grad,
)
from lerobot_mlx.compat.einops_mlx import repeat

from lerobot_mlx.policies.sac.configuration_sac import (
    SACConfig,
    is_image_feature,
    ACTION,
    OBS_ENV_STATE,
    OBS_STATE,
)

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class SACPolicy(Module):
    """Soft Actor-Critic policy — MLX implementation.

    Mirrors upstream SACPolicy with identical class names, method names, and signatures.
    """

    config_class = SACConfig
    name = "sac"

    def __init__(
        self,
        config: SACConfig | None = None,
    ):
        super().__init__()
        config.validate_features()
        self.config = config

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_critics(continuous_action_dim)
        self._init_actor(continuous_action_dim)
        self._init_temperature()

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor": self.actor.parameters(),
            "critic": self.critic_ensemble.parameters(),
            "temperature": {"log_alpha": self.log_alpha},
        }
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass

    def select_action(self, batch: dict) -> Tensor:
        """Select action for inference/evaluation"""
        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observations_features = self.actor.encoder.get_cached_image_features(batch)

        actions, _, _ = self.actor(batch, observations_features)
        return actions

    def critic_forward(
        self,
        observations: dict,
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through a critic network ensemble."""
        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def forward(
        self,
        batch: dict,
        model: Literal["actor", "critic", "temperature"] = "critic",
    ) -> dict:
        """Compute the loss for the given model."""
        actions = batch[ACTION]
        observations = batch["state"]
        observation_features = batch.get("observation_feature")

        if model == "critic":
            rewards = batch["reward"]
            next_observations = batch["next_state"]
            done = batch["done"]
            next_observation_features = batch.get("next_observation_feature")

            loss_critic = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )
            return {"loss_critic": loss_critic}

        if model == "actor":
            return {
                "loss_actor": self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        if model == "temperature":
            return {
                "loss_temperature": self.compute_loss_temperature(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average (Polyak averaging)."""
        tau = self.config.critic_target_update_weight
        from mlx.utils import tree_flatten

        source_params = dict(tree_flatten(self.critic_ensemble.parameters()))
        target_params = dict(tree_flatten(self.critic_target.parameters()))

        updated = [
            (k, tau * source_params[k] + (1.0 - tau) * target_params[k])
            for k in target_params
            if k in source_params
        ]
        self.critic_target.load_weights(updated)

    @property
    def temperature(self) -> float:
        """Return the current temperature value, always in sync with log_alpha."""
        return mx.exp(self.log_alpha).item()

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
    ) -> Tensor:
        # 1- compute next actions and log probs (no grad)
        next_action_preds, next_log_probs, _ = self.actor(next_observations, next_observation_features)

        # Stop gradient on actor outputs for target computation
        next_action_preds = mx.stop_gradient(next_action_preds)
        next_log_probs = mx.stop_gradient(next_log_probs)

        # 2- compute q targets
        q_targets = self.critic_forward(
            observations=next_observations,
            actions=next_action_preds,
            use_target=True,
            observation_features=next_observation_features,
        )
        q_targets = mx.stop_gradient(q_targets)

        # subsample critics to prevent overfitting if use high UTD
        if self.config.num_subsample_critics is not None:
            indices = mx.array(np.random.permutation(self.config.num_critics))
            indices = indices[: self.config.num_subsample_critics]
            q_targets = q_targets[indices]

        # critics subsample size
        min_q = mx.min(q_targets, axis=0)
        if self.config.use_backup_entropy:
            min_q = min_q - (self.temperature * next_log_probs)

        td_target = rewards + (1 - done) * self.config.discount * min_q
        td_target = mx.stop_gradient(td_target)

        # 3- compute predicted qs
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- Calculate loss
        # Repeat td_target for each critic: shape [num_critics, batch_size]
        td_target_duplicate = repeat(
            td_target, "b -> e b", e=q_preds.shape[0]
        )
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            )
        )
        # Mean over batch for each critic, then sum across critics
        critics_loss = mx.mean(critics_loss, axis=1)
        critics_loss = mx.sum(critics_loss)
        return critics_loss

    def compute_loss_temperature(self, observations, observation_features=None) -> Tensor:
        """Compute the temperature loss."""
        _, log_probs, _ = self.actor(observations, observation_features)
        log_probs = mx.stop_gradient(log_probs)
        temperature_loss = mx.mean(-mx.exp(self.log_alpha) * (log_probs + self.target_entropy))
        return temperature_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features=None,
    ) -> Tensor:
        actions_pi, log_probs, _ = self.actor(observations, observation_features)

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = mx.min(q_preds, axis=0)

        actor_loss = mx.mean((self.temperature * log_probs) - min_q_preds)
        return actor_loss

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else SACObservationEncoder(self.config)
        )

    def _init_critics(self, continuous_action_dim):
        """Build critic ensemble, targets."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=self.encoder_critic, ensemble=heads)
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=self.encoder_critic, ensemble=target_heads)
        # Copy weights from ensemble to target
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = continuous_action_dim
            self.target_entropy = -np.prod(dim) / 2

    def _init_temperature(self) -> None:
        """Set up temperature parameter (log_alpha)."""
        temp_init = self.config.temperature_init
        self.log_alpha = mx.array([math.log(temp_init)])


class SACObservationEncoder(Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.config = config
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()

    def _init_image_layers(self) -> None:
        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            return

        self.image_encoder = DefaultImageEncoder(self.config)

        if self.config.freeze_vision_encoder:
            freeze_image_encoder(self.image_encoder)

        dummy = zeros(1, *self.config.input_features[self.image_keys[0]].shape)
        _, channels, height, width = self.image_encoder(dummy).shape

        self.spatial_embeddings = ModuleDict()
        self.post_encoders = ModuleDict()

        for key in self.image_keys:
            name = key.replace(".", "_")
            self.spatial_embeddings[name] = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.config.image_embedding_pooling_dim,
            )
            self.post_encoders[name] = Sequential(
                Dropout(0.1),
                Linear(
                    in_features=channels * self.config.image_embedding_pooling_dim,
                    out_features=self.config.latent_dim,
                ),
                LayerNorm(self.config.latent_dim),
                Tanh(),
            )

    def _init_state_layers(self) -> None:
        self.has_env = OBS_ENV_STATE in self.config.input_features
        self.has_state = OBS_STATE in self.config.input_features
        if self.has_env:
            dim = self.config.input_features[OBS_ENV_STATE].shape[0]
            self.env_encoder = Sequential(
                Linear(dim, self.config.latent_dim),
                LayerNorm(self.config.latent_dim),
                Tanh(),
            )
        if self.has_state:
            dim = self.config.input_features[OBS_STATE].shape[0]
            self.state_encoder = Sequential(
                Linear(dim, self.config.latent_dim),
                LayerNorm(self.config.latent_dim),
                Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images:
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env:
            out += self.config.latent_dim
        if self.has_state:
            out += self.config.latent_dim
        self._out_dim = out

    def __call__(
        self, obs: dict, cache: dict | None = None, detach: bool = False
    ) -> Tensor:
        parts = []
        if self.has_images:
            if cache is None:
                cache = self.get_cached_image_features(obs)
            parts.append(self._encode_images(cache, detach))
        if self.has_env:
            parts.append(self.env_encoder(obs[OBS_ENV_STATE]))
        if self.has_state:
            parts.append(self.state_encoder(obs[OBS_STATE]))
        if parts:
            return cat(parts, dim=-1)

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state"
        )

    def get_cached_image_features(self, obs: dict) -> dict:
        """Extract and optionally cache image features from observations."""
        batched = cat([obs[k] for k in self.image_keys], dim=0)
        out = self.image_encoder(batched)
        n = len(self.image_keys)
        chunk_size = out.shape[0] // n
        chunks = [out[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
        return dict(zip(self.image_keys, chunks))

    def _encode_images(self, cache: dict, detach: bool) -> Tensor:
        """Encode image features from cached observations."""
        feats = []
        for k, feat in cache.items():
            safe_key = k.replace(".", "_")
            x = self.spatial_embeddings[safe_key](feat)
            x = self.post_encoders[safe_key](x)
            if detach:
                x = mx.stop_gradient(x)
            feats.append(x)
        return cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class MLP(Module):
    """Multi-layer perceptron builder.

    Dynamically constructs a sequence of layers based on `hidden_dims`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: str | None = None,
        activate_final: bool = False,
        dropout_rate: float | None = None,
        final_activation: str | None = None,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            layers.append(Linear(in_dim, out_dim))

            is_last = idx == total - 1
            if not is_last or activate_final:
                if dropout_rate and dropout_rate > 0:
                    layers.append(Dropout(p=dropout_rate))
                layers.append(LayerNorm(out_dim))
                if is_last and final_activation:
                    act = _get_activation(final_activation)
                else:
                    act = _get_activation(activations)
                layers.append(act)

            in_dim = out_dim

        self.net = Sequential(*layers)

    def __call__(self, x: Tensor) -> Tensor:
        return self.net(x)


def _get_activation(act) -> Module:
    """Resolve activation by name or return default SiLU."""
    if act is None:
        return SiLU()
    if isinstance(act, Module):
        return act
    if isinstance(act, str):
        name_map = {
            "SiLU": SiLU,
            "silu": SiLU,
            "ReLU": ReLU,
            "relu": ReLU,
            "Tanh": Tanh,
            "tanh": Tanh,
        }
        cls = name_map.get(act)
        if cls:
            return cls()
        raise ValueError(f"Unknown activation: {act}")
    # If it's already callable (e.g., a Module instance), return it
    return SiLU()


class CriticHead(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: str | None = None,
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: str | None = None,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = Linear(hidden_dims[-1], 1)
        if init_final is not None:
            self.output_layer.weight = mx.random.uniform(
                low=-init_final, high=init_final, shape=self.output_layer.weight.shape
            )
            self.output_layer.bias = mx.random.uniform(
                low=-init_final, high=init_final, shape=self.output_layer.bias.shape
            )
        else:
            _orthogonal_init(self.output_layer)

    def __call__(self, x: Tensor) -> Tensor:
        return self.output_layer(self.net(x))


class CriticEnsemble(Module):
    """CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list,
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.critics = ModuleList(ensemble)

    def __call__(
        self,
        observations: dict,
        actions: Tensor,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        obs_enc = self.encoder(observations, cache=observation_features)
        inputs = cat([obs_enc, actions], dim=-1)

        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to shape [num_critics, batch_size]
        q_values = mx.stack([mx.squeeze(q, axis=-1) for q in q_values], axis=0)
        return q_values


class Policy(Module):
    """SAC stochastic policy (actor).

    Outputs actions via a tanh-squashed multivariate normal distribution.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: Module,
        action_dim: int,
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        out_features = None
        for layer in reversed(network.net.layers):
            if isinstance(layer, Linear):
                out_features = layer.weight.shape[0]
                break
        if out_features is None:
            raise ValueError("Could not find a Linear layer in the MLP network")

        # Mean layer
        self.mean_layer = Linear(out_features, action_dim)
        if init_final is not None:
            self.mean_layer.weight = mx.random.uniform(
                low=-init_final, high=init_final, shape=self.mean_layer.weight.shape
            )
            self.mean_layer.bias = mx.random.uniform(
                low=-init_final, high=init_final, shape=self.mean_layer.bias.shape
            )
        else:
            _orthogonal_init(self.mean_layer)

        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = Linear(out_features, action_dim)
            if init_final is not None:
                self.std_layer.weight = mx.random.uniform(
                    low=-init_final, high=init_final, shape=self.std_layer.weight.shape
                )
                self.std_layer.bias = mx.random.uniform(
                    low=-init_final, high=init_final, shape=self.std_layer.bias.shape
                )
            else:
                _orthogonal_init(self.std_layer)

    def __call__(
        self,
        observations: dict,
        observation_features: Tensor | None = None,
    ) -> tuple:
        # We detach the encoder if it is shared to avoid backprop through it
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = mx.exp(log_std)
            std = mx.clip(std, self.std_min, self.std_max)
        else:
            std = mx.broadcast_to(self.fixed_std, means.shape)

        # Build distribution and sample (TanhMultivariateNormalDiag)
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        # Sample actions (reparameterized)
        actions = dist.rsample()

        # Compute log_probs
        log_probs = dist.log_prob(actions)

        return actions, log_probs, means


class DefaultImageEncoder(Module):
    def __init__(self, config: SACConfig):
        super().__init__()
        image_key = next(key for key in config.input_features if is_image_feature(key))
        self.image_enc_layers = Sequential(
            Conv2d(
                in_channels=config.input_features[image_key].shape[0],
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=7,
                stride=2,
            ),
            ReLU(),
            Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=5,
                stride=2,
            ),
            ReLU(),
            Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            ReLU(),
            Conv2d(
                in_channels=config.image_encoder_hidden_dim,
                out_channels=config.image_encoder_hidden_dim,
                kernel_size=3,
                stride=2,
            ),
            ReLU(),
        )

    def __call__(self, x):
        return self.image_enc_layers(x)


def freeze_image_encoder(image_encoder: Module):
    """Freeze all parameters in the encoder — no-op in MLX (handled by grad filter)."""
    pass


def _orthogonal_init(layer: Linear, gain: float = 1.0):
    """Apply orthogonal initialization to a Linear layer using numpy."""
    shape = layer.weight.shape
    # For MLX Linear, weight is (out_features, in_features)
    rows, cols = shape
    if rows < cols:
        flat = np.random.randn(cols, rows)
    else:
        flat = np.random.randn(rows, cols)
    q, r = np.linalg.qr(flat)
    # Make Q uniform (remove sign ambiguity)
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph
    if rows < cols:
        q = q.T
    q = q[:rows, :cols]
    layer.weight = mx.array(gain * q.astype(np.float32))


class SpatialLearnedEmbeddings(Module):
    def __init__(self, height, width, channel, num_features=8):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        # Initialize kernel with kaiming normal
        fan_in = channel * height * width
        std = math.sqrt(2.0 / fan_in)
        self.kernel = mx.random.normal(shape=(channel, height, width, num_features)) * std

    def __call__(self, features):
        """Forward pass for spatial embedding.

        Args:
            features: Input tensor of shape [B, C, H, W]
        Returns:
            Output tensor of shape [B, C*F]
        """
        features_expanded = mx.expand_dims(features, axis=-1)  # [B, C, H, W, 1]
        kernel_expanded = mx.expand_dims(self.kernel, axis=0)  # [1, C, H, W, F]

        # Element-wise multiplication and spatial reduction
        output = mx.sum(features_expanded * kernel_expanded, axis=(2, 3))  # Sum over H,W

        # Reshape to combine channel and feature dimensions
        batch_size = output.shape[0]
        output = mx.reshape(output, (batch_size, -1))  # [B, C*F]

        return output


class TanhMultivariateNormalDiag:
    """Tanh-squashed multivariate normal distribution with diagonal covariance.

    This replaces torch.distributions.TransformedDistribution with TanhTransform.
    The distribution samples z ~ N(loc, diag(scale_diag)), then returns tanh(z).
    """

    def __init__(self, loc: Tensor, scale_diag: Tensor):
        self.loc = loc
        self.scale_diag = scale_diag

    def rsample(self) -> Tensor:
        """Reparameterized sample: z ~ N(loc, scale), return tanh(z)."""
        eps = mx.random.normal(shape=self.loc.shape)
        z = self.loc + self.scale_diag * eps
        self._last_z = z
        return mx.tanh(z)

    def sample(self) -> Tensor:
        """Sample (same as rsample in MLX since all ops are differentiable)."""
        return self.rsample()

    def log_prob(self, actions: Tensor) -> Tensor:
        """Compute log probability with tanh squashing correction.

        log p(a) = log p(z) - sum(log(1 - tanh(z)^2 + eps))

        where z = atanh(a) but we use the cached z from rsample for numerical stability.
        """
        if hasattr(self, '_last_z'):
            z = self._last_z
        else:
            # Inverse tanh (atanh) with clamping for numerical stability
            actions_clamped = mx.clip(actions, -0.999999, 0.999999)
            z = 0.5 * mx.log((1 + actions_clamped) / (1 - actions_clamped))

        # Log prob of the base normal distribution (multivariate diagonal)
        # log N(z; loc, scale) = -0.5 * sum((z - loc)^2 / var) - sum(log(scale)) - 0.5 * d * log(2*pi)
        var = self.scale_diag ** 2
        log_scale = mx.log(self.scale_diag)
        d = self.loc.shape[-1]
        log_prob_normal = (
            -0.5 * mx.sum((z - self.loc) ** 2 / var, axis=-1)
            - mx.sum(log_scale, axis=-1)
            - 0.5 * d * math.log(2 * math.pi)
        )

        # Tanh squashing correction: -sum(log(1 - tanh(z)^2 + eps))
        tanh_z = mx.tanh(z)
        log_det = mx.sum(mx.log(1 - tanh_z ** 2 + 1e-6), axis=-1)

        return log_prob_normal - log_det

    def mode(self) -> Tensor:
        """Mode of the distribution (deterministic action)."""
        return mx.tanh(self.loc)

    @property
    def mean(self) -> Tensor:
        """Mean of the base distribution."""
        return self.loc


def polyak_update(source: Module, target: Module, tau: float = 0.005):
    """Soft update: target = tau * source + (1 - tau) * target.

    Utility function for updating target networks.
    """
    from mlx.utils import tree_flatten
    source_params = dict(tree_flatten(source.parameters()))
    target_params = dict(tree_flatten(target.parameters()))
    updated = [
        (k, tau * source_params[k] + (1 - tau) * target_params[k])
        for k in target_params
        if k in source_params
    ]
    target.load_weights(updated)
