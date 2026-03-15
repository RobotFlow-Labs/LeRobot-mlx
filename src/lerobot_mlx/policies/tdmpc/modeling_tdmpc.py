#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
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
"""Implementation of TD-MPC / FOWM — MLX port.

Mirrors upstream lerobot.policies.tdmpc.modeling_tdmpc exactly, with all class
names, method names, and signatures preserved. Uses MLX compat layer.

References:
    TD-MPC paper: Temporal Difference Learning for Model Predictive Control
    FOWM paper: Finetuning Offline World Models in the Real World
"""

# ruff: noqa: N806

from collections import deque
from collections.abc import Callable
from copy import deepcopy
from functools import partial

import numpy as np

import mlx.core as mx
import mlx.nn as _nn

from lerobot_mlx.compat import nn, F, Tensor
from lerobot_mlx.compat.nn_modules import Module
from lerobot_mlx.compat.nn_layers import (
    Linear, LayerNorm, Conv2d, ModuleList,
    Sequential, Sigmoid, Tanh, ELU, Mish, ReLU, Flatten,
)
from lerobot_mlx.compat.tensor_ops import (
    zeros, ones, cat, stack, no_grad, zeros_like, ones_like,
    clamp, where, float32,
)
from lerobot_mlx.compat.einops_mlx import rearrange, repeat

from lerobot_mlx.policies.tdmpc.configuration_tdmpc import (
    TDMPCConfig, ACTION, OBS_STATE, OBS_ENV_STATE, OBS_IMAGE, OBS_PREFIX,
    OBS_STR, REWARD,
)


class TDMPCPolicy(Module):
    """Implementation of TD-MPC learning + inference — MLX port.

    Mirrors upstream TDMPCPolicy exactly.
    """

    config_class = TDMPCConfig
    name = "tdmpc"

    def __init__(self, config: TDMPCConfig, **kwargs):
        super().__init__()
        config.validate_features()
        self.config = config

        self.model = TDMPCTOLD(config)
        self.model_target = deepcopy(self.model)
        # In MLX we don't have requires_grad, targets are excluded from grad graph
        # by using mx.stop_gradient when needed.

        self.reset()

    def reset(self):
        """Clear observation and action queues. Clear previous means for warm starting."""
        self._queues = {
            OBS_STATE: deque(maxlen=1),
            ACTION: deque(maxlen=max(self.config.n_action_steps, self.config.n_action_repeats)),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGE] = deque(maxlen=1)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=1)
        self._prev_mean = None

    def predict_action_chunk(self, batch: dict[str, mx.array]) -> mx.array:
        """Predict a chunk of actions given environment observations."""
        batch = {key: mx.stack(list(self._queues[key]), axis=1) for key in batch if key in self._queues}

        # Remove the time dimensions as it is not handled yet.
        for key in batch:
            assert batch[key].shape[1] == 1
            batch[key] = batch[key][:, 0]

        # NOTE: Order of observations matters here.
        encode_keys = []
        if self.config.image_features:
            encode_keys.append(OBS_IMAGE)
        if self.config.env_state_feature:
            encode_keys.append(OBS_ENV_STATE)
        encode_keys.append(OBS_STATE)
        z = self.model.encode({k: batch[k] for k in encode_keys})
        if self.config.use_mpc:
            actions = self.plan(z)  # (horizon, batch, action_dim)
        else:
            # Plan with the policy alone.
            actions = mx.expand_dims(self.model.pi(z), axis=0)

        actions = mx.clip(actions, -1, +1)
        return actions

    def select_action(self, batch: dict[str, mx.array]) -> mx.array:
        """Select a single action given environment observations."""
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGE] = batch[next(iter(self.config.image_features))]
        if ACTION in batch:
            batch.pop(ACTION)

        self._queues = _populate_queues(self._queues, batch)

        # When the action queue is depleted, populate it again.
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)

            if self.config.n_action_repeats > 1:
                for _ in range(self.config.n_action_repeats):
                    self._queues[ACTION].append(actions[0])
            else:
                for i in range(self.config.n_action_steps):
                    self._queues[ACTION].append(actions[i])

        action = self._queues[ACTION].popleft()
        return action

    def plan(self, z: mx.array) -> mx.array:
        """Plan sequence of actions using TD-MPC inference (MPPI/CEM).

        Args:
            z: (batch, latent_dim) tensor for the initial state.
        Returns:
            (horizon, batch, action_dim) tensor for the planned trajectory.
        """
        batch_size = z.shape[0]
        action_dim = self.config.action_feature.shape[0]

        # Sample N_pi trajectories from the policy.
        pi_actions = mx.zeros((self.config.horizon, self.config.n_pi_samples, batch_size, action_dim))
        if self.config.n_pi_samples > 0:
            _z = repeat(z, "b d -> n b d", n=self.config.n_pi_samples)
            pi_action_list = []
            for t in range(self.config.horizon):
                pi_a = self.model.pi(_z, self.config.min_std)
                pi_action_list.append(pi_a)
                _z = self.model.latent_dynamics(_z, pi_a)
            pi_actions = mx.stack(pi_action_list, axis=0)

        # Expand z for CEM
        z = repeat(z, "b d -> n b d", n=self.config.n_gaussian_samples + self.config.n_pi_samples)

        # CEM loop
        mean = mx.zeros((self.config.horizon, batch_size, action_dim))
        if self._prev_mean is not None:
            # Warm start
            mean = mx.concatenate([self._prev_mean[1:], mx.zeros((1, batch_size, action_dim))], axis=0)
        std = self.config.max_std * mx.ones_like(mean)

        for _ in range(self.config.cem_iterations):
            # Sample action trajectories from gaussian
            std_normal_noise = mx.random.normal(
                shape=(self.config.horizon, self.config.n_gaussian_samples, batch_size, action_dim)
            )
            gaussian_actions = mx.clip(
                mx.expand_dims(mean, axis=1) + mx.expand_dims(std, axis=1) * std_normal_noise,
                -1, 1
            )

            # Compute elite actions
            actions = mx.concatenate([gaussian_actions, pi_actions], axis=1)
            value = self.estimate_value(z, actions)
            # Replace NaN with 0
            value = mx.where(mx.isnan(value), mx.zeros_like(value), value)

            # topk: get indices of top n_elites values
            # MLX doesn't have topk, so we use argsort
            sorted_indices = mx.argsort(-value, axis=0)  # descending
            elite_idxs = sorted_indices[:self.config.n_elites]  # (n_elites, batch)

            # Gather elite values
            elite_value = _take_along_dim(value, elite_idxs, dim=0)  # (n_elites, batch)

            # Gather elite actions: actions is (horizon, n_samples, batch, action_dim)
            # elite_idxs is (n_elites, batch)
            elite_actions = _gather_elite_actions(actions, elite_idxs)
            # elite_actions: (horizon, n_elites, batch, action_dim)

            # Update gaussian PDF parameters
            max_value = mx.max(elite_value, axis=0, keepdims=True)  # (1, batch)
            score = mx.exp(self.config.elite_weighting_temperature * (elite_value - max_value))
            score = score / mx.sum(score, axis=0, keepdims=True)
            # score: (n_elites, batch)

            # Weighted mean: (horizon, batch, action_dim)
            score_expanded = mx.expand_dims(score, axis=-1)  # (n_elites, batch, 1)
            # elite_actions: (horizon, n_elites, batch, action_dim)
            # We need: sum over n_elites dim (dim=1)
            # score needs shape (1, n_elites, batch, 1) for broadcasting
            score_4d = mx.expand_dims(score_expanded, axis=0)  # (1, n_elites, batch, 1)
            _mean = mx.sum(score_4d * elite_actions, axis=1)  # (horizon, batch, action_dim)

            # Weighted std
            _mean_expanded = mx.expand_dims(_mean, axis=1)  # (horizon, 1, batch, action_dim)
            _std = mx.sqrt(
                mx.sum(score_4d * (elite_actions - _mean_expanded) ** 2, axis=1)
            )

            # EMA update for mean
            mean = self.config.gaussian_mean_momentum * mean + (1 - self.config.gaussian_mean_momentum) * _mean
            std = mx.clip(_std, self.config.min_std, self.config.max_std)

        # Keep track of mean for warm-starting
        self._prev_mean = mean

        # Select from elites using multinomial sampling
        # score: (n_elites, batch), sample one index per batch element
        actions_out = _multinomial_select(elite_actions, score, batch_size)
        return actions_out

    def estimate_value(self, z: mx.array, actions: mx.array) -> mx.array:
        """Estimate trajectory value as per eqn 4 of FOWM.

        Args:
            z: (n_samples, batch, latent_dim) initial latent states.
            actions: (horizon, n_samples, batch, action_dim) action trajectories.
        Returns:
            (n_samples, batch) tensor of values.
        """
        G = mx.zeros(z.shape[:2])  # (n_samples, batch)
        running_discount = 1.0

        for t in range(actions.shape[0]):
            if self.config.uncertainty_regularizer_coeff > 0:
                qs = self.model.Qs(z, actions[t])  # (ensemble, n_samples, batch)
                regularization = -(self.config.uncertainty_regularizer_coeff * mx.std(qs, axis=0))
            else:
                regularization = 0

            z, reward = self.model.latent_dynamics_and_reward(z, actions[t])
            G = G + running_discount * (reward + regularization)
            running_discount *= self.config.discount

        # Terminal value
        next_action = self.model.pi(z, self.config.min_std)
        terminal_values = self.model.Qs(z, next_action)  # (ensemble, n_samples, batch)

        if self.config.q_ensemble_size > 2:
            # Randomly choose 2 Q functions
            idx = np.random.choice(self.config.q_ensemble_size, size=2, replace=False)
            selected = mx.stack([terminal_values[int(i)] for i in idx], axis=0)
            G = G + running_discount * mx.min(selected, axis=0)
        else:
            G = G + running_discount * mx.min(terminal_values, axis=0)

        if self.config.uncertainty_regularizer_coeff > 0:
            G = G - running_discount * self.config.uncertainty_regularizer_coeff * mx.std(terminal_values, axis=0)

        return G

    def forward(self, batch: dict[str, mx.array]) -> tuple[mx.array, dict]:
        """Run the batch through the model and compute the loss."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGE] = batch[next(iter(self.config.image_features))]

        info = {}

        # (b, t) -> (t, b)
        for key in batch:
            if isinstance(batch[key], mx.array) and batch[key].ndim > 1:
                axes = list(range(batch[key].ndim))
                axes[0], axes[1] = axes[1], axes[0]
                batch[key] = mx.transpose(batch[key], axes=axes)

        action = batch[ACTION]  # (t, b, action_dim)
        reward = batch[REWARD]  # (t, b)
        observations = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}

        # Apply random image augmentations.
        if self.config.image_features and self.config.max_random_shift_ratio > 0:
            observations[OBS_IMAGE] = flatten_forward_unflatten(
                partial(random_shifts_aug, max_random_shift_ratio=self.config.max_random_shift_ratio),
                observations[OBS_IMAGE],
            )

        # Split current vs next observations
        current_observation, next_observations = {}, {}
        for k in observations:
            current_observation[k] = observations[k][0]
            next_observations[k] = observations[k][1:]

        if self.config.image_features:
            horizon = next_observations[OBS_IMAGE].shape[0]
            batch_size = next_observations[OBS_IMAGE].shape[1]
        elif self.config.env_state_feature:
            horizon = next_observations[OBS_ENV_STATE].shape[0]
            batch_size = next_observations[OBS_ENV_STATE].shape[1]
        else:
            horizon = next_observations[OBS_STATE].shape[0]
            batch_size = next_observations[OBS_STATE].shape[1]

        # Run latent rollout
        z_preds_list = [self.model.encode(current_observation)]
        reward_preds_list = []
        for t in range(horizon):
            z_next, r_pred = self.model.latent_dynamics_and_reward(z_preds_list[-1], action[t])
            z_preds_list.append(z_next)
            reward_preds_list.append(r_pred)
        z_preds = mx.stack(z_preds_list, axis=0)  # (horizon+1, batch, latent_dim)
        reward_preds = mx.stack(reward_preds_list, axis=0)  # (horizon, batch)

        # Q and V predictions
        q_preds_ensemble = self.model.Qs(z_preds[:-1], action)  # (ensemble, horizon, batch)
        v_preds = self.model.V(z_preds[:-1])  # (horizon, batch)
        info.update({"Q": float(mx.mean(q_preds_ensemble)), "V": float(mx.mean(v_preds))})

        # Compute targets with stopgrad
        z_targets = mx.stop_gradient(self.model_target.encode(next_observations))
        # TD targets
        q_targets = mx.stop_gradient(
            reward + self.config.discount * self.model.V(self.model.encode(next_observations))
        )
        # V targets
        v_targets = mx.stop_gradient(
            self.model_target.Qs(mx.stop_gradient(z_preds[:-1]), action, return_min=True)
        )

        # Compute losses
        temporal_loss_coeffs = mx.power(
            self.config.temporal_decay_coeff, mx.arange(horizon)
        )
        temporal_loss_coeffs = mx.expand_dims(temporal_loss_coeffs, axis=-1)  # (horizon, 1)

        # Consistency loss
        consistency_loss_raw = F.mse_loss(z_preds[1:], z_targets, reduction="none")
        consistency_loss_raw = mx.mean(consistency_loss_raw, axis=-1)  # mean over latent dim
        consistency_loss = mx.mean(mx.sum(temporal_loss_coeffs * consistency_loss_raw, axis=0))

        # Reward loss
        reward_loss_raw = F.mse_loss(reward_preds, reward, reduction="none")
        reward_loss = mx.mean(mx.sum(temporal_loss_coeffs * reward_loss_raw, axis=0))

        # Q value loss
        q_targets_expanded = repeat(q_targets, "t b -> e t b", e=q_preds_ensemble.shape[0])
        q_value_loss_raw = F.mse_loss(q_preds_ensemble, q_targets_expanded, reduction="none")
        q_value_loss_raw = mx.sum(q_value_loss_raw, axis=0)  # sum over ensemble
        q_value_loss = mx.mean(mx.sum(temporal_loss_coeffs * q_value_loss_raw, axis=0))

        # V value loss (expectile regression)
        diff = v_targets - v_preds
        raw_v_value_loss = mx.where(
            diff > 0,
            self.config.expectile_weight * (diff ** 2),
            (1 - self.config.expectile_weight) * (diff ** 2),
        )
        v_value_loss = mx.mean(mx.sum(temporal_loss_coeffs * raw_v_value_loss, axis=0))

        # Pi loss (advantage weighted regression)
        z_preds_stopped = mx.stop_gradient(z_preds)
        advantage = mx.stop_gradient(
            self.model_target.Qs(z_preds_stopped[:-1], action, return_min=True) -
            self.model.V(z_preds_stopped[:-1])
        )
        info["advantage"] = advantage[0]
        exp_advantage = mx.clip(mx.exp(advantage * self.config.advantage_scaling), a_min=None, a_max=100.0)

        action_preds = self.model.pi(z_preds_stopped[:-1])  # (t, b, a)
        mse = mx.sum(F.mse_loss(action_preds, action, reduction="none"), axis=-1)  # (t, b)
        pi_loss = mx.mean(exp_advantage * mse * temporal_loss_coeffs)

        loss = (
            self.config.consistency_coeff * consistency_loss
            + self.config.reward_coeff * reward_loss
            + self.config.value_coeff * q_value_loss
            + self.config.value_coeff * v_value_loss
            + self.config.pi_coeff * pi_loss
        )

        info.update({
            "consistency_loss": float(consistency_loss),
            "reward_loss": float(reward_loss),
            "Q_value_loss": float(q_value_loss),
            "V_value_loss": float(v_value_loss),
            "pi_loss": float(pi_loss),
            "sum_loss": float(loss) * self.config.horizon,
        })

        # Undo (b, t) -> (t, b)
        for key in batch:
            if isinstance(batch[key], mx.array) and batch[key].ndim > 1:
                axes = list(range(batch[key].ndim))
                axes[0], axes[1] = axes[1], axes[0]
                batch[key] = mx.transpose(batch[key], axes=axes)

        return loss, info

    def update(self):
        """Update the target model's parameters with an EMA step."""
        update_ema_parameters(self.model_target, self.model, self.config.target_model_momentum)


class TDMPCTOLD(Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, config: TDMPCConfig):
        super().__init__()
        self.config = config
        action_dim = config.action_feature.shape[0]

        self.encoder = TDMPCObservationEncoder(config)
        self.dynamics = Sequential(
            Linear(config.latent_dim + action_dim, config.mlp_dim),
            LayerNorm(config.mlp_dim),
            Mish(),
            Linear(config.mlp_dim, config.mlp_dim),
            LayerNorm(config.mlp_dim),
            Mish(),
            Linear(config.mlp_dim, config.latent_dim),
            LayerNorm(config.latent_dim),
            Sigmoid(),
        )
        self.reward_net = Sequential(
            Linear(config.latent_dim + action_dim, config.mlp_dim),
            LayerNorm(config.mlp_dim),
            Mish(),
            Linear(config.mlp_dim, config.mlp_dim),
            LayerNorm(config.mlp_dim),
            Mish(),
            Linear(config.mlp_dim, 1),
        )
        self.pi_net = Sequential(
            Linear(config.latent_dim, config.mlp_dim),
            LayerNorm(config.mlp_dim),
            Mish(),
            Linear(config.mlp_dim, config.mlp_dim),
            LayerNorm(config.mlp_dim),
            Mish(),
            Linear(config.mlp_dim, action_dim),
        )
        self.q_nets = ModuleList([
            Sequential(
                Linear(config.latent_dim + action_dim, config.mlp_dim),
                LayerNorm(config.mlp_dim),
                Tanh(),
                Linear(config.mlp_dim, config.mlp_dim),
                ELU(),
                Linear(config.mlp_dim, 1),
            )
            for _ in range(config.q_ensemble_size)
        ])
        self.v_net = Sequential(
            Linear(config.latent_dim, config.mlp_dim),
            LayerNorm(config.mlp_dim),
            Tanh(),
            Linear(config.mlp_dim, config.mlp_dim),
            ELU(),
            Linear(config.mlp_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights.

        Orthogonal initialization for linear layers, zero biases.
        Final layers of reward and Q networks get zero weights.
        """
        def _apply_fn(path, module):
            if isinstance(module, _nn.Linear):
                w = module.weight
                shape = w.shape
                if len(shape) >= 2:
                    # QR decomposition for orthogonal matrix
                    flat = np.random.randn(max(shape), max(shape))
                    q, r = np.linalg.qr(flat)
                    q = q[:shape[0], :shape[1]]
                    module.weight = mx.array(q.astype(np.float32))
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias = mx.zeros_like(module.bias)

        self.apply_to_modules(_apply_fn)

        # Zero init final layers of reward and Q networks
        for m in [self.reward_net, *self.q_nets]:
            last_layer = m.layers[-1] if hasattr(m, 'layers') else None
            if last_layer is not None and isinstance(last_layer, _nn.Linear):
                last_layer.weight = mx.zeros_like(last_layer.weight)
                if last_layer.bias is not None:
                    last_layer.bias = mx.zeros_like(last_layer.bias)

    def encode(self, obs: dict[str, mx.array]) -> mx.array:
        """Encodes an observation into its latent representation."""
        return self.encoder(obs)

    def latent_dynamics_and_reward(self, z: mx.array, a: mx.array) -> tuple[mx.array, mx.array]:
        """Predict next latent state and reward.

        Args:
            z: (*, latent_dim) current latent.
            a: (*, action_dim) action.
        Returns:
            Tuple of (next_z, reward).
        """
        x = mx.concatenate([z, a], axis=-1)
        return self.dynamics(x), self.reward_net(x).squeeze(-1)

    def latent_dynamics(self, z: mx.array, a: mx.array) -> mx.array:
        """Predict next latent state.

        Args:
            z: (*, latent_dim) current latent.
            a: (*, action_dim) action.
        Returns:
            (*, latent_dim) next latent.
        """
        x = mx.concatenate([z, a], axis=-1)
        return self.dynamics(x)

    def pi(self, z: mx.array, std: float = 0.0) -> mx.array:
        """Sample action from the learned policy.

        Args:
            z: (*, latent_dim) current latent.
            std: Standard deviation of injected noise.
        Returns:
            (*, action_dim) sampled action.
        """
        action = mx.tanh(self.pi_net(z))
        if std > 0:
            noise = mx.random.normal(shape=action.shape) * std
            action = action + noise
        return action

    def V(self, z: mx.array) -> mx.array:
        """Predict state value.

        Args:
            z: (*, latent_dim) current latent.
        Returns:
            (*,) value estimate.
        """
        return self.v_net(z).squeeze(-1)

    def Qs(self, z: mx.array, a: mx.array, return_min: bool = False) -> mx.array:
        """Predict state-action values for Q ensemble.

        Args:
            z: (*, latent_dim) current latent.
            a: (*, action_dim) action.
            return_min: If True, return min of randomly selected 2 Qs.
        Returns:
            (q_ensemble, *) values or (*,) if return_min.
        """
        x = mx.concatenate([z, a], axis=-1)
        if not return_min:
            return mx.stack([q(x).squeeze(-1) for q in self.q_nets], axis=0)
        else:
            if len(self.q_nets) > 2:
                idx = np.random.choice(len(self.q_nets), size=2, replace=False)
                Qs = [self.q_nets[int(i)] for i in idx]
            else:
                Qs = list(self.q_nets)
            return mx.min(mx.stack([q(x).squeeze(-1) for q in Qs], axis=0), axis=0)


class TDMPCObservationEncoder(Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: TDMPCConfig):
        super().__init__()
        self.config = config

        if config.image_features:
            image_shape = next(iter(config.image_features.values())).shape
            in_channels = image_shape[0]

            self.image_enc_layers = Sequential(
                Conv2d(in_channels, config.image_encoder_hidden_dim, 7, stride=2),
                ReLU(),
                Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 5, stride=2),
                ReLU(),
                Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                ReLU(),
                Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                ReLU(),
            )
            # Compute output shape by running a dummy input
            dummy = mx.zeros((1, in_channels, image_shape[1], image_shape[2]))
            # Conv2d expects NCHW, our wrapper handles the conversion
            out = self.image_enc_layers(dummy)
            out_features = out.shape[1] * out.shape[2] * out.shape[3]

            self.image_fc = Sequential(
                Flatten(),
                Linear(out_features, config.latent_dim),
                LayerNorm(config.latent_dim),
                Sigmoid(),
            )

        if config.robot_state_feature:
            state_dim = config.robot_state_feature.shape[0]
            self.state_enc_layers = Sequential(
                Linear(state_dim, config.state_encoder_hidden_dim),
                ELU(),
                Linear(config.state_encoder_hidden_dim, config.latent_dim),
                LayerNorm(config.latent_dim),
                Sigmoid(),
            )

        if config.env_state_feature:
            env_dim = config.env_state_feature.shape[0]
            self.env_state_enc_layers = Sequential(
                Linear(env_dim, config.state_encoder_hidden_dim),
                ELU(),
                Linear(config.state_encoder_hidden_dim, config.latent_dim),
                LayerNorm(config.latent_dim),
                Sigmoid(),
            )

    def __call__(self, obs_dict: dict[str, mx.array]) -> mx.array:
        """Encode the observation(s) into a latent vector.

        Each modality is encoded, then averaged.
        """
        feat = []
        if self.config.image_features:
            img_key = next(iter(self.config.image_features))
            img = obs_dict.get(img_key, obs_dict.get(OBS_IMAGE))
            img_feat = flatten_forward_unflatten(
                lambda x: self.image_fc(self.image_enc_layers(x)),
                img,
            )
            feat.append(img_feat)
        if self.config.env_state_feature:
            feat.append(self.env_state_enc_layers(obs_dict[OBS_ENV_STATE]))
        if self.config.robot_state_feature:
            feat.append(self.state_enc_layers(obs_dict[OBS_STATE]))
        return mx.mean(mx.stack(feat, axis=0), axis=0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def random_shifts_aug(x: mx.array, max_random_shift_ratio: float) -> mx.array:
    """Randomly shift images horizontally and vertically.

    Adapted from https://github.com/facebookresearch/drqv2
    """
    b, c, h, w = x.shape
    assert h == w, "non-square images not handled yet"
    pad = int(round(max_random_shift_ratio * h))
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = mx.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad)[:h]
    arange = mx.expand_dims(arange, axis=(0, 2))  # (1, h, 1)
    arange_h = mx.broadcast_to(arange, (h, h, 1))
    arange_w = mx.transpose(arange_h, axes=(1, 0, 2))
    base_grid = mx.concatenate([arange_h, arange_w], axis=2)
    base_grid = mx.broadcast_to(mx.expand_dims(base_grid, axis=0), (b, h, h, 2))

    shift = mx.random.uniform(shape=(b, 1, 1, 2)) * (2 * pad + 1)
    shift = shift.astype(mx.int32).astype(mx.float32)
    shift = shift * 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def update_ema_parameters(ema_net: Module, net: Module, alpha: float):
    """Update EMA parameters: ema_param <- alpha * ema_param + (1 - alpha) * param."""
    ema_params = dict(ema_net.named_parameters())
    net_params = dict(net.named_parameters())
    updated = []
    for name, p in net_params.items():
        if name in ema_params:
            p_ema = ema_params[name]
            new_val = alpha * p_ema + (1 - alpha) * p
            updated.append((name, new_val))
    if updated:
        ema_net.load_weights(updated)


def flatten_forward_unflatten(fn: Callable, image_tensor: mx.array) -> mx.array:
    """Helper to temporarily flatten extra dims at the start of the image tensor.

    Args:
        fn: Callable that accepts (B, C, H, W) and returns (B, *).
        image_tensor: (**, C, H, W) tensor.
    Returns:
        (**, *) tensor.
    """
    if image_tensor.ndim == 4:
        return fn(image_tensor)
    start_dims = image_tensor.shape[:-3]
    inp = image_tensor.reshape(-1, *image_tensor.shape[-3:])
    flat_out = fn(inp)
    return flat_out.reshape(*start_dims, *flat_out.shape[1:])


def _populate_queues(queues: dict, batch: dict) -> dict:
    """Populate observation queues from batch."""
    for key in queues:
        if key in batch:
            queues[key].append(batch[key])
    return queues


def _take_along_dim(arr: mx.array, indices: mx.array, dim: int = 0) -> mx.array:
    """Take values from arr along dim using indices (like torch.take_along_dim)."""
    if dim == 0:
        # indices: (n_elites, batch), arr: (n_samples, batch)
        # Gather manually
        result = mx.zeros((indices.shape[0], arr.shape[1]))
        results = []
        for b in range(arr.shape[1]):
            idx = indices[:, b]
            results.append(arr[:, b][idx])
        return mx.stack(results, axis=1)
    raise NotImplementedError(f"_take_along_dim for dim={dim}")


def _gather_elite_actions(actions: mx.array, elite_idxs: mx.array) -> mx.array:
    """Gather elite actions from actions tensor.

    Args:
        actions: (horizon, n_samples, batch, action_dim)
        elite_idxs: (n_elites, batch)
    Returns:
        (horizon, n_elites, batch, action_dim)
    """
    horizon = actions.shape[0]
    n_elites = elite_idxs.shape[0]
    batch_size = elite_idxs.shape[1]
    action_dim = actions.shape[3]

    results = []
    for t in range(horizon):
        batch_results = []
        for b in range(batch_size):
            idx = elite_idxs[:, b]  # (n_elites,)
            selected = actions[t, :, b, :]  # (n_samples, action_dim)
            batch_results.append(selected[idx])  # (n_elites, action_dim)
        results.append(mx.stack(batch_results, axis=1))  # (n_elites, batch, action_dim)
    return mx.stack(results, axis=0)  # (horizon, n_elites, batch, action_dim)


def _multinomial_select(elite_actions: mx.array, score: mx.array, batch_size: int) -> mx.array:
    """Select actions from elites using multinomial sampling.

    Args:
        elite_actions: (horizon, n_elites, batch, action_dim)
        score: (n_elites, batch)
        batch_size: batch dimension size
    Returns:
        (horizon, batch, action_dim)
    """
    results = []
    for b in range(batch_size):
        probs = score[:, b]  # (n_elites,)
        probs = probs / mx.sum(probs)  # normalize
        # Sample one index from the categorical distribution
        cumsum = mx.cumsum(probs, axis=0)
        u = mx.random.uniform(shape=(1,))
        idx = int(mx.sum(cumsum < u).item())
        idx = min(idx, elite_actions.shape[1] - 1)
        results.append(elite_actions[:, idx, b, :])  # (horizon, action_dim)
    return mx.stack(results, axis=1)  # (horizon, batch, action_dim)


