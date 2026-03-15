#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
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
"""VQ-BeT Policy -- MLX implementation.

Direct port of upstream lerobot.policies.vqbet.modeling_vqbet and
lerobot.policies.vqbet.vqbet_utils to MLX, preserving all class names,
method names, and signatures.
"""

import math
import warnings
from collections import deque
from collections.abc import Callable
from functools import partial
from math import ceil
from random import randrange

import numpy as np

import mlx.core as mx
import mlx.nn as _nn

from lerobot_mlx.compat import nn, F, Tensor
from lerobot_mlx.compat.nn_modules import Module
from lerobot_mlx.compat.nn_layers import (
    Linear, LayerNorm, Embedding, Dropout, Conv2d, ModuleList, ModuleDict,
    Identity, Sequential, GELU, ReLU, Parameter,
)
from lerobot_mlx.compat.tensor_ops import (
    zeros, ones, cat, stack, no_grad, zeros_like, ones_like,
    tensor, arange, full, float32,
)
from lerobot_mlx.compat.einops_mlx import rearrange, repeat, reduce
from lerobot_mlx.compat.vision import (
    ResNet, _BasicBlock, _channel_first_to_last, _channel_last_to_first, _max_pool_2d,
)

from lerobot_mlx.policies.vqbet.configuration_vqbet import (
    VQBeTConfig, ACTION, OBS_STATE, OBS_IMAGES,
)

# ruff: noqa: N806


# =============================================================================
# Utility functions (from vqbet_utils.py)
# =============================================================================

def _identity(t):
    return t


def _noop(*args, **kwargs):
    pass


def _log(t, eps=1e-20):
    return mx.log(mx.clip(t, a_min=eps, a_max=None))


def _ema_inplace(old, new, decay):
    """In-place EMA update: old = old * decay + new * (1 - decay)."""
    updated = old * decay + new * (1 - decay)
    # MLX doesn't have in-place ops; we return the updated value
    return updated


def _uniform_init(*shape):
    """Kaiming uniform initialization."""
    t = mx.random.normal(shape=shape) * 0.02
    return t


def _sample_vectors(samples, num):
    num_samples = samples.shape[0]
    if num_samples >= num:
        indices = mx.random.permutation(num_samples)[:num]
    else:
        indices = mx.random.randint(0, num_samples, shape=(num,))
    return samples[indices]


def _batched_sample_vectors(samples, num):
    results = []
    for i in range(samples.shape[0]):
        results.append(_sample_vectors(samples[i], num))
    return mx.stack(results, axis=0)


def _gumbel_noise(t):
    noise = mx.random.uniform(shape=tuple(t.shape))
    return -_log(-_log(noise))


def _gumbel_sample(
    logits,
    temperature=1.0,
    stochastic=False,
    straight_through=False,
    reinmax=False,
    dim=-1,
    training=True,
):
    dtype = logits.dtype
    size = logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + _gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = mx.argmax(sampling_logits, axis=dim)
    # Build one-hot encoding using F.one_hot from compat
    from lerobot_mlx.compat.functional import one_hot as _one_hot
    one_hot = _one_hot(ind.astype(mx.int32), num_classes=size).astype(dtype)

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    if reinmax:
        pi0 = mx.softmax(logits, axis=dim)
        pi1 = (one_hot + mx.softmax(logits / temperature, axis=dim)) / 2
        pi1 = mx.softmax(mx.stop_gradient(_log(pi1) - logits) + logits, axis=1)
        pi2 = 2 * pi1 - 0.5 * pi0
        one_hot = mx.stop_gradient(pi2 - pi2) + one_hot
    else:
        pi1 = mx.softmax(logits / temperature, axis=dim)
        one_hot = one_hot + pi1 - mx.stop_gradient(pi1)

    return ind, one_hot


def _laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = mx.sum(x, axis=dim, keepdims=True)
    return (x + eps) / (denom + n_categories * eps)


def _batched_bincount(x, minlength):
    """Batched bincount: for each row in x, count occurrences of each value."""
    batch = x.shape[0]
    target = mx.zeros((batch, minlength), dtype=x.dtype)
    for b_idx in range(batch):
        for n_idx in range(x.shape[1]):
            idx = int(x[b_idx, n_idx].item())
            target = target.at[b_idx, idx].add(mx.array(1, dtype=x.dtype))
    return target


def _cdist(x, y):
    """Compute pairwise distances between x and y.

    x: (b, n, d), y: (b, m, d) -> (b, n, m)
    """
    x2 = mx.sum(x ** 2, axis=-1, keepdims=True)  # (b, n, 1)
    y2 = mx.sum(y ** 2, axis=-1, keepdims=True)  # (b, m, 1)
    xy = mx.matmul(x, mx.transpose(y, axes=(0, 2, 1)))  # (b, n, m)
    dist_sq = x2 - 2 * xy + mx.transpose(y2, axes=(0, 2, 1))
    return mx.sqrt(mx.maximum(dist_sq, mx.array(0.0)))


def _batched_embedding(indices, embeds):
    """Gather embeddings by indices.

    indices: (h, b, n), embeds: (h, c, d) -> (h, b, n, d)
    """
    h, b, n = indices.shape
    d = embeds.shape[-1]
    results = []
    for hi in range(h):
        batch_results = []
        for bi in range(b):
            idx = indices[hi, bi]  # (n,)
            batch_results.append(embeds[hi][idx])  # (n, d)
        results.append(mx.stack(batch_results, axis=0))  # (b, n, d)
    return mx.stack(results, axis=0)  # (h, b, n, d)


def _kmeans(samples, num_clusters, num_iters=10, sample_fn=_batched_sample_vectors):
    """Simple K-means clustering.

    samples: (num_codebooks, num_samples, dim)
    Returns: means (num_codebooks, num_clusters, dim), bins (num_codebooks, num_clusters)
    """
    num_codebooks = samples.shape[0]
    dim = samples.shape[-1]

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        # Compute negative distances
        dists = -_cdist(samples, means)
        buckets = mx.argmax(dists, axis=-1)  # (num_codebooks, num_samples)
        bins = _batched_bincount(buckets, minlength=num_clusters)

        zero_mask = bins == 0
        bins_clamped = mx.where(zero_mask, mx.ones_like(bins), bins)

        new_means = mx.zeros((num_codebooks, num_clusters, dim))
        # Accumulate
        for ci in range(num_codebooks):
            for ni in range(samples.shape[1]):
                b = int(buckets[ci, ni].item())
                new_means = new_means.at[ci, b].add(samples[ci, ni])

        new_means = new_means / mx.expand_dims(bins_clamped, axis=-1)

        mask_expanded = mx.expand_dims(zero_mask, axis=-1)
        means = mx.where(mask_expanded, means, new_means)

    return means, bins


# =============================================================================
# Gumbel Softmax (standalone utility)
# =============================================================================

def gumbel_softmax(logits, tau=1.0, hard=False):
    """Gumbel-Softmax for differentiable discrete sampling.

    Args:
        logits: Unnormalized log probabilities.
        tau: Temperature parameter.
        hard: If True, returns hard one-hot vectors with straight-through gradients.
    """
    gumbels = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape) + 1e-20) + 1e-20)
    y = mx.softmax((logits + gumbels) / tau, axis=-1)
    if hard:
        index = mx.argmax(y, axis=-1)
        # one-hot encoding via scatter
        num_classes = y.shape[-1]
        flat_idx = index.reshape(-1).astype(mx.int32)
        n = flat_idx.shape[0]
        flat_y = mx.zeros((n, num_classes))
        for i in range(n):
            idx_val = int(flat_idx[i].item())
            flat_y = flat_y.at[i, idx_val].add(mx.array(1.0))
        y_hard = flat_y.reshape(y.shape)
        y = mx.stop_gradient(y_hard - y) + y
    return y


# =============================================================================
# EuclideanCodebook
# =============================================================================

class EuclideanCodebook(Module):
    """Euclidean codebook for vector quantization."""

    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        learnable_codebook=False,
        sample_codebook_temp=1.0,
        ema_update=True,
        **kwargs,
    ):
        super().__init__()
        self.transform_input = _identity

        self.decay = decay
        self.ema_update = ema_update
        self.kmeans_init = kmeans_init

        if not kmeans_init:
            embed = mx.random.normal(shape=(num_codebooks, codebook_size, dim)) * 0.02
        else:
            embed = mx.zeros((num_codebooks, codebook_size, dim))

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.gumbel_sample = partial(
            _gumbel_sample,
            stochastic=False,
            reinmax=False,
            straight_through=False,
        )
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = _batched_sample_vectors

        self.initted = mx.array([not kmeans_init], dtype=mx.float32)
        self.cluster_size = mx.zeros((num_codebooks, codebook_size))
        self.embed_avg = mx.array(embed)

        self.learnable_codebook = learnable_codebook
        self.embed = mx.array(embed)

    def _init_embed(self, data):
        if self.initted.item() > 0:
            return

        embed, cluster_size = _kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn=self.sample_fn,
        )

        embed_sum = embed * mx.expand_dims(cluster_size, axis=-1)

        self.embed = mx.array(embed)
        self.embed_avg = mx.array(embed_sum)
        self.cluster_size = mx.array(cluster_size)
        self.initted = mx.array([1.0])

    def __call__(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = (
            sample_codebook_temp if sample_codebook_temp is not None else self.sample_codebook_temp
        )

        x = x.astype(mx.float32)

        if needs_codebook_dim:
            x = mx.expand_dims(x, axis=0)

        # flatten: (h, b*n, d)
        orig_shape = x.shape
        flatten = x.reshape(x.shape[0], -1, x.shape[-1])

        self._init_embed(flatten)

        embed = self.embed

        dist = -_cdist(flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )

        # Reshape embed_ind back
        embed_ind = embed_ind.reshape(orig_shape[:-1])

        if self.training:
            unpacked_onehot = embed_onehot.reshape(*orig_shape[:-1], embed_onehot.shape[-1])
            if unpacked_onehot.ndim == 3:
                # (h, n, c) x (h, c, d) -> (h, n, d)
                quantize = mx.einsum("hnc,hcd->hnd", unpacked_onehot, embed)
            else:
                # (h, b, n, c) x (h, c, d) -> (h, b, n, d)
                quantize = mx.einsum("hbnc,hcd->hbnd", unpacked_onehot, embed)
        else:
            # Eval mode: look up codebook entries directly by index
            # embed_ind: shape matching orig_shape[:-1], embed: (h, c, d)
            # Eval mode: look up codebook entries directly
            # embed_ind has shape orig_shape[:-1], embed has shape (h, codebook_size, d)
            # Flatten to (h, total_samples) for lookup, then reshape
            h = embed_ind.shape[0] if embed_ind.ndim >= 2 else 1
            flat_ind = embed_ind.reshape(h, -1).astype(mx.int32)
            # Lookup: for each codebook head, gather embeddings
            results = []
            for hi in range(h):
                results.append(embed[hi if hi < embed.shape[0] else 0][flat_ind[hi]])
            quantize = mx.stack(results, axis=0)
            quantize = quantize.reshape(orig_shape)

        if self.training and self.ema_update and not freeze_codebook:
            cluster_size = mx.sum(embed_onehot, axis=1)
            self.cluster_size = _ema_inplace(self.cluster_size, cluster_size, self.decay)

            embed_sum = mx.einsum("hnd,hnc->hcd", flatten, embed_onehot)
            self.embed_avg = _ema_inplace(self.embed_avg, embed_sum, self.decay)

            cluster_size_smooth = _laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps
            ) * mx.sum(self.cluster_size, axis=-1, keepdims=True)

            embed_normalized = self.embed_avg / mx.expand_dims(cluster_size_smooth, axis=-1)
            self.embed = mx.array(embed_normalized)

        if needs_codebook_dim:
            quantize = quantize[0]
            embed_ind = embed_ind[0]

        dist = dist.reshape(*orig_shape[:-1], dist.shape[-1])
        if needs_codebook_dim:
            dist = dist[0]

        return quantize, embed_ind, dist


# =============================================================================
# VectorQuantize
# =============================================================================

class VectorQuantize(Module):
    """Single-layer vector quantization module."""

    def __init__(
        self,
        dim,
        codebook_size=None,
        codebook_dim=None,
        heads=1,
        decay=0.8,
        eps=1e-5,
        commitment_weight=1.0,
        ema_update=True,
        learnable_codebook=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads

        codebook_dim = codebook_dim if codebook_dim is not None else dim
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = Linear(dim, codebook_input_dim) if requires_projection else Identity()
        self.project_out = Linear(codebook_input_dim, dim) if requires_projection else Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.learnable_codebook = learnable_codebook

        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            num_codebooks=1,
            codebook_size=codebook_size,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=0,
            learnable_codebook=learnable_codebook,
            ema_update=ema_update,
        )

        self.codebook_size = codebook_size

    @property
    def codebook(self):
        codebook = self._codebook.embed
        return codebook[0]  # Remove the num_codebooks dim (always 1 head)

    def get_codebook_vector_from_indices(self, indices):
        codebook = self.codebook  # (codebook_size, dim)
        codes = codebook[indices.astype(mx.int32)]
        return codes

    def __call__(
        self,
        x,
        indices=None,
        mask=None,
        sample_codebook_temp=None,
        freeze_codebook=False,
    ):
        orig_input = x
        only_one = x.ndim == 2

        if only_one:
            x = mx.expand_dims(x, axis=1)

        shape = x.shape

        # project input
        x = self.project_in(x)
        x = self._codebook.transform_input(x)

        codebook_forward_kwargs = {
            "sample_codebook_temp": sample_codebook_temp,
            "mask": mask,
            "freeze_codebook": freeze_codebook,
        }

        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.training:
            commit_quantize = mx.stop_gradient(quantize) if not self.learnable_codebook or freeze_codebook else quantize
            # straight-through
            quantize = x + mx.stop_gradient(quantize - x)

        # handle return_loss case
        if indices is not None:
            # Cross entropy loss on codes
            ce_loss = _nn.losses.cross_entropy(
                distances.reshape(-1, distances.shape[-1]),
                indices.reshape(-1).astype(mx.int32),
                reduction="mean",
            )
            return quantize, ce_loss

        # project out
        quantize = self.project_out(quantize)

        if only_one:
            quantize = quantize[:, 0, :]
            embed_ind = embed_ind  # already correct shape

        # aggregate loss
        loss = mx.array([0.0])

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = mx.mean((mx.stop_gradient(quantize) - orig_input) ** 2)
                loss = loss + commit_loss * self.commitment_weight

        return quantize, embed_ind, loss


# =============================================================================
# ResidualVQ
# =============================================================================

class ResidualVQ(Module):
    """Residual Vector Quantization with multiple VQ layers.

    Follows Algorithm 1 in https://huggingface.co/papers/2107.03312
    """

    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_dim=None,
        codebook_size=None,
        **kwargs,
    ):
        super().__init__()
        codebook_dim = codebook_dim if codebook_dim is not None else dim
        codebook_input_dim = codebook_dim

        requires_projection = codebook_input_dim != dim
        self.project_in = Linear(dim, codebook_input_dim) if requires_projection else Identity()
        self.project_out = Linear(codebook_input_dim, dim) if requires_projection else Identity()

        self.num_quantizers = num_quantizers
        self.layers = [
            VectorQuantize(dim=codebook_dim, codebook_dim=codebook_dim, codebook_size=codebook_size, **kwargs)
            for _ in range(num_quantizers)
        ]

        self.freeze_codebook = mx.array(False)

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = mx.stack(codebooks, axis=0)
        # shape: (q, 1, c, d) -> (q, c, d)
        return codebooks[:, 0]

    def get_codebook_vector_from_indices(self, indices):
        """Get codebook vectors from indices.

        indices: (batch, num_quantizers) or (batch, n, num_quantizers)
        Returns: (num_quantizers, batch, ..., dim)
        """
        batch = indices.shape[0]
        quantize_dim = indices.shape[-1]

        codebooks = self.codebooks  # (q, c, d)

        all_codes = []
        for q in range(self.num_quantizers):
            if q < quantize_dim:
                idx = indices[..., q].astype(mx.int32)  # (batch,) or (batch, n)
                codes = codebooks[q][idx]  # (batch, d) or (batch, n, d)
            else:
                codes = mx.zeros_like(all_codes[-1])
            all_codes.append(codes)

        return mx.stack(all_codes, axis=0)  # (q, batch, ..., d)

    def __call__(self, x, indices=None, return_all_codes=False, sample_codebook_temp=None):
        return_loss = indices is not None

        x = self.project_in(x)

        quantized_out = mx.zeros_like(x)
        residual = mx.array(x)

        all_losses = []
        all_indices = []

        if return_loss:
            ce_losses = []

        freeze = bool(self.freeze_codebook.item()) if isinstance(self.freeze_codebook, mx.array) else bool(self.freeze_codebook)

        for quantizer_index, layer in enumerate(self.layers):
            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            result = layer(
                residual,
                indices=layer_indices,
                sample_codebook_temp=sample_codebook_temp,
                freeze_codebook=freeze,
            )

            if return_loss:
                quantized, ce_loss = result
                ce_losses.append(ce_loss)
            else:
                quantized, embed_indices, loss = result
                all_indices.append(embed_indices)
                all_losses.append(loss)

            residual = residual - mx.stop_gradient(quantized)
            quantized_out = quantized_out + quantized

        quantized_out = self.project_out(quantized_out)

        if return_loss:
            total_ce_loss = sum(ce_losses)
            return quantized_out, total_ce_loss

        # Stack indices and losses
        all_indices_stacked = mx.stack(all_indices, axis=-1)
        all_losses_stacked = mx.stack(all_losses, axis=-1)

        return quantized_out, all_indices_stacked, all_losses_stacked


# =============================================================================
# GPT components (from vqbet_utils.py)
# =============================================================================

class CausalSelfAttention(Module):
    def __init__(self, config):
        super().__init__()
        assert config.gpt_hidden_dim % config.gpt_n_head == 0
        self.c_attn = Linear(config.gpt_hidden_dim, 3 * config.gpt_hidden_dim)
        self.c_proj = Linear(config.gpt_hidden_dim, config.gpt_hidden_dim)
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)
        # Build causal mask
        mask = mx.tril(mx.ones((config.gpt_block_size, config.gpt_block_size)))
        self.bias = mask.reshape(1, 1, config.gpt_block_size, config.gpt_block_size)
        self.gpt_n_head = config.gpt_n_head
        self.gpt_hidden_dim = config.gpt_hidden_dim

    def __call__(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        # Split into q, k, v
        q = qkv[..., :self.gpt_hidden_dim]
        k = qkv[..., self.gpt_hidden_dim:2*self.gpt_hidden_dim]
        v = qkv[..., 2*self.gpt_hidden_dim:]

        hs = C // self.gpt_n_head
        q = q.reshape(B, T, self.gpt_n_head, hs).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.gpt_n_head, hs).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.gpt_n_head, hs).transpose(0, 2, 1, 3)

        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(hs))
        causal_mask = self.bias[:, :, :T, :T]
        att = mx.where(causal_mask == 0, mx.array(float("-inf")), att)
        att = mx.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(Module):
    """Causal self-attention block for GPT."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.gpt_hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.gpt_hidden_dim)
        self.mlp_linear1 = Linear(config.gpt_hidden_dim, 4 * config.gpt_hidden_dim)
        self.mlp_gelu = GELU()
        self.mlp_linear2 = Linear(4 * config.gpt_hidden_dim, config.gpt_hidden_dim)
        self.mlp_dropout = Dropout(config.dropout)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.mlp_linear1(self.ln_2(x))
        h = self.mlp_gelu(h)
        h = self.mlp_linear2(h)
        h = self.mlp_dropout(h)
        x = x + h
        return x


class GPT(Module):
    """nanoGPT-style transformer for VQ-BeT."""

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        assert config.gpt_output_dim is not None
        assert config.gpt_block_size is not None
        self.config = config

        self.wte = Linear(config.gpt_input_dim, config.gpt_hidden_dim)
        self.wpe = Embedding(config.gpt_block_size, config.gpt_hidden_dim)
        self.drop = Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.gpt_n_layer)]
        self.ln_f = LayerNorm(config.gpt_hidden_dim)
        self.lm_head = Linear(config.gpt_hidden_dim, config.gpt_output_dim, bias=False)

    def __call__(self, input, targets=None):
        b, t, d = input.shape
        assert t <= self.config.gpt_block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.gpt_block_size}"
        )

        pos = mx.arange(0, t).reshape(1, -1)

        tok_emb = self.wte(input)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def configure_parameters(self):
        """Separate parameters into decay and no_decay sets."""
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                no_decay.append(param)
            elif "ln_" in name or "wpe" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay


# =============================================================================
# MLP
# =============================================================================

class MLP(Module):
    """Multi-layer perceptron (mirrors upstream torch.nn.Sequential-based MLP)."""

    def __init__(self, in_channels: int, hidden_channels: list[int]):
        super().__init__()
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(Linear(in_dim, hidden_dim))
            layers.append(ReLU())
            in_dim = hidden_dim
        layers.append(Linear(in_dim, hidden_channels[-1]))
        self.seq = _nn.Sequential(*layers)

    def __call__(self, x):
        return self.seq(x)


# =============================================================================
# SpatialSoftmax
# =============================================================================

class SpatialSoftmax(Module):
    """Spatial Soft Argmax operation.

    Takes 2D feature maps and returns the "center of mass" of activations
    of each channel (keypoints in image space).
    """

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = pos_x.reshape(self._in_h * self._in_w, 1).astype(np.float32)
        pos_y = pos_y.reshape(self._in_h * self._in_w, 1).astype(np.float32)
        self.pos_grid = mx.array(np.concatenate([pos_x, pos_y], axis=1))

    def __call__(self, features):
        """
        Args:
            features: (B, C, H, W) in channels-first format.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        B = features.shape[0]
        # Flatten spatial dims: (B, K, H*W)
        if features.ndim == 4:
            features_flat = features.reshape(B * self._out_c, self._in_h * self._in_w)
        else:
            features_flat = features.reshape(-1, self._in_h * self._in_w)

        attention = mx.softmax(features_flat, axis=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.reshape(B, self._out_c, 2)
        return feature_keypoints


# =============================================================================
# FocalLoss
# =============================================================================

class FocalLoss(Module):
    """Focal Loss from miniBET."""

    def __init__(self, gamma: float = 0, size_average: bool = True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def __call__(self, input, target):
        if len(input.shape) == 3:
            N, T, _ = input.shape
            logpt = F.log_softmax(input, dim=-1)
            # gather
            target_idx = target.reshape(N, T, 1).astype(mx.int32)
            logpt = mx.take_along_axis(logpt, target_idx, axis=-1).reshape(N, T)
        elif len(input.shape) == 2:
            logpt = F.log_softmax(input, dim=-1)
            target_idx = target.reshape(-1, 1).astype(mx.int32)
            logpt = mx.take_along_axis(logpt, target_idx, axis=-1).reshape(-1)
        pt = mx.exp(logpt)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return mx.mean(loss)
        else:
            return mx.sum(loss)


# =============================================================================
# VqVae
# =============================================================================

class VqVae(Module):
    """VQ-VAE with residual VQ, encoder, and decoder."""

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        self.config = config
        self.discretized = mx.array(False)
        self.optimized_steps = 0
        self.vqvae_num_layers = 2

        self.vq_layer = ResidualVQ(
            dim=config.vqvae_embedding_dim,
            num_quantizers=self.vqvae_num_layers,
            codebook_size=config.vqvae_n_embed,
        )

        action_dim = config.action_feature.shape[0]
        self.encoder = MLP(
            in_channels=action_dim * config.action_chunk_size,
            hidden_channels=[
                config.vqvae_enc_hidden_dim,
                config.vqvae_enc_hidden_dim,
                config.vqvae_embedding_dim,
            ],
        )
        self.decoder = MLP(
            in_channels=config.vqvae_embedding_dim,
            hidden_channels=[
                config.vqvae_enc_hidden_dim,
                config.vqvae_enc_hidden_dim,
                action_dim * config.action_chunk_size,
            ],
        )

    def get_embeddings_from_code(self, encoding_indices):
        """Get embeddings from code indices and sum across RVQ layers."""
        z_embed = self.vq_layer.get_codebook_vector_from_indices(encoding_indices)
        z_embed = mx.sum(z_embed, axis=0)
        return z_embed

    def get_action_from_latent(self, latent):
        """Decode latent vector to action chunks."""
        output = self.decoder(latent)
        action_dim = self.config.action_feature.shape[0]
        return output.reshape(output.shape[0], self.config.action_chunk_size, action_dim)

    def get_code(self, state):
        """Get VQ codes for given action sequence (used in training phase 2)."""
        # state: (N, T, A) -> (N, T*A)
        state = state.reshape(state.shape[0], -1)
        state_rep = self.encoder(state)
        state_rep_flat = mx.expand_dims(state_rep, axis=1)  # (N, 1, dim)
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat[:, 0, :]  # (N, dim)
        vq_code = vq_code[:, 0, :]  # (N, num_layers)
        return state_vq, vq_code

    def vqvae_forward(self, state):
        """Forward pass for VQ-VAE training (phase 1)."""
        # state: (N, T, A) -> (N, T*A)
        state_flat = state.reshape(state.shape[0], -1)
        state_rep = self.encoder(state_flat)
        state_rep_flat = mx.expand_dims(state_rep, axis=1)  # (N, 1, dim)
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat[:, 0, :]  # (N, dim)
        vq_code = vq_code[:, 0, :]  # (N, num_layers)
        vq_loss_state = mx.sum(vq_loss_state)

        dec_out = self.decoder(state_vq)
        encoder_loss = mx.mean(mx.abs(state_flat - dec_out))
        rep_loss = encoder_loss + vq_loss_state * 5

        metric = (
            mx.array(encoder_loss),
            mx.array(vq_loss_state),
            vq_code,
            float(rep_loss.item()),
        )
        return rep_loss, metric


# =============================================================================
# VQBeTHead
# =============================================================================

class VQBeTHead(Module):
    """Action head for VQ-BeT: bin prediction + offset prediction."""

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        self.config = config
        self.vqvae_model = VqVae(config)

        action_dim = config.action_feature.shape[0]

        if config.sequentially_select:
            self.map_to_cbet_preds_primary_bin = MLP(
                in_channels=config.gpt_output_dim,
                hidden_channels=[config.vqvae_n_embed],
            )
            self.map_to_cbet_preds_secondary_bin = MLP(
                in_channels=config.gpt_output_dim + config.vqvae_n_embed,
                hidden_channels=[config.vqvae_n_embed],
            )
        else:
            self.map_to_cbet_preds_bin = MLP(
                in_channels=config.gpt_output_dim,
                hidden_channels=[self.vqvae_model.vqvae_num_layers * config.vqvae_n_embed],
            )

        self.map_to_cbet_preds_offset = MLP(
            in_channels=config.gpt_output_dim,
            hidden_channels=[
                self.vqvae_model.vqvae_num_layers
                * config.vqvae_n_embed
                * config.action_chunk_size
                * action_dim,
            ],
        )

        self._focal_loss_fn = FocalLoss(gamma=2.0)

    def discretize(self, n_vqvae_training_steps, actions):
        """Train the VQ-VAE (phase 1)."""
        # Sliding window to create action chunks
        chunks = []
        for j in range(actions.shape[1] + 1 - self.config.action_chunk_size):
            chunks.append(actions[:, j:j + self.config.action_chunk_size, :])
        actions = mx.concatenate(chunks, axis=0)

        loss, metric = self.vqvae_model.vqvae_forward(actions)

        vq_code = metric[2]
        n_different_codes = 0
        for i in range(self.vqvae_model.vqvae_num_layers):
            n_different_codes += len(set(vq_code[:, i].tolist()))
        n_different_combinations = len(set(tuple(row) for row in vq_code.tolist()))
        recon_l1_error = float(metric[0].item())

        self.vqvae_model.optimized_steps += 1
        if self.vqvae_model.optimized_steps >= n_vqvae_training_steps:
            self.vqvae_model.discretized = mx.array(True)
            self.vqvae_model.vq_layer.freeze_codebook = mx.array(True)
            print("Finished discretizing action data!")
            self.vqvae_model.eval()

        return loss, n_different_codes, n_different_combinations, recon_l1_error

    def __call__(self, x, **kwargs):
        N, T, _ = x.shape
        NT = N * T
        x_flat = x.reshape(NT, -1)

        action_dim = self.config.action_feature.shape[0]
        G = self.vqvae_model.vqvae_num_layers
        C = self.config.vqvae_n_embed

        # Predict offsets
        cbet_offsets = self.map_to_cbet_preds_offset(x_flat)
        cbet_offsets = cbet_offsets.reshape(NT, G, C, -1)

        if self.config.sequentially_select:
            cbet_primary_logits = self.map_to_cbet_preds_primary_bin(x_flat)
            cbet_primary_probs = mx.softmax(cbet_primary_logits / self.config.bet_softmax_temperature, axis=-1)
            choices = cbet_primary_probs.shape[-1]

            # Sample primary centers
            sampled_primary = []
            for i in range(NT):
                probs = cbet_primary_probs[i]
                idx = mx.random.categorical(mx.log(probs + 1e-10))
                sampled_primary.append(idx)
            sampled_primary_centers = mx.stack(sampled_primary, axis=0)

            # Secondary prediction conditioned on primary
            primary_onehot = mx.zeros((NT, C))
            for i in range(NT):
                idx_val = int(sampled_primary_centers[i].item())
                primary_onehot = primary_onehot.at[i, idx_val].add(mx.array(1.0))

            secondary_input = mx.concatenate([x_flat, primary_onehot], axis=1)
            cbet_secondary_logits = self.map_to_cbet_preds_secondary_bin(secondary_input)
            cbet_secondary_probs = mx.softmax(cbet_secondary_logits / self.config.bet_softmax_temperature, axis=-1)

            sampled_secondary = []
            for i in range(NT):
                probs = cbet_secondary_probs[i]
                idx = mx.random.categorical(mx.log(probs + 1e-10))
                sampled_secondary.append(idx)
            sampled_secondary_centers = mx.stack(sampled_secondary, axis=0)

            sampled_centers = mx.stack([sampled_primary_centers, sampled_secondary_centers], axis=1)
            cbet_logits = mx.stack([cbet_primary_logits, cbet_secondary_logits], axis=1)
        else:
            cbet_logits = self.map_to_cbet_preds_bin(x_flat)
            cbet_logits = cbet_logits.reshape(NT, G, C)
            cbet_probs = mx.softmax(cbet_logits / self.config.bet_softmax_temperature, axis=-1)

            # Sample from multinomial
            sampled = []
            for g in range(G):
                g_samples = []
                for i in range(NT):
                    probs = cbet_probs[i, g]
                    idx = mx.random.categorical(mx.log(probs + 1e-10))
                    g_samples.append(idx)
                sampled.append(mx.stack(g_samples, axis=0))
            sampled_centers = mx.stack(sampled, axis=1)  # (NT, G)

        # Extract offsets for sampled codes using advanced indexing
        sampled_offsets = mx.zeros((NT, G, cbet_offsets.shape[-1]))
        for i in range(NT):
            for g in range(G):
                c_idx = int(sampled_centers[i, g].item())
                sampled_offsets = sampled_offsets.at[i, g].add(
                    cbet_offsets[i, g, c_idx] - sampled_offsets[i, g]
                )

        sampled_offsets = mx.sum(sampled_offsets, axis=1)  # (NT, W*A)

        # Decode actions from sampled codes
        return_decoder_input = mx.stop_gradient(
            self.vqvae_model.get_embeddings_from_code(sampled_centers.astype(mx.int32))
        )
        decoded_action = mx.stop_gradient(
            self.vqvae_model.get_action_from_latent(return_decoder_input)
        )

        sampled_offsets = sampled_offsets.reshape(NT, self.config.action_chunk_size, action_dim)
        predicted_action = decoded_action + sampled_offsets
        predicted_action = predicted_action.reshape(N, T, self.config.action_chunk_size * action_dim)

        return {
            "cbet_logits": cbet_logits,
            "predicted_action": predicted_action,
            "sampled_centers": sampled_centers,
            "decoded_action": decoded_action,
        }

    def loss_fn(self, pred, target, **kwargs):
        """Compute VQ-BeT loss (focal loss on code prediction + offset L1 loss)."""
        action_seq = target
        predicted_action = pred["predicted_action"]
        sampled_centers = pred["sampled_centers"]
        decoded_action = pred["decoded_action"]
        NT = predicted_action.shape[0] * predicted_action.shape[1]
        cbet_logits = pred["cbet_logits"]

        action_dim = self.config.action_feature.shape[0]
        predicted_action = predicted_action.reshape(
            predicted_action.shape[0] * predicted_action.shape[1],
            self.config.action_chunk_size,
            action_dim,
        )
        action_seq = action_seq.reshape(
            action_seq.shape[0] * action_seq.shape[1],
            self.config.action_chunk_size,
            action_dim,
        )

        # Get ground truth codes
        state_vq, action_bins = self.vqvae_model.get_code(action_seq)

        # Offset loss (L1)
        offset_loss = mx.mean(mx.abs(action_seq - predicted_action))

        # Focal loss for code prediction
        cbet_loss1 = self._focal_loss_fn(cbet_logits[:, 0, :], action_bins[:, 0])
        cbet_loss2 = self._focal_loss_fn(cbet_logits[:, 1, :], action_bins[:, 1])
        cbet_loss = (
            cbet_loss1 * self.config.primary_code_loss_weight
            + cbet_loss2 * self.config.secondary_code_loss_weight
        )

        equal_primary_code_rate = float(
            mx.sum((action_bins[:, 0] == sampled_centers[:, 0]).astype(mx.int32)).item()
        ) / NT
        equal_secondary_code_rate = float(
            mx.sum((action_bins[:, 1] == sampled_centers[:, 1]).astype(mx.int32)).item()
        ) / NT

        action_mse_error = float(mx.mean((action_seq - predicted_action) ** 2).item())
        vq_action_error = float(mx.mean(mx.abs(action_seq - decoded_action)).item())
        offset_action_error = float(mx.mean(mx.abs(action_seq - predicted_action)).item())
        action_error_max = float(mx.max(mx.abs(action_seq - predicted_action)).item())

        loss = cbet_loss + self.config.offset_loss_weight * offset_loss

        loss_dict = {
            "loss": loss,
            "classification_loss": float(cbet_loss.item()),
            "offset_loss": float(offset_loss.item()),
            "equal_primary_code_rate": equal_primary_code_rate,
            "equal_secondary_code_rate": equal_secondary_code_rate,
            "vq_action_error": vq_action_error,
            "offset_action_error": offset_action_error,
            "action_error_max": action_error_max,
            "action_mse_error": action_mse_error,
        }
        return loss_dict


# =============================================================================
# VQBeTRgbEncoder
# =============================================================================

class VQBeTRgbEncoder(Module):
    """Encode an RGB image into a 1D feature vector.

    Uses ResNet backbone + SpatialSoftmax pooling.
    """

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        self.do_crop = config.crop_shape is not None

        # Set up backbone (ResNet without final avgpool + fc)
        if config.vision_backbone == "resnet18":
            from lerobot_mlx.compat.vision import resnet18
            backbone_model = resnet18(pretrained=config.pretrained_backbone_weights is not None)
        else:
            from lerobot_mlx.compat.vision import resnet34
            backbone_model = resnet34(pretrained=config.pretrained_backbone_weights is not None)

        self.backbone = backbone_model

        # Determine feature map shape with a dry run
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        # Create dummy input in NCHW format - ResNet handles conversion internally
        dummy = mx.zeros((1, images_shape[0], *dummy_shape_h_w))
        # Use forward_features to get spatial feature maps (returns NHWC)
        feat_nhwc, _ = self.backbone.forward_features(dummy)
        # Convert to NCHW for SpatialSoftmax
        feat_cf = _channel_last_to_first(feat_nhwc)
        feature_map_shape = feat_cf.shape[1:]  # (C, H, W)

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = ReLU()

        self.crop_shape = config.crop_shape

    def __call__(self, x):
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Crop if needed
        if self.do_crop and self.crop_shape is not None:
            _, _, H, W = x.shape
            ch, cw = self.crop_shape
            if self.training:
                # Random crop
                top = int(mx.random.randint(0, H - ch + 1, shape=()).item())
                left = int(mx.random.randint(0, W - cw + 1, shape=()).item())
            else:
                # Center crop
                top = (H - ch) // 2
                left = (W - cw) // 2
            x = x[:, :, top:top+ch, left:left+cw]

        # ResNet expects NCHW and converts internally; use forward_features for spatial maps
        feat_nhwc, _ = self.backbone.forward_features(x)
        # Convert back to NCHW for SpatialSoftmax
        feat_cf = _channel_last_to_first(feat_nhwc)

        # Pool + flatten
        keypoints = self.pool(feat_cf)
        flat = keypoints.reshape(x.shape[0], -1)

        # Final linear + relu
        out = self.relu(self.out(flat))
        return out


# =============================================================================
# VQBeTModel
# =============================================================================

class VQBeTModel(Module):
    """VQ-BeT: The underlying neural network."""

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        self.config = config

        self.rgb_encoder = VQBeTRgbEncoder(config)
        self.num_images = len(config.image_features)

        # Action query token
        self.action_token = mx.random.normal(shape=(1, 1, config.gpt_input_dim))

        # Projectors
        self.state_projector = MLP(
            config.robot_state_feature.shape[0],
            hidden_channels=[config.gpt_input_dim],
        )
        self.rgb_feature_projector = MLP(
            self.rgb_encoder.feature_dim,
            hidden_channels=[config.gpt_input_dim],
        )

        # GPT
        self.policy = GPT(config)
        # Action head
        self.action_head = VQBeTHead(config)

        # Build target action indices
        num_tokens = config.n_action_pred_token + config.n_obs_steps - 1
        indices_list = []
        for i in range(num_tokens):
            indices_list.append(mx.arange(i, i + config.action_chunk_size))
        self.select_target_actions_indices = mx.stack(indices_list, axis=0)

    def __call__(self, batch, rollout=False):
        assert OBS_STATE in batch and OBS_IMAGES in batch

        batch_size = batch[OBS_STATE].shape[0]
        n_obs_steps = batch[OBS_STATE].shape[1]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode images
        obs_images = batch[OBS_IMAGES]
        # obs_images: (B, S, N, C, H, W) - flatten batch, seq, and num_images
        B, S, N = obs_images.shape[:3]
        img_flat = obs_images.reshape(B * S * N, *obs_images.shape[3:])
        img_features = self.rgb_encoder(img_flat)
        img_features = img_features.reshape(B, S, N, -1)

        # Project features
        rgb_tokens = self.rgb_feature_projector(img_features)

        # Build input tokens: [rgb_cam0, ..., state, action_query] per timestep
        input_token_list = []
        for i in range(rgb_tokens.shape[2]):
            input_token_list.append(rgb_tokens[:, :, i])

        input_token_list.append(self.state_projector(batch[OBS_STATE]))

        # Action token repeated for each obs step
        action_tok = mx.broadcast_to(
            self.action_token,
            (batch_size, n_obs_steps, self.config.gpt_input_dim),
        )
        input_token_list.append(action_tok)

        # Interleave: stack along token dim then flatten
        input_tokens = mx.stack(input_token_list, axis=2)  # (B, S, num_token_types, D)
        input_tokens = input_tokens.reshape(batch_size, n_obs_steps * len(input_token_list), -1)

        # Add future action query tokens
        len_additional = self.config.n_action_pred_token - 1
        if len_additional > 0:
            future_tokens = mx.broadcast_to(
                self.action_token,
                (batch_size, len_additional, self.config.gpt_input_dim),
            )
            input_tokens = mx.concatenate([input_tokens, future_tokens], axis=1)

        # GPT forward
        features = self.policy(input_tokens)

        # Extract action prediction token indices
        num_input_features = len(self.config.input_features) + 1  # +1 for action token type
        historical_act_pred_index = np.arange(0, n_obs_steps) * num_input_features + (num_input_features - 1)

        # Use mx array indexing (numpy arrays don't work as MLX indices)
        hist_idx = mx.array(historical_act_pred_index.tolist(), dtype=mx.int32)
        hist_features = features[:, hist_idx]
        if len_additional > 0:
            features = mx.concatenate(
                [hist_features, features[:, -len_additional:]],
                axis=1,
            )
        else:
            features = hist_features

        # Action head
        action_head_output = self.action_head(features)

        if rollout:
            action_dim = self.config.action_feature.shape[0]
            return action_head_output["predicted_action"][:, n_obs_steps - 1, :].reshape(
                batch_size, self.config.action_chunk_size, action_dim
            )
        else:
            # Gather target actions
            target = batch[ACTION][:, self.select_target_actions_indices.astype(mx.int32)]
            loss = self.action_head.loss_fn(action_head_output, target, reduction="mean")
            return action_head_output, loss


# =============================================================================
# VQBeTPolicy (top-level policy)
# =============================================================================

class VQBeTPolicy(Module):
    """VQ-BeT Policy as per 'Behavior Generation with Latent Actions'."""

    config_class = VQBeTConfig
    name = "vqbet"

    def __init__(self, config: VQBeTConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = VQBeTConfig()
        config.validate_features()
        self.config = config

        self.vqbet = VQBeTModel(config)
        self.reset()

    def reset(self):
        """Clear observation and action queues."""
        self._queues = {
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.action_chunk_size),
        }

    def predict_action_chunk(self, batch):
        batch_stacked = {
            k: mx.stack(list(self._queues[k]), axis=1)
            for k in batch if k in self._queues
        }
        actions = self.vqbet(batch_stacked, rollout=True)[:, :self.config.action_chunk_size]
        return actions

    def select_action(self, batch):
        """Select a single action given environment observations."""
        if ACTION in batch:
            batch = {k: v for k, v in batch.items() if k != ACTION}
        batch = dict(batch)
        batch[OBS_IMAGES] = mx.stack(
            [batch[key] for key in self.config.image_features], axis=-4
        )

        # Populate queues
        for key in self._queues:
            if key in batch:
                self._queues[key].append(batch[key])

        if not bool(self.vqbet.action_head.vqvae_model.discretized.item()):
            warnings.warn(
                "To evaluate in the environment, your VQ-BeT model should contain a pretrained Residual VQ.",
                stacklevel=1,
            )

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            # Fill action queue
            for i in range(actions.shape[1]):
                self._queues[ACTION].append(actions[:, i])

        action = self._queues[ACTION].popleft()
        return action

    def __call__(self, batch):
        """Run the batch through the model and compute the loss."""
        batch = dict(batch)
        batch[OBS_IMAGES] = mx.stack(
            [batch[key] for key in self.config.image_features], axis=-4
        )

        if not bool(self.vqbet.action_head.vqvae_model.discretized.item()):
            loss, n_different_codes, n_different_combinations, recon_l1_error = (
                self.vqbet.action_head.discretize(
                    self.config.n_vqvae_training_steps, batch[ACTION]
                )
            )
            return loss, {
                "n_different_codes": n_different_codes,
                "n_different_combinations": n_different_combinations,
                "recon_l1_error": recon_l1_error,
            }

        _, loss_dict = self.vqbet(batch, rollout=False)
        loss = loss_dict.pop("loss")
        return loss, loss_dict
