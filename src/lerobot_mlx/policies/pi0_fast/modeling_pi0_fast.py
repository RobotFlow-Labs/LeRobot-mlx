#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
"""Pi0-FAST: Token-based action generation with autoregressive decoding.

Instead of continuous flow matching (Pi0), Pi0-FAST discretizes actions
into tokens and generates them autoregressively via the Gemma expert.

Architecture:
    Pi0FastPolicy -> Pi0FastModel -> GemmaExpert (from Pi0)
                                  -> action token embedding
                                  -> token prediction head
                                  -> action detokenizer (tokens -> continuous)

Training:
    1. Quantize continuous actions to discrete tokens
    2. Embed tokens and concatenate with state conditioning
    3. Teacher-forced autoregressive prediction
    4. Cross-entropy loss on next-token prediction

Inference:
    1. Start with state conditioning tokens
    2. Autoregressively generate action tokens (greedy or sampled)
    3. Detokenize: convert discrete tokens back to continuous actions
"""

from collections import deque

import mlx.core as mx
import mlx.nn as _nn

from lerobot_mlx.policies.pi0.modeling_pi0 import GemmaExpert
from lerobot_mlx.policies.pi0_fast.configuration_pi0_fast import Pi0FastConfig


# ---------------------------------------------------------------------------
# Pi0-FAST Model -- autoregressive action token generation
# ---------------------------------------------------------------------------

class Pi0FastModel(_nn.Module):
    """Autoregressive action token model.

    Uses a Gemma expert (shared architecture with Pi0) to predict action
    tokens one at a time. Actions are discretized via uniform quantization
    and detokenized back to continuous values.
    """

    def __init__(self, config: Pi0FastConfig):
        super().__init__()
        self.config = config
        hidden = config.expert_hidden_size

        # State projection (same as Pi0)
        self.state_proj = _nn.Linear(config.max_state_dim, hidden)

        # Token embedding for actions
        self.action_token_embed = _nn.Embedding(config.action_vocab_size, hidden)

        # Expert transformer (reuse architecture from Pi0)
        self.expert = GemmaExpert(config)

        # Token prediction head
        self.token_head = _nn.Linear(hidden, config.action_vocab_size)

        # Action detokenizer (discrete tokens -> continuous actions)
        self.action_detokenizer = _nn.Linear(config.action_vocab_size, config.max_action_dim)

    def __call__(self, batch: dict) -> dict:
        """Training forward: predict action tokens via cross-entropy.

        Args:
            batch: Dictionary with keys:
                - 'observation.state': (B, state_dim) or (B, 1, state_dim)
                - 'action': (B, T, A) continuous actions

        Returns:
            Dict with 'loss' and 'logits'.
        """
        state = batch.get("observation.state")
        actions = batch.get("action")  # (B, T, A)

        B, T, A = actions.shape

        # Quantize actions to tokens (simple uniform quantization)
        action_tokens = self._tokenize_actions(actions)  # (B, T)

        # Embed tokens
        token_embs = self.action_token_embed(action_tokens)  # (B, T, H)

        # Add state conditioning
        if state.ndim == 3:
            state = state.squeeze(1)
        state_emb = self.state_proj(state)  # (B, H)
        state_emb = mx.expand_dims(state_emb, axis=1)  # (B, 1, H)

        # Concatenate state + action tokens (teacher forcing)
        # Input: [state, tok_0, tok_1, ..., tok_{T-2}]
        # Target: [tok_0, tok_1, ..., tok_{T-1}]
        input_seq = mx.concatenate([state_emb, token_embs[:, :-1]], axis=1)  # (B, T, H)

        # Create causal mask
        seq_len = input_seq.shape[1]
        causal_mask = _nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        # Allow all tokens to attend to the first token (state)
        causal_mask = causal_mask.at[:, 0].add(-causal_mask[:, 0])

        # Run through expert
        expert_out = self.expert(input_seq, mask=causal_mask)  # (B, T, H)

        # Predict next token logits
        action_logits = self.token_head(expert_out)  # (B, T, V)

        # Cross-entropy loss
        targets = action_tokens  # (B, T) -- shifted alignment: input[i] predicts target[i]
        # Reshape for cross-entropy computation
        logits_flat = action_logits.reshape(-1, self.config.action_vocab_size)  # (B*T, V)
        targets_flat = targets.reshape(-1)  # (B*T,)

        # Numerically stable cross-entropy via log-softmax
        log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
        # Gather log probs for target tokens
        target_log_probs = mx.take_along_axis(
            log_probs, targets_flat[:, None], axis=-1
        ).squeeze(-1)
        loss = -mx.mean(target_log_probs)

        return {"loss": loss, "logits": action_logits}

    def generate_actions(
        self,
        batch: dict,
        temperature: float | None = None,
    ) -> mx.array:
        """Autoregressive action generation.

        Args:
            batch: Dictionary with 'observation.state'.
            temperature: Sampling temperature. 0 = greedy (default from config).

        Returns:
            (B, chunk_size, max_action_dim) continuous actions.
        """
        if temperature is None:
            temperature = self.config.temperature

        state = batch.get("observation.state")
        if state.ndim == 3:
            state = state.squeeze(1)
        B = state.shape[0]

        state_emb = self.state_proj(state)  # (B, H)
        state_emb = mx.expand_dims(state_emb, axis=1)  # (B, 1, H)

        # Start with state tokens
        generated_tokens = []
        input_seq = state_emb

        n_steps = min(self.config.max_decoding_steps, self.config.chunk_size)
        for step in range(n_steps):
            # Create causal mask for current sequence length
            seq_len = input_seq.shape[1]
            causal_mask = _nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            causal_mask = causal_mask.at[:, 0].add(-causal_mask[:, 0])

            expert_out = self.expert(input_seq, mask=causal_mask)
            next_logits = self.token_head(expert_out[:, -1:])  # (B, 1, V)

            if temperature <= 0:
                next_token = mx.argmax(next_logits, axis=-1)  # (B, 1) -- Greedy
            else:
                probs = mx.softmax(next_logits / temperature, axis=-1)
                # Sample from categorical distribution
                next_token = mx.random.categorical(
                    mx.log(probs + 1e-8).reshape(B, -1)
                )  # (B,)
                next_token = next_token[:, None]  # (B, 1)

            generated_tokens.append(next_token.squeeze(-1))  # (B,)

            # Append to input sequence
            next_emb = self.action_token_embed(next_token)  # (B, 1, H)
            input_seq = mx.concatenate([input_seq, next_emb], axis=1)
            mx.eval(input_seq)

        # Stack tokens: (B, T)
        token_ids = mx.stack(generated_tokens, axis=1)  # (B, T)

        # Detokenize: convert token indices to continuous actions
        one_hot = mx.zeros((B, token_ids.shape[1], self.config.action_vocab_size))
        # Build one-hot encoding
        for t in range(token_ids.shape[1]):
            indices = token_ids[:, t]  # (B,)
            scatter = mx.zeros((B, self.config.action_vocab_size))
            scatter = scatter.at[mx.arange(B), indices].add(1.0)
            one_hot = one_hot.at[:, t, :].add(scatter)

        actions = self.action_detokenizer(one_hot)  # (B, T, A)

        return actions

    def _tokenize_actions(self, actions: mx.array) -> mx.array:
        """Simple uniform quantization of continuous actions to discrete tokens.

        Maps actions assumed in [-1, 1] to integer tokens in [0, vocab_size-1].

        Args:
            actions: (B, T, A) continuous actions.

        Returns:
            (B, T) integer tokens (uses first action dimension for simplicity).
        """
        # Use first action dimension for token sequence
        first_dim = actions[:, :, 0]  # (B, T)
        # Normalize to [0, 1] range (assume actions in [-1, 1])
        normalized = (first_dim + 1) / 2  # [0, 1]
        # Quantize to vocab_size bins
        tokens = mx.clip(
            (normalized * (self.config.action_vocab_size - 1)).astype(mx.int32),
            0,
            self.config.action_vocab_size - 1,
        )
        return tokens


# ---------------------------------------------------------------------------
# Pi0-FAST Policy (top-level wrapper)
# ---------------------------------------------------------------------------

class Pi0FastPolicy(_nn.Module):
    """Top-level Pi0-FAST policy -- wraps autoregressive action token model.

    Provides select_action (inference with action queue) and forward (training).
    """

    config_class = Pi0FastConfig
    name = "pi0_fast"

    def __init__(self, config: Pi0FastConfig | None = None):
        super().__init__()
        if config is None:
            config = Pi0FastConfig()
        self.config = config
        self.model = Pi0FastModel(config)
        self._action_queue: deque = deque(maxlen=config.n_action_steps)

    def reset(self):
        """Reset the action queue (call between episodes)."""
        self._action_queue.clear()

    def select_action(self, batch: dict) -> mx.array:
        """Generate actions via autoregressive token generation, with action queue.

        On first call (or after queue depletes), generates a full chunk
        of actions and caches them. Subsequent calls pop from the queue.

        Args:
            batch: Dictionary with observation data.

        Returns:
            Single action array of shape (B, action_dim).
        """
        if len(self._action_queue) == 0:
            actions = self.model.generate_actions(batch)
            # actions: (B, chunk_size, action_dim)
            n_steps = min(self.config.n_action_steps, actions.shape[1])
            for i in range(n_steps):
                self._action_queue.append(actions[:, i, :])

        return self._action_queue.popleft()

    def __call__(self, batch: dict) -> dict:
        """Training forward: compute cross-entropy loss on action tokens.

        Args:
            batch: Dictionary with 'observation.state' and 'action'.

        Returns:
            Dict with 'loss' and 'logits' keys.
        """
        return self.model(batch)
