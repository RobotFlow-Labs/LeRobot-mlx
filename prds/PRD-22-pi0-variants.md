# PRD-22: Pi0.5 + Pi0-FAST Variants

> **Status:** TODO
> **Priority:** P2 — Completes the Pi0 family
> **Dependencies:** PRD-16 (Pi0), PRD-19 (VLM backbone)
> **Estimated LOC:** ~400

---

## Objective

Port the Pi0.5 and Pi0-FAST variants. Both share 90% of Pi0's architecture.

### Pi0.5
- Same as Pi0 but uses QUANTILES normalization instead of MEAN_STD
- Different tokenizer max length (200 vs 48)
- Minimal code changes from Pi0

### Pi0-FAST
- Uses action tokenization instead of continuous values
- Autoregressive decoding with KV cache
- `lerobot/fast-action-tokenizer` for discretizing actions
- More complex inference loop (token-by-token generation)

---

## Deliverables

### Pi0.5
1. `src/lerobot_mlx/policies/pi05/__init__.py`
2. `src/lerobot_mlx/policies/pi05/configuration_pi05.py`
3. `src/lerobot_mlx/policies/pi05/modeling_pi05.py` (inherits from Pi0, overrides normalization)
4. `tests/test_pi05.py` — 10+ tests

### Pi0-FAST
1. `src/lerobot_mlx/policies/pi0_fast/__init__.py`
2. `src/lerobot_mlx/policies/pi0_fast/configuration_pi0_fast.py`
3. `src/lerobot_mlx/policies/pi0_fast/modeling_pi0_fast.py` (adds token-based action generation)
4. `tests/test_pi0_fast.py` — 15+ tests

---

## Acceptance Criteria

1. Pi0.5 forward pass works with QUANTILES normalization
2. Pi0-FAST autoregressive action generation works
3. Both share Pi0's expert architecture
4. 25+ tests total
