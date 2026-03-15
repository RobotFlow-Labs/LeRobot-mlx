# PRD-15: SARM Policy

> **Status:** TODO
> **Priority:** P2 — TransformerEncoder-heavy, good coverage
> **Dependencies:** PRD-02, PRD-03, PRD-04
> **Estimated LOC:** ~600 (upstream ~2729 LOC)
> **Phase:** 5 (Extended Policies)

---

## Objective

Port SARM — reward-based policy using TransformerEncoder for state prediction. Medium complexity, exercises TransformerEncoder compat layer thoroughly.

---

## Key Components

- TransformerEncoder (multiple layers)
- TransformerEncoderLayer with pre-norm
- Custom state prediction heads
- Reward modeling

## Acceptance Criteria

1. Forward pass produces correct output shapes
2. TransformerEncoder compat handles all attention patterns
3. Training loss decreases on synthetic data
4. 10+ tests passing
