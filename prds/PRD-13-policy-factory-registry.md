# PRD-13: Policy Factory & Registry

> **Status:** TODO
> **Priority:** P1 — Required for unified API
> **Dependencies:** PRD-05 through PRD-11 (policies)
> **Estimated LOC:** ~200
> **Phase:** 4 (Infrastructure)

---

## Objective

Mirror upstream's `factory.py` — the policy registry that maps config types to policy classes. This enables `make_policy(config)` to work with any ported policy.

---

## Upstream Pattern

```python
# Upstream factory.py
POLICY_CLASSES = {
    "act": ACTPolicy,
    "diffusion": DiffusionPolicy,
    "tdmpc": TDMPCPolicy,
    "vqbet": VQBeTPolicy,
    "sac": SACPolicy,
}

def make_policy(config, **kwargs):
    policy_type = config.type
    cls = POLICY_CLASSES[policy_type]
    return cls(config, **kwargs)
```

## Our Mirror

```python
# src/lerobot_mlx/policies/factory.py

from lerobot_mlx.policies.act.modeling_act import ACTPolicy
from lerobot_mlx.policies.diffusion.modeling_diffusion import DiffusionPolicy
# ... lazy imports to avoid loading all policies

POLICY_REGISTRY = {}

def register_policy(name, cls):
    POLICY_REGISTRY[name] = cls

def make_policy(config, **kwargs):
    """Create policy from config. Matches upstream API."""
    policy_type = getattr(config, 'type', None) or config.__class__.__name__.lower().replace('config', '')
    if policy_type not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy type: {policy_type}. "
                         f"Available: {list(POLICY_REGISTRY.keys())}")
    cls = POLICY_REGISTRY[policy_type]
    return cls(config, **kwargs)

# Register all ported policies
def _register_all():
    """Lazy registration to avoid import-time heavy loads."""
    try:
        from lerobot_mlx.policies.act.modeling_act import ACTPolicy
        register_policy("act", ACTPolicy)
    except ImportError:
        pass
    # ... repeat for each policy

_register_all()
```

---

## Acceptance Criteria

1. `make_policy(act_config)` returns ACTPolicy instance
2. `make_policy(diffusion_config)` returns DiffusionPolicy instance
3. Unknown policy type raises clear error with available options
4. Lazy imports: importing factory.py doesn't load all policy code
5. 10+ tests passing
