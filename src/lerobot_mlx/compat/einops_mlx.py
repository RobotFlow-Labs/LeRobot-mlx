# Copyright (c) 2025 AIFLOW LABS / RobotFlow Labs
# LeRobot-MLX: einops rearrange/repeat replacement using MLX
#
# Implements the specific einops patterns used in LeRobot (~6 patterns),
# not the full einops API. Patterns are parsed and executed using
# mx.reshape and mx.transpose.

from typing import Any, Dict, List, Tuple, Union

import mlx.core as mx


__all__ = ["rearrange", "repeat", "reduce"]


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

_Token = Tuple[str, Union[str, List[str]]]  # ('dim', 'b') or ('group', ['c', 'h', 'w'])


# ---------------------------------------------------------------------------
# Pattern parser
# ---------------------------------------------------------------------------

def _parse_pattern(pattern: str) -> List[_Token]:
    """Parse a pattern string like 'b (h w) c' into structured tokens.

    Returns a list of tokens where each is either:
      ('dim', name)       - a single dimension like 'b'
      ('group', [names])  - a grouped dimension like '(h w)'
      ('lit', '1')        - a literal size-1 dimension
    """
    tokens: List[_Token] = []
    i = 0
    s = pattern.strip()
    while i < len(s):
        ch = s[i]
        if ch == " ":
            i += 1
            continue
        if ch == "(":
            j = s.index(")", i)
            group_names = s[i + 1 : j].split()
            tokens.append(("group", group_names))
            i = j + 1
        elif ch.isdigit():
            # literal like '1'
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(("lit", s[i:j]))
            i = j
        elif ch.isalpha() or ch == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(("dim", s[i:j]))
            i = j
        else:
            i += 1
    return tokens


def _flat_names(tokens: List[_Token]) -> List[str]:
    """Flatten tokens into a list of dimension names."""
    names: List[str] = []
    for kind, val in tokens:
        if kind == "dim":
            names.append(val)
        elif kind == "group":
            names.extend(val)
        elif kind == "lit":
            names.append(f"__lit_{val}")
    return names


def _resolve_sizes(
    tokens: List[_Token],
    shape: Tuple[int, ...],
    axes_lengths: Dict[str, int],
) -> Dict[str, int]:
    """Resolve each dimension name to a concrete size.

    Uses the tensor shape and any provided axes_lengths kwargs.
    """
    sizes: Dict[str, int] = dict(axes_lengths)
    shape_idx = 0

    for kind, val in tokens:
        if kind == "dim":
            name = val
            if name not in sizes:
                sizes[name] = shape[shape_idx]
            shape_idx += 1
        elif kind == "group":
            # The group occupies one axis in the input shape
            group_size = shape[shape_idx]
            shape_idx += 1
            # Try to resolve sub-dimensions
            unknown = [n for n in val if n not in sizes]
            known_product = 1
            for n in val:
                if n in sizes:
                    known_product *= sizes[n]
            if len(unknown) == 1:
                sizes[unknown[0]] = group_size // known_product
            elif len(unknown) == 0:
                # All known, verify
                pass
            else:
                raise ValueError(
                    f"Cannot resolve grouped dims {val} with shape axis "
                    f"{group_size}. Provide more axes_lengths."
                )
        elif kind == "lit":
            lit_val = int(val)
            sizes[f"__lit_{val}"] = lit_val
            shape_idx += 1

    return sizes


# ---------------------------------------------------------------------------
# Rearrange
# ---------------------------------------------------------------------------

def rearrange(tensor: mx.array, pattern: str, **axes_lengths: int) -> mx.array:
    """Rearrange tensor dimensions following an einops-style pattern.

    Supports the common patterns used in LeRobot:
      1. 'b c h w -> b (c h w)'           — flatten spatial dims
      2. 'b (h w) c -> b h w c'           — unflatten with known h, w
      3. 'b c h w -> b h w c'             — channel reorder (transpose)
      4. 'b t c -> (b t) c'               — merge batch and time
      5. '(b t) c -> b t c'               — split batch and time
      6. '1 c -> b c'                      — broadcast (use repeat instead)

    Args:
        tensor: Input MLX array.
        pattern: Einops-style pattern string like 'b c h w -> b (c h w)'.
        **axes_lengths: Named dimension sizes for ambiguous dimensions.

    Returns:
        Rearranged MLX array.
    """
    if "->" not in pattern:
        raise ValueError(f"Pattern must contain '->': {pattern}")

    lhs_str, rhs_str = pattern.split("->")
    lhs_tokens = _parse_pattern(lhs_str)
    rhs_tokens = _parse_pattern(rhs_str)

    # Resolve all dimension sizes
    sizes = _resolve_sizes(lhs_tokens, tensor.shape, axes_lengths)

    # Get flat name lists
    lhs_flat = _flat_names(lhs_tokens)
    rhs_flat = _flat_names(rhs_tokens)

    # Step 1: If LHS has groups, reshape to fully expanded form
    expanded_shape = tuple(sizes[n] for n in lhs_flat)
    if len(expanded_shape) != len(tensor.shape):
        tensor = mx.reshape(tensor, expanded_shape)

    # Step 2: If dim order differs, transpose
    if lhs_flat != rhs_flat:
        # Check that sets are equal (rearrange, not reduce/repeat)
        if set(lhs_flat) != set(rhs_flat):
            raise ValueError(
                f"Rearrange requires same dims on both sides. "
                f"LHS: {lhs_flat}, RHS: {rhs_flat}"
            )
        # Build permutation
        perm = [lhs_flat.index(n) for n in rhs_flat]
        tensor = mx.transpose(tensor, axes=perm)

    # Step 3: If RHS has groups, reshape to merge
    rhs_shape = []
    for kind, val in rhs_tokens:
        if kind == "dim":
            rhs_shape.append(sizes[val])
        elif kind == "group":
            product = 1
            for n in val:
                product *= sizes[n]
            rhs_shape.append(product)
        elif kind == "lit":
            rhs_shape.append(int(val))

    if tuple(tensor.shape) != tuple(rhs_shape):
        tensor = mx.reshape(tensor, rhs_shape)

    return tensor


# ---------------------------------------------------------------------------
# Repeat
# ---------------------------------------------------------------------------

def repeat(tensor: mx.array, pattern: str, **axes_lengths: int) -> mx.array:
    """Repeat/broadcast tensor dimensions following an einops-style pattern.

    Common patterns:
      - '1 c -> b c'    with b=32  — broadcast and repeat
      - 'b c -> b t c'  with t=10  — insert and repeat new axis

    Args:
        tensor: Input MLX array.
        pattern: Einops-style repeat pattern.
        **axes_lengths: Named dimension sizes, including new dimensions.

    Returns:
        Repeated MLX array.
    """
    if "->" not in pattern:
        raise ValueError(f"Pattern must contain '->': {pattern}")

    lhs_str, rhs_str = pattern.split("->")
    lhs_tokens = _parse_pattern(lhs_str)
    rhs_tokens = _parse_pattern(rhs_str)

    # Build a mapping from LHS positional dims to RHS dims.
    # For literals like '1', we map to the corresponding RHS dim name.
    lhs_flat = _flat_names(lhs_tokens)
    rhs_flat = _flat_names(rhs_tokens)

    # Resolve sizes from LHS using tensor shape
    sizes = _resolve_sizes(lhs_tokens, tensor.shape, axes_lengths)

    # Add any new dims from axes_lengths that appear only on RHS
    for name in rhs_flat:
        if name not in sizes:
            if name in axes_lengths:
                sizes[name] = axes_lengths[name]
            else:
                raise ValueError(
                    f"Cannot determine size of new dimension '{name}'. "
                    f"Provide it in axes_lengths."
                )

    # Build a name mapping: literal dims on LHS correspond to RHS dims
    # by position (e.g., '1 c -> b c' means __lit_1 maps to b)
    # Create a working list of names for the current tensor
    # Strategy: rename literals to their RHS counterparts if they are
    # at the same position and the literal has size 1.
    working_names = list(lhs_flat)
    for i, name in enumerate(working_names):
        if name.startswith("__lit_"):
            # Find which RHS dim this corresponds to by looking at
            # dims shared between LHS and RHS
            # The literal occupies a position — find a RHS dim not in
            # working_names that should go at this position
            for rhs_name in rhs_flat:
                if rhs_name not in working_names and rhs_name not in lhs_flat:
                    if rhs_name in sizes or rhs_name in axes_lengths:
                        working_names[i] = rhs_name
                        break

    # Step 1: Expand LHS groups if needed
    expanded_shape = tuple(sizes.get(n, 1) for n in lhs_flat)
    current_tensor = tensor
    if len(expanded_shape) != len(tensor.shape):
        current_tensor = mx.reshape(current_tensor, expanded_shape)

    # Step 2: Insert new dimensions not present in working_names
    current_names = list(working_names)
    for i, name in enumerate(rhs_flat):
        if name not in current_names:
            current_tensor = mx.expand_dims(current_tensor, axis=i)
            current_names.insert(i, name)

    # Step 3: Reorder if needed
    if current_names != rhs_flat:
        perm = [current_names.index(n) for n in rhs_flat]
        current_tensor = mx.transpose(current_tensor, axes=perm)

    # Step 4: Build broadcast shape and broadcast
    broadcast_shape = tuple(sizes[name] for name in rhs_flat)
    current_tensor = mx.broadcast_to(current_tensor, broadcast_shape)

    # Step 5: Merge groups on RHS if any
    target_shape = []
    for kind, val in rhs_tokens:
        if kind == "dim":
            target_shape.append(sizes[val])
        elif kind == "group":
            product = 1
            for n in val:
                product *= sizes[n]
            target_shape.append(product)
        elif kind == "lit":
            target_shape.append(int(val))

    if tuple(current_tensor.shape) != tuple(target_shape):
        current_tensor = mx.reshape(current_tensor, target_shape)

    return current_tensor


# ---------------------------------------------------------------------------
# Reduce (minimal implementation)
# ---------------------------------------------------------------------------

def reduce(
    tensor: mx.array,
    pattern: str,
    reduction: str = "mean",
    **axes_lengths: int,
) -> mx.array:
    """Reduce tensor dimensions following an einops-style pattern.

    Args:
        tensor: Input MLX array.
        pattern: Einops-style pattern where RHS has fewer dims than LHS.
        reduction: One of 'mean', 'sum', 'max', 'min'.
        **axes_lengths: Named dimension sizes.

    Returns:
        Reduced MLX array.
    """
    if "->" not in pattern:
        raise ValueError(f"Pattern must contain '->': {pattern}")

    lhs_str, rhs_str = pattern.split("->")
    lhs_tokens = _parse_pattern(lhs_str)
    rhs_tokens = _parse_pattern(rhs_str)

    lhs_flat = _flat_names(lhs_tokens)
    rhs_flat = _flat_names(rhs_tokens)

    # Resolve sizes
    sizes = _resolve_sizes(lhs_tokens, tensor.shape, axes_lengths)

    # Expand groups
    expanded_shape = tuple(sizes[n] for n in lhs_flat)
    if len(expanded_shape) != len(tensor.shape):
        tensor = mx.reshape(tensor, expanded_shape)

    # Find axes to reduce
    reduce_axes = []
    for i, name in enumerate(lhs_flat):
        if name not in rhs_flat:
            reduce_axes.append(i)

    if not reduce_axes:
        return tensor

    reduce_fn = {
        "mean": mx.mean,
        "sum": mx.sum,
        "max": lambda x, axis: mx.max(x, axis=axis),
        "min": lambda x, axis: mx.min(x, axis=axis),
    }

    if reduction not in reduce_fn:
        raise ValueError(f"Unknown reduction: {reduction}")

    result = reduce_fn[reduction](tensor, axis=tuple(reduce_axes))

    # Reorder remaining dims if needed
    remaining = [n for n in lhs_flat if n not in set(lhs_flat[i] for i in reduce_axes)]
    if remaining != rhs_flat:
        perm = [remaining.index(n) for n in rhs_flat]
        result = mx.transpose(result, axes=perm)

    return result
