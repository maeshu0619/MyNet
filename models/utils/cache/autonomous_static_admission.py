"""Predict CPU clone bytes without performing an otherwise rejected transfer."""
import torch


def static_clone_bytes(value, seen=None):
    """Match _clone_static_cache_value's tensor-object alias semantics.

    Distinct views are cloned into distinct storages, while repeated references
    to the same Tensor share one clone. Counting source storage pointers would
    underestimate the former, so it is deliberately not used here.
    """
    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(static_clone_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(static_clone_bytes(v, seen) for v in value)
    return 0
