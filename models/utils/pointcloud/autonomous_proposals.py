"""Isolated replay of the 2026-10-27 autonomous legal candidate pool.

The default den6 pool stays in ana_den6_online.py without numerical changes.
"""
from __future__ import annotations

import hashlib
import math
import json
import time
from typing import Any, Mapping

import numpy as np
import torch

from .ana_den6_online import (
    FAST_UNORDERED_SOURCE, _FAST_NEIGHBOR_OFFSETS, _fast_spatial_source_indices,
)

_POOL_RAM = None
_POOL_RAM_CONTEXT = None


def build_autonomous_pool(
    context: Mapping[str, Any], args: Any, identity: Mapping[str, Any]
) -> dict[str, Any]:
    from ..training.sparsepcgc_autonomous_policy import autonomous_search_enabled, autonomous_pool_phase
    capacity = max(int(getattr(args, 'test_route_pool_cache_memory_mb', 128)), 0)*1024**2
    from ..loss.autonomous_codec_context import codec_context_enabled
    if codec_context_enabled(args):
        capacity = max(int(getattr(args, 'test_route_context_pool_cache_memory_mb', 768)), 0)*1024**2
    if (not autonomous_search_enabled(args) or not capacity
            or not bool(getattr(args, '_autonomous_policy_training', True))):
        return _build_autonomous_pool(context, args, identity)
    started = time.perf_counter()
    coords = context.get('full_global_voxel_coords', context.get('global_voxel_coords'))
    if not torch.is_tensor(coords) or coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[-1] == 0:
        return _build_autonomous_pool(context, args, identity)
    rows = coords[0].transpose(0, 1).detach().to(dtype=torch.long).contiguous()
    phase = autonomous_pool_phase(args)
    global _POOL_RAM, _POOL_RAM_CONTEXT
    if _POOL_RAM is None or _POOL_RAM_CONTEXT != (capacity, phase):
        from ..cache.lossless_tensor_ram import LosslessObjectRAM
        _POOL_RAM = LosslessObjectRAM(capacity)
        _POOL_RAM_CONTEXT = (capacity, phase)
    settings = dict(identity=identity, phase=phase,
        maximum=float(getattr(args, 'test_route_max_operation_ratio', .03)),
        reserve=float(getattr(args, 'heuristic_guidance_online_fast_pool_reserve_factor', 2.)),
        cap=int(getattr(args, 'heuristic_guidance_online_fast_pool_max_per_operation', 8192)),
        resolution=int(getattr(args, 'sparsepcgc_native_resolution', 1023)))
    if codec_context_enabled(args):
        settings['native_context'] = str(getattr(args, '_autonomous_native_context_path', '')).rsplit('/',1)[-1]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    # Content AND order: source sampling depends on row indices. Do not trust
    # just a file path, timestamp, tensor pointer, or input voxel count.
    digest.update(memoryview(rows.cpu().numpy()).cast('B'))
    key = digest.hexdigest()
    payload = _POOL_RAM.get(key)
    hit = payload is not None
    if not hit:
        payload = _build_autonomous_pool(context, args, identity, rows)
        _POOL_RAM.put(key, payload)
    elapsed = time.perf_counter()-started
    payload['autonomous_pool_cache_hit'] = hit
    payload['autonomous_pool_cache_stats'] = _POOL_RAM.stats()
    payload['elapsed_sec'] = elapsed
    if hit:
        payload['proposal_timing'] = dict(total=elapsed, setup=0., spatial_sampling=0.,
            candidate_enumeration=0., occupancy_screening=0., codec_proxy=0., geometry_risk=0.,
            deduplication_and_pool_build=0., lossless_cache_restore=elapsed)
    else:
        payload['proposal_timing']['total'] = elapsed
    return payload


def _build_autonomous_pool(
    context: Mapping[str, Any], args: Any, identity: Mapping[str, Any], rows=None
) -> dict[str, Any]:
    """Generate a legal, unordered pool without codec calls or full ranking."""
    started = time.perf_counter()
    coords = context.get("full_global_voxel_coords", context.get("global_voxel_coords"))
    if not torch.is_tensor(coords) or coords.ndim != 3 or coords.shape[0] != 1:
        raise RuntimeError("fast unordered proposal requires [1,3,N] voxel coords")
    if rows is None:
        rows = coords[0].transpose(0, 1).detach().to(dtype=torch.long).contiguous()
    point_count = int(rows.shape[0])
    if point_count <= 0:
        raise RuntimeError("fast unordered proposal received an empty cloud")

    maximum_ratio = float(getattr(args, 'test_route_max_operation_ratio', .03))
    total_ratio, shares = 1.5 * maximum_ratio, {name: 1. / 3. for name in ('Add', 'Prune', 'Adjust')}
    from ..training.sparsepcgc_autonomous_policy import autonomous_search_enabled, autonomous_pool_phase
    autonomous = autonomous_search_enabled(args)
    requested_total = max(int(round(point_count * total_ratio)), 1)
    reserve = max(float(getattr(
        args, "heuristic_guidance_online_fast_pool_reserve_factor", 2.0
    )), 1.0)
    configured_cap = int(getattr(
        args, "heuristic_guidance_online_fast_pool_max_per_operation", 8192
    ))
    budgets = {}
    for operation in ("Add", "Prune", "Adjust"):
        required = max(int(round(requested_total * shares[operation])), 1)
        if autonomous:
            # Every family can consume the full legal total budget. Pool size
            # is independent of the old dataset-specific ratio/share tables.
            required = max(int(math.ceil(point_count * maximum_ratio)), 1)
        budget = max(int(np.ceil(required * reserve)), required)
        if configured_cap > 0 and not autonomous:
            budget = min(budget, configured_cap)
        budgets[operation] = min(budget, point_count)
    setup_finished = time.perf_counter()

    seed = int(str(identity["input_sha256"])[:16], 16) & 0x7FFFFFFF
    if autonomous:
        seed = (seed + 104729 * autonomous_pool_phase(args)) & 0x7FFFFFFF
    sample_count = min(
        point_count,
        max(max(budgets.values()) * 2, max(budgets.values()) + 256),
    )
    sample_indices = _fast_spatial_source_indices(
        point_count, sample_count, seed, rows.device
    )
    sampled_rows = rows.index_select(0, sample_indices)
    sampling_finished = time.perf_counter()
    offsets = torch.as_tensor(
        _FAST_NEIGHBOR_OFFSETS, device=rows.device, dtype=torch.long
    )
    targets = sampled_rows[:, None, :] + offsets[None, :, :]
    enumeration_finished = time.perf_counter()

    membership_started = time.perf_counter()
    lower = rows.amin(dim=0) - 1
    upper = rows.amax(dim=0) + 1
    span = (upper - lower + 1).clamp_min(1)
    span_product = int(span[0].item()) * int(span[1].item()) * int(span[2].item())
    if span_product >= (1 << 62):
        raise RuntimeError("fast proposal coordinate span exceeds int64")

    def packed_keys(values: torch.Tensor) -> torch.Tensor:
        shifted = values - lower.view(*([1] * (values.ndim - 1)), 3)
        return (
            shifted[..., 0] * span[1] * span[2]
            + shifted[..., 1] * span[2]
            + shifted[..., 2]
        )

    occupied_keys = torch.sort(packed_keys(rows)).values
    target_keys = packed_keys(targets)
    flat_keys = target_keys.reshape(-1)
    positions = torch.searchsorted(occupied_keys, flat_keys)
    inside = positions < occupied_keys.numel()
    safe_positions = positions.clamp_max(max(int(occupied_keys.numel()) - 1, 0))
    occupied = inside & occupied_keys.index_select(0, safe_positions).eq(flat_keys)
    native_resolution = max(int(getattr(
        args, "sparsepcgc_native_resolution", 1023
    )), 1)
    in_domain = ((targets >= 0) & (targets <= native_resolution)).all(dim=-1)
    if autonomous:
        in_domain &= ((targets >= rows.amin(dim=0)) & (targets <= rows.amax(dim=0))).all(dim=-1)
    interior = ((sampled_rows > rows.amin(dim=0)) &
                (sampled_rows < rows.amax(dim=0))).all(dim=-1)
    empty = (~occupied.reshape_as(target_keys)) & in_domain
    neighbour_count = occupied.reshape_as(target_keys).sum(dim=1)
    membership_finished = time.perf_counter()

    def choose_pairs(budget: int, phase: int, protect_source: bool = False):
        starts = (sample_indices * 2654435761 + int(phase)) % len(_FAST_NEIGHBOR_OFFSETS)
        directions = torch.arange(
            len(_FAST_NEIGHBOR_OFFSETS), device=rows.device, dtype=torch.long
        ).view(1, -1)
        priority = (directions - starts.view(-1, 1)) % len(_FAST_NEIGHBOR_OFFSETS)
        priority = priority.masked_fill(~empty, len(_FAST_NEIGHBOR_OFFSETS) + 1)
        selected_direction = priority.argmin(dim=1)
        valid = empty.any(dim=1)
        if protect_source:
            valid = valid & interior
        valid_rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if int(valid_rows.numel()) > int(budget):
            valid_rows = valid_rows[: int(budget)]
        source_index = sample_indices.index_select(0, valid_rows)
        direction_index = selected_direction.index_select(0, valid_rows)
        source = rows.index_select(0, source_index)
        target = source + offsets.index_select(0, direction_index)
        degree = neighbour_count.index_select(0, valid_rows).float()
        return direction_index, source, target, degree

    serialization_started = time.perf_counter()
    pools: dict[str, list[dict[str, Any]]] = {}
    prune_indices = sample_indices[: budgets["Prune"]]
    if autonomous:
        prune_indices = sample_indices[interior][:budgets['Prune']]
    prune_rows = rows.index_select(0, prune_indices)
    prune_degree = (neighbour_count[interior][:int(prune_indices.numel())].float()
                    if autonomous else neighbour_count[:int(prune_indices.numel())].float())
    from ..loss.autonomous_codec_context import codec_context_enabled
    native_arrays=codec_context_enabled(args)
    if native_arrays:
        from .autonomous_context_arrays import SpatialCandidates,first_rows
        source_array=prune_rows.detach().cpu().numpy()
        pools['Prune']=SpatialCandidates(np.concatenate((source_array,np.zeros_like(source_array)),1),
            prune_degree.detach().cpu().numpy().astype(np.float64)/26.)
    else:
        pools["Prune"] = [
        {
            "candidate_id": f"fast:Prune:{x}:{y}:{z}",
            "operation": "Prune", "pool_rank": index, "rank_score": 0.0,
            "heuristic_score": 0.0, "remove_coords": [[x, y, z]],
            "add_coords": [], "affected_voxel_cells": 1,
            "operation_count": 1,
            "optimistic_gain_bits": float((26.0 - degree) / 26.0),
            "neighbor_bit_risk": 0.0, "geometry_cost": float(degree / 26.0),
        }
        for index, ((x, y, z), degree) in enumerate(
            zip(prune_rows.cpu().tolist(), prune_degree.cpu().tolist())
        )
        ]
    for operation, phase in (("Add", seed + 17), ("Adjust", seed + 43)):
        direction_index, source, target, degree = choose_pairs(
            budgets[operation], phase, protect_source=(autonomous and operation == 'Adjust')
        )
        if native_arrays:
            source_array=source.detach().cpu().numpy()
            target_array=target.detach().cpu().numpy()
            # The original spatial generator deduplicates TARGETS, even for
            # Adjust. Preserve the first source/direction for each target.
            keep=first_rows(target_array)
            geo=degree.detach().cpu().numpy().astype(np.float64)
            geo=(26.-geo)/26. if operation=='Add' else geo/26.
            remove=source_array if operation=='Adjust' else np.zeros_like(target_array)
            pools[operation]=SpatialCandidates(np.concatenate((remove,target_array),1)[keep],geo[keep])
            continue
        pool = []
        seen_targets = set()
        for source_row, target_row, local_degree, direction in zip(
            source.cpu().tolist(), target.cpu().tolist(), degree.cpu().tolist(),
            direction_index.cpu().tolist(),
        ):
            # Tensor.tolist already produced Python ints. Preserve exactly
            # the same coordinates without rebuilding six scalar generators.
            source_tuple = tuple(source_row)
            target_tuple = tuple(target_row)
            if target_tuple in seen_targets:
                continue
            seen_targets.add(target_tuple)
            index = len(pool)
            pool.append({
                "candidate_id": (
                    f"fast:{operation}:{source_tuple[0]}:{source_tuple[1]}:"
                    f"{source_tuple[2]}:{int(direction)}"
                ),
                "operation": operation, "pool_rank": index, "rank_score": 0.0,
                "heuristic_score": 0.0,
                "remove_coords": [list(source_tuple)] if operation == "Adjust" else [],
                "add_coords": [list(target_tuple)], "affected_voxel_cells": 1,
                "operation_count": 1,
                "optimistic_gain_bits": float(abs(local_degree - 13.0) / 13.0),
                "neighbor_bit_risk": 0.0,
                "geometry_cost": float(
                    (26.0 - local_degree) / 26.0
                    if operation == "Add" else local_degree / 26.0
                ),
            })
        pools[operation] = pool
    native_context_path = ''
    context_diagnostics = {}
    if codec_context_enabled(args):
        native_context_path = str(getattr(args, '_autonomous_native_context_path', ''))
        if not native_context_path:
            raise RuntimeError('Prepare the current GT native codec context before autonomous Network.forward')
        from .autonomous_context_proposals import context_proposals
        pools, context_diagnostics = context_proposals(rows.detach().cpu().numpy(), native_context_path,
            max(budgets.values()), int(str(identity['input_sha256'])[:16],16)&0x7fffffff,
            random_pools=pools)
    if any(not pools[name] for name in ("Add", "Prune", "Adjust")):
        raise RuntimeError("fast unordered proposal produced an empty operation pool")
    serialization_finished = time.perf_counter()

    elapsed = time.perf_counter() - started
    cache_signature = hashlib.sha256("|".join((
        FAST_UNORDERED_SOURCE, str(identity["input_sha256"]),
        str(identity["setting_id"]), str(reserve), str(configured_cap),
        *(('autonomous', str(seed), str(budgets)) if autonomous else ()),
        *((native_context_path.rsplit('/',1)[-1],) if native_context_path else ()),
    )).encode("utf-8")).hexdigest()
    return {
        **dict(identity),
        "schema_version": FAST_UNORDERED_SOURCE,
        "source": FAST_UNORDERED_SOURCE,
        # Candidate legality is checked from the current canonical Tensor
        # above.  Re-sorting all ~1M voxels on CPU merely to make another hash
        # would dominate this lightweight proposal, so the already verified
        # input content hash identifies the frame here.
        "input_voxel_hash": str(identity["input_sha256"]),
        "total_ratio": float(total_ratio),
        "operation_shares": shares,
        "operation_heuristics": {
            name: "safety_spatial_sampling_no_rank" for name in pools
        },
        "operation_priority": ["Add", "Prune", "Adjust"],
        "operation_edit_units": pools,
        "heuristic_anchor_plan": {"available": False},
        "proposal_policy": "fast_spatial_unordered_safe_pool_v1",
        "full_plan_candidate_count": 1,
        "actual_candidate_encode_count": 0,
        "cache_path": "", "cache_signature": cache_signature,
        "elapsed_sec": float(elapsed),
        "proposal_timing": {
            "total": float(elapsed),
            "setup": float(setup_finished - started),
            "spatial_sampling": float(sampling_finished - setup_finished),
            "candidate_enumeration": float(
                enumeration_finished - sampling_finished
            ),
            "occupancy_screening": float(
                membership_finished - membership_started
            ),
            # The fast path deliberately has neither a SparsePCGC candidate
            # probe nor a codec-derived ranking pass.
            "codec_proxy": 0.0,
            "geometry_risk": 0.0,
            "deduplication_and_pool_build": float(
                serialization_finished - serialization_started
            ),
        },
        "proposal_pool_counts": {name: len(value) for name, value in pools.items()},
        "proposal_sampled_source_count": int(sample_count),
        "contains_candidate_actual": False,
        "contains_heuristic_ranking": False,
        "autonomous_search": autonomous,
        "native_codec_context": bool(native_context_path),
        "native_context_diagnostics": context_diagnostics,
        "pool_phase": autonomous_pool_phase(args) if autonomous else 0,
    }

