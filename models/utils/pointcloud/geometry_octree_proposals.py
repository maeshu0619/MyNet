"""Codec-independent legal Add/Prune/Adjust pools on conventional octrees.

No SparsePCGC cache, learned SparsePCGC symbol probability or codec worker is
consulted.  Local occupancy costs are explicitly proxies; the selected codec's
executed composite plan supplies the Actual training target.
"""
import hashlib
import itertools
import json
import math
import time

import torch
from .geometry_octree import codec_key, geometry_step

SOURCE = 'mynet_geometry_octree_safe_pool_v1'


def attach_geometry_octree_guidance(context, args, *, device):
    started = time.perf_counter()
    coords = context.get('full_global_voxel_coords', context.get('global_voxel_coords'))
    if not torch.is_tensor(coords) or coords.ndim != 3 or coords.shape[:2] != (1, 3):
        raise ValueError('conventional candidate pools require [1,3,N] canonical coordinates')
    rows = torch.unique(coords[0].T.long(), dim=0, sorted=True)
    n = int(rows.shape[0])
    if n == 0:
        raise ValueError('cannot propose edits for an empty octree')
    total = float(getattr(args, 'geometry_octree_initial_total_ratio', 0.0025))
    override = float(getattr(args, 'heuristic_guidance_total_ratio_percent', -1.))
    if override >= 0:
        total = override / 100.
    raw = str(getattr(args, 'heuristic_guidance_operation_shares', '') or '0.4,0.4,0.2')
    shares = [float(v) for v in raw.split(',')]
    if len(shares) != 3 or min(shares) <= 0 or not all(math.isfinite(v) for v in shares):
        raise ValueError('operation shares must be three finite positive values')
    shares = [v / sum(shares) for v in shares]
    reserve = max(float(getattr(args, 'geometry_octree_pool_reserve_factor', 4.)), 1.)
    cap = max(int(getattr(args, 'geometry_octree_pool_max_per_operation', 8192)), 1)
    budgets = {op: min(n, cap, max(int(math.ceil(n * total * share * reserve)), 1))
               for op, share in zip(('Add', 'Prune', 'Adjust'), shares)}
    # Spatial strata are deterministic for a fixed cloud, so Actual comparisons
    # revisit the same pool.  Exploration remains in the shared Actor sampler.
    requested = min(n, max(budgets.values()) * 3 + 256)
    ids = torch.div(torch.arange(requested, device=rows.device) * n, requested, rounding_mode='floor')
    sources = rows.index_select(0, ids)
    offsets = torch.tensor([v for v in itertools.product((-1, 0, 1), repeat=3) if v != (0, 0, 0)], device=rows.device)
    targets = sources[:, None, :] + offsets[None, :, :]
    minimum = rows.amin(0) - 1
    span = rows.amax(0) - minimum + 2
    if math.prod(int(v) for v in span.tolist()) >= (1 << 62):
        raise ValueError('coordinate range exceeds int64 candidate keys')

    def keys(values):
        x = values - minimum
        return (x[..., 0] * span[1] + x[..., 1]) * span[2] + x[..., 2]

    occupied = torch.sort(keys(rows)).values
    query = keys(targets)
    pos = torch.searchsorted(occupied, query.reshape(-1)).clamp_max(n - 1)
    occupied_target = occupied[pos].reshape_as(query).eq(query)
    depth = max(int(rows.max()).bit_length(), 1)
    empty = ~occupied_target & (targets >= 0).all(-1) & (targets < (1 << depth)).all(-1)
    degree = occupied_target.sum(-1).float()
    pools = {}
    source_cpu = sources.cpu().tolist()
    degree_cpu = degree.cpu().tolist()
    # These local costs are occupancy-topology priors, not claimed codec bits.
    pools['Prune'] = [dict(candidate_id=f'{SOURCE}:Prune:{x}:{y}:{z}', operation='Prune',
                           pool_rank=i, rank_score=float((26-d)/26), heuristic_score=float((26-d)/26),
                           remove_coords=[[x,y,z]], add_coords=[], affected_voxel_cells=1,
                           operation_count=1, optimistic_gain_bits=float((26-d)/26),
                           neighbor_bit_risk=0., geometry_cost=float(1/(d+1)))
                       for i, ((x,y,z), d) in enumerate(zip(source_cpu[:budgets['Prune']], degree_cpu))]
    for op, phase in (('Add', 7), ('Adjust', 19)):
        start = (ids * 1103515245 + phase) % 26
        priority = (torch.arange(26, device=rows.device)[None, :] - start[:, None]) % 26
        direction = priority.masked_fill(~empty, 27).argmin(-1)
        valid = torch.nonzero(empty.any(-1), as_tuple=False).flatten()
        destination = targets[valid, direction[valid]].cpu().tolist()
        candidate_source = sources[valid].cpu().tolist()
        local_degrees = degree[valid].cpu().tolist()
        seen = set()
        pool = []
        for source, target, d in zip(candidate_source, destination, local_degrees):
            if tuple(target) in seen:
                continue
            seen.add(tuple(target))
            score = float(d / 26.) if op == 'Add' else float(abs(d - 13) / 26.)
            pool.append(dict(candidate_id=f'{SOURCE}:{op}:{source}:{target}', operation=op,
                             pool_rank=len(pool), rank_score=score, heuristic_score=score,
                             remove_coords=[source] if op == 'Adjust' else [], add_coords=[target],
                             affected_voxel_cells=1, operation_count=1, optimistic_gain_bits=score,
                             neighbor_bit_risk=0., geometry_cost=float(1/(d+1))))
            if len(pool) >= budgets[op]:
                break
        pools[op] = pool
    if any(not pools[op] for op in ('Add', 'Prune', 'Adjust')):
        raise ValueError('geometry pool is empty for at least one operation')
    digest = hashlib.sha256(rows.detach().cpu().numpy().tobytes()).hexdigest()
    signature = hashlib.sha256(json.dumps(dict(codec=codec_key(args), qs=geometry_step(args),
        voxels=digest, total=total, shares=shares, reserve=reserve, cap=cap), sort_keys=True).encode()).hexdigest()
    elapsed = time.perf_counter() - started
    payload = dict(source=SOURCE, schema_version=SOURCE, codec=codec_key(args),
                   input_voxel_hash=digest, cache_signature=signature, cache_path='',
                   total_ratio=total, operation_shares=dict(zip(('Add','Prune','Adjust'), shares)),
                   operation_heuristics={op:'conventional_occupancy_proxy' for op in pools},
                   operation_priority=['Add','Prune','Adjust'], operation_edit_units=pools,
                   heuristic_anchor_plan=dict(available=False), actual_candidate_encode_count=0,
                   full_plan_candidate_count=1, elapsed_sec=elapsed,
                   proposal_pool_counts={op:len(pool) for op,pool in pools.items()},
                   proposal_policy=SOURCE, contains_candidate_actual=False,
                   proposal_timing=dict(total=elapsed), teacher_bootstrap_active=False)
    out = dict(context)
    out['ana_den6_ranked_candidate_guidance'] = payload  # shared candidate interface only
    out['ana_den6_online_cache_used'] = False
    args._ana_den6_online_last_timing = dict(heuristic_pool_wall_time=elapsed,
        proposal_mode=SOURCE, proposal_pool_counts=payload['proposal_pool_counts'],
        proposal_timing=payload['proposal_timing'])
    return out
