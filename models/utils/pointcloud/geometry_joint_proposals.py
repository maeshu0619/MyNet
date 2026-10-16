"""Signed occupancy-delta proposals across the full cloud, all child slots."""
import hashlib
import json
import math
import time
import numpy as np
import torch
from .geometry_octree import codec_key, geometry_step
from .geometry_octree_rate import remember_rate_model

SOURCE = 'mynet_geometry_octree_safe_pool_v1'  # shared interface discriminator
SCHEMA = 'geometry_joint_occupancy_v3'


def attach_joint_geometry_guidance(context, args, *, device):
    started = time.perf_counter()
    coords = context.get('full_global_voxel_coords', context.get('global_voxel_coords'))
    if not torch.is_tensor(coords) or coords.ndim != 3 or coords.shape[:2] != (1, 3):
        raise ValueError('conventional candidate pools require [1,3,N] canonical coordinates')
    rows = np.unique(coords[0].T.detach().cpu().numpy().astype(np.int64), axis=0)
    n = len(rows)
    if not n:
        raise ValueError('cannot propose edits for an empty octree')
    total = float(getattr(args, 'geometry_octree_initial_total_ratio', .0025))
    override = float(getattr(args, 'heuristic_guidance_total_ratio_percent', -1.))
    if override >= 0:
        total = override / 100.
    shares = [float(v) for v in str(getattr(args, 'heuristic_guidance_operation_shares', '')
                                    or '0.4,0.4,0.2').split(',')]
    if len(shares) != 3 or min(shares) <= 0 or not all(math.isfinite(v) for v in shares):
        raise ValueError('operation shares must be three finite positive values')
    shares = [v / sum(shares) for v in shares]
    reserve = max(float(getattr(args, 'geometry_octree_pool_reserve_factor', 4.)), 1.)
    cap = max(int(getattr(args, 'geometry_octree_pool_max_per_operation', 8192)), 1)
    budgets = {op: min(n, cap, max(int(math.ceil(n * total * share * reserve)), 1))
               for op, share in zip(('Add', 'Prune', 'Adjust'), shares)}
    digest = hashlib.sha256(rows.tobytes()).hexdigest()
    weight = float(getattr(args, 'geometry_octree_distortion_weight', .05))
    signature = hashlib.sha256(json.dumps(dict(schema=SCHEMA, codec=codec_key(args),
        qs=geometry_step(args), voxels=digest, total=total, shares=shares,
        reserve=reserve, cap=cap, distortion_weight=weight, bundles=True,
        rate_model=getattr(args, 'geometry_octree_rate_model', 'empirical')), sort_keys=True).encode()).hexdigest()
    model = remember_rate_model(args, signature, rows)
    parents = np.unique(rows // 2, axis=0)
    count = min(len(parents), max(budgets.values()) * 2 + 256)
    parent_ids = np.arange(count, dtype=np.int64) * len(parents) // count
    slots = [(i >> 2, (i >> 1) & 1, i & 1) for i in range(8)]
    pools = {op: [] for op in budgets}
    qs2 = geometry_step(args) ** 2
    for parent in parents[parent_ids].tolist():
        mask = model.masks[(1,) + tuple(parent)]
        occupied = [i for i in range(8) if mask & (1 << i)]
        empty = [i for i in range(8) if not mask & (1 << i)]
        points = [tuple(2 * parent[d] + slots[i][d] for d in range(3)) for i in range(8)]
        def distance(i, choices):
            return min((sum((slots[i][d] - slots[j][d]) ** 2 for d in range(3))
                        for j in choices), default=4) * qs2
        for op in pools:
            pairs = ([(i, None) for i in occupied] if op == 'Prune' else
                     [(None, j) for j in empty] if op == 'Add' else
                     [(i, j) for i in occupied for j in empty])
            for source, target in pairs:
                rem = [list(points[source])] if source is not None else []
                add = [list(points[target])] if target is not None else []
                geometry = ((distance(source, [i for i in occupied if i != source])
                             if source is not None else 0.)
                            + (distance(target, occupied) if target is not None else 0.))
                candidate = dict(operation=op, remove_coords=rem, add_coords=add,
                    affected_voxel_cells=1, operation_count=1, neighbor_bit_risk=0.,
                    geometry_cost=geometry, joint_parent=parent)
                _, gain = model.patch(candidate)
                utility = gain - weight * geometry
                candidate.update(candidate_id=f'{SCHEMA}:{op}:{rem}:{add}',
                    optimistic_gain_bits=gain, rank_score=utility, heuristic_score=utility)
                pools[op].append(candidate)
        # A whole sibling pattern is a single coherent candidate, not eight
        # independently estimated edits. Actual voxel costs remain explicit.
        for op, indices in (('Add', empty), ('Prune', occupied)):
            if len(indices) <= 1:
                continue
            rem = [list(points[i]) for i in indices] if op == 'Prune' else []
            add = [list(points[i]) for i in indices] if op == 'Add' else []
            geometry = sum(distance(i, occupied) for i in indices) if op == 'Add' else len(indices) * qs2 * 4
            candidate = dict(operation=op, remove_coords=rem, add_coords=add,
                affected_voxel_cells=len(indices), operation_count=len(indices),
                neighbor_bit_risk=0., geometry_cost=geometry, joint_parent=parent)
            _, gain = model.patch(candidate)
            utility = (gain - weight * geometry) / len(indices)
            candidate.update(candidate_id=f'{SCHEMA}:{op}:siblings:{parent}',
                optimistic_gain_bits=gain, rank_score=utility, heuristic_score=utility)
            pools[op].append(candidate)
    for level in (2, 3):
        groups, inverse = np.unique(rows >> level, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind='stable')
        lengths = np.bincount(inverse)
        starts = np.r_[0, np.cumsum(lengths)]
        sampled = np.arange(min(len(groups), 512)) * len(groups) // min(len(groups), 512)
        for group in sampled:
            rem = rows[order[starts[group]:starts[group + 1]]].tolist()
            if not 2 <= len(rem) <= max(2, int(n * .004)):
                continue
            geometry = len(rem) * qs2 * (2 ** (level - 1)) ** 2
            candidate = dict(operation='Prune', remove_coords=rem, add_coords=[],
                affected_voxel_cells=len(rem), operation_count=len(rem),
                neighbor_bit_risk=0., geometry_cost=geometry, joint_parent=groups[group].tolist())
            _, gain = model.patch(candidate)
            utility = (gain - weight * geometry) / len(rem)
            candidate.update(candidate_id=f'{SCHEMA}:Prune:subtree{level}:{groups[group].tolist()}',
                optimistic_gain_bits=gain, rank_score=utility, heuristic_score=utility)
            pools['Prune'].append(candidate)
    # A solid rectangular cloud may have no holes inside occupied parents.
    # Retain legal boundary Add/Adjust instead of silently losing a family.
    missing = [op for op in ('Add', 'Adjust') if not pools[op]]
    if missing:
        count = min(n, max(budgets.values()) * 2 + 256)
        source_ids = np.arange(count) * n // count
        seen = {op: set() for op in missing}
        for source in rows[source_ids].tolist():
            for axis in range(3):
                for direction in (-1, 1):
                    target = list(source); target[axis] += direction
                    if min(target) < 0 or max(target) >= 2 ** model.depth:
                        continue
                    slot = (target[0] & 1) * 4 + (target[1] & 1) * 2 + (target[2] & 1)
                    if model.masks.get(model.node(target), 0) & (1 << slot):
                        continue
                    for op in missing:
                        identity = (tuple(target), tuple(source) if op == 'Adjust' else ())
                        if identity in seen[op]:
                            continue
                        seen[op].add(identity)
                        rem = [source] if op == 'Adjust' else []
                        candidate = dict(operation=op, remove_coords=rem, add_coords=[target],
                            affected_voxel_cells=1, operation_count=1, neighbor_bit_risk=0.,
                            geometry_cost=qs2 * (2 if op == 'Adjust' else 1))
                        _, gain = model.patch(candidate)
                        utility = gain - weight * candidate['geometry_cost']
                        candidate.update(candidate_id=f'{SCHEMA}:{op}:boundary:{source}:{target}',
                            optimistic_gain_bits=gain, rank_score=utility, heuristic_score=utility)
                        pools[op].append(candidate)
    for op, pool in pools.items():
        ranked = sorted(pool, key=lambda p: (-p['rank_score'], p['candidate_id']))
        keep = min(budgets[op], len(ranked))
        top = max(1, int(keep * .8))
        tail = ranked[top:]
        indices = np.arange(keep - top) * len(tail) // max(keep - top, 1)
        chosen = ranked[:top] + [tail[i] for i in indices]
        chosen.sort(key=lambda p: (-p['rank_score'], p['candidate_id']))
        for rank, candidate in enumerate(chosen):
            candidate['pool_rank'] = rank
        pools[op] = chosen
    if any(not pool for pool in pools.values()):
        raise ValueError('geometry pool is empty for at least one operation')
    elapsed = time.perf_counter() - started
    payload = dict(source=SOURCE, schema_version=SCHEMA, codec=codec_key(args),
        input_voxel_hash=digest, cache_signature=signature, geometry_rate_model_key=signature,
        cache_path='', total_ratio=total,
        operation_shares=dict(zip(('Add', 'Prune', 'Adjust'), shares)),
        operation_heuristics={op: 'signed_multilevel_occupancy_proxy' for op in pools},
        operation_priority=['Add', 'Prune', 'Adjust'], operation_edit_units=pools,
        heuristic_anchor_plan=dict(available=False), actual_candidate_encode_count=0,
        full_plan_candidate_count=1, elapsed_sec=elapsed,
        proposal_pool_counts={op: len(pool) for op, pool in pools.items()},
        proposal_policy=SCHEMA, contains_candidate_actual=False,
        proposal_timing=dict(total=elapsed), teacher_bootstrap_active=False)
    out = dict(context, ana_den6_ranked_candidate_guidance=payload,
               ana_den6_online_cache_used=False)
    args._ana_den6_online_last_timing = dict(heuristic_pool_wall_time=elapsed,
        proposal_mode=SCHEMA, proposal_pool_counts=payload['proposal_pool_counts'],
        proposal_timing=payload['proposal_timing'])
    return out
