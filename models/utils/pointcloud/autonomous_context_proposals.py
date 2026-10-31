"""Dataset-independent proposals covering native coded occupancy changes.

The native fixed-context likelihood change is an INPUT observation, never an
Actual counterfactual label or a fixed final policy score. Unordered spatial
candidates remain available alongside these codec-visible alternatives.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def cell_counts(rows):
    low, high = rows.min(0), rows.max(0)
    span = [int(b)-int(a)+1 for a,b in zip(low,high)]
    if span[0]*span[1]*span[2] >= 2**63:
        return np.unique(rows,axis=0,return_inverse=True,return_counts=True)
    shifted=rows-low
    keys=(shifted[:,0]*span[1]+shifted[:,1])*span[2]+shifted[:,2]
    _,first,inverse,counts=np.unique(keys,return_index=True,return_inverse=True,return_counts=True)
    return rows[first],inverse,counts


def canonical_voxel_sha(rows):
    from ..loss.autonomous_native_loader import lexicographic_unique_rows
    rows = np.asarray(rows, dtype=np.int64)
    unique = lexicographic_unique_rows(rows)
    if unique is None:
        unique = np.unique(rows, axis=0)
    return hashlib.sha256(unique.tobytes()).hexdigest()


def context_proposals(source, context_path, budget, seed=0, random_pools=None):
    if int(budget)<=0:
        return _context_proposals_reference(source,context_path,budget,seed,random_pools)
    from .autonomous_context_arrays import context_proposals_arrays
    return context_proposals_arrays(source,context_path,budget,seed,random_pools)


def _context_proposals_reference(source, context_path, budget, seed=0, random_pools=None):
    from scipy.spatial import cKDTree
    from ..loss.autonomous_native_loader import lexicographic_unique_rows
    source = np.asarray(source, dtype=np.int64)
    canonical = lexicographic_unique_rows(source)
    source = canonical if canonical is not None else np.unique(source,axis=0)
    with np.load(context_path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'].reshape(())))
        expected = metadata.get('input_voxel_sha256', metadata.get('input_sha'))
        if expected != canonical_voxel_sha(source):
            raise ValueError('Native codec context belongs to a different GT voxel set')
        scale = int(metadata['scale'])
        cells = data['coords'].astype(np.int64)
        p = data['probability'].astype(float)
        occupied = data['occupied'].astype(bool)
    if scale <= 0 or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('Invalid native coded-level probabilities/scale')
    if len(cells) != len({tuple(x) for x in cells}):
        raise ValueError('Native coded-level context has duplicate symbols')
    logits = np.log2(p)-np.log2(1-p)
    cell_prob = {tuple(c): float(v) for c, v in zip(cells, logits)}
    unique, inverse, counts = cell_counts(source//scale)
    count_map = {tuple(c): int(n) for c, n in zip(unique, counts)}
    pools = {op: [] for op in ('Add', 'Prune', 'Adjust')}
    cache_path=Path(str(context_path)+'.proposals_v1_%d_%d.npz'%(int(budget),int(seed)&0x7fffffff))
    cached_core=None
    if cache_path.is_file():
        with np.load(cache_path,allow_pickle=False) as data:
            cached_core={op:data[op].astype(np.int64) for op in pools}
    lower, upper = source.min(0), source.max(0)
    interior = ((source > lower) & (source < upper)).all(1)
    if cached_core is None:
        for point in source[(counts[inverse] == 1) & interior]:
            pools['Prune'].append(dict(remove_coords=[point.tolist()], add_coords=[]))
    # Project nearest GT voxels onto each empty coded cell's integer box.
    # The raw target stays within one voxel per axis, matching the existing
    # 26-neighbour operation domain. This is not a new larger edit radius.
    empty = cells[~occupied] if cached_core is None else cells[:0]
    lo = empty*scale
    hi = lo+scale-1
    if cached_core is None:
        _, near = cKDTree(source).query((lo+hi)/2., k=min(8, len(source)), workers=2)
    else:
        near=np.empty((0,min(8,len(source))),dtype=np.int64)
    if near.ndim == 1:
        near = near[:, None]
    raw = source[near]
    target = np.maximum(np.minimum(raw, hi[:, None, :]), lo[:, None, :])
    delta = np.max(np.abs(target-raw), axis=2)
    valid = (delta > 0) & (delta <= 1)
    valid &= ((target >= lower) & (target <= upper)).all(2)
    for src, dst, index in zip(raw[valid], target[valid], near[valid]):
        pools['Add'].append(dict(remove_coords=[], add_coords=[dst.tolist()]))
        if interior[index]:
            pools['Adjust'].append(dict(remove_coords=[src.tolist()], add_coords=[dst.tolist()]))
    rng = np.random.RandomState(int(seed) & 0x7fffffff)

    def observation(candidate):
        remove, add = candidate.get('remove_coords', []), candidate.get('add_coords', [])
        rc = tuple(np.asarray(remove[0])//scale) if remove else None
        ac = tuple(np.asarray(add[0])//scale) if add else None
        gain, known = 0., True
        if rc != ac:
            if rc is not None and count_map.get(rc) == 1:
                if rc in cell_prob: gain -= cell_prob[rc]
                else: known = False
            if ac is not None and ac not in count_map:
                if ac in cell_prob: gain += cell_prob[ac]
                else: known = False
        return gain, known

    result, diagnostics, core_arrays = {}, {}, {}
    for operation_index, (op, proposals) in enumerate(pools.items()):
        rng=np.random.RandomState((int(seed)+operation_index*7919)&0x7fffffff)
        seen, unique_proposals = set(), []
        for c in proposals:
            token=(tuple(map(tuple,c['remove_coords'])),tuple(map(tuple,c['add_coords'])))
            if token not in seen:
                seen.add(token); unique_proposals.append(c)
        # Uniformly sample in signed native log-likelihood bands, preserving
        # both adverse and promising candidates without native-rank sorting.
        band_edges=np.array([-8.,-4.,-1.,0.,1.,4.,8.])
        bands=np.digitize([observation(c)[0] for c in unique_proposals],band_edges)
        ids=[]
        groups=[rng.permutation(np.flatnonzero(bands==b)).tolist() for b in range(8)]
        cap=min(len(unique_proposals), max(int(budget)//2,1))
        while len(ids)<cap and any(groups):
            for group in groups:
                if group and len(ids)<cap:ids.append(group.pop())
        selected=[unique_proposals[i] for i in ids]
        if cached_core is not None:
            selected=[dict(remove_coords=[v[:3].tolist()] if op!='Add' else [],
                           add_coords=[v[3:].tolist()] if op!='Prune' else []) for v in cached_core[op]]
        core_arrays[op]=np.asarray([(c.get('remove_coords') or [[0,0,0]])[0]
                                   +(c.get('add_coords') or [[0,0,0]])[0] for c in selected],dtype=np.int64).reshape(-1,6)
        selected_tokens={(tuple(map(tuple,c.get('remove_coords',[]))),tuple(map(tuple,c.get('add_coords',[]))))
                         for c in selected}
        existing=(random_pools or {}).get(op, [])
        rng=np.random.RandomState((int(seed)+operation_index*7919+104729)&0x7fffffff)
        for i in rng.permutation(len(existing)):
            if len(selected)>=int(budget):break
            c=existing[i]
            token=(tuple(map(tuple,c.get('remove_coords',[]))),tuple(map(tuple,c.get('add_coords',[]))))
            if token not in selected_tokens:
                selected.append(c)
                selected_tokens.add(token)
        built=[]
        for index,c in enumerate(selected):
            c=dict(c);gain,known=observation(c)
            remove,add=c.get('remove_coords',[]),c.get('add_coords',[])
            payload=json.dumps([op,remove,add],separators=(',',':'))
            built.append(dict(c,candidate_id='ctx:'+hashlib.sha256(payload.encode()).hexdigest()[:24],
                operation=op,pool_rank=index,rank_score=0.,heuristic_score=0.,
                optimistic_gain_bits=gain,neighbor_bit_risk=0.,
                geometry_cost=float(c.get('geometry_cost',1.)),
                operation_count=1,affected_voxel_cells=1,
                native_fixed_context_observed=known))
        result[op]=built
        diagnostics[op]=dict(codec_candidates=len(unique_proposals) if cached_core is None else None,
            retained_codec_candidates=len(core_arrays[op]),core_cache_hit=cached_core is not None,
            total=len(built),positive_fixed_gain=sum(x['optimistic_gain_bits']>0 for x in built),
            negative_fixed_gain=sum(x['optimistic_gain_bits']<0 for x in built))
    if cached_core is None:
        with tempfile.NamedTemporaryFile(dir=str(cache_path.parent),delete=False) as f:
            temp=Path(f.name);np.savez_compressed(f,**core_arrays)
        try:
            os.link(str(temp),str(cache_path))
        except FileExistsError:
            pass
        finally:
            temp.unlink()
    return result,diagnostics
