#!/usr/bin/env python
"""CPU-only, joint final-occupancy research. Never imports train.py.

All 255 nonempty child masks are considered before assigning operation names.
Raw-point coverage is tested on the *whole new mask*, not on isolated Prunes.
G-PCC selection uses full-cloud Actual encodes, not OA scores as ground truth.
This is an offline codec oracle experiment, not trained network performance.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from octree_codec_research import OAActual, ActualCodec, unique, difference, load_ply, quality

SLOTS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int64)
POPS = np.array([bin(i).count('1') for i in range(256)])
MASKS = np.arange(1, 256, dtype=np.int64)


def coverage_valid(covers, masks=MASKS):
    """A raw point may be rescued by *any* voxel in the final joint mask."""
    covers = np.unique(np.asarray(covers, dtype=np.int64))
    if not len(covers):
        return np.ones(len(masks), dtype=bool)
    if (covers == 0).any():
        raise ValueError('Missing assigned-point coverage')
    return ((np.asarray(masks)[:, None] & covers[None, :]) != 0).all(1)


def parts(before, after, parent):
    """Explicit executable local moves: all pairs are at most one cell/axis."""
    rem = [tuple(parent + SLOTS[s]) for s in range(8) if before & (1 << s) and not after & (1 << s)]
    add = [tuple(parent + SLOTS[s]) for s in range(8) if after & (1 << s) and not before & (1 << s)]
    moves = min(len(rem), len(add))
    return dict(Adjust=list(zip(rem[:moves], add[:moves])), Prune=rem[moves:], Add=add[moves:])


def apply_plan(q, seq, chosen, operations=('Add', 'Prune', 'Adjust')):
    occupied = set(map(tuple, q.tolist()))
    counts = dict(Add=0, Prune=0, Adjust=0)
    seen = set()
    for row, after in chosen:
        if row in seen:
            raise AssertionError('Duplicate leaf rewrite')
        seen.add(row)
        before = int(seq[row, -1, 0])
        split = parts(before, int(after), seq[row, -1, 3:6])
        for op in operations:
            for item in split[op]:
                if op == 'Add':
                    occupied.add(item)
                elif op == 'Prune':
                    occupied.remove(item)
                else:
                    source, target = item
                    if np.max(np.abs(np.array(source)-target)) > 1:
                        raise AssertionError('Nonlocal Adjust')
                    occupied.remove(source)
                    occupied.add(target)
                counts[op] += 1
    out = unique(list(occupied))
    if operations == ('Add', 'Prune', 'Adjust'):
        nr, na = len(difference(q, out)), len(difference(out, q))
        if nr != counts['Prune'] + counts['Adjust'] or na != counts['Add'] + counts['Adjust']:
            raise AssertionError('Cancellation or operation accounting mismatch')
        if not np.array_equal(q.min(0), out.min(0)) or not np.array_equal(q.max(0), out.max(0)):
            raise AssertionError('Coding phase/bounding box changed')
    return out, counts


def proposals(q, raw, origin, seq, probs, report):
    started = time.perf_counter()
    rows = np.flatnonzero(seq[:, -1, 1] == seq[:, -1, 1].max())
    parent_origins = seq[rows, -1, 3:6].astype(np.int64)
    parent_index = {tuple(p): i for i, p in enumerate(parent_origins.tolist())}
    q_parent = np.array([parent_index[tuple(p)] for p in q//2*2], dtype=np.int64)
    d, near = cKDTree((q+origin)*2.).query(raw, workers=2)
    raw_parent = q_parent[near]
    order = np.argsort(raw_parent, kind='stable')
    starts = np.r_[0, np.bincount(raw_parent, minlength=len(rows)).cumsum()]
    raw_tree = cKDTree(raw)
    slot_dist = raw_tree.query(((parent_origins[:, None]+SLOTS[None]+origin)*2.).reshape(-1, 3), workers=2)[0].reshape(-1, 8)
    protected = set(map(tuple, q[np.unique(np.r_[q.argmin(0), q.argmax(0)])].tolist()))
    bbox_min, bbox_max = q.min(0), q.max(0)
    # Marginal occupancy frequency is only a proposal feature, NOT a G-PCC probability.
    histogram = np.bincount(seq[rows, -1, 0], minlength=256).astype(np.float64)+1.
    occupied_masks = {tuple(p): int(seq[row, -1, 0]) for p, row in zip(parent_origins, rows)}
    result = []
    joint_only, unsupported = 0, 0
    for index, row in enumerate(rows):
        before = int(seq[row, -1, 0])
        parent = parent_origins[index]
        points = parent+SLOTS
        found = sum(1 << s for s, p in enumerate(points) if tuple(p) in protected)
        assigned = order[starts[index]:starts[index+1]]
        unsupported += int(not len(assigned))
        distance_sq = np.sum((((points+origin)*2.)[None]-raw[assigned, None])**2, axis=2)
        covers = ((distance_sq <= d[assigned, None]**2+1e-8) * (1 << np.arange(8))[None]).sum(1)
        valid = coverage_valid(covers)
        unsafe_add = sum(1 << s for s in range(8) if slot_dist[index, s] > np.sqrt(3)+1e-9
                         or (points[s] < bbox_min).any() or (points[s] > bbox_max).any())
        valid &= (MASKS & found) == found
        valid &= (MASKS & unsafe_add & ~before) == 0
        valid &= MASKS != before
        targets = MASKS[valid]
        if not len(targets):
            continue
        costs = np.maximum(POPS[before & ~targets], POPS[targets & ~before])
        oa_gain = np.log2(np.maximum(probs[row, targets-1], 1e-30)/max(float(probs[row, before-1]), 1e-30))
        freq_gain = np.log2(histogram[targets]/histogram[before])
        # Neighbor-mask agreement includes spatial distribution, but is not native OBUF.
        neighbors = [occupied_masks[tuple(parent+2*delta)] for delta in
                     np.vstack([np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)])
                     if tuple(parent+2*delta) in occupied_masks]
        agreement = np.zeros(len(targets))
        for neighbor in neighbors:
            agreement += POPS[before ^ neighbor]-POPS[targets ^ neighbor]
        agreement /= max(1, len(neighbors))
        for j, after in enumerate(targets):
            # Can all removals stand alone? False means Adds rescue this deletion.
            retained = before & int(after)
            rescue = not bool(coverage_valid(covers, np.array([retained]))[0])
            joint_only += int(rescue)
            result.append((int(row), int(after), int(costs[j]), float(oa_gain[j]),
                           float(freq_gain[j]), float(agreement[j]), int(POPS[before]-POPS[after]), rescue))
        if (index+1) % 10000 == 0:
            report(dict(proposal_leaves=index+1, candidates=len(result)))
    out = np.array(result, dtype=[('row','i4'), ('mask','i4'), ('cost','i4'),
        ('oa','f8'), ('freq','f8'), ('neighbor','f8'), ('prune','f8'), ('rescue','?')])
    audit = dict(candidate_count=len(out), joint_rescue_candidates=joint_only,
                 leaf_without_exclusive_nn_assignment=unsupported,
                 generation_sec=time.perf_counter()-started, nonpositive_oa_candidates=int((out['oa'] <= 0).sum()),
                 legality='per-raw-point joint coverage; safe Add distance; bbox anchors; nonempty leaves')
    report(audit)
    return out, audit


def select(candidates, weights, k, seed=0):
    score = sum(candidates[name]*weight for name, weight in weights.items())
    score = score/candidates['cost']
    # Deterministic tie handling; seed only used for an explicit random control.
    if seed:
        score = score+np.random.RandomState(seed).gumbel(size=len(score))*.25
    ranked = np.argsort(-score, kind='stable')
    chosen, used, rows = [], 0, set()
    for index in ranked:
        item = candidates[index]
        if score[index] <= 0:
            break
        row, cost = int(item['row']), int(item['cost'])
        if row in rows or used+cost > k:
            continue
        rows.add(row)
        chosen.append((row, int(item['mask'])))
        used += cost
        if used == k:
            break
    lookup = {(int(x['row']), int(x['mask'])): bool(x['rescue']) for x in candidates if x['rescue']}
    return chosen, dict(cost=used, joint_rescue_rewrites=sum(lookup.get(x, False) for x in chosen),
                        frozen_oa_gain=float(sum(candidates['oa'][
                            (candidates['row']==row)&(candidates['mask']==mask)][0] for row, mask in chosen)))


def quality_ok(base, edited):
    return all(edited[key] <= base[key]+1e-9 for key in base)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--budgets', nargs='+', type=float, default=[.0015, .003])
    p.add_argument('--skip-ablations', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'metadata.json').write_text(json.dumps(dict(args=vars(a),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_only=True, baseline_commit='deb783be385bd764d1bbb87e683655b2d9399ebb',
        scope='isolated offline research; no training changes'), indent=2))
    oa, gpcc = OAActual('cpu'), ActualCodec(2., 2)
    profiles = [('oa', dict(oa=1.)), ('freq', dict(freq=1.)),
        ('neighbor', dict(neighbor=1.)), ('prune', dict(prune=1.)),
        ('oa_freq', dict(oa=1.,freq=1.)), ('oa_neighbor', dict(oa=1.,neighbor=1.)),
        ('oa_prune', dict(oa=1.,prune=2.)), ('freq_neighbor', dict(freq=1.,neighbor=1.)),
        ('freq_prune', dict(freq=1.,prune=2.)), ('neighbor_prune', dict(neighbor=1.,prune=2.)),
        ('combined', dict(oa=.5,freq=1.,neighbor=1.,prune=1.))]
    with (folder/'trials.jsonl').open('w', buffering=1) as log:
        def emit(row):
            log.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        for frame in a.frames:
            raw = load_ply(frame, return_color=False).astype(np.float64)
            absolute = unique(np.rint(raw/2.))
            origin = absolute.min(0)
            q = absolute-origin
            base_quality = quality(raw, q, origin, 2.)
            base_oa, seq, probs = oa.encode(q, detail=True, roundtrip=True)
            base_gpcc = gpcc.encode(q, verify=True)
            emit(dict(frame=frame, trial='baseline', voxels=len(q), gpcc=base_gpcc, oa=base_oa, quality=base_quality))
            candidates, audit = proposals(q, raw, origin, seq, probs, lambda x: print(json.dumps(x), flush=True))
            np.savez_compressed(folder/(Path(frame).stem+'_proposals.npz'), candidates=candidates, seq=seq, origin=origin)
            for budget in a.budgets:
                k = round(len(q)*budget)
                trials = []
                fingerprints = {}
                for name, weights in profiles:
                    chosen, extra = select(candidates, weights, k)
                    out, counts = apply_plan(q, seq, chosen)
                    geom = quality(raw, out, origin, 2.)
                    fingerprint = hashlib.sha256(out.tobytes()).hexdigest()
                    stats = fingerprints.get(fingerprint)
                    if stats is None:
                        stats = gpcc.encode(out)
                        fingerprints[fingerprint] = stats
                    result = dict(frame=frame, trial=name, budget=budget, counts=counts,
                        operation_percent=100*sum(counts.values())/len(q), quality=geom,
                        quality_ok=quality_ok(base_quality, geom), gpcc=stats,
                        gpcc_C=100*(stats['bits']/base_gpcc['bits']-1.), **extra)
                    if name == 'oa':
                        oa_stats = oa.encode(out, roundtrip=True)
                        result.update(oa=oa_stats, oa_C=100*(oa_stats['bits']/base_oa['bits']-1.))
                    emit(result)
                    trials.append((result, chosen, out))
                valid = [x for x in trials if x[0]['quality_ok'] and x[0]['gpcc']['bits'] < base_gpcc['bits']]
                if not valid:
                    emit(dict(frame=frame, trial='no_quality_safe_improvement', budget=budget))
                    continue
                best, chosen, out = min(valid, key=lambda x: x[0]['gpcc']['bits'])
                stem = Path(frame).stem+'_%.5f'%budget
                np.savez_compressed(folder/(stem+'_best.npz'), canonical_voxels=out, origin=origin,
                                    chosen=np.array(chosen, dtype=np.int64), qs=2.)
                encoded = gpcc.encode(out, verify=True)
                if encoded['stream_sha256'] != best['gpcc']['stream_sha256']:
                    raise AssertionError('Repeated Actual encode changed')
                oa_stats = oa.encode(out, roundtrip=True)
                emit(dict(frame=frame, trial='gpcc_oracle_best', profile=best['trial'], budget=budget,
                    gpcc=encoded, gpcc_C=best['gpcc_C'], oa=oa_stats,
                    oa_C=100*(oa_stats['bits']/base_oa['bits']-1.), counts=best['counts'],
                    operation_percent=best['operation_percent'], quality=best['quality'],
                    joint_rescue_rewrites=best['joint_rescue_rewrites'], generation=audit))
                if not a.skip_ablations and budget == a.budgets[0]:
                    for size in (1, 2):
                        for ops in itertools.combinations(('Add', 'Prune', 'Adjust'), size):
                            partial, counts = apply_plan(q, seq, chosen, ops)
                            stats = gpcc.encode(partial)
                            emit(dict(frame=frame, trial='ablation', operations=list(ops), budget=budget,
                                gpcc=stats, gpcc_C=100*(stats['bits']/base_gpcc['bits']-1.),
                                counts=counts, quality=quality(raw, partial, origin, 2.)))
    print('RESULTS '+str(folder), flush=True)


if __name__ == '__main__':
    main()
