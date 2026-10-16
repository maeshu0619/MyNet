#!/usr/bin/env python
"""Standalone, CPU-only Actual G-PCC micro-edit experiments; no training writes.

Every rate is total stream bits relative to the SAME unedited qs baseline.
The quality metric is symmetric mean squared NN distance, in original units.
Search results are codec-oracle experiments, NOT trained-network performance.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.utils.compression.gpcc_tmc3 import GPCCGeometryEncoder
from models.utils.data.dataset import load_ply


def unique(rows):
    return np.unique(np.asarray(rows, dtype=np.int64).reshape(-1, 3), axis=0)


def cells(rows):
    return np.ascontiguousarray(rows, dtype=np.int64).view(np.dtype((np.void, 24))).ravel()


def difference(a, b):
    return np.setdiff1d(cells(a), cells(b)).view(np.int64).reshape(-1, 3)


class ActualCodec:
    """Same encoder command as training, plus optional independent decoding."""
    def __init__(self, qs, threads, extra_flags=()):
        self.encoder = GPCCGeometryEncoder(str(ROOT.parent / 'compress/octree/G-PCC'),
                                          '', '', effective_qs=qs)
        self.encoder.validate()
        self.env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        self.calls = 0
        self.extra_flags = list(extra_flags)

    def encode(self, rows, verify=False):
        # Match the teacher's bbox normalization without changing physical
        # positions used for distortion measurement.
        coords = unique(rows - rows.min(axis=0))
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='gpcc_micro_') as directory:
            inp, stream, rec = [str(Path(directory) / name) for name in
                                ('input.ply', 'stream.bin', 'recon.ply')]
            self.encoder._write_ascii_ply(inp, coords)
            command = self.encoder._command(inp, stream, rec) + self.extra_flags
            result = subprocess.run(command, cwd=self.encoder.root, env=self.env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(result.stderr[-3000:] + result.stdout[-3000:])
            self.calls += 1
            payload = Path(stream).read_bytes()
            if verify:
                command = [self.encoder.encoder_path, '--mode=1',
                           '--compressedStreamPath=' + stream, '--reconstructedDataPath=' + rec]
                result = subprocess.run(command, cwd=self.encoder.root, env=self.env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, timeout=120)
                if result.returncode:
                    raise RuntimeError(result.stderr[-3000:] + result.stdout[-3000:])
                decoded = unique(load_ply(rec, return_color=False))
                if not np.array_equal(decoded, coords):
                    raise AssertionError('Decoded geometry differs from encoded integer voxels')
        return dict(bits=len(payload) * 8, stream_sha256=hashlib.sha256(payload).hexdigest(),
                    codec_sec=time.perf_counter() - started, decoded_exact=bool(verify))


def phase_add(q, shift):
    """At most three nearby boundary anchors; original points do not move.

    Each anchor lies 1..k voxel steps beyond an existing minimum-plane point.
    The teacher's bbox normalization then shifts the coding grid globally.
    """
    anchors = []
    for axis, amount in enumerate(shift):
        if amount:
            anchor = q[np.argmin(q[:, axis])].copy()
            anchor[axis] -= amount
            anchors.append(anchor)
    return unique(np.vstack([q, np.asarray(anchors).reshape(-1, 3)])) if anchors else q


def phase_prune(q, shift):
    return q[np.all(q >= np.asarray(shift), axis=1)]


def phase_adjust(q, shift):
    moves = {}
    for axis, amount in enumerate(shift):
        if amount:
            candidates = np.flatnonzero(q[:, axis] == q[:, axis].min())
            i = next((int(j) for j in candidates if int(j) not in moves), int(candidates[0]))
            if i not in moves:
                moves[i] = q[i].copy()
            moves[i][axis] -= amount
    mask = np.ones(len(q), dtype=bool)
    mask[list(moves)] = False
    return unique(np.vstack([q[mask], np.asarray(list(moves.values())).reshape(-1, 3)])), len(moves)


def local_variants(q, budgets, rng):
    tree = cKDTree(q)
    degree = tree.query_ball_point(q, np.sqrt(3) + 1e-6, return_length=True, workers=2) - 1
    for budget in budgets:
        k = max(1, int(round(len(q) * budget)))
        for mode, order in [('random', rng.permutation(len(q))),
                            ('sparse', np.argsort(degree, kind='stable')),
                            ('dense', np.argsort(-degree, kind='stable'))]:
            ids = order[:k]
            yield 'prune_%s_%.5f' % (mode, budget), np.delete(q, ids, axis=0), dict(prune=k)
        # Remove coherent, spatially compact whole subtrees. This is not
        # independent top-K point removal and charges every removed voxel.
        for level in (1, 2, 3):
            groups, inv = np.unique(q >> level, axis=0, return_inverse=True)
            lengths = np.bincount(inv)
            ids = np.argsort(inv, kind='stable')
            starts = np.r_[0, lengths.cumsum()]
            for strategy in ('prefix', 'small'):
                order = np.arange(len(groups)) if strategy == 'prefix' else np.argsort(lengths, kind='stable')
                selected, used = [], 0
                for g in order:
                    if used + lengths[g] <= k:
                        selected.extend(ids[starts[g]:starts[g + 1]])
                        used += int(lengths[g])
                    if used == k:
                        break
                mask = np.ones(len(q), dtype=bool)
                mask[selected] = False
                yield 'subtree%d_%s_%.5f' % (level, strategy, budget), q[mask], dict(prune=used)
        # Parent occupancy completion, scored by number of existing siblings.
        parents, inv = np.unique(q // 2, axis=0, return_inverse=True)
        lengths = np.bincount(inv)
        occupied = set(map(tuple, q.tolist()))
        additions = []
        for g in np.argsort(-lengths, kind='stable'):
            if lengths[g] == 8:
                continue
            for slot in itertools.product((0, 1), repeat=3):
                p = tuple(parents[g] * 2 + slot)
                if p not in occupied:
                    additions.append(p)
                    if len(additions) >= k:
                        break
            if len(additions) >= k:
                break
        added = unique(additions)
        yield 'add_complete_%.5f' % budget, unique(np.vstack([q, added])), dict(add=len(added))
        # Move single-child points to an adjacent occupied parent's empty slot.
        # Missing moves are not filled with unrelated random operations.
        occupied_parents = set(map(tuple, parents.tolist()))
        moved_src, moved_dst, used_dst = [], [], set()
        for i in np.flatnonzero(lengths[inv] == 1):
            for axis, step in itertools.product(range(3), (-1, 1)):
                p = q[i].copy()
                p[axis] += step
                key = tuple(p)
                if (p >= 0).all() and key not in occupied and key not in used_dst and tuple(p // 2) in occupied_parents:
                    moved_src.append(i)
                    moved_dst.append(p)
                    used_dst.add(key)
                    break
            if len(moved_src) >= k:
                break
        mask = np.ones(len(q), dtype=bool)
        mask[moved_src] = False
        yield 'adjust_merge_parent_%.5f' % budget, unique(np.vstack([q[mask], np.asarray(moved_dst).reshape(-1, 3)])), dict(adjust=len(moved_src))


def controls(q):
    parents, inv = np.unique(q // 2, axis=0, return_inverse=True)
    _, first = np.unique(inv, return_index=True)
    for fraction in (.05, .30, 1.):
        mask = inv >= int(len(parents) * fraction)
        mask[first] = True
        out = q[mask]
        yield 'large_prune_parent_fraction_%.2f' % fraction, out, dict(prune=len(q) - len(out))


def current_prior_variants(q, budgets):
    """Read-only use of current proposal/selection, with ZERO learned residual.

    This is an initialization audit, not evaluation of the running checkpoint.
    """
    from models.utils.pointcloud.geometry_joint_proposals import attach_joint_geometry_guidance
    from models.utils.pointcloud.geometry_joint_policy import select_joint_geometry_plan
    args = SimpleNamespace(compress='G-PCC', gpcc_prequantize=True,
                           gpcc_effective_qs=2., geometry_octree_initial_total_ratio=.0025)
    coords = torch.from_numpy(q.T.copy()).unsqueeze(0)
    context = attach_joint_geometry_guidance(dict(full_global_voxel_coords=coords), args, device='cpu')
    exact = context['ana_den6_ranked_candidate_guidance']
    pools = exact['operation_edit_units']
    scores = {op: torch.zeros(len(pool)) for op, pool in pools.items()}
    for budget in budgets:
        k = max(1, int(round(len(q) * budget)))
        amounts = dict(Add=int(k*.4), Prune=int(k*.4), Adjust=k-2*int(k*.4))
        for mode in ('joint', 'add', 'prune', 'adjust'):
            counts = {op: n if mode == 'joint' or mode.lower() == op.lower() else 0
                      for op, n in amounts.items()}
            with torch.no_grad():
                result = select_joint_geometry_plan(coords, exact, pools, counts, scores,
                                                     args, exploration=0., heuristic_only=True)
            _, rem, add, _, actual, _, _, audit = result
            occupied = set(map(tuple, q.tolist()))
            occupied.difference_update(rem)
            occupied.update(add)
            out = unique(list(occupied))
            metadata = {op.lower(): n for op, n in actual.items()}
            metadata.update(predicted_gain_bits=audit['joint_gain'], pool_counts={op: len(v) for op,v in pools.items()})
            yield 'current_prior_%s_%.5f' % (mode, budget), out, metadata


def combo_variants(q, budgets):
    tree = cKDTree(q)
    degree = tree.query_ball_point(q, np.sqrt(3) + 1e-6, return_length=True, workers=2) - 1
    order = np.argsort(degree, kind='stable')
    for budget in budgets:
        k = max(1, int(round(len(q) * budget)))
        pruned = np.delete(q, order[:k], axis=0)
        # Fixed recipe chosen on the discovery frame, not per-frame oracle.
        # Do not let removal of extrema silently change the requested anchor:
        # anchors are calculated from the ORIGINAL cloud.
        anchors = difference(phase_add(q, (0, 1, 1)), q)
        out = unique(np.vstack([pruned, anchors]))
        yield 'sparse_plus_fixed_anchor011_%.5f' % budget, out, dict(prune=k, add=len(anchors))


def quality_guard_variants(q, raw, origin, qs, budgets, preserve_bbox=False):
    """Prune sparse voxels without increasing the GT->recon max distance.

    Reserve every deleted voxel's second-nearest replacement so that two
    individually safe deletions cannot invalidate one another's guarantee.
    This explicitly checks the original input geometry; it is a hand-designed
    experiment, not a learned policy or a trained-network performance result.
    """
    distances, indices = cKDTree((q + origin)*qs).query(raw, k=2, workers=2)
    original_max = distances[:, 0].max()
    per_voxel_max = np.zeros(len(q))
    np.maximum.at(per_voxel_max, indices[:, 0], distances[:, 1])
    valid = per_voxel_max <= original_max + 1e-9
    assignment_order = np.argsort(indices[:, 0], kind='stable')
    starts = np.r_[0, np.bincount(indices[:, 0], minlength=len(q)).cumsum()]
    degree = cKDTree(q).query_ball_point(q, np.sqrt(3)+1e-6, workers=2, return_length=True)-1
    order = np.argsort(degree, kind='stable')
    for budget in budgets:
        k = max(1, round(len(q)*budget))
        anchors = set(q.argmin(0).tolist() + q.argmax(0).tolist()) if preserve_bbox else set()
        selected, protected = set(), set(anchors)
        for i in order:
            i = int(i)
            if not valid[i] or i in protected:
                continue
            depend = set(indices[assignment_order[starts[i]:starts[i+1]], 1].tolist())
            if depend & selected:
                continue
            selected.add(i)
            protected.update(depend)
            if len(selected) == k:
                break
        mask = np.ones(len(q), dtype=bool)
        mask[list(selected)] = False
        prefix = 'quality_bbox_guard_prune_' if preserve_bbox else 'quality_guard_prune_'
        yield prefix + '%.5f' % budget, q[mask], dict(prune=len(selected),
                original_max_gt_distance=float(original_max), eligible_voxels=int(valid.sum()),
                bbox_protected=preserve_bbox)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--qs', type=float, default=2.)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=21)
    p.add_argument('--modes', nargs='+', default=['phase', 'local', 'control'],
                   choices=['phase', 'local', 'control', 'prior', 'combo', 'adjustphase',
                            'qualityguard', 'qualityguardbbox', 'bboxcounterfactual',
                            'bboxaxes', 'matchedprune'])
    p.add_argument('--phases', type=int, default=2, help='0..phases-1 shifts per axis')
    p.add_argument('--anchor-only', action='store_true', help='Skip min-plane deletion in phase sweep')
    p.add_argument('--budgets', nargs='+', type=float, default=[.0005, .0015, .003])
    p.add_argument('--max-ratio', type=float, default=.0099)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.qs <= 0 or a.phases < 1 or any(not 0 < v <= a.max_ratio for v in a.budgets):
        p.error('positive qs/phases and local budgets within max-ratio required')
    output = Path(a.out)
    output.mkdir(parents=True, exist_ok=False)
    codec = ActualCodec(a.qs, a.threads)
    meta = dict(arguments=vars(a), encoder=codec.encoder.encoder_path, cfg=codec.encoder.cfg_path,
                experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                encoder_sha256=hashlib.sha256(Path(codec.encoder.encoder_path).read_bytes()).hexdigest(),
                cfg_sha256=hashlib.sha256(Path(codec.encoder.cfg_path).read_bytes()).hexdigest(),
                rate_definition='100*(edited_total_bits-baseline_total_bits)/baseline_total_bits',
                quality_definition='mean(sqNN(raw,recon))+mean(sqNN(recon,raw))',
                operation_denominator='unedited unique qs-grid voxels', training_modified=False)
    (output / 'metadata.json').write_text(json.dumps(meta, indent=2))
    summaries = []
    with (output / 'trials.jsonl').open('w', buffering=1) as log:
        for frame in a.frames:
            raw = load_ply(frame, return_color=False).astype(np.float64)
            absolute = unique(np.rint(raw / a.qs))
            origin = absolute.min(axis=0)
            q = absolute - origin
            raw_tree = cKDTree(raw)
            rng = np.random.RandomState(a.seed)
            frame_rows, best, best_q = [], None, None
            baseline_bits = None
            for_baseline = codec.encode(q, verify=True)
            baseline_bits = for_baseline['bits']

            def record(name, out, counts, sanity=False, preencoded=None):
                nonlocal best, best_q
                out = unique(out)
                removed, added = difference(q, out), difference(out, q)
                nadd, nprune, nadjust = [int(counts.get(k, 0)) for k in ('add', 'prune', 'adjust')]
                if not sanity and (len(removed) != nprune + nadjust or len(added) != nadd + nadjust):
                    raise AssertionError('Operation accounting mismatch: ' + name)
                rec = (out + origin) * a.qs
                d1 = cKDTree(rec).query(raw, workers=a.threads)[0]
                d2 = raw_tree.query(rec, workers=a.threads)[0]
                stats = codec.encode(out) if preencoded is None else preencoded
                row = dict(frame=str(frame), trial=name, **stats,
                    baseline_bits=baseline_bits, compression_percent=100. * (stats['bits'] / baseline_bits - 1.),
                    input_points=len(raw), baseline_voxels=len(q), edited_voxels=len(out),
                    add=nadd, prune=nprune, adjust=nadjust,
                    operation_percent=100. * (nadd + nprune + nadjust) / len(q),
                    removed_cells=len(removed), added_cells=len(added),
                    changed_cells_percent=100. * (len(removed) + len(added)) / len(q),
                    bbox_min=out.min(axis=0).tolist(), bbox_max=out.max(axis=0).tolist(),
                    chamfer_sq=float(np.mean(d1*d1) + np.mean(d2*d2)),
                    d1_gt_to_rec_sq=float(np.mean(d1*d1)), d1_rec_to_gt_sq=float(np.mean(d2*d2)),
                    d1_max_direction_mse=float(max(np.mean(d1*d1), np.mean(d2*d2))),
                    max_gt_to_rec_distance=float(d1.max()), max_rec_to_gt_distance=float(d2.max()),
                    sanity_only=sanity,
                    prediction={k: v for k,v in counts.items() if k not in ('add','prune','adjust')})
                log.write(json.dumps(row) + '\n')
                frame_rows.append(row)
                print(json.dumps({k: row[k] for k in ('trial', 'bits', 'compression_percent', 'operation_percent', 'chamfer_sq')}), flush=True)
                if not sanity and row['operation_percent'] <= 100 * a.max_ratio and (best is None or row['bits'] < best['bits']):
                    best, best_q = row, out.copy()
                return row

            record('baseline', q, {}, preencoded=for_baseline)
            if 'phase' in a.modes:
                # A global translation is a sanity check, NOT a micro edit.
                translated = record('sanity_translate_111', q + 1, dict(adjust=len(q)), sanity=True)
                shuffled = codec.encode(q[rng.permutation(len(q))])
                if translated['bits'] != baseline_bits or shuffled['bits'] != baseline_bits:
                    raise AssertionError('Translation/permutation invariant failed')
                for shift in itertools.product(range(a.phases), repeat=3):
                    if not any(shift):
                        continue
                    out = phase_add(q, shift)
                    record('anchor_add_' + ''.join(map(str, shift)), out, dict(add=len(out)-len(q)))
                    if a.anchor_only:
                        continue
                    out = phase_prune(q, shift)
                    if len(out) and (len(q) - len(out)) / len(q) <= a.max_ratio:
                        record('min_plane_prune_' + ''.join(map(str, shift)), out, dict(prune=len(q)-len(out)))
            if 'local' in a.modes:
                for name, out, counts in local_variants(q, a.budgets, rng):
                    record(name, out, counts)
            if 'control' in a.modes:
                for name, out, counts in controls(q):
                    record(name, out, counts)
            if 'prior' in a.modes:
                if a.qs != 2.:
                    raise ValueError('Current-prior audit is explicitly configured for qs=2')
                for name, out, counts in current_prior_variants(q, a.budgets):
                    record(name, out, counts)
            if 'combo' in a.modes:
                for name, out, counts in combo_variants(q, a.budgets):
                    record(name, out, counts)
            if 'adjustphase' in a.modes:
                for shift in itertools.product((0, 1), repeat=3):
                    if any(shift):
                        out, count = phase_adjust(q, shift)
                        record('anchor_adjust_' + ''.join(map(str, shift)), out, dict(adjust=count))
            if 'qualityguard' in a.modes:
                for name, out, counts in quality_guard_variants(q, raw, origin, a.qs, a.budgets):
                    row = record(name, out, counts)
                    if row['max_gt_to_rec_distance'] > counts['original_max_gt_distance'] + 1e-8:
                        raise AssertionError('Quality guard failed')
            if 'qualityguardbbox' in a.modes:
                for name, out, counts in quality_guard_variants(q, raw, origin, a.qs, a.budgets, preserve_bbox=True):
                    row = record(name, out, counts)
                    if (row['max_gt_to_rec_distance'] > counts['original_max_gt_distance'] + 1e-8
                            or not np.array_equal(q.min(0), out.min(0))
                            or not np.array_equal(q.max(0), out.max(0))):
                        raise AssertionError('Quality/bbox guard failed')
            if 'matchedprune' in a.modes:
                # Exactly match the Prune counts of current_prior_variants
                # (not only its total three-operation budget).
                for total_budget in a.budgets:
                    target = int(max(1, round(len(q)*total_budget))*.4)
                    for _, out, counts in quality_guard_variants(q, raw, origin, a.qs,
                            [target/len(q)], preserve_bbox=True):
                        record('matched_prune_for_total_%.5f' % total_budget, out, counts)
            if 'bboxcounterfactual' in a.modes:
                anchors = q[np.unique(np.r_[q.argmin(0),q.argmax(0)])]
                for name, out, counts in quality_guard_variants(q, raw, origin, a.qs, a.budgets):
                    restored = unique(np.vstack([out, anchors]))
                    record('bbox_restore_' + name, restored, dict(prune=len(q)-len(restored),
                           restored_anchor_count=len(restored)-len(out)))
            if 'bboxaxes' in a.modes:
                for name, out, counts in quality_guard_variants(q, raw, origin, a.qs, a.budgets):
                    for direction, chooser in (('min',np.argmin),('max',np.argmax)):
                        for axis in range(3):
                            index = chooser(q[:,axis])
                            if len(difference(q[index:index+1],out)):
                                restored = unique(np.vstack([out,q[index]]))
                                record('restore_%s_axis%d_' % (direction,axis)+name, restored,
                                       dict(prune=len(q)-len(restored),restored_anchor_count=1))
            verified = codec.encode(best_q, verify=True)
            if verified['bits'] != best['bits']:
                raise AssertionError('Repeated best-plan encode changed bits')
            np.savez_compressed(output / (Path(frame).stem + '_best.npz'),
                                canonical_voxels=best_q, original_grid_origin=origin, qs=a.qs)
            summaries.append(dict(frame=str(frame), best=best, best_decoded_exact=True,
                                  trials=len(frame_rows), baseline_decoded_exact=True,
                                  negative_trials=sum(r['compression_percent'] < 0 for r in frame_rows if not r['sanity_only']),
                                  candidate_actual_encodes=codec.calls))
            (output / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print('RESULTS ' + str(output), flush=True)


if __name__ == '__main__':
    main()
