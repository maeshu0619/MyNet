#!/usr/bin/env python
"""Quality-checked coding-grid phase anchors composed with joint mask edits.

An added boundary voxel may shift all normalized encoding coordinates. Existing
physical points never move. Phase-only and composition are both encoded, with
the same total operation cap. This diagnoses global context effects, not a
claim of independently decoded absolute coordinates without origin metadata.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from octree_joint_mask_research import (OAActual, ActualCodec, quality, quality_ok,
    unique, difference, load_ply, apply_plan, POPS)


def anchors(q, raw, origin):
    raw_tree = cKDTree(raw)
    forward = cKDTree((q+origin)*2.).query(raw, workers=2)[0]
    backward = raw_tree.query((q+origin)*2., workers=2)[0]
    result, audits = {}, []
    for axis in range(3):
        targets = q[q[:, axis] == q[:, axis].min()].copy()
        targets[:, axis] -= 1
        distances = raw_tree.query((targets+origin)*2., workers=2)[0]
        eligible = distances <= backward.max()+1e-9
        candidates = targets[eligible]
        scores = []
        for target, distance in zip(candidates, distances[eligible]):
            physical = (target+origin)*2.
            ids = raw_tree.query_ball_point(physical, forward.max()+1e-9)
            new_sq = np.sum((raw[ids]-physical)**2, axis=1)
            saved = np.maximum(forward[ids]**2-new_sq, 0.).sum()/len(raw)
            back_delta = (np.sum(backward**2)+distance**2)/(len(q)+1)-np.mean(backward**2)
            scores.append(float(back_delta-saved))
        if len(candidates):
            best = int(np.argmin(scores))
            result[axis] = candidates[best]
        audits.append(dict(axis=axis, candidates=len(targets), within_original_max_distance=len(candidates),
                           best_predicted_chamfer_delta=min(scores) if scores else None))
    return result, audits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', nargs='+', required=True)
    p.add_argument('--joint-results', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--budget', type=float, default=.0015)
    a = p.parse_args()
    torch.set_num_threads(2)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=False)
    gpcc, oa = ActualCodec(2.,2), OAActual('cpu')
    with (folder/'trials.jsonl').open('w', buffering=1) as log:
        def emit(x):
            log.write(json.dumps(x)+'\n')
            print(json.dumps(x),flush=True)
        for frame in a.frames:
            raw = load_ply(frame, return_color=False).astype(np.float64)
            absolute = unique(np.rint(raw/2.))
            origin = absolute.min(0)
            q = absolute-origin
            cap = round(len(q)*a.budget)
            bq = quality(raw,q,origin,2.)
            bg = gpcc.encode(q,verify=True)
            bo = oa.encode(q,roundtrip=True)
            picks, audit = anchors(q,raw,origin)
            emit(dict(frame=frame,trial='baseline',gpcc=bg,oa=bo,quality=bq,anchor_audit=audit))
            stem = Path(frame).stem
            proposal = np.load(str(Path(a.joint_results)/(stem+'_proposals.npz')))
            seq = proposal['seq']
            joint = np.load(str(Path(a.joint_results)/(stem+'_%.5f_best.npz'%a.budget)))['chosen']
            trial_records = []
            for count in range(len(picks)+1):
                for axes in itertools.combinations(sorted(picks),count):
                    extra = np.array([picks[x] for x in axes],dtype=np.int64).reshape(-1,3)
                    for compose in (False,True):
                        if not compose and not axes:
                            continue
                        chosen, used = [],len(extra)
                        if compose:
                            for row,mask in joint:
                                before = int(seq[row,-1,0])
                                cost = max(POPS[before & ~mask],POPS[mask & ~before])
                                if used+cost <= cap:
                                    chosen.append((int(row),int(mask)))
                                    used += cost
                        middle,counts = apply_plan(q,seq,chosen)
                        out = unique(np.vstack([middle,extra]))
                        counts['Add'] += len(extra)
                        if len(out) != len(middle)+len(extra):
                            raise AssertionError('Anchor is not a new voxel')
                        if sum(counts.values()) > cap:
                            raise AssertionError('Operation cap exceeded')
                        nr,na=len(difference(q,out)),len(difference(out,q))
                        if nr!=counts['Prune']+counts['Adjust'] or na!=counts['Add']+counts['Adjust']:
                            raise AssertionError('Final voxel accounting differs')
                        geom = quality(raw,out,origin,2.)
                        stats = gpcc.encode(out)
                        # Extra origin/phase side info is not packed by this diagnostic.
                        # Show an explicit conservative byte overhead, not free metadata.
                        side_bits = 8 if axes else 0
                        result = dict(frame=frame,trial='joint_phase' if compose else 'phase_only',
                            axes=list(axes),counts=counts,operation_percent=100*sum(counts.values())/len(q),
                            gpcc=stats,gpcc_C=100*(stats['bits']/bg['bits']-1.),
                            estimated_side_bits=side_bits, estimated_net_gpcc_C=100*((stats['bits']+side_bits)/bg['bits']-1.),
                            quality=geom,quality_ok=quality_ok(bq,geom))
                        emit(result)
                        trial_records.append((result,out))
            safe = [x for x in trial_records if x[0]['quality_ok']]
            if not safe:
                emit(dict(frame=frame,trial='no_safe_phase_plan'))
                continue
            best,out = min(safe,key=lambda x:x[0]['estimated_net_gpcc_C'])
            encoded = gpcc.encode(out,verify=True)
            if encoded['stream_sha256']!=best['gpcc']['stream_sha256']:
                raise AssertionError('Nondeterministic Actual')
            ostat = oa.encode(out,roundtrip=True)
            emit(dict(frame=frame,trial='best_validated',selection=best,gpcc=encoded,oa=ostat,
                oa_C=100*(ostat['bits']/bo['bits']-1.),
                estimated_net_oa_C=100*((ostat['bits']+best['estimated_side_bits'])/bo['bits']-1.)))
            np.savez_compressed(folder/(stem+'_best.npz'),canonical_voxels=out,origin=origin,qs=2.)


if __name__=='__main__':
    main()
