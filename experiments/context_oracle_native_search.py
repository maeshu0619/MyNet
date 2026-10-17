#!/usr/bin/env python
"""Native G-PCC full-cloud oracle over fixed-window occupancy menus.

No OA independence is assumed for G-PCC: every proposal is a full native
encode in the current joint state. SparsePCGC and codec sources are untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from context_oracle_core import ActualCodec,unique,load_ply,apply_plan,quality,quality_ok
from context_oracle_study import edit_list


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',required=True);p.add_argument('--wide',required=True)
    p.add_argument('--out',required=True);p.add_argument('--budget',type=float,default=.05)
    p.add_argument('--max-options',type=int,default=5)
    a=p.parse_args();folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    (folder/'arguments.json').write_text(json.dumps(vars(a),indent=2))
    raw=load_ply(a.frame,return_color=False).astype(np.float64)
    absolute=unique(np.rint(raw/2.));origin=absolute.min(0);q=absolute-origin
    sub=Path(a.wide)/Path(a.frame).stem
    features=np.load(str(sub/'candidate_features.npz'));c=features['candidates']
    seq=np.load(str(Path('experiments/results/20261017_octree_joint_masks_v2')/(Path(a.frame).stem+'_proposals.npz')))['seq']
    menus=json.loads((sub/'menus.json').read_text())
    rows=[json.loads(s) for s in (Path(a.wide)/'results.jsonl').read_text().splitlines()]
    plans=[r for r in rows if r.get('frame')==a.frame and r.get('trial')=='full_plan'
           and r['operation_count']<=round(len(q)*a.budget) and r['quality_ok']]
    if not plans:
        raise ValueError('Wide experiment has not written its full plans yet')
    seed=min(plans,key=lambda r:r['gpcc']['bits'])
    seed_edits=np.load(str(sub/('plan_%03d.npz'%seed['id'])))['edits'].reshape(-1,2)
    identity={(int(x['row']),int(x['mask'])):i for i,x in enumerate(c)}
    chosen=[identity[tuple(map(int,e))] for e in seed_edits]
    gpcc=ActualCodec(2.,2);base=gpcc.encode(q,verify=True)
    baseline_quality=quality(raw,q,origin,2.)
    best=seed['gpcc']['bits'];initial=best
    cap=round(len(q)*a.budget);cache={};accepted=0;calls=0
    with (folder/'results.jsonl').open('w',buffering=1) as log:
        def emit(x):
            log.write(json.dumps(x)+'\n');print(json.dumps(x),flush=True)
        emit(dict(trial='baseline',frame=a.frame,gpcc=base,quality=baseline_quality,
                  initial_bits=best,initial_plan=seed['id'],operation_cap=cap))
        for number,options in enumerate(menus):
            # Finite mixed menu: no-op, best contextual, best empirical, best
            # self-rate and a high-cost state. Every option is available to
            # all selectors; this is coordinate search, not exhaustive oracle.
            ranked=[]
            for key in ('total','prior','own','cost'):
                ranked.append(max(range(len(options)),key=lambda i:options[i][key]))
            ranked=[0]+list(dict.fromkeys(ranked))
            group=int(c[options[-1]['selected'][0]]['row'])//1024 if options[-1]['selected'] else None
            if group is None:
                continue
            remaining=[i for i in chosen if int(c[i]['row'])//1024!=group]
            winner=chosen;winner_bits=best
            for index in ranked[:a.max_options]:
                proposed=remaining+options[index]['selected']
                if int(c['cost'][proposed].sum())>cap or features['slack'][proposed].sum()>1e-8:
                    continue
                key=tuple(sorted(proposed))
                if key in cache:
                    bits=cache[key]
                else:
                    out,counts=apply_plan(q,seq,edit_list(c,proposed))
                    encoded=gpcc.encode(out);bits=encoded['bits'];cache[key]=bits;calls+=1
                improve=bits<winner_bits
                emit(dict(trial='native_candidate',window=number,group=group,option=index,bits=bits,
                    gain_from_current=best-bits,accepted_within_window=improve,
                    operation_count=int(c['cost'][proposed].sum())))
                if improve:
                    winner=proposed;winner_bits=bits
            if winner_bits<best:
                accepted+=1;chosen=winner;best=winner_bits
            if (number+1)%10==0:
                emit(dict(trial='progress',windows=number+1,actual_encodes=calls,bits=best,
                          gpcc_C=100*(best/base['bits']-1.),accepted_windows=accepted))
        out,counts=apply_plan(q,seq,edit_list(c,chosen));geom=quality(raw,out,origin,2.)
        if not quality_ok(baseline_quality,geom):
            raise AssertionError('Native oracle output violates geometry constraint')
        encoded=gpcc.encode(out,verify=True)
        if encoded['bits']!=best:
            raise AssertionError('Native final rate differs')
        emit(dict(trial='final',frame=a.frame,initial_bits=initial,gpcc=encoded,
            gpcc_C=100*(best/base['bits']-1.),counts=counts,
            operation_percent=100*sum(counts.values())/len(q),quality=geom,
            unique_actual_encodes=calls,accepted_windows=accepted))
        np.savez_compressed(folder/'best.npz',canonical_voxels=out,origin=origin,qs=2.,
                            edits=np.array(edit_list(c,chosen),dtype=np.int64).reshape(-1,2))


if __name__=='__main__':
    main()
