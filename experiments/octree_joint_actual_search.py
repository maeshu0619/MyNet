#!/usr/bin/env python
"""Full-cloud G-PCC oracle search over coupled occupancy rewrite blocks.

No per-operation turn taking. A state is the complete final leaf-mask set;
Where and Amount change together under a shared cap, never a mandatory count.
Native Actual bytes and final geometry, not a proxy gain, decide acceptance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from octree_joint_mask_research import (OAActual,ActualCodec,load_ply,unique,quality,
    quality_ok,apply_plan,select,POPS)


def capacity_trim(chosen, seq, cap):
    out, used = [],0
    for row,after in chosen:
        before=int(seq[row,-1,0])
        cost=int(max(POPS[before & ~after],POPS[after & ~before]))
        if used+cost<=cap:
            out.append((int(row),int(after)))
            used+=cost
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',required=True)
    p.add_argument('--joint-results',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--budget',type=float,default=.0015)
    p.add_argument('--blocks',type=int,default=16)
    p.add_argument('--rounds',type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(2)
    folder=Path(a.out)
    folder.mkdir(parents=True,exist_ok=False)
    raw=load_ply(a.frame,return_color=False).astype(np.float64)
    absolute=unique(np.rint(raw/2.))
    origin=absolute.min(0)
    q=absolute-origin
    cap=round(len(q)*a.budget)
    stem=Path(a.frame).stem
    data=np.load(str(Path(a.joint_results)/(stem+'_proposals.npz')))
    candidates,seq=data['candidates'],data['seq']
    state=[tuple(map(int,p)) for p in np.load(str(Path(a.joint_results)/(stem+'_%.5f_best.npz'%a.budget)))['chosen']]
    gpcc,oa=ActualCodec(2.,2),OAActual('cpu')
    base_quality=quality(raw,q,origin,2.)
    base_gpcc=gpcc.encode(q,verify=True)
    base_oa=oa.encode(q,roundtrip=True)
    best_out,counts=apply_plan(q,seq,state)
    best=gpcc.encode(best_out,verify=True)['bits']
    initial=best
    accepted=0
    cache={}
    donors=[]
    for weights in (dict(oa=1.),dict(oa=1.,freq=1.),dict(oa=.5,freq=1.,neighbor=1.,prune=1.),
                    dict(oa=1.,neighbor=1.),dict(oa=1.,prune=2.)):
        chosen,_=select(candidates,weights,cap)
        donors.append(chosen)
    row_min=int(np.flatnonzero(seq[:,-1,1]==seq[:,-1,1].max()).min())
    row_span=len(seq)-row_min

    def group(row):
        return min(a.blocks-1, (row-row_min)*a.blocks//row_span)

    with (folder/'trials.jsonl').open('w',buffering=1) as log:
        def emit(x):
            log.write(json.dumps(x)+'\n')
            print(json.dumps(x),flush=True)

        def evaluate(proposed,kind,round_index,block):
            nonlocal best,best_out,state,accepted
            proposed=capacity_trim(proposed,seq,cap)
            key=tuple(sorted(proposed))
            cached=key in cache
            if cached:
                bits,geom,count=cache[key]
                out=None
            else:
                out,count=apply_plan(q,seq,proposed)
                geom=quality(raw,out,origin,2.)
                bits=gpcc.encode(out)['bits']
                cache[key]=bits,geom,count
            safe=quality_ok(base_quality,geom)
            improve=bool(safe and bits<best)
            emit(dict(trial=kind,round=round_index,block=block,bits=bits,
                gpcc_C=100*(bits/base_gpcc['bits']-1.),before_best_bits=best,
                quality_ok=safe,quality=geom,counts=count,accepted=improve,cached=cached))
            if improve:
                best=bits
                state=proposed
                best_out=out if out is not None else apply_plan(q,seq,state)[0]
                accepted+=1

        emit(dict(trial='baseline',frame=a.frame,gpcc=base_gpcc,oa=base_oa,quality=base_quality,
                  initial_joint_bits=initial,cap=cap))
        for iteration in range(a.rounds):
            # Alter mixed-operation spatial/sequence blocks, not operation families.
            for block in range(a.blocks):
                evaluate([x for x in state if group(x[0])!=block], 'drop_joint_block',iteration,block)
            for block in range(a.blocks):
                donor=donors[(block+iteration)%len(donors)]
                remaining=[x for x in state if group(x[0])!=block]
                inserts=[x for x in donor if group(x[0])==block]
                evaluate(remaining+inserts,'replace_joint_block',iteration,block)
        verified=gpcc.encode(best_out,verify=True)
        if verified['bits']!=best:
            raise AssertionError('Final native re-encode differs')
        edited_oa=oa.encode(best_out,roundtrip=True)
        counts=apply_plan(q,seq,state)[1]
        emit(dict(trial='final',frame=a.frame,initial_bits=initial,gpcc=verified,
            gpcc_C=100*(best/base_gpcc['bits']-1.),oa=edited_oa,
            oa_C=100*(edited_oa['bits']/base_oa['bits']-1.),counts=counts,
            operation_percent=100*sum(counts.values())/len(q),accepted_steps=accepted,
            unique_proposals=len(cache),quality=quality(raw,best_out,origin,2.)))
        np.savez_compressed(folder/'best.npz',canonical_voxels=best_out,origin=origin,
                            chosen=np.array(state,dtype=np.int64),qs=2.)


if __name__=='__main__':
    main()
