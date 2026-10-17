#!/usr/bin/env python
"""Single-edit + adjacent-parent + topology controlled Actual benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from context_oracle_core import (OAActual,ActualCodec,WindowCodec,costs,quality,
    unique,load_ply,apply_plan,quality_ok,protect_snapshot)
from context_oracle_study import features,edit_list
from context_oracle_patches import patch_pool,compose
from models.utils.pointcloud.geometry_octree_rate import GeometryOctreeRate


def topology_cost(w,seq,prob):
    key=lambda s:[(int(row[-1,1]),)+tuple(map(int,row[-1,3:6])) for row in s]
    old_keys=key(w.seq);new_keys=key(seq)
    old={k:i for i,k in enumerate(old_keys)};new={k:i for i,k in enumerate(new_keys)}
    if set(new)-set(old):
        raise AssertionError('This benchmark requires deletion-only internal topology')
    new_symbols=(seq[:,-1,0]-1).astype(np.int16)
    new_cost=costs(prob,new_symbols)
    deleted=[old[k] for k in set(old)-set(new)]
    frozen=float(w.base_cost[deleted].sum());edited=frozen;unedited=0.
    changed=0;affected_prob=0;affected_cost=0
    for k,j in new.items():
        i=old[k]
        if w.symbols[i]!=new_symbols[j]:
            frozen+=float(w.all_cost[i,w.symbols[i]]-w.all_cost[i,new_symbols[j]])
            edited+=float(w.base_cost[i]-new_cost[j]);changed+=1
        else:
            delta=float(w.base_cost[i]-new_cost[j]);unedited+=delta
            affected_prob+=int(np.max(np.abs(w.probs[i]-prob[j]))>1e-7)
            affected_cost+=int(abs(delta)>1e-9)
    total=float(w.base_cost.sum()-new_cost.sum())
    if abs(edited+unedited-total)>1e-6:
        raise AssertionError('Topology cost decomposition failed')
    return dict(frozen_self_cdf_gain=frozen,edited_node_cdf_gain=edited,
        unedited_context_cdf_gain=unedited,total_cdf_gain=total,changed_nodes=changed,
        deleted_nodes=len(deleted),affected_unedited_probability_nodes=affected_prob,
        affected_unedited_cost_nodes=affected_cost)


def samples(c,f,seq,seed=73):
    rng=np.random.RandomState(seed);chosen=[];seen=set()
    def add(indices,n,kind):
        count=0
        for i in indices:
            i=int(i);key=(int(c[i]['row']),int(c[i]['mask']))
            if key in seen:
                continue
            seen.add(key);chosen.append((kind,[i]));count+=1
            if count>=n:
                break
    one=np.flatnonzero(c['cost']==1)
    add(one[np.argsort(-f['own'][one])],12,'single_top_own')
    add(one[np.argsort(-f['prior'][one])],12,'single_top_current')
    modest=one[(f['own'][one]>.1)&(f['own'][one]<3.)]
    add(rng.permutation(modest),16,'single_modest_own')
    add(rng.permutation(one),12,'single_random')
    multi=np.flatnonzero((c['cost']>1)&(c['cost']<=4))
    add(multi[np.argsort(-f['own'][multi]/c['cost'][multi])],4,'mask_multi_top')
    add(rng.permutation(multi),4,'mask_multi_random')
    # Adjacent physical parents, not just nearby sequence rows.
    by_row={}
    for i in np.argsort(-f['own']/c['cost'],kind='stable'):
        by_row.setdefault(int(c[i]['row']),int(i))
    pos_to_row={tuple(seq[r,-1,3:6]):r for r in by_row}
    count=0
    for i in np.argsort(-f['own']/c['cost'],kind='stable'):
        row=int(c[i]['row']);pos=seq[row,-1,3:6]
        found=False
        for axis in range(3):
            for sign in (-1,1):
                p=pos.copy();p[axis]+=2*sign
                other=pos_to_row.get(tuple(p))
                if other is not None:
                    chosen.append(('adjacent_parents',[int(i),by_row[other]]));count+=1;found=True
                    break
            if found:
                break
        if count>=8:
            break
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--candidates',default='experiments/results/20261017_octree_joint_masks_v2')
    a=p.parse_args();torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    protected=protect_snapshot(root)
    (folder/'protected_before.json').write_text(json.dumps(protected,indent=2))
    (folder/'arguments.json').write_text(json.dumps(vars(a),indent=2))
    oa,gpcc=OAActual('cpu'),ActualCodec(2.,2)
    with (folder/'results.jsonl').open('w',buffering=1) as log:
        def emit(x):
            log.write(json.dumps(x)+'\n');print(json.dumps(x),flush=True)
        for frame in a.frames:
            sub=folder/Path(frame).stem;sub.mkdir()
            raw=load_ply(frame,return_color=False).astype(np.float64)
            absolute=unique(np.rint(raw/2.));origin=absolute.min(0);q=absolute-origin
            w=WindowCodec(oa,q);geom_base=quality(raw,q,origin,2.);gbase=gpcc.encode(q,verify=True)
            cache=np.load(str(Path(a.candidates)/(Path(frame).stem+'_proposals.npz')))
            if not np.array_equal(cache['seq'],w.seq):
                raise AssertionError('Candidate source mismatch')
            c=cache['candidates'];c=c[c['row']%w.bptt!=0]
            model=GeometryOctreeRate(q,'gpcc_categorical')
            f,_=features(c,w.seq,q,raw,origin,w,model)
            emit(dict(frame=frame,trial='baseline',oa=w.baseline,gpcc=gbase,quality=geom_base))
            records=[]
            for number,(kind,indices) in enumerate(samples(c,f,w.seq)):
                edits=edit_list(c,indices)
                out,counts=apply_plan(q,w.seq,edits)
                geom=quality(raw,out,origin,2.)
                result=w.evaluate(edits,actual=True,return_probs=True)
                chunks=result.pop('_chunks');node_rows=result.pop('_node_rows')
                encoded=gpcc.encode(out)
                row=dict(frame=frame,trial='candidate',id=number,kind=kind,edits=edits,counts=counts,
                    operation_count=sum(counts.values()),current_prior_score=float(f['prior'][indices].sum()),
                    quality=geom,quality_ok=quality_ok(geom_base,geom),gpcc=encoded,
                    gpcc_gain=gbase['bits']-encoded['bits'],
                    gpcc_C=100*(encoded['bits']/gbase['bits']-1.),
                    oa_C=100*(result['actual_oa']['bits']/w.baseline['bits']-1.),
                    window_fraction_mean=float(np.mean([r%w.bptt/w.bptt for r,_ in edits])),
                    original_mask_population=float(np.mean([POPCOUNT(int(w.seq[r,-1,0])) for r,_ in edits])),
                    **result)
                emit(row);records.append(row)
                np.savez_compressed(sub/('candidate_%03d_costs.npz'%number),
                    ids=np.concatenate([x[0] for x in node_rows]),
                    gain=np.concatenate([x[1] for x in node_rows]),edited=np.concatenate([x[2] for x in node_rows]))
            patches,audit=patch_pool(q,raw,origin,6000)
            emit(dict(frame=frame,trial='patch_pool',**audit))
            topology=[]
            for level in (1,2,3):
                subset=[i for i,x in enumerate(patches) if x['level']==level]
                subset.sort(key=lambda i:(-patches[i]['depth']/patches[i]['cost'],-patches[i]['rescued']))
                topology.extend(subset[:2])
            topology.extend([i for i,x in enumerate(patches) if x['rescued']][:2])
            for index in dict.fromkeys(topology):
                patch=patches[index];occupied=set(map(tuple,q.tolist()))
                occupied.difference_update(map(tuple,patch['rem'].tolist()));occupied.update(map(tuple,patch['add'].tolist()))
                out=unique(list(occupied));geom=quality(raw,out,origin,2.)
                stat,seq,prob=oa.encode(out,detail=True)
                dec=topology_cost(w,seq,prob)
                encoded=gpcc.encode(out)
                _,prior=model.patch(dict(remove_coords=patch['rem'],add_coords=patch['add']))
                score=float(np.tanh((prior-.05*16*patch['cost'])/(4*patch['cost'])))
                row=dict(frame=frame,trial='candidate',id=len(records),kind='topology_level_%d'%patch['level'],
                    counts=patch['counts'],operation_count=patch['cost'],current_prior_score=score,
                    remove=patch['rem'].tolist(),add=patch['add'].tolist(),quality=geom,
                    quality_ok=quality_ok(geom_base,geom),gpcc=encoded,gpcc_gain=gbase['bits']-encoded['bits'],
                    gpcc_C=100*(encoded['bits']/gbase['bits']-1.),actual_oa=stat,
                    actual_oa_gain=w.baseline['bits']-stat['bits'],
                    oa_C=100*(stat['bits']/w.baseline['bits']-1.),**dec)
                emit(row);records.append(row)
            # Exact same admissible candidate IDs for all four strategies.
            eligible=[r for r in records if r['quality_ok']]
            noop=dict(id=-1,actual_oa=w.baseline,gpcc=gbase,oa_C=0.,gpcc_C=0.,quality=geom_base,
                current_prior_score=0.,frozen_self_cdf_gain=0.,total_cdf_gain=0.)
            eligible.append(noop)
            for name,key in dict(current_prior=lambda r:r['current_prior_score'],
                node_only=lambda r:r['frozen_self_cdf_gain'],total_context=lambda r:r['total_cdf_gain'],
                actual_oa_oracle=lambda r:-r['actual_oa']['bits'],actual_gpcc_oracle=lambda r:-r['gpcc']['bits']).items():
                winner=max(eligible,key=key)
                emit(dict(frame=frame,trial='same_pool_single_comparison',method=name,winner=winner['id'],
                    candidate_ids=[r['id'] for r in eligible],oa_C=winner['oa_C'],gpcc_C=winner['gpcc_C'],quality=winner['quality']))
            # Larger budgets, topology changes allowed, no-op retained explicitly.
            families=dict(node=np.array([x['depth']/x['cost'] for x in patches]),
                rescue=np.array([(2*bool(x['rescued'])+.25*x['depth'])/x['cost'] for x in patches]))
            for budget in (.01,.03,.05):
                for name,score in families.items():
                    out,audit,chosen=compose(q,patches,round(len(q)*budget),score)
                    geom=quality(raw,out,origin,2.)
                    stat=oa.encode(out,roundtrip=True);encoded=gpcc.encode(out,verify=True)
                    emit(dict(frame=frame,trial='topology_budget',method=name,budget=budget,**audit,
                        operation_percent=100*audit['operation_count']/len(q),quality=geom,
                        quality_ok=quality_ok(geom_base,geom),oa=stat,gpcc=encoded,
                        oa_C=100*(stat['bits']/w.baseline['bits']-1.),gpcc_C=100*(encoded['bits']/gbase['bits']-1.)))
                    np.savez_compressed(sub/('topology_%s_%.3f.npz'%(name,budget)),
                                        canonical_voxels=out,origin=origin,qs=2.)
    after=protect_snapshot(root)
    (folder/'protected_after.json').write_text(json.dumps(after,indent=2))
    changed=[p for p in protected if protected[p]!=after.get(p)]
    if changed:
        raise AssertionError('Protected source/codec/weights changed: '+repr(changed))
    print(json.dumps(dict(complete=True,protected_files=len(protected),changed=[])),flush=True)


def POPCOUNT(x):
    return bin(x).count('1')


if __name__=='__main__':
    main()
