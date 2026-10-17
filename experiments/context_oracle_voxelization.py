#!/usr/bin/env python
"""Quality-safe local tie-breaking voxelization; parent creation is allowed.

Only integer raw points exactly halfway between qs=2 sites can change their
assignment. Each raw point keeps a nearest grid site. Shared support cells are
reserved jointly; final geometry and complete native rates decide success.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from context_oracle_core import OAActual,ActualCodec,load_ply,unique,quality,quality_ok,difference
from octree_joint_topology_research import operation_counts
from models.utils.pointcloud.geometry_octree_rate import GeometryOctreeRate


def make_patches(q,raw,origin,level,rule):
    original=set(map(tuple,q.tolist()))
    protected=set(map(tuple,q[np.unique(np.r_[q.argmin(0),q.argmax(0)])].tolist()))
    assigned=np.rint(raw/2.).astype(np.int64)-origin
    alternate=(np.floor(raw/2.) if rule=='floor' else np.floor(raw/2.+.5)).astype(np.int64)-origin
    groups,inv=np.unique(assigned>>level,axis=0,return_inverse=True)
    order=np.argsort(inv,kind='stable');starts=np.r_[0,np.bincount(inv).cumsum()]
    all_points=unique(np.vstack([q,alternate]));dist=cKDTree(raw).query((all_points+origin)*2.,workers=2)[0]**2
    backward={tuple(p):float(d) for p,d in zip(all_points.tolist(),dist)}
    mean=float(np.mean([backward[tuple(p)] for p in q]))
    model=GeometryOctreeRate(q,'gpcc_categorical')
    patches=[];bbox_min,bbox_max=q.min(0),q.max(0)
    for g in range(len(groups)):
        ids=order[starts[g]:starts[g+1]]
        old=unique(assigned[ids]);new=unique(alternate[ids])
        rem=difference(old,new);add=np.array([p for p in new if tuple(p) not in original],dtype=np.int64).reshape(-1,3)
        if not len(rem) and not len(add):
            continue
        remove=set(map(tuple,rem.tolist()))
        if remove & protected or (len(add) and ((add<bbox_min).any() or (add>bbox_max).any())):
            continue
        slack=sum(backward[tuple(p)]-mean for p in add)-sum(backward[tuple(p)]-mean for p in rem)
        if slack>1e-9:
            continue
        counts=operation_counts(rem,add);cost=sum(counts.values())
        if not cost:
            continue
        kept=set(map(tuple,new.tolist()))&original
        _,gain=model.patch(dict(remove_coords=rem,add_coords=add))
        patches.append(dict(remove=remove,add=set(map(tuple,add.tolist())),keep=kept,
            counts=counts,cost=cost,slack=slack,gain=gain,source_group=groups[g].tolist()))
    return patches,dict(source_level=level,tie_rule=rule,total_groups=len(groups),
        legal_patches=len(patches),parent_creation_allowed=True,
        total_patch_operation_capacity=sum(x['cost'] for x in patches))


def compose(q,patches,cap):
    occupied=set(map(tuple,q.tolist()));removed,added,keep=set(),set(),set();chosen=[];cost=0
    # Proposal prior only. Actual codec is always evaluated after composition.
    for x in sorted(patches,key=lambda x:x['gain']/x['cost'],reverse=True):
        if x['gain']<=0:
            break
        if x['remove'] & (removed|keep) or x['keep'] & removed or cost+x['cost']>cap:
            continue
        removed.update(x['remove']);added.update(x['add']);keep.update(x['keep'])
        cost+=x['cost'];chosen.append(x)
    out=unique(list((occupied-removed)|added))
    rem=difference(q,out);add=difference(out,q)
    # Each selected patch has an explicit executable local matching. Duplicate
    # added sites may make global minimum operation count lower, never higher.
    counts=operation_counts(rem,add)
    if sum(counts.values())>cap:
        raise AssertionError('Final voxelization exceeded operation cap')
    return out,dict(counts=counts,operation_count=sum(counts.values()),
        conservative_patch_cost=cost,selected_patches=len(chosen),
        removed_cells=len(rem),added_cells=len(add))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    (folder/'arguments.json').write_text(json.dumps(vars(a),indent=2))
    oa,gpcc=OAActual('cpu'),ActualCodec(2.,2)
    with (folder/'results.jsonl').open('w',buffering=1) as log:
        def emit(x):
            log.write(json.dumps(x)+'\n');print(json.dumps(x),flush=True)
        for frame in a.frames:
            raw=load_ply(frame,return_color=False).astype(np.float64)
            if not np.array_equal(raw,np.rint(raw)):
                raise ValueError('Tie-breaking proof assumes integer raw geometry')
            absolute=unique(np.rint(raw/2.));origin=absolute.min(0);q=absolute-origin
            bq=quality(raw,q,origin,2.);bo=oa.encode(q,roundtrip=True);bg=gpcc.encode(q,verify=True)
            emit(dict(frame=frame,trial='baseline',oa=bo,gpcc=bg,quality=bq))
            for level in (2,3):
                for rule in ('floor','halfup'):
                    patches,audit=make_patches(q,raw,origin,level,rule)
                    emit(dict(frame=frame,trial='proposal_audit',**audit))
                    for budget in (.01,.03,.05):
                        out,details=compose(q,patches,round(len(q)*budget))
                        geom=quality(raw,out,origin,2.)
                        if not quality_ok(bq,geom):
                            raise AssertionError('Tie-break support proof failed final geometry')
                        stat,seq,prob=oa.encode(out,detail=True,roundtrip=True)
                        encoded=gpcc.encode(out,verify=True)
                        old_parents={tuple(p) for p in np.unique(q//2,axis=0)}
                        new_parents={tuple(p) for p in np.unique(out//2,axis=0)}
                        emit(dict(frame=frame,trial='local_voxelization',budget=budget,
                            source_level=level,tie_rule=rule,**details,quality=geom,quality_ok=True,
                            created_leaf_parents=len(new_parents-old_parents),deleted_leaf_parents=len(old_parents-new_parents),
                            operation_percent=100*details['operation_count']/len(q),oa=stat,gpcc=encoded,
                            oa_C=100*(stat['bits']/bo['bits']-1.),gpcc_C=100*(encoded['bits']/bg['bits']-1.)))
                        name=Path(frame).stem+'_L%d_%s_%.3f.npz'%(level,rule,budget)
                        np.savez_compressed(folder/name,canonical_voxels=out,origin=origin,qs=2.)


if __name__=='__main__':
    main()
