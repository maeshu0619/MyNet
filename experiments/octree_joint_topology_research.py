#!/usr/bin/env python
"""Joint cross-parent cover/reroute proposals, allowing Octree node deletion.

For every proposed deleted parent, assigned raw points must remain at least as
well covered by retained neighbors or newly added voxels. All operations of a
patch are selected atomically. Replacement parents are reserved jointly.
"""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from octree_joint_mask_research import OAActual,ActualCodec,load_ply,unique,quality,quality_ok,difference


def operation_counts(rem,add):
    rem=np.asarray(rem,dtype=np.int64).reshape(-1,3)
    add=np.asarray(add,dtype=np.int64).reshape(-1,3)
    moves=0
    if len(rem) and len(add):
        legal=np.max(np.abs(rem[:,None]-add[None]),axis=2)<=1
        matching=maximum_bipartite_matching(csr_matrix(legal),perm_type='column')
        moves=int((matching>=0).sum())
    return dict(Add=len(add)-moves,Prune=len(rem)-moves,Adjust=moves)


def generate(q,raw,origin,max_parents):
    started=time.perf_counter()
    parents,inverse=np.unique(q//2,axis=0,return_inverse=True)
    lookup={tuple(p):i for i,p in enumerate(parents.tolist())}
    lengths=np.bincount(inverse)
    order=np.argsort(inverse,kind='stable')
    starts=np.r_[0,lengths.cumsum()]
    distance,near=cKDTree((q+origin)*2.).query(raw,k=9,workers=2)
    raw_parent=inverse[near[:,0]]
    order_raw=np.argsort(raw_parent,kind='stable')
    starts_raw=np.r_[0,np.bincount(raw_parent,minlength=len(parents)).cumsum()]
    occupied=set(map(tuple,q.tolist()))
    rt=cKDTree(raw)
    bbox_min,bbox_max=q.min(0),q.max(0)
    protected={int(i) for i in inverse[np.unique(np.r_[q.argmin(0),q.argmax(0)])]}
    offsets=np.array(list(itertools.product((-1,0,1),repeat=3)))
    child_counts=[]
    for level in range(10):
        nodes=np.unique(parents//(1<<level),axis=0)
        outer,nchild=np.unique(nodes//2,axis=0,return_counts=True)
        child_counts.append({tuple(p):int(c) for p,c in zip(outer,nchild)})
    # Sample all spatial ranges uniformly; this finite search is not exhaustive.
    sampled=np.unique(np.arange(min(len(parents),max_parents))*len(parents)//min(len(parents),max_parents))
    result=[]
    for g in sampled:
        if g in protected:
            continue
        rem=q[order[starts[g]:starts[g+1]]]
        assigned=order_raw[starts_raw[g]:starts_raw[g+1]]
        outside=inverse[near[assigned]]!=g
        if not outside.any(1).all():
            raise AssertionError('Nine nearest voxels should include an outside parent')
        first=outside.argmax(1)
        fallback_ids=near[assigned,first]
        fallback_distance=distance[assigned,first]
        baseline=distance[assigned,0]
        retained_cover=fallback_distance<=baseline+1e-9
        uncovered=~retained_cover
        reserved=set(inverse[fallback_ids[retained_cover]].tolist())
        targets=unique((rem[:,None]+offsets[None]).reshape(-1,3))
        targets=np.array([p for p in targets if tuple(p) not in occupied
                          and tuple(p//2) in lookup and lookup[tuple(p//2)]!=g
                          and (p>=bbox_min).all() and (p<=bbox_max).all()],dtype=np.int64).reshape(-1,3)
        additions=[]
        if uncovered.any():
            if not len(targets):
                continue
            target_distance=rt.query((targets+origin)*2.,workers=2)[0]
            targets=targets[target_distance<=np.sqrt(3)+1e-9]
            if not len(targets):
                continue
            distance_sq=np.sum((raw[assigned,None]-((targets+origin)*2.)[None])**2,axis=2)
            cover=distance_sq<=baseline[:,None]**2+1e-8
            while uncovered.any():
                gain=cover[uncovered].sum(0)
                best=int(gain.argmax())
                if gain[best]==0:
                    break
                additions.append(targets[best])
                uncovered &= ~cover[:,best]
            if uncovered.any():
                continue
        add=unique(additions)
        reserved.update(lookup[tuple(p//2)] for p in add)
        counts=operation_counts(rem,add)
        # Ancestor occupancy chain removed by erasing this whole leaf parent.
        depth=1
        current=parents[g].copy()
        for level in range(10):
            if np.all(current==0) or child_counts[level][tuple(current//2)]!=1:
                break
            depth+=1
            current//=2
        result.append(dict(source=int(g),rem=rem,add=add,reserved=reserved,counts=counts,
                           cost=sum(counts.values()),depth=depth,
                           rescued_raw_points=int((~retained_cover).sum())))
    return result,dict(sampled_parents=len(sampled),total_parents=len(parents),
                       candidates=len(result),generation_sec=time.perf_counter()-started,
                       candidates_requiring_add_rescue=sum(bool(x['rescued_raw_points']) for x in result))


def compose(q,candidates,k,ranking):
    def score(x):
        if ranking=='rescue':
            return (2*bool(x['rescued_raw_points'])+.25*x['depth'])/x['cost']
        if ranking=='node':
            return x['depth']/x['cost']
        if ranking=='prune':
            return (len(x['rem'])-len(x['add'])+.1*x['depth'])/x['cost']
        return (x['depth']+.5*len(x['rem'])-.25*len(x['add']))/x['cost']
    chosen=[]
    sources,reserved=set(),set()
    used=0
    occupied=set(map(tuple,q.tolist()))
    counts=dict(Add=0,Prune=0,Adjust=0)
    for item in sorted(candidates,key=score,reverse=True):
        if item['source'] in reserved or item['reserved'] & sources or used+item['cost']>k:
            continue
        if any(tuple(p) in occupied for p in item['add']):
            continue
        occupied.difference_update(map(tuple,item['rem'].tolist()))
        occupied.update(map(tuple,item['add'].tolist()))
        chosen.append(item)
        sources.add(item['source'])
        reserved.update(item['reserved'])
        used+=item['cost']
        for op in counts:
            counts[op]+=item['counts'][op]
    out=unique(list(occupied))
    nr,na=len(difference(q,out)),len(difference(out,q))
    if nr!=counts['Prune']+counts['Adjust'] or na!=counts['Add']+counts['Adjust']:
        raise AssertionError('Overlapping edits / accounting error')
    if not np.array_equal(q.min(0),out.min(0)) or not np.array_equal(q.max(0),out.max(0)):
        raise AssertionError('Bounding box changed')
    return out,dict(counts=counts,removed_leaf_parents=len(chosen),
        joint_rescue_patches=sum(bool(x['rescued_raw_points']) for x in chosen),
        rescued_raw_points=sum(x['rescued_raw_points'] for x in chosen))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--budget',type=float,default=.0015)
    p.add_argument('--max-parents',type=int,default=12000)
    p.add_argument('--rankings',nargs='+',choices=['node','prune','balanced','rescue'],
                   default=['node','prune','balanced'])
    a=p.parse_args()
    torch.set_num_threads(2)
    folder=Path(a.out)
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'arguments.json').write_text(json.dumps(vars(a),indent=2))
    gpcc,oa=ActualCodec(2.,2),OAActual('cpu')
    with (folder/'trials.jsonl').open('w',buffering=1) as log:
        def emit(x):
            log.write(json.dumps(x)+'\n')
            print(json.dumps(x),flush=True)
        for frame in a.frames:
            raw=load_ply(frame,return_color=False).astype(np.float64)
            absolute=unique(np.rint(raw/2.))
            origin=absolute.min(0)
            q=absolute-origin
            bq=quality(raw,q,origin,2.)
            bg=gpcc.encode(q,verify=True)
            bo=oa.encode(q,roundtrip=True)
            emit(dict(frame=frame,trial='baseline',gpcc=bg,oa=bo,quality=bq))
            candidates,audit=generate(q,raw,origin,a.max_parents)
            emit(dict(frame=frame,trial='proposal_audit',**audit))
            for ranking in a.rankings:
                out,counts=compose(q,candidates,round(len(q)*a.budget),ranking)
                geom=quality(raw,out,origin,2.)
                if not quality_ok(bq,geom):
                    emit(dict(frame=frame,trial=ranking,quality=geom,quality_ok=False,**counts))
                    continue
                stats=gpcc.encode(out,verify=True)
                ostat=oa.encode(out,roundtrip=True)
                emit(dict(frame=frame,trial=ranking,quality=geom,quality_ok=True,**counts,
                    operation_percent=100*sum(counts['counts'].values())/len(q),gpcc=stats,oa=ostat,
                    gpcc_C=100*(stats['bits']/bg['bits']-1.),oa_C=100*(ostat['bits']/bo['bits']-1.)))
                np.savez_compressed(folder/(Path(frame).stem+'_'+ranking+'.npz'),canonical_voxels=out,origin=origin,qs=2.)


if __name__=='__main__':
    main()
