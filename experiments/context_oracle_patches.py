"""Independent topology patches, including 4^3 and 8^3 source regions."""
import numpy as np
from scipy.spatial import cKDTree
from octree_joint_topology_research import generate,operation_counts
from context_oracle_core import unique,difference


def patch_pool(q,raw,origin,maximum=6000):
    leaves,audit=generate(q,raw,origin,maximum)
    parents,inv=np.unique(q//2,axis=0,return_inverse=True)
    ordering=np.argsort(inv,kind='stable');starts=np.r_[0,np.bincount(inv).cumsum()]
    group_ids=[ordering[starts[g]:starts[g+1]] for g in range(len(parents))]
    qid={tuple(p):i for i,p in enumerate(q.tolist())}
    patches=[]
    for x in leaves:
        patches.append(dict(rem=x['rem'],add=x['add'],
            remove_ids={qid[tuple(p)] for p in x['rem']},
            keep_ids=set(i for g in x['reserved'] for i in group_ids[g]),
            counts=x['counts'],cost=x['cost'],level=1,depth=x['depth'],rescued=x['rescued_raw_points']))
    distances,neighbors=cKDTree((q+origin)*2.).query(raw,k=17,workers=2)
    protected=set(np.unique(np.r_[q.argmin(0),q.argmax(0)]).tolist())
    summaries=[]
    for level in (2,3):
        groups,inv=np.unique(q>>level,axis=0,return_inverse=True)
        lengths=np.bincount(inv)
        ids=np.argsort(inv,kind='stable');starts=np.r_[0,lengths.cumsum()]
        assigned=inv[neighbors[:,0]]
        raw_order=np.argsort(assigned,kind='stable')
        raw_starts=np.r_[0,np.bincount(assigned,minlength=len(groups)).cumsum()]
        count=0
        for g in np.flatnonzero(lengths<=16):
            remove=ids[starts[g]:starts[g+1]]
            if set(remove)&protected:
                continue
            raw_ids=raw_order[raw_starts[g]:raw_starts[g+1]]
            outside=inv[neighbors[raw_ids]]!=g
            if not outside.any(1).all():
                raise AssertionError('Missing outside fallback')
            first=outside.argmax(1)
            if not (distances[raw_ids,first]<=distances[raw_ids,0]+1e-9).all():
                continue
            keep=set(neighbors[raw_ids,first].tolist())
            patches.append(dict(rem=q[remove],add=np.zeros((0,3),dtype=np.int64),
                remove_ids=set(remove.tolist()),keep_ids=keep,
                counts=dict(Add=0,Prune=len(remove),Adjust=0),cost=len(remove),level=level,
                depth=level,rescued=0))
            count+=1
        summaries.append(dict(source_level=level,eligible_source_groups=int((lengths<=16).sum()),
                              quality_safe_candidates=count))
    return patches,dict(leaf=audit,higher_level=summaries,total_patches=len(patches))


def compose(q,patches,cap,score):
    occupied=set(map(tuple,q.tolist()));removed,keep=set(),set()
    counts=dict(Add=0,Prune=0,Adjust=0);chosen=[];used=0
    for i in np.argsort(-np.asarray(score),kind='stable'):
        x=patches[i]
        if used+x['cost']>cap or x['remove_ids']&(removed|keep) or x['keep_ids']&removed:
            continue
        if any(tuple(p) in occupied for p in x['add']):
            continue
        occupied.difference_update(map(tuple,x['rem'].tolist()));occupied.update(map(tuple,x['add'].tolist()))
        removed.update(x['remove_ids']);keep.update(x['keep_ids'])
        for op in counts:
            counts[op]+=x['counts'][op]
        chosen.append(int(i));used+=x['cost']
    out=unique(list(occupied))
    if len(difference(q,out))!=counts['Prune']+counts['Adjust'] or len(difference(out,q))!=counts['Add']+counts['Adjust']:
        raise AssertionError('Patch accounting overlap')
    if not np.array_equal(out.min(0),q.min(0)) or not np.array_equal(out.max(0),q.max(0)):
        raise AssertionError('Patch coding origin changed')
    return out,dict(counts=counts,selected_patches=len(chosen),
        source_levels={str(level):sum(patches[i]['level']==level for i in chosen) for level in (1,2,3)},
        operation_count=used),chosen
