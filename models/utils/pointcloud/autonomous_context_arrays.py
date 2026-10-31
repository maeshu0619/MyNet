"""Array implementation of native-context proposals, preserving v1 ordering.

No probability model, sampling budget, random draw or candidate ID changes.
Only selected candidates become Python dictionaries. Coarse-cell joins and
first-occurrence deduplication use integer arrays, including overflow fallback.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


class SpatialCandidates:
    """Internal generated spatial rows; final candidate fields are unchanged."""
    def __init__(self,rows,geometry):
        self.rows=np.asarray(rows,dtype=np.int64).reshape(-1,6)
        self.geometry=np.asarray(geometry,dtype=np.float64)

    def __len__(self):
        return len(self.rows)


def _keys(rows, lower=None, spans=None):
    rows=np.ascontiguousarray(rows,dtype=np.int64)
    if lower is None:
        lower=rows.min(0)
        spans=[int(b)-int(a)+1 for a,b in zip(lower,rows.max(0))]
    product=1
    for span in spans:product*=span
    if product >= 2**63:
        return rows.view(np.dtype((np.void,rows.shape[1]*8))).reshape(-1),None,None
    shifted=rows-lower
    result=np.zeros(len(rows),dtype=np.int64)
    for axis,span in enumerate(spans):result=result*span+shifted[:,axis]
    return result,lower,spans


def first_rows(rows):
    if not len(rows):return np.empty(0,dtype=np.int64)
    keys,_,_=_keys(rows)
    _,first=np.unique(keys,return_index=True)
    return np.sort(first)


class RowIndex:
    def __init__(self,rows):
        self.rows=np.asarray(rows,dtype=np.int64)
        if not len(self.rows):
            self.keys=self.sorted=self.order=np.empty(0,dtype=np.int64)
            self.lower=self.spans=None
            return
        self.keys,self.lower,self.spans=_keys(self.rows)
        self.order=np.argsort(self.keys)
        self.sorted=self.keys[self.order]

    def find(self,queries):
        queries=np.ascontiguousarray(queries,dtype=np.int64)
        if not len(queries):return np.empty(0,dtype=np.int64)
        if not len(self.rows):return np.full(len(queries),-1,dtype=np.int64)
        if self.lower is None:
            keys=queries.view(np.dtype((np.void,queries.shape[1]*8))).reshape(-1)
            in_range=np.ones(len(queries),dtype=bool)
        else:
            in_range=((queries>=self.lower)&(queries<=self.rows.max(0))).all(1)
            bounded=np.where(in_range[:,None],queries,self.lower)
            keys,_,_=_keys(bounded,self.lower,self.spans)
        position=np.searchsorted(self.sorted,keys)
        safe=np.minimum(position,len(self.sorted)-1)
        valid=in_range&(position<len(self.sorted))&(self.sorted[safe]==keys)
        return np.where(valid,self.order[safe],-1)


def context_proposals_arrays(source,context_path,budget,seed=0,random_pools=None):
    from scipy.spatial import cKDTree
    from ..loss.autonomous_native_loader import lexicographic_unique_rows
    from .autonomous_context_proposals import canonical_voxel_sha,cell_counts
    source=np.asarray(source,dtype=np.int64)
    canonical=lexicographic_unique_rows(source)
    source=canonical if canonical is not None else np.unique(source,axis=0)
    with np.load(context_path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata'].reshape(())))
        if metadata.get('input_voxel_sha256',metadata.get('input_sha'))!=canonical_voxel_sha(source):
            raise ValueError('Native codec context belongs to a different GT voxel set')
        scale=int(metadata['scale']);cells=data['coords'].astype(np.int64)
        p=data['probability'].astype(float);occupied=data['occupied'].astype(bool)
    if scale<=0 or not np.isfinite(p).all() or np.any((p<=0)|(p>=1)):
        raise ValueError('Invalid native coded-level probabilities/scale')
    if len(first_rows(cells))!=len(cells):raise ValueError('Native coded-level context has duplicate symbols')
    logit=np.log2(p)-np.log2(1-p)
    native_index=RowIndex(cells)
    unique,inverse,counts=cell_counts(source//scale)
    source_index=RowIndex(unique)
    operations=('Add','Prune','Adjust')
    cache_path=Path(str(context_path)+'.proposals_v1_%d_%d.npz'%(int(budget),int(seed)&0x7fffffff))
    cached=None
    if cache_path.is_file():
        with np.load(cache_path,allow_pickle=False) as data:cached={op:data[op].astype(np.int64) for op in operations}

    def observe(rows,op):
        gain=np.zeros(len(rows));known=np.ones(len(rows),dtype=bool)
        rc,ac=rows[:,:3]//scale,rows[:,3:]//scale
        different=(rc!=ac).any(1) if op=='Adjust' else np.ones(len(rows),dtype=bool)
        if op!='Add':
            cell=source_index.find(rc)
            changed=different&(cell>=0)&(counts[np.maximum(cell,0)]==1)
            indices=np.flatnonzero(changed);prob=native_index.find(rc[changed]);valid=prob>=0
            known[indices]&=valid;gain[indices[valid]]-=logit[prob[valid]]
        if op!='Prune':
            changed=different&(source_index.find(ac)<0)
            indices=np.flatnonzero(changed);prob=native_index.find(ac[changed]);valid=prob>=0
            known[indices]&=valid;gain[indices[valid]]+=logit[prob[valid]]
        return gain,known

    native={}
    if cached is None:
        lower,upper=source.min(0),source.max(0)
        interior=((source>lower)&(source<upper)).all(1)
        prune=source[(counts[inverse]==1)&interior]
        empty=cells[~occupied];lo=empty*scale;hi=lo+scale-1
        _,near=cKDTree(source).query((lo+hi)/2.,k=min(8,len(source)),workers=2)
        if near.ndim==1:near=near[:,None]
        raw=source[near];target=np.maximum(np.minimum(raw,hi[:,None,:]),lo[:,None,:])
        delta=np.max(np.abs(target-raw),axis=2)
        valid=(delta>0)&(delta<=1)&((target>=lower)&(target<=upper)).all(2)
        src,dst,nearby=raw[valid],target[valid],near[valid]
        native=dict(Add=np.concatenate((np.zeros_like(dst),dst),axis=1),
            Prune=np.concatenate((prune,np.zeros_like(prune)),axis=1),
            Adjust=np.concatenate((src[interior[nearby]],dst[interior[nearby]]),axis=1))
    results,diagnostics,core_arrays={},{},{}
    for op_index,op in enumerate(operations):
        unique_native=None
        if cached is not None:
            core=cached[op]
        else:
            unique_native=native[op][first_rows(native[op])]
            bands=np.digitize(observe(unique_native,op)[0],np.array([-8.,-4.,-1.,0.,1.,4.,8.]))
            rng=np.random.RandomState((int(seed)+op_index*7919)&0x7fffffff)
            groups=[rng.permutation(np.flatnonzero(bands==b)).tolist() for b in range(8)]
            cap=min(len(unique_native),max(int(budget)//2,1));ids=[]
            while len(ids)<cap and any(groups):
                for group in groups:
                    if group and len(ids)<cap:ids.append(group.pop())
            core=unique_native[np.asarray(ids,dtype=np.int64)]
        core_arrays[op]=core
        existing=(random_pools or {}).get(op,[])
        rng=np.random.RandomState((int(seed)+op_index*7919+104729)&0x7fffffff)
        order=rng.permutation(len(existing))
        typed=isinstance(existing,SpatialCandidates)
        spatial=(existing.rows if typed else np.asarray([(c.get('remove_coords') or [[0,0,0]])[0]
                            +(c.get('add_coords') or [[0,0,0]])[0] for c in existing],dtype=np.int64).reshape(-1,6))
        combined=np.concatenate((core,spatial[order]),axis=0)
        origins=np.concatenate((np.full(len(core),-1,dtype=np.int64),order))
        selected=first_rows(combined)[:int(budget)]
        rows,origins=combined[selected],origins[selected]
        gains,known=observe(rows,op)
        built=[]
        for index,(row,origin,gain,observed) in enumerate(zip(rows,origins,gains,known)):
            c=(dict(existing[int(origin)]) if origin>=0 and not typed else
               dict(remove_coords=[row[:3].tolist()] if op!='Add' else [],
                    add_coords=[row[3:].tolist()] if op!='Prune' else []))
            if typed and origin>=0:c['geometry_cost']=float(existing.geometry[int(origin)])
            remove,add=c.get('remove_coords',[]),c.get('add_coords',[])
            payload=json.dumps([op,remove,add],separators=(',',':'))
            built.append(dict(c,candidate_id='ctx:'+hashlib.sha256(payload.encode()).hexdigest()[:24],
                operation=op,pool_rank=index,rank_score=0.,heuristic_score=0.,
                optimistic_gain_bits=float(gain),neighbor_bit_risk=0.,
                geometry_cost=float(c.get('geometry_cost',1.)),operation_count=1,affected_voxel_cells=1,
                native_fixed_context_observed=bool(observed)))
        results[op]=built
        diagnostics[op]=dict(codec_candidates=len(unique_native) if cached is None else None,
            retained_codec_candidates=len(core),core_cache_hit=cached is not None,total=len(built),
            positive_fixed_gain=int((gains>0).sum()),negative_fixed_gain=int((gains<0).sum()))
    if cached is None:
        with tempfile.NamedTemporaryFile(dir=str(cache_path.parent),delete=False) as f:
            temp=Path(f.name);np.savez_compressed(f,**core_arrays)
        try:os.link(str(temp),str(cache_path))
        except FileExistsError:pass
        finally:temp.unlink()
    return results,diagnostics
