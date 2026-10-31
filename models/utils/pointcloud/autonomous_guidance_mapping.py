"""Exact batched mapping for the autonomous single-voxel action vocabulary.

This only replaces Python coordinate sets/loops. Candidate order, all Add
neighbours, direction order and static legality are the legacy mapper's.
Other vocabularies (including multi-voxel bundles) use the original mapper.
"""
from collections.abc import Mapping

import numpy as np
import torch


def map_single_voxel_candidates(pools, coords, like, offsets, coordinate_indices):
    if coords.shape[0] != 1:
        return None
    arrays = {}
    for op in ('Add', 'Prune', 'Adjust'):
        pool = pools.get(op)
        if not isinstance(pool, list):
            return None
        removes, adds = [], []
        for c in pool:
            if not isinstance(c, Mapping):
                return None
            try:
                if float(c.get('rank_score',float('nan'))) != 0.:return None
            except (TypeError,ValueError,OverflowError):
                return None
            r,a=c.get('remove_coords',()),c.get('add_coords',())
            if len(r) != int(op != 'Add') or len(a) != int(op != 'Prune'):
                return None
            rows=list(r)+list(a)
            if any(len(v)!=3 or any(int(x)!=x for x in v) for v in rows):
                return None
            removes.append(r[0] if r else (0,0,0))
            adds.append(a[0] if a else (0,0,0))
        arrays[op]=(torch.as_tensor(np.asarray(removes,dtype=np.int64).reshape(-1,3),device=coords.device),
                    torch.as_tensor(np.asarray(adds,dtype=np.int64).reshape(-1,3),device=coords.device))
    reference=coords[0].T.contiguous()
    offset_tensor=torch.tensor(offsets,device=coords.device,dtype=torch.long)
    # One packed-key join for all queried rows. Repeated rows are intentional:
    # reshaping retains the exact candidate-major, direction-minor order.
    add_targets=arrays['Add'][1]
    neighbour_rows=(add_targets[:,None,:]-offset_tensor[None,:,:]).reshape(-1,3)
    queries=(neighbour_rows,add_targets,arrays['Prune'][0],arrays['Adjust'][0],arrays['Adjust'][1])
    sizes=[len(q) for q in queries]
    indices=coordinate_indices(torch.cat(queries),reference).split(sizes)
    neighbour_index=indices[0].reshape(-1,len(offsets))
    add_target_index,prune_source,adjust_source,adjust_target=indices[1:]
    delta=arrays['Adjust'][1]-arrays['Adjust'][0]
    valid_direction=(delta.abs()<=1).all(1)&(delta!=0).any(1)
    code=(delta[:,0]+1)*9+(delta[:,1]+1)*3+delta[:,2]+1
    adjust_direction=code-(code>13).to(torch.long)
    adjust_valid=valid_direction&(adjust_source>=0)
    pair_valid=neighbour_index>=0
    add_candidate,add_direction=torch.nonzero(pair_valid,as_tuple=True)
    add_source=neighbour_index[pair_valid]
    empty=coords.new_empty(0)
    def pairs(source,direction):
        return dict(batch_index=torch.zeros_like(source),source_index=source,
                    direction_index=direction,rank_score=like.new_zeros(source.shape))
    where_prior={op:like.new_zeros((1,1,coords.shape[-1])) for op in arrays}
    masks={op:torch.zeros_like(value,dtype=torch.bool) for op,value in where_prior.items()}
    sources={'Add':add_source,'Prune':prune_source[prune_source>=0],
             'Adjust':adjust_source[adjust_valid]}
    for op,source in sources.items():
        masks[op].view(-1).index_fill_(0,source,True)
    maps={}
    for op in arrays:
        count=len(pools[op])
        maps[op]=dict(pool_rank=torch.tensor([int(c.get('pool_rank',i)) for i,c in enumerate(pools[op])],
                        device=coords.device,dtype=torch.long),rank_score=like.new_zeros(count),
            source_index=torch.full((count,),-1,device=coords.device,dtype=torch.long),
            direction_index=torch.full((count,),-1,device=coords.device,dtype=torch.long),
            pair_candidate_index=empty,pair_source_index=empty,pair_direction_index=empty)
    maps['Add'].update(pair_candidate_index=add_candidate,pair_source_index=add_source,
        pair_direction_index=add_direction,static_compatible=add_target_index<0)
    maps['Prune'].update(source_index=prune_source,static_compatible=prune_source>=0)
    maps['Adjust'].update(source_index=torch.where(adjust_valid,adjust_source,-1),
        direction_index=torch.where(adjust_valid,adjust_direction,-1),
        static_compatible=(adjust_source>=0)&(adjust_target<0))
    mapped=dict(Add=int(pair_valid.any(1).sum()),Prune=int((prune_source>=0).sum()),
                Adjust=int(adjust_valid.sum()))
    return dict(where_prior=where_prior,candidate_mask=masks,
        target_direction_sparse=dict(Add=pairs(add_source,add_direction),
            Adjust=pairs(adjust_source[adjust_valid],adjust_direction[adjust_valid])),
        candidate_tensor_map=maps,exact_candidate_mapped_count=mapped,
        exact_candidate_unmapped_count={op:len(pools[op])-n for op,n in mapped.items()})
