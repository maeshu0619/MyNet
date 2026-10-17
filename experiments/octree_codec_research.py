#!/usr/bin/env python
"""Isolated G-PCC/OctAttention research, never imports a training entry point.

OctAttention rates below are actual arithmetic-stream lengths, NOT CE estimates.
Its sequence and model are the same as the existing teacher. An arithmetic
roundtrip uses encoder probabilities; this is not a full autonomous OA decoder.
"""
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
import torch
from geometry_micro_edit_probe import ROOT, ActualCodec, unique, difference, load_ply


class OAActual:
    def __init__(self, device='cpu'):
        from models.utils.loss.actual_encoder import _OctAttentionActualEncoder
        self.teacher = _OctAttentionActualEncoder(SimpleNamespace(qs=1.,
            octattention_actualcode=True, octattention_teacher_device=device, bptt=1024))
        self.teacher._lazy_init()
        self.calls = 0

    def sequence(self, rows):
        from Octree import GenOctree, GenKparentSeq
        rows = unique(rows-rows.min(0))
        _, tree, _ = GenOctree(rows)
        data = GenKparentSeq(tree,4)
        return np.concatenate([data['Seq'][:,:,None],data['Level'],data['Pos']],axis=2).astype(np.int32)

    def probabilities(self, seq):
        teacher = self.teacher
        n = len(seq)
        bptt = min(teacher._oa_bptt,n-1)
        if bptt < 32:
            raise ValueError('OA sequence too short')
        ids, padded = teacher._batchify(seq,bptt,n)
        mask = teacher._generate_square_subsequent_mask(bptt)
        chunks = []
        with torch.inference_mode():
            for start in range(0,len(padded)-bptt,bptt):
                valid = ids[start+1:start+bptt+1].reshape(-1)>=0
                if not valid.any():
                    continue
                output = teacher._model(padded[start:start+bptt].to(teacher._oa_device),mask,[])
                prob = torch.softmax(output.reshape(-1,255),dim=1)
                chunks.append(prob[valid.to(prob.device)].cpu().numpy())
        probs = np.vstack(chunks)[:n]
        if len(probs)!=n:
            raise AssertionError('Incomplete OA probability rows')
        return probs

    def encode(self,rows,detail=False,roundtrip=False):
        import numpyAc
        started = time.perf_counter()
        seq = self.sequence(rows)
        probs = self.probabilities(seq)
        symbols = (seq[:,-1,0]-1).astype(np.int16)
        payload,bits = numpyAc.arithmeticCoding().encode(probs,symbols)
        nll = -np.log2(np.maximum(probs[np.arange(len(seq)),symbols],1e-30))
        if roundtrip:
            decoder = numpyAc.arithmeticDeCoding(payload,len(seq),255)
            decoded = np.array([decoder.decode(probs[i:i+1]) for i in range(len(seq))]).reshape(-1)
            if not np.array_equal(decoded,symbols):
                raise AssertionError('OA arithmetic roundtrip failed')
            from Octree import DeOctree
            reconstructed = unique(DeOctree(decoded+1))
            if not np.array_equal(reconstructed,unique(rows-rows.min(0))):
                raise AssertionError('OA occupancy-to-geometry roundtrip failed')
        self.calls += 1
        levels = seq[:,-1,1]
        stats = dict(bits=int(bits), codec_sec=time.perf_counter()-started,
            ce_bits=float(nll.sum()), nodes=len(seq), arithmetic_roundtrip=roundtrip,
            stream_sha256=hashlib.sha256(payload).hexdigest(),
            levels={str(int(level)):dict(nodes=int((levels==level).sum()),
                       ce_bits=float(nll[levels==level].sum())) for level in np.unique(levels)})
        if detail:
            return stats,seq,probs
        return stats


def quality(raw,q,origin,qs):
    rec = (q+origin)*qs
    forward = cKDTree(rec).query(raw,workers=2)[0]
    backward = cKDTree(raw).query(rec,workers=2)[0]
    return dict(chamfer_sq=float(np.mean(forward**2)+np.mean(backward**2)),
                d1_mse=float(max(np.mean(forward**2),np.mean(backward**2))),
                max_gt_to_rec=float(forward.max()),max_rec_to_gt=float(backward.max()))


def safe_prune_order(q,raw,origin,qs,strict=True,score=None):
    """Reserve replacement voxels: globally joint-safe, not isolated deletions."""
    distances, indices = cKDTree((q+origin)*qs).query(raw,k=2,workers=2)
    per_voxel = np.zeros(len(q))
    slack = distances[:,1]-distances[:,0] if strict else distances[:,1]-distances[:,0].max()
    np.maximum.at(per_voxel,indices[:,0],slack)
    valid = per_voxel <= 1e-9
    assignment_order = np.argsort(indices[:,0],kind='stable')
    starts = np.r_[0,np.bincount(indices[:,0],minlength=len(q)).cumsum()]
    if score is None:
        score = -(cKDTree(q).query_ball_point(q,np.sqrt(3)+1e-6,return_length=True,workers=2)-1)
    selected, protected = [], set(q.argmin(0).tolist()+q.argmax(0).tolist())
    selected_set = set()
    for index in np.argsort(-score,kind='stable'):
        i = int(index)
        if not valid[i] or i in protected:
            continue
        replacements = set(indices[assignment_order[starts[i]:starts[i+1]],1].tolist())
        if replacements & selected_set:
            continue
        selected.append(i)
        selected_set.add(i)
        protected.update(replacements)
    return np.array(selected,dtype=np.int64),dict(eligible=int(valid.sum()),
            safe_deletion_capacity=len(selected),strict_nearest_distance=strict)


def leaf_plan(q,raw,origin,qs,seq,probs,budget,return_candidates=False):
    """Actual OA conditional-probability leaf masks; ancestors stay identical.

    Contexts are frozen for proposal only. Edited cloud is fully re-encoded.
    Bbox preserved; deletion replacements reserved; Add within original max NN.
    """
    max_level = seq[:,-1,1].max()
    bbox_min,bbox_max = q.min(0),q.max(0)
    leaves = np.flatnonzero(seq[:,-1,1]==max_level)
    q_index = {tuple(p):i for i,p in enumerate(q.tolist())}
    raw_tree = cKDTree(raw)
    safe_ids,_ = safe_prune_order(q,raw,origin,qs)
    # Any subset of this reserved set remains quality-safe for Prune.
    safe = {tuple(p) for p in q[safe_ids].tolist()}
    extrema = set(map(tuple,q[np.unique(np.r_[q.argmin(0),q.argmax(0)])].tolist()))
    leaf_origins = seq[leaves,-1,3:6].astype(np.int64)
    slots = np.array(list(itertools.product((0,1),repeat=3)))
    holes = set(map(tuple,(leaf_origins[:,None,:]+slots[None,:,:]).reshape(-1,3).tolist()))-set(q_index)
    holes = np.array(sorted(holes),dtype=np.int64).reshape(-1,3)
    hole_distances = raw_tree.query((holes+origin)*qs,workers=2)[0]
    safe_add = {tuple(point) for point,distance in zip(holes.tolist(),hole_distances)
                if distance<=np.sqrt(3)+1e-9}
    candidates = []
    for row in leaves:
        parent_min = seq[row,-1,3:6].astype(np.int64)
        mask = int(seq[row,-1,0])
        coords = [tuple(parent_min+[(slot>>2)&1,(slot>>1)&1,slot&1]) for slot in range(8)]
        # Validate coordinate/slot conventions against all leaf masks.
        found = sum(1<<slot for slot,p in enumerate(coords) if p in q_index)
        if found != mask:
            raise AssertionError('Leaf coordinate convention mismatch')
        for op in ('Prune','Add','Adjust'):
            pairs = ([(s,None) for s in range(8) if mask&(1<<s)] if op=='Prune' else
                     [(None,t) for t in range(8) if not mask&(1<<t)] if op=='Add' else
                     [(s,t) for s in range(8) if mask&(1<<s) for t in range(8) if not mask&(1<<t)])
            for source,target in pairs:
                if source is not None and (coords[source] not in safe or coords[source] in extrema):
                    continue
                after = mask if source is None else mask & ~(1<<source)
                if target is not None:
                    point = np.asarray(coords[target])
                    if (point<bbox_min).any() or (point>bbox_max).any():
                        continue
                    if tuple(point) not in safe_add:
                        continue
                    after |= 1<<target
                if after==0:
                    continue
                gain = float(np.log2(max(float(probs[row,after-1]),1e-30)/max(float(probs[row,mask-1]),1e-30)))
                if gain>0:
                    candidates.append((gain,int(row),op,coords[source] if source is not None else None,
                                       coords[target] if target is not None else None))
    candidates.sort(reverse=True)
    if return_candidates:
        return candidates
    occupied = set(q_index)
    used = set()
    counts = dict(Add=0,Prune=0,Adjust=0)
    predicted = 0.
    k = round(len(q)*budget)
    for gain,row,op,remove,add in candidates:
        if sum(counts.values())>=k:
            break
        if row in used:
            continue
        used.add(row)
        if remove is not None:
            occupied.remove(remove)
        if add is not None:
            occupied.add(add)
        counts[op] += 1
        predicted += gain
    return unique(list(occupied)),dict(counts=counts,predicted_local_gain=predicted,
            candidate_count=len(candidates),edited_leaf_nodes=len(used))


def pattern_leaf_plan(q,raw,origin,qs,seq,probs,budget):
    """Joint eight-child mask rewrites with actual voxel-operation accounting."""
    leaves=np.flatnonzero(seq[:,-1,1]==seq[:,-1,1].max())
    slots=np.array(list(itertools.product((0,1),repeat=3)))
    occupied=set(map(tuple,q.tolist()))
    safe_ids,_=safe_prune_order(q,raw,origin,qs)
    safe_remove=set(map(tuple,q[safe_ids].tolist()))
    origins=seq[leaves,-1,3:6].astype(np.int64)
    holes=set(map(tuple,(origins[:,None]+slots[None]).reshape(-1,3).tolist()))-occupied
    holes=np.array(sorted(holes),dtype=np.int64).reshape(-1,3)
    distances=cKDTree(raw).query((holes+origin)*qs,workers=2)[0]
    bbox_min,bbox_max=q.min(0),q.max(0)
    safe_add={tuple(p) for p,d in zip(holes.tolist(),distances)
              if d<=np.sqrt(3)+1e-9 and (p>=bbox_min).all() and (p<=bbox_max).all()}
    all_masks=np.arange(1,256)
    pops=np.array([bin(i).count('1') for i in range(256)])
    candidates=[]
    for index,row in enumerate(leaves):
        mask=int(seq[row,-1,0])
        points=[tuple(p) for p in origins[index]+slots]
        removable=sum(1<<s for s,p in enumerate(points) if p in safe_remove)
        addable=sum(1<<s for s,p in enumerate(points) if p in safe_add)
        rem=mask & ~all_masks
        add=all_masks & ~mask
        cost=np.maximum(pops[rem],pops[add])
        valid=((rem & ~removable)==0)&((add & ~addable)==0)&(cost>0)
        gain=np.log2(np.maximum(probs[row],1e-30)/max(float(probs[row,mask-1]),1e-30))
        score=np.where(valid&(gain>0),gain/np.maximum(cost,1),-np.inf)
        choice=int(score.argmax())
        if not np.isfinite(score[choice]):
            continue
        target=choice+1
        removes=[p for s,p in enumerate(points) if mask&(1<<s) and not target&(1<<s)]
        adds=[p for s,p in enumerate(points) if target&(1<<s) and not mask&(1<<s)]
        candidates.append((float(score[choice]),int(cost[choice]),float(gain[choice]),removes,adds))
    candidates.sort(key=lambda x:x[0],reverse=True)
    counts=dict(Add=0,Prune=0,Adjust=0)
    predicted,selected=0.,0
    k=round(len(q)*budget)
    for _,cost,gain,rem,add in candidates:
        if cost>k-sum(counts.values()):
            continue
        occupied.difference_update(rem)
        occupied.update(add)
        moves=min(len(rem),len(add))
        counts['Adjust']+=moves
        counts['Add']+=len(add)-moves
        counts['Prune']+=len(rem)-moves
        predicted+=gain
        selected+=1
    return unique(list(occupied)),dict(counts=counts,predicted_local_gain=predicted,
            candidate_count=len(candidates),edited_leaf_nodes=selected,
            coherent_eight_child_masks=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--codecs',nargs='+',choices=['gpcc','oa'],default=['gpcc'])
    p.add_argument('--modes',nargs='+',choices=['orientation','prune','leaf','pattern','quant'],required=True)
    p.add_argument('--budgets',nargs='+',type=float,default=[.0006,.0015,.003,.01,.03,.1,.3])
    p.add_argument('--oa-device',default='cpu')
    p.add_argument('--qs',type=float,default=2.)
    p.add_argument('--reflections',action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    folder = Path(a.out)
    folder.mkdir(parents=True,exist_ok=False)
    backends = {name:(ActualCodec(a.qs,2) if name=='gpcc' else OAActual(a.oa_device)) for name in a.codecs}
    (folder/'metadata.json').write_text(json.dumps(dict(arguments=vars(a),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        coordinate_protocol='Common G-PCC qs grid, OA canonical integer input with qs=1',
        existing_training_modified=False),indent=2))
    with (folder/'trials.jsonl').open('w',buffering=1) as log:
        for frame in a.frames:
            raw = load_ply(frame,return_color=False).astype(np.float64)
            absolute = unique(np.rint(raw/a.qs))
            origin = absolute.min(0)
            q = absolute-origin
            baseline_quality = quality(raw,q,origin,a.qs)
            baselines = {}
            seq,probs = None,None

            def record(name,out,details,quality_stats=None):
                removed,added = len(difference(q,out)),len(difference(out,q))
                geom = quality(raw,out,origin,a.qs) if quality_stats is None else quality_stats
                for codec,backend in backends.items():
                    if codec=='oa' and name=='baseline':
                        nonlocal seq,probs
                        stats,seq,probs = backend.encode(out,detail=True,roundtrip=True)
                    else:
                        stats = backend.encode(out,verify=name=='baseline') if codec=='gpcc' else backend.encode(out)
                    if name=='baseline':
                        baselines[codec]=stats['bits']
                    physical_removed = 0 if details.get('physical_voxel_edits')==0 else removed
                    physical_added = 0 if details.get('physical_voxel_edits')==0 else added
                    side_bits = int(details.get('side_information_bits',0))
                    operation_count = (sum(details['counts'].values()) if 'counts' in details
                                       else max(physical_removed,physical_added))
                    row = dict(frame=frame,codec=codec,trial=name,**stats,**geom,details=details,
                        baseline_bits=baselines[codec],compression_percent=100.*(stats['bits']/baselines[codec]-1.),
                        net_bits=stats['bits']+side_bits, net_compression_percent=100.*((stats['bits']+side_bits)/baselines[codec]-1.),
                        baseline_voxels=len(q),edited_voxels=len(out),removed=physical_removed,added=physical_added,
                        code_space_removed=removed,code_space_added=added,
                        operation_percent=100.*operation_count/len(q),
                        min_operation_percent=100.*max(physical_removed,physical_added)/len(q))
                    log.write(json.dumps(row)+'\n')
                    print(json.dumps({k:row[k] for k in ['codec','trial','bits','compression_percent','chamfer_sq']}),flush=True)
            record('baseline',q,{},baseline_quality)
            if 'orientation' in a.modes:
                signs = list(itertools.product((1,-1),repeat=3)) if a.reflections else [(1,1,1)]
                for perm in itertools.permutations(range(3)):
                    for sign in signs:
                        if perm==(0,1,2) and sign==(1,1,1):
                            continue
                        transformed = q[:,perm]*np.array(sign)
                        # Same physical point cloud after exact inverse; this
                        # is an invertible coding transform, not voxel editing.
                        record('orientation_%s_%s'%(''.join(map(str,perm)),''.join('p' if s==1 else 'n' for s in sign)),
                            unique(transformed),dict(physical_voxel_edits=0,requires_inverse_transform=True,
                            side_information_bits=8,not_Add_Prune_Adjust=True),baseline_quality)
            if 'prune' in a.modes:
                order,audit = safe_prune_order(q,raw,origin,a.qs)
                for budget in a.budgets:
                    mask = np.ones(len(q),dtype=bool)
                    mask[order[:round(len(q)*budget)]]=False
                    record('strict_prune_%.5f'%budget,q[mask],dict(**audit,requested_ratio=budget))
            if 'leaf' in a.modes:
                if probs is None:
                    raise ValueError('leaf mode requires OA backend')
                for budget in a.budgets:
                    out,audit = leaf_plan(q,raw,origin,a.qs,seq,probs,budget)
                    record('oa_conditional_leaf_%.5f'%budget,out,audit)
            if 'pattern' in a.modes:
                if probs is None:
                    raise ValueError('pattern mode requires OA backend')
                for budget in a.budgets:
                    out,audit=pattern_leaf_plan(q,raw,origin,a.qs,seq,probs,budget)
                    record('oa_pattern_leaf_%.5f'%budget,out,audit)
            if 'quant' in a.modes:
                for method in ['halfup','floor','ceil']:
                    values = raw/a.qs
                    absolute_new = unique(np.floor(values+.5) if method=='halfup' else np.floor(values) if method=='floor' else np.ceil(values))
                    record('quant_'+method,absolute_new-origin,dict(not_micro=True))
    print('RESULTS '+str(folder),flush=True)


if __name__=='__main__':
    main()
