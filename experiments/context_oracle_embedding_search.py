#!/usr/bin/env python
"""Frozen OA input-embedding gradient proposals, accepted by integer CDF only.

No optimizer, no weight update, no training Network load. The CE derivative is
only a proposal heuristic for collateral context changes, never Actual bits.
Every accepted mask set must improve the re-evaluated full-window CDF cost.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from context_oracle_core import (OAActual,ActualCodec,WindowCodec,costs,load_ply,
    unique,apply_plan,quality,quality_ok)
from context_oracle_study import pick,edit_list


def digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def gradient(w,group,edits):
    start=(group+1)*w.bptt
    inp=w.padded[start:start+w.bptt].clone()
    ids=w.ids[start+1:start+w.bptt+1].reshape(-1).numpy()
    valid=ids>=0;ids=ids[valid];target=w.symbols[ids].copy()
    for row,after in edits:
        inp[row%w.bptt,0,-1,0]=after-1
        target[row-(group*w.bptt+1)]=after-1
    captured=[]
    def hook(module,inputs,output):
        replacement=output.detach().clone().requires_grad_(True)
        captured.append(replacement)
        return replacement
    model=w.oa.teacher._model
    handle=model.encoder.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits=model(inp.to(w.oa.teacher._oa_device),w.mask,[]).reshape(-1,255)
            good=torch.from_numpy(valid).to(logits.device)
            labels=torch.from_numpy(target.astype(np.int64)).to(logits.device)
            loss=torch.nn.functional.cross_entropy(logits[good],labels,reduction='sum')/math.log(2.)
            g=torch.autograd.grad(loss,captured[0])[0].detach().cpu().numpy()[:,0,-1]
            prob=torch.softmax(logits[good].detach(),1).cpu().numpy()
    finally:
        handle.remove()
    return ids,prob,g


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',required=True);p.add_argument('--wide',required=True)
    p.add_argument('--seed-plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--passes',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(2)
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    (folder/'arguments.json').write_text(json.dumps(vars(a),indent=2))
    raw=load_ply(a.frame,return_color=False).astype(np.float64)
    absolute=unique(np.rint(raw/2.));origin=absolute.min(0);q=absolute-origin
    oa=OAActual('cpu');w=WindowCodec(oa,q);model=oa.teacher._model
    before_hash=digest(model)
    for param in model.parameters():
        param.requires_grad_(False)
    features=np.load(str(Path(a.wide)/Path(a.frame).stem/'candidate_features.npz'))
    c=features['candidates'];seed=np.load(a.seed_plan)
    state=[tuple(map(int,x)) for x in seed['edits'].reshape(-1,2)]
    groups={}
    for row,after in state:
        groups.setdefault(w.group(row),[]).append((row,after))
    lookup={(int(x['row']),int(x['mask'])):i for i,x in enumerate(c)}
    weight=model.encoder.weight.detach().cpu().numpy()
    initial=w.evaluate(state,actual=True)
    accepted=0
    with (folder/'results.jsonl').open('w',buffering=1) as log:
        def emit(x):
            log.write(json.dumps(x)+'\n');print(json.dumps(x),flush=True)
        emit(dict(trial='initial',frame=a.frame,seed=a.seed_plan,**initial))
        for iteration in range(a.passes):
            for count,(group,items) in enumerate(sorted(groups.items())):
                indices=np.flatnonzero(c['row']//w.bptt==group)
                old_indices=[lookup[x] for x in items]
                cap=int(c['cost'][old_indices].sum())
                current=w.evaluate(items)
                ids,prob,g=gradient(w,group,items)
                all_cost=costs(prob)
                pos=c['row'][indices]-(group*w.bptt+1)
                old=w.symbols[c['row'][indices]];after=c['mask'][indices]-1
                own=all_cost[pos,old]-all_cost[pos,after]
                context=-(g[c['row'][indices]%w.bptt]*(weight[after]-weight[old])).sum(1)
                best=items;best_gain=current['total_cdf_gain'];alpha_best=None
                for alpha in (.1,.25,.5,1.):
                    score=np.zeros(len(c));score[indices]=own+alpha*context-1.*np.maximum(features['slack'][indices],0.)
                    selected=pick(indices,c,score,cap)
                    if features['slack'][selected].sum()>1e-8:
                        continue
                    edits=edit_list(c,selected)
                    measured=w.evaluate(edits)
                    if measured['total_cdf_gain']>best_gain+1e-7:
                        best=edits;best_gain=measured['total_cdf_gain'];alpha_best=alpha
                if alpha_best is not None:
                    groups[group]=best;accepted+=1
                    emit(dict(trial='accepted_window',iteration=iteration,group=group,
                        before_gain=current['total_cdf_gain'],after_gain=best_gain,alpha=alpha_best))
                if (count+1)%10==0:
                    emit(dict(trial='progress',iteration=iteration,windows=count+1,accepted=accepted))
        state=[x for items in groups.values() for x in items]
        out,counts=apply_plan(q,w.seq,state);geom=quality(raw,out,origin,2.)
        base_geom=quality(raw,q,origin,2.)
        if not quality_ok(base_geom,geom):
            raise AssertionError('Gradient proposal violated final geometry constraints')
        evaluated=w.evaluate(state,actual=True)
        fresh=oa.encode(out,roundtrip=True)
        if fresh['stream_sha256']!=evaluated['actual_oa']['stream_sha256']:
            raise AssertionError('Final gradient proposal not full-model reproducible')
        gpcc=ActualCodec(2.,2);base_gpcc=gpcc.encode(q,verify=True);encoded=gpcc.encode(out,verify=True)
        after_hash=digest(model)
        if before_hash!=after_hash:
            raise AssertionError('Frozen codec model weights changed')
        selected=[lookup[x] for x in state]
        emit(dict(trial='final',frame=a.frame,counts=counts,operation_count=sum(counts.values()),
            operation_percent=100*sum(counts.values())/len(q),quality=geom,
            baseline_quality=base_geom,oa=fresh,gpcc=encoded,
            oa_C=100*(fresh['bits']/w.baseline['bits']-1.),gpcc_C=100*(encoded['bits']/base_gpcc['bits']-1.),
            current_prior_score=float(features['prior'][selected].sum()),own_score=float(features['own'][selected].sum()),
            accepted_windows=accepted,model_weights_sha256_before=before_hash,
            model_weights_sha256_after=after_hash,**evaluated))
        np.savez_compressed(folder/'best.npz',canonical_voxels=out,origin=origin,qs=2.,edits=np.array(state,dtype=np.int64).reshape(-1,2))


if __name__=='__main__':
    main()
