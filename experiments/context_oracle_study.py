#!/usr/bin/env python
"""Same-pool OA context / Actual oracle comparison, CPU-only, new artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from context_oracle_core import (OAActual,ActualCodec,WindowCodec,costs,stream,
    unique,quality,load_ply,apply_plan,quality_ok,POPS,protect_snapshot)
from octree_joint_mask_research import SLOTS
from models.utils.pointcloud.geometry_octree_rate import GeometryOctreeRate


def features(c,seq,q,raw,origin,w,model):
    before=seq[c['row'],-1,0]
    parent=seq[c['row'],-1,3:6]//2
    orientation=(parent[:,0]&1)*4+(parent[:,1]&1)*2+(parent[:,2]&1)
    table=np.stack([model.costs[(1,o)] for o in range(8)])
    empirical=table[orientation,before]-table[orientation,c['mask']]
    geom=np.zeros((256,256),dtype=np.float64)
    distance=np.sum((SLOTS[:,None]-SLOTS[None])**2,axis=2)
    for old in range(1,256):
        occupied=np.flatnonzero(old & (1<<np.arange(8)))
        rem_cost=np.zeros(8)
        for s in occupied:
            other=occupied[occupied!=s]
            rem_cost[s]=distance[s,other].min() if len(other) else 4.
        add_cost=distance[:,occupied].min(1)
        for after in range(1,256):
            rem=np.array([bool(old&(1<<s) and not after&(1<<s)) for s in range(8)])
            add=np.array([bool(after&(1<<s) and not old&(1<<s)) for s in range(8)])
            geom[old,after]=4.*(rem_cost[rem].sum()+add_cost[add].sum())
    prior=np.tanh((empirical-.05*geom[before,c['mask']])/(4.*c['cost']))
    own=w.all_cost[c['row'],before-1]-w.all_cost[c['row'],c['mask']-1]
    leaf=np.flatnonzero(seq[:,-1,1]==seq[:,-1,1].max())
    leaf_index=np.full(len(seq),-1,dtype=np.int64)
    leaf_index[leaf]=np.arange(len(leaf))
    pos=seq[leaf,-1,3:6]
    d=cKDTree(raw).query(((pos[:,None]+SLOTS[None]+origin)*2.).reshape(-1,3),workers=2)[0].reshape(-1,8)**2
    incidence=((np.arange(256)[:,None] & (1<<np.arange(8))[None])!=0).astype(float)
    slot_sum=d.dot(incidence.T)
    backward=cKDTree(raw).query((q+origin)*2.,workers=2)[0]**2
    mean=backward.mean()
    idx=leaf_index[c['row']]
    delta=slot_sum[idx,c['mask']]-slot_sum[idx,before]
    dn=POPS[c['mask']]-POPS[before]
    slack=delta-mean*dn
    return dict(prior=prior,own=own,slack=slack,empirical=empirical),dict(backward_mse=float(mean))


def pick(indices,c,score,cap):
    order=indices[np.argsort(-score[indices]/c['cost'][indices],kind='stable')]
    chosen,seen,used=[],set(),0
    for i in order:
        if score[i]<=0:
            break
        row=int(c[i]['row'])
        cost=int(c[i]['cost'])
        if row in seen or used+cost>cap:
            continue
        seen.add(row);chosen.append(int(i));used+=cost
        if used==cap:
            break
    return chosen


def edit_list(c,selected):
    return [(int(c[i]['row']),int(c[i]['mask'])) for i in selected]


def knapsack(menus,cap,objective):
    dp=np.full(cap+1,-np.inf);dp[0]=0.
    histories=[]
    for options in menus:
        nxt=np.full_like(dp,-np.inf)
        history=np.full(cap+1,-1,dtype=np.int32)
        for index,o in enumerate(options):
            n=o['cost'];value=o[objective]
            if n>cap:
                continue
            v=dp[:cap+1-n]+value
            target=nxt[n:];better=v>target
            target[better]=v[better];history[n:][better]=index
        dp=nxt;histories.append(history)
    amount=int(np.argmax(dp));value=float(dp[amount]);used=amount;selected=[]
    for options,history in zip(menus[::-1],histories[::-1]):
        index=int(history[amount])
        if index<0:
            raise AssertionError('DP backtrack failed')
        selected.extend(options[index]['selected']);amount-=options[index]['cost']
    if amount:
        raise AssertionError('DP budget mismatch')
    return selected,dict(dp_objective=objective,dp_value=value,used_cost=used)


def build_menus(c,f,w,folder,report):
    groups={}
    for group in np.unique(c['row']//w.bptt):
        groups[int(group)]=np.flatnonzero(c['row']//w.bptt==group)
    caps=(1,4,16,64,192,384)
    menus=[]
    diagnostics=[]
    started=time.perf_counter()
    for group,indices in sorted(groups.items()):
        options=[dict(cost=0,prior=0.,own=0.,total=0.,selected=[],slack=0.)]
        fingerprints={tuple()}
        refinements=[]
        def add(selected,kind):
            key=tuple(sorted(selected))
            if key in fingerprints:
                return None
            fingerprints.add(key)
            if float(f['slack'][selected].sum())>1e-8:
                return None
            evaluation=w.evaluate(edit_list(c,selected),return_probs=True)
            option=dict(cost=int(c['cost'][selected].sum()),
                prior=float(f['prior'][selected].sum()),own=float(f['own'][selected].sum()),
                total=evaluation['total_cdf_gain'],selected=selected,
                slack=float(f['slack'][selected].sum()),kind=kind)
            options.append(option)
            return evaluation
        for cap in caps:
            for name,score in [('prior',f['prior']*c['cost']),('own',f['own']),
                               ('own_geometry',f['own']-2.*np.maximum(f['slack'],0.))]:
                selected=pick(indices,c,score,cap)
                result=add(selected,name)
                if result is not None and name in ('own','own_geometry'):
                    refinements.append((cap,selected,result))
        # Re-propose masks in edited contexts, including originally nonpositive
        # candidates, and accept only full-window integer-CDF improvements.
        for cap,selected,result in refinements:
            for iteration in range(2):
                fresh=f['own'].copy()
                for target_ids,prob,targets in result['_chunks']:
                    all_cost=costs(prob)
                    lookup=np.full(len(w.seq),-1,dtype=np.int64)
                    lookup[target_ids]=np.arange(len(target_ids))
                    pos=lookup[c['row'][indices]]
                    if (pos<0).any():
                        raise AssertionError('Window candidate/target mismatch')
                    old=w.symbols[c['row'][indices]]
                    after=c['mask'][indices]-1
                    fresh[indices]=all_cost[pos,old]-all_cost[pos,after]
                candidate=pick(indices,c,fresh-1.*np.maximum(f['slack'],0.),cap)
                new=add(candidate,'context_refine')
                if new is None or new['total_cdf_gain']<=result['total_cdf_gain']:
                    break
                selected,result=candidate,new
        menus.append(options)
        diagnostics.append(dict(group=group,options=len(options),
            best_context_gain=max(x['total'] for x in options),
            best_own_gain=max(x['own'] for x in options)))
        if len(menus)%5==0:
            report(dict(stage='window_menus',completed=len(menus),total=len(groups),
                model_chunk_forwards=w.calls,elapsed_sec=time.perf_counter()-started))
    (folder/'window_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    # Save the precise finite search space, not just its best outputs.
    serial=[[{**o,'selected':list(map(int,o['selected']))} for o in options] for options in menus]
    (folder/'menus.json').write_text(json.dumps(serial))
    return menus


def wide(frame,a,folder,oa,gpcc,emit):
    raw=load_ply(frame,return_color=False).astype(np.float64)
    absolute=unique(np.rint(raw/2.));origin=absolute.min(0);q=absolute-origin
    w=WindowCodec(oa,q)
    baseline_geom=quality(raw,q,origin,2.)
    baseline_gpcc=gpcc.encode(q,verify=True)
    cache=np.load(str(Path(a.candidates)/(Path(frame).stem+'_proposals.npz')))
    if not np.array_equal(cache['seq'],w.seq):
        raise AssertionError('Cached candidate sequence no longer matches current codec')
    c=cache['candidates']
    c=c[c['row']%w.bptt!=0]
    model=GeometryOctreeRate(q,'gpcc_categorical')
    f,audit=features(c,w.seq,q,raw,origin,w,model)
    emit(dict(frame=frame,trial='baseline',oa=w.baseline,gpcc=baseline_gpcc,
              quality=baseline_geom,candidates=len(c),**audit))
    np.savez_compressed(folder/'candidate_features.npz',candidates=c,**f)
    menus=build_menus(c,f,w,folder,emit)
    plans={tuple():dict(selected=[],producers=['no_op'])}
    for budget in a.budgets:
        for objective in ('prior','own','total'):
            chosen,info=knapsack(menus,round(len(q)*budget),objective)
            key=tuple(sorted(chosen))
            plans.setdefault(key,dict(selected=chosen,producers=[]))['producers'].append(dict(budget=budget,**info))
    evaluated=[]
    best_total_key=None
    best_total_gain=-np.inf
    for number,plan in enumerate(plans.values()):
        selected=plan['selected'];edits=edit_list(c,selected)
        out,counts=apply_plan(q,w.seq,edits)
        geom=quality(raw,out,origin,2.)
        if not quality_ok(baseline_geom,geom):
            raise AssertionError('Window quality slack proof does not match final geometry')
        eval_oa=w.evaluate(edits,actual=True)
        encoded=gpcc.encode(out,verify=True)
        result=dict(frame=frame,trial='full_plan',id=number,producers=plan['producers'],
            operation_count=sum(counts.values()),operation_percent=100*sum(counts.values())/len(q),
            counts=counts,quality=geom,quality_ok=True,
            current_prior_score=float(f['prior'][selected].sum()),own_score=float(f['own'][selected].sum()),
            gpcc=encoded,gpcc_C=100*(encoded['bits']/baseline_gpcc['bits']-1.),
            oa_C=100*(eval_oa['actual_oa']['bits']/w.baseline['bits']-1.),**eval_oa)
        if eval_oa['total_cdf_gain']>best_total_gain:
            best_total_key=number;best_total_gain=eval_oa['total_cdf_gain']
        emit(result);evaluated.append(result)
        np.savez_compressed(folder/('plan_%03d.npz'%number),canonical_voxels=out,origin=origin,qs=2.,edits=np.array(edits,dtype=np.int64).reshape(-1,2))
    comparisons=[]
    for budget in a.budgets:
        eligible=[r for r in evaluated if r['operation_count']<=round(len(q)*budget)]
        selectors=dict(current_prior=lambda r:r['current_prior_score'],node_only=lambda r:r['own_score'],
            total_context=lambda r:r['total_cdf_gain'],actual_oa_oracle=lambda r:-r['actual_oa']['bits'],
            actual_gpcc_oracle=lambda r:-r['gpcc']['bits'])
        for method,key in selectors.items():
            winner=max(eligible,key=key)
            record=dict(frame=frame,trial='same_pool_comparison',budget=budget,method=method,
                winner=winner['id'],candidate_plan_ids=[r['id'] for r in eligible],
                operation_percent=winner['operation_percent'],counts=winner['counts'],
                oa_C=winner['oa_C'],gpcc_C=winner['gpcc_C'],quality=winner['quality'])
            comparisons.append(record);emit(record)
    # Independent full-sequence model rerun for each scorer's widest winner.
    final_ids={r['winner'] for r in comparisons if r['budget']==max(a.budgets)}
    for index in sorted(final_ids):
        data=np.load(str(folder/('plan_%03d.npz'%index)))
        fresh=oa.encode(data['canonical_voxels'],roundtrip=True)
        if fresh['stream_sha256']!=evaluated[index]['actual_oa']['stream_sha256']:
            raise AssertionError('Partial-context result differs from full Actual OA')
        emit(dict(frame=frame,trial='full_oa_parity',plan=index,oa=fresh,exact_match=True))
    (folder/'comparisons.json').write_text(json.dumps(comparisons,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--candidates',default='experiments/results/20261017_octree_joint_masks_v2')
    p.add_argument('--budgets',nargs='+',type=float,default=[.0015,.003,.01,.03,.05])
    a=p.parse_args();torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    protected=protect_snapshot(root)
    (folder/'protected_before.json').write_text(json.dumps(protected,indent=2))
    (folder/'arguments.json').write_text(json.dumps(dict(args=vars(a),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(root),text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        current_score='zero-Actor empirical prior logits; no Network checkpoint loaded',
        gpcc_node_probabilities='unavailable from existing wrapper; full Actual stream teacher only'),indent=2))
    oa,gpcc=OAActual('cpu'),ActualCodec(2.,2)
    with (folder/'results.jsonl').open('w',buffering=1) as log:
        def emit(x):
            log.write(json.dumps(x)+'\n');print(json.dumps(x),flush=True)
        for frame in a.frames:
            sub=folder/Path(frame).stem;sub.mkdir()
            wide(frame,a,sub,oa,gpcc,emit)
    after=protect_snapshot(root)
    changed=[p for p in protected if after.get(p)!=protected[p]]
    (folder/'protected_after.json').write_text(json.dumps(after,indent=2))
    if changed:
        raise AssertionError('Protected files changed externally or by experiment: '+repr(changed))
    print(json.dumps(dict(complete=True,protected_files=len(protected),changed=[])),flush=True)


if __name__=='__main__':
    main()
