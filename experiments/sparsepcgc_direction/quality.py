"""Reconstruct real Geometry and RD for already measured native oracle labels.

No proxy bits are substituted for codec measurements. Candidate scores are
compared within the same frame, stage and operation, never pooled across scenes.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.sparsepcgc_direction.common import (
    append_json,canonical_points,correlations,geometry_with_fit,materialize,write_json,
)


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def effect(plan,selected):
    import numpy as np
    r,a=set(),set()
    for op,indices in selected.items():
        for i in indices:
            c=plan['pools'][op][i]
            r.update(map(tuple,c.get('remove_coords') or ()))
            a.update(map(tuple,c.get('add_coords') or ()))
    return np.asarray(sorted((set(map(tuple,plan['source']))-r)|a),dtype=np.int64).reshape(-1,3)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--oracle',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    o=p.parse_args();out=o.out.resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    args=SimpleNamespace(**json.loads((o.run/'effective_args.json').read_text()))
    wc=float(args.w_com);wg=float(args.cp_lambda_geom);tau=float(args.cp_tau_geom)
    edit_weight=max(float(getattr(args,'heuristic_guidance_online_edit_tiebreak_weight',0.)),0.)
    def penalty(g):return wg*max(g['training_G_reconstructed']-tau,0.)
    def edit_penalty(plan,selected):
        return edit_weight*sum((2 if op=='Adjust' else 1)*len(indices)
                               for op,indices in selected.items())/len(plan['source'])
    plans=read_rows(o.oracle/'plans.jsonl');candidates=read_rows(o.oracle/'candidates.jsonl')
    enriched=[];checks=[]
    keys=sorted({(r['stage'],r['frame']) for r in plans})
    for stage,frame in keys:
        plan=torch.load(str(o.run/('validation_%03d_%s_plan.pt'%(stage,Path(frame).stem))),map_location='cpu')
        all_selected=plan['selected'];normal=geometry_with_fit(plan,plan['final'],all_selected,args)
        source_geom=geometry_with_fit(plan,plan['source'],{op:[] for op in all_selected},args)
        for row in [r for r in plans if (r['stage'],r['frame'])==(stage,frame)]:
            kind=row['kind']
            if kind=='actual_selected':selected=all_selected;coords=plan['final']
            elif kind.endswith('_only'):
                selected={op:(idx if kind==op+'_only' else []) for op,idx in all_selected.items()}
                coords=effect(plan,selected)
            else:
                ordered={}
                for op in all_selected:
                    actor=torch.tensor(plan['arrays']['candidate_actor_logits'][op]).float()
                    centered=actor-actor.mean();residual=.5*torch.tanh(centered/centered.square().mean().sqrt().clamp_min(1.))
                    authority=1. if kind.endswith('authority1') else plan['authority'][op]
                    logits=torch.tensor(plan['arrays']['rank_score'][op]).float()*plan['heuristic_weight']+residual*authority
                    logits=logits.masked_fill(~torch.tensor(plan['arrays']['candidate_proposal_masks'][op]).bool(),float('-inf'))
                    ordered[op]=torch.argsort(logits,descending=True).tolist()
                coords,selected=materialize(plan['source'],plan['pools'],plan['order'],plan['requested'],ordered)
            g=geometry_with_fit(plan,coords,selected,args)
            row.update(full_geometry=g,J=wc*row['C']+penalty(g)+edit_penalty(plan,selected),
                       training_quality_no_worse=g['training_G_reconstructed']<=normal['training_G_reconstructed']+2e-7)
            append_json(out/'plans.jsonl',row)
        finalset=set(map(tuple,plan['final']))
        for raw in [r for r in candidates if (r['stage'],r['frame'])==(stage,frame)]:
            row=dict(raw);op=row['operation']
            index=next(i for i,c in enumerate(plan['pools'][op]) if c['candidate_id']==row['candidate_id'])
            c=plan['pools'][op][index];r=set(map(tuple,c.get('remove_coords') or ()));a=set(map(tuple,c.get('add_coords') or ()))
            chosen={name:[] for name in all_selected};chosen[op]=[index]
            isolated=effect(plan,chosen);g=geometry_with_fit(plan,isolated,chosen,args)
            row['intrinsic_geometry']=g
            row['intrinsic_RD_gain']=wc*100*row['intrinsic_gain_bits']/row['baseline_bits']+penalty(source_geom)-penalty(g)-edit_penalty(plan,chosen)
            if row['contextual_gain_bits'] is not None:
                chosen={name:list(idx) for name,idx in all_selected.items()}
                if row['selected']:
                    chosen[op].remove(index);coords=np.asarray(sorted((finalset-a)|r),dtype=np.int64).reshape(-1,3)
                    other=geometry_with_fit(plan,coords,chosen,args)
                    geometry_gain=penalty(other)-penalty(normal)+edit_penalty(plan,chosen)-edit_penalty(plan,all_selected)
                else:
                    chosen[op].append(index);coords=np.asarray(sorted((finalset-r)|a),dtype=np.int64).reshape(-1,3)
                    other=geometry_with_fit(plan,coords,chosen,args)
                    geometry_gain=penalty(normal)-penalty(other)+edit_penalty(plan,all_selected)-edit_penalty(plan,chosen)
                row.update(contextual_geometry_counterfactual=other,
                    contextual_geometry_gain_weighted=geometry_gain,
                    contextual_RD_gain=wc*100*row['contextual_gain_bits']/row['baseline_bits']+geometry_gain)
            enriched.append(row);append_json(out/'candidates.jsonl',row)
        print('[QUALITY]',stage,Path(frame).stem,flush=True)
    strata=[]
    for stage,frame,op in sorted({(r['stage'],r['frame'],r['operation']) for r in enriched}):
        rows=[r for r in enriched if (r['stage'],r['frame'],r['operation'])==(stage,frame,op)]
        for target in ['intrinsic_gain_bits','contextual_gain_bits','contextual_RD_gain']:
            usable=[r for r in rows if r.get(target) is not None]
            for score in ['q','actor','proposal']:
                row=dict(stage=stage,frame=frame,operation=op,target=target,score=score,
                         statistics=correlations([r[score] for r in usable],[r[target] for r in usable]))
                if score=='q' and target=='contextual_RD_gain':
                    errors=[]
                    scale=max(float(args.heuristic_guidance_online_actual_actor_min_delta),.01)
                    for i,j in itertools.combinations(range(len(usable)),2):
                        errors.append((usable[i]['q']-usable[j]['q']-math.tanh((usable[i][target]-usable[j][target])/scale))**2)
                    row['diagnostic_pair_normalized_RMSE']=float(np.sqrt(np.mean(errors))) if errors else None
                    row['warning']='Q is trained on group MEAN contrasts, not individual marginals; this is out-of-training-task diagnostic error'
                strata.append(row)
    write_json(out/'rank_statistics.json',strata)
    # Check reconstruction independently on ALL four validation frames.
    for stage in sorted({r['stage'] for r in plans}):
        values=[]
        for path in sorted(o.run.glob('validation_%03d_*_plan.pt'%stage)):
            plan=torch.load(str(path),map_location='cpu')
            values.append(geometry_with_fit(plan,plan['final'],plan['selected'],args)['training_G_reconstructed'])
        observed=json.loads((o.run/('validation_%03d.json'%stage)).read_text())
        error=float(np.mean(values))-observed['geometry_value']
        assert abs(error)<2e-6,('Geometry reconstruction mismatch',stage,error)
        checks.append(dict(stage=stage,reconstructed=float(np.mean(values)),native=observed,absolute_error=abs(error)))
    write_json(out/'reconstruction_checks.json',checks)
    write_json(out/'conditions.json',dict(w_com=wc,cp_lambda_geom=wg,tau=tau,edit_tiebreak_weight=edit_weight,rows=len(enriched)))


if __name__=='__main__':main()
