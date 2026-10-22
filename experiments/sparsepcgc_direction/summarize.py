"""Aggregate independent real measurements, never invent missing runs/values."""
import argparse
import collections
import csv
import json
from pathlib import Path
import sys
import math
import re

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.sparsepcgc_direction.common import correlations,source_hashes,write_json,runtime_digest


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x] if path.exists() else []


def mean(x):return sum(x)/len(x) if x else None


def export(path,values):
    keys=list(dict.fromkeys(k for r in values for k in r))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(values)


def main():
    import torch
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    o=p.parse_args();out=o.out.resolve();out.mkdir(parents=True,exist_ok=False)
    summary={};metrics=[];sources=[];update_csv=[];teacher_csv=[];decisions_csv=[];cosines_csv=[];pair_csv=[]
    completed=[d for d in sorted(o.root.iterdir()) if (d/'completed.json').exists() and (d/'effective_args.json').exists()]
    for d in completed:
        teachers=rows(d/'teachers.jsonl');decisions=rows(d/'decisions.jsonl');grads=rows(d/'gradients.jsonl')
        # Early instrument versions also saw the emulator Adam. Last row is
        # the real decision optimizer and agrees with post-update snapshots.
        updates={r['step']:r for r in rows(d/'updates.jsonl')}
        opts=json.loads((d/'options.json').read_text());info=dict(options=opts)
        info['integrity']=json.loads((d/'integrity.json').read_text())
        info['source_current_unchanged']=json.loads((d/'source_sha256.json').read_text())==source_hashes()
        info['initial_weights']=json.loads((d/'initial_weights.json').read_text())
        info['training_time']=json.loads((d/'training_time.json').read_text())
        validation={int(path.stem.split('_')[1]):json.loads(path.read_text()) for path in sorted(d.glob('validation_[0-9][0-9][0-9].json'))}
        info['validation']=validation
        if teachers:
            active=[t for t in teachers if t['debug']['actual_group_pair_available']]
            reused=[t for t in teachers if t['scalars'].get('matched_reused')]
            where=[t for t in teachers if t['debug']['actual_actor_replay_exact_pairs']>0]
            info['teachers']=dict(steps=len(teachers),pair_count=len(active),reused=len(reused),
                where_pair_count=len(where),groups=dict(collections.Counter(t['scalars']['factorized_group'] for t in active)),
                actor_accuracy=mean([t['debug']['actual_actor_replay_accuracy'] for t in where]),
                proposal_accuracy=mean([t['debug']['actual_proposal_replay_accuracy'] for t in where]),
                candidate_critic_accuracy=mean([t['debug']['actual_candidate_critic_replay_accuracy'] for t in where]),
                unique_frames=len({t['scalars']['input_file'] for t in teachers}),
                plan_critic=correlations([t['debug']['plan_critic_value'] for t in teachers],[t['debug']['actor_rd_target'] for t in teachers]),
                plan_critic_RMSE=math.sqrt(mean([(t['debug']['plan_critic_value']-t['debug']['actor_rd_target'])**2 for t in teachers])))
            anchors={};pairs=[]
            for t in teachers:
                s=t['scalars'];debug=t['debug'];key=(s['input_file'],s['factorized_group'],s.get('matched_anchor_hash',''))
                if not s.get('matched_reused'):
                    anchors[key]=t;continue
                previous=anchors.pop(key,None)
                if previous is None:continue
                ps=previous['scalars'];pd=previous['debug']
                dc=ps['current_compression_value']-s['current_compression_value']
                dg=ps['current_geometry_value']-s['current_geometry_value']
                dj=pd['actor_rd_target']-debug['actor_rd_target']
                row=dict(run=d.name,step=t['step'],group=s['factorized_group'],delta_C=dc,delta_G=dg,
                    delta_J=dj,weighted_C=3*dc,weighted_G=150*dg,
                    tiebreak_and_roundoff=dj-3*dc-150*dg,
                    compression_direction_conflict=bool(abs(dc)>1e-7 and dc*dj<0),
                    compression_unchanged=abs(dc)<=1e-7,usable=debug['actual_group_pair_available'])
                pairs.append(row);pair_csv.append(row)
            info['pair_decomposition']=dict(pairs=len(pairs),compression_direction_conflicts=sum(r['compression_direction_conflict'] for r in pairs),
                compression_unchanged=sum(r['compression_unchanged'] for r in pairs))
            for t in teachers:
                s=t['scalars'];debug=t['debug'];teacher_csv.append(dict(run=d.name,step=t['step'],
                    frame=s['input_file'],group=s['factorized_group'],reused=s.get('matched_reused'),
                    pair=debug['actual_group_pair_available'],delta=debug['actual_group_pair_delta'],
                    C=s['current_compression_value'],G=s['current_geometry_value'],J=debug['actor_rd_target'],
                    plan_Q=debug['plan_critic_value'],where_pairs=debug['actual_actor_replay_exact_pairs'],
                    actor_accuracy=debug['actual_actor_replay_accuracy'],proposal_accuracy=debug['actual_proposal_replay_accuracy'],
                    candidate_critic_accuracy=debug['actual_candidate_critic_replay_accuracy']))
        for r in grads:
            for group,v in r['groups'].items():
                sources.append(dict(run=d.name,step=r['step'],stage=r['stage'],source=r.get('source',''),module=group,**v))
        for step,r in updates.items():
            for group,v in r['update'].items():
                theta=r['theta'][group]['l2']
                update_csv.append(dict(run=d.name,step=step,module=group,update_l2=v['l2'],update_rms=v['rms'],
                    update_ratio=v['l2']/theta if theta>1e-15 else None,theta_l2=theta,
                    prestep_gradient_l2=r['gradient'][group]['l2'],prestep_gradient_rms=r['gradient'][group]['rms']))
        for group in (next(iter(updates.values()))['update'] if updates else {}):
            src={}
            for source in sorted({r.get('source') for r in grads if r['stage']=='source'}):
                subset=[r['groups'][group] for r in grads if r['stage']=='source' and r['source']==source]
                src[source]=dict(mean_l2=mean([r['l2'] for r in subset]),max_l2=max(r['l2'] for r in subset),
                    mean_rms=mean([r['rms'] for r in subset]),nonzero=sum(r['l2']>0 for r in subset),probes=len(subset))
            group_updates=[r for r in update_csv if r['run']==d.name and r['module']==group]
            post=[r['groups'][group] for r in grads if r['stage']=='clip_model_gradients:after']
            before=[r['groups'][group] for r in grads if r['stage']=='_balance_actual_operation_head_gradients:before']
            after=[r['groups'][group] for r in grads if r['stage']=='_balance_actual_operation_head_gradients:after']
            row=dict(run=d.name,module=group,source_summary=src,
                mean_update_l2=mean([r['update_l2'] for r in group_updates]),
                mean_update_rms=mean([r['update_rms'] for r in group_updates]),
                mean_update_ratio=mean([r['update_ratio'] for r in group_updates if r['update_ratio'] is not None]),
                nonzero_updates=sum(r['update_l2']>0 for r in group_updates),updates=len(group_updates),
                mean_postclip_l2=mean([r['l2'] for r in post]),mean_postclip_rms=mean([r['rms'] for r in post]),
                balance_clip_probes=sum(b['l2']>a['l2']*(1+1e-5) for b,a in zip(before,after)),balance_probes=len(before))
            metrics.append(row)
        globalclips=[r for r in grads if r['stage']=='clip_model_gradients:after']
        info['global_clip']=dict(probes=len(globalclips),clipped=sum(bool(r.get('debug',{}).get('train_grad_clip_applied')) for r in globalclips))
        logged_clips=[]
        for path in (d/'log').rglob('*.txt'):
            for line in path.read_text(errors='replace').splitlines():
                if line.startswith('Den6OnlineAudit:'):
                    match=re.search(r'grad_before_clip=([^,]+), clip_applied=(True|False)',line)
                    if match:logged_clips.append(dict(norm=float(match.group(1)),clipped=match.group(2)=='True'))
        info['global_clip_all_updates_from_native_audit']=dict(count=len(logged_clips),clipped=sum(x['clipped'] for x in logged_clips))
        for r in decisions:
            for op,c in r['candidate'].items():
                row={k:v for k,v in c.items() if k!='selected_ids'}
                decisions_csv.append(dict(run=d.name,step=r['step'],phase=r['phase'],frame=r['frame'],operation=op,
                    **row,noise=r['debug'].get('effective_where_gumbel_scale'),
                    confidence=r['debug'].get('network_plan_confidence')))
        for r in {r['step']:r for r in rows(d/'cosines.jsonl')}.values():
            cosines_csv.extend(dict(run=d.name,step=r['step'],**v) for v in r['values'])
        info['validation_counts']={str(stage):{op:mean([r['counts'][op] for r in decisions if r['phase']=='validation_%03d'%stage])
                                  for op in ['Add','Prune','Adjust']} for stage in validation}
        summary[d.name]=info
    matched=[]
    for phase,steps,suffix in [('cold',64,''),('mature',48,'_storage_v2')]:
        for seed in (21,22):
            names=[phase+'_'+variant+'_s'+str(seed)+suffix for variant in ['before','current']]
            if not all(n in summary for n in names):continue
            a,b=[summary[n] for n in names]
            ta,tb=[rows(o.root/n/'teachers.jsonl') for n in names]
            da,db=[rows(o.root/n/'decisions.jsonl') for n in names]
            order_a=[(r['frame'],r['source_hash']) for r in da if r['phase']=='train']
            order_b=[(r['frame'],r['source_hash']) for r in db if r['phase']=='train']
            same=a['initial_weights']==b['initial_weights']
            runtime_hashes=[runtime_digest(torch.load(str(o.root/n/'model_000.pth'),map_location='cpu')['runtime']) for n in names]
            matched.append(dict(phase=phase,seed=seed,initial_weights_identical=same,input_order_identical=order_a==order_b,
                initial_runtime_identical=runtime_hashes[0]==runtime_hashes[1],runtime_hashes=runtime_hashes,
                final_steps=steps,validation_same=a['validation'][steps]['sample_signature']==b['validation'][steps]['sample_signature']))
            assert same and order_a==order_b and runtime_hashes[0]==runtime_hashes[1]
    for seed in (21,22):
        names=['cold_current_s'+str(seed),'proxy_distill_off_s'+str(seed)]
        if not all(n in summary for n in names):continue
        a,b=[summary[n] for n in names]
        streams=[[ (r['frame'],r['source_hash']) for r in rows(o.root/n/'decisions.jsonl') if r['phase']=='train'] for n in names]
        same=a['initial_weights']==b['initial_weights']
        matched.append(dict(phase='proxy_distill_ablation',seed=seed,initial_weights_identical=same,
            input_order_identical=streams[0]==streams[1],final_steps=64,
            validation_same=a['validation'][64]['sample_signature']==b['validation'][64]['sample_signature']))
        assert same and streams[0]==streams[1]
    write_json(out/'matched_conditions.json',matched);write_json(out/'summary.json',summary)
    write_json(out/'module_summary.json',metrics)
    export(out/'gradients.csv',sources);export(out/'updates.csv',update_csv);export(out/'teachers.csv',teacher_csv)
    export(out/'decisions.csv',decisions_csv);export(out/'cosines.csv',cosines_csv)
    export(out/'pair_decomposition.csv',pair_csv)
    print(json.dumps({n:{str(k):[v['value'],v['geometry_value']] for k,v in i.get('validation',{}).items()}
                      for n,i in summary.items()},indent=2))


if __name__=='__main__':main()
