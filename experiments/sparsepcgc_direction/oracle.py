"""Native full-cloud oracle for individual and contextual candidate effects.

Evaluates saved real forward plans. Never trains or changes Network/codec code.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.sparsepcgc_direction.common import (
    append_json,canonical_points,correlations,digest,geometry,materialize,write_json,
)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--stages',default='0,64')
    parser.add_argument('--frames',type=int,default=2)
    parser.add_argument('--per-op',type=int,default=4)
    parser.add_argument('--teacher-pairs',action='store_true')
    opts=parser.parse_args();out=opts.out.resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from models.utils.loss.actual_encoder import build_actual_encoder
    from models.utils.loss.compression import CompressionLossMixin
    args=SimpleNamespace(**json.loads((opts.run/'effective_args.json').read_text()))
    args.sparsepcgc_device='cpu';args.sparsepcgc_tmp_dir=str(out/'tmp')
    args.sparsepcgc_actual_result_cache=False  # duplicates test real encode repeatability
    args._den6_online_training_step_active=False
    args._current_exact_teacher_mode='full_cloud'
    write_json(out/'options.json',vars(opts))
    encoder=build_actual_encoder(args)
    measured={};fixed={};records=[]
    def points(plan,coords):
        state={'final_voxel_coords':torch.as_tensor(coords,dtype=torch.long).T.unsqueeze(0),
               'voxel_step':plan.get('voxel_step'),'voxel_offset':plan.get('voxel_offset')}
        xyz,debug=CompressionLossMixin._voxel_state_to_codec_xyz(None,args,state,torch.zeros(1,3,1))
        assert debug['voxel_state_codec_xyz_used']
        return xyz[0]
    def encode(plan,coords,force=False):
        xyz=points(plan,coords)
        key=digest(canonical_points(xyz.T.numpy()))
        reused=key in measured and not force
        if reused:stats=measured[key]
        else:stats=dict(encoder.encode_bits(xyz));measured[key]=stats
        append_json(out/'native_encodes.jsonl',dict(hash=key,bit=stats['bit'],cache_reuse=reused,
            n=int(xyz.shape[-1]),seconds=stats.get('encode_time')))
        return float(stats['bit'])
    def selected_effect(plan,which):
        r,a=set(),set()
        for op,indices in which.items():
            for index in indices:
                c=plan['pools'][op][index]
                r.update(map(tuple,c.get('remove_coords') or ()))
                a.update(map(tuple,c.get('add_coords') or ()))
        source=set(map(tuple,plan['source']))
        return np.asarray(sorted((source-r)|a),dtype=np.int64).reshape(-1,3)
    try:
        if opts.teacher_pairs:
            files=sorted(opts.run.glob('train_*_plan.pt'))
            assert len(files)>=2,'Need saved training anchor/variant plans'
            pair=[]
            for file in files[:2]:
                p=torch.load(str(file),map_location='cpu')
                normal,selected=materialize(p['source'],p['pools'],p['order'],p['requested'],p['ordered'])
                assert np.array_equal(normal,canonical_points(p['final']))
                baseline=encode(p,p['source']);bit=encode(p,p['final'],force=True)
                repeated=encode(p,p['final'],force=True)
                row=dict(file=file.name,frame=p['frame'],bit=bit,repeat_bit=repeated,gt_bit=baseline,
                    C=100*(bit/baseline-1),source_hash=digest(p['source']),
                    counts=p['counts'],selected_ids={op:[p['pools'][op][i]['candidate_id'] for i in indices]
                                                   for op,indices in p['selected'].items()})
                pair.append(row)
            write_json(out/'teacher_pair_reencode.json',pair)
        for stage in [int(x) for x in opts.stages.split(',') if x]:
            choices=sorted(opts.run.glob('validation_%03d_*_plan.pt'%stage))
            seen=set();paths=[]
            for path in choices:
                plan=torch.load(str(path),map_location='cpu')
                seq=Path(plan['frame']).parent.name
                if seq in seen:continue
                seen.add(seq);paths.append(path)
                if len(paths)>=opts.frames:break
            for path in paths:
                p=torch.load(str(path),map_location='cpu');frame=p['frame']
                source=canonical_points(p['source']);final=canonical_points(p['final'])
                reconstructed,selected=materialize(source,p['pools'],p['order'],p['requested'],p['ordered'])
                assert np.array_equal(reconstructed,final),'Experimental materializer must reproduce real plan'
                base=encode(p,source);full=encode(p,final)
                gt=points(p,source).T.numpy();full_geom=geometry(gt,points(p,final).T.numpy())
                append_json(out/'plans.jsonl',dict(stage=stage,frame=frame,kind='actual_selected',
                    bit=full,baseline_bit=base,C=100*(full/base-1),geometry=full_geom,counts=p['counts']))
                family_gain=[]
                for op in p['order']:
                    only={name:(p['selected'][name] if name==op else []) for name in p['order']}
                    family=selected_effect(p,only);bit=encode(p,family);family_gain.append(base-bit)
                    append_json(out/'plans.jsonl',dict(stage=stage,frame=frame,kind=op+'_only',bit=bit,
                        baseline_bit=base,C=100*(bit/base-1),geometry=geometry(gt,points(p,family).T.numpy())))
                append_json(out/'interactions.jsonl',dict(stage=stage,frame=frame,
                    full_gain=base-full,family_gain_sum=sum(family_gain),nonadditivity=base-full-sum(family_gain)))
                # Deployment ablation: same Proposal, counts and conflict rules;
                # remove only the per-candidate LCB veto in a shadow plan.
                for authority_override in [False,True]:
                    ordered={}
                    for op in p['pools']:
                        arrays=p['arrays'];actor=torch.tensor(arrays['candidate_actor_logits'][op]).float()
                        centered=actor-actor.mean();scale=centered.square().mean().sqrt().clamp_min(1.)
                        residual=.5*torch.tanh(centered/scale)
                        authority=1. if authority_override else p['authority'][op]
                        logits=torch.tensor(arrays['rank_score'][op]).float()*p['heuristic_weight']+residual*authority
                        logits=logits.masked_fill(~torch.tensor(arrays['candidate_proposal_masks'][op]).bool(),float('-inf'))
                        ordered[op]=torch.argsort(logits,descending=True).tolist()
                    shadow,chosen=materialize(source,p['pools'],p['order'],p['requested'],ordered)
                    bit=encode(p,shadow);g=geometry(gt,points(p,shadow).T.numpy())
                    append_json(out/'plans.jsonl',dict(stage=stage,frame=frame,
                        kind='shadow_no_lcb_authority1' if authority_override else 'shadow_no_lcb',
                        bit=bit,baseline_bit=base,C=100*(bit/base-1),geometry=g,
                        changed={op:len(set(chosen[op])^set(p['selected'][op])) for op in chosen},
                        quality_success=(bit<=full and all(g[k]<=full_geom[k]+1e-12 for k in g))))
                for op,candidates in p['pools'].items():
                    ids={c['candidate_id']:i for i,c in enumerate(candidates)}
                    q=p['arrays']['candidate_critic_values'][op].reshape(-1)
                    if (frame,op) not in fixed:
                        selected_indices=p['selected'][op]
                        unselected=sorted(set(range(len(candidates)))-set(selected_indices))
                        chosen=[]
                        for subset in (selected_indices,unselected):
                            if subset:
                                ranked=sorted(subset,key=lambda i:q[i])
                                chosen.extend([ranked[0],ranked[-1]])
                        fixed[(frame,op)]=[candidates[i]['candidate_id'] for i in dict.fromkeys(chosen)][:opts.per_op]
                    for cid in fixed[(frame,op)]:
                        if cid not in ids:continue
                        index=ids[cid];c=candidates[index]
                        r=set(map(tuple,c.get('remove_coords') or ()));a=set(map(tuple,c.get('add_coords') or ()))
                        occupied=set(map(tuple,source));finalset=set(map(tuple,final))
                        if not r.issubset(occupied) or a&occupied:continue
                        isolated=np.asarray(sorted((occupied-r)|a),dtype=np.int64).reshape(-1,3)
                        bit=encode(p,isolated);gain=base-bit
                        is_selected=index in p['selected'][op]
                        if is_selected:
                            modified=np.asarray(sorted((finalset-a)|r),dtype=np.int64).reshape(-1,3)
                            marginal=encode(p,modified)-full
                        elif r.issubset(finalset) and not a&finalset:
                            modified=np.asarray(sorted((finalset-r)|a),dtype=np.int64).reshape(-1,3)
                            marginal=full-encode(p,modified)
                        else:
                            marginal=None
                        row=dict(stage=stage,frame=frame,operation=op,candidate_id=cid,
                            selected=is_selected,removed=len(r),added=len(a),
                            q=float(q[index]),sigma=float(p['arrays']['candidate_critic_uncertainties'][op][index]),
                            actor=float(p['arrays']['candidate_actor_logits'][op][index]),
                            proposal=float(p['arrays']['candidate_proposal_logits'][op][index]),
                            intrinsic_gain_bits=gain,contextual_gain_bits=marginal,
                            intrinsic_C=100*(bit/base-1),baseline_bits=base)
                        records.append(row);append_json(out/'candidates.jsonl',row)
                print('[ORACLE]',stage,Path(frame).stem,'labels',len(records),flush=True)
        summary={}
        for stage in sorted({r['stage'] for r in records}):
            for op in ['Add','Prune','Adjust']:
                rows=[r for r in records if r['stage']==stage and r['operation']==op]
                summary[str(stage)+'_'+op]={}
                for score in ['q','actor','proposal']:
                    summary[str(stage)+'_'+op][score]=correlations([r[score] for r in rows],[r['intrinsic_gain_bits'] for r in rows])
        write_json(out/'summary.json',summary)
    finally:encoder.close()


if __name__=='__main__':main()
