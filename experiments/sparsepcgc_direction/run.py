"""Controlled before/after training plus read-only decision/Actual observations.

All changes below are in-process experimental instrumentation. The original
Network, loss and codec source files are never written. Output must be new.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.sparsepcgc_direction.common import (
    FrozenSources, append_json, canonical_points, digest, module_digest,
    source_hashes, write_json,
)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--variant',choices=['before','current'],default='current')
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--seed',type=int,default=21)
    parser.add_argument('--views',type=int,choices=[1,2],default=2)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--validation',type=int,default=4)
    parser.add_argument('--seed-cache',type=Path,default=ROOT/'experiments/results/20261022_credit_fixes/16_verified_initial/cache')
    parser.add_argument('--probe-steps',default='2,4,6,8,24,64')
    parser.add_argument('--evaluate-only',type=Path)
    parser.add_argument('--disable-proxy-distill',action='store_true',
                        help='One-factor experimental ablation; production args are not edited')
    parser.add_argument('--post-window-probe-floor',type=float)
    parser.add_argument('--before-source',type=Path,default=ROOT/'experiments/results/20261022_credit_fixes/00_baseline_source')
    opts=parser.parse_args()
    out=opts.out.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(str(Path(__file__)),str(out/'runner.py.txt'))
    shutil.copy2(str(Path(__file__).with_name('common.py')),str(out/'common.py.txt'))
    write_json(out/'options.json',vars(opts)); initial_sources=source_hashes()
    write_json(out/'source_sha256.json',initial_sources)
    for command,name in [(['git','status','--short'],'git_status.txt'),
                         (['git','diff','--binary'],'git_diff.txt'),
                         (['git','rev-parse','HEAD'],'git_head.txt')]:
        (out/name).write_bytes(subprocess.check_output(command,cwd=str(ROOT)))
    if opts.variant=='before':
        sys.meta_path.insert(0,FrozenSources(opts.before_source.resolve()))
        manifest=json.loads((opts.before_source/'manifest.json').read_text())
        assert all(hashlib.sha256((opts.before_source/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
        write_json(out/'frozen_sources.json',manifest)
    if opts.checkpoint:
        inputs=[opts.checkpoint,Path(str(opts.checkpoint)+'.actor_critic_runtime.pth')]
        checkpoint_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.is_file()}
        write_json(out/'checkpoint_sha256.json',checkpoint_hashes)
    else:
        checkpoint_hashes={}
    # Static inputs only: the teacher GT and edited Actual caches are fresh.
    for folder in ['features','online','structure','tensors']:
        source=opts.seed_cache/folder
        if source.is_dir():shutil.copytree(str(source),str(out/'cache'/folder))
    import numpy as np
    import torch
    import train as training
    from models.utils.training import train_runtime as runtime
    from models.utils.training.sparsepcgc_actual_credit import MatchedFrameDataset
    state=dict(step=0,phase='train',latest_plan=None,latest_gt=None,latest_gen=None,
               snapshots={},model=None,args=None,loss=None,step_started=None)
    probes={int(x) for x in opts.probe_steps.split(',') if x}
    original_parse=training.parse_pugan_args
    def parse(*a,**kw):
        args=original_parse(*a,**kw)
        args.seed=opts.seed
        args.sparsepcgc_cpu_actual_overlap=False
        # Both sources receive exactly the same external 2-view input stream
        # and the same schedule estimate. Do not pair current a second time.
        args.sparsepcgc_matched_frame_repeats=1
        args.train_full_cloud_val_frames=0  # evaluate saved snapshots only
        args.sparsepcgc_full_cloud_amount_reset_heads_on_more_training=False
        if opts.disable_proxy_distill:
            args.heuristic_guidance_online_actor_critic_distill_weight=0.
        if opts.post_window_probe_floor is not None:
            args.heuristic_guidance_online_post_window_exploration_floor=opts.post_window_probe_floor
        if opts.checkpoint:
            args.more_training=True;args.more_training_ckpt=str(opts.checkpoint.resolve())
        return args
    training.parse_pugan_args=parse
    original_window=training.apply_epoch_file_window
    def window(dataset,args,episode):
        base=original_window(dataset,args,episode)
        return MatchedFrameDataset(base,opts.views) if opts.views>1 else base
    training.apply_epoch_file_window=window
    original_optimizer=training.build_optimizer_and_scheduler
    def make_optimizer(model,args,writer):
        optimizer,scheduler=original_optimizer(model,args,writer)
        if opts.offset:
            # Terminal values read from the original 043722 training log.
            # Adam moments are absent from that checkpoint: warm-start only.
            for group in optimizer.param_groups:
                group['lr']=6.25e-5 if group.get('name','main')=='main' else 1e-5
            write_json(out/'late_optimizer_lrs.json',[dict(name=g.get('name'),lr=g['lr']) for g in optimizer.param_groups])
        return optimizer,scheduler
    training.build_optimizer_and_scheduler=make_optimizer

    original_network=training.Network
    def make_network(*a,**kw):
        model=original_network(*a,**kw);state['model']=model
        for op,name in [('Add','add_amount_head'),('Prune','drop_amount_head'),('Adjust','move_amount_head')]:
            head=getattr(model.actuator,name,None)
            if head is not None:
                def amount_observe(module,inputs,output,op=op):
                    value=output.detach().double()
                    state.setdefault('amount_raw',{})[op]=dict(min=float(value.min().cpu()),
                        max=float(value.max().cpu()),mean=float(value.mean().cpu()))
                head.register_forward_hook(amount_observe)
        original_build=model.actuator._build_exact_den6_residual_plan
        code=original_build.__func__.__code__
        def build(*a,**kw):
            capture={}
            def profile(frame,event,value):
                if event=='return' and frame.f_code is code:
                    capture.update(frame.f_locals)
            previous=sys.getprofile();sys.setprofile(profile)
            try: result=original_build(*a,**kw)
            finally:sys.setprofile(previous)
            if result is None:return result
            coords,debug=result
            arrays={}
            for key in ['candidate_critic_values','candidate_critic_uncertainties',
                        'candidate_actor_logits','candidate_proposal_logits',
                        'candidate_actor_features','candidate_critic_features',
                        'combined_logits','candidate_proposal_masks','network_terms']:
                arrays[key]={op:t.detach().cpu().numpy().copy() for op,t in capture.get(key,{}).items()}
            arrays['rank_score']={op:m['rank_score'].detach().cpu().numpy().copy()
                                  for op,m in capture['guidance']['candidate_tensor_map'].items()
                                  if 'rank_score' in m}
            pools=capture['pools']
            # Keep only the exact edit description, never modify cached pool.
            compact={op:[{k:c.get(k) for k in ['candidate_id','remove_coords','add_coords','rank_score','operation']}
                         for c in candidates] for op,candidates in pools.items()}
            plan=dict(source=capture['voxel_coords'][0].T.detach().cpu().numpy().copy(),
                final=coords[0].T.detach().cpu().numpy().copy(),arrays=arrays,pools=compact,
                order=list(capture['operation_order']),requested=dict(capture['requested_counts']),
                selected=copy.deepcopy(capture['selected_indices_by_operation']),
                ordered=copy.deepcopy(capture['ordered_indices']),
                authority={op:float(model.actuator._den6_actual_residual_authority(op+'Where')) for op in pools},
                debug={k:v for k,v in debug.items() if isinstance(v,(str,int,float,bool))},
                counts=dict(debug['selected_counts']),
                fine_means=debug.get('operation_amount_mean_log_residuals',{}),
                bin_probabilities=debug.get('amount_bin_probabilities',[]),
                gate_probabilities=debug.get('operation_gate_probabilities',{}),
                heuristic_weight=float(getattr(model.args,'heuristic_guidance_final_where_weight',.01)),
                plan_feature=capture['plan_feature'].detach().cpu().numpy().copy(),
                plan_q=float(capture['plan_critic_mean'].detach().cpu()),
                frame=str(getattr(model.args,'_current_input_file','')))
            state['latest_plan']=plan
            candidate_stats={}
            for op in pools:
                q=arrays['candidate_critic_values'][op].reshape(-1)
                sigma=arrays['candidate_critic_uncertainties'][op].reshape(-1)
                rank=arrays['rank_score'][op].reshape(-1)*plan['heuristic_weight']
                valid=np.isfinite(rank)
                order=np.argsort(-rank[valid]);indices=np.flatnonzero(valid)
                cut=indices[order[min(max(plan['requested'][op],1),len(order))-1]]
                lcb=q-q[cut]-float(getattr(model.args,'heuristic_guidance_online_actual_lcb_kappa',1.))*np.sqrt(sigma*sigma+sigma[cut]**2)
                net=arrays['network_terms'][op]
                candidate_stats[op]=dict(pool=len(q),proposal=int(arrays['candidate_proposal_masks'][op].sum()),
                    selected=len(plan['selected'][op]),q_min=float(q.min()),q_max=float(q.max()),
                    sigma_mean=float(sigma.mean()),lcb_positive=int((lcb>0).sum()),
                    authority=plan['authority'][op],network_nonzero=int((np.abs(net)>1e-9).sum()),
                    selected_ids=[compact[op][i]['candidate_id'] for i in plan['selected'][op]])
            append_json(out/'decisions.jsonl',dict(step=state['step'],phase=state['phase'],
                frame=plan['frame'],source_hash=digest(canonical_points(plan['source'])),
                final_hash=digest(canonical_points(plan['final'])),candidate=candidate_stats,
                counts=plan['counts'],debug=plan['debug'],plan_q=plan['plan_q']))
            return result
        model.actuator._build_exact_den6_residual_plan=build
        original_forward=model.forward
        if hasattr(model,'point_transformer_feature_adapter'):
            def adapter_observe(module,inputs,output):
                v=output.detach().float()
                state['adapter']=dict(rms=float(v.square().mean().sqrt().cpu()),
                    l2=float(v.norm().cpu()),n=v.numel(),hash=digest(v.cpu().numpy()))
                if state['phase']!='train':state['adapter_value']=v.cpu().numpy().copy()
            model.point_transformer_feature_adapter.register_forward_hook(adapter_observe)
        def forward(*a,**kw):
            if model.training:
                raw=int(getattr(model.args,'_global_train_step',0))
                model.args._global_train_step=raw+opts.offset
                # Common random numbers at each forward entry; no evaluation
                # can perturb future training RNG state.
                runtime.set_seed(opts.seed*1000003+state['step'],deterministic=False)
                state['step_started']=time.perf_counter()
            result=original_forward(*a,**kw)
            if model.training:
                state['step']+=1
            voxel=model.last_actuator_voxel_state
            for key in ['voxel_step','voxel_offset']:
                if torch.is_tensor(voxel.get(key)):
                    state['latest_plan'][key]=voxel[key].detach().cpu()
            if state['phase']!='train' or state['step'] in (1,2):
                plan=state['latest_plan']
                label=(state['phase'].replace(':','_')+('_%03d'%state['step'] if model.training else '')
                       +'_'+Path(plan['frame']).stem)
                torch.save(plan,str(out/(label+'_plan.pt')))
            append_json(out/'features.jsonl',dict(step=state['step'],phase=state['phase'],
                adapter=state.get('adapter'),amount_raw=state.get('amount_raw'),
                gate_parameter=model.point_transformer_feature_gate.detach().cpu().reshape(-1).tolist()))
            if state['phase']!='train' and state.get('adapter_value') is not None:
                np.savez_compressed(out/(state['phase']+'_'+Path(state['latest_plan']['frame']).stem+'_adapter.npz'),
                                    feature=state['adapter_value'])
            return result
        model.forward=forward
        original_policy=model.discrete_policy_loss
        pcode=original_policy.__func__.__code__
        def policy(*a,**kw):
            captured={}
            def profile(frame,event,value):
                if event=='return' and frame.f_code is pcode:captured.update(frame.f_locals)
            previous=sys.getprofile();sys.setprofile(profile)
            try:result=original_policy(*a,**kw)
            finally:sys.setprofile(previous)
            scalars={k:v for k,v in captured.items() if isinstance(v,(str,int,float,bool))}
            append_json(out/'teachers.jsonl',dict(step=state['step'],scalars=scalars,
                debug=dict(model.last_discrete_policy_debug)))
            state['components']={k:captured[k] for k in ['plan_critic_loss_weighted',
                'actual_candidate_critic_replay_weighted','actual_actor_replay_weighted',
                'actual_proposal_replay_weighted','actual_group_critic_weighted',
                'amount_local_credit_weighted','fine_local_credit_weighted',
                'operation_local_credit_weighted','adaptive_amount_entropy_weighted'] if k in captured}
            return result
        model.discrete_policy_loss=policy
        return model
    training.Network=make_network

    original_loss=training.Loss
    def make_loss(*a,**kw):
        loss=original_loss(*a,**kw);state['loss']=loss
        original_encode=loss._encode_actual_batch
        def encode(args,xyz,final_w=None):
            selected=loss._select_actual_points(xyz[0].float(),final_w,args,0).detach().cpu()
            stats=original_encode(args,xyz,final_w=final_w)
            array=selected.T.contiguous().numpy()
            append_json(out/'encodes.jsonl',dict(step=state['step'],phase=state['phase'],
                frame=str(getattr(args,'_current_input_file','')),hash=digest(canonical_points(array)),
                n=len(array),bit=float(stats['bit']),seconds=float(stats.get('encode_time',0.))))
            if state['phase']!='train' or state['step'] in (1,2):
                # These are exactly the unmodified production encoder inputs.
                count=int(state.get('encode_counter',0));state['encode_counter']=count+1
                np.savez_compressed(out/(state['phase'].replace(':','_')+('_%03d'%state['step'] if state['phase']=='train' else '')+'_'+
                    Path(str(getattr(args,'_current_input_file',''))).stem+('_encode_%05d.npz'%count)),points=array)
            return stats
        loss._encode_actual_batch=encode
        return loss
    training.Loss=make_loss

    def focus():
        return {name:p for name,p in state['model'].named_parameters() if p.requires_grad}
    def grouped(named):
        groups={'CandidateActor':['actuator.den6_candidate_actor.'],
            'CandidateCritic':['actuator.den6_candidate_critic.'],
            'PlanCritic':['actuator.den6_plan_critic.'],'Proposal':['actuator.den6_candidate_proposal.'],
            'AddWhere':['actuator.add_head.','actuator.add_voxel_head.'],
            'PruneWhere':['actuator.drop_head.'],
            'AdjustWhere':['actuator.move_voxel_head.','actuator.subtree_move_source_head.'],
            'AddAmount':['actuator.add_amount_head.'],'PruneAmount':['actuator.drop_amount_head.'],
            'AdjustAmount':['actuator.move_amount_head.'],'CoarseAmount':['actuator.algorithmic_amount_selector_head.'],
            'Adapter':['point_transformer_feature_adapter.'],'Gate':['actuator.operation_gate_head.']}
        result={}
        for group,prefixes in groups.items():
            parts=[value.detach().reshape(-1).double().cpu() for name,value in named.items()
                   if any(name.startswith(prefix) for prefix in prefixes)]
            v=torch.cat(parts) if parts else torch.zeros(0,dtype=torch.double)
            result[group]=dict(l2=float(v.norm()),rms=float(v.norm())/max(v.numel(),1)**.5,n=v.numel())
        return result
    def gradients():return {n:(p.grad if p.grad is not None else torch.zeros_like(p)) for n,p in focus().items()}
    def probe(source,value):
        named=focus();params=list(named.values())
        retained=runtime._den6_retained_decision_tensors(state['model'])
        saved=[(t,t.grad) for t in retained]
        for t in retained:t.grad=None
        try:
            grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True) if torch.is_tensor(value) and value.requires_grad else [None]*len(params)
            values={name:(g.detach() if g is not None else torch.zeros_like(param))
                    for (name,param),g in zip(named.items(),grads)}
            state.setdefault('probe_vectors',{})[source]={n:g.detach().cpu().clone() for n,g in values.items()}
            append_json(out/'gradients.jsonl',dict(step=state['step'],stage='source',source=source,
                groups=grouped(values)))
        finally:
            for t,g in saved:t.grad=g
    original_capture=training._capture_den6_policy_only_gradients
    def capture(policy_loss,model):
        if state['step'] in probes:
            state['probe_vectors']={}
            scale=float(model.args.heuristic_guidance_online_policy_backward_scale)
            sources={'actual':model._den6_policy_source_losses['actual_compression_policy']*scale,
                     'proxy':model._den6_policy_source_losses['proxy']*scale,
                     'entropy':model._den6_policy_source_losses['entropy']*scale,
                     'geometry_policy':model._den6_policy_source_losses['geometry']*scale}
            sources.update({k:v*scale for k,v in state.get('components',{}).items()})
            sources['policy_total']=policy_loss
            sources['unclassified']=policy_loss-sum(v*scale for v in model._den6_policy_source_losses.values())
            for source,value in sources.items():
                probe(source,value)
        return original_capture(policy_loss,model)
    training._capture_den6_policy_only_gradients=capture
    def loss_rows(args,model,items,**kw):
        if state['step'] in probes:
            for name,value in items:
                if name in ['L_downstream','weighted_geometry_primary','L_total']:
                    probe(name,value)
        return []
    training.build_step_grad_rows=loss_rows
    for name in ['_restore_den6_policy_only_gradients','_balance_actual_operation_head_gradients','clip_model_gradients']:
        if not hasattr(training,name):continue
        original=getattr(training,name)
        def observed(*a,original=original,name=name,**kw):
            measure=state['step'] in probes
            if measure:append_json(out/'gradients.jsonl',dict(step=state['step'],stage=name+':before',groups=grouped(gradients())))
            result=original(*a,**kw)
            if measure:append_json(out/'gradients.jsonl',dict(step=state['step'],stage=name+':after',groups=grouped(gradients()),debug=result))
            return result
        setattr(training,name,observed)
    original_step=torch.optim.Adam.step
    def adam_step(optimizer,*a,**kw):
        # The registered fast emulator is also a child of the Network, but
        # has its own Adam. Identify the main optimizer by decision heads.
        model_ids={id(p) for n,p in focus().items()
                   if n.startswith(('actuator.den6_candidate_actor.',
                                    'actuator.algorithmic_amount_selector_head.'))}
        main=any(id(p) in model_ids for g in optimizer.param_groups for p in g['params'])
        if not main:return original_step(optimizer,*a,**kw)
        params=focus();before={n:p.detach().clone() for n,p in params.items()}
        grad=grouped(gradients());result=original_step(optimizer,*a,**kw)
        update=grouped({n:p.detach()-before[n] for n,p in params.items()})
        theta=grouped(before)
        append_json(out/'updates.jsonl',dict(step=state['step'],gradient=grad,update=update,theta=theta,
            seconds=time.perf_counter()-state['step_started'] if state['step_started'] else None))
        if state['step'] in probes:
            comparisons=[];vectors=state.get('probe_vectors',{})
            for left,right in [('actual','proxy'),('proxy','entropy'),('actual','weighted_geometry_primary'),
                               ('L_downstream','weighted_geometry_primary'),
                               ('plan_critic_loss_weighted','actual_candidate_critic_replay_weighted')]:
                if left not in vectors or right not in vectors:continue
                for group,prefix in [('Actor','actuator.den6_candidate_actor.'),('PruneWhere','actuator.drop_head.'),
                    ('CandidateCritic','actuator.den6_candidate_critic.'),
                    ('AddWhere','actuator.add_voxel_head.'),('AdjustWhere','actuator.move_voxel_head.'),
                    ('AddAmount','actuator.add_amount_head.'),('PruneAmount','actuator.drop_amount_head.'),
                    ('AdjustAmount','actuator.move_amount_head.'),('CoarseAmount','actuator.algorithmic_amount_selector_head.'),
                    ('Adapter','point_transformer_feature_adapter.')]:
                    names=[n for n in params if n.startswith(prefix)]
                    x=torch.cat([vectors[left][n].reshape(-1).double() for n in names])
                    y=torch.cat([vectors[right][n].reshape(-1).double() for n in names])
                    den=float(x.norm()*y.norm())
                    comparisons.append(dict(group=group,left=left,right=right,
                        cosine=float(torch.dot(x,y))/den if den>1e-30 else None))
            append_json(out/'cosines.jsonl',dict(step=state['step'],values=comparisons))
            direction=[]
            for source,vector in vectors.items():
                for group,prefix in [('Actor','actuator.den6_candidate_actor.'),('PruneWhere','actuator.drop_head.'),
                    ('AddWhere','actuator.add_voxel_head.'),('AdjustWhere','actuator.move_voxel_head.'),
                    ('Proposal','actuator.den6_candidate_proposal.'),
                    ('AddAmount','actuator.add_amount_head.'),('PruneAmount','actuator.drop_amount_head.'),
                    ('AdjustAmount','actuator.move_amount_head.'),('CoarseAmount','actuator.algorithmic_amount_selector_head.'),
                    ('Adapter','point_transformer_feature_adapter.')]:
                    names=[n for n in params if n.startswith(prefix)]
                    if not names:continue
                    g=torch.cat([vector[n].reshape(-1).double() for n in names])
                    delta=torch.cat([(params[n].detach()-before[n]).reshape(-1).double().cpu() for n in names])
                    den=float(g.norm()*delta.norm())
                    direction.append(dict(source=source,group=group,
                        descent_update_cosine=float(torch.dot(-g,delta))/den if den>1e-30 else None,
                        first_order_loss_change=float(torch.dot(g,delta))))
            append_json(out/'update_directions.jsonl',dict(step=state['step'],values=direction))
        if state['step'] in {24,32,opts.steps}:
            snapshot(state['step'])
        if state['step']%8==0:print('[DIRECTION] update',state['step'],flush=True)
        return result
    torch.optim.Adam.step=adam_step

    def snapshot(step):
        model=state['model']
        payload=dict(model={n:p.detach().cpu().clone() for n,p in model.state_dict().items()},
            runtime=copy.deepcopy(model.actor_critic_runtime_state_dict()),step=step,
            global_step=opts.offset+step)
        state['snapshots'][step]=payload
        torch.save(payload,str(out/('model_%03d.pth'%step)))

    def load_runtime(model,value):
        current=model.actor_critic_runtime_state_dict()
        for part,owner in [('network',model),('actuator',model.actuator)]:
            for name in current.get(part,{}):
                if name not in value.get(part,{}) and hasattr(owner,name):delattr(owner,name)
        model.load_actor_critic_runtime_state_dict(copy.deepcopy(value))

    original_train=training.train
    def train(model,args,loss,writer,plot,notifier=None):
        state.update(model=model,args=args,loss=loss)
        write_json(out/'effective_args.json',vars(args))
        write_json(out/'initial_weights.json',dict(sha256=module_digest(model)))
        snapshot(0)
        # Read-only reuse of precomputed candidate JSONs for both seeds.
        dirs=runtime._limit_training_seq_dirs(runtime.collect_seq_dirs2(args.input_dir,dataset_name=args.dataname),args)
        datasets=[(d,runtime.PlyDirDataset(args,d)) for d in dirs]
        needed=[];validation=[]
        for d,ds in datasets:
            needed.extend(str(x) for x in ds.all_files[:int(args.train_frames_per_sequence)])
        for d,ds in datasets[:2]:
            validation.extend((ds,str(x)) for x in ds.all_files[int(args.train_frames_per_sequence):int(args.train_frames_per_sequence)+2])
        validation=validation[:opts.validation]
        args._fixed_full_cloud_validation_records=tuple(validation)
        needed.extend(x for _,x in validation)
        cache=out/'cache/online';cache.mkdir(parents=True,exist_ok=True)
        for path in needed:
            pattern='exact_single_plan_UVG_'+Path(path).stem+'_vs1_pq1_ae0_sr1_m8_*_rf4p0.json'
            for source in Path('/data/maejima/log/mynet_den6_online_cache').glob(pattern):
                target=cache/source.name
                if not target.exists():shutil.copy2(str(source),str(target))
        start=time.perf_counter()
        if opts.evaluate_only:
            target=opts.evaluate_only.resolve()
            initial=torch.load(str(target/'model_000.pth'),map_location='cpu')
            final_file=max(target.glob('model_[0-9][0-9][0-9].pth'),key=lambda p:int(p.stem.split('_')[-1]))
            final=torch.load(str(final_file),map_location='cpu')
            schedule=json.loads((target/'schedule.json').read_text())
            for key,value in schedule.items():setattr(args,key,value)
            specs={'full':final}
            frozen=copy.deepcopy(initial);frozen['global_step']=final['global_step']
            specs['frozen_clock_final']=frozen
            frozen_runtime=copy.deepcopy(frozen);frozen_runtime['runtime']=copy.deepcopy(final['runtime'])
            specs['initial_weights_final_runtime']=frozen_runtime
            blocks={'PruneWhere':['actuator.drop_head.'],
                'AddWhere':['actuator.add_head.','actuator.add_voxel_head.'],
                'AdjustWhere':['actuator.move_voxel_head.','actuator.subtree_move_source_head.'],
                'Actor':['actuator.den6_candidate_actor.'],'Proposal':['actuator.den6_candidate_proposal.'],
                'FineAmounts':['actuator.add_amount_head.','actuator.drop_amount_head.','actuator.move_amount_head.'],
                'CoarseAmount':['actuator.algorithmic_amount_selector_head.'],
                'Adapter':['point_transformer_feature_adapter.','point_transformer_feature_gate']}
            for name,prefixes in blocks.items():
                ablated=copy.deepcopy(final)
                for key in ablated['model']:
                    if any(key.startswith(prefix) for prefix in prefixes):ablated['model'][key]=initial['model'][key].clone()
                specs['rollback_'+name]=ablated
            state['snapshots']={900+i:payload for i,payload in enumerate(specs.values())}
            write_json(out/'evaluation_labels.json',{str(900+i):name for i,name in enumerate(specs)})
            result=None
        else:
            result=original_train(model,args,loss,writer,plot,notifier=notifier)
        write_json(out/'training_time.json',dict(seconds=time.perf_counter()-start,updates=state['step']))
        args.train_full_cloud_val_frames=len(validation)
        write_json(out/'schedule.json',{k:v for k,v in vars(args).items() if 'schedule' in k or k in ['_total_train_steps_estimate','_global_train_step']})
        write_json(out/'validation_manifest.json',[path for _,path in validation])
        for step,payload in sorted(state['snapshots'].items()):
            if not validation:break
            model.load_state_dict(payload['model'],strict=True)
            load_runtime(model,payload['runtime'])
            args._global_train_step=payload['global_step']
            state['phase']='validation_%03d'%step
            value=runtime.run_episode_full_cloud_validation(model=model,args=args,loss=loss,
                seq_datasets=datasets,writer=writer,episode=0,global_step=payload['global_step'],
                use_cuda=True,use_amp=False,amp_dtype=torch.float16)
            write_json(out/('validation_%03d.json'%step),value)
            print('[DIRECTION]',state['phase'],json.dumps(value),flush=True)
        return result
    training.train=train
    argv=['train.py','--compression','SparsePCGC','--max_train_steps',str(opts.steps),
          '--seed',str(opts.seed),'--log_root',str(out/'log'),'--out_path',str(out/'checkpoints'),
          '--mail_notify','false','--sparsepcgc_device','cpu','--heuristic_guidance_online_grad_audit','true']
    for flag,folder in [('point_transformer_feature_disk_cache_dir','features'),
        ('sparsepcgc_actual_gt_disk_cache_dir','actual_gt'),('exact_teacher_cache_root','exact'),
        ('heuristic_guidance_online_cache_dir','online'),('heuristic_guidance_tensor_disk_cache_dir','tensors'),
        ('structure_fixed_disk_cache_dir','structure'),('surrogate_pretrain_cache_dir','surrogate'),('sparsepcgc_tmp_dir','tmp')]:
        argv += ['--'+flag,str(out/'cache'/folder)]
    write_json(out/'invocation.json',argv);sys.argv=argv;os.chdir(str(ROOT))
    try:
        training.main()
        write_json(out/'completed.json',dict(updates=state['step']))
    finally:
        current=source_hashes()
        changes=[p for p,h in initial_sources.items() if current.get(p)!=h]
        checkpoints=[p for p,h in checkpoint_hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
        write_json(out/'integrity.json',dict(source_changes=changes,checkpoint_changes=checkpoints))
        assert not changes and not checkpoints


if __name__=='__main__':main()
