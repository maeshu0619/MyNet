"""Read-only weight ablations using the existing full-cloud validation flow.

Uses a fresh private Loss/worker/cache. No optimizer update is performed.
Each mode uses the same GT split, native codec, pool and deterministic eval.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--step',type=int,default=64)
    p.add_argument('--modes',default='full,initial,rollback_where,rollback_amount')
    p.add_argument('--fixed-amount',type=float,default=.001)
    p.add_argument('--fixed-amounts',help='Diagnostic Add,Prune,Adjust ratios; overrides the common fixed amount')
    o=p.parse_args();o.out.mkdir(exist_ok=False,parents=True)
    from experiments.sparsepcgc_direction.common import source_hashes
    protected=source_hashes()
    for folder in ('features','online','structure','tensors','actual_gt'):
        source=o.run/'cache'/folder
        if source.is_dir():shutil.copytree(str(source),str(o.out/'cache'/folder))
    registry=o.out/'registry/SparsePCGC';registry.mkdir(parents=True)
    shutil.copy2('/data/maejima/data/pretrained_surrogate/SparsePCGC/20260729_001750.pth',registry)
    import torch
    import train as training
    from models.utils.training import train_runtime as runtime
    initial=torch.load(o.run/'initial.pth',map_location='cpu')
    path=o.run/('step_%03d.pth'%o.step)
    final=torch.load(path,map_location='cpu')['model']
    checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    saved=json.loads((o.run/'effective_args.json').read_text())
    parse=training.parse_pugan_args
    def settings(*args,**kwargs):
        value=parse(*args,**kwargs)
        for key,item in saved.items():
            if not key.startswith('_'):setattr(value,key,item)
        value.more_training=False;value.log_root=str(o.out/'log');value.out_path=str(o.out/'checkpoints')
        value.surrogate_pretrained_root=str(o.out/'registry');value.mail_notify=False
        value.sparsepcgc_device='cuda';value.sparsepcgc_worker_cpu_trim_interval=1
        value._sparsepcgc_force_fresh_actual_encode=True
        for flag,folder in (
            ('point_transformer_feature_disk_cache_dir','features'),('heuristic_guidance_online_cache_dir','online'),
            ('heuristic_guidance_tensor_disk_cache_dir','tensors'),('structure_fixed_disk_cache_dir','structure'),
            ('sparsepcgc_actual_gt_disk_cache_dir','actual_gt'),('exact_teacher_cache_root','exact'),
            ('surrogate_pretrain_cache_dir','surrogate'),('sparsepcgc_tmp_dir','tmp')):
            setattr(value,flag,str(o.out/'cache'/folder))
        return value
    training.parse_pugan_args=settings
    state={}
    from models.utils.training import autonomous_joint_learning as joint
    original_amount=joint.sample_joint_amounts
    def amount(selector,means,gate,bins,maximum,phase,training,*args,**kwargs):
        result=original_amount(selector,means,gate,bins,maximum,phase,training,*args,**kwargs)
        if state.get('mode') in ('fixed_amount','initial_where_fixed_amount'):
            if training:raise ValueError('Amount intervention is evaluation-only')
            values=[float(x) for x in o.fixed_amounts.split(',')] if o.fixed_amounts else [o.fixed_amount]*3
            if len(values)!=3:raise ValueError('Need Add,Prune,Adjust diagnostic amounts')
            ratios=dict(zip(('Add','Prune','Adjust'),values))
            result['ratios']={op:mean.new_tensor(min(max(ratios[op],0.),float(maximum))) for op,mean in means.items()}
            result['indices']={op:torch.argmin((bins-ratios[op]).abs()) for op in means}
        return result
    joint.sample_joint_amounts=amount
    def evaluate(model,args,loss,writer,plot,notifier=None):
        (o.out/'effective_args.json').write_text(json.dumps(vars(args),indent=2,default=str))
        directories=runtime._limit_training_seq_dirs(runtime.collect_seq_dirs2(args.input_dir,dataset_name=args.dataname),args)
        datasets=[(d,runtime.PlyDirDataset(args,d)) for d in directories]
        records=[];train_paths=set()
        for _,dataset in datasets:
            limit=int(args.train_frames_per_sequence)
            train_paths.update(str(Path(x).resolve()) for x in dataset.all_files[:limit])
            records.append((dataset,str(Path(dataset.all_files[limit]).resolve())))
        assert not train_paths.intersection(path for _,path in records)
        args._fixed_full_cloud_validation_records=tuple(records)
        args._fixed_full_cloud_validation_train_paths=tuple(sorted(train_paths))
        args.train_full_cloud_val_frames=len(records)
        forward=model.forward
        def observed(*positional,**named):
            result=forward(*positional,**named)
            debug=model.last_actuator_voxel_state['ana_den6_exact_residual_plan_debug']
            row=dict(mode=state['mode'],frame=str(args._current_input_file),
                counts=debug['selected_counts'],ratios=debug['selected_amount_ratios'],
                requested=debug.get('joint_requested_ratios'),
                amount_bins=debug.get('joint_amount_bins_by_operation'),
                voxel_hash=debug.get('final_edited_voxel_hash'),
                input_hash=debug.get('autonomous_input_sha256'))
            with (o.out/'decisions.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            return result
        model.forward=observed
        blocks=dict(rollback_where=('actuator.add_head.','actuator.add_voxel_head.','actuator.drop_head.',
                    'actuator.move_voxel_head.','actuator.subtree_move_source_head.',
                    'actuator.den6_candidate_actor.','actuator.den6_candidate_proposal.'),
            rollback_amount=('actuator.algorithmic_amount_selector_head.','actuator.add_amount_head.',
                'actuator.drop_amount_head.','actuator.move_amount_head.','actuator.operation_gate_head.'))
        for mode in o.modes.split(','):
            if mode not in ('full','initial','fixed_amount','initial_where_fixed_amount','zero_native_preference',*blocks):raise ValueError(mode)
            weights=copy.deepcopy(initial if mode=='initial' else final)
            for name in weights:
                prefixes=blocks['rollback_where'] if mode=='initial_where_fixed_amount' else blocks.get(mode,())
                if name.startswith(prefixes):weights[name]=initial[name].clone()
                if mode=='zero_native_preference' and name.endswith('.native_context_preference'):
                    weights[name].zero_()
            model.load_state_dict(weights,strict=True)
            state['mode']=mode;args._global_train_step=o.step
            result=runtime.run_episode_full_cloud_validation(model=model,args=args,loss=loss,writer=writer,
                seq_datasets=datasets,episode=0,global_step=o.step,use_cuda=True,use_amp=False,amp_dtype=torch.float16)
            measurements=getattr(args,'_autonomous_validation_measurements',[])
            with (o.out/(mode+'.json')).open('x') as stream:
                json.dump(dict(result=result,measurements=measurements),stream,indent=2)
            print(mode,json.dumps(result),flush=True)
        return 0.
    training.train=evaluate
    sys.argv=['train.py','--test_route','true','--more_training','false','--mail_notify','false',
        '--log_root',str(o.out/'log'),'--out_path',str(o.out/'checkpoints')]
    try:
        training.main()
    finally:
        current=source_hashes()
        changed=[p for p,h in protected.items() if current.get(p)!=h]
        assert not changed
        assert hashlib.sha256(path.read_bytes()).hexdigest()==checkpoint_hash
        (o.out/'integrity.json').write_text(json.dumps(dict(source_changes=changed,checkpoint_unchanged=True)))


if __name__=='__main__':main()
