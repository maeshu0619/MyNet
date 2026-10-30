"""Private real-training probe for a saved autonomous policy and its pool.

Checkpoint reads are hash checked. The optional phase override is confined to
this process; no production state or log is written by this wrapper.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diagnostic-checkpoint', type=Path)
    parser.add_argument('--diagnostic-runtime', type=Path)
    parser.add_argument('--diagnostic-phase', type=float)
    parser.add_argument('--capture-steps', default='1')
    parser.add_argument('--diagnostic-legacy-policy',action='store_true')
    parser.add_argument('--diagnostic-recovery-probability',type=float)
    parser.add_argument('--final-validation',type=int,default=0)
    parser.add_argument('--balanced-validation',action='store_true')
    parser.add_argument('--diagnostic-gradients',action='store_true')
    options, remainder = parser.parse_known_args()
    out = Path(remainder[remainder.index('--out')+1])
    captures = {int(x) for x in options.capture_steps.split(',') if x}
    import torch
    import train as training
    parse = training.parse_pugan_args
    def settings(*args,**kwargs):
        value=parse(*args,**kwargs)
        if options.diagnostic_legacy_policy:
            value.test_route_codec_context=False
            value.test_route_amount_uniform_fraction=0.
            value.test_route_noop_recovery_probability=0.
            value.test_route_structured_where_noise=0.
            value.test_route_actual_plan_replay=False
        if options.diagnostic_recovery_probability is not None:
            value.test_route_noop_recovery_probability=options.diagnostic_recovery_probability
        if options.diagnostic_gradients:
            value.heuristic_guidance_online_source_grad_probe=True
            value.heuristic_guidance_online_source_grad_probe_interval=1
        value.train_full_cloud_val_frames=options.final_validation
        return value
    training.parse_pugan_args=settings
    if options.diagnostic_gradients:
        from experiments.autonomous_gradient_observer import install_observers
        install_observers(training,out)
    validation_arguments={}
    if options.final_validation:
        from models.utils.training import train_runtime as runtime
        if options.balanced_validation:
            def balanced_records(args,seq_datasets,max_frames):
                existing=getattr(args,'_fixed_full_cloud_validation_records',None)
                if existing is not None:return tuple(existing),False
                choices=[];train_paths=set()
                limit=int(args.train_frames_per_sequence)
                if limit <= 0:raise ValueError('Balanced private validation requires an explicit split')
                for _,dataset in seq_datasets:
                    files=list(getattr(dataset,'all_files',dataset.files))
                    train_paths.update(str(Path(x).resolve()) for x in files[:limit])
                    choices.append([(dataset,str(Path(x).resolve())) for x in files[limit:]])
                records=[];index=0
                while len(records)<max_frames and any(index<len(x) for x in choices):
                    for group in choices:
                        if index<len(group) and len(records)<max_frames:records.append(group[index])
                    index+=1
                assert len(records)==max_frames
                assert not train_paths.intersection(path for _,path in records)
                args._fixed_full_cloud_validation_records=tuple(records)
                args._fixed_full_cloud_validation_train_paths=tuple(sorted(train_paths))
                return tuple(records),True
            runtime.fixed_full_cloud_validation_records=balanced_records
        original_validation=runtime.run_episode_full_cloud_validation
        target=int(remainder[remainder.index('--steps')+1]) if '--steps' in remainder else 4
        def validation(**kwargs):
            validation_arguments.update(kwargs)
            args=kwargs['args'];frames=args.train_full_cloud_val_frames
            final=int(kwargs['global_step'])>=target
            if not final:args.train_full_cloud_val_frames=0
            try:result=original_validation(**kwargs)
            finally:args.train_full_cloud_val_frames=frames
            if final and kwargs.get('diagnostic_policy_override') is None:
                path=out/'final_validation.json'
                with path.open('x') as f:json.dump(result,f,indent=2,default=str)
            return result
        training.run_episode_full_cloud_validation=runtime.run_episode_full_cloud_validation=validation
    original = training.Network
    checkpoint_hash = (hashlib.sha256(options.diagnostic_checkpoint.read_bytes()).hexdigest()
                       if options.diagnostic_checkpoint else None)

    def network(*args, **kwargs):
        model = original(*args, **kwargs)
        if options.diagnostic_checkpoint:
            payload = torch.load(str(options.diagnostic_checkpoint), map_location='cpu')
            weights = payload.get('state_dict', payload.get('model', payload))
            model.load_state_dict(weights, strict=True)
        if options.diagnostic_runtime:
            model.load_actor_critic_runtime_state_dict(torch.load(str(options.diagnostic_runtime),map_location='cpu'))
        if options.diagnostic_phase is not None:
            model.actuator._exploration_phase = lambda: options.diagnostic_phase
        build = model.actuator._build_exact_den6_residual_plan
        code = build.__func__.__code__
        count = [0]

        def observed(*positional, **named):
            count[0] += 1
            if count[0] not in captures:
                return build(*positional, **named)
            capture = {}

            def profile(frame, event, result):
                if frame.f_code is code and event == 'return':
                    capture.update(frame.f_locals)

            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                result = build(*positional, **named)
            finally:
                sys.setprofile(previous)
            if result is None:
                return result
            arrays = {}
            for name in ('combined_logits', 'candidate_actor_features', 'network_scores'):
                arrays[name] = {k: v.detach().cpu() for k, v in capture.get(name, {}).items()}
            torch.save(dict(source=capture['voxel_coords'].detach().cpu(),
                final=result[0].detach().cpu(), pools=capture['pools'], arrays=arrays,
                frame=str(getattr(model.args, '_current_input_file', ''))),
                out/('captured_%03d.pth' % count[0]))
            return result

        model.actuator._build_exact_den6_residual_plan = observed
        forward = model.forward

        def observed_forward(*positional, **named):
            value = forward(*positional, **named)
            if not model.training:
                debug=model.last_actuator_voxel_state.get('ana_den6_exact_residual_plan_debug',{})
                row=dict(frame=str(getattr(model.args,'_current_input_file','')),
                    counts=debug.get('selected_counts'),ratios=debug.get('selected_amount_ratios'),
                    requested=debug.get('joint_requested_ratios'),hash=debug.get('final_edited_voxel_hash'))
                with (out/'validation_decisions.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if count[0] in captures:
                state = model.last_actuator_voxel_state
                torch.save({k: v.detach().cpu() for k, v in state.items()
                            if torch.is_tensor(v) and k in (
                                'voxel_step', 'voxel_offset', 'final_voxel_coords')},
                           out/('geometry_context_%03d.pth' % count[0]))
            return value
        model.forward = observed_forward
        return model

    training.Network = network
    sys.argv = ['autonomous_speed_probe.py']+remainder
    runpy.run_module('experiments.autonomous_speed_probe', run_name='__main__')
    if options.final_validation and not (out/'final_validation.json').exists():
        if not validation_arguments:
            raise RuntimeError('No completed diagnostic Episode supplied fixed-validation inputs')
        class PrivateWriter:
            def write(self,message):
                with (out/'post_validation.txt').open('a') as stream:
                    stream.write(str(message)+'\n')
        validation_arguments.update(writer=PrivateWriter(),global_step=target,
                                    diagnostic_policy_override=None)
        result=original_validation(**validation_arguments)
        with (out/'final_validation.json').open('x') as stream:
            json.dump(result,stream,indent=2,default=str)
    if validation_arguments:
        args=validation_arguments['args']
        (out/'final_validation_measurements.json').write_text(json.dumps(
            getattr(args,'_autonomous_validation_measurements',[]),indent=2))
        torch.save(validation_arguments['model'].actor_critic_runtime_state_dict(),out/'final_runtime.pth')
    if options.diagnostic_checkpoint:
        assert hashlib.sha256(options.diagnostic_checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    (out/'diagnostic_conditions.json').write_text(json.dumps(dict(
        checkpoint=str(options.diagnostic_checkpoint), checkpoint_sha256=checkpoint_hash,
        phase_override=options.diagnostic_phase,legacy_policy=options.diagnostic_legacy_policy,
        recovery_probability_override=options.diagnostic_recovery_probability,
        optimizer='fresh Adam, warm-start weights only'), indent=2))


if __name__ == '__main__':
    main()
