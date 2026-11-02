"""Private real-train probe with source freezing and gradient/update evidence.

Existing checkpoints and seed caches are read-only. Copies and new codec
results are written only beneath --out. A phase override is diagnostic only.
"""
import argparse
import json
import math
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-source-snapshot', type=Path)
    parser.add_argument('--probe-learned-exploration', choices=('true','false'), default='true')
    options, remainder = parser.parse_known_args()
    out = Path(remainder[remainder.index('--out')+1])
    seed_cache = Path(remainder[remainder.index('--seed-cache')+1])
    if options.probe_source_snapshot:
        from experiments.sparsepcgc_direction.common import FrozenSources
        sys.meta_path.insert(0, FrozenSources(options.probe_source_snapshot))
    import torch
    import train as training
    original_parse, original_network, original_step = training.parse_pugan_args, training.Network, torch.optim.Adam.step
    state = {}

    def settings(*args, **kwargs):
        value = original_parse(*args, **kwargs)
        value.test_route_learned_exploration = options.probe_learned_exploration == 'true'
        source, target = seed_cache/'actual_gt', out/'cache/actual_gt'
        if source.is_dir() and not target.exists():
            shutil.copytree(str(source), str(target))
        return value
    training.parse_pugan_args = settings

    def network(*args, **kwargs):
        model = original_network(*args, **kwargs)
        state['model'] = model
        return model
    training.Network = network
    from models.utils.training import autonomous_joint_learning as joint
    original_clip = joint.clip_joint_gradients

    def clip(model, maximum):
        result = original_clip(model, maximum)
        state['clip'] = result
        return result
    joint.clip_joint_gradients = clip
    groups = dict(actor=('actuator.den6_candidate_actor.',),
        proposal=('actuator.den6_candidate_proposal.',),
        add_where=('actuator.add_head.','actuator.add_voxel_head.'),
        prune_where=('actuator.drop_head.',),
        adjust_where=('actuator.move_voxel_head.','actuator.subtree_move_source_head.'),
        add_amount=('actuator.add_amount_head.',), prune_amount=('actuator.drop_amount_head.',),
        adjust_amount=('actuator.move_amount_head.',), coarse_amount=('actuator.algorithmic_amount_selector_head.',))

    def step(optimizer, *args, **kwargs):
        model = state.get('model')
        if model is None:
            return original_step(optimizer, *args, **kwargs)
        coarse = {id(p) for p in model.actuator.algorithmic_amount_selector_head.parameters()}
        if not any(id(p) in coarse for group in optimizer.param_groups for p in group['params']):
            return original_step(optimizer, *args, **kwargs)
        params = {n:p for n,p in model.named_parameters() if any(n.startswith(v) for v in groups.values())}
        before = {n:p.detach().cpu().clone() for n,p in params.items()}
        grads = {n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in params.items()}
        result = original_step(optimizer, *args, **kwargs)
        measured = {}
        for role, prefixes in groups.items():
            names = [n for n in params if n.startswith(prefixes)]
            elements = sum(params[n].numel() for n in names)
            gn = math.sqrt(sum(float(grads[n].double().square().sum()) for n in names if grads[n] is not None))
            wn = math.sqrt(sum(float(before[n].double().square().sum()) for n in names))
            dn = math.sqrt(sum(float((params[n].detach().cpu()-before[n]).double().square().sum()) for n in names))
            measured[role] = dict(gradient_l2=gn, gradient_rms=gn/math.sqrt(elements) if elements else None,
                delta_l2=dn, update_rate=dn/wn if wn else None)
        parameter = getattr(model.actuator.den6_candidate_actor, 'exploration_log_scales', None)
        controls = None
        if parameter is not None:
            from models.utils.training.autonomous_exploration import exploration_controls
            controls = {k:float(v.detach()) for k,v in exploration_controls(model.actuator.den6_candidate_actor, model.args).items()}
        row = dict(step=int(model.args._global_train_step)+1, modules=measured,
                   clip=state.get('clip'), learned_controls_after_update=controls)
        with (out/'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        return result
    torch.optim.Adam.step = step
    sys.argv = ['autonomous_collapse_probe.py']+remainder
    runpy.run_module('experiments.autonomous_collapse_probe', run_name='__main__')
    (out/'probe_conditions.json').write_text(json.dumps(dict(
        learned_exploration=options.probe_learned_exploration,
        source_snapshot=str(options.probe_source_snapshot),
        gt_cache='private copy of fingerprint-checked measured GT cache',
        optimizer='fresh Adam; optional warm-start weights/runtime, not full optimizer resume'), indent=2))


if __name__ == '__main__':
    main()
