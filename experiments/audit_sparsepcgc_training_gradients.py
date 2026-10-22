"""Observational, isolated train.py gradient audit; never patch training sources.

Run in the training Python environment. Each probe uses retained backward,
compatible with checkpointed heads, then restores Parameter and retained
decision gradients. Optimizer, objective, scheduling and data remain unchanged.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=3)
    opts = parser.parse_args()
    out = Path(opts.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import train as training
    from models.utils.training import train_runtime as runtime

    state = {'step': 0}
    records = out / 'measurements.jsonl'

    def emit(kind, **data):
        row = dict(kind=kind, step=state['step'], time=time.time(), **data)
        with records.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False, default=str) + '\n')

    def named():
        return list(state['model'].named_parameters())

    def group_specs():
        specs = [('ALL', '')]
        specs += [('module:' + n, n + '.') for n, _ in state['model'].named_children()]
        for prefix in ('actuator', 'full_cloud_amount_selector', 'structure_analyzer'):
            module = getattr(state['model'], prefix, None)
            if module is not None:
                specs += [(prefix + ':' + n, prefix + '.' + n + '.')
                          for n, _ in module.named_children()]
        specs += [('head:' + n, tuple(keys)) for n, keys in runtime._step_grad_group_specs()
                  if keys]
        return specs

    def stats(gradients):
        rows = []
        for label, prefix in group_specs():
            selected = [(n, p) for n, p in named() if
                        (any(k in n for k in prefix) if isinstance(prefix, tuple)
                         else n.startswith(prefix))]
            trainable = [(n, p) for n, p in selected if p.requires_grad]
            grads = [gradients.get(n) for n, _ in trainable]
            used = [g.detach() for g in grads if g is not None]
            count = sum(g.numel() for g in used)
            squares = sum(float(g.double().square().sum()) for g in used)
            absolute = sum(float(g.double().abs().sum()) for g in used)
            maximum = max([float(g.abs().max()) for g in used] or [0.])
            finite = math.isfinite(squares) and math.isfinite(absolute)
            status = ('absent' if not selected else 'frozen' if not trainable else
                      'disconnected' if not used else 'nonfinite' if not finite else
                      'zero' if squares == 0 else 'nonzero')
            rows.append(dict(group=label, status=status,
                             parameters=sum(p.numel() for _, p in selected),
                             trainable=sum(p.numel() for _, p in trainable),
                             tensors_connected=len(used), tensors_none=len(grads)-len(used),
                             l2=math.sqrt(squares) if finite else None,
                             rms=math.sqrt(squares/count) if count and finite else 0.,
                             mean_abs=absolute/count if count and finite else 0.,
                             max_abs=maximum if finite else None))
        return rows

    def current():
        return {n: p.grad for n, p in named()}

    def probe(label, tensor):
        scalar = float(tensor.detach()) if torch.is_tensor(tensor) and tensor.numel() == 1 else None
        value = scalar if scalar is not None and math.isfinite(scalar) else None
        if not torch.is_tensor(tensor) or not tensor.requires_grad:
            emit('loss_gradient', loss=label, value=value, requires_grad=False,
                 groups=stats({}))
            return
        params = named()
        saved = [(p, p.grad) for _, p in params]
        decisions = runtime._den6_retained_decision_tensors(state['model'])
        saved_decisions = [(t, t.grad) for t in decisions]
        state['model'].zero_grad(set_to_none=True)
        for t in decisions:
            t.grad = None
        try:
            tensor.backward(retain_graph=True)
            emit('loss_gradient', loss=label, value=value, requires_grad=True,
                 groups=stats(current()))
        except Exception as exc:
            emit('probe_error', loss=label, error=repr(exc))
            raise
        finally:
            for p, grad in saved:
                p.grad = grad
            for t, grad in saved_decisions:
                t.grad = grad

    original_network = training.Network
    def network(*args, **kwargs):
        model = original_network(*args, **kwargs)
        state['model'] = model
        return model
    training.Network = network

    original_loss = training.Loss
    def loss_factory(*args, **kwargs):
        loss = original_loss(*args, **kwargs)
        state['loss'] = loss
        return loss
    training.Loss = loss_factory

    original_train = training.train
    def train_entry(model, args, *rest, **kwargs):
        state['args'] = args
        (out / 'effective_args.json').write_text(json.dumps(vars(args), indent=2, default=str))
        inventory = [dict(name=n, shape=list(p.shape), requires_grad=p.requires_grad)
                     for n, p in named()]
        (out / 'parameter_inventory.json').write_text(json.dumps(inventory, indent=2))
        return original_train(model, args, *rest, **kwargs)
    training.train = train_entry

    original_capture = training._capture_den6_policy_only_gradients
    def capture(policy_loss, model):
        state['step'] += 1
        print('[AUDIT] step {} policy sources'.format(state['step']), flush=True)
        scale = float(getattr(state['args'], 'heuristic_guidance_online_policy_backward_scale', 1.))
        sources = getattr(model, '_den6_policy_source_losses', {})
        for key, tensor in sources.items():
            if torch.is_tensor(tensor):
                probe('policy_source:' + key, scale * tensor)
        probe('policy_only_scaled', policy_loss)
        return original_capture(policy_loss, model)
    training._capture_den6_policy_only_gradients = capture

    def loss_rows(args, model, items, **where):
        emit('step_location', location=where)
        print('[AUDIT] step {} objective gradients'.format(state['step']), flush=True)
        # Include every original probe tensor. Identical aliases are measured
        # once and explicitly linked, not recorded as independent measurements.
        seen = {}
        for label, tensor in items:
            if torch.is_tensor(tensor) and id(tensor) in seen:
                emit('loss_alias', loss=label, alias=seen[id(tensor)])
                continue
            if torch.is_tensor(tensor):
                seen[id(tensor)] = label
            probe(label, tensor)
        return []
    training.build_step_grad_rows = loss_rows

    original_restore = training._restore_den6_policy_only_gradients
    def restore(snapshot, model=None):
        emit('joint_backward_before_routing', groups=stats(current()))
        result = original_restore(snapshot, model=model)
        emit('after_policy_routing', groups=stats(current()), routing=result)
        return result
    training._restore_den6_policy_only_gradients = restore

    original_clip = training.clip_model_gradients
    def clip(model, *args, **kwargs):
        emit('optimizer_preclip', groups=stats(current()))
        result = original_clip(model, *args, **kwargs)
        emit('optimizer_postclip', groups=stats(current()), clip=str(result))
        return result
    training.clip_model_gradients = clip

    original_step = torch.optim.Adam.step
    def adam_step(optimizer, *args, **kwargs):
        ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        is_main = any(id(p) in ids for _, p in named())
        active = [(n, p) for n, p in named() if id(p) in ids and p.grad is not None] if is_main else []
        before = [(n, p, p.detach().clone()) for n, p in active]
        if not is_main:
            grads = [p.grad.detach() for g in optimizer.param_groups for p in g['params']
                     if p.grad is not None]
            emit('surrogate_optimizer_gradient', l2=math.sqrt(sum(float(g.double().square().sum()) for g in grads)),
                 connected_tensors=len(grads))
        result = original_step(optimizer, *args, **kwargs)
        if is_main:
            delta = {n: p.detach()-old for n, p, old in before}
            emit('actual_parameter_update', groups=stats(delta))
            print('[AUDIT] step {} optimizer update measured'.format(state['step']), flush=True)
        return result
    torch.optim.Adam.step = adam_step

    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT / 'train.py', *sorted((ROOT / 'models').rglob('*.py'))]
              if p.is_file()}
    (out / 'source_sha256_before.json').write_text(json.dumps(hashes, indent=2))
    argv = ['train.py', '--compression', 'SparsePCGC', '--max_train_steps', str(opts.steps),
            '--log_root', str(out / 'log'), '--out_path', str(out / 'checkpoints'),
            '--mail_notify', 'false']
    for flag, folder in [
        ('point_transformer_feature_disk_cache_dir', 'features'),
        ('sparsepcgc_actual_gt_disk_cache_dir', 'actual_gt'),
        ('exact_teacher_cache_root', 'exact'),
        ('heuristic_guidance_online_cache_dir', 'online'),
        ('heuristic_guidance_tensor_disk_cache_dir', 'tensors'),
        ('structure_fixed_disk_cache_dir', 'structure'),
        ('surrogate_pretrain_cache_dir', 'surrogate'),
        ('sparsepcgc_tmp_dir', 'tmp'),
    ]:
        argv += ['--' + flag, str(out / 'cache' / folder)]
    (out / 'invocation.json').write_text(json.dumps(argv, indent=2))
    sys.argv = argv
    os.chdir(str(ROOT))
    try:
        training.main()
        emit('completed')
    finally:
        changed = [name for name, digest in hashes.items()
                   if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
        emit('source_integrity', changed=changed)


if __name__ == '__main__':
    main()
