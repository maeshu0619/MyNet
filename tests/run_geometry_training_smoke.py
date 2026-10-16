"""Isolated real training diagnostic: pass normal train.py CLI arguments.

Records decision gradients and actual parameter differences without changing
the optimizer update. Use a separate --log_root/--out_path/Surrogate root.
"""
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train

models = []
original_init = train.Network.__init__
original_step = torch.optim.Adam.step


def model_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    models.append(self)


def measured_step(self, *args, **kwargs):
    model = models[-1] if models else None
    if model is None:
        return original_step(self, *args, **kwargs)
    owned = {id(p) for group in self.param_groups for p in group['params']}
    prefixes = ('den6_candidate_actor', 'algorithmic_amount_selector_head',
                'add_amount_head', 'drop_amount_head', 'move_amount_head', 'operation_gate_head',
                'geometry_joint_context')
    tracked = {name: p for name, p in model.named_parameters()
               if id(p) in owned and any(prefix in name for prefix in prefixes)}
    if not tracked:
        return original_step(self, *args, **kwargs)
    before = {name: p.detach().cpu().clone() for name, p in tracked.items()}
    gradients = {name: None if p.grad is None else p.grad.detach().float().cpu()
                 for name, p in tracked.items()}
    result = original_step(self, *args, **kwargs)
    rows = {}
    for prefix in prefixes:
        names = [n for n in tracked if prefix in n]
        grads = [gradients[n] for n in names if gradients[n] is not None]
        rows[prefix] = dict(
            grad_l2=sum(float((g*g).sum()) for g in grads)**.5,
            finite=all(bool(torch.isfinite(g).all()) for g in grads),
            update_l2=sum(float((tracked[n].detach().cpu()-before[n]).square().sum())
                          for n in names)**.5)
    path = os.path.join(model.args.log_root, 'decision_updates.jsonl')
    with open(path, 'a') as stream:
        stream.write(json.dumps(dict(step=int(getattr(model.args, '_global_train_step', -1)), heads=rows))+'\n')
    return result


if __name__ == '__main__':
    train.Network.__init__ = model_init
    torch.optim.Adam.step = measured_step
    train.main()
