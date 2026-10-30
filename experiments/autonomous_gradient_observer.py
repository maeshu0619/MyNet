"""Optional read-only gradient observation in an isolated training probe.

Every diagnostic traversal is before the existing policy capture/backward.
Gradients are cleared afterwards. The optimizer, objective and clip bounds
remain the production ones; this observer must not be used for speed claims.
"""
import json
import math

import torch


def install_observers(training, output):
    from models.utils.training import autonomous_joint_learning as joint
    groups = {
        'Add Where': ('actuator.add_head.', 'actuator.add_voxel_head.'),
        'Prune Where': ('actuator.drop_head.',),
        'Adjust Where': ('actuator.move_voxel_head.', 'actuator.subtree_move_source_head.'),
        'Add Amount': ('actuator.add_amount_head.',),
        'Prune Amount': ('actuator.drop_amount_head.',),
        'Adjust Amount': ('actuator.move_amount_head.',),
        'Coarse Amount': ('actuator.algorithmic_amount_selector_head.',),
        'Actor': ('actuator.den6_candidate_actor.',),
        'Proposal': ('actuator.den6_candidate_proposal.',),
        'Candidate Critic': ('actuator.den6_candidate_critic.',),
        'Plan Critic': ('actuator.den6_plan_critic.',),
        'Gate': ('actuator.operation_gate_head.',),
        'Adapter': ('point_transformer_feature_adapter.',),
    }

    def measure(model, losses):
        base = model.module if hasattr(model, 'module') else model
        params = dict(base.named_parameters())
        names = {role: [name for name,p in params.items()
                        if name.startswith(prefix) and p.requires_grad]
                 for role,prefix in groups.items()}
        rows, vectors, result = {}, {}, {}
        for source, loss in losses.items():
            model.zero_grad(set_to_none=True)
            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward(retain_graph=True)
            rows[source], vectors[source] = {}, {}
            for role, keys in names.items():
                vector = torch.cat([params[k].grad.detach().float().cpu().reshape(-1)
                                    if params[k].grad is not None else
                                    torch.zeros(params[k].numel()) for k in keys]) if keys else torch.empty(0)
                norm = float(vector.double().norm())
                rows[source][role] = dict(l2=norm, rms=norm/math.sqrt(vector.numel())
                    if vector.numel() else None, elements=vector.numel(), finite=bool(torch.isfinite(vector).all()))
                result[source+'_'+role.replace(' ','_').lower()] = norm
                vectors[source][role] = vector
        model.zero_grad(set_to_none=True)
        cosines={}
        for first,second in (('actual_compression_policy','geometry'),
                             ('actual_joint_replay','geometry'),
                             ('actual_joint_replay','actual_compression_policy')):
            if first not in vectors or second not in vectors:
                continue
            cosines[first+'__'+second]={}
            for role in groups:
                a,b=vectors[first][role].double(),vectors[second][role].double()
                denominator=float(a.norm()*b.norm())
                cosines[first+'__'+second][role]=float(a.dot(b))/denominator if denominator else None
        with (output/'source_gradients.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=int(base.args._global_train_step)+1,
                sources=rows,cosines=cosines))+'\n')
        return result

    training._probe_den6_policy_source_gradients=measure
    original=joint.clip_joint_gradients

    def clip(model, maximum):
        report=original(model,maximum)
        base=model.module if hasattr(model,'module') else model
        with (output/'clip_stages.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=int(base.args._global_train_step)+1,report=report))+'\n')
        return report

    joint.clip_joint_gradients=clip
