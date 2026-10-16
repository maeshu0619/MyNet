"""One interaction-aware policy over all three operation pools.

The conditional distribution is recomputed after every edit. No candidate
Actual encodes, no best-of-N codec oracle, and no change to SparsePCGC.
"""
from collections import defaultdict
import time
import numpy as np
import torch
from .geometry_octree_rate import remember_rate_model


class GeometryJointContext(torch.nn.Module):
    """Actual-trained coefficients on rate, distortion and interaction gain.

Only nine parameters; sees the evolving joint state, not just original nodes.
Zero initialization retains the prior behavior before any Actual experience.
"""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, 3))


def select_joint_geometry_plan(voxel_coords, exact, pools, counts, actor_scores,
                               args, exploration=0., heuristic_only=False, masks=None, context_head=None,
                               random_only=False):
    started = time.perf_counter()
    operations = ('Add', 'Prune', 'Adjust')
    signature = exact['geometry_rate_model_key']
    model = remember_rate_model(args, signature)
    if model is None:
        rows = voxel_coords[0].T.detach().cpu().numpy()
        model = remember_rate_model(args, signature, rows)
    state = dict(model.masks)
    entries, actor = [], []
    for op in operations:
        entries.extend((op, i, c) for i, c in enumerate(pools[op]))
        score = actor_scores[op].reshape(-1)
        centered = score - score.mean()
        actor.append(.5 * torch.tanh(centered / centered.detach().square().mean().sqrt().clamp_min(1.)))
    actor = torch.cat(actor)
    actor_weight = 0. if heuristic_only or random_only else float(getattr(args, 'geometry_octree_actor_weight', .25))
    actor_numpy = actor.detach().float().cpu().numpy() * actor_weight
    trust = float(getattr(args, '_geometry_octree_prior_trust', 1.))
    if random_only:
        trust = 0.
    distortion_weight = float(getattr(args, 'geometry_octree_distortion_weight', .05))
    scale = max(float(getattr(args, 'geometry_octree_score_scale', 4.)), 1e-6)
    temperature = max(float(getattr(args, 'geometry_octree_joint_temperature', .25)), 0.) * exploration
    if random_only:
        temperature = 1.
    gradient_temperature = temperature if temperature > 1e-4 else 1.
    seed = int(torch.randint(0, 2**31 - 1, (), device=actor.device).cpu()) if temperature > 0 else 0
    rng = np.random.RandomState(seed)
    node_users, coord_users = defaultdict(set), defaultdict(set)
    prior = np.empty(len(entries), dtype=np.float32)
    context_features = np.zeros((len(entries), 3), dtype=np.float32)
    op_indices = np.array([operations.index(op) for op, _, _ in entries])
    coefficients = (context_head.weight.tanh() if context_head is not None and not heuristic_only and not random_only
                    else actor.new_zeros((3, 3)))
    coefficient_numpy = coefficients.detach().cpu().numpy()[op_indices]
    eligible = np.ones(len(entries), dtype=bool)
    if masks is not None:
        eligible = np.concatenate([masks[op].detach().cpu().numpy().astype(bool)
                                   for op in operations])
    for j, (_, _, candidate) in enumerate(entries):
        for key in model.dependencies(candidate):
            node_users[key].add(j)
        for field in ('remove_coords', 'add_coords'):
            for point in candidate[field]:
                coord_users[tuple(point)].add(j)
        _, gain = model.patch(candidate, state)
        unit = scale * candidate.get('operation_count', 1)
        prior[j] = (gain - distortion_weight * candidate['geometry_cost']) / unit
        context_features[j, :2] = np.tanh([gain / unit, -candidate['geometry_cost'] / unit])
    selected, choices, prior_rows, context_rows = [], [], [], []
    selected_counts = dict.fromkeys(operations, 0)
    removes, adds, true_adds = set(), set(), set()
    gain_sum, static_sum, evaluations = 0., 0., len(entries)
    for _ in range(sum(max(int(counts[op]), 0) for op in operations)):
        for j, (op, _, _) in enumerate(entries):
            if entries[j][2].get('operation_count', 1) > counts[op] - selected_counts[op]:
                eligible[j] = False
        if not eligible.any():
            break
        row = np.where(eligible, trust * np.tanh(prior), -np.inf)
        scores = row + actor_numpy + (context_features * coefficient_numpy).sum(1)
        if temperature > 0:
            scores = scores + temperature * rng.gumbel(size=len(entries))
        chosen = int(np.argmax(scores))
        op, index, candidate = entries[chosen]
        patch, gain = model.patch(candidate, state)
        prior_rows.append(row.copy())
        context_rows.append(context_features.copy())
        choices.append(chosen)
        selected.append((op, index, candidate))
        selected_counts[op] += candidate.get('operation_count', 1)
        gain_sum += gain
        static_sum += candidate['optimistic_gain_bits']
        state.update(patch)
        invalid = {chosen}
        for field, target in (('remove_coords', removes), ('add_coords', adds)):
            for point in candidate[field]:
                point = tuple(point)
                target.add(point)
                invalid.update(coord_users[point])
                if op == 'Add' and field == 'add_coords':
                    true_adds.add(point)
        eligible[list(invalid)] = False
        affected = set()
        for key in patch:
            affected.update(node_users[key])
        for j in affected:
            if eligible[j]:
                _, new_gain = model.patch(entries[j][2], state)
                unit = scale * entries[j][2].get('operation_count', 1)
                prior[j] = (new_gain - distortion_weight * entries[j][2]['geometry_cost']) / unit
                context_features[j, 0] = np.tanh(new_gain / unit)
                context_features[j, 2] = np.tanh((new_gain - entries[j][2]['optimistic_gain_bits']) / unit)
                evaluations += 1
    if choices:
        # One batched differentiable calculation, rather than hundreds of
        # tiny GPU softmax/backward nodes. Masks describe the actual history.
        logits = torch.as_tensor(np.stack(prior_rows), device=actor.device, dtype=actor.dtype)
        features = torch.as_tensor(np.stack(context_rows), device=actor.device, dtype=actor.dtype)
        operation_index = torch.as_tensor(op_indices, device=actor.device, dtype=torch.long)
        interaction = (features * coefficients.index_select(0, operation_index)[None]).sum(-1)
        logits = (logits + actor_weight * actor[None] + interaction) / gradient_temperature
        logs = torch.log_softmax(logits, dim=1)
        chosen_tensor = torch.tensor(choices, device=actor.device, dtype=torch.long)
        log_prob = logs.gather(1, chosen_tensor[:, None]).mean()
        finite_logs = torch.where(torch.isfinite(logs), logs, torch.zeros_like(logs))
        entropy = -(logs.exp() * finite_logs).sum(1).mean()
    else:
        log_prob = actor.sum() * 0.
        entropy = actor.sum() * 0.
    audit = dict(schema='joint_conditional_occupancy_v1', selected_counts=selected_counts,
                 static_gain_sum=static_sum, joint_gain=gain_sum,
                 interaction_gain=gain_sum-static_sum, marginal_evaluations=evaluations,
                 prior_trust=trust,
                 random_diagnostic=random_only,
                 learned_context_coefficients=coefficients.detach().cpu().tolist(),
                 temperature=temperature, elapsed_sec=time.perf_counter()-started,
                 actual_candidate_encodes=0)
    return selected, removes, adds, true_adds, selected_counts, log_prob, entropy, audit
