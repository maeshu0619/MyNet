"""Joint Amount/Where policy for test_route=True only.

Scores are trained from measured COMPLETE plans. A plan's gain is never
presented as the isolated Actual gain of each selected candidate.
"""
import math
import json
from collections import OrderedDict

import torch
import torch.nn.functional as F

OPERATIONS = ('Add', 'Prune', 'Adjust')


def sample_joint_amounts(selector, means, gate_logits, bins, maximum, phase,
                         training, fine_sigma=.30):
    """Three independent categorical+lognormal actions; no fixed total/share.

    The sampled latent action's exact likelihood is used even when a ratio
    reaches the safety cap. No gradient through a detached random sample is
    used in the likelihood. Evaluation and phase=0 use deterministic actions.
    """
    stochastic = bool(training and phase > 0.)
    temperature = max(2. * float(phase), .05)
    sigma = (max(float(fine_sigma) * float(phase), .02)
             if stochastic and fine_sigma > 0. else 0.)
    basis = bins / float(maximum)
    ratios, log_probs, indices, logits_by_op = {}, {}, {}, {}
    entropies = []
    for i, name in enumerate(OPERATIONS):
        logits = selector + means[name] * basis
        edit_mask = (bins > 0.).to(logits)
        logits = logits + gate_logits[i] * edit_mask
        log_probs_all = F.log_softmax(logits / temperature if stochastic else logits, dim=0)
        if stochastic:
            uniform = torch.rand_like(logits).clamp_(1e-7, 1. - 1e-7)
            index = torch.argmax(logits - temperature * torch.log(-torch.log(uniform)))
            sample = (means[name] + sigma * torch.randn_like(means[name])
                      if sigma > 0. else means[name])
            # Evaluate the detached sampled action, not a reparameterized
            # sample whose mean derivative would cancel the score function.
            fine_lp = (torch.distributions.Normal(means[name], sigma).log_prob(sample.detach())
                       if sigma > 0. else means[name] * 0.)
            lp = log_probs_all[index] + torch.where(bins[index] > 0., fine_lp, fine_lp * 0.)
        else:
            index = torch.argmax(logits)
            sample = means[name]
            lp = logits.sum() * 0.
        ratios[name] = (bins[index] * sample.exp()).clamp(max=float(maximum))
        log_probs[name], indices[name], logits_by_op[name] = lp, index, logits
        entropies.append(-(log_probs_all.exp() * log_probs_all).sum())
    return dict(ratios=ratios, log_probs=log_probs, indices=indices,
                logits=logits_by_op, entropy=torch.stack(entropies).sum(),
                temperature=temperature if stochastic else 0., stochastic=stochastic)


def sample_where_order(logits, phase, training):
    """Gumbel top-k gives a Plackett-Luce permutation on the legal pool."""
    valid = torch.nonzero(torch.isfinite(logits), as_tuple=False).reshape(-1)
    score = logits.index_select(0, valid)
    if training and phase > 0.:
        temperature = max(float(phase), .05)
        uniform = torch.rand_like(score).clamp_(1e-7, 1. - 1e-7)
        score = score - temperature * torch.log(-torch.log(uniform))
    return valid[torch.argsort(score, descending=True)]


def where_prefix_log_prob(logits, order, attempted, phase, training):
    """Exact log probability of the attempted prefix, including rejections.

    Conflict resolution is a deterministic function of sampled permutations.
    Rejected candidates still belong to the latent sampled prefix, therefore
    they must remain in its likelihood. Unvisited suffixes are marginalized.
    """
    if not training or phase <= 0. or attempted <= 0:
        return logits[torch.isfinite(logits)].sum() * 0.
    weights = logits.index_select(0, order) / max(float(phase), .05)
    denominators = torch.logcumsumexp(weights.flip(0), dim=0).flip(0)
    return (weights[:attempted] - denominators[:attempted]).sum()


def joint_actual_policy_loss(model, compression, geometry):
    """REINFORCE with separate real rate/geometry credit and a frame EMA.

    A zero control variate on first sight is NOT an encoded comparison. Its
    action independence makes the score estimator valid without inventing a
    measured teacher. Later visits use only that exact input/codec's EMA.
    """
    args, state = model.args, model.last_actuator_voxel_state
    debug = state['ana_den6_exact_residual_plan_debug']
    lp = debug['_joint_policy_log_prob']
    zero = lp * 0.
    c = compression.detach().float().mean()
    g = geometry.detach().float().mean() if torch.is_tensor(geometry) else c * 0.
    if not bool(torch.isfinite(c) & torch.isfinite(g)):
        raise ValueError('Non-finite measured joint Actual/geometry target')
    penalty = torch.relu(g - float(getattr(args, 'cp_tau_geom', 0.)))
    wc, wg = float(getattr(args, 'w_com', 1.)), float(getattr(args, 'cp_lambda_geom', 1.))
    key = '|'.join(str(v) for v in (
        'autonomous_joint_v2', getattr(args, '_current_input_file', ''),
        debug['autonomous_input_sha256'], debug.get('autonomous_codec_setting', ''),
        json.dumps({k: v for k, v in vars(args).items() if k.startswith(('geometry_', 'geom_'))
                    or k in ('loss_type', 'discrete_loss_mode', 'cp_tau_geom')}, sort_keys=True, default=str),
        getattr(args, 'sparsepcgc_scale_ae', 0),
        getattr(args, 'sparsepcgc_scale_sr', 2), getattr(args, 'sparsepcgc_scale_m', 8),
        getattr(args, 'sparsepcgc_voxel_size', 1.), getattr(args, 'sparsepcgc_pos_quantscale', 1)))
    memory = getattr(model, '_den6_online_efficiency_baseline', None)
    if memory is None:
        memory = OrderedDict()
        model._den6_online_efficiency_baseline = memory
    previous = memory.get(key)
    bc, bg = ((previous['ema_c'], previous['ema_g']) if previous is not None else (0., 0.))
    scale = max(float(getattr(args, 'heuristic_guidance_online_policy_backward_scale', 1.)), 1.)
    policy_weight = max(float(getattr(args, 'heuristic_guidance_online_policy_weight', 1.)), 0.)
    # train.py's pre-existing backward-only scale is canceled exactly once.
    compression_policy = -policy_weight * (wc * (float(bc) - c)) * lp / scale
    geometry_policy = -policy_weight * (wg * (float(bg) - penalty)) * lp / scale
    ratio = float(debug['_actor_critic_selected_operation_ratio'].detach().cpu())
    efficiency = zero
    guarded = False
    if previous is not None:
        guarded = (float(c) <= min(0., previous['compression'] + float(getattr(
            args, 'heuristic_guidance_online_efficiency_compression_tolerance', .02)))
            and float(g) <= previous['geometry'] * (1. + float(getattr(
                args, 'heuristic_guidance_online_efficiency_geometry_relative_tolerance', .005))) + 1e-12)
        if guarded:
            advantage = min(max((previous['operation_ratio'] - ratio)
                                / max(previous['operation_ratio'], 1e-8), -.25), .25)
            efficiency = -float(getattr(args, 'heuristic_guidance_online_efficiency_policy_weight', .25)) * advantage * lp / scale
    alpha = float(getattr(args, 'heuristic_guidance_online_reward_ema', .1))
    memory[key] = dict(compression=float(c), geometry=float(g), operation_ratio=ratio,
                       ema_c=(1. - alpha) * bc + alpha * float(c) if previous else float(c),
                       ema_g=(1. - alpha) * bg + alpha * float(penalty) if previous else float(penalty))
    memory.move_to_end(key)
    while len(memory) > 4096:
        memory.popitem(last=False)
    # Train a set-level value. Only candidate-Q summary slots retain their
    # graph; action logits/features are detached to avoid Critic leakage into
    # the Actor/Amount heads. Individual candidate Qs are latent set features,
    # not independently measured candidate gains.
    feature = debug['_actor_critic_plan_feature']
    mask = feature.new_zeros(feature.shape)
    mask[[0, 1, 4, 5, 8, 9]] = 1.
    critic_feature = feature.detach() + mask * (feature - feature.detach())
    prediction = model.actuator.den6_plan_critic(critic_feature.unsqueeze(0))[0, 0]
    rd = wc * c + wg * penalty
    critic = F.smooth_l1_loss(prediction, rd, beta=1.) / scale * float(getattr(
        args, 'heuristic_guidance_online_plan_critic_weight', .10))
    phase = float(debug['exploration_multiplier'])
    entropy = -float(getattr(args, 'heuristic_guidance_online_entropy_weight', .001)) * phase * debug['_joint_entropy'] / scale
    loss = compression_policy + geometry_policy + efficiency + critic + entropy
    model._den6_policy_source_losses = dict(
        actual_compression_policy=compression_policy + critic,
        geometry=geometry_policy, proxy=zero, efficiency=efficiency, entropy=entropy)
    model.last_discrete_policy_debug = dict(
        joint_policy=True, actual_group_pair_available=False,
        joint_frame_baseline_available=previous is not None,
        objective_baseline_source='same_frame_ema' if previous else 'zero_control_variate_not_measurement',
        objective_baseline_factorized_group='all', current_compression_value=float(c),
        current_geometry_value=float(g), actor_rd_target=float(rd),
        efficiency_guard_passed=guarded, efficiency_operation_ratio=ratio,
        baseline=float(wc * bc + wg * bg), advantage=float(wc * (bc - c) + wg * (bg - penalty)),
        policy_loss=float(loss.detach()), actual_teacher_scope='complete_joint_plan_cost')
    return loss.to(compression.dtype)


def clip_joint_gradients(model, max_norm):
    """Clip disjoint roles without letting one role shrink all other roles.

    This only lowers norms above the pre-existing decision clip bound. Small
    and zero gradients remain unchanged; no target-norm normalization is used.
    The backbone retains the original global max_norm bound separately.
    """
    base = model.module if hasattr(model, 'module') else model
    actuator = base.actuator
    limit = max(float(getattr(base.args, 'repair_online_decision_grad_max_norm', 1.)), 0.)
    roles = dict(
        where=('drop_head', 'add_head', 'add_voxel_head', 'move_voxel_head', 'subtree_move_source_head'),
        actor_proposal=('den6_candidate_actor', 'den6_candidate_proposal'),
        amount_fine=('add_amount_head', 'drop_amount_head', 'move_amount_head'),
        amount_coarse=('algorithmic_amount_selector_head',), gate=('operation_gate_head',),
        critic=('den6_candidate_critic', 'den6_candidate_proxy_critic', 'den6_plan_critic'))
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    total_before = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters))
    seen, groups = set(), {}
    for role, names in roles.items():
        params = []
        for name in names:
            module = getattr(actuator, name, None)
            if module is None:
                continue
            for p in module.parameters():
                if p.grad is not None and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        groups[role] = (params, limit if limit > 0. else float(max_norm))
    groups['backbone'] = ([p for p in parameters if id(p) not in seen], float(max_norm))
    report, applied = {}, False
    for role, (params, bound) in groups.items():
        if not params:
            continue
        before = float(torch.nn.utils.clip_grad_norm_(params, bound)) if bound > 0. else 0.
        after = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in params))
        report[role] = dict(before=before, after=after, bound=bound)
        applied |= before > bound > 0.
    return dict(train_grad_clip_max_norm=float(max_norm),
                train_grad_clip_parameter_count=len(parameters),
                train_grad_clip_applied=applied,
                train_grad_total_norm_before_clip=total_before,
                autonomous_gradient_groups=report)
