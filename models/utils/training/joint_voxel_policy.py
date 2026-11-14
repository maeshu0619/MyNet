"""共同Voxel編集のsampling・実測cost学習・役割別gradient clip。

読む順序: sample_joint_amounts -> sample_where_order ->
where_prefix_log_prob -> compute_joint_voxel_policy_loss -> clip_joint_gradients。
L_joint_policyは圧縮方策、幾何方策、集合価値回帰、entropy、編集効率、
実測plan自己模倣の和。個別候補のActual寄与を測定した教師ではない。
runtime dictの旧den6キーは既存checkpoint・ログとの互換のため保持する。
"""
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F

OPERATIONS = ('Add', 'Prune', 'Adjust')


def sample_joint_amounts(selector, means, gate_logits, bins, maximum, phase,
                         training, fine_sigma=.30, uniform_fraction=0.,
                         temperature_override=None, fine_std_override=None, neighbourhood=None):
    """Three independent categorical+lognormal actions; no fixed total/share.

    The sampled latent action's exact likelihood is used even when a ratio
    reaches the safety cap. No gradient through a detached random sample is
    used in the likelihood. Evaluation and phase=0 use deterministic actions.
    """
    stochastic = bool(training and phase > 0.)
    temperature = (temperature_override if stochastic and temperature_override is not None
                   else max(2. * float(phase), .05))
    sigma = (fine_std_override if stochastic and fine_std_override is not None else
             max(float(fine_sigma) * float(phase), .02)
             if stochastic and fine_sigma > 0. else 0.)
    basis = bins / float(maximum)
    ratios, log_probs, indices, logits_by_op = {}, {}, {}, {}
    entropies = []
    samples = {}
    epsilon = (uniform_fraction.clamp(0., 1.) if torch.is_tensor(uniform_fraction)
               else min(max(float(uniform_fraction), 0.), 1.)) if stochastic else 0.
    for i, name in enumerate(OPERATIONS):
        logits = selector + means[name] * basis
        edit_mask = (bins > 0.).to(logits)
        logits = logits + gate_logits[i] * edit_mask
        behaviour_logits, behaviour_mean = logits, means[name]
        if stochastic and neighbourhood and neighbourhood['active']:
            from .autonomous_local_search import amount_neighbourhood
            behaviour_logits, behaviour_mean = amount_neighbourhood(
                logits, means[name], bins, name, neighbourhood, temperature)
        log_probs_all = F.log_softmax(behaviour_logits / temperature if stochastic else logits, dim=0)
        if stochastic:
            uniform = torch.rand_like(logits).clamp_(1e-7, 1. - 1e-7)
            index = torch.argmax(behaviour_logits - temperature * torch.log(-torch.log(uniform)))
            broad = epsilon > 0. and bool(torch.rand((), device=logits.device) < epsilon)
            if broad:
                index = torch.randint(len(bins), (), device=logits.device)
            sample_mean = behaviour_mean * 0. if broad and sigma > 0. else behaviour_mean
            sample = (sample_mean + sigma * torch.randn_like(means[name])
                      if sigma > 0. else means[name])
            # Evaluate the detached sampled action, not a reparameterized
            # sample whose mean derivative would cancel the score function.
            fine_lp = (torch.distributions.Normal(behaviour_mean, sigma).log_prob(sample.detach())
                       if sigma > 0. else means[name] * 0.)
            lp = log_probs_all[index] + torch.where(bins[index] > 0., fine_lp, fine_lp * 0.)
            if epsilon > 0.:
                broad_fine_lp = (torch.distributions.Normal(means[name]*0., sigma).log_prob(sample.detach())
                                 if sigma > 0. else fine_lp*0.)
                broad_lp = logits.sum()*0.-math.log(len(bins)) + torch.where(
                    bins[index] > 0., broad_fine_lp, broad_fine_lp*0.)
                if epsilon == 1.:
                    lp = broad_lp
                else:
                    log_policy = torch.log1p(-epsilon) if torch.is_tensor(epsilon) else math.log1p(-epsilon)
                    log_broad = epsilon.log() if torch.is_tensor(epsilon) else math.log(epsilon)
                    lp = torch.logaddexp(lp + log_policy, broad_lp + log_broad)
        else:
            index = torch.argmax(logits)
            sample = means[name]
            lp = logits.sum() * 0.
        ratios[name] = (bins[index] * sample.exp()).clamp(max=float(maximum))
        samples[name] = sample
        log_probs[name], indices[name], logits_by_op[name] = lp, index, logits
        probabilities = (1.-epsilon)*log_probs_all.exp()+epsilon/len(bins)
        entropies.append(-(probabilities * probabilities.clamp_min(1e-20).log()).sum())
    return dict(ratios=ratios, log_probs=log_probs, indices=indices,
                logits=logits_by_op, entropy=torch.stack(entropies).sum(),
                samples=samples, means=means, epsilon=epsilon,
                temperature=temperature if stochastic else 0., stochastic=stochastic)


def sample_where_order(logits, phase, training, temperature_override=None):
    """Gumbel top-k gives a Plackett-Luce permutation on the legal pool."""
    valid = torch.nonzero(torch.isfinite(logits), as_tuple=False).reshape(-1)
    score = logits.index_select(0, valid)
    if training and phase > 0.:
        temperature = temperature_override if temperature_override is not None else max(float(phase), .05)
        uniform = torch.rand_like(score).clamp_(1e-7, 1. - 1e-7)
        score = score - temperature * torch.log(-torch.log(uniform))
    return valid[torch.argsort(score, descending=True)]


def where_prefix_log_prob(logits, order, attempted, phase, training, temperature_override=None):
    """Exact log probability of the attempted prefix, including rejections.

    Conflict resolution is a deterministic function of sampled permutations.
    Rejected candidates still belong to the latent sampled prefix, therefore
    they must remain in its likelihood. Unvisited suffixes are marginalized.
    """
    if not training or phase <= 0. or attempted <= 0:
        return logits[torch.isfinite(logits)].sum() * 0.
    temperature = temperature_override if temperature_override is not None else max(float(phase), .05)
    weights = logits.index_select(0, order) / temperature
    denominators = torch.logcumsumexp(weights.flip(0), dim=0).flip(0)
    return (weights[:attempted] - denominators[:attempted]).sum()


def compute_joint_voxel_policy_loss(model, compression, geometry):
    """REINFORCE with separate real rate/geometry credit and a frame EMA.

    A zero control variate on first sight is NOT an encoded comparison. Its
    action independence makes the score estimator valid without inventing a
    measured teacher. Later visits use only that exact input/codec's EMA.
    """
    args, state = model.args, model.last_actuator_voxel_state
    debug = state['ana_den6_exact_residual_plan_debug']
    lp = debug['_joint_policy_log_prob']
    zero = lp * 0.
    actual_compression_percent = compression.detach().float().mean()
    geometry_distortion = geometry.detach().float().mean() if torch.is_tensor(geometry) else actual_compression_percent * 0.
    if not bool(torch.isfinite(actual_compression_percent) & torch.isfinite(geometry_distortion)):
        raise ValueError('Non-finite measured joint Actual/geometry target')
    geometry_penalty = torch.relu(geometry_distortion - float(getattr(args, 'cp_tau_geom', 0.)))
    compression_loss_weight, geometry_loss_weight = float(getattr(args, 'w_com', 1.)), float(getattr(args, 'cp_lambda_geom', 1.))
    from .autonomous_local_search import policy_memory_key
    key = policy_memory_key(args, debug)
    memory = getattr(model, '_den6_online_efficiency_baseline', None)
    if memory is None:
        memory = OrderedDict()
        model._den6_online_efficiency_baseline = memory
    previous = memory.get(key)
    bc, bg = ((previous['ema_c'], previous['ema_g']) if previous is not None else (0., 0.))
    global_bc, global_bg = bc, bg
    from .autonomous_exploration import learned_exploration_enabled
    conditional_baseline = learned_exploration_enabled(args)
    family = str(debug.get('joint_credit_family', 'joint'))
    family_history = dict(previous.get('credit_family_ema', {})) if previous else {}
    family_previous = family_history.get(family)
    if conditional_baseline:
        # The clean-Where Amount trial and sampled-Where trial have very
        # different cost distributions. The independent family coin is drawn
        # before the credited actions: its own past EMA is a valid baseline,
        # and avoids mixing expensive Where trials into Amount advantages.
        bc, bg = ((family_previous['c'], family_previous['g'])
                  if family_previous is not None else (0., 0.))
    policy_backward_scale = max(float(getattr(args, 'heuristic_guidance_online_policy_backward_scale', 1.)), 1.)
    policy_weight = max(float(getattr(args, 'heuristic_guidance_online_policy_weight', 1.)), 0.)
    # train.py's pre-existing backward-only scale is canceled exactly once.
    L_compression_policy = -policy_weight * (compression_loss_weight * (float(bc) - actual_compression_percent)) * lp / policy_backward_scale
    L_geometry_policy = -policy_weight * (geometry_loss_weight * (float(bg) - geometry_penalty)) * lp / policy_backward_scale
    alpha = float(getattr(args, 'heuristic_guidance_online_reward_ema', .1))
    from .autonomous_conditional_baseline import conditional_where_baseline
    (where_bc, where_bg), where_control = conditional_where_baseline(
        debug, previous, args, actual_compression_percent, geometry_penalty, (bc, bg), alpha)
    if where_control['enabled']:
        # Keep the Native Normal score's family baseline unchanged. Only the
        # conditional permutation score gets the action-independent baseline.
        where_lp = debug['_joint_where_position_log_prob']
        L_compression_policy = L_compression_policy + policy_weight*compression_loss_weight*(bc-where_bc)*where_lp/policy_backward_scale
        L_geometry_policy = L_geometry_policy + policy_weight*geometry_loss_weight*(bg-where_bg)*where_lp/policy_backward_scale
    ratio = float(debug['_actor_critic_selected_operation_ratio'].detach().cpu())
    L_edit_efficiency = zero
    guarded = False
    if previous is not None:
        guarded = (float(actual_compression_percent) <= min(0., previous['compression'] + float(getattr(
            args, 'heuristic_guidance_online_efficiency_compression_tolerance', .02)))
            and float(geometry_distortion) <= previous['geometry'] * (1. + float(getattr(
                args, 'heuristic_guidance_online_efficiency_geometry_relative_tolerance', .005))) + 1e-12)
        if guarded:
            advantage = min(max((previous['operation_ratio'] - ratio)
                                / max(previous['operation_ratio'], 1e-8), -.25), .25)
            L_edit_efficiency = -float(getattr(args, 'heuristic_guidance_online_efficiency_policy_weight', .25)) * advantage * lp / policy_backward_scale
    memory[key] = dict(compression=float(actual_compression_percent), geometry=float(geometry_distortion), operation_ratio=ratio,
                       ema_c=(1. - alpha) * global_bc + alpha * float(actual_compression_percent) if previous else float(actual_compression_percent),
                       ema_g=(1. - alpha) * global_bg + alpha * float(geometry_penalty) if previous else float(geometry_penalty))
    if conditional_baseline:
        family_history[family] = dict(
            c=(1.-alpha)*bc+alpha*float(actual_compression_percent) if family_previous else float(actual_compression_percent),
            g=(1.-alpha)*bg+alpha*float(geometry_penalty) if family_previous else float(geometry_penalty))
        memory[key]['credit_family_ema'] = family_history
    if where_control['history'] is not None:
        memory[key]['where_native_condition_ema'] = where_control['history']
    elif previous and previous.get('where_native_condition_ema'):
        # Clean Amount trials must not erase the separate Where history.
        memory[key]['where_native_condition_ema'] = previous['where_native_condition_ema']
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
    predicted_plan_cost = model.actuator.den6_plan_critic(critic_feature.unsqueeze(0))[0, 0]
    measured_rd_cost = compression_loss_weight * actual_compression_percent + geometry_loss_weight * geometry_penalty
    from .autonomous_actual_replay import measured_plan_replay, bound_replay_memory
    L_measured_plan_replay, replacement, replay_debug = measured_plan_replay(debug, previous, actual_compression_percent, geometry_distortion, args)
    if previous is not None and previous.get('best_action'):
        memory[key]['best_action'] = previous['best_action']
    if replacement is not None:
        memory[key]['best_action'] = replacement
    replay_bytes = bound_replay_memory(memory, max(int(getattr(
        args, 'test_route_actual_plan_replay_memory_mb', 32)), 0)*1024**2)
    L_measured_plan_replay = L_measured_plan_replay/policy_backward_scale*policy_weight
    L_plan_value = F.smooth_l1_loss(predicted_plan_cost, measured_rd_cost, beta=1.) / policy_backward_scale * float(getattr(
        args, 'heuristic_guidance_online_plan_critic_weight', .10))
    phase = float(debug['exploration_multiplier'])
    L_policy_entropy = -float(getattr(args, 'heuristic_guidance_online_entropy_weight', .001)) * phase * debug['_joint_entropy'] / policy_backward_scale
    L_joint_policy = L_compression_policy + L_geometry_policy + L_edit_efficiency + L_plan_value + L_policy_entropy + L_measured_plan_replay
    model._den6_policy_source_losses = dict(
        actual_compression_policy=L_compression_policy + L_plan_value,
        geometry=L_geometry_policy, proxy=zero, efficiency=L_edit_efficiency, entropy=L_policy_entropy,
        actual_joint_replay=L_measured_plan_replay)
    model.last_discrete_policy_debug = dict(
        joint_policy=True, actual_group_pair_available=False,
        joint_frame_baseline_available=previous is not None,
        objective_baseline_source=(('same_frame_credit_family_ema' if family_previous else
            'zero_family_control_variate_not_measurement') if conditional_baseline else
            'same_frame_ema' if previous else 'zero_control_variate_not_measurement'),
        objective_baseline_factorized_group='all', current_compression_value=float(actual_compression_percent),
        current_geometry_value=float(geometry_distortion), actor_rd_target=float(measured_rd_cost),
        efficiency_guard_passed=guarded, efficiency_operation_ratio=ratio,
        baseline=float(compression_loss_weight * bc + geometry_loss_weight * bg), advantage=float(compression_loss_weight * (bc - actual_compression_percent) + geometry_loss_weight * (bg - geometry_penalty)),
        policy_loss=float(L_joint_policy.detach()), actual_teacher_scope='complete_joint_plan_cost',
        actual_plan_replay=replay_debug, actual_plan_replay_bytes=replay_bytes,
        where_control_variate={k:v for k,v in where_control.items() if k!='history'},
        where_baseline=float(compression_loss_weight*where_bc+geometry_loss_weight*where_bg),
        actual_plan_replay_loss=float(L_measured_plan_replay.detach()))
    return L_joint_policy.to(compression.dtype)


def _host_scalars(values):
    """Transfer diagnostics once per device, preserving scalar/order arithmetic."""
    result, devices = [None] * len(values), {}
    for index, value in enumerate(values):
        devices.setdefault(value.device, []).append((index, value))
    for entries in devices.values():
        numbers = torch.stack([value for _, value in entries]).detach().cpu().tolist()
        for (index, _), number in zip(entries, numbers):
            result[index] = number
    return result


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
    total_before = math.sqrt(sum(_host_scalars([
        p.grad.detach().double().square().sum() for p in parameters])))
    seen, groups = set(), {}
    from .autonomous_exploration import exploration_parameter
    exploration = exploration_parameter(model)
    exploration_report = None
    if exploration is not None and exploration.grad is not None:
        # These five coordinates control different sampling distributions.
        # Bound each scalar independently so a long Where prefix cannot
        # suppress the Native/Fine scale updates. No gradient is amplified.
        seen.add(id(exploration))
        control_bound = limit if limit > 0. else float(max_norm)
        control_before = exploration.grad.detach().clone()
        if control_bound > 0.:
            exploration.grad.clamp_(min=-control_bound, max=control_bound)
        exploration_report = dict(before=control_before.cpu().tolist(),
            after=exploration.grad.detach().cpu().tolist(), bound_per_coordinate=control_bound)
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
    measurements = []
    for role, (params, bound) in groups.items():
        if not params:
            continue
        before = torch.nn.utils.clip_grad_norm_(params, bound) if bound > 0. else params[0].grad.new_zeros(())
        measurements.append((role, params, bound, before))
    # The clipping operations and their order are unchanged. Only reporting
    # transfers are batched; double per-parameter sums and Python summation
    # order match the original diagnostics.
    before_values = _host_scalars([value for _, _, _, value in measurements])
    after_values = iter(_host_scalars([
        p.grad.detach().double().square().sum()
        for _, params, _, _ in measurements for p in params]))
    report, applied = {}, bool(exploration_report and
        exploration_report['before'] != exploration_report['after'])
    for (role, params, bound, _), before in zip(measurements, before_values):
        after = math.sqrt(sum(next(after_values) for _ in params))
        report[role] = dict(before=before, after=after, bound=bound)
        applied |= before > bound > 0.
    return dict(train_grad_clip_max_norm=float(max_norm),
                train_grad_clip_parameter_count=len(parameters),
                train_grad_clip_applied=applied,
                train_grad_total_norm_before_clip=total_before,
                autonomous_gradient_groups=report,
                autonomous_exploration_clip=exploration_report)

# Compatibility names: serialized files and external scripts may still use these.
joint_actual_policy_loss = compute_joint_voxel_policy_loss
