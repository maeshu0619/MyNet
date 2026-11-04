"""Broad search, followed by a soft neighbourhood of self-measured plans.

Only the SparsePCGC learned True route uses this training behaviour. A past
complete plan is a conditioning input, never an isolated candidate label or
an action substituted after sampling. Evaluation has no access to this prior.
"""
import json
import math
import zlib

import torch


def phased_search_enabled(args):
    from .autonomous_exploration import learned_exploration_enabled
    return learned_exploration_enabled(args) and bool(
        getattr(args, 'test_route_phased_exploration', False))


def search_stage(args, training=True):
    if not training or not phased_search_enabled(args):
        return dict(enabled=False, progress=0., strength=0.)
    warmup = int(getattr(args, 'test_route_broad_search_episodes', 60))
    duration = int(getattr(args, 'test_route_local_search_episodes', 120))
    steps = max(int(getattr(args, '_fixed_episode_steps', 100)), 1)
    step = max(int(getattr(args, '_global_train_step', 0)), 0)
    u = min(max((step / steps - warmup) / duration, 0.), 1.)
    # Zero slope at both ends, and no hard argmax/zero-likelihood boundary.
    strength = .98 * u * u * (3. - 2. * u)
    return dict(enabled=True, progress=u, strength=strength,
                episode_position=step / steps + 1., broad_episodes=warmup,
                local_episodes=duration)


def policy_identity(args, exact):
    return dict(autonomous_input_sha256=str(exact.get(
        'input_sha256', exact.get('input_voxel_hash', ''))),
        autonomous_codec_setting=json.dumps(dict({key: exact.get(key) for key in (
            'setting_id', 'codec_mode', 'checkpoint_identifier', 'voxel_size',
            'pos_quantscale', 'scale_ae', 'scale_sr', 'scale_m')},
            native_context_file=str(getattr(args, '_autonomous_native_context_path', '')
                                    ).rsplit('/', 1)[-1]), sort_keys=True))


def policy_memory_key(args, identity):
    # Keep the existing runtime/checkpoint key byte-for-byte compatible.
    return '|'.join(str(v) for v in (
        'autonomous_joint_v2', getattr(args, '_current_input_file', ''),
        identity['autonomous_input_sha256'], identity.get('autonomous_codec_setting', ''),
        json.dumps({k: v for k, v in vars(args).items() if k.startswith(('geometry_', 'geom_'))
                    or k in ('loss_type', 'discrete_loss_mode', 'cp_tau_geom')},
                   sort_keys=True, default=str),
        getattr(args, 'sparsepcgc_scale_ae', 0), getattr(args, 'sparsepcgc_scale_sr', 2),
        getattr(args, 'sparsepcgc_scale_m', 8), getattr(args, 'sparsepcgc_voxel_size', 1.),
        getattr(args, 'sparsepcgc_pos_quantscale', 1)))


def prepare_neighbourhood(args, exact, memory, training, candidate_ids=None):
    stage = search_stage(args, training)
    result = dict(stage=stage, active=False, reason='broad_or_disabled')
    if stage['strength'] <= 0.:
        return result
    previous = (memory or {}).get(policy_memory_key(args, policy_identity(args, exact)))
    if not previous or not previous.get('best_action'):
        result['reason'] = 'no_measured_profitable_plan_keep_broad'
        return result
    try:
        best = json.loads(zlib.decompress(previous['best_action']))
        c, g = float(best['compression']), float(best['geometry'])
        cost = float(args.w_com) * c + float(args.cp_lambda_geom) * max(
            g - float(getattr(args, 'cp_tau_geom', 0.)), 0.)
        if not math.isfinite(c + g + cost) or c >= 0. or cost >= 0.:
            raise ValueError('not a finite profitable measurement')
        pools = (exact.get('operation_edit_units') or
                 exact.get('operation_candidate_shortlists') or exact.get('ranked_candidate_pools'))
        bins = (0.,) + tuple(args.sparsepcgc_algorithmic_amount_bin_values)
        action = best['action']
        indices = {}
        for op in ('Add', 'Prune', 'Adjust'):
            index = int(action['amount_indices'][op])
            fine = float(action['fine_samples'][op])
            if not 0 <= index < len(bins) or not math.isfinite(fine):
                raise ValueError('incompatible Amount action')
            ids_in_pool = (candidate_ids[op] if candidate_ids is not None else
                           (str(c['candidate_id']) for c in pools[op]))
            mapping = {name: i for i, name in enumerate(ids_in_pool)}
            ids = action['where_prefix_ids'][op]
            if len(set(ids)) != len(ids) or any(i not in mapping for i in ids):
                raise ValueError('measured plan unavailable in current candidate space')
            indices[op] = [mapping[i] for i in ids]
    except (KeyError, ValueError, TypeError, zlib.error, OverflowError):
        result['reason'] = 'incompatible_measurement_keep_broad'
        return result
    result.update(active=True, reason='past_same_frame_codec_profitable_plan',
                  best=best, prefix_indices=indices)
    return result


def neighbourhood_controls(controls, neighbourhood):
    if controls is None or not neighbourhood['active']:
        return controls
    strength = neighbourhood['stage']['strength']
    floors = dict(native_std=.05, where_temperature=.10, amount_temperature=.15,
                  fine_std=.03, uniform_fraction=.005)
    # Geometric interpolation shrinks large/small scales proportionately.
    # Explicitly disabled noise remains zero; gradients to learned scales
    # remain connected (2% exponent at the final, nonzero exploration stage).
    return {key: torch.where(value > 0.,
                ((1. - strength) * value.clamp_min(1e-20).log()
                 + strength * math.log(floors[key])).exp(), value)
            for key, value in controls.items()}


def amount_neighbourhood(logits, mean, bins, operation, neighbourhood, temperature=1.):
    if not neighbourhood or not neighbourhood['active']:
        return logits, mean
    strength = neighbourhood['stage']['strength']
    action = neighbourhood['best']['action']
    index = int(action['amount_indices'][operation])
    reference_fine = float(action['fine_samples'][operation])
    # A convex mean keeps at least 51% of the Fine head's derivative.
    behaviour_mean = (1. - .5 * strength) * mean + .5 * strength * reference_fine
    minimum = bins[bins > 0.].min() * .5
    log_ratio = bins.clamp_min(minimum).log() + torch.where(
        bins > 0., behaviour_mean, behaviour_mean * 0.)
    reference = bins[index].clamp_min(minimum).log() + (reference_fine if index else 0.)
    width = 1.5 * (1. - strength) + .35 * strength
    prior = -.5 * strength * ((log_ratio - reference) / width).square()
    # The prior is a log probability, not a score to amplify by 1/T again.
    return logits + temperature * prior, behaviour_mean


def where_neighbourhood(logits, operation, neighbourhood, temperature=1.):
    if not neighbourhood or not neighbourhood['active']:
        return logits
    indices = neighbourhood['prefix_indices'][operation]
    if not indices:
        return logits
    index = torch.as_tensor(indices, dtype=torch.long, device=logits.device)
    # Preserve current legality; no masked candidate is resurrected.
    bias = torch.zeros_like(logits)
    # Express locality as finite odds in the sampling distribution. A raw
    # constant bias / a small T would almost hard-lock the stored prefix.
    odds = math.log(max(len(logits) / len(indices), 1.)) + 3.
    bias[index] = temperature * odds * neighbourhood['stage']['strength']
    return logits + bias


def neighbourhood_audit(neighbourhood):
    result = dict(neighbourhood['stage'], active=neighbourhood['active'],
                  reason=neighbourhood['reason'])
    if neighbourhood['active']:
        best = neighbourhood['best']
        result.update(reference_compression=best['compression'], reference_geometry=best['geometry'],
            reference_prefix_counts={op: len(v) for op, v in neighbourhood['prefix_indices'].items()})
    return result
