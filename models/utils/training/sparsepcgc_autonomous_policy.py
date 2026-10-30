"""Dataset-independent search controls, separate from the den6 comparator.

This is an opt-in policy space. Increasing that space is not a performance
certificate; promotion requires matched Actual/geometry validation.
"""
import math
import hashlib
import json
from pathlib import Path


def autonomous_search_enabled(args):
    codec = str(getattr(args, 'compress', '')).lower().replace('-', '').replace('_', '')
    return (codec == 'sparsepcgc'
            and str(getattr(args, 'heuristic_guidance_mode', '')).lower() == 'ana_den6_online'
            and bool(getattr(args, 'test_route', False)))


def autonomous_amount_bins(max_ratio):
    """Positive search grid plus the existing no-op, without dataset profiles."""
    maximum = float(max_ratio)
    if not math.isfinite(maximum) or not 0.0 < maximum <= .10:
        raise ValueError('test_route_max_operation_ratio must be in (0, 0.10]')
    return [maximum * (value / .03) for value in (
        .0001, .0003, .0006, .001, .002, .003, .005, .007, .009,
        .01, .012, .015, .02, .025, .03)]


def configure_autonomous_search(args):
    if not bool(getattr(args, 'test_route', False)):
        return args
    if not autonomous_search_enabled(args):
        raise ValueError('test_route requires SparsePCGC ana_den6_online')
    for name in ('test_route_amount_uniform_fraction', 'test_route_noop_recovery_probability'):
        value = float(getattr(args, name, 0.))
        if not math.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError(name+' must be finite in [0, 1]')
    noise = float(getattr(args, 'test_route_structured_where_noise', 0.))
    if not math.isfinite(noise) or noise < 0.:
        raise ValueError('test_route_structured_where_noise must be finite and nonnegative')
    maximum = float(getattr(args, 'test_route_max_operation_ratio', .03))
    autonomous_amount_bins(maximum)  # validate the per-operation safety bound
    raw_bins = str(getattr(args, 'test_route_amount_bins', '')).strip()
    values = ([float(v) for v in raw_bins.split(',')] if raw_bins
              else autonomous_amount_bins(maximum))
    if (not values or len(set(values)) != len(values)
            or any(not math.isfinite(v) or not 0 < v <= maximum for v in values)):
        raise ValueError('test_route_amount_bins requires distinct positive ratios <= the per-operation maximum')
    values = sorted(values)
    args.test_route_max_operation_ratio = maximum
    args.heuristic_guidance_online_max_total_ratio = 3. * maximum
    args.heuristic_guidance_online_max_changed_ratio = 4. * maximum
    args.heuristic_guidance_online_amount_bins = ','.join(str(value) for value in values)
    args.sparsepcgc_algorithmic_amount_bin_values = tuple(values)
    args.sparsepcgc_algorithmic_amount_bins = args.heuristic_guidance_online_amount_bins
    args.sparsepcgc_amount_exploration = True
    args.sparsepcgc_amount_residual_span = math.log(32.)
    args.heuristic_guidance_online_amount_logit_std_cap = 2.
    args.heuristic_guidance_exact_anchor_steps = 0
    args.heuristic_guidance_anchor_steps = 0
    args.heuristic_guidance_online_actor_use_rank_feature = False
    args.heuristic_guidance_online_proposal_mode = 'fast_unordered'
    args.heuristic_guidance_network_only_inference = False
    args.heuristic_guidance_require_exact_single_plan_teacher = False
    args.heuristic_guidance_auto_build_exact_single_plan_teacher = False
    args.heuristic_guidance_fixed_symbol_features = False
    args.heuristic_guidance_where_weight = 0.
    args.heuristic_guidance_final_where_weight = 0.
    args.leaf_pattern_add_target_direction_weight = 0.
    args.leaf_pattern_move_target_direction_weight = 0.
    # Neutral head/reference initialization from the search bound, not den6.
    args.repair_init_add_ratio = args.repair_init_drop_ratio = maximum / 2.
    args.repair_init_move_ratio = maximum / 2.
    args.repair_operation_gate_init_drop = .5
    args.repair_operation_gate_init_add = .5
    args.repair_operation_gate_init_move = .5
    args.max_add_ratio = args.max_drop_ratio = maximum
    args.max_move_ratio = maximum
    # All legal pool entries participate. Proposal is a trainable logit in
    # the joint Where distribution, not a hard shortlist bottleneck.
    args.heuristic_guidance_online_learned_proposal = False
    args.heuristic_guidance_online_factorized_exploration = False
    args.heuristic_guidance_online_post_window_exploration_floor = 0.
    # Do not use the spatial pool's uncalibrated degree proxy as an Actor
    # improvement label. Actual plan contrasts remain the policy teacher.
    args.heuristic_guidance_online_actor_critic_distill_weight = 0.
    return args


def autonomous_pool_phase(args):
    if not bool(getattr(args, '_autonomous_policy_training', True)):
        return 0
    # Keep a reservoir for two complete passes, so same-frame counterfactuals
    # can finish before expanding coverage to another set of source voxels.
    episode_steps = max(int(getattr(args, '_fixed_episode_steps', 100)), 1)
    step = max(int(getattr(args, '_global_train_step', 0)), 0)
    if str(getattr(args, 'repair_policy_exploration_mode', 'annealed')) == 'annealed':
        total = int(getattr(args, '_exploration_schedule_steps_estimate',
                            getattr(args, '_total_train_steps_estimate', 0)))
        if total > 0:
            fraction = min(max(float(getattr(args, 'repair_exploration_fraction', .25)), 0.), 1.)
            # Reservoir rotation is exploration too. Do not keep changing
            # the action space after all explicit sampling noise has ended.
            step = min(step, max(int(math.ceil(total * fraction)) - 1, 0))
    # Codec-visible alternatives cover the full GT coded context already.
    # Keep the auxiliary spatial reservoir long enough to learn/replay real
    # same-frame outcomes instead of invalidating it after only two visits.
    from ..loss.autonomous_codec_context import codec_context_enabled
    passes = max(int(getattr(args,'test_route_context_reservoir_passes',8)),2) if codec_context_enabled(args) else 2
    return step // (passes * episode_steps)


def align_autonomous_context(context, node_coords):
    """Use the neural node row order for recomputed occupancy observations.

    This mode uses the minimal full-cloud context, whose metadata describes
    quantization rather than cached per-point predictions. Reject unsupported
    point-key metadata instead of silently permuting only part of a subtree.
    """
    if not isinstance(context, dict):
        return context
    if context.get('global_morton_keys') is not None:
        raise ValueError('autonomous alignment requires the minimal full-cloud context')
    result = dict(context)
    for key in ('global_voxel_coords', 'full_global_voxel_coords', 'full_occupied_voxel_coords'):
        result[key] = node_coords
    return result


def proposal_coverage_index(visits, key, *, training, autonomous):
    """Exploratory coverage advances in training, never in autonomous eval."""
    if autonomous and not training:
        return 0
    index = int(visits.get(key, 0))
    visits[key] = index + 1
    visits.move_to_end(key)
    return index


def validation_promotion(candidate, reference):
    """Require per-frame non-regression of BOTH actual rate and geometry.

    Inputs are measured rows, never Proxy predictions or filtered winners.
    Missing/extra/duplicate frames, codec changes and non-finite values fail
    closed. This only certifies the measured set, not unseen data.
    """
    def index(rows):
        values = {}
        for row in rows:
            key = (row['frame'], row['codec_setting'], row.get('input_sha256', ''),
                   row.get('geometry_setting', ''))
            if key in values:
                raise ValueError('duplicate validation frame/codec')
            c, g = float(row['actual_percent']), float(row['geometry'])
            if not math.isfinite(c) or not math.isfinite(g):
                raise ValueError('non-finite measurement')
            values[key] = (c, g)
        return values

    try:
        trial, anchor = index(candidate), index(reference)
    except (KeyError, TypeError, ValueError) as error:
        return {'eligible': False, 'reason': str(error)}
    if not anchor or trial.keys() != anchor.keys():
        return {'eligible': False, 'reason': 'validation frame/codec mismatch or empty set'}
    failures = [dict(frame=key[0], codec_setting=key[1],
                     delta_actual=trial[key][0]-anchor[key][0],
                     delta_geometry=trial[key][1]-anchor[key][1])
                for key in anchor
                if trial[key][0] > anchor[key][0] or trial[key][1] > anchor[key][1]]
    return {'eligible': not failures, 'reason': 'non-regression on measured frames' if not failures
            else 'Actual rate or geometry regressed', 'failures': failures,
            'frames': len(anchor)}


def validation_fingerprint(args, loss, frame):
    """Fingerprint the input, native codec and geometry definition."""
    path = Path(frame).resolve()
    from ..pointcloud.ana_den6_online import _sha256_file
    geometry_options = {name: value for name, value in vars(args).items()
                        if name.startswith(('geometry_', 'geom_')) or name in (
                            'loss_type', 'discrete_loss_mode', 'minimal_loss_objective')}
    geometry_file = Path(__file__).resolve().parents[1] / 'loss' / 'geometry.py'
    geometry_options['implementation_sha256'] = hashlib.sha256(geometry_file.read_bytes()).hexdigest()
    return dict(frame=str(path), input_sha256=_sha256_file(path),
                codec_setting=loss._get_actual_encoder(args)._codec_cache_fingerprint(),
                geometry_setting=hashlib.sha256(json.dumps(geometry_options, sort_keys=True,
                    default=str).encode()).hexdigest())


def validation_measurement(args, loss, frame, actual_percent, geometry):
    return dict(validation_fingerprint(args, loss, frame), actual_percent=float(actual_percent),
                geometry=float(geometry))


def autonomous_checkpoint_status(args):
    """No unverified autonomous policy may replace the released best model."""
    reference_path = str(getattr(args, 'sparsepcgc_autonomous_reference_json', '') or '')
    if not reference_path:
        return dict(eligible=False, reason='autonomous_reference_not_provided')
    try:
        reference = json.loads(Path(reference_path).read_text())['measurements']
        candidate = getattr(args, '_autonomous_validation_measurements', [])
        expected = int(getattr(args, 'train_full_cloud_val_frames', len(reference)))
        if expected <= 0 or len(reference) != expected or len(candidate) != expected:
            return dict(eligible=False, reason='incomplete fixed validation or reference')
        for row in reference + list(candidate):
            if not row.get('input_sha256') or not row.get('geometry_setting'):
                raise ValueError('validation fingerprint missing')
        return validation_promotion(candidate, reference)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return dict(eligible=False, reason='invalid autonomous reference: ' + str(error))
