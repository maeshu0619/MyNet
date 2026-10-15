"""Conventional-codec configuration, deliberately isolated from SparsePCGC."""
import math


def configure_geometry_codec(args, explicit):
    key = str(args.compress).lower().replace('-', '').replace('_', '')
    if key not in {'gpcc', 'octattention'}:
        return args
    if str(args.heuristic_guidance_mode) != 'ana_den6_online':
        raise ValueError('G-PCC/OctAttention pooled training currently requires --heuristic_guidance_mode ana_den6_online')
    if not args.compression_loss_backend.startswith(key + '_'):
        raise ValueError('compression loss backend must match the selected conventional codec')
    if not 0 < args.geometry_octree_initial_total_ratio < 1:
        raise ValueError('geometry_octree_initial_total_ratio must be between 0 and 1')
    if not math.isfinite(args.geometry_octree_pool_reserve_factor) or args.geometry_octree_pool_reserve_factor < 1 or args.geometry_octree_pool_max_per_operation < 1:
        raise ValueError('geometry candidate reserve must be >=1 and pool cap positive')
    bins = sorted(set(float(v) for v in args.heuristic_guidance_online_amount_bins.split(',')))
    if not bins or any(not math.isfinite(v) or v <= 0 or v > 0.0099 for v in bins):
        raise ValueError('conventional Amount bins must be finite positive ratios <=0.0099')
    args.geometry_octree_amount_bin_values = tuple(bins)
    shares = [float(v) for v in (args.heuristic_guidance_operation_shares or '0.4,0.4,0.2').split(',')]
    if len(shares) != 3 or any(not math.isfinite(v) or v <= 0 for v in shares):
        raise ValueError('operation shares must contain 3 finite positive values')
    total = args.geometry_octree_initial_total_ratio
    for flag, name, share in zip(
        ('--repair_init_add_ratio', '--repair_init_drop_ratio', '--repair_init_move_ratio'),
        ('repair_init_add_ratio', 'repair_init_drop_ratio', 'repair_init_move_ratio'), shares
    ):
        if not explicit(flag):
            setattr(args, name, total * share / sum(shares))
    if not explicit('--heuristic_guidance_anchor_steps'):
        args.heuristic_guidance_anchor_steps = 0
    if not explicit('--repair_amount_target_mode'):
        args.repair_amount_target_mode = 'none'
    # Guard old SparsePCGC-only teacher/oracle branches without changing the
    # shared one-plan Actor/Proposal/Critic, exploration or Amount/Fine system.
    for name in (
        'enable_sparsepcgc_exact_occupancy_teacher', 'sparsepcgc_outcome_imitation',
        'sparsepcgc_multi_subtree_train', 'sparsepcgc_full_cloud_amount_enable',
        'sparsepcgc_actual_oracle_enable', 'sparsepcgc_full_cloud_actual_primary',
        'sparsepcgc_codec_prune_prior', 'sparsepcgc_subtree_actual_filter',
    ):
        setattr(args, name, False)
    args.train_full_cloud_actual_interval = 0
    args.sparsepcgc_training_mode = 'legacy'
    args.compression_surrogate_target_cache_entries = 0
    args.compression_surrogate_replay_entries = 0
    args.compression_surrogate_replay_steps = 0
    args.compression_surrogate_reuse_last_target = False
    args.compression_surrogate_refresh_interval = 1
    args.disable_actual_codec_during_train = False
    args.actual_codec_fallback_to_proxy_on_error = False
    args.actual_eval_interval = 1
    # The retained score-function capture runs before GradScaler's joint
    # backward. FP16 silently underflowed Where gradients in real smoke runs.
    # Use the same FP32 policy-head contract as the SparsePCGC online path.
    if explicit('--use_amp') and args.use_amp:
        raise ValueError('pooled conventional-codec training currently requires --use_amp false')
    args.use_amp = False
    for name, value in (
        ('heuristic_guidance_network_residual_weight', 1.0),
        ('heuristic_guidance_network_residual_weight_max', 1.0),
        ('heuristic_guidance_network_residual_weight_increment', 0.0),
    ):
        if not explicit('--' + name):
            setattr(args, name, value)
    args.repair_move_require_empty_target = True
    args.repair_move_prefer_occupied_target = False
    # Geometry, Surrogate and Actual must observe the same edited occupied
    # set. The legacy non-SparsePCGC default returned pre-edit node positions,
    # so Geometry stayed constant even when the executed plan changed.
    args.repair_output_voxel_restored_points = True
    # Inference must use the same pools/grids, with noise disabled by eval().
    args.heuristic_guidance_network_only_inference = False
    if key == 'octattention' and not explicit('--octattention_actualcode'):
        args.octattention_actualcode = True  # arithmetic-coded bits, not NLL
    return args
