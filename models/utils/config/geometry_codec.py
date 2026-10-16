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
    if not math.isfinite(args.geometry_octree_max_edit_ratio) or not 0 < args.geometry_octree_max_edit_ratio < 1:
        raise ValueError('geometry_octree_max_edit_ratio must be between 0 and 1')
    if key == 'octattention' and args.geometry_octree_rate_model == 'neighbour_binary':
        raise ValueError('neighbour_binary is an experimental G-PCC prior, not an OctAttention probability model')
    if args.geometry_octree_joint_policy:
        # The shared initializer reads this legacy field. Its SparsePCGC 3%
        # default would always initialize geometry's largest positive bin.
        if not explicit('--sparsepcgc_algorithmic_amount_init_ratio'):
            args.sparsepcgc_algorithmic_amount_init_ratio = args.geometry_octree_initial_total_ratio
        for name, value in (
            ('heuristic_guidance_online_actor_proxy_weight', 0.),
            ('heuristic_guidance_online_actor_critic_distill_weight', 0.),
            ('cp_geometry_backward_scale', 1.),
            ('heuristic_guidance_online_global_actual_credit_weight', 0.5),
        ):
            if not explicit('--' + name):
                setattr(args, name, value)
        if not explicit('--heuristic_guidance_online_amount_bins'):
            args.heuristic_guidance_online_amount_bins = '0.0005,0.0015,0.003,0.006,0.009'
        if not explicit('--cp_tau_geom'):
            args.cp_tau_geom = 0.  # excess distortion, measured in voxel^2
    for name in ('geometry_octree_distortion_weight', 'geometry_octree_actor_weight',
                 'geometry_octree_score_scale', 'geometry_octree_joint_temperature'):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0 or (name.endswith('scale') and value == 0):
            raise ValueError(name + ' must be finite and nonnegative (scale positive)')
    bins = sorted(set(float(v) for v in args.heuristic_guidance_online_amount_bins.split(',')))
    maximum = args.geometry_octree_max_edit_ratio if args.geometry_octree_joint_policy else .0099
    if not bins or any(not math.isfinite(v) or v <= 0 or v > maximum for v in bins):
        raise ValueError('conventional Amount bins must be finite positive ratios <=geometry_octree_max_edit_ratio')
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
