# models/utils/training/log_setup.py

from models.utils.pointcloud.quant_noise import resolve_uniform_noise_delta
from models.utils.training.utils import uses_actual_total_bit_objective


def log_basic_setup(writer, args, file_day, file_time):
    writer.write(f"Date of Training: {file_day}-{file_time}")
    writer.write(f"Log Root: {args.log_root}")
    writer.write(f"Checkpoint Root: {args.out_path}")
    writer.write(f"Method Name: {getattr(args, 'method_name', 'Mine')}")
    writer.write(
        f"Surrogate Name: "
        f"{getattr(args, 'surrogate_name', getattr(args, 'compress', 'OctAttention'))}"
    )
    writer.write(f"Geometry Loss Type: {args.loss_type}")
    writer.write(f"Discrete Loss Mode: {args.discrete_loss_mode}")
    writer.write(
        "Optimization Modes: "
        f"geometry={'ste_hard' if args.discrete_loss_mode == 'ste_hard' else ('weighted_soft' if args.discrete_loss_mode == 'weighted_soft' else 'hard')}, "
        f"compression={args.compression_loss_backend}"
    )
    if str(getattr(args, "discrete_loss_mode", "")).strip().lower() == "hard":
        writer.write(
            "Discrete Loss Warning: hard mode sets final_w_for_loss=None in train.py; "
            "delete weights are not used by geometry/compression losses. Prefer ste_hard for learning."
        )
    writer.write(
        "Compression Delta Sign: "
        "delta_percent=(after_bits-before_bits)/before_bits*100; negative means improved compression."
    )
    writer.write(
        f"Compression Loss Mode: compression_loss_delta={bool(getattr(args, 'compression_loss_delta', True))} "
        "(True keeps the standard delta, False uses 100-delta for the objective)."
    )
    writer.write(
        f"Minimal Objective: minimal_loss_objective={bool(getattr(args, 'minimal_loss_objective', True))}, "
        f"geometry_fit_weight={float(getattr(args, 'geometry_fit_weight', 0.05)):.6g}, "
        f"geometry_use_d2={bool(getattr(args, 'geometry_use_d2', False))}"
    )
    writer.write(
        "Gradient Scale Groups: "
        f"prune_where_compression={float(getattr(args, 'grad_scale_prune_where_compression', 1.0)):.6g}, "
        f"prune_where_actuator={float(getattr(args, 'grad_scale_prune_where_actuator', 1.0)):.6g}, "
        f"operation_amount={float(getattr(args, 'grad_scale_operation_amount', 1.0)):.6g}"
    )
    writer.write(
        "Gradient Diagnostics: "
        f"compression_grad_probe={bool(getattr(args, 'compression_grad_probe', False))}"
        f"(every={int(getattr(args, 'compression_grad_probe_every', 1))}), "
        f"debug_grad_flow={bool(getattr(args, 'debug_grad_flow', False))}"
        f"(rate={int(getattr(args, 'debug_grad_flow_rate', 1))})"
    )


def log_loss_weight_setup(writer, args):
    compression_backend = str(
        getattr(args, "compression_loss_backend", "proxy")
    ).strip().lower()
    actual_total_bit_backend = uses_actual_total_bit_objective(args)
    surrogate_backend = compression_backend.endswith("_surrogate")

    if surrogate_backend:
        writer.write(
            "Compression Forward Metric: actual_total_bit_percent=100*(Mine_bits-GT_bits)/GT_bits "
            "with surrogate backward; soft auxiliaries are diagnostics unless explicitly enabled."
        )
        writer.write(
            "Compression Teacher Refresh: "
            f"periodic(interval={int(getattr(args, 'compression_surrogate_refresh_interval', 0))}, "
            f"actual_eval_interval={int(getattr(args, 'actual_eval_interval', 1000))}, "
            f"warmup_steps={int(getattr(args, 'compression_surrogate_warmup_steps', 0))}, "
            f"replay_steps={int(getattr(args, 'compression_surrogate_replay_steps', 0))}, "
            f"replay_batch={int(getattr(args, 'compression_surrogate_replay_batch', 0))}, "
            f"forward={getattr(args, 'compression_surrogate_forward_mode', 'teacher_ste')}, "
            f"aux_node={float(getattr(args, 'compression_surrogate_aux_node_weight', 0.0))}, "
            f"aux_single={float(getattr(args, 'compression_surrogate_aux_single_weight', 0.0))}, "
            f"aux_in_objective={bool(getattr(args, 'compression_surrogate_aux_in_objective', False))}, "
            f"log_soft_aux={bool(getattr(args, 'compression_surrogate_log_soft_aux', True))}, "
            f"reuse_last_target={bool(getattr(args, 'compression_surrogate_reuse_last_target', True))})"
        )
        writer.write(
            "Surrogate Pretrain/Online Update: "
            f"pretrain_steps={int(getattr(args, 'surrogate_pretrain_steps', 0))}, "
            f"pretrain_lr={float(getattr(args, 'surrogate_pretrain_lr', 1e-4))}, "
            f"pretrain_mode={getattr(args, 'surrogate_pretrain_mode', 'full')}, "
            f"pretrain_subtree_teacher={getattr(args, 'surrogate_pretrain_subtree_teacher_type', 'local_actual')}, "
            f"pretrain_depth_percent={float(getattr(args, 'surrogate_pretrain_subtree_depth_percent_min', 0.0)):.3g}-"
            f"{float(getattr(args, 'surrogate_pretrain_subtree_depth_percent_max', 0.50)):.3g}, "
            f"pretrain_full_calibration_interval={int(getattr(args, 'surrogate_pretrain_full_calibration_interval', 0))}, "
            f"pretrain_full_calibration_steps={int(getattr(args, 'surrogate_pretrain_full_calibration_steps', 1))}, "
            f"pretrain_actual_refresh_interval={int(getattr(args, 'surrogate_pretrain_actual_refresh_interval', 0))}, "
            f"pretrain_replay={bool(getattr(args, 'surrogate_pretrain_use_replay', True))}, "
            f"pretrain_replay_steps={int(getattr(args, 'surrogate_pretrain_replay_steps', 0))}, "
            f"pretrain_replay_batch={int(getattr(args, 'surrogate_pretrain_replay_batch_size', 0))}, "
            f"pretrain_sparsepcgc_debug_interval={int(getattr(args, 'surrogate_pretrain_sparsepcgc_debug_interval', 0))}, "
            f"update_during_training={bool(getattr(args, 'surrogate_update_during_training', True))}, "
            f"update_interval={int(getattr(args, 'surrogate_update_interval', 1))}, "
            f"joint_lr_scale={float(getattr(args, 'surrogate_joint_lr_scale', 0.1))}, "
            f"teacher_refresh_only={bool(getattr(args, 'surrogate_update_on_teacher_refresh_only', False))}"
        )
        writer.write("Compression Surrogate Backward: enabled")
        writer.write(
            f"Surrogate compression objective on {compression_backend}: "
            "actual_total_bit_percent = 100*(Mine_bits-GT_bits)/GT_bits with surrogate backward; "
            "soft auxiliary proxies are diagnostics unless explicitly enabled."
        )

    if actual_total_bit_backend:
        writer.write(
            "Loss Weight: "
            f"geom={args.w_geom}, "
            f"comp_total={args.w_com} (actual_total_bit_percent), "
            f"attr={args.w_attr}, policy={args.w_policy}, actuator={args.w_actuator}"
        )
    else:
        writer.write(
            "Loss Weight: "
            f"geom={args.w_geom}, "
            f"comp_total={args.w_com}, comp_bit={args.com_bit}, comp_single={args.com_sin}, "
            f"comp_node={args.com_node}, comp_bpn={getattr(args, 'com_bpn', 0.0)}, "
            f"comp_sparsepcgc={getattr(args, 'com_sparsepcgc', 0.0)}, comp_lowprob={args.com_lowprob}, "
            f"attr={args.w_attr}, policy={args.w_policy}, actuator={args.w_actuator}"
        )


def log_runtime_setup(writer, args):
    writer.write(
        f"Runtime Tradeoff: k={args.k}, encoder_query_chunk={args.encoder_query_chunk}, "
        f"structure_geo_max_points={args.structure_geo_max_points}, "
        f"knn_backend={getattr(args, 'knn_backend', 'pointops_cuda')}"
    )
    writer.write(
        "Point Edit Count Thresholds: "
        f"drop_keep_threshold={float(getattr(args, 'operation_count_drop_threshold', 0.5))}, "
        f"adjust_threshold={float(getattr(args, 'operation_count_adjust_threshold', 1e-6))}"
    )
    writer.write(
        "Uniform Quantization Noise: "
        f"use={bool(getattr(args, 'use_uniform_noise', True))}, "
        f"delta={resolve_uniform_noise_delta(args):.6g}, "
        "scope=train_only_after_edit_before_rate_struct, "
        "shape_loss=clean_X_edit, rate_struct_loss=noisy_X_edit, "
        "actual_codec_eval=clean_X_edit, "
        "supported_compress=gpcc,draco,octattention,sparsepcgc"
    )
    writer.write(
        "Profiling: "
        f"log_step_time={bool(getattr(args, 'log_step_time', True))}, "
        f"log_gpu_memory={bool(getattr(args, 'log_gpu_memory', True))}, "
        f"profile_interval={int(getattr(args, 'profile_interval', 100))}, "
        f"actual_eval_interval={int(getattr(args, 'actual_eval_interval', 1000))}, "
        f"disable_actual_codec_during_train={bool(getattr(args, 'disable_actual_codec_during_train', False))}, "
        f"actual_codec_fallback_to_proxy_on_error={bool(getattr(args, 'actual_codec_fallback_to_proxy_on_error', True))}, "
        f"skip_optimizer_on_actual_fallback={bool(getattr(args, 'skip_optimizer_on_actual_fallback', True))}"
    )
    writer.write(
        "Checkpoint Selection: "
        "primary=actual_total_bit_percent(fresh only), "
        f"geom_gate={bool(getattr(args, 'checkpoint_geom_gate', True))}"
        f"(rel={float(getattr(args, 'checkpoint_geom_rel_factor', 1.5)):.6g}, "
        f"abs={float(getattr(args, 'checkpoint_geom_abs_max', 0.0)):.6g}), "
        f"safety_gate={bool(getattr(args, 'checkpoint_safety_gate', True))}"
        f"(repair_abs={float(getattr(args, 'checkpoint_repair_abs_max', 10.0)):.6g}, "
        f"node_abs={float(getattr(args, 'checkpoint_node_abs_max', 100.0)):.6g}, "
        f"single_abs={float(getattr(args, 'checkpoint_single_abs_max', 100.0)):.6g}, "
        f"op_ratio_max={float(getattr(args, 'checkpoint_operation_ratio_max', 100.0)):.6g}), "
        f"metric_csv[compression={bool(getattr(args, 'save_compression_metric_csv', True))}, "
        f"operation={bool(getattr(args, 'save_operation_metric_csv', True))}, "
        f"checkpoint={bool(getattr(args, 'save_checkpoint_metric_csv', True))}]"
    )
    writer.write(
        "Actual Compression Guard: "
        f"enabled={bool(getattr(args, 'actual_compression_guard', True))}, "
        f"patience={int(getattr(args, 'actual_guard_patience', 2))}, "
        f"tolerance={float(getattr(args, 'actual_guard_tolerance', 0.25)):.6g}, "
        f"lr_decay={float(getattr(args, 'actual_guard_lr_decay', 0.5)):.6g}, "
        f"restore_best={bool(getattr(args, 'actual_guard_restore_best', True))}"
    )


def log_codec_setup(writer, args):
    compression_backend = str(
        getattr(args, "compression_loss_backend", "proxy")
    ).strip().lower()

    writer.write(f"Compression Codec: {getattr(args, 'compress', 'OctAttention')}")
    writer.write(f"OctAttention Teacher Device: {args.octattention_teacher_device}")

    if compression_backend.startswith("sparsepcgc"):
        writer.write(
            "SparsePCGC Teacher: "
            f"env={getattr(args, 'sparsepcgc_env', 'sparsepcgc')}, "
            f"python={getattr(args, 'sparsepcgc_python', '') or '(auto)'}, "
            f"mode={getattr(args, 'sparsepcgc_mode', 'dense_lossless')}, "
            f"device={getattr(args, 'sparsepcgc_device', 'auto')}, "
            f"match_qs={bool(getattr(args, 'sparsepcgc_match_qs', True))}, "
            f"voxel_size={float(getattr(args, 'sparsepcgc_voxel_size', 1.0))}, "
            f"pos_quantscale={int(getattr(args, 'sparsepcgc_pos_quantscale', 1))}, "
            f"effective_qs={float(getattr(args, 'sparsepcgc_effective_qs', 0.0))}, "
            f"root={getattr(args, 'sparsepcgc_root', '')}, "
            f"skip_decode={bool(getattr(args, 'sparsepcgc_skip_decode', True))}"
        )
        writer.write(
            "SparsePCGC Operation Safety: "
            f"disable_add={bool(getattr(args, 'sparsepcgc_disable_add', True))}, "
            f"add_experiment={bool(getattr(args, 'sparsepcgc_enable_add_experiment', False))}, "
            f"add_only_cp={bool(getattr(args, 'sparsepcgc_add_only_when_compression_primary', True))}, "
            f"move_existing_target_only={bool(getattr(args, 'sparsepcgc_move_existing_target_only', True))}, "
            f"target_add_ratio={float(getattr(args, 'target_add_ratio', 0.0))}, "
            f"max_add_ratio={float(getattr(args, 'max_add_ratio', 0.0))}, "
            f"add_hard_threshold={float(getattr(args, 'repair_add_hard_threshold', 0.5))}, "
            f"move_hard_threshold={float(getattr(args, 'repair_move_hard_threshold', 0.5))}, "
            f"move_source_prior_weight={float(getattr(args, 'sparsepcgc_move_source_prior_weight', 0.0))}, "
            f"exp_target={float(getattr(args, 'sparsepcgc_add_target_ratio', 0.005))}, "
            f"exp_max={float(getattr(args, 'sparsepcgc_add_max_ratio', 0.10))}"
        )
        writer.write(
            "SparsePCGC Quantization Alignment: "
            "teacher=round(x/voxel_size)->unique->round(coord/posQuantscale)->unique, "
            f"network_sparse_quant=sparsepcgc_twostep, "
            f"surrogate_effective_qs={float(getattr(args, 'sparsepcgc_effective_qs', 0.0))}, "
            f"surrogate_levels={getattr(args, 'compression_surrogate_levels', '4,6,8')}"
        )
        writer.write(
            "SparsePCGC Prune Prior/Floor: "
            f"prune_after_prior_mode={getattr(args, 'sparsepcgc_prune_after_prior_mode', 'oracle')}, "
            f"codec_prior_enabled={bool(getattr(args, 'sparsepcgc_codec_prune_prior', False))}, "
            f"codec_prior_warmup_steps={int(getattr(args, 'sparsepcgc_codec_prune_prior_warmup_steps', 0))}, "
            f"network_prune_ratio_floor={float(getattr(args, 'sparsepcgc_network_prune_ratio_floor', 0.0)):.6g}, "
            f"network_prune_min_hard_count={int(getattr(args, 'sparsepcgc_network_prune_min_hard_count', 0))}, "
            f"network_prune_floor_steps={int(getattr(args, 'sparsepcgc_network_prune_floor_steps', 0))}, "
            f"network_prune_floor_decay_steps={int(getattr(args, 'sparsepcgc_network_prune_floor_decay_steps', 0))}"
        )
        writer.write(
            "SparsePCGC Exact Occupancy Teacher: "
            f"enabled={bool(getattr(args, 'enable_sparsepcgc_exact_occupancy_teacher', False))}, "
            f"interval={int(getattr(args, 'sparsepcgc_exact_occupancy_interval', 1))}, "
            f"mode={getattr(args, 'sparsepcgc_exact_teacher_mode', 'auto')}, "
            f"loss_enabled={bool(getattr(args, 'enable_sparsepcgc_exact_occupancy_loss', False))}, "
            f"nll_weight={float(getattr(args, 'sparsepcgc_exact_occupancy_loss_weight', 0.0))}, "
            f"bits_weight={float(getattr(args, 'sparsepcgc_exact_bits_loss_weight', 0.0))}"
        )

    if compression_backend.startswith("gpcc"):
        writer.write(
            "G-PCC Teacher: "
            f"encoder={getattr(args, 'gpcc_encoder_path', '')}, "
            f"cfg={getattr(args, 'gpcc_cfg_dir', '')}, "
            f"match_qs={bool(getattr(args, 'gpcc_match_qs', True))}, "
            f"prequantize={bool(getattr(args, 'gpcc_prequantize', True))}, "
            f"effective_qs={float(getattr(args, 'gpcc_effective_qs', 0.0))}, "
            f"geometry_only={bool(getattr(args, 'gpcc_disable_attribute_coding', True))}, "
            f"merge_duplicates={bool(getattr(args, 'gpcc_merge_duplicated_points', True))}, "
            f"timeout={float(getattr(args, 'gpcc_timeout', 120.0))}"
        )

    if compression_backend.startswith("draco"):
        writer.write(
            "Draco Teacher: "
            f"encoder={getattr(args, 'draco_encoder_path', '')}, "
            f"decoder={getattr(args, 'draco_decoder_path', '')}, "
            f"match_qs={bool(getattr(args, 'draco_match_qs', True))}, "
            f"prequantize={bool(getattr(args, 'draco_prequantize', True))}, "
            f"effective_qs={float(getattr(args, 'draco_effective_qs', 0.0))}, "
            f"qp={int(getattr(args, 'draco_position_quantization_bits', 0))}, "
            f"cl={int(getattr(args, 'draco_compression_level', 7))}, "
            f"point_cloud={bool(getattr(args, 'draco_force_point_cloud', True))}, "
            f"merge_duplicates={bool(getattr(args, 'draco_merge_duplicated_points', True))}, "
            f"skip_decode={bool(getattr(args, 'draco_skip_decode', True))}"
        )

    writer.write(f"Compression Rate Metric: {args.compression_rate_metric}")
    writer.write(f"Compression Loss Backend: {args.compression_loss_backend}")


def log_training_setup(writer, args, file_day, file_time):
    log_basic_setup(writer, args, file_day, file_time)
    log_loss_weight_setup(writer, args)
    log_runtime_setup(writer, args)
    log_codec_setup(writer, args, )
    writer.write(
        f"Two-Stage Training: enabled={bool(getattr(args, 'two_stage_training', False))}, "
        f"base_stage={args.training_stage}, diagnosis_ratio={getattr(args, 'diagnosis_episode_ratio', 0.0)}, "
        f"diagnosis_episodes={getattr(args, 'diagnosis_episodes', 0)}"
    )
    writer.write(f"Module BatchNorm Running Stats: {args.module_bn_use_running_stats}")
