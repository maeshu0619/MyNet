import os
_TMPDIR = os.environ.get("TMPDIR") or "/dev/shm/mynet_tmp"
try:
    os.makedirs(_TMPDIR, exist_ok=True)
    os.environ["TMPDIR"] = _TMPDIR
    os.environ["TEMP"] = _TMPDIR
    os.environ["TMP"] = _TMPDIR
except OSError:
    pass
import torch
import torch.optim as optim
import argparse
import hashlib
import math
from collections import OrderedDict

import time
import datetime
from contextlib import nullcontext

from models.utils.pointcloud.utils_repkpu import *
from models.utils.pointcloud.octree_subtree import (
    assign_octree_subtree_keys,
    build_subtree_index_map,
    build_octree_subtree_reference,
    sample_train_subtree_depth,
    select_octree_subtree_keys,
    should_use_full_cloud_anchor,
)
from models.utils.pointcloud.quant_noise import add_uniform_quantization_noise, resolve_uniform_noise_delta
from models.utils.data.dataset import *
from models.utils.patching.patch import (
    build_patch_info,
    denormalize_patch_output,
    merge_patch_outputs,
    merge_patch_subset_outputs,
    patch_info_to_cpu,
    patch_info_to_device,
)
from models.utils.compression.octree_stats import hard_octree_occupancy_stats
from models.utils.training.utils_grad import *
from models.network import Network
from models.utils.loss.loss import Loss
from models.utils.notify.mail_notify import TrainingMailNotifier
from record.write import Writing
from record.plot import PlotMaker

from cfgs.utils import str2bool
from models.utils.config.args import parse_pugan_args

import multiprocessing as mp

from models.utils.training.utils import (_adapt_encoder_state_dict_for_sparse_input,
                                         _resolve_amp_dtype,
                                         _warmup_whole_cloud_caches,
                                         _use_memory_safe_loader_workers,
                                         _resolve_training_stage_for_episode,
                                         _stage_loss_factors,
                                         _make_step_cache_key,
                                         _should_log_step,
                                         _downsample_input_batch,
                                         _get_patch_info,
                                         _effective_patch_batch_size,
                                         select_patch_subset_ids,
                                         make_patch_subset_cache_key,
                                         _accumulate_grouped_patch_geometry,
                                         _run_geometry_audit,
                                         _new_metric_sums,
                                         _add_metric_sums,
                                         _surrogate_plot_metrics,
                                         _sync_for_timing,
                                         _prepare_whole_cloud_inputs,
                                         _metric_avgs_to_floats,
                                         _format_metric_summary,
                                         _new_point_edit_sums,
                                         _add_point_edit_sums,
                                         _finalize_point_edit_sums,
                                         _summarize_point_edits,
                                         _cuda_bf16_ops_safe,
                                         _log_grad_flow, 
                                         _capture_param_update_snapshots,
                                         _log_param_updates,
                                         _format_named_float_map,
                                         _uses_actual_total_bit_objective,
                                         _write_structure_decision_debug,
                                         _compression_stat_qs,
                                         _format_triplet,
                                         _summarize_subtree_octree_stats,)
from models.utils.training.utils import *
from models.utils.training.noise_debug import (
    empty_noise_debug,
    merge_noise_debug_values,
    prepare_compression_points,
    accumulate_compression_terms,
)
from models.utils.training.correlation import (
    finite_float_or_none,
    rolling_pearson,
    push_rolling_correlation,
    format_corr,
)
from models.utils.training.optim_amp import (
    build_optimizer_and_scheduler,
    setup_amp,
    build_loader_kwargs,
)
from models.utils.training.checkpointing import save_episode_checkpoint
from models.utils.training.train_logging import (
    log_backend_summary,
    log_input_mode,
    log_structure_debug,
    log_point_edit_stats,
)
from models.utils.training.log_step import (
    log_step_loss,
    log_compression_stats,
    log_compression_train_debug,
    log_codec_actual_correlation,
    log_sparsepcgc_train_debug,
    log_step_timing,
)
from models.utils.training.log_epoch_episode import (
    log_epoch_point_edit_average,
    log_episode_point_edit_average,
    log_plot_skip_epoch,
    log_plot_skip_episode,
)
from models.utils.training.log_setup import log_training_setup

def train(model, args, loss, writer, plot, notifier=None):
    """==========================================================="""
    """セットアップ"""
    """==========================================================="""
    set_seed(args.seed, deterministic=getattr(args, "deterministic", False))
    best_loss = float('inf')

    # ===== Loss histogram (fixed size, safe) =====
    seq_dirs = collect_seq_dirs2(args.input_dir, dataset_name=args.dataname)
    num_seq = len(seq_dirs)

    writer.write(f"Total seq directories: {num_seq}")
    seq_datasets = [(seq_dir, PlyDirDataset(args, seq_dir)) for seq_dir in seq_dirs]
    total_train_files = sum(len(dataset) for _, dataset in seq_datasets)
    args._total_train_steps_estimate = max(int(getattr(args, "episodes", 1)), 1) * max(int(total_train_files), 1)
    set_cache_expected = getattr(model, "set_expected_input_cache_entries", None)
    if callable(set_cache_expected):
        set_cache_expected(total_train_files)
    patch_info_cache = OrderedDict()
    sparsepcgc_proxy_actual_pairs = []
    codec_actual_metric_pairs = {}

    # モデル保存先ファイルのセットアップ
    output_dir = os.path.join(args.out_path)
    ckpt_dir = os.path.join(output_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    optimizer, scheduler_steplr = build_optimizer_and_scheduler(
        model,
        args,
        writer,
    )

    amp_state = setup_amp(
        model,
        args,
        writer,
    )

    use_cuda = amp_state["use_cuda"]
    use_amp = amp_state["use_amp"]
    amp_dtype = amp_state["amp_dtype"]
    amp_scaler_enabled = amp_state["amp_scaler_enabled"]
    scaler = amp_state["scaler"]
    amp_overflow_patience = amp_state["amp_overflow_patience"]
    consecutive_amp_skips = amp_state["consecutive_amp_skips"]

    _warmup_whole_cloud_caches(model, args, loss, seq_datasets, writer, use_cuda, use_amp, amp_dtype)
    loader_kwargs = build_loader_kwargs(
        args,
        model,
        writer,
        use_cuda,
    )
    """==========================================================="""
    """トレーニング"""
    """==========================================================="""
    prev_stage = None
    global_train_step = 0
    global_epoch = 0
    for episode in range(args.episodes):
        current_stage = _resolve_training_stage_for_episode(args, episode)
        args.training_stage = current_stage
        if current_stage != prev_stage:
            stage_factors = _stage_loss_factors(args)
            writer.write(f"Training Stage Switch: episode={episode + 1}, stage={current_stage}")
            writer.write(
                "Stage Loss Factors: "
                f"geom={stage_factors['geom']}, com={stage_factors['com']}, "
                f"attr={stage_factors['attr']}, policy={stage_factors['policy']}, repair={stage_factors['repair']}"
            )
            prev_stage = current_stage
        writer.write(f"◆◆◆ Episode {episode + 1} / {args.episodes} ◆◆◆")
        model.train()
        episode_metric_sums = None
        for epoch, (seq_dir, dataset) in enumerate(seq_datasets):
            writer.write(f"●●● Epoch {epoch + 1}/{num_seq} : {seq_dir} ●●●")
            loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
            num_steps = len(dataset)
            epoch_has_optimizer_step = False
            epoch_metric_sums = None

            for step, pts in enumerate(loader):
                st_step = time.time()
                file_path = dataset.files[step]
                cache_key = _make_step_cache_key(file_path, args)
                log_this_step = _should_log_step(step + 1, num_steps, args.print_rate)
                profile_this_step = _should_log_step(
                    global_train_step + 1,
                    max(int(getattr(args, "_total_train_steps_estimate", num_steps)), 1),
                    int(getattr(args, "profile_interval", 100)),
                )
                timing_enabled = bool(
                    (getattr(args, "debug_timing", False) and log_this_step)
                    or (
                        (
                            getattr(args, "log_step_time", True)
                            or getattr(args, "log_gpu_memory", True)
                        )
                        and profile_this_step
                    )
                )
                args._global_train_step = int(global_train_step)
                args._log_this_step = False
                args._collect_sparsepcgc_debug = bool(log_this_step or profile_this_step)
                detail_log_this_step = False
                raw_pts_num = int(pts.shape[1] if pts.dim() == 3 else pts.shape[0])
                if timing_enabled and use_cuda and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                # pts: [1, N, 3]
                if timing_enabled:
                    _sync_for_timing(use_cuda)
                    timing_data_start = time.time()
                subtree_mode = bool(getattr(args, "train_patch_subset_enable", False))
                if subtree_mode:
                    input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
                    input_pcd = _downsample_input_batch(input_pcd, args, cache_key)
                    if use_cuda:
                        input_pcd = input_pcd.cuda(non_blocking=True)
                    input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous()
                    input_xyz = input_pcd[:, :3, :]
                elif args.split2patch:
                    input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
                    input_pcd = _downsample_input_batch(input_pcd, args, cache_key)
                    if use_cuda:
                        input_pcd = input_pcd.cuda(non_blocking=True)
                    input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous()
                    input_xyz = input_pcd[:, :3, :]
                else:
                    input_xyz, patches, centroid_xyz, fd_xyz = _prepare_whole_cloud_inputs(pts, args, cache_key, use_cuda)
                    input_pcd = input_xyz

                pcd_pts_num = input_xyz.shape[-1]
                if timing_enabled:
                    _sync_for_timing(use_cuda)
                    timing_data_end = time.time()
                    timing_model_start = timing_data_end

                clear_policy_terms = getattr(model, "clear_discrete_policy_terms", None)
                if callable(clear_policy_terms):
                    clear_policy_terms()
                stage_factors = _stage_loss_factors(args)
                compute_compression = stage_factors["com"] != 0.0
                refresh_actual_gen = not bool(getattr(args, "disable_actual_codec_during_train", False))

                subset_step = False
                subset_enabled = False
                is_anchor_step = True
                compression_cache_key = cache_key
                compression_gt_pts = input_xyz
                train_edit_stats = None
                noise_debug = empty_noise_debug()

                if subtree_mode:
                    optimizer.zero_grad(set_to_none=True)
                    subset_enabled = True
                    input_attr_full = input_pcd[:, 3:, :].contiguous() if input_pcd.shape[1] > 3 else None
                    subtree_depth_meta = sample_train_subtree_depth(
                        input_xyz,
                        args,
                        global_step=global_train_step,
                        cache_key=cache_key,
                    )
                    subtree_ref = build_octree_subtree_reference(
                        input_xyz,
                        args,
                        depth=int(subtree_depth_meta["depth"]),
                    )
                    full_subtree_keys = assign_octree_subtree_keys(input_xyz, subtree_ref)
                    all_subtree_keys, subtree_index_lists = build_subtree_index_map(full_subtree_keys)
                    total_subtree_count = int(all_subtree_keys.numel())
                    min_subtree_points = max(int(getattr(args, "train_subtree_min_points", 1)), 1)
                    eligible_groups = [
                        (int(subtree_key.detach().cpu()), point_idx)
                        for subtree_key, point_idx in zip(all_subtree_keys, subtree_index_lists)
                        if int(point_idx.numel()) >= min_subtree_points
                    ]
                    if eligible_groups:
                        candidate_subtree_keys = all_subtree_keys.new_tensor(
                            [subtree_key for subtree_key, _ in eligible_groups]
                        )
                    else:
                        candidate_subtree_keys = all_subtree_keys
                    eligible_subtree_count = int(candidate_subtree_keys.numel())
                    is_anchor_step, anchor_reason = should_use_full_cloud_anchor(
                        args,
                        global_step=global_train_step,
                        cache_key=cache_key,
                    )
                    selected_subtree_keys = candidate_subtree_keys
                    if eligible_subtree_count > 0 and not is_anchor_step:
                        selected_subtree_keys = select_octree_subtree_keys(candidate_subtree_keys, global_train_step, args)
                    selected_subtree_count = int(selected_subtree_keys.numel())
                    subset_step = (not is_anchor_step) and selected_subtree_count < eligible_subtree_count
                    encoder_debug_chunks = [] if detail_log_this_step else None
                    selected_groups = None
                    if not is_anchor_step:
                        selected_key_set = set(selected_subtree_keys.detach().cpu().tolist())
                        group_source = eligible_groups
                        if not group_source:
                            group_source = [
                                (int(subtree_key.detach().cpu()), point_idx)
                                for subtree_key, point_idx in zip(all_subtree_keys, subtree_index_lists)
                            ]
                        selected_groups = [
                            (subtree_key, point_idx)
                            for subtree_key, point_idx in group_source
                            if subtree_key in selected_key_set
                        ]
                        if not selected_groups and group_source:
                            selected_groups = [max(group_source, key=lambda item: int(item[1].numel()))]
                        if not selected_groups:
                            raise RuntimeError("Subtree mode did not select any subtree group.")
                    if log_this_step and bool(getattr(args, "train_patch_subset_log", True)):
                        if is_anchor_step:
                            point_counts = [int(point_idx.numel()) for _, point_idx in (eligible_groups or [])]
                            if not point_counts:
                                point_counts = [int(input_xyz.shape[-1])]
                            stat_groups = eligible_groups or [(0, torch.arange(input_xyz.shape[-1], device=input_xyz.device))]
                            loss_scope = "full_cloud_output_vs_full_cloud_input"
                        else:
                            point_counts = [int(point_idx.numel()) for _, point_idx in selected_groups]
                            stat_groups = selected_groups
                            loss_scope = "subtree_output_vs_subtree_input"
                        mean_points = sum(point_counts) / float(max(len(point_counts), 1))
                        octree_stat = _summarize_subtree_octree_stats(input_xyz, stat_groups, args)
                        octree_stat_text = ""
                        if octree_stat is not None:
                            octree_stat_text = (
                                f", octree_node[min/mean/max]={octree_stat['node']}, "
                                f"octree_single[min/mean/max]={octree_stat['single']}, "
                                f"octree_depth[min/mean/max]={octree_stat['depth']}, "
                                f"octree_stat_count={int(octree_stat['count'])}"
                            )
                        writer.write(
                            "SubtreeSelection: "
                            f"depth={int(subtree_depth_meta['depth'])} "
                            f"(base={int(subtree_depth_meta['base_depth'])}, "
                            f"range={int(subtree_depth_meta['min_depth'])}-{int(subtree_depth_meta['max_depth'])}, "
                            f"uncapped_range={int(subtree_depth_meta.get('uncapped_min_depth', subtree_depth_meta['min_depth']))}-"
                            f"{int(subtree_depth_meta.get('uncapped_max_depth', subtree_depth_meta['max_depth']))}, "
                            f"curriculum_phase={float(subtree_depth_meta.get('curriculum_phase', 1.0)):.3f}, "
                            f"data_max={int(subtree_depth_meta['data_max_depth'])}, "
                            f"percent_mode={bool(subtree_depth_meta.get('depth_percent_curriculum', False))}, "
                            f"percent_range={subtree_depth_meta.get('depth_percent_range', 'n/a')}), "
                            f"selected={selected_subtree_count}/{eligible_subtree_count} eligible "
                            f"(total={total_subtree_count}, min_points={min_subtree_points}), "
                            f"points[min/mean/max]={min(point_counts)}/{mean_points:.1f}/{max(point_counts)}, "
                            f"anchor_refresh={bool(is_anchor_step)}({anchor_reason}), "
                            f"loss_scope={loss_scope}"
                            f"{octree_stat_text}"
                        )

                    L_geom = input_xyz.new_zeros(())
                    L_com = input_xyz.new_zeros(())
                    L_attr = input_xyz.new_zeros(())
                    L_policy = input_xyz.new_zeros(())
                    L_actuator = input_xyz.new_zeros(())
                    Lp_out = input_xyz.new_zeros(())
                    La_fit = input_xyz.new_zeros(())
                    La_rep = input_xyz.new_zeros(())
                    loss_bit = input_xyz.new_zeros(())
                    loss_single = input_xyz.new_zeros(())
                    loss_nodes = input_xyz.new_zeros(())
                    gen_xyz = None
                    final_w = None
                    out_label = None
                    prev_log_flag = getattr(args, "_log_this_step", False)
                    try:
                        args._log_this_step = bool(getattr(args, "verbose_step_logs", False) and detail_log_this_step)
                        if is_anchor_step:
                            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                            with autocast_ctx:
                                (
                                    gen_pts,
                                    L_attr,
                                    L_policy,
                                    L_actuator,
                                    final_w,
                                    Lp_out,
                                    La_fit,
                                    La_rep,
                                    out_label,
                                ) = model.forward(
                                    input_xyz,
                                    input_attr_full,
                                    cache_key=cache_key,
                                    return_attr_output=False,
                                    subtree_ref=subtree_ref,
                                    selected_subtree_keys=None,
                                )
                            if final_w is not None and not torch.isfinite(final_w).all():
                                writer.write(
                                    "Warning: final_w contains NaN/Inf. "
                                    "It will be sanitized before point-edit summary and losses."
                                )
                                final_w = torch.nan_to_num(final_w, nan=0.0, posinf=1.0, neginf=0.0)
                                final_w = final_w.clamp(0.0, 1.0)
                            if detail_log_this_step:
                                base_model = model.module if hasattr(model, "module") else model
                                encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))
                            gen_xyz = gen_pts[:, :3, :]
                            train_edit_stats = _summarize_point_edits(
                                input_xyz=input_xyz[:, :3, :],
                                gen_pts=gen_pts,
                                final_w=final_w,
                                args=args,
                            )
                            final_w_for_loss = None
                            if str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() != "hard":
                                final_w_for_loss = final_w
                            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                            with autocast_ctx:
                                L_geom = loss.get_geometry_loss(
                                    args,
                                    gen_pts=gen_xyz,
                                    gt_pts=input_xyz[:, :3, :],
                                    final_w=final_w_for_loss,
                                    out_label=out_label,
                                )
                                if stage_factors["com"] != 0.0:
                                    compression_gen_xyz, noise_debug = prepare_compression_points(
                                        gen_xyz,
                                        args,
                                        model,
                                        collect_stats=bool(log_this_step or profile_this_step),
                                    )
                                    L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss(
                                        args,
                                        gen_xyz=compression_gen_xyz,
                                        gt_xyz=input_xyz[:, :3, :],
                                        final_w=final_w_for_loss,
                                        cache_key=cache_key,
                                        refresh_actual_gen=refresh_actual_gen,
                                        actual_gen_xyz=gen_xyz,
                                    )
                                else:
                                    zero = input_xyz.new_zeros(())
                                    L_com = zero
                                    loss_bit = zero
                                    loss_single = zero
                                    loss_nodes = zero
                        else:
                            num_selected = float(max(len(selected_groups), 1))
                            subtree_edit_sums = _new_point_edit_sums()
                            subtree_noise_debug_values = []
                            subtree_compression_term_sums = {}
                            for subtree_key, point_idx in selected_groups:
                                subtree_xyz = input_xyz.index_select(2, point_idx).contiguous()
                                subtree_attr = None
                                if input_attr_full is not None:
                                    subtree_attr = input_attr_full.index_select(2, point_idx).contiguous()
                                subtree_cache_key = (
                                    f"{cache_key}|subtree_depth={int(subtree_ref['depth'][0].item())}|subtree_key={subtree_key}"
                                )
                                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                                with autocast_ctx:
                                    (
                                        gen_subtree_pts,
                                        L_attr_sub,
                                        L_policy_sub,
                                        L_actuator_sub,
                                        final_w_sub,
                                        Lp_out_sub,
                                        La_fit_sub,
                                        La_rep_sub,
                                        out_label_sub,
                                    ) = model.forward(
                                        subtree_xyz,
                                        subtree_attr,
                                        cache_key=subtree_cache_key,
                                        return_attr_output=False,
                                    )
                                if detail_log_this_step:
                                    base_model = model.module if hasattr(model, "module") else model
                                    encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))

                                gen_subtree_xyz = gen_subtree_pts[:, :3, :]
                                subtree_edit_stats = _summarize_point_edits(
                                    input_xyz=subtree_xyz[:, :3, :],
                                    gen_pts=gen_subtree_pts,
                                    final_w=final_w_sub,
                                    args=args,
                                )
                                _add_point_edit_sums(subtree_edit_sums, subtree_edit_stats)
                                final_w_sub_loss = None
                                if str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() != "hard":
                                    final_w_sub_loss = final_w_sub

                                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                                with autocast_ctx:
                                    L_geom_sub = loss.get_geometry_loss(
                                        args,
                                        gen_pts=gen_subtree_xyz,
                                        gt_pts=subtree_xyz[:, :3, :],
                                        final_w=final_w_sub_loss,
                                        out_label=out_label_sub,
                                    )
                                    if stage_factors["com"] != 0.0:
                                        compression_subtree_xyz, noise_debug_sub = prepare_compression_points(
                                            gen_subtree_xyz,
                                            args,
                                            model,
                                            collect_stats=bool(log_this_step or profile_this_step),
                                        )
                                        subtree_noise_debug_values.append(noise_debug_sub)
                                        L_com_sub, loss_bit_sub, loss_single_sub, loss_nodes_sub, _, _ = loss.get_compression_loss(
                                            args,
                                            gen_xyz=compression_subtree_xyz,
                                            gt_xyz=subtree_xyz[:, :3, :],
                                            final_w=final_w_sub_loss,
                                            cache_key=subtree_cache_key,
                                            refresh_actual_gen=refresh_actual_gen,
                                            actual_gen_xyz=gen_subtree_xyz,
                                        )
                                        accumulate_compression_terms(
                                            subtree_compression_term_sums,
                                            getattr(loss, "last_compression_terms", {}) or {},
                                            1.0 / num_selected,
                                        )
                                    else:
                                        zero = subtree_xyz.new_zeros(())
                                        L_com_sub = zero
                                        loss_bit_sub = zero
                                        loss_single_sub = zero
                                        loss_nodes_sub = zero

                                L_geom = L_geom + (L_geom_sub / num_selected)
                                L_com = L_com + (L_com_sub / num_selected)
                                L_attr = L_attr + (L_attr_sub / num_selected)
                                L_policy = L_policy + (L_policy_sub / num_selected)
                                L_actuator = L_actuator + (L_actuator_sub / num_selected)
                                Lp_out = Lp_out + (Lp_out_sub / num_selected)
                                La_fit = La_fit + (La_fit_sub / num_selected)
                                La_rep = La_rep + (La_rep_sub / num_selected)
                                loss_bit = loss_bit + (loss_bit_sub / num_selected)
                                loss_single = loss_single + (loss_single_sub / num_selected)
                                loss_nodes = loss_nodes + (loss_nodes_sub / num_selected)
                                gen_xyz = gen_subtree_xyz
                                final_w = final_w_sub
                                out_label = out_label_sub
                            train_edit_stats = _finalize_point_edit_sums(subtree_edit_sums)
                            noise_debug = merge_noise_debug_values(subtree_noise_debug_values)
                            if subtree_compression_term_sums:
                                loss.last_compression_terms = subtree_compression_term_sums
                    finally:
                        args._log_this_step = prev_log_flag
                elif args.split2patch:
                    optimizer.zero_grad(set_to_none=True)
                    patch_info = _get_patch_info(input_pcd, args, cache_key, patch_info_cache)
                    total_patch_count = int(patch_info["num_patches"])
                    subset_enabled = bool(getattr(args, "train_patch_subset_enable", False))
                    selected_patch_ids = torch.arange(
                        total_patch_count,
                        device=patch_info["patch_xyz"].device,
                        dtype=torch.long,
                    )
                    if subset_enabled:
                        is_anchor_step, _ = should_use_full_cloud_anchor(
                            args,
                            global_step=global_train_step,
                            cache_key=cache_key,
                        )
                        if not is_anchor_step:
                            selected_patch_ids = select_patch_subset_ids(patch_info, global_train_step, args)
                    selected_patch_count = int(selected_patch_ids.numel())
                    subset_step = bool(
                        subset_enabled
                        and (not is_anchor_step)
                        and selected_patch_count < total_patch_count
                    )
                    encoder_debug_chunks = [] if detail_log_this_step else None
                    pb = _effective_patch_batch_size(
                        args,
                        patch_count=selected_patch_count,
                        patch_size=args.num_points,
                        is_train=True,
                        writer=writer,
                    )
                    patch_outputs = []
                    patch_count = selected_patch_count
                    geom_weight_sum = 0.0
                    L_geom = input_pcd.new_zeros(())
                    L_attr = input_pcd.new_zeros(())
                    L_policy = input_pcd.new_zeros(())
                    L_actuator = input_pcd.new_zeros(())
                    Lp_out = input_pcd.new_zeros(())
                    La_fit = input_pcd.new_zeros(())
                    La_rep = input_pcd.new_zeros(())
                    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                    with autocast_ctx:
                        prev_patch_geom_log = getattr(args, "_log_this_step", True)
                        args._log_this_step = False
                        try:
                            for i in range(0, patch_count, pb):
                                chunk_patch_ids = selected_patch_ids[i:i+pb]
                                chunk_patch_ids_list = chunk_patch_ids.detach().cpu().tolist()
                                patch_xyz = patch_info["patch_xyz"].index_select(0, chunk_patch_ids)
                                patch_attr = patch_info["patch_attr"].index_select(0, chunk_patch_ids)
                                patch_centroid = patch_info["patch_centroid"].index_select(0, chunk_patch_ids)
                                patch_scale = patch_info["patch_scale"].index_select(0, chunk_patch_ids)
                                patch_cache_keys = [
                                    f"{cache_key}|patch={patch_id}"
                                    for patch_id in chunk_patch_ids_list
                                ]
                                (
                                    gen_chunk,
                                    L_attr_chunk,
                                    L_policy_chunk,
                                    L_actuator_chunk,
                                    final_w_chunk,
                                    Lp_out_chunk,
                                    La_fit_chunk,
                                    La_rep_chunk,
                                    _,
                                    patch_meta_chunk,
                                ) = model.forward(
                                    patch_xyz,
                                    patch_attr,
                                    cache_key=patch_cache_keys,
                                    return_patch_meta=True,
                                    coord_scale=patch_scale,
                                    return_attr_output=False,
                                )
                                if detail_log_this_step:
                                    base_model = model.module if hasattr(model, "module") else model
                                    encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))
                                gen_chunk = denormalize_patch_output(
                                    gen_chunk,
                                    patch_centroid,
                                    patch_scale,
                                )

                                chunk_size = patch_xyz.shape[0]
                                geom_groups = {}
                                L_attr = L_attr + L_attr_chunk * chunk_size
                                L_policy = L_policy + L_policy_chunk * chunk_size
                                L_actuator = L_actuator + L_actuator_chunk * chunk_size
                                Lp_out = Lp_out + Lp_out_chunk * chunk_size
                                La_fit = La_fit + La_fit_chunk * chunk_size
                                La_rep = La_rep + La_rep_chunk * chunk_size

                                for local_idx in range(chunk_size):
                                    patch_id = int(chunk_patch_ids_list[local_idx])
                                    patch_input_idx = patch_info["patch_input_idx"][patch_id]
                                    owned_input_mask = patch_info["owned_input_mask"][patch_id]
                                    anchor_idx_local = patch_meta_chunk["anchor_idx_local"][local_idx].clamp_(0, patch_input_idx.shape[0] - 1)
                                    valid_mask = patch_meta_chunk["output_valid_mask"][local_idx]
                                    owned_output_mask = owned_input_mask.index_select(0, anchor_idx_local)
                                    select_mask = valid_mask & owned_output_mask
                                    selected_pts = gen_chunk[local_idx, :, select_mask]
                                    selected_w = None
                                    if final_w_chunk is not None:
                                        selected_w = final_w_chunk[local_idx, :, select_mask]
                                    represented_owned_mask = torch.zeros_like(owned_input_mask)
                                    if select_mask.any():
                                        represented_owned_mask[anchor_idx_local[select_mask]] = True
                                    missing_owned_mask = owned_input_mask & (~represented_owned_mask)
                                    fallback_pts = None
                                    fallback_w = None
                                    if missing_owned_mask.any():
                                        patch_input_xyz_world = (
                                            patch_info["patch_centroid"][patch_id:patch_id+1]
                                            + patch_info["patch_xyz"][patch_id:patch_id+1] * patch_info["patch_scale"][patch_id:patch_id+1]
                                        )
                                        fallback_pts = patch_input_xyz_world[0, :, missing_owned_mask]
                                        if final_w_chunk is not None:
                                            fallback_w = final_w_chunk.new_ones((1, int(missing_owned_mask.sum().item())))

                                    owned_local_idx = torch.nonzero(owned_input_mask, as_tuple=False).flatten()
                                    owned_global_idx = None
                                    owned_out_label = None
                                    if owned_local_idx.numel() > 0:
                                        owned_global_idx = patch_input_idx.index_select(0, owned_local_idx)
                                        if patch_meta_chunk["out_label"] is not None:
                                            owned_out_label = patch_meta_chunk["out_label"][local_idx, owned_local_idx]

                                    if valid_mask.any():
                                        gen_patch_valid = gen_chunk[local_idx:local_idx+1, :3, valid_mask]
                                        if str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() == "hard":
                                            final_w_owned = None
                                        else:
                                            final_w_owned = None if final_w_chunk is None else final_w_chunk[local_idx:local_idx+1, :, valid_mask]

                                        gt_patch_owned = input_pcd[:, :3, patch_input_idx[owned_input_mask]].contiguous()
                                        local_weight = float(max(int(owned_input_mask.sum().item()), 1))
                                        can_batch_geom = (
                                            owned_out_label is None
                                            or int(torch.count_nonzero(owned_out_label).detach().cpu()) == 0
                                        )
                                        if can_batch_geom:
                                            geom_key = (
                                                int(gen_patch_valid.shape[-1]),
                                                int(gt_patch_owned.shape[-1]),
                                                final_w_owned is not None,
                                            )
                                            group = geom_groups.get(geom_key)
                                            if group is None:
                                                group = {
                                                    "gen": [],
                                                    "gt": [],
                                                    "final_w": [] if final_w_owned is not None else None,
                                                    "weight": 0.0,
                                                }
                                                geom_groups[geom_key] = group
                                            group["gen"].append(gen_patch_valid)
                                            group["gt"].append(gt_patch_owned)
                                            if final_w_owned is not None:
                                                group["final_w"].append(final_w_owned)
                                            group["weight"] += local_weight
                                        else:
                                            out_label_owned = owned_out_label.unsqueeze(0)
                                            L_geom = L_geom + loss.get_geometry_loss(
                                                args,
                                                gen_pts=gen_patch_valid,
                                                gt_pts=gt_patch_owned,
                                                final_w=final_w_owned,
                                                out_label=out_label_owned,
                                            ) * local_weight
                                            geom_weight_sum += local_weight

                                    patch_outputs.append(
                                        {
                                            "patch_id": patch_id,
                                            "selected_pts": selected_pts,
                                            "selected_w": selected_w,
                                            "fallback_pts": fallback_pts,
                                            "fallback_w": fallback_w,
                                            "owned_global_idx": owned_global_idx,
                                            "owned_out_label": owned_out_label,
                                            "patch_meta": {
                                                "anchor_idx_local": anchor_idx_local,
                                                "output_valid_mask": valid_mask,
                                                "out_label": None if patch_meta_chunk["out_label"] is None else patch_meta_chunk["out_label"][local_idx],
                                            },
                                        }
                                    )
                                geom_chunk, geom_chunk_weight = _accumulate_grouped_patch_geometry(
                                    geom_groups,
                                    loss,
                                    args,
                                )
                                if geom_chunk is not None and geom_chunk_weight > 0.0:
                                    L_geom = L_geom + geom_chunk
                                    geom_weight_sum += geom_chunk_weight
                        finally:
                            args._log_this_step = prev_patch_geom_log
                        if subset_step:
                            gen_pts, compression_gt_pts, final_w, out_label = merge_patch_subset_outputs(
                                patch_info,
                                patch_outputs,
                                input_pcd=input_pcd,
                                device=input_pcd.device,
                                dtype=input_pcd.dtype,
                            )
                            compression_cache_key = make_patch_subset_cache_key(
                                cache_key,
                                selected_patch_ids,
                                total_patch_count=total_patch_count,
                            )
                        else:
                            gen_pts, final_w, out_label = merge_patch_outputs(
                                patch_info,
                                patch_outputs,
                                device=input_pcd.device,
                                dtype=input_pcd.dtype,
                            )
                            compression_gt_pts = input_xyz

                        norm = float(max(patch_count, 1))
                        L_attr = L_attr / norm
                        L_policy = L_policy / norm
                        L_actuator = L_actuator / norm
                        Lp_out = Lp_out / norm
                        La_fit = La_fit / norm
                        La_rep = La_rep / norm
                        if geom_weight_sum > 0:
                            L_geom = L_geom / geom_weight_sum
                    gen_xyz = gen_pts[:, :3, :]
                    train_edit_stats = _summarize_point_edits(
                        input_xyz=compression_gt_pts[:, :3, :],
                        gen_pts=gen_pts,
                        final_w=final_w,
                        args=args,
                    )

                else:
                    optimizer.zero_grad(set_to_none=True)
                    args._log_this_step = bool(getattr(args, "verbose_step_logs", False) and detail_log_this_step)
                    encoder_debug_chunks = [] if detail_log_this_step else None
                    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                    with autocast_ctx:
                        gen_patches, L_attr, L_policy, L_actuator, final_w, Lp_out, La_fit, La_rep, out_label = model.forward(
                            patches,
                            None,
                            cache_key=cache_key,
                            coord_scale=fd_xyz,
                            return_attr_output=False,
                        )
                    if detail_log_this_step:
                        base_model = model.module if hasattr(model, "module") else model
                        encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))

                    # 元スケールに戻す
                    gen_xyz = centroid_xyz + gen_patches[:, :3, :] * fd_xyz
                    gen_pts = gen_xyz.contiguous()
                    gen_xyz = gen_pts[:, :3, :]
                    train_edit_stats = _summarize_point_edits(
                        input_xyz=input_xyz[:, :3, :],
                        gen_pts=gen_pts,
                        final_w=final_w,
                        args=args,
                    )
                    L_geom = None

                if timing_enabled:
                    _sync_for_timing(use_cuda)
                    timing_model_end = time.time()

                # ---------- Loss計算と最適化 ----------
                if timing_enabled:
                    timing_loss_start = time.time()
                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                with autocast_ctx:
                    final_w_for_loss = None
                    if str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() != "hard":
                        final_w_for_loss = final_w
                    if timing_enabled:
                        _sync_for_timing(use_cuda)
                        timing_noise_start = time.time()
                    if subtree_mode:
                        compression_gen_xyz = gen_xyz
                    else:
                        # 入力や診断前ではなく、編集後・量子化前にだけ一様ノイズを加える。
                        # 形状損失はcleanなgen_xyz、rate/structure損失はcompression_gen_xyzを見る。
                        compression_gen_xyz, noise_debug = prepare_compression_points(
                            gen_xyz,
                            args,
                            model,
                            collect_stats=bool(log_this_step or profile_this_step),
                        )
                    if timing_enabled:
                        _sync_for_timing(use_cuda)
                        timing_noise_end = time.time()
                    if subtree_mode:
                        pass
                    elif args.split2patch:
                        if compute_compression:
                            L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss(
                                args,
                                gen_xyz=compression_gen_xyz,
                                gt_xyz=compression_gt_pts[:, :3, :],
                                final_w=final_w_for_loss,
                                cache_key=compression_cache_key,
                                refresh_actual_gen=refresh_actual_gen,
                                actual_gen_xyz=gen_xyz,
                            )
                        else:
                            zero = gen_xyz.new_zeros(())
                            L_com = zero
                            loss_bit = zero
                            loss_single = zero
                            loss_nodes = zero
                            loss.last_compression_debug = {}
                            loss.last_compression_terms = {}
                    else:
                        L_geom = loss.get_geometry_loss(
                            args,
                            gen_pts=gen_xyz,
                            gt_pts=input_xyz,
                            final_w=final_w_for_loss,
                            out_label=out_label,
                        )
                        if compute_compression:
                            L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss(
                                args,
                                gen_xyz=compression_gen_xyz,
                                gt_xyz=input_xyz[:, :3, :],
                                final_w=final_w_for_loss,
                                cache_key=cache_key,
                                refresh_actual_gen=refresh_actual_gen,
                                actual_gen_xyz=gen_xyz,
                            )
                        else:
                            zero = gen_xyz.new_zeros(())
                            L_com = zero
                            loss_bit = zero
                            loss_single = zero
                            loss_nodes = zero
                            loss.last_compression_debug = {}
                            loss.last_compression_terms = {}

                if compute_compression:
                    comp_debug_for_noise = getattr(loss, "last_compression_debug", {}) or {}
                    comp_debug_for_noise.update(
                        {
                            "uniform_noise_enabled": bool(noise_debug.get("enabled", False)),
                            "uniform_noise_applied": bool(noise_debug.get("applied", False)),
                            "uniform_noise_delta": float(noise_debug.get("delta", 0.0)),
                            "uniform_noise_mean_abs": float(noise_debug.get("mean_abs", 0.0)),
                            "compression_input_noisy": bool(noise_debug.get("applied", False)),
                        }
                    )
                    loss.last_compression_debug = comp_debug_for_noise

                actual_total_bit_backend = _uses_actual_total_bit_objective(args)
                if actual_total_bit_backend:
                    L_com_objective = float(getattr(args, "w_com", 1.0)) * L_com
                else:
                    terms = getattr(loss, "last_compression_terms", {}) or {}
                    bit_term = terms.get("bit", L_com.new_zeros(()))
                    single_term = terms.get("single", L_com.new_zeros(()))
                    node_term = terms.get("node", L_com.new_zeros(()))
                    bpn_term = terms.get("bpn", L_com.new_zeros(()))
                    sparsepcgc_term = terms.get("sparsepcgc", L_com.new_zeros(()))
                    lowprob_term = La_fit if torch.is_tensor(La_fit) else L_com.new_zeros(())
                    L_com_objective = float(getattr(args, "w_com", 1.0)) * (
                        float(getattr(args, "com_bit", 0.0)) * bit_term
                        + float(getattr(args, "com_sin", 0.0)) * single_term
                        + float(getattr(args, "com_node", 0.0)) * node_term
                        + float(getattr(args, "com_bpn", 0.0)) * bpn_term
                        + float(getattr(args, "com_sparsepcgc", 0.0)) * sparsepcgc_term
                        + float(getattr(args, "com_lowprob", 0.0)) * lowprob_term
                    )
                L_downstream = (
                    stage_factors["geom"] * args.w_geom * L_geom
                    + stage_factors["com"] * L_com_objective
                )
                L = (
                    L_downstream
                    + stage_factors["attr"] * args.w_attr * L_attr
                    + stage_factors["policy"] * args.w_policy * L_policy
                    + stage_factors["repair"] * args.w_actuator * L_actuator
                )
                L_discrete_policy = L.new_zeros(())
                if str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() == "hard":
                    policy_loss_fn = getattr(model, "discrete_policy_loss", None)
                    if callable(policy_loss_fn):
                        L_discrete_policy = policy_loss_fn(L_downstream.detach())
                        L = L + L_discrete_policy

                if log_this_step:
                    comp_debug = getattr(loss, "last_compression_debug", {}) or {}
                    base_model = model.module if hasattr(model, "module") else model
                    structure_debug = getattr(base_model, "last_structure_debug", {}) or {}

                    log_step_loss(
                        writer,
                        step,
                        num_steps,
                        L,
                        L_geom,
                        L_com,
                        L_com_objective,
                        L_attr,
                        L_policy,
                        L_actuator,
                        Lp_out,
                        La_fit,
                        La_rep,
                        L_discrete_policy,
                        loss_bit,
                        loss_single,
                        loss_nodes,
                    )

                    log_compression_stats(
                        writer,
                        step,
                        num_steps,
                        comp_debug,
                    )

                    before_node, after_node, before_single, after_single = log_compression_train_debug(
                        writer,
                        step,
                        num_steps,
                        args,
                        comp_debug,
                        loss,
                        L_com,
                    )

                    log_codec_actual_correlation(
                        writer,
                        step,
                        num_steps,
                        args,
                        comp_debug,
                        codec_actual_metric_pairs,
                        before_node,
                        after_node,
                        before_single,
                        after_single,
                    )

                    log_sparsepcgc_train_debug(
                        writer,
                        step,
                        num_steps,
                        args,
                        comp_debug,
                        sparsepcgc_proxy_actual_pairs,
                    )

                    if structure_debug:
                        log_structure_debug(
                            writer,
                            structure_debug,
                            step,
                            num_steps,
                        )

                        _write_structure_decision_debug(
                            writer,
                            f"StructureDecision step={step + 1}/{num_steps}",
                            structure_debug,
                        )
                if timing_enabled:
                    _sync_for_timing(use_cuda)
                    timing_loss_end = time.time()
                
                """--- どのlossがどの勾配を作っているか確認 ---"""
                # backward_and_measure("geom", args.w_geom * L_geom, model, optimizer, writer, args)                
                # backward_and_measure("com", args.w_com  * L_com,  model, optimizer, writer, args)
                # backward_and_measure("attr", args.w_attr * L_attr, model, optimizer, writer, args)
                # backward_and_measure("policy" , args.w_policy  * L_policy,  model, optimizer, writer, args)

                step_completed = False
                total_loss_finite = bool(torch.isfinite(L.detach()).all().item())
                param_update_snapshots = None
                if total_loss_finite:
                    param_update_snapshots = _capture_param_update_snapshots(
                        args,
                        model,
                        step + 1,
                        num_steps,
                    )
                if not total_loss_finite:
                    writer.write(
                        f"Skipped optimizer step due to non-finite total loss at "
                        f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}."
                    )
                elif amp_scaler_enabled:
                    scale_before = float(scaler.get_scale())
                    scaler.scale(L).backward()
                    scaler.unscale_(optimizer)
                    grad_clip = float(getattr(args, "train_grad_clip", 0.0))
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_norm=grad_clip,
                        )
                    if bool(getattr(args, "debug_grad_flow", False)):
                        _log_grad_flow(args, writer, model, step + 1, num_steps)
                    scaler.step(optimizer)
                    optimizer_state = scaler._per_optimizer_states[id(optimizer)]
                    found_inf = 0.0
                    if optimizer_state["found_inf_per_device"]:
                        found_inf = float(
                            sum(v.item() for v in optimizer_state["found_inf_per_device"].values())
                        )
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    step_completed = found_inf == 0.0 and scale_after >= scale_before
                    if step_completed:
                        consecutive_amp_skips = 0
                    else:
                        consecutive_amp_skips += 1
                        if consecutive_amp_skips >= amp_overflow_patience:
                            consecutive_amp_skips = 0
                            if use_cuda and _cuda_bf16_ops_safe():
                                amp_dtype = torch.bfloat16
                                amp_scaler_enabled = False
                                writer.write(
                                    "float16 AMP overflow persisted; switched AMP autocast to bfloat16."
                                )
                            else:
                                use_amp = False
                                amp_scaler_enabled = False
                                scaler = torch.cuda.amp.GradScaler(enabled=False)
                                writer.write(
                                    "float16 AMP overflow persisted; disabled AMP and continue in float32."
                                )
                else:
                    L.backward()
                    grad_clip = float(getattr(args, "train_grad_clip", 0.0))
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_norm=grad_clip,
                        )
                    _log_grad_flow(args, writer, model, step + 1, num_steps)
                    optimizer.step()
                    step_completed = True
                    consecutive_amp_skips = 0
                if step_completed:
                    _log_param_updates(
                        args,
                        writer,
                        model,
                        param_update_snapshots,
                        step + 1,
                        num_steps,
                    )
                if timing_enabled:
                    _sync_for_timing(use_cuda)
                    timing_step_end = time.time()
                epoch_has_optimizer_step = epoch_has_optimizer_step or step_completed
                
                if epoch_metric_sums is None:
                    epoch_metric_sums = _new_metric_sums(L.device, plot.num_loss)
                _add_metric_sums(
                    epoch_metric_sums,
                        [
                            L,
                            L_geom,
                            L_com,
                            L_attr,
                            L_policy,
                            loss_single,
                        loss_nodes,
                        Lp_out,
                        La_fit,
                        La_rep,
                        L_actuator,
                        *_surrogate_plot_metrics(loss),
                    ],
                    L.device,
                )
                if episode_metric_sums is None:
                    episode_metric_sums = _new_metric_sums(L.device, plot.num_loss)
                step_metric_values = [
                    L,
                    L_geom,
                    L_com,
                    L_attr,
                    L_policy,
                    loss_single,
                    loss_nodes,
                    Lp_out,
                    La_fit,
                    La_rep,
                    L_actuator,
                    *_surrogate_plot_metrics(loss),
                ]
                _add_metric_sums(episode_metric_sums, step_metric_values, L.device)
                if train_edit_stats is None:
                    train_edit_stats = _summarize_point_edits(
                        input_xyz=input_xyz[:, :3, :],
                        gen_pts=gen_pts,
                        final_w=final_w,
                        args=args,
                    )
                plot.record_point_edits("step", global_train_step + 1, train_edit_stats)
                plot_step_info = plot.record_metrics("step", global_train_step + 1, step_metric_values)
                if plot_step_info.get("skipped", False):
                    threshold_text = f"{plot_step_info.get('threshold', float('nan')):.6g}"
                    baseline = plot_step_info.get("baseline", None)
                    baseline_text = ""
                    if baseline is not None:
                        baseline_text = f", baseline={float(baseline):.6g}"
                    writer.write(
                        "PlotSkipStep: "
                        f"global_step={global_train_step + 1}, "
                        f"episode={episode + 1}, "
                        f"epoch={epoch + 1}, "
                        f"metric={plot_step_info.get('metric_key', 'unknown')}, "
                        f"value={float(plot_step_info.get('value', float('nan'))):.6g}, "
                        f"rule={plot_step_info.get('reason', 'unknown')}, "
                        f"threshold={threshold_text}"
                        f"{baseline_text}"
                    )
                if timing_enabled:
                    _sync_for_timing(use_cuda)
                    en_step = time.time()

                    log_step_timing(
                        writer=writer,
                        args=args,
                        step=step,
                        num_steps=num_steps,
                        epoch=epoch,
                        global_train_step=global_train_step,
                        use_cuda=use_cuda,
                        st_step=st_step,
                        timing_data_start=timing_data_start,
                        timing_data_end=timing_data_end,
                        timing_model_start=timing_model_start,
                        timing_model_end=timing_model_end,
                        timing_noise_start=timing_noise_start,
                        timing_noise_end=timing_noise_end,
                        timing_loss_start=timing_loss_start,
                        timing_loss_end=timing_loss_end,
                        timing_step_end=timing_step_end,
                        en_step=en_step,
                        loss=loss,
                        model=model,
                        KNN_BACKEND=KNN_BACKEND,
                    )
                else:
                    en_step = time.time()
                if log_this_step:
                    log_point_edit_stats(
                        writer,
                        train_edit_stats,
                        step,
                        num_steps,
                    )
                    print(
                        f"Epi{episode + 1}/Epo{epoch + 1}/Step{step + 1}:"
                        f"{en_step-st_step:.4f}s   |   "
                        f"{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}"
                    )
                global_train_step += 1

            # lr scheduler
            if epoch_has_optimizer_step:
                scheduler_steplr.step()
            else:
                writer.write("No successful optimizer step in this epoch; lr_scheduler.step() was skipped.")

            # ログの記録
            if epoch_metric_sums is not None:
                epoch_avgs = _metric_avgs_to_floats(epoch_metric_sums)
                plot.epo_avg = epoch_avgs
                plot_epoch_info = plot.record_metrics("epo", global_epoch + 1, epoch_avgs)
                log_plot_skip_epoch(
                    writer,
                    plot_epoch_info,
                    global_epoch,
                )
                writer.write(_format_metric_summary("EpochAvg", plot.metric_keys, epoch_avgs))
            epoch_edit_info = plot.record_point_edits("epo", global_epoch + 1)
            log_epoch_point_edit_average(
                writer,
                epoch_edit_info,
                global_epoch,
            )
            global_epoch += 1
            plot.plot_loss_curve("step")
            plot.plot_loss_curve("epo")
            plot.plot_point_edit_curve("step")
            plot.plot_point_edit_curve("epo")
            writer.write(f"Saved step/epoch plots/csv: {plot.save_dir}")
            writer.flush()
        if episode_metric_sums is not None:
            plot.epi_avg = _metric_avgs_to_floats(episode_metric_sums)
            plot_episode_info = plot.record_metrics("epi", episode + 1, plot.epi_avg)
            log_plot_skip_episode(
                writer,
                plot_episode_info,
                episode,
            )
        else:
            plot.epi_avg = [None for _ in range(plot.num_loss)]
        writer.write(_format_metric_summary("EpisodeAvg", plot.metric_keys, plot.epi_avg))
        episode_edit_info = plot.record_point_edits("epi", episode + 1)
        log_episode_point_edit_average(
            writer,
            episode_edit_info,
            episode,
        )
        plot.plot_loss_curve("epi")
        plot.plot_point_edit_curve("epi")
        writer.write(f"Saved episode plots/csv: {plot.save_dir}")
        writer.flush()

        # 毎エピソードと最高スコアのモデルを保存
        best_loss, model_path = save_episode_checkpoint(
            model=model,
            ckpt_dir=ckpt_dir,
            plot=plot,
            writer=writer,
            episode=episode,
            best_loss=best_loss,
        )
        if notifier is not None:
            notifier.episode_finished(
                episode=episode + 1,
                total_episodes=args.episodes,
                loss_value=float(plot.epi_loss_return()),
                model_path=model_path,
                log_path=getattr(writer, "file_path", None),
            )

    return best_loss

if __name__ == '__main__':
    """=== セットアップ ==="""
    setup_t0 = time.time()
    # トレーニングInfoのセットアップ
    file_day = datetime.datetime.now().strftime('%Y%m%d')
    file_time = datetime.datetime.now().strftime('%H%M%S')

    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--trainORtest', default="train", type=str, help='date')
    args = parse_pugan_args(parser, file_day, file_time)
    requested_mp_method = str(getattr(args, "mp_start_method", "auto")).strip().lower()
    if requested_mp_method != "auto":
        current_mp_method = mp.get_start_method(allow_none=True)
        if current_mp_method != requested_mp_method:
            mp.set_start_method(requested_mp_method, force=True)

    if torch.cuda.is_available() and not args.cpu and args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not bool(getattr(args, "deterministic", False))
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    
    # ログのセットアップ
    writer = Writing(
        args,
        file_day,
        file_time,
        filename="MyNetwork_train",
        flush_every=args.log_flush_every,
        sync_every=args.log_sync_every,
        log_root=args.log_root,
    )
    writer.write(f"SetupTiming: writer_init={time.time() - setup_t0:.3f}s")
    setup_plot_t0 = time.time()
    plot = PlotMaker(args)
    writer.write(f"SetupTiming: plot_init={time.time() - setup_plot_t0:.3f}s")

    log_training_setup(
        writer,
        args,
        file_day,
        file_time,
    )

    writer.write(
        f"Two-Stage Training: enabled={bool(getattr(args, 'two_stage_training', False))}, "
        f"base_stage={args.training_stage}, diagnosis_ratio={getattr(args, 'diagnosis_episode_ratio', 0.0)}, "
        f"diagnosis_episodes={getattr(args, 'diagnosis_episodes', 0)}"
    )
    writer.write(f"Module BatchNorm Running Stats: {args.module_bn_use_running_stats}")
    notifier = TrainingMailNotifier.from_args(args, writer=writer)

    setup_model_t0 = time.time()
    model = Network(args, writer)
    writer.write(f"SetupTiming: model_init={time.time() - setup_model_t0:.3f}s")

    setup_ckpt_t0 = time.time()
    repkpu_ckpt = os.path.join(os.path.dirname(__file__), "repkpu_model", "ckpt-best.pth")
    ckpt = torch.load(repkpu_ckpt, map_location="cpu")
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in ckpt.items()
        if k.startswith("encoder.")
    }
    encoder_state = _adapt_encoder_state_dict_for_sparse_input(model, encoder_state, writer=writer)
    model.encoder.load_state_dict(encoder_state, strict=False)
    for p in model.encoder.parameters():
        p.requires_grad = False
    writer.write("RepKPU encoder loaded: repkpu_model/ckpt-best.pth")
    writer.write(f"SetupTiming: encoder_ckpt_load={time.time() - setup_ckpt_t0:.3f}s")

    if args.cpu is False and torch.cuda.is_available():
        setup_cuda_t0 = time.time()
        model = model.cuda()
        writer.write(f"SetupTiming: model_to_cuda={time.time() - setup_cuda_t0:.3f}s")

    setup_loss_t0 = time.time()
    loss = Loss(args, file_day + "-" + file_time, writer)
    writer.write(f"SetupTiming: loss_init={time.time() - setup_loss_t0:.3f}s")
    writer.write(f"SetupTiming: total_before_train={time.time() - setup_t0:.3f}s")

    st = time.time()
    writer.write("=== Start Training ===")
    notifier.training_started(
        start_date=datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'),
        log_path=getattr(writer, "file_path", None),
    )
    best_loss = None
    try:
        best_loss = train(model, args, loss, writer, plot, notifier=notifier)
        en = time.time()
        finish_date = datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S')
        writer.write(f"Training time: {en - st}")
        writer.write(f"Date of finishing training: {finish_date}")
        notifier.training_finished(
            elapsed_sec=en - st,
            finish_date=finish_date,
            best_loss=best_loss,
            log_path=getattr(writer, "file_path", None),
        )
    except Exception as exc:
        try:
            writer.write(f"Training error: {type(exc).__name__}: {exc}")
        finally:
            notifier.training_error(exc, log_path=getattr(writer, "file_path", None))
        raise
    finally:
        writer.close()
