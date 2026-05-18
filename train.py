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
import csv
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

CASE_DEBUG_COLUMNS = [
    "case_type",
    "global_step",
    "episode",
    "epoch",
    "step",
    "sample_name",
    "codec",
    "actual_delta",
    "surrogate_delta",
    "surrogate_abs_error",
    "surrogate_signed_error",
    "actual_bits_before",
    "actual_bits_after",
    "point_count_before",
    "point_count_after",
    "unique_coord_before",
    "unique_coord_after",
    "active_coord_before",
    "active_coord_after",
    "octree_node_before",
    "octree_node_after",
    "single_before",
    "single_after",
    "add_points",
    "delete_points",
    "adjust_points",
    "preserve_ratio",
    "same_voxel_adjust",
    "different_voxel_move",
    "move_source_emptied",
    "move_target_new",
    "move_source_not_emptied",
    "shape_loss",
    "compression_loss",
    "actuator_loss",
    "total_loss",
    "teacher_refresh",
    "teacher_target_age",
]

COMPRESSION_METRIC_COLUMNS = [
    "global_step",
    "episode",
    "epoch",
    "step",
    "stage",
    "codec",
    "backend",
    "actual_value_source",
    "fresh_actual",
    "cached_actual",
    "actual_total_bit_percent",
    "actual_total_bit_percent_fresh",
    "actual_total_bit_percent_cached",
    "compression_loss_L_com",
    "lcom_main",
    "lcom_aux",
    "lcom_sparsepcgc_aux",
    "sparsepcgc_aux_raw",
    "sparsepcgc_aux_weighted",
    "lcom_without_sparsepcgc_aux",
    "lcom_with_sparsepcgc_aux",
    "lcom_objective",
    "com_sparsepcgc_weight",
    "sparsepcgc_aux_weight",
    "sparsepcgc_active_coord_loss",
    "sparsepcgc_isolated_loss",
    "sparsepcgc_entropy_loss",
    "sparsepcgc_density_loss",
    "sparsepcgc_single_aux",
    "sparsepcgc_node_aux",
    "compression_objective",
    "compression_main_loss",
    "compression_aux_loss",
    "sparsepcgc_aux_loss",
    "surrogate_pred_bit_percent",
    "surrogate_target_bit_percent",
    "surrogate_abs_bit_error",
    "surrogate_signed_bit_error",
    "surrogate_train_loss",
    "proxy_delta_percent",
    "actual_bits_before",
    "actual_bits_after",
    "point_count_before",
    "point_count_after",
    "unique_coord_before",
    "unique_coord_after",
    "node_delta",
    "single_delta",
    "teacher_refresh",
    "teacher_cache_hit",
    "teacher_target_age",
    "actual_codec_disabled",
    "actual_codec_skipped_by_interval",
    "actual_codec_fallback_to_proxy",
    "loss_mode",
    "cp_main_source",
    "cp_warmup",
    "cp_L_com_main",
    "cp_L_com_primary",
    "cp_P_geom",
    "cp_P_single",
    "cp_P_nodes",
    "cp_P_sparsepcgc",
    "cp_P_actuator",
    "cp_P_op",
    "cp_total",
    "cp_main_requires_grad",
    "cp_geom_requires_grad",
    "cp_single_requires_grad",
    "cp_nodes_requires_grad",
    "cp_sparsepcgc_requires_grad",
    "cp_actuator_requires_grad",
    "cp_op_requires_grad",
    "cp_main_finite",
    "cp_geom_finite",
    "cp_single_finite",
    "cp_nodes_finite",
    "cp_sparsepcgc_finite",
    "cp_actuator_finite",
    "cp_op_finite",
    "corr_surrogate_actual",
    "corr_lcom_actual",
    "corr_cp_main_actual",
    "corr_sparsepcgc_aux_actual",
    "corr_lcom_without_sparsepcgc_aux_actual",
    "sign_match_surrogate_actual",
    "sign_match_lcom_actual",
    "sign_match_cp_main_actual",
    "sign_match_sparsepcgc_aux_actual",
    "sign_match_lcom_without_sparsepcgc_aux_actual",
    "rolling_corr_window",
    "rolling_sign_match_window",
    "active_coord_before",
    "active_coord_after",
    "active_coord_delta",
    "isolated_voxel_count",
    "isolated_voxel_delta",
    "sparse_density_before",
    "sparse_density_after",
    "sparse_density_delta",
    "occupancy_entropy",
    "occupancy_nll_proxy",
    "lowprob_occupancy_ratio",
    "entropy_delta",
    "nll_delta",
]

COMPRESSION_EPISODE_METRIC_COLUMNS = [
    "episode",
    "stage",
    "codec",
    "backend",
    "row_count",
    "fresh_actual_count",
] + [
    key for key in COMPRESSION_METRIC_COLUMNS
    if key not in {"global_step", "episode", "epoch", "step", "stage", "codec", "backend", "fresh_actual"}
]

OPERATION_METRIC_COLUMNS = [
    "global_step",
    "episode",
    "epoch",
    "step",
    "stage",
    "codec",
    "fresh_actual",
    "actual_total_bit_percent",
    "train_or_eval_mode",
    "hardening_mode",
    "selection_threshold",
    "topk_selected_count",
    "sparsepcgc_add_experiment_enabled",
    "add_enabled",
    "prune_enabled",
    "disp_enabled",
    "repair_ratio",
    "preserve_ratio",
    "add_prob_mean",
    "add_prob_max",
    "add_priority_mean",
    "add_priority_max",
    "add_score_mean",
    "add_score_max",
    "add_ratio",
    "add_candidate_ratio",
    "add_candidate_count",
    "add_hard_count",
    "add_effective_count",
    "add_actual_point_count",
    "add_target_voxels",
    "add_target_ratio",
    "add_max_ratio",
    "add_warmup",
    "soft_add_count",
    "hard_add_count",
    "drop_prob_mean",
    "hard_drop_ratio",
    "hard_drop_count",
    "delete_target_voxels",
    "delete_emptied_voxels",
    "move_score_mean",
    "hard_move_ratio",
    "hard_move_count",
    "move_source_voxels",
    "move_target_voxels",
    "move_source_emptied",
    "move_target_new",
    "move_source_not_emptied",
    "same_voxel_adjust",
    "different_voxel_move",
    "input_points",
    "pre_output_points",
    "output_points",
    "added_ratio_percent",
    "deleted_ratio_percent",
    "adjusted_ratio_percent",
    "codec_points_after",
    "codec_points_before",
    "codec_unique_after",
    "codec_unique_before",
    "unique_coord_delta",
    "add_after_quant_unique_count",
    "add_removed_by_unique_count",
    "active_coord_before",
    "active_coord_after",
    "active_coord_delta",
    "isolated_voxel_count",
    "isolated_voxel_delta",
    "sparse_density_before",
    "sparse_density_after",
    "sparse_density_delta",
    "occupancy_entropy",
    "occupancy_nll_proxy",
    "lowprob_occupancy_ratio",
    "entropy_delta",
    "nll_delta",
    "depth_node_count_summary",
    "depth_single_child_count_summary",
    "depth_entropy_summary",
    "depth_lowprob_summary",
    "subtree_depth",
    "subtree_node_count",
    "subtree_single_child_count",
    "single_child_delta",
    "cp_L_com_main",
    "cp_total",
]

OPERATION_EPISODE_METRIC_COLUMNS = [
    "episode",
    "stage",
    "codec",
    "row_count",
    "fresh_actual_count",
] + [
    key for key in OPERATION_METRIC_COLUMNS
    if key not in {"global_step", "episode", "epoch", "step", "stage", "codec", "fresh_actual"}
]

CHECKPOINT_METRIC_COLUMNS = [
    "episode",
    "stage",
    "total_loss",
    "geom_loss",
    "compression_loss_L_com",
    "repair_loss",
    "single_loss",
    "node_loss",
    "fresh_actual_delta",
    "fresh_actual_count",
    "cached_actual_delta",
    "cached_actual_count",
    "surrogate_pred_bit_percent",
    "proxy_delta_percent",
    "corr_surrogate_actual",
    "corr_lcom_actual",
    "corr_cp_main_actual",
    "corr_sparsepcgc_aux_actual",
    "corr_lcom_without_sparsepcgc_aux_actual",
    "sign_match_surrogate_actual",
    "sign_match_lcom_actual",
    "sign_match_cp_main_actual",
    "sign_match_sparsepcgc_aux_actual",
    "sign_match_lcom_without_sparsepcgc_aux_actual",
    "lcom_main",
    "lcom_aux",
    "lcom_sparsepcgc_aux",
    "sparsepcgc_aux_raw",
    "sparsepcgc_aux_weighted",
    "lcom_without_sparsepcgc_aux",
    "lcom_with_sparsepcgc_aux",
    "sparsepcgc_active_coord_loss",
    "sparsepcgc_isolated_loss",
    "sparsepcgc_entropy_loss",
    "sparsepcgc_density_loss",
    "geometry_ok",
    "safety_ok",
    "repair_ok",
    "node_ok",
    "single_ok",
    "operation_ok",
    "geom_reference",
    "repair_reference",
    "added_ratio_percent",
    "deleted_ratio_percent",
    "adjusted_ratio_percent",
    "active_coord_delta",
    "unique_coord_delta",
    "add_effective_count",
]

CHECKPOINT_AVG_KEYS = [
    "total_loss",
    "geom_loss",
    "compression_loss_L_com",
    "repair_loss",
    "single_loss",
    "node_loss",
    "actual_total_bit_percent_fresh",
    "actual_total_bit_percent_cached",
    "surrogate_pred_bit_percent",
    "proxy_delta_percent",
    "added_ratio_percent",
    "deleted_ratio_percent",
    "adjusted_ratio_percent",
    "add_prob_mean",
    "drop_prob_mean",
    "hard_move_ratio",
    "corr_surrogate_actual",
    "corr_lcom_actual",
    "corr_cp_main_actual",
    "corr_sparsepcgc_aux_actual",
    "corr_lcom_without_sparsepcgc_aux_actual",
    "sign_match_surrogate_actual",
    "sign_match_lcom_actual",
    "sign_match_cp_main_actual",
    "sign_match_sparsepcgc_aux_actual",
    "sign_match_lcom_without_sparsepcgc_aux_actual",
    "lcom_main",
    "lcom_aux",
    "lcom_sparsepcgc_aux",
    "sparsepcgc_aux_raw",
    "sparsepcgc_aux_weighted",
    "lcom_without_sparsepcgc_aux",
    "lcom_with_sparsepcgc_aux",
    "sparsepcgc_active_coord_loss",
    "sparsepcgc_isolated_loss",
    "sparsepcgc_entropy_loss",
    "sparsepcgc_density_loss",
    "active_coord_delta",
    "unique_coord_delta",
    "add_effective_count",
]

SURROGATE_PRETRAIN_COLUMNS = [
    "surrogate_pretrain_step",
    "pretrain_mode",
    "pretrain_teacher_type",
    "sample_name",
    "codec",
    "backend",
    "surrogate_pretrain_loss",
    "surrogate_pretrain_pred_bit_percent",
    "surrogate_pretrain_actual_bit_percent",
    "surrogate_pretrain_abs_error",
    "surrogate_pretrain_corr",
    "surrogate_pretrain_sign_match",
    "surrogate_pretrain_teacher_refresh",
    "surrogate_pretrain_target_age",
    "surrogate_pretrain_fresh_actual_count",
    "pretrain_step_time",
    "pretrain_actual_eval_time",
    "pretrain_surrogate_update_time",
    "pretrain_data_time",
    "pretrain_subtree_sampling_time",
    "pretrain_model_time",
    "pretrain_log_time",
    "pretrain_eta_seconds",
    "pretrain_gpu_alloc_mb",
    "pretrain_cpu_rss_mb",
    "teacher_mode",
    "teacher_refreshed",
    "teacher_replayed",
    "teacher_stale",
    "teacher_skipped",
    "teacher_target_age",
    "replay_buffer_size",
    "replay_sample_count",
    "fresh_actual_count",
    "sparsepcgc_debug_collected",
    "sparsepcgc_debug_time",
    "pretrain_subtree_enabled",
    "pretrain_subtree_depth",
    "pretrain_subtree_point_count",
    "pretrain_subtree_bbox_min",
    "pretrain_subtree_bbox_max",
    "pretrain_subtree_retry_count",
    "pretrain_subtree_skip_reason",
    "pretrain_subtree_key",
    "pretrain_subtree_total_count",
    "pretrain_subtree_eligible_count",
    "pretrain_subtree_selected_count",
    "pretrain_full_calibration",
    "pretrain_actual_scope",
    "surrogate_param_norm",
    "surrogate_pretrain_lr",
]


def _case_float(value, default=0.0):
    if torch.is_tensor(value):
        try:
            return float(value.detach().cpu())
        except Exception:
            return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _case_int(value, default=0):
    try:
        return int(round(_case_float(value, default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _format_duration_seconds(seconds):
    seconds = _case_float(seconds, 0.0)
    if not math.isfinite(seconds) or seconds < 0.0:
        seconds = 0.0
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _process_rss_mb():
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            parts = handle.read().strip().split()
        if len(parts) >= 2:
            return float(int(parts[1]) * page_size) / float(1024 ** 2)
    except Exception:
        return None
    return None


def _cuda_alloc_mb(use_cuda):
    if use_cuda and torch.cuda.is_available():
        return float(torch.cuda.memory_allocated()) / float(1024 ** 2)
    return None


def _surrogate_param_norm(loss):
    surrogate = getattr(loss, "compression_surrogate", None)
    if surrogate is None:
        return None
    total_sq = 0.0
    with torch.no_grad():
        for param in surrogate.parameters():
            if param is None:
                continue
            try:
                total_sq += float(param.detach().float().pow(2).sum().cpu())
            except Exception:
                return None
    return math.sqrt(total_sq) if total_sq >= 0.0 else None


def _format_xyz_triplet(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().flatten().cpu().tolist()
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(values) < 3:
        return None
    return ",".join(f"{values[i]:.6g}" for i in range(3))


def _mean_finite(values):
    finite_values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite_values:
        return None
    return sum(finite_values) / float(len(finite_values))


def _summarize_octree_level_debug(level_debug, value_key):
    if not level_debug:
        return None
    chunks = []
    for item in level_debug:
        if not isinstance(item, dict):
            continue
        level = item.get("level", None)
        value = item.get(value_key, None)
        if level is None or value is None:
            continue
        try:
            chunks.append(f"d{int(level)}:{float(value):.6g}")
        except (TypeError, ValueError):
            continue
    return ";".join(chunks) if chunks else None


def _infer_octree_depth_from_xyz(input_xyz, args):
    """
    input_xyz から点群全体をOctree分割したときの最大深さを推定する。
    量子化座標を想定し、最大座標スパンから ceil(log2(span + 1)) を計算する。
    失敗した場合は args 側の既存設定にフォールバックする。
    """
    fallback_candidates = [
        "octree_depth",
        "max_octree_depth",
        "max_depth",
        "depth",
        "bitdepth",
        "bit_depth",
        "coord_bit_depth",
        "train_subtree_level_max",
        "train_subtree_level_min",
    ]

    fallback_depth = None
    for name in fallback_candidates:
        value = getattr(args, name, None)
        try:
            value = int(value)
            if value > 0:
                fallback_depth = value
                break
        except (TypeError, ValueError):
            pass

    if input_xyz is None:
        return fallback_depth if fallback_depth is not None else 1

    try:
        with torch.no_grad():
            xyz = input_xyz.detach()
            if xyz.dim() == 2:
                # [3, N] or [N, 3] を想定
                if xyz.shape[0] == 3:
                    coord_min = xyz.amin(dim=1)
                    coord_max = xyz.amax(dim=1)
                else:
                    coord_min = xyz.amin(dim=0)
                    coord_max = xyz.amax(dim=0)
            elif xyz.dim() == 3:
                # [B, 3, N] を想定
                coord_min = xyz[:, :3, :].amin(dim=2)
                coord_max = xyz[:, :3, :].amax(dim=2)
            else:
                return fallback_depth if fallback_depth is not None else 1

            max_span = coord_max.sub(coord_min).amax()
            if torch.is_tensor(max_span):
                max_span = float(max_span.detach().cpu())

            if not math.isfinite(max_span) or max_span <= 0:
                return fallback_depth if fallback_depth is not None else 1

            # 座標範囲が 0〜1023 なら span=1023, span+1=1024, depth=10
            estimated_depth = int(math.ceil(math.log2(max(max_span + 1.0, 2.0))))
            estimated_depth = max(1, estimated_depth)

            # args側に明示的な最大depthがある場合は、それを超えないようにする
            if fallback_depth is not None:
                estimated_depth = min(estimated_depth, int(fallback_depth))

            return estimated_depth
    except Exception:
        return fallback_depth if fallback_depth is not None else 1


def _with_pretrain_subtree_depth_overrides(args, callback, input_xyz=None):
    saved = {
        "train_subtree_level_min": getattr(args, "train_subtree_level_min", 0),
        "train_subtree_level_max": getattr(args, "train_subtree_level_max", 0),
        "train_subtree_randomize_level": getattr(args, "train_subtree_randomize_level", False),
        "_train_subtree_depth_cli_override": getattr(args, "_train_subtree_depth_cli_override", False),
    }
    try:
        # 点群全体をOctree分割したときの最大深さを推定
        full_octree_depth = _infer_octree_depth_from_xyz(input_xyz, args)

        # 浅すぎ・深すぎを避けるため、最大深さの10%〜80%を使う
        depth_min = int(math.ceil(float(full_octree_depth) * 0.10))
        depth_max = int(math.floor(float(full_octree_depth) * 0.80))

        # depth=0 や範囲崩壊を防ぐ
        depth_min = max(1, depth_min)
        depth_max = max(depth_min, depth_max)

        # 念のため最大深さを超えないようにする
        depth_min = min(depth_min, full_octree_depth)
        depth_max = min(depth_max, full_octree_depth)

        if depth_min > depth_max:
            depth_min, depth_max = depth_max, depth_min

        args.train_subtree_level_min = int(depth_min)
        args.train_subtree_level_max = int(depth_max)
        args._train_subtree_depth_cli_override = True

        # 10%〜80%の範囲からランダムに選ばせる
        if bool(getattr(args, "surrogate_pretrain_subtree_random_depth", True)):
            args.train_subtree_randomize_level = True
        else:
            args.train_subtree_randomize_level = False

        return callback()
    finally:
        for key, value in saved.items():
            setattr(args, key, value)


def _build_surrogate_pretrain_subtree_sample(pts, args, cache_key, use_cuda, global_step):
    sample_t0 = time.perf_counter()
    input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
    input_pcd = _downsample_input_batch(input_pcd, args, cache_key)
    if use_cuda:
        input_pcd = input_pcd.cuda(non_blocking=True)
    input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous()
    input_xyz = input_pcd[:, :3, :]
    input_attr_full = input_pcd[:, 3:, :].contiguous() if input_pcd.shape[1] > 3 else None

    def _sample_depth():
        return sample_train_subtree_depth(
            input_xyz,
            args,
            global_step=global_step,
            cache_key=cache_key,
        )

    subtree_depth_meta = _with_pretrain_subtree_depth_overrides(
        args,
        _sample_depth,
        input_xyz=input_xyz,
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
    all_groups = [
        (int(subtree_key.detach().cpu()), point_idx)
        for subtree_key, point_idx in zip(all_subtree_keys, subtree_index_lists)
    ]
    eligible_groups = [
        (subtree_key, point_idx)
        for subtree_key, point_idx in all_groups
        if int(point_idx.numel()) >= min_subtree_points
    ]
    group_source = eligible_groups or all_groups
    skip_reason = "none"
    if not group_source:
        return {
            "skip_reason": "no_valid_subtree",
            "sampling_time": time.perf_counter() - sample_t0,
            "depth": int(subtree_depth_meta.get("depth", 0)),
            "point_count": 0,
            "total_subtree_count": total_subtree_count,
            "eligible_subtree_count": 0,
            "selected_subtree_count": 0,
            "retry_count": 0,
        }

    selected_groups = group_source
    if bool(getattr(args, "surrogate_pretrain_subtree_reuse_train_sampler", True)):
        candidate_keys = all_subtree_keys.new_tensor([subtree_key for subtree_key, _ in group_source])
        selected_keys = select_octree_subtree_keys(candidate_keys, int(global_step), args)
        selected_key_set = set(selected_keys.detach().cpu().tolist())
        selected_groups = [
            (subtree_key, point_idx)
            for subtree_key, point_idx in group_source
            if subtree_key in selected_key_set
        ]
        if not selected_groups:
            selected_groups = [max(group_source, key=lambda item: int(item[1].numel()))]
            skip_reason = "sampler_empty_fallback_largest"
    else:
        selected_groups = [max(group_source, key=lambda item: int(item[1].numel()))]

    chosen_index = int(global_step) % max(len(selected_groups), 1)
    subtree_key, point_idx = selected_groups[chosen_index]
    subtree_xyz = input_xyz.index_select(2, point_idx).contiguous()
    subtree_attr = input_attr_full.index_select(2, point_idx).contiguous() if input_attr_full is not None else None
    point_count = int(subtree_xyz.shape[-1])
    if point_count <= 0:
        skip_reason = "empty_selected_subtree"

    bbox_min = subtree_xyz[:, :3, :].amin(dim=2).squeeze(0) if point_count > 0 else None
    bbox_max = subtree_xyz[:, :3, :].amax(dim=2).squeeze(0) if point_count > 0 else None
    subtree_cache_key = (
        f"{cache_key}|pretrain_subtree_depth={int(subtree_ref['depth'][0].item())}|subtree_key={int(subtree_key)}"
    )
    return {
        "input_xyz": input_xyz,
        "input_attr_full": input_attr_full,
        "subtree_xyz": subtree_xyz,
        "subtree_attr": subtree_attr,
        "subtree_ref": subtree_ref,
        "subtree_depth_meta": subtree_depth_meta,
        "subtree_cache_key": subtree_cache_key,
        "subtree_key": int(subtree_key),
        "point_count": point_count,
        "bbox_min": _format_xyz_triplet(bbox_min),
        "bbox_max": _format_xyz_triplet(bbox_max),
        "retry_count": 0,
        "skip_reason": skip_reason,
        "total_subtree_count": total_subtree_count,
        "eligible_subtree_count": int(len(eligible_groups)),
        "selected_subtree_count": int(len(selected_groups)),
        "depth": int(subtree_ref["depth"][0].item()),
        "sampling_time": time.perf_counter() - sample_t0,
    }


def _sign_label(value, eps=1e-12):
    value = finite_float_or_none(value)
    if value is None:
        return None
    if abs(value) <= eps:
        return 0
    return 1 if value > 0.0 else -1


def _sign_match_value(metric_value, actual_value):
    metric_sign = _sign_label(metric_value)
    actual_sign = _sign_label(actual_value)
    if metric_sign is None or actual_sign is None:
        return None
    return 1.0 if metric_sign == actual_sign else 0.0


def _append_corr_pair(store, key, metric_value, actual_value, max_samples):
    metric = finite_float_or_none(metric_value)
    actual = finite_float_or_none(actual_value)
    if metric is None or actual is None:
        return None, None, 0
    pairs = store.setdefault(key, [])
    pairs.append((metric, actual))
    if len(pairs) > max_samples:
        del pairs[:-max_samples]
    corr = rolling_pearson(pairs)
    sign_values = [
        _sign_match_value(metric_item, actual_item)
        for metric_item, actual_item in pairs
    ]
    sign_values = [value for value in sign_values if value is not None]
    sign_match = (sum(sign_values) / len(sign_values)) if sign_values else None
    return corr, sign_match, len(pairs)


def _update_actual_correlation_debug(args, comp_debug, L_com, corr_store):
    actual_value = finite_float_or_none(comp_debug.get("actual_total_bit_percent", None))
    if actual_value is None or not _is_fresh_actual(args, comp_debug):
        return {}

    max_samples = max(int(getattr(args, "sparsepcgc_corr_window", 100)), 2)
    metric_values = {
        "surrogate_actual": comp_debug.get("surrogate_pred_bit", None),
        "lcom_actual": L_com,
        "cp_main_actual": comp_debug.get("cp_L_com_main", None),
        "sparsepcgc_aux_actual": comp_debug.get("sparsepcgc_aux_weighted", comp_debug.get("sparsepcgc_aux_loss", None)),
        "lcom_without_sparsepcgc_aux_actual": comp_debug.get("lcom_without_sparsepcgc_aux", None),
    }
    result = {
        "rolling_corr_window": max_samples,
        "rolling_sign_match_window": max_samples,
    }
    key_map = {
        "surrogate_actual": ("corr_surrogate_actual", "sign_match_surrogate_actual"),
        "lcom_actual": ("corr_lcom_actual", "sign_match_lcom_actual"),
        "cp_main_actual": ("corr_cp_main_actual", "sign_match_cp_main_actual"),
        "sparsepcgc_aux_actual": ("corr_sparsepcgc_aux_actual", "sign_match_sparsepcgc_aux_actual"),
        "lcom_without_sparsepcgc_aux_actual": (
            "corr_lcom_without_sparsepcgc_aux_actual",
            "sign_match_lcom_without_sparsepcgc_aux_actual",
        ),
    }
    for key, metric_value in metric_values.items():
        corr, sign_match, _count = _append_corr_pair(corr_store, key, metric_value, actual_value, max_samples)
        corr_name, sign_name = key_map[key]
        result[corr_name] = corr
        result[sign_name] = sign_match
    return result


def _sparsepcgc_add_experiment_active(args):
    codec = str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "")
    if codec != "sparsepcgc":
        return False
    if not bool(getattr(args, "sparsepcgc_enable_add_experiment", False)):
        return False
    if bool(getattr(args, "sparsepcgc_add_only_when_compression_primary", True)):
        return _loss_mode(args) == "compression_primary"
    return True


def _add_warmup_factor(args):
    steps = max(int(getattr(args, "sparsepcgc_add_warmup_steps", 0)), 0)
    if steps <= 0:
        return 1.0
    step = int(getattr(args, "_global_train_step", 0)) + 1
    return min(1.0, max(0.0, float(step) / float(steps)))


def _loss_mode(args):
    return str(getattr(args, "loss_mode", "legacy_total")).strip().lower()


def _as_scalar_loss_tensor(value):
    if not torch.is_tensor(value):
        return None
    if value.numel() == 1:
        return value.reshape(())
    return value.mean()


def _zero_like_loss(reference):
    if torch.is_tensor(reference):
        return reference.new_zeros(())
    return torch.zeros((), dtype=torch.float32)


def _relu_penalty(term, tau):
    tau_t = term.new_tensor(float(tau))
    return torch.relu(term - tau_t)


def _term_requires_grad(value):
    return bool(torch.is_tensor(value) and value.requires_grad)


def _term_is_finite(value):
    if not torch.is_tensor(value):
        return False
    try:
        return bool(torch.isfinite(value.detach()).all().item())
    except Exception:
        return False


def _select_compression_primary_main(terms, L_com):
    candidates = [
        ("main", terms.get("main", None)),
        ("bit", terms.get("bit", None)),
        ("objective", terms.get("objective", None)),
        ("L_com_fallback", L_com),
    ]
    tensor_candidates = []
    for name, value in candidates:
        tensor_value = _as_scalar_loss_tensor(value)
        if tensor_value is not None:
            tensor_candidates.append((name, tensor_value))
    for name, value in tensor_candidates:
        if value.requires_grad:
            return name, value
    if tensor_candidates:
        return tensor_candidates[0]
    return "zero_fallback", _zero_like_loss(L_com)


def _build_compression_primary_loss(
    args,
    *,
    terms,
    L_com,
    L_geom,
    L_actuator,
    global_train_step,
    stage_factors,
):
    # actual codec値は微分不能なので、actual_total_bit_percent_fresh等はlossに入れない。
    # compression_primaryの学習信号はsurrogate/proxy/soft auxのtensorだけから作る。
    # debugやCSVの.item()済み値はcheckpoint/log/teacher用であり、backward対象にしない。
    main_source, L_com_main = _select_compression_primary_main(terms, L_com)
    warmup_steps = int(getattr(args, "compression_primary_warmup_steps", 0))
    if warmup_steps > 0:
        warmup = min(1.0, float(int(global_train_step) + 1) / float(warmup_steps))
    else:
        warmup = 1.0
    L_com_primary = float(getattr(args, "w_com", 1.0)) * float(warmup) * L_com_main

    zero = _zero_like_loss(L_com_primary)
    L_single = _as_scalar_loss_tensor(terms.get("single", None))
    L_nodes = _as_scalar_loss_tensor(terms.get("node", None))
    L_sparsepcgc = _as_scalar_loss_tensor(terms.get("sparsepcgc", None))
    L_op = _as_scalar_loss_tensor(terms.get("op", None))

    P_geom = _relu_penalty(_as_scalar_loss_tensor(L_geom), getattr(args, "cp_tau_geom", 0.06))
    P_single = _relu_penalty(L_single, getattr(args, "cp_tau_single", 0.0)) if L_single is not None else zero
    P_nodes = _relu_penalty(L_nodes, getattr(args, "cp_tau_nodes", 0.0)) if L_nodes is not None else zero
    P_sparsepcgc = (
        _relu_penalty(L_sparsepcgc, getattr(args, "cp_tau_sparsepcgc", 0.0))
        if L_sparsepcgc is not None
        else zero
    )
    P_actuator = _relu_penalty(_as_scalar_loss_tensor(L_actuator), getattr(args, "cp_tau_actuator", 0.0))
    P_op = _relu_penalty(L_op, 0.0) if L_op is not None else zero

    use_stage = bool(getattr(args, "cp_use_stage_factors", False))
    sf_com = float(stage_factors.get("com", 1.0)) if use_stage else 1.0
    sf_geom = float(stage_factors.get("geom", 1.0)) if use_stage else 1.0
    sf_repair = float(stage_factors.get("repair", 1.0)) if use_stage else 1.0

    L = (
        sf_com * L_com_primary
        + sf_geom * float(getattr(args, "cp_lambda_geom", 1.0)) * P_geom
        + sf_com * float(getattr(args, "cp_lambda_single", 1.0)) * P_single
        + sf_com * float(getattr(args, "cp_lambda_nodes", 1.0)) * P_nodes
        + sf_com * float(getattr(args, "cp_lambda_sparsepcgc", 1.0)) * P_sparsepcgc
        + sf_repair * float(getattr(args, "cp_lambda_actuator", 0.1)) * P_actuator
        + sf_repair * float(getattr(args, "cp_lambda_op", 0.0)) * P_op
    )

    debug = {
        "loss_mode": "compression_primary",
        "cp_main_source": main_source,
        "cp_warmup": float(warmup),
        "cp_L_com_main": _case_float(L_com_main, float("nan")),
        "cp_L_com_primary": _case_float(L_com_primary, float("nan")),
        "cp_P_geom": _case_float(P_geom, float("nan")),
        "cp_P_single": _case_float(P_single, float("nan")),
        "cp_P_nodes": _case_float(P_nodes, float("nan")),
        "cp_P_sparsepcgc": _case_float(P_sparsepcgc, float("nan")),
        "cp_P_actuator": _case_float(P_actuator, float("nan")),
        "cp_P_op": _case_float(P_op, float("nan")),
        "cp_total": _case_float(L, float("nan")),
        "cp_main_requires_grad": _term_requires_grad(L_com_main),
        "cp_geom_requires_grad": _term_requires_grad(L_geom),
        "cp_single_requires_grad": _term_requires_grad(L_single),
        "cp_nodes_requires_grad": _term_requires_grad(L_nodes),
        "cp_sparsepcgc_requires_grad": _term_requires_grad(L_sparsepcgc),
        "cp_actuator_requires_grad": _term_requires_grad(L_actuator),
        "cp_op_requires_grad": _term_requires_grad(L_op),
        "cp_main_finite": _term_is_finite(L_com_main),
        "cp_geom_finite": _term_is_finite(L_geom),
        "cp_single_finite": _term_is_finite(L_single) if L_single is not None else True,
        "cp_nodes_finite": _term_is_finite(L_nodes) if L_nodes is not None else True,
        "cp_sparsepcgc_finite": _term_is_finite(L_sparsepcgc) if L_sparsepcgc is not None else True,
        "cp_actuator_finite": _term_is_finite(L_actuator),
        "cp_op_finite": _term_is_finite(L_op) if L_op is not None else True,
    }
    return L, sf_com * L_com_primary, debug


def _log_compression_primary_terms(writer, step, num_steps, cp_debug):
    writer.write(
        f"CompressionPrimaryLoss step={step + 1}/{num_steps}: "
        f"main_source={cp_debug.get('cp_main_source', 'unknown')}, "
        f"warmup={float(cp_debug.get('cp_warmup', 1.0)):.6f}, "
        f"L_com_main={float(cp_debug.get('cp_L_com_main', 0.0)):.6f}, "
        f"L_com_primary={float(cp_debug.get('cp_L_com_primary', 0.0)):.6f}, "
        f"P_geom={float(cp_debug.get('cp_P_geom', 0.0)):.6f}, "
        f"P_single={float(cp_debug.get('cp_P_single', 0.0)):.6f}, "
        f"P_nodes={float(cp_debug.get('cp_P_nodes', 0.0)):.6f}, "
        f"P_sparsepcgc={float(cp_debug.get('cp_P_sparsepcgc', 0.0)):.6f}, "
        f"P_actuator={float(cp_debug.get('cp_P_actuator', 0.0)):.6f}, "
        f"P_op={float(cp_debug.get('cp_P_op', 0.0)):.6f}, "
        f"total={float(cp_debug.get('cp_total', 0.0)):.6f}, "
        "requires_grad["
        f"main={bool(cp_debug.get('cp_main_requires_grad', False))}, "
        f"geom={bool(cp_debug.get('cp_geom_requires_grad', False))}, "
        f"single={bool(cp_debug.get('cp_single_requires_grad', False))}, "
        f"nodes={bool(cp_debug.get('cp_nodes_requires_grad', False))}, "
        f"sparsepcgc={bool(cp_debug.get('cp_sparsepcgc_requires_grad', False))}, "
        f"actuator={bool(cp_debug.get('cp_actuator_requires_grad', False))}, "
        f"op={bool(cp_debug.get('cp_op_requires_grad', False))}], "
        "finite["
        f"main={bool(cp_debug.get('cp_main_finite', False))}, "
        f"geom={bool(cp_debug.get('cp_geom_finite', False))}, "
        f"single={bool(cp_debug.get('cp_single_finite', False))}, "
        f"nodes={bool(cp_debug.get('cp_nodes_finite', False))}, "
        f"sparsepcgc={bool(cp_debug.get('cp_sparsepcgc_finite', False))}, "
        f"actuator={bool(cp_debug.get('cp_actuator_finite', False))}, "
        f"op={bool(cp_debug.get('cp_op_finite', False))}]"
    )


def _init_case_debug_csv(args, plot, writer):
    if not bool(getattr(args, "save_good_bad_cases", False)):
        return None
    os.makedirs(plot.save_dir, exist_ok=True)
    path = os.path.join(plot.save_dir, f"{args.time}_good_bad_cases.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CASE_DEBUG_COLUMNS).writeheader()
    writer.write(f"GoodBadCaseDebug: enabled path={path}")
    return path


def _maybe_record_case_debug(
    args,
    writer,
    case_debug_path,
    case_debug_counts,
    *,
    global_step,
    episode,
    epoch,
    step,
    file_path,
    comp_debug,
    structure_debug,
    edit_stats,
    L,
    L_geom,
    L_com,
    L_actuator,
):
    if not case_debug_path or int(getattr(args, "max_saved_cases", 0)) <= 0:
        return
    actual_delta = _case_float(comp_debug.get("actual_total_bit_percent", comp_debug.get("total_bit", float("nan"))), float("nan"))
    if not math.isfinite(actual_delta):
        return
    case_type = None
    if actual_delta <= float(getattr(args, "good_case_delta_threshold", -5.0)):
        case_type = "good"
    elif actual_delta >= float(getattr(args, "bad_case_delta_threshold", 20.0)):
        case_type = "bad"
    if case_type is None:
        return
    if int(case_debug_counts.get(case_type, 0)) >= int(getattr(args, "max_saved_cases", 64)):
        return

    row = {
        "case_type": case_type,
        "global_step": int(global_step),
        "episode": int(episode) + 1,
        "epoch": int(epoch) + 1,
        "step": int(step) + 1,
        "sample_name": os.path.basename(str(file_path)),
        "codec": str(comp_debug.get("teacher_codec", getattr(args, "compress", "unknown"))),
        "actual_delta": actual_delta,
        "surrogate_delta": _case_float(comp_debug.get("rate_proxy_delta", comp_debug.get("surrogate_pred_bit", 0.0))),
        "surrogate_abs_error": _case_float(comp_debug.get("surrogate_abs_bit_error", 0.0)),
        "surrogate_signed_error": _case_float(comp_debug.get("surrogate_signed_bit_error", 0.0)),
        "actual_bits_before": _case_float(comp_debug.get("gt_actual_bit", float("nan")), float("nan")),
        "actual_bits_after": _case_float(comp_debug.get("gen_actual_bit", float("nan")), float("nan")),
        "point_count_before": _case_int(comp_debug.get("gt_points", edit_stats.get("input_points", 0) if edit_stats else 0)),
        "point_count_after": _case_int(comp_debug.get("gen_points", edit_stats.get("output_points", 0) if edit_stats else 0)),
        "unique_coord_before": _case_int(comp_debug.get("gt_unique_coord_count", 0)),
        "unique_coord_after": _case_int(comp_debug.get("gen_unique_coord_count", 0)),
        "active_coord_before": _case_int(comp_debug.get("sparsepcgc_before_active_coords", 0)),
        "active_coord_after": _case_int(comp_debug.get("sparsepcgc_after_active_coords", 0)),
        "octree_node_before": _case_float(comp_debug.get("gt_octree_node", 0.0)),
        "octree_node_after": _case_float(comp_debug.get("gen_octree_node", 0.0)),
        "single_before": _case_float(comp_debug.get("gt_octree_single", 0.0)),
        "single_after": _case_float(comp_debug.get("gen_octree_single", 0.0)),
        "add_points": _case_int(edit_stats.get("added_points", 0) if edit_stats else 0),
        "delete_points": _case_int(edit_stats.get("deleted_points", 0) if edit_stats else 0),
        "adjust_points": _case_int(edit_stats.get("adjusted_points", 0) if edit_stats else 0),
        "preserve_ratio": _case_float(structure_debug.get("preserve_ratio", 0.0)),
        "same_voxel_adjust": _case_int(structure_debug.get("same_voxel_adjust_count", 0)),
        "different_voxel_move": _case_int(structure_debug.get("moved_different_voxel_count", 0)),
        "move_source_emptied": _case_int(structure_debug.get("move_source_emptied_voxel_count", 0)),
        "move_target_new": _case_int(structure_debug.get("move_target_new_voxel_count", 0)),
        "move_source_not_emptied": _case_int(structure_debug.get("move_source_not_emptied_count", 0)),
        "shape_loss": _case_float(L_geom),
        "compression_loss": _case_float(L_com),
        "actuator_loss": _case_float(L_actuator),
        "total_loss": _case_float(L),
        "teacher_refresh": bool(comp_debug.get("teacher_refresh", False)),
        "teacher_target_age": _case_int(comp_debug.get("teacher_target_age", 0)),
    }
    with open(case_debug_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CASE_DEBUG_COLUMNS).writerow(row)
    case_debug_counts[case_type] = int(case_debug_counts.get(case_type, 0)) + 1
    writer.write(
        "GoodBadCaseDebug: "
        f"type={case_type}, step={int(global_step) + 1}, sample={row['sample_name']}, "
        f"actual_delta={actual_delta:.6f}, path={case_debug_path}"
    )


def _init_csv_file(path, columns, writer, label):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()
    writer.write(f"{label}: enabled path={path}")


def _append_csv_row(path, columns, row):
    if not path:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=columns).writerow(
            {key: row.get(key, None) for key in columns}
        )


def _init_metric_csvs(args, plot, writer):
    paths = {
        "compression_step": None,
        "compression_episode": None,
        "operation_step": None,
        "operation_episode": None,
        "checkpoint_episode": None,
        "surrogate_pretrain_step": None,
    }
    os.makedirs(plot.save_dir, exist_ok=True)
    if bool(getattr(args, "save_compression_metric_csv", True)):
        path = os.path.join(plot.save_dir, f"{args.time}_compression_metrics_step.csv")
        _init_csv_file(path, COMPRESSION_METRIC_COLUMNS, writer, "CompressionMetricCSV")
        paths["compression_step"] = path
        epi_path = os.path.join(plot.save_dir, f"{args.time}_compression_metrics_epi.csv")
        _init_csv_file(epi_path, COMPRESSION_EPISODE_METRIC_COLUMNS, writer, "CompressionEpisodeMetricCSV")
        paths["compression_episode"] = epi_path
    if bool(getattr(args, "save_operation_metric_csv", True)):
        path = os.path.join(plot.save_dir, f"{args.time}_operation_metrics_step.csv")
        _init_csv_file(path, OPERATION_METRIC_COLUMNS, writer, "OperationMetricCSV")
        paths["operation_step"] = path
        epi_path = os.path.join(plot.save_dir, f"{args.time}_operation_metrics_epi.csv")
        _init_csv_file(epi_path, OPERATION_EPISODE_METRIC_COLUMNS, writer, "OperationEpisodeMetricCSV")
        paths["operation_episode"] = epi_path
    if bool(getattr(args, "save_checkpoint_metric_csv", True)):
        path = os.path.join(plot.save_dir, f"{args.time}_checkpoint_metrics_epi.csv")
        _init_csv_file(path, CHECKPOINT_METRIC_COLUMNS, writer, "CheckpointMetricCSV")
        paths["checkpoint_episode"] = path
    if int(getattr(args, "surrogate_pretrain_steps", 0)) > 0:
        path = os.path.join(plot.save_dir, f"{args.time}_surrogate_pretrain_metrics_step.csv")
        _init_csv_file(path, SURROGATE_PRETRAIN_COLUMNS, writer, "SurrogatePretrainMetricCSV")
        paths["surrogate_pretrain_step"] = path
    return paths


def _is_actual_training_backend(args):
    backend = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    return backend.endswith("_surrogate") or "_actual" in backend


def _is_fresh_actual(args, comp_debug):
    backend = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    has_actual = math.isfinite(
        _case_float(comp_debug.get("actual_total_bit_percent", float("nan")), float("nan"))
    )
    not_fallback = not bool(comp_debug.get("actual_codec_fallback_to_proxy", False))
    not_disabled = not bool(comp_debug.get("actual_codec_disabled_during_train", False))
    if not has_actual or not not_fallback:
        return False
    if bool(comp_debug.get("actual_value_is_fresh", False)):
        return True
    if backend.endswith("_surrogate"):
        return bool(comp_debug.get("teacher_refresh", False))
    if "_actual" in backend:
        return not_disabled and not bool(comp_debug.get("actual_codec_skipped_by_interval", False))
    return False


def _build_compression_metric_row(
    args,
    *,
    global_step,
    episode,
    epoch,
    step,
    stage,
    comp_debug,
    L_com,
):
    actual_delta = _case_float(comp_debug.get("actual_total_bit_percent", float("nan")), float("nan"))
    fresh_actual = _is_fresh_actual(args, comp_debug)
    cached_actual = math.isfinite(actual_delta) and not fresh_actual
    return {
        "global_step": int(global_step) + 1,
        "episode": int(episode) + 1,
        "epoch": int(epoch) + 1,
        "step": int(step) + 1,
        "stage": str(stage),
        "codec": str(comp_debug.get("teacher_codec", getattr(args, "compress", "unknown"))),
        "backend": str(getattr(args, "compression_loss_backend", "proxy")),
        "actual_value_source": str(comp_debug.get("actual_value_source", "unknown")),
        "fresh_actual": bool(fresh_actual),
        "cached_actual": bool(cached_actual),
        "actual_total_bit_percent": actual_delta if math.isfinite(actual_delta) else None,
        "actual_total_bit_percent_fresh": actual_delta if fresh_actual else None,
        "actual_total_bit_percent_cached": actual_delta if cached_actual else None,
        "compression_loss_L_com": _case_float(L_com, float("nan")),
        "lcom_main": _case_float(comp_debug.get("compression_main_loss", float("nan")), float("nan")),
        "lcom_aux": _case_float(comp_debug.get("compression_aux_loss", float("nan")), float("nan")),
        "lcom_sparsepcgc_aux": _case_float(comp_debug.get("sparsepcgc_aux_loss", float("nan")), float("nan")),
        "sparsepcgc_aux_raw": _case_float(comp_debug.get("sparsepcgc_aux_raw", comp_debug.get("sparsepcgc_aux_loss", float("nan"))), float("nan")),
        "sparsepcgc_aux_weighted": _case_float(comp_debug.get("sparsepcgc_aux_weighted", float("nan")), float("nan")),
        "lcom_without_sparsepcgc_aux": _case_float(comp_debug.get("lcom_without_sparsepcgc_aux", float("nan")), float("nan")),
        "lcom_with_sparsepcgc_aux": _case_float(comp_debug.get("lcom_with_sparsepcgc_aux", comp_debug.get("compression_objective", float("nan"))), float("nan")),
        "lcom_objective": _case_float(comp_debug.get("compression_objective", comp_debug.get("total_bit", float("nan"))), float("nan")),
        "com_sparsepcgc_weight": _case_float(getattr(args, "com_sparsepcgc", float("nan")), float("nan")),
        "sparsepcgc_aux_weight": _case_float(getattr(args, "com_sparsepcgc", float("nan")), float("nan")),
        "sparsepcgc_active_coord_loss": _case_float(comp_debug.get("sparsepcgc_active_coord_loss", float("nan")), float("nan")),
        "sparsepcgc_isolated_loss": _case_float(comp_debug.get("sparsepcgc_isolated_proxy_loss", float("nan")), float("nan")),
        "sparsepcgc_entropy_loss": _case_float(comp_debug.get("sparsepcgc_entropy_proxy_loss", float("nan")), float("nan")),
        "sparsepcgc_density_loss": _case_float(comp_debug.get("sparsepcgc_density_proxy_loss", float("nan")), float("nan")),
        "sparsepcgc_single_aux": _case_float(comp_debug.get("soft_single_percent", float("nan")), float("nan")),
        "sparsepcgc_node_aux": _case_float(comp_debug.get("soft_node_percent", float("nan")), float("nan")),
        "compression_objective": _case_float(comp_debug.get("compression_objective", comp_debug.get("total_bit", float("nan"))), float("nan")),
        "compression_main_loss": _case_float(comp_debug.get("compression_main_loss", float("nan")), float("nan")),
        "compression_aux_loss": _case_float(comp_debug.get("compression_aux_loss", float("nan")), float("nan")),
        "sparsepcgc_aux_loss": _case_float(comp_debug.get("sparsepcgc_aux_loss", float("nan")), float("nan")),
        "surrogate_pred_bit_percent": _case_float(comp_debug.get("surrogate_pred_bit", float("nan")), float("nan")),
        "surrogate_target_bit_percent": _case_float(comp_debug.get("surrogate_target_bit", float("nan")), float("nan")),
        "surrogate_abs_bit_error": _case_float(comp_debug.get("surrogate_abs_bit_error", float("nan")), float("nan")),
        "surrogate_signed_bit_error": _case_float(comp_debug.get("surrogate_signed_bit_error", float("nan")), float("nan")),
        "surrogate_train_loss": _case_float(comp_debug.get("surrogate_train_loss", float("nan")), float("nan")),
        "proxy_delta_percent": _case_float(comp_debug.get("rate_proxy_delta", comp_debug.get("total_bit", float("nan"))), float("nan")),
        "actual_bits_before": _case_float(comp_debug.get("gt_actual_bit", comp_debug.get("gt_bit_abs", float("nan"))), float("nan")),
        "actual_bits_after": _case_float(comp_debug.get("gen_actual_bit", comp_debug.get("gen_bit_abs", float("nan"))), float("nan")),
        "point_count_before": _case_int(comp_debug.get("gt_points", 0)),
        "point_count_after": _case_int(comp_debug.get("gen_points", 0)),
        "unique_coord_before": _case_int(comp_debug.get("gt_unique_coord_count", 0)),
        "unique_coord_after": _case_int(comp_debug.get("gen_unique_coord_count", 0)),
        "node_delta": _case_float(comp_debug.get("node_delta", float("nan")), float("nan")),
        "single_delta": _case_float(comp_debug.get("single_delta", float("nan")), float("nan")),
        "teacher_refresh": bool(comp_debug.get("teacher_refresh", False)),
        "teacher_cache_hit": comp_debug.get("teacher_cache_hit", None),
        "teacher_target_age": _case_int(comp_debug.get("teacher_target_age", 0)),
        "actual_codec_disabled": bool(comp_debug.get("actual_codec_disabled_during_train", False)),
        "actual_codec_skipped_by_interval": bool(comp_debug.get("actual_codec_skipped_by_interval", False)),
        "actual_codec_fallback_to_proxy": bool(comp_debug.get("actual_codec_fallback_to_proxy", False)),
        "loss_mode": str(comp_debug.get("loss_mode", getattr(args, "loss_mode", "legacy_total"))),
        "cp_main_source": str(comp_debug.get("cp_main_source", "")),
        "cp_warmup": _case_float(comp_debug.get("cp_warmup", float("nan")), float("nan")),
        "cp_L_com_main": _case_float(comp_debug.get("cp_L_com_main", float("nan")), float("nan")),
        "cp_L_com_primary": _case_float(comp_debug.get("cp_L_com_primary", float("nan")), float("nan")),
        "cp_P_geom": _case_float(comp_debug.get("cp_P_geom", float("nan")), float("nan")),
        "cp_P_single": _case_float(comp_debug.get("cp_P_single", float("nan")), float("nan")),
        "cp_P_nodes": _case_float(comp_debug.get("cp_P_nodes", float("nan")), float("nan")),
        "cp_P_sparsepcgc": _case_float(comp_debug.get("cp_P_sparsepcgc", float("nan")), float("nan")),
        "cp_P_actuator": _case_float(comp_debug.get("cp_P_actuator", float("nan")), float("nan")),
        "cp_P_op": _case_float(comp_debug.get("cp_P_op", float("nan")), float("nan")),
        "cp_total": _case_float(comp_debug.get("cp_total", float("nan")), float("nan")),
        "cp_main_requires_grad": comp_debug.get("cp_main_requires_grad", None),
        "cp_geom_requires_grad": comp_debug.get("cp_geom_requires_grad", None),
        "cp_single_requires_grad": comp_debug.get("cp_single_requires_grad", None),
        "cp_nodes_requires_grad": comp_debug.get("cp_nodes_requires_grad", None),
        "cp_sparsepcgc_requires_grad": comp_debug.get("cp_sparsepcgc_requires_grad", None),
        "cp_actuator_requires_grad": comp_debug.get("cp_actuator_requires_grad", None),
        "cp_op_requires_grad": comp_debug.get("cp_op_requires_grad", None),
        "cp_main_finite": comp_debug.get("cp_main_finite", None),
        "cp_geom_finite": comp_debug.get("cp_geom_finite", None),
        "cp_single_finite": comp_debug.get("cp_single_finite", None),
        "cp_nodes_finite": comp_debug.get("cp_nodes_finite", None),
        "cp_sparsepcgc_finite": comp_debug.get("cp_sparsepcgc_finite", None),
        "cp_actuator_finite": comp_debug.get("cp_actuator_finite", None),
        "cp_op_finite": comp_debug.get("cp_op_finite", None),
        "corr_surrogate_actual": _case_float(comp_debug.get("corr_surrogate_actual", float("nan")), float("nan")),
        "corr_lcom_actual": _case_float(comp_debug.get("corr_lcom_actual", float("nan")), float("nan")),
        "corr_cp_main_actual": _case_float(comp_debug.get("corr_cp_main_actual", float("nan")), float("nan")),
        "corr_sparsepcgc_aux_actual": _case_float(comp_debug.get("corr_sparsepcgc_aux_actual", float("nan")), float("nan")),
        "corr_lcom_without_sparsepcgc_aux_actual": _case_float(comp_debug.get("corr_lcom_without_sparsepcgc_aux_actual", float("nan")), float("nan")),
        "sign_match_surrogate_actual": _case_float(comp_debug.get("sign_match_surrogate_actual", float("nan")), float("nan")),
        "sign_match_lcom_actual": _case_float(comp_debug.get("sign_match_lcom_actual", float("nan")), float("nan")),
        "sign_match_cp_main_actual": _case_float(comp_debug.get("sign_match_cp_main_actual", float("nan")), float("nan")),
        "sign_match_sparsepcgc_aux_actual": _case_float(comp_debug.get("sign_match_sparsepcgc_aux_actual", float("nan")), float("nan")),
        "sign_match_lcom_without_sparsepcgc_aux_actual": _case_float(comp_debug.get("sign_match_lcom_without_sparsepcgc_aux_actual", float("nan")), float("nan")),
        "rolling_corr_window": _case_int(comp_debug.get("rolling_corr_window", getattr(args, "sparsepcgc_corr_window", 100))),
        "rolling_sign_match_window": _case_int(comp_debug.get("rolling_sign_match_window", getattr(args, "sparsepcgc_corr_window", 100))),
        "active_coord_before": _case_int(comp_debug.get("sparsepcgc_before_active_coords", 0)),
        "active_coord_after": _case_int(comp_debug.get("sparsepcgc_after_active_coords", 0)),
        "active_coord_delta": _case_int(comp_debug.get("sparsepcgc_active_coord_delta", 0)),
        "isolated_voxel_count": _case_int(comp_debug.get("sparsepcgc_after_isolated_voxels", 0)),
        "isolated_voxel_delta": _case_int(comp_debug.get("sparsepcgc_isolated_delta", 0)),
        "sparse_density_before": _case_float(comp_debug.get("sparsepcgc_before_sparse_density", float("nan")), float("nan")),
        "sparse_density_after": _case_float(comp_debug.get("sparsepcgc_after_sparse_density", float("nan")), float("nan")),
        "sparse_density_delta": _case_float(comp_debug.get("sparsepcgc_sparse_density_delta", float("nan")), float("nan")),
        "occupancy_entropy": _case_float(comp_debug.get("occupancy_entropy", comp_debug.get("sparsepcgc_entropy_proxy_loss", float("nan"))), float("nan")),
        "occupancy_nll_proxy": _case_float(comp_debug.get("occupancy_nll_proxy", float("nan")), float("nan")),
        "lowprob_occupancy_ratio": _case_float(comp_debug.get("lowprob_occupancy_ratio", float("nan")), float("nan")),
        "entropy_delta": _case_float(comp_debug.get("sparsepcgc_entropy_proxy_loss", float("nan")), float("nan")),
        "nll_delta": _case_float(comp_debug.get("nll_delta", comp_debug.get("occupancy_nll_proxy", float("nan"))), float("nan")),
    }


def _build_operation_metric_row(
    args,
    *,
    global_step,
    episode,
    epoch,
    step,
    stage,
    comp_debug,
    structure_debug,
    edit_stats,
):
    actual_delta = _case_float(comp_debug.get("actual_total_bit_percent", float("nan")), float("nan"))
    unique_before = _case_int(comp_debug.get("gt_unique_coord_count", 0))
    unique_after = _case_int(comp_debug.get("gen_unique_coord_count", 0))
    edit_stats = edit_stats or {}
    input_points = _case_float(edit_stats.get("input_points_avg", edit_stats.get("input_points", float("nan"))), float("nan"))
    add_candidate_ratio = _case_float(structure_debug.get("add_candidate_ratio", float("nan")), float("nan"))
    add_candidate_count = None
    if math.isfinite(input_points) and math.isfinite(add_candidate_ratio):
        add_candidate_count = int(round(input_points * add_candidate_ratio))
        if add_candidate_ratio > 0.0 and input_points > 0.0:
            add_candidate_count = max(add_candidate_count, 1)
    add_effective_count = _case_int(structure_debug.get("add_effective_count", structure_debug.get("add_actual_point_count", 0)))
    unique_delta = unique_after - unique_before
    positive_unique_delta = max(unique_delta, 0)
    add_removed_by_unique = max(add_effective_count - positive_unique_delta, 0)
    active_before = _case_int(comp_debug.get("sparsepcgc_before_active_coords", 0))
    active_after = _case_int(comp_debug.get("sparsepcgc_after_active_coords", 0))
    level_debug = structure_debug.get("octree_level_debug") or []
    return {
        "global_step": int(global_step) + 1,
        "episode": int(episode) + 1,
        "epoch": int(epoch) + 1,
        "step": int(step) + 1,
        "stage": str(stage),
        "codec": str(comp_debug.get("teacher_codec", getattr(args, "compress", "unknown"))),
        "fresh_actual": bool(_is_fresh_actual(args, comp_debug)),
        "actual_total_bit_percent": actual_delta if math.isfinite(actual_delta) else None,
        "train_or_eval_mode": "train",
        "hardening_mode": str(edit_stats.get("keep_mode", "")),
        "selection_threshold": _case_float(getattr(args, "operation_count_drop_threshold", 0.5), float("nan")),
        "topk_selected_count": _case_int(edit_stats.get("output_points", 0)),
        "sparsepcgc_add_experiment_enabled": bool(_sparsepcgc_add_experiment_active(args)),
        "add_enabled": bool(structure_debug.get("add_enabled", False)),
        "prune_enabled": bool(structure_debug.get("prune_enabled", False)),
        "disp_enabled": bool(structure_debug.get("disp_enabled", False)),
        "repair_ratio": _case_float(structure_debug.get("repair_ratio", float("nan")), float("nan")),
        "preserve_ratio": _case_float(structure_debug.get("preserve_ratio", float("nan")), float("nan")),
        "add_prob_mean": _case_float(structure_debug.get("add_prob_mean", float("nan")), float("nan")),
        "add_prob_max": _case_float(structure_debug.get("add_prob_max", float("nan")), float("nan")),
        "add_priority_mean": _case_float(structure_debug.get("add_priority_mean", float("nan")), float("nan")),
        "add_priority_max": _case_float(structure_debug.get("add_priority_max", float("nan")), float("nan")),
        "add_score_mean": _case_float(structure_debug.get("add_priority_mean", float("nan")), float("nan")),
        "add_score_max": _case_float(structure_debug.get("add_priority_max", float("nan")), float("nan")),
        "add_ratio": _case_float(structure_debug.get("add_ratio", float("nan")), float("nan")),
        "add_candidate_ratio": add_candidate_ratio,
        "add_candidate_count": add_candidate_count,
        "add_hard_count": _case_int(structure_debug.get("add_count", 0)),
        "add_effective_count": add_effective_count,
        "add_actual_point_count": add_effective_count,
        "add_target_voxels": _case_int(structure_debug.get("add_target_voxel_count", 0)),
        "add_target_ratio": _case_float(getattr(args, "target_add_ratio", 0.0), float("nan")),
        "add_max_ratio": _case_float(getattr(args, "max_add_ratio", 0.0), float("nan")),
        "add_warmup": _add_warmup_factor(args),
        "soft_add_count": _case_float(structure_debug.get("add_ratio", 0.0), 0.0) * input_points if math.isfinite(input_points) else None,
        "hard_add_count": _case_int(structure_debug.get("add_count", 0)),
        "drop_prob_mean": _case_float(structure_debug.get("drop_ratio", float("nan")), float("nan")),
        "hard_drop_ratio": _case_float(structure_debug.get("hard_drop_ratio", float("nan")), float("nan")),
        "hard_drop_count": _case_int(structure_debug.get("hard_drop_count", structure_debug.get("delete_removed_point_count", 0))),
        "delete_target_voxels": _case_int(structure_debug.get("delete_target_voxel_count", 0)),
        "delete_emptied_voxels": _case_int(structure_debug.get("delete_emptied_voxel_count", 0)),
        "move_score_mean": _case_float(structure_debug.get("move_score_mean", float("nan")), float("nan")),
        "hard_move_ratio": _case_float(structure_debug.get("move_ratio", float("nan")), float("nan")),
        "hard_move_count": _case_int(structure_debug.get("hard_move_count", 0)),
        "move_source_voxels": _case_int(structure_debug.get("move_source_voxel_count", 0)),
        "move_target_voxels": _case_int(structure_debug.get("move_target_voxel_count", 0)),
        "move_source_emptied": _case_int(structure_debug.get("move_source_emptied_voxel_count", 0)),
        "move_target_new": _case_int(structure_debug.get("move_target_new_voxel_count", 0)),
        "move_source_not_emptied": _case_int(structure_debug.get("move_source_not_emptied_count", 0)),
        "same_voxel_adjust": _case_int(structure_debug.get("same_voxel_adjust_count", 0)),
        "different_voxel_move": _case_int(structure_debug.get("moved_different_voxel_count", 0)),
        "input_points": input_points,
        "pre_output_points": _case_float(edit_stats.get("pre_output_points_avg", edit_stats.get("pre_output_points", float("nan"))), float("nan")),
        "output_points": _case_float(edit_stats.get("output_points_avg", edit_stats.get("output_points", float("nan"))), float("nan")),
        "added_ratio_percent": _case_float(edit_stats.get("added_ratio_percent", float("nan")), float("nan")),
        "deleted_ratio_percent": _case_float(edit_stats.get("deleted_ratio_percent", float("nan")), float("nan")),
        "adjusted_ratio_percent": _case_float(edit_stats.get("adjusted_ratio_percent", float("nan")), float("nan")),
        "codec_points_after": _case_int(comp_debug.get("gen_points", 0)),
        "codec_points_before": _case_int(comp_debug.get("gt_points", 0)),
        "codec_unique_after": unique_after,
        "codec_unique_before": unique_before,
        "unique_coord_delta": unique_delta,
        "add_after_quant_unique_count": positive_unique_delta,
        "add_removed_by_unique_count": add_removed_by_unique,
        "active_coord_before": active_before,
        "active_coord_after": active_after,
        "active_coord_delta": active_after - active_before,
        "isolated_voxel_count": _case_int(comp_debug.get("sparsepcgc_after_isolated_voxels", 0)),
        "isolated_voxel_delta": _case_int(comp_debug.get("sparsepcgc_isolated_delta", 0)),
        "sparse_density_before": _case_float(comp_debug.get("sparsepcgc_before_sparse_density", float("nan")), float("nan")),
        "sparse_density_after": _case_float(comp_debug.get("sparsepcgc_after_sparse_density", float("nan")), float("nan")),
        "sparse_density_delta": _case_float(comp_debug.get("sparsepcgc_sparse_density_delta", float("nan")), float("nan")),
        "occupancy_entropy": _case_float(structure_debug.get("occupancy_entropy", comp_debug.get("occupancy_entropy", float("nan"))), float("nan")),
        "occupancy_nll_proxy": _case_float(structure_debug.get("occupancy_nll_proxy", comp_debug.get("occupancy_nll_proxy", float("nan"))), float("nan")),
        "lowprob_occupancy_ratio": _case_float(structure_debug.get("lowprob_occupancy_ratio", comp_debug.get("lowprob_occupancy_ratio", float("nan"))), float("nan")),
        "entropy_delta": _case_float(comp_debug.get("sparsepcgc_entropy_proxy_loss", float("nan")), float("nan")),
        "nll_delta": _case_float(comp_debug.get("nll_delta", structure_debug.get("occupancy_nll_proxy", float("nan"))), float("nan")),
        "depth_node_count_summary": _summarize_octree_level_debug(level_debug, "occupied_mean"),
        "depth_single_child_count_summary": _summarize_octree_level_debug(level_debug, "single_mean"),
        "depth_entropy_summary": _summarize_octree_level_debug(level_debug, "std_children_mean"),
        "depth_lowprob_summary": _summarize_octree_level_debug(level_debug, "single_ratio_mean"),
        "subtree_depth": _case_int(structure_debug.get("subtree_depth", getattr(args, "_current_subtree_depth", 0))),
        "subtree_node_count": _case_float(structure_debug.get("subtree_node_count", float("nan")), float("nan")),
        "subtree_single_child_count": _case_float(structure_debug.get("subtree_single_child_count", float("nan")), float("nan")),
        "single_child_delta": _case_float(comp_debug.get("single_delta", float("nan")), float("nan")),
        "cp_L_com_main": _case_float(comp_debug.get("cp_L_com_main", float("nan")), float("nan")),
        "cp_total": _case_float(comp_debug.get("cp_total", float("nan")), float("nan")),
    }


def _new_operation_episode_sums():
    return {
        "sums": {key: 0.0 for key in OPERATION_EPISODE_METRIC_COLUMNS},
        "counts": {key: 0 for key in OPERATION_EPISODE_METRIC_COLUMNS},
        "row_count": 0,
        "fresh_actual_count": 0,
        "codec": None,
    }


def _accumulate_operation_episode(metric_sums, operation_row):
    metric_sums["row_count"] += 1
    if bool(operation_row.get("fresh_actual", False)):
        metric_sums["fresh_actual_count"] += 1
    if metric_sums.get("codec") is None:
        metric_sums["codec"] = operation_row.get("codec")
    for key in OPERATION_EPISODE_METRIC_COLUMNS:
        if key in {"episode", "stage", "codec", "row_count", "fresh_actual_count"}:
            continue
        value = _case_float(operation_row.get(key), float("nan"))
        if math.isfinite(value):
            metric_sums["sums"][key] = float(metric_sums["sums"].get(key, 0.0)) + value
            metric_sums["counts"][key] = int(metric_sums["counts"].get(key, 0)) + 1


def _finalize_operation_episode_metrics(episode, stage, metric_sums):
    row = {
        "episode": int(episode) + 1,
        "stage": str(stage),
        "codec": metric_sums.get("codec"),
        "row_count": int(metric_sums.get("row_count", 0)),
        "fresh_actual_count": int(metric_sums.get("fresh_actual_count", 0)),
    }
    for key in OPERATION_EPISODE_METRIC_COLUMNS:
        if key in row:
            continue
        count = int(metric_sums["counts"].get(key, 0))
        row[key] = None if count <= 0 else float(metric_sums["sums"].get(key, 0.0)) / float(count)
    return row


def _new_compression_episode_sums():
    return {
        "sums": {key: 0.0 for key in COMPRESSION_EPISODE_METRIC_COLUMNS},
        "counts": {key: 0 for key in COMPRESSION_EPISODE_METRIC_COLUMNS},
        "row_count": 0,
        "fresh_actual_count": 0,
        "codec": None,
        "backend": None,
    }


def _accumulate_compression_episode(metric_sums, compression_row):
    metric_sums["row_count"] += 1
    if bool(compression_row.get("fresh_actual", False)):
        metric_sums["fresh_actual_count"] += 1
    if metric_sums.get("codec") is None:
        metric_sums["codec"] = compression_row.get("codec")
    if metric_sums.get("backend") is None:
        metric_sums["backend"] = compression_row.get("backend")
    for key in COMPRESSION_EPISODE_METRIC_COLUMNS:
        if key in {"episode", "stage", "codec", "backend", "row_count", "fresh_actual_count"}:
            continue
        value = _case_float(compression_row.get(key), float("nan"))
        if math.isfinite(value):
            metric_sums["sums"][key] = float(metric_sums["sums"].get(key, 0.0)) + value
            metric_sums["counts"][key] = int(metric_sums["counts"].get(key, 0)) + 1


def _finalize_compression_episode_metrics(episode, stage, metric_sums):
    row = {
        "episode": int(episode) + 1,
        "stage": str(stage),
        "codec": metric_sums.get("codec"),
        "backend": metric_sums.get("backend"),
        "row_count": int(metric_sums.get("row_count", 0)),
        "fresh_actual_count": int(metric_sums.get("fresh_actual_count", 0)),
    }
    for key in COMPRESSION_EPISODE_METRIC_COLUMNS:
        if key in row:
            continue
        count = int(metric_sums["counts"].get(key, 0))
        row[key] = None if count <= 0 else float(metric_sums["sums"].get(key, 0.0)) / float(count)
    return row


def _new_checkpoint_metric_sums():
    return {
        "sums": {key: 0.0 for key in CHECKPOINT_AVG_KEYS},
        "counts": {key: 0 for key in CHECKPOINT_AVG_KEYS},
        "corr_pairs": {
            "surrogate_actual": [],
            "lcom_actual": [],
            "cp_main_actual": [],
            "sparsepcgc_aux_actual": [],
            "lcom_without_sparsepcgc_aux_actual": [],
        },
    }


def _add_checkpoint_metric(metric_sums, key, value):
    value = _case_float(value, float("nan"))
    if not math.isfinite(value):
        return
    metric_sums["sums"][key] = float(metric_sums["sums"].get(key, 0.0)) + value
    metric_sums["counts"][key] = int(metric_sums["counts"].get(key, 0)) + 1


def _accumulate_checkpoint_metrics(metric_sums, compression_row, operation_row, step_metric_values):
    explicit_values = {
        "total_loss": step_metric_values[0],
        "geom_loss": step_metric_values[1],
        "compression_loss_L_com": step_metric_values[2],
        "single_loss": step_metric_values[5],
        "node_loss": step_metric_values[6],
        "repair_loss": step_metric_values[10],
        "actual_total_bit_percent_fresh": compression_row.get("actual_total_bit_percent_fresh"),
        "actual_total_bit_percent_cached": compression_row.get("actual_total_bit_percent_cached"),
        "surrogate_pred_bit_percent": compression_row.get("surrogate_pred_bit_percent"),
        "proxy_delta_percent": compression_row.get("proxy_delta_percent"),
        "added_ratio_percent": operation_row.get("added_ratio_percent"),
        "deleted_ratio_percent": operation_row.get("deleted_ratio_percent"),
        "adjusted_ratio_percent": operation_row.get("adjusted_ratio_percent"),
        "add_prob_mean": operation_row.get("add_prob_mean"),
        "drop_prob_mean": operation_row.get("drop_prob_mean"),
        "hard_move_ratio": operation_row.get("hard_move_ratio"),
        "corr_surrogate_actual": compression_row.get("corr_surrogate_actual"),
        "corr_lcom_actual": compression_row.get("corr_lcom_actual"),
        "corr_cp_main_actual": compression_row.get("corr_cp_main_actual"),
        "corr_sparsepcgc_aux_actual": compression_row.get("corr_sparsepcgc_aux_actual"),
        "corr_lcom_without_sparsepcgc_aux_actual": compression_row.get("corr_lcom_without_sparsepcgc_aux_actual"),
        "sign_match_surrogate_actual": compression_row.get("sign_match_surrogate_actual"),
        "sign_match_lcom_actual": compression_row.get("sign_match_lcom_actual"),
        "sign_match_cp_main_actual": compression_row.get("sign_match_cp_main_actual"),
        "sign_match_sparsepcgc_aux_actual": compression_row.get("sign_match_sparsepcgc_aux_actual"),
        "sign_match_lcom_without_sparsepcgc_aux_actual": compression_row.get("sign_match_lcom_without_sparsepcgc_aux_actual"),
        "lcom_main": compression_row.get("lcom_main"),
        "lcom_aux": compression_row.get("lcom_aux"),
        "lcom_sparsepcgc_aux": compression_row.get("lcom_sparsepcgc_aux"),
        "sparsepcgc_aux_raw": compression_row.get("sparsepcgc_aux_raw"),
        "sparsepcgc_aux_weighted": compression_row.get("sparsepcgc_aux_weighted"),
        "lcom_without_sparsepcgc_aux": compression_row.get("lcom_without_sparsepcgc_aux"),
        "lcom_with_sparsepcgc_aux": compression_row.get("lcom_with_sparsepcgc_aux"),
        "sparsepcgc_active_coord_loss": compression_row.get("sparsepcgc_active_coord_loss"),
        "sparsepcgc_isolated_loss": compression_row.get("sparsepcgc_isolated_loss"),
        "sparsepcgc_entropy_loss": compression_row.get("sparsepcgc_entropy_loss"),
        "sparsepcgc_density_loss": compression_row.get("sparsepcgc_density_loss"),
        "active_coord_delta": operation_row.get("active_coord_delta"),
        "unique_coord_delta": operation_row.get("unique_coord_delta"),
        "add_effective_count": operation_row.get("add_effective_count"),
    }
    for key, value in explicit_values.items():
        _add_checkpoint_metric(metric_sums, key, value)

    actual_value = finite_float_or_none(compression_row.get("actual_total_bit_percent_fresh"))
    if actual_value is not None:
        pair_sources = {
            "surrogate_actual": compression_row.get("surrogate_pred_bit_percent"),
            "lcom_actual": compression_row.get("compression_loss_L_com"),
            "cp_main_actual": compression_row.get("cp_L_com_main"),
            "sparsepcgc_aux_actual": compression_row.get("sparsepcgc_aux_weighted"),
            "lcom_without_sparsepcgc_aux_actual": compression_row.get("lcom_without_sparsepcgc_aux"),
        }
        for key, metric_value in pair_sources.items():
            metric = finite_float_or_none(metric_value)
            if metric is not None:
                metric_sums["corr_pairs"].setdefault(key, []).append((metric, actual_value))


def _checkpoint_average(metric_sums, key):
    count = int(metric_sums["counts"].get(key, 0))
    if count <= 0:
        return None
    return float(metric_sums["sums"].get(key, 0.0)) / float(count)


def _checkpoint_corr(metric_sums, key):
    pairs = metric_sums.get("corr_pairs", {}).get(key, [])
    return rolling_pearson(pairs)


def _checkpoint_sign_match(metric_sums, key):
    pairs = metric_sums.get("corr_pairs", {}).get(key, [])
    values = [_sign_match_value(metric, actual) for metric, actual in pairs]
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def _gate_with_relative_reference(value, reference, rel_factor):
    value = _case_float(value, float("nan"))
    reference = _case_float(reference, float("nan"))
    rel_factor = float(rel_factor)
    if not math.isfinite(value) or rel_factor <= 0.0:
        return True
    if not math.isfinite(reference) or abs(reference) <= 1e-12:
        return True
    return value <= abs(reference) * rel_factor


def _gate_with_abs_max(value, abs_max):
    value = _case_float(value, float("nan"))
    abs_max = float(abs_max)
    if not math.isfinite(value) or abs_max <= 0.0:
        return True
    return value <= abs_max


def _finalize_checkpoint_metrics(args, stage, episode, plot, metric_sums, gate_refs):
    metrics = {
        "episode": int(episode) + 1,
        "stage": str(stage),
        "total_loss": plot.epi_loss_return(),
        "geom_loss": plot.epi_avg[1] if len(plot.epi_avg) > 1 else None,
        "compression_loss_L_com": plot.epi_avg[2] if len(plot.epi_avg) > 2 else None,
        "single_loss": plot.epi_avg[5] if len(plot.epi_avg) > 5 else None,
        "node_loss": plot.epi_avg[6] if len(plot.epi_avg) > 6 else None,
        "repair_loss": plot.epi_avg[10] if len(plot.epi_avg) > 10 else None,
        "fresh_actual_delta": _checkpoint_average(metric_sums, "actual_total_bit_percent_fresh"),
        "fresh_actual_count": int(metric_sums["counts"].get("actual_total_bit_percent_fresh", 0)),
        "cached_actual_delta": _checkpoint_average(metric_sums, "actual_total_bit_percent_cached"),
        "cached_actual_count": int(metric_sums["counts"].get("actual_total_bit_percent_cached", 0)),
        "surrogate_pred_bit_percent": _checkpoint_average(metric_sums, "surrogate_pred_bit_percent"),
        "proxy_delta_percent": _checkpoint_average(metric_sums, "proxy_delta_percent"),
        "corr_surrogate_actual": _checkpoint_corr(metric_sums, "surrogate_actual"),
        "corr_lcom_actual": _checkpoint_corr(metric_sums, "lcom_actual"),
        "corr_cp_main_actual": _checkpoint_corr(metric_sums, "cp_main_actual"),
        "corr_sparsepcgc_aux_actual": _checkpoint_corr(metric_sums, "sparsepcgc_aux_actual"),
        "corr_lcom_without_sparsepcgc_aux_actual": _checkpoint_corr(metric_sums, "lcom_without_sparsepcgc_aux_actual"),
        "sign_match_surrogate_actual": _checkpoint_sign_match(metric_sums, "surrogate_actual"),
        "sign_match_lcom_actual": _checkpoint_sign_match(metric_sums, "lcom_actual"),
        "sign_match_cp_main_actual": _checkpoint_sign_match(metric_sums, "cp_main_actual"),
        "sign_match_sparsepcgc_aux_actual": _checkpoint_sign_match(metric_sums, "sparsepcgc_aux_actual"),
        "sign_match_lcom_without_sparsepcgc_aux_actual": _checkpoint_sign_match(metric_sums, "lcom_without_sparsepcgc_aux_actual"),
        "lcom_main": _checkpoint_average(metric_sums, "lcom_main"),
        "lcom_aux": _checkpoint_average(metric_sums, "lcom_aux"),
        "lcom_sparsepcgc_aux": _checkpoint_average(metric_sums, "lcom_sparsepcgc_aux"),
        "sparsepcgc_aux_raw": _checkpoint_average(metric_sums, "sparsepcgc_aux_raw"),
        "sparsepcgc_aux_weighted": _checkpoint_average(metric_sums, "sparsepcgc_aux_weighted"),
        "lcom_without_sparsepcgc_aux": _checkpoint_average(metric_sums, "lcom_without_sparsepcgc_aux"),
        "lcom_with_sparsepcgc_aux": _checkpoint_average(metric_sums, "lcom_with_sparsepcgc_aux"),
        "sparsepcgc_active_coord_loss": _checkpoint_average(metric_sums, "sparsepcgc_active_coord_loss"),
        "sparsepcgc_isolated_loss": _checkpoint_average(metric_sums, "sparsepcgc_isolated_loss"),
        "sparsepcgc_entropy_loss": _checkpoint_average(metric_sums, "sparsepcgc_entropy_loss"),
        "sparsepcgc_density_loss": _checkpoint_average(metric_sums, "sparsepcgc_density_loss"),
        "added_ratio_percent": _checkpoint_average(metric_sums, "added_ratio_percent"),
        "deleted_ratio_percent": _checkpoint_average(metric_sums, "deleted_ratio_percent"),
        "adjusted_ratio_percent": _checkpoint_average(metric_sums, "adjusted_ratio_percent"),
        "active_coord_delta": _checkpoint_average(metric_sums, "active_coord_delta"),
        "unique_coord_delta": _checkpoint_average(metric_sums, "unique_coord_delta"),
        "add_effective_count": _checkpoint_average(metric_sums, "add_effective_count"),
    }
    stage_refs = gate_refs.setdefault(str(stage), {})
    for key in ("geom_loss", "repair_loss"):
        value = _case_float(metrics.get(key), float("nan"))
        if key not in stage_refs and math.isfinite(value):
            stage_refs[key] = value

    geom_ref = stage_refs.get("geom_loss")
    repair_ref = stage_refs.get("repair_loss")
    geom_ok = True
    if bool(getattr(args, "checkpoint_geom_gate", True)):
        geom_ok = _gate_with_relative_reference(
            metrics.get("geom_loss"),
            geom_ref,
            float(getattr(args, "checkpoint_geom_rel_factor", 1.5)),
        ) and _gate_with_abs_max(
            metrics.get("geom_loss"),
            float(getattr(args, "checkpoint_geom_abs_max", 0.0)),
        )

    repair_ok = _gate_with_relative_reference(
        metrics.get("repair_loss"),
        repair_ref,
        float(getattr(args, "checkpoint_repair_rel_factor", 0.0)),
    ) and _gate_with_abs_max(
        metrics.get("repair_loss"),
        float(getattr(args, "checkpoint_repair_abs_max", 10.0)),
    )
    node_ok = _gate_with_abs_max(metrics.get("node_loss"), float(getattr(args, "checkpoint_node_abs_max", 100.0)))
    single_ok = _gate_with_abs_max(metrics.get("single_loss"), float(getattr(args, "checkpoint_single_abs_max", 100.0)))
    op_limit = float(getattr(args, "checkpoint_operation_ratio_max", 100.0))
    operation_ok = True
    if op_limit >= 0.0:
        for key in ("added_ratio_percent", "deleted_ratio_percent", "adjusted_ratio_percent"):
            value = _case_float(metrics.get(key), float("nan"))
            if math.isfinite(value) and value > op_limit:
                operation_ok = False
                break

    if not bool(getattr(args, "checkpoint_safety_gate", True)):
        repair_ok = node_ok = single_ok = operation_ok = True

    metrics.update(
        {
            "geometry_ok": bool(geom_ok),
            "safety_ok": bool(geom_ok and repair_ok and node_ok and single_ok and operation_ok),
            "repair_ok": bool(repair_ok),
            "node_ok": bool(node_ok),
            "single_ok": bool(single_ok),
            "operation_ok": bool(operation_ok),
            "geom_reference": geom_ref,
            "repair_reference": repair_ref,
        }
    )
    return metrics


def _optimizer_lrs(optimizer):
    if optimizer is None:
        return []
    return [float(group.get("lr", 0.0)) for group in optimizer.param_groups]


def _set_optimizer_lrs(optimizer, lrs):
    if optimizer is None:
        return
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = float(lr)


def _run_surrogate_pretrain(
    *,
    model,
    args,
    loss,
    seq_datasets,
    loader_kwargs,
    metric_csv_paths,
    ckpt_dir,
    writer,
    use_cuda,
    use_amp,
    amp_dtype,
):
    print(f"Surrogate pretrain step: {int(getattr(args, 'surrogate_pretrain_steps', 0))}")
    steps = max(int(getattr(args, "surrogate_pretrain_steps", 0)), 0)
    if steps <= 0:
        return
    backend = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    if not backend.endswith("_surrogate"):
        writer.write(f"SurrogatePretrain skipped: backend={backend} is not a surrogate backend.")
        return
    pretrain_mode = str(getattr(args, "surrogate_pretrain_mode", "full")).strip().lower()
    if pretrain_mode not in {"full", "subtree", "hybrid"}:
        raise ValueError("--surrogate_pretrain_mode must be one of: full, subtree, hybrid")

    refresh_interval = max(int(getattr(args, "surrogate_pretrain_actual_refresh_interval", 10)), 0)
    replay_enabled = bool(getattr(args, "surrogate_pretrain_use_replay", True))
    replay_steps = max(int(getattr(args, "surrogate_pretrain_replay_steps", 4)), 0)
    replay_batch = max(int(getattr(args, "surrogate_pretrain_replay_batch_size", 16)), 1)
    replay_buffer_size = max(int(getattr(args, "surrogate_pretrain_replay_buffer_size", 256)), 0)
    debug_interval = int(getattr(args, "surrogate_pretrain_sparsepcgc_debug_interval", 10))
    teacher_type = str(getattr(args, "surrogate_pretrain_subtree_teacher_type", "local_proxy")).strip().lower()
    full_calibration_interval = max(int(getattr(args, "surrogate_pretrain_full_calibration_interval", 50)), 1)
    full_calibration_steps = max(int(getattr(args, "surrogate_pretrain_full_calibration_steps", 1)), 1)
    max_wall_time_sec = max(float(getattr(args, "surrogate_pretrain_max_wall_time_sec", 0.0)), 0.0)
    writer.write(
        "SurrogatePretrain start: "
        f"steps={steps}, lr={float(getattr(args, 'surrogate_pretrain_lr', 1e-4)):.6g}, "
        f"freeze_network={bool(getattr(args, 'surrogate_pretrain_freeze_network', True))}, "
        f"refresh_interval={refresh_interval}, "
        f"replay_enabled={replay_enabled}, replay_steps={replay_steps}, "
        f"replay_batch={replay_batch}, replay_buffer={replay_buffer_size}, "
        f"sparsepcgc_debug_interval={debug_interval}, mode={pretrain_mode}, "
        f"teacher_type={teacher_type}, full_calibration_interval={full_calibration_interval}, "
        f"full_calibration_steps={full_calibration_steps}, "
        f"subtree_steps_per_full={int(getattr(args, 'surrogate_pretrain_subtree_steps_per_full', full_calibration_interval))}, "
        f"max_wall_time_sec={max_wall_time_sec:.1f}"
    )
    if pretrain_mode == "full" and steps >= 1000:
        writer.write(
            "[WARN] surrogate_pretrain_mode=full with "
            f"{steps} steps can be extremely slow. Consider --surrogate_pretrain_mode subtree or hybrid."
        )
    elif pretrain_mode == "subtree":
        writer.write(
            "[SurrogatePretrain] mode=subtree uses "
            f"{teacher_type} teacher. Full SparsePCGC actual codec will not be called every step."
        )
    elif pretrain_mode == "hybrid":
        writer.write(
            "[SurrogatePretrain] mode=hybrid uses subtree steps plus full calibration. "
            f"Full actual calibration interval={full_calibration_interval}, steps_per_window={full_calibration_steps}."
        )
    if teacher_type == "local_actual":
        writer.write(
            "[WARN] surrogate_pretrain_subtree_teacher_type=local_actual encodes subtree-only point clouds. "
            "Subtree actual bit is a local teacher and is not identical to full-cloud SparsePCGC bit because "
            "bbox/origin/header/global density/context can differ."
        )
    if teacher_type == "inherited_full":
        writer.write(
            "[WARN] surrogate_pretrain_subtree_teacher_type=inherited_full assigns a full-cloud teacher/cache "
            "to subtree steps. This is biased and should be used only for calibration experiments."
        )

    model_was_training = model.training
    param_states = [(param, bool(param.requires_grad)) for param in model.parameters()]
    surrogate_optimizer = getattr(loss, "surrogate_optimizer", None)
    original_surrogate_lrs = _optimizer_lrs(surrogate_optimizer)
    original_replay_max_entries = getattr(loss, "surrogate_replay_max_entries", None)
    pretrain_lr = float(getattr(args, "surrogate_pretrain_lr", 1e-4))
    if pretrain_lr > 0.0 and original_surrogate_lrs:
        _set_optimizer_lrs(surrogate_optimizer, [pretrain_lr for _ in original_surrogate_lrs])

    saved_args = {
        "_global_train_step": getattr(args, "_global_train_step", 0),
        "_collect_sparsepcgc_debug": getattr(args, "_collect_sparsepcgc_debug", False),
        "_surrogate_pretrain_timing_enabled": getattr(args, "_surrogate_pretrain_timing_enabled", False),
        "_surrogate_pretrain_active": getattr(args, "_surrogate_pretrain_active", False),
        "_surrogate_pretrain_mode": getattr(args, "_surrogate_pretrain_mode", None),
        "_surrogate_pretrain_teacher_type": getattr(args, "_surrogate_pretrain_teacher_type", None),
        "_surrogate_pretrain_actual_scope": getattr(args, "_surrogate_pretrain_actual_scope", None),
        "_surrogate_pretrain_full_calibration": getattr(args, "_surrogate_pretrain_full_calibration", False),
        "compression_surrogate_refresh_interval": getattr(args, "compression_surrogate_refresh_interval", 0),
        "compression_surrogate_replay_steps": getattr(args, "compression_surrogate_replay_steps", 0),
        "compression_surrogate_replay_batch": getattr(args, "compression_surrogate_replay_batch", 1),
        "compression_surrogate_replay_entries": getattr(args, "compression_surrogate_replay_entries", 0),
        "compression_surrogate_reuse_last_target": getattr(args, "compression_surrogate_reuse_last_target", True),
    }
    corr_pairs = {}
    abs_error_history = []
    step_times = []
    fresh_actual_count = 0
    completed_steps = 0
    last_corr = None
    last_sign_match = None
    early_stop_hits = 0
    early_stop_reason = None
    eta_warned = False
    log_interval = max(int(getattr(args, "surrogate_pretrain_log_interval", 10)), 1)
    print_interval = max(int(getattr(args, "surrogate_pretrain_print_interval", log_interval)), 1)
    pretrain_start_time = time.perf_counter()
    last_log_time = 0.0

    try:
        if bool(getattr(args, "surrogate_pretrain_freeze_network", True)):
            for param, _old_state in param_states:
                param.requires_grad_(False)
        args._surrogate_pretrain_active = True
        args._surrogate_pretrain_timing_enabled = True
        args._surrogate_pretrain_mode = pretrain_mode
        args._surrogate_pretrain_teacher_type = teacher_type
        args._surrogate_pretrain_actual_scope = "full"
        args._surrogate_pretrain_full_calibration = False
        args.compression_surrogate_refresh_interval = refresh_interval
        args.compression_surrogate_replay_steps = replay_steps if replay_enabled else 0
        args.compression_surrogate_replay_batch = replay_batch
        args.compression_surrogate_replay_entries = replay_buffer_size
        args.compression_surrogate_reuse_last_target = bool(
            getattr(args, "surrogate_pretrain_allow_stale_target", True)
        )
        if original_replay_max_entries is not None:
            loss.surrogate_replay_max_entries = replay_buffer_size
            if replay_buffer_size > 0 and len(getattr(loss, "surrogate_replay", [])) > replay_buffer_size:
                loss.surrogate_replay = list(loss.surrogate_replay[-replay_buffer_size:])
                loss.surrogate_replay_next = len(loss.surrogate_replay) % replay_buffer_size
            elif replay_buffer_size <= 0:
                loss.surrogate_replay = []
                loss.surrogate_replay_next = 0
        model.train()

        while completed_steps < steps and early_stop_reason is None:
            progressed = False
            for _seq_dir, dataset in seq_datasets:
                loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
                data_wait_t0 = time.perf_counter()
                for local_step, pts in enumerate(loader):
                    surrogate_st = time.time()
                    data_time = time.perf_counter() - data_wait_t0
                    if completed_steps >= steps or early_stop_reason is not None:
                        break
                    progressed = True
                    step_zero = int(completed_steps)
                    step_number = step_zero + 1
                    step_t0 = data_wait_t0
                    file_path = dataset.files[local_step]
                    base_cache_key = f"surrogate_pretrain|{_make_step_cache_key(file_path, args)}"
                    cache_key = base_cache_key
                    args._global_train_step = step_zero
                    full_window_pos = (step_number - 1) % full_calibration_interval
                    full_calibration = bool(
                        pretrain_mode == "hybrid"
                        and full_window_pos >= max(full_calibration_interval - full_calibration_steps, 0)
                    )
                    subtree_enabled = bool(pretrain_mode in {"subtree", "hybrid"} and not full_calibration)
                    actual_scope = "subtree" if subtree_enabled else "full"
                    effective_teacher_type = teacher_type if subtree_enabled else "full_actual"
                    if full_calibration:
                        cache_key = f"{base_cache_key}|hybrid_full_calibration"
                    if not subtree_enabled:
                        should_refresh_actual = (
                            full_calibration
                            or step_zero == 0
                            or refresh_interval == 1
                            or (refresh_interval > 1 and step_zero % refresh_interval == 0)
                        )
                    elif teacher_type == "local_actual":
                        should_refresh_actual = (
                            step_zero == 0
                            or refresh_interval == 1
                            or (refresh_interval > 1 and step_zero % refresh_interval == 0)
                        )
                    else:
                        should_refresh_actual = False
                    debug_collect = bool(
                        debug_interval > 0
                        and (step_number == 1 or step_number % debug_interval == 0)
                    )
                    args._collect_sparsepcgc_debug = debug_collect
                    args._surrogate_pretrain_mode = pretrain_mode
                    args._surrogate_pretrain_teacher_type = effective_teacher_type
                    args._surrogate_pretrain_actual_scope = actual_scope
                    args._surrogate_pretrain_full_calibration = full_calibration

                    model_t0 = time.perf_counter()
                    subtree_sampling_time = 0.0
                    subtree_meta = {
                        "depth": 0,
                        "point_count": 0,
                        "bbox_min": None,
                        "bbox_max": None,
                        "retry_count": 0,
                        "skip_reason": "none",
                        "subtree_key": None,
                        "total_subtree_count": 0,
                        "eligible_subtree_count": 0,
                        "selected_subtree_count": 0,
                    }
                    comp_debug = None
                    refresh_actual_arg = "always" if full_calibration else should_refresh_actual
                    if should_refresh_actual:
                        writer.write(
                            "[SurrogatePretrainActual] start "
                            f"mode={pretrain_mode} step={step_number}/{steps} "
                            f"scope={actual_scope} teacher={effective_teacher_type}"
                        )
                    with torch.enable_grad():
                        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                        if subtree_enabled:
                            subtree_sample = _build_surrogate_pretrain_subtree_sample(
                                pts,
                                args,
                                base_cache_key,
                                use_cuda,
                                global_step=step_zero,
                            )
                            subtree_sampling_time = _case_float(subtree_sample.get("sampling_time", 0.0), 0.0)
                            subtree_meta.update(
                                {
                                    "depth": _case_int(subtree_sample.get("depth", 0)),
                                    "point_count": _case_int(subtree_sample.get("point_count", 0)),
                                    "bbox_min": subtree_sample.get("bbox_min"),
                                    "bbox_max": subtree_sample.get("bbox_max"),
                                    "retry_count": _case_int(subtree_sample.get("retry_count", 0)),
                                    "skip_reason": str(subtree_sample.get("skip_reason", "none")),
                                    "subtree_key": subtree_sample.get("subtree_key"),
                                    "total_subtree_count": _case_int(subtree_sample.get("total_subtree_count", 0)),
                                    "eligible_subtree_count": _case_int(subtree_sample.get("eligible_subtree_count", 0)),
                                    "selected_subtree_count": _case_int(subtree_sample.get("selected_subtree_count", 0)),
                                }
                            )
                            if subtree_meta["point_count"] <= 0 or subtree_meta["skip_reason"] == "empty_selected_subtree":
                                model_time = time.perf_counter() - model_t0
                                comp_debug = {
                                    "teacher_mode": "skip",
                                    "teacher_skipped": True,
                                    "actual_value_source": f"subtree_skip:{subtree_meta['skip_reason']}",
                                    "surrogate_replay_size": len(getattr(loss, "surrogate_replay", [])),
                                    "surrogate_replay_sample_count": 0,
                                    "timing": {},
                                }
                            else:
                                subtree_xyz = subtree_sample["subtree_xyz"]
                                subtree_attr = subtree_sample.get("subtree_attr")
                                cache_key = subtree_sample["subtree_cache_key"]
                                with autocast_ctx:
                                    (
                                        gen_subtree_pts,
                                        _L_attr,
                                        _L_policy,
                                        _L_actuator,
                                        final_w,
                                        _Lp_out,
                                        _La_fit,
                                        _La_rep,
                                        _out_label,
                                    ) = model.forward(
                                        subtree_xyz,
                                        subtree_attr,
                                        cache_key=cache_key,
                                        return_attr_output=False,
                                    )
                                    gen_xyz = gen_subtree_pts[:, :3, :]
                                    final_w_for_loss = None
                                    if str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() != "hard":
                                        final_w_for_loss = final_w
                                    compression_gen_xyz, _noise_debug = prepare_compression_points(
                                        gen_xyz,
                                        args,
                                        model,
                                        collect_stats=debug_collect,
                                    )
                                    model_time = time.perf_counter() - model_t0
                                    saved_stale = getattr(args, "surrogate_pretrain_allow_stale_target", True)
                                    saved_reuse = getattr(args, "compression_surrogate_reuse_last_target", True)
                                    if teacher_type in {"local_proxy", "none"}:
                                        args.surrogate_pretrain_allow_stale_target = False
                                        args.compression_surrogate_reuse_last_target = False
                                    try:
                                        loss.get_compression_loss(
                                            args,
                                            gen_xyz=compression_gen_xyz,
                                            gt_xyz=subtree_xyz[:, :3, :],
                                            final_w=final_w_for_loss,
                                            cache_key=cache_key,
                                            refresh_actual_gen=refresh_actual_arg,
                                            actual_gen_xyz=gen_xyz,
                                        )
                                    finally:
                                        args.surrogate_pretrain_allow_stale_target = saved_stale
                                        args.compression_surrogate_reuse_last_target = saved_reuse
                        else:
                            input_xyz, patches, centroid_xyz, fd_xyz = _prepare_whole_cloud_inputs(
                                pts,
                                args,
                                cache_key,
                                use_cuda,
                            )
                            with autocast_ctx:
                                gen_patches, _L_attr, _L_policy, _L_actuator, final_w, _Lp_out, _La_fit, _La_rep, _out_label = model.forward(
                                    patches,
                                    None,
                                    cache_key=cache_key,
                                    coord_scale=fd_xyz,
                                    return_attr_output=False,
                                )
                                gen_xyz = (centroid_xyz + gen_patches[:, :3, :] * fd_xyz).contiguous()
                                final_w_for_loss = None
                                if str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() != "hard":
                                    final_w_for_loss = final_w
                                compression_gen_xyz, _noise_debug = prepare_compression_points(
                                    gen_xyz,
                                    args,
                                    model,
                                    collect_stats=True,
                                )
                                model_time = time.perf_counter() - model_t0
                                loss.get_compression_loss(
                                    args,
                                    gen_xyz=compression_gen_xyz,
                                    gt_xyz=input_xyz[:, :3, :],
                                    final_w=final_w_for_loss,
                                    cache_key=cache_key,
                                    refresh_actual_gen=refresh_actual_arg,
                                    actual_gen_xyz=gen_xyz,
                                )

                    if comp_debug is None:
                        comp_debug = dict(getattr(loss, "last_compression_debug", {}) or {})
                    comp_timing = comp_debug.get("timing", {}) or {}
                    actual_value = finite_float_or_none(comp_debug.get("actual_total_bit_percent", None))
                    pred_value = finite_float_or_none(comp_debug.get("surrogate_pred_bit", None))
                    teacher_mode_value = str(comp_debug.get("teacher_mode", "refresh" if should_refresh_actual else "skip"))
                    actual_source = str(comp_debug.get("actual_value_source", ""))
                    if actual_source.startswith(("missing", "target_missing", "local_proxy", "subtree_skip")):
                        actual_value = None
                    if actual_value is not None and pred_value is not None and _is_fresh_actual(args, comp_debug):
                        fresh_actual_count += 1
                        last_corr, last_sign_match, _count = _append_corr_pair(
                            corr_pairs,
                            "surrogate_actual",
                            pred_value,
                            actual_value,
                            max(int(getattr(args, "sparsepcgc_corr_window", 100)), 2),
                        )
                    abs_error = None
                    if actual_value is not None and pred_value is not None:
                        abs_error = abs(pred_value - actual_value)
                    if abs_error is not None and _is_fresh_actual(args, comp_debug):
                        abs_error_history.append(float(abs_error))
                        max_abs_history = max(int(getattr(args, "sparsepcgc_corr_window", 100)), 2)
                        if len(abs_error_history) > max_abs_history:
                            del abs_error_history[:-max_abs_history]

                    step_time = time.perf_counter() - step_t0
                    step_times.append(float(step_time))
                    avg_step_time = sum(step_times) / max(len(step_times), 1)
                    eta_seconds = max(steps - step_number, 0) * avg_step_time
                    actual_eval_time = _case_float(comp_timing.get("actual_encode", 0.0), 0.0)
                    if actual_eval_time <= 0.0 and bool(comp_debug.get("teacher_refresh", False)):
                        actual_eval_time = _case_float(comp_debug.get("actual_encode_time_total", 0.0), 0.0)
                    if bool(comp_debug.get("teacher_refreshed", comp_debug.get("teacher_refresh", False))):
                        writer.write(
                            "[SurrogatePretrainActual] done "
                            f"mode={pretrain_mode} step={step_number}/{steps} "
                            f"scope={actual_scope} teacher={effective_teacher_type} "
                            f"time={actual_eval_time:.2f}s"
                        )
                    surrogate_update_time = (
                        _case_float(comp_timing.get("surrogate_fit", 0.0), 0.0)
                        + _case_float(comp_timing.get("surrogate_replay", 0.0), 0.0)
                        + (
                            _case_float(comp_timing.get("target_cache", 0.0), 0.0)
                            if bool(comp_debug.get("teacher_stale", False))
                            else 0.0
                        )
                    )
                    gpu_alloc_mb = _cuda_alloc_mb(use_cuda)
                    cpu_rss_mb = _process_rss_mb()
                    current_lr = _optimizer_lrs(surrogate_optimizer)
                    param_norm = _surrogate_param_norm(loss)
                    row = {
                        "surrogate_pretrain_step": step_number,
                        "pretrain_mode": pretrain_mode,
                        "pretrain_teacher_type": effective_teacher_type,
                        "sample_name": os.path.basename(str(file_path)),
                        "codec": comp_debug.get("teacher_codec", getattr(args, "compress", "unknown")),
                        "backend": backend,
                        "surrogate_pretrain_loss": _case_float(comp_debug.get("surrogate_train_loss", float("nan")), float("nan")),
                        "surrogate_pretrain_pred_bit_percent": pred_value,
                        "surrogate_pretrain_actual_bit_percent": actual_value,
                        "surrogate_pretrain_abs_error": abs_error,
                        "surrogate_pretrain_corr": last_corr,
                        "surrogate_pretrain_sign_match": last_sign_match,
                        "surrogate_pretrain_teacher_refresh": bool(comp_debug.get("teacher_refresh", False)),
                        "surrogate_pretrain_target_age": _case_int(comp_debug.get("teacher_target_age", 0)),
                        "surrogate_pretrain_fresh_actual_count": fresh_actual_count,
                        "pretrain_step_time": step_time,
                        "pretrain_actual_eval_time": actual_eval_time,
                        "pretrain_surrogate_update_time": surrogate_update_time,
                        "pretrain_data_time": data_time,
                        "pretrain_subtree_sampling_time": subtree_sampling_time,
                        "pretrain_model_time": model_time,
                        "pretrain_log_time": last_log_time,
                        "pretrain_eta_seconds": eta_seconds,
                        "pretrain_gpu_alloc_mb": gpu_alloc_mb,
                        "pretrain_cpu_rss_mb": cpu_rss_mb,
                        "teacher_mode": teacher_mode_value,
                        "teacher_refreshed": bool(comp_debug.get("teacher_refreshed", comp_debug.get("teacher_refresh", False))),
                        "teacher_replayed": bool(comp_debug.get("teacher_replayed", False)),
                        "teacher_stale": bool(comp_debug.get("teacher_stale", False)),
                        "teacher_skipped": bool(comp_debug.get("teacher_skipped", False)),
                        "teacher_target_age": _case_int(comp_debug.get("teacher_target_age", 0)),
                        "replay_buffer_size": _case_int(comp_debug.get("surrogate_replay_size", len(getattr(loss, "surrogate_replay", [])))),
                        "replay_sample_count": _case_int(comp_debug.get("surrogate_replay_sample_count", 0)),
                        "fresh_actual_count": fresh_actual_count,
                        "sparsepcgc_debug_collected": bool(comp_debug.get("sparsepcgc_debug_collected", False)),
                        "sparsepcgc_debug_time": _case_float(comp_debug.get("sparsepcgc_debug_time", 0.0), 0.0),
                        "pretrain_subtree_enabled": bool(subtree_enabled),
                        "pretrain_subtree_depth": subtree_meta["depth"] if subtree_enabled else None,
                        "pretrain_subtree_point_count": subtree_meta["point_count"] if subtree_enabled else None,
                        "pretrain_subtree_bbox_min": subtree_meta["bbox_min"],
                        "pretrain_subtree_bbox_max": subtree_meta["bbox_max"],
                        "pretrain_subtree_retry_count": subtree_meta["retry_count"],
                        "pretrain_subtree_skip_reason": subtree_meta["skip_reason"],
                        "pretrain_subtree_key": subtree_meta["subtree_key"],
                        "pretrain_subtree_total_count": subtree_meta["total_subtree_count"],
                        "pretrain_subtree_eligible_count": subtree_meta["eligible_subtree_count"],
                        "pretrain_subtree_selected_count": subtree_meta["selected_subtree_count"],
                        "pretrain_full_calibration": bool(full_calibration),
                        "pretrain_actual_scope": actual_scope,
                        "surrogate_param_norm": param_norm,
                        "surrogate_pretrain_lr": current_lr[0] if current_lr else None,
                    }

                    log_t0 = time.perf_counter()
                    should_print = step_number == 1 or step_number % print_interval == 0 or step_number >= steps
                    if should_print:
                        actual_text = "NA" if actual_value is None else f"{_case_float(actual_value, float('nan')):.6f}"
                        abs_text = "NA" if abs_error is None else f"{_case_float(abs_error, float('nan')):.6f}"
                        writer.write(
                            "[SurrogatePretrain] "
                            f"mode={pretrain_mode} step={step_number}/{steps} "
                            f"depth={row['pretrain_subtree_depth'] if subtree_enabled else 'NA'} "
                            f"pts={row['pretrain_subtree_point_count'] if subtree_enabled else 'NA'} "
                            f"teacher={row['teacher_mode']} "
                            f"elapsed={_format_duration_seconds(time.perf_counter() - pretrain_start_time)} "
                            f"step_time={step_time:.2f}s avg={avg_step_time:.2f}s "
                            f"eta={_format_duration_seconds(eta_seconds)} "
                            f"fit={_case_float(row['surrogate_pretrain_loss'], float('nan')):.6f} "
                            f"pred={_case_float(pred_value, float('nan')):.6f}, "
                            f"actual={actual_text}, "
                            f"abs={abs_text}, "
                            f"corr={format_corr(last_corr, len(corr_pairs.get('surrogate_actual', [])))}, "
                            f"sign={_case_float(last_sign_match, float('nan')):.6f}, "
                            f"age={row['teacher_target_age']} "
                            f"fresh={fresh_actual_count} replay={row['replay_sample_count']} "
                            f"actual_time={actual_eval_time:.2f}s model={model_time:.2f}s "
                            f"sur_fit={surrogate_update_time:.2f}s data={data_time:.2f}s "
                            f"subtree={subtree_sampling_time:.2f}s "
                            f"log={last_log_time:.4f}s "
                            f"gpu={_case_float(gpu_alloc_mb, 0.0):.1f}MB "
                            f"cpu={_case_float(cpu_rss_mb, 0.0):.1f}MB "
                            f"debug={bool(row['sparsepcgc_debug_collected'])} "
                            f"subtree_skip={row['pretrain_subtree_skip_reason']}"
                        )
                    row["pretrain_log_time"] = time.perf_counter() - log_t0
                    last_log_time = row["pretrain_log_time"]
                    _append_csv_row(
                        metric_csv_paths.get("surrogate_pretrain_step"),
                        SURROGATE_PRETRAIN_COLUMNS,
                        row,
                    )

                    if step_number in {1, 3} or (step_number % max(print_interval, 1) == 0):
                        estimated_total = avg_step_time * float(steps)
                        if estimated_total > 24.0 * 3600.0 and not eta_warned:
                            eta_warned = True
                            writer.write(
                                "[WARN] Surrogate pretrain estimated time is "
                                f"{estimated_total / 3600.0:.1f} hours. Consider increasing "
                                "--surrogate_pretrain_actual_refresh_interval, enabling replay, "
                                "using --surrogate_pretrain_mode subtree/hybrid, or reducing --surrogate_pretrain_steps."
                            )

                    min_corr = float(getattr(args, "surrogate_pretrain_min_corr", -1.0))
                    min_sign = float(getattr(args, "surrogate_pretrain_min_sign_match", -1.0))
                    min_abs_error = float(getattr(args, "surrogate_pretrain_min_abs_error", -1.0))
                    min_fresh = max(int(getattr(args, "surrogate_pretrain_min_fresh_samples", 30)), 0)
                    patience = max(int(getattr(args, "surrogate_pretrain_early_stop_patience", 0)), 0)
                    early_enabled = patience > 0 and (min_corr >= 0.0 or min_sign >= 0.0 or min_abs_error >= 0.0)
                    mean_abs_error = _mean_finite(abs_error_history)
                    corr_ok_step = min_corr < 0.0 or (last_corr is not None and last_corr >= min_corr)
                    sign_ok_step = min_sign < 0.0 or (last_sign_match is not None and last_sign_match >= min_sign)
                    abs_ok_step = min_abs_error < 0.0 or (mean_abs_error is not None and mean_abs_error <= min_abs_error)
                    fresh_ok_step = fresh_actual_count >= min_fresh
                    if early_enabled and fresh_ok_step and corr_ok_step and sign_ok_step and abs_ok_step:
                        early_stop_hits += 1
                        if early_stop_hits >= patience:
                            early_stop_reason = (
                                f"corr={format_corr(last_corr, len(corr_pairs.get('surrogate_actual', [])))}, "
                                f"sign={_case_float(last_sign_match, float('nan')):.6f}, "
                                f"mean_abs={_case_float(mean_abs_error, float('nan')):.6f}, "
                                f"fresh={fresh_actual_count}, patience={patience}"
                            )
                            writer.write(f"SurrogatePretrainEarlyStop: {early_stop_reason}")
                    elif early_enabled:
                        early_stop_hits = 0

                    completed_steps += 1
                    surrogate_en = time.time()
                    print(f"Surrogate Step: {completed_steps + 1}/{steps} | {surrogate_en - surrogate_st}sec")
                    if max_wall_time_sec > 0.0:
                        elapsed_wall = time.perf_counter() - pretrain_start_time
                        if elapsed_wall >= max_wall_time_sec:
                            early_stop_reason = (
                                f"max_wall_time_sec={max_wall_time_sec:.1f}, "
                                f"elapsed={elapsed_wall:.1f}"
                            )
                            writer.write(f"SurrogatePretrainEarlyStop: {early_stop_reason}")
                    data_wait_t0 = time.perf_counter()
                if completed_steps >= steps or early_stop_reason is not None:
                    break
            if not progressed:
                writer.write("SurrogatePretrain stopped early: no training samples were available.")
                break

        min_corr = float(getattr(args, "surrogate_pretrain_min_corr", -1.0))
        min_sign = float(getattr(args, "surrogate_pretrain_min_sign_match", -1.0))
        corr_ok = min_corr < 0.0 or (last_corr is not None and last_corr >= min_corr)
        sign_ok = min_sign < 0.0 or (last_sign_match is not None and last_sign_match >= min_sign)
        final_param_norm = _surrogate_param_norm(loss)
        writer.write(
            "SurrogatePretrainSummary: "
            f"mode={pretrain_mode}, completed_steps={completed_steps}, fresh_actual_count={fresh_actual_count}, "
            f"corr={format_corr(last_corr, len(corr_pairs.get('surrogate_actual', [])))}, "
            f"sign_match={_case_float(last_sign_match, float('nan')):.6f}, "
            f"mean_abs_error={_case_float(_mean_finite(abs_error_history), float('nan')):.6f}, "
            f"corr_ok={bool(corr_ok)}, sign_match_ok={bool(sign_ok)}, "
            f"early_stop={early_stop_reason or 'none'}, "
            f"surrogate_param_norm={_case_float(final_param_norm, float('nan')):.6f}"
        )
        writer.write(
            "[SurrogatePretrain] complete "
            f"mode={pretrain_mode} steps={completed_steps} "
            f"surrogate_param_norm={_case_float(final_param_norm, float('nan')):.6f} "
            f"lr={_optimizer_lrs(surrogate_optimizer)[0] if _optimizer_lrs(surrogate_optimizer) else 'NA'}"
        )
        if bool(getattr(args, "surrogate_pretrain_checkpoint", True)):
            path = os.path.join(ckpt_dir, "surrogate_pretrain.pth")
            torch.save(
                {
                    "compression_surrogate": loss.compression_surrogate.state_dict(),
                    "completed_steps": completed_steps,
                    "fresh_actual_count": fresh_actual_count,
                    "corr": last_corr,
                    "sign_match": last_sign_match,
                    "surrogate_param_norm": final_param_norm,
                    "early_stop_reason": early_stop_reason,
                },
                path,
            )
            writer.write(f"SurrogatePretrainCheckpoint: {path}")
    finally:
        for param, old_state in param_states:
            param.requires_grad_(old_state)
        for key, value in saved_args.items():
            setattr(args, key, value)
        if original_replay_max_entries is not None:
            loss.surrogate_replay_max_entries = original_replay_max_entries
        if original_surrogate_lrs:
            joint_scale = float(getattr(args, "surrogate_joint_lr_scale", 0.1))
            joint_lrs = [lr * joint_scale for lr in original_surrogate_lrs]
            _set_optimizer_lrs(surrogate_optimizer, joint_lrs)
            writer.write(
                "SurrogatePretrainJointLR: "
                f"original={','.join(f'{lr:.6g}' for lr in original_surrogate_lrs)}, "
                f"scale={joint_scale:.6g}, "
                f"joint={','.join(f'{lr:.6g}' for lr in joint_lrs)}"
            )
        if model_was_training:
            model.train()
        else:
            model.eval()


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
    case_debug_path = _init_case_debug_csv(args, plot, writer)
    case_debug_counts = {"good": 0, "bad": 0}
    metric_csv_paths = _init_metric_csvs(args, plot, writer)
    checkpoint_gate_refs = {}
    best_trackers = None

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
    _run_surrogate_pretrain(
        model=model,
        args=args,
        loss=loss,
        seq_datasets=seq_datasets,
        loader_kwargs=loader_kwargs,
        metric_csv_paths=metric_csv_paths,
        ckpt_dir=ckpt_dir,
        writer=writer,
        use_cuda=use_cuda,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    post_pretrain_norm = _surrogate_param_norm(loss)
    surrogate_optimizer = getattr(loss, "surrogate_optimizer", None)
    surrogate_lrs = _optimizer_lrs(surrogate_optimizer)
    pretrain_label = (
        "start after surrogate pretrain"
        if int(getattr(args, "surrogate_pretrain_steps", 0)) > 0
        else "start"
    )
    writer.write(
        f"[Training] {pretrain_label} "
        f"surrogate_param_norm={_case_float(post_pretrain_norm, float('nan')):.6f} "
        f"lr={surrogate_lrs[0] if surrogate_lrs else 'NA'}"
    )
    optimizer.zero_grad(set_to_none=True)
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
        episode_checkpoint_sums = _new_checkpoint_metric_sums()
        episode_compression_sums = _new_compression_episode_sums()
        episode_operation_sums = _new_operation_episode_sums()
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
                sparsepcgc_csv_debug = (
                    str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "") == "sparsepcgc"
                    and bool(getattr(args, "save_compression_metric_csv", True))
                )
                operation_csv_debug = bool(
                    getattr(args, "save_operation_metric_csv", getattr(args, "save_operation_metrics_csv", True))
                )
                args._collect_sparsepcgc_debug = bool(log_this_step or profile_this_step or sparsepcgc_csv_debug)
                args._collect_structure_debug = bool(
                    log_this_step
                    or profile_this_step
                    or operation_csv_debug
                    or _sparsepcgc_add_experiment_active(args)
                )
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
                loss_mode = _loss_mode(args)
                compression_primary_mode = loss_mode == "compression_primary"
                stage_factors = _stage_loss_factors(args)
                if compression_primary_mode and not bool(getattr(args, "cp_use_stage_factors", False)):
                    stage_factors = {name: 1.0 for name in stage_factors}
                compute_compression = True if compression_primary_mode else stage_factors["com"] != 0.0
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

                # legacy_totalは既存互換の総合lossとして残す。
                # compression_primaryはactual codec値ではなく、last_compression_termsのgrad tensorを主目的に使う。
                terms = getattr(loss, "last_compression_terms", {}) or {}
                actual_total_bit_backend = _uses_actual_total_bit_objective(args)
                if actual_total_bit_backend:
                    L_com_objective = float(getattr(args, "w_com", 1.0)) * L_com
                else:
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
                legacy_L_downstream = (
                    stage_factors["geom"] * args.w_geom * L_geom
                    + stage_factors["com"] * L_com_objective
                )
                legacy_L_total = (
                    legacy_L_downstream
                    + stage_factors["attr"] * args.w_attr * L_attr
                    + stage_factors["policy"] * args.w_policy * L_policy
                    + stage_factors["repair"] * args.w_actuator * L_actuator
                )
                L = legacy_L_total
                L_downstream = legacy_L_downstream
                L_discrete_policy = L.new_zeros(())
                cp_debug = {}
                if compression_primary_mode:
                    L, L_com_objective, cp_debug = _build_compression_primary_loss(
                        args,
                        terms=terms,
                        L_com=L_com,
                        L_geom=L_geom,
                        L_actuator=L_actuator,
                        global_train_step=global_train_step,
                        stage_factors=stage_factors,
                    )
                    L_downstream = L_com_objective
                    L_discrete_policy = L.new_zeros(())
                elif str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() == "hard":
                    policy_loss_fn = getattr(model, "discrete_policy_loss", None)
                    if callable(policy_loss_fn):
                        L_discrete_policy = policy_loss_fn(L_downstream.detach())
                        L = L + L_discrete_policy

                comp_debug = dict(getattr(loss, "last_compression_debug", {}) or {})
                if cp_debug:
                    comp_debug.update(cp_debug)
                    loss.last_compression_debug = comp_debug
                base_model = model.module if hasattr(model, "module") else model
                structure_debug = getattr(base_model, "last_structure_debug", {}) or {}
                if train_edit_stats is None:
                    train_edit_stats = _summarize_point_edits(
                        input_xyz=input_xyz[:, :3, :],
                        gen_pts=gen_pts,
                        final_w=final_w,
                        args=args,
                    )
                corr_debug = _update_actual_correlation_debug(args, comp_debug, L_com, codec_actual_metric_pairs)
                if corr_debug:
                    comp_debug.update(corr_debug)
                    loss.last_compression_debug = comp_debug
                    corr_value = finite_float_or_none(corr_debug.get("corr_surrogate_actual"))
                    if (
                        log_this_step
                        and bool(getattr(args, "surrogate_realign_on_low_corr", False))
                        and corr_value is not None
                        and corr_value < float(getattr(args, "surrogate_realign_min_corr", 0.3))
                    ):
                        writer.write(
                            "SurrogateRealignNotice: "
                            f"corr_surrogate_actual={corr_value:.6f} below "
                            f"{float(getattr(args, 'surrogate_realign_min_corr', 0.3)):.6f}; "
                            f"realign_steps={int(getattr(args, 'surrogate_realign_steps', 0))} "
                            "(current implementation logs the trigger; extra realign steps are not run unless added later)."
                        )
                compression_metric_row = _build_compression_metric_row(
                    args,
                    global_step=global_train_step,
                    episode=episode,
                    epoch=epoch,
                    step=step,
                    stage=current_stage,
                    comp_debug=comp_debug,
                    L_com=L_com,
                )
                operation_metric_row = _build_operation_metric_row(
                    args,
                    global_step=global_train_step,
                    episode=episode,
                    epoch=epoch,
                    step=step,
                    stage=current_stage,
                    comp_debug=comp_debug,
                    structure_debug=structure_debug,
                    edit_stats=train_edit_stats,
                )
                _append_csv_row(
                    metric_csv_paths.get("compression_step"),
                    COMPRESSION_METRIC_COLUMNS,
                    compression_metric_row,
                )
                _accumulate_compression_episode(episode_compression_sums, compression_metric_row)
                _append_csv_row(
                    metric_csv_paths.get("operation_step"),
                    OPERATION_METRIC_COLUMNS,
                    operation_metric_row,
                )
                _accumulate_operation_episode(episode_operation_sums, operation_metric_row)
                _maybe_record_case_debug(
                    args,
                    writer,
                    case_debug_path,
                    case_debug_counts,
                    global_step=global_train_step,
                    episode=episode,
                    epoch=epoch,
                    step=step,
                    file_path=file_path,
                    comp_debug=comp_debug,
                    structure_debug=structure_debug,
                    edit_stats=train_edit_stats,
                    L=L,
                    L_geom=L_geom,
                    L_com=L_com,
                    L_actuator=L_actuator,
                )

                if log_this_step:
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
                    if cp_debug and bool(getattr(args, "cp_log_grad_terms", True)):
                        _log_compression_primary_terms(writer, step, num_steps, cp_debug)

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
                _accumulate_checkpoint_metrics(
                    episode_checkpoint_sums,
                    compression_metric_row,
                    operation_metric_row,
                    step_metric_values,
                )
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
                max_train_steps = int(getattr(args, "max_train_steps", 0))
                if max_train_steps > 0 and global_train_step >= max_train_steps:
                    writer.write(f"MaxTrainSteps reached: {global_train_step}/{max_train_steps}; stopping debug run.")
                    writer.flush()
                    return

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
        checkpoint_metrics = _finalize_checkpoint_metrics(
            args,
            current_stage,
            episode,
            plot,
            episode_checkpoint_sums,
            checkpoint_gate_refs,
        )
        _append_csv_row(
            metric_csv_paths.get("checkpoint_episode"),
            CHECKPOINT_METRIC_COLUMNS,
            checkpoint_metrics,
        )
        compression_episode_metrics = _finalize_compression_episode_metrics(
            episode,
            current_stage,
            episode_compression_sums,
        )
        _append_csv_row(
            metric_csv_paths.get("compression_episode"),
            COMPRESSION_EPISODE_METRIC_COLUMNS,
            compression_episode_metrics,
        )
        operation_episode_metrics = _finalize_operation_episode_metrics(
            episode,
            current_stage,
            episode_operation_sums,
        )
        _append_csv_row(
            metric_csv_paths.get("operation_episode"),
            OPERATION_EPISODE_METRIC_COLUMNS,
            operation_episode_metrics,
        )

        # 毎エピソードと最高スコアのモデルを保存
        best_loss, model_path, best_trackers = save_episode_checkpoint(
            model=model,
            ckpt_dir=ckpt_dir,
            plot=plot,
            writer=writer,
            episode=episode,
            best_loss=best_loss,
            args=args,
            stage=current_stage,
            checkpoint_metrics=checkpoint_metrics,
            best_trackers=best_trackers,
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
