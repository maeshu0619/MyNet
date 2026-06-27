import torch
import multiprocessing as mp
import torch.optim as optim
import argparse
import hashlib
import math
from collections import OrderedDict

import time
import datetime
from contextlib import nullcontext
from models.utils.pointcloud.utils_repkpu import *
from models.utils.data.dataset import *
from models.utils.patching.patch import (
    build_patch_info,
    denormalize_patch_output,
    merge_patch_outputs,
    patch_info_to_cpu,
    patch_info_to_device,
)
from models.utils.training.utils_grad import *
from models.network import Network
from models.utils.loss.loss import Loss
from models.utils.notify.mail_notify import TrainingMailNotifier
from models.utils.compression.octree_stats import hard_octree_occupancy_stats



def format_named_float_map(values, max_items=None):
    if not values:
        return "n/a"
    items = list(values.items())
    if max_items is not None:
        items = items[:max_items]
    return ", ".join(f"{key}={float(value):.4f}" for key, value in items)


def uses_actual_total_bit_objective(args):
    backend_name = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    return backend_name in {
        "octattention_actual",
        "octattention_actual_ste",
        "octattention_surrogate",
        "sparsepcgc_actual",
        "sparsepcgc_actual_ste",
        "sparsepcgc_surrogate",
        "gpcc_actual",
        "gpcc_actual_ste",
        "gpcc_surrogate",
        "draco_actual",
        "draco_actual_ste",
        "draco_surrogate",
    }


def write_structure_decision_debug(writer, prefix, structure_debug):
    if not structure_debug:
        return

    cause_mean = structure_debug.get("subtree_cause_mean") or structure_debug.get("cause_mean") or {}
    policy_mean = structure_debug.get("policy_mean") or {}

    top_cause = max(cause_mean, key=cause_mean.get) if cause_mean else "n/a"
    top_policy = max(policy_mean, key=policy_mean.get) if policy_mean else "n/a"

    writer.write(
        f"{prefix}: "
        f"top_cause={top_cause}, top_policy={top_policy}, "
        f"policy_entropy={float(structure_debug.get('policy_entropy', 0.0)):.6f}, "
        f"policy_diversity={int(structure_debug.get('policy_diversity', 0))}, "
        f"add_ratio={float(structure_debug.get('add_ratio', 0.0)):.6f}, "
        f"cause_mean=[{format_named_float_map(cause_mean)}], "
        f"policy_mean=[{format_named_float_map(policy_mean)}]"
    )

    operation_by_cause = structure_debug.get("operation_by_cause") or {}
    if operation_by_cause:
        parts = []
        for cause_name, info in operation_by_cause.items():
            parts.append(
                f"{cause_name}->{info.get('operation', 'none')}"
                f"(count={int(info.get('count', 0))}, conf={float(info.get('confidence', 0.0)):.4f})"
            )
        writer.write(f"{prefix}ByCause: " + "; ".join(parts))

    level_debug = structure_debug.get("octree_level_debug") or []
    if level_debug:
        parts = []
        for item in level_debug:
            parts.append(
                f"L{int(item.get('level', 0))}:"
                f"occ={float(item.get('occupied_mean', 0.0)):.1f},"
                f"single_ratio={float(item.get('single_ratio_mean', 0.0)):.4f},"
                f"children={float(item.get('mean_children_mean', 0.0)):.3f}"
            )
        writer.write(f"{prefix}OctreeLevels: " + "; ".join(parts))


def compression_stat_qs(args):
    codec = str(getattr(args, "compress", "OctAttention")).strip().lower().replace("_", "").replace("-", "")

    if codec == "sparsepcgc":
        return max(float(getattr(args, "sparsepcgc_voxel_size", 1.0)), 1e-9)

    if codec == "gpcc":
        return max(float(getattr(args, "gpcc_effective_qs", getattr(args, "qs", 1.0))), 1e-9)

    return max(float(getattr(args, "qs", 1.0)), 1e-9)


def format_triplet(values):
    if not values:
        return "0/0.0/0"

    mean = sum(values) / float(max(len(values), 1))
    return f"{min(values):.0f}/{mean:.1f}/{max(values):.0f}"


def summarize_subtree_octree_stats(input_xyz, groups, args):
    limit = max(int(getattr(args, "train_subtree_stat_log_limit", 16)), 0)
    if limit <= 0 or not groups:
        return None

    qs = compression_stat_qs(args)

    nodes = []
    singles = []
    depths = []

    for _subtree_key, point_idx in groups[:limit]:
        pts = input_xyz[0, :3, :].index_select(1, point_idx).contiguous()

        stats = hard_octree_occupancy_stats(
            pts,
            qs=qs,
            max_depth=int(getattr(args, "compression_octree_stat_depth", 0)),
            quant_mode="sparsepcgc"
            if str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "") == "sparsepcgc"
            else "round",
            pos_quantscale=int(getattr(args, "sparsepcgc_pos_quantscale", 1)),
        )

        nodes.append(float(stats["node_count"]))
        singles.append(float(stats["single_child_count"]))
        depths.append(float(stats["max_depth"]))

    return {
        "count": len(nodes),
        "node": format_triplet(nodes),
        "single": format_triplet(singles),
        "depth": format_triplet(depths),
    }

def should_log_step(step_idx, total_count, rate):
    rate = int(rate)
    if total_count <= 0:
        return False
    if step_idx <= 1 or step_idx >= total_count:
        return True
    if rate <= 0:
        return False
    return step_idx % rate == 0

def new_metric_sums(device, num_metrics):
    return {
        "sums": [torch.zeros((), device=device, dtype=torch.float32) for _ in range(num_metrics)],
        "counts": [0 for _ in range(num_metrics)],
    }

def metric_tensor(value, device):
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=torch.float32)
        if tensor.numel() != 1:
            tensor = tensor.mean()
        if not bool(torch.isfinite(tensor).all().item()):
            return None
        return tensor.reshape(())
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return torch.tensor(scalar, device=device, dtype=torch.float32)


def add_metric_sums(metric_sums, values, device):
    for idx, value in enumerate(values):
        tensor = metric_tensor(value, device)
        if tensor is None:
            continue
        metric_sums["sums"][idx] = metric_sums["sums"][idx] + tensor
        metric_sums["counts"][idx] += 1


def metric_avgs_to_floats(metric_sums, count=None):
    avgs = []
    for value, valid_count in zip(metric_sums["sums"], metric_sums["counts"]):
        if valid_count <= 0:
            avgs.append(None)
            continue
        avgs.append(float((value / float(valid_count)).detach().cpu()))
    return avgs


def surrogate_plot_metrics(loss_obj):
    comp_debug = getattr(loss_obj, "last_compression_debug", {}) or {}
    return [
        float(comp_debug.get("surrogate_train_loss", 0.0)),
        float(comp_debug.get("surrogate_abs_bit_error", 0.0)),
        float(comp_debug.get("surrogate_abs_mean_error", 0.0)),
    ]


def actual_compression_plot_metric(loss_obj, device):
    comp_debug = getattr(loss_obj, "last_compression_debug", {}) or {} # 直近Stepの圧縮debug辞書を取り出す
    if "surrogate_teacher_is_actual" in comp_debug and not bool(comp_debug.get("surrogate_teacher_is_actual", False)): # local_proxyなど実codecでない教師は除外する
        return None # 実圧縮ではない値をactual_compressionグラフへ混ぜない
    actual_value = comp_debug.get(
        "compression_loss_L_com",
        comp_debug.get(
            "actual_train_objective_percent",
            comp_debug.get("actual_total_bit_percent", None),
        ),
    ) # L_comに使った実codec objectiveを取り出す
    if actual_value is None:
        return None # 実圧縮値が無いStepはplot集計から除外する
    return metric_tensor(actual_value, device) # plot/CSVに渡せるscalar tensorへ正規化する


def policy_actual_compression_plot_metric(loss_obj, device):
    comp_debug = getattr(loss_obj, "last_compression_debug", {}) or {}
    actual_value = comp_debug.get("policy_actual_percent", comp_debug.get("policy_final_full_cloud_actual_bit_percent", None))
    if actual_value is None:
        return None
    return metric_tensor(actual_value, device)


def oracle_teacher_compression_plot_metric(loss_obj, device):
    comp_debug = getattr(loss_obj, "last_compression_debug", {}) or {}
    actual_value = comp_debug.get("oracle_teacher_actual_percent", comp_debug.get("oracle_full_cloud_actual_bit_percent", None))
    if actual_value is None:
        return None
    return metric_tensor(actual_value, device)


def actual_compression_ratio_plot_metric(loss_obj, device):
    comp_debug = getattr(loss_obj, "last_compression_debug", {}) or {}
    if "surrogate_teacher_is_actual" in comp_debug and not bool(comp_debug.get("surrogate_teacher_is_actual", False)):
        return None

    gt_bits = comp_debug.get("gt_actual_bit", comp_debug.get("gt_bit_abs", None))
    gen_bits = comp_debug.get(
        "gen_total_bit_with_edit_record",
        comp_debug.get("gen_actual_bit", comp_debug.get("actual_total_bits", None)),
    )
    try:
        gt_bits = float(gt_bits)
        gen_bits = float(gen_bits)
    except (TypeError, ValueError):
        gt_bits = float("nan")
        gen_bits = float("nan")

    if math.isfinite(gt_bits) and gt_bits > 0.0 and math.isfinite(gen_bits):
        return metric_tensor(100.0 * gen_bits / gt_bits, device)

    actual_value = comp_debug.get("actual_total_bit_percent", None)
    if actual_value is None:
        return None
    try:
        actual_value = float(actual_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(actual_value):
        return None
    return metric_tensor(100.0 + actual_value, device)


def surrogate_compression_plot_metric(loss_obj, fallback_value, device):
    comp_debug = getattr(loss_obj, "last_compression_debug", {}) or {} # 直近Stepの圧縮debug辞書を取り出す
    surrogate_value = comp_debug.get("surrogate_pred_bit", comp_debug.get("rate_proxy_delta", None)) # Surrogateが予測した(Mine-GT)*100/GTを取り出す
    if surrogate_value is None:
        surrogate_value = fallback_value # Surrogate値が無いbackendでは従来のL_com表示へ戻す
    return metric_tensor(surrogate_value, device) # plot/CSVに渡せるscalar tensorへ正規化する


def format_metric_summary(prefix, metric_keys, values):
    parts = []
    for key, value in zip(metric_keys, values):
        try:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError
            parts.append(f"{key}={numeric:.6g}")
        except (TypeError, ValueError):
            parts.append(f"{key}=n/a")
    return f"{prefix}: " + ", ".join(parts)


def point_ratio_percent(numerator, denominator):
    denom = max(int(denominator), 1)
    return 100.0 * float(numerator) / float(denom)


def new_point_edit_sums():
    return {
        "input_points": 0,
        "pre_output_points": 0,
        "output_points": 0,
        "added_points": 0,
        "deleted_points": 0,
        "adjusted_points": 0,
        "net_change": 0,
        "adjust_mean_sum": 0.0,
        "adjust_max": 0.0,
        "voxel_edit_input_count": 0,
        "voxel_edit_add_count": 0,
        "voxel_edit_drop_count": 0,
        "voxel_edit_move_count": 0,
        "voxel_edit_final_count": 0,
        "full_cloud_voxel_count": 0,
        "count": 0,
    }


def add_point_edit_sums(edit_sums, edit_stats):
    if edit_stats is None:
        return edit_sums
    if edit_sums is None:
        edit_sums = new_point_edit_sums()
    for key in (
        "input_points",
        "pre_output_points",
        "output_points",
        "added_points",
        "deleted_points",
        "adjusted_points",
        "net_change",
        "voxel_edit_input_count",
        "voxel_edit_add_count",
        "voxel_edit_drop_count",
        "voxel_edit_move_count",
        "voxel_edit_final_count",
    ):
        edit_sums[key] += int(edit_stats.get(key, 0))
    edit_sums["full_cloud_voxel_count"] = max(
        int(edit_sums.get("full_cloud_voxel_count", 0)),
        int(edit_stats.get("full_cloud_voxel_count", 0) or 0),
    )
    edit_sums["adjust_mean_sum"] += float(edit_stats.get("adjust_mean", 0.0))
    edit_sums["adjust_max"] = max(float(edit_sums["adjust_max"]), float(edit_stats.get("adjust_max", 0.0)))
    edit_sums["count"] += 1
    return edit_sums


def finalize_point_edit_sums(edit_sums):
    if edit_sums is None:
        return None
    count = max(int(edit_sums.get("count", 0)), 1)
    finalized = dict(edit_sums)
    finalized["input_points_avg"] = float(edit_sums.get("input_points", 0)) / float(count)
    finalized["pre_output_points_avg"] = float(edit_sums.get("pre_output_points", 0)) / float(count)
    finalized["output_points_avg"] = float(edit_sums.get("output_points", 0)) / float(count)
    finalized["added_points_avg"] = float(edit_sums.get("added_points", 0)) / float(count)
    finalized["deleted_points_avg"] = float(edit_sums.get("deleted_points", 0)) / float(count)
    finalized["adjusted_points_avg"] = float(edit_sums.get("adjusted_points", 0)) / float(count)
    finalized["net_change_avg"] = float(edit_sums.get("net_change", 0)) / float(count)
    finalized["adjust_mean"] = float(edit_sums.get("adjust_mean_sum", 0.0)) / float(count)
    finalized["added_ratio_percent"] = point_ratio_percent(
        finalized.get("added_points", 0),
        finalized.get("input_points", 0),
    )
    finalized["deleted_ratio_percent"] = point_ratio_percent(
        finalized.get("deleted_points", 0),
        finalized.get("input_points", 0),
    )
    finalized["adjusted_ratio_percent"] = point_ratio_percent(
        finalized.get("adjusted_points", 0),
        finalized.get("input_points", 0),
    )
    voxel_input_count = int(finalized.get("voxel_edit_input_count", 0) or 0)
    voxel_drop_count = int(finalized.get("voxel_edit_drop_count", 0) or 0)
    full_cloud_voxel_count = int(finalized.get("full_cloud_voxel_count", 0) or 0)
    full_cloud_denominator = full_cloud_voxel_count if full_cloud_voxel_count > 0 else voxel_input_count
    finalized["voxel_add_ratio_percent"] = point_ratio_percent(
        finalized.get("voxel_edit_add_count", 0),
        voxel_input_count,
    )
    finalized["voxel_drop_ratio_percent"] = point_ratio_percent(
        voxel_drop_count,
        voxel_input_count,
    )
    finalized["voxel_move_ratio_percent"] = point_ratio_percent(
        finalized.get("voxel_edit_move_count", 0),
        voxel_input_count,
    )
    finalized["full_cloud_voxel_drop_ratio_percent"] = point_ratio_percent(
        voxel_drop_count,
        full_cloud_denominator,
    )
    return finalized


def aligned_edit_ref_xyz(input_xyz, output_points):
    ref_xyz = input_xyz[:, :3, :]
    output_points = int(output_points)
    if ref_xyz.shape[-1] == output_points:
        return ref_xyz.contiguous()
    if ref_xyz.shape[-1] > output_points:
        return ref_xyz[:, :, :output_points].contiguous()
    pad = ref_xyz.new_full((ref_xyz.shape[0], 3, output_points - ref_xyz.shape[-1]), float("nan"))
    return torch.cat([ref_xyz, pad], dim=2).contiguous()


def compute_edit_keep_mask(final_w, args):
    if final_w is None:
        return None, None
    flat_w = final_w.detach().reshape(-1)

    # final_w に NaN / Inf が混じると int(NaN) で落ちるため、集計前に安全化する
    flat_w = torch.nan_to_num(flat_w, nan=0.0, posinf=1.0, neginf=0.0)
    flat_w = flat_w.clamp(0.0, 1.0)
    total_count = int(flat_w.numel())
    drop_threshold = float(
        getattr(args, "operation_count_drop_threshold", getattr(args, "test_drop_threshold", 0.5))
    )
    keep_mask = flat_w >= drop_threshold
    keep_count = int(keep_mask.sum().item())
    mode = "threshold"
    if total_count > 0 and (keep_count <= 0 or keep_count >= total_count):
        expected_keep_value = float(flat_w.sum().item())

        # 念のため、ここでも有限値チェックを行う
        if not math.isfinite(expected_keep_value):
            expected_keep_value = float(keep_count)

        expected_keep = int(round(expected_keep_value))
        expected_keep = min(max(expected_keep, 1), total_count)
        if 0 < expected_keep < total_count:
            topk_idx = torch.topk(flat_w, k=expected_keep, largest=True, sorted=False).indices
            keep_mask = torch.zeros_like(flat_w, dtype=torch.bool)
            keep_mask.scatter_(0, topk_idx, True)
            keep_count = expected_keep
            mode = "expected_keep"
    return keep_mask, {
        "mode": mode,
        "threshold": drop_threshold,
        "keep_count": keep_count,
        "total_count": total_count,
    }


def summarize_point_edits(input_xyz, gen_pts, final_w=None, args=None, edit_ref_xyz=None):
    if gen_pts is None:
        return None
    input_points = int(input_xyz.shape[-1])
    pre_output_points = int(gen_pts.shape[-1])
    added_points = max(pre_output_points - input_points, 0)
    keep_mask, keep_info = compute_edit_keep_mask(final_w, args) if args is not None else (None, None)
    if keep_info is not None and keep_info.get("keep_count") is not None:
        output_points = int(keep_info["keep_count"])
        deleted_points = max(pre_output_points - output_points, 0)
    else:
        output_points = pre_output_points
        deleted_points = max(input_points + added_points - output_points, 0)

    if edit_ref_xyz is None:
        edit_ref_xyz = aligned_edit_ref_xyz(input_xyz, pre_output_points)
    compare_points = min(int(gen_pts.shape[-1]), int(edit_ref_xyz.shape[-1]))
    adjusted_points = 0
    max_adjust = 0.0
    mean_adjust = 0.0
    threshold = max(
        float(getattr(args, "operation_count_adjust_threshold", getattr(args, "test_adjust_threshold", 1e-6)))
        if args is not None
        else 1e-6,
        0.0,
    )
    if compare_points > 0:
        gen_xyz = gen_pts[:, :3, :compare_points].detach()
        ref_xyz = edit_ref_xyz[:, :3, :compare_points].to(device=gen_xyz.device, dtype=gen_xyz.dtype)
        delta_norm = torch.linalg.norm(gen_xyz - ref_xyz, dim=1)
        finite_mask = torch.isfinite(delta_norm)
        moved_mask = finite_mask & (delta_norm > threshold)
        if keep_mask is not None and int(keep_mask.numel()) == int(delta_norm.numel()):
            moved_mask = moved_mask.reshape(-1) & keep_mask.reshape(-1)[:moved_mask.numel()]
            moved_mask = moved_mask.reshape(delta_norm.shape)
        valid_delta = delta_norm[finite_mask]
        if valid_delta.numel() > 0:
            max_adjust = float(valid_delta.max().detach().cpu())
            mean_adjust = float(valid_delta.mean().detach().cpu())
        adjusted_points = int(moved_mask.sum().item())

    voxel_state = getattr(args, "_last_actuator_voxel_state", None) if args is not None else None
    voxel_input_count = 0
    voxel_add_count = 0
    voxel_drop_count = 0
    voxel_move_count = 0
    voxel_final_count = 0
    voxel_mode = False
    full_cloud_voxel_count = 0
    if args is not None:
        for attr_name in (
            "_full_cloud_canonical_coords_count",
            "_full_cloud_input_voxel_count",
            "_full_cloud_voxel_count",
        ):
            try:
                attr_value = getattr(args, attr_name, None)
                if attr_value is not None:
                    full_cloud_voxel_count = int(float(attr_value))
                    break
            except Exception:
                full_cloud_voxel_count = 0
    if isinstance(voxel_state, dict):
        def _state_int(*keys):
            for key in keys:
                value = voxel_state.get(key, None)
                if value is not None:
                    try:
                        return int(float(value))
                    except Exception:
                        return 0
            return 0

        voxel_mode = bool(voxel_state.get("voxel_edit_state_enabled", False))
        voxel_input_count = _state_int("input_voxel_count", "before_occupied_voxel_count", "voxel_edit_initial_count")
        voxel_add_count = _state_int("voxel_edit_add_count", "add_target_voxel_count")
        voxel_drop_count = _state_int("voxel_edit_drop_count", "delete_target_voxel_count")
        voxel_move_count = _state_int("voxel_edit_move_count", "move_source_voxel_count")
        voxel_final_count = _state_int("final_voxel_count", "after_occupied_voxel_count", "voxel_edit_final_count")
        if full_cloud_voxel_count <= 0:
            full_cloud_voxel_count = _state_int(
                "full_cloud_voxel_count",
                "full_input_voxel_count",
                "input_voxel_count",
                "before_occupied_voxel_count",
            )

    point_added_debug = int(added_points)
    point_deleted_debug = int(deleted_points)
    point_adjusted_debug = int(adjusted_points)
    ratio_denominator = input_points
    metric_input_points = input_points
    if voxel_mode and voxel_input_count > 0:
        metric_input_points = int(voxel_input_count)
        added_points = int(voxel_add_count)
        deleted_points = int(voxel_drop_count)
        adjusted_points = int(voxel_move_count)
        output_points = int(voxel_final_count)
        pre_output_points = int(voxel_final_count)
        ratio_denominator = metric_input_points

    return {
        "input_points": int(metric_input_points),
        "pre_output_points": int(pre_output_points),
        "output_points": int(output_points),
        "added_points": int(added_points),
        "deleted_points": int(deleted_points),
        "adjusted_points": int(adjusted_points),
        "input_points_avg": float(metric_input_points),
        "pre_output_points_avg": float(pre_output_points),
        "output_points_avg": float(output_points),
        "added_points_avg": float(added_points),
        "deleted_points_avg": float(deleted_points),
        "adjusted_points_avg": float(adjusted_points),
        "added_ratio_percent": point_ratio_percent(added_points, ratio_denominator),
        "deleted_ratio_percent": point_ratio_percent(deleted_points, ratio_denominator),
        "adjusted_ratio_percent": point_ratio_percent(adjusted_points, ratio_denominator),
        "added_ratio_percent_point_debug": point_ratio_percent(point_added_debug, input_points),
        "deleted_ratio_percent_point_debug": point_ratio_percent(point_deleted_debug, input_points),
        "adjusted_ratio_percent_point_debug": point_ratio_percent(point_adjusted_debug, input_points),
        "input_points_point_debug": int(input_points),
        "voxel_edit_input_count": int(voxel_input_count),
        "voxel_edit_add_count": int(voxel_add_count),
        "voxel_edit_drop_count": int(voxel_drop_count),
        "voxel_edit_move_count": int(voxel_move_count),
        "voxel_edit_final_count": int(voxel_final_count),
        "voxel_add_ratio_percent": point_ratio_percent(voxel_add_count, voxel_input_count),
        "voxel_drop_ratio_percent": point_ratio_percent(voxel_drop_count, voxel_input_count),
        "voxel_move_ratio_percent": point_ratio_percent(voxel_move_count, voxel_input_count),
        "full_cloud_voxel_count": int(full_cloud_voxel_count),
        "full_cloud_voxel_drop_ratio_percent": point_ratio_percent(
            voxel_drop_count,
            full_cloud_voxel_count if full_cloud_voxel_count > 0 else voxel_input_count,
        ),
        "voxel_operation_mode": bool(voxel_mode),
        "adjust_threshold": float(threshold),
        "adjust_mean": float(mean_adjust),
        "adjust_max": float(max_adjust),
        "net_change": int(output_points - metric_input_points),
        "net_change_avg": float(output_points - metric_input_points),
        "keep_mode": None if keep_info is None else keep_info.get("mode"),
    }


def make_step_cache_key(file_path, args):
    return (
        f"{file_path}"
        f"|max_input_points={int(getattr(args, 'max_input_points', 0))}"
        f"|safe_max_input_points={int(getattr(args, 'safe_max_input_points', 0))}"
        f"|allow_unbounded_input={bool(getattr(args, 'allow_unbounded_input', False))}"
        f"|sampling={getattr(args, 'input_sampling', 'random')}"
        f"|split2patch={bool(getattr(args, 'split2patch', False))}"
        f"|num_points={int(getattr(args, 'num_points', 0))}"
        f"|patch_rate={float(getattr(args, 'patch_rate', 1.0))}"
        f"|patch_build_mode={getattr(args, 'patch_build_mode', 'spatial_sort')}"
        f"|patch_owned_ratio={float(getattr(args, 'patch_owned_ratio', 0.875))}"
        f"|patch_sort_grid_size={int(getattr(args, 'patch_sort_grid_size', 1024))}"
    )

def effective_patch_batch_size(args, patch_count=None, patch_size=None, is_train=True, writer=None):
    patch_count = max(int(patch_count or getattr(args, "patch_batch_size", 1)), 1)
    mode = str(getattr(args, "patch_parallel_mode", "auto")).strip().lower()
    fixed = max(int(getattr(args, "patch_batch_size", 1)), 1)
    if mode == "all":
        return patch_count
    if mode == "fixed":
        return min(fixed, patch_count)

    budget_name = "patch_parallel_points_budget_train" if is_train else "patch_parallel_points_budget_test"
    budget_points = int(getattr(args, budget_name, 0))
    patch_size = max(int(patch_size or getattr(args, "num_points", fixed)), 1)
    if budget_points <= 0:
        return min(fixed, patch_count)
    auto_batch = max(budget_points // patch_size, 1)
    return min(max(auto_batch, 1), patch_count)


def select_patch_subset_ids(patch_info, global_step, args):
    total_patches = int(patch_info["num_patches"])
    if total_patches <= 0:
        raise ValueError("Patch subset selection received zero patches.")

    ref_tensor = patch_info.get("patch_input_idx")
    device = ref_tensor.device if torch.is_tensor(ref_tensor) else "cpu"
    if not bool(getattr(args, "train_patch_subset_enable", False)):
        return torch.arange(total_patches, device=device, dtype=torch.long)

    sampling = str(getattr(args, "train_patch_subset_sampling", "coverage_cycle")).strip().lower()
    if sampling != "coverage_cycle":
        raise ValueError(f"Unsupported train patch subset sampling mode: {sampling}")

    patches_per_step = int(getattr(args, "train_patch_subset_patches_per_step", total_patches))
    if patches_per_step < 1:
        raise ValueError("train_patch_subset_patches_per_step must be >= 1")
    if patches_per_step >= total_patches:
        return torch.arange(total_patches, device=device, dtype=torch.long)

    stride = max(int(math.ceil(total_patches / float(patches_per_step))), 1)
    offset = int(global_step) % stride
    selected = []
    seen = set()
    for class_shift in range(stride):
        start = (offset + class_shift) % stride
        for patch_id in range(start, total_patches, stride):
            if patch_id in seen:
                continue
            selected.append(patch_id)
            seen.add(patch_id)
            if len(selected) >= patches_per_step:
                break
        if len(selected) >= patches_per_step:
            break

    if len(selected) < patches_per_step:
        for patch_id in range(total_patches):
            if patch_id in seen:
                continue
            selected.append(patch_id)
            if len(selected) >= patches_per_step:
                break

    return torch.tensor(selected, device=device, dtype=torch.long)


def make_patch_subset_cache_key(cache_key, selected_patch_ids, total_patch_count=None):
    if cache_key is None:
        cache_key = ""
    if selected_patch_ids is None:
        return cache_key

    if torch.is_tensor(selected_patch_ids):
        subset_ids = selected_patch_ids.detach().to(torch.long).cpu().tolist()
    else:
        subset_ids = [int(patch_id) for patch_id in selected_patch_ids]

    if total_patch_count is not None and len(subset_ids) >= int(total_patch_count):
        return cache_key

    subset_text = ",".join(str(patch_id) for patch_id in subset_ids)
    subset_hash = hashlib.sha1(subset_text.encode("utf-8")).hexdigest()[:16]
    return f"{cache_key}|subset={subset_hash}"


def format_bytes(num_bytes):
    num_bytes = float(max(num_bytes, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024.0 or unit == "TB":
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0


def process_rss_bytes():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def log_cache_status(model, writer, prefix):
    stats_fn = getattr(model, "input_cache_stats", None)
    if not callable(stats_fn):
        return
    stats = stats_fn()
    max_bytes = stats.get("max_bytes", 0)
    mem_limit = "unlimited" if max_bytes <= 0 else format_bytes(max_bytes)
    rss = process_rss_bytes()
    rss_text = "" if rss is None else f", rss={format_bytes(rss)}"
    writer.write(
        f"{prefix}: frozen_cache={stats.get('entries', 0)}/{stats.get('max_entries', 0)} "
        f"entries, memory={format_bytes(stats.get('bytes', 0))}/{mem_limit}{rss_text}"
    )


def adapt_encoder_state_dict_for_sparse_input(model, encoder_state, writer=None):
    key = "stem.0.weight"
    model_state = model.encoder.state_dict()
    if key not in encoder_state or key not in model_state:
        return encoder_state

    saved_weight = encoder_state[key]
    target_weight = model_state[key]
    if tuple(saved_weight.shape) == tuple(target_weight.shape):
        return encoder_state

    adapted = dict(encoder_state)
    new_weight = target_weight.clone()
    if int(new_weight.shape[1]) == 1 and int(saved_weight.shape[1]) > 1:
        new_weight[:, :1, :] = saved_weight.mean(dim=1, keepdim=True)
    else:
        copy_channels = min(int(saved_weight.shape[1]), int(new_weight.shape[1]))
        new_weight[:, :copy_channels, :] = saved_weight[:, :copy_channels, :]
    if int(new_weight.shape[1]) > int(saved_weight.shape[1]):
        filler = saved_weight.mean(dim=1, keepdim=True)
        for channel in range(int(saved_weight.shape[1]), int(new_weight.shape[1])):
            new_weight[:, channel:channel + 1, :] = filler
    adapted[key] = new_weight
    if writer is not None and hasattr(writer, "write"):
        writer.write(
            f"RepKPU encoder stem adapted for sparse input: {tuple(saved_weight.shape)} -> {tuple(new_weight.shape)}"
        )
    return adapted


def module_grad_stats(module):
    if module is None:
        return {"missing_module": True, "norm": 0.0, "max": 0.0, "mean": 0.0, "rms": 0.0, "active": 0, "none": 0, "finite": False, "param_count": 0, "status": "module_missing"}
    total_sq = 0.0
    total_abs = 0.0 # 勾配の平均絶対値を出すために絶対値合計を保持する
    total_numel = 0 # 勾配の平均/RMSを出すために要素数を数える
    max_abs = 0.0
    active = 0
    missing = 0
    finite = True
    param_count = 0
    for param in module.parameters():
        if not param.requires_grad:
            continue
        param_count += 1
        grad = param.grad
        if grad is None:
            missing += 1
            continue
        grad_det = grad.detach()
        finite = finite and bool(torch.isfinite(grad_det).all().item())
        grad_float = grad_det.float()
        grad_abs = grad_float.abs() # 勾配の絶対値統計を計算する
        total_sq += float(grad_float.pow(2).sum().detach().cpu())
        total_abs += float(grad_abs.sum().detach().cpu()) # モジュール全体の平均絶対勾配用に加算する
        total_numel += int(grad_float.numel()) # モジュール全体の勾配要素数を加算する
        max_abs = max(max_abs, float(grad_abs.max().detach().cpu()))
        active += 1
    norm = total_sq ** 0.5
    mean_abs = (total_abs / float(total_numel)) if total_numel > 0 else 0.0 # 平均絶対勾配を計算する
    rms = (total_sq / float(total_numel)) ** 0.5 if total_numel > 0 else 0.0 # RMS勾配を計算する
    if param_count <= 0:
        status = "no_trainable_params"
    elif not finite:
        status = "nonfinite_grad"
    elif active <= 0 and missing > 0:
        status = "all_grad_none"
    elif active <= 0:
        status = "no_grad"
    elif norm <= 0.0:
        status = "zero_grad"
    else:
        status = "ok"
    return {"missing_module": False, "norm": norm, "max": max_abs, "mean": mean_abs, "rms": rms, "active": active, "none": missing, "finite": finite, "param_count": param_count, "status": status}


def module_grad_summary(module):
    stats = module_grad_stats(module)
    status = str(stats.get("status", "unknown"))
    norm = float(stats.get("norm", 0.0))
    max_abs = float(stats.get("max", 0.0))
    mean_abs = float(stats.get("mean", 0.0))
    rms = float(stats.get("rms", 0.0))
    active = int(stats.get("active", 0))
    missing = int(stats.get("none", 0))
    param_count = int(stats.get("param_count", 0))
    finite = bool(stats.get("finite", False))
    return f"status={status}, norm={norm:.3e}, max={max_abs:.3e}, mean={mean_abs:.3e}, rms={rms:.3e}, active={active}, none={missing}, params={param_count}, finite={finite}"


def _first_existing_module(root, *names):
    for name in names:
        module = getattr(root, name, None)
        if module is not None:
            return module
    return None


def named_trainable_child_modules(base_model):
    actuator = _first_existing_module(base_model, "actuator", "disp_module")
    modules = [
        ("encoder", getattr(base_model, "encoder", None)),
        ("cost_attr", _first_existing_module(base_model, "cost_attributor", "prun_module")),
        ("repair_policy", _first_existing_module(base_model, "policy_module", "adding_module")),
        ("actuator", actuator),
    ]
    if actuator is not None:
        modules.extend(
            [
                ("delete_branch", getattr(actuator, "drop_head", None)),
                ("delete_amount", getattr(actuator, "drop_amount_head", None)),
                ("add_branch", getattr(actuator, "add_head", None)),
                ("add_amount", getattr(actuator, "add_amount_head", None)),
                ("add_target_branch", getattr(actuator, "add_voxel_head", None)),
                ("move_branch", getattr(actuator, "move_voxel_head", None)),
                ("move_amount", getattr(actuator, "move_amount_head", None)),
            ]
        )
    return modules


def trainable_parameters(module):
    if module is None:
        return []
    return [param for param in module.parameters() if param.requires_grad]


def first_trainable_parameter(module):
    for param in trainable_parameters(module):
        return param
    return None


def snapshot_module_parameters(module):
    params = trainable_parameters(module)
    if not params:
        return None
    return [param.detach().clone() for param in params]


def capture_param_update_snapshots(args, model, step_idx, total_count):
    if bool(getattr(args, "compact_step_text_log", False)):
        return None
    if not bool(getattr(args, "debug_grad_flow", False)):
        return None
    if not should_log_step(step_idx, total_count, getattr(args, "debug_grad_flow_rate", 1)):
        return None
    base_model = model.module if hasattr(model, "module") else model
    snapshots = {}
    for name, module in named_trainable_child_modules(base_model):
        if name == "encoder" and bool(getattr(args, "encoder_0grad", True)):
            continue
        snapshot = snapshot_module_parameters(module)
        if snapshot is None:
            continue
        # debug時だけ、そのstep内でdetach済みcloneを保持する。
        # ログ後に参照を捨てるため、過去stepのGPU tensorや計算グラフは蓄積しない。
        snapshots[name] = snapshot
    return snapshots


def log_param_updates(args, writer, model, snapshots, step_idx, total_count):
    if not snapshots:
        return
    if bool(getattr(args, "compact_step_text_log", False)):
        return
    if not bool(getattr(args, "debug_grad_flow", False)):
        args._last_grad_flow = {}
        return
    if not should_log_step(step_idx, total_count, getattr(args, "debug_grad_flow_rate", 1)):
        args._last_grad_flow = {}
        return
    base_model = model.module if hasattr(model, "module") else model
    parts = []
    for name, module in named_trainable_child_modules(base_model):
        if name == "encoder" and bool(getattr(args, "encoder_0grad", True)):
            parts.append("encoder(skipped_encoder_0grad)")
            continue
        before = snapshots.get(name)
        params = trainable_parameters(module)
        if before is None or not params:
            parts.append(f"{name}(missing)")
            continue
        if len(before) != len(params):
            parts.append(f"{name}(shape_changed)")
            continue
        total_sq = 0.0
        max_abs = 0.0
        shape_changed = False
        for param, before_param in zip(params, before):
            after = param.detach()
            if tuple(after.shape) != tuple(before_param.shape):
                shape_changed = True
                break
            diff = (after - before_param.to(device=after.device, dtype=after.dtype)).float()
            total_sq += float(diff.pow(2).sum().detach().cpu())
            max_abs = max(max_abs, float(diff.abs().max().detach().cpu()))
        if shape_changed:
            parts.append(f"{name}(shape_changed)")
            continue
        parts.append(
            f"{name}(delta_norm={(total_sq ** 0.5):.3e}, "
            f"delta_max={max_abs:.3e})"
        )
    writer.write("ParamUpdate: " + " | ".join(parts))


def log_grad_flow(args, writer, model, step_idx, total_count, global_step=None):
    compact_step_log = bool(getattr(args, "compact_step_text_log", False))
    if not bool(getattr(args, "debug_grad_flow", False)) and not compact_step_log:
        return
    should_write = (
        should_log_step(step_idx, total_count, getattr(args, "debug_grad_flow_rate", 1))
        and not compact_step_log
    )
    base_model = model.module if hasattr(model, "module") else model
    grad_map = {}
    parts = []
    for name, module in named_trainable_child_modules(base_model):
        stats = module_grad_stats(module)
        grad_map[f"{name}_grad_norm"] = float(stats.get("norm", 0.0))
        grad_map[f"{name}_grad_max"] = float(stats.get("max", 0.0))
        grad_map[f"{name}_grad_mean"] = float(stats.get("mean", 0.0))
        grad_map[f"{name}_grad_rms"] = float(stats.get("rms", 0.0))
        grad_map[f"{name}_grad_status"] = str(stats.get("status", "unknown"))
        grad_map[f"{name}_grad_active_count"] = int(stats.get("active", 0))
        grad_map[f"{name}_grad_none_count"] = int(stats.get("none", 0))
        grad_map[f"{name}_grad_finite"] = bool(stats.get("finite", False))
        parts.append(f"{name}({module_grad_summary(module)})")
    args._last_grad_flow = grad_map
    step_text = int(global_step) + 1 if global_step is not None else step_idx
    if should_write:
        writer.write(f"GradFlow: global_step={step_text} | " + " | ".join(parts))
    threshold = max(float(getattr(args, "operation_dead_grad_warn_threshold", 1e-12)), 0.0)
    patience = max(int(getattr(args, "operation_dead_grad_warn_patience", 20)), 1)
    for key in ("add_branch_grad_norm", "add_amount_grad_norm"):
        norm = float(grad_map.get(key, 0.0))
        streak_key = f"_{key}_low_streak"
        streak = int(getattr(args, streak_key, 0))
        streak = streak + 1 if norm <= threshold else 0
        setattr(args, streak_key, streak)
        if should_write and streak == patience:
            writer.write(f"GradFlowWarning: global_step={step_text}, {key}={norm:.3e} stayed <= {threshold:.3e} for {patience} logged checks.")


def sync_for_timing(use_cuda):
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def use_memory_safe_loader_workers(args, model, writer):
    requested_workers = max(int(args.num_workers), 0)
    if requested_workers <= 0:
        return 0

    requested_mp_method = str(getattr(args, "mp_start_method", "auto")).strip().lower()
    current_mp_method = mp.get_start_method(allow_none=True) or "fork"
    fork_like = requested_mp_method in {"auto", "fork"} and current_mp_method == "fork"
    if not fork_like:
        return requested_workers

    compression_backend = str(getattr(args, "compression_loss_backend", "")).strip().lower()
    actual_teacher_backends = {
        "octattention_surrogate",
        "octattention_actual",
        "octattention_actual_ste",
        "sparsepcgc_surrogate",
        "sparsepcgc_actual",
        "sparsepcgc_actual_ste",
        "gpcc_surrogate",
        "gpcc_actual",
        "gpcc_actual_ste",
    }
    if compression_backend in actual_teacher_backends:
        writer.write(
            "DataLoader workers were disabled because fork workers would duplicate the compression "
            f"teacher/cache state ({requested_workers} requested). Use --mp_start_method spawn to keep workers."
        )
        return 0

    warmup_frozen = bool(getattr(args, "warmup_frozen_cache", True) and getattr(model, "cache_enabled", False))
    if not warmup_frozen:
        return requested_workers

    writer.write(
        "DataLoader workers were disabled because fork would inherit the prewarmed frozen cache "
        f"({requested_workers} requested). Use --mp_start_method spawn to keep workers."
    )
    return 0


def resolve_training_stage_for_episode(args, episode_idx):
    if not bool(getattr(args, "two_stage_training", False)):
        return str(getattr(args, "training_stage", "joint")).strip().lower()
    diagnosis_episodes = int(getattr(args, "diagnosis_episodes", 0))
    if diagnosis_episodes <= 0:
        ratio = float(getattr(args, "diagnosis_episode_ratio", 0.25))
        diagnosis_episodes = max(int(round(float(getattr(args, "episodes", 1)) * ratio)), 1)
    return "diagnosis" if int(episode_idx) < diagnosis_episodes else "joint"


def stage_loss_factors(args):
    stage = str(getattr(args, "training_stage", "joint")).strip().lower()
    if stage == "diagnosis":
        return {
            "geom": float(getattr(args, "diagnosis_geom_factor", 0.0)),
            "com": float(getattr(args, "diagnosis_com_factor", 0.0)),
            "attr": float(getattr(args, "diagnosis_attr_factor", 1.0)),
            "policy": float(getattr(args, "diagnosis_policy_factor", 1.0)),
            "repair": float(getattr(args, "diagnosis_repair_factor", 0.25)),
        }
    return {
        "geom": 1.0,
        "com": 1.0,
        "attr": 1.0,
        "policy": 1.0,
        "repair": 1.0,
    }


def get_patch_info(input_pcd, args, cache_key, patch_info_cache):
    cache_enabled = bool(getattr(args, "patch_info_cache", True))
    cache_max_entries = max(int(getattr(args, "cache_max_entries", 64)), 0)
    if cache_enabled and cache_key:
        cached = patch_info_cache.get(cache_key)
        if cached is not None:
            patch_info_cache.move_to_end(cache_key)
            return patch_info_to_device(cached, device=input_pcd.device, dtype=input_pcd.dtype)

    patch_info = build_patch_info(input_pcd, args)

    if cache_enabled and cache_key and cache_max_entries > 0:
        patch_info_cache[cache_key] = patch_info_to_cpu(patch_info)
        patch_info_cache.move_to_end(cache_key)
        while len(patch_info_cache) > cache_max_entries:
            patch_info_cache.popitem(last=False)

    return patch_info


def accumulate_grouped_patch_geometry(
    geom_groups,
    loss,
    args,
):
    if not geom_groups:
        return None, 0.0

    ref_tensor = next(iter(geom_groups.values()))["gen"][0]
    total = ref_tensor.new_zeros(())
    total_weight = 0.0
    for group in geom_groups.values():
        gen_batch = torch.cat(group["gen"], dim=0)
        gt_batch = torch.cat(group["gt"], dim=0)
        final_w_batch = None
        if group["final_w"] is not None:
            final_w_batch = torch.cat(group["final_w"], dim=0)
        group_loss = loss.get_geometry_loss(
            args,
            gen_pts=gen_batch,
            gt_pts=gt_batch,
            final_w=final_w_batch,
            out_label=None,
        )
        weight = float(group["weight"])
        total = total + group_loss * weight
        total_weight += weight
    return total, total_weight


def stable_index_subset(num_points, max_points, method, key, seed):
    if max_points <= 0 or num_points <= max_points:
        return None

    method = str(method).strip().lower()
    if method == "head":
        return torch.arange(max_points, dtype=torch.long)
    if method == "stride":
        step = max(num_points // max_points, 1)
        return torch.arange(0, num_points, step, dtype=torch.long)[:max_points]
    if method != "random":
        raise ValueError(f"Unsupported input_sampling mode: {method}")

    seed_text = f"{key}|{int(seed)}"
    seed_value = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_value)
    return torch.randperm(num_points, generator=generator)[:max_points]


def downsample_input_batch(input_pcd, args, cache_key):
    max_points = int(getattr(args, "max_input_points", 0))
    if max_points <= 0 and not bool(getattr(args, "allow_unbounded_input", False)):
        max_points = int(getattr(args, "safe_max_input_points", 0))
    if max_points <= 0:
        return input_pcd
    if input_pcd.dim() != 3:
        raise ValueError(f"Expected input_pcd to have shape [B, N, C], got {tuple(input_pcd.shape)}")

    num_points = input_pcd.shape[1]
    idx = stable_index_subset(
        num_points=num_points,
        max_points=max_points,
        method=getattr(args, "input_sampling", "random"),
        key=cache_key,
        seed=args.seed,
    )
    if idx is None:
        return input_pcd
    return input_pcd.index_select(1, idx)


def sample_geometry_audit_tensors(gen_pts, gt_pts, final_w, out_label, args, cache_key):
    max_points = max(int(getattr(args, "geometry_audit_max_points", 0)), 0)
    if max_points <= 0:
        return gen_pts, gt_pts, final_w, out_label

    def _sample_pts(tensor, key_suffix):
        idx = stable_index_subset(
            num_points=tensor.shape[-1],
            max_points=max_points,
            method=getattr(args, "input_sampling", "random"),
            key=f"{cache_key}|{key_suffix}",
            seed=args.seed,
        )
        if idx is None:
            return tensor, None
        idx = idx.to(device=tensor.device, dtype=torch.long)
        return tensor.index_select(2, idx), idx

    gen_pts_audit, gen_idx = _sample_pts(gen_pts, "geom_audit_gen")
    gt_pts_audit, gt_idx = _sample_pts(gt_pts, "geom_audit_gt")
    final_w_audit = None
    out_label_audit = out_label
    if final_w is not None and gen_idx is not None:
        final_w_audit = final_w.index_select(2, gen_idx)
    elif final_w is not None:
        final_w_audit = final_w
    if out_label is not None and gt_idx is not None:
        dim = 2 if out_label.dim() == 3 else 1
        out_label_audit = out_label.index_select(dim, gt_idx)
    return gen_pts_audit, gt_pts_audit, final_w_audit, out_label_audit


def run_geometry_audit(loss_obj, args, gen_pts, gt_pts, final_w, out_label, cache_key):
    if gen_pts.dim() == 3 and gen_pts.shape[1] > 3:
        gen_pts = gen_pts[:, :3, :]
    if gt_pts.dim() == 3 and gt_pts.shape[1] > 3:
        gt_pts = gt_pts[:, :3, :]
    gen_a, gt_a, w_a, label_a = sample_geometry_audit_tensors(
        gen_pts,
        gt_pts,
        final_w,
        out_label,
        args,
        cache_key,
    )
    with torch.no_grad():
        geom_value = loss_obj.get_geometry_loss(
            args,
            gen_pts=gen_a,
            gt_pts=gt_a,
            final_w=w_a,
            out_label=label_a,
        )
    geom_debug = dict(getattr(loss_obj, "last_geometry_debug", {}) or {})
    geom_debug["audit_value"] = float(geom_value.detach().cpu())
    geom_debug["audit_gen_points"] = int(gen_a.shape[-1])
    geom_debug["audit_gt_points"] = int(gt_a.shape[-1])
    return geom_debug


def move_xyz_to_device(pts, use_cuda):
    input_xyz = pts[..., :3]
    if use_cuda:
        input_xyz = input_xyz.cuda(non_blocking=True)
    return rearrange(input_xyz, 'b n c -> b c n').contiguous()


def prepare_whole_cloud_inputs(pts, args, cache_key, use_cuda):
    input_xyz = pts if pts.dim() == 3 else pts.unsqueeze(0)
    input_xyz = downsample_input_batch(input_xyz, args, cache_key)
    input_xyz = move_xyz_to_device(input_xyz, use_cuda)
    patches, centroid, furthest_distance = normalize_point_cloud(input_xyz)
    return input_xyz, patches, centroid[:, :3, :], furthest_distance


def warmup_whole_cloud_caches(model, args, loss, seq_datasets, writer, use_cuda, use_amp, amp_dtype):
    if args.split2patch:
        return

    total_files = sum(len(dataset) for _, dataset in seq_datasets)
    if total_files <= 0:
        return

    if bool(getattr(args, "warmup_gt_cache", True)) and getattr(loss, "gt_cache_enabled", False):
        loss.gt_cache_max_entries = max(int(getattr(loss, "gt_cache_max_entries", 0)), total_files)

    warmup_frozen = bool(getattr(args, "warmup_frozen_cache", True) and getattr(model, "cache_enabled", False))
    warmup_gt = bool(getattr(args, "warmup_gt_cache", True) and getattr(loss, "gt_cache_enabled", False))
    if not warmup_frozen and not warmup_gt:
        return

    warmup_max_files = max(int(getattr(args, "warmup_max_files", 0)), 0)
    warmup_max_seconds = max(float(getattr(args, "warmup_max_seconds", 0.0)), 0.0)
    target_files = total_files if warmup_max_files <= 0 else min(total_files, warmup_max_files)

    writer.write(f"=== Warmup Cache Start ({target_files}/{total_files} files) ===")
    if warmup_frozen:
        log_cache_status(model, writer, "Warmup cache initial")
    warmup_log_rate = max(int(getattr(args, "warmup_log_rate", 0)), 0)
    auto_disable_partial_cache = bool(getattr(args, "auto_disable_partial_frozen_cache", True))
    frozen_cache_checked = False
    stop_warmup = False
    processed = 0
    warmup_start = time.time()
    stop_reason = None

    with torch.inference_mode():
        for _, dataset in seq_datasets:
            for step, pts in enumerate(dataset):
                if warmup_max_files > 0 and processed >= warmup_max_files:
                    stop_warmup = True
                    stop_reason = f"file limit reached ({warmup_max_files})"
                    break
                if warmup_max_seconds > 0 and (time.time() - warmup_start) >= warmup_max_seconds:
                    stop_warmup = True
                    stop_reason = f"time limit reached ({warmup_max_seconds:.1f}s)"
                    break

                file_path = dataset.files[step]
                cache_key = make_step_cache_key(file_path, args)
                input_xyz, patches, _, _ = prepare_whole_cloud_inputs(pts, args, cache_key, use_cuda)
                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=bool(use_cuda and use_amp)) if use_cuda else nullcontext()
                with autocast_ctx:
                    if warmup_frozen:
                        model.warmup_frozen_input(
                            patches,
                            cache_key=cache_key,
                        )
                        if auto_disable_partial_cache and not frozen_cache_checked:
                            frozen_cache_checked = True
                            if not getattr(model, "cache_enabled", False):
                                warmup_frozen = False
                                stop_warmup = not warmup_gt
                                if stop_warmup:
                                    stop_reason = "frozen cache disabled and no GT warmup requested"
                            else:
                                estimate_fn = getattr(model, "estimate_input_cache_capacity_entries", None)
                                estimated_capacity = estimate_fn() if callable(estimate_fn) else total_files
                                if estimated_capacity < total_files:
                                    clear_fn = getattr(model, "disable_input_cache", None)
                                    if callable(clear_fn):
                                        clear_fn()
                                    warmup_frozen = False
                                    writer.write(
                                        "Frozen input cache was disabled because the estimated capacity "
                                        f"({estimated_capacity} files) is smaller than the training pass "
                                        f"({total_files} files). This avoids prewarming entries that LRU "
                                        "evicts before reuse."
                                    )
                                    stop_warmup = not warmup_gt
                                    if stop_warmup:
                                        stop_reason = "frozen cache disabled and no GT warmup requested"
                    if warmup_gt:
                        loss.warmup_gt_cache(input_xyz, cache_key=cache_key)
                processed += 1
                elapsed = time.time() - warmup_start
                if warmup_log_rate > 0 and (processed % warmup_log_rate == 0 or processed >= target_files):
                    sec_per_file = elapsed / max(processed, 1)
                    writer.write(
                        f"Warmup cache progress: {processed}/{target_files} "
                        f"(total={total_files}, elapsed={elapsed:.1f}s, {sec_per_file:.2f}s/file)"
                    )
                    if warmup_frozen:
                        log_cache_status(model, writer, "Warmup cache status")
                if warmup_max_seconds > 0 and elapsed >= warmup_max_seconds:
                    stop_warmup = True
                    stop_reason = f"time limit reached ({warmup_max_seconds:.1f}s)"
                if stop_warmup:
                    break
            if stop_warmup:
                break

    if use_cuda:
        torch.cuda.empty_cache()
    if warmup_frozen:
        log_cache_status(model, writer, "Warmup cache final")
    elapsed = time.time() - warmup_start
    if stop_reason is not None:
        writer.write(f"Warmup cache stopped: {stop_reason}")
    writer.write(f"Warmup cache summary: processed={processed}/{target_files}, elapsed={elapsed:.1f}s")
    writer.write("=== Warmup Cache Done ===")


def cuda_bf16_ops_safe():
    if not (torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported")):
        return False
    if not torch.cuda.is_bf16_supported():
        return False
    version_text = str(getattr(torch, "__version__", "0.0")).split("+", 1)[0]
    try:
        major_text, minor_text, *_ = version_text.split(".")
        return (int(major_text), int(minor_text)) >= (2, 0)
    except (TypeError, ValueError):
        return False


def resolve_amp_dtype(args, use_cuda):
    if not use_cuda:
        return torch.float16

    requested = str(getattr(args, "amp_dtype", "auto")).strip().lower()
    bf16_safe = cuda_bf16_ops_safe()

    if requested == "auto":
        return torch.bfloat16 if bf16_safe else torch.float16
    if requested in {"bf16", "bfloat16"}:
        if not bf16_safe:
            raise ValueError(
                "Requested bf16 AMP, but this environment does not safely support "
                "the bf16 CUDA ops used by the model. Use --amp_dtype fp16."
            )
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {requested}")
