import os
import gc
_TMPDIR = os.environ.get("TMPDIR") or "/dev/shm/mynet_tmp"
try:
    os.makedirs(_TMPDIR, exist_ok=True)
    os.environ["TMPDIR"] = _TMPDIR
    os.environ["TEMP"] = _TMPDIR
    os.environ["TMP"] = _TMPDIR
except OSError:
    pass
import shutil
import torch
import torch.optim as optim
import sys
import argparse
import hashlib
import numpy as np
from pathlib import Path

import time
import datetime
from contextlib import nullcontext

from glob import glob
from models.utils.pointcloud.utils_repkpu import *
from models.utils.pointcloud.utils_repkpu import *
from models.utils.pointcloud.octree_subtree import (
    assign_octree_subtree_keys,
    build_octree_subtree_reference,
    build_subtree_index_map,
)
from models.utils.data.dataset import PlyDirDataset
from models.utils.patching.patch import build_patch_info, denormalize_patch_output, merge_patch_outputs
from record.write import Writing
from models.network import Network
from models.utils.loss.loss import Loss
from models.utils.config.args import parse_pugan_args
from models.utils.io.utils_ply import write_ply
from models.utils.testing.metrics import compute_pointcloud_metrics
from torch.utils.data import DataLoader


# 点群映像のみを使ったトレーニングか否か
Flag_video = 2
if Flag_video == 1:
    dir_input = "video_scaled"
elif Flag_video == 2:
    dir_input = "video_noised"
def _effective_patch_batch_size(args, patch_count=None, patch_size=None, is_train=False, writer=None):
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


def _adapt_encoder_state_dict_for_sparse_input(model, encoder_state, writer=None):
    key = "stem.0.weight"
    model_state = model.encoder.state_dict()
    if key not in encoder_state or key not in model_state:
        return encoder_state

    saved_weight = encoder_state[key].detach().cpu()
    target_weight = model_state[key].detach().cpu()
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


def _adapt_model_state_dict_for_sparse_input(model, state_dict, writer=None):
    key = "encoder.stem.0.weight"
    model_state = model.state_dict()
    if key not in state_dict or key not in model_state:
        return state_dict

    saved_weight = state_dict[key].detach().cpu()
    target_weight = model_state[key].detach().cpu()
    if tuple(saved_weight.shape) == tuple(target_weight.shape):
        return state_dict

    adapted = dict(state_dict)
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
            f"Model checkpoint stem adapted for sparse input: {tuple(saved_weight.shape)} -> {tuple(new_weight.shape)}"
        )
    return adapted


def _adapt_state_dict_to_model_shapes(state_dict, target_state_dict, writer=None, label="Checkpoint"):
    adapted = dict(state_dict)
    adapted_msgs = []
    skipped_msgs = []

    for key, saved_tensor in state_dict.items():
        target_tensor = target_state_dict.get(key)
        if target_tensor is None or not torch.is_tensor(saved_tensor) or not torch.is_tensor(target_tensor):
            continue
        if tuple(saved_tensor.shape) == tuple(target_tensor.shape):
            continue
        if saved_tensor.ndim != target_tensor.ndim:
            skipped_msgs.append(
                f"{key}: ndim mismatch {saved_tensor.ndim} -> {target_tensor.ndim}"
            )
            continue

        new_tensor = target_tensor.detach().cpu().clone()
        source_tensor = saved_tensor.detach().cpu().to(dtype=new_tensor.dtype)
        copy_slices = tuple(
            slice(0, min(int(src_dim), int(dst_dim)))
            for src_dim, dst_dim in zip(source_tensor.shape, new_tensor.shape)
        )
        new_tensor[copy_slices] = source_tensor[copy_slices]
        adapted[key] = new_tensor
        adapted_msgs.append(
            f"{key}: {tuple(saved_tensor.shape)} -> {tuple(target_tensor.shape)}"
        )

    if writer is not None and hasattr(writer, "write"):
        if adapted_msgs:
            writer.write(f"{label} shape-adapted {len(adapted_msgs)} tensors.")
            for msg in adapted_msgs[:20]:
                writer.write(f"  {msg}")
            if len(adapted_msgs) > 20:
                writer.write(f"  ... {len(adapted_msgs) - 20} more")
        if skipped_msgs:
            writer.write(f"{label} skipped {len(skipped_msgs)} tensors due to ndim mismatch.")
            for msg in skipped_msgs[:10]:
                writer.write(f"  {msg}")
            if len(skipped_msgs) > 10:
                writer.write(f"  ... {len(skipped_msgs) - 10} more")
    return adapted


def _stable_index_subset(num_points, max_points, method, key, seed):
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
    seed_text = f"{key}|test|{int(seed)}"
    seed_value = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_value)
    return torch.randperm(num_points, generator=generator)[:max_points]


def _downsample_input_batch(input_pcd, args, cache_key):
    max_points = int(getattr(args, "max_input_points", 0))
    if max_points <= 0 and not bool(getattr(args, "allow_unbounded_input", False)):
        max_points = int(getattr(args, "safe_max_input_points", 0))
    if max_points <= 0:
        return input_pcd
    if input_pcd.dim() != 3:
        raise ValueError(f"Expected input_pcd to have shape [B, N, C], got {tuple(input_pcd.shape)}")
    idx = _stable_index_subset(
        num_points=input_pcd.shape[1],
        max_points=max_points,
        method=getattr(args, "input_sampling", "random"),
        key=cache_key,
        seed=args.seed,
    )
    if idx is None:
        return input_pcd
    return input_pcd.index_select(1, idx)


def _sample_geometry_audit_tensors(gen_pts, gt_pts, final_w, out_label, args, cache_key):
    max_points = max(int(getattr(args, "geometry_audit_max_points", 0)), 0)
    if max_points <= 0:
        return gen_pts, gt_pts, final_w, out_label

    def _sample_pts(tensor, key_suffix):
        idx = _stable_index_subset(
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


def _resolved_test_save_paths(args, step_idx):
    base_dir = Path(os.path.abspath(os.path.expanduser(args.save_ply_dir)))
    run_dir = base_dir / "runs" / f"{args.date}_{args.time}_{Path(args.ckpt).stem}"
    filename = f"{step_idx:04d}_{getattr(args, 'method_name', 'Mine')}.ply"
    return base_dir, base_dir / filename, run_dir, run_dir / filename


def _format_log_value(value, kind="float"):
    if value is None:
        return "nan"
    if isinstance(value, (str, Path)):
        return str(value)
    if kind == "int":
        return str(int(round(float(value))))
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    if kind == "time":
        return f"{value:.6f}"
    return f"{value:.6g}"


def _emit_table(title, headers, rows, writer):
    rows = [tuple(str(value) for value in row) for row in rows]
    headers = tuple(str(value) for value in headers)
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def _row(row):
        return "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"

    writer.write(title)
    writer.write(_row(headers))
    writer.write("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows:
        writer.write(_row(row))


def _format_named_float_map(values):
    if not values:
        return "n/a"
    return ", ".join(f"{key}={float(value):.4f}" for key, value in values.items())


def _average_named_float_maps(chunks, key):
    sums = {}
    counts = {}
    for chunk in chunks:
        mapping = chunk.get(key) or {}
        for name, value in mapping.items():
            sums[name] = sums.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    return {
        name: sums[name] / float(max(counts[name], 1))
        for name in sums
    }


def _sum_named_counts(chunks, key):
    totals = {}
    for chunk in chunks:
        mapping = chunk.get(key) or {}
        for name, value in mapping.items():
            totals[name] = totals.get(name, 0) + int(value)
    return totals


def _merge_operation_by_cause(chunks):
    merged = {}
    for chunk in chunks:
        for cause_name, info in (chunk.get("operation_by_cause") or {}).items():
            entry = merged.setdefault(cause_name, {"count": 0, "ops": {}, "confidence_sum": 0.0, "confidence_count": 0})
            count = int(info.get("count", 0))
            operation = info.get("operation", "none")
            entry["count"] += count
            entry["ops"][operation] = entry["ops"].get(operation, 0) + count
            entry["confidence_sum"] += float(info.get("confidence", 0.0))
            entry["confidence_count"] += 1
    result = {}
    for cause_name, entry in merged.items():
        operation = max(entry["ops"], key=entry["ops"].get) if entry["ops"] else "none"
        result[cause_name] = {
            "count": int(entry["count"]),
            "operation": operation,
            "confidence": entry["confidence_sum"] / float(max(entry["confidence_count"], 1)),
        }
    return result


def _merge_octree_level_debug(chunks):
    buckets = {}
    for chunk in chunks:
        for item in chunk.get("octree_level_debug") or []:
            level = int(item.get("level", 0))
            bucket = buckets.setdefault(level, {"count": 0, "occupied": 0.0, "single_ratio": 0.0, "children": 0.0})
            bucket["count"] += 1
            bucket["occupied"] += float(item.get("occupied_mean", 0.0))
            bucket["single_ratio"] += float(item.get("single_ratio_mean", 0.0))
            bucket["children"] += float(item.get("mean_children_mean", 0.0))
    merged = []
    for level in sorted(buckets):
        bucket = buckets[level]
        count = float(max(bucket["count"], 1))
        merged.append(
            {
                "level": level,
                "occupied_mean": bucket["occupied"] / count,
                "single_ratio_mean": bucket["single_ratio"] / count,
                "mean_children_mean": bucket["children"] / count,
            }
        )
    return merged


def _aggregate_structure_debug_chunks(chunks):
    chunks = [chunk for chunk in (chunks or []) if chunk]
    if not chunks:
        return {}
    if len(chunks) == 1:
        return dict(chunks[0])
    merged = dict(chunks[-1])
    merged["cause_mean"] = _average_named_float_maps(chunks, "cause_mean")
    merged["subtree_cause_mean"] = _average_named_float_maps(chunks, "subtree_cause_mean")
    merged["policy_mean"] = _average_named_float_maps(chunks, "policy_mean")
    merged["cause_argmax_counts"] = _sum_named_counts(chunks, "cause_argmax_counts")
    merged["policy_argmax_counts"] = _sum_named_counts(chunks, "policy_argmax_counts")
    merged["operation_by_cause"] = _merge_operation_by_cause(chunks)
    merged["policy_entropy"] = float(np.mean([float(chunk.get("policy_entropy", 0.0)) for chunk in chunks]))
    merged["policy_diversity"] = max(int(chunk.get("policy_diversity", 0)) for chunk in chunks)
    merged["octree_level_debug"] = _merge_octree_level_debug(chunks)
    merged["chunk_count"] = len(chunks)
    return merged


def _write_structure_decision_debug(writer, prefix, structure_debug):
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
        f"chunks={int(structure_debug.get('chunk_count', 1))}, "
        f"cause_mean=[{_format_named_float_map(cause_mean)}], "
        f"policy_mean=[{_format_named_float_map(policy_mean)}]"
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


def _emit_step_result_table(step_idx, step_record, writer):
    _emit_table(
        f"Step {step_idx + 1} Result",
        ["field", "value"],
        [
            ("input_path", step_record["input_path"]),
            ("output_path", step_record["output_path"]),
            ("deleted_points", _format_log_value(step_record["deleted_points"], "int")),
            ("added_points", _format_log_value(step_record["added_points"], "int")),
            ("adjusted_points", _format_log_value(step_record["adjusted_points"], "int")),
            ("output_points", _format_log_value(step_record["output_points"], "int")),
            ("cd", _format_log_value(step_record["cd"])),
            ("d1_psnr", _format_log_value(step_record["d1_psnr"])),
            ("d2_psnr", _format_log_value(step_record["d2_psnr"])),
            ("model_time", _format_log_value(step_record["model_time"], "time")),
        ],
        writer,
    )
    writer.write(f"\n")


def _summary_stat(values, fn):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(fn(arr))


def _emit_step_summary_table(step_records, writer):
    if not step_records:
        writer.write("StepResultSummary: no values recorded")
        return
    metric_specs = [
        ("deleted_points", "Deleted Points", "int"),
        ("added_points", "Added Points", "int"),
        ("adjusted_points", "Adjusted Points", "int"),
        ("output_points", "Output Points", "int"),
        ("cd", "CD", "float"),
        ("d1_psnr", "D1PSNR", "float"),
        ("d2_psnr", "D2PSNR", "float"),
        ("model_time", "Model Time", "time"),
    ]
    rows = []
    for key, label, kind in metric_specs:
        values = [record[key] for record in step_records]
        average_kind = "float" if kind == "int" else kind
        rows.append(
            (
                label,
                _format_log_value(_summary_stat(values, np.mean), average_kind),
                _format_log_value(_summary_stat(values, np.max), kind),
                _format_log_value(_summary_stat(values, np.min), kind),
            )
        )
    _emit_table("Step Result Summary", ["metric", "average", "max", "min"], rows, writer)


def _autocast_context(use_cuda, use_amp, amp_dtype):
    if not use_cuda:
        return nullcontext()
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            return amp_mod.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        except TypeError:
            pass
    cuda_amp_mod = getattr(torch.cuda, "amp", None)
    if cuda_amp_mod is not None and hasattr(cuda_amp_mod, "autocast"):
        return cuda_amp_mod.autocast(dtype=amp_dtype, enabled=use_amp)
    return nullcontext()


def _extract_input_attr(input_pcd):
    if input_pcd.dim() != 3 or input_pcd.shape[1] <= 3:
        return None
    return input_pcd[:, 3:, :].contiguous()


def _aligned_edit_ref_xyz(input_xyz, output_points):
    ref_xyz = input_xyz[:, :3, :]
    output_points = int(output_points)
    if ref_xyz.shape[-1] == output_points:
        return ref_xyz.contiguous()
    if ref_xyz.shape[-1] > output_points:
        return ref_xyz[:, :, :output_points].contiguous()
    pad = ref_xyz.new_full((ref_xyz.shape[0], 3, output_points - ref_xyz.shape[-1]), float("nan"))
    return torch.cat([ref_xyz, pad], dim=2).contiguous()


def _compute_drop_hardening(final_w, args):
    if final_w is None:
        return None, {
            "mode": "none",
            "threshold": float(getattr(args, "test_drop_threshold", 0.5)),
            "keep_count": None,
            "total_count": None,
        }

    flat_w = final_w.reshape(-1)
    drop_th = float(getattr(args, "test_drop_threshold", 0.5))
    total_count = int(flat_w.numel())
    keep_mask = flat_w >= drop_th
    keep_count = int(keep_mask.sum().item())
    hardening_mode = "threshold"

    if total_count > 0 and (keep_count <= 0 or keep_count >= total_count):
        expected_keep = int(round(float(flat_w.clamp(0.0, 1.0).sum().item())))
        expected_keep = min(max(expected_keep, 1), total_count)
        if 0 < expected_keep < total_count:
            topk_idx = torch.topk(flat_w, k=expected_keep, largest=True, sorted=False).indices
            keep_mask = torch.zeros_like(flat_w, dtype=torch.bool)
            keep_mask.scatter_(0, topk_idx, True)
            keep_count = expected_keep
            hardening_mode = "expected_keep"

    return keep_mask, {
        "mode": hardening_mode,
        "threshold": drop_th,
        "keep_count": keep_count,
        "total_count": total_count,
    }


def _summarize_point_edits(input_xyz, pre_harden_gen_pts, final_gen_pts, edit_ref_xyz, keep_mask, args):
    input_points = int(input_xyz.shape[-1])
    pre_output_points = int(pre_harden_gen_pts.shape[-1])
    output_points = int(final_gen_pts.shape[-1])
    added_points = max(pre_output_points - input_points, 0)
    deleted_points = max(input_points + added_points - output_points, 0)

    if edit_ref_xyz is None:
        edit_ref_xyz = _aligned_edit_ref_xyz(input_xyz, pre_output_points)

    compare_points = min(int(pre_harden_gen_pts.shape[-1]), int(edit_ref_xyz.shape[-1]))
    adjusted_points = 0
    max_adjust = 0.0
    mean_adjust = 0.0
    threshold = max(float(getattr(args, "test_adjust_threshold", 1e-6)), 0.0)
    if compare_points > 0:
        gen_xyz = pre_harden_gen_pts[:, :3, :compare_points]
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

    return {
        "input_points": input_points,
        "pre_output_points": pre_output_points,
        "output_points": output_points,
        "added_points": int(added_points),
        "deleted_points": int(deleted_points),
        "adjusted_points": int(adjusted_points),
        "adjust_threshold": threshold,
        "adjust_mean": mean_adjust,
        "adjust_max": max_adjust,
        "net_change": output_points - input_points,
    }


def _compression_efficiency_value(comp_debug, fallback=None):
    if comp_debug:
        for key in ("actual_total_bit_percent", "total_bit"):
            value = comp_debug.get(key)
            if value is not None:
                return float(value)
    if fallback is None:
        return None
    if torch.is_tensor(fallback):
        return float(fallback.detach().cpu())
    return float(fallback)


def _build_inference_subtree_ref(input_xyz, args):
    depth = int(getattr(args, "test_subtree_level", 0))
    if depth > 0:
        return build_octree_subtree_reference(input_xyz, args, depth=depth)
    return build_octree_subtree_reference(input_xyz, args)


def _build_test_subtree_groups(input_xyz, args):
    subtree_ref = _build_inference_subtree_ref(input_xyz, args)
    min_points = max(
        int(getattr(args, "test_subtree_min_points", getattr(args, "train_subtree_min_points", 1))),
        1,
    )
    depth = int(subtree_ref["depth"][0].item())
    while True:
        subtree_keys = assign_octree_subtree_keys(input_xyz, subtree_ref)
        unique_keys, subtree_index_lists = build_subtree_index_map(subtree_keys)
        if int(unique_keys.numel()) <= 0:
            raise RuntimeError("No valid subtrees were found for subtree_merge inference.")
        point_counts = [int(point_idx.numel()) for point_idx in subtree_index_lists]
        if min(point_counts) >= min_points or depth <= 1:
            break
        depth -= 1
        subtree_ref = build_octree_subtree_reference(input_xyz, args, depth=depth)

    groups = [
        (int(subtree_key.detach().cpu()), point_idx)
        for subtree_key, point_idx in zip(unique_keys, subtree_index_lists)
    ]
    point_counts = [int(point_idx.numel()) for _, point_idx in groups]
    subtree_stats = {
        "depth": int(subtree_ref["depth"][0].item()),
        "count": len(groups),
        "min_points": min(point_counts),
        "mean_points": sum(point_counts) / float(max(len(point_counts), 1)),
        "max_points": max(point_counts),
        "target_min_points": min_points,
    }
    return subtree_ref, groups, subtree_stats


def _sync_cuda_if_needed(use_cuda):
    if use_cuda:
        torch.cuda.synchronize()


def _format_peak_memory_bytes(num_bytes):
    if not num_bytes:
        return "0B"
    mib = float(num_bytes) / float(1024 ** 2)
    if mib >= 1024.0:
        return f"{mib / 1024.0:.2f}GiB"
    return f"{mib:.1f}MiB"


def _is_oom_error(exc):
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _run_full_cloud_inference(model, input_pcd, args, cache_key, use_cuda, use_amp, amp_dtype):
    input_xyz = input_pcd[:, :3, :]
    input_attr = _extract_input_attr(input_pcd)
    encoder_debug_chunks = []
    subtree_ref = _build_inference_subtree_ref(input_xyz, args) if bool(getattr(args, "train_patch_subset_enable", False)) else None
    with _autocast_context(use_cuda, use_amp, amp_dtype):
        model_out = model.forward(
            input_xyz,
            input_attr,
            cache_key=cache_key,
            compute_internal_losses=False,
            subtree_ref=subtree_ref,
            selected_subtree_keys=None,
        )
    base_model = model.module if hasattr(model, "module") else model
    encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))
    structure_debug_chunks = [dict(getattr(base_model, "last_structure_debug", {}) or {})]
    result = {
        "mode": "full_cloud",
        "gen_pts": model_out[0],
        "final_w": model_out[4],
        "encoder_debug_chunks": encoder_debug_chunks,
        "structure_debug_chunks": structure_debug_chunks,
        "edit_ref_xyz": _aligned_edit_ref_xyz(input_xyz, model_out[0].shape[-1]),
    }
    if subtree_ref is not None:
        subtree_keys = assign_octree_subtree_keys(input_xyz, subtree_ref)
        _, subtree_index_lists = build_subtree_index_map(subtree_keys)
        point_counts = [int(point_idx.numel()) for point_idx in subtree_index_lists]
        result["subtree_stats"] = {
            "depth": int(subtree_ref["depth"][0].item()),
            "count": len(point_counts),
            "min_points": min(point_counts) if point_counts else 0,
            "mean_points": (sum(point_counts) / float(max(len(point_counts), 1))) if point_counts else 0.0,
            "max_points": max(point_counts) if point_counts else 0,
            "target_min_points": max(
                int(getattr(args, "test_subtree_min_points", getattr(args, "train_subtree_min_points", 1))),
                1,
            ),
        }
    return result


def _run_subtree_merge_inference(model, input_pcd, args, cache_key, use_cuda, use_amp, amp_dtype):
    input_xyz = input_pcd[:, :3, :]
    input_attr = _extract_input_attr(input_pcd)
    _, groups, subtree_stats = _build_test_subtree_groups(input_xyz, args)
    encoder_debug_chunks = []
    structure_debug_chunks = []
    merged_pts = []
    merged_weights = []
    merged_refs = []
    prev_log_flag = getattr(args, "_log_this_step", False)
    try:
        args._log_this_step = bool(getattr(args, "verbose_step_logs", False))
        for subtree_key, point_idx in groups:
            subtree_xyz = input_xyz.index_select(2, point_idx).contiguous()
            subtree_attr = input_attr.index_select(2, point_idx).contiguous() if input_attr is not None else None
            subtree_cache_key = (
                f"{cache_key}|test_subtree_depth={int(subtree_stats['depth'])}|subtree_key={subtree_key}"
            )
            with _autocast_context(use_cuda, use_amp, amp_dtype):
                model_out = model.forward(
                    subtree_xyz,
                    subtree_attr,
                    cache_key=subtree_cache_key,
                    compute_internal_losses=False,
            )
            base_model = model.module if hasattr(model, "module") else model
            encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))
            structure_debug_chunks.append(dict(getattr(base_model, "last_structure_debug", {}) or {}))
            merged_pts.append(model_out[0].squeeze(0))
            merged_refs.append(_aligned_edit_ref_xyz(subtree_xyz, model_out[0].shape[-1]).squeeze(0))
            if model_out[4] is not None:
                merged_weights.append(model_out[4].squeeze(0))
    finally:
        args._log_this_step = prev_log_flag

    gen_pts = torch.cat(merged_pts, dim=1).unsqueeze(0).contiguous()
    final_w = torch.cat(merged_weights, dim=1).unsqueeze(0).contiguous() if merged_weights else None
    edit_ref_xyz = torch.cat(merged_refs, dim=1).unsqueeze(0).contiguous() if merged_refs else None
    return {
        "mode": "subtree_merge",
        "gen_pts": gen_pts,
        "final_w": final_w,
        "edit_ref_xyz": edit_ref_xyz,
        "encoder_debug_chunks": encoder_debug_chunks,
        "structure_debug_chunks": structure_debug_chunks,
        "subtree_stats": subtree_stats,
    }


def _run_patch_inference(model, input_pcd, args, cache_key, use_cuda, use_amp, amp_dtype, writer=None):
    encoder_debug_chunks = []
    structure_debug_chunks = []
    patch_info = build_patch_info(input_pcd, args)
    pb = _effective_patch_batch_size(
        args,
        patch_count=patch_info["num_patches"],
        patch_size=args.num_points,
        is_train=False,
        writer=writer,
    )
    patch_outputs = []
    edit_ref_chunks = []
    patch_count = patch_info["num_patches"]

    with _autocast_context(use_cuda, use_amp, amp_dtype):
        for i in range(0, patch_count, pb):
            patch_xyz = patch_info["patch_xyz"][i:i+pb]
            patch_attr = patch_info["patch_attr"][i:i+pb]
            patch_cache_keys = [
                f"{cache_key}|patch={patch_id}"
                for patch_id in range(i, i + patch_xyz.shape[0])
            ]
            model_out = model.forward(
                patch_xyz,
                patch_attr,
                cache_key=patch_cache_keys,
                return_patch_meta=True,
                coord_scale=patch_info["patch_scale"][i:i+pb],
                compute_internal_losses=False,
            )
            base_model = model.module if hasattr(model, "module") else model
            encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))
            structure_debug_chunks.append(dict(getattr(base_model, "last_structure_debug", {}) or {}))
            gen_chunk = model_out[0]
            final_w_chunk = model_out[4]
            patch_meta_chunk = model_out[-1]
            gen_chunk = denormalize_patch_output(
                gen_chunk,
                patch_info["patch_centroid"][i:i+pb],
                patch_info["patch_scale"][i:i+pb],
            )

            chunk_size = patch_xyz.shape[0]
            for local_idx in range(chunk_size):
                patch_id = i + local_idx
                patch_input_idx = patch_info["patch_input_idx"][patch_id]
                owned_input_mask = patch_info["owned_input_mask"][patch_id]
                anchor_idx_local = patch_meta_chunk["anchor_idx_local"][local_idx].clamp_(0, patch_input_idx.shape[0] - 1)
                valid_mask = patch_meta_chunk["output_valid_mask"][local_idx]
                owned_output_mask = owned_input_mask.index_select(0, anchor_idx_local)
                select_mask = valid_mask & owned_output_mask
                selected_pts = gen_chunk[local_idx, :, select_mask]
                selected_w = None if final_w_chunk is None else final_w_chunk[local_idx, :, select_mask]
                patch_input_xyz_world = (
                    patch_info["patch_centroid"][patch_id:patch_id+1]
                    + patch_info["patch_xyz"][patch_id:patch_id+1] * patch_info["patch_scale"][patch_id:patch_id+1]
                )
                if select_mask.any():
                    selected_ref_idx = anchor_idx_local[select_mask]
                    edit_ref_chunks.append(patch_input_xyz_world[local_idx, :, selected_ref_idx])
                represented_owned_mask = torch.zeros_like(owned_input_mask)
                if select_mask.any():
                    represented_owned_mask[anchor_idx_local[select_mask]] = True
                missing_owned_mask = owned_input_mask & (~represented_owned_mask)
                fallback_pts = None
                fallback_w = None
                if missing_owned_mask.any():
                    patch_input_pts_world = torch.cat(
                        [patch_input_xyz_world, patch_info["patch_attr"][patch_id:patch_id+1]],
                        dim=1,
                    )
                    fallback_pts = patch_input_pts_world[local_idx, :, missing_owned_mask]
                    fallback_w = gen_chunk.new_ones((1, int(missing_owned_mask.sum().item())))
                    edit_ref_chunks.append(patch_input_xyz_world[local_idx, :, missing_owned_mask])

                owned_local_idx = torch.nonzero(owned_input_mask, as_tuple=False).flatten()
                owned_global_idx = None
                owned_out_label = None
                if owned_local_idx.numel() > 0:
                    owned_global_idx = patch_input_idx.index_select(0, owned_local_idx)
                    if patch_meta_chunk["out_label"] is not None:
                        owned_out_label = patch_meta_chunk["out_label"][local_idx, owned_local_idx]
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

    gen_pts, final_w, _ = merge_patch_outputs(
        patch_info,
        patch_outputs,
        device=input_pcd.device,
        dtype=input_pcd.dtype,
    )
    edit_ref_xyz = torch.cat(edit_ref_chunks, dim=1).unsqueeze(0).contiguous() if edit_ref_chunks else None
    return {
        "mode": "patch",
        "gen_pts": gen_pts,
        "final_w": final_w,
        "edit_ref_xyz": edit_ref_xyz,
        "encoder_debug_chunks": encoder_debug_chunks,
        "structure_debug_chunks": structure_debug_chunks,
        "patch_stats": {
            "count": int(patch_count),
            "batch_size": int(pb),
        },
    }


def _run_direct_inference(model, input_pcd, args, use_cuda, use_amp, amp_dtype):
    encoder_debug_chunks = []
    patches_xyz = input_pcd[:, :3, :]
    patches_attr = _extract_input_attr(input_pcd)
    patches, centroid, furthest_distance = normalize_point_cloud(patches_xyz)

    with _autocast_context(use_cuda, use_amp, amp_dtype):
        model_out = model.forward(
            patches,
            patches_attr,
            coord_scale=furthest_distance,
            compute_internal_losses=False,
        )
    base_model = model.module if hasattr(model, "module") else model
    encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))
    structure_debug_chunks = [dict(getattr(base_model, "last_structure_debug", {}) or {})]
    gen_patches = model_out[0]
    final_w = model_out[4]
    centroid_xyz = centroid[:, :3, :]
    gen_xyz = centroid_xyz + gen_patches[:, :3, :] * furthest_distance
    if gen_patches.shape[1] > 3:
        gen_patches = torch.cat([gen_xyz, gen_patches[:, 3:, :]], dim=1)
    else:
        gen_patches = gen_xyz
    gen_pts = rearrange(gen_patches, 'b c n -> 1 c (b n)').contiguous()
    return {
        "mode": "direct",
        "gen_pts": gen_pts,
        "final_w": final_w,
        "edit_ref_xyz": _aligned_edit_ref_xyz(input_pcd[:, :3, :], gen_pts.shape[-1]),
        "encoder_debug_chunks": encoder_debug_chunks,
        "structure_debug_chunks": structure_debug_chunks,
    }


def _legacy_inference_mode(args):
    if bool(getattr(args, "train_patch_subset_enable", False)):
        return "full_cloud"
    if bool(getattr(args, "split2patch", False)):
        return "patch"
    return "direct"


def _run_named_inference_mode(mode_name, model, input_pcd, args, cache_key, use_cuda, use_amp, amp_dtype, writer=None):
    if mode_name == "legacy":
        mode_name = _legacy_inference_mode(args)
    if mode_name == "full_cloud":
        return _run_full_cloud_inference(model, input_pcd, args, cache_key, use_cuda, use_amp, amp_dtype)
    if mode_name == "subtree_merge":
        return _run_subtree_merge_inference(model, input_pcd, args, cache_key, use_cuda, use_amp, amp_dtype)
    if mode_name == "patch":
        return _run_patch_inference(model, input_pcd, args, cache_key, use_cuda, use_amp, amp_dtype, writer=writer)
    if mode_name == "direct":
        return _run_direct_inference(model, input_pcd, args, use_cuda, use_amp, amp_dtype)
    raise ValueError(f"Unsupported inference mode: {mode_name}")


def _measure_inference_mode(mode_name, model, input_pcd, args, cache_key, use_cuda, use_amp, amp_dtype, writer=None):
    device = next(model.parameters()).device
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        _sync_cuda_if_needed(use_cuda)
    st = time.perf_counter()
    result = _run_named_inference_mode(
        mode_name,
        model,
        input_pcd,
        args,
        cache_key,
        use_cuda,
        use_amp,
        amp_dtype,
        writer=writer,
    )
    _sync_cuda_if_needed(use_cuda)
    elapsed = time.perf_counter() - st
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0
    return {
        "ok": True,
        "mode": mode_name,
        "elapsed": elapsed,
        "peak_memory": peak_memory,
        "result": result,
    }


def _select_auto_inference_mode(measurements, args):
    successful = [item for item in measurements if item.get("ok", False)]
    if not successful:
        raise RuntimeError("Automatic inference-mode selection did not find any successful mode.")
    fastest = min(successful, key=lambda item: item["elapsed"])
    time_tol = float(getattr(args, "test_auto_time_tolerance", 0.10))
    time_cutoff = fastest["elapsed"] * (1.0 + time_tol)
    close_in_time = [item for item in successful if item["elapsed"] <= time_cutoff]
    return min(close_in_time, key=lambda item: (item["peak_memory"], item["elapsed"]))


def test(model, loss, args, writer):
    """==========================================================="""
    """セットアップ"""
    """==========================================================="""
    model.eval()

    writer.write(f"model: {args.ckpt}")
    writer.write(f"input dir: {args.input_dir_test}")
    writer.write(f"output ply dir: {args.save_ply_dir}")
    writer.write(f"Method Name: {getattr(args, 'method_name', 'Mine')}")
    writer.write(f"Surrogate Name: {getattr(args, 'surrogate_name', getattr(args, 'compress', 'OctAttention'))}")
    writer.write(
        "Optimization Modes: "
        f"geometry={'ste_hard' if args.discrete_loss_mode == 'ste_hard' else ('weighted_soft' if args.discrete_loss_mode == 'weighted_soft' else 'hard')}, "
        f"compression={args.compression_loss_backend}"
    )
    writer.write(f"Compression Codec: {getattr(args, 'compress', 'OctAttention')}")
    if str(getattr(args, "compression_loss_backend", "proxy")).strip().lower().startswith("sparsepcgc"):
        writer.write(
            "SparsePCGC Teacher: "
            f"env={getattr(args, 'sparsepcgc_env', 'sparsepcgc')}, "
            f"python={getattr(args, 'sparsepcgc_python', '') or '(auto)'}, "
            f"mode={getattr(args, 'sparsepcgc_mode', 'dense_lossless')}, "
            f"device={getattr(args, 'sparsepcgc_device', 'auto')}, "
            f"skip_decode={bool(getattr(args, 'sparsepcgc_skip_decode', True))}"
        )
    if str(getattr(args, "compression_loss_backend", "proxy")).strip().lower().startswith("gpcc"):
        writer.write(
            "G-PCC Teacher: "
            f"encoder={getattr(args, 'gpcc_encoder_path', '')}, "
            f"cfg={getattr(args, 'gpcc_cfg_dir', '')}, "
            f"match_qs={bool(getattr(args, 'gpcc_match_qs', True))}, "
            f"prequantize={bool(getattr(args, 'gpcc_prequantize', True))}, "
            f"effective_qs={float(getattr(args, 'gpcc_effective_qs', 0.0))}, "
            f"geometry_only={bool(getattr(args, 'gpcc_disable_attribute_coding', True))}, "
            f"merge_duplicates={bool(getattr(args, 'gpcc_merge_duplicated_points', True))}"
        )
    writer.write(
        f"Input Sampling: max_input_points={args.max_input_points}, "
        f"safe_max_input_points={getattr(args, 'safe_max_input_points', 0)}, "
        f"allow_unbounded_input={getattr(args, 'allow_unbounded_input', False)}, "
        f"encoder_sparse_tensor={bool(getattr(args, 'encoder_sparse_tensor', True))}, "
        f"sparse_tensor_keep_after_encoder={bool(getattr(args, 'sparse_tensor_keep_after_encoder', True))}, "
        f"encoder_raw_downsample_factor={float(getattr(args, 'encoder_raw_downsample_factor', 1.0))}, "
        f"encoder_pre_downsample={bool(getattr(args, 'encoder_pre_downsample', False))}, "
        f"encoder_pre_downsample_mode={getattr(args, 'encoder_pre_downsample_mode', 'voxel')}, "
        f"encoder_pre_downsample_max_points={int(getattr(args, 'encoder_pre_downsample_max_points', 0))}, "
        f"encoder_feature_propagation={getattr(args, 'encoder_feature_propagation', 'knn_inverse_distance')}, "
        f"encoder_feature_propagation_k={int(getattr(args, 'encoder_feature_propagation_k', 3))}, "
        f"sampling={args.input_sampling}, "
        f"patch_parallel_mode={getattr(args, 'patch_parallel_mode', 'auto')}, "
        f"patch_batch_size={int(getattr(args, 'patch_batch_size', 1))}, "
        f"patch_budget_test={int(getattr(args, 'patch_parallel_points_budget_test', 0))}"
    )
    writer.write(f"Requested Inference Mode: {args.test_inference_mode}")
    writer.write(
        "Quality Metrics: "
        f"max_points={int(getattr(args, 'test_metric_max_points', 8192))}, "
        f"normal_k={int(getattr(args, 'test_metric_normal_k', 16))}, "
        "device=cpu"
    )

    print(f"Pruning Module used: {args.prune}")
    writer.write(f"Pruning Module used: {args.prune}")

    dataset = PlyDirDataset(args, args.input_dir_test)
    use_cuda = next(model.parameters()).is_cuda
    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=max(int(args.num_workers), 0),
        pin_memory=bool(use_cuda and args.pin_memory),
    )
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    loader = DataLoader(dataset, **loader_kwargs)
    use_amp = bool(use_cuda and getattr(args, "use_amp", False))
    amp_dtype = torch.float16

    with torch.inference_mode():
        # Lossの平均計算用の配列定義
        L_his = []
        L_geom_his = []
        L_bit_his = []
        L_nodes_his = []
        L_single_his = []
        compression_efficiency_his = []
        step_result_records = []
        selected_inference_mode = None
        for step, pts in enumerate(loader):
            st_step = time.time()
            writer.write(f"＊＊＊ Step {step + 1} ＊＊＊")
            print(f"*** step {step + 1}/{len(loader)} ***")
            cache_key = dataset.files[step]
            input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
            raw_points = int(input_pcd.shape[1])
            input_pcd = _downsample_input_batch(input_pcd, args, cache_key)
            sampled_points = int(input_pcd.shape[1])
            if raw_points != sampled_points:
                msg = (
                    f"Full-pipeline input truncated by max_input_points: "
                    f"{raw_points} -> {sampled_points} (sampling={args.input_sampling})"
                )
                print(msg)
                writer.write(msg)
            if use_cuda:
                input_pcd = input_pcd.cuda(non_blocking=True)
            input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous()

            # 入力点群データのファイル名
            print(f"Input PCD: {dataset.files[step]}")
            writer.write(f"Input PCD: {dataset.files[step]}")

            st_model = time.time()
            benchmark_measurements = None
            requested_mode = str(getattr(args, "test_inference_mode", "legacy")).strip().lower()
            if requested_mode == "auto":
                if selected_inference_mode is None:
                    benchmark_measurements = []
                    for candidate_mode in ("full_cloud", "subtree_merge"):
                        try:
                            measurement = _measure_inference_mode(
                                candidate_mode,
                                model,
                                input_pcd,
                                args,
                                cache_key,
                                use_cuda,
                                use_amp,
                                amp_dtype,
                                writer=writer,
                            )
                        except Exception as exc:
                            if use_cuda:
                                torch.cuda.empty_cache()
                            benchmark_measurements.append(
                                {
                                    "ok": False,
                                    "mode": candidate_mode,
                                    "error": str(exc),
                                    "oom": bool(use_cuda and _is_oom_error(exc)),
                                }
                            )
                            continue
                        benchmark_measurements.append(measurement)
                    successful = [item for item in benchmark_measurements if item.get("ok", False)]
                    if not successful:
                        details = "; ".join(
                            f"{item['mode']}={item.get('error', 'unknown error')}"
                            for item in benchmark_measurements
                        )
                        raise RuntimeError(f"All auto inference candidates failed: {details}")
                    selected_measurement = _select_auto_inference_mode(benchmark_measurements, args)
                    selected_inference_mode = selected_measurement["mode"]
                    for item in benchmark_measurements:
                        if item.get("ok", False) and item["mode"] != selected_inference_mode:
                            item.pop("result", None)
                    inference_measurement = selected_measurement
                else:
                    try:
                        inference_measurement = _measure_inference_mode(
                            selected_inference_mode,
                            model,
                            input_pcd,
                            args,
                            cache_key,
                            use_cuda,
                            use_amp,
                            amp_dtype,
                            writer=writer,
                        )
                    except Exception as exc:
                        if not (use_cuda and _is_oom_error(exc)):
                            raise
                        fallback_mode = "subtree_merge" if selected_inference_mode == "full_cloud" else "full_cloud"
                        writer.write(
                            "InferenceModeFallback: "
                            f"previous={selected_inference_mode}, "
                            f"reason=oom, "
                            f"retry={fallback_mode}"
                        )
                        torch.cuda.empty_cache()
                        inference_measurement = _measure_inference_mode(
                            fallback_mode,
                            model,
                            input_pcd,
                            args,
                            cache_key,
                            use_cuda,
                            use_amp,
                            amp_dtype,
                            writer=writer,
                        )
                        selected_inference_mode = inference_measurement["mode"]
            else:
                inference_measurement = _measure_inference_mode(
                    requested_mode,
                    model,
                    input_pcd,
                    args,
                    cache_key,
                    use_cuda,
                    use_amp,
                    amp_dtype,
                    writer=writer,
                )
                selected_inference_mode = inference_measurement["result"]["mode"]
            en_model = time.time()
            inference_result = inference_measurement["result"]
            gen_pts = inference_result["gen_pts"]
            final_w = inference_result["final_w"]
            encoder_debug_chunks = inference_result.get("encoder_debug_chunks", [])
            structure_debug = _aggregate_structure_debug_chunks(
                inference_result.get("structure_debug_chunks", [])
            )

            st_fp = time.time()
            pre_harden_gen_pts = gen_pts
            keep_mask, hardening_info = _compute_drop_hardening(final_w, args)
            if final_w is not None:
                keep_count = hardening_info["keep_count"]
                total_count = hardening_info["total_count"]
                if 0 < keep_count < total_count:
                    gen_pts = gen_pts[:, :, keep_mask].contiguous()

            edit_stats = _summarize_point_edits(
                input_xyz=input_pcd[:, :3, :],
                pre_harden_gen_pts=pre_harden_gen_pts,
                final_gen_pts=gen_pts,
                edit_ref_xyz=inference_result.get("edit_ref_xyz"),
                keep_mask=keep_mask,
                args=args,
            )
            out = gen_pts.squeeze(0).transpose(0, 1).detach().cpu().numpy()
            xyz = out[:, :3]
            rgb = out[:, 3:6] if out.shape[1] >= 6 else None
            input_points = int(input_pcd.shape[-1])
            output_points = int(gen_pts.shape[-1])
            point_ratio = output_points / max(float(input_points), 1.0)
            point_msg = (
                f"Point count: input={input_points}, output={output_points}, "
                f"ratio={point_ratio:.6f}"
            )
            print(point_msg)
            writer.write(point_msg)

            base_model = model.module if hasattr(model, "module") else model
            encoder_debug = {}
            if encoder_debug_chunks:
                coarse = []
                full = []
                raw = []
                pre_sparse = []
                analysis = []
                for item in encoder_debug_chunks:
                    raw.extend(item.get("raw_points", []))
                    pre_sparse.extend(item.get("pre_sparse_points", []))
                    analysis.extend(item.get("analysis_points", item.get("pre_sparse_points", [])))
                    coarse.extend(item.get("coarse_points", []))
                    full.extend(item.get("full_points", []))
                if coarse and full:
                    encoder_debug = {
                        "enabled": encoder_debug_chunks[-1].get("enabled", False),
                        "mode": encoder_debug_chunks[-1].get("mode", "none"),
                        "raw_points": raw,
                        "pre_sparse_points": pre_sparse,
                        "analysis_points": analysis,
                        "coarse_points": coarse,
                        "full_points": full,
                        "propagation": encoder_debug_chunks[-1].get("propagation", "none"),
                        "propagation_k": encoder_debug_chunks[-1].get("propagation_k", 0),
                        "raw_downsample_factor": encoder_debug_chunks[-1].get("raw_downsample_factor", 1.0),
                    }
            if not encoder_debug:
                encoder_debug = getattr(base_model, "last_encoder_debug", {}) or {}
            if not structure_debug:
                structure_debug = getattr(base_model, "last_structure_debug", {}) or {}
            _write_structure_decision_debug(
                writer,
                f"TestStructureDecision step={step + 1}",
                structure_debug,
            )

            if bool(getattr(args, "test_compute_loss", True)):
                L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss(
                    args,
                    gen_xyz=gen_pts[:, :3, :],
                    gt_xyz=input_pcd[:, :3, :],
                    final_w=None,
                    cache_key=cache_key,
                    refresh_actual_gen=False,
                )
                gen_a, gt_a, w_a, label_a = _sample_geometry_audit_tensors(
                    gen_pts,
                    input_pcd,
                    None,
                    None,
                    args,
                    cache_key,
                )
                with torch.no_grad():
                    geom_audit = loss.get_geometry_loss(
                        args,
                        gen_pts=gen_a,
                        gt_pts=gt_a,
                        final_w=w_a,
                        out_label=label_a,
                    )
                L_geom = geom_audit
                geom_debug = getattr(loss, "last_geometry_debug", {}) or {}
                writer.write(
                    "TestLoss: "
                    f"geom={float(L_geom.detach().cpu()):.6g}, "
                    f"com={float(L_com.detach().cpu()):.6g}, "
                    f"bit={float(loss_bit.detach().cpu()):.6g}, "
                    f"single={float(loss_single.detach().cpu()):.6g}, "
                    f"nodes={float(loss_nodes.detach().cpu()):.6g}"
                )
                comp_debug = getattr(loss, "last_compression_debug", {}) or {}
                compression_efficiency = _compression_efficiency_value(comp_debug, fallback=loss_bit)
                if compression_efficiency is not None:
                    compression_efficiency_his.append(compression_efficiency)
                    writer.write(
                        "CompressionEfficiency: "
                        f"total_bit_diff_percent={compression_efficiency:.6g}, "
                        f"metric={comp_debug.get('metric', 'total_bit_diff_percent')}, "
                        f"teacher_codec={comp_debug.get('teacher_codec', 'unknown')}, "
                        f"bpp_diff_percent={float(comp_debug.get('bpp', 0.0)):.6g}"
                    )
                writer.write(
                    "GeometryAudit:"
                    f" mode={geom_debug.get('mode', 'unknown')},"
                    f" train_value={float(L_geom.detach().cpu()):.6g},"
                    f" audit_value={float(geom_audit.detach().cpu()):.6g},"
                    f" hard={float(geom_debug.get('hard', 0.0)):.6g},"
                    f" weighted={float(geom_debug.get('weighted', 0.0)):.6g},"
                    f" surrogate={float(geom_debug.get('surrogate', 0.0)):.6g},"
                    f" points={gen_a.shape[-1]}->{gt_a.shape[-1]}"
                )

            field_list = [xyz.astype(np.float32)]
            field_names = ["x", "y", "z"]
            if rgb is not None and rgb.shape[1] >= 3:
                rgb = np.clip(rgb[:, :3] * 255.0, 0, 255).astype(np.uint8)
                field_list.append(rgb)
                field_names.extend(["red", "green", "blue"])

            en_fp = time.time()

            save_dir, save_path, run_dir, run_save_path = _resolved_test_save_paths(args, step)
            try:
                quality_metrics = compute_pointcloud_metrics(
                    input_pcd[:, :3, :],
                    gen_pts[:, :3, :],
                    max_points=int(getattr(args, "test_metric_max_points", 8192)),
                    normal_k=int(getattr(args, "test_metric_normal_k", 16)),
                    seed=int(getattr(args, "seed", 0)) + step,
                )
            except Exception as exc:
                writer.write(f"QualityMetricError: {exc}")
                quality_metrics = {
                    "cd": float("nan"),
                    "d1_psnr": float("nan"),
                    "d2_psnr": float("nan"),
                }
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(run_dir, exist_ok=True)
            print(f"save path: {save_path}")

            ok = write_ply(
                str(save_path),
                field_list,
                field_names,
            )
            if not ok:
                raise RuntimeError(f"write_ply returned False: {save_path}")
            if not save_path.exists():
                raise FileNotFoundError(f"Saved PLY was not found after write: {save_path}")
            shutil.copy2(save_path, run_save_path)
            save_stat = save_path.stat()
            run_stat = run_save_path.stat()
            en_step = time.time()
            model_time = en_model - st_model
            step_record = {
                "input_path": str(dataset.files[step]),
                "output_path": str(save_path),
                "deleted_points": int(edit_stats["deleted_points"]),
                "added_points": int(edit_stats["added_points"]),
                "adjusted_points": int(edit_stats["adjusted_points"]),
                "output_points": int(output_points),
                "cd": float(quality_metrics["cd"]),
                "d1_psnr": float(quality_metrics["d1_psnr"]),
                "d2_psnr": float(quality_metrics["d2_psnr"]),
                "model_time": float(model_time),
            }
            step_result_records.append(step_record)

            print(f"model time: {model_time}")
            writer.write(f"{step+1} Step time: {en_step - st_step}")
            writer.write(f"  - Model time: {model_time}")
            writer.write(f"  - Forward time: {en_fp - en_model}")
            writer.write(f"  - Sum time: {en_fp - st_fp}")
            writer.write(f"  - Step time: {en_step - st_step}")
            writer.write(
                "Saved PLY: "
                f"primary={save_path} (size={save_stat.st_size}B, mtime={save_stat.st_mtime:.6f}), "
                f"run_copy={run_save_path} (size={run_stat.st_size}B, mtime={run_stat.st_mtime:.6f})\n"
            )
            _emit_step_result_table(step, step_record, writer)

        if compression_efficiency_his:
            comp_arr = np.asarray(compression_efficiency_his, dtype=np.float64)
            writer.write("=== Compression Efficiency Summary ===")
            writer.write(
                "CompressionEfficiencySummary: "
                f"count={int(comp_arr.size)}, "
                f"metric=total_bit_diff_percent, "
                f"average={float(comp_arr.mean()):.6g}, "
                f"max={float(comp_arr.max()):.6g}, "
                f"min={float(comp_arr.min()):.6g}"
            )
        else:
            writer.write("CompressionEfficiencySummary: no values recorded")
        _emit_step_summary_table(step_result_records, writer)

        # L_avg = np.mean(L_his)
        # # L_geom_avg = np.mean(L_geom_his)
        # L_geom_avg = np.mean([x.cpu().numpy() for x in L_geom_his])
        # L_bit_avg = np.mean(L_bit_his)
        # L_sin_avg = np.mean(L_single_his)
        # L_node_avg = np.mean(L_nodes_his)
        
        # L_max = np.max(L_his)
        # # L_geom_max = np.max(L_geom_his)
        # L_geom_max = np.max([x.cpu().numpy() for x in L_geom_his])
        # L_bit_max = np.max(L_bit_his)
        # L_sin_max = np.max(L_single_his)
        # L_node_max = np.max(L_nodes_his)

        # L_min = np.min(L_his)
        # # L_geom_min = np.min(L_geom_his)
        # L_geom_min = np.min([x.cpu().numpy() for x in L_geom_his])
        # L_bit_min = np.min(L_bit_his)
        # L_sin_min = np.min(L_single_his)
        # L_node_min = np.min(L_nodes_his)

        # writer.write(f"=== Average ===")
        # writer.write(f"Loss         : {L_avg}")
        # writer.write(f"Loss Geom    : {L_geom_avg}")
        # writer.write(f"Loss Bit     : {L_bit_avg}")
        # writer.write(f"Loss Single  : {L_sin_avg}")
        # writer.write(f"Loss Nodes   : {L_node_avg}\n")
        
        # writer.write(f"=== MAX ===")
        # writer.write(f"Loss         : {L_max}")
        # writer.write(f"Loss Geom    : {L_geom_max}")
        # writer.write(f"Loss Bit     : {L_bit_max}")
        # writer.write(f"Loss Single  : {L_sin_max}")
        # writer.write(f"Loss Nodes   : {L_node_max}\n")

        # writer.write("=== Min ===")
        # writer.write(f"Loss         : {L_min}")
        # writer.write(f"Loss Geom    : {L_geom_min}")
        # writer.write(f"Loss Bit     : {L_bit_min}")
        # writer.write(f"Loss Single  : {L_sin_min}")
        # writer.write(f"Loss Nodes   : {L_node_min}\n")

if __name__ == '__main__':
    """=== セットアップ ==="""
    # テストInfoのセットアップ
    file_day = datetime.datetime.now().strftime('%Y%m%d')
    file_time = datetime.datetime.now().strftime('%H%M%S')

    parser = argparse.ArgumentParser(description='Testing Arguments')
    parser.add_argument('--trainORtest', default="test", type=str, help='date')
    args = parse_pugan_args(parser, file_day, file_time)
    if torch.cuda.is_available() and not args.cpu and args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    
    # ログのセットアップ
    writer = Writing(
        file_day,
        file_time,
        filename="MyNetwork_test",
        flush_every=args.log_flush_every,
        sync_every=args.log_sync_every,
        log_root=args.log_root,
    )
    writer.write(f"Date of Testing: {file_day}-{file_time}")
    writer.write(f"Log Root: {args.log_root}")
    writer.write(f"Output PLY Root: {args.save_ply_dir}")
    writer.write(f"Method Name: {getattr(args, 'method_name', 'Mine')}")
    writer.write(f"Surrogate Name: {getattr(args, 'surrogate_name', getattr(args, 'compress', 'OctAttention'))}")
    writer.write(f"Geometry Loss Type: {args.loss_type}")
    writer.write(f"Discrete Loss Mode: {args.discrete_loss_mode}")
    writer.write(f"Checkpoint Path: {args.ckpt}")


    # モデルのセットアップ
    print(f"Model Setting: {datetime.datetime.now()}")
    model = Network(args, writer)
    # ===== RepKPU Encoder の重みをロード =====
    repkpu_ckpt = os.path.join(os.path.dirname(__file__), "repkpu_model", "ckpt-best.pth")
    ckpt = torch.load(repkpu_ckpt, map_location="cpu")
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in ckpt.items()
        if k.startswith("encoder.")
    }
    encoder_state = _adapt_encoder_state_dict_for_sparse_input(model, encoder_state, writer=writer)
    encoder_state = _adapt_state_dict_to_model_shapes(
        encoder_state,
        model.encoder.state_dict(),
        writer=writer,
        label="RepKPU encoder",
    )
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.load_state_dict(encoder_state, strict=False)

    model_state = torch.load(args.ckpt, map_location="cpu")
    model_state = _adapt_model_state_dict_for_sparse_input(model, model_state, writer=writer)
    model_state = _adapt_state_dict_to_model_shapes(
        model_state,
        model.state_dict(),
        writer=writer,
        label="Model checkpoint",
    )
    model.load_state_dict(model_state, strict=False)
    writer.write("Model checkpoint loaded and shape-adapted on CPU before CUDA transfer.")
    del ckpt, encoder_state, model_state
    gc.collect()

    if args.cpu == False:
        print(f"Using GPU for testing")
        model = model.cuda()
        torch.cuda.empty_cache()
    print(f"Model Setted: {datetime.datetime.now()}\n")
    
    # 損失計算（推論時にどうなるのか計算するため）
    loss = Loss(args, file_day+"-"+file_time, writer)

    # テスト開始
    st = time.time()
    print(f"=== Start Testing ===")
    writer.write(f"=== Start Testing ===")
    test(model, loss, args, writer)
    en = time.time()

    FinishDate = datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S')

    # テスト時間の記録
    print(f"Testing time: {en - st}")
    print(f"Date of finishing testing: {FinishDate}")
    writer.write(f"Testing time: {en - st}")
    writer.write(f"Date of finishing testing: {FinishDate}")
    writer.close()
