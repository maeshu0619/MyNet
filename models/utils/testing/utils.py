import os
import shutil
import hashlib
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import numpy as np

from models.utils.pointcloud.utils_repkpu import *
from models.utils.pointcloud.octree_subtree import (
    assign_octree_subtree_keys,
    build_octree_subtree_reference,
    build_subtree_index_map,
)
from models.utils.patching.patch import (
    build_patch_info,
    denormalize_patch_output,
    merge_patch_outputs,
)



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
    codec_eval_dir = Path(
        os.path.abspath(
            os.path.expanduser(getattr(args, "codec_eval_dir", args.save_ply_dir))
        )
    )
    run_dir = base_dir / "runs" / f"{args.date}_{args.time}_{Path(args.ckpt).stem}"
    filename = f"{step_idx:04d}_{getattr(args, 'method_name', 'Mine')}.ply"
    return {
        "primary_dir": base_dir,
        "primary_path": base_dir / filename,
        "run_dir": run_dir,
        "run_path": run_dir / filename,
        "codec_eval_dir": codec_eval_dir,
        "codec_eval_path": codec_eval_dir / filename,
    }


def _copy_file_if_needed(src_path, dst_path):
    src = Path(src_path)
    dst = Path(dst_path)
    if src.resolve() == dst.resolve():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _cleanup_codec_eval_slot(codec_eval_dir, step_idx, keep_path):
    keep = Path(keep_path).resolve()
    slot_prefix = f"{int(step_idx):04d}_"
    for path in Path(codec_eval_dir).glob(f"{slot_prefix}*.ply"):
        if path.resolve() == keep:
            continue
        try:
            path.unlink()
        except OSError:
            pass


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
    for key in ("add_ratio", "drop_ratio", "keep_ratio", "add_candidate_ratio"):
        merged[key] = float(np.mean([float(chunk.get(key, 0.0)) for chunk in chunks]))
    for key in ("add_count", "add_effective_count"):
        merged[key] = int(sum(int(chunk.get(key, 0)) for chunk in chunks))
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
        f"add_ratio={float(structure_debug.get('add_ratio', 0.0)):.6f}, "
        f"add_count={int(structure_debug.get('add_count', 0))}, "
        f"add_effective_count={int(structure_debug.get('add_effective_count', 0))}, "
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
            ("codec_eval_path", step_record["codec_eval_path"]),
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
    expected_keep = None
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
        "expected_keep": expected_keep,
        "above_threshold": int((flat_w >= drop_th).sum().item()),
        "weight_min": float(flat_w.min().detach().cpu()) if total_count > 0 else None,
        "weight_mean": float(flat_w.mean().detach().cpu()) if total_count > 0 else None,
        "weight_max": float(flat_w.max().detach().cpu()) if total_count > 0 else None,
    }


def _summarize_hardening_counts(input_points, pre_output_points, keep_mask):
    input_points = int(input_points)
    pre_output_points = int(pre_output_points)
    candidate_added = max(pre_output_points - input_points, 0)
    if keep_mask is None or int(keep_mask.numel()) == 0:
        kept_original = min(input_points, pre_output_points)
        kept_added = candidate_added
    else:
        flat_keep = keep_mask.reshape(-1)
        original_end = min(input_points, int(flat_keep.numel()), pre_output_points)
        added_start = min(input_points, int(flat_keep.numel()), pre_output_points)
        added_end = min(input_points + candidate_added, int(flat_keep.numel()), pre_output_points)
        kept_original = int(flat_keep[:original_end].sum().item()) if original_end > 0 else 0
        kept_added = int(flat_keep[added_start:added_end].sum().item()) if added_end > added_start else 0
    deleted_original = max(input_points - kept_original, 0)
    deleted_added = max(candidate_added - kept_added, 0)
    return {
        "candidate_added": int(candidate_added),
        "kept_original": int(kept_original),
        "deleted_original": int(deleted_original),
        "kept_added": int(kept_added),
        "deleted_added": int(deleted_added),
        "net_change_after_hardening": int(kept_added - deleted_original),
    }


def _format_optional_float(value, digits=6):
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


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


def _should_force_actual_codec_eval(args):
    backend = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    return backend.endswith("_surrogate")


def _should_run_actual_codec_eval(args, step_idx=None):
    if bool(getattr(args, "skip_actual_codec", True)):
        return False
    interval = max(int(getattr(args, "codec_eval_interval", 0)), 0)
    if interval <= 0:
        return False
    if step_idx is None:
        return True
    return (int(step_idx) + 1) % interval == 0


def _compression_eval_refresh_mode(args, step_idx=None):
    if not _should_run_actual_codec_eval(args, step_idx=step_idx):
        return False
    return "always" if _should_force_actual_codec_eval(args) else True


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
