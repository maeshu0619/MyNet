import csv
import math
import os

from .metric_columns import CASE_DEBUG_COLUMNS
from .scalar_utils import case_float, case_int

def init_case_debug_csv(args, plot, writer):
    if not bool(getattr(args, "save_good_bad_cases", False)):
        return None
    os.makedirs(plot.save_dir, exist_ok=True)
    path = os.path.join(plot.save_dir, f"{args.time}_good_bad_cases.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CASE_DEBUG_COLUMNS).writeheader()
    writer.write(f"GoodBadCaseDebug: enabled path={path}")
    return path


def maybe_record_case_debug(
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
    actual_delta = case_float(comp_debug.get("actual_total_bit_percent", comp_debug.get("total_bit", float("nan"))), float("nan"))
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
        "surrogate_delta": case_float(comp_debug.get("rate_proxy_delta", comp_debug.get("surrogate_pred_bit", 0.0))),
        "surrogate_abs_error": case_float(comp_debug.get("surrogate_abs_bit_error", 0.0)),
        "surrogate_signed_error": case_float(comp_debug.get("surrogate_signed_bit_error", 0.0)),
        "actual_bits_before": case_float(comp_debug.get("gt_actual_bit", float("nan")), float("nan")),
        "actual_bits_after": case_float(comp_debug.get("gen_actual_bit", float("nan")), float("nan")),
        "point_count_before": case_int(comp_debug.get("gt_points", edit_stats.get("input_points", 0) if edit_stats else 0)),
        "point_count_after": case_int(comp_debug.get("gen_points", edit_stats.get("output_points", 0) if edit_stats else 0)),
        "unique_coord_before": case_int(comp_debug.get("gt_unique_coord_count", 0)),
        "unique_coord_after": case_int(comp_debug.get("gen_unique_coord_count", 0)),
        "active_coord_before": case_int(comp_debug.get("sparsepcgc_before_active_coords", 0)),
        "active_coord_after": case_int(comp_debug.get("sparsepcgc_after_active_coords", 0)),
        "octree_node_before": case_float(comp_debug.get("gt_octree_node", 0.0)),
        "octree_node_after": case_float(comp_debug.get("gen_octree_node", 0.0)),
        "single_before": case_float(comp_debug.get("gt_octree_single", 0.0)),
        "single_after": case_float(comp_debug.get("gen_octree_single", 0.0)),
        "add_points": case_int(edit_stats.get("added_points", 0) if edit_stats else 0),
        "delete_points": case_int(edit_stats.get("deleted_points", 0) if edit_stats else 0),
        "adjust_points": case_int(edit_stats.get("adjusted_points", 0) if edit_stats else 0),
        "preserve_ratio": case_float(structure_debug.get("preserve_ratio", 0.0)),
        "same_voxel_adjust": case_int(structure_debug.get("same_voxel_adjust_count", 0)),
        "different_voxel_move": case_int(structure_debug.get("moved_different_voxel_count", 0)),
        "move_source_emptied": case_int(structure_debug.get("move_source_emptied_voxel_count", 0)),
        "move_target_new": case_int(structure_debug.get("move_target_new_voxel_count", 0)),
        "move_source_not_emptied": case_int(structure_debug.get("move_source_not_emptied_count", 0)),
        "shape_loss": case_float(L_geom),
        "compression_loss": case_float(L_com),
        "actuator_loss": case_float(L_actuator),
        "total_loss": case_float(L),
        "teacher_refresh": bool(comp_debug.get("teacher_refresh", False)),
        "teacher_target_age": case_int(comp_debug.get("teacher_target_age", 0)),
    }
    with open(case_debug_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CASE_DEBUG_COLUMNS).writerow(row)
    case_debug_counts[case_type] = int(case_debug_counts.get(case_type, 0)) + 1
    writer.write(
        "GoodBadCaseDebug: "
        f"type={case_type}, step={int(global_step) + 1}, sample={row['sample_name']}, "
        f"actual_delta={actual_delta:.6f}, path={case_debug_path}"
    )
