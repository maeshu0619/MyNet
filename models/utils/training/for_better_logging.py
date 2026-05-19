import datetime
import json
import math
import os

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in training, but logging should stay optional.
    torch = None

from .scalar_utils import case_float, case_int
from .sparsepcgc_controls import sparsepcgc_add_control_status


FOR_BETTER_STATES = {}


def for_better_scalar(value, default=None):
    if value is None:
        return default
    if torch is not None and torch.is_tensor(value):
        try:
            if value.numel() == 0:
                return default
            value = value.detach().reshape(-1)[0].float().cpu().item()
        except Exception:
            return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def for_better_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def for_better_param_norm(module_or_loss):
    module = getattr(module_or_loss, "compression_surrogate", module_or_loss)
    if module is None or torch is None:
        return None
    total_sq = 0.0
    try:
        params = module.parameters()
    except Exception:
        return None
    with torch.no_grad():
        for param in params:
            if param is None:
                continue
            try:
                data = param.detach().float()
                if not torch.isfinite(data).all():
                    return None
                total_sq += float(data.pow(2).sum().cpu())
            except Exception:
                return None
    return math.sqrt(total_sq) if total_sq >= 0.0 else None


def for_better_grad_norm(model):
    if model is None or torch is None:
        return None
    total_sq = 0.0
    found = False
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        try:
            grad_data = grad.detach().float()
            if not torch.isfinite(grad_data).all():
                return None
            total_sq += float(grad_data.pow(2).sum().cpu())
            found = True
        except Exception:
            return None
    return math.sqrt(total_sq) if found and total_sq >= 0.0 else None


def for_better_lrs(optimizer):
    if optimizer is None:
        return []
    return [float(group.get("lr", 0.0)) for group in getattr(optimizer, "param_groups", [])]


def for_better_clean(value):
    if torch is not None and torch.is_tensor(value):
        scalar = for_better_scalar(value, None)
        if scalar is not None:
            return scalar
        try:
            return {"shape": list(value.shape), "finite": bool(torch.isfinite(value.detach()).all().item())}
        except Exception:
            return str(type(value).__name__)
    if isinstance(value, dict):
        return {str(k): for_better_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [for_better_clean(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def format_for_better_line(kind, fields):
    payload = {"kind": kind, "time": datetime.datetime.now().isoformat(timespec="seconds")}
    payload.update({str(k): for_better_clean(v) for k, v in fields.items()})
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def write_for_better_log(path, message):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(str(message).rstrip() + "\n")
    except Exception:
        return


def log_for_better_event(path, event, **fields):
    if not path:
        return
    event_fields = {"event": str(event)}
    event_fields.update(fields)
    write_for_better_log(path, format_for_better_line("event", event_fields))


def init_for_better_logger(args, plot, writer=None):
    if not for_better_bool(getattr(args, "for_better_log", True), True):
        return None
    save_dir = getattr(plot, "save_dir", None) or getattr(args, "out_path", ".")
    args_time = str(getattr(args, "time", "") or "").strip()
    filename = f"{args_time}_ForBetter.txt" if args_time else "ForBetter.txt"
    path = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(format_for_better_line("header", {"message": "ForBetter diagnostic log"}) + "\n")
    except Exception:
        return None
    setattr(args, "_for_better_path", path)
    setup_fields = {
        "args_time": getattr(args, "time", None),
        "compress": getattr(args, "compress", None),
        "compression_loss_backend": getattr(args, "compression_loss_backend", None),
        "loss_mode": getattr(args, "loss_mode", None),
        "surrogate_pretrain_steps": getattr(args, "surrogate_pretrain_steps", None),
        "surrogate_pretrain_mode": getattr(args, "surrogate_pretrain_mode", None),
        "surrogate_pretrain_subtree_teacher_type": getattr(args, "surrogate_pretrain_subtree_teacher_type", None),
        "surrogate_pretrain_store_local_proxy_replay": getattr(args, "surrogate_pretrain_store_local_proxy_replay", None),
        "surrogate_pretrain_skip_min_points_miss": getattr(args, "surrogate_pretrain_skip_min_points_miss", None),
        "surrogate_update_during_training": getattr(args, "surrogate_update_during_training", None),
        "surrogate_update_interval": getattr(args, "surrogate_update_interval", None),
        "surrogate_auto_freeze": getattr(args, "surrogate_auto_freeze", None),
        "surrogate_freeze_abs_error": getattr(args, "surrogate_freeze_abs_error", None),
        "compression_surrogate_refresh_interval": getattr(args, "compression_surrogate_refresh_interval", None),
        "compression_surrogate_reuse_last_target": getattr(args, "compression_surrogate_reuse_last_target", None),
        "compression_surrogate_aux_in_objective": getattr(args, "compression_surrogate_aux_in_objective", None),
        "compression_good_step_boost": getattr(args, "compression_good_step_boost", None),
        "surrogate_pretrain_allow_stale_target": getattr(args, "surrogate_pretrain_allow_stale_target", None),
        "train_patch_subset_enable": getattr(args, "train_patch_subset_enable", None),
        "train_subtree_level_min": getattr(args, "train_subtree_level_min", None),
        "train_subtree_level_max": getattr(args, "train_subtree_level_max", None),
        "train_subtree_randomize_level": getattr(args, "train_subtree_randomize_level", None),
        "train_subtree_anchor_on_min_points_miss": getattr(args, "train_subtree_anchor_on_min_points_miss", None),
        "train_subtree_level_curriculum": getattr(args, "train_subtree_level_curriculum", None),
        "train_subtree_level_curriculum_direction": getattr(args, "train_subtree_level_curriculum_direction", None),
        "train_subtree_depth_percent_curriculum": getattr(args, "train_subtree_depth_percent_curriculum", None),
        "train_subtree_depth_percent_start_min": getattr(args, "train_subtree_depth_percent_start_min", None),
        "train_subtree_depth_percent_start_max": getattr(args, "train_subtree_depth_percent_start_max", None),
        "train_subtree_depth_percent_end_min": getattr(args, "train_subtree_depth_percent_end_min", None),
        "train_subtree_depth_percent_end_max": getattr(args, "train_subtree_depth_percent_end_max", None),
        "w_com": getattr(args, "w_com", None),
        "w_geom": getattr(args, "w_geom", None),
        "com_bit": getattr(args, "com_bit", None),
        "com_sin": getattr(args, "com_sin", None),
        "com_node": getattr(args, "com_node", None),
        "com_sparsepcgc": getattr(args, "com_sparsepcgc", None),
        "sparsepcgc_aux_backprop": getattr(args, "sparsepcgc_aux_backprop", None),
        "sparsepcgc_add_status": sparsepcgc_add_control_status(args),
    }
    log_for_better_event(path, "setup", **setup_fields)
    if str(getattr(args, "surrogate_pretrain_subtree_teacher_type", "")).strip().lower() == "local_proxy":
        log_for_better_event(
            path,
            "surrogate_local_proxy_warning",
            message="local_proxy is a differentiable local proxy target, not actual SparsePCGC bit.",
        )
    if writer is not None and hasattr(writer, "write"):
        writer.write(f"ForBetterLog: {path}")
    return path


def for_better_state(path):
    return FOR_BETTER_STATES.setdefault(path, {"series": {}, "last": {}})


def detect_for_better_spikes(path, args, metrics):
    state = for_better_state(path)
    ratio = max(float(getattr(args, "for_better_spike_ratio", 2.0)), 1.0)
    window = max(int(getattr(args, "for_better_spike_window", 20)), 2)
    events = []
    for name, value in metrics.items():
        value = for_better_scalar(value, None)
        if value is None:
            events.append((name, "nonfinite", None, None))
            continue
        series = state["series"].setdefault(name, [])
        prev = series[-1] if series else None
        baseline = None
        if series:
            recent = [abs(v) for v in series[-window:] if v is not None and math.isfinite(v)]
            if recent:
                baseline = sum(recent) / float(len(recent))
        if baseline is not None and baseline > 1e-9 and abs(value) >= ratio * baseline:
            events.append((name, "rolling_ratio", value, baseline))
        elif prev is not None and abs(prev) > 1e-9 and abs(value) >= ratio * abs(prev):
            events.append((name, "prev_ratio", value, prev))
        series.append(value)
        if len(series) > window * 4:
            del series[:-window * 2]
    return events


def log_for_better_step(
    path,
    *,
    args,
    model=None,
    loss_obj=None,
    optimizer=None,
    global_step=0,
    episode=0,
    epoch=0,
    step=0,
    stage=None,
    stage_factors=None,
    compression_row=None,
    operation_row=None,
    comp_debug=None,
    structure_debug=None,
    edit_stats=None,
    subtree_meta=None,
    loss_values=None,
    step_completed=None,
    total_loss_finite=None,
    amp_info=None,
    timing=None,
):
    if not path:
        return
    step_number = int(global_step) + 1
    interval = max(int(getattr(args, "for_better_log_interval", 1)), 1)
    compression_row = compression_row or {}
    operation_row = operation_row or {}
    comp_debug = comp_debug or {}
    structure_debug = structure_debug or {}
    edit_stats = edit_stats or {}
    subtree_meta = subtree_meta or {}
    loss_values = loss_values or {}
    amp_info = amp_info or {}
    timing = timing or {}

    lr_values = for_better_lrs(optimizer)
    fields = {
        "global_step": step_number,
        "episode": int(episode) + 1,
        "epoch": int(epoch) + 1,
        "local_step": int(step) + 1,
        "stage": stage,
        "stage_factors": stage_factors,
        "current_lr": lr_values,
        "subtree": subtree_meta,
        "L_total": for_better_scalar(loss_values.get("L")),
        "L_geom": for_better_scalar(loss_values.get("L_geom")),
        "L_com": for_better_scalar(loss_values.get("L_com")),
        "L_com_objective": for_better_scalar(loss_values.get("L_com_objective")),
        "L_attr": for_better_scalar(loss_values.get("L_attr")),
        "L_policy": for_better_scalar(loss_values.get("L_policy")),
        "L_actuator": for_better_scalar(loss_values.get("L_actuator")),
        "loss_bit": for_better_scalar(loss_values.get("loss_bit")),
        "loss_single": for_better_scalar(loss_values.get("loss_single")),
        "loss_nodes": for_better_scalar(loss_values.get("loss_nodes")),
        "sparsepcgc_aux_raw": compression_row.get("sparsepcgc_aux_raw"),
        "sparsepcgc_aux_weighted": compression_row.get("sparsepcgc_aux_weighted"),
        "lcom_without_sparsepcgc_aux": compression_row.get("lcom_without_sparsepcgc_aux"),
        "lcom_with_sparsepcgc_aux": compression_row.get("lcom_with_sparsepcgc_aux"),
        "surrogate_pred_bit_percent": compression_row.get("surrogate_pred_bit_percent"),
        "surrogate_target_bit_percent": compression_row.get("surrogate_target_bit_percent"),
        "surrogate_target_train_bit_percent": comp_debug.get("surrogate_target_train_bit"),
        "surrogate_target_raw_bit_percent": comp_debug.get("surrogate_target_raw_bit"),
        "surrogate_teacher_source": comp_debug.get("surrogate_teacher_source"),
        "surrogate_teacher_is_actual": comp_debug.get("surrogate_teacher_is_actual"),
        "surrogate_teacher_is_local_proxy": comp_debug.get("surrogate_teacher_is_local_proxy"),
        "surrogate_target_clamped": comp_debug.get("surrogate_target_clamped"),
        "surrogate_pred_clip": comp_debug.get("surrogate_pred_clip"),
        "actual_total_bit_percent_fresh": compression_row.get("actual_total_bit_percent_fresh"),
        "actual_total_bit_percent_cached": compression_row.get("actual_total_bit_percent_cached"),
        "proxy_delta_percent": compression_row.get("proxy_delta_percent"),
        "corr_surrogate_actual": compression_row.get("corr_surrogate_actual"),
        "corr_lcom_actual": compression_row.get("corr_lcom_actual"),
        "corr_sparsepcgc_aux_actual": compression_row.get("corr_sparsepcgc_aux_actual"),
        "sign_match_surrogate_actual": compression_row.get("sign_match_surrogate_actual"),
        "sign_match_lcom_actual": compression_row.get("sign_match_lcom_actual"),
        "sign_match_sparsepcgc_aux_actual": compression_row.get("sign_match_sparsepcgc_aux_actual"),
        "add_enabled": operation_row.get("add_enabled"),
        "delete_enabled": operation_row.get("prune_enabled"),
        "adjust_enabled": operation_row.get("disp_enabled"),
        "add_prob_mean": operation_row.get("add_prob_mean"),
        "drop_prob_mean": operation_row.get("drop_prob_mean"),
        "added_ratio_percent": operation_row.get("added_ratio_percent"),
        "deleted_ratio_percent": operation_row.get("deleted_ratio_percent"),
        "adjusted_ratio_percent": operation_row.get("adjusted_ratio_percent"),
        "add_effective_count": operation_row.get("add_effective_count"),
        "active_coord_before": compression_row.get("active_coord_before"),
        "active_coord_after": compression_row.get("active_coord_after"),
        "active_coord_delta": compression_row.get("active_coord_delta"),
        "unique_coord_before": compression_row.get("unique_coord_before"),
        "unique_coord_after": compression_row.get("unique_coord_after"),
        "unique_coord_delta": case_int(compression_row.get("unique_coord_after", 0)) - case_int(compression_row.get("unique_coord_before", 0)),
        "teacher_mode": comp_debug.get("teacher_mode", compression_row.get("teacher_cache_hit")),
        "teacher_cache_hit": comp_debug.get("teacher_cache_hit"),
        "teacher_target_age": comp_debug.get("teacher_target_age"),
        "surrogate_replay_size": comp_debug.get("surrogate_replay_size"),
        "surrogate_replay_sample_count": comp_debug.get("surrogate_replay_sample_count"),
        "gradient_norm": for_better_grad_norm(model),
        "surrogate_param_norm": for_better_param_norm(loss_obj),
        "nonfinite_skip": not for_better_bool(total_loss_finite, True),
        "amp": amp_info,
        "optimizer_step": for_better_bool(step_completed, False),
        "timing": timing,
        "edit_stats": edit_stats,
        "sparsepcgc_add_status": sparsepcgc_add_control_status(args),
    }

    state = for_better_state(path)
    last = state["last"]
    depth_range = None
    if isinstance(subtree_meta, dict):
        depth_range = (subtree_meta.get("min_depth"), subtree_meta.get("max_depth"))
    if last.get("stage") != stage:
        log_for_better_event(path, "stage_changed", global_step=step_number, previous=last.get("stage"), current=stage)
    if depth_range != last.get("depth_range") and any(v is not None for v in (depth_range or [])):
        log_for_better_event(path, "subtree_depth_range_changed", global_step=step_number, previous=last.get("depth_range"), current=depth_range)
    if lr_values != last.get("lr_values") and lr_values:
        log_for_better_event(path, "learning_rate_changed", global_step=step_number, previous=last.get("lr_values"), current=lr_values)
    op_state = (fields["add_enabled"], fields["delete_enabled"], fields["adjust_enabled"])
    if op_state != last.get("operation_enabled_state"):
        log_for_better_event(path, "operation_enabled_state", global_step=step_number, add_enabled=op_state[0], delete_enabled=op_state[1], adjust_enabled=op_state[2])
    last["stage"] = stage
    last["depth_range"] = depth_range
    last["lr_values"] = lr_values
    last["operation_enabled_state"] = op_state

    spike_metrics = {
        "L_total": fields["L_total"],
        "L_com": fields["L_com"],
        "actual_total_bit_percent_fresh": fields["actual_total_bit_percent_fresh"],
        "surrogate_train_loss": comp_debug.get("surrogate_train_loss"),
    }
    for metric, reason, value, baseline in detect_for_better_spikes(path, args, spike_metrics):
        log_for_better_event(path, "loss_or_metric_spike", global_step=step_number, metric=metric, reason=reason, value=value, baseline=baseline)

    if (step_number == 1) or (step_number % interval == 0) or fields["nonfinite_skip"]:
        write_for_better_log(path, format_for_better_line("step", fields))


def log_for_better_pretrain_step(path, row, comp_debug=None, extra=None):
    if not path:
        return
    comp_debug = comp_debug or {}
    extra = extra or {}
    fields = {
        "row": row,
        "teacher_mode": row.get("teacher_mode"),
        "actual_value_source": comp_debug.get("actual_value_source"),
        "actual_value_source_detail": comp_debug.get("actual_value_source_detail"),
        "surrogate_teacher_source": comp_debug.get("surrogate_teacher_source"),
        "surrogate_teacher_is_actual": comp_debug.get("surrogate_teacher_is_actual"),
        "surrogate_teacher_is_local_proxy": comp_debug.get("surrogate_teacher_is_local_proxy"),
        "surrogate_target_raw_bit": comp_debug.get("surrogate_target_raw_bit"),
        "surrogate_target_train_bit": comp_debug.get("surrogate_target_train_bit"),
        "surrogate_target_clamped": comp_debug.get("surrogate_target_clamped"),
        "surrogate_target_scale": comp_debug.get("surrogate_target_scale"),
        "surrogate_pred_clip": comp_debug.get("surrogate_pred_clip"),
        "surrogate_local_proxy_replay_stored": comp_debug.get("surrogate_local_proxy_replay_stored"),
        "target_cache_hit": comp_debug.get("teacher_cache_hit"),
        "target_age": comp_debug.get("teacher_target_age"),
        "replay_size": comp_debug.get("surrogate_replay_size"),
        "replay_sample_count": comp_debug.get("surrogate_replay_sample_count"),
        "reset_flag": comp_debug.get("surrogate_reset", False),
        "nonfinite": not for_better_bool(comp_debug.get("inputs_finite", True), True),
        "extra": extra,
    }
    write_for_better_log(path, format_for_better_line("pretrain_step", fields))


def log_for_better_pretrain_complete(path, **fields):
    log_for_better_event(path, "surrogate_pretrain_complete", **fields)


def log_for_better_episode(
    path,
    *,
    args,
    episode,
    stage,
    checkpoint_metrics=None,
    compression_episode_metrics=None,
    operation_episode_metrics=None,
    best_trackers=None,
    model_path=None,
):
    if not path:
        return
    fields = {
        "episode": int(episode) + 1,
        "stage": stage,
        "checkpoint_metrics": checkpoint_metrics or {},
        "compression_episode_metrics": compression_episode_metrics or {},
        "operation_episode_metrics": operation_episode_metrics or {},
        "best_trackers": best_trackers or {},
        "model_path": model_path,
        "sparsepcgc_add_status": sparsepcgc_add_control_status(args),
    }
    write_for_better_log(path, format_for_better_line("episode", fields))
