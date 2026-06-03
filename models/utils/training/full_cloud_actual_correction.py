import math
import torch


def _zero(reference=None):
    if torch.is_tensor(reference):
        return reference.new_zeros(())
    return torch.zeros((), dtype=torch.float32)


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            out = float(value.detach().float().mean().cpu())
        else:
            out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _first_number(debug, keys, default=None):
    if not isinstance(debug, dict):
        return default
    for key in keys:
        if key in debug:
            value = _safe_float(debug.get(key), None)
            if value is not None:
                return value
    return default


def _ema(prev, value, alpha):
    if value is None:
        return prev
    if prev is None:
        return float(value)
    return float(alpha) * float(prev) + (1.0 - float(alpha)) * float(value)


def _state_get_float(state, key, default=None):
    if not isinstance(state, dict):
        return default
    return _safe_float(state.get(key), default)


def _count_from_state(actuator_voxel_state, keys):
    return _first_number(actuator_voxel_state, keys, 0.0)


def _soft_tensor_from_args(args, keys):
    soft_terms = getattr(args, "_last_actuator_soft_terms", None)
    if not isinstance(soft_terms, dict):
        return None
    for key in keys:
        value = soft_terms.get(key, None)
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.reshape(())
            return value.float().mean()
    return None


def update_full_cloud_actual_correction_state(
    args,
    state,
    full_cloud_debug=None,
    subtree_debug=None,
    full_context_debug=None,
    actuator_voxel_state=None,
    reference=None,
    global_step=None,
):
    """
    periodic full cloud actualとsubtree/proxy/full-context deltaのgapをEMAで記録する。
    full cloud actualが存在しないstepではstateを更新しない。
    """
    if state is None or not isinstance(state, dict):
        state = {}

    debug = {
        "full_cloud_corr_update_used": False,
        "full_cloud_corr_update_reason": "not_initialized",
    }

    if not bool(getattr(args, "full_cloud_actual_correction", True)):
        debug["full_cloud_corr_update_reason"] = "disabled_by_args"
        return state, debug

    full_delta = _first_number(
        full_cloud_debug,
        (
            "full_cloud_actual_delta",
            "full_cloud_actual_percent",
            "actual_total_bit_percent",
            "actual_bit_percent",
        ),
        None,
    )

    if full_delta is None:
        debug["full_cloud_corr_update_reason"] = "missing_full_cloud_actual_delta"
        return state, debug

    subtree_delta = _first_number(
        subtree_debug,
        (
            "subtree_local_actual_delta",
            "subtree_teacher_percent",
            "actual_total_bit_percent",
            "actual_bit_percent",
        ),
        None,
    )

    context_delta = _first_number(
        full_context_debug,
        (
            "full_context_subtree_delta_value",
            "cp_full_context_subtree_delta",
        ),
        None,
    )

    proxy_delta = _first_number(
        subtree_debug,
        (
            "surrogate_pred_percent",
            "surrogate_prediction",
            "proxy_bit_percent",
            "predicted_bit_percent",
            "L_com",
        ),
        None,
    )

    full_vs_subtree_gap = None
    if subtree_delta is not None:
        full_vs_subtree_gap = float(full_delta) - float(subtree_delta)

    full_vs_context_gap = None
    if context_delta is not None:
        full_vs_context_gap = float(full_delta) - float(context_delta)

    full_vs_proxy_gap = None
    if proxy_delta is not None:
        full_vs_proxy_gap = float(full_delta) - float(proxy_delta)

    alpha = min(max(float(getattr(args, "full_cloud_actual_correction_ema", 0.90)), 0.0), 0.999)

    state["ema_full_vs_subtree_gap"] = _ema(
        _state_get_float(state, "ema_full_vs_subtree_gap", None),
        full_vs_subtree_gap,
        alpha,
    )
    state["ema_full_vs_context_gap"] = _ema(
        _state_get_float(state, "ema_full_vs_context_gap", None),
        full_vs_context_gap,
        alpha,
    )
    state["ema_full_vs_proxy_gap"] = _ema(
        _state_get_float(state, "ema_full_vs_proxy_gap", None),
        full_vs_proxy_gap,
        alpha,
    )
    state["ema_full_actual_delta"] = _ema(
        _state_get_float(state, "ema_full_actual_delta", None),
        full_delta,
        alpha,
    )

    state["last_full_actual_delta"] = float(full_delta)
    state["last_subtree_actual_delta"] = subtree_delta
    state["last_full_context_delta"] = context_delta
    state["last_subtree_proxy_delta"] = proxy_delta
    state["last_update_step"] = int(global_step) if global_step is not None else -1

    state["last_full_actual_bit"] = _first_number(
        full_cloud_debug,
        ("actual_total_bit", "actual_bit", "bit", "full_cloud_actual_bit"),
        None,
    )
    state["last_full_actual_bpp"] = _first_number(
        full_cloud_debug,
        ("actual_bpp", "bpp", "full_cloud_actual_bpp"),
        None,
    )
    state["last_full_cloud_occupancy_nll_delta"] = _first_number(
        full_cloud_debug,
        ("actual_occupancy_nll_delta", "exact_occ_nll_delta", "full_cloud_occupancy_nll_delta"),
        None,
    )
    state["last_full_cloud_node_delta"] = _first_number(
        full_cloud_debug,
        ("actual_node_percent", "node_percent", "full_cloud_node_delta"),
        None,
    )
    state["last_full_cloud_single_child_delta"] = _first_number(
        full_cloud_debug,
        ("actual_single_percent", "single_percent", "full_cloud_single_child_delta"),
        None,
    )

    state["last_move_count"] = _count_from_state(
        actuator_voxel_state,
        ("voxel_edit_move_count", "hard_move_count", "adjusted_point_count"),
    )
    state["last_add_count"] = _count_from_state(
        actuator_voxel_state,
        ("voxel_edit_add_count", "added_point_count", "add_count"),
    )
    state["last_drop_count"] = _count_from_state(
        actuator_voxel_state,
        ("voxel_edit_drop_count", "hard_drop_count", "selected_drop_count_hard"),
    )
    state["last_same_voxel_move_rejected"] = _count_from_state(
        actuator_voxel_state,
        ("voxel_edit_same_voxel_move_rejected",),
    )
    state["last_existing_target_rejected"] = _count_from_state(
        actuator_voxel_state,
        ("voxel_edit_existing_target_rejected",),
    )
    state["last_duplicate_target_rejected"] = _count_from_state(
        actuator_voxel_state,
        ("voxel_edit_duplicate_target_rejected",),
    )
    state["last_child_slot_rejected"] = _count_from_state(
        actuator_voxel_state,
        ("voxel_edit_child_slot_rejected",),
    )
    state["last_empty_target_rejected"] = _count_from_state(
        actuator_voxel_state,
        ("voxel_edit_empty_target_rejected",),
    )

    debug.update(
        {
            "full_cloud_corr_update_used": True,
            "full_cloud_corr_update_reason": "updated",
            "full_cloud_corr_last_full_actual_delta": state["last_full_actual_delta"],
            "full_cloud_corr_last_subtree_actual_delta": state["last_subtree_actual_delta"],
            "full_cloud_corr_last_full_context_delta": state["last_full_context_delta"],
            "full_cloud_corr_last_subtree_proxy_delta": state["last_subtree_proxy_delta"],
            "full_cloud_corr_ema_full_vs_subtree_gap": state["ema_full_vs_subtree_gap"],
            "full_cloud_corr_ema_full_vs_context_gap": state["ema_full_vs_context_gap"],
            "full_cloud_corr_ema_full_vs_proxy_gap": state["ema_full_vs_proxy_gap"],
            "full_cloud_corr_ema_full_actual_delta": state["ema_full_actual_delta"],
            "full_cloud_corr_last_update_step": state["last_update_step"],
            "full_cloud_corr_move_count": state["last_move_count"],
            "full_cloud_corr_add_count": state["last_add_count"],
            "full_cloud_corr_drop_count": state["last_drop_count"],
        }
    )

    return state, debug


def build_full_cloud_actual_correction_loss(
    args,
    correction_state=None,
    actuator_voxel_state=None,
    reference=None,
    global_step=None,
):
    """
    full cloud actual悪化時に、Move/Add/Drop量へ弱い補正ペナルティを作る。
    デフォルトではlossへ足さない。
    """
    zero = _zero(reference)
    debug = {
        "full_cloud_corr_loss_used": False,
        "full_cloud_corr_loss_reason": "not_initialized",
        "full_cloud_corr_loss_value": 0.0,
        "full_cloud_corr_loss_enabled": bool(getattr(args, "full_cloud_actual_correction_loss_enable", False)),
    }

    if not bool(getattr(args, "full_cloud_actual_correction", True)):
        debug["full_cloud_corr_loss_reason"] = "disabled_by_args"
        return zero, debug

    if correction_state is None or not isinstance(correction_state, dict):
        debug["full_cloud_corr_loss_reason"] = "missing_correction_state"
        return zero, debug

    warmup = int(getattr(args, "full_cloud_actual_correction_warmup_steps", 100))
    if global_step is not None and int(global_step) < warmup:
        debug["full_cloud_corr_loss_reason"] = "warmup"
        return zero, debug

    full_delta = _state_get_float(correction_state, "ema_full_actual_delta", None)
    if full_delta is None:
        full_delta = _state_get_float(correction_state, "last_full_actual_delta", None)

    if full_delta is None:
        debug["full_cloud_corr_loss_reason"] = "missing_full_actual_delta"
        return zero, debug

    if full_delta <= 0.0:
        debug["full_cloud_corr_loss_reason"] = "full_cloud_actual_not_worse"
        return zero, debug

    clip_value = max(float(getattr(args, "full_cloud_actual_correction_clip", 5.0)), 0.0)
    severity_value = max(float(full_delta), 0.0)
    if clip_value > 0.0:
        severity_value = min(severity_value, clip_value)
    severity = zero + zero.new_tensor(float(severity_value))

    move_term = _soft_tensor_from_args(args, ("learned_move_ratio", "move_ratio_soft", "soft_move_voxel_sum"))
    add_term = _soft_tensor_from_args(args, ("learned_add_ratio", "add_ratio_soft", "soft_add_sum"))
    drop_term = _soft_tensor_from_args(args, ("soft_drop_mass", "drop_prob_proxy", "drop_prob_mean", "learned_drop_prob"))

    penalty = zero

    if bool(getattr(args, "full_cloud_actual_correction_penalize_move", True)):
        if move_term is not None:
            penalty = penalty + move_term.to(device=zero.device, dtype=zero.dtype).reshape(())

    if bool(getattr(args, "full_cloud_actual_correction_penalize_add", True)):
        if add_term is not None:
            penalty = penalty + add_term.to(device=zero.device, dtype=zero.dtype).reshape(())

    if bool(getattr(args, "full_cloud_actual_correction_penalize_drop", False)):
        if drop_term is not None:
            penalty = penalty + drop_term.to(device=zero.device, dtype=zero.dtype).reshape(())

    if not torch.is_tensor(penalty) or penalty.numel() == 0:
        debug["full_cloud_corr_loss_reason"] = "missing_soft_operation_terms"
        return zero, debug

    correction_loss = severity.detach() * penalty

    debug.update(
        {
            "full_cloud_corr_loss_used": True,
            "full_cloud_corr_loss_reason": "built",
            "full_cloud_corr_loss_value": _safe_float(correction_loss, 0.0),
            "full_cloud_corr_loss_severity": float(severity_value),
            "full_cloud_corr_loss_enabled": bool(getattr(args, "full_cloud_actual_correction_loss_enable", False)),
            "full_cloud_corr_ema_full_vs_subtree_gap": _state_get_float(correction_state, "ema_full_vs_subtree_gap", 0.0),
            "full_cloud_corr_ema_full_vs_context_gap": _state_get_float(correction_state, "ema_full_vs_context_gap", 0.0),
            "full_cloud_corr_ema_full_vs_proxy_gap": _state_get_float(correction_state, "ema_full_vs_proxy_gap", 0.0),
            "full_cloud_corr_ema_full_actual_delta": _state_get_float(correction_state, "ema_full_actual_delta", 0.0),
            "full_cloud_corr_last_full_actual_delta": _state_get_float(correction_state, "last_full_actual_delta", 0.0),
            "full_cloud_corr_last_subtree_actual_delta": _state_get_float(correction_state, "last_subtree_actual_delta", 0.0),
            "full_cloud_corr_last_full_context_delta": _state_get_float(correction_state, "last_full_context_delta", 0.0),
            "full_cloud_corr_last_subtree_proxy_delta": _state_get_float(correction_state, "last_subtree_proxy_delta", 0.0),
            "full_cloud_corr_move_count": _state_get_float(correction_state, "last_move_count", 0.0),
            "full_cloud_corr_add_count": _state_get_float(correction_state, "last_add_count", 0.0),
            "full_cloud_corr_drop_count": _state_get_float(correction_state, "last_drop_count", 0.0),
            "full_cloud_corr_same_voxel_move_rejected": _state_get_float(correction_state, "last_same_voxel_move_rejected", 0.0),
            "full_cloud_corr_existing_target_rejected": _state_get_float(correction_state, "last_existing_target_rejected", 0.0),
            "full_cloud_corr_duplicate_target_rejected": _state_get_float(correction_state, "last_duplicate_target_rejected", 0.0),
            "full_cloud_corr_child_slot_rejected": _state_get_float(correction_state, "last_child_slot_rejected", 0.0),
            "full_cloud_corr_empty_target_rejected": _state_get_float(correction_state, "last_empty_target_rejected", 0.0),
        }
    )

    return correction_loss, debug