import torch

from models.utils.compression.octree_stats import hard_octree_occupancy_stats_from_voxel_coords


def _zero(reference):
    if torch.is_tensor(reference):
        return reference.new_zeros(())
    return torch.zeros((), dtype=torch.float32)


def _safe_float(value, default=0.0):
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            return float(value.detach().float().mean().cpu())
        return float(value)
    except Exception:
        return float(default)


def _normalize_coords_b3n(coords, device=None):
    if coords is None:
        return None
    if not torch.is_tensor(coords):
        coords = torch.as_tensor(coords)
    if device is not None:
        coords = coords.to(device=device)

    if coords.ndim == 2:
        if coords.shape[0] == 3:
            coords = coords.unsqueeze(0)
        elif coords.shape[1] == 3:
            coords = coords.transpose(0, 1).contiguous().unsqueeze(0)
        else:
            return None
    elif coords.ndim == 3:
        if coords.shape[1] == 3:
            coords = coords.contiguous()
        elif coords.shape[2] == 3:
            coords = coords.permute(0, 2, 1).contiguous()
        else:
            return None
    else:
        return None

    return coords.to(dtype=torch.long).contiguous()


def _valid_coords_n3(coords_b3n, valid_mask=None, batch_index=0):
    if coords_b3n is None:
        return None
    b = min(int(batch_index), int(coords_b3n.shape[0]) - 1)
    coords_n3 = coords_b3n[b].transpose(0, 1).contiguous()
    if valid_mask is None:
        return torch.unique(coords_n3, dim=0, sorted=True) if coords_n3.numel() > 0 else coords_n3

    mask = valid_mask
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, device=coords_n3.device)
    mask = mask.to(device=coords_n3.device, dtype=torch.bool)
    if mask.ndim == 3:
        mask = mask.squeeze(1)
    if mask.ndim == 1:
        mask_b = mask
    else:
        mask_b = mask[min(b, mask.shape[0] - 1)]
    mask_b = mask_b.reshape(-1)
    if mask_b.numel() != coords_n3.shape[0]:
        return torch.unique(coords_n3, dim=0, sorted=True) if coords_n3.numel() > 0 else coords_n3

    coords_n3 = coords_n3[mask_b]
    return torch.unique(coords_n3, dim=0, sorted=True) if coords_n3.numel() > 0 else coords_n3


def _coord_keys(coords, mins, spans):
    shifted = coords - mins.view(1, 3)
    return (
        shifted[:, 0] * spans[1].clamp_min(1) * spans[2].clamp_min(1)
        + shifted[:, 1] * spans[2].clamp_min(1)
        + shifted[:, 2]
    )


def _coords_membership(query_coords, reference_coords):
    if query_coords is None or reference_coords is None:
        return None
    if query_coords.numel() == 0:
        return torch.zeros((query_coords.shape[0],), device=query_coords.device, dtype=torch.bool)
    if reference_coords.numel() == 0:
        return torch.zeros((query_coords.shape[0],), device=query_coords.device, dtype=torch.bool)

    combined = torch.cat([query_coords.to(torch.long), reference_coords.to(torch.long)], dim=0)
    mins = combined.amin(dim=0)
    spans = (combined.amax(dim=0) - mins + 1).to(torch.long).clamp_min(1)

    query_keys = _coord_keys(query_coords.to(torch.long), mins, spans)
    reference_keys = torch.unique(_coord_keys(reference_coords.to(torch.long), mins, spans), sorted=True)

    pos = torch.searchsorted(reference_keys, query_keys)
    in_bounds = pos < reference_keys.numel()
    safe_pos = pos.clamp(max=max(int(reference_keys.numel()) - 1, 0))
    return in_bounds & (reference_keys[safe_pos] == query_keys)


def _remove_coords(base_coords, remove_coords):
    base_coords = base_coords.to(dtype=torch.long).reshape(-1, 3)
    if base_coords.numel() == 0 or remove_coords is None or remove_coords.numel() == 0:
        return base_coords
    remove_coords = torch.unique(remove_coords.to(device=base_coords.device, dtype=torch.long).reshape(-1, 3), dim=0, sorted=True)
    remove_mask = _coords_membership(base_coords, remove_coords)
    return base_coords[~remove_mask].contiguous()


def _append_coords(base_coords, add_coords):
    base_coords = base_coords.to(dtype=torch.long).reshape(-1, 3)
    if add_coords is None or add_coords.numel() == 0:
        return torch.unique(base_coords, dim=0, sorted=True) if base_coords.numel() > 0 else base_coords
    add_coords = add_coords.to(device=base_coords.device, dtype=torch.long).reshape(-1, 3)
    if base_coords.numel() == 0:
        return torch.unique(add_coords, dim=0, sorted=True)
    return torch.unique(torch.cat([base_coords, add_coords], dim=0), dim=0, sorted=True)


def _first_existing_tensor(dict_obj, keys):
    if not isinstance(dict_obj, dict):
        return None
    for key in keys:
        value = dict_obj.get(key, None)
        if torch.is_tensor(value):
            return value
    return None


def _first_existing_value(dict_obj, keys):
    if not isinstance(dict_obj, dict):
        return None
    for key in keys:
        if key in dict_obj and dict_obj.get(key, None) is not None:
            return dict_obj.get(key)
    return None

def _first_grad_tensor(dict_obj, keys):
    """
    actuator_voxel_stateから、勾配を持つsoft scalar/tensorを取り出す。
    hard countやdetach済み値は学習用proxyには使わない。
    """
    if not isinstance(dict_obj, dict):
        return None
    for key in keys:
        value = dict_obj.get(key, None)
        if torch.is_tensor(value) and value.requires_grad:
            if value.numel() == 0:
                continue
            value = torch.nan_to_num(value.float().mean(), nan=0.0, posinf=0.0, neginf=0.0)
            return value
    return None


def _soft_proxy_term_or_zero(reference, value):
    if torch.is_tensor(value) and value.requires_grad:
        return value.to(device=reference.device, dtype=reference.dtype).reshape(())
    return reference.new_zeros(())

def _stats_delta(before_stats, after_stats, key, normalize_by_before=False):
    before = float(before_stats.get(key, 0.0))
    after = float(after_stats.get(key, 0.0))
    delta = after - before
    if normalize_by_before:
        delta = delta / max(abs(before), 1.0)
    return before, after, delta


def build_full_context_subtree_delta_loss(
    args,
    full_octree_context=None,
    subtree_tree=None,
    actuator_voxel_state=None,
    reference=None,
):
    """
    full cloud文脈上で、Subtree編集前後のoccupancy統計差分をproxy lossとして返す。
    Phase4ではactual SparsePCGCを呼ばず、hard occupancy statsだけを使う。
    """
    zero = _zero(reference)
    debug = {
        "full_context_subtree_delta_used": False,
        "full_context_subtree_delta_reason": "not_initialized",
        "full_context_subtree_delta_value": 0.0,
        "full_context_subtree_delta_before_nodes": 0.0,
        "full_context_subtree_delta_after_nodes": 0.0,
        "full_context_subtree_delta_before_single": 0.0,
        "full_context_subtree_delta_after_single": 0.0,
        "full_context_subtree_delta_before_entropy": 0.0,
        "full_context_subtree_delta_after_entropy": 0.0,
        "full_context_subtree_delta_before_lowprob": 0.0,
        "full_context_subtree_delta_after_lowprob": 0.0,
        "full_context_subtree_delta_before_nll": 0.0,
        "full_context_subtree_delta_after_nll": 0.0,
        "full_context_subtree_delta_before_count": 0.0,
        "full_context_subtree_delta_after_count": 0.0,
        "full_context_subtree_delta_before_isolated": 0.0,
        "full_context_subtree_delta_after_isolated": 0.0,
        "full_context_subtree_delta_grad_used": False,
        "full_context_subtree_hard_loss_value": 0.0,
        "full_context_subtree_soft_proxy_used": False,
        "full_context_subtree_soft_proxy_loss_value": 0.0,
        "full_context_subtree_soft_proxy_severity": 0.0,
        "full_context_subtree_soft_proxy_move_mean": 0.0,
        "full_context_subtree_soft_proxy_add_mean": 0.0,
        "full_context_subtree_soft_proxy_drop_mean": 0.0,
        "full_context_subtree_soft_proxy_weight": float(getattr(args, "full_context_subtree_soft_proxy_weight", 0.05)),
    }

    if not bool(getattr(args, "full_context_subtree_loss", True)):
        debug["full_context_subtree_delta_reason"] = "disabled_by_args"
        return zero, debug

    if bool(getattr(args, "full_context_subtree_loss_require_context", True)):
        if full_octree_context is None or subtree_tree is None:
            debug["full_context_subtree_delta_reason"] = "missing_subtree_tree_or_full_octree_context"
            return zero, debug

    if not isinstance(actuator_voxel_state, dict):
        debug["full_context_subtree_delta_reason"] = "missing_actuator_voxel_state"
        return zero, debug

    device = reference.device if torch.is_tensor(reference) else None

    full_coords_raw = _first_existing_tensor(
        full_octree_context,
        (
            "full_occupied_voxel_coords",
            "full_global_voxel_coords",
            "global_voxel_coords",
            "occupied_voxel_coords",
        ),
    )
    before_subtree_raw = _first_existing_tensor(
        actuator_voxel_state,
        ("initial_voxel_coords", "point_aligned_initial_voxel_coords"),
    )
    if before_subtree_raw is None:
        before_subtree_raw = _first_existing_tensor(
            subtree_tree,
            ("subtree_global_voxel_coords", "global_voxel_coords"),
        )
    after_subtree_raw = _first_existing_tensor(
        actuator_voxel_state,
        ("final_voxel_coords", "point_aligned_final_voxel_coords"),
    )
    after_valid_mask = _first_existing_tensor(
        actuator_voxel_state,
        ("final_voxel_valid_mask",),
    )

    if full_coords_raw is None:
        debug["full_context_subtree_delta_reason"] = "missing_full_cloud_voxel_coords"
        return zero, debug
    if before_subtree_raw is None:
        debug["full_context_subtree_delta_reason"] = "missing_before_subtree_voxel_coords"
        return zero, debug
    if after_subtree_raw is None:
        debug["full_context_subtree_delta_reason"] = "missing_after_subtree_voxel_coords"
        return zero, debug

    full_coords_b3n = _normalize_coords_b3n(full_coords_raw, device=device)
    before_subtree_b3n = _normalize_coords_b3n(before_subtree_raw, device=device)
    after_subtree_b3n = _normalize_coords_b3n(after_subtree_raw, device=device)
    if full_coords_b3n is None or before_subtree_b3n is None or after_subtree_b3n is None:
        debug["full_context_subtree_delta_reason"] = "invalid_voxel_coord_shape"
        return zero, debug

    max_depth = int(getattr(args, "full_context_subtree_loss_max_depth", getattr(args, "sparsepcgc_occupancy_max_depth", 0)))
    batch_count = int(max(full_coords_b3n.shape[0], before_subtree_b3n.shape[0], after_subtree_b3n.shape[0]))

    losses = []
    before_stats_list = []
    after_stats_list = []

    for b in range(batch_count):
        full_n3 = _valid_coords_n3(full_coords_b3n, batch_index=b)
        before_subtree_n3 = _valid_coords_n3(before_subtree_b3n, batch_index=b)
        after_subtree_n3 = _valid_coords_n3(after_subtree_b3n, valid_mask=after_valid_mask, batch_index=b)

        if full_n3 is None or before_subtree_n3 is None or after_subtree_n3 is None:
            continue
        if full_n3.numel() == 0:
            continue

        before_full_n3 = torch.unique(full_n3, dim=0, sorted=True)
        after_full_n3 = _append_coords(
            _remove_coords(before_full_n3, before_subtree_n3),
            after_subtree_n3,
        )

        grid_origin = before_full_n3.amin(dim=0).reshape(1, 3) if before_full_n3.numel() > 0 else None

        with torch.no_grad():
            before_stats = hard_octree_occupancy_stats_from_voxel_coords(
                before_full_n3,
                max_depth=max_depth,
                grid_origin=grid_origin,
            )
            after_stats = hard_octree_occupancy_stats_from_voxel_coords(
                after_full_n3,
                max_depth=max_depth,
                grid_origin=grid_origin,
            )

        before_stats_list.append(before_stats)
        after_stats_list.append(after_stats)

        _, _, node_delta = _stats_delta(before_stats, after_stats, "node_count", normalize_by_before=True)
        _, _, single_delta = _stats_delta(before_stats, after_stats, "single_ratio", normalize_by_before=False)
        _, _, entropy_delta = _stats_delta(before_stats, after_stats, "occupancy_entropy", normalize_by_before=False)
        _, _, nll_delta = _stats_delta(before_stats, after_stats, "occupancy_nll", normalize_by_before=False)
        _, _, lowprob_delta = _stats_delta(before_stats, after_stats, "lowprob_occupancy_ratio", normalize_by_before=False)
        _, _, count_delta = _stats_delta(before_stats, after_stats, "leaf_count", normalize_by_before=True)
        _, _, isolated_delta = _stats_delta(before_stats, after_stats, "isolated_voxel_ratio", normalize_by_before=False)

        hard_obj = (
            float(getattr(args, "full_context_subtree_loss_node_weight", 0.05)) * max(node_delta, 0.0)
            + float(getattr(args, "full_context_subtree_loss_single_weight", 0.10)) * max(single_delta, 0.0)
            + float(getattr(args, "full_context_subtree_loss_entropy_weight", 0.20)) * max(entropy_delta, 0.0)
            + float(getattr(args, "full_context_subtree_loss_nll_weight", 0.00)) * max(nll_delta, 0.0)
            + float(getattr(args, "full_context_subtree_loss_lowprob_weight", 0.20)) * max(lowprob_delta, 0.0)
            + float(getattr(args, "full_context_subtree_loss_count_weight", 0.02)) * max(count_delta, 0.0)
            + float(getattr(args, "full_context_subtree_loss_fragment_weight", 0.05)) * max(isolated_delta, 0.0)
        )
        losses.append(zero.new_tensor(float(hard_obj)))

    if not losses:
        debug["full_context_subtree_delta_reason"] = "no_valid_batch"
        return zero, debug

    hard_loss = losses[0]
    for item in losses[1:]:
        hard_loss = hard_loss + item
    hard_loss = hard_loss / float(max(len(losses), 1))

    loss_value = hard_loss.detach()

    # Phase7-2:
    # hard_lossはoccupancy stats由来なのでdebug / teacher値としてdetachしたまま扱う。
    # 実際の勾配はActuator由来のsoft edit量へ弱く返す。
    soft_proxy_loss = zero

    if bool(getattr(args, "full_context_subtree_soft_proxy", True)):
        raw_hard_severity = hard_loss.detach().clamp_min(0.0)
        severity_floor = max(
            float(getattr(args, "full_context_subtree_soft_proxy_severity_floor", 0.0)),
            0.0,
        )
        if severity_floor > 0.0:
            hard_severity = torch.where(
                raw_hard_severity > 0.0,
                torch.maximum(raw_hard_severity, raw_hard_severity.new_tensor(severity_floor)),
                raw_hard_severity,
            )
        else:
            hard_severity = raw_hard_severity

        soft_move = _first_grad_tensor(
            actuator_voxel_state,
            (
                "voxel_soft_move_amount",
                "voxel_soft_move_score",
                "move_ratio_soft",
                "learned_move_ratio",
                "soft_move_voxel_sum",
            ),
        )
        soft_add = _first_grad_tensor(
            actuator_voxel_state,
            (
                "voxel_soft_add_amount",
                "voxel_soft_add_score",
                "add_ratio_soft",
                "learned_add_ratio",
                "soft_add_sum",
            ),
        )
        soft_drop = _first_grad_tensor(
            actuator_voxel_state,
            (
                "voxel_soft_drop_amount",
                "voxel_soft_drop_score",
                "drop_ratio_soft",
                "learned_drop_ratio",
                "soft_drop_mass",
                "drop_prob_proxy",
            ),
        )

        soft_move_term = _soft_proxy_term_or_zero(zero, soft_move)
        soft_add_term = _soft_proxy_term_or_zero(zero, soft_add)
        soft_drop_term = _soft_proxy_term_or_zero(zero, soft_drop)

        soft_penalty = (
            float(getattr(args, "full_context_subtree_soft_proxy_move_weight", 1.0)) * soft_move_term
            + float(getattr(args, "full_context_subtree_soft_proxy_add_weight", 0.5)) * soft_add_term
            + float(getattr(args, "full_context_subtree_soft_proxy_drop_weight", 0.0)) * soft_drop_term
        )

        if torch.is_tensor(soft_penalty) and soft_penalty.requires_grad:
            soft_proxy_loss = (
                float(getattr(args, "full_context_subtree_soft_proxy_weight", 0.05))
                * hard_severity
                * soft_penalty
            )
            loss_value = loss_value + soft_proxy_loss
            debug["full_context_subtree_delta_grad_used"] = True
            debug["full_context_subtree_soft_proxy_used"] = True
            debug["full_context_subtree_soft_proxy_loss_value"] = _safe_float(soft_proxy_loss)
            debug["full_context_subtree_soft_proxy_severity"] = _safe_float(hard_severity)
            debug["full_context_subtree_soft_proxy_move_mean"] = _safe_float(soft_move_term)
            debug["full_context_subtree_soft_proxy_add_mean"] = _safe_float(soft_add_term)
            debug["full_context_subtree_soft_proxy_drop_mean"] = _safe_float(soft_drop_term)

    final_weights = _first_existing_tensor(actuator_voxel_state, ("final_voxel_weights", "point_aligned_final_voxel_weights"))
    if (
        torch.is_tensor(final_weights)
        and final_weights.requires_grad
        and float(getattr(args, "full_context_subtree_loss_grad_weight", 0.1)) > 0.0
    ):
        proxy = torch.nan_to_num(final_weights.float().mean(), nan=0.0, posinf=0.0, neginf=0.0)
        loss_value = loss_value + float(getattr(args, "full_context_subtree_loss_grad_weight", 0.1)) * (proxy - proxy.detach())
        debug["full_context_subtree_delta_grad_used"] = True
        debug["full_context_subtree_final_weight_proxy_used"] = True
        debug["full_context_subtree_final_weight_proxy_value"] = _safe_float(proxy)
    else:
        debug["full_context_subtree_final_weight_proxy_used"] = False
        debug["full_context_subtree_final_weight_proxy_value"] = 0.0

    weight = float(getattr(args, "full_context_subtree_loss_weight", 0.2))
    loss_value = weight * loss_value

    def _mean_stat(stats_list, key):
        if not stats_list:
            return 0.0
        return sum(float(item.get(key, 0.0)) for item in stats_list) / float(max(len(stats_list), 1))

    before_nodes = _mean_stat(before_stats_list, "node_count")
    after_nodes = _mean_stat(after_stats_list, "node_count")
    before_single = _mean_stat(before_stats_list, "single_ratio")
    after_single = _mean_stat(after_stats_list, "single_ratio")
    before_entropy = _mean_stat(before_stats_list, "occupancy_entropy")
    after_entropy = _mean_stat(after_stats_list, "occupancy_entropy")
    before_lowprob = _mean_stat(before_stats_list, "lowprob_occupancy_ratio")
    after_lowprob = _mean_stat(after_stats_list, "lowprob_occupancy_ratio")
    before_nll = _mean_stat(before_stats_list, "occupancy_nll")
    after_nll = _mean_stat(after_stats_list, "occupancy_nll")
    before_count = _mean_stat(before_stats_list, "leaf_count")
    after_count = _mean_stat(after_stats_list, "leaf_count")
    before_isolated = _mean_stat(before_stats_list, "isolated_voxel_ratio")
    after_isolated = _mean_stat(after_stats_list, "isolated_voxel_ratio")

    debug.update(
        {
            "full_context_subtree_delta_used": True,
            "full_context_subtree_delta_reason": "ok",
            "full_context_subtree_delta_value": _safe_float(loss_value),
            "full_context_subtree_hard_loss_value": _safe_float(hard_loss.detach()),
            "full_context_subtree_soft_proxy_used": bool(debug.get("full_context_subtree_soft_proxy_used", False)),
            "full_context_subtree_soft_proxy_loss_value": float(debug.get("full_context_subtree_soft_proxy_loss_value", 0.0)),
            "full_context_subtree_hard_loss": _safe_float(hard_loss.detach()),
            "full_context_subtree_loss_total": _safe_float(loss_value),
            "full_context_hard_loss": _safe_float(hard_loss.detach()),
            "full_context_soft_proxy_loss": float(debug.get("full_context_subtree_soft_proxy_loss_value", 0.0)),
            "full_context_subtree_soft_proxy_severity": float(debug.get("full_context_subtree_soft_proxy_severity", 0.0)),
            "full_context_subtree_soft_proxy_move_mean": float(debug.get("full_context_subtree_soft_proxy_move_mean", 0.0)),
            "full_context_subtree_soft_proxy_add_mean": float(debug.get("full_context_subtree_soft_proxy_add_mean", 0.0)),
            "full_context_subtree_soft_proxy_drop_mean": float(debug.get("full_context_subtree_soft_proxy_drop_mean", 0.0)),
            "full_context_subtree_soft_proxy_weight": float(getattr(args, "full_context_subtree_soft_proxy_weight", 0.05)),
            "full_context_subtree_delta_before_nodes": float(before_nodes),
            "full_context_subtree_delta_after_nodes": float(after_nodes),
            "full_context_subtree_delta_node_delta_norm": (after_nodes - before_nodes) / max(abs(before_nodes), 1.0),
            "full_context_subtree_delta_before_single": float(before_single),
            "full_context_subtree_delta_after_single": float(after_single),
            "full_context_subtree_delta_single_delta": float(after_single - before_single),
            "full_context_subtree_delta_before_entropy": float(before_entropy),
            "full_context_subtree_delta_after_entropy": float(after_entropy),
            "full_context_subtree_delta_entropy_delta": float(after_entropy - before_entropy),
            "full_context_subtree_delta_before_lowprob": float(before_lowprob),
            "full_context_subtree_delta_after_lowprob": float(after_lowprob),
            "full_context_subtree_delta_lowprob_delta": float(after_lowprob - before_lowprob),
            "full_context_subtree_delta_before_nll": float(before_nll),
            "full_context_subtree_delta_after_nll": float(after_nll),
            "full_context_subtree_delta_nll_delta": float(after_nll - before_nll),
            "full_context_subtree_delta_before_count": float(before_count),
            "full_context_subtree_delta_after_count": float(after_count),
            "full_context_subtree_delta_count_delta_norm": (after_count - before_count) / max(abs(before_count), 1.0),
            "full_context_subtree_delta_before_isolated": float(before_isolated),
            "full_context_subtree_delta_after_isolated": float(after_isolated),
            "full_context_subtree_delta_isolated_delta": float(after_isolated - before_isolated),
            "full_context_subtree_delta_weight": float(weight),
            "full_context_subtree_delta_batch_count": int(len(losses)),
            "full_context_subtree_voxel_state_source": "actuator_voxel_state",
            "full_context_subtree_uses_initial_voxel_coords": bool(before_subtree_raw is not None),
            "full_context_subtree_uses_final_voxel_coords": bool(after_subtree_raw is not None),
            "full_context_subtree_uses_final_valid_mask": bool(after_valid_mask is not None),
            "full_context_subtree_same_state_contract": bool(
                before_subtree_raw is not None and after_subtree_raw is not None
            ),
        }
    )

    return loss_value, debug
