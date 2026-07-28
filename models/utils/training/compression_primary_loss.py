import math

import torch

from .scalar_utils import case_float
from .train_flow import compose_train_compression_main, _actual_total_bit_objective_mix_state
from .utils import uses_actual_total_bit_objective

def as_scalar_loss_tensor(value):
    if not torch.is_tensor(value):
        return None
    if value.numel() == 1:
        return value.reshape(())
    return value.mean()


def zero_like_loss(reference):
    if torch.is_tensor(reference):
        return reference.new_zeros(())
    return torch.zeros((), dtype=torch.float32)


def relu_penalty(term, tau):
    tau_t = term.new_tensor(float(tau))
    return torch.relu(term - tau_t)


def _compression_primary_support_balance(
    args,
    primary_value,
    support_value,
    *,
    enabled=True,
    target_ratio_name="compression_primary_aux_target_ratio",
    min_scale_name="compression_primary_aux_balance_min_scale",
    max_scale_name="compression_primary_aux_balance_max_scale",
    disabled_reason="actual_total_bit_balance_disabled",
):
    """
    compression_primary で、圧縮主目的に対して support block が強くなりすぎたときだけ
    support block を弱めるためのバランス情報を返す。

    返り値:
      {
        "scale": support block に掛ける係数,
        "reason": ログ用の簡潔な説明,
        "target_ratio": support / |primary| の目標上限,
        "primary_mag": |primary|,
        "support_mag": |support|,
        "scaled_support_mag": scale * |support|,
        "dominant": "compression" / "support" / "neutral",
      }
    """
    if not enabled:
        return {
            "scale": 1.0,
            "reason": str(disabled_reason),
            "target_ratio": None,
            "primary_mag": None,
            "support_mag": None,
            "scaled_support_mag": None,
            "dominant": "neutral",
        }

    target_ratio = max(float(getattr(args, target_ratio_name, 0.25)), 0.0)
    min_scale = min(max(float(getattr(args, min_scale_name, 0.0)), 0.0), 1.0)
    max_scale = min(max(float(getattr(args, max_scale_name, 1.0)), min_scale), 1.0)
    if primary_value is None or support_value is None:
        return {
            "scale": 1.0,
            "reason": "balance_value_missing",
            "target_ratio": float(target_ratio),
            "primary_mag": None,
            "support_mag": None,
            "scaled_support_mag": None,
            "dominant": "neutral",
        }
    if not torch.is_tensor(primary_value) or not torch.is_tensor(support_value):
        return {
            "scale": 1.0,
            "reason": "balance_tensor_missing",
            "target_ratio": float(target_ratio),
            "primary_mag": None,
            "support_mag": None,
            "scaled_support_mag": None,
            "dominant": "neutral",
        }

    primary_mag = float(torch.nan_to_num(primary_value.detach().abs(), nan=0.0, posinf=0.0, neginf=0.0).cpu())
    support_mag = float(torch.nan_to_num(support_value.detach().abs(), nan=0.0, posinf=0.0, neginf=0.0).cpu())
    if not math.isfinite(primary_mag) or not math.isfinite(support_mag):
        return {
            "scale": 1.0,
            "reason": "balance_non_finite",
            "target_ratio": float(target_ratio),
            "primary_mag": None,
            "support_mag": None,
            "scaled_support_mag": None,
            "dominant": "neutral",
        }
    if support_mag <= 1e-12:
        dominant = "compression" if primary_mag > 1e-12 else "neutral"
        return {
            "scale": float(max_scale),
            "reason": "support_block_zero",
            "target_ratio": float(target_ratio),
            "primary_mag": float(primary_mag),
            "support_mag": float(support_mag),
            "scaled_support_mag": 0.0,
            "dominant": dominant,
        }

    budget_mag = target_ratio * primary_mag
    raw_scale = 0.0 if budget_mag <= 1e-12 else (budget_mag / support_mag)
    scale = min(max(raw_scale, min_scale), max_scale)
    scaled_support_mag = scale * support_mag

    if raw_scale < min_scale:
        reason = "support_over_budget"
    elif raw_scale > max_scale:
        reason = "support_under_budget"
    else:
        reason = "balanced_to_target"

    if primary_mag <= 1e-12 and scaled_support_mag <= 1e-12:
        dominant = "neutral"
    elif primary_mag + 1e-12 >= scaled_support_mag:
        dominant = "compression"
    else:
        dominant = "support"

    return {
        "scale": float(scale),
        "reason": str(reason),
        "target_ratio": float(target_ratio),
        "primary_mag": float(primary_mag),
        "support_mag": float(support_mag),
        "scaled_support_mag": float(scaled_support_mag),
        "dominant": dominant,
    }


def monotonic_support_scale(previous_scale, proposed_scale):
    """初期の支配を抑えたscaleを、loss低下に合わせて逆増幅しない。"""
    proposed = float(proposed_scale)
    previous = float(previous_scale)
    if not math.isfinite(proposed) or proposed < 0.0:
        raise ValueError("proposed_scaleは有限の非負値である必要がある")
    if not math.isfinite(previous) or previous < 0.0:
        return proposed
    return min(previous, proposed)


def term_requires_grad(value):
    return bool(torch.is_tensor(value) and value.requires_grad)


def term_is_finite(value):
    if not torch.is_tensor(value):
        return False
    try:
        return bool(torch.isfinite(value.detach()).all().item())
    except Exception:
        return False


def select_compression_primary_main(terms, L_com):
    candidates = [
        ("main", terms.get("main", None)),
        ("bit", terms.get("bit", None)),
        ("objective", terms.get("objective", None)),
        ("L_com_fallback", L_com),
    ]
    tensor_candidates = []
    for name, value in candidates:
        tensor_value = as_scalar_loss_tensor(value)
        if tensor_value is not None:
            tensor_candidates.append((name, tensor_value))
    for name, value in tensor_candidates:
        if value.requires_grad:
            return name, value
    if tensor_candidates:
        return tensor_candidates[0]
    return "zero_fallback", zero_like_loss(L_com)


def build_compression_primary_loss(
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
    L_sparsepcgc = as_scalar_loss_tensor(terms.get("sparsepcgc", None))
    if bool(getattr(args, "_sparsepcgc_full_cloud_actual_primary_active", False)) and torch.is_tensor(L_com):
        main_source = "full_cloud_actual_primary_lcom"
        L_com_main = as_scalar_loss_tensor(L_com)
    else:
        main_source, L_com_main = select_compression_primary_main(terms, L_com)
    if uses_actual_total_bit_objective(args):
        # actual/surrogate系ではL_com直結と圧縮内訳を半々で混ぜた主目的にする。
        _, main_zero_fallback_used = _actual_total_bit_objective_mix_state(args, terms, L_com)
        L_com_main = compose_train_compression_main(args, terms, L_com_main, zero_like_loss(L_com_main))
        # Debugで混合主目的を使ったことを追えるようにsource名へ印を付ける。
        main_source = f"{main_source}+mixed_terms"
        if main_zero_fallback_used:
            main_source = f"{main_source}+zero_actual_fallback"
        # SparsePCGC補助項はsurrogate側でgate済みのTensorだけがterms["sparsepcgc"]へ入る。
        # actual_total_bit_objective_mix=1.0でもactualが0のstepだけproxy側へ少し戻すため、
        # forward値はactualのまま、backwardだけSparsePCGC補助へ戻す。
        sparsepcgc_main_grad_weight = max(
            float(getattr(args, "cp_sparsepcgc_aux_main_grad_weight", 0.0)),
            0.0,
        )
        if sparsepcgc_main_grad_weight > 0.0 and term_requires_grad(L_sparsepcgc):
            L_com_main = L_com_main + sparsepcgc_main_grad_weight * (
                L_sparsepcgc - L_sparsepcgc.detach()
            )
            main_source = f"{main_source}+sparsepcgc_aux_grad"
    else:
        sparsepcgc_main_grad_weight = 0.0
    warmup_steps = int(getattr(args, "compression_primary_warmup_steps", 0))
    if warmup_steps > 0:
        warmup = min(1.0, float(int(global_train_step) + 1) / float(warmup_steps))
    else:
        warmup = 1.0
    L_com_primary = float(getattr(args, "w_com", 1.0)) * float(warmup) * L_com_main

    zero = zero_like_loss(L_com_primary)
    L_single = as_scalar_loss_tensor(terms.get("single", None))
    L_nodes = as_scalar_loss_tensor(terms.get("node", None))
    L_op = as_scalar_loss_tensor(terms.get("op", None))

    L_full_context_subtree_delta = as_scalar_loss_tensor(
        terms.get("full_context_subtree_delta", None)
    )
    L_full_cloud_actual_correction = as_scalar_loss_tensor(
        terms.get("full_cloud_actual_correction", None)
    )

    full_cloud_correction_cp_weight = max(
        float(getattr(args, "cp_full_cloud_actual_correction_weight", 1.0)),
        0.0,
    )

    if L_full_cloud_actual_correction is not None:
        L_full_cloud_actual_correction = torch.nan_to_num(
            L_full_cloud_actual_correction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    # ============================================================
    # Phase3:
    # full-context subtree delta を compression primary の主勾配として扱う。
    # 既存の full_context_subtree_loss_weight は loss生成側の内部重みなので、
    # ここでは compression primary へ混ぜる外側の重みを別に持つ。
    # ============================================================
    full_context_cp_weight = max(
        float(getattr(args, "cp_full_context_subtree_delta_weight", 1.0)),
        0.0,
    )

    if L_full_context_subtree_delta is not None:
        L_full_context_subtree_delta = torch.nan_to_num(
            L_full_context_subtree_delta,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    P_geom = relu_penalty(as_scalar_loss_tensor(L_geom), getattr(args, "cp_tau_geom", 0.06))
    P_single = relu_penalty(L_single, getattr(args, "cp_tau_single", 0.0)) if L_single is not None else zero
    P_nodes = relu_penalty(L_nodes, getattr(args, "cp_tau_nodes", 0.0)) if L_nodes is not None else zero
    P_sparsepcgc = (
        relu_penalty(L_sparsepcgc, getattr(args, "cp_tau_sparsepcgc", 0.0))
        if L_sparsepcgc is not None
        else zero
    )
    P_actuator = relu_penalty(as_scalar_loss_tensor(L_actuator), getattr(args, "cp_tau_actuator", 0.0))
    P_op = relu_penalty(L_op, 0.0) if L_op is not None else zero
    gating_debug = getattr(args, "_node_single_gating_debug", {}) or {}
    node_weight_raw = float(getattr(args, "cp_lambda_nodes", 1.0))
    single_weight_raw = float(getattr(args, "cp_lambda_single", 1.0))
    node_weight_effective = node_weight_raw * float(gating_debug.get("node_scale", 1.0))
    single_weight_effective = single_weight_raw * float(gating_debug.get("single_scale", 1.0))
    single_delta_penalty = relu_penalty(L_single, 0.0) if L_single is not None else zero
    single_delta_corr = getattr(args, "_last_single_delta_actual_corr", None)
    single_delta_penalty_used = bool(
        getattr(args, "single_delta_penalty_backprop", False)
        and single_delta_corr is not None
        and float(single_delta_corr) >= float(getattr(args, "single_delta_penalty_min_corr", 0.30))
    )
    single_delta_penalty_weight = float(getattr(args, "single_delta_penalty_weight", 0.0)) if single_delta_penalty_used else 0.0

    use_stage = bool(getattr(args, "cp_use_stage_factors", False))
    sf_com = float(stage_factors.get("com", 1.0)) if use_stage else 1.0
    sf_geom = float(stage_factors.get("geom", 1.0)) if use_stage else 1.0
    sf_repair = float(stage_factors.get("repair", 1.0)) if use_stage else 1.0

    main_block = sf_com * L_com_primary
    geom_block = sf_geom * float(getattr(args, "cp_lambda_geom", 1.0)) * P_geom
    single_block = sf_com * single_weight_effective * P_single
    node_block = sf_com * node_weight_effective * P_nodes
    sparsepcgc_block = sf_com * float(getattr(args, "cp_lambda_sparsepcgc", 1.0)) * P_sparsepcgc
    actuator_block = sf_repair * float(getattr(args, "cp_lambda_actuator", 0.1)) * P_actuator
    op_block = sf_repair * float(getattr(args, "cp_lambda_op", 0.0)) * P_op
    single_delta_block = sf_com * single_delta_penalty_weight * single_delta_penalty
    full_context_block = sf_com * full_context_cp_weight * (
        L_full_context_subtree_delta
        if L_full_context_subtree_delta is not None
        else zero
    )
    full_cloud_correction_block = sf_com * full_cloud_correction_cp_weight * (
        L_full_cloud_actual_correction
        if L_full_cloud_actual_correction is not None
        else zero
    )
    aux_block = (
        geom_block
        + single_block
        + node_block
        + sparsepcgc_block
        + actuator_block
        + op_block
        + single_delta_block
        + full_context_block
        + full_cloud_correction_block
    )
    aux_balance = _compression_primary_support_balance(
        args,
        main_block,
        aux_block,
        enabled=uses_actual_total_bit_objective(args),
        target_ratio_name="compression_primary_aux_target_ratio",
        min_scale_name="compression_primary_aux_balance_min_scale",
        max_scale_name="compression_primary_aux_balance_max_scale",
        disabled_reason="actual_total_bit_balance_disabled",
    )
    aux_balance_scale = float(aux_balance["scale"])
    aux_balance_reason = str(aux_balance["reason"])
    aux_block_scaled = aux_balance_scale * aux_block
    L = main_block + aux_block_scaled
    main_grad_scale = 1.0 / max(float(aux_balance_scale), 1e-12)

    debug = {
        "loss_mode": "compression_primary",
        "cp_main_source": main_source,
        "cp_warmup": float(warmup),
        "cp_L_com_main": case_float(L_com_main, float("nan")),
        "cp_L_com_primary": case_float(L_com_primary, float("nan")),
        "cp_P_geom": case_float(P_geom, float("nan")),
        "cp_P_single": case_float(P_single, float("nan")),
        "cp_P_nodes": case_float(P_nodes, float("nan")),
        "cp_P_sparsepcgc": case_float(P_sparsepcgc, float("nan")),
        "cp_P_actuator": case_float(P_actuator, float("nan")),
        "cp_P_op": case_float(P_op, float("nan")),
        "cp_main_block": case_float(main_block, float("nan")),
        "cp_aux_block_raw": case_float(aux_block, float("nan")),
        "cp_aux_block_scaled": case_float(aux_block_scaled, float("nan")),
        "cp_aux_target_ratio": (
            float(aux_balance["target_ratio"])
            if aux_balance.get("target_ratio", None) is not None
            else float("nan")
        ),
        "cp_aux_balance_primary_abs": (
            float(aux_balance["primary_mag"])
            if aux_balance.get("primary_mag", None) is not None
            else float("nan")
        ),
        "cp_aux_balance_support_abs": (
            float(aux_balance["support_mag"])
            if aux_balance.get("support_mag", None) is not None
            else float("nan")
        ),
        "cp_aux_balance_scaled_support_abs": (
            float(aux_balance["scaled_support_mag"])
            if aux_balance.get("scaled_support_mag", None) is not None
            else float("nan")
        ),
        "cp_aux_balance_scale": float(aux_balance_scale),
        "cp_aux_balance_reason": str(aux_balance_reason),
        "cp_aux_balance_dominant": str(aux_balance.get("dominant", "neutral")),
        "cp_full_context_subtree_delta": case_float(
            L_full_context_subtree_delta if L_full_context_subtree_delta is not None else zero,
            0.0,
        ),
        "cp_full_context_subtree_delta_weight": float(full_context_cp_weight),
        "cp_full_context_subtree_delta_added": bool(L_full_context_subtree_delta is not None),
        "cp_full_cloud_actual_correction": case_float(
            L_full_cloud_actual_correction
            if L_full_cloud_actual_correction is not None
            else zero,
            0.0,
        ),
        "cp_full_cloud_actual_correction_weight": float(full_cloud_correction_cp_weight),
        "cp_full_cloud_actual_correction_added": bool(L_full_cloud_actual_correction is not None),
        "cp_full_cloud_actual_correction_requires_grad": term_requires_grad(L_full_cloud_actual_correction),
        "cp_full_cloud_actual_correction_used_for_backprop": bool(
            L_full_cloud_actual_correction is not None
            and term_requires_grad(L_full_cloud_actual_correction)
            and full_cloud_correction_cp_weight > 0.0
        ),
        "node_loss_weight_raw": float(node_weight_raw),
        "node_loss_weight_effective": float(node_weight_effective),
        "single_loss_weight_raw": float(single_weight_raw),
        "single_loss_weight_effective": float(single_weight_effective),
        "node_single_gating_reason": str(gating_debug.get("reason", "not_initialized")),
        "single_delta_penalty": case_float(single_delta_penalty, float("nan")),
        "single_delta_penalty_weight": float(single_delta_penalty_weight),
        "single_delta_penalty_used_for_backprop": bool(single_delta_penalty_used),
        "cp_sparsepcgc_main_grad_weight": float(sparsepcgc_main_grad_weight),
        "compression_main_grad_scale": float(main_grad_scale),
        "compression_main_grad_scale_reason": str(aux_balance_reason),
        "compression_aux_in_objective": True,
        "compression_main_loss": case_float(L_com_main, float("nan")),
        "compression_aux_loss": case_float(aux_block_scaled, float("nan")),
        "compression_objective": case_float(L, float("nan")),
        "cp_total": case_float(L, float("nan")),
        "cp_main_requires_grad": term_requires_grad(L_com_main),
        "cp_geom_requires_grad": term_requires_grad(L_geom),
        "cp_single_requires_grad": term_requires_grad(L_single),
        "cp_nodes_requires_grad": term_requires_grad(L_nodes),
        "cp_sparsepcgc_requires_grad": term_requires_grad(L_sparsepcgc),
        "cp_actuator_requires_grad": term_requires_grad(L_actuator),
        "cp_op_requires_grad": term_requires_grad(L_op),
        "cp_full_context_subtree_delta_requires_grad": term_requires_grad(L_full_context_subtree_delta),
        "cp_full_context_subtree_delta_used_for_backprop": bool(
            L_full_context_subtree_delta is not None
            and term_requires_grad(L_full_context_subtree_delta)
            and full_context_cp_weight > 0.0
        ),
        "cp_main_finite": term_is_finite(L_com_main),
        "cp_geom_finite": term_is_finite(L_geom),
        "cp_single_finite": term_is_finite(L_single) if L_single is not None else True,
        "cp_nodes_finite": term_is_finite(L_nodes) if L_nodes is not None else True,
        "cp_sparsepcgc_finite": term_is_finite(L_sparsepcgc) if L_sparsepcgc is not None else True,
        "cp_actuator_finite": term_is_finite(L_actuator),
        "cp_op_finite": term_is_finite(L_op) if L_op is not None else True,
        "cp_full_context_subtree_delta_finite": (
            term_is_finite(L_full_context_subtree_delta)
            if L_full_context_subtree_delta is not None
            else True
        ),
    }
    return L, sf_com * L_com_primary, debug


def log_compression_primary_terms(writer, step, num_steps, cp_debug):
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
        f"main_block={float(cp_debug.get('cp_main_block', 0.0)):.6f}, "
        f"aux_raw={float(cp_debug.get('cp_aux_block_raw', 0.0)):.6f}, "
        f"aux_scaled={float(cp_debug.get('cp_aux_block_scaled', 0.0)):.6f}, "
        f"target_ratio={float(cp_debug.get('cp_aux_target_ratio', 0.0)):.6f}, "
        f"aux_balance_scale={float(cp_debug.get('cp_aux_balance_scale', 1.0)):.6f}, "
        f"main_grad_scale={float(cp_debug.get('compression_main_grad_scale', 1.0)):.6f}, "
        f"balance_reason={cp_debug.get('compression_main_grad_scale_reason', 'n/a')}, "
        f"dominant={cp_debug.get('cp_aux_balance_dominant', 'neutral')}, "
        f"full_context_subtree_delta={float(cp_debug.get('cp_full_context_subtree_delta', 0.0)):.6f}, "
        f"node_w={float(cp_debug.get('node_loss_weight_effective', 0.0)):.6f}/"
        f"{float(cp_debug.get('node_loss_weight_raw', 0.0)):.6f}, "
        f"single_w={float(cp_debug.get('single_loss_weight_effective', 0.0)):.6f}/"
        f"{float(cp_debug.get('single_loss_weight_raw', 0.0)):.6f}, "
        f"single_delta_penalty={float(cp_debug.get('single_delta_penalty', 0.0)):.6f}, "
        f"total={float(cp_debug.get('cp_total', 0.0)):.6f}, "
        "requires_grad["
        f"main={bool(cp_debug.get('cp_main_requires_grad', False))}, "
        f"geom={bool(cp_debug.get('cp_geom_requires_grad', False))}, "
        f"single={bool(cp_debug.get('cp_single_requires_grad', False))}, "
        f"nodes={bool(cp_debug.get('cp_nodes_requires_grad', False))}, "
        f"sparsepcgc={bool(cp_debug.get('cp_sparsepcgc_requires_grad', False))}, "
        f"full_context_subtree_delta={bool(cp_debug.get('cp_full_context_subtree_delta_requires_grad', False))}, "
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
