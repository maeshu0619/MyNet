import torch

from .scalar_utils import case_float

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
    main_source, L_com_main = select_compression_primary_main(terms, L_com)
    warmup_steps = int(getattr(args, "compression_primary_warmup_steps", 0))
    if warmup_steps > 0:
        warmup = min(1.0, float(int(global_train_step) + 1) / float(warmup_steps))
    else:
        warmup = 1.0
    L_com_primary = float(getattr(args, "w_com", 1.0)) * float(warmup) * L_com_main

    zero = zero_like_loss(L_com_primary)
    L_single = as_scalar_loss_tensor(terms.get("single", None))
    L_nodes = as_scalar_loss_tensor(terms.get("node", None))
    L_sparsepcgc = as_scalar_loss_tensor(terms.get("sparsepcgc", None))
    L_op = as_scalar_loss_tensor(terms.get("op", None))

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
        "cp_L_com_main": case_float(L_com_main, float("nan")),
        "cp_L_com_primary": case_float(L_com_primary, float("nan")),
        "cp_P_geom": case_float(P_geom, float("nan")),
        "cp_P_single": case_float(P_single, float("nan")),
        "cp_P_nodes": case_float(P_nodes, float("nan")),
        "cp_P_sparsepcgc": case_float(P_sparsepcgc, float("nan")),
        "cp_P_actuator": case_float(P_actuator, float("nan")),
        "cp_P_op": case_float(P_op, float("nan")),
        "cp_total": case_float(L, float("nan")),
        "cp_main_requires_grad": term_requires_grad(L_com_main),
        "cp_geom_requires_grad": term_requires_grad(L_geom),
        "cp_single_requires_grad": term_requires_grad(L_single),
        "cp_nodes_requires_grad": term_requires_grad(L_nodes),
        "cp_sparsepcgc_requires_grad": term_requires_grad(L_sparsepcgc),
        "cp_actuator_requires_grad": term_requires_grad(L_actuator),
        "cp_op_requires_grad": term_requires_grad(L_op),
        "cp_main_finite": term_is_finite(L_com_main),
        "cp_geom_finite": term_is_finite(L_geom),
        "cp_single_finite": term_is_finite(L_single) if L_single is not None else True,
        "cp_nodes_finite": term_is_finite(L_nodes) if L_nodes is not None else True,
        "cp_sparsepcgc_finite": term_is_finite(L_sparsepcgc) if L_sparsepcgc is not None else True,
        "cp_actuator_finite": term_is_finite(L_actuator),
        "cp_op_finite": term_is_finite(L_op) if L_op is not None else True,
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
