import math

from .actual_codec_status import is_fresh_actual
from .correlation import finite_float_or_none, rolling_pearson

def sign_label(value, eps=1e-12):
    value = finite_float_or_none(value)
    if value is None:
        return None
    if abs(value) <= eps:
        return 0
    return 1 if value > 0.0 else -1


def sign_match_value(metric_value, actual_value):
    metric_sign = sign_label(metric_value)
    actual_sign = sign_label(actual_value)
    if metric_sign is None or actual_sign is None:
        return None
    return 1.0 if metric_sign == actual_sign else 0.0


def append_corr_pair(store, key, metric_value, actual_value, max_samples):
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
        sign_match_value(metric_item, actual_item)
        for metric_item, actual_item in pairs
    ]
    sign_values = [value for value in sign_values if value is not None]
    sign_match = (sum(sign_values) / len(sign_values)) if sign_values else None
    return corr, sign_match, len(pairs)


def update_actual_correlation_debug(args, comp_debug, L_com, corr_store):
    actual_value = finite_float_or_none(comp_debug.get("actual_total_bit_percent", None))
    if actual_value is None or not is_fresh_actual(args, comp_debug):
        return {}

    max_samples = max(
        int(getattr(args, "sparsepcgc_corr_window", 100)),
        int(getattr(args, "node_single_gating_window", 100)),
        int(getattr(args, "lr_decay_actual_window", 100)),
        2,
    )
    actual_history = getattr(args, "_actual_delta_history", [])
    actual_history.append(float(actual_value))
    actual_history = actual_history[-max(int(getattr(args, "lr_decay_actual_window", 100)), 2):]
    setattr(args, "_actual_delta_history", actual_history)
    actual_moving_avg = sum(actual_history) / float(max(len(actual_history), 1))
    prev_actual_moving_avg = getattr(args, "_last_actual_moving_avg", None)
    actual_moving_avg_delta = None if prev_actual_moving_avg is None else actual_moving_avg - float(prev_actual_moving_avg)
    setattr(args, "_last_actual_moving_avg", float(actual_moving_avg))
    actual_improvement_threshold = float(getattr(args, "lr_decay_actual_threshold", 0.0))
    actual_moving_avg_improving = bool(actual_moving_avg_delta is not None and actual_moving_avg_delta < -abs(actual_improvement_threshold))
    actual_negative_stable = bool(len(actual_history) >= 2 and actual_moving_avg < 0.0 and max(actual_history[-min(len(actual_history), 5):]) < 0.0)
    lr_decay_allowed_by_actual = bool(
        (not bool(getattr(args, "lr_decay_requires_actual_improvement", False)))
        or actual_moving_avg_improving
        or actual_negative_stable
    )
    metric_values = {
        "surrogate_actual": comp_debug.get("surrogate_pred_bit", None),
        "lcom_actual": L_com,
        "cp_main_actual": comp_debug.get("cp_L_com_main", None),
        "sparsepcgc_aux_actual": comp_debug.get("sparsepcgc_aux_weighted", comp_debug.get("sparsepcgc_aux_loss", None)),
        "lcom_without_sparsepcgc_aux_actual": comp_debug.get("lcom_without_sparsepcgc_aux", None),
        "node_actual": comp_debug.get("node_delta", comp_debug.get("soft_node_percent", None)),
        "single_actual": comp_debug.get("single_delta", comp_debug.get("soft_single_percent", None)),
        "single_delta_actual": comp_debug.get("single_delta", None),
        "point_delta_actual": (
            finite_float_or_none(comp_debug.get("gen_points", None)) - finite_float_or_none(comp_debug.get("gt_points", None))
            if finite_float_or_none(comp_debug.get("gen_points", None)) is not None and finite_float_or_none(comp_debug.get("gt_points", None)) is not None
            else None
        ),
        "occupancy_actual": comp_debug.get("occupancy_nll_delta", comp_debug.get("nll_delta", None)),
    }
    result = {
        "rolling_corr_window": max_samples,
        "rolling_sign_match_window": max_samples,
        "actual_moving_avg": actual_moving_avg,
        "actual_moving_avg_delta": actual_moving_avg_delta,
        "actual_moving_avg_improving": actual_moving_avg_improving,
        "actual_negative_stable": actual_negative_stable,
        "lr_decay_allowed_by_actual": lr_decay_allowed_by_actual,
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
        "node_actual": ("corr_node_actual", "sign_match_node_actual"),
        "single_actual": ("corr_single_actual", "sign_match_single_actual"),
        "single_delta_actual": ("corr_single_delta_actual", "sign_match_single_delta_actual"),
        "point_delta_actual": ("point_delta_actual_corr", "sign_match_point_delta_actual"),
        "occupancy_actual": ("corr_occupancy_actual", "sign_match_occupancy_actual"),
    }
    for key, metric_value in metric_values.items():
        corr, sign_match, _count = append_corr_pair(corr_store, key, metric_value, actual_value, max_samples)
        corr_name, sign_name = key_map[key]
        result[corr_name] = corr
        result[sign_name] = sign_match
        if corr_name == "corr_single_delta_actual":
            setattr(args, "_last_single_delta_actual_corr", corr)
    point_reduction_history = getattr(args, "_point_reduction_actual_improved_history", [])
    point_delta = metric_values.get("point_delta_actual")
    if point_delta is not None and point_delta < 0.0:
        point_reduction_history.append(1.0 if actual_value < 0.0 else 0.0)
        point_reduction_history = point_reduction_history[-max_samples:]
    setattr(args, "_point_reduction_actual_improved_history", point_reduction_history)
    result["point_reduction_actual_improved_ratio"] = (
        sum(point_reduction_history) / float(len(point_reduction_history))
        if point_reduction_history
        else None
    )
    single_positive_history = getattr(args, "_single_delta_positive_history", [])
    single_delta = finite_float_or_none(comp_debug.get("single_delta", None))
    if single_delta is not None:
        single_positive_history.append(1.0 if single_delta > 0.0 else 0.0)
        single_positive_history = single_positive_history[-max_samples:]
    setattr(args, "_single_delta_positive_history", single_positive_history)
    result["single_delta_positive_ratio"] = (
        sum(single_positive_history) / float(len(single_positive_history))
        if single_positive_history
        else None
    )
    if bool(getattr(args, "node_single_actual_gating", True)):
        floor = min(max(float(getattr(args, "node_single_weight_floor", 0.10)), 0.0), 1.0)
        min_corr = float(getattr(args, "node_single_min_corr", 0.10))
        min_sign = float(getattr(args, "node_single_min_sign_match", 0.50))
        node_corr = result.get("corr_node_actual")
        node_sign = result.get("sign_match_node_actual")
        single_corr = result.get("corr_single_actual")
        single_sign = result.get("sign_match_single_actual")
        node_scale = 1.0
        single_scale = 1.0
        reasons = []
        if node_corr is not None and node_sign is not None and (node_corr < min_corr or node_sign < min_sign):
            node_scale = floor
            reasons.append("node_low_actual_alignment")
        if single_corr is not None and single_sign is not None and (single_corr < min_corr or single_sign < min_sign):
            single_scale = floor
            reasons.append("single_low_actual_alignment")
        if node_corr is None or node_sign is None or single_corr is None or single_sign is None:
            reasons.append("warming_up")
        gating_debug = {
            "node_scale": float(node_scale),
            "single_scale": float(single_scale),
            "reason": ",".join(reasons) if reasons else "passed",
        }
    else:
        gating_debug = {"node_scale": 1.0, "single_scale": 1.0, "reason": "disabled"}
    setattr(args, "_node_single_gating_debug", gating_debug)
    result["node_single_gating_reason"] = gating_debug["reason"]
    return result
