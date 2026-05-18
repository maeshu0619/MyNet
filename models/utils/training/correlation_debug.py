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

    max_samples = max(int(getattr(args, "sparsepcgc_corr_window", 100)), 2)
    metric_values = {
        "surrogate_actual": comp_debug.get("surrogate_pred_bit", None),
        "lcom_actual": L_com,
        "cp_main_actual": comp_debug.get("cp_L_com_main", None),
        "sparsepcgc_aux_actual": comp_debug.get("sparsepcgc_aux_weighted", comp_debug.get("sparsepcgc_aux_loss", None)),
        "lcom_without_sparsepcgc_aux_actual": comp_debug.get("lcom_without_sparsepcgc_aux", None),
    }
    result = {
        "rolling_corr_window": max_samples,
        "rolling_sign_match_window": max_samples,
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
    }
    for key, metric_value in metric_values.items():
        corr, sign_match, _count = append_corr_pair(corr_store, key, metric_value, actual_value, max_samples)
        corr_name, sign_name = key_map[key]
        result[corr_name] = corr
        result[sign_name] = sign_match
    return result
