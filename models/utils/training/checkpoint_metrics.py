import math

from .correlation import finite_float_or_none, rolling_pearson
from .correlation_debug import sign_match_value
from .metric_columns import (
    CHECKPOINT_AVG_KEYS,
    )
from .scalar_utils import case_float

def new_checkpoint_metric_sum():
    return {
        "sums": {key: 0.0 for key in CHECKPOINT_AVG_KEYS},
        "counts": {key: 0 for key in CHECKPOINT_AVG_KEYS},
        "corr_pairs": {
            "surrogate_actual": [],
            "lcom_actual": [],
            "cp_main_actual": [],
            "sparsepcgc_aux_actual": [],
            "lcom_without_sparsepcgc_aux_actual": [],
        },
    }


def add_checkpoint_metric(metric_sums, key, value):
    value = case_float(value, float("nan"))
    if not math.isfinite(value):
        return
    metric_sums["sums"][key] = float(metric_sums["sums"].get(key, 0.0)) + value
    metric_sums["counts"][key] = int(metric_sums["counts"].get(key, 0)) + 1


def _is_sparsepcgc_backend(args):
    compress_key = str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "")
    backend = str(getattr(args, "compression_loss_backend", "")).strip().lower()
    return compress_key == "sparsepcgc" or backend.startswith("sparsepcgc_")


def resolve_checkpoint_actual_metric(args, metrics):
    source = str(getattr(args, "checkpoint_actual_source", "auto")).strip().lower()
    if source == "auto":
        source = "full_cloud" if _is_sparsepcgc_backend(args) else "fresh"
    if source == "full_cloud":
        return {
            "source": "full_cloud",
            "delta": finite_float_or_none(metrics.get("full_cloud_actual_delta")),
            "count": int(metrics.get("full_cloud_actual_count") or 0),
            "min_count": max(int(getattr(args, "checkpoint_full_cloud_min_count", 1)), 0),
        }
    return {
        "source": "fresh",
        "delta": finite_float_or_none(metrics.get("fresh_actual_delta")),
        "count": int(metrics.get("fresh_actual_count") or 0),
        "min_count": 1,
    }


def accumulate_checkpoint_metrics(metric_sums, compression_row, operation_row, step_metric_values):
    explicit_values = {
        "total_loss": step_metric_values[0],
        "geom_loss": step_metric_values[1],
        "compression_loss_L_com": compression_row.get("compression_loss_L_com", step_metric_values[2]), # plot列がSurrogate表示でも実際のL_comをcheckpointへ使う
        "single_loss": step_metric_values[6], # actual_compression列追加後のsingle-child損失位置を参照する
        "node_loss": step_metric_values[7], # actual_compression列追加後のnode損失位置を参照する
        "repair_loss": step_metric_values[11], # actual_compression列追加後のrepair損失位置を参照する
        "actual_total_bit_percent_fresh": compression_row.get("actual_total_bit_percent_fresh"),
        "actual_total_bit_percent_cached": compression_row.get("actual_total_bit_percent_cached"),
        "surrogate_pred_bit_percent": compression_row.get("surrogate_pred_bit_percent"),
        "surrogate_abs_bit_error": compression_row.get("surrogate_abs_bit_error"),
        "proxy_delta_percent": compression_row.get("proxy_delta_percent"),
        "added_ratio_percent": operation_row.get("added_ratio_percent"),
        "deleted_ratio_percent": operation_row.get("deleted_ratio_percent"),
        "adjusted_ratio_percent": operation_row.get("adjusted_ratio_percent"),
        "add_prob_mean": operation_row.get("add_prob_mean"),
        "drop_prob_mean": operation_row.get("drop_prob_mean"),
        "hard_move_ratio": operation_row.get("hard_move_ratio"),
        "corr_surrogate_actual": compression_row.get("corr_surrogate_actual"),
        "corr_lcom_actual": compression_row.get("corr_lcom_actual"),
        "corr_cp_main_actual": compression_row.get("corr_cp_main_actual"),
        "corr_sparsepcgc_aux_actual": compression_row.get("corr_sparsepcgc_aux_actual"),
        "corr_lcom_without_sparsepcgc_aux_actual": compression_row.get("corr_lcom_without_sparsepcgc_aux_actual"),
        "sign_match_surrogate_actual": compression_row.get("sign_match_surrogate_actual"),
        "sign_match_lcom_actual": compression_row.get("sign_match_lcom_actual"),
        "sign_match_cp_main_actual": compression_row.get("sign_match_cp_main_actual"),
        "sign_match_sparsepcgc_aux_actual": compression_row.get("sign_match_sparsepcgc_aux_actual"),
        "sign_match_lcom_without_sparsepcgc_aux_actual": compression_row.get("sign_match_lcom_without_sparsepcgc_aux_actual"),
        "lcom_main": compression_row.get("lcom_main"),
        "lcom_aux": compression_row.get("lcom_aux"),
        "lcom_sparsepcgc_aux": compression_row.get("lcom_sparsepcgc_aux"),
        "sparsepcgc_aux_raw": compression_row.get("sparsepcgc_aux_raw"),
        "sparsepcgc_aux_weighted": compression_row.get("sparsepcgc_aux_weighted"),
        "lcom_without_sparsepcgc_aux": compression_row.get("lcom_without_sparsepcgc_aux"),
        "lcom_with_sparsepcgc_aux": compression_row.get("lcom_with_sparsepcgc_aux"),
        "sparsepcgc_active_coord_loss": compression_row.get("sparsepcgc_active_coord_loss"),
        "sparsepcgc_isolated_loss": compression_row.get("sparsepcgc_isolated_loss"),
        "sparsepcgc_entropy_loss": compression_row.get("sparsepcgc_entropy_loss"),
        "sparsepcgc_density_loss": compression_row.get("sparsepcgc_density_loss"),
        "active_coord_delta": operation_row.get("active_coord_delta"),
        "unique_coord_delta": operation_row.get("unique_coord_delta"),
        "add_effective_count": operation_row.get("add_effective_count"),
    }
    for key, value in explicit_values.items():
        add_checkpoint_metric(metric_sums, key, value)

    if bool(compression_row.get("full_cloud_teacher_used", False)):
        full_cloud_value = finite_float_or_none(compression_row.get("full_cloud_actual_percent"))
        if full_cloud_value is not None:
            add_checkpoint_metric(metric_sums, "full_cloud_actual_percent", full_cloud_value)
            if bool(compression_row.get("fresh_actual", False)):
                add_checkpoint_metric(metric_sums, "full_cloud_actual_percent_fresh", full_cloud_value)

    actual_value = finite_float_or_none(compression_row.get("actual_total_bit_percent_fresh"))
    if actual_value is not None:
        pair_sources = {
            "surrogate_actual": compression_row.get("surrogate_pred_bit_percent"),
            "lcom_actual": compression_row.get("compression_loss_L_com"),
            "cp_main_actual": compression_row.get("cp_L_com_main"),
            "sparsepcgc_aux_actual": compression_row.get("sparsepcgc_aux_weighted"),
            "lcom_without_sparsepcgc_aux_actual": compression_row.get("lcom_without_sparsepcgc_aux"),
        }
        for key, metric_value in pair_sources.items():
            metric = finite_float_or_none(metric_value)
            if metric is not None:
                metric_sums["corr_pairs"].setdefault(key, []).append((metric, actual_value))


def checkpoint_average(metric_sums, key):
    count = int(metric_sums["counts"].get(key, 0))
    if count <= 0:
        return None
    return float(metric_sums["sums"].get(key, 0.0)) / float(count)


def checkpoint_corr(metric_sums, key):
    pairs = metric_sums.get("corr_pairs", {}).get(key, [])
    return rolling_pearson(pairs)


def checkpoint_sign_match(metric_sums, key):
    pairs = metric_sums.get("corr_pairs", {}).get(key, [])
    values = [sign_match_value(metric, actual) for metric, actual in pairs]
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def gate_with_relative_reference(value, reference, rel_factor):
    value = case_float(value, float("nan"))
    reference = case_float(reference, float("nan"))
    rel_factor = float(rel_factor)
    if not math.isfinite(value) or rel_factor <= 0.0:
        return True
    if not math.isfinite(reference) or abs(reference) <= 1e-12:
        return True
    return value <= abs(reference) * rel_factor


def gate_with_abs_max(value, abs_max):
    value = case_float(value, float("nan"))
    abs_max = float(abs_max)
    if not math.isfinite(value) or abs_max <= 0.0:
        return True
    return value <= abs_max


def finalize_checkpoint_metrics(args, stage, episode, plot, metric_sums, gate_refs):
    metrics = {
        "episode": int(episode) + 1,
        "stage": str(stage),
        "total_loss": plot.epi_loss_return(),
        "geom_loss": plot.epi_avg[1] if len(plot.epi_avg) > 1 else None,
        "compression_loss_L_com": checkpoint_average(metric_sums, "compression_loss_L_com"), # Surrogate表示列ではなく実際のL_com平均をCheckpoint metricへ使う
        "single_loss": plot.epi_avg[6] if len(plot.epi_avg) > 6 else None, # actual_compression列追加後のsingle-child平均を参照する
        "node_loss": plot.epi_avg[7] if len(plot.epi_avg) > 7 else None, # actual_compression列追加後のnode平均を参照する
        "repair_loss": plot.epi_avg[11] if len(plot.epi_avg) > 11 else None, # actual_compression列追加後のrepair平均を参照する
        "fresh_actual_delta": checkpoint_average(metric_sums, "actual_total_bit_percent_fresh"),
        "fresh_actual_count": int(metric_sums["counts"].get("actual_total_bit_percent_fresh", 0)),
        "cached_actual_delta": checkpoint_average(metric_sums, "actual_total_bit_percent_cached"),
        "cached_actual_count": int(metric_sums["counts"].get("actual_total_bit_percent_cached", 0)),
        "full_cloud_actual_delta": (
            checkpoint_average(metric_sums, "full_cloud_actual_percent_fresh")
            if int(metric_sums["counts"].get("full_cloud_actual_percent_fresh", 0)) > 0
            else checkpoint_average(metric_sums, "full_cloud_actual_percent")
        ),
        "full_cloud_actual_count": (
            int(metric_sums["counts"].get("full_cloud_actual_percent_fresh", 0))
            if int(metric_sums["counts"].get("full_cloud_actual_percent_fresh", 0)) > 0
            else int(metric_sums["counts"].get("full_cloud_actual_percent", 0))
        ),
        "surrogate_pred_bit_percent": checkpoint_average(metric_sums, "surrogate_pred_bit_percent"),
        "surrogate_abs_bit_error": checkpoint_average(metric_sums, "surrogate_abs_bit_error"),
        "proxy_delta_percent": checkpoint_average(metric_sums, "proxy_delta_percent"),
        "corr_surrogate_actual": checkpoint_corr(metric_sums, "surrogate_actual"),
        "corr_lcom_actual": checkpoint_corr(metric_sums, "lcom_actual"),
        "corr_cp_main_actual": checkpoint_corr(metric_sums, "cp_main_actual"),
        "corr_sparsepcgc_aux_actual": checkpoint_corr(metric_sums, "sparsepcgc_aux_actual"),
        "corr_lcom_without_sparsepcgc_aux_actual": checkpoint_corr(metric_sums, "lcom_without_sparsepcgc_aux_actual"),
        "sign_match_surrogate_actual": checkpoint_sign_match(metric_sums, "surrogate_actual"),
        "sign_match_lcom_actual": checkpoint_sign_match(metric_sums, "lcom_actual"),
        "sign_match_cp_main_actual": checkpoint_sign_match(metric_sums, "cp_main_actual"),
        "sign_match_sparsepcgc_aux_actual": checkpoint_sign_match(metric_sums, "sparsepcgc_aux_actual"),
        "sign_match_lcom_without_sparsepcgc_aux_actual": checkpoint_sign_match(metric_sums, "lcom_without_sparsepcgc_aux_actual"),
        "lcom_main": checkpoint_average(metric_sums, "lcom_main"),
        "lcom_aux": checkpoint_average(metric_sums, "lcom_aux"),
        "lcom_sparsepcgc_aux": checkpoint_average(metric_sums, "lcom_sparsepcgc_aux"),
        "sparsepcgc_aux_raw": checkpoint_average(metric_sums, "sparsepcgc_aux_raw"),
        "sparsepcgc_aux_weighted": checkpoint_average(metric_sums, "sparsepcgc_aux_weighted"),
        "lcom_without_sparsepcgc_aux": checkpoint_average(metric_sums, "lcom_without_sparsepcgc_aux"),
        "lcom_with_sparsepcgc_aux": checkpoint_average(metric_sums, "lcom_with_sparsepcgc_aux"),
        "sparsepcgc_active_coord_loss": checkpoint_average(metric_sums, "sparsepcgc_active_coord_loss"),
        "sparsepcgc_isolated_loss": checkpoint_average(metric_sums, "sparsepcgc_isolated_loss"),
        "sparsepcgc_entropy_loss": checkpoint_average(metric_sums, "sparsepcgc_entropy_loss"),
        "sparsepcgc_density_loss": checkpoint_average(metric_sums, "sparsepcgc_density_loss"),
        "added_ratio_percent": checkpoint_average(metric_sums, "added_ratio_percent"),
        "deleted_ratio_percent": checkpoint_average(metric_sums, "deleted_ratio_percent"),
        "adjusted_ratio_percent": checkpoint_average(metric_sums, "adjusted_ratio_percent"),
        "active_coord_delta": checkpoint_average(metric_sums, "active_coord_delta"),
        "unique_coord_delta": checkpoint_average(metric_sums, "unique_coord_delta"),
        "add_effective_count": checkpoint_average(metric_sums, "add_effective_count"),
    }
    stage_refs = gate_refs.setdefault(str(stage), {})
    for key in ("geom_loss", "repair_loss"):
        value = case_float(metrics.get(key), float("nan"))
        if key not in stage_refs and math.isfinite(value):
            stage_refs[key] = value

    geom_ref = stage_refs.get("geom_loss")
    repair_ref = stage_refs.get("repair_loss")
    geom_ok = True
    if bool(getattr(args, "checkpoint_geom_gate", True)):
        geom_ok = gate_with_relative_reference(
            metrics.get("geom_loss"),
            geom_ref,
            float(getattr(args, "checkpoint_geom_rel_factor", 1.5)),
        ) and gate_with_abs_max(
            metrics.get("geom_loss"),
            float(getattr(args, "checkpoint_geom_abs_max", 0.0)),
        )

    repair_ok = gate_with_relative_reference(
        metrics.get("repair_loss"),
        repair_ref,
        float(getattr(args, "checkpoint_repair_rel_factor", 0.0)),
    ) and gate_with_abs_max(
        metrics.get("repair_loss"),
        float(getattr(args, "checkpoint_repair_abs_max", 10.0)),
    )
    node_ok = gate_with_abs_max(metrics.get("node_loss"), float(getattr(args, "checkpoint_node_abs_max", 100.0)))
    single_ok = gate_with_abs_max(metrics.get("single_loss"), float(getattr(args, "checkpoint_single_abs_max", 100.0)))
    op_limit = float(getattr(args, "checkpoint_operation_ratio_max", 100.0))
    operation_ok = True
    if op_limit >= 0.0:
        for key in ("added_ratio_percent", "deleted_ratio_percent", "adjusted_ratio_percent"):
            value = case_float(metrics.get(key), float("nan"))
            if math.isfinite(value) and value > op_limit:
                operation_ok = False
                break

    if not bool(getattr(args, "checkpoint_safety_gate", True)):
        repair_ok = node_ok = single_ok = operation_ok = True

    metrics.update(
        {
            "geometry_ok": bool(geom_ok),
            "safety_ok": bool(geom_ok and repair_ok and node_ok and single_ok and operation_ok),
            "repair_ok": bool(repair_ok),
            "node_ok": bool(node_ok),
            "single_ok": bool(single_ok),
            "operation_ok": bool(operation_ok),
            "geom_reference": geom_ref,
            "repair_reference": repair_ref,
        }
    )
    actual_metric = resolve_checkpoint_actual_metric(args, metrics)
    actual_ok = (
        actual_metric["delta"] is not None
        and int(actual_metric["count"]) >= int(actual_metric["min_count"])
    )
    metrics.update(
        {
            "checkpoint_actual_delta": actual_metric["delta"],
            "checkpoint_actual_count": int(actual_metric["count"]),
            "checkpoint_actual_source": actual_metric["source"],
            "checkpoint_eligible": bool(actual_ok),
            "checkpoint_ineligible_reason": "" if actual_ok else "insufficient_checkpoint_actual",
        }
    )
    return metrics
