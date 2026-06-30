# models/utils/training/log_step.py

import math

import torch

from models.utils.training.correlation import (
    finite_float_or_none,
    push_rolling_correlation,
    format_corr,
    rolling_pearson,
)


def _to_float(value, default=0.0):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _fmt(value, digits=4, default="n/a"):
    try:
        value = _to_float(value, float("nan"))
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    if abs(value) >= 100000.0 or (0.0 < abs(value) < 0.0001):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _fmt_int(value, default="n/a"):
    try:
        value = _to_float(value, float("nan"))
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return str(int(round(value)))


def _first_value(mapping, keys, default=None):
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return default


def _count_delta_text(before, after):
    before_f = _to_float(before, float("nan"))
    after_f = _to_float(after, float("nan"))
    if not (math.isfinite(before_f) and math.isfinite(after_f)):
        return "n/a"
    delta = after_f - before_f
    ratio = 0.0 if abs(before_f) < 1e-12 else 100.0 * delta / before_f
    return f"{_fmt_int(before_f)}->{_fmt_int(after_f)}(d={_fmt(delta, 2)}, {ratio:.2f}%)"


def _compression_debug_fresh_actual(args, comp_debug):
    backend = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    if bool(comp_debug.get("actual_value_is_fresh", False)):
        return True
    if backend.endswith("_surrogate"):
        return bool(comp_debug.get("teacher_refresh", False))
    if "_actual" in backend:
        return (
            not bool(comp_debug.get("actual_codec_disabled_during_train", False))
            and not bool(comp_debug.get("actual_codec_skipped_by_interval", False))
            and not bool(comp_debug.get("actual_codec_fallback_to_proxy", False))
        )
    return False


def _resolve_actual_compression_loss(comp_debug):
    return _first_value(
        comp_debug,
        (
            "actual_total_bit_percent",
            "actual_train_objective_percent",
            "actual_bit_percent",
        ),
        float("nan"),
    )


def _resolve_actual_bits_before_after(comp_debug):
    gt_bits = _first_value(
        comp_debug,
        ("gt_actual_bit", "before_bits", "actual_gt_bits"),
        float("nan"),
    )
    gen_bits = _first_value(
        comp_debug,
        (
            "gen_total_bit_with_edit_record",
            "actual_total_bits",
            "gen_actual_bit",
            "after_bits",
        ),
        float("nan"),
    )
    return _to_float(gt_bits, float("nan")), _to_float(gen_bits, float("nan"))


def log_actual_compression_loss(writer, comp_debug):
    if writer is None or not hasattr(writer, "write"):
        return
    comp_debug = comp_debug if isinstance(comp_debug, dict) else {}
    actual_compression_loss = _resolve_actual_compression_loss(comp_debug)
    actual_compression_loss_raw = _first_value(
        comp_debug,
        ("compression_loss_raw", "actual_bit_percent_raw", "actual_raw_percent", "actual_bit_percent"),
        float("nan"),
    )
    actual_compression_loss_used = _first_value(
        comp_debug,
        ("compression_loss_used", "actual_bit_percent_used_for_loss", "actual_forward_value", "actual_bit_percent"),
        float("nan"),
    )
    source = str(comp_debug.get("actual_value_source", "unknown"))
    freshness = "fresh" if bool(comp_debug.get("actual_value_is_fresh", False)) else "stale"
    extra = ""
    if not math.isfinite(_to_float(actual_compression_loss, float("nan"))):
        if bool(comp_debug.get("actual_codec_fallback_to_proxy", False)):
            extra = " (proxy_fallback)"
        elif bool(comp_debug.get("actual_codec_skipped_by_interval", False)):
            extra = " (skipped_by_interval)"
        elif source == "local_proxy":
            extra = " (local_proxy)"
        elif source in {"target_cache", "stale_target"}:
            extra = " (cached_target)"
    writer.write(
        f"Actual Compression Loss: {_fmt(actual_compression_loss, 6)} "
        f"(raw={_fmt(actual_compression_loss_raw, 6)}, used={_fmt(actual_compression_loss_used, 6)}) "
        f"[source={source}, {freshness}]{extra}"
    )


def log_step_loss(
    writer,
    step,
    num_steps,
    L,
    L_geom,
    L_com,
    L_com_objective,
    L_attr,
    L_policy,
    L_actuator,
    Lp_out,
    La_fit,
    La_rep,
    L_discrete_policy,
    loss_bit,
    loss_single,
    loss_nodes,
):
    writer.write(
        f"StepLoss step={step + 1}/{num_steps}: "
        f"L={_to_float(L):.6f}, "
        f"L_geom={_to_float(L_geom):.6f}, "
        f"L_com={_to_float(L_com):.6f}, "
        f"L_com_obj={_to_float(L_com_objective):.6f}, "
        f"L_attr={_to_float(L_attr):.6f}, "
        f"L_policy={_to_float(L_policy):.6f}, "
        f"L_actuator={_to_float(L_actuator):.6f}, "
        f"Lp_out={_to_float(Lp_out):.6f}, "
        f"La_fit={_to_float(La_fit):.6f}, "
        f"La_rep={_to_float(La_rep):.6f}, "
        f"L_discrete_policy={_to_float(L_discrete_policy):.6f}, "
        f"bit={_to_float(loss_bit):.6f}, "
        f"single={_to_float(loss_single):.6f}, "
        f"nodes={_to_float(loss_nodes):.6f}"
    )


def log_compression_stats(writer, step, num_steps, comp_debug):
    gt_actual_bit, gen_actual_bit = _resolve_actual_bits_before_after(comp_debug)
    actual_total_bit_percent = _first_value(
        comp_debug,
        ("actual_total_bit_percent", "actual_train_objective_percent", "actual_bit_percent"),
        float("nan"),
    )
    actual_raw_percent = _first_value(
        comp_debug,
        ("actual_raw_percent", "actual_total_bit_percent"),
        float("nan"),
    )
    actual_used_percent = _first_value(
        comp_debug,
        ("actual_bit_percent_used_for_loss", "actual_forward_value", "actual_total_bit_percent"),
        float("nan"),
    )
    compression_loss_raw = _first_value(
        comp_debug,
        ("compression_loss_raw", "actual_bit_percent_raw", "actual_raw_percent", "actual_total_bit_percent"),
        float("nan"),
    )
    compression_loss_used = _first_value(
        comp_debug,
        ("compression_loss_used", "actual_bit_percent_used_for_loss", "actual_forward_value", "actual_total_bit_percent"),
        float("nan"),
    )
    log_actual_compression_loss(writer, comp_debug)
    writer.write(
        f"CompressionStats step={step + 1}/{num_steps}: "
        f"actual_bit:{_fmt(gt_actual_bit, 6)}"
        f"->{_fmt(gen_actual_bit, 6)}, "
        f"actual_bit_percent={_fmt(actual_total_bit_percent, 6)}, "
        f"actual_raw_percent={_fmt(actual_raw_percent, 6)}, "
        f"actual_used_for_loss={_fmt(actual_used_percent, 6)}, "
        f"compression_loss_raw={_fmt(compression_loss_raw, 6)}, "
        f"compression_loss_used={_fmt(compression_loss_used, 6)}, "
        f"policy_actual_noop_guard_used={bool(comp_debug.get('policy_actual_noop_guard_used', False))}, "
        f"policy_actual_noop_guard_replaced_in_loss={bool(comp_debug.get('policy_actual_noop_guard_replaced_in_loss', False))}, "
        f"actual_oracle_force_no_edit_used={bool(comp_debug.get('actual_oracle_force_no_edit_used', False))}, "
        f"edit_record_bits={float(comp_debug.get('actual_edit_record_bits', 0.0)):.3f}, "
        f"codec_points={int(comp_debug.get('gt_points', 0))}->{int(comp_debug.get('gen_points', 0))}, "
        f"unique_coords={int(comp_debug.get('gt_unique_coord_count', 0))}->{int(comp_debug.get('gen_unique_coord_count', 0))}, "
        f"actual_encode_time={float(comp_debug.get('actual_encode_time_total', 0.0)):.4f}s"
        f"(before={float(comp_debug.get('gt_actual_encode_time', 0.0)):.4f}, "
        f"after={float(comp_debug.get('gen_actual_encode_time', 0.0)):.4f}), "
        f"objective={float(comp_debug.get('compression_objective', comp_debug.get('total_bit', 0.0))):.6f}, "
        f"surrogate_bit_percent={float(comp_debug.get('surrogate_pred_bit', 0.0)):.6f}, "
        f"surrogate_target_bit={float(comp_debug.get('surrogate_target_bit', 0.0)):.6f}, "
        f"surrogate_abs_bit_error={float(comp_debug.get('surrogate_abs_bit_error', 0.0)):.6f}, "
        f"surrogate_signed_bit_error={float(comp_debug.get('surrogate_signed_bit_error', 0.0)):.6f}, "
        f"surrogate_train_loss={float(comp_debug.get('surrogate_train_loss', 0.0)):.6f}, "
        f"actual_forward={float(comp_debug.get('actual_forward_value', 0.0)):.6f}, "
        f"actual_forward_raw={float(comp_debug.get('actual_forward_raw_value', comp_debug.get('actual_bit_percent', 0.0))):.6f}, "
        f"actual_forward_clamped={bool(comp_debug.get('actual_forward_clamped', False))}, "
        f"lcom_main={float(comp_debug.get('compression_main_loss', 0.0)):.6f}, "
        f"lcom_aux={float(comp_debug.get('compression_aux_loss', 0.0)):.6f}, "
        f"lcom_sparsepcgc_aux={float(comp_debug.get('sparsepcgc_aux_loss', 0.0)):.6f}, "
        f"sparsepcgc_aux_weighted={float(comp_debug.get('sparsepcgc_aux_weighted', 0.0)):.6f}, "
        f"sparsepcgc_aux_used_for_backprop={bool(comp_debug.get('sparsepcgc_aux_used_for_backprop', False))}, "
        f"sparsepcgc_aux_weight_effective={float(comp_debug.get('sparsepcgc_aux_weight_effective', 0.0)):.6f}, "
        f"sparsepcgc_aux_gate={comp_debug.get('sparsepcgc_aux_gating_reason', '')}, "
        f"grad_source={comp_debug.get('grad_source', '')}, "
        f"surrogate_grad={float(comp_debug.get('surrogate_loss_for_grad', 0.0)):.6f}, "
        f"proxy_grad={float(comp_debug.get('proxy_aux_for_grad', 0.0)):.6f}, "
        f"lcom_without_sparse={float(comp_debug.get('lcom_without_sparsepcgc_aux', 0.0)):.6f}, "
        f"soft_node={float(comp_debug.get('soft_node_percent', 0.0)):.6f}, "
        f"soft_single={float(comp_debug.get('soft_single_percent', 0.0)):.6f}, "
        f"octree_node:{float(comp_debug.get('gt_octree_node', 0.0)):.1f}->{float(comp_debug.get('gen_octree_node', 0.0)):.1f}, "
        f"octree_single:{float(comp_debug.get('gt_octree_single', 0.0)):.1f}->{float(comp_debug.get('gen_octree_single', 0.0)):.1f}, "
        f"occupancy_entropy[pred_delta={_to_float(comp_debug.get('occupancy_entropy_delta', float('nan')), float('nan')):.6f}, "
        f"actual={_to_float(comp_debug.get('actual_occupancy_entropy_before', float('nan')), float('nan')):.6f}"
        f"->{_to_float(comp_debug.get('actual_occupancy_entropy_after', float('nan')), float('nan')):.6f}, "
        f"actual_delta={_to_float(comp_debug.get('actual_occupancy_entropy_delta', float('nan')), float('nan')):.6f}], "
        f"occupancy_nll[pred_delta={_to_float(comp_debug.get('occupancy_nll_delta', float('nan')), float('nan')):.6f}, "
        f"actual_delta={_to_float(comp_debug.get('actual_occupancy_nll_delta', float('nan')), float('nan')):.6f}], "
        f"lowprob_occupancy[pred={_to_float(comp_debug.get('lowprob_occupancy_ratio', float('nan')), float('nan')):.6f}, "
        f"actual={_to_float(comp_debug.get('actual_lowprob_occupancy_ratio_after', float('nan')), float('nan')):.6f}], "
        f"sparsepcgc_candidate_occ[debug={bool(comp_debug.get('sparsepcgc_occupancy_debug_available', False))}, "
        f"candidates={int(_to_float(comp_debug.get('sparsepcgc_candidate_count_after', 0), 0))}, "
        f"label_ratio={_to_float(comp_debug.get('sparsepcgc_actual_occupancy_label_ratio_after', float('nan')), float('nan')):.6f}, "
        f"nll={_to_float(comp_debug.get('sparsepcgc_pred_occupancy_nll_after', float('nan')), float('nan')):.6f}, "
        f"bits_delta={_to_float(comp_debug.get('sparsepcgc_estimated_occupancy_bits_delta', float('nan')), float('nan')):.6f}, "
        f"prob_true_mean={_to_float(comp_debug.get('sparsepcgc_prob_true_mean_after', float('nan')), float('nan')):.6f}, "
        f"low_true={_to_float(comp_debug.get('sparsepcgc_prob_true_low_ratio_after', float('nan')), float('nan')):.6f}], "
        f"sparsepcgc_exact_occ[mode={comp_debug.get('sparsepcgc_exact_teacher_mode', '')}, "
        f"uses_full={bool(comp_debug.get('exact_teacher_uses_full_context', False))}, "
        f"candidates={int(_to_float(comp_debug.get('sparsepcgc_exact_candidate_count_after', comp_debug.get('sparsepcgc_exact_candidate_count', 0)), 0))}, "
        f"nll={_to_float(comp_debug.get('sparsepcgc_exact_occupancy_nll_after', comp_debug.get('sparsepcgc_exact_occupancy_nll', float('nan'))), float('nan')):.6f}, "
        f"bits={_to_float(comp_debug.get('sparsepcgc_exact_estimated_bits_after', comp_debug.get('sparsepcgc_exact_estimated_bits', float('nan'))), float('nan')):.6f}, "
        f"bce_bits={_to_float(comp_debug.get('sparsepcgc_exact_bce_bits_after', comp_debug.get('sparsepcgc_exact_bce_bits', float('nan'))), float('nan')):.6f}, "
        f"bits_match={bool(comp_debug.get('exact_bits_match', False))}, "
        f"fallback={comp_debug.get('exact_teacher_fallback_reason', '')}], "
        f"actual_occupancy_predictability={_to_float(comp_debug.get('actual_occupancy_predictability_after', float('nan')), float('nan')):.6f}, "
        f"teacher_codec={comp_debug.get('teacher_codec', 'unknown')}, "
        f"teacher_refresh={bool(comp_debug.get('teacher_refresh', False))}, "
        f"teacher_mode={comp_debug.get('teacher_mode', 'unknown')}, "
        f"teacher_type={comp_debug.get('teacher_type', 'unknown')}, "
        f"full_cloud_teacher_used={bool(comp_debug.get('full_cloud_teacher_used', False))}, "
        f"target_clamped={bool(comp_debug.get('target_clamped', False))}, "
        f"target_clamp_rate={float(comp_debug.get('target_clamp_rate', 0.0)):.6f}, "
        f"teacher_cache_hit={comp_debug.get('teacher_cache_hit', 'unknown')}, "
        f"teacher_target_age={int(comp_debug.get('teacher_target_age', 0))}, "
        f"actual_value_source={comp_debug.get('actual_value_source', 'unknown')}, "
        f"actual_value_is_fresh={bool(comp_debug.get('actual_value_is_fresh', False))}, "
        f"teacher_interval={int(comp_debug.get('teacher_refresh_interval', 0))}, "
        f"actual_eval_interval={int(comp_debug.get('actual_eval_interval', 0))}, "
        f"reuse_last_target={bool(comp_debug.get('reuse_last_target', True))}, "
        f"replay={int(comp_debug.get('surrogate_replay_size', 0))}, "
        f"replay_samples={int(comp_debug.get('surrogate_replay_sample_count', 0))}, "
        f"replay_age={float(comp_debug.get('replay_age', 0.0)):.3f}, "
        f"teacher_policy={comp_debug.get('teacher_refresh_policy', 'unknown')}"
    )


def log_compression_train_debug(writer, step, num_steps, args, comp_debug, loss, L_com):
    before_node = float(comp_debug.get("gt_octree_node", comp_debug.get("gt_node_abs", 0.0)))
    after_node = float(comp_debug.get("gen_octree_node", comp_debug.get("gen_node_abs", 0.0)))
    before_single = float(comp_debug.get("gt_octree_single", comp_debug.get("gt_single_abs", 0.0)))
    after_single = float(comp_debug.get("gen_octree_single", comp_debug.get("gen_single_abs", 0.0)))
    rate_before = float(comp_debug.get("rate_proxy_before", comp_debug.get("gt_actual_bit", comp_debug.get("gt_bit_abs", 0.0))))
    rate_after = float(comp_debug.get("rate_proxy_after", comp_debug.get("gen_actual_bit", comp_debug.get("gen_bit_abs", 0.0))))
    terms = getattr(loss, "last_compression_terms", {}) or {}

    writer.write(
        f"CompressionTrainDebug step={step + 1}/{num_steps}: "
        f"before_points={int(comp_debug.get('gt_points', 0))}, "
        f"after_points={int(comp_debug.get('gen_points', 0))}, "
        f"before_node={before_node:.3f}, after_node={after_node:.3f}, "
        f"node_delta={after_node - before_node:.3f}, "
        f"before_single={before_single:.3f}, after_single={after_single:.3f}, "
        f"single_delta={after_single - before_single:.3f}, "
        f"rate_proxy_before={rate_before:.6f}, rate_proxy_after={rate_after:.6f}, "
        f"rate_proxy_delta={float(comp_debug.get('rate_proxy_delta', comp_debug.get('total_bit', 0.0))):.6f}, "
        f"compression_proxy_input_mode={comp_debug.get('compression_proxy_input_mode', '')}, "
        f"L_com_source={comp_debug.get('L_com_source', '')}, "
        f"loss_nodes_source={comp_debug.get('loss_nodes_source', '')}, "
        f"loss_single_source={comp_debug.get('loss_single_source', '')}, "
        f"prebuilt_node_count_used={float(comp_debug.get('prebuilt_node_count_used', 0.0)):.3f}, "
        f"prebuilt_single_child_count_used={float(comp_debug.get('prebuilt_single_child_count_used', 0.0)):.3f}, "
        f"actual_train_objective_delta={float(comp_debug.get('actual_train_objective_percent', comp_debug.get('actual_total_bit_percent', comp_debug.get('total_bit', 0.0)))):.6f}, "
        f"uniform_noise[enabled={bool(comp_debug.get('uniform_noise_enabled', False))}, "
        f"applied={bool(comp_debug.get('uniform_noise_applied', False))}, "
        f"delta={float(comp_debug.get('uniform_noise_delta', 0.0)):.6g}, "
        f"mean_abs={float(comp_debug.get('uniform_noise_mean_abs', 0.0)):.6g}], "
        f"actual_codec[disabled={bool(comp_debug.get('actual_codec_disabled_during_train', False))}, "
        f"skipped_by_interval={bool(comp_debug.get('actual_codec_skipped_by_interval', False))}, "
        f"fallback_to_proxy={bool(comp_debug.get('actual_codec_fallback_to_proxy', False))}, "
        f"fresh={bool(comp_debug.get('actual_value_is_fresh', False))}, "
        f"source={comp_debug.get('actual_value_source', 'unknown')}, "
        f"error={str(comp_debug.get('actual_codec_error', ''))[:160]}], "
        f"weights[w_com={float(getattr(args, 'w_com', 0.0)):.6g}, "
        f"com_bit={float(getattr(args, 'com_bit', 0.0)):.6g}, "
        f"com_single={float(getattr(args, 'com_sin', 0.0)):.6g}, "
        f"com_node={float(getattr(args, 'com_node', 0.0)):.6g}, "
        f"com_sparsepcgc={float(getattr(args, 'com_sparsepcgc', 0.0)):.6g}, "
        f"aux_node={float(getattr(args, 'compression_surrogate_aux_node_weight', 0.0)):.6g}, "
        f"aux_single={float(getattr(args, 'compression_surrogate_aux_single_weight', 0.0)):.6g}], "
        f"terms[bit={_to_float(terms.get('bit', L_com.new_zeros(()))):.6f}, "
        f"single={_to_float(terms.get('single', L_com.new_zeros(()))):.6f}, "
        f"node={_to_float(terms.get('node', L_com.new_zeros(()))):.6f}, "
        f"sparsepcgc={_to_float(terms.get('sparsepcgc', L_com.new_zeros(()))):.6f}], "
        f"teacher_ste[forward={float(comp_debug.get('actual_forward_value', 0.0)):.6f}, "
        f"surrogate_grad={float(comp_debug.get('surrogate_loss_for_grad', 0.0)):.6f}, "
        f"proxy_grad={float(comp_debug.get('proxy_aux_for_grad', 0.0)):.6f}, "
        f"source={comp_debug.get('grad_source', '')}], "
        f"corr[surrogate={_to_float(comp_debug.get('corr_surrogate_actual', float('nan')), float('nan')):.6f}, "
        f"lcom={_to_float(comp_debug.get('corr_lcom_actual', float('nan')), float('nan')):.6f}, "
        f"cp_main={_to_float(comp_debug.get('corr_cp_main_actual', float('nan')), float('nan')):.6f}, "
        f"sparse_aux={_to_float(comp_debug.get('corr_sparsepcgc_aux_actual', float('nan')), float('nan')):.6f}, "
        f"lcom_no_sparse={_to_float(comp_debug.get('corr_lcom_without_sparsepcgc_aux_actual', float('nan')), float('nan')):.6f}]"
    )

    return before_node, after_node, before_single, after_single


def log_codec_actual_correlation(
    writer,
    step,
    num_steps,
    args,
    comp_debug,
    codec_actual_metric_pairs,
    before_node,
    after_node,
    before_single,
    after_single,
):
    actual_delta_for_corr = finite_float_or_none(
        comp_debug.get("actual_total_bit_percent", comp_debug.get("total_bit", None))
    )

    if actual_delta_for_corr is None:
        return
    if not _compression_debug_fresh_actual(args, comp_debug):
        writer.write(
            f"CodecActualCorrelation step={step + 1}/{num_steps}: "
            "skipped because actual delta is cached or proxy-derived."
        )
        return

    codec_for_corr = str(
        comp_debug.get("teacher_codec", getattr(args, "compress", "unknown"))
    ).lower()
    max_corr_samples = max(int(getattr(args, "sparsepcgc_corr_window", 100)), 2)

    corr_metrics = {
        "proxy_delta": comp_debug.get("rate_proxy_delta", comp_debug.get("total_bit", None)),
        "node_delta": after_node - before_node,
        "single_delta": after_single - before_single,
        "point_delta": int(comp_debug.get("gen_points", 0)) - int(comp_debug.get("gt_points", 0)),
        "unique_coord_delta": int(comp_debug.get("gen_unique_coord_count", 0)) - int(comp_debug.get("gt_unique_coord_count", 0)),
        "surrogate_pred": comp_debug.get("surrogate_pred_bit", None),
        "sparsepcgc_aux": comp_debug.get("sparsepcgc_aux_weighted", comp_debug.get("sparsepcgc_aux_loss", None)),
        "lcom_without_sparsepcgc_aux": comp_debug.get("lcom_without_sparsepcgc_aux", None),
    }

    if "sparsepcgc_before_active_coords" in comp_debug:
        corr_metrics.update({
            "sparse_active_delta": comp_debug.get("sparsepcgc_active_coord_delta", None),
            "sparse_occupied_delta": comp_debug.get("sparsepcgc_occupied_voxel_delta", None),
            "sparse_isolated_delta": comp_debug.get("sparsepcgc_isolated_delta", None),
            "sparse_density_var_delta": comp_debug.get("sparsepcgc_local_density_var_delta", None),
            "sparse_density_delta": comp_debug.get("sparsepcgc_sparse_density_delta", None),
            "sparse_neighbor_delta": comp_debug.get("sparsepcgc_mean_neighbors_delta", None),
            "sparse_duplicate_delta": comp_debug.get("sparsepcgc_duplicate_delta", None),
        })

    corr_chunks = []
    for metric_name, metric_value in corr_metrics.items():
        corr, count = push_rolling_correlation(
            codec_actual_metric_pairs,
            f"{codec_for_corr}:{metric_name}",
            metric_value,
            actual_delta_for_corr,
            max_corr_samples,
        )
        corr_chunks.append(f"{metric_name}={format_corr(corr, count)}")

    writer.write(
        f"CodecActualCorrelation step={step + 1}/{num_steps}: "
        f"codec={codec_for_corr}, "
        f"actual_delta_percent={actual_delta_for_corr:.6f}, "
        + ", ".join(corr_chunks)
    )


def log_sparsepcgc_train_debug(
    writer,
    step,
    num_steps,
    args,
    comp_debug,
    sparsepcgc_proxy_actual_pairs,
):
    if "sparsepcgc_before_active_coords" not in comp_debug:
        return

    actual_delta_for_corr = finite_float_or_none(
        comp_debug.get("actual_total_bit_percent", None)
    )
    proxy_delta_for_corr = finite_float_or_none(
        comp_debug.get("rate_proxy_delta", comp_debug.get("total_bit", None))
    )

    if (
        actual_delta_for_corr is not None
        and proxy_delta_for_corr is not None
        and _compression_debug_fresh_actual(args, comp_debug)
    ):
        sparsepcgc_proxy_actual_pairs.append((proxy_delta_for_corr, actual_delta_for_corr))
        max_corr_samples = max(int(getattr(args, "sparsepcgc_corr_window", 100)), 2)
        if len(sparsepcgc_proxy_actual_pairs) > max_corr_samples:
            del sparsepcgc_proxy_actual_pairs[:-max_corr_samples]

    proxy_actual_corr = rolling_pearson(sparsepcgc_proxy_actual_pairs)
    proxy_actual_corr_text = "n/a" if proxy_actual_corr is None else f"{proxy_actual_corr:.6f}"

    writer.write(
        f"SparsePCGCTrainDebug step={step + 1}/{num_steps}: "
        f"actual_bits={float(comp_debug.get('gt_actual_bit', float('nan'))):.6f}"
        f"->{float(comp_debug.get('gen_actual_bit', float('nan'))):.6f}, "
        f"actual_delta_percent={float(comp_debug.get('actual_total_bit_percent', comp_debug.get('total_bit', 0.0))):.6f}, "
        f"proxy_actual_corr={proxy_actual_corr_text}, "
        f"unique={int(comp_debug.get('gt_unique_coord_count', 0))}"
        f"->{int(comp_debug.get('gen_unique_coord_count', 0))}, "
        f"active={int(comp_debug.get('sparsepcgc_before_active_coords', 0))}"
        f"->{int(comp_debug.get('sparsepcgc_after_active_coords', 0))}, "
        f"active_delta={int(comp_debug.get('sparsepcgc_active_coord_delta', 0))}, "
        f"isolated={int(comp_debug.get('sparsepcgc_before_isolated_voxels', 0))}"
        f"->{int(comp_debug.get('sparsepcgc_after_isolated_voxels', 0))}, "
        f"isolated_delta={int(comp_debug.get('sparsepcgc_isolated_delta', 0))}, "
        f"density_var={float(comp_debug.get('sparsepcgc_before_local_density_var', 0.0)):.6f}"
        f"->{float(comp_debug.get('sparsepcgc_after_local_density_var', 0.0)):.6f}, "
        f"mean_neighbors={float(comp_debug.get('sparsepcgc_before_mean_neighbors', 0.0)):.6f}"
        f"->{float(comp_debug.get('sparsepcgc_after_mean_neighbors', 0.0)):.6f}, "
        f"aux={float(comp_debug.get('sparsepcgc_aux_loss', 0.0)):.6f}, "
        f"aux_used_for_backprop={bool(comp_debug.get('sparsepcgc_aux_used_for_backprop', False))}, "
        f"aux_gate={comp_debug.get('sparsepcgc_aux_gating_reason', '')}, "
        f"aux_corr_roll={_to_float(comp_debug.get('corr_sparsepcgc_aux_actual_rolling', float('nan')), float('nan')):.6f}, "
        f"aux_sign_roll={_to_float(comp_debug.get('sign_match_sparsepcgc_aux_actual_rolling', float('nan')), float('nan')):.6f}, "
        f"active_loss={float(comp_debug.get('sparsepcgc_active_coord_loss', 0.0)):.6f}, "
        f"isolated_proxy={float(comp_debug.get('sparsepcgc_isolated_proxy_loss', 0.0)):.6f}, "
        f"entropy_proxy={float(comp_debug.get('sparsepcgc_entropy_proxy_loss', 0.0)):.6f}, "
        f"density_proxy={float(comp_debug.get('sparsepcgc_density_proxy_loss', 0.0)):.6f}"
    )


def log_compact_step_summary(
    writer,
    step,
    num_steps,
    args,
    loss_obj,
    comp_debug,
    structure_debug,
    edit_stats,
    *,
    L,
    L_geom,
    L_com,
    L_com_objective,
    L_attr,
    L_policy,
    L_actuator,
    loss_bit,
    loss_single,
    loss_nodes,
    stage_factors=None,
    step_completed=None,
):
    if writer is None or not hasattr(writer, "write"):
        return
    comp_debug = comp_debug if isinstance(comp_debug, dict) else {}
    structure_debug = structure_debug if isinstance(structure_debug, dict) else {}
    edit_stats = edit_stats if isinstance(edit_stats, dict) else {}
    terms = getattr(loss_obj, "last_compression_terms", {}) or {}
    geom_debug = getattr(loss_obj, "last_geometry_debug", {}) or {}
    stage_factors = stage_factors or {}

    full_cloud = bool(
        str(comp_debug.get("actual_scope", "")).strip().lower() == "full_cloud"
        or bool(comp_debug.get("full_cloud_actual_primary_used", False))
        or bool(comp_debug.get("full_cloud_teacher_used", False))
        or bool(comp_debug.get("full_cloud_anchor_shadow_train_used", False))
    )
    eval_scope = str(
        comp_debug.get(
            "actual_oracle_eval_scope",
            comp_debug.get("teacher_scope", comp_debug.get("actual_scope", "unknown")),
        )
    )
    source = str(
        comp_debug.get(
            "policy_action_source",
            comp_debug.get("actual_value_source", comp_debug.get("L_com_source", "")),
        )
    )
    writer.write(
        f"StepScope {step + 1}/{num_steps}: "
        f"FullCloud={full_cloud}, scope={eval_scope}, source={source}"
    )

    weighted_geom = _to_float(L_geom, 0.0) * _to_float(stage_factors.get("geom", 1.0), 1.0) * _to_float(getattr(args, "w_geom", 1.0), 1.0)
    weighted_com = _to_float(L_com_objective, 0.0) * _to_float(stage_factors.get("com", 1.0), 1.0) * _to_float(getattr(args, "w_com", 1.0), 1.0)
    weighted_attr = _to_float(L_attr, 0.0) * _to_float(stage_factors.get("attr", 1.0), 1.0) * _to_float(getattr(args, "w_attr", 1.0), 1.0)
    weighted_policy = _to_float(L_policy, 0.0) * _to_float(stage_factors.get("policy", 1.0), 1.0) * _to_float(getattr(args, "w_policy", 1.0), 1.0)
    weighted_act = _to_float(L_actuator, 0.0) * _to_float(stage_factors.get("repair", 1.0), 1.0) * _to_float(getattr(args, "w_actuator", 1.0), 1.0)
    raw_lcom = _first_value(
        comp_debug,
        (
            "compression_loss_L_com",
            "compression_objective",
            "actual_total_bit_percent",
            "actual_bit_percent",
            "compression_forward_teacher_percent",
            "total_bit",
        ),
        _to_float(L_com, 0.0),
    )
    raw_geom = _first_value(geom_debug, ("value", "hard", "weighted"), _to_float(L_geom, 0.0))
    sparse_raw = _first_value(
        comp_debug,
        ("sparsepcgc_aux_loss", "sparsepcgc_aux_weighted", "compression_aux_loss"),
        terms.get("sparsepcgc", 0.0),
    )
    writer.write(
        f"StepLoss {step + 1}/{num_steps}: "
        f"weighted[L={_fmt(L)}, geom={_fmt(weighted_geom)}, com={_fmt(weighted_com)}, "
        f"attr={_fmt(weighted_attr)}, policy={_fmt(weighted_policy)}, act={_fmt(weighted_act)}] "
        f"raw[geom={_fmt(raw_geom)}, L_com={_fmt(raw_lcom)}, bit={_fmt(loss_bit)}, "
        f"single={_fmt(loss_single)}, node={_fmt(loss_nodes)}, sparse={_fmt(sparse_raw)}, "
        f"attr={_fmt(L_attr)}, policy={_fmt(L_policy)}, act={_fmt(L_actuator)}]"
    )
    if str(comp_debug.get("loss_mode", "")).strip().lower() == "compression_primary":
        writer.write(
            f"StepBalance {step + 1}/{num_steps}: "
            f"main={_fmt(comp_debug.get('cp_main_block', float('nan')))}"
            f"(src={comp_debug.get('cp_main_source', 'n/a')}), "
            f"cp_aux={_fmt(comp_debug.get('cp_aux_block_scaled', float('nan')))}"
            f"(raw={_fmt(comp_debug.get('cp_aux_block_raw', float('nan')))}, "
            f"scale={_fmt(comp_debug.get('cp_aux_balance_scale', float('nan')))}, "
            f"target={_fmt(comp_debug.get('cp_aux_target_ratio', float('nan')))}, "
            f"dom={comp_debug.get('cp_aux_balance_dominant', 'n/a')}), "
            f"tail={_fmt(comp_debug.get('cp_support_tail_scaled', float('nan')))}"
            f"(raw={_fmt(comp_debug.get('cp_support_tail_raw', float('nan')))}, "
            f"scale={_fmt(comp_debug.get('cp_support_tail_scale', float('nan')))}, "
            f"target={_fmt(comp_debug.get('cp_support_tail_target_ratio', float('nan')))}, "
            f"dom={comp_debug.get('cp_support_tail_dominant', 'n/a')}), "
            f"corr={_fmt(comp_debug.get('cp_support_correction_scaled', float('nan')))}"
            f"(raw={_fmt(comp_debug.get('cp_support_correction_raw', float('nan')))}, "
            f"scale={_fmt(comp_debug.get('cp_support_correction_scale', float('nan')))}), "
            f"support_total={_fmt(comp_debug.get('cp_support_total_scaled', float('nan')))}, "
            f"support/main={_fmt(comp_debug.get('cp_support_total_ratio_to_main', float('nan')))}, "
            f"dominant={comp_debug.get('cp_support_dominant', 'n/a')}"
        )

    log_actual_compression_loss(writer, comp_debug)

    gt_bits, oracle_gen_bits = _resolve_actual_bits_before_after(comp_debug)
    policy_mine_bits = _first_value(
        comp_debug,
        (
            "policy_final_full_cloud_total_bit_with_edit_record",
            "policy_final_full_cloud_gen_bit",
            "gen_total_bit_with_edit_record",
            "actual_total_bits",
            "gen_actual_bit",
        ),
        float("nan"),
    )
    oracle_mine_bits = oracle_gen_bits
    if (
        (not math.isfinite(_to_float(policy_mine_bits, float("nan"))))
        and not bool(comp_debug.get("oracle_full_cloud_override_used", False))
    ):
        policy_mine_bits = oracle_mine_bits
    train_objective = _first_value(
        comp_debug,
        ("compression_loss_L_com", "actual_train_objective_percent", "actual_total_bit_percent", "actual_bit_percent", "total_bit"),
        float("nan"),
    )
    policy_actual = _first_value(
        comp_debug,
        ("policy_actual_percent", "policy_final_full_cloud_actual_bit_percent"),
        float("nan"),
    )
    oracle_actual = _first_value(
        comp_debug,
        ("oracle_teacher_actual_percent", "oracle_full_cloud_actual_bit_percent"),
        float("nan"),
    )
    raw_delta = _first_value(comp_debug, ("actual_raw_percent", "actual_oracle_raw_percent"), train_objective)
    edit_bits = _first_value(comp_debug, ("actual_edit_record_bits", "actual_oracle_edit_record_bits"), 0.0)
    writer.write(
        f"StepBits {step + 1}/{num_steps}: "
        f"GT={_fmt(gt_bits, 1)}, PolicyMINE={_fmt(policy_mine_bits, 1)}, "
        f"OracleMINE={_fmt(oracle_mine_bits, 1)}, "
        f"train_objective={_fmt(train_objective)}, policy_actual={_fmt(policy_actual)}, "
        f"oracle_actual={_fmt(oracle_actual)}, raw={_fmt(raw_delta)}, edit_bits={_fmt(edit_bits, 1)}"
    )

    before_node = _first_value(comp_debug, ("gt_octree_node", "gt_node_abs", "prebuilt_node_count_before"), float("nan"))
    after_node = _first_value(comp_debug, ("gen_octree_node", "gen_node_abs", "prebuilt_node_count_after"), float("nan"))
    before_single = _first_value(comp_debug, ("gt_octree_single", "gt_single_abs", "prebuilt_single_child_count_before"), float("nan"))
    after_single = _first_value(comp_debug, ("gen_octree_single", "gen_single_abs", "prebuilt_single_child_count_after"), float("nan"))
    occ_entropy_delta = _first_value(
        comp_debug,
        ("actual_occupancy_entropy_delta", "exact_occ_entropy_delta", "occupancy_entropy_delta"),
        float("nan"),
    )
    occ_nll_delta = _first_value(
        comp_debug,
        ("actual_occupancy_nll_delta", "exact_occ_nll_delta", "occupancy_nll_delta"),
        float("nan"),
    )
    predictability = _first_value(
        comp_debug,
        ("actual_occupancy_predictability_after", "sparsepcgc_prob_true_mean_after"),
        float("nan"),
    )
    lowprob = _first_value(
        comp_debug,
        ("actual_lowprob_occupancy_ratio_after", "sparsepcgc_prob_true_low_ratio_after", "lowprob_occupancy_ratio"),
        float("nan"),
    )
    writer.write(
        f"StepStruct {step + 1}/{num_steps}: "
        f"node={_count_delta_text(before_node, after_node)}, "
        f"single={_count_delta_text(before_single, after_single)}, "
        f"occ_entropy_d={_fmt(occ_entropy_delta)}, occ_nll_d={_fmt(occ_nll_delta)}, "
        f"occ_pred={_fmt(predictability)}, lowprob={_fmt(lowprob)}"
    )

    input_points = _first_value(
        edit_stats,
        ("input_points_avg", "input_points"),
        _first_value(comp_debug, ("gt_points", "point_count_before"), float("nan")),
    )
    output_points = _first_value(
        edit_stats,
        ("output_points_avg", "output_points"),
        _first_value(comp_debug, ("gen_points", "point_count_after"), float("nan")),
    )
    add_count = _first_value(
        structure_debug,
        ("add_target_voxel_count", "add_actual_point_count", "voxel_edit_add_count"),
        _first_value(comp_debug, ("voxel_edit_add_count", "actual_oracle_accepted_add_count"), 0),
    )
    prune_count = _first_value(
        structure_debug,
        ("delete_target_voxel_count", "delete_removed_point_count", "voxel_edit_drop_count"),
        _first_value(comp_debug, ("voxel_edit_drop_count", "actual_oracle_accepted_prune_count"), 0),
    )
    adjust_count = _first_value(
        structure_debug,
        ("move_source_voxel_count", "moved_different_voxel_count", "voxel_edit_move_count"),
        _first_value(comp_debug, ("voxel_edit_move_count", "actual_oracle_accepted_adjust_count"), 0),
    )
    oracle_full_drop = _first_value(
        comp_debug,
        (
            "actual_oracle_full_cloud_macro_best_drop_count",
            "actual_oracle_override_drop_count",
            "oracle_full_cloud_override_drop_count",
        ),
        0,
    )
    oracle_full_ratio = _first_value(
        comp_debug,
        ("actual_oracle_full_cloud_macro_best_ratio", "oracle_full_cloud_drop_ratio"),
        0.0,
    )
    if not bool(comp_debug.get("oracle_full_cloud_override_used", False)):
        oracle_full_drop = 0
        oracle_full_ratio = 0.0
    local_prune_ratio = _first_value(
        edit_stats,
        ("voxel_drop_ratio_percent", "deleted_ratio_percent"),
        float("nan"),
    )
    full_cloud_prune_ratio = _first_value(
        edit_stats,
        ("full_cloud_voxel_drop_ratio_percent",),
        float("nan"),
    )
    def _prefer_text(key):
        comp_value = str(comp_debug.get(key, "") or "").strip()
        if comp_value:
            return comp_value
        return str(structure_debug.get(key, "") or "").strip()
    def _prefer_number(key, default=float("nan")):
        comp_value = comp_debug.get(key, None)
        comp_float = _to_float(comp_value, float("nan"))
        if math.isfinite(comp_float):
            return comp_float
        struct_value = structure_debug.get(key, default)
        return _to_float(struct_value, default)
    hard_drop_reason = _prefer_text("hard_drop_block_reason")
    prune_after_prior_mode = _prefer_text("prune_after_prior_mode")
    hard_drop_count_trace = _prefer_text("hard_drop_count_trace")
    collapse_reason = _prefer_text("collapse_reason")
    hard_target_source_id = _prefer_number("hard_drop_target_ratio_source_id")
    hard_target_ratio = _prefer_number("hard_drop_target_ratio_value")
    hard_target_network = _prefer_number("hard_drop_target_ratio_network_value")
    hard_target_prior = _prefer_number("hard_drop_target_ratio_codec_prior_value")
    post_amount_used = _prefer_number("post_warmup_amount_hybrid_applied", 0.0)
    post_amount_alpha = _prefer_number("post_warmup_amount_alpha")
    post_amount_proposal = _prefer_number("post_warmup_amount_proposal_ratio")
    post_amount_teacher = _prefer_number("post_warmup_amount_teacher_loss")
    collapse_detected = bool(
        comp_debug.get("phase0_noop_only_collapse_detected", False)
        or structure_debug.get("phase0_noop_only_collapse_detected", False)
    )
    writer.write(
        f"StepOps {step + 1}/{num_steps}: "
        f"input={_fmt_int(input_points)}, output={_fmt_int(output_points)}, "
        f"LocalAdd={_fmt_int(add_count)}, LocalPrune={_fmt_int(prune_count)}, "
        f"LocalAdjust={_fmt_int(adjust_count)}, "
        f"PruneMode={prune_after_prior_mode}, "
        f"HardPruneReason={hard_drop_reason}, "
        f"AmountSrc={_fmt_int(hard_target_source_id)}, "
        f"AmountTarget={_fmt(hard_target_ratio, 5)}, "
        f"AmountNet={_fmt(hard_target_network, 5)}, "
        f"AmountPrior={_fmt(hard_target_prior, 5)}, "
        f"PostAmountUsed={bool(round(_to_float(post_amount_used, 0.0)))}, "
        f"PostAlpha={_fmt(post_amount_alpha, 5)}, "
        f"PostProposal={_fmt(post_amount_proposal, 5)}, "
        f"PostTeacher={_fmt(post_amount_teacher, 6)}, "
        f"CollapseDetected={collapse_detected}, "
        f"CollapseReason={collapse_reason}, "
        f"HardDropTrace={hard_drop_count_trace}, "
        f"OracleFullPrune={_fmt_int(oracle_full_drop)}({100.0 * _to_float(oracle_full_ratio, 0.0):.2f}%)"
    )
    writer.write(
        f"StepTrainDiag {step + 1}/{num_steps}: "
        f"anchor={bool(comp_debug.get('is_anchor_refresh_step', False))}, "
        f"subtree={bool(comp_debug.get('is_subtree_step', False))}, "
        f"subtree_filter_label={int(_to_float(comp_debug.get('subtree_actual_filter_label_id', 0), 0))}, "
        f"subtree_filter_weight={_fmt(comp_debug.get('subtree_actual_filter_weight', float('nan')))}, "
        f"anchor_teacher_used={bool(comp_debug.get('anchor_success_teacher_used', False))}, "
        f"anchor_teacher_amount={_fmt(comp_debug.get('anchor_success_teacher_amount', float('nan')))}, "
        f"outcome_good={_fmt(comp_debug.get('outcome_good_weight', float('nan')))}, "
        f"outcome_bad={_fmt(comp_debug.get('outcome_bad_weight', float('nan')))}, "
        f"surrogate_trust={_fmt(comp_debug.get('surrogate_trust_value', float('nan')))}, "
        f"stage_guard={bool(comp_debug.get('stage_switch_guard_used', False))}, "
        f"com_factor={_fmt(comp_debug.get('compression_loss_factor_effective', float('nan')))}, "
        f"policy_factor={_fmt(comp_debug.get('policy_loss_factor_effective', float('nan')))}"
    )


def log_compact_step_grad(writer, step, num_steps, args):
    if writer is None or not hasattr(writer, "write"):
        return
    grad_map = getattr(args, "_last_grad_flow", {}) or {}

    def grad(name):
        return _fmt(grad_map.get(f"{name}_grad_norm", float("nan")), digits=3)

    writer.write(
        f"StepVoxelGrad {step + 1}/{num_steps}: "
        f"Prune(where={grad('delete_branch')}, amount={grad('delete_amount')}); "
        f"Add(where={grad('add_branch')}, amount={grad('add_amount')}, target={grad('add_target_branch')}); "
        f"Adjust(where={grad('move_branch')}, amount={grad('move_amount')})\n"
    )


def log_step_timing(
    writer,
    args,
    step,
    num_steps,
    epoch,
    global_train_step,
    use_cuda,
    st_step,
    timing_data_start,
    timing_data_end,
    timing_model_start,
    timing_model_end,
    timing_noise_start,
    timing_noise_end,
    timing_loss_start,
    timing_loss_end,
    timing_step_end,
    en_step,
    loss,
    model,
    KNN_BACKEND,
):
    base_model = model.module if hasattr(model, "module") else model
    comp_debug = getattr(loss, "last_compression_debug", {}) or {}
    comp_timing = comp_debug.get("timing", {}) or {}
    runtime_timing = getattr(base_model, "last_runtime_timing", {}) or {}
    encoder_debug = getattr(base_model, "last_encoder_debug", {}) or {}

    cuda_peak_mb = 0.0
    cuda_alloc_mb = 0.0
    cuda_reserved_mb = 0.0

    if use_cuda and torch.cuda.is_available():
        cuda_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        cuda_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        cuda_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

    writer.write(
        "StepTiming "
        f"step={step + 1}/{num_steps}: "
        f"compress={getattr(args, 'compress', 'unknown')}, "
        f"data={timing_data_end - timing_data_start:.4f}s, "
        f"model={timing_model_end - timing_model_start:.4f}s, "
        f"noise={timing_noise_end - timing_noise_start:.6f}s, "
        f"loss={timing_loss_end - timing_loss_start:.4f}s, "
        f"original_actual_encode={float(comp_debug.get('gt_actual_encode_time', 0.0)):.4f}s, "
        f"edited_actual_encode={float(comp_debug.get('gen_actual_encode_time', 0.0)):.4f}s, "
        f"actual_encode_total={float(comp_debug.get('actual_encode_time_total', 0.0)):.4f}s, "
        f"candidate_actual_oracle={float(comp_debug.get('actual_oracle_time', 0.0)):.4f}s, "
        f"octree_build={float(comp_debug.get('octree_build_time', comp_debug.get('sparsepcgc_octree_build_time', 0.0))):.4f}s, "
        f"full_cloud_canonical={float(comp_debug.get('full_cloud_canonical_build_time', 0.0)):.4f}s, "
        f"subtree_group={float(comp_debug.get('subtree_group_build_time', 0.0)):.4f}s, "
        f"subtree_potential={float(comp_debug.get('subtree_potential_select_time', 0.0)):.4f}s, "
        f"selected_metadata_oracle={float(comp_debug.get('selected_metadata_oracle_time', 0.0)):.4f}s, "
        f"full_cloud_anchor_block={float(comp_debug.get('full_cloud_anchor_block_time', 0.0)):.4f}s, "
        f"full_cloud_anchor_runtime={comp_debug.get('full_cloud_anchor_runtime_timing', {})}, "
        f"geometry_loss={float(comp_debug.get('geometry_loss_time', 0.0)):.4f}s, "
        f"proxy_loss={float(comp_debug.get('proxy_loss_time', comp_debug.get('sparsepcgc_proxy_time', 0.0))):.4f}s, "
        f"backward_opt={timing_step_end - timing_loss_end:.4f}s, "
        f"metrics_log={en_step - timing_step_end:.4f}s, "
        f"total={en_step - st_step:.4f}s, "
        f"cuda_peak_mb={cuda_peak_mb:.1f}, "
        f"cuda_alloc_mb={cuda_alloc_mb:.1f}, "
        f"cuda_reserved_mb={cuda_reserved_mb:.1f}, "
        f"knn_backend={KNN_BACKEND}, "
        f"encoder_raw={encoder_debug.get('raw_points', 'n/a')}, "
        f"encoder_coarse={encoder_debug.get('coarse_points', 'n/a')}, "
        f"runtime={runtime_timing}, "
        f"compression_timing={comp_timing}"
    )

    if bool(getattr(args, "log_gpu_memory", True)) and use_cuda and torch.cuda.is_available():
        writer.write(
            "[GPU] "
            f"compress={getattr(args, 'compress', 'unknown')} "
            f"epoch={epoch + 1} iter={global_train_step + 1} "
            f"allocated={cuda_alloc_mb:.1f}MB "
            f"reserved={cuda_reserved_mb:.1f}MB "
            f"max_allocated={cuda_peak_mb:.1f}MB"
        )
