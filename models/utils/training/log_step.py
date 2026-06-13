# models/utils/training/log_step.py

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
    writer.write(
        f"CompressionStats step={step + 1}/{num_steps}: "
        f"actual_bit:{float(comp_debug.get('gt_actual_bit', float('nan'))):.6f}"
        f"->{float(comp_debug.get('gen_actual_bit', float('nan'))):.6f}, "
        f"actual_bit_percent={float(comp_debug.get('actual_total_bit_percent', comp_debug.get('total_bit', 0.0))):.6f}, "
        f"actual_raw_percent={float(comp_debug.get('actual_raw_percent', comp_debug.get('actual_total_bit_percent', 0.0))):.6f}, "
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
        f"lcom_main={float(comp_debug.get('compression_main_loss', 0.0)):.6f}, "
        f"lcom_aux={float(comp_debug.get('compression_aux_loss', 0.0)):.6f}, "
        f"lcom_sparsepcgc_aux={float(comp_debug.get('sparsepcgc_aux_loss', 0.0)):.6f}, "
        f"sparsepcgc_aux_weighted={float(comp_debug.get('sparsepcgc_aux_weighted', 0.0)):.6f}, "
        f"sparsepcgc_aux_used_for_backprop={bool(comp_debug.get('sparsepcgc_aux_used_for_backprop', False))}, "
        f"sparsepcgc_aux_weight_effective={float(comp_debug.get('sparsepcgc_aux_weight_effective', 0.0)):.6f}, "
        f"sparsepcgc_aux_gate={comp_debug.get('sparsepcgc_aux_gating_reason', '')}, "
        f"grad_source={comp_debug.get('grad_source', '')}, "
        f"actual_forward={float(comp_debug.get('actual_forward_value', 0.0)):.6f}, "
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
        f"actual_compression_delta={float(comp_debug.get('actual_total_bit_percent', comp_debug.get('total_bit', 0.0))):.6f}, "
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
