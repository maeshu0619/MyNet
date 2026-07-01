import math

from .actual_codec_status import is_fresh_actual
from .scalar_utils import case_float, case_int, summarize_octree_level_debug
from .sparsepcgc_controls import add_warmup_factor, sparsepcgc_add_experiment_active


def _ratio_percent_or_nan(numerator, denominator):
    numerator = case_float(numerator, float("nan"))
    denominator = case_float(denominator, float("nan"))
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    if denominator <= 0.0:
        return 0.0 if abs(numerator) <= 1e-12 else float("nan")
    return 100.0 * numerator / denominator


def build_compression_metric_row(
    args,
    *,
    global_step,
    episode,
    epoch,
    step,
    stage,
    comp_debug,
    L_com,
    sequence_name=None,
    sequence_step=None,
):
    actual_delta = case_float(comp_debug.get("actual_total_bit_percent", float("nan")), float("nan"))
    actual_objective_percent = case_float(
        comp_debug.get("actual_objective_percent", comp_debug.get("actual_train_objective_percent", actual_delta)),
        float("nan"),
    )
    gt_actual_bits = case_float(comp_debug.get("gt_actual_bit", comp_debug.get("gt_bit_abs", float("nan"))), float("nan"))
    gen_actual_bits_raw = case_float(comp_debug.get("gen_actual_bit", comp_debug.get("gen_bit_abs", float("nan"))), float("nan"))
    gen_actual_bits = case_float(
        comp_debug.get(
            "gen_total_bit_with_edit_record",
            comp_debug.get("gen_actual_bit", comp_debug.get("actual_total_bits", float("nan"))),
        ),
        float("nan"),
    )
    actual_raw_formula_percent = float("nan")
    actual_billed_formula_percent = float("nan")
    if math.isfinite(gt_actual_bits) and gt_actual_bits > 0.0:
        if math.isfinite(gen_actual_bits_raw):
            actual_raw_formula_percent = 100.0 * (gen_actual_bits_raw - gt_actual_bits) / gt_actual_bits
        if math.isfinite(gen_actual_bits):
            actual_billed_formula_percent = 100.0 * (gen_actual_bits - gt_actual_bits) / gt_actual_bits
    actual_ratio = float("nan")
    if math.isfinite(gt_actual_bits) and gt_actual_bits > 0.0 and math.isfinite(gen_actual_bits):
        actual_ratio = 100.0 * gen_actual_bits / gt_actual_bits
    elif math.isfinite(actual_delta):
        actual_ratio = 100.0 + actual_delta
    oracle_override_used = (
        bool(comp_debug.get("oracle_full_cloud_override_used", False))
        or str(comp_debug.get("policy_action_source", "")) == "actual_oracle_full_cloud_override"
    )
    policy_actual_percent = case_float(comp_debug.get("policy_final_full_cloud_actual_bit_percent", float("nan")), float("nan"))
    if not math.isfinite(policy_actual_percent) and not oracle_override_used:
        policy_actual_percent = case_float(
            comp_debug.get(
                "policy_full_cloud_actual_bit_percent",
                comp_debug.get("full_cloud_actual_bit_percent", actual_delta),
            ),
            float("nan"),
        )
    oracle_teacher_actual_percent = case_float(
        comp_debug.get("oracle_full_cloud_actual_bit_percent", float("nan")),
        float("nan"),
    )
    actual_formula_mismatch = None
    if math.isfinite(actual_delta) and math.isfinite(actual_billed_formula_percent):
        actual_formula_mismatch = abs(actual_delta - actual_billed_formula_percent) > 1e-5
    actual_value_matches_policy_output = None
    if math.isfinite(actual_delta) and math.isfinite(policy_actual_percent):
        actual_value_matches_policy_output = abs(actual_delta - policy_actual_percent) <= 1e-5
    fresh_actual = is_fresh_actual(args, comp_debug)
    cached_actual = math.isfinite(actual_delta) and not fresh_actual
    return {
        "global_step": int(global_step) + 1,
        "episode": int(episode) + 1,
        "epoch": int(epoch) + 1,
        "step": int(step) + 1,
        "episode_index": int(episode),
        "epoch_index": int(epoch),
        "sequence_step": (int(sequence_step) + 1) if sequence_step is not None else int(step) + 1,
        "sequence_name": str(sequence_name or ""),
        "stage": str(stage),
        "codec": str(comp_debug.get("teacher_codec", getattr(args, "compress", "unknown"))),
        "backend": str(getattr(args, "compression_loss_backend", "proxy")),
        "actual_value_source": str(comp_debug.get("actual_value_source", "unknown")),
        "fresh_actual": bool(fresh_actual),
        "cached_actual": bool(cached_actual),
        "actual_total_bit_percent": actual_delta if math.isfinite(actual_delta) else None,
        "actual_train_objective_percent": (
            actual_objective_percent if math.isfinite(actual_objective_percent) else None
        ),
        "actual_objective_percent": (
            actual_objective_percent if math.isfinite(actual_objective_percent) else None
        ),
        "actual_bit_objective": str(comp_debug.get("actual_bit_objective", getattr(args, "sparsepcgc_actual_bit_objective", "raw"))),
        "actual_objective_bit_source": str(comp_debug.get("actual_objective_bit_source", "")),
        "policy_actual_percent": policy_actual_percent if math.isfinite(policy_actual_percent) else None,
        "oracle_teacher_actual_percent": (
            oracle_teacher_actual_percent if math.isfinite(oracle_teacher_actual_percent) else None
        ),
        "actual_total_bit_ratio_percent": actual_ratio if math.isfinite(actual_ratio) else None,
        "actual_raw_formula_percent": (
            actual_raw_formula_percent if math.isfinite(actual_raw_formula_percent) else None
        ),
        "actual_billed_formula_percent": (
            actual_billed_formula_percent if math.isfinite(actual_billed_formula_percent) else None
        ),
        "actual_formula_mismatch": actual_formula_mismatch,
        "actual_value_is_oracle_override": bool(oracle_override_used),
        "actual_value_matches_policy_output": actual_value_matches_policy_output,
        "proposal_selector_enabled": bool(comp_debug.get("proposal_selector_enabled", False)),
        "proposal_candidate_count": case_int(comp_debug.get("proposal_candidate_count", 0), 0),
        "proposal_actual_eval_count": case_int(comp_debug.get("proposal_actual_eval_count", 0), 0),
        "proposal_surrogate_prefilter_count": case_int(
            comp_debug.get("proposal_surrogate_prefilter_count", 0),
            0,
        ),
        "proposal_applied_subtree_count": case_int(
            comp_debug.get("proposal_applied_subtree_count", 0),
            0,
        ),
        "proposal_selected_subtree_count": case_int(
            comp_debug.get("proposal_selected_subtree_count", 0),
            0,
        ),
        "proposal_noop_count": case_int(comp_debug.get("proposal_noop_count", 0), 0),
        "proposal_best_actual_percent": case_float(
            comp_debug.get("proposal_best_actual_percent", float("nan")),
            float("nan"),
        ),
        "proposal_chosen_actual_percent": case_float(
            comp_debug.get("proposal_chosen_actual_percent", float("nan")),
            float("nan"),
        ),
        "proposal_predicted_delta": case_float(
            comp_debug.get("proposal_predicted_delta", float("nan")),
            float("nan"),
        ),
        "proposal_amount_bin": case_float(comp_debug.get("proposal_amount_bin", float("nan")), float("nan")),
        "proposal_amount_residual": case_float(
            comp_debug.get("proposal_amount_residual", float("nan")),
            float("nan"),
        ),
        "proposal_final_amount": case_float(
            comp_debug.get("proposal_final_amount", float("nan")),
            float("nan"),
        ),
        "proposal_cls_loss": case_float(comp_debug.get("proposal_cls_loss", float("nan")), float("nan")),
        "proposal_value_loss": case_float(comp_debug.get("proposal_value_loss", float("nan")), float("nan")),
        "proposal_rank_loss": case_float(comp_debug.get("proposal_rank_loss", float("nan")), float("nan")),
        "proposal_geom_loss": case_float(comp_debug.get("proposal_geom_loss", float("nan")), float("nan")),
        "proposal_total_loss": case_float(comp_debug.get("proposal_total_loss", float("nan")), float("nan")),
        "proposal_teacher_source": str(comp_debug.get("proposal_teacher_source", "")),
        "verified_noop_guard_used": bool(comp_debug.get("verified_noop_guard_used", False)),
        "full_cloud_verified_noop_guard_used": bool(
            comp_debug.get("full_cloud_verified_noop_guard_used", False)
        ),
        "sparsepcgc_training_mode": str(
            comp_debug.get(
                "sparsepcgc_training_mode",
                getattr(args, "sparsepcgc_training_mode", "subtree_selector"),
            )
        ),
        "full_cloud_amount_enabled": bool(comp_debug.get("full_cloud_amount_enabled", False)),
        "full_cloud_amount_fresh_actual_every_step": bool(
            comp_debug.get(
                "full_cloud_amount_fresh_actual_every_step",
                getattr(args, "sparsepcgc_full_cloud_amount_fresh_actual_every_step", True),
            )
        ),
        "full_cloud_amount_actual_step": bool(comp_debug.get("full_cloud_amount_actual_step", False)),
        "full_cloud_amount_actual_interval": case_int(
            comp_debug.get(
                "full_cloud_amount_actual_interval",
                getattr(args, "_full_cloud_amount_actual_interval_active", 0),
            ),
            0,
        ),
        "full_cloud_amount_input_points": case_int(comp_debug.get("full_cloud_amount_input_points", 0), 0),
        "full_cloud_amount_bin": case_float(comp_debug.get("full_cloud_amount_bin", float("nan")), float("nan")),
        "full_cloud_amount_residual": case_float(
            comp_debug.get("full_cloud_amount_residual", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_pred_residual": case_float(
            comp_debug.get("full_cloud_amount_pred_residual", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_pred_residual_raw": case_float(
            comp_debug.get("full_cloud_amount_pred_residual_raw", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_selected_base_bin": case_float(
            comp_debug.get("full_cloud_amount_selected_base_bin", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_selected_residual": case_float(
            comp_debug.get("full_cloud_amount_selected_residual", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_final_ratio": case_float(
            comp_debug.get("full_cloud_amount_final_ratio", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_drop_count": case_int(comp_debug.get("full_cloud_amount_drop_count", 0), 0),
        "full_cloud_amount_noop_selected": bool(comp_debug.get("full_cloud_amount_noop_selected", False)),
        "full_cloud_amount_candidate_count": case_int(comp_debug.get("full_cloud_amount_candidate_count", 0), 0),
        "full_cloud_amount_actual_eval_count": case_int(comp_debug.get("full_cloud_amount_actual_eval_count", 0), 0),
        "full_cloud_amount_actual_requested_count": case_int(
            comp_debug.get("full_cloud_amount_actual_requested_count", 0),
            0,
        ),
        "full_cloud_amount_actual_finished_count": case_int(
            comp_debug.get("full_cloud_amount_actual_finished_count", 0),
            0,
        ),
        "full_cloud_amount_teacher_source": str(comp_debug.get("full_cloud_amount_teacher_source", "")),
        "full_cloud_amount_teacher_ratio": case_float(
            comp_debug.get("full_cloud_amount_teacher_ratio", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_teacher_base_bin": case_float(
            comp_debug.get("full_cloud_amount_teacher_base_bin", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_teacher_residual": case_float(
            comp_debug.get("full_cloud_amount_teacher_residual", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_oracle_best_ratio": case_float(
            comp_debug.get("full_cloud_amount_oracle_best_ratio", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_raw_oracle_best_ratio": case_float(
            comp_debug.get("full_cloud_amount_raw_oracle_best_ratio", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_oracle_best_actual_delta": case_float(
            comp_debug.get("full_cloud_amount_oracle_best_actual_delta", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_oracle_best_objective_delta": case_float(
            comp_debug.get(
                "full_cloud_amount_oracle_best_objective_delta",
                comp_debug.get("full_cloud_amount_oracle_best_actual_delta", float("nan")),
            ),
            float("nan"),
        ),
        "full_cloud_amount_selected_ratio": case_float(
            comp_debug.get("full_cloud_amount_selected_ratio", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_selected_actual_delta": case_float(
            comp_debug.get("full_cloud_amount_selected_actual_delta", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_selected_objective_delta": case_float(
            comp_debug.get(
                "full_cloud_amount_selected_objective_delta",
                comp_debug.get("full_cloud_amount_selected_actual_delta", float("nan")),
            ),
            float("nan"),
        ),
        "full_cloud_amount_oracle_gap": case_float(
            comp_debug.get("full_cloud_amount_oracle_gap", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_selected_is_best": bool(
            comp_debug.get("full_cloud_amount_selected_is_best", False)
        ),
        "full_cloud_amount_selected_is_raw_best": bool(
            comp_debug.get("full_cloud_amount_selected_is_raw_best", False)
        ),
        "full_cloud_amount_raw_oracle_gap": case_float(
            comp_debug.get("full_cloud_amount_raw_oracle_gap", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_actual_finished_nonselected_count": case_int(
            comp_debug.get("full_cloud_amount_actual_finished_nonselected_count", 0),
            0,
        ),
        "full_cloud_amount_wide_probe_due": bool(
            comp_debug.get("full_cloud_amount_wide_probe_due", False)
        ),
        "full_cloud_amount_wide_probe_actual_count": case_int(
            comp_debug.get("full_cloud_amount_wide_probe_actual_count", 0),
            0,
        ),
        "full_cloud_amount_sequence_memory_ratio": case_float(
            comp_debug.get("full_cloud_amount_sequence_memory_ratio", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_entropy": case_float(
            comp_debug.get("full_cloud_amount_entropy", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_predicted_delta": case_float(
            comp_debug.get("full_cloud_amount_predicted_delta", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_actual_delta": case_float(
            comp_debug.get("full_cloud_amount_actual_delta", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_surrogate_delta": case_float(
            comp_debug.get("full_cloud_amount_surrogate_delta", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_geom_loss": case_float(
            comp_debug.get("full_cloud_amount_geom_loss", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_cls_loss": case_float(comp_debug.get("full_cloud_amount_cls_loss", 0.0), 0.0),
        "full_cloud_amount_value_loss": case_float(comp_debug.get("full_cloud_amount_value_loss", 0.0), 0.0),
        "full_cloud_amount_rank_loss": case_float(comp_debug.get("full_cloud_amount_rank_loss", 0.0), 0.0),
        "full_cloud_amount_geom_guard_loss": case_float(
            comp_debug.get("full_cloud_amount_geom_guard_loss", 0.0),
            0.0,
        ),
        "full_cloud_amount_ratio_reg_loss": case_float(
            comp_debug.get("full_cloud_amount_ratio_reg_loss", 0.0),
            0.0,
        ),
        "full_cloud_amount_noop_guard_loss": case_float(
            comp_debug.get("full_cloud_amount_noop_guard_loss", 0.0),
            0.0,
        ),
        "full_cloud_amount_entropy_loss": case_float(
            comp_debug.get("full_cloud_amount_entropy_loss", 0.0),
            0.0,
        ),
        "full_cloud_amount_residual_loss": case_float(
            comp_debug.get("full_cloud_amount_residual_loss", 0.0),
            0.0,
        ),
        "full_cloud_amount_residual_error": case_float(
            comp_debug.get("full_cloud_amount_residual_error", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_residual_enabled": bool(
            comp_debug.get("full_cloud_amount_residual_enabled", False)
        ),
        "full_cloud_amount_residual_max": case_float(
            comp_debug.get("full_cloud_amount_residual_max", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_residual_teacher_clamped": bool(
            comp_debug.get("full_cloud_amount_residual_teacher_clamped", False)
        ),
        "full_cloud_amount_ratio_hist_selected": str(
            comp_debug.get("full_cloud_amount_ratio_hist_selected", "")
        ),
        "full_cloud_amount_ratio_hist_teacher": str(
            comp_debug.get("full_cloud_amount_ratio_hist_teacher", "")
        ),
        "full_cloud_amount_fine_probe_enabled": bool(
            comp_debug.get("full_cloud_amount_fine_probe_enabled", False)
        ),
        "full_cloud_amount_residual_probe_enabled": bool(
            comp_debug.get("full_cloud_amount_residual_probe_enabled", False)
        ),
        "full_cloud_amount_total_loss": case_float(comp_debug.get("full_cloud_amount_total_loss", 0.0), 0.0),
        "full_cloud_amount_step_time": case_float(comp_debug.get("full_cloud_amount_step_time", 0.0), 0.0),
        "full_cloud_amount_actual_wall_time_total": case_float(
            comp_debug.get("full_cloud_amount_actual_wall_time_total", 0.0),
            0.0,
        ),
        "full_cloud_amount_actual_wall_time_max": case_float(
            comp_debug.get("full_cloud_amount_actual_wall_time_max", 0.0),
            0.0,
        ),
        "full_cloud_amount_actual_dispatch_time": case_float(
            comp_debug.get("full_cloud_amount_actual_dispatch_time", 0.0),
            0.0,
        ),
        "full_cloud_amount_actual_gather_time": case_float(
            comp_debug.get("full_cloud_amount_actual_gather_time", 0.0),
            0.0,
        ),
        "full_cloud_amount_reuse_where_ranking": bool(
            comp_debug.get("full_cloud_amount_reuse_where_ranking", False)
        ),
        "full_cloud_amount_reuse_where_ranking_reason": str(
            comp_debug.get("full_cloud_amount_reuse_where_ranking_reason", "")
        ),
        "sparsepcgc_actual_parallel_mode": str(
            comp_debug.get(
                "sparsepcgc_actual_parallel_mode",
                getattr(args, "sparsepcgc_actual_parallel_mode", "single"),
            )
        ),
        "sparsepcgc_actual_parallel_candidates": case_int(
            comp_debug.get(
                "sparsepcgc_actual_parallel_candidates",
                getattr(args, "sparsepcgc_actual_parallel_candidates", 1),
            ),
            1,
        ),
        "sparsepcgc_actual_worker_pool_used": bool(
            comp_debug.get("sparsepcgc_actual_worker_pool_used", False)
        ),
        "actual_total_bit_percent_fresh": actual_delta if fresh_actual else None,
        "actual_total_bit_percent_cached": actual_delta if cached_actual else None,
        "full_cloud_actual_bit_percent": case_float(comp_debug.get("full_cloud_actual_bit_percent", float("nan")), float("nan")),
        "policy_full_cloud_actual_bit_percent": case_float(comp_debug.get("policy_full_cloud_actual_bit_percent", float("nan")), float("nan")),
        "policy_final_full_cloud_raw_bit_percent": case_float(comp_debug.get("policy_final_full_cloud_raw_bit_percent", float("nan")), float("nan")),
        "policy_final_full_cloud_actual_bit_percent": case_float(comp_debug.get("policy_final_full_cloud_actual_bit_percent", float("nan")), float("nan")),
        "policy_final_full_cloud_gt_bit": case_float(comp_debug.get("policy_final_full_cloud_gt_bit", float("nan")), float("nan")),
        "policy_final_full_cloud_gen_bit": case_float(comp_debug.get("policy_final_full_cloud_gen_bit", float("nan")), float("nan")),
        "policy_final_full_cloud_total_bit_with_edit_record": case_float(
            comp_debug.get("policy_final_full_cloud_total_bit_with_edit_record", float("nan")),
            float("nan"),
        ),
        "oracle_full_cloud_raw_bit_percent": case_float(comp_debug.get("oracle_full_cloud_raw_bit_percent", float("nan")), float("nan")),
        "oracle_full_cloud_actual_bit_percent": case_float(comp_debug.get("oracle_full_cloud_actual_bit_percent", float("nan")), float("nan")),
        "oracle_full_cloud_override_used": bool(oracle_override_used),
        "policy_action_source": str(comp_debug.get("policy_action_source", "")),
        "policy_actual_noop_guard_used": bool(comp_debug.get("policy_actual_noop_guard_used", False)),
        "policy_actual_noop_guard_percent": case_float(comp_debug.get("policy_actual_noop_guard_percent", float("nan")), float("nan")),
        "policy_actual_noop_guard_raw_percent": case_float(comp_debug.get("policy_actual_noop_guard_raw_percent", float("nan")), float("nan")),
        "policy_actual_noop_guard_replaced_in_loss": bool(comp_debug.get("policy_actual_noop_guard_replaced_in_loss", False)),
        "policy_actual_noop_guard_reason": str(comp_debug.get("policy_actual_noop_guard_reason", "")),
        "policy_actual_noop_guard_margin": case_float(comp_debug.get("policy_actual_noop_guard_margin", 0.0), 0.0),
        "subtree_actual_bit_percent": case_float(comp_debug.get("subtree_actual_bit_percent", float("nan")), float("nan")),
        "local_proxy_percent": case_float(comp_debug.get("local_proxy_percent", float("nan")), float("nan")),
        "actual_scope": str(comp_debug.get("actual_scope", "")),
        "teacher_scope": str(comp_debug.get("teacher_scope", "")),
        "full_cloud_actual_primary_used": bool(comp_debug.get("full_cloud_actual_primary_used", False)),
        "full_cloud_actual_primary_forward_value": case_float(comp_debug.get("full_cloud_actual_primary_forward_value", float("nan")), float("nan")),
        "full_cloud_actual_primary_subtree_forward_before": case_float(comp_debug.get("full_cloud_actual_primary_subtree_forward_before", float("nan")), float("nan")),
        "full_cloud_actual_primary_reason": str(comp_debug.get("full_cloud_actual_primary_reason", "")),
        "full_cloud_actual_primary_raw_policy_value": case_float(comp_debug.get("full_cloud_actual_primary_raw_policy_value", float("nan")), float("nan")),
        "full_cloud_actual_primary_noop_guard_used": bool(comp_debug.get("full_cloud_actual_primary_noop_guard_used", False)),
        "full_cloud_actual_primary_oracle_source": bool(comp_debug.get("full_cloud_actual_primary_oracle_source", False)),
        "full_cloud_actual_primary_suppressed_zero": bool(comp_debug.get("full_cloud_actual_primary_suppressed_zero", False)),
        "full_cloud_actual_primary_grad_source": str(comp_debug.get("full_cloud_actual_primary_grad_source", "")),
        "full_cloud_actual_primary_grad_fn": str(comp_debug.get("full_cloud_actual_primary_grad_fn", "")),
        "full_vs_subtree_actual_gap": case_float(comp_debug.get("full_vs_subtree_actual_gap", float("nan")), float("nan")),
        "proxy_full_actual_gap": case_float(comp_debug.get("proxy_full_actual_gap", float("nan")), float("nan")),
        "sign_match_subtree_full": comp_debug.get("sign_match_subtree_full", None),
        "sign_match_proxy_full": comp_debug.get("sign_match_proxy_full", None),
        "actual_oracle_force_no_edit_used": bool(comp_debug.get("actual_oracle_force_no_edit_used", False)),
        "actual_raw_percent": case_float(comp_debug.get("actual_raw_percent", actual_delta), float("nan")),
        "actual_objective_percent": case_float(
            comp_debug.get("actual_objective_percent", comp_debug.get("actual_train_objective_percent", actual_delta)),
            float("nan"),
        ),
        "actual_bit_objective": str(comp_debug.get("actual_bit_objective", getattr(args, "sparsepcgc_actual_bit_objective", "raw"))),
        "actual_objective_bit_source": str(comp_debug.get("actual_objective_bit_source", "")),
        "actual_bit_percent_raw": case_float(comp_debug.get("actual_bit_percent_raw", comp_debug.get("actual_bit_percent", actual_delta)), float("nan")),
        "actual_bit_percent_used_for_loss": case_float(comp_debug.get("actual_bit_percent_used_for_loss", comp_debug.get("actual_forward_value", actual_delta)), float("nan")),
        "actual_edit_record_bits": case_float(comp_debug.get("actual_edit_record_bits", 0.0), 0.0),
        "gen_total_bit_with_edit_record": case_float(
            comp_debug.get("gen_total_bit_with_edit_record", comp_debug.get("actual_total_bits", float("nan"))),
            float("nan"),
        ),
        "compression_loss_L_com": case_float(L_com, float("nan")),
        "compression_loss_raw": case_float(comp_debug.get("compression_loss_raw", comp_debug.get("actual_bit_percent_raw", actual_delta)), float("nan")),
        "compression_loss_used": case_float(comp_debug.get("compression_loss_used", comp_debug.get("actual_forward_value", actual_delta)), float("nan")),
        "compression_loss_noop_replaced": bool(comp_debug.get("compression_loss_noop_replaced", False)),
        "compression_loss_tensor_value": case_float(comp_debug.get("compression_loss_tensor_value", L_com), float("nan")),
        "compression_loss_requires_grad": bool(comp_debug.get("compression_loss_requires_grad", comp_debug.get("lcom_requires_grad", False))),
        "compression_loss_grad_fn": str(comp_debug.get("compression_loss_grad_fn", "")),
        "is_anchor_refresh_step": bool(comp_debug.get("is_anchor_refresh_step", False)),
        "is_subtree_step": bool(comp_debug.get("is_subtree_step", False)),
        "anchor_step_count": 1 if bool(comp_debug.get("is_anchor_refresh_step", False)) else 0,
        "subtree_step_count": 1 if bool(comp_debug.get("is_subtree_step", False)) else 0,
        "anchor_actual_raw": case_float(
            comp_debug.get(
                "full_cloud_anchor_actual_total_bit_percent",
                comp_debug.get("full_cloud_anchor_actual_bit_percent", float("nan")),
            ),
            float("nan"),
        ),
        "subtree_actual_raw": case_float(
            comp_debug.get(
                "subtree_actual_filter_raw_percent",
                comp_debug.get("subtree_actual_bit_percent", float("nan")),
            ),
            float("nan"),
        ),
        "subtree_good_count": case_int(comp_debug.get("subtree_good_count", 0), 0),
        "subtree_neutral_count": case_int(comp_debug.get("subtree_neutral_count", 0), 0),
        "subtree_bad_count": case_int(comp_debug.get("subtree_bad_count", 0), 0),
        "subtree_actual_filter_used": bool(comp_debug.get("subtree_actual_filter_used", False)),
        "subtree_actual_filter_label_id": case_int(comp_debug.get("subtree_actual_filter_label_id", 0), 0),
        "subtree_actual_filter_weight": case_float(comp_debug.get("subtree_actual_filter_weight", float("nan")), float("nan")),
        "subtree_actual_filter_raw_percent": case_float(comp_debug.get("subtree_actual_filter_raw_percent", float("nan")), float("nan")),
        "subtree_actual_filter_used_percent": case_float(comp_debug.get("subtree_actual_filter_used_percent", float("nan")), float("nan")),
        "anchor_success_teacher_used": bool(comp_debug.get("anchor_success_teacher_used", False)),
        "anchor_success_teacher_percent": case_float(comp_debug.get("anchor_success_teacher_percent", float("nan")), float("nan")),
        "anchor_success_teacher_amount": case_float(comp_debug.get("anchor_success_teacher_amount", float("nan")), float("nan")),
        "anchor_success_teacher_loss": case_float(comp_debug.get("anchor_success_teacher_loss", float("nan")), float("nan")),
        "anchor_success_memory_count": case_int(comp_debug.get("anchor_success_memory_count", 0), 0),
        "outcome_imitation_used": bool(comp_debug.get("outcome_imitation_used", False)),
        "outcome_good_weight": case_float(comp_debug.get("outcome_good_weight", float("nan")), float("nan")),
        "outcome_bad_weight": case_float(comp_debug.get("outcome_bad_weight", float("nan")), float("nan")),
        "outcome_good_count": 1 if case_float(comp_debug.get("outcome_good_weight", 0.0), 0.0) > 0.0 else 0,
        "outcome_bad_count": 1 if case_float(comp_debug.get("outcome_bad_weight", 0.0), 0.0) > 0.0 else 0,
        "outcome_neutral_count": 1 if (
            case_float(comp_debug.get("outcome_good_weight", 0.0), 0.0) <= 0.0
            and case_float(comp_debug.get("outcome_bad_weight", 0.0), 0.0) <= 0.0
        ) else 0,
        "outcome_amount_anticollapse_loss": case_float(comp_debug.get("outcome_amount_anticollapse_loss", float("nan")), float("nan")),
        "outcome_success_amount_teacher": case_float(comp_debug.get("outcome_success_amount_teacher", float("nan")), float("nan")),
        "bad_amount_loss_disabled_no_success_memory": bool(
            comp_debug.get("bad_amount_loss_disabled_no_success_memory", False)
        ),
        "outcome_bad_amount_policy_id": case_int(comp_debug.get("outcome_bad_amount_policy_id", 0), 0),
        "amount_outcome_memory_saved": bool(comp_debug.get("amount_outcome_memory_saved", False)),
        "amount_outcome_memory_label_id": case_int(comp_debug.get("amount_outcome_memory_label_id", 0), 0),
        "amount_outcome_memory_used_ratio": case_float(comp_debug.get("amount_outcome_memory_used_ratio", float("nan")), float("nan")),
        "amount_outcome_memory_bucket_ratio": case_float(comp_debug.get("amount_outcome_memory_bucket_ratio", float("nan")), float("nan")),
        "amount_outcome_memory_best_ratio": case_float(comp_debug.get("amount_outcome_memory_best_ratio", float("nan")), float("nan")),
        "amount_outcome_memory_best_score": case_float(comp_debug.get("amount_outcome_memory_best_score", float("nan")), float("nan")),
        "amount_outcome_memory_best_count": case_int(comp_debug.get("amount_outcome_memory_best_count", 0), 0),
        "amount_outcome_memory_good_count": case_int(comp_debug.get("amount_outcome_memory_good_count", 0), 0),
        "amount_outcome_memory_bad_count": case_int(comp_debug.get("amount_outcome_memory_bad_count", 0), 0),
        "amount_outcome_memory_entry_count": case_int(comp_debug.get("amount_outcome_memory_entry_count", 0), 0),
        "subtree_outcome_memory_saved": bool(comp_debug.get("subtree_outcome_memory_saved", False)),
        "subtree_outcome_memory_score": case_float(comp_debug.get("subtree_outcome_memory_score", float("nan")), float("nan")),
        "subtree_outcome_memory_count": case_int(comp_debug.get("subtree_outcome_memory_count", 0), 0),
        "subtree_outcome_memory_good_count": case_int(comp_debug.get("subtree_outcome_memory_good_count", 0), 0),
        "subtree_outcome_memory_bad_count": case_int(comp_debug.get("subtree_outcome_memory_bad_count", 0), 0),
        "surrogate_trust_gate_used": bool(comp_debug.get("surrogate_trust_gate_used", False)),
        "surrogate_bit_error_for_trust": case_float(comp_debug.get("surrogate_bit_error_for_trust", float("nan")), float("nan")),
        "surrogate_trust_value": case_float(comp_debug.get("surrogate_trust_value", 1.0), 1.0),
        "surrogate_loss_before_trust": case_float(comp_debug.get("surrogate_loss_before_trust", float("nan")), float("nan")),
        "surrogate_loss_after_trust": case_float(comp_debug.get("surrogate_loss_after_trust", float("nan")), float("nan")),
        "stage_switch_guard_used": bool(comp_debug.get("stage_switch_guard_used", False)),
        "stage_original": str(comp_debug.get("stage_original", stage)),
        "stage_effective": str(comp_debug.get("stage_effective", stage)),
        "compression_loss_factor_original": case_float(comp_debug.get("compression_loss_factor_original", float("nan")), float("nan")),
        "compression_loss_factor_effective": case_float(comp_debug.get("compression_loss_factor_effective", float("nan")), float("nan")),
        "policy_loss_factor_original": case_float(comp_debug.get("policy_loss_factor_original", float("nan")), float("nan")),
        "policy_loss_factor_effective": case_float(comp_debug.get("policy_loss_factor_effective", float("nan")), float("nan")),
        "compression_objective_tensor_value": case_float(comp_debug.get("compression_objective_tensor_value", comp_debug.get("compression_objective", float("nan"))), float("nan")),
        "compression_objective_requires_grad": bool(comp_debug.get("compression_objective_requires_grad", False)),
        "compression_objective_grad_fn": str(comp_debug.get("compression_objective_grad_fn", "")),
        "loss_bit_tensor_value": case_float(comp_debug.get("loss_bit_tensor_value", float("nan")), float("nan")),
        "loss_bit_requires_grad": bool(comp_debug.get("loss_bit_requires_grad", False)),
        "loss_bit_grad_fn": str(comp_debug.get("loss_bit_grad_fn", "")),
        "lcom_main": case_float(comp_debug.get("compression_main_loss", float("nan")), float("nan")),
        "lcom_aux": case_float(comp_debug.get("compression_aux_loss", float("nan")), float("nan")),
        "lcom_sparsepcgc_aux": case_float(comp_debug.get("sparsepcgc_aux_loss", float("nan")), float("nan")),
        "sparsepcgc_aux_raw": case_float(comp_debug.get("sparsepcgc_aux_raw", comp_debug.get("sparsepcgc_aux_loss", float("nan"))), float("nan")),
        "sparsepcgc_aux_weighted": case_float(comp_debug.get("sparsepcgc_aux_weighted", float("nan")), float("nan")),
        "sparsepcgc_aux_backprop": bool(comp_debug.get("sparsepcgc_aux_backprop", False)),
        "sparsepcgc_aux_value": case_float(comp_debug.get("sparsepcgc_aux_value", comp_debug.get("sparsepcgc_aux_weighted", float("nan"))), float("nan")),
        "sparsepcgc_aux_used_for_backprop": bool(comp_debug.get("sparsepcgc_aux_used_for_backprop", False)),
        "sparsepcgc_aux_weight_raw": case_float(comp_debug.get("sparsepcgc_aux_weight_raw", comp_debug.get("sparsepcgc_aux_weight", float("nan"))), float("nan")),
        "sparsepcgc_aux_weight_effective": case_float(comp_debug.get("sparsepcgc_aux_weight_effective", float("nan")), float("nan")),
        "corr_sparsepcgc_aux_actual_rolling": case_float(comp_debug.get("corr_sparsepcgc_aux_actual_rolling", float("nan")), float("nan")),
        "sign_match_sparsepcgc_aux_actual_rolling": case_float(comp_debug.get("sign_match_sparsepcgc_aux_actual_rolling", float("nan")), float("nan")),
        "sparsepcgc_aux_gate_mode": str(comp_debug.get("sparsepcgc_aux_gate_mode", "")),
        "sparsepcgc_aux_gate_multiplier": case_float(comp_debug.get("sparsepcgc_aux_gate_multiplier", float("nan")), float("nan")),
        "sparsepcgc_aux_gating_reason": str(comp_debug.get("sparsepcgc_aux_gating_reason", "")),
        "lcom_without_sparsepcgc_aux": case_float(comp_debug.get("lcom_without_sparsepcgc_aux", float("nan")), float("nan")),
        "lcom_with_sparsepcgc_aux": case_float(comp_debug.get("lcom_with_sparsepcgc_aux", comp_debug.get("compression_objective", float("nan"))), float("nan")),
        "lcom_objective": case_float(comp_debug.get("compression_objective", comp_debug.get("total_bit", float("nan"))), float("nan")),
        "compression_main_grad_scale": case_float(comp_debug.get("compression_main_grad_scale", 1.0), 1.0),
        "compression_main_grad_scale_reason": str(comp_debug.get("compression_main_grad_scale_reason", "")),
        "compression_aux_in_objective": bool(comp_debug.get("compression_aux_in_objective", False)),
        "surrogate_auto_frozen": bool(comp_debug.get("surrogate_auto_frozen", False)),
        "surrogate_auto_freeze_streak": case_int(comp_debug.get("surrogate_auto_freeze_streak", 0), 0),
        "com_sparsepcgc_weight": case_float(getattr(args, "com_sparsepcgc", float("nan")), float("nan")),
        "sparsepcgc_aux_weight": case_float(getattr(args, "com_sparsepcgc", float("nan")), float("nan")),
        "sparsepcgc_active_coord_loss": case_float(comp_debug.get("sparsepcgc_active_coord_loss", float("nan")), float("nan")),
        "sparsepcgc_isolated_loss": case_float(comp_debug.get("sparsepcgc_isolated_proxy_loss", float("nan")), float("nan")),
        "sparsepcgc_entropy_loss": case_float(comp_debug.get("sparsepcgc_entropy_proxy_loss", float("nan")), float("nan")),
        "sparsepcgc_density_loss": case_float(comp_debug.get("sparsepcgc_density_proxy_loss", float("nan")), float("nan")),
        "sparsepcgc_single_aux": case_float(comp_debug.get("soft_single_percent", float("nan")), float("nan")),
        "sparsepcgc_node_aux": case_float(comp_debug.get("soft_node_percent", float("nan")), float("nan")),
        "compression_objective": case_float(comp_debug.get("compression_objective", comp_debug.get("total_bit", float("nan"))), float("nan")),
        "compression_main_loss": case_float(comp_debug.get("compression_main_loss", float("nan")), float("nan")),
        "compression_aux_loss": case_float(comp_debug.get("compression_aux_loss", float("nan")), float("nan")),
        "sparsepcgc_aux_loss": case_float(comp_debug.get("sparsepcgc_aux_loss", float("nan")), float("nan")),
        "surrogate_pred_bit_percent": case_float(comp_debug.get("surrogate_pred_bit", float("nan")), float("nan")),
        "surrogate_target_bit_percent": case_float(comp_debug.get("surrogate_target_bit", float("nan")), float("nan")),
        "surrogate_abs_bit_error": case_float(comp_debug.get("surrogate_abs_bit_error", float("nan")), float("nan")),
        "surrogate_signed_bit_error": case_float(comp_debug.get("surrogate_signed_bit_error", float("nan")), float("nan")),
        "surrogate_train_loss": case_float(comp_debug.get("surrogate_train_loss", float("nan")), float("nan")),
        "actual_codec_raw_percent": case_float(comp_debug.get("actual_raw_percent", comp_debug.get("actual_total_bit_percent", float("nan"))), float("nan")),
        "actual_clamped_percent": case_float(comp_debug.get("actual_clamped_percent", comp_debug.get("surrogate_target_train_bit", float("nan"))), float("nan")),
        "surrogate_train_target_percent": case_float(comp_debug.get("surrogate_train_target_percent", comp_debug.get("surrogate_target_train_bit", float("nan"))), float("nan")),
        "surrogate_train_target_log_ratio": case_float(comp_debug.get("surrogate_train_target_log_ratio", float("nan")), float("nan")),
        "surrogate_train_target_value": case_float(comp_debug.get("surrogate_train_target_value", comp_debug.get("surrogate_target_train_bit", float("nan"))), float("nan")),
        "surrogate_target_mode": str(comp_debug.get("surrogate_target_mode", "")),
        "surrogate_pred_raw_percent": case_float(comp_debug.get("surrogate_pred_raw_percent", comp_debug.get("surrogate_pred", float("nan"))), float("nan")),
        "surrogate_pred_clipped_percent": case_float(comp_debug.get("surrogate_pred_clipped_percent", comp_debug.get("surrogate_pred", float("nan"))), float("nan")),
        "surrogate_pred_log_ratio": case_float(comp_debug.get("surrogate_pred_log_ratio", float("nan")), float("nan")),
        "pred_clipped": bool(comp_debug.get("pred_clipped", False)),
        "target_clamp_min": case_float(comp_debug.get("target_clamp_min", float("nan")), float("nan")),
        "target_clamp_max": case_float(comp_debug.get("target_clamp_max", float("nan")), float("nan")),
        "pred_clip_min": case_float(comp_debug.get("pred_clip_min", float("nan")), float("nan")),
        "pred_clip_max": case_float(comp_debug.get("pred_clip_max", float("nan")), float("nan")),
        "raw_target_gap_percent": case_float(comp_debug.get("raw_target_gap_percent", float("nan")), float("nan")),
        "raw_actual_vs_train_target_gap": case_float(comp_debug.get("raw_actual_vs_train_target_gap", float("nan")), float("nan")),
        "surrogate_loss_against_raw_actual": case_float(comp_debug.get("surrogate_loss_against_raw_actual", float("nan")), float("nan")),
        "surrogate_loss_against_train_target": case_float(comp_debug.get("surrogate_loss_against_train_target", float("nan")), float("nan")),
        "actual_target": case_float(comp_debug.get("actual_target", float("nan")), float("nan")),
        "surrogate_pred": case_float(comp_debug.get("surrogate_pred", comp_debug.get("surrogate_pred_bit", float("nan"))), float("nan")),
        "surrogate_rel_error": case_float(comp_debug.get("surrogate_rel_error", float("nan")), float("nan")),
        "surrogate_loss_for_grad": case_float(comp_debug.get("surrogate_loss_for_grad", float("nan")), float("nan")),
        "proxy_aux_for_grad": case_float(comp_debug.get("proxy_aux_for_grad", float("nan")), float("nan")),
        "actual_forward_value": case_float(comp_debug.get("actual_forward_value", float("nan")), float("nan")),
        "actual_forward_raw_value": case_float(comp_debug.get("actual_forward_raw_value", comp_debug.get("actual_bit_percent", float("nan"))), float("nan")),
        "actual_forward_clamped": bool(comp_debug.get("actual_forward_clamped", False)),
        "actual_forward_source": str(comp_debug.get("actual_forward_source", "")),
        "compression_forward_teacher_percent": case_float(comp_debug.get("compression_forward_teacher_percent", float("nan")), float("nan")),
        "compression_forward_teacher_source": str(comp_debug.get("compression_forward_teacher_source", "")),
        "local_proxy_target_percent": case_float(comp_debug.get("local_proxy_target_percent", float("nan")), float("nan")),
        "local_proxy_rate_target_percent": case_float(comp_debug.get("local_proxy_rate_target_percent", float("nan")), float("nan")),
        "local_proxy_aux_target_percent": case_float(comp_debug.get("local_proxy_aux_target_percent", float("nan")), float("nan")),
        "local_proxy_rate_error": str(comp_debug.get("local_proxy_rate_error", "")),
        "forward_display_value": case_float(comp_debug.get("forward_display_value", float("nan")), float("nan")),
        "final_L_com_value": case_float(comp_debug.get("final_L_com_value", float("nan")), float("nan")),
        "grad_source": str(comp_debug.get("grad_source", "")),
        "detach_surrogate_from_network": bool(comp_debug.get("detach_surrogate_from_network", False)),
        "surrogate_weight_effective": case_float(comp_debug.get("surrogate_weight_effective", float("nan")), float("nan")),
        "surrogate_pred_requires_grad": bool(comp_debug.get("surrogate_pred_requires_grad", False)),
        "surrogate_input_requires_grad": bool(comp_debug.get("surrogate_input_requires_grad", False)),
        "loss_bit_proxy_requires_grad": bool(comp_debug.get("loss_bit_proxy_requires_grad", False)),
        "soft_rate_proxy_requires_grad": bool(comp_debug.get("soft_rate_proxy_requires_grad", False)),
        "soft_rate_proxy_ste_requires_grad": bool(comp_debug.get("soft_rate_proxy_ste_requires_grad", False)),
        "soft_prune_proxy_requires_grad": bool(comp_debug.get("soft_prune_proxy_requires_grad", False)),
        "soft_prune_rate_ste_requires_grad": bool(comp_debug.get("soft_prune_rate_ste_requires_grad", False)),
        "main_loss_requires_grad": bool(comp_debug.get("main_loss_requires_grad", False)),
        "lcom_requires_grad": bool(comp_debug.get("lcom_requires_grad", False)),
        "stored_main_requires_grad": bool(comp_debug.get("stored_main_requires_grad", False)),
        "stored_bit_requires_grad": bool(comp_debug.get("stored_bit_requires_grad", False)),
        "stored_node_requires_grad": bool(comp_debug.get("stored_node_requires_grad", False)),
        "stored_single_requires_grad": bool(comp_debug.get("stored_single_requires_grad", False)),
        "stored_op_requires_grad": bool(comp_debug.get("stored_op_requires_grad", False)),
        "stored_surrogate_requires_grad": bool(comp_debug.get("stored_surrogate_requires_grad", False)),
        "network_grad_from_surrogate_pred": bool(comp_debug.get("network_grad_from_surrogate_pred", False)),
        "network_grad_from_soft_rate_proxy": bool(comp_debug.get("network_grad_from_soft_rate_proxy", False)),
        "network_grad_from_soft_prune_proxy": bool(comp_debug.get("network_grad_from_soft_prune_proxy", False)),
        "network_grad_from_sparsepcgc_aux": bool(comp_debug.get("network_grad_from_sparsepcgc_aux", False)),
        "network_grad_component_summary": str(comp_debug.get("network_grad_component_summary", "")),
        "compression_soft_prune_rate_proxy_grad_weight": case_float(comp_debug.get("compression_soft_prune_rate_proxy_grad_weight", float("nan")), float("nan")),
        "target_clamped": bool(comp_debug.get("target_clamped", comp_debug.get("surrogate_target_clamped", False))),
        "target_clamp_rate": case_float(comp_debug.get("target_clamp_rate", float("nan")), float("nan")),
        "teacher_type": str(comp_debug.get("teacher_type", "")),
        "full_cloud_teacher_used": bool(comp_debug.get("full_cloud_teacher_used", False)),
        "full_cloud_actual_percent": case_float(comp_debug.get("full_cloud_actual_percent", float("nan")), float("nan")),
        "subtree_teacher_percent": case_float(comp_debug.get("subtree_teacher_percent", float("nan")), float("nan")),
        "teacher_gap_percent": case_float(comp_debug.get("teacher_gap_percent", float("nan")), float("nan")),
        "teacher_gap_status": str(comp_debug.get("teacher_gap_status", "")),
        "full_cloud_teacher_count": case_int(comp_debug.get("full_cloud_teacher_count", 0)),
        "subtree_teacher_count": case_int(comp_debug.get("subtree_teacher_count", 0)),
        "teacher_type_counts": str(comp_debug.get("teacher_type_counts", "")),
        "full_cloud_replay_ratio": case_float(comp_debug.get("full_cloud_replay_ratio", float("nan")), float("nan")),
        "replay_full_cloud_count": case_int(comp_debug.get("replay_full_cloud_count", comp_debug.get("replay_full_cloud_count", 0))),
        "full_cloud_calib_interval": case_int(comp_debug.get("full_cloud_calib_interval", getattr(args, "surrogate_full_cloud_calib_interval", 0))),
        "full_cloud_calib_triggered": bool(comp_debug.get("full_cloud_calib_triggered", False)),
        "teacher_anchor_reason": str(comp_debug.get("teacher_anchor_reason", "")),
        "train_full_cloud_actual_interval": case_int(getattr(args, "train_full_cloud_actual_interval", 0)),
        "before_bits": case_float(comp_debug.get("before_bits", comp_debug.get("gt_actual_bit", float("nan"))), float("nan")),
        "after_bits": case_float(comp_debug.get("after_bits", comp_debug.get("gen_actual_bit", float("nan"))), float("nan")),
        "log_bit_ratio": case_float(comp_debug.get("log_bit_ratio", float("nan")), float("nan")),
        "replay_age": case_float(comp_debug.get("replay_age", float("nan")), float("nan")),
        "replay_sample_count": case_int(comp_debug.get("replay_sample_count", comp_debug.get("surrogate_replay_sample_count", 0))),
        "replay_is_full_cloud": bool(comp_debug.get("replay_is_full_cloud", False)),
        "sample_name": str(comp_debug.get("sample_name", getattr(args, "_current_sample_name", ""))),
        "subtree_id": str(comp_debug.get("subtree_id", getattr(args, "_current_subtree_id", ""))),
        "proxy_delta_percent": case_float(comp_debug.get("rate_proxy_delta", comp_debug.get("total_bit", float("nan"))), float("nan")),
        "actual_bits_before": case_float(comp_debug.get("gt_actual_bit", comp_debug.get("gt_bit_abs", float("nan"))), float("nan")),
        "actual_bits_after": case_float(comp_debug.get("gen_actual_bit", comp_debug.get("gen_bit_abs", float("nan"))), float("nan")),
        "gt_actual_bit": case_float(comp_debug.get("gt_actual_bit", comp_debug.get("gt_bit_abs", float("nan"))), float("nan")),
        "gen_actual_bit": case_float(comp_debug.get("gen_actual_bit", comp_debug.get("gen_bit_abs", float("nan"))), float("nan")),
        "point_count_before": case_int(comp_debug.get("gt_points", 0)),
        "point_count_after": case_int(comp_debug.get("gen_points", 0)),
        "unique_coord_before": case_int(comp_debug.get("gt_unique_coord_count", 0)),
        "unique_coord_after": case_int(comp_debug.get("gen_unique_coord_count", 0)),
        "actual_sparsepcgc_bit": case_float(comp_debug.get("actual_sparsepcgc_bit", float("nan")), float("nan")),
        "actual_sparsepcgc_gt_bit": case_float(comp_debug.get("actual_sparsepcgc_gt_bit", float("nan")), float("nan")),
        "actual_sparsepcgc_bit_delta": case_float(comp_debug.get("actual_sparsepcgc_bit_delta", float("nan")), float("nan")),
        "proxy_sparsepcgc_bit": case_float(comp_debug.get("proxy_sparsepcgc_bit", float("nan")), float("nan")),
        "proxy_sparsepcgc_bit_percent": case_float(comp_debug.get("proxy_sparsepcgc_bit_percent", float("nan")), float("nan")),
        "estimated_occupancy_bits": case_float(comp_debug.get("estimated_occupancy_bits", float("nan")), float("nan")),
        "mean_prob_true": case_float(comp_debug.get("mean_prob_true", float("nan")), float("nan")),
        "low_prob_true_count": case_float(comp_debug.get("low_prob_true_count", float("nan")), float("nan")),
        "low_prob_true_ratio": case_float(comp_debug.get("low_prob_true_ratio", float("nan")), float("nan")),
        "proxy_actual_bit_gap": case_float(comp_debug.get("proxy_actual_bit_gap", float("nan")), float("nan")),
        "proxy_actual_bit_gap_percent": case_float(comp_debug.get("proxy_actual_bit_gap_percent", float("nan")), float("nan")),
        "use_subtree_tree": bool(comp_debug.get("use_subtree_tree", False)),
        "use_full_octree_context": bool(comp_debug.get("use_full_octree_context", False)),
        "octree_input_mode": str(comp_debug.get("octree_input_mode", "")),

        # Section2:
        # leaf pattern candidate診断。
        "leaf_pattern_available": bool(comp_debug.get("leaf_pattern_available", False)),
        "leaf_pattern_source": str(comp_debug.get("leaf_pattern_source", "")),
        "leaf_pattern_reason": str(comp_debug.get("leaf_pattern_reason", "")),
        "leaf_unique_parent_count": case_int(comp_debug.get("leaf_unique_parent_count", 0)),
        "leaf_unique_pattern_count": case_int(comp_debug.get("leaf_unique_pattern_count", 0)),
        "leaf_mean_child_count": case_float(comp_debug.get("leaf_mean_child_count", float("nan")), float("nan")),
        "leaf_single_child_parent_ratio": case_float(comp_debug.get("leaf_single_child_parent_ratio", float("nan")), float("nan")),
        "leaf_max_pattern_frequency": case_float(comp_debug.get("leaf_max_pattern_frequency", float("nan")), float("nan")),
        "leaf_candidate_available": bool(comp_debug.get("leaf_candidate_available", False)),
        "leaf_delete_gain_mean": case_float(comp_debug.get("leaf_delete_gain_mean", float("nan")), float("nan")),
        "leaf_add_gain_mean": case_float(comp_debug.get("leaf_add_gain_mean", float("nan")), float("nan")),
        "leaf_move_gain_mean": case_float(comp_debug.get("leaf_move_gain_mean", float("nan")), float("nan")),
        "leaf_high_gain_candidate_ratio": case_float(comp_debug.get("leaf_high_gain_candidate_ratio", float("nan")), float("nan")),

        # Section3:
        "leaf_feature_integration_used": bool(comp_debug.get("leaf_feature_integration_used", False)),
        "leaf_feature_best_gain_mean": case_float(comp_debug.get("leaf_feature_best_gain_mean", float("nan")), float("nan")),
        "leaf_feature_best_gain_max": case_float(comp_debug.get("leaf_feature_best_gain_max", float("nan")), float("nan")),

        # Section4:
        "leaf_actuator_prior_enabled": bool(comp_debug.get("leaf_actuator_prior_enabled", False)),
        "leaf_actuator_drop_prior_mean": case_float(comp_debug.get("leaf_actuator_drop_prior_mean", float("nan")), float("nan")),
        "leaf_actuator_add_prior_mean": case_float(comp_debug.get("leaf_actuator_add_prior_mean", float("nan")), float("nan")),
        "leaf_actuator_move_prior_mean": case_float(comp_debug.get("leaf_actuator_move_prior_mean", float("nan")), float("nan")),
        "leaf_actuator_best_prior_mean": case_float(comp_debug.get("leaf_actuator_best_prior_mean", float("nan")), float("nan")),
        "leaf_actuator_best_prior_max": case_float(comp_debug.get("leaf_actuator_best_prior_max", float("nan")), float("nan")),

        "structural_voxel_mode": str(comp_debug.get("structural_voxel_mode", "")),
        "point_feature_voxel_mode": str(comp_debug.get("point_feature_voxel_mode", "")),
        "selected_subtree_key": str(comp_debug.get("selected_subtree_key", "")),
        "selected_subtree_path": str(comp_debug.get("selected_subtree_path", "")),
        "root_to_subtree_path": str(comp_debug.get("root_to_subtree_path", "")),
        "global_offset": str(comp_debug.get("global_offset", "")),
        "local_offset": str(comp_debug.get("local_offset", "")),
        "global_depth": case_int(comp_debug.get("global_depth", 0)),
        "local_depth": case_int(comp_debug.get("local_depth", 0)),
        "parent_occupancy_code": str(comp_debug.get("parent_occupancy_code", "")),
        "sibling_count": case_int(comp_debug.get("sibling_count", 0)),
        "enable_sparsepcgc_exact_occupancy_teacher": bool(comp_debug.get("enable_sparsepcgc_exact_occupancy_teacher", False)),
        "sparsepcgc_exact_teacher_mode": str(comp_debug.get("sparsepcgc_exact_teacher_mode", "")),
        "exact_teacher_uses_full_context": bool(comp_debug.get("exact_teacher_uses_full_context", False)),
        "exact_teacher_fallback_reason": str(comp_debug.get("exact_teacher_fallback_reason", "")),
        "compression_proxy_input_mode": str(comp_debug.get("compression_proxy_input_mode", "")),
        "compression_proxy_uses_subtree_tree": bool(comp_debug.get("compression_proxy_uses_subtree_tree", False)),
        "compression_proxy_uses_full_context": bool(comp_debug.get("compression_proxy_uses_full_context", False)),
        "rate_proxy_source": str(comp_debug.get("rate_proxy_source", "")),
        "L_com_source": str(comp_debug.get("L_com_source", "")),
        "loss_nodes_source": str(comp_debug.get("loss_nodes_source", "")),
        "loss_single_source": str(comp_debug.get("loss_single_source", "")),
        "compression_proxy_fallback_reason": str(comp_debug.get("compression_proxy_fallback_reason", "")),
        "prebuilt_node_count_used": case_float(comp_debug.get("prebuilt_node_count_used", float("nan")), float("nan")),
        "prebuilt_single_child_count_used": case_float(comp_debug.get("prebuilt_single_child_count_used", float("nan")), float("nan")),
        "rate_proxy_node_count_used": case_float(comp_debug.get("rate_proxy_node_count_used", float("nan")), float("nan")),
        "loss_nodes_node_count_used": case_float(comp_debug.get("loss_nodes_node_count_used", float("nan")), float("nan")),
        "exact_teacher_candidate_source": str(comp_debug.get("exact_teacher_candidate_source", "")),
        "exact_teacher_label_source": str(comp_debug.get("exact_teacher_label_source", "")),
        "sparsepcgc_exact_candidate_count": case_int(comp_debug.get("sparsepcgc_exact_candidate_count", 0)),
        "sparsepcgc_exact_occupied_count": case_int(comp_debug.get("sparsepcgc_exact_occupied_count", 0)),
        "sparsepcgc_exact_occupancy_label_ratio": case_float(comp_debug.get("sparsepcgc_exact_occupancy_label_ratio", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_mean": case_float(comp_debug.get("sparsepcgc_exact_prob_mean", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_entropy": case_float(comp_debug.get("sparsepcgc_exact_prob_entropy", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_true_mean": case_float(comp_debug.get("sparsepcgc_exact_prob_true_mean", float("nan")), float("nan")),
        "sparsepcgc_exact_occupancy_nll": case_float(comp_debug.get("sparsepcgc_exact_occupancy_nll", float("nan")), float("nan")),
        "sparsepcgc_exact_estimated_bits": case_float(comp_debug.get("sparsepcgc_exact_estimated_bits", float("nan")), float("nan")),
        "sparsepcgc_exact_estimated_bpp": case_float(comp_debug.get("sparsepcgc_exact_estimated_bpp", float("nan")), float("nan")),
        "sparsepcgc_exact_low_prob_ratio": case_float(comp_debug.get("sparsepcgc_exact_low_prob_ratio", float("nan")), float("nan")),
        "sparsepcgc_exact_bce_bits": case_float(comp_debug.get("sparsepcgc_exact_bce_bits", float("nan")), float("nan")),
        "sparsepcgc_exact_actual_bitstream_bits": case_float(comp_debug.get("sparsepcgc_exact_actual_bitstream_bits", float("nan")), float("nan")),
        "sparsepcgc_exact_occupancy_nll_delta": case_float(comp_debug.get("sparsepcgc_exact_occupancy_nll_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_estimated_bits_delta": case_float(comp_debug.get("sparsepcgc_exact_estimated_bits_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_bpp_delta": case_float(comp_debug.get("sparsepcgc_exact_bpp_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_loss_candidate": case_float(comp_debug.get("sparsepcgc_exact_loss_candidate", float("nan")), float("nan")),
        "sparsepcgc_exact_loss_enabled": bool(comp_debug.get("sparsepcgc_exact_loss_enabled", False)),
        "exact_estimated_vs_actual_bit_gap": case_float(comp_debug.get("exact_estimated_vs_actual_bit_gap", float("nan")), float("nan")),
        "exact_estimated_vs_actual_bit_gap_percent": case_float(comp_debug.get("exact_estimated_vs_actual_bit_gap_percent", float("nan")), float("nan")),
        "exact_bits_impl": case_float(comp_debug.get("exact_bits_impl", float("nan")), float("nan")),
        "exact_bits_sparsepcgc_estimate_bitrate": case_float(comp_debug.get("exact_bits_sparsepcgc_estimate_bitrate", float("nan")), float("nan")),
        "exact_bits_abs_diff": case_float(comp_debug.get("exact_bits_abs_diff", float("nan")), float("nan")),
        "exact_bits_rel_diff": case_float(comp_debug.get("exact_bits_rel_diff", float("nan")), float("nan")),
        "exact_bits_match": bool(comp_debug.get("exact_bits_match", False)),
        "voxel_collision_input_gt_raw_point_count": case_int(comp_debug.get("voxel_collision_input_gt_raw_point_count", 0)),
        "voxel_collision_input_gt_unique_voxel_count": case_int(comp_debug.get("voxel_collision_input_gt_unique_voxel_count", 0)),
        "voxel_collision_input_gt_duplicate_point_count": case_int(comp_debug.get("voxel_collision_input_gt_duplicate_point_count", 0)),
        "voxel_collision_input_gt_duplicate_rate": case_float(comp_debug.get("voxel_collision_input_gt_duplicate_rate", float("nan")), float("nan")),
        "voxel_collision_input_gt_max_points_per_voxel": case_int(comp_debug.get("voxel_collision_input_gt_max_points_per_voxel", 0)),
        "voxel_collision_input_gt_point_reduction_rate": case_float(comp_debug.get("voxel_collision_input_gt_point_reduction_rate", float("nan")), float("nan")),
        "voxel_collision_model_output_raw_raw_point_count": case_int(comp_debug.get("voxel_collision_model_output_raw_raw_point_count", 0)),
        "voxel_collision_model_output_raw_unique_voxel_count": case_int(comp_debug.get("voxel_collision_model_output_raw_unique_voxel_count", 0)),
        "voxel_collision_model_output_raw_duplicate_point_count": case_int(comp_debug.get("voxel_collision_model_output_raw_duplicate_point_count", 0)),
        "voxel_collision_model_output_raw_duplicate_rate": case_float(comp_debug.get("voxel_collision_model_output_raw_duplicate_rate", float("nan")), float("nan")),
        "voxel_collision_model_output_raw_max_points_per_voxel": case_int(comp_debug.get("voxel_collision_model_output_raw_max_points_per_voxel", 0)),
        "voxel_collision_model_output_raw_point_reduction_rate": case_float(comp_debug.get("voxel_collision_model_output_raw_point_reduction_rate", float("nan")), float("nan")),
        "voxel_collision_compression_input_raw_point_count": case_int(comp_debug.get("voxel_collision_compression_input_raw_point_count", 0)),
        "voxel_collision_compression_input_unique_voxel_count": case_int(comp_debug.get("voxel_collision_compression_input_unique_voxel_count", 0)),
        "voxel_collision_compression_input_duplicate_point_count": case_int(comp_debug.get("voxel_collision_compression_input_duplicate_point_count", 0)),
        "voxel_collision_compression_input_duplicate_rate": case_float(comp_debug.get("voxel_collision_compression_input_duplicate_rate", float("nan")), float("nan")),
        "voxel_collision_compression_input_max_points_per_voxel": case_int(comp_debug.get("voxel_collision_compression_input_max_points_per_voxel", 0)),
        "voxel_collision_compression_input_point_reduction_rate": case_float(comp_debug.get("voxel_collision_compression_input_point_reduction_rate", float("nan")), float("nan")),
        "node_delta": case_float(comp_debug.get("node_delta", float("nan")), float("nan")),
        "single_delta": case_float(comp_debug.get("single_delta", float("nan")), float("nan")),
        "teacher_refresh": bool(comp_debug.get("teacher_refresh", False)),
        "teacher_cache_hit": comp_debug.get("teacher_cache_hit", None),
        "teacher_target_age": case_int(comp_debug.get("teacher_target_age", 0)),
        "actual_codec_disabled": bool(comp_debug.get("actual_codec_disabled_during_train", False)),
        "actual_codec_skipped_by_interval": bool(comp_debug.get("actual_codec_skipped_by_interval", False)),
        "actual_codec_fallback_to_proxy": bool(comp_debug.get("actual_codec_fallback_to_proxy", False)),
        "optimizer_step": bool(comp_debug.get("optimizer_step", False)),
        "optimizer_skip_reason": str(comp_debug.get("optimizer_skip_reason", "")),
        "optimizer_step_success_rate_episode": case_float(comp_debug.get("optimizer_step_success_rate_episode", float("nan")), float("nan")),
        "nonfinite_grad_bad_element_count": case_int(comp_debug.get("nonfinite_grad_bad_element_count", 0)),
        "nonfinite_grad_checked_param_count": case_int(comp_debug.get("nonfinite_grad_checked_param_count", 0)),
        "nonfinite_grad_checked_element_count": case_int(comp_debug.get("nonfinite_grad_checked_element_count", 0)),
        "consecutive_nonfinite_grad_skips": case_int(comp_debug.get("consecutive_nonfinite_grad_skips", 0)),
        "nonfinite_grad_summary": str(comp_debug.get("nonfinite_grad_summary", "")),
        "loss_mode": str(comp_debug.get("loss_mode", getattr(args, "loss_mode", "legacy_total"))),
        "cp_main_source": str(comp_debug.get("cp_main_source", "")),
        "cp_warmup": case_float(comp_debug.get("cp_warmup", float("nan")), float("nan")),
        "cp_L_com_main": case_float(comp_debug.get("cp_L_com_main", float("nan")), float("nan")),
        "cp_L_com_primary": case_float(comp_debug.get("cp_L_com_primary", float("nan")), float("nan")),
        "cp_P_geom": case_float(comp_debug.get("cp_P_geom", float("nan")), float("nan")),
        "cp_P_single": case_float(comp_debug.get("cp_P_single", float("nan")), float("nan")),
        "cp_P_nodes": case_float(comp_debug.get("cp_P_nodes", float("nan")), float("nan")),
        "cp_P_sparsepcgc": case_float(comp_debug.get("cp_P_sparsepcgc", float("nan")), float("nan")),
        "cp_P_actuator": case_float(comp_debug.get("cp_P_actuator", float("nan")), float("nan")),
        "cp_P_op": case_float(comp_debug.get("cp_P_op", float("nan")), float("nan")),
        "cp_total": case_float(comp_debug.get("cp_total", float("nan")), float("nan")),
        "cp_main_requires_grad": comp_debug.get("cp_main_requires_grad", None),
        "cp_geom_requires_grad": comp_debug.get("cp_geom_requires_grad", None),
        "cp_single_requires_grad": comp_debug.get("cp_single_requires_grad", None),
        "cp_nodes_requires_grad": comp_debug.get("cp_nodes_requires_grad", None),
        "cp_sparsepcgc_requires_grad": comp_debug.get("cp_sparsepcgc_requires_grad", None),
        "cp_actuator_requires_grad": comp_debug.get("cp_actuator_requires_grad", None),
        "cp_op_requires_grad": comp_debug.get("cp_op_requires_grad", None),
        "cp_main_finite": comp_debug.get("cp_main_finite", None),
        "cp_geom_finite": comp_debug.get("cp_geom_finite", None),
        "cp_single_finite": comp_debug.get("cp_single_finite", None),
        "cp_nodes_finite": comp_debug.get("cp_nodes_finite", None),
        "cp_sparsepcgc_finite": comp_debug.get("cp_sparsepcgc_finite", None),
        "cp_actuator_finite": comp_debug.get("cp_actuator_finite", None),
        "cp_op_finite": comp_debug.get("cp_op_finite", None),
        "corr_surrogate_actual": case_float(comp_debug.get("corr_surrogate_actual", float("nan")), float("nan")),
        "corr_lcom_actual": case_float(comp_debug.get("corr_lcom_actual", float("nan")), float("nan")),
        "corr_cp_main_actual": case_float(comp_debug.get("corr_cp_main_actual", float("nan")), float("nan")),
        "corr_sparsepcgc_aux_actual": case_float(comp_debug.get("corr_sparsepcgc_aux_actual", float("nan")), float("nan")),
        "corr_lcom_without_sparsepcgc_aux_actual": case_float(comp_debug.get("corr_lcom_without_sparsepcgc_aux_actual", float("nan")), float("nan")),
        "sign_match_surrogate_actual": case_float(comp_debug.get("sign_match_surrogate_actual", float("nan")), float("nan")),
        "sign_match_lcom_actual": case_float(comp_debug.get("sign_match_lcom_actual", float("nan")), float("nan")),
        "sign_match_cp_main_actual": case_float(comp_debug.get("sign_match_cp_main_actual", float("nan")), float("nan")),
        "sign_match_sparsepcgc_aux_actual": case_float(comp_debug.get("sign_match_sparsepcgc_aux_actual", float("nan")), float("nan")),
        "sign_match_lcom_without_sparsepcgc_aux_actual": case_float(comp_debug.get("sign_match_lcom_without_sparsepcgc_aux_actual", float("nan")), float("nan")),
        "rolling_corr_window": case_int(comp_debug.get("rolling_corr_window", getattr(args, "sparsepcgc_corr_window", 100))),
        "rolling_sign_match_window": case_int(comp_debug.get("rolling_sign_match_window", getattr(args, "sparsepcgc_corr_window", 100))),
        "active_coord_before": case_int(comp_debug.get("sparsepcgc_before_active_coords", 0)),
        "active_coord_after": case_int(comp_debug.get("sparsepcgc_after_active_coords", 0)),
        "active_coord_delta": case_int(comp_debug.get("sparsepcgc_active_coord_delta", 0)),
        "isolated_voxel_count": case_int(comp_debug.get("sparsepcgc_after_isolated_voxels", 0)),
        "isolated_voxel_delta": case_int(comp_debug.get("sparsepcgc_isolated_delta", 0)),
        "sparse_density_before": case_float(comp_debug.get("sparsepcgc_before_sparse_density", float("nan")), float("nan")),
        "sparse_density_after": case_float(comp_debug.get("sparsepcgc_after_sparse_density", float("nan")), float("nan")),
        "sparse_density_delta": case_float(comp_debug.get("sparsepcgc_sparse_density_delta", float("nan")), float("nan")),
        "occupancy_entropy": case_float(comp_debug.get("occupancy_entropy", comp_debug.get("sparsepcgc_entropy_proxy_loss", float("nan"))), float("nan")),
        "occupancy_nll_proxy": case_float(comp_debug.get("occupancy_nll_proxy", float("nan")), float("nan")),
        "lowprob_occupancy_ratio": case_float(comp_debug.get("lowprob_occupancy_ratio", float("nan")), float("nan")),
        "entropy_delta": case_float(comp_debug.get("sparsepcgc_entropy_proxy_loss", float("nan")), float("nan")),
        "nll_delta": case_float(comp_debug.get("nll_delta", comp_debug.get("occupancy_nll_proxy", float("nan"))), float("nan")),
        "occupancy_pattern_before": case_float(comp_debug.get("occupancy_pattern_before", float("nan")), float("nan")),
        "occupancy_pattern_after": case_float(comp_debug.get("occupancy_pattern_after", float("nan")), float("nan")),
        "occupancy_pattern_delta": case_float(comp_debug.get("occupancy_pattern_delta", float("nan")), float("nan")),
        "lowprob_occupancy_count_before": case_float(comp_debug.get("lowprob_occupancy_count_before", float("nan")), float("nan")),
        "lowprob_occupancy_count_after": case_float(comp_debug.get("lowprob_occupancy_count_after", float("nan")), float("nan")),
        "occupancy_entropy_before": case_float(comp_debug.get("occupancy_entropy_before", float("nan")), float("nan")),
        "occupancy_entropy_after": case_float(comp_debug.get("occupancy_entropy_after", float("nan")), float("nan")),
        "occupancy_entropy_delta": case_float(comp_debug.get("occupancy_entropy_delta", float("nan")), float("nan")),
        "occupancy_nll_before": case_float(comp_debug.get("occupancy_nll_before", float("nan")), float("nan")),
        "occupancy_nll_after": case_float(comp_debug.get("occupancy_nll_after", float("nan")), float("nan")),
        "occupancy_nll_delta": case_float(comp_debug.get("occupancy_nll_delta", comp_debug.get("nll_delta", float("nan"))), float("nan")),
        "predicted_occupancy_pattern_delta": case_float(comp_debug.get("occupancy_pattern_delta", float("nan")), float("nan")),
        "predicted_occupancy_entropy_delta": case_float(comp_debug.get("occupancy_entropy_delta", float("nan")), float("nan")),
        "predicted_occupancy_nll_delta": case_float(comp_debug.get("occupancy_nll_delta", comp_debug.get("nll_delta", float("nan"))), float("nan")),
        "predicted_lowprob_occupancy_ratio": case_float(comp_debug.get("lowprob_occupancy_ratio", float("nan")), float("nan")),
        "actual_occupancy_pattern_before": case_float(comp_debug.get("actual_occupancy_pattern_before", float("nan")), float("nan")),
        "actual_occupancy_pattern_after": case_float(comp_debug.get("actual_occupancy_pattern_after", float("nan")), float("nan")),
        "actual_occupancy_pattern_delta": case_float(comp_debug.get("actual_occupancy_pattern_delta", float("nan")), float("nan")),
        "actual_occupancy_entropy_before": case_float(comp_debug.get("actual_occupancy_entropy_before", float("nan")), float("nan")),
        "actual_occupancy_entropy_after": case_float(comp_debug.get("actual_occupancy_entropy_after", float("nan")), float("nan")),
        "actual_occupancy_entropy_delta": case_float(comp_debug.get("actual_occupancy_entropy_delta", float("nan")), float("nan")),
        "actual_occupancy_nll_before": case_float(comp_debug.get("actual_occupancy_nll_before", float("nan")), float("nan")),
        "actual_occupancy_nll_after": case_float(comp_debug.get("actual_occupancy_nll_after", float("nan")), float("nan")),
        "actual_occupancy_nll_delta": case_float(comp_debug.get("actual_occupancy_nll_delta", float("nan")), float("nan")),
        "actual_lowprob_occupancy_count_before": case_float(comp_debug.get("actual_lowprob_occupancy_count_before", float("nan")), float("nan")),
        "actual_lowprob_occupancy_count_after": case_float(comp_debug.get("actual_lowprob_occupancy_count_after", float("nan")), float("nan")),
        "actual_lowprob_occupancy_count_delta": case_float(comp_debug.get("actual_lowprob_occupancy_count_delta", float("nan")), float("nan")),
        "actual_lowprob_occupancy_ratio_before": case_float(comp_debug.get("actual_lowprob_occupancy_ratio_before", float("nan")), float("nan")),
        "actual_lowprob_occupancy_ratio_after": case_float(comp_debug.get("actual_lowprob_occupancy_ratio_after", float("nan")), float("nan")),
        "actual_lowprob_occupancy_ratio_delta": case_float(comp_debug.get("actual_lowprob_occupancy_ratio_delta", float("nan")), float("nan")),
        "actual_occupancy_predictability_before": case_float(comp_debug.get("actual_occupancy_predictability_before", float("nan")), float("nan")),
        "actual_occupancy_predictability_after": case_float(comp_debug.get("actual_occupancy_predictability_after", float("nan")), float("nan")),
        "actual_occupancy_predictability_delta": case_float(comp_debug.get("actual_occupancy_predictability_delta", float("nan")), float("nan")),
        "sparsepcgc_occupancy_debug_available": bool(comp_debug.get("sparsepcgc_occupancy_debug_available", False)),
        "sparsepcgc_candidate_count_before": case_int(comp_debug.get("sparsepcgc_candidate_count_before", 0)),
        "sparsepcgc_candidate_count_after": case_int(comp_debug.get("sparsepcgc_candidate_count_after", 0)),
        "sparsepcgc_candidate_count_delta": case_int(comp_debug.get("sparsepcgc_candidate_count_delta", 0)),
        "sparsepcgc_occupied_candidate_count_before": case_int(comp_debug.get("sparsepcgc_occupied_candidate_count_before", 0)),
        "sparsepcgc_occupied_candidate_count_after": case_int(comp_debug.get("sparsepcgc_occupied_candidate_count_after", 0)),
        "sparsepcgc_actual_occupancy_label_ratio_before": case_float(comp_debug.get("sparsepcgc_actual_occupancy_label_ratio_before", float("nan")), float("nan")),
        "sparsepcgc_actual_occupancy_label_ratio_after": case_float(comp_debug.get("sparsepcgc_actual_occupancy_label_ratio_after", float("nan")), float("nan")),
        "sparsepcgc_actual_occupancy_label_ratio_delta": case_float(comp_debug.get("sparsepcgc_actual_occupancy_label_ratio_delta", float("nan")), float("nan")),
        "sparsepcgc_pred_prob_entropy_before": case_float(comp_debug.get("sparsepcgc_pred_prob_entropy_before", float("nan")), float("nan")),
        "sparsepcgc_pred_prob_entropy_after": case_float(comp_debug.get("sparsepcgc_pred_prob_entropy_after", float("nan")), float("nan")),
        "sparsepcgc_pred_prob_entropy_delta": case_float(comp_debug.get("sparsepcgc_pred_prob_entropy_delta", float("nan")), float("nan")),
        "sparsepcgc_pred_occupancy_nll_before": case_float(comp_debug.get("sparsepcgc_pred_occupancy_nll_before", float("nan")), float("nan")),
        "sparsepcgc_pred_occupancy_nll_after": case_float(comp_debug.get("sparsepcgc_pred_occupancy_nll_after", float("nan")), float("nan")),
        "sparsepcgc_pred_occupancy_nll_delta": case_float(comp_debug.get("sparsepcgc_pred_occupancy_nll_delta", float("nan")), float("nan")),
        "sparsepcgc_estimated_occupancy_bits_before": case_float(comp_debug.get("sparsepcgc_estimated_occupancy_bits_before", float("nan")), float("nan")),
        "sparsepcgc_estimated_occupancy_bits_after": case_float(comp_debug.get("sparsepcgc_estimated_occupancy_bits_after", float("nan")), float("nan")),
        "sparsepcgc_estimated_occupancy_bits_delta": case_float(comp_debug.get("sparsepcgc_estimated_occupancy_bits_delta", float("nan")), float("nan")),
        "sparsepcgc_estimated_occupancy_bpp_before": case_float(comp_debug.get("sparsepcgc_estimated_occupancy_bpp_before", float("nan")), float("nan")),
        "sparsepcgc_estimated_occupancy_bpp_after": case_float(comp_debug.get("sparsepcgc_estimated_occupancy_bpp_after", float("nan")), float("nan")),
        "sparsepcgc_estimated_occupancy_bpp_delta": case_float(comp_debug.get("sparsepcgc_estimated_occupancy_bpp_delta", float("nan")), float("nan")),
        "sparsepcgc_prob_true_mean_before": case_float(comp_debug.get("sparsepcgc_prob_true_mean_before", float("nan")), float("nan")),
        "sparsepcgc_prob_true_mean_after": case_float(comp_debug.get("sparsepcgc_prob_true_mean_after", float("nan")), float("nan")),
        "sparsepcgc_prob_true_mean_delta": case_float(comp_debug.get("sparsepcgc_prob_true_mean_delta", float("nan")), float("nan")),
        "sparsepcgc_prob_true_low_count_before": case_float(comp_debug.get("sparsepcgc_prob_true_low_count_before", float("nan")), float("nan")),
        "sparsepcgc_prob_true_low_count_after": case_float(comp_debug.get("sparsepcgc_prob_true_low_count_after", float("nan")), float("nan")),
        "sparsepcgc_prob_true_low_ratio_before": case_float(comp_debug.get("sparsepcgc_prob_true_low_ratio_before", float("nan")), float("nan")),
        "sparsepcgc_prob_true_low_ratio_after": case_float(comp_debug.get("sparsepcgc_prob_true_low_ratio_after", float("nan")), float("nan")),
        "sparsepcgc_prob_true_low_ratio_delta": case_float(comp_debug.get("sparsepcgc_prob_true_low_ratio_delta", float("nan")), float("nan")),
        "sparsepcgc_occupied_low_prob_count_before": case_float(comp_debug.get("sparsepcgc_occupied_low_prob_count_before", float("nan")), float("nan")),
        "sparsepcgc_occupied_low_prob_count_after": case_float(comp_debug.get("sparsepcgc_occupied_low_prob_count_after", float("nan")), float("nan")),
        "sparsepcgc_occupied_low_prob_ratio_before": case_float(comp_debug.get("sparsepcgc_occupied_low_prob_ratio_before", float("nan")), float("nan")),
        "sparsepcgc_occupied_low_prob_ratio_after": case_float(comp_debug.get("sparsepcgc_occupied_low_prob_ratio_after", float("nan")), float("nan")),
        "sparsepcgc_occupied_low_prob_ratio_delta": case_float(comp_debug.get("sparsepcgc_occupied_low_prob_ratio_delta", float("nan")), float("nan")),
        "sparsepcgc_low_prob_threshold": case_float(comp_debug.get("sparsepcgc_low_prob_threshold", float("nan")), float("nan")),
        "sparsepcgc_exact_candidate_count_before": case_int(comp_debug.get("sparsepcgc_exact_candidate_count_before", 0)),
        "sparsepcgc_exact_candidate_count_after": case_int(comp_debug.get("sparsepcgc_exact_candidate_count_after", 0)),
        "sparsepcgc_exact_candidate_count_delta": case_int(comp_debug.get("sparsepcgc_exact_candidate_count_delta", 0)),
        "sparsepcgc_exact_occupied_count_before": case_int(comp_debug.get("sparsepcgc_exact_occupied_count_before", 0)),
        "sparsepcgc_exact_occupied_count_after": case_int(comp_debug.get("sparsepcgc_exact_occupied_count_after", 0)),
        "sparsepcgc_exact_occupancy_label_ratio_before": case_float(comp_debug.get("sparsepcgc_exact_occupancy_label_ratio_before", float("nan")), float("nan")),
        "sparsepcgc_exact_occupancy_label_ratio_after": case_float(comp_debug.get("sparsepcgc_exact_occupancy_label_ratio_after", float("nan")), float("nan")),
        "sparsepcgc_exact_occupancy_label_ratio_delta": case_float(comp_debug.get("sparsepcgc_exact_occupancy_label_ratio_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_mean_before": case_float(comp_debug.get("sparsepcgc_exact_prob_mean_before", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_mean_after": case_float(comp_debug.get("sparsepcgc_exact_prob_mean_after", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_mean_delta": case_float(comp_debug.get("sparsepcgc_exact_prob_mean_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_entropy_before": case_float(comp_debug.get("sparsepcgc_exact_prob_entropy_before", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_entropy_after": case_float(comp_debug.get("sparsepcgc_exact_prob_entropy_after", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_entropy_delta": case_float(comp_debug.get("sparsepcgc_exact_prob_entropy_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_true_mean_before": case_float(comp_debug.get("sparsepcgc_exact_prob_true_mean_before", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_true_mean_after": case_float(comp_debug.get("sparsepcgc_exact_prob_true_mean_after", float("nan")), float("nan")),
        "sparsepcgc_exact_prob_true_mean_delta": case_float(comp_debug.get("sparsepcgc_exact_prob_true_mean_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_occupancy_nll_before": case_float(comp_debug.get("sparsepcgc_exact_occupancy_nll_before", float("nan")), float("nan")),
        "sparsepcgc_exact_occupancy_nll_after": case_float(comp_debug.get("sparsepcgc_exact_occupancy_nll_after", float("nan")), float("nan")),
        "sparsepcgc_exact_occupancy_nll_delta": case_float(comp_debug.get("sparsepcgc_exact_occupancy_nll_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_estimated_bits_before": case_float(comp_debug.get("sparsepcgc_exact_estimated_bits_before", float("nan")), float("nan")),
        "sparsepcgc_exact_estimated_bits_after": case_float(comp_debug.get("sparsepcgc_exact_estimated_bits_after", float("nan")), float("nan")),
        "sparsepcgc_exact_estimated_bits_delta": case_float(comp_debug.get("sparsepcgc_exact_estimated_bits_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_estimated_bpp_before": case_float(comp_debug.get("sparsepcgc_exact_estimated_bpp_before", float("nan")), float("nan")),
        "sparsepcgc_exact_estimated_bpp_after": case_float(comp_debug.get("sparsepcgc_exact_estimated_bpp_after", float("nan")), float("nan")),
        "sparsepcgc_exact_estimated_bpp_delta": case_float(comp_debug.get("sparsepcgc_exact_estimated_bpp_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_low_prob_ratio_before": case_float(comp_debug.get("sparsepcgc_exact_low_prob_ratio_before", float("nan")), float("nan")),
        "sparsepcgc_exact_low_prob_ratio_after": case_float(comp_debug.get("sparsepcgc_exact_low_prob_ratio_after", float("nan")), float("nan")),
        "sparsepcgc_exact_low_prob_ratio_delta": case_float(comp_debug.get("sparsepcgc_exact_low_prob_ratio_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_bce_bits_before": case_float(comp_debug.get("sparsepcgc_exact_bce_bits_before", float("nan")), float("nan")),
        "sparsepcgc_exact_bce_bits_after": case_float(comp_debug.get("sparsepcgc_exact_bce_bits_after", float("nan")), float("nan")),
        "sparsepcgc_exact_bce_bits_delta": case_float(comp_debug.get("sparsepcgc_exact_bce_bits_delta", float("nan")), float("nan")),
        "sparsepcgc_exact_actual_bitstream_bits_before": case_float(comp_debug.get("sparsepcgc_exact_actual_bitstream_bits_before", float("nan")), float("nan")),
        "sparsepcgc_exact_actual_bitstream_bits_after": case_float(comp_debug.get("sparsepcgc_exact_actual_bitstream_bits_after", float("nan")), float("nan")),
        "sparsepcgc_exact_actual_bitstream_bits_delta": case_float(comp_debug.get("sparsepcgc_exact_actual_bitstream_bits_delta", float("nan")), float("nan")),
        "exact_bits_impl_before": case_float(comp_debug.get("exact_bits_impl_before", float("nan")), float("nan")),
        "exact_bits_impl_after": case_float(comp_debug.get("exact_bits_impl_after", float("nan")), float("nan")),
        "exact_bits_impl_delta": case_float(comp_debug.get("exact_bits_impl_delta", float("nan")), float("nan")),
        "exact_bits_sparsepcgc_estimate_bitrate_before": case_float(comp_debug.get("exact_bits_sparsepcgc_estimate_bitrate_before", float("nan")), float("nan")),
        "exact_bits_sparsepcgc_estimate_bitrate_after": case_float(comp_debug.get("exact_bits_sparsepcgc_estimate_bitrate_after", float("nan")), float("nan")),
        "exact_bits_sparsepcgc_estimate_bitrate_delta": case_float(comp_debug.get("exact_bits_sparsepcgc_estimate_bitrate_delta", float("nan")), float("nan")),
        "exact_bits_abs_diff_before": case_float(comp_debug.get("exact_bits_abs_diff_before", float("nan")), float("nan")),
        "exact_bits_abs_diff_after": case_float(comp_debug.get("exact_bits_abs_diff_after", float("nan")), float("nan")),
        "exact_bits_abs_diff_delta": case_float(comp_debug.get("exact_bits_abs_diff_delta", float("nan")), float("nan")),
        "exact_bits_rel_diff_before": case_float(comp_debug.get("exact_bits_rel_diff_before", float("nan")), float("nan")),
        "exact_bits_rel_diff_after": case_float(comp_debug.get("exact_bits_rel_diff_after", float("nan")), float("nan")),
        "exact_bits_rel_diff_delta": case_float(comp_debug.get("exact_bits_rel_diff_delta", float("nan")), float("nan")),
        "exact_estimated_vs_actual_bit_gap_before": case_float(comp_debug.get("exact_estimated_vs_actual_bit_gap_before", float("nan")), float("nan")),
        "exact_estimated_vs_actual_bit_gap_after": case_float(comp_debug.get("exact_estimated_vs_actual_bit_gap_after", float("nan")), float("nan")),
        "exact_estimated_vs_actual_bit_gap_delta": case_float(comp_debug.get("exact_estimated_vs_actual_bit_gap_delta", float("nan")), float("nan")),
        "exact_estimated_vs_actual_bit_gap_percent_before": case_float(comp_debug.get("exact_estimated_vs_actual_bit_gap_percent_before", float("nan")), float("nan")),
        "exact_estimated_vs_actual_bit_gap_percent_after": case_float(comp_debug.get("exact_estimated_vs_actual_bit_gap_percent_after", float("nan")), float("nan")),
        "exact_estimated_vs_actual_bit_gap_percent_delta": case_float(comp_debug.get("exact_estimated_vs_actual_bit_gap_percent_delta", float("nan")), float("nan")),
        "octree_pattern_entropy_before": case_float(comp_debug.get("octree_pattern_entropy_before", float("nan")), float("nan")),
        "octree_pattern_entropy_after": case_float(comp_debug.get("octree_pattern_entropy_after", float("nan")), float("nan")),
        "octree_pattern_entropy_delta": case_float(comp_debug.get("octree_pattern_entropy_delta", float("nan")), float("nan")),
        "octree_pattern_nll_before": case_float(comp_debug.get("octree_pattern_nll_before", float("nan")), float("nan")),
        "octree_pattern_nll_after": case_float(comp_debug.get("octree_pattern_nll_after", float("nan")), float("nan")),
        "octree_pattern_nll_delta": case_float(comp_debug.get("octree_pattern_nll_delta", float("nan")), float("nan")),
        "octree_pattern_lowprob_ratio": case_float(comp_debug.get("octree_pattern_lowprob_ratio", float("nan")), float("nan")),
        "occupancy_proxy_definition": str(comp_debug.get("occupancy_proxy_definition", "")),
        "actual_occupancy_definition": str(comp_debug.get("actual_occupancy_definition", "")),
        "predicted_occupancy_definition": str(comp_debug.get("predicted_occupancy_definition", "")),
        "single_child_chain_length_before": case_float(comp_debug.get("single_child_chain_length_before", float("nan")), float("nan")),
        "single_child_chain_length_after": case_float(comp_debug.get("single_child_chain_length_after", float("nan")), float("nan")),
        "sibling_occupancy_balance_before": case_float(comp_debug.get("sibling_occupancy_balance_before", float("nan")), float("nan")),
        "sibling_occupancy_balance_after": case_float(comp_debug.get("sibling_occupancy_balance_after", float("nan")), float("nan")),
        "corr_occupancy_actual": case_float(comp_debug.get("corr_occupancy_actual", float("nan")), float("nan")),
        "sign_match_occupancy_actual": case_float(comp_debug.get("sign_match_occupancy_actual", float("nan")), float("nan")),
        "heuristic_cause_score_node": case_float(comp_debug.get("heuristic_cause_score_node", comp_debug.get("soft_node_percent", float("nan"))), float("nan")),
        "heuristic_cause_score_single": case_float(comp_debug.get("heuristic_cause_score_single", comp_debug.get("soft_single_percent", float("nan"))), float("nan")),
        "heuristic_cause_score_lowprob": case_float(comp_debug.get("heuristic_cause_score_lowprob", comp_debug.get("lowprob_occupancy_ratio", float("nan"))), float("nan")),
        "heuristic_sparse_proxy": case_float(comp_debug.get("sparsepcgc_aux_value", comp_debug.get("sparsepcgc_aux_weighted", float("nan"))), float("nan")),
        "heuristic_quant_proxy": case_float(comp_debug.get("sparsepcgc_active_coord_loss", float("nan")), float("nan")),
        "heuristic_node_proxy": case_float(comp_debug.get("soft_node_percent", float("nan")), float("nan")),
        "actual_bit_delta": case_float(comp_debug.get("gen_actual_bit", 0.0), 0.0) - case_float(comp_debug.get("gt_actual_bit", 0.0), 0.0),
        "actual_bit_delta_percent": actual_delta if math.isfinite(actual_delta) else None,
        "cause_actual_corr": case_float(comp_debug.get("cause_actual_corr", comp_debug.get("corr_sparsepcgc_aux_actual", float("nan"))), float("nan")),
        "cause_actual_sign_match": case_float(comp_debug.get("cause_actual_sign_match", comp_debug.get("sign_match_sparsepcgc_aux_actual", float("nan"))), float("nan")),
        "corr_node_actual": case_float(comp_debug.get("corr_node_actual", float("nan")), float("nan")),
        "corr_single_actual": case_float(comp_debug.get("corr_single_actual", float("nan")), float("nan")),
        "sign_match_node_actual": case_float(comp_debug.get("sign_match_node_actual", float("nan")), float("nan")),
        "sign_match_single_actual": case_float(comp_debug.get("sign_match_single_actual", float("nan")), float("nan")),
        "node_loss_weight_raw": case_float(comp_debug.get("node_loss_weight_raw", float("nan")), float("nan")),
        "node_loss_weight_effective": case_float(comp_debug.get("node_loss_weight_effective", float("nan")), float("nan")),
        "single_loss_weight_raw": case_float(comp_debug.get("single_loss_weight_raw", float("nan")), float("nan")),
        "single_loss_weight_effective": case_float(comp_debug.get("single_loss_weight_effective", float("nan")), float("nan")),
        "node_single_gating_reason": str(comp_debug.get("node_single_gating_reason", "")),
        "single_delta_penalty": case_float(comp_debug.get("single_delta_penalty", float("nan")), float("nan")),
        "single_delta_penalty_weight": case_float(comp_debug.get("single_delta_penalty_weight", float("nan")), float("nan")),
        "single_delta_penalty_used_for_backprop": bool(comp_debug.get("single_delta_penalty_used_for_backprop", False)),
        "single_delta_positive_ratio": case_float(comp_debug.get("single_delta_positive_ratio", float("nan")), float("nan")),
        "corr_single_delta_actual": case_float(comp_debug.get("corr_single_delta_actual", float("nan")), float("nan")),
        "point_delta_actual_corr": case_float(comp_debug.get("point_delta_actual_corr", float("nan")), float("nan")),
        "point_reduction_actual_improved_ratio": case_float(comp_debug.get("point_reduction_actual_improved_ratio", float("nan")), float("nan")),
        "actual_moving_avg": case_float(comp_debug.get("actual_moving_avg", float("nan")), float("nan")),
        "actual_moving_avg_delta": case_float(comp_debug.get("actual_moving_avg_delta", float("nan")), float("nan")),
        "actual_moving_avg_improving": bool(comp_debug.get("actual_moving_avg_improving", False)),
        "actual_negative_stable": bool(comp_debug.get("actual_negative_stable", False)),
        "lr_decay_allowed_by_actual": bool(comp_debug.get("lr_decay_allowed_by_actual", True)),
        "cause_score_used_for_backprop": bool(comp_debug.get("sparsepcgc_aux_used_for_backprop", False)),
        "cause_score_is_actual_teacher": False,
        "cause_score_is_heuristic": True,
        "sparsepcgc_condition_voxel_size": case_float(comp_debug.get("sparsepcgc_condition_voxel_size", float("nan")), float("nan")),
        "sparsepcgc_condition_pos_quantscale": case_int(comp_debug.get("sparsepcgc_condition_pos_quantscale", 0)),
        "sparsepcgc_condition_actual_quant_mode": str(comp_debug.get("sparsepcgc_condition_actual_quant_mode", "")),
        "sparsepcgc_condition_proxy_quant_mode": str(comp_debug.get("sparsepcgc_condition_proxy_quant_mode", "")),
        "sparsepcgc_condition_rounding": str(comp_debug.get("sparsepcgc_condition_rounding", "")),
        "sparsepcgc_condition_dedup": str(comp_debug.get("sparsepcgc_condition_dedup", "")),
        "sparsepcgc_condition_teacher_scope": str(comp_debug.get("sparsepcgc_condition_teacher_scope", "")),
        "sparsepcgc_condition_gt_bbox_min": str(comp_debug.get("sparsepcgc_condition_gt_bbox_min", "")),
        "sparsepcgc_condition_gt_bbox_max": str(comp_debug.get("sparsepcgc_condition_gt_bbox_max", "")),
        "sparsepcgc_condition_gen_bbox_min": str(comp_debug.get("sparsepcgc_condition_gen_bbox_min", "")),
        "sparsepcgc_condition_gen_bbox_max": str(comp_debug.get("sparsepcgc_condition_gen_bbox_max", "")),
        "sparsepcgc_condition_local_min_offset": str(comp_debug.get("sparsepcgc_condition_local_min_offset", "")),
        "sparsepcgc_condition_warning": str(comp_debug.get("sparsepcgc_condition_warning", "")),
        "surrogate_input_feature_names": str(comp_debug.get("surrogate_input_feature_names", "")),
        "surrogate_input_feature_dim": case_int(comp_debug.get("surrogate_input_feature_dim", 0)),
        "surrogate_uses_operation_features": bool(comp_debug.get("surrogate_uses_operation_features", False)),
        "surrogate_uses_codec_condition": bool(comp_debug.get("surrogate_uses_codec_condition", False)),
        "surrogate_uses_quant_condition": bool(comp_debug.get("surrogate_uses_quant_condition", False)),
        "surrogate_uses_occupancy_features": bool(comp_debug.get("surrogate_uses_occupancy_features", False)),
        "surrogate_uses_before_after_delta": bool(comp_debug.get("surrogate_uses_before_after_delta", False)),
        "surrogate_input_version": str(comp_debug.get("surrogate_input_version", "")),
    }


def build_operation_metric_row(
    args,
    *,
    global_step,
    episode,
    epoch,
    step,
    stage,
    comp_debug,
    structure_debug,
    edit_stats,
    sequence_name=None,
    sequence_step=None,
):
    actual_delta = case_float(comp_debug.get("actual_total_bit_percent", float("nan")), float("nan"))
    unique_before = case_int(comp_debug.get("gt_unique_coord_count", 0))
    unique_after = case_int(comp_debug.get("gen_unique_coord_count", 0))
    edit_stats = edit_stats or {}
    input_points = case_float(edit_stats.get("input_points_avg", edit_stats.get("input_points", float("nan"))), float("nan"))
    add_candidate_ratio = case_float(structure_debug.get("add_candidate_ratio", float("nan")), float("nan"))
    add_candidate_count = None
    debug_add_candidate_count = structure_debug.get("add_candidate_count", None)
    if debug_add_candidate_count is not None:
        add_candidate_count = case_int(debug_add_candidate_count, 0)
    elif math.isfinite(input_points) and math.isfinite(add_candidate_ratio):
        add_candidate_count = int(round(input_points * add_candidate_ratio))
    add_effective_count = case_int(structure_debug.get("add_effective_count", structure_debug.get("add_actual_point_count", 0)))
    unique_delta = unique_after - unique_before
    positive_unique_delta = max(unique_delta, 0)
    add_removed_by_unique = max(add_effective_count - positive_unique_delta, 0)
    active_before = case_int(comp_debug.get("sparsepcgc_before_active_coords", 0))
    active_after = case_int(comp_debug.get("sparsepcgc_after_active_coords", 0))
    level_debug = structure_debug.get("octree_level_debug") or []
    actual_oracle_accepted_candidate_count = case_int(structure_debug.get("actual_oracle_accepted_candidate_count", 0))
    actual_oracle_accepted_prune_count = case_int(structure_debug.get("actual_oracle_accepted_prune_count", 0))
    actual_oracle_eval_count = case_int(structure_debug.get("actual_oracle_eval_count", 0))
    actual_oracle_time = case_float(structure_debug.get("actual_oracle_time", 0.0), 0.0)
    actual_oracle_delta_actual = case_float(structure_debug.get("actual_oracle_delta_actual_percent", float("nan")), float("nan"))
    actual_oracle_proxy = case_float(structure_debug.get("actual_oracle_proxy_percent", float("nan")), float("nan"))
    oracle_full_cloud_override_used = (
        bool(comp_debug.get("oracle_full_cloud_override_used", False))
        or str(comp_debug.get("policy_action_source", "")) == "actual_oracle_full_cloud_override"
    )
    oracle_full_cloud_prune_count = 0
    oracle_full_cloud_prune_ratio_percent = 0.0
    if oracle_full_cloud_override_used:
        oracle_full_cloud_prune_count = case_int(
            structure_debug.get(
                "actual_oracle_full_cloud_macro_best_drop_count",
                comp_debug.get("actual_oracle_full_cloud_macro_best_drop_count", 0),
            )
        )
        oracle_full_cloud_prune_ratio_percent = 100.0 * case_float(
            structure_debug.get(
                "actual_oracle_full_cloud_macro_best_ratio",
                comp_debug.get("actual_oracle_full_cloud_macro_best_ratio", 0.0),
            ),
            0.0,
        )
    voxel_edit_input_count = case_int(edit_stats.get("voxel_edit_input_count", 0), 0)
    if voxel_edit_input_count <= 0:
        voxel_edit_input_count = case_int(
            structure_debug.get(
                "input_voxel_count",
                structure_debug.get(
                    "before_occupied_voxel_count",
                    structure_debug.get("voxel_edit_initial_count", active_before if active_before > 0 else unique_before),
                ),
            ),
            0,
        )
    voxel_edit_add_count = case_int(
        edit_stats.get("voxel_edit_add_count", structure_debug.get("add_target_voxel_count", 0)),
        0,
    )
    voxel_edit_drop_count = case_int(
        edit_stats.get("voxel_edit_drop_count", structure_debug.get("delete_target_voxel_count", 0)),
        0,
    )
    voxel_edit_move_count = case_int(
        edit_stats.get("voxel_edit_move_count", structure_debug.get("move_source_voxel_count", 0)),
        0,
    )
    voxel_edit_final_count = case_int(
        edit_stats.get("voxel_edit_final_count", structure_debug.get("after_occupied_voxel_count", 0)),
        0,
    )
    if voxel_edit_final_count <= 0 and voxel_edit_input_count > 0:
        voxel_edit_final_count = max(voxel_edit_input_count + voxel_edit_add_count - voxel_edit_drop_count, 0)
    full_cloud_voxel_count = case_int(edit_stats.get("full_cloud_voxel_count", 0), 0)
    if full_cloud_voxel_count <= 0:
        full_cloud_voxel_count = case_int(
            getattr(args, "_full_cloud_canonical_coords_count", 0),
            0,
        )
    if full_cloud_voxel_count <= 0:
        full_cloud_voxel_count = case_int(
            comp_debug.get(
                "full_cloud_anchor_unique_coord_before",
                comp_debug.get("gt_unique_coord_count", voxel_edit_input_count),
            ),
            0,
        )
    voxel_add_ratio_percent = case_float(edit_stats.get("voxel_add_ratio_percent", float("nan")), float("nan"))
    if not math.isfinite(voxel_add_ratio_percent):
        voxel_add_ratio_percent = _ratio_percent_or_nan(voxel_edit_add_count, voxel_edit_input_count)
    voxel_drop_ratio_percent = case_float(edit_stats.get("voxel_drop_ratio_percent", float("nan")), float("nan"))
    if not math.isfinite(voxel_drop_ratio_percent):
        voxel_drop_ratio_percent = _ratio_percent_or_nan(voxel_edit_drop_count, voxel_edit_input_count)
    voxel_move_ratio_percent = case_float(edit_stats.get("voxel_move_ratio_percent", float("nan")), float("nan"))
    if not math.isfinite(voxel_move_ratio_percent):
        voxel_move_ratio_percent = _ratio_percent_or_nan(voxel_edit_move_count, voxel_edit_input_count)
    full_cloud_voxel_drop_ratio_percent = case_float(
        edit_stats.get("full_cloud_voxel_drop_ratio_percent", float("nan")),
        float("nan"),
    )
    if not math.isfinite(full_cloud_voxel_drop_ratio_percent):
        full_cloud_voxel_drop_ratio_percent = _ratio_percent_or_nan(
            voxel_edit_drop_count,
            full_cloud_voxel_count if full_cloud_voxel_count > 0 else voxel_edit_input_count,
        )
    fast_diagnostic_used_fallback = (
        actual_oracle_accepted_candidate_count > 0
        and actual_oracle_accepted_prune_count > 0
        and actual_oracle_eval_count == 0
        and abs(actual_oracle_time) <= 1e-9
        and (not math.isfinite(actual_oracle_delta_actual) or abs(actual_oracle_delta_actual) <= 1e-9)
        and math.isfinite(actual_oracle_proxy)
    )
    actual_oracle_fast_diagnostic_used = bool(
        structure_debug.get("actual_oracle_fast_diagnostic_used", False)
    ) or bool(fast_diagnostic_used_fallback)
    actual_oracle_fast_diagnostic_local_count = case_int(
        structure_debug.get("actual_oracle_fast_diagnostic_local_drop_count", 0)
    )
    if actual_oracle_fast_diagnostic_used and actual_oracle_fast_diagnostic_local_count <= 0:
        actual_oracle_fast_diagnostic_local_count = actual_oracle_accepted_prune_count
    actual_oracle_fast_diagnostic_local_add_count = case_int(
        structure_debug.get("actual_oracle_fast_diagnostic_local_add_count", 0)
    )
    if actual_oracle_fast_diagnostic_used and actual_oracle_fast_diagnostic_local_add_count <= 0:
        actual_oracle_fast_diagnostic_local_add_count = case_int(
            structure_debug.get("actual_oracle_accepted_add_count", 0)
        )
    return {
        "episode_index": int(episode),
        "epoch_index": int(epoch),
        "sequence_step": (int(sequence_step) + 1) if sequence_step is not None else int(step) + 1,
        "sequence_name": str(sequence_name or ""),
        "global_step": int(global_step) + 1,
        "episode": int(episode) + 1,
        "epoch": int(epoch) + 1,
        "step": int(step) + 1,
        "stage": str(stage),
        "codec": str(comp_debug.get("teacher_codec", getattr(args, "compress", "unknown"))),
        "fresh_actual": bool(is_fresh_actual(args, comp_debug)),
        "actual_total_bit_percent": actual_delta if math.isfinite(actual_delta) else None,
        "train_or_eval_mode": "train",
        "hardening_mode": str(edit_stats.get("keep_mode", "")),
        "selection_threshold": case_float(getattr(args, "operation_count_drop_threshold", 0.5), float("nan")),
        "topk_selected_count": case_int(edit_stats.get("output_points", 0)),
        "sparsepcgc_add_experiment_enabled": bool(sparsepcgc_add_experiment_active(args)),
        "add_enabled": bool(structure_debug.get("add_enabled", False)),
        "prune_enabled": bool(structure_debug.get("prune_enabled", False)),
        "disp_enabled": bool(structure_debug.get("disp_enabled", False)),
        "drop_operation_gate": case_float(structure_debug.get("drop_operation_gate", float("nan")), float("nan")),
        "add_operation_gate": case_float(structure_debug.get("add_operation_gate", float("nan")), float("nan")),
        "move_operation_gate": case_float(structure_debug.get("move_operation_gate", float("nan")), float("nan")),
        "operation_gate_grad_norm": case_float((getattr(args, "_last_grad_flow", {}) or {}).get("operation_gate_head_grad_norm", float("nan")), float("nan")),
        "operation_gate_grad_status": str((getattr(args, "_last_grad_flow", {}) or {}).get("operation_gate_head_grad_status", "")),
        "operation_gate_oracle_loss": case_float(structure_debug.get("operation_gate_oracle_loss", float("nan")), float("nan")),
        "actual_oracle_candidate_where_loss": case_float(structure_debug.get("actual_oracle_candidate_where_loss", float("nan")), float("nan")),
        "actual_oracle_bad_candidate_count": case_int(structure_debug.get("actual_oracle_bad_candidate_count", 0)),
        "actual_oracle_improving_candidate_count": case_int(structure_debug.get("actual_oracle_improving_candidate_count", 0)),
        "actual_oracle_combo_extra_count": case_int(structure_debug.get("actual_oracle_combo_extra_count", 0)),
        "actual_oracle_generated_candidate_count": case_int(structure_debug.get("actual_oracle_generated_candidate_count", 0)),
        "actual_oracle_accepted_candidate_count": actual_oracle_accepted_candidate_count,
        "actual_oracle_accepted_generated_ratio": (
            case_float(structure_debug.get("actual_oracle_accepted_candidate_count", 0), 0.0)
            / max(case_float(structure_debug.get("actual_oracle_generated_candidate_count", 0), 0.0), 1.0)
        ),
        "actual_oracle_noop_label_count": case_int(structure_debug.get("actual_oracle_noop_label_count", 0)),
        "actual_oracle_noop_label_weight": case_float(structure_debug.get("actual_oracle_noop_label_weight", 0.0), 0.0),
        "actual_oracle_accepted_prune_count": actual_oracle_accepted_prune_count,
        "actual_oracle_accepted_add_count": case_int(structure_debug.get("actual_oracle_accepted_add_count", 0)),
        "actual_oracle_accepted_adjust_count": case_int(structure_debug.get("actual_oracle_accepted_adjust_count", 0)),
        "actual_oracle_accepted_subtree_move_count": case_int(structure_debug.get("actual_oracle_accepted_subtree_move_count", 0)),
        "actual_oracle_accepted_parent_collapse_count": case_int(structure_debug.get("actual_oracle_accepted_parent_collapse_count", 0)),
        "actual_oracle_accepted_pattern_canonicalize_count": case_int(structure_debug.get("actual_oracle_accepted_pattern_canonicalize_count", 0)),
        "actual_oracle_high_rate_mppov_count": case_int(structure_debug.get("actual_oracle_high_rate_mppov_count", 0)),
        "actual_oracle_low_prob_occupied_count": case_int(structure_debug.get("actual_oracle_low_prob_occupied_count", 0)),
        "actual_oracle_single_child_chain_count": case_int(structure_debug.get("actual_oracle_single_child_chain_count", 0)),
        "actual_oracle_context_pattern_candidate_count": case_int(structure_debug.get("actual_oracle_context_pattern_candidate_count", 0)),
        "actual_oracle_eval_count": actual_oracle_eval_count,
        "actual_oracle_eval_max_configured": case_int(structure_debug.get("actual_oracle_eval_max_configured", 0)),
        "actual_oracle_eval_max": case_int(structure_debug.get("actual_oracle_eval_max", 0)),
        "actual_oracle_eval_scope": str(structure_debug.get("actual_oracle_eval_scope", "")),
        "actual_oracle_eval_full_coord_count": case_int(structure_debug.get("actual_oracle_eval_full_coord_count", 0)),
        "actual_oracle_full_cloud_teacher_required": bool(
            structure_debug.get(
                "actual_oracle_full_cloud_teacher_required",
                getattr(args, "sparsepcgc_require_full_cloud_actual_teacher", True),
            )
        ),
        "actual_oracle_full_cloud_teacher_eval_available": bool(structure_debug.get("actual_oracle_full_cloud_teacher_eval_available", False)),
        "actual_oracle_time": actual_oracle_time,
        "actual_oracle_original_actual_cache_hit": bool(structure_debug.get("actual_oracle_original_actual_cache_hit", False)),
        "actual_oracle_original_actual_encode_time": case_float(structure_debug.get("actual_oracle_original_actual_encode_time", 0.0), 0.0),
        "actual_oracle_candidate_actual_encode_time": case_float(structure_debug.get("actual_oracle_candidate_actual_encode_time", 0.0), 0.0),
        "actual_oracle_released_main_cuda_cache": bool(structure_debug.get("actual_oracle_released_main_cuda_cache", False)),
        "actual_oracle_drop_bad_count": case_int(structure_debug.get("actual_oracle_drop_bad_count", 0)),
        "actual_oracle_add_bad_count": case_int(structure_debug.get("actual_oracle_add_bad_count", 0)),
        "actual_oracle_move_bad_count": case_int(structure_debug.get("actual_oracle_move_bad_count", 0)),
        "actual_oracle_drop_reason": str(structure_debug.get("actual_oracle_drop_reason", "")),
        "actual_oracle_operation": str(structure_debug.get("actual_oracle_operation", "")),
        "actual_oracle_scheduled_operation": str(structure_debug.get("actual_oracle_scheduled_operation", "")),
        "actual_oracle_edit_record_bits": case_float(structure_debug.get("actual_oracle_edit_record_bits", 0.0), 0.0),
        "actual_oracle_best_edit_record_bits": case_float(structure_debug.get("actual_oracle_best_edit_record_bits", float("nan")), float("nan")),
        "actual_oracle_raw_percent": case_float(structure_debug.get("actual_oracle_raw_percent", float("nan")), float("nan")),
        "actual_oracle_best_raw_percent": case_float(structure_debug.get("actual_oracle_best_raw_percent", float("nan")), float("nan")),
        "actual_oracle_delta_actual_percent": actual_oracle_delta_actual,
        "actual_oracle_best_actual_percent": case_float(structure_debug.get("actual_oracle_best_actual_percent", float("nan")), float("nan")),
        "actual_oracle_proxy_percent": actual_oracle_proxy,
        "actual_oracle_best_proxy_percent": case_float(structure_debug.get("actual_oracle_best_proxy_percent", float("nan")), float("nan")),
        "actual_oracle_geometry_percent": case_float(structure_debug.get("actual_oracle_geometry_percent", float("nan")), float("nan")),
        "actual_oracle_original_actual_bits": case_float(structure_debug.get("actual_oracle_original_actual_bits", float("nan")), float("nan")),
        "actual_oracle_edited_actual_bits": case_float(structure_debug.get("actual_oracle_edited_actual_bits", float("nan")), float("nan")),
        "actual_oracle_proxy_actual_gap": (
            actual_oracle_proxy
            - actual_oracle_delta_actual
        ),
        "actual_oracle_fast_diagnostic_used": actual_oracle_fast_diagnostic_used,
        "actual_oracle_fast_diagnostic_full_drop_count": case_int(structure_debug.get("actual_oracle_fast_diagnostic_full_drop_count", 0)),
        "actual_oracle_fast_diagnostic_local_drop_count": actual_oracle_fast_diagnostic_local_count,
        "actual_oracle_fast_diagnostic_full_drop_ratio": case_float(structure_debug.get("actual_oracle_fast_diagnostic_full_drop_ratio", 0.0), 0.0),
        "actual_oracle_fast_diagnostic_local_drop_ratio": case_float(structure_debug.get("actual_oracle_fast_diagnostic_local_drop_ratio", 0.0), 0.0),
        "actual_oracle_fast_diagnostic_full_add_count": case_int(structure_debug.get("actual_oracle_fast_diagnostic_full_add_count", 0)),
        "actual_oracle_fast_diagnostic_local_add_count": actual_oracle_fast_diagnostic_local_add_count,
        "actual_oracle_fast_diagnostic_full_add_ratio": case_float(structure_debug.get("actual_oracle_fast_diagnostic_full_add_ratio", 0.0), 0.0),
        "actual_oracle_fast_diagnostic_local_add_ratio": case_float(structure_debug.get("actual_oracle_fast_diagnostic_local_add_ratio", 0.0), 0.0),
        "actual_oracle_joint_tested_count": case_int(structure_debug.get("actual_oracle_joint_tested_count", 0)),
        "actual_oracle_joint_improving_count": case_int(structure_debug.get("actual_oracle_joint_improving_count", 0)),
        "actual_oracle_group_tested_count": case_int(structure_debug.get("actual_oracle_group_tested_count", 0)),
        "actual_oracle_group_improving_count": case_int(structure_debug.get("actual_oracle_group_improving_count", 0)),
        "actual_oracle_full_cloud_macro_fallback_triggered": bool(structure_debug.get("actual_oracle_full_cloud_macro_fallback_triggered", False)),
        "actual_oracle_full_cloud_macro_fail_extra_eval_max": case_int(structure_debug.get("actual_oracle_full_cloud_macro_fail_extra_eval_max", 0)),
        "actual_oracle_full_cloud_macro_fallback_candidate_generation_enabled": bool(
            structure_debug.get("actual_oracle_full_cloud_macro_fallback_candidate_generation_enabled", False)
        ),
        "actual_oracle_full_cloud_macro_tested_count": case_int(structure_debug.get("actual_oracle_full_cloud_macro_tested_count", 0)),
        "actual_oracle_full_cloud_macro_improving_count": case_int(structure_debug.get("actual_oracle_full_cloud_macro_improving_count", 0)),
        "actual_oracle_full_cloud_macro_best_percent": case_float(structure_debug.get("actual_oracle_full_cloud_macro_best_percent", float("nan")), float("nan")),
        "actual_oracle_full_cloud_macro_best_ratio": case_float(structure_debug.get("actual_oracle_full_cloud_macro_best_ratio", float("nan")), float("nan")),
        "actual_oracle_full_cloud_macro_best_drop_count": case_int(structure_debug.get("actual_oracle_full_cloud_macro_best_drop_count", 0)),
        "actual_oracle_macro_prune_tested_count": case_int(structure_debug.get("actual_oracle_macro_prune_tested_count", 0)),
        "actual_oracle_macro_prune_improving_count": case_int(structure_debug.get("actual_oracle_macro_prune_improving_count", 0)),
        "actual_oracle_macro_prune_best_percent": case_float(structure_debug.get("actual_oracle_macro_prune_best_percent", float("nan")), float("nan")),
        "actual_oracle_macro_prune_best_ratio": case_float(structure_debug.get("actual_oracle_macro_prune_best_ratio", float("nan")), float("nan")),
        "actual_oracle_macro_prune_best_drop_count": case_int(structure_debug.get("actual_oracle_macro_prune_best_drop_count", 0)),
        "actual_oracle_macro_prune_best_variant": str(structure_debug.get("actual_oracle_macro_prune_best_variant", "")),
        "actual_oracle_macro_prune_best_proxy_percent": case_float(structure_debug.get("actual_oracle_macro_prune_best_proxy_percent", float("nan")), float("nan")),
        "actual_oracle_parent_prune_tested_count": case_int(structure_debug.get("actual_oracle_parent_prune_tested_count", 0)),
        "actual_oracle_parent_prune_improving_count": case_int(structure_debug.get("actual_oracle_parent_prune_improving_count", 0)),
        "actual_oracle_pattern_plan_tested_count": case_int(structure_debug.get("actual_oracle_pattern_plan_tested_count", 0)),
        "actual_oracle_pattern_plan_improving_count": case_int(structure_debug.get("actual_oracle_pattern_plan_improving_count", 0)),
        "actual_oracle_subtree_move_tested_count": case_int(structure_debug.get("actual_oracle_subtree_move_tested_count", 0)),
        "actual_oracle_subtree_move_improving_count": case_int(structure_debug.get("actual_oracle_subtree_move_improving_count", 0)),
        "actual_oracle_apply_teacher_actions": bool(
            structure_debug.get("actual_oracle_apply_teacher_actions", False)
        ),
        "actual_oracle_force_no_edit_used": bool(structure_debug.get("actual_oracle_force_no_edit_used", False)),
        "actual_oracle_has_drop": bool(structure_debug.get("actual_oracle_has_drop", False)),
        "prune_after_prior_mode": str(structure_debug.get("prune_after_prior_mode", "")),
        "phase0_network_prune_mode": bool(structure_debug.get("phase0_network_prune_mode", False)),
        "algorithmic_proposal_selector_enabled": bool(
            structure_debug.get("algorithmic_proposal_selector_enabled", False)
        ),
        "algorithmic_proposal_selector_active": bool(
            structure_debug.get("algorithmic_proposal_selector_active", False)
        ),
        "algorithmic_proposal_where_source_id": case_int(
            structure_debug.get("algorithmic_proposal_where_source_id", 0),
            0,
        ),
        "algorithmic_proposal_noop_selected": bool(
            structure_debug.get("algorithmic_proposal_noop_selected", False)
        ),
        "algorithmic_amount_selected_class": case_int(
            structure_debug.get("algorithmic_amount_selected_class", -1),
            -1,
        ),
        "algorithmic_amount_selected_bin_ratio": case_float(
            structure_debug.get("algorithmic_amount_selected_bin_ratio", float("nan")),
            float("nan"),
        ),
        "algorithmic_amount_residual": case_float(
            structure_debug.get("algorithmic_amount_residual", float("nan")),
            float("nan"),
        ),
        "algorithmic_amount_final_ratio": case_float(
            structure_debug.get("algorithmic_amount_final_ratio", float("nan")),
            float("nan"),
        ),
        "algorithmic_amount_noop_prob": case_float(
            structure_debug.get("algorithmic_amount_noop_prob", float("nan")),
            float("nan"),
        ),
        "algorithmic_amount_selected_prob": case_float(
            structure_debug.get("algorithmic_amount_selected_prob", float("nan")),
            float("nan"),
        ),
        "algorithmic_amount_teacher_class": case_int(
            structure_debug.get("algorithmic_amount_teacher_class", -1),
            -1,
        ),
        "algorithmic_amount_teacher_ratio": case_float(
            structure_debug.get("algorithmic_amount_teacher_ratio", float("nan")),
            float("nan"),
        ),
        "algorithmic_amount_selector_teacher_loss": case_float(
            structure_debug.get("algorithmic_amount_selector_teacher_loss", float("nan")),
            float("nan"),
        ),
        "algorithmic_amount_residual_teacher_loss": case_float(
            structure_debug.get("algorithmic_amount_residual_teacher_loss", float("nan")),
            float("nan"),
        ),
        "proposal_selector_enabled": bool(
            comp_debug.get(
                "proposal_selector_enabled",
                structure_debug.get("proposal_selector_enabled", False),
            )
        ),
        "proposal_candidate_count": case_int(comp_debug.get("proposal_candidate_count", 0), 0),
        "proposal_actual_eval_count": case_int(comp_debug.get("proposal_actual_eval_count", 0), 0),
        "proposal_surrogate_prefilter_count": case_int(
            comp_debug.get("proposal_surrogate_prefilter_count", 0),
            0,
        ),
        "proposal_applied_subtree_count": case_int(
            comp_debug.get("proposal_applied_subtree_count", 0),
            0,
        ),
        "proposal_selected_subtree_count": case_int(
            comp_debug.get("proposal_selected_subtree_count", 0),
            0,
        ),
        "proposal_noop_count": case_int(comp_debug.get("proposal_noop_count", 0), 0),
        "proposal_best_actual_percent": case_float(
            comp_debug.get("proposal_best_actual_percent", float("nan")),
            float("nan"),
        ),
        "proposal_chosen_actual_percent": case_float(
            comp_debug.get("proposal_chosen_actual_percent", float("nan")),
            float("nan"),
        ),
        "proposal_predicted_delta": case_float(
            comp_debug.get("proposal_predicted_delta", float("nan")),
            float("nan"),
        ),
        "proposal_amount_bin": case_float(comp_debug.get("proposal_amount_bin", float("nan")), float("nan")),
        "proposal_amount_residual": case_float(
            comp_debug.get("proposal_amount_residual", float("nan")),
            float("nan"),
        ),
        "proposal_final_amount": case_float(
            comp_debug.get("proposal_final_amount", float("nan")),
            float("nan"),
        ),
        "proposal_cls_loss": case_float(comp_debug.get("proposal_cls_loss", float("nan")), float("nan")),
        "proposal_value_loss": case_float(comp_debug.get("proposal_value_loss", float("nan")), float("nan")),
        "proposal_rank_loss": case_float(comp_debug.get("proposal_rank_loss", float("nan")), float("nan")),
        "proposal_geom_loss": case_float(comp_debug.get("proposal_geom_loss", float("nan")), float("nan")),
        "proposal_total_loss": case_float(comp_debug.get("proposal_total_loss", float("nan")), float("nan")),
        "proposal_teacher_source": str(comp_debug.get("proposal_teacher_source", "")),
        "verified_noop_guard_used": bool(comp_debug.get("verified_noop_guard_used", False)),
        "full_cloud_verified_noop_guard_used": bool(
            comp_debug.get("full_cloud_verified_noop_guard_used", False)
        ),
        "sparsepcgc_training_mode": str(
            comp_debug.get(
                "sparsepcgc_training_mode",
                getattr(args, "sparsepcgc_training_mode", "subtree_selector"),
            )
        ),
        "full_cloud_amount_enabled": bool(
            comp_debug.get(
                "full_cloud_amount_enabled",
                structure_debug.get("full_cloud_amount_enabled", False),
            )
        ),
        "full_cloud_amount_fresh_actual_every_step": bool(
            comp_debug.get(
                "full_cloud_amount_fresh_actual_every_step",
                structure_debug.get(
                    "full_cloud_amount_fresh_actual_every_step",
                    getattr(args, "sparsepcgc_full_cloud_amount_fresh_actual_every_step", True),
                ),
            )
        ),
        "full_cloud_amount_actual_step": bool(
            comp_debug.get(
                "full_cloud_amount_actual_step",
                structure_debug.get("full_cloud_amount_actual_step", False),
            )
        ),
        "full_cloud_amount_actual_interval": case_int(
            comp_debug.get(
                "full_cloud_amount_actual_interval",
                structure_debug.get(
                    "full_cloud_amount_actual_interval",
                    getattr(args, "_full_cloud_amount_actual_interval_active", 0),
                ),
            ),
            0,
        ),
        "full_cloud_amount_input_points": case_int(
            comp_debug.get(
                "full_cloud_amount_input_points",
                structure_debug.get("full_cloud_amount_input_points", 0),
            ),
            0,
        ),
        "full_cloud_amount_bin": case_float(
            comp_debug.get("full_cloud_amount_bin", structure_debug.get("full_cloud_amount_bin", float("nan"))),
            float("nan"),
        ),
        "full_cloud_amount_residual": case_float(
            comp_debug.get(
                "full_cloud_amount_residual",
                structure_debug.get("full_cloud_amount_residual", float("nan")),
            ),
            float("nan"),
        ),
        "full_cloud_amount_final_ratio": case_float(
            comp_debug.get(
                "full_cloud_amount_final_ratio",
                structure_debug.get("full_cloud_amount_final_ratio", float("nan")),
            ),
            float("nan"),
        ),
        "full_cloud_amount_drop_count": case_int(comp_debug.get("full_cloud_amount_drop_count", 0), 0),
        "full_cloud_amount_noop_selected": bool(
            comp_debug.get(
                "full_cloud_amount_noop_selected",
                structure_debug.get("full_cloud_amount_noop_selected", False),
            )
        ),
        "full_cloud_amount_candidate_count": case_int(comp_debug.get("full_cloud_amount_candidate_count", 0), 0),
        "full_cloud_amount_actual_eval_count": case_int(comp_debug.get("full_cloud_amount_actual_eval_count", 0), 0),
        "full_cloud_amount_teacher_source": str(comp_debug.get("full_cloud_amount_teacher_source", "")),
        "full_cloud_amount_predicted_delta": case_float(
            comp_debug.get(
                "full_cloud_amount_predicted_delta",
                structure_debug.get("full_cloud_amount_predicted_delta", float("nan")),
            ),
            float("nan"),
        ),
        "full_cloud_amount_actual_delta": case_float(
            comp_debug.get("full_cloud_amount_actual_delta", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_surrogate_delta": case_float(
            comp_debug.get("full_cloud_amount_surrogate_delta", float("nan")),
            float("nan"),
        ),
        "full_cloud_amount_geom_loss": case_float(comp_debug.get("full_cloud_amount_geom_loss", float("nan")), float("nan")),
        "full_cloud_amount_cls_loss": case_float(comp_debug.get("full_cloud_amount_cls_loss", 0.0), 0.0),
        "full_cloud_amount_value_loss": case_float(comp_debug.get("full_cloud_amount_value_loss", 0.0), 0.0),
        "full_cloud_amount_rank_loss": case_float(comp_debug.get("full_cloud_amount_rank_loss", 0.0), 0.0),
        "full_cloud_amount_ratio_reg_loss": case_float(comp_debug.get("full_cloud_amount_ratio_reg_loss", 0.0), 0.0),
        "full_cloud_amount_noop_guard_loss": case_float(
            comp_debug.get("full_cloud_amount_noop_guard_loss", 0.0),
            0.0,
        ),
        "full_cloud_amount_total_loss": case_float(comp_debug.get("full_cloud_amount_total_loss", 0.0), 0.0),
        "full_cloud_amount_step_time": case_float(comp_debug.get("full_cloud_amount_step_time", 0.0), 0.0),
        "actual_gate_prune_enabled": bool(structure_debug.get("actual_gate_prune_enabled", False)),
        "actual_gate_prune_allowed": bool(structure_debug.get("actual_gate_prune_allowed", False)),
        "hard_prune_actual_allowed": bool(structure_debug.get("hard_prune_actual_allowed", structure_debug.get("actual_gate_prune_allowed", False))),
        "hard_drop_block_reason": str(structure_debug.get("hard_drop_block_reason", "")),
        "phase0_network_mode_but_hard_drop_zero": bool(structure_debug.get("phase0_network_mode_but_hard_drop_zero", False)),
        "phase0_noop_only_collapse_detected": bool(structure_debug.get("phase0_noop_only_collapse_detected", False)),
        "collapse_reason": str(structure_debug.get("collapse_reason", "")),
        "hard_drop_count_trace": str(structure_debug.get("hard_drop_count_trace", "")),
        "codec_prune_prior_enabled": bool(structure_debug.get("codec_prune_prior_enabled", False)),
        "codec_prune_prior_phase": case_float(structure_debug.get("codec_prune_prior_phase", 0.0), 0.0),
        "codec_prune_prior_ratio": case_float(structure_debug.get("codec_prune_prior_ratio", 0.0), 0.0),
        "codec_prune_prior_base_ratio": case_float(
            structure_debug.get("codec_prune_prior_base_ratio", structure_debug.get("codec_prune_prior_ratio", 0.0)),
            0.0,
        ),
        "codec_prune_prior_active_ratio": case_float(
            structure_debug.get("codec_prune_prior_active_ratio", structure_debug.get("codec_prune_prior_ratio", 0.0)),
            0.0,
        ),
        "codec_prune_prior_count_alpha": case_float(
            structure_debug.get("codec_prune_prior_count_alpha", 0.0),
            0.0,
        ),
        "codec_prune_prior_block_size": case_int(structure_debug.get("codec_prune_prior_block_size", 0), 0),
        "codec_prune_prior_block_count_mean": case_float(
            structure_debug.get("codec_prune_prior_block_count_mean", 0.0),
            0.0,
        ),
        "prune_ratio_monotonic_floor": case_float(structure_debug.get("prune_ratio_monotonic_floor", float("nan")), float("nan")),
        "network_prune_ratio_floor": case_float(structure_debug.get("network_prune_ratio_floor", 0.0), 0.0),
        "network_prune_min_hard_count": case_int(structure_debug.get("network_prune_min_hard_count", 0), 0),
        "network_prune_floor_steps": case_int(structure_debug.get("network_prune_floor_steps", 0), 0),
        "network_prune_floor_decay_steps": case_int(structure_debug.get("network_prune_floor_decay_steps", 0), 0),
        "drop_score_gate_applied_to_hard_selection": bool(structure_debug.get("drop_score_gate_applied_to_hard_selection", True)),
        "parent_collapse_grad_norm": case_float((getattr(args, "_last_grad_flow", {}) or {}).get("parent_collapse_grad_norm", float("nan")), float("nan")),
        "parent_collapse_grad_status": str((getattr(args, "_last_grad_flow", {}) or {}).get("parent_collapse_grad_status", "not_implemented_or_no_grad")),
        "pattern_canonicalize_grad_norm": case_float((getattr(args, "_last_grad_flow", {}) or {}).get("pattern_canonicalize_grad_norm", float("nan")), float("nan")),
        "pattern_canonicalize_grad_status": str((getattr(args, "_last_grad_flow", {}) or {}).get("pattern_canonicalize_grad_status", "not_implemented_or_no_grad")),
        "raw_learned_drop_ratio": case_float(structure_debug.get("raw_learned_drop_ratio", float("nan")), float("nan")),
        "raw_learned_add_ratio": case_float(structure_debug.get("raw_learned_add_ratio", float("nan")), float("nan")),
        "learned_drop_ratio_before_floor": case_float(structure_debug.get("learned_drop_ratio_before_floor", float("nan")), float("nan")),
        "learned_drop_ratio_after_floor": case_float(structure_debug.get("learned_drop_ratio_after_floor", float("nan")), float("nan")),
        "learned_drop_ratio_before_gate": case_float(structure_debug.get("learned_drop_ratio_before_gate", float("nan")), float("nan")),
        "learned_drop_ratio_after_gate": case_float(structure_debug.get("learned_drop_ratio_after_gate", float("nan")), float("nan")),
        "learned_drop_ratio_value": case_float(structure_debug.get("learned_drop_ratio_value", float("nan")), float("nan")),
        "effective_drop_ratio_for_hard_count": case_float(structure_debug.get("effective_drop_ratio_for_hard_count", float("nan")), float("nan")),
        "hard_drop_target_ratio_source_id": case_int(
            structure_debug.get("hard_drop_target_ratio_source_id", 0),
            0,
        ),
        "hard_drop_target_ratio_value": case_float(
            structure_debug.get("hard_drop_target_ratio_value", float("nan")),
            float("nan"),
        ),
        "hard_drop_target_ratio_network_value": case_float(
            structure_debug.get("hard_drop_target_ratio_network_value", float("nan")),
            float("nan"),
        ),
        "hard_drop_target_ratio_codec_prior_value": case_float(
            structure_debug.get("hard_drop_target_ratio_codec_prior_value", float("nan")),
            float("nan"),
        ),
        "post_warmup_amount_hybrid_applied": bool(
            structure_debug.get("post_warmup_amount_hybrid_applied", False)
        ),
        "post_warmup_amount_mode_id": case_int(
            structure_debug.get("post_warmup_amount_mode_id", 0),
            0,
        ),
        "post_warmup_amount_strategy_id": case_int(
            structure_debug.get("post_warmup_amount_strategy_id", 0),
            0,
        ),
        "post_warmup_amount_tail_phase": case_float(
            structure_debug.get("post_warmup_amount_tail_phase", float("nan")),
            float("nan"),
        ),
        "post_warmup_amount_alpha": case_float(
            structure_debug.get("post_warmup_amount_alpha", float("nan")),
            float("nan"),
        ),
        "post_warmup_amount_proposal_ratio": case_float(
            structure_debug.get("post_warmup_amount_proposal_ratio", float("nan")),
            float("nan"),
        ),
        "post_warmup_amount_teacher_loss": case_float(
            structure_debug.get("post_warmup_amount_teacher_loss", float("nan")),
            float("nan"),
        ),
        "post_warmup_amount_teacher_weight_effective": case_float(
            structure_debug.get("post_warmup_amount_teacher_weight_effective", float("nan")),
            float("nan"),
        ),
        "amount_explore_step": bool(structure_debug.get("amount_explore_step", False)),
        "amount_explore_prob": case_float(
            structure_debug.get("amount_explore_prob", float("nan")),
            float("nan"),
        ),
        "amount_explore_candidate_ratio": case_float(
            structure_debug.get("amount_explore_candidate_ratio", float("nan")),
            float("nan"),
        ),
        "amount_explore_candidate_index": case_int(
            structure_debug.get("amount_explore_candidate_index", -1),
            -1,
        ),
        "amount_explore_teacher_ratio": case_float(
            structure_debug.get("amount_explore_teacher_ratio", float("nan")),
            float("nan"),
        ),
        "amount_explore_teacher_count": case_int(
            structure_debug.get("amount_explore_teacher_count", 0),
            0,
        ),
        "amount_explore_teacher_score": case_float(
            structure_debug.get("amount_explore_teacher_score", float("nan")),
            float("nan"),
        ),
        "amount_explore_teacher_alpha": case_float(
            structure_debug.get("amount_explore_teacher_alpha", float("nan")),
            float("nan"),
        ),
        "amount_explore_used_teacher": bool(
            structure_debug.get("amount_explore_used_teacher", False)
        ),
        "amount_mode_id": case_int(structure_debug.get("amount_mode_id", 0), 0),
        "amount_mode_network": bool(structure_debug.get("amount_mode_network", False)),
        "voxel_count": case_int(structure_debug.get("voxel_count", 0), 0),
        "codec_block_valid_point_count": case_int(
            structure_debug.get("codec_block_valid_point_count", 0),
            0,
        ),
        "codec_block_budget_points": case_float(
            structure_debug.get("codec_block_budget_points", float("nan")),
            float("nan"),
        ),
        "codec_block_count": case_int(structure_debug.get("codec_block_count", 0), 0),
        "codec_block_selected_block_count": case_int(
            structure_debug.get("codec_block_selected_block_count", 0),
            0,
        ),
        "codec_block_selected_point_count": case_int(
            structure_debug.get("codec_block_selected_point_count", 0),
            0,
        ),
        "codec_block_budget_zero": bool(structure_debug.get("codec_block_budget_zero", False)),
        "codec_block_target_drop_ratio": case_float(
            structure_debug.get("codec_block_target_drop_ratio", float("nan")),
            float("nan"),
        ),
        "codec_block_under_selected": bool(structure_debug.get("codec_block_under_selected", False)),
        "delete_candidate_count": case_int(structure_debug.get("delete_candidate_count", 0), 0),
        "delete_candidate_point_count": case_int(structure_debug.get("delete_candidate_point_count", 0), 0),
        "delete_candidate_empty_reason": str(structure_debug.get("delete_candidate_empty_reason", "")),
        "hard_delete_selection_count": case_int(structure_debug.get("hard_delete_selection_count", 0), 0),
        "pre_round_target_count": case_float(structure_debug.get("pre_round_target_count", float("nan")), float("nan")),
        "post_round_target_count": case_float(structure_debug.get("post_round_target_count", float("nan")), float("nan")),
        "min_hard_drop_count_floor_applied": bool(structure_debug.get("min_hard_drop_count_floor_applied", False)),
        "hard_mask_count": case_int(structure_debug.get("hard_mask_count", 0), 0),
        "final_hard_drop_count": case_int(structure_debug.get("final_hard_drop_count", 0), 0),
        "selected_drop_count_hard": case_int(
            structure_debug.get(
                "selected_drop_count_hard",
                structure_debug.get("hard_drop_count", structure_debug.get("final_hard_drop_count", 0)),
            ),
            0,
        ),
        "drop_ratio_hard": case_float(
            structure_debug.get("drop_ratio_hard", structure_debug.get("hard_drop_ratio", float("nan"))),
            float("nan"),
        ),
        "raw_learned_move_ratio": case_float(structure_debug.get("raw_learned_move_ratio", float("nan")), float("nan")),
        "repair_ratio": case_float(structure_debug.get("repair_ratio", float("nan")), float("nan")),
        "preserve_ratio": case_float(structure_debug.get("preserve_ratio", float("nan")), float("nan")),
        "add_prob_mean": case_float(structure_debug.get("add_prob_mean", float("nan")), float("nan")),
        "add_prob_max": case_float(structure_debug.get("add_prob_max", float("nan")), float("nan")),
        "add_priority_mean": case_float(structure_debug.get("add_priority_mean", float("nan")), float("nan")),
        "add_priority_max": case_float(structure_debug.get("add_priority_max", float("nan")), float("nan")),
        "add_score_mean": case_float(structure_debug.get("add_priority_mean", float("nan")), float("nan")),
        "add_score_max": case_float(structure_debug.get("add_priority_max", float("nan")), float("nan")),
        "add_ratio": case_float(structure_debug.get("add_ratio", float("nan")), float("nan")),
        "add_soft_ratio": case_float(structure_debug.get("add_ratio", float("nan")), float("nan")),
        "add_hard_ratio": case_float(edit_stats.get("added_ratio_percent", float("nan")), float("nan")) / 100.0 if edit_stats else None,
        "add_candidate_ratio": add_candidate_ratio,
        "add_candidate_count": add_candidate_count,
        "add_hard_count": case_int(structure_debug.get("add_count", 0)),
        "add_effective_count": add_effective_count,
        "add_amount_mean": case_float(structure_debug.get("add_amount_mean", structure_debug.get("learned_add_ratio", structure_debug.get("add_ratio", float("nan")))), float("nan")),
        "add_amount_std": case_float(structure_debug.get("add_amount_std", structure_debug.get("learned_add_ratio_std", float("nan"))), float("nan")),
        "add_branch_grad_norm": case_float((getattr(args, "_last_grad_flow", {}) or {}).get("add_branch_grad_norm", float("nan")), float("nan")),
        "add_amount_grad_norm": case_float((getattr(args, "_last_grad_flow", {}) or {}).get("add_amount_grad_norm", float("nan")), float("nan")),
        "add_branch_grad_status": str((getattr(args, "_last_grad_flow", {}) or {}).get("add_branch_grad_status", "")),
        "add_amount_grad_status": str((getattr(args, "_last_grad_flow", {}) or {}).get("add_amount_grad_status", "")),
        "actual_oracle_drop_amount_loss": case_float(structure_debug.get("actual_oracle_drop_amount_loss", float("nan")), float("nan")),
        "actual_oracle_add_amount_loss": case_float(structure_debug.get("actual_oracle_add_amount_loss", float("nan")), float("nan")),
        "actual_oracle_move_amount_loss": case_float(structure_debug.get("actual_oracle_move_amount_loss", float("nan")), float("nan")),
        "actual_oracle_drop_amount_logit_loss": case_float(structure_debug.get("actual_oracle_drop_amount_logit_loss", float("nan")), float("nan")),
        "actual_oracle_add_amount_logit_loss": case_float(structure_debug.get("actual_oracle_add_amount_logit_loss", float("nan")), float("nan")),
        "actual_oracle_amount_supervision_loss": case_float(structure_debug.get("actual_oracle_amount_supervision_loss", float("nan")), float("nan")),
        "add_actual_point_count": add_effective_count,
        "add_target_voxels": case_int(structure_debug.get("add_target_voxel_count", 0)),
        "add_target_ratio": case_float(getattr(args, "target_add_ratio", 0.0), float("nan")),
        "add_max_ratio": case_float(getattr(args, "max_add_ratio", 0.0), float("nan")),
        "add_warmup": add_warmup_factor(args),
        "soft_add_count": case_float(structure_debug.get("add_ratio", 0.0), 0.0) * input_points if math.isfinite(input_points) else None,
        "hard_add_count": case_int(structure_debug.get("add_count", 0)),
        "drop_prob_mean": case_float(structure_debug.get("drop_ratio", float("nan")), float("nan")),
        "prune_soft_ratio": case_float(structure_debug.get("drop_ratio", float("nan")), float("nan")),
        "prune_hard_ratio": case_float(structure_debug.get("hard_drop_ratio", float("nan")), float("nan")),
        "prune_effective_count": case_int(structure_debug.get("hard_drop_count", structure_debug.get("delete_removed_point_count", 0))),
        "prune_amount_mean": case_float(structure_debug.get("prune_amount_mean", structure_debug.get("learned_drop_ratio", structure_debug.get("drop_ratio", float("nan")))), float("nan")),
        "prune_amount_std": case_float(structure_debug.get("prune_amount_std", structure_debug.get("learned_drop_ratio_std", float("nan"))), float("nan")),
        "prune_branch_grad_norm": case_float((getattr(args, "_last_grad_flow", {}) or {}).get("delete_branch_grad_norm", float("nan")), float("nan")),
        "prune_amount_grad_norm": case_float((getattr(args, "_last_grad_flow", {}) or {}).get("delete_amount_grad_norm", float("nan")), float("nan")),
        "prune_branch_grad_status": str((getattr(args, "_last_grad_flow", {}) or {}).get("delete_branch_grad_status", "")),
        "prune_amount_grad_status": str((getattr(args, "_last_grad_flow", {}) or {}).get("delete_amount_grad_status", "")),
        "hard_drop_ratio": case_float(structure_debug.get("hard_drop_ratio", float("nan")), float("nan")),
        "hard_drop_count": case_int(structure_debug.get("hard_drop_count", structure_debug.get("delete_removed_point_count", 0))),
        "delete_target_voxels": case_int(structure_debug.get("delete_target_voxel_count", 0)),
        "delete_emptied_voxels": case_int(structure_debug.get("delete_emptied_voxel_count", 0)),
        "move_score_mean": case_float(structure_debug.get("move_score_mean", float("nan")), float("nan")),
        "adjust_soft_ratio": case_float(structure_debug.get("move_ratio", structure_debug.get("move_score_mean", float("nan"))), float("nan")),
        "adjust_hard_ratio": case_float(structure_debug.get("hard_move_ratio", float("nan")), float("nan")),
        "adjust_effective_count": case_int(structure_debug.get("hard_move_count", 0)),
        "adjust_amount_mean": case_float(structure_debug.get("adjust_amount_mean", structure_debug.get("learned_move_ratio", structure_debug.get("move_ratio", float("nan")))), float("nan")),
        "adjust_amount_std": case_float(structure_debug.get("adjust_amount_std", structure_debug.get("learned_move_ratio_std", float("nan"))), float("nan")),
        "adjust_branch_grad_norm": case_float((getattr(args, "_last_grad_flow", {}) or {}).get("move_branch_grad_norm", float("nan")), float("nan")),
        "adjust_amount_grad_norm": case_float((getattr(args, "_last_grad_flow", {}) or {}).get("move_amount_grad_norm", float("nan")), float("nan")),
        "adjust_branch_grad_status": str((getattr(args, "_last_grad_flow", {}) or {}).get("move_branch_grad_status", "")),
        "adjust_amount_grad_status": str((getattr(args, "_last_grad_flow", {}) or {}).get("move_amount_grad_status", "")),
        "move_source_prior_mean": case_float(structure_debug.get("move_source_prior_mean", float("nan")), float("nan")),
        "move_ratio": case_float(structure_debug.get("move_ratio", float("nan")), float("nan")),
        "hard_move_ratio": case_float(structure_debug.get("move_ratio", float("nan")), float("nan")),
        "hard_move_count": case_int(structure_debug.get("hard_move_count", 0)),
        "adjusted_point_count": case_int(structure_debug.get("adjusted_point_count", structure_debug.get("hard_move_count", 0))),
        "adjusted_point_rate": case_float(structure_debug.get("adjusted_point_rate", structure_debug.get("move_ratio", float("nan"))), float("nan")),
        "move_source_voxels": case_int(structure_debug.get("move_source_voxel_count", 0)),
        "move_target_voxels": case_int(structure_debug.get("move_target_voxel_count", 0)),
        "source_unique_voxel_count": case_int(structure_debug.get("source_unique_voxel_count", 0)),
        "target_unique_voxel_count": case_int(structure_debug.get("target_unique_voxel_count", 0)),
        "target_duplicate_voxel_count": case_int(structure_debug.get("target_duplicate_voxel_count", 0)),
        "target_voxel_duplicate_rate": case_float(structure_debug.get("target_voxel_duplicate_rate", float("nan")), float("nan")),
        "target_existing_occupied_count": case_int(structure_debug.get("target_existing_occupied_count", 0)),
        "target_existing_occupied_rate": case_float(structure_debug.get("target_existing_occupied_rate", float("nan")), float("nan")),
        "target_empty_voxel_count": case_int(structure_debug.get("target_empty_voxel_count", 0)),
        "target_empty_voxel_rate": case_float(structure_debug.get("target_empty_voxel_rate", float("nan")), float("nan")),
        "raw_hard_move_count_before_sparsepcgc_guard": case_int(structure_debug.get("raw_hard_move_count_before_sparsepcgc_guard", 0)),
        "empty_target_violation_loss": case_float(structure_debug.get("empty_target_violation_loss", float("nan")), float("nan")),
        "target_duplicate_voxel_loss": case_float(structure_debug.get("target_duplicate_voxel_loss", float("nan")), float("nan")),
        "enable_sparsepcgc_empty_target_guard": bool(structure_debug.get("enable_sparsepcgc_empty_target_guard", False)),
        "enable_sparsepcgc_target_duplicate_guard": bool(structure_debug.get("enable_sparsepcgc_target_duplicate_guard", False)),
        "sparsepcgc_empty_target_guard_rejected_count": case_int(structure_debug.get("sparsepcgc_empty_target_guard_rejected_count", 0)),
        "sparsepcgc_target_duplicate_guard_rejected_count": case_int(structure_debug.get("sparsepcgc_target_duplicate_guard_rejected_count", 0)),
        "sparsepcgc_guard_rejected_count": case_int(structure_debug.get("sparsepcgc_guard_rejected_count", 0)),
        "sparsepcgc_move_existing_target_only": bool(structure_debug.get("sparsepcgc_move_existing_target_only", False)),
        "repair_move_require_empty_target": bool(structure_debug.get("repair_move_require_empty_target", True)),
        "repair_move_require_empty_target_effective": bool(structure_debug.get("repair_move_require_empty_target_effective", True)),
        "repair_move_max_points_per_voxel": case_int(structure_debug.get("repair_move_max_points_per_voxel", 0)),
        "repair_move_warmup": case_float(structure_debug.get("repair_move_warmup", float("nan")), float("nan")),
        "target_move_ratio": case_float(structure_debug.get("target_move_ratio", float("nan")), float("nan")),
        "max_move_ratio": case_float(structure_debug.get("max_move_ratio", float("nan")), float("nan")),
        "repair_move_hard_threshold": case_float(structure_debug.get("repair_move_hard_threshold", float("nan")), float("nan")),
        "move_source_emptied": case_int(structure_debug.get("move_source_emptied_voxel_count", 0)),
        "move_target_new": case_int(structure_debug.get("move_target_new_voxel_count", 0)),
        "move_source_not_emptied": case_int(structure_debug.get("move_source_not_emptied_count", 0)),
        "same_voxel_adjust": case_int(structure_debug.get("same_voxel_adjust_count", 0)),
        "different_voxel_move": case_int(structure_debug.get("moved_different_voxel_count", 0)),
        "input_points": input_points,
        "pre_output_points": case_float(edit_stats.get("pre_output_points_avg", edit_stats.get("pre_output_points", float("nan"))), float("nan")),
        "output_points": case_float(edit_stats.get("output_points_avg", edit_stats.get("output_points", float("nan"))), float("nan")),
        "added_ratio_percent": case_float(edit_stats.get("added_ratio_percent", float("nan")), float("nan")),
        "deleted_ratio_percent": case_float(edit_stats.get("deleted_ratio_percent", float("nan")), float("nan")),
        "local_policy_deleted_ratio_percent": case_float(edit_stats.get("deleted_ratio_percent", float("nan")), float("nan")),
        "oracle_full_cloud_prune_ratio_percent": oracle_full_cloud_prune_ratio_percent,
        "oracle_full_cloud_prune_count": oracle_full_cloud_prune_count,
        "oracle_full_cloud_override_used": bool(oracle_full_cloud_override_used),
        "full_cloud_voxel_drop_ratio_percent": full_cloud_voxel_drop_ratio_percent,
        "full_cloud_voxel_count": case_int(full_cloud_voxel_count, 0),
        "adjusted_ratio_percent": case_float(edit_stats.get("adjusted_ratio_percent", float("nan")), float("nan")),
        "adjusted_ratio_percent_point_debug": case_float(edit_stats.get("adjusted_ratio_percent_point_debug", edit_stats.get("adjusted_ratio_percent", float("nan"))), float("nan")),
        "voxel_edit_input_count": case_int(voxel_edit_input_count, 0),
        "voxel_edit_add_count": case_int(voxel_edit_add_count, 0),
        "voxel_edit_drop_count": case_int(voxel_edit_drop_count, 0),
        "voxel_edit_move_count": case_int(voxel_edit_move_count, 0),
        "voxel_edit_final_count": case_int(voxel_edit_final_count, 0),
        "voxel_add_ratio_percent": voxel_add_ratio_percent,
        "voxel_drop_ratio_percent": voxel_drop_ratio_percent,
        "voxel_move_ratio_percent": voxel_move_ratio_percent,
        "codec_points_after": case_int(comp_debug.get("gen_points", 0)),
        "codec_points_before": case_int(comp_debug.get("gt_points", 0)),
        "codec_unique_after": unique_after,
        "codec_unique_before": unique_before,
        "unique_coord_delta": unique_delta,
        "add_after_quant_unique_count": positive_unique_delta,
        "add_removed_by_unique_count": add_removed_by_unique,
        "active_coord_before": active_before,
        "active_coord_after": active_after,
        "active_coord_delta": active_after - active_before,
        "isolated_voxel_count": case_int(comp_debug.get("sparsepcgc_after_isolated_voxels", 0)),
        "isolated_voxel_delta": case_int(comp_debug.get("sparsepcgc_isolated_delta", 0)),
        "sparse_density_before": case_float(comp_debug.get("sparsepcgc_before_sparse_density", float("nan")), float("nan")),
        "sparse_density_after": case_float(comp_debug.get("sparsepcgc_after_sparse_density", float("nan")), float("nan")),
        "sparse_density_delta": case_float(comp_debug.get("sparsepcgc_sparse_density_delta", float("nan")), float("nan")),
        "occupancy_entropy": case_float(structure_debug.get("occupancy_entropy", comp_debug.get("occupancy_entropy", float("nan"))), float("nan")),
        "occupancy_nll_proxy": case_float(structure_debug.get("occupancy_nll_proxy", comp_debug.get("occupancy_nll_proxy", float("nan"))), float("nan")),
        "lowprob_occupancy_ratio": case_float(structure_debug.get("lowprob_occupancy_ratio", comp_debug.get("lowprob_occupancy_ratio", float("nan"))), float("nan")),
        "entropy_delta": case_float(comp_debug.get("sparsepcgc_entropy_proxy_loss", float("nan")), float("nan")),
        "nll_delta": case_float(comp_debug.get("nll_delta", structure_debug.get("occupancy_nll_proxy", float("nan"))), float("nan")),
        "occupancy_pattern_delta": case_float(comp_debug.get("occupancy_pattern_delta", float("nan")), float("nan")),
        "occupancy_entropy_delta": case_float(comp_debug.get("occupancy_entropy_delta", float("nan")), float("nan")),
        "occupancy_nll_delta": case_float(comp_debug.get("occupancy_nll_delta", comp_debug.get("nll_delta", float("nan"))), float("nan")),
        "corr_occupancy_actual": case_float(comp_debug.get("corr_occupancy_actual", float("nan")), float("nan")),
        "sign_match_occupancy_actual": case_float(comp_debug.get("sign_match_occupancy_actual", float("nan")), float("nan")),
        "depth_node_count_summary": summarize_octree_level_debug(level_debug, "occupied_mean"),
        "depth_single_child_count_summary": summarize_octree_level_debug(level_debug, "single_mean"),
        "depth_entropy_summary": summarize_octree_level_debug(level_debug, "std_children_mean"),
        "depth_lowprob_summary": summarize_octree_level_debug(level_debug, "single_ratio_mean"),
        "subtree_depth": case_int(structure_debug.get("subtree_depth", getattr(args, "_current_subtree_depth", 0))),
        "subtree_node_count": case_float(structure_debug.get("subtree_node_count", float("nan")), float("nan")),
        "subtree_single_child_count": case_float(structure_debug.get("subtree_single_child_count", float("nan")), float("nan")),
        "single_child_delta": case_float(comp_debug.get("single_delta", float("nan")), float("nan")),
        "cp_L_com_main": case_float(comp_debug.get("cp_L_com_main", float("nan")), float("nan")),
        "cp_total": case_float(comp_debug.get("cp_total", float("nan")), float("nan")),
        "temperature": case_float(structure_debug.get("temperature", structure_debug.get("repair_temperature", float("nan"))), float("nan")),
        "exploration_noise": case_float(structure_debug.get("exploration_noise", structure_debug.get("move_score_noise", float("nan"))), float("nan")),
        "operation_regularization": case_float(structure_debug.get("operation_regularization", structure_debug.get("operation_amount_consistency_loss", float("nan"))), float("nan")),
        "operation_entropy": case_float(structure_debug.get("operation_entropy", float("nan")), float("nan")),
        "operation_entropy_loss": case_float(structure_debug.get("operation_entropy_loss", float("nan")), float("nan")),
        "operation_entropy_weight_effective": case_float(structure_debug.get("operation_entropy_weight_effective", float("nan")), float("nan")),
        "operation_prob_floor_applied": bool(structure_debug.get("operation_prob_floor_applied", False)),
        "exploration_alive": bool(case_float(structure_debug.get("exploration_noise", 0.0), 0.0) > 0.0 or case_float(structure_debug.get("operation_entropy", 0.0), 0.0) > 0.0),
        "operation_entropy_moving_avg": case_float(comp_debug.get("operation_entropy_moving_avg", float("nan")), float("nan")),
    }


def attach_grad_flow_to_operation_row(row, args):
    grad = getattr(args, "_last_grad_flow", {}) or {}
    row["add_branch_grad_norm"] = case_float(grad.get("add_branch_grad_norm", row.get("add_branch_grad_norm", float("nan"))), float("nan"))
    row["add_amount_grad_norm"] = case_float(grad.get("add_amount_grad_norm", row.get("add_amount_grad_norm", float("nan"))), float("nan"))
    row["add_branch_grad_status"] = str(grad.get("add_branch_grad_status", row.get("add_branch_grad_status", "")))
    row["add_amount_grad_status"] = str(grad.get("add_amount_grad_status", row.get("add_amount_grad_status", "")))
    row["prune_branch_grad_norm"] = case_float(grad.get("delete_branch_grad_norm", row.get("prune_branch_grad_norm", float("nan"))), float("nan"))
    row["prune_amount_grad_norm"] = case_float(grad.get("delete_amount_grad_norm", row.get("prune_amount_grad_norm", float("nan"))), float("nan"))
    row["prune_branch_grad_status"] = str(grad.get("delete_branch_grad_status", row.get("prune_branch_grad_status", "")))
    row["prune_amount_grad_status"] = str(grad.get("delete_amount_grad_status", row.get("prune_amount_grad_status", "")))
    row["adjust_branch_grad_norm"] = case_float(grad.get("move_branch_grad_norm", row.get("adjust_branch_grad_norm", float("nan"))), float("nan"))
    row["adjust_amount_grad_norm"] = case_float(grad.get("move_amount_grad_norm", row.get("adjust_amount_grad_norm", float("nan"))), float("nan"))
    row["adjust_branch_grad_status"] = str(grad.get("move_branch_grad_status", row.get("adjust_branch_grad_status", "")))
    row["adjust_amount_grad_status"] = str(grad.get("move_amount_grad_status", row.get("adjust_amount_grad_status", "")))
    row["operation_gate_grad_norm"] = case_float(grad.get("operation_gate_head_grad_norm", row.get("operation_gate_grad_norm", float("nan"))), float("nan"))
    row["operation_gate_grad_status"] = str(grad.get("operation_gate_head_grad_status", row.get("operation_gate_grad_status", "")))
    row["parent_collapse_grad_norm"] = case_float(
        grad.get("parent_collapse_grad_norm", row.get("parent_collapse_grad_norm", float("nan"))),
        float("nan"),
    )
    row["parent_collapse_grad_status"] = str(
        grad.get("parent_collapse_grad_status", row.get("parent_collapse_grad_status", "not_implemented_or_no_grad"))
    )
    row["pattern_canonicalize_grad_norm"] = case_float(
        grad.get("pattern_canonicalize_grad_norm", row.get("pattern_canonicalize_grad_norm", float("nan"))),
        float("nan"),
    )
    row["pattern_canonicalize_grad_status"] = str(
        grad.get("pattern_canonicalize_grad_status", row.get("pattern_canonicalize_grad_status", "not_implemented_or_no_grad"))
    )
    return row
