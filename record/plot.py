"""
loss: 全損失
geom: 幾何損失
compression: Surrogateが予測したtotal-bit差百分率
actual_compression: L_comに使った実codec objective差百分率
policy_actual_compression: Network自身の最終出力actual差百分率
oracle_teacher_compression: Oracle teacher候補のactual差百分率
actual_compression_ratio: L_comに使った実codec objectiveの100*Mine/GT百分率
attr: 原因分解損失
policy: 修復ポリシー損失
single: single-child 指標の変化率
nodes: node 数指標の変化率
single_attr: single-child 原因スコア平均
lowprob_attr: low-probability occupancy 原因スコア平均
node_attr: node-count 原因スコア平均
repair: 修復アクチュエータ損失
surrogate_train: surrogate 教師学習損失
surrogate_bit_error: surrogate の bit 予測誤差
surrogate_mean_error: surrogate の平均予測誤差
"""

import os
import math
import statistics

plt = None


def _get_pyplot():
    global plt
    if plt is not None:
        return plt
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except ModuleNotFoundError:
        return None
    plt = _plt
    return plt


class PlotMaker():
    def __init__(self, args):
        self.args = args
        log_root = getattr(args, "log_root", None)
        if log_root is None:
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../")
            )
            log_root = os.path.join(base_dir, "log")
        else:
            log_root = os.path.abspath(os.path.expanduser(str(log_root)))
        self.save_dir = os.path.join(log_root, self.args.date, f"MyNetwork_train/{args.compress}")
        self.num_loss = 18 # train objective / policy / oracle actual系列を分離したmetric総数
        self.x_len = 10
        self.y_len = 4
        self.step_loss_his = [[] for _ in range(self.num_loss)]
        self.epo_loss_his = [[] for _ in range(self.num_loss)]
        self.epi_loss_his = [[] for _ in range(self.num_loss)]
        self.step_x_his = []
        self.epo_x_his = []
        self.epi_x_his = []
        self.edit_keys = [
            "added_ratio_percent",
            "deleted_ratio_percent",
            "full_cloud_voxel_drop_ratio_percent",
            "adjusted_ratio_percent",
            "oracle_full_cloud_prune_ratio_percent",
            "full_cloud_amount_final_ratio",
            "full_cloud_amount_drop_count",
            "full_cloud_amount_actual_delta",
            "full_cloud_amount_surrogate_delta",
            "full_cloud_amount_geom_loss",
            "full_cloud_amount_total_loss",
            "full_cloud_amount_noop_selected",
            "full_cloud_amount_actual_eval_count",
        ]
        self.step_edit_his = [[] for _ in self.edit_keys]
        self.epo_edit_his = [[] for _ in self.edit_keys]
        self.epi_edit_his = [[] for _ in self.edit_keys]
        self.step_edit_x_his = []
        self.epo_edit_x_his = []
        self.epi_edit_x_his = []
        self._reset_edit_running("epo")
        self._reset_edit_running("epi")
        self.occupancy_keys = [
            "predicted_occupancy_entropy_delta",
            "actual_occupancy_entropy_delta",
            "predicted_occupancy_nll_delta",
            "actual_occupancy_nll_delta",
            "predicted_occupancy_pattern_delta",
            "actual_occupancy_pattern_delta",
            "predicted_lowprob_occupancy_ratio",
            "actual_lowprob_occupancy_ratio_after",
            "actual_occupancy_predictability_after",
        ]
        self.occupancy_titles = {
            "predicted_occupancy_entropy_delta": "Predicted Occupancy Entropy Delta",
            "actual_occupancy_entropy_delta": "Actual Occupancy Entropy Delta",
            "predicted_occupancy_nll_delta": "Predicted Occupancy NLL Delta",
            "actual_occupancy_nll_delta": "Actual Occupancy NLL Delta",
            "predicted_occupancy_pattern_delta": "Predicted Occupancy Pattern Delta",
            "actual_occupancy_pattern_delta": "Actual Occupancy Pattern Delta",
            "predicted_lowprob_occupancy_ratio": "Predicted Low-Probability Occupancy Ratio",
            "actual_lowprob_occupancy_ratio_after": "Actual Low-Probability Occupancy Ratio",
            "actual_occupancy_predictability_after": "Actual Occupancy Predictability",
        }
        self.occupancy_plot_groups = [
            (
                "Occupancy Entropy Delta",
                [
                    ("Predicted", "predicted_occupancy_entropy_delta", "tab:blue"),
                    ("Actual", "actual_occupancy_entropy_delta", "tab:orange"),
                ],
            ),
            (
                "Occupancy NLL Delta",
                [
                    ("Predicted", "predicted_occupancy_nll_delta", "tab:blue"),
                    ("Actual", "actual_occupancy_nll_delta", "tab:orange"),
                ],
            ),
            (
                "Occupancy Pattern Delta",
                [
                    ("Predicted", "predicted_occupancy_pattern_delta", "tab:blue"),
                    ("Actual", "actual_occupancy_pattern_delta", "tab:orange"),
                ],
            ),
            (
                "Low-Probability Occupancy Ratio",
                [
                    ("Predicted", "predicted_lowprob_occupancy_ratio", "tab:blue"),
                    ("Actual", "actual_lowprob_occupancy_ratio_after", "tab:orange"),
                ],
            ),
            (
                "Actual Occupancy Predictability",
                [
                    ("Actual", "actual_occupancy_predictability_after", "tab:green"),
                ],
            ),
        ]
        self.step_occupancy_his = [[] for _ in self.occupancy_keys]
        self.epo_occupancy_his = [[] for _ in self.occupancy_keys]
        self.epi_occupancy_his = [[] for _ in self.occupancy_keys]
        self.step_occupancy_x_his = []
        self.epo_occupancy_x_his = []
        self.epi_occupancy_x_his = []
        self._reset_occupancy_running("epo")
        self._reset_occupancy_running("epi")
        self.voxel_collision_keys = [
            "voxel_collision_input_gt_point_reduction_rate",
            "voxel_collision_model_output_raw_point_reduction_rate",
            "voxel_collision_compression_input_point_reduction_rate",
        ]
        self.voxel_collision_titles = {
            "voxel_collision_input_gt_point_reduction_rate": "GT",
            "voxel_collision_model_output_raw_point_reduction_rate": "Mine",
            "voxel_collision_compression_input_point_reduction_rate": "Mine Codec Input",
        }
        self.voxel_collision_colors = {
            "voxel_collision_input_gt_point_reduction_rate": "tab:orange",
            "voxel_collision_model_output_raw_point_reduction_rate": "tab:blue",
            "voxel_collision_compression_input_point_reduction_rate": "tab:green",
        }
        self.step_voxel_collision_his = [[] for _ in self.voxel_collision_keys]
        self.epo_voxel_collision_his = [[] for _ in self.voxel_collision_keys]
        self.epi_voxel_collision_his = [[] for _ in self.voxel_collision_keys]
        self.step_voxel_collision_x_his = []
        self.epo_voxel_collision_x_his = []
        self.epi_voxel_collision_x_his = []
        self._reset_voxel_collision_running("epo")
        self._reset_voxel_collision_running("epi")
        self.epo_loss = [0 for _ in range(self.num_loss)]
        self.epi_loss = [0 for _ in range(self.num_loss)]
        self.epo_count = [0 for _ in range(self.num_loss)]
        self.epi_count = [0 for _ in range(self.num_loss)]
        self.epo_avg = [0 for _ in range(self.num_loss)]
        self.epi_avg = [0 for _ in range(self.num_loss)]
        self.plot_max_points = max(int(getattr(args, "plot_max_points", 512)), 2)
        self.plot_skip_outlier_steps = bool(getattr(args, "plot_skip_outlier_steps", True))
        self.plot_outlier_abs_threshold = float(getattr(args, "plot_outlier_abs_threshold", 0.0))
        self.plot_outlier_rel_factor = float(getattr(args, "plot_outlier_rel_factor", 0.0))
        self.plot_outlier_min_history = max(int(getattr(args, "plot_outlier_min_history", 8)), 0)
        self.plot_outlier_history_window = max(int(getattr(args, "plot_outlier_history_window", 64)), 0)
        self.plot_outlier_min_scale = max(float(getattr(args, "plot_outlier_min_scale", 1.0)), 0.0)

        self.filename_step = f"{args.time}_step"
        self.filename_epo = f"{args.time}_epo"
        self.filename_epi = f"{args.time}_epi"
        self.title_group = [[0, 1, 13], [2, 3, 4, 5, 8, 9], [14, 15, 16], [6, 10], [7, 11, 12]] # actual objective / policy / oracleを同じ圧縮グループに並べる
        self.group_title = [
            "other", 
            "compression", 
            "surrogate",
            "structure_attribution",
            "repair_policy",
        ]
        self.filename = [
            "", 
            "_geom", 
            "_com", 
            "_actual_train_objective", # L_comに使った実codec objectiveの個別plot名
            "_policy_actual",
            "_oracle_teacher_actual",
            "_attr",
            "_policy",
            "_single", 
            "_nodes", 
            "_single_attr",
            "_lowprob_attr",
            "_node_attr",
            "_repair",
            "_sur_train",
            "_sur_bit_err",
            "_sur_mean_err",
            "_actual_ratio",
        ]
        self.title = [
            "Loss", 
            "Loss of Geometry", 
            "Surrogate Predicted Delta (100*(Mine-GT)/GT)",
            "Training Actual Objective Delta",
            "Policy Actual Delta",
            "Oracle Teacher Actual Delta",
            "Loss of Octree Cost Attribution",
            "Loss of Structure Repair Policy",
            "Loss of Single Child Nodes", 
            "Loss of Nodes", 
            "Mean Single-Child Chain Attribution",
            "Mean Low-Probability Occupancy Attribution",
            "Mean Node-Count Attribution",
            "Loss of Structure Repair Actuator",
            "Surrogate Teacher Fit Loss (SmoothL1)",
            "Surrogate Bit Prediction Error (%)",
            "Surrogate Mean Prediction Error (bit/node/single/bpn, %)",
            "Training Actual Objective Ratio (100*Mine/GT)",
        ]
        self.metric_keys = [
            "loss",
            "geom",
            "compression",
            "actual_compression", # L_comに使った実codec objectiveをCSVへ保存する
            "policy_actual_compression",
            "oracle_teacher_compression",
            "attr",
            "policy",
            "single",
            "nodes",
            "single_attr",
            "lowprob_attr",
            "node_attr",
            "repair",
            "surrogate_train",
            "surrogate_bit_error",
            "surrogate_mean_error",
            "actual_compression_ratio",
        ]
        self.latest_episode_loss = None
        self.step_plot_skipped = 0
        self.epoch_plot_skipped = 0
        self.episode_plot_skipped = 0
        self.step_plot_recorded = 0
        self._recent_metric_abs = [[] for _ in range(self.num_loss)]
        self._reset_plot_running("epo")
        self._reset_plot_running("epi")

    def _reset_edit_running(self, mode):
        if mode == "epo":
            self._edit_epo_sums = [0.0 for _ in self.edit_keys]
            self._edit_epo_counts = [0 for _ in self.edit_keys]
            self._edit_epo_steps = 0
        elif mode == "epi":
            self._edit_epi_sums = [0.0 for _ in self.edit_keys]
            self._edit_epi_counts = [0 for _ in self.edit_keys]
            self._edit_epi_steps = 0
        else:
            raise ValueError(f"Unknown edit running mode: {mode}")

    def _edit_running_state(self, mode):
        if mode == "epo":
            return self._edit_epo_sums, self._edit_epo_counts, self._edit_epo_steps
        if mode == "epi":
            return self._edit_epi_sums, self._edit_epi_counts, self._edit_epi_steps
        raise ValueError(f"Unknown edit running mode: {mode}")

    def _set_edit_running_step_count(self, mode, value):
        if mode == "epo":
            self._edit_epo_steps = int(value)
        elif mode == "epi":
            self._edit_epi_steps = int(value)
        else:
            raise ValueError(f"Unknown edit running mode: {mode}")

    def _reset_occupancy_running(self, mode):
        if mode == "epo":
            self._occupancy_epo_sums = [0.0 for _ in self.occupancy_keys]
            self._occupancy_epo_counts = [0 for _ in self.occupancy_keys]
            self._occupancy_epo_steps = 0
        elif mode == "epi":
            self._occupancy_epi_sums = [0.0 for _ in self.occupancy_keys]
            self._occupancy_epi_counts = [0 for _ in self.occupancy_keys]
            self._occupancy_epi_steps = 0
        else:
            raise ValueError(f"Unknown occupancy running mode: {mode}")

    def _occupancy_running_state(self, mode):
        if mode == "epo":
            return self._occupancy_epo_sums, self._occupancy_epo_counts, self._occupancy_epo_steps
        if mode == "epi":
            return self._occupancy_epi_sums, self._occupancy_epi_counts, self._occupancy_epi_steps
        raise ValueError(f"Unknown occupancy running mode: {mode}")

    def _set_occupancy_running_step_count(self, mode, value):
        if mode == "epo":
            self._occupancy_epo_steps = int(value)
        elif mode == "epi":
            self._occupancy_epi_steps = int(value)
        else:
            raise ValueError(f"Unknown occupancy running mode: {mode}")

    def _reset_voxel_collision_running(self, mode):
        if mode == "epo":
            self._voxel_collision_epo_sums = [0.0 for _ in self.voxel_collision_keys]
            self._voxel_collision_epo_counts = [0 for _ in self.voxel_collision_keys]
            self._voxel_collision_epo_steps = 0
        elif mode == "epi":
            self._voxel_collision_epi_sums = [0.0 for _ in self.voxel_collision_keys]
            self._voxel_collision_epi_counts = [0 for _ in self.voxel_collision_keys]
            self._voxel_collision_epi_steps = 0
        else:
            raise ValueError(f"Unknown voxel collision running mode: {mode}")

    def _voxel_collision_running_state(self, mode):
        if mode == "epo":
            return self._voxel_collision_epo_sums, self._voxel_collision_epo_counts, self._voxel_collision_epo_steps
        if mode == "epi":
            return self._voxel_collision_epi_sums, self._voxel_collision_epi_counts, self._voxel_collision_epi_steps
        raise ValueError(f"Unknown voxel collision running mode: {mode}")

    def _set_voxel_collision_running_step_count(self, mode, value):
        if mode == "epo":
            self._voxel_collision_epo_steps = int(value)
        elif mode == "epi":
            self._voxel_collision_epi_steps = int(value)
        else:
            raise ValueError(f"Unknown voxel collision running mode: {mode}")

    def _reset_plot_running(self, mode):
        if mode == "epo":
            self._plot_epo_sums = [0.0 for _ in range(self.num_loss)]
            self._plot_epo_counts = [0 for _ in range(self.num_loss)]
            self._plot_epo_steps = 0
        elif mode == "epi":
            self._plot_epi_sums = [0.0 for _ in range(self.num_loss)]
            self._plot_epi_counts = [0 for _ in range(self.num_loss)]
            self._plot_epi_steps = 0
        else:
            raise ValueError(f"Unknown running-average mode: {mode}")

    def _running_state(self, mode):
        if mode == "epo":
            return self._plot_epo_sums, self._plot_epo_counts, self._plot_epo_steps
        if mode == "epi":
            return self._plot_epi_sums, self._plot_epi_counts, self._plot_epi_steps
        raise ValueError(f"Unknown running-average mode: {mode}")

    def _set_running_step_count(self, mode, value):
        if mode == "epo":
            self._plot_epo_steps = int(value)
        elif mode == "epi":
            self._plot_epi_steps = int(value)
        else:
            raise ValueError(f"Unknown running-average mode: {mode}")

    def _append_list(self, appended_list, appending_list):
        for i in range(len(appended_list)):
            appended_list[i].append(self._metric_float(appending_list[i]))

    def _append_history_entry(self, x_history, loss_history, x_value, values):
        x_history.append(int(x_value))
        self._append_list(loss_history, values)

    def _normalized_metric_list(self, values):
        return [self._metric_float(value) for value in values]

    def _accumulate_plot_running(self, mode, values):
        sums, counts, step_count = self._running_state(mode)
        has_any_value = False
        for idx, value in enumerate(values):
            if value is None:
                continue
            sums[idx] += value
            counts[idx] += 1
            has_any_value = True
        if has_any_value:
            self._set_running_step_count(mode, step_count + 1)

    def _finalize_plot_running(self, mode):
        sums, counts, step_count = self._running_state(mode)
        avgs = []
        for sum_value, count_value in zip(sums, counts):
            if count_value > 0:
                avgs.append(sum_value / float(count_value))
            else:
                avgs.append(None)
        return avgs, int(step_count)

    def _trim_recent_history(self, idx):
        window = int(self.plot_outlier_history_window)
        if window > 0 and len(self._recent_metric_abs[idx]) > window:
            self._recent_metric_abs[idx] = self._recent_metric_abs[idx][-window:]

    def _update_recent_metric_history(self, values):
        for idx, value in enumerate(values):
            if value is None:
                continue
            self._recent_metric_abs[idx].append(abs(value))
            self._trim_recent_history(idx)

    def _detect_step_outlier(self, values):
        finite_metrics = [(idx, value) for idx, value in enumerate(values) if value is not None]
        if not finite_metrics:
            return None

        abs_threshold = float(self.plot_outlier_abs_threshold)
        if abs_threshold > 0.0:
            trigger_idx, trigger_value = max(finite_metrics, key=lambda item: abs(item[1]))
            trigger_abs = abs(trigger_value)
            if trigger_abs > abs_threshold:
                return {
                    "reason": "abs_threshold",
                    "metric_idx": int(trigger_idx),
                    "metric_key": self.metric_keys[trigger_idx],
                    "value": float(trigger_value),
                    "threshold": float(abs_threshold),
                }

        rel_factor = float(self.plot_outlier_rel_factor)
        if rel_factor > 0.0:
            min_history = int(self.plot_outlier_min_history)
            min_scale = max(float(self.plot_outlier_min_scale), 0.0)
            for idx, value in finite_metrics:
                history = self._recent_metric_abs[idx]
                if len(history) < min_history:
                    continue
                baseline = statistics.median(history)
                baseline = max(float(baseline), min_scale)
                rel_threshold = baseline * rel_factor
                if abs(value) > rel_threshold:
                    return {
                        "reason": "relative_threshold",
                        "metric_idx": int(idx),
                        "metric_key": self.metric_keys[idx],
                        "value": float(value),
                        "threshold": float(rel_threshold),
                        "baseline": float(baseline),
                    }
        return None

    def record_metrics(self, mode, x_value, values):
        numeric_values = self._normalized_metric_list(values)
        info = {
            "mode": mode,
            "x_value": int(x_value),
            "recorded": True,
            "skipped": False,
            "plot_values": numeric_values,
        }

        if mode == "step":
            outlier_info = None
            if bool(self.plot_skip_outlier_steps):
                outlier_info = self._detect_step_outlier(numeric_values)
            if outlier_info is not None:
                self.step_plot_skipped += 1
                info.update(
                    {
                        "recorded": False,
                        "skipped": True,
                        "reason": outlier_info["reason"],
                        "metric_idx": outlier_info["metric_idx"],
                        "metric_key": outlier_info["metric_key"],
                        "value": outlier_info["value"],
                        "threshold": outlier_info["threshold"],
                    }
                )
                if "baseline" in outlier_info:
                    info["baseline"] = outlier_info["baseline"]
                return info

            self._append_history_entry(self.step_x_his, self.step_loss_his, x_value, numeric_values)
            self._accumulate_plot_running("epo", numeric_values)
            self._accumulate_plot_running("epi", numeric_values)
            self._update_recent_metric_history(numeric_values)
            self.step_plot_recorded += 1
            return info

        if mode == "epo":
            plot_values, accepted_steps = self._finalize_plot_running("epo")
            self._reset_plot_running("epo")
            info["plot_values"] = plot_values
            info["accepted_steps"] = accepted_steps
            if accepted_steps <= 0:
                self.epoch_plot_skipped += 1
                info.update(
                    {
                        "recorded": False,
                        "skipped": True,
                        "reason": "no_valid_plot_steps",
                    }
                )
                return info
            self._append_history_entry(self.epo_x_his, self.epo_loss_his, x_value, plot_values)
            return info

        if mode == "epi":
            self.latest_episode_loss = self._metric_float(values[0])
            plot_values, accepted_steps = self._finalize_plot_running("epi")
            self._reset_plot_running("epi")
            info["plot_values"] = plot_values
            info["accepted_steps"] = accepted_steps
            if accepted_steps <= 0:
                self.episode_plot_skipped += 1
                info.update(
                    {
                        "recorded": False,
                        "skipped": True,
                        "reason": "no_valid_plot_steps",
                    }
                )
                return info
            self._append_history_entry(self.epi_x_his, self.epi_loss_his, x_value, plot_values)
            return info

        raise ValueError(f"Unknown plot mode: {mode}")

    def _normalize_edit_values(self, edit_stats):
        edit_stats = edit_stats or {}
        values = []
        for key in self.edit_keys:
            values.append(self._metric_float(edit_stats.get(key)))
        return values

    def _append_edit_history_entry(self, x_history, edit_history, x_value, values):
        x_history.append(int(x_value))
        for idx, value in enumerate(values):
            edit_history[idx].append(self._metric_float(value))

    def _accumulate_edit_running(self, mode, values):
        sums, counts, step_count = self._edit_running_state(mode)
        has_any_value = False
        for idx, value in enumerate(values):
            if value is None:
                continue
            sums[idx] += float(value)
            counts[idx] += 1
            has_any_value = True
        if has_any_value:
            self._set_edit_running_step_count(mode, step_count + 1)

    def record_point_edits(self, mode, x_value, edit_stats=None):
        info = {
            "mode": mode,
            "x_value": int(x_value),
            "recorded": True,
            "skipped": False,
            "plot_values": None,
        }
        if mode == "step":
            values = self._normalize_edit_values(edit_stats)
            info["plot_values"] = values
            self._append_edit_history_entry(self.step_edit_x_his, self.step_edit_his, x_value, values)
            self._accumulate_edit_running("epo", values)
            self._accumulate_edit_running("epi", values)
            return info
        if mode == "epo":
            sums, counts, accepted_steps = self._edit_running_state("epo")
            values = [
                (sum_value / float(count_value)) if count_value > 0 else None
                for sum_value, count_value in zip(sums, counts)
            ]
            self._reset_edit_running("epo")
            info["plot_values"] = values
            info["accepted_steps"] = accepted_steps
            if accepted_steps <= 0:
                info.update({"recorded": False, "skipped": True, "reason": "no_edit_steps"})
                return info
            self._append_edit_history_entry(self.epo_edit_x_his, self.epo_edit_his, x_value, values)
            return info
        if mode == "epi":
            sums, counts, accepted_steps = self._edit_running_state("epi")
            values = [
                (sum_value / float(count_value)) if count_value > 0 else None
                for sum_value, count_value in zip(sums, counts)
            ]
            self._reset_edit_running("epi")
            info["plot_values"] = values
            info["accepted_steps"] = accepted_steps
            if accepted_steps <= 0:
                info.update({"recorded": False, "skipped": True, "reason": "no_edit_steps"})
                return info
            self._append_edit_history_entry(self.epi_edit_x_his, self.epi_edit_his, x_value, values)
            return info
        raise ValueError(f"Unknown point edit mode: {mode}")

    def _normalize_occupancy_values(self, row):
        row = row or {}
        values = []
        for key in self.occupancy_keys:
            values.append(self._metric_float(row.get(key)))
        return values

    def _append_occupancy_history_entry(self, x_history, occupancy_history, x_value, values):
        x_history.append(int(x_value))
        for idx, value in enumerate(values):
            occupancy_history[idx].append(self._metric_float(value))

    def _accumulate_occupancy_running(self, mode, values):
        sums, counts, step_count = self._occupancy_running_state(mode)
        has_any_value = False
        for idx, value in enumerate(values):
            if value is None:
                continue
            sums[idx] += float(value)
            counts[idx] += 1
            has_any_value = True
        if has_any_value:
            self._set_occupancy_running_step_count(mode, step_count + 1)

    def record_occupancy_metrics(self, mode, x_value, row=None):
        info = {
            "mode": mode,
            "x_value": int(x_value),
            "recorded": True,
            "skipped": False,
            "plot_values": None,
        }
        if mode == "step":
            values = self._normalize_occupancy_values(row)
            info["plot_values"] = values
            self._append_occupancy_history_entry(self.step_occupancy_x_his, self.step_occupancy_his, x_value, values)
            self._accumulate_occupancy_running("epo", values)
            self._accumulate_occupancy_running("epi", values)
            return info
        if mode == "epo":
            sums, counts, accepted_steps = self._occupancy_running_state("epo")
            values = [
                (sum_value / float(count_value)) if count_value > 0 else None
                for sum_value, count_value in zip(sums, counts)
            ]
            self._reset_occupancy_running("epo")
            info["plot_values"] = values
            info["accepted_steps"] = accepted_steps
            if accepted_steps <= 0:
                info.update({"recorded": False, "skipped": True, "reason": "no_occupancy_steps"})
                return info
            self._append_occupancy_history_entry(self.epo_occupancy_x_his, self.epo_occupancy_his, x_value, values)
            return info
        if mode == "epi":
            sums, counts, accepted_steps = self._occupancy_running_state("epi")
            values = [
                (sum_value / float(count_value)) if count_value > 0 else None
                for sum_value, count_value in zip(sums, counts)
            ]
            self._reset_occupancy_running("epi")
            info["plot_values"] = values
            info["accepted_steps"] = accepted_steps
            if accepted_steps <= 0:
                info.update({"recorded": False, "skipped": True, "reason": "no_occupancy_steps"})
                return info
            self._append_occupancy_history_entry(self.epi_occupancy_x_his, self.epi_occupancy_his, x_value, values)
            return info
        raise ValueError(f"Unknown occupancy mode: {mode}")

    def _normalize_voxel_collision_values(self, row):
        row = row or {}
        values = []
        for key in self.voxel_collision_keys:
            value = self._metric_float(row.get(key))
            if value is None:
                stage = key.replace("voxel_collision_", "").replace("_point_reduction_rate", "")
                alias_key = f"voxel_{stage}_point_reduction_rate"
                value = self._metric_float(row.get(alias_key))
            if value is None:
                stage = key.replace("voxel_collision_", "").replace("_point_reduction_rate", "")
                for prefix in (f"voxel_collision_{stage}", f"voxel_{stage}"):
                    raw_count = self._metric_float(row.get(f"{prefix}_finite_point_count"))
                    if raw_count is None:
                        raw_count = self._metric_float(row.get(f"{prefix}_raw_point_count"))
                    unique_count = self._metric_float(row.get(f"{prefix}_unique_voxel_count"))
                    if raw_count is not None and unique_count is not None:
                        value = 1.0 - float(unique_count) / max(float(raw_count), 1.0)
                        break
            values.append(value)
        return values

    def _append_voxel_collision_history_entry(self, x_history, voxel_collision_history, x_value, values):
        x_history.append(int(x_value))
        for idx, value in enumerate(values):
            voxel_collision_history[idx].append(self._metric_float(value))

    def _accumulate_voxel_collision_running(self, mode, values):
        sums, counts, step_count = self._voxel_collision_running_state(mode)
        has_any_value = False
        for idx, value in enumerate(values):
            if value is None:
                continue
            sums[idx] += float(value)
            counts[idx] += 1
            has_any_value = True
        if has_any_value:
            self._set_voxel_collision_running_step_count(mode, step_count + 1)

    def record_voxel_collision_metrics(self, mode, x_value, row=None):
        info = {
            "mode": mode,
            "x_value": int(x_value),
            "recorded": True,
            "skipped": False,
            "plot_values": None,
        }
        if mode == "step":
            values = self._normalize_voxel_collision_values(row)
            info["plot_values"] = values
            self._append_voxel_collision_history_entry(self.step_voxel_collision_x_his, self.step_voxel_collision_his, x_value, values)
            self._accumulate_voxel_collision_running("epo", values)
            self._accumulate_voxel_collision_running("epi", values)
            return info
        if mode == "epo":
            sums, counts, accepted_steps = self._voxel_collision_running_state("epo")
            values = [
                (sum_value / float(count_value)) if count_value > 0 else None
                for sum_value, count_value in zip(sums, counts)
            ]
            self._reset_voxel_collision_running("epo")
            info["plot_values"] = values
            info["accepted_steps"] = accepted_steps
            if accepted_steps <= 0:
                info.update({"recorded": False, "skipped": True, "reason": "no_voxel_collision_steps"})
                return info
            self._append_voxel_collision_history_entry(self.epo_voxel_collision_x_his, self.epo_voxel_collision_his, x_value, values)
            return info
        if mode == "epi":
            sums, counts, accepted_steps = self._voxel_collision_running_state("epi")
            values = [
                (sum_value / float(count_value)) if count_value > 0 else None
                for sum_value, count_value in zip(sums, counts)
            ]
            self._reset_voxel_collision_running("epi")
            info["plot_values"] = values
            info["accepted_steps"] = accepted_steps
            if accepted_steps <= 0:
                info.update({"recorded": False, "skipped": True, "reason": "no_voxel_collision_steps"})
                return info
            self._append_voxel_collision_history_entry(self.epi_voxel_collision_x_his, self.epi_voxel_collision_his, x_value, values)
            return info
        raise ValueError(f"Unknown voxel collision mode: {mode}")

    @staticmethod
    def _metric_float(value):
        if hasattr(value, "detach"):
            value = value.detach()
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    def _plot_values(self, values):
        plotted = []
        for value in values:
            numeric = self._metric_float(value)
            plotted.append(float("nan") if numeric is None else numeric)
        return plotted

    def _downsample_series(self, epochs, values):
        if len(values) <= self.plot_max_points:
            return epochs, values
        span = len(values) - 1
        indices = []
        last_idx = -1
        for i in range(self.plot_max_points):
            idx = round(i * span / float(max(self.plot_max_points - 1, 1)))
            if idx != last_idx:
                indices.append(idx)
                last_idx = idx
        if indices[-1] != span:
            indices.append(span)
        return [epochs[idx] for idx in indices], [values[idx] for idx in indices]

    def _write_csv(self, loss_history, x_history, filename_front, x_label):
        os.makedirs(self.save_dir, exist_ok=True)
        max_len = len(x_history)
        save_path = os.path.join(self.save_dir, f"{filename_front}_metrics.csv")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"{x_label}," + ",".join(self.metric_keys) + "\n")
            for step in range(max_len):
                row = [str(int(x_history[step]))]
                for metric in loss_history:
                    if step < len(metric):
                        value = self._metric_float(metric[step])
                        row.append("" if value is None else f"{value:.10g}")
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")

    def _write_edit_csv(self, edit_history, x_history, filename_front, x_label):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"{filename_front}_point_edits.csv")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"{x_label}," + ",".join(self.edit_keys) + "\n")
            for step in range(len(x_history)):
                row = [str(int(x_history[step]))]
                for metric in edit_history:
                    if step < len(metric):
                        value = self._metric_float(metric[step])
                        row.append("" if value is None else f"{value:.10g}")
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")

    def _write_occupancy_csv(self, occupancy_history, x_history, filename_front, x_label):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"{filename_front}_occupancy_metrics.csv")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"{x_label}," + ",".join(self.occupancy_keys) + "\n")
            for step in range(len(x_history)):
                row = [str(int(x_history[step]))]
                for metric in occupancy_history:
                    if step < len(metric):
                        value = self._metric_float(metric[step])
                        row.append("" if value is None else f"{value:.10g}")
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")

    def _write_voxel_collision_csv(self, voxel_collision_history, x_history, filename_front, x_label):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"{filename_front}_voxel_collision_metrics.csv")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"{x_label}," + ",".join(self.voxel_collision_keys) + "\n")
            for step in range(len(x_history)):
                row = [str(int(x_history[step]))]
                for metric in voxel_collision_history:
                    if step < len(metric):
                        value = self._metric_float(metric[step])
                        row.append("" if value is None else f"{value:.10g}")
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")

    def _plot_single_axis(self, ax, epochs, values, loss_idx, xl):
        values = self._plot_values(values)
        epochs, values = self._downsample_series(epochs, values)
        if any(math.isfinite(value) for value in values):
            ax.plot(epochs, values, marker="o", linewidth=2, markersize=3)
        else:
            ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
        ax.set_xlabel(xl)
        ax.set_ylabel(self.metric_keys[loss_idx])
        ax.set_title(self.title[loss_idx])
        ax.grid(True, alpha=0.35)
        if len(epochs) >= 2:
            ax.set_xlim(min(epochs), max(epochs))
            try:
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            except Exception:
                pass

    def _plot_compression_compare_axis(self, ax, loss_history, x_history, xl):
        key_to_index = {key: index for index, key in enumerate(self.metric_keys)}
        surrogate_idx = key_to_index.get("compression")
        actual_idx = key_to_index.get("actual_compression")
        policy_idx = key_to_index.get("policy_actual_compression")
        oracle_idx = key_to_index.get("oracle_teacher_compression")
        if surrogate_idx is None or actual_idx is None:
            return
        plotted = False
        handles = []
        series = [
            ("Surrogate", loss_history[surrogate_idx], "tab:blue"),
            ("TrainObjective", loss_history[actual_idx], "tab:orange"),
        ]
        if policy_idx is not None:
            series.append(("PolicyActual", loss_history[policy_idx], "tab:green"))
        if oracle_idx is not None:
            series.append(("OracleTeacher", loss_history[oracle_idx], "tab:red"))
        for label, values, color in series:
            plot_values = self._plot_values(values)
            plot_x, plot_values = self._downsample_series(list(x_history), plot_values)
            if any(math.isfinite(value) for value in plot_values):
                line, = ax.plot(plot_x, plot_values, marker="o", linewidth=2, markersize=3, color=color, label=label)
                handles.append(line)
                plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "no compression data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
        ax.set_xlabel(xl)
        ax.set_ylabel("Delta [%]")
        ax.set_title("Surrogate / Train Objective / Policy / Oracle Delta")
        if handles:
            ax.legend(handles=handles, loc="best")
        ax.grid(True, alpha=0.35)
        if len(x_history) >= 2:
            ax.set_xlim(min(x_history), max(x_history))
            try:
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            except Exception:
                pass

    def _plot_actual_ratio_axis(self, ax, loss_history, x_history, xl):
        key_to_index = {key: index for index, key in enumerate(self.metric_keys)}
        ratio_idx = key_to_index.get("actual_compression_ratio")
        plotted = False
        if ratio_idx is not None and ratio_idx < len(loss_history):
            ratio_values = self._plot_values(loss_history[ratio_idx])
            ratio_x, ratio_values = self._downsample_series(list(x_history), ratio_values)
            if any(math.isfinite(value) for value in ratio_values):
                ax.plot(
                    ratio_x,
                    ratio_values,
                    marker="o",
                    linewidth=2,
                    markersize=3,
                    color="tab:green",
                    label="Actual Ratio",
                )
                plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "no actual ratio data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
        ax.axhline(100.0, color="black", linewidth=0.7, alpha=0.4)
        ax.set_xlabel(xl)
        ax.set_ylabel("Ratio [%]")
        ax.set_title("Training Actual Objective Ratio (100*Mine/GT)")
        if plotted:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.35)
        if len(x_history) >= 2:
            ax.set_xlim(min(x_history), max(x_history))
            try:
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            except Exception:
                pass

    def plot_loss_curve(self, epoORepi):
        if epoORepi == "step":
            loss_history = self.step_loss_his
            x_history = self.step_x_his
            filename_front = self.filename_step
            xl = "Train Step"
            x_label = "step"
        elif epoORepi == "epo":
            loss_history = self.epo_loss_his
            x_history = self.epo_x_his
            filename_front = self.filename_epo
            xl = "Epoch"
            x_label = "epoch"
        else:
            loss_history = self.epi_loss_his
            x_history = self.epi_x_his
            filename_front = self.filename_epi
            xl = "Episode"
            x_label = "episode"

        os.makedirs(self.save_dir, exist_ok=True)
        self._write_csv(loss_history, x_history, filename_front, x_label)
        plot_mod = _get_pyplot()
        if plot_mod is None:
            return

        for group_idx, group in enumerate(self.title_group):
            save_path = os.path.join(self.save_dir, f"{filename_front}_{self.group_title[group_idx]}.png")
            save_dir_full = os.path.dirname(save_path)
            if save_dir_full != "":
                os.makedirs(save_dir_full, exist_ok=True)

            is_compression_group = self.group_title[group_idx] == "compression"
            axis_count = len(group) + (2 if is_compression_group else 0)
            fig, axes = plot_mod.subplots(axis_count, 1, figsize=(self.x_len, self.y_len * axis_count))

            if axis_count == 1:
                axes = [axes]

            axis_offset = 0
            if is_compression_group:
                # 従来のdelta比較と新しいratio表示を同じPNG内の別サブプロットに分ける。
                self._plot_compression_compare_axis(axes[0], loss_history, x_history, xl)
                self._plot_actual_ratio_axis(axes[1], loss_history, x_history, xl)
                axis_offset = 2

            for ax, loss_idx in zip(axes[axis_offset:], group):
                epochs = list(x_history)
                self._plot_single_axis(ax, epochs, loss_history[loss_idx], loss_idx, xl)

            plot_mod.tight_layout()
            plot_mod.savefig(save_path, dpi=140)
            plot_mod.close(fig)


    def plot_point_edit_curve(self, epoORepi):
        if epoORepi == "step":
            edit_history = self.step_edit_his
            x_history = self.step_edit_x_his
            filename_front = self.filename_step
            xl = "Train Step"
            x_label = "step"
        elif epoORepi == "epo":
            edit_history = self.epo_edit_his
            x_history = self.epo_edit_x_his
            filename_front = self.filename_epo
            xl = "Epoch"
            x_label = "epoch"
        else:
            edit_history = self.epi_edit_his
            x_history = self.epi_edit_x_his
            filename_front = self.filename_epi
            xl = "Episode"
            x_label = "episode"

        os.makedirs(self.save_dir, exist_ok=True)
        self._write_edit_csv(edit_history, x_history, filename_front, x_label)
        plot_mod = _get_pyplot()
        if plot_mod is None:
            return

        save_path = os.path.join(self.save_dir, f"{filename_front}_point_edits.png")
        edit_titles = {
            "added_ratio_percent": "Add",
            "deleted_ratio_percent": "Prune",
            "full_cloud_voxel_drop_ratio_percent": "FullCloudVoxelPrune",
            "adjusted_ratio_percent": "Adjust",
            "oracle_full_cloud_prune_ratio_percent": "OracleFullPrune",
        }
        edit_colors = {
            "added_ratio_percent": "#2ca02c",
            "deleted_ratio_percent": "#d62728",
            "full_cloud_voxel_drop_ratio_percent": "#ff7f0e",
            "adjusted_ratio_percent": "#1f77b4",
            "oracle_full_cloud_prune_ratio_percent": "#9467bd",
        }
        fig, axes = plot_mod.subplots(
            len(self.edit_keys),
            1,
            figsize=(self.x_len, self.y_len * len(self.edit_keys)),
            sharex=False,
        )
        if len(self.edit_keys) == 1:
            axes = [axes]
        x_values = list(x_history)
        for ax, key, values in zip(axes, self.edit_keys, edit_history):
            plotted = False
            if values:
                plot_values = self._plot_values(values)
                epochs, plot_values = self._downsample_series(x_values, plot_values)
                if any(math.isfinite(value) for value in plot_values):
                    ax.plot(
                        epochs,
                        plot_values,
                        marker="o",
                        linewidth=2,
                        markersize=3,
                        color=edit_colors.get(key),
                    )
                    plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "no point edit data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
            ax.set_xlabel(xl)
            ax.set_ylabel(f"{edit_titles.get(key, key)} Ratio [%]")
            ax.set_title(f"{edit_titles.get(key, key)} Ratio")
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
            ax.grid(True, alpha=0.35)
            if len(x_history) >= 2:
                ax.set_xlim(min(x_history), max(x_history))
                try:
                    from matplotlib.ticker import MaxNLocator
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                except Exception:
                    pass
        fig.tight_layout(h_pad=2.0)
        plot_mod.savefig(save_path, dpi=140)
        plot_mod.close(fig)


    def plot_voxel_collision_curve(self, epoORepi):
        if epoORepi == "step":
            voxel_collision_history = self.step_voxel_collision_his
            x_history = self.step_voxel_collision_x_his
            filename_front = self.filename_step
            xl = "Train Step"
            x_label = "step"
        elif epoORepi == "epo":
            voxel_collision_history = self.epo_voxel_collision_his
            x_history = self.epo_voxel_collision_x_his
            filename_front = self.filename_epo
            xl = "Epoch"
            x_label = "epoch"
        else:
            voxel_collision_history = self.epi_voxel_collision_his
            x_history = self.epi_voxel_collision_x_his
            filename_front = self.filename_epi
            xl = "Episode"
            x_label = "episode"

        os.makedirs(self.save_dir, exist_ok=True)
        self._write_voxel_collision_csv(voxel_collision_history, x_history, filename_front, x_label)
        plot_mod = _get_pyplot()
        if plot_mod is None:
            return

        save_path = os.path.join(self.save_dir, f"{filename_front}_voxel_collision.png")
        fig, ax = plot_mod.subplots(1, 1, figsize=(self.x_len, self.y_len))
        plotted = False
        x_values = list(x_history)
        for key, values in zip(self.voxel_collision_keys, voxel_collision_history):
            plot_values = self._plot_values(values)
            plot_x, plot_values = self._downsample_series(x_values, plot_values)
            if any(math.isfinite(value) for value in plot_values):
                ax.plot(
                    plot_x,
                    plot_values,
                    marker="o",
                    linewidth=2,
                    markersize=3,
                    color=self.voxel_collision_colors.get(key),
                    label=self.voxel_collision_titles.get(key, key),
                )
                plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "no voxel collision data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
        ax.set_xlabel(xl)
        ax.set_ylabel("Point Reduction Rate")
        ax.set_title("SparsePCGC Quantized Voxel Collapse Rate")
        if plotted:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.35)
        if len(x_values) >= 2:
            ax.set_xlim(min(x_values), max(x_values))
            try:
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            except Exception:
                pass
        fig.tight_layout()
        plot_mod.savefig(save_path, dpi=140)
        plot_mod.close(fig)


    def plot_occupancy_curve(self, epoORepi):
        if epoORepi == "step":
            occupancy_history = self.step_occupancy_his
            x_history = self.step_occupancy_x_his
            filename_front = self.filename_step
            xl = "Train Step"
            x_label = "step"
        elif epoORepi == "epo":
            occupancy_history = self.epo_occupancy_his
            x_history = self.epo_occupancy_x_his
            filename_front = self.filename_epo
            xl = "Epoch"
            x_label = "epoch"
        else:
            occupancy_history = self.epi_occupancy_his
            x_history = self.epi_occupancy_x_his
            filename_front = self.filename_epi
            xl = "Episode"
            x_label = "episode"

        os.makedirs(self.save_dir, exist_ok=True)
        self._write_occupancy_csv(occupancy_history, x_history, filename_front, x_label)
        plot_mod = _get_pyplot()
        if plot_mod is None:
            return

        save_path = os.path.join(self.save_dir, f"{filename_front}_occupancy.png")
        key_to_history = {
            key: occupancy_history[index]
            for index, key in enumerate(self.occupancy_keys)
        }
        plot_groups = [
            group
            for group in self.occupancy_plot_groups
            if any(key in key_to_history for _, key, _ in group[1])
        ]
        if not plot_groups:
            plot_groups = [
                (
                    self.occupancy_titles.get(key, key),
                    [(self.occupancy_titles.get(key, key), key, "tab:blue")],
                )
                for key in self.occupancy_keys
            ]
        fig, axes = plot_mod.subplots(
            len(plot_groups),
            1,
            figsize=(self.x_len, max(self.y_len * len(plot_groups), self.y_len)),
            sharex=False,
        )
        if len(plot_groups) == 1:
            axes = [axes]
        x_values = list(x_history)
        for ax, (title, series_list) in zip(axes, plot_groups):
            plotted = False
            plot_x_ref = []
            for label, key, color in series_list:
                values = key_to_history.get(key)
                if values is None:
                    continue
                plot_values = self._plot_values(values)
                plot_x, plot_values = self._downsample_series(x_values, plot_values)
                plot_x_ref = plot_x
                if any(math.isfinite(value) for value in plot_values):
                    ax.plot(
                        plot_x,
                        plot_values,
                        marker="o",
                        linewidth=2,
                        markersize=3,
                        color=color,
                        label=label,
                    )
                    plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "no occupancy data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
            ax.set_xlabel(xl)
            ax.set_ylabel(title)
            ax.set_title(title)
            if plotted and len(series_list) > 1:
                ax.legend(loc="best")
            ax.grid(True, alpha=0.35)
            if len(plot_x_ref) >= 2:
                ax.set_xlim(min(plot_x_ref), max(plot_x_ref))
                try:
                    from matplotlib.ticker import MaxNLocator
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                except Exception:
                    pass
        fig.tight_layout(h_pad=2.0)
        plot_mod.savefig(save_path, dpi=140)
        plot_mod.close(fig)


    def epi_loss_return(self):
        loss = self.latest_episode_loss
        if loss is None and self.epi_loss_his[0]:
            loss = self._metric_float(self.epi_loss_his[0][-1])
        return float("inf") if loss is None else loss

    def record_surrogate_pretrain(self, step, row):
        """
        Surrogate事前学習のうち、本番学習のSurrogateプロットと同じ3指標だけを記録する。
        """
        if not hasattr(self, "surrogate_pretrain_keys"):
            self.surrogate_pretrain_keys = [
                "surrogate_pretrain_loss",
                "surrogate_pretrain_abs_error",
                "surrogate_pretrain_mean_error",
            ]
            self.surrogate_pretrain_titles = {
                "surrogate_pretrain_loss": "Surrogate Teacher Fit Loss (SmoothL1)",
                "surrogate_pretrain_abs_error": "Surrogate Bit Prediction Error (%)",
                "surrogate_pretrain_mean_error": "Surrogate Mean Prediction Error (bit/node/single/bpn, %)",
            }
            self.surrogate_pretrain_x_his = []
            self.surrogate_pretrain_his = {
                key: [] for key in self.surrogate_pretrain_keys
            }

        self.surrogate_pretrain_x_his.append(int(step))

        self.surrogate_pretrain_his["surrogate_pretrain_loss"].append(
            self._metric_float(row.get("surrogate_pretrain_loss"))
        )
        self.surrogate_pretrain_his["surrogate_pretrain_abs_error"].append(
            self._metric_float(row.get("surrogate_pretrain_abs_error"))
        )

        mean_error = row.get("surrogate_pretrain_mean_error", None)
        if mean_error is None:
            pred_value = self._metric_float(row.get("surrogate_pretrain_pred_bit_percent"))
            actual_value = self._metric_float(row.get("surrogate_pretrain_actual_bit_percent"))
            if pred_value is not None and actual_value is not None:
                mean_error = abs(pred_value - actual_value)
            else:
                mean_error = None

        self.surrogate_pretrain_his["surrogate_pretrain_mean_error"].append(
            self._metric_float(mean_error)
        )


    def plot_surrogate_pretrain_curve(self):
        """
        Surrogate事前学習の3指標を、本番学習のSurrogateグラフと同じ形式で保存する。
        """
        if not hasattr(self, "surrogate_pretrain_x_his"):
            return
        if len(self.surrogate_pretrain_x_his) == 0:
            return

        os.makedirs(self.save_dir, exist_ok=True)

        csv_path = os.path.join(
            self.save_dir,
            f"{self.args.time}_surrogate_pretrain_metrics.csv"
        )
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step," + ",".join(self.surrogate_pretrain_keys) + "\n")
            for row_idx, step in enumerate(self.surrogate_pretrain_x_his):
                values = [str(int(step))]
                for key in self.surrogate_pretrain_keys:
                    value = self.surrogate_pretrain_his[key][row_idx]
                    values.append("" if value is None else f"{value:.10g}")
                f.write(",".join(values) + "\n")

        plot_mod = _get_pyplot()
        if plot_mod is None:
            return

        fig, axes = plot_mod.subplots(
            len(self.surrogate_pretrain_keys),
            1,
            figsize=(self.x_len, self.y_len * len(self.surrogate_pretrain_keys)),
        )

        if len(self.surrogate_pretrain_keys) == 1:
            axes = [axes]

        x_values = list(self.surrogate_pretrain_x_his)

        for ax, key in zip(axes, self.surrogate_pretrain_keys):
            values = self._plot_values(self.surrogate_pretrain_his[key])
            plot_x, plot_values = self._downsample_series(x_values, values)

            if any(math.isfinite(value) for value in plot_values):
                ax.plot(
                    plot_x,
                    plot_values,
                    marker="o",
                    linewidth=2,
                    markersize=3,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "no finite data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    alpha=0.7,
                )

            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
            ax.set_xlabel("Surrogate Pretrain Step")
            ax.set_ylabel(key)
            ax.set_title(self.surrogate_pretrain_titles.get(key, key))
            ax.grid(True, alpha=0.35)

            if len(plot_x) >= 2:
                ax.set_xlim(min(plot_x), max(plot_x))
                try:
                    from matplotlib.ticker import MaxNLocator
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                except Exception:
                    pass

        fig.tight_layout()
        save_path = os.path.join(
            self.save_dir,
            f"{self.args.time}_surrogate_pretrain.png"
        )
        plot_mod.savefig(save_path, dpi=140)
        plot_mod.close(fig)
