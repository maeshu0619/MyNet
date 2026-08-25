"""固定検証に基づいて、訓練を終了してよいかを判定する。"""

import math


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _mean(values):
    return sum(values) / float(len(values))


def _slope(values):
    """Episodeをx軸にした最小二乗傾き。"""
    count = len(values)
    if count < 2:
        return float("inf")
    x_mean = (count - 1) / 2.0
    y_mean = _mean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(count))
    if denominator <= 0.0:
        return 0.0
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator


def convergence_episode_limit(args):
    """通常実行では従来Episode数、収束制御時は安全上限を返す。"""
    minimum = max(
        int(getattr(args, "convergence_min_episodes", 0)),
        int(getattr(args, "episodes", 1)),
        1,
    )
    if not bool(getattr(args, "train_until_converged", False)):
        return int(getattr(args, "episodes", minimum))
    return max(int(getattr(args, "convergence_max_episodes", minimum)), minimum)


def exploration_schedule_step_estimate(args, total_train_files):
    """明示固定されていなければ実際の最大訓練長へ探索を追従させる。"""
    schedule_episodes = int(getattr(args, "exploration_schedule_episodes", 0))
    if schedule_episodes <= 0:
        schedule_episodes = convergence_episode_limit(args)
    return max(schedule_episodes, 1) * max(int(total_train_files), 1)


class TrainingConvergenceMonitor:
    """Rate-Distortionと訓練損失の安定を低コストに監視する。"""

    def __init__(self, args):
        self.args = args
        self.enabled = bool(getattr(args, "train_until_converged", False))
        self.minimum_episodes = max(
            int(getattr(args, "convergence_min_episodes", 0)),
            int(getattr(args, "episodes", 1)),
            1,
        )
        self.window = max(int(getattr(args, "convergence_window_episodes", 16)), 4)
        self.patience = max(int(getattr(args, "convergence_patience_episodes", 32)), 1)
        self.history = []
        self.stable_episodes = 0
        self.last_rollback_episode = None
        self.converged = False
        self.last_event = {}

    def _exploration_finished(self, global_step):
        total_steps = max(int(getattr(self.args, "_total_train_steps_estimate", 0)), 1)
        fraction = min(max(float(getattr(
            self.args, "repair_exploration_fraction", 0.0
        )), 0.0), 1.0)
        return int(global_step) >= int(math.ceil(total_steps * fraction))

    def update(self, checkpoint_metrics, guard_event, global_step):
        if not self.enabled:
            return {"enabled": False, "converged": False}

        metrics = checkpoint_metrics or {}
        episode = int(metrics.get("episode") or (len(self.history) + 1))
        guard_action = str((guard_event or {}).get("action", "")).strip().lower()
        if guard_action == "rollback":
            self.last_rollback_episode = episode

        record = {
            "episode": episode,
            "total_loss": _finite(metrics.get("total_loss")),
            "compression": _finite(metrics.get("compression_loss_L_com")),
            "fixed_objective": _finite(metrics.get("full_cloud_val_fixed_objective")),
            "fixed_actual": _finite(metrics.get("full_cloud_val_actual_percent")),
            "fixed_geometry": _finite(metrics.get("full_cloud_val_geometry")),
            "signature": str(metrics.get("full_cloud_val_sample_signature") or ""),
            "optimizer_ok": bool(metrics.get("optimizer_success_ok", False)),
            "geometry_ok": bool(metrics.get("geometry_ok", False)),
            "safety_ok": bool(metrics.get("safety_ok", False)),
        }
        self.history.append(record)
        self.history = self.history[-max(self.window * 2, self.patience + self.window):]

        reasons = []
        stats = {}
        if episode < self.minimum_episodes:
            reasons.append("minimum_episodes")
        # args.episodesを「最低訓練長」として扱い、その後にも独立した安定期間を
        # 必ず要求する。収束した直後に固定長で終了する問題を防ぐ。
        if episode <= self.minimum_episodes:
            reasons.append("post_minimum_evidence")
        if not self._exploration_finished(global_step):
            reasons.append("exploration_active")
        if len(self.history) < self.window:
            reasons.append("window_not_full")

        recent = self.history[-self.window:]
        required = (
            "total_loss", "compression", "fixed_objective",
            "fixed_actual", "fixed_geometry",
        )
        for name in required:
            if len(recent) < self.window or any(row[name] is None for row in recent):
                reasons.append(f"missing_{name}")

        signatures = {row["signature"] for row in recent if row["signature"]}
        if len(recent) == self.window and len(signatures) != 1:
            reasons.append("validation_signature_changed")

        cooldown = max(int(getattr(
            self.args, "convergence_guard_cooldown_episodes", 16
        )), 0)
        if (
            self.last_rollback_episode is not None
            and episode - self.last_rollback_episode < cooldown
        ):
            reasons.append("guard_cooldown")

        actual_target = float(getattr(
            self.args, "convergence_actual_compression_target", -3.5
        ))
        if record["fixed_actual"] is None or record["fixed_actual"] > actual_target:
            reasons.append("actual_target")
        if not record["optimizer_ok"]:
            reasons.append("optimizer")
        if not record["geometry_ok"] or not record["safety_ok"]:
            reasons.append("safety_gate")

        if len(recent) == self.window and not any(
            reason.startswith("missing_") for reason in reasons
        ):
            half = self.window // 2
            for name in required:
                values = [row[name] for row in recent]
                stats[f"{name}_slope"] = _slope(values)
                stats[f"{name}_half_delta"] = (
                    _mean(values[half:]) - _mean(values[:half])
                )

            if abs(stats["total_loss_slope"]) > float(getattr(
                self.args, "convergence_total_loss_slope_max", 0.02
            )):
                reasons.append("total_loss_slope")
            if abs(stats["total_loss_half_delta"]) > float(getattr(
                self.args, "convergence_total_loss_half_delta_max", 0.12
            )):
                reasons.append("total_loss_level")
            if abs(stats["compression_slope"]) > float(getattr(
                self.args, "convergence_compression_slope_max", 0.01
            )):
                reasons.append("compression_slope")
            if abs(stats["compression_half_delta"]) > float(getattr(
                self.args, "convergence_compression_half_delta_max", 0.08
            )):
                reasons.append("compression_level")
            if abs(stats["fixed_objective_slope"]) > float(getattr(
                self.args, "convergence_fixed_objective_slope_max", 0.001
            )):
                reasons.append("fixed_objective_slope")
            if abs(stats["fixed_objective_half_delta"]) > float(getattr(
                self.args, "convergence_fixed_objective_half_delta_max", 0.01
            )):
                reasons.append("fixed_objective_level")

            previous_geometry = _mean([row["fixed_geometry"] for row in recent[:half]])
            current_geometry = _mean([row["fixed_geometry"] for row in recent[half:]])
            relative_worsening = (
                (current_geometry - previous_geometry)
                / max(abs(previous_geometry), 1e-12)
            )
            stats["fixed_geometry_relative_worsening"] = relative_worsening
            if relative_worsening > float(getattr(
                self.args, "convergence_geometry_relative_worsening_max", 0.0025
            )):
                reasons.append("geometry_worsening")

        stable_now = not reasons
        self.stable_episodes = self.stable_episodes + 1 if stable_now else 0
        self.converged = self.stable_episodes >= self.patience
        event = {
            "enabled": True,
            "episode": episode,
            "converged": bool(self.converged),
            "stable_now": bool(stable_now),
            "stable_episodes": int(self.stable_episodes),
            "patience": int(self.patience),
            "window": int(self.window),
            "minimum_episodes": int(self.minimum_episodes),
            "reasons": reasons,
            "stats": stats,
        }
        self.last_event = event
        return event


def format_convergence_event(event):
    if not event.get("enabled", False):
        return "ConvergenceControl: enabled=False"
    stats = event.get("stats") or {}
    return (
        "ConvergenceStatus: "
        f"episode={int(event.get('episode', 0))}, "
        f"stable={bool(event.get('stable_now', False))}, "
        f"evidence={int(event.get('stable_episodes', 0))}/"
        f"{int(event.get('patience', 0))}, "
        f"converged={bool(event.get('converged', False))}, "
        f"reasons={','.join(event.get('reasons') or ('none',))}, "
        f"total_slope={stats.get('total_loss_slope', float('nan')):.6g}, "
        f"fixed_rd_slope={stats.get('fixed_objective_slope', float('nan')):.6g}, "
        f"geometry_rel_worsening={stats.get('fixed_geometry_relative_worsening', float('nan')):.6g}"
    )
