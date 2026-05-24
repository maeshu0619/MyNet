# models/utils/training/checkpointing.py

import math
import os
import torch

from ..surrogate.checkpoint import maybe_save_best_surrogate_registry, surrogate_state_payload


def _finite_float(value, default=None):
    if torch.is_tensor(value):
        try:
            value = float(value.detach().cpu())
        except Exception:
            return default
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def surrogate_sidecar_filename(filename):
    stem, ext = os.path.splitext(str(filename))
    if not ext:
        ext = ".pth"
    return f"{stem}_surrogate{ext}"


def _save_surrogate_state(loss, ckpt_dir, filename):
    if loss is None:
        return None
    surrogate = getattr(loss, "compression_surrogate", None)
    if surrogate is None:
        return None
    loss_args = getattr(loss, "args", None)
    payload = surrogate_state_payload(loss_args if loss_args is not None else object(), loss, source=f"checkpoint_sidecar:{filename}") # 通常Checkpoint横にもCPU化済みSurrogate stateを保存する
    if payload is None:
        return None
    path = os.path.join(ckpt_dir, surrogate_sidecar_filename(filename))
    torch.save(payload, path)
    return path


def _save_state_dict(model, ckpt_dir, filename, loss=None):
    path = os.path.join(ckpt_dir, filename)
    torch.save(model.state_dict(), path)
    _save_surrogate_state(loss, ckpt_dir, filename)
    return path


def _ensure_best_trackers(best_trackers, best_loss):
    if best_trackers is None:
        best_trackers = {}
    best_trackers.setdefault("legacy_loss", _finite_float(best_loss, float("inf")))
    best_trackers.setdefault("loss_by_stage", {})
    best_trackers.setdefault("actual_candidate", float("inf"))
    best_trackers.setdefault("actual_improved", float("inf"))
    best_trackers.setdefault("actual_by_stage", {})
    best_trackers.setdefault("has_actual_candidate", False)
    best_trackers.setdefault("has_actual_improved", False)
    best_trackers.setdefault("best_pth_source", None)
    best_trackers.setdefault("surrogate_best_metric", float("inf"))
    best_trackers.setdefault("surrogate_best_metric_name", None)
    return best_trackers


def _is_actual_backend(args):
    backend = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    return backend.endswith("_surrogate") or "_actual" in backend


def _format_metric(value):
    value = _finite_float(value, None)
    return "n/a" if value is None else f"{value:.6f}"


def save_episode_checkpoint(
    model,
    ckpt_dir,
    plot,
    writer,
    episode,
    best_loss,
    *,
    args=None,
    stage=None,
    checkpoint_metrics=None,
    best_trackers=None,
    loss=None,
):
    # 既存仕様を維持するため、保存名は train.py の元コードと同じにする
    model_name = f"{episode}.pth"
    model_path = _save_state_dict(model, ckpt_dir, model_name, loss=loss)

    checkpoint_metrics = checkpoint_metrics or {}
    best_trackers = _ensure_best_trackers(best_trackers, best_loss)
    current_loss = _finite_float(checkpoint_metrics.get("total_loss"), None)
    if current_loss is None:
        current_loss = plot.epi_loss_return()
    stage_name = str(stage or checkpoint_metrics.get("stage") or "unknown").strip().lower() or "unknown"
    actual_backend = _is_actual_backend(args) if args is not None else False
    checkpoint_updates = []
    not_updated_reasons = []

    if current_loss < best_trackers["legacy_loss"]:
        best_trackers["legacy_loss"] = current_loss
        best_loss = current_loss
        model_path = _save_state_dict(model, ckpt_dir, "best_loss_legacy.pth", loss=loss)
        checkpoint_updates.append("best_loss_legacy")
        writer.write(
            f"New legacy loss best at episode {episode + 1}, "
            f"avg_epi_loss={current_loss:.6f}"
        )
        if not actual_backend:
            model_path = _save_state_dict(model, ckpt_dir, "best.pth", loss=loss)
            best_trackers["best_pth_source"] = "best_loss_legacy"

    loss_by_stage = best_trackers["loss_by_stage"]
    stage_loss_best = _finite_float(loss_by_stage.get(stage_name), float("inf"))
    if current_loss < stage_loss_best:
        loss_by_stage[stage_name] = current_loss
        filename = f"best_loss_{stage_name}.pth"
        model_path = _save_state_dict(model, ckpt_dir, filename, loss=loss)
        checkpoint_updates.append(filename.replace(".pth", ""))
        writer.write(
            f"New {stage_name} loss best at episode {episode + 1}, "
            f"avg_epi_loss={current_loss:.6f}, path={filename}"
        )
        if (not actual_backend) and stage_name == "joint":
            model_path = _save_state_dict(model, ckpt_dir, "best.pth", loss=loss)
            best_trackers["best_pth_source"] = filename

    actual_delta = _finite_float(checkpoint_metrics.get("fresh_actual_delta"), None)
    fresh_count = int(checkpoint_metrics.get("fresh_actual_count") or 0)
    cached_count = int(checkpoint_metrics.get("cached_actual_count") or 0)
    geometry_ok = bool(checkpoint_metrics.get("geometry_ok", False))
    safety_ok = bool(checkpoint_metrics.get("safety_ok", False))
    if actual_backend and actual_delta is not None and fresh_count > 0:
        if actual_delta < best_trackers["actual_candidate"]:
            best_trackers["actual_candidate"] = actual_delta
            best_trackers["has_actual_candidate"] = True
            model_path = _save_state_dict(model, ckpt_dir, "best_actual_delta_candidate.pth", loss=loss)
            checkpoint_updates.append("best_actual_delta_candidate")
            writer.write(
                f"New actual-delta candidate at episode {episode + 1}, "
                f"fresh_actual_delta={actual_delta:.6f}, fresh_count={fresh_count}, "
                f"geom_ok={geometry_ok}, safety_ok={safety_ok}"
            )
            if not best_trackers["has_actual_improved"]:
                model_path = _save_state_dict(model, ckpt_dir, "best.pth", loss=loss)
                best_trackers["best_pth_source"] = "best_actual_delta_candidate"
        else:
            not_updated_reasons.append("actual_not_improved")

        actual_by_stage = best_trackers["actual_by_stage"]
        stage_actual_best = _finite_float(actual_by_stage.get(stage_name), float("inf"))
        if actual_delta < stage_actual_best:
            actual_by_stage[stage_name] = actual_delta
            filename = f"best_actual_delta_{stage_name}.pth"
            model_path = _save_state_dict(model, ckpt_dir, filename, loss=loss)
            checkpoint_updates.append(filename.replace(".pth", ""))
            writer.write(
                f"New {stage_name} actual-delta best at episode {episode + 1}, "
                f"fresh_actual_delta={actual_delta:.6f}, path={filename}"
            )

        if actual_delta < 0.0 and geometry_ok and safety_ok and actual_delta < best_trackers["actual_improved"]:
            best_trackers["actual_improved"] = actual_delta
            best_trackers["has_actual_improved"] = True
            model_path = _save_state_dict(model, ckpt_dir, "best_actual_delta_improved.pth", loss=loss)
            model_path = _save_state_dict(model, ckpt_dir, "best.pth", loss=loss)
            best_trackers["best_pth_source"] = "best_actual_delta_improved"
            checkpoint_updates.append("best_actual_delta_improved")
            writer.write(
                f"New improved actual-delta best at episode {episode + 1}, "
                f"fresh_actual_delta={actual_delta:.6f}, fresh_count={fresh_count}, "
                "path=best_actual_delta_improved.pth and best.pth"
            )
        else:
            if actual_delta >= 0.0 or actual_delta >= best_trackers["actual_improved"]:
                not_updated_reasons.append("actual_not_improved")
            if not geometry_ok:
                not_updated_reasons.append("geometry_gate_failed")
            if not safety_ok:
                not_updated_reasons.append("safety_gate_failed")
    elif actual_backend:
        if actual_delta is None:
            not_updated_reasons.append("nonfinite_metric")
        if fresh_count <= 0:
            not_updated_reasons.append("cached_actual_only" if cached_count > 0 else "no_fresh_actual")
        fallback_path = None
        if not best_trackers["has_actual_candidate"]:
            joint_loss = _finite_float(loss_by_stage.get("joint"), None)
            if joint_loss is not None and stage_name == "joint" and current_loss <= joint_loss:
                fallback_path = "best_loss_joint.pth"
            elif best_trackers["legacy_loss"] == current_loss:
                fallback_path = "best_loss_legacy.pth"
        if fallback_path:
            model_path = _save_state_dict(model, ckpt_dir, "best.pth", loss=loss)
            best_trackers["best_pth_source"] = fallback_path
            checkpoint_updates.append("fallback")
        else:
            not_updated_reasons.append("fallback_not_allowed")
    if not checkpoint_updates and not not_updated_reasons:
        not_updated_reasons.append("actual_not_improved" if actual_backend else "loss_not_improved")
    surrogate_abs_error = _finite_float(checkpoint_metrics.get("surrogate_abs_bit_error"), None) # Surrogate保存判定用の平均abs errorを取り出す
    surrogate_registry_path = None # 共有Surrogate保存先を初期化する
    if loss is not None and surrogate_abs_error is not None:
        surrogate_registry_path = maybe_save_best_surrogate_registry( args, loss, best_trackers, writer, metric_name="surrogate_abs_bit_error", metric_value=surrogate_abs_error, source=f"episode_{episode + 1}") # Surrogateが改善した場合だけ指定形式の共有重みを保存する
        if surrogate_registry_path is not None:
            checkpoint_updates.append("surrogate_registry_best") # CheckpointSummaryにSurrogate保存更新を表示する

    writer.write(
        "CheckpointSummary: "
        f"episode={episode + 1}, stage={stage_name}, "
        f"loss={_format_metric(current_loss)}, "
        f"fresh_actual_delta={_format_metric(actual_delta)}, "
        f"fresh_count={fresh_count}, "
        f"geom_ok={geometry_ok}, safety_ok={safety_ok}, "
        f"best_pth_source={best_trackers.get('best_pth_source')}, "
        f"surrogate_best={_format_metric(best_trackers.get('surrogate_best_metric'))}, "
        f"updates={','.join(dict.fromkeys(checkpoint_updates)) or 'none'}, "
        f"not_updated_reasons={','.join(dict.fromkeys(not_updated_reasons)) or 'none'}"
    )

    return best_trackers["legacy_loss"], model_path, best_trackers
