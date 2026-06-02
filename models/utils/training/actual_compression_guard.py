import math
import os

import torch

from .checkpointing import surrogate_sidecar_filename
from .lr_control import apply_optimizer_lr_floor, optimizer_lrs_safe


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


def _is_actual_backend(args):
    backend = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    return backend.endswith("_surrogate") or "_actual" in backend


def _selected_actual_metric(checkpoint_metrics):
    metrics = checkpoint_metrics or {}
    source = str(metrics.get("checkpoint_actual_source") or "fresh").strip().lower() or "fresh"
    delta = _finite_float(metrics.get("checkpoint_actual_delta"), None)
    count = int(metrics.get("checkpoint_actual_count") or 0)
    if delta is None and source == "full_cloud":
        delta = _finite_float(metrics.get("full_cloud_actual_delta"), None)
        count = int(metrics.get("full_cloud_actual_count") or 0)
    if delta is None:
        source = "fresh"
        delta = _finite_float(metrics.get("fresh_actual_delta"), None)
        count = int(metrics.get("fresh_actual_count") or 0)
    return source, delta, count


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            state = payload.get(key)
            if isinstance(state, dict):
                return state
    return payload


def _load_model_state(model, path):
    payload = torch.load(path, map_location="cpu")
    state = _extract_state_dict(payload)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state, strict=False)


def _load_surrogate_state(loss, model_path):
    if loss is None:
        return False
    sidecar_path = os.path.join(os.path.dirname(model_path), surrogate_sidecar_filename(os.path.basename(model_path)))
    if not os.path.exists(sidecar_path):
        return False
    payload = torch.load(sidecar_path, map_location="cpu")
    surrogate = getattr(loss, "compression_surrogate", None)
    if surrogate is None:
        return False
    state = payload.get("compression_surrogate_state_dict") or payload.get("compression_surrogate")
    if isinstance(state, dict):
        surrogate.load_state_dict(state, strict=False)
    optimizer = getattr(loss, "surrogate_optimizer", None)
    opt_state = payload.get("surrogate_optimizer_state_dict")
    if optimizer is not None and isinstance(opt_state, dict):
        optimizer.load_state_dict(opt_state)
        device = next(surrogate.parameters()).device
        for state_values in optimizer.state.values():
            for key, value in list(state_values.items()):
                if torch.is_tensor(value):
                    state_values[key] = value.to(device=device)
    if "surrogate_step" in payload:
        loss._surrogate_step = int(payload.get("surrogate_step") or 0)
    return True


def _decay_optimizer_lrs(optimizer, factor, args=None, writer=None, global_step=None):
    if optimizer is None:
        return []
    factor = min(max(float(factor), 0.0), 1.0)
    lrs = []
    for group in optimizer.param_groups:
        group["lr"] = float(group.get("lr", 0.0)) * factor
        lrs.append(float(group["lr"]))
    if args is not None:
        apply_optimizer_lr_floor(
            optimizer,
            args,
            label="main",
            writer=writer,
            global_step=global_step,
            reason="actual_compression_guard",
        )
        lrs = optimizer_lrs_safe(optimizer)
    return lrs


def _clear_loss_caches(loss):
    if loss is None:
        return
    for name in ("actual_gt_cache", "surrogate_target_cache", "surrogate_replay_buffer"):
        cache = getattr(loss, name, None)
        if hasattr(cache, "clear"):
            cache.clear()
    if hasattr(loss, "last_surrogate_target_entry"):
        loss.last_surrogate_target_entry = None


def apply_actual_compression_guard(
    *,
    args,
    model,
    loss,
    optimizer,
    writer,
    guard_state,
    checkpoint_metrics,
    ckpt_dir,
    episode,
):
    if not bool(getattr(args, "actual_compression_guard", True)):
        return None
    if not _is_actual_backend(args):
        return None
    if not bool(checkpoint_metrics.get("checkpoint_eligible", True)):
        reason = str(checkpoint_metrics.get("checkpoint_ineligible_reason") or "checkpoint_ineligible")
        if writer is not None and hasattr(writer, "write"):
            writer.write(
                "ActualCompressionGuard: skipped "
                f"episode={episode + 1}, reason={reason}"
            )
        return {
            "action": "skipped",
            "reason": reason,
            "checkpoint_eligible": False,
        }

    actual_source, actual_delta, actual_count = _selected_actual_metric(checkpoint_metrics)
    min_count = (
        max(int(getattr(args, "checkpoint_full_cloud_min_count", 1)), 0)
        if actual_source == "full_cloud"
        else max(int(getattr(args, "actual_guard_min_fresh", 1)), 1)
    )
    if actual_count < min_count:
        return None

    if actual_delta is None:
        return None

    guard_state.setdefault("best_delta", float("inf"))
    guard_state.setdefault("best_path", None)
    guard_state.setdefault("bad_count", 0)

    episode_path = os.path.join(ckpt_dir, f"{episode}.pth")
    eps = max(float(getattr(args, "actual_guard_improvement_epsilon", 1e-6)), 0.0)
    if actual_delta < float(guard_state["best_delta"]) - eps:
        guard_state["best_delta"] = actual_delta
        guard_state["best_path"] = episode_path
        guard_state["bad_count"] = 0
        message = (
            "ActualCompressionGuard: new_best "
            f"episode={episode + 1}, actual_source={actual_source}, "
            f"actual_delta={actual_delta:.6f}, path={episode_path}"
        )
        if writer is not None and hasattr(writer, "write"):
            writer.write(message)
        return {
            "action": "new_best",
            "fresh_actual_delta": actual_delta,
            "actual_delta": actual_delta,
            "actual_source": actual_source,
            "best_delta": actual_delta,
            "best_path": episode_path,
            "bad_count": 0,
        }

    tolerance = max(float(getattr(args, "actual_guard_tolerance", 0.25)), 0.0)
    best_delta = float(guard_state["best_delta"])
    if actual_delta <= best_delta + tolerance:
        guard_state["bad_count"] = 0
        return {
            "action": "within_tolerance",
            "fresh_actual_delta": actual_delta,
            "actual_delta": actual_delta,
            "actual_source": actual_source,
            "best_delta": best_delta,
            "bad_count": 0,
        }

    guard_state["bad_count"] = int(guard_state.get("bad_count", 0)) + 1
    patience = max(int(getattr(args, "actual_guard_patience", 2)), 1)
    event = {
        "action": "worse",
        "fresh_actual_delta": actual_delta,
        "actual_delta": actual_delta,
        "actual_source": actual_source,
        "best_delta": best_delta,
        "bad_count": int(guard_state["bad_count"]),
        "patience": patience,
    }
    if guard_state["bad_count"] < patience:
        if writer is not None and hasattr(writer, "write"):
            writer.write(
                "ActualCompressionGuard: worse "
                f"episode={episode + 1}, actual_source={actual_source}, actual_delta={actual_delta:.6f}, "
                f"best={best_delta:.6f}, bad_count={guard_state['bad_count']}/{patience}"
            )
        return event

    best_path = guard_state.get("best_path")
    restored = False
    surrogate_restored = False
    if bool(getattr(args, "actual_guard_restore_best", True)) and best_path and os.path.exists(best_path):
        _load_model_state(model, best_path)
        surrogate_restored = _load_surrogate_state(loss, best_path)
        _clear_loss_caches(loss)
        clear_input_cache = getattr(model, "clear_input_cache", None)
        if callable(clear_input_cache):
            clear_input_cache()
        model.train()
        restored = True

    global_step = int(getattr(args, "_global_train_step", 0))
    old_lrs = optimizer_lrs_safe(optimizer)
    decay_lr = bool(getattr(args, "actual_guard_decay_lr", False))
    if decay_lr:
        new_lrs = _decay_optimizer_lrs(
            optimizer,
            float(getattr(args, "actual_guard_lr_decay", 0.5)),
            args=args,
            writer=writer,
            global_step=global_step,
        )
    else:
        apply_optimizer_lr_floor(
            optimizer,
            args,
            label="main",
            writer=writer,
            global_step=global_step,
            reason="actual_guard_no_decay",
        )
        new_lrs = optimizer_lrs_safe(optimizer)
    surrogate_floor_event = apply_optimizer_lr_floor(
        getattr(loss, "surrogate_optimizer", None),
        args,
        label="surrogate",
        writer=writer,
        global_step=global_step,
        reason="actual_compression_guard",
    )
    guard_state["bad_count"] = 0
    lr_floor_applied = any(after > before for before, after in zip(old_lrs, new_lrs)) if old_lrs and new_lrs else False
    guard_action = "rollback" if restored else ("lr_decay" if decay_lr else "guard_no_lr_decay")
    event.update(
        {
            "action": guard_action,
            "guard_rollback": bool(restored),
            "guard_lr_changed": bool(decay_lr and old_lrs != new_lrs),
            "lr_floor_applied": bool(lr_floor_applied),
            "rollback_reason": "fresh_actual_delta_worse_than_guard_tolerance",
            "restored": restored,
            "surrogate_restored": surrogate_restored,
            "restore_path": best_path,
            "old_lrs": old_lrs,
            "new_lrs": new_lrs,
            "surrogate_lrs": surrogate_floor_event.get("lr_after_floor", []),
            "surrogate_lr_floor_applied": bool(surrogate_floor_event.get("lr_floor_applied", False)),
            "actual_total_bit_percent_fresh": actual_delta,
            "checkpoint_actual_delta": actual_delta,
            "checkpoint_actual_source": actual_source,
        }
    )
    if writer is not None and hasattr(writer, "write"):
        writer.write(
            "ActualCompressionGuard: "
            f"{event['action']} episode={episode + 1}, actual_source={actual_source}, actual_delta={actual_delta:.6f}, "
            f"best={best_delta:.6f}, restored={restored}, surrogate_restored={surrogate_restored}, "
            f"guard_lr_changed={event['guard_lr_changed']}, new_lrs={new_lrs}"
        )
    return event
