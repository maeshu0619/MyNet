import math
import os

import torch

from .checkpointing import surrogate_sidecar_filename


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


def _decay_optimizer_lrs(optimizer, factor):
    if optimizer is None:
        return []
    factor = min(max(float(factor), 0.0), 1.0)
    lrs = []
    for group in optimizer.param_groups:
        group["lr"] = float(group.get("lr", 0.0)) * factor
        lrs.append(float(group["lr"]))
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
    fresh_count = int(checkpoint_metrics.get("fresh_actual_count") or 0)
    min_fresh = max(int(getattr(args, "actual_guard_min_fresh", 1)), 1)
    if fresh_count < min_fresh:
        return None

    fresh_delta = _finite_float(checkpoint_metrics.get("fresh_actual_delta"), None)
    if fresh_delta is None:
        return None

    guard_state.setdefault("best_delta", float("inf"))
    guard_state.setdefault("best_path", None)
    guard_state.setdefault("bad_count", 0)

    episode_path = os.path.join(ckpt_dir, f"{episode}.pth")
    eps = max(float(getattr(args, "actual_guard_improvement_epsilon", 1e-6)), 0.0)
    if fresh_delta < float(guard_state["best_delta"]) - eps:
        guard_state["best_delta"] = fresh_delta
        guard_state["best_path"] = episode_path
        guard_state["bad_count"] = 0
        message = (
            "ActualCompressionGuard: new_best "
            f"episode={episode + 1}, fresh_actual_delta={fresh_delta:.6f}, path={episode_path}"
        )
        if writer is not None and hasattr(writer, "write"):
            writer.write(message)
        return {
            "action": "new_best",
            "fresh_actual_delta": fresh_delta,
            "best_delta": fresh_delta,
            "best_path": episode_path,
            "bad_count": 0,
        }

    tolerance = max(float(getattr(args, "actual_guard_tolerance", 0.25)), 0.0)
    best_delta = float(guard_state["best_delta"])
    if fresh_delta <= best_delta + tolerance:
        guard_state["bad_count"] = 0
        return {
            "action": "within_tolerance",
            "fresh_actual_delta": fresh_delta,
            "best_delta": best_delta,
            "bad_count": 0,
        }

    guard_state["bad_count"] = int(guard_state.get("bad_count", 0)) + 1
    patience = max(int(getattr(args, "actual_guard_patience", 2)), 1)
    event = {
        "action": "worse",
        "fresh_actual_delta": fresh_delta,
        "best_delta": best_delta,
        "bad_count": int(guard_state["bad_count"]),
        "patience": patience,
    }
    if guard_state["bad_count"] < patience:
        if writer is not None and hasattr(writer, "write"):
            writer.write(
                "ActualCompressionGuard: worse "
                f"episode={episode + 1}, fresh_actual_delta={fresh_delta:.6f}, "
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

    new_lrs = _decay_optimizer_lrs(optimizer, float(getattr(args, "actual_guard_lr_decay", 0.5)))
    guard_state["bad_count"] = 0
    event.update(
        {
            "action": "rollback" if restored else "lr_decay",
            "restored": restored,
            "surrogate_restored": surrogate_restored,
            "restore_path": best_path,
            "new_lrs": new_lrs,
        }
    )
    if writer is not None and hasattr(writer, "write"):
        writer.write(
            "ActualCompressionGuard: "
            f"{event['action']} episode={episode + 1}, fresh_actual_delta={fresh_delta:.6f}, "
            f"best={best_delta:.6f}, restored={restored}, surrogate_restored={surrogate_restored}, "
            f"new_lrs={new_lrs}"
        )
    return event
