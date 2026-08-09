import math
import os
import random

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy無しでもguard自体は動かす
    np = None

from .checkpointing import surrogate_sidecar_filename
from .lr_control import apply_optimizer_lr_floor, optimizer_lrs_safe


_RUNTIME_ARG_STATE_NAMES = (
    "_sparsepcgc_anchor_success_memory",
    "_sparsepcgc_amount_outcome_memory",
    "_sparsepcgc_full_cloud_sequence_amount_memory",
    "_sparsepcgc_full_cloud_sequence_baseline_memory",
    "_heuristic_guidance_network_residual_weight_current",
)


def update_network_autonomy_from_guard(args, guard_event):
    """固定validationが改善した時だけden6 Pool内のNetwork裁量を広げる。"""
    start = max(
        float(getattr(args, "heuristic_guidance_network_residual_weight", 0.05)),
        0.0,
    )
    maximum = max(
        float(getattr(args, "heuristic_guidance_network_residual_weight_max", 0.25)),
        start,
    )
    increment = max(
        float(getattr(args, "heuristic_guidance_network_residual_weight_increment", 0.025)),
        0.0,
    )
    current = min(
        max(
            float(getattr(
                args,
                "_heuristic_guidance_network_residual_weight_current",
                start,
            )),
            start,
        ),
        maximum,
    )
    action = str((guard_event or {}).get("action", "")).strip().lower()
    previous = current
    # new_bestは同一固定frameで圧縮性能が改善した証拠である。候補Actualを
    # 増やさず、次EpisodeからPool全体の再順位付け幅を一段だけ広げる。
    if action == "new_best":
        current = min(current + increment, maximum)
    elif action == "rollback":
        # 完全state restoreで保存時の裁量へ戻っている。念のため範囲だけ拘束する。
        current = min(max(current, start), maximum)
    setattr(args, "_heuristic_guidance_network_residual_weight_current", current)
    event = {
        "action": action or "none",
        "previous": float(previous),
        "current": float(current),
        "start": float(start),
        "maximum": float(maximum),
        "increment": float(increment),
        "changed": bool(abs(current - previous) > 1e-12),
    }
    if isinstance(guard_event, dict):
        guard_event["network_autonomy_previous"] = event["previous"]
        guard_event["network_autonomy_current"] = event["current"]
        guard_event["network_autonomy_changed"] = event["changed"]
    return event


def _training_state_path(model_path):
    stem, ext = os.path.splitext(str(model_path))
    return f"{stem}_training_state{ext or '.pth'}"


def _state_dict_or_none(value):
    state_dict = getattr(value, "state_dict", None)
    return state_dict() if callable(state_dict) else None


def _optimizer_state_to_parameter_device(optimizer):
    if optimizer is None:
        return
    parameter_device = None
    for group in optimizer.param_groups:
        for parameter in group.get("params", ()):
            if torch.is_tensor(parameter):
                parameter_device = parameter.device
                break
        if parameter_device is not None:
            break
    if parameter_device is None:
        return
    for state_values in optimizer.state.values():
        for key, value in list(state_values.items()):
            if torch.is_tensor(value):
                state_values[key] = value.to(device=parameter_device)


def _save_training_state(model_path, runtime_state, args):
    """Actual guardで巻き戻す学習状態をsidecarへ保存する。"""
    runtime_state = runtime_state or {}
    payload = {
        "version": 1,
        "main_optimizer": _state_dict_or_none(runtime_state.get("optimizer")),
        "main_scheduler": _state_dict_or_none(runtime_state.get("scheduler")),
        "main_scaler": _state_dict_or_none(runtime_state.get("scaler")),
        "emulator_optimizer": _state_dict_or_none(runtime_state.get("emulator_optimizer")),
        "emulator_scheduler": _state_dict_or_none(runtime_state.get("emulator_scheduler")),
        "emulator_scaler": _state_dict_or_none(runtime_state.get("emulator_scaler")),
        "runtime_args": {
            name: getattr(args, name)
            for name in _RUNTIME_ARG_STATE_NAMES
            if hasattr(args, name)
        },
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if np is not None:
        payload["numpy_rng_state"] = np.random.get_state()
    mutable_mappings = runtime_state.get("mutable_mappings", {})
    payload["mutable_mappings"] = {
        str(name): dict(value)
        for name, value in mutable_mappings.items()
        if isinstance(value, dict)
    }
    path = _training_state_path(model_path)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


def _load_training_state(model_path, runtime_state, args):
    """モデル重みと同じ時点のoptimizer等を復元し、部分rollbackを防ぐ。"""
    path = _training_state_path(model_path)
    if not os.path.exists(path):
        return False, path
    payload = torch.load(path, map_location="cpu")
    runtime_state = runtime_state or {}
    object_keys = (
        ("optimizer", "main_optimizer"),
        ("scheduler", "main_scheduler"),
        ("scaler", "main_scaler"),
        ("emulator_optimizer", "emulator_optimizer"),
        ("emulator_scheduler", "emulator_scheduler"),
        ("emulator_scaler", "emulator_scaler"),
    )
    for runtime_key, payload_key in object_keys:
        obj = runtime_state.get(runtime_key)
        state = payload.get(payload_key)
        load_state_dict = getattr(obj, "load_state_dict", None)
        if callable(load_state_dict) and isinstance(state, dict):
            load_state_dict(state)
            if "optimizer" in runtime_key:
                _optimizer_state_to_parameter_device(obj)
    for name, value in (payload.get("runtime_args") or {}).items():
        if name in _RUNTIME_ARG_STATE_NAMES:
            setattr(args, name, value)
    for name, value in (payload.get("mutable_mappings") or {}).items():
        target = (runtime_state.get("mutable_mappings") or {}).get(name)
        if isinstance(target, dict) and isinstance(value, dict):
            target.clear()
            target.update(value)
    torch_rng_state = payload.get("torch_rng_state")
    if torch.is_tensor(torch_rng_state):
        torch.set_rng_state(torch_rng_state)
    cuda_rng_state_all = payload.get("cuda_rng_state_all")
    if torch.cuda.is_available() and isinstance(cuda_rng_state_all, (list, tuple)):
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
    python_rng_state = payload.get("python_rng_state")
    if python_rng_state is not None:
        random.setstate(python_rng_state)
    numpy_rng_state = payload.get("numpy_rng_state")
    if np is not None and numpy_rng_state is not None:
        np.random.set_state(numpy_rng_state)
    return True, path


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
    runtime_state=None,
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
    if bool(getattr(args, "actual_guard_require_fixed_validation", True)) and actual_source != "full_cloud":
        reason = "fixed_full_cloud_validation_required"
        if writer is not None and hasattr(writer, "write"):
            writer.write(
                "ActualCompressionGuard: skipped "
                f"episode={episode + 1}, reason={reason}, actual_source={actual_source}"
            )
        return {"action": "skipped", "reason": reason, "actual_source": actual_source}
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
        training_state_path = _save_training_state(episode_path, runtime_state, args)
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
            "training_state_path": training_state_path,
            "training_state_saved": True,
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
    training_state_restored = False
    training_state_path = _training_state_path(best_path) if best_path else None
    if bool(getattr(args, "actual_guard_restore_best", True)) and best_path and os.path.exists(best_path):
        training_state_restored, training_state_path = _load_training_state(
            best_path, runtime_state, args
        )
        if training_state_restored or not bool(
            getattr(args, "actual_guard_require_full_state_restore", True)
        ):
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
            "rollback_reason": "fixed_validation_actual_worse_than_guard_tolerance",
            "restored": restored,
            "surrogate_restored": surrogate_restored,
            "training_state_restored": training_state_restored,
            "training_state_path": training_state_path,
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
            f"training_state_restored={training_state_restored}, "
            f"guard_lr_changed={event['guard_lr_changed']}, new_lrs={new_lrs}"
        )
    return event
