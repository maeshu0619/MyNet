import math

import torch

from .utils import named_trainable_child_modules, trainable_parameters


def _scalar_loss(value):
    if not torch.is_tensor(value):
        return None
    if value.numel() == 1:
        return value.reshape(())
    return value.mean()


def _grad_stats_from_list(grads, param_count):
    total_sq = 0.0
    total_abs = 0.0
    total_numel = 0
    max_abs = 0.0
    active = 0
    none_count = 0
    finite = True
    for grad in grads:
        if grad is None:
            none_count += 1
            continue
        grad_det = grad.detach()
        finite = finite and bool(torch.isfinite(grad_det).all().item())
        grad_float = grad_det.float()
        grad_abs = grad_float.abs()
        total_sq += float(grad_float.pow(2).sum().detach().cpu())
        total_abs += float(grad_abs.sum().detach().cpu())
        total_numel += int(grad_float.numel())
        max_abs = max(max_abs, float(grad_abs.max().detach().cpu()))
        active += 1
    norm = total_sq ** 0.5
    if param_count <= 0:
        status = "no_trainable_params"
    elif not finite:
        status = "nonfinite_grad"
    elif active <= 0 and none_count > 0:
        status = "all_grad_none"
    elif active <= 0:
        status = "no_grad"
    elif norm <= 0.0:
        status = "zero_grad"
    else:
        status = "ok"
    return {
        "grad_norm": norm,
        "grad_max": max_abs,
        "grad_mean": (total_abs / float(total_numel)) if total_numel > 0 else 0.0,
        "grad_rms": (total_sq / float(total_numel)) ** 0.5 if total_numel > 0 else 0.0,
        "active_grad_param_count": active,
        "none_grad_param_count": none_count,
        "finite_grad": finite,
        "grad_status": status,
        "grad_available": status == "ok",
    }


def should_run_loss_grad_probe(args, global_step):
    if not bool(getattr(args, "loss_grad_probe_enabled", False)):
        return False
    interval = max(int(getattr(args, "loss_grad_probe_interval", 0)), 0)
    return interval > 0 and ((int(global_step) + 1) % interval) == 0


def build_loss_grad_probe_rows(args, model, loss_items, *, global_step, episode, epoch, step, stage):
    if not should_run_loss_grad_probe(args, global_step):
        return []
    base_model = model.module if hasattr(model, "module") else model
    modules = list(named_trainable_child_modules(base_model))
    rows = []
    for loss_name, raw_loss in loss_items:
        loss_value = _scalar_loss(raw_loss)
        loss_finite = bool(loss_value is not None and torch.isfinite(loss_value.detach()).all().item())
        loss_requires_grad = bool(loss_value is not None and loss_value.requires_grad)
        for module_name, module in modules:
            params = trainable_parameters(module)
            row = {
                "global_step": int(global_step) + 1,
                "episode": int(episode) + 1,
                "epoch": int(epoch) + 1,
                "step": int(step) + 1,
                "stage": str(stage),
                "loss_name": str(loss_name),
                "module_name": str(module_name),
                "loss_value": None if loss_value is None else float(loss_value.detach().float().mean().cpu()),
                "loss_requires_grad": bool(loss_requires_grad),
                "grad_norm": 0.0,
                "grad_max": 0.0,
                "grad_mean": 0.0,
                "grad_rms": 0.0,
                "active_grad_param_count": 0,
                "none_grad_param_count": len(params),
                "finite_grad": False,
                "grad_status": "loss_missing",
                "grad_available": False,
                "probe_skipped_reason": "",
            }
            if module is None:
                row["grad_status"] = "module_missing"
                row["probe_skipped_reason"] = "module_missing"
                rows.append(row)
                continue
            if not params:
                row["grad_status"] = "no_trainable_params"
                row["probe_skipped_reason"] = "no_trainable_params"
                rows.append(row)
                continue
            if loss_value is None:
                row["probe_skipped_reason"] = "loss_missing"
                rows.append(row)
                continue
            if not loss_finite:
                row["grad_status"] = "loss_nonfinite"
                row["probe_skipped_reason"] = "loss_nonfinite"
                rows.append(row)
                continue
            if not loss_requires_grad:
                row["grad_status"] = "loss_no_grad"
                row["probe_skipped_reason"] = "loss_no_grad"
                rows.append(row)
                continue
            try:
                grads = torch.autograd.grad(
                    loss_value,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=False,
                )
                row.update(_grad_stats_from_list(grads, len(params)))
                row["probe_skipped_reason"] = ""
            except RuntimeError as exc:
                row["grad_status"] = "probe_error"
                row["probe_skipped_reason"] = str(exc).splitlines()[0][:240]
            rows.append(row)
    return rows


def summarize_loss_grad_probe_rows(rows):
    if not rows:
        return "LossGradProbe: no rows"
    ok_rows = sum(1 for row in rows if row.get("grad_status") == "ok")
    finite_rows = sum(1 for row in rows if bool(row.get("finite_grad", False)))
    max_norm = 0.0
    for row in rows:
        value = row.get("grad_norm", 0.0)
        if value is not None and math.isfinite(float(value)):
            max_norm = max(max_norm, float(value))
    return f"LossGradProbe: rows={len(rows)}, ok={ok_rows}, finite={finite_rows}, max_norm={max_norm:.3e}"
