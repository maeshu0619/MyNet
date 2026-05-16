# models/utils/training/optim_amp.py

import torch
import torch.optim as optim

from models.utils.data.dataset import clear_ply_cache
from models.utils.training.utils import (
    _resolve_amp_dtype,
    _use_memory_safe_loader_workers,
)


def split_trainable_params(model, args):
    # encoderは固定する前提なので除外する。
    if args.deform:
        deform_params = [
            p for n, p in model.named_parameters()
            if (("disp_module" in n) or ("actuator" in n)) and p.requires_grad
        ]
        other_params = [
            p for n, p in model.named_parameters()
            if ("disp_module" not in n)
            and ("actuator" not in n)
            and ("encoder" not in n)
            and p.requires_grad
        ]
    else:
        deform_params = []
        other_params = [
            p for n, p in model.named_parameters()
            if ("encoder" not in n) and p.requires_grad
        ]

    return other_params, deform_params


def build_optimizer_and_scheduler(model, args, writer):
    other_params, deform_params = split_trainable_params(model, args)

    num_enc_trainable = sum(p.requires_grad for p in model.encoder.parameters())
    writer.write(f"Trainable encoder params: {num_enc_trainable} (should be 0)")

    assert args.optim in ["adam", "sgd"]

    if args.optim == "adam":
        optimizer = optim.Adam(
            [
                {"params": other_params},
                {"params": deform_params, "lr": args.lr * 0.1},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        args.lr = args.lr * 100
        optimizer = optim.SGD(
            [
                {"params": other_params},
                {"params": deform_params, "lr": args.lr * 0.1},
            ],
            lr=args.lr,
        )

    scheduler_steplr = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step,
        gamma=args.gamma,
    )

    return optimizer, scheduler_steplr


def setup_amp(model, args, writer):
    use_cuda = next(model.parameters()).is_cuda
    use_amp = bool(use_cuda and getattr(args, "use_amp", False))

    amp_dtype = _resolve_amp_dtype(args, use_cuda) if use_amp else torch.float16
    amp_scaler_enabled = bool(use_amp and amp_dtype == torch.float16)

    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_scaler_enabled,
        init_scale=float(getattr(args, "amp_init_scale", 1.0)),
    )

    amp_overflow_patience = max(
        int(getattr(args, "amp_overflow_patience", 2)),
        1,
    )

    writer.write(
        f"AMP: {'enabled' if use_amp else 'disabled'}"
        + (
            f" ({'bf16' if amp_dtype == torch.bfloat16 else 'fp16'})"
            if use_amp
            else ""
        )
    )

    return {
        "use_cuda": use_cuda,
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
        "amp_scaler_enabled": amp_scaler_enabled,
        "scaler": scaler,
        "amp_overflow_patience": amp_overflow_patience,
        "consecutive_amp_skips": 0,
    }


def build_loader_kwargs(args, model, writer, use_cuda):
    loader_num_workers = _use_memory_safe_loader_workers(args, model, writer)

    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=loader_num_workers,
        pin_memory=bool(use_cuda and args.pin_memory),
    )

    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)

        if bool(getattr(args, "clear_main_ply_cache_for_workers", True)):
            clear_ply_cache()
            writer.write(
                "Main-process PLY cache was cleared before worker DataLoaders "
                "to avoid duplicated CPU memory."
            )

    return loader_kwargs