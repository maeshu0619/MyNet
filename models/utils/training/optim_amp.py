# models/utils/training/optim_amp.py

import torch
import torch.optim as optim

from models.utils.data.dataset import clear_ply_cache
from models.utils.training.utils import (
    resolve_amp_dtype,
    use_memory_safe_loader_workers,
)


def split_trainable_params(model, args):
    # 通常はencoderを固定する。Single-Plan Stage 1だけは明示設定に従って含める。
    exclude_encoder = bool(getattr(args, "encoder_0grad", True))
    if args.deform:
        deform_params = [
            p for n, p in model.named_parameters()
            if (("disp_module" in n) or ("actuator" in n)) and p.requires_grad
        ]
        other_params = [
            p for n, p in model.named_parameters()
            if ("disp_module" not in n)
            and ("actuator" not in n)
            and (("encoder" not in n) or not exclude_encoder)
            and p.requires_grad
        ]
    else:
        deform_params = []
        other_params = [
            p for n, p in model.named_parameters()
            if (("encoder" not in n) or not exclude_encoder) and p.requires_grad
        ]

    return other_params, deform_params


def build_optimizer_and_scheduler(model, args, writer):
    other_params, deform_params = split_trainable_params(model, args)

    # K-policyは巨大な離散空間を探索するActorと、毎StepのActualへ素早く
    # 追従すべきCriticで適切な学習速度が異なるため、LR groupを分離する。
    named_trainable = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    k_critic_names = {
        name for name in named_trainable
        if "network_k_proposal_policy." in name
        and any(token in name for token in (
            ".critic.", ".critic_gain_head.", ".critic_geometry_head.",
            ".critic_interaction_head.", ".critic_uncertainty_head.",
        ))
    }
    k_actor_names = {
        name for name in named_trainable
        if "network_k_proposal_policy." in name and name not in k_critic_names
    }
    k_actor_params = [named_trainable[name] for name in sorted(k_actor_names)]
    k_critic_params = [named_trainable[name] for name in sorted(k_critic_names)]
    k_param_ids = {id(parameter) for parameter in (*k_actor_params, *k_critic_params)}
    other_params = [parameter for parameter in other_params if id(parameter) not in k_param_ids]
    deform_params = [parameter for parameter in deform_params if id(parameter) not in k_param_ids]

    single_names = {
        name for name in named_trainable
        if "single_plan_student." in name
    }
    single_params = [named_trainable[name] for name in sorted(single_names)]
    single_param_ids = {id(parameter) for parameter in single_params}
    other_params = [parameter for parameter in other_params if id(parameter) not in single_param_ids]
    deform_params = [parameter for parameter in deform_params if id(parameter) not in single_param_ids]
    separate_emulator = (
        str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
        == "ana_den6_online"
        and bool(getattr(args, "single_plan_shadow_distillation", True))
    )

    num_enc_trainable = sum(p.requires_grad for p in model.encoder.parameters())
    writer.write(
        f"Trainable encoder params: {num_enc_trainable} "
        f"(expected={'0' if bool(getattr(args, 'encoder_0grad', True)) else '>0'})"
    )

    assert args.optim in ["adam", "sgd"]

    if args.optim == "adam":
        groups = [
            {"params": other_params, "name": "main"},
            {"params": deform_params, "lr": args.lr * 0.1, "name": "deform"},
        ]
        if k_actor_params:
            groups.append({
                "params": k_actor_params,
                "lr": args.lr * float(getattr(args, "network_k_actor_lr_scale", 0.1)),
                "name": "network_k_actor",
            })
        if k_critic_params:
            groups.append({
                "params": k_critic_params,
                "lr": args.lr * float(getattr(args, "network_k_critic_lr_scale", 1.0)),
                "name": "network_k_critic",
            })
        if single_params and not separate_emulator:
            groups.append({
                "params": single_params,
                "lr": args.lr * float(getattr(args, "single_plan_student_lr_scale", 1.0)),
                "name": "single_plan_student",
            })
        optimizer = optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        args.lr = args.lr * 100
        groups = [
            {"params": other_params, "name": "main"},
            {"params": deform_params, "lr": args.lr * 0.1, "name": "deform"},
        ]
        if k_actor_params:
            groups.append({
                "params": k_actor_params,
                "lr": args.lr * float(getattr(args, "network_k_actor_lr_scale", 0.1)),
                "name": "network_k_actor",
            })
        if k_critic_params:
            groups.append({
                "params": k_critic_params,
                "lr": args.lr * float(getattr(args, "network_k_critic_lr_scale", 1.0)),
                "name": "network_k_critic",
            })
        if single_params and not separate_emulator:
            groups.append({
                "params": single_params,
                "lr": args.lr * float(getattr(args, "single_plan_student_lr_scale", 1.0)),
                "name": "single_plan_student",
            })
        optimizer = optim.SGD(
            groups,
            lr=args.lr,
        )

    scheduler_steplr = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step,
        gamma=args.gamma,
    )

    writer.write(
        "OptimizerGroups: "
        + ", ".join(
            f"{group.get('name', 'unnamed')}[lr={float(group['lr']):.6g},params={len(group['params'])}]"
            for group in optimizer.param_groups
        )
    )

    return optimizer, scheduler_steplr


def build_emulator_optimizer_and_scheduler(model, args, writer):
    """Exact主経路と共有しないEmulator専用optimizerを作る。"""
    mode = str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
    if mode != "ana_den6_online" or not bool(
        getattr(args, "single_plan_shadow_distillation", True)
    ):
        return None, None
    base_model = model.module if hasattr(model, "module") else model
    emulator = getattr(base_model, "single_plan_student", None)
    if emulator is None:
        raise RuntimeError("Fast Heuristic Emulatorが初期化されていない")
    parameters = [parameter for parameter in emulator.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Fast Heuristic Emulatorに学習可能parameterがない")
    lr = float(args.lr) * float(getattr(args, "single_plan_student_lr_scale", 1.0))
    if str(args.optim).strip().lower() == "adam":
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=float(args.weight_decay))
    else:
        optimizer = optim.SGD(parameters, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step,
        gamma=args.gamma,
    )
    writer.write(
        "FastHeuristicEmulatorOptimizer: "
        f"separate=True, lr={lr:.6g}, params={len(parameters)}"
    )
    return optimizer, scheduler


def setup_amp(model, args, writer):
    use_cuda = next(model.parameters()).is_cuda
    use_amp = bool(use_cuda and getattr(args, "use_amp", False))

    amp_dtype = resolve_amp_dtype(args, use_cuda) if use_amp else torch.float16
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
    loader_num_workers = use_memory_safe_loader_workers(args, model, writer)

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
