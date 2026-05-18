def loss_mode(args):
    return str(getattr(args, "loss_mode", "legacy_total")).strip().lower()


def lossmode(args):
    return loss_mode(args)

def sparsepcgc_add_experiment_active(args):
    codec = str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "")
    if codec != "sparsepcgc":
        return False
    if not bool(getattr(args, "sparsepcgc_enable_add_experiment", False)):
        return False
    if bool(getattr(args, "sparsepcgc_add_only_when_compression_primary", True)):
        return loss_mode(args) == "compression_primary"
    return True


def sparsepcgc_add_control_status(args):
    codec = str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "")
    experiment_active = sparsepcgc_add_experiment_active(args)
    disable_add = bool(getattr(args, "sparsepcgc_disable_add", True))
    loss = loss_mode(args)
    if codec != "sparsepcgc":
        reason = "not_sparsepcgc"
        add_allowed = not disable_add
    elif experiment_active:
        reason = "sparsepcgc_add_experiment_active"
        add_allowed = True
    elif disable_add:
        reason = "sparsepcgc_disable_add_true"
        add_allowed = False
    else:
        reason = "sparsepcgc_disable_add_false"
        add_allowed = True
    return {
        "codec": codec,
        "loss_mode": loss,
        "sparsepcgc_disable_add": disable_add,
        "sparsepcgc_enable_add_experiment": bool(getattr(args, "sparsepcgc_enable_add_experiment", False)),
        "sparsepcgc_add_only_when_compression_primary": bool(
            getattr(args, "sparsepcgc_add_only_when_compression_primary", True)
        ),
        "experiment_active": bool(experiment_active),
        "add_allowed_by_sparsepcgc_control": bool(add_allowed),
        "reason": reason,
        "target_add_ratio": float(getattr(args, "target_add_ratio", 0.0)),
        "max_add_ratio": float(getattr(args, "max_add_ratio", 0.0)),
        "sparsepcgc_add_target_ratio": float(getattr(args, "sparsepcgc_add_target_ratio", 0.0)),
        "sparsepcgc_add_max_ratio": float(getattr(args, "sparsepcgc_add_max_ratio", 0.0)),
    }


def add_warmup_factor(args):
    steps = max(int(getattr(args, "sparsepcgc_add_warmup_steps", 0)), 0)
    if steps <= 0:
        return 1.0
    step = int(getattr(args, "_global_train_step", 0)) + 1
    return min(1.0, max(0.0, float(step) / float(steps)))
