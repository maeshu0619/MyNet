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


def add_warmup_factor(args):
    steps = max(int(getattr(args, "sparsepcgc_add_warmup_steps", 0)), 0)
    if steps <= 0:
        return 1.0
    step = int(getattr(args, "_global_train_step", 0)) + 1
    return min(1.0, max(0.0, float(step) / float(steps)))
