import math


def sparsepcgc_effective_edit_record_bit_scale(args):
    base_scale = max(float(getattr(args, "sparsepcgc_edit_record_bit_scale", 1.0)), 0.0)
    if base_scale <= 0.0:
        return 0.0
    if not bool(getattr(args, "sparsepcgc_edit_record_train_curriculum", True)):
        return base_scale

    # The edit record is part of the final codec accounting, but applying the
    # full cost from step 0 makes all small raw SparsePCGC improvements look bad
    # and collapses the actuator to no-op.  During train.py, _global_train_step
    # is set on args; evaluation/test code normally has no such field and keeps
    # the full accounting scale.
    if not hasattr(args, "_global_train_step"):
        return base_scale

    warmup_steps = max(int(getattr(args, "sparsepcgc_edit_record_train_warmup_steps", 3000)), 1)
    step = max(int(getattr(args, "_global_train_step", 0)), 0)
    phase = min(float(step) / float(warmup_steps), 1.0)
    start = max(float(getattr(args, "sparsepcgc_edit_record_train_start_scale", 0.0)), 0.0)
    end = max(float(getattr(args, "sparsepcgc_edit_record_train_end_scale", 1.0)), 0.0)
    scale = start + (end - start) * phase
    if not math.isfinite(scale):
        scale = end
    return base_scale * max(scale, 0.0)
