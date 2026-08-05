import math

from .scalar_utils import case_float



def is_fresh_actual(args, comp_debug):
    backend = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    has_actual = math.isfinite(
        case_float(comp_debug.get("actual_total_bit_percent", float("nan")), float("nan"))
    )
    not_fallback = not bool(comp_debug.get("actual_codec_fallback_to_proxy", False))
    not_disabled = not bool(comp_debug.get("actual_codec_disabled_during_train", False))
    if not has_actual or not not_fallback:
        return False
    if bool(comp_debug.get("actual_value_is_fresh", False)):
        return True
    if backend.endswith("_surrogate"):
        return bool(comp_debug.get("teacher_refresh", False))
    if "_actual" in backend:
        return not_disabled and not bool(comp_debug.get("actual_codec_skipped_by_interval", False))
    return False
