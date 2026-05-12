from __future__ import annotations

import numpy as np


def point_to_point_mse(ref_to_dec_sq: np.ndarray, dec_to_ref_sq: np.ndarray) -> float:
    if ref_to_dec_sq.size == 0 or dec_to_ref_sq.size == 0:
        return float("nan")
    return float(max(np.mean(ref_to_dec_sq), np.mean(dec_to_ref_sq)))
