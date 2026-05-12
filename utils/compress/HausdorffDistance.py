from __future__ import annotations

import numpy as np


def hausdorff_distance(ref_to_dec: np.ndarray, dec_to_ref: np.ndarray) -> float:
    if ref_to_dec.size == 0 or dec_to_ref.size == 0:
        return float("nan")
    return float(max(np.max(ref_to_dec), np.max(dec_to_ref)))
