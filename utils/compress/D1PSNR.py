from __future__ import annotations

import math


def d1_psnr(point_to_point_mse_value: float, peak: float) -> float:
    if point_to_point_mse_value <= 0.0:
        return float("inf")
    peak = max(float(peak), 1e-12)
    return float(10.0 * math.log10((peak * peak) / float(point_to_point_mse_value)))
