from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .nearest import deterministic_subsample


def earth_mover_distance(reference_points: np.ndarray, decoded_points: np.ndarray, max_points: int = 2048) -> float:
    reference_points = deterministic_subsample(reference_points, max_points)
    decoded_points = deterministic_subsample(decoded_points, max_points)
    count = min(reference_points.shape[0], decoded_points.shape[0])
    if count <= 0:
        return float("nan")
    reference_points = deterministic_subsample(reference_points, count)
    decoded_points = deterministic_subsample(decoded_points, count)
    if reference_points.shape[0] != count:
        reference_points = reference_points[:count]
    if decoded_points.shape[0] != count:
        decoded_points = decoded_points[:count]
    cost = cdist(reference_points, decoded_points, metric="euclidean")
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())
