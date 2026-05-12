from __future__ import annotations

import numpy as np


def normal_consistency(
    reference_normals: np.ndarray,
    decoded_normals: np.ndarray,
    ref_to_dec_indices: np.ndarray,
    dec_to_ref_indices: np.ndarray,
) -> float:
    if reference_normals.size == 0 or decoded_normals.size == 0:
        return float("nan")
    ref_score = np.abs(np.sum(reference_normals * decoded_normals[ref_to_dec_indices], axis=1))
    dec_score = np.abs(np.sum(decoded_normals * reference_normals[dec_to_ref_indices], axis=1))
    return float(0.5 * (np.mean(ref_score) + np.mean(dec_score)))
