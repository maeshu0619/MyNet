from __future__ import annotations

import numpy as np


def point_to_plane_mse(
    reference_points: np.ndarray,
    decoded_points: np.ndarray,
    reference_normals: np.ndarray,
    ref_to_dec_indices: np.ndarray,
    dec_to_ref_indices: np.ndarray,
) -> float:
    if reference_points.size == 0 or decoded_points.size == 0:
        return float("nan")
    ref_vectors = decoded_points[ref_to_dec_indices] - reference_points
    ref_proj_sq = np.sum(ref_vectors * reference_normals, axis=1) ** 2
    matched_ref = reference_points[dec_to_ref_indices]
    matched_normals = reference_normals[dec_to_ref_indices]
    dec_vectors = decoded_points - matched_ref
    dec_proj_sq = np.sum(dec_vectors * matched_normals, axis=1) ** 2
    return float(max(np.mean(ref_proj_sq), np.mean(dec_proj_sq)))


def point_to_plane_components(
    reference_points: np.ndarray,
    decoded_points: np.ndarray,
    reference_normals: np.ndarray,
    ref_to_dec_indices: np.ndarray,
    dec_to_ref_indices: np.ndarray,
):
    ref_vectors = decoded_points[ref_to_dec_indices] - reference_points
    ref_proj_sq = np.sum(ref_vectors * reference_normals, axis=1) ** 2
    matched_ref = reference_points[dec_to_ref_indices]
    matched_normals = reference_normals[dec_to_ref_indices]
    dec_vectors = decoded_points - matched_ref
    dec_proj_sq = np.sum(dec_vectors * matched_normals, axis=1) ** 2
    return float(np.mean(ref_proj_sq)), float(np.mean(dec_proj_sq))
