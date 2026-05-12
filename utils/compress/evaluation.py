from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from .ChamferDistance import chamfer_distance
from .D1PSNR import d1_psnr
from .D2PSNR import d2_psnr
from .EarthMoverDistance import earth_mover_distance
from .HausdorffDistance import hausdorff_distance
from .NormalConsistency import normal_consistency
from .PointToPlane import point_to_plane_components, point_to_plane_mse
from .PointToPoint import point_to_point_mse
from .nearest import deterministic_subsample, estimate_normals, nearest_distances
from .ply_io import read_ply_xyz


SHAPE_METRIC_KEYS = (
    "cd",
    "hd",
    "p2point",
    "p2plane",
    "emd",
    "nc",
    "d1_psnr",
    "d2_psnr",
)

HIGHER_IS_BETTER = {"nc", "d1_psnr", "d2_psnr"}


@dataclass
class EvaluationConfig:
    max_points: int = 0
    normal_max_points: int = 100000
    emd_points: int = 2048
    normal_k: int = 16
    psnr_peak: float = 0.0


def _auto_peak(reference_points: np.ndarray) -> float:
    if reference_points.size == 0:
        return 1.0
    extent = np.ptp(reference_points, axis=0)
    peak = float(np.max(extent))
    return peak if peak > 0.0 else 1.0


def evaluate_decoded_geometry(
    reference_path: str | Path,
    decoded_path: str | Path,
    config: EvaluationConfig | None = None,
) -> Dict[str, float]:
    config = config or EvaluationConfig()
    reference_points = read_ply_xyz(reference_path)
    decoded_points = read_ply_xyz(decoded_path)

    ref_eval = deterministic_subsample(reference_points, config.max_points)
    dec_eval = deterministic_subsample(decoded_points, config.max_points)

    ref_to_dec, ref_to_dec_indices = nearest_distances(ref_eval, dec_eval, squared=False)
    dec_to_ref, dec_to_ref_indices = nearest_distances(dec_eval, ref_eval, squared=False)
    ref_to_dec_sq = ref_to_dec * ref_to_dec
    dec_to_ref_sq = dec_to_ref * dec_to_ref

    normal_ref = deterministic_subsample(reference_points, config.normal_max_points)
    normal_dec = deterministic_subsample(decoded_points, config.normal_max_points)
    normal_ref_to_dec, normal_ref_to_dec_indices = nearest_distances(normal_ref, normal_dec, squared=False)
    normal_dec_to_ref, normal_dec_to_ref_indices = nearest_distances(normal_dec, normal_ref, squared=False)
    del normal_ref_to_dec, normal_dec_to_ref

    ref_normals = estimate_normals(normal_ref, k=config.normal_k)
    dec_normals = estimate_normals(normal_dec, k=config.normal_k)
    p2plane_fwd, p2plane_bwd = point_to_plane_components(
        normal_ref,
        normal_dec,
        ref_normals,
        normal_ref_to_dec_indices,
        normal_dec_to_ref_indices,
    )
    p2plane_value = point_to_plane_mse(
        normal_ref,
        normal_dec,
        ref_normals,
        normal_ref_to_dec_indices,
        normal_dec_to_ref_indices,
    )

    p2point_value = point_to_point_mse(ref_to_dec_sq, dec_to_ref_sq)
    peak = float(config.psnr_peak) if float(config.psnr_peak) > 0.0 else _auto_peak(reference_points)

    return {
        "reference_point_count": float(reference_points.shape[0]),
        "decoded_point_count": float(decoded_points.shape[0]),
        "cd": chamfer_distance(ref_to_dec_sq, dec_to_ref_sq),
        "hd": hausdorff_distance(ref_to_dec, dec_to_ref),
        "p2point": p2point_value,
        "p2plane": p2plane_value,
        "emd": earth_mover_distance(reference_points, decoded_points, max_points=config.emd_points),
        "nc": normal_consistency(ref_normals, dec_normals, normal_ref_to_dec_indices, normal_dec_to_ref_indices),
        "d1_psnr": d1_psnr(p2point_value, peak),
        "d2_psnr": d2_psnr(max(p2plane_fwd, p2plane_bwd), peak),
    }
