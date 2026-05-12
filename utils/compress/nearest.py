from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def deterministic_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    max_points = int(max_points or 0)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[indices]


def nearest_distances(source: np.ndarray, target: np.ndarray, squared: bool = False):
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.size == 0 or target.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty.astype(np.int64)
    distances, indices = cKDTree(target).query(source, k=1, workers=-1)
    if squared:
        distances = distances * distances
    return distances.astype(np.float64, copy=False), indices.astype(np.int64, copy=False)


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return points.copy()
    if points.shape[0] <= 3:
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0
        return normals
    k = min(max(int(k), 3), points.shape[0])
    _, indices = cKDTree(points).query(points, k=k, workers=-1)
    if indices.ndim == 1:
        indices = indices[:, None]
    neighborhoods = points[indices]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
    _, _, vh = np.linalg.svd(cov, full_matrices=False)
    normals = vh[:, -1, :]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)
