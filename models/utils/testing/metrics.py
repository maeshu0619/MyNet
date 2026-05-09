import math

import numpy as np


def _to_xyz_array(points):
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and torch.is_tensor(points):
        pts = points.detach().to(device="cpu", dtype=torch.float32)
        if pts.ndim == 3:
            pts = pts[0]
        if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] != 3:
            pts = pts[:3].transpose(0, 1)
        return pts.numpy().astype(np.float64, copy=False)

    arr = np.asarray(points)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] != 3:
        arr = arr[:3].T
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected point cloud shape (N, 3+) or (3+, N), got {arr.shape}")
    return arr[:, :3].astype(np.float64, copy=False)


def _stable_sample(points, max_points, seed):
    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(int(seed) % (2**32))
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    idx.sort()
    return points[idx]


def _stable_sample_pair(ref, rec, max_points, seed):
    max_points = int(max_points)
    if max_points <= 0:
        return ref, rec
    if ref.shape[0] == rec.shape[0] and ref.shape[0] > max_points:
        rng = np.random.default_rng(int(seed) % (2**32))
        idx = rng.choice(ref.shape[0], size=max_points, replace=False)
        idx.sort()
        return ref[idx], rec[idx]
    return (
        _stable_sample(ref, max_points=max_points, seed=int(seed) + 17),
        _stable_sample(rec, max_points=max_points, seed=int(seed) + 31),
    )


def _nearest_squared(src, dst):
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(dst)
        try:
            dist, idx = tree.query(src, k=1, workers=1)
        except TypeError:
            dist, idx = tree.query(src, k=1)
        return np.square(dist).astype(np.float64, copy=False), idx.astype(np.int64, copy=False)
    except Exception:
        chunk = 1024
        all_dist = []
        all_idx = []
        for start in range(0, src.shape[0], chunk):
            block = src[start:start + chunk]
            diff = block[:, None, :] - dst[None, :, :]
            dist2 = np.einsum("ijk,ijk->ij", diff, diff)
            idx = np.argmin(dist2, axis=1)
            all_dist.append(dist2[np.arange(block.shape[0]), idx])
            all_idx.append(idx.astype(np.int64, copy=False))
        return np.concatenate(all_dist), np.concatenate(all_idx)


def _estimate_normals(points, k):
    count = int(points.shape[0])
    if count < 3:
        normals = np.zeros((count, 3), dtype=np.float64)
        normals[:, 2] = 1.0
        return normals

    k = min(max(int(k), 3), count)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        try:
            _, idx = tree.query(points, k=k, workers=1)
        except TypeError:
            _, idx = tree.query(points, k=k)
    except Exception:
        _, idx = _nearest_k_fallback(points, k)

    if idx.ndim == 1:
        idx = idx[:, None]
    neighbors = points[idx]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(float(k), 1.0)
    _, eigvec = np.linalg.eigh(cov)
    normals = eigvec[:, :, 0]
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-12)


def _nearest_k_fallback(points, k):
    dist2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    idx = np.argpartition(dist2, kth=min(k - 1, points.shape[0] - 1), axis=1)[:, :k]
    dist = np.take_along_axis(dist2, idx, axis=1)
    return dist, idx


def _psnr_from_mse(mse, peak2):
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 1e-30:
        return float("inf")
    peak2 = max(float(peak2), 1e-30)
    return float(10.0 * math.log10(peak2 / float(mse)))


def compute_pointcloud_metrics(input_points, output_points, max_points=8192, normal_k=16, seed=0):
    """Compute lightweight CPU-only CD, D1 PSNR, and D2 PSNR.

    Inputs may be torch tensors or numpy arrays. Large clouds are sampled
    deterministically before KD-tree queries to keep testing logs cheap.
    """
    ref = _to_xyz_array(input_points)
    rec = _to_xyz_array(output_points)
    if ref.shape[0] == 0 or rec.shape[0] == 0:
        return {"cd": float("nan"), "d1_psnr": float("nan"), "d2_psnr": float("nan")}

    ref, rec = _stable_sample_pair(ref, rec, max_points=max_points, seed=int(seed))

    ref_to_rec_dist2, ref_to_rec_idx = _nearest_squared(ref, rec)
    rec_to_ref_dist2, rec_to_ref_idx = _nearest_squared(rec, ref)
    ref_mse = float(ref_to_rec_dist2.mean())
    rec_mse = float(rec_to_ref_dist2.mean())
    cd = ref_mse + rec_mse
    d1_mse = 0.5 * (ref_mse + rec_mse)

    union = np.concatenate([ref, rec], axis=0)
    span = union.max(axis=0) - union.min(axis=0)
    peak2 = float(np.dot(span, span))

    ref_normals = _estimate_normals(ref, normal_k)
    rec_normals = _estimate_normals(rec, normal_k)
    ref_to_rec_vec = ref - rec[ref_to_rec_idx]
    rec_to_ref_vec = rec - ref[rec_to_ref_idx]
    ref_to_rec_plane = np.sum(ref_to_rec_vec * rec_normals[ref_to_rec_idx], axis=1) ** 2
    rec_to_ref_plane = np.sum(rec_to_ref_vec * ref_normals[rec_to_ref_idx], axis=1) ** 2
    d2_mse = 0.5 * (float(ref_to_rec_plane.mean()) + float(rec_to_ref_plane.mean()))

    return {
        "cd": float(cd),
        "d1_psnr": _psnr_from_mse(d1_mse, peak2),
        "d2_psnr": _psnr_from_mse(d2_mse, peak2),
    }
