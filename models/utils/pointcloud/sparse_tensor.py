from typing import Dict

import torch


def _normalize_coord_scale_like(pts_xyz: torch.Tensor, coord_scale):
    if coord_scale is None:
        return pts_xyz.new_tensor(1.0)
    if torch.is_tensor(coord_scale):
        scale = coord_scale.to(device=pts_xyz.device, dtype=pts_xyz.dtype).reshape(-1)[0]
        return scale.clamp_min(1e-9)
    return pts_xyz.new_tensor(max(float(coord_scale), 1e-9))


def _voxelize_once(pts_xyz: torch.Tensor, voxel_size: float):
    mins = pts_xyz.amin(dim=1, keepdim=True)
    discrete = torch.floor((pts_xyz - mins) / max(float(voxel_size), 1e-9)).long().transpose(0, 1).contiguous()
    unique_coords, inverse = torch.unique(discrete, dim=0, sorted=True, return_inverse=True)
    voxel_count = int(unique_coords.shape[0])

    counts = torch.bincount(inverse, minlength=voxel_count).to(device=pts_xyz.device, dtype=pts_xyz.dtype)
    centers = mins + (unique_coords.transpose(0, 1).to(dtype=pts_xyz.dtype) + 0.5) * float(voxel_size)
    return unique_coords, inverse, centers, counts


def _full_sparse_tensor_points(
    pts_xyz: torch.Tensor,
    coord_scale,
    qs: float,
    voxel_scale: float = 1.0,
    quant_mode: str = "floor_relative",
    pos_quantscale: int = 1,
):
    scale = _normalize_coord_scale_like(pts_xyz, coord_scale)
    voxel_size = max(float(voxel_scale) * float(qs) / max(float(scale.item()), 1e-9), 1e-9)
    quant_mode = str(quant_mode).strip().lower()
    if quant_mode in {"sparsepcgc_twostep"}:
        pos_q = max(int(pos_quantscale), 1)
        first = torch.round(pts_xyz / voxel_size)
        discrete = torch.round(first / float(pos_q)).long()
        coords_xyz = discrete.to(dtype=pts_xyz.dtype) * voxel_size * float(pos_q)
    elif quant_mode in {"round_absolute", "sparsepcgc"}:
        discrete = torch.round(pts_xyz / voxel_size).long()
        coords_xyz = discrete.to(dtype=pts_xyz.dtype) * voxel_size
    elif quant_mode in {"floor_absolute"}:
        discrete = torch.floor(pts_xyz / voxel_size).long()
        coords_xyz = (discrete.to(dtype=pts_xyz.dtype) + 0.5) * voxel_size
    else:
        mins = pts_xyz.amin(dim=1, keepdim=True)
        discrete = torch.floor((pts_xyz - mins) / voxel_size).long()
        coords_xyz = mins + (discrete.to(dtype=pts_xyz.dtype) + 0.5) * voxel_size
    feat = pts_xyz.new_ones((1, int(coords_xyz.shape[-1])))
    return coords_xyz, feat, float(voxel_size)


def _downsample_sparse_tensor_points(
    sparse_xyz: torch.Tensor,
    target_points: int,
    base_voxel_size: float,
    growth: float = 1.5,
    max_iters: int = 8,
):
    num_points = int(sparse_xyz.shape[-1])
    target_points = max(int(target_points), 1)
    if num_points <= target_points:
        feat = sparse_xyz.new_ones((1, num_points))
        counts = sparse_xyz.new_ones((num_points,))
        return sparse_xyz, feat, counts, float(base_voxel_size)

    span = (sparse_xyz.amax(dim=1, keepdim=True) - sparse_xyz.amin(dim=1, keepdim=True)).amax().clamp_min(1e-9)
    base_from_budget = float(span.item()) / max(float(target_points), 1.0) ** (1.0 / 3.0)
    voxel_size = max(float(base_voxel_size), base_from_budget, 1e-9)
    growth = max(float(growth), 1.05)
    max_iters = max(int(max_iters), 1)

    coords = None
    centers = None
    counts = None
    for _ in range(max_iters):
        coords, _, centers, counts = _voxelize_once(sparse_xyz, voxel_size)
        if int(coords.shape[0]) <= target_points:
            break
        voxel_size *= growth

    if int(centers.shape[-1]) > target_points:
        keep = torch.argsort(counts, descending=True)[:target_points]
        keep = torch.sort(keep).values
        centers = centers.index_select(1, keep)
        counts = counts.index_select(0, keep)
    feat = sparse_xyz.new_ones((1, int(centers.shape[-1])))
    return centers, feat, counts, float(voxel_size)


def build_sparse_point_tensor_single(
    pts_xyz: torch.Tensor,
    coord_scale,
    max_points: int,
    qs: float,
    raw_downsample_factor: float = 1.0,
    voxel_scale: float = 1.0,
    growth: float = 1.5,
    max_iters: int = 8,
    quant_mode: str = "floor_relative",
    pos_quantscale: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Build a sparse-tensor-like representation in two stages.

    1) Convert every input point into a sparse-grid coordinate/feature pair
       without reducing point count.
    2) Downsample that sparse tensor only for the encoder path.

    Returns:
        {
            "sparse_xyz": [3, N] sparse-tensor coordinates before encoder downsampling,
            "sparse_feat": [1, N] binary occupancy feature before encoder downsampling,
            "coords_xyz": [3, M] encoder-side downsampled sparse coordinates,
            "feat": [1, M] encoder-side downsampled occupancy feature,
            "counts": [M],
            "raw_points": int,
            "pre_downsample_points": int,  # sparse tensor point count before encoder downsampling
            "voxel_size": float,
        }
    """
    raw_points = int(pts_xyz.shape[-1])
    sparse_xyz, sparse_feat, sparse_voxel_size = _full_sparse_tensor_points(
        pts_xyz,
        coord_scale=coord_scale,
        qs=qs,
        voxel_scale=voxel_scale,
        quant_mode=quant_mode,
        pos_quantscale=pos_quantscale,
    )
    num_points = int(sparse_xyz.shape[-1])
    max_points = max(int(max_points), 0)
    factor = max(float(raw_downsample_factor), 1.0)
    target_points = max(int(round(num_points / factor)), 1)
    if max_points > 0:
        target_points = min(target_points, max_points)

    centers, feat, counts, voxel_size = _downsample_sparse_tensor_points(
        sparse_xyz,
        target_points=target_points,
        base_voxel_size=sparse_voxel_size,
        growth=growth,
        max_iters=max_iters,
    )

    return {
        "sparse_xyz": sparse_xyz,
        "sparse_feat": sparse_feat,
        "coords_xyz": centers,
        "feat": feat,
        "counts": counts,
        "raw_points": raw_points,
        "pre_downsample_points": num_points,
        "voxel_size": float(voxel_size),
        "sparse_voxel_size": float(sparse_voxel_size),
    }
