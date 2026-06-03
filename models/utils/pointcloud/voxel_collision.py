import math
import torch
from .sparsepcgc_voxel import canonical_sparsepcgc_voxel_coords

def _as_points_n3(pts):
    if pts is None:
        return None
    if not torch.is_tensor(pts):
        pts = torch.as_tensor(pts)
    if pts.ndim != 2:
        raise ValueError(f"points must be 2D [3,N] or [N,3], got shape={tuple(pts.shape)}")
    if pts.shape[0] == 3 or (pts.shape[0] > 3 and pts.shape[1] != 3):
        return pts[:3, :].transpose(0, 1).contiguous()
    if pts.shape[1] >= 3:
        return pts[:, :3].contiguous()
    raise ValueError(f"points must contain xyz coordinates, got shape={tuple(pts.shape)}")


def _finite_filter(xyz, drop_nonfinite=True):
    finite_mask = torch.isfinite(xyz).all(dim=1)
    if drop_nonfinite:
        return xyz[finite_mask], int(finite_mask.sum().item())
    return torch.nan_to_num(xyz), int(finite_mask.sum().item())


def sparsepcgc_quantized_coords(
    pts_3n,
    voxel_size,
    pos_quantscale,
    *,
    drop_nonfinite=True,
    shift_min_to_zero=False,
):
    """SparsePCGCのround+unique前の量子化座標を返す。"""
    with torch.no_grad():
        xyz = _as_points_n3(pts_3n)
        if xyz is None:
            return torch.empty((0, 3), dtype=torch.long)
        xyz = xyz.detach().to(dtype=torch.float32)
        xyz, _ = _finite_filter(xyz, drop_nonfinite=drop_nonfinite)
        if xyz.numel() == 0:
            return torch.empty((0, 3), device=xyz.device, dtype=torch.long)

        voxel = max(float(voxel_size), 1e-9)
        pos_q = max(float(pos_quantscale), 1.0)
        coords = torch.round(xyz / voxel)
        if pos_q > 1.0:
            coords = torch.round(coords / pos_q)
        coords = coords.to(torch.long)
        if shift_min_to_zero and coords.numel() > 0:
            # 座標全体の平行移動はunique数や重複率を変えない。
            # Sparse Tensorの入力座標を非負にそろえたい確認用途だけで使う。
            coords = coords - coords.amin(dim=0, keepdim=True)
        return coords


def _quantize_xyz_n3(xyz, voxel_size, pos_quantscale):
    return canonical_sparsepcgc_voxel_coords(
        xyz,
        args=None,
        coord_scale=None,
        voxel_size=float(voxel_size),
        pos_quantscale=int(pos_quantscale),
        global_offset=None,
    ).to(torch.long)


def _sample_points_if_needed(xyz, max_points):
    total = int(xyz.shape[0])
    if max_points is None or int(max_points) <= 0 or total <= int(max_points):
        return xyz, False, total, 1
    max_points = max(int(max_points), 1)
    stride = max(int(math.ceil(total / float(max_points))), 1)
    sampled = xyz[::stride]
    if sampled.shape[0] > max_points:
        sampled = sampled[:max_points]
    return sampled.contiguous(), True, int(sampled.shape[0]), stride


def compute_voxel_collision_stats(pts_3n, voxel_size, pos_quantscale, max_points=None):
    """1点群のSparsePCGC互換Voxel衝突統計を計算する。"""
    with torch.no_grad():
        xyz = _as_points_n3(pts_3n)
        raw_point_count = 0 if xyz is None else int(xyz.shape[0])
        if xyz is None:
            finite_xyz = torch.empty((0, 3), dtype=torch.float32)
            original_finite = 0
        else:
            xyz = xyz.detach().to(dtype=torch.float32)
            finite_xyz, original_finite = _finite_filter(xyz, drop_nonfinite=True)

        sampled_xyz, sampled, sampled_count, stride = _sample_points_if_needed(finite_xyz, max_points)
        coords = _quantize_xyz_n3(sampled_xyz, voxel_size, pos_quantscale)
        finite_point_count = int(coords.shape[0])
        if finite_point_count > 0:
            unique_coords, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
            unique_voxel_count = int(unique_coords.shape[0])
            counts = torch.bincount(inverse, minlength=unique_voxel_count).to(torch.float32)
            max_points_per_voxel = int(counts.max().item()) if counts.numel() > 0 else 0
            mean_points_per_occupied_voxel = float(counts.mean().item()) if counts.numel() > 0 else 0.0
        else:
            unique_voxel_count = 0
            max_points_per_voxel = 0
            mean_points_per_occupied_voxel = 0.0

        duplicate_point_count = max(int(finite_point_count - unique_voxel_count), 0)
        denom = max(float(finite_point_count), 1.0)
        sampling_note = ""
        if sampled:
            sampling_note = (
                f"sampled_by_stride={stride}; original_finite_point_count={original_finite}; "
                f"sampled_point_count={sampled_count}"
            )
        return {
            "raw_point_count": int(raw_point_count),
            "finite_point_count": int(finite_point_count),
            "original_finite_point_count": int(original_finite),
            "unique_voxel_count": int(unique_voxel_count),
            "duplicate_point_count": int(duplicate_point_count),
            "duplicate_rate": float(duplicate_point_count) / denom,
            "max_points_per_voxel": int(max_points_per_voxel),
            "mean_points_per_occupied_voxel": float(mean_points_per_occupied_voxel),
            "point_reduction_rate": 1.0 - float(unique_voxel_count) / denom,
            "voxel_size": float(voxel_size),
            "pos_quantscale": int(max(int(pos_quantscale), 1)),
            "quant_mode": "round_xyz_div_voxel_then_round_div_pos_quantscale",
            "sampled": bool(sampled),
            "sampled_point_count": int(sampled_count),
            "sampling_stride": int(stride),
            "sampling_note": sampling_note,
        }


def compute_voxel_collision_stats_batch(
    pts_b3n,
    voxel_size,
    pos_quantscale,
    *,
    max_points=None,
    first_batch_only=True,
):
    """Batch点群のSparsePCGC互換Voxel衝突統計を合算する。"""
    if pts_b3n is None:
        empty = compute_voxel_collision_stats(None, voxel_size, pos_quantscale, max_points=max_points)
        empty.update(
            {
                "batch_count": 0,
                "processed_batch_count": 0,
                "first_batch_only": bool(first_batch_only),
                "sampled_batch_count": 0,
            }
        )
        return empty
    if not torch.is_tensor(pts_b3n):
        pts_b3n = torch.as_tensor(pts_b3n)
    if pts_b3n.ndim == 2:
        stats = compute_voxel_collision_stats(pts_b3n, voxel_size, pos_quantscale, max_points=max_points)
        stats.update(
            {
                "batch_count": 1,
                "processed_batch_count": 1,
                "first_batch_only": bool(first_batch_only),
                "sampled_batch_count": int(bool(stats.get("sampled", False))),
            }
        )
        return stats
    if pts_b3n.ndim != 3:
        raise ValueError(f"batched points must be [B,3,N] or [B,N,3], got shape={tuple(pts_b3n.shape)}")

    batch_count = int(pts_b3n.shape[0])
    indices = [0] if first_batch_only and batch_count > 0 else list(range(batch_count))
    per_batch = []
    for b in indices:
        item = pts_b3n[b]
        per_batch.append(compute_voxel_collision_stats(item, voxel_size, pos_quantscale, max_points=max_points))

    raw_total = sum(int(item["raw_point_count"]) for item in per_batch)
    finite_total = sum(int(item["finite_point_count"]) for item in per_batch)
    original_finite_total = sum(int(item.get("original_finite_point_count", item["finite_point_count"])) for item in per_batch)
    unique_total = sum(int(item["unique_voxel_count"]) for item in per_batch)
    duplicate_total = sum(int(item["duplicate_point_count"]) for item in per_batch)
    max_per_voxel = max([int(item["max_points_per_voxel"]) for item in per_batch] or [0])
    denom = max(float(finite_total), 1.0)
    mean_points = float(finite_total) / max(float(unique_total), 1.0) if unique_total > 0 else 0.0
    sampled_batch_count = sum(1 for item in per_batch if bool(item.get("sampled", False)))
    notes = [str(item.get("sampling_note", "")) for item in per_batch if item.get("sampling_note")]
    return {
        "raw_point_count": int(raw_total),
        "finite_point_count": int(finite_total),
        "original_finite_point_count": int(original_finite_total),
        "unique_voxel_count": int(unique_total),
        "duplicate_point_count": int(duplicate_total),
        "duplicate_rate": float(duplicate_total) / denom,
        "max_points_per_voxel": int(max_per_voxel),
        "mean_points_per_occupied_voxel": float(mean_points),
        "point_reduction_rate": 1.0 - float(unique_total) / denom,
        "voxel_size": float(voxel_size),
        "pos_quantscale": int(max(int(pos_quantscale), 1)),
        "quant_mode": "round_xyz_div_voxel_then_round_div_pos_quantscale",
        "batch_count": int(batch_count),
        "processed_batch_count": int(len(per_batch)),
        "first_batch_only": bool(first_batch_only),
        "sampled": bool(sampled_batch_count > 0),
        "sampled_batch_count": int(sampled_batch_count),
        "sampling_note": " | ".join(notes),
    }


def flatten_voxel_collision_stats(prefix, stats):
    """CSV/debugへ入れやすい平坦なkeyへ変換する。"""
    if not stats:
        return {}
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def format_voxel_collision_summary(stage, stats):
    return (
        f"VoxelCollision[{stage}]: "
        f"raw={int(stats.get('raw_point_count', 0))}, "
        f"finite={int(stats.get('finite_point_count', 0))}, "
        f"unique={int(stats.get('unique_voxel_count', 0))}, "
        f"dup={int(stats.get('duplicate_point_count', 0))}, "
        f"dup_rate={float(stats.get('duplicate_rate', 0.0)):.6f}, "
        f"max_per_voxel={int(stats.get('max_points_per_voxel', 0))}, "
        f"reduction={float(stats.get('point_reduction_rate', 0.0)):.6f}"
    )
