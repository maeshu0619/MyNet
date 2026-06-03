import math

import torch


def hard_octree_occupancy_stats(pts_3n, qs=1.0, max_depth=0, quant_mode="round", pos_quantscale=1):
    if pts_3n.ndim != 2 or pts_3n.shape[0] != 3:
        raise ValueError(f"pts_3n must have shape [3, N], got {tuple(pts_3n.shape)}")

    pts = pts_3n.detach().to(torch.float32)
    finite = torch.isfinite(pts).all(dim=0)
    if finite.any():
        pts = torch.nan_to_num(pts[:, finite], nan=0.0, posinf=0.0, neginf=0.0)
    else:
        return {
            "point_count": 0,
            "leaf_count": 0,
            "node_count": 0,
            "single_child_count": 0,
            "max_depth": 0,
            "single_ratio": 0.0,
            "mean_children": 0.0,
            "occupancy_pattern_count": 0,
            "occupancy_entropy": 0.0,
            "occupancy_nll": 0.0,
            "lowprob_occupancy_count": 0.0,
            "lowprob_occupancy_ratio": 0.0,
            "occupancy_predictability": 0.0,
        }

    qs = max(float(qs), 1e-9)
    quant_mode = str(quant_mode).strip().lower()
    if quant_mode == "sparsepcgc":
        pos_q = max(int(pos_quantscale), 1)
        q = torch.round(pts / qs)
        if pos_q != 1:
            q = torch.round(q / float(pos_q))
        q = q.to(torch.long)
    else:
        q = torch.round(pts / qs).to(torch.long)
    q = q - q.amin(dim=1, keepdim=True)
    coords = torch.unique(q.transpose(0, 1).contiguous(), dim=0, sorted=False)
    if coords.numel() == 0:
        return {
            "point_count": int(pts.shape[-1]),
            "leaf_count": 0,
            "node_count": 0,
            "single_child_count": 0,
            "max_depth": 0,
            "single_ratio": 0.0,
            "mean_children": 0.0,
            "occupancy_pattern_count": 0,
            "occupancy_entropy": 0.0,
            "occupancy_nll": 0.0,
            "lowprob_occupancy_count": 0.0,
            "lowprob_occupancy_ratio": 0.0,
            "occupancy_predictability": 0.0,
        }

    max_coord = int(coords.max().detach().cpu()) if coords.numel() > 0 else 0
    inferred_depth = max(int(math.ceil(math.log2(max(max_coord + 1, 2)))), 1)
    depth_arg = int(max_depth)
    depth = depth_arg if depth_arg > 0 else inferred_depth
    depth = max(int(depth), 1)

    current = coords
    node_count = 0
    single_child_count = 0
    child_count_sum = 0.0
    parent_count_sum = 0
    pattern_hist = torch.zeros((256,), device=current.device, dtype=torch.float64)
    pattern_weights = (2 ** torch.arange(8, device=current.device, dtype=torch.long)).view(1, 8)
    for _level in range(depth, 0, -1):
        if current.numel() == 0:
            break
        child_bits = current & 1
        child_index = child_bits[:, 0] * 4 + child_bits[:, 1] * 2 + child_bits[:, 2]
        parents = torch.div(current, 2, rounding_mode="floor")
        unique_parents, inverse = torch.unique(parents, dim=0, sorted=False, return_inverse=True)
        occupancy = torch.zeros((unique_parents.shape[0], 8), device=current.device, dtype=torch.bool)
        occupancy[inverse, child_index] = True
        child_counts = occupancy.sum(dim=1)
        parent_count = int(unique_parents.shape[0])
        node_count += parent_count
        single_child_count += int((child_counts == 1).sum().detach().cpu())
        child_count_sum += float(child_counts.to(torch.float32).sum().detach().cpu())
        parent_count_sum += parent_count
        patterns = (occupancy.to(torch.long) * pattern_weights).sum(dim=1)
        pattern_hist += torch.bincount(patterns, minlength=256).to(torch.float64)
        current = unique_parents

    observed_patterns = pattern_hist[pattern_hist > 0]
    if observed_patterns.numel() > 0:
        probs = observed_patterns / observed_patterns.sum().clamp_min(1.0)
        occupancy_entropy = float((-(probs * torch.log2(probs.clamp_min(1e-12))).sum()).detach().cpu())
        lowprob_mask = probs < (1.0 / 16.0)
        lowprob_count = float(observed_patterns[lowprob_mask].sum().detach().cpu())
        occupancy_pattern_count = int(observed_patterns.numel())
    else:
        occupancy_entropy = 0.0
        lowprob_count = 0.0
        occupancy_pattern_count = 0
    lowprob_ratio = lowprob_count / max(float(node_count), 1.0)
    occupancy_predictability = max(0.0, min(1.0, 1.0 - occupancy_entropy / 8.0))

    return {
        "point_count": int(pts.shape[-1]),
        "leaf_count": int(coords.shape[0]),
        "node_count": int(node_count),
        "single_child_count": int(single_child_count),
        "max_depth": int(depth),
        "single_ratio": float(single_child_count) / max(float(node_count), 1.0),
        "mean_children": float(child_count_sum) / max(float(parent_count_sum), 1.0),
        "occupancy_pattern_count": int(occupancy_pattern_count),
        "occupancy_entropy": float(occupancy_entropy),
        "occupancy_nll": float(occupancy_entropy),
        "lowprob_occupancy_count": float(lowprob_count),
        "lowprob_occupancy_ratio": float(lowprob_ratio),
        "occupancy_predictability": float(occupancy_predictability),
    }

def _empty_hard_octree_coord_stats(point_count=0):
    return {
        "point_count": int(point_count),
        "leaf_count": 0,
        "node_count": 0,
        "single_child_count": 0,
        "max_depth": 0,
        "single_ratio": 0.0,
        "mean_children": 0.0,
        "occupancy_pattern_count": 0,
        "occupancy_entropy": 0.0,
        "occupancy_nll": 0.0,
        "lowprob_occupancy_count": 0.0,
        "lowprob_occupancy_ratio": 0.0,
        "occupancy_predictability": 0.0,
        "isolated_voxel_count": 0.0,
        "isolated_voxel_ratio": 0.0,
        "fragmentation_ratio": 0.0,
    }


def _normalize_voxel_coords_for_stats(coords):
    """
    voxel coordsを[B, 3, N]のlong tensorに正規化する。
    入力は[N, 3], [3, N], [B, N, 3], [B, 3, N]を許容する。
    """
    if coords is None:
        raise ValueError("coords must not be None.")
    if not torch.is_tensor(coords):
        coords = torch.as_tensor(coords)

    if coords.ndim == 2:
        if coords.shape[0] == 3:
            coords = coords.unsqueeze(0)
        elif coords.shape[1] == 3:
            coords = coords.transpose(0, 1).contiguous().unsqueeze(0)
        else:
            raise ValueError(f"coords must be [3,N] or [N,3], got {tuple(coords.shape)}")
    elif coords.ndim == 3:
        if coords.shape[1] == 3:
            coords = coords.contiguous()
        elif coords.shape[2] == 3:
            coords = coords.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"coords must be [B,3,N] or [B,N,3], got {tuple(coords.shape)}")
    else:
        raise ValueError(f"coords must be 2D or 3D, got {tuple(coords.shape)}")

    return coords.to(dtype=torch.long).contiguous()


def _coord_keys_for_stats(coords, mins, spans):
    shifted = coords - mins.view(1, 3)
    return (
        shifted[:, 0] * spans[1].clamp_min(1) * spans[2].clamp_min(1)
        + shifted[:, 1] * spans[2].clamp_min(1)
        + shifted[:, 2]
    )


def _coords_membership_for_stats(query_coords, reference_coords):
    if query_coords.numel() == 0:
        return torch.zeros((query_coords.shape[0],), device=query_coords.device, dtype=torch.bool)
    if reference_coords.numel() == 0:
        return torch.zeros((query_coords.shape[0],), device=query_coords.device, dtype=torch.bool)

    combined = torch.cat([query_coords.to(torch.long), reference_coords.to(torch.long)], dim=0)
    mins = combined.amin(dim=0)
    spans = (combined.amax(dim=0) - mins + 1).to(torch.long).clamp_min(1)

    query_keys = _coord_keys_for_stats(query_coords.to(torch.long), mins, spans)
    reference_keys = torch.unique(
        _coord_keys_for_stats(reference_coords.to(torch.long), mins, spans),
        sorted=True,
    )
    pos = torch.searchsorted(reference_keys, query_keys)
    in_bounds = pos < reference_keys.numel()
    safe_pos = pos.clamp(max=max(int(reference_keys.numel()) - 1, 0))
    return in_bounds & (reference_keys[safe_pos] == query_keys)


def _isolated_voxel_stats_from_coords(coords_n3):
    """
    26近傍にoccupied voxelが1つもないvoxelを孤立voxelとして数える。
    """
    coords_n3 = coords_n3.to(dtype=torch.long).reshape(-1, 3)
    if coords_n3.numel() == 0:
        return 0.0, 0.0

    unique_coords = torch.unique(coords_n3, dim=0, sorted=True)
    if unique_coords.shape[0] <= 1:
        return float(unique_coords.shape[0]), 1.0 if unique_coords.shape[0] == 1 else 0.0

    offsets = torch.tensor(
        [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ],
        device=unique_coords.device,
        dtype=torch.long,
    )

    has_neighbor = torch.zeros((unique_coords.shape[0],), device=unique_coords.device, dtype=torch.bool)
    for offset in offsets:
        target = unique_coords + offset.view(1, 3)
        has_neighbor = has_neighbor | _coords_membership_for_stats(target, unique_coords)

    isolated = (~has_neighbor).sum()
    isolated_count = float(isolated.detach().cpu())
    isolated_ratio = isolated_count / max(float(unique_coords.shape[0]), 1.0)
    return isolated_count, isolated_ratio


def _hard_octree_occupancy_stats_from_single_coords(coords_n3, max_depth=0, grid_origin=None):
    coords_n3 = coords_n3.detach().to(dtype=torch.long).reshape(-1, 3)
    if coords_n3.numel() == 0:
        return _empty_hard_octree_coord_stats(0)

    coords = torch.unique(coords_n3, dim=0, sorted=False)
    point_count = int(coords_n3.shape[0])

    if grid_origin is not None:
        if not torch.is_tensor(grid_origin):
            grid_origin = torch.as_tensor(grid_origin, device=coords.device)
        grid_origin = grid_origin.to(device=coords.device, dtype=torch.long).reshape(1, 3)
        coords = coords - grid_origin
    else:
        coords = coords - coords.amin(dim=0, keepdim=True)

    coords = coords - coords.amin(dim=0, keepdim=True)

    max_coord = int(coords.max().detach().cpu()) if coords.numel() > 0 else 0
    inferred_depth = max(int(math.ceil(math.log2(max(max_coord + 1, 2)))), 1)
    depth_arg = int(max_depth)
    depth = depth_arg if depth_arg > 0 else inferred_depth
    depth = max(int(depth), 1)

    current = coords
    node_count = 0
    single_child_count = 0
    child_count_sum = 0.0
    parent_count_sum = 0
    pattern_hist = torch.zeros((256,), device=current.device, dtype=torch.float64)
    pattern_weights = (2 ** torch.arange(8, device=current.device, dtype=torch.long)).view(1, 8)

    for _level in range(depth, 0, -1):
        if current.numel() == 0:
            break
        child_bits = current & 1
        child_index = child_bits[:, 0] * 4 + child_bits[:, 1] * 2 + child_bits[:, 2]
        parents = torch.div(current, 2, rounding_mode="floor")
        unique_parents, inverse = torch.unique(parents, dim=0, sorted=False, return_inverse=True)

        occupancy = torch.zeros((unique_parents.shape[0], 8), device=current.device, dtype=torch.bool)
        occupancy[inverse, child_index] = True

        child_counts = occupancy.sum(dim=1)
        parent_count = int(unique_parents.shape[0])
        node_count += parent_count
        single_child_count += int((child_counts == 1).sum().detach().cpu())
        child_count_sum += float(child_counts.to(torch.float32).sum().detach().cpu())
        parent_count_sum += parent_count

        patterns = (occupancy.to(torch.long) * pattern_weights).sum(dim=1)
        pattern_hist += torch.bincount(patterns, minlength=256).to(torch.float64)
        current = unique_parents

    observed_patterns = pattern_hist[pattern_hist > 0]
    if observed_patterns.numel() > 0:
        probs = observed_patterns / observed_patterns.sum().clamp_min(1.0)
        occupancy_entropy = float((-(probs * torch.log2(probs.clamp_min(1e-12))).sum()).detach().cpu())
        lowprob_mask = probs < (1.0 / 16.0)
        lowprob_count = float(observed_patterns[lowprob_mask].sum().detach().cpu())
        occupancy_pattern_count = int(observed_patterns.numel())
    else:
        occupancy_entropy = 0.0
        lowprob_count = 0.0
        occupancy_pattern_count = 0

    lowprob_ratio = lowprob_count / max(float(node_count), 1.0)
    occupancy_predictability = max(0.0, min(1.0, 1.0 - occupancy_entropy / 8.0))
    isolated_count, isolated_ratio = _isolated_voxel_stats_from_coords(coords_n3)

    return {
        "point_count": int(point_count),
        "leaf_count": int(coords.shape[0]),
        "node_count": int(node_count),
        "single_child_count": int(single_child_count),
        "max_depth": int(depth),
        "single_ratio": float(single_child_count) / max(float(node_count), 1.0),
        "mean_children": float(child_count_sum) / max(float(parent_count_sum), 1.0),
        "occupancy_pattern_count": int(occupancy_pattern_count),
        "occupancy_entropy": float(occupancy_entropy),
        "occupancy_nll": float(occupancy_entropy),
        "lowprob_occupancy_count": float(lowprob_count),
        "lowprob_occupancy_ratio": float(lowprob_ratio),
        "occupancy_predictability": float(occupancy_predictability),
        "isolated_voxel_count": float(isolated_count),
        "isolated_voxel_ratio": float(isolated_ratio),
        "fragmentation_ratio": float(isolated_ratio),
    }


def hard_octree_occupancy_stats_from_voxel_coords(
    coords,
    max_depth=0,
    grid_origin=None,
):
    """
    量子化済みvoxel coordsからoccupancy統計を計算する。
    coordsは[N, 3], [3, N], [B, N, 3], [B, 3, N]を許容する。
    既存のhard_octree_occupancy_statsと同じ主要keyを返し、
    isolated_voxel_count / fragmentation_ratioも追加で返す。
    """
    coords_b3n = _normalize_voxel_coords_for_stats(coords)

    stats_list = []
    for b in range(int(coords_b3n.shape[0])):
        origin_b = None
        if grid_origin is not None:
            origin = grid_origin
            if not torch.is_tensor(origin):
                origin = torch.as_tensor(origin, device=coords_b3n.device)
            origin = origin.to(device=coords_b3n.device, dtype=torch.long)
            if origin.ndim == 1:
                origin_b = origin.reshape(1, 3)
            elif origin.ndim == 2 and origin.shape[-1] == 3:
                origin_b = origin[min(b, origin.shape[0] - 1)].reshape(1, 3)
            elif origin.ndim == 3 and origin.shape[1] == 3:
                origin_b = origin[min(b, origin.shape[0] - 1), :, 0].reshape(1, 3)

        coords_n3 = coords_b3n[b].transpose(0, 1).contiguous()
        stats_list.append(
            _hard_octree_occupancy_stats_from_single_coords(
                coords_n3,
                max_depth=max_depth,
                grid_origin=origin_b,
            )
        )

    if len(stats_list) == 1:
        return stats_list[0]

    out = {}
    keys = stats_list[0].keys()
    for key in keys:
        values = [float(item.get(key, 0.0)) for item in stats_list]
        out[key] = sum(values) / float(max(len(values), 1))
    return out