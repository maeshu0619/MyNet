import math

import torch


def hard_octree_occupancy_stats(pts_3n, qs=1.0, max_depth=0):
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
        }

    qs = max(float(qs), 1e-9)
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
        current = unique_parents

    return {
        "point_count": int(pts.shape[-1]),
        "leaf_count": int(coords.shape[0]),
        "node_count": int(node_count),
        "single_child_count": int(single_child_count),
        "max_depth": int(depth),
        "single_ratio": float(single_child_count) / max(float(node_count), 1.0),
        "mean_children": float(child_count_sum) / max(float(parent_count_sum), 1.0),
    }
