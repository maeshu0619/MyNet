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
