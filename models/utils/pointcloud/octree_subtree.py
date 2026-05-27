import math
import hashlib
import time
from typing import Dict, Optional

import torch


def resolve_subtree_level(args, default: Optional[int] = None) -> int:
    level = int(getattr(args, "train_subtree_level", 0))
    if level <= 0:
        if default is None:
            default = int(getattr(args, "repair_unit_level", max(1, int(getattr(args, "octree_ctx_level", 5)))))
        level = int(default)
    return max(level, 1)


def resolve_subtree_level_range(args, default: Optional[int] = None):
    base_level = resolve_subtree_level(args, default=default)
    jitter = max(int(getattr(args, "train_subtree_level_jitter", 0)), 0)
    min_level = int(getattr(args, "train_subtree_level_min", 0))
    max_level = int(getattr(args, "train_subtree_level_max", 0))
    if min_level <= 0:
        min_level = max(1, base_level - jitter)
    if max_level <= 0:
        max_level = base_level + jitter
    if min_level > max_level:
        min_level, max_level = max_level, min_level
    return {
        "base": int(base_level),
        "min": max(int(min_level), 1),
        "max": max(int(max_level), 1),
    }


def _effective_subtree_qs(args) -> float:
    compress_key = (
        str(getattr(args, "compress", ""))
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    if compress_key == "sparsepcgc":
        return max(
            float(getattr(args, "sparsepcgc_effective_qs", 0.0))
            or float(getattr(args, "sparsepcgc_voxel_size", 1.0)) * float(getattr(args, "sparsepcgc_pos_quantscale", 1)),
            1e-9,
        )
    if compress_key in {"gpcc", "gpcctmc3"}:
        return max(float(getattr(args, "gpcc_effective_qs", getattr(args, "qs", 2.0))), 1e-9)
    return max(float(getattr(args, "qs", 2.0)), 1e-9)


def _percent_range(values, fallback):
    if isinstance(values, (list, tuple)) and len(values) >= 2:
        lo, hi = float(values[0]), float(values[1])
    else:
        parts = [item.strip() for item in str(values).split(",") if item.strip()]
        if len(parts) >= 2:
            lo, hi = float(parts[0]), float(parts[1])
        else:
            lo, hi = fallback
    lo, hi = sorted((lo, hi))
    lo = min(max(lo, 0.0), 1.0)
    hi = min(max(hi, 0.0), 1.0)
    if hi <= 0.0:
        hi = max(float(fallback[1]), 1e-6)
    return lo, hi


def surrogate_pretrain_depth_percent_range(args, fallback=(0.0, 0.50)):
    """Depth percent range shared by surrogate pretrain and train subtree sampling."""
    lo = float(getattr(args, "surrogate_pretrain_subtree_depth_percent_min", fallback[0]))
    hi = float(getattr(args, "surrogate_pretrain_subtree_depth_percent_max", fallback[1]))
    return _percent_range((lo, hi), fallback)


def percent_depth_bounds(data_max_level, pct_lo, pct_hi, *, min_depth=1):
    data_max_level = max(int(data_max_level), 1)
    pct_lo, pct_hi = sorted((float(pct_lo), float(pct_hi)))
    pct_lo = min(max(pct_lo, 0.0), 1.0)
    pct_hi = min(max(pct_hi, 0.0), 1.0)
    min_level = max(int(min_depth), min(int(math.ceil(float(data_max_level) * pct_lo)), data_max_level))
    max_level = max(int(min_depth), min(int(math.floor(float(data_max_level) * pct_hi)), data_max_level))
    if min_level > max_level:
        min_level, max_level = max_level, min_level
    return int(min_level), int(max_level)


def _sample_depth_from_range(min_level, max_level, args, global_step, cache_key):
    min_level = int(min_level)
    max_level = int(max_level)
    if max_level <= min_level:
        return max(min_level, 1)
    mode = str(getattr(args, "train_subtree_level_sampling", "uniform_random")).strip().lower()
    span = max_level - min_level + 1
    if mode == "coverage_cycle":
        return min_level + (int(global_step) % span)
    if mode == "uniform_random":
        seed_text = f"{cache_key or ''}|step={int(global_step)}|seed={int(getattr(args, 'seed', 0))}"
        seed_value = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16], 16)
        return min_level + (seed_value % span)
    raise ValueError(
        "--train_subtree_level_sampling must be one of: uniform_random, coverage_cycle "
        f"(got {mode})"
    )


def _estimate_subtree_max_level_single(valid_pts: torch.Tensor, qs: float) -> int:
    offset = valid_pts.amin(dim=1)
    qpts = torch.round((valid_pts - offset.unsqueeze(1)) / qs).to(torch.long)
    max_coord = int(qpts.max().detach().cpu()) if qpts.numel() > 0 else 0
    return max(int(math.ceil(math.log2(max(max_coord + 1, 1)))), 1)


def sample_train_subtree_depth(pts_xyz: torch.Tensor, args, global_step: int = 0, cache_key: Optional[str] = None):
    if pts_xyz.ndim != 3 or pts_xyz.shape[1] != 3:
        raise ValueError(f"pts_xyz must have shape [B, 3, N], got {tuple(pts_xyz.shape)}")

    qs = _effective_subtree_qs(args)
    level_range = resolve_subtree_level_range(args)
    data_max_level = None
    for b in range(pts_xyz.shape[0]):
        pts_b = pts_xyz[b].to(torch.float32)
        finite_mask = torch.isfinite(pts_b).all(dim=0)
        if finite_mask.any():
            valid_pts = torch.nan_to_num(pts_b[:, finite_mask], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            valid_pts = pts_b.new_zeros((3, 1))
        max_level_b = _estimate_subtree_max_level_single(valid_pts, qs)
        data_max_level = max_level_b if data_max_level is None else min(data_max_level, max_level_b)

    if data_max_level is None:
        data_max_level = level_range["base"]
    percent_curriculum = (
        bool(getattr(args, "train_subtree_depth_percent_curriculum", False))
        and not bool(getattr(args, "_train_subtree_depth_cli_override", False))
    )
    if percent_curriculum:
        total_steps = max(int(getattr(args, "_total_train_steps_estimate", 0)), 0)
        curriculum_fraction = min(max(float(getattr(args, "train_subtree_curriculum_fraction", 1.0)), 0.0), 1.0)
        curriculum_steps = int(round(float(total_steps) * curriculum_fraction)) if total_steps > 0 else 0
        if curriculum_steps > 0:
            phase_denom = max(curriculum_steps - 1, 1)
            curriculum_phase = min(max(float(global_step) / float(phase_denom), 0.0), 1.0)
        else:
            curriculum_phase = 0.0
        shared_lo, shared_hi = surrogate_pretrain_depth_percent_range(args)
        start_lo, start_hi = _percent_range(
            getattr(args, "train_subtree_depth_percent_start", f"{shared_lo},{shared_hi}"),
            (shared_lo, shared_hi),
        )
        end_lo, end_hi = _percent_range(
            getattr(args, "train_subtree_depth_percent_end", f"{shared_lo},{shared_hi}"),
            (shared_lo, shared_hi),
        )
        pct_lo = start_lo + (end_lo - start_lo) * curriculum_phase
        pct_hi = start_hi + (end_hi - start_hi) * curriculum_phase
        pct_lo, pct_hi = sorted((pct_lo, pct_hi))
        min_level, max_level = percent_depth_bounds(data_max_level, pct_lo, pct_hi)
        chosen = _sample_depth_from_range(min_level, max_level, args, global_step, cache_key)
        return {
            "depth": int(chosen),
            "base_depth": int(level_range["base"]),
            "min_depth": int(min_level),
            "max_depth": int(max_level),
            "uncapped_min_depth": int(min_level),
            "uncapped_max_depth": int(max_level),
            "curriculum_phase": float(curriculum_phase),
            "curriculum_steps": int(curriculum_steps),
            "data_max_depth": int(data_max_level),
            "depth_percent_curriculum": True,
            "depth_percent_range": (float(pct_lo), float(pct_hi)),
        }
    explicit_range = int(getattr(args, "train_subtree_level_min", 0)) > 0 or int(getattr(args, "train_subtree_level_max", 0)) > 0
    use_full_random_range = (
        bool(getattr(args, "train_subtree_randomize_level", False))
        and bool(getattr(args, "train_subtree_random_full_range", True))
        and not explicit_range
    )
    if use_full_random_range:
        min_level = 1
        max_level = max(1, int(data_max_level))
    else:
        min_level = max(1, min(level_range["min"], data_max_level))
        max_level = max(min_level, min(level_range["max"], data_max_level))
    uncapped_min_level = int(min_level)
    uncapped_max_level = int(max_level)
    curriculum_phase = 1.0
    curriculum_steps = 0
    if bool(getattr(args, "train_subtree_level_curriculum", True)) and max_level > min_level:
        total_steps = max(int(getattr(args, "_total_train_steps_estimate", 0)), 0)
        curriculum_fraction = min(max(float(getattr(args, "train_subtree_curriculum_fraction", 1.0)), 0.0), 1.0)
        curriculum_steps = int(round(float(total_steps) * curriculum_fraction)) if total_steps > 0 else 0
        if curriculum_steps > 0:
            phase_denom = max(curriculum_steps - 1, 1)
            curriculum_phase = min(max(float(global_step) / float(phase_denom), 0.0), 1.0)
            depth_span = uncapped_max_level - uncapped_min_level
            direction = str(getattr(args, "train_subtree_curriculum_direction", "deep_to_shallow")).strip().lower()
            if direction == "shallow_to_deep":
                curriculum_max = uncapped_min_level + int(math.floor(float(depth_span) * curriculum_phase + 1e-9))
                max_level = max(min_level, min(max_level, curriculum_max))
            else:
                curriculum_min = uncapped_max_level - int(math.floor(float(depth_span) * curriculum_phase + 1e-9))
                min_level = min(max_level, max(min_level, curriculum_min))
    chosen = max(min(level_range["base"], max_level), min_level)

    if bool(getattr(args, "train_subtree_randomize_level", False)) and max_level > min_level:
        chosen = _sample_depth_from_range(min_level, max_level, args, global_step, cache_key)

    return {
        "depth": int(chosen),
        "base_depth": int(level_range["base"]),
        "min_depth": int(min_level),
        "max_depth": int(max_level),
        "uncapped_min_depth": int(uncapped_min_level),
        "uncapped_max_depth": int(uncapped_max_level),
        "curriculum_phase": float(curriculum_phase),
        "curriculum_steps": int(curriculum_steps),
        "data_max_depth": int(data_max_level),
        "depth_percent_curriculum": False,
    }


def should_use_full_cloud_anchor(args, global_step: int = 0, cache_key: Optional[str] = None):
    """Return whether this step should train on the full cloud instead of one subtree."""
    step = int(global_step)
    interval = max(int(getattr(args, "train_patch_subset_anchor_interval", 0)), 0)
    if interval > 0 and ((step + 1) % interval) == 0:
        return True, "interval"

    calib_interval = max(int(getattr(args, "surrogate_full_cloud_calib_interval", 0)), 0)
    if calib_interval > 0 and ((step + 1) % calib_interval) == 0:
        return True, "surrogate_full_cloud_calib"

    prob = float(getattr(args, "train_subtree_full_cloud_prob", 0.0))
    if prob <= 0.0:
        return False, "subtree"
    if prob >= 1.0:
        return True, "probability"

    seed_text = f"{cache_key or ''}|anchor_step={step}|seed={int(getattr(args, 'seed', 0))}"
    seed_value = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    threshold = seed_value / float(16 ** 16)
    if threshold < prob:
        return True, "probability"
    return False, "subtree"


def build_octree_subtree_reference(pts_xyz: torch.Tensor, args, depth: Optional[int] = None) -> Dict[str, torch.Tensor]:
    if pts_xyz.ndim != 3 or pts_xyz.shape[1] != 3:
        raise ValueError(f"pts_xyz must have shape [B, 3, N], got {tuple(pts_xyz.shape)}")

    qs = _effective_subtree_qs(args)
    base_depth = max(int(depth), 1) if depth is not None else resolve_subtree_level(args)

    offsets = []
    max_levels = []
    depths = []
    for b in range(pts_xyz.shape[0]):
        pts_b = pts_xyz[b].to(torch.float32)
        finite_mask = torch.isfinite(pts_b).all(dim=0)
        if finite_mask.any():
            valid_pts = torch.nan_to_num(pts_b[:, finite_mask], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            valid_pts = pts_b.new_zeros((3, 1))

        offset_b = valid_pts.amin(dim=1)
        qpts = torch.round((valid_pts - offset_b.unsqueeze(1)) / qs).to(torch.long)
        max_level = _estimate_subtree_max_level_single(valid_pts, qs)
        depth_b = max(1, min(base_depth, max_level))

        offsets.append(offset_b.to(device=pts_xyz.device, dtype=pts_xyz.dtype))
        max_levels.append(max_level)
        depths.append(depth_b)

    return {
        "offset": torch.stack(offsets, dim=0),
        "max_level": torch.tensor(max_levels, device=pts_xyz.device, dtype=torch.long),
        "depth": torch.tensor(depths, device=pts_xyz.device, dtype=torch.long),
        "qs": float(qs),
    }


def assign_octree_subtree_keys(pts_xyz: torch.Tensor, subtree_ref: Dict[str, torch.Tensor]) -> torch.Tensor:
    if pts_xyz.ndim != 3 or pts_xyz.shape[1] != 3:
        raise ValueError(f"pts_xyz must have shape [B, 3, N], got {tuple(pts_xyz.shape)}")
    if subtree_ref is None:
        raise ValueError("subtree_ref must not be None.")

    offsets = subtree_ref["offset"].to(device=pts_xyz.device, dtype=torch.float32)
    max_levels = subtree_ref["max_level"].to(device=pts_xyz.device, dtype=torch.long)
    depths = subtree_ref["depth"].to(device=pts_xyz.device, dtype=torch.long)
    qs = max(float(subtree_ref.get("qs", 1.0)), 1e-9)

    if offsets.shape[0] != pts_xyz.shape[0]:
        if offsets.shape[0] == 1 and pts_xyz.shape[0] > 1:
            offsets = offsets.expand(pts_xyz.shape[0], -1)
            max_levels = max_levels.expand(pts_xyz.shape[0])
            depths = depths.expand(pts_xyz.shape[0])
        else:
            raise ValueError(
                f"subtree_ref batch size {offsets.shape[0]} does not match pts batch size {pts_xyz.shape[0]}"
            )

    keys = []
    for b in range(pts_xyz.shape[0]):
        pts_b = pts_xyz[b].to(torch.float32)
        safe_pts = torch.nan_to_num(pts_b, nan=0.0, posinf=0.0, neginf=0.0)
        qpts = torch.round((safe_pts - offsets[b].unsqueeze(1)) / qs).to(torch.long).clamp_min_(0)
        max_level = int(max_levels[b].item())
        depth = int(depths[b].item())
        shift = max(max_level - depth, 0)
        if shift > 0:
            qpts = torch.bitwise_right_shift(qpts, shift)
        grid = 1 << depth
        qpts = qpts.clamp_(0, grid - 1)
        key_b = qpts[0] + grid * (qpts[1] + grid * qpts[2])
        invalid_mask = ~torch.isfinite(pts_xyz[b]).all(dim=0)
        if invalid_mask.any():
            key_b = key_b.masked_fill(invalid_mask, -1)
        keys.append(key_b)
    return torch.stack(keys, dim=0)


def subtree_membership_mask(unit_keys: torch.Tensor, selected_keys: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if unit_keys is None or selected_keys is None:
        return None
    selected_keys = selected_keys.to(device=unit_keys.device, dtype=unit_keys.dtype).reshape(-1)
    if selected_keys.numel() == 0:
        return torch.zeros_like(unit_keys, dtype=torch.bool)
    return unit_keys.unsqueeze(-1).eq(selected_keys.view(1, 1, -1)).any(dim=-1)


def sorted_unique_subtree_keys(unit_keys: torch.Tensor) -> torch.Tensor:
    if unit_keys.ndim == 2:
        if unit_keys.shape[0] != 1:
            raise ValueError("sorted_unique_subtree_keys currently expects batch size 1.")
        unit_keys = unit_keys[0]
    valid = unit_keys[unit_keys >= 0]
    if valid.numel() == 0:
        return unit_keys.new_empty((0,), dtype=unit_keys.dtype)
    return torch.unique(valid, sorted=True)


def build_subtree_index_map(unit_keys: torch.Tensor):
    if unit_keys.ndim == 2:
        if unit_keys.shape[0] != 1:
            raise ValueError("build_subtree_index_map currently expects batch size 1.")
        unit_keys = unit_keys[0]
    if unit_keys.ndim != 1:
        raise ValueError(f"unit_keys must be 1D or [1, N], got {tuple(unit_keys.shape)}")

    valid_mask = unit_keys >= 0
    valid_keys = unit_keys[valid_mask]
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_keys.numel() == 0:
        return unit_keys.new_empty((0,), dtype=unit_keys.dtype), []

    sorted_keys, order = torch.sort(valid_keys)
    sorted_idx = valid_idx.index_select(0, order)
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)

    index_lists = []
    start = 0
    for count in counts.detach().cpu().tolist():
        end = start + int(count)
        index_lists.append(sorted_idx[start:end])
        start = end
    return unique_keys, index_lists


def _cpu_long(value):
    return torch.as_tensor(value, dtype=torch.long, device="cpu")


def _decode_subtree_key(subtree_key: int, depth: int):
    grid = 1 << max(int(depth), 1)
    key = int(subtree_key)
    x = key % grid
    y = (key // grid) % grid
    z = key // (grid * grid)
    return torch.tensor([x, y, z], dtype=torch.long)


def _path_from_cell(cell_xyz, depth: int):
    depth = max(int(depth), 1)
    cell = _cpu_long(cell_xyz).reshape(3)
    path = []
    for bit_pos in range(depth - 1, -1, -1):
        child = int(((cell[0] >> bit_pos) & 1) * 4 + ((cell[1] >> bit_pos) & 1) * 2 + ((cell[2] >> bit_pos) & 1))
        path.append(child)
    return tuple(path)


def _path_to_text(path):
    if not path:
        return "root"
    return "/".join(f"{int(child)}:{int(child):03b}" for child in path)


def _cell_from_path(path):
    xyz = torch.zeros((3,), dtype=torch.long)
    for child in path:
        child = int(child)
        xyz = xyz * 2
        xyz[0] += (child >> 2) & 1
        xyz[1] += (child >> 1) & 1
        xyz[2] += child & 1
    return xyz


def _occupancy_code(children):
    code = 0
    for child in children:
        code |= 1 << int(child)
    return int(code)


def _morton_keys(coords, depth: int):
    coords = _cpu_long(coords).reshape(-1, 3)
    depth = max(int(depth), 1)
    keys = torch.zeros((coords.shape[0],), dtype=torch.long)
    for bit_pos in range(depth):
        shift = depth - 1 - bit_pos
        child = ((coords[:, 0] >> shift) & 1) * 4 + ((coords[:, 1] >> shift) & 1) * 2 + ((coords[:, 2] >> shift) & 1)
        keys = keys * 8 + child
    return keys


def _global_node_id(path, global_depth: int):
    depth = len(path)
    if depth <= 0:
        morton = 0
    else:
        morton = int(_morton_keys(_cell_from_path(path).view(1, 3), depth)[0].item())
    # 深さを上位側に混ぜ、同じMorton値でも階層が違えば別ノードとして扱う。
    return int(depth) * int(1 << (3 * max(int(global_depth), 1))) + morton


def _path_from_morton_code(code: int, depth: int):
    path = []
    code = int(code)
    for bit_pos in range(max(int(depth), 0) - 1, -1, -1):
        path.append((code >> (3 * bit_pos)) & 7)
    return tuple(path)


def _child_map_from_coords(global_coords, global_depth: int):
    coords = _cpu_long(global_coords).reshape(-1, 3)
    global_depth = max(int(global_depth), 1)
    child_map = {}
    if coords.numel() == 0:
        return child_map
    coords = torch.unique(coords, dim=0, sorted=True)
    prefix_codes = torch.zeros((coords.shape[0],), dtype=torch.long)
    for level in range(global_depth):
        bit_pos = global_depth - 1 - level
        child = ((coords[:, 0] >> bit_pos) & 1) * 4 + ((coords[:, 1] >> bit_pos) & 1) * 2 + ((coords[:, 2] >> bit_pos) & 1)
        pairs = torch.unique(torch.stack([prefix_codes, child.to(torch.long)], dim=1), dim=0, sorted=True)
        for prefix_code, child_id in pairs.tolist():
            prefix = _path_from_morton_code(prefix_code, level)
            child_map.setdefault(prefix, set()).add(int(child_id))
        prefix_codes = prefix_codes * 8 + child.to(torch.long)
    return child_map


def _node_geometry_from_path(path, global_depth: int):
    depth = len(path)
    cell = _cell_from_path(path)
    remaining = max(int(global_depth) - int(depth), 0)
    span = 1 << remaining
    origin = cell * span
    bbox_min = origin.clone()
    bbox_max = origin + span - 1
    return origin, bbox_min, bbox_max


def _build_single_subtree_tree(subtree_key, point_idx, subtree_ref, global_coords, global_depth: int):
    depth = int(subtree_ref["depth"][0].detach().cpu().item())
    global_depth = max(int(global_depth), depth)
    qs = float(subtree_ref.get("qs", 1.0))
    offset = subtree_ref["offset"][0].detach().to("cpu", dtype=torch.float32)
    cell = _decode_subtree_key(int(subtree_key), depth)
    path = _path_from_cell(cell, depth)
    point_idx_cpu = point_idx.detach().to("cpu", dtype=torch.long)
    selected_coords = global_coords.index_select(0, point_idx_cpu)
    child_map = _child_map_from_coords(selected_coords, global_depth)
    node_paths = [
        node_path
        for node_path in child_map.keys()
        if len(node_path) >= depth and node_path[:depth] == path
    ]
    node_paths = sorted(node_paths, key=lambda item: (len(item), item))
    path_to_idx = {node_path: idx for idx, node_path in enumerate(node_paths)}

    child_masks = torch.zeros((len(node_paths), 8), dtype=torch.bool)
    child_indices = torch.full((len(node_paths), 8), -1, dtype=torch.long)
    parent_indices = torch.full((len(node_paths),), -1, dtype=torch.long)
    occupancy_codes = torch.zeros((len(node_paths),), dtype=torch.long)
    node_depths = torch.zeros((len(node_paths),), dtype=torch.long)
    node_origins = torch.zeros((len(node_paths), 3), dtype=torch.long)
    node_bbox_min = torch.zeros((len(node_paths), 3), dtype=torch.long)
    node_bbox_max = torch.zeros((len(node_paths), 3), dtype=torch.long)
    global_node_ids = torch.zeros((len(node_paths),), dtype=torch.long)

    for idx, node_path in enumerate(node_paths):
        children = sorted(child_map.get(node_path, set()))
        occupancy_codes[idx] = _occupancy_code(children)
        node_depths[idx] = len(node_path)
        global_node_ids[idx] = _global_node_id(node_path, global_depth)
        origin, bbox_min, bbox_max = _node_geometry_from_path(node_path, global_depth)
        node_origins[idx] = origin
        node_bbox_min[idx] = bbox_min
        node_bbox_max[idx] = bbox_max
        if node_path[:-1] in path_to_idx:
            parent_indices[idx] = path_to_idx[node_path[:-1]]
        for child in children:
            child_masks[idx, int(child)] = True
            child_path = node_path + (int(child),)
            if child_path in path_to_idx:
                child_indices[idx, int(child)] = path_to_idx[child_path]

    subtree_origin, subtree_bbox_min, subtree_bbox_max = _node_geometry_from_path(path, global_depth)
    return {
        "subtree_key": int(subtree_key),
        "subtree_depth": int(depth),
        "subtree_path": _path_to_text(path),
        "subtree_depth_in_full_octree": int(depth),
        "remaining_depth_to_leaf": int(max(global_depth - depth, 0)),
        "global_offset": offset,
        "global_qs": float(qs),
        "global_depth": int(global_depth),
        "subtree_global_origin": subtree_origin,
        "subtree_global_bbox_min": subtree_bbox_min,
        "subtree_global_bbox_max": subtree_bbox_max,
        "node_ids": torch.arange(len(node_paths), dtype=torch.long),
        "global_node_ids": global_node_ids,
        "node_depths": node_depths,
        "node_origins": node_origins,
        "node_bbox_min": node_bbox_min,
        "node_bbox_max": node_bbox_max,
        "parent_indices": parent_indices,
        "child_indices": child_indices,
        "child_masks": child_masks,
        "occupancy_codes": occupancy_codes,
        "leaf_point_indices": point_idx_cpu,
        "global_voxel_coords": selected_coords,
        "global_morton_keys": _morton_keys(selected_coords, global_depth),
    }


def _build_single_full_octree_context(subtree_key, subtree_ref, full_child_map, global_depth: int):
    depth = int(subtree_ref["depth"][0].detach().cpu().item())
    global_depth = max(int(global_depth), depth)
    qs = float(subtree_ref.get("qs", 1.0))
    offset = subtree_ref["offset"][0].detach().to("cpu", dtype=torch.float32)
    cell = _decode_subtree_key(int(subtree_key), depth)
    path = _path_from_cell(cell, depth)
    parent_path = path[:-1]
    selected_child = path[-1] if path else 0
    parent_children = sorted(full_child_map.get(parent_path, set()))
    sibling_paths = [parent_path + (child,) for child in parent_children if int(child) != int(selected_child)]
    sibling_codes = [_occupancy_code(full_child_map.get(sibling_path, set())) for sibling_path in sibling_paths]
    sibling_masks = []
    for code in sibling_codes:
        sibling_masks.append([(int(code) >> child) & 1 for child in range(8)])
    subtree_origin, subtree_bbox_min, subtree_bbox_max = _node_geometry_from_path(path, global_depth)
    ancestor_paths = [path[:level] for level in range(0, len(path))]
    ancestor_codes = [_occupancy_code(full_child_map.get(item, set())) for item in ancestor_paths]
    return {
        "root_to_subtree_path": [_path_to_text(path[:level]) for level in range(0, len(path) + 1)],
        "ancestor_node_ids": torch.tensor([_global_node_id(item, global_depth) for item in ancestor_paths], dtype=torch.long),
        "ancestor_paths": [_path_to_text(item) for item in ancestor_paths],
        "ancestor_occupancy_codes": torch.tensor(ancestor_codes, dtype=torch.long),
        "parent_node_id": int(_global_node_id(parent_path, global_depth)),
        "parent_path": _path_to_text(parent_path),
        "parent_occupancy_code": int(_occupancy_code(parent_children)),
        "sibling_node_ids": torch.tensor([_global_node_id(item, global_depth) for item in sibling_paths], dtype=torch.long),
        "sibling_paths": [_path_to_text(item) for item in sibling_paths],
        "sibling_occupancy_codes": torch.tensor(sibling_codes, dtype=torch.long),
        "sibling_child_masks": torch.tensor(sibling_masks, dtype=torch.bool) if sibling_masks else torch.zeros((0, 8), dtype=torch.bool),
        "selected_subtree_key": int(subtree_key),
        "selected_subtree_path": _path_to_text(path),
        "selected_subtree_global_bbox_min": subtree_bbox_min,
        "selected_subtree_global_bbox_max": subtree_bbox_max,
        "global_offset": offset,
        "global_qs": float(qs),
        "global_depth": int(global_depth),
    }


def _build_group_octree_metadata(pts_xyz, subtree_ref, all_groups):
    if subtree_ref is None or not all_groups:
        return {}, {}, {}
    if pts_xyz.ndim != 3 or pts_xyz.shape[0] != 1:
        return {}, {}, {}
    qs = max(float(subtree_ref.get("qs", 1.0)), 1e-9)
    offset = subtree_ref["offset"][0].to(device=pts_xyz.device, dtype=torch.float32)
    global_depth = int(subtree_ref["max_level"][0].detach().cpu().item())
    pts_b = torch.nan_to_num(pts_xyz[0].detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    global_coords = torch.round((pts_b - offset.view(3, 1)) / qs).to(torch.long).transpose(0, 1).contiguous().detach().to("cpu")
    global_coords = global_coords.clamp_min(0)
    full_child_map = _child_map_from_coords(global_coords, global_depth)
    subtree_trees = {}
    full_octree_contexts = {}
    group_meta = {}
    for subtree_key, point_idx in all_groups:
        key = int(subtree_key)
        tree = _build_single_subtree_tree(key, point_idx, subtree_ref, global_coords, global_depth)
        context = _build_single_full_octree_context(key, subtree_ref, full_child_map, global_depth)
        subtree_trees[key] = tree
        full_octree_contexts[key] = context
        group_meta[key] = {
            "subtree_key": key,
            "subtree_path": tree["subtree_path"],
            "point_count": int(point_idx.numel()),
            "node_count": int(tree["occupancy_codes"].numel()),
            "single_child_count": int((tree["child_masks"].sum(dim=1) == 1).sum().item()),
            "global_depth": int(global_depth),
            "global_offset": tree["global_offset"].tolist() if hasattr(tree["global_offset"], "tolist") else tree["global_offset"],
            "subtree_depth": int(tree["subtree_depth"]),
            "local_depth": int(tree["subtree_depth"]),
        }
    return subtree_trees, full_octree_contexts, group_meta

def build_selected_group_octree_metadata(pts_xyz, subtree_ref, selected_groups):
    """
    選択済みSubtreeだけに対して subtree_tree / full_octree_context / group_meta を作る。
    all_groups全体ではなく、train.pyで確定した selected_groups だけを渡すこと。
    """
    return _build_group_octree_metadata(
        pts_xyz,
        subtree_ref,
        selected_groups,
    )

def build_octree_subtree_groups_with_retry(
    pts_xyz: torch.Tensor,
    args,
    requested_depth: int,
    min_points: int = 1,
    *,
    allow_largest_fallback: bool = True,
):
    requested_depth = max(int(requested_depth), 1)
    min_points = max(int(min_points), 1)
    fallback = None
    retry_count = 0

    for depth in range(requested_depth, 0, -1):
        subtree_ref = build_octree_subtree_reference(pts_xyz, args, depth=int(depth))
        unit_keys = assign_octree_subtree_keys(pts_xyz, subtree_ref)
        unique_keys, index_lists = build_subtree_index_map(unit_keys)
        all_groups = [
            (int(subtree_key.detach().cpu()), point_idx)
            for subtree_key, point_idx in zip(unique_keys, index_lists)
        ]
        eligible_groups = [
            (subtree_key, point_idx)
            for subtree_key, point_idx in all_groups
            if int(point_idx.numel()) >= min_points
        ]

        if all_groups:
            largest_group = max(all_groups, key=lambda item: int(item[1].numel()))
            fallback_points = int(fallback["groups"][0][1].numel()) if fallback is not None else -1
            if fallback is None or int(largest_group[1].numel()) > fallback_points:
                fallback = {
                    "subtree_ref": subtree_ref,
                    "unique_keys": unique_keys,
                    "index_lists": index_lists,
                    "all_groups": all_groups,
                    "groups": [largest_group],
                    "eligible_groups": [],
                    "eligible_count": 0,
                    "retry_count": int(retry_count),
                    "depth": int(depth),
                    "requested_depth": int(requested_depth),
                    "selection_reason": "min_points_miss_fallback_largest",
                    "total_subtree_count": int(unique_keys.numel()),

                    # metadataはここでは作らない
                    # 実際に選ばれたSubtreeだけtrain.py側で作る
                    "subtree_trees": {},
                    "full_octree_contexts": {},
                    "group_meta": {},
                }

        if eligible_groups:
            return {
                "subtree_ref": subtree_ref,
                "unique_keys": unique_keys,
                "index_lists": index_lists,
                "all_groups": all_groups,
                "groups": eligible_groups,
                "eligible_groups": eligible_groups,
                "eligible_count": int(len(eligible_groups)),
                "retry_count": int(retry_count),
                "depth": int(depth),
                "requested_depth": int(requested_depth),
                "selection_reason": "depth_retry" if int(depth) != int(requested_depth) else "none",
                "total_subtree_count": int(unique_keys.numel()),

                # metadataはここでは作らない
                # 実際に選ばれたSubtreeだけtrain.py側で作る
                "subtree_trees": {},
                "full_octree_contexts": {},
                "group_meta": {},
            }
        retry_count += 1

    if fallback is not None and allow_largest_fallback:
        fallback = dict(fallback)
        fallback["retry_count"] = int(retry_count)
        return fallback

    empty = pts_xyz.new_empty((0,), dtype=torch.long)
    return {
        "subtree_ref": None,
        "unique_keys": empty,
        "index_lists": [],
        "all_groups": [],
        "groups": [],
        "eligible_groups": [],
        "eligible_count": 0,
        "retry_count": int(retry_count),
        "depth": int(requested_depth),
        "requested_depth": int(requested_depth),
        "selection_reason": "no_valid_subtree",
        "total_subtree_count": 0,
        "subtree_trees": {},
        "full_octree_contexts": {},
        "group_meta": {},
    }


def select_octree_subtree_keys(sorted_keys: torch.Tensor, global_step: int, args) -> torch.Tensor:
    if sorted_keys.ndim != 1:
        raise ValueError(f"sorted_keys must be 1D, got {tuple(sorted_keys.shape)}")
    total = int(sorted_keys.numel())
    if total <= 0:
        raise ValueError("Subtree subset selection received zero subtrees.")

    sampling = str(getattr(args, "train_patch_subset_sampling", "coverage_cycle")).strip().lower()
    if sampling != "coverage_cycle":
        raise ValueError(f"Unsupported train subtree subset sampling mode: {sampling}")

    per_step = int(getattr(args, "train_patch_subset_patches_per_step", total))
    if per_step < 1:
        raise ValueError("train_patch_subset_patches_per_step must be >= 1")
    if per_step >= total:
        return sorted_keys

    stride = max(int(math.ceil(total / float(per_step))), 1)
    offset = int(global_step) % stride
    selected_positions = []
    seen = set()
    for class_shift in range(stride):
        start = (offset + class_shift) % stride
        for idx in range(start, total, stride):
            if idx in seen:
                continue
            selected_positions.append(idx)
            seen.add(idx)
            if len(selected_positions) >= per_step:
                break
        if len(selected_positions) >= per_step:
            break

    if len(selected_positions) < per_step:
        for idx in range(total):
            if idx in seen:
                continue
            selected_positions.append(idx)
            if len(selected_positions) >= per_step:
                break

    selected_idx = torch.tensor(selected_positions, device=sorted_keys.device, dtype=torch.long)
    return sorted_keys.index_select(0, selected_idx)
