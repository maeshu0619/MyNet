#!/usr/bin/env python
"""Phase2N voxel-level context-aware RD optimizer.

Research-only.  This script reuses the existing Phase2J/K/M actual
SparsePCGC encode/decode and quality evaluation helpers, and adds only
voxel-level candidate generators.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.utils.config.args import parse_pugan_args
from models.utils.data.dataset import load_ply
from models.utils.loss.actual_encoder import build_actual_encoder

from tools.context_aware_where_probe import (
    _coord_key_setup,
    _coords_to_xyz,
    _lookup_occupied,
    _neighbor_count,
    _parent_info,
    _parse_csv_text,
    _safe_float,
    _unique_coords,
)
from tools.phase2_rdo_beam_probe import (
    _coords_signature,
    _high_bit_voxel_scores,
    _hist_json_from_tensor,
    _select_aggressive_high_bit_mask,
    _write_csv,
)
from tools.phase2k_context_rewrite_rd_probe import _candidate_to_coords
from tools.phase2m_multi_operator_context_rewriter import _eval_decoded_row


BASELINE_CANDIDATES = (
    "block_only",
    "high_bit_raw_prune",
    "move_snap_context_projection",
)
DEFAULT_CANDIDATES = (
    "block_only",
    "high_bit_raw_prune",
    "move_snap_context_projection",
    "high_bit_voxel_prune_veto_fill",
    "high_bit_relocate_swap",
    "parent_pattern_projection_v2",
    "hybrid_voxel_context_beam",
)


def _prepare_args(_cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2N voxel context RD optimizer"),
            time.strftime("%Y%m%d"),
            time.strftime("%H%M%S"),
        )
    finally:
        sys.argv = old_argv
    for attr, value in {
        "compress": "SparsePCGC",
        "compression_loss_backend": "sparsepcgc_surrogate",
        "sparsepcgc_worker_gpu_stats": False,
        "enable_sparsepcgc_occupancy_debug": True,
        "enable_sparsepcgc_exact_occupancy_teacher": False,
        "sparsepcgc_exact_occupancy_interval": 1,
    }.items():
        setattr(args, attr, value)
    return args


def _debug_for_mask(
    *,
    name: str,
    family: str,
    coords: torch.Tensor,
    mask: torch.Tensor,
    base_stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    edit_unit_type: str,
    selection_unit_type: str,
    extra: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    _score, bit_score, depth_score = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    selected_bits = bit_score[mask]
    selected_depth = depth_score[mask]
    parent = torch.div(coords, 2, rounding_mode="floor")
    block = torch.div(coords, int(block_size), rounding_mode="floor")
    unique_parent, inverse_parent = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    unique_block, inverse_block = torch.unique(block, dim=0, sorted=True, return_inverse=True)
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    selected_neigh = neigh[mask]
    edited_count = int(mask.sum().item())
    parent_counts = torch.bincount(inverse_parent[mask], minlength=int(unique_parent.shape[0])) if edited_count else torch.zeros((int(unique_parent.shape[0]),), device=coords.device, dtype=torch.long)
    block_counts_all = torch.bincount(inverse_block, minlength=int(unique_block.shape[0])).to(dtype=torch.float32).clamp_min(1.0)
    block_counts_edit = torch.bincount(inverse_block[mask], minlength=int(unique_block.shape[0])).to(dtype=torch.float32) if edited_count else torch.zeros((int(unique_block.shape[0]),), device=coords.device)
    debug = {
        "candidate_family": family,
        "candidate_variant": name,
        "canonical_method": name,
        "operation_type": "voxel_context_rewrite",
        "edit_unit_type": edit_unit_type,
        "selection_unit_type": selection_unit_type,
        "actual_prune_ratio": float(edited_count) / max(float(coords.shape[0]), 1.0),
        "budget_reached": bool(edited_count >= max(int(math.ceil(float(coords.shape[0]) * float(budget) * 0.98)), 1)),
        "candidate_pool_size": int(pool),
        "actual_edited_voxel_count": int(edited_count),
        "affected_parent_count": int((parent_counts > 0).sum().item()),
        "affected_block_count": int((block_counts_edit > 0).sum().item()),
        "same_parent_edit_max": int(parent_counts.max().item()) if edited_count else 0,
        "same_block_edit_ratio_max": float((block_counts_edit / block_counts_all).max().item()) if edited_count else 0.0,
        "selected_bit_sum": float(selected_bits.sum().item()) if edited_count else 0.0,
        "selected_bit_mean": float(selected_bits.mean().item()) if edited_count else 0.0,
        "selected_depth_hist_json": _hist_json_from_tensor(selected_depth[selected_depth >= 0]),
        "density_drop_mean": float((1.0 / (selected_neigh + 1.0)).mean().item()) if edited_count else 0.0,
        "hole_risk": float((selected_neigh <= 2).to(dtype=torch.float32).mean().item()) if edited_count else 0.0,
        "boundary_removed_ratio": float((selected_neigh <= 2).to(dtype=torch.float32).mean().item()) if edited_count else 0.0,
        "prune_count": int(edited_count),
        "add_count": 0,
        "move_count": 0,
        "swap_count": 0,
        "operation_counts_json": json.dumps([name], sort_keys=True),
        "pattern_simplification_gain": 0.0,
        "move_distance_mean": 0.0,
        "target_low_bit_score_mean": 0.0,
    }
    if extra:
        debug.update(dict(extra))
    return debug


def _high_bit_voxel_prune_veto_fill(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    target = min(max(int(math.ceil(float(coords.shape[0]) * float(budget))), 0), max(int(coords.shape[0]) - 1, 0))
    score, bit_score, _depth = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    valid = torch.isfinite(score) & (bit_score > 0)
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    parent = torch.div(coords, 2, rounding_mode="floor")
    block = torch.div(coords, int(block_size), rounding_mode="floor")
    unique_parent, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    unique_block, inverse_block = torch.unique(block, dim=0, sorted=True, return_inverse=True)
    block_counts = torch.bincount(inverse_block, minlength=int(unique_block.shape[0])).to(dtype=torch.float32).clamp_min(1.0)
    parent_selected = torch.zeros((int(unique_parent.shape[0]),), device=coords.device, dtype=torch.long)
    block_selected = torch.zeros((int(unique_block.shape[0]),), device=coords.device, dtype=torch.float32)
    parent_pop_voxel = parent_pop.index_select(0, inverse_parent).to(dtype=torch.float32)
    selected = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    vetoed = {"hole": 0, "thin_parent": 0, "same_parent": 0, "same_block": 0}
    order = torch.argsort(torch.where(valid, score, score.new_full(score.shape, -1e12)), descending=True)
    parent_cap = 2
    block_ratio_cap = max(float(budget) * 2.5, 0.08)
    for idx in order.tolist():
        if int(selected.sum().item()) >= target:
            break
        if not bool(valid[idx].item()):
            continue
        p = int(inverse_parent[idx].item())
        b = int(inverse_block[idx].item())
        if float(neigh[idx].item()) <= 1.0:
            vetoed["hole"] += 1
            continue
        if float(parent_pop_voxel[idx].item()) <= 1.0:
            vetoed["thin_parent"] += 1
            continue
        if int(parent_selected[p].item()) >= parent_cap:
            vetoed["same_parent"] += 1
            continue
        if float((block_selected[b] + 1.0) / block_counts[b]) > block_ratio_cap:
            vetoed["same_block"] += 1
            continue
        selected[idx] = True
        parent_selected[p] += 1
        block_selected[b] += 1.0
    # Minimal fill: if veto is too strict, fill from next high-bit voxels while
    # still avoiding true isolated samples.
    if int(selected.sum().item()) < target:
        for idx in order.tolist():
            if int(selected.sum().item()) >= target:
                break
            if bool(selected[idx].item()) or not bool(valid[idx].item()):
                continue
            if float(neigh[idx].item()) <= 0.0:
                continue
            selected[idx] = True
    cand = torch.unique(coords[~selected].to(dtype=torch.long), dim=0, sorted=True)
    debug = _debug_for_mask(
        name="high_bit_voxel_prune_veto_fill",
        family="voxel_prune_veto_fill",
        coords=coords,
        mask=selected,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        edit_unit_type="voxel",
        selection_unit_type="high_bit_node",
        extra={
            "veto_reason_counts_json": json.dumps(vetoed, sort_keys=True),
            "fill_from_lower_bit_count": max(0, int(selected.sum().item()) - (target - sum(vetoed.values()))),
        },
    )
    return cand, selected, debug


def _offsets_for_radius(radius_mode: str, device) -> torch.Tensor:
    offsets = []
    max_r = 2 if str(radius_mode) == "2" else 1
    limit = 4.01 if str(radius_mode) == "2" else (2.01 if str(radius_mode) == "sqrt2" else 1.01)
    for dx in range(-max_r, max_r + 1):
        for dy in range(-max_r, max_r + 1):
            for dz in range(-max_r, max_r + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                d2 = dx * dx + dy * dy + dz * dz
                if float(d2) <= limit:
                    offsets.append((dx, dy, dz))
    return torch.tensor(offsets, device=device, dtype=torch.long)


def _relocate_swap(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
    same_parent_only: bool = False,
    radius_mode: str = "sqrt2",
    name: str = "high_bit_relocate_swap",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    target = min(max(int(math.ceil(float(coords.shape[0]) * float(budget))), 0), max(int(coords.shape[0]) - 1, 0))
    score, bit_score, _depth = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    valid = torch.isfinite(score) & (bit_score > 0)
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    keys, occupied = _coord_key_setup(coords)
    offsets = _offsets_for_radius(radius_mode, coords.device)
    selected = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    add_targets: List[torch.Tensor] = []
    used_target_keys = set()
    move_dists = []
    target_support = []
    order = torch.argsort(torch.where(valid, score, score.new_full(score.shape, -1e12)), descending=True)
    for idx in order.tolist():
        if len(add_targets) >= target:
            break
        if not bool(valid[idx].item()) or bool(selected[idx].item()):
            continue
        src = coords[idx].reshape(1, 3)
        candidates = src + offsets
        occ = _lookup_occupied(keys(candidates), occupied)
        candidates = candidates[~occ]
        if candidates.numel() <= 0:
            continue
        if same_parent_only:
            candidates = candidates[(torch.div(candidates, 2, rounding_mode="floor") == torch.div(src, 2, rounding_mode="floor")).all(dim=1)]
            if candidates.numel() <= 0:
                continue
        # Target should be supported by nearby occupied voxels.  This keeps the
        # operation as a local surface/context rewrite rather than random Add.
        support = []
        for cand in candidates:
            query = cand.reshape(1, 3) + offsets
            support.append(int(_lookup_occupied(keys(query), occupied).sum().item()))
        support_t = torch.tensor(support, device=coords.device, dtype=torch.float32)
        best_order = torch.argsort(support_t, descending=True)
        picked = None
        for j in best_order.tolist():
            if float(support_t[j].item()) < 2.0:
                continue
            key = int(keys(candidates[j].reshape(1, 3))[0].item())
            if key in used_target_keys:
                continue
            picked = candidates[j]
            used_target_keys.add(key)
            target_support.append(float(support_t[j].item()))
            move_dists.append(float(torch.norm((picked - src.reshape(3)).to(dtype=torch.float32)).item()))
            break
        if picked is None:
            continue
        selected[idx] = True
        add_targets.append(picked.reshape(1, 3))
    if add_targets:
        target_coords = torch.cat(add_targets, dim=0).to(dtype=torch.long)
        cand = torch.unique(torch.cat([coords[~selected], target_coords], dim=0).to(dtype=torch.long), dim=0, sorted=True)
    else:
        cand = coords.clone()
    debug = _debug_for_mask(
        name=name,
        family="relocate_swap" if not same_parent_only else "parent_pattern_projection",
        coords=coords,
        mask=selected,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        edit_unit_type="voxel_swap",
        selection_unit_type="high_bit_node" if not same_parent_only else "high_bit_parent_pattern",
        extra={
            "prune_count": int(selected.sum().item()),
            "add_count": int(len(add_targets)),
            "move_count": int(len(add_targets)),
            "swap_count": int(len(add_targets)),
            "operation_counts_json": json.dumps([name], sort_keys=True),
            "move_distance_mean": float(sum(move_dists) / len(move_dists)) if move_dists else 0.0,
            "target_low_bit_score_mean": float(sum(target_support) / len(target_support)) if target_support else 0.0,
            "pattern_simplification_gain": float(sum(target_support) / len(target_support)) if target_support else 0.0,
            "actual_prune_ratio": float(selected.sum().item()) / max(float(coords.shape[0]), 1.0),
            "budget_reached": bool(int(selected.sum().item()) >= max(int(math.ceil(float(coords.shape[0]) * float(budget) * 0.98)), 1)),
        },
    )
    return cand, selected, debug


def _hybrid_voxel_context_beam(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    # Cheap two-candidate beam: compare a conservative veto-fill path with a
    # relocate path, then prefer the one with higher selected bit per risk.
    cand_a, mask_a, dbg_a = _high_bit_voxel_prune_veto_fill(coords, base_stats, budget=budget * 0.75, pool=pool, block_size=block_size)
    cand_b, mask_b, dbg_b = _relocate_swap(coords, base_stats, budget=budget * 0.75, pool=pool, block_size=block_size)
    score_a = _safe_float(dbg_a.get("selected_bit_sum"), 0.0) - 500.0 * _safe_float(dbg_a.get("hole_risk"), 0.0)
    score_b = _safe_float(dbg_b.get("selected_bit_sum"), 0.0) - 500.0 * _safe_float(dbg_b.get("hole_risk"), 0.0)
    if score_b > score_a:
        dbg_b["candidate_variant"] = "hybrid_voxel_context_beam"
        dbg_b["candidate_family"] = "hybrid_voxel_context_beam"
        dbg_b["operation_counts_json"] = json.dumps(["high_bit_relocate_swap"], sort_keys=True)
        return cand_b, mask_b, dbg_b
    dbg_a["candidate_variant"] = "hybrid_voxel_context_beam"
    dbg_a["candidate_family"] = "hybrid_voxel_context_beam"
    dbg_a["operation_counts_json"] = json.dumps(["high_bit_voxel_prune_veto_fill"], sort_keys=True)
    return cand_a, mask_a, dbg_a


def _candidate_to_coords_n(
    *,
    candidate: str,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    if candidate in {"block_only", "high_bit_raw_prune", "move_snap_context_projection"}:
        cand, mask, debug = _candidate_to_coords(
            candidate=candidate,
            coords=coords,
            base_stats=base_stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            seed=seed,
        )
        debug = dict(debug)
        if candidate == "block_only":
            debug.setdefault("edit_unit_type", "block")
            debug.setdefault("selection_unit_type", "block")
        elif candidate == "high_bit_raw_prune":
            debug.setdefault("edit_unit_type", "voxel")
            debug.setdefault("selection_unit_type", "high_bit_node")
        else:
            debug.setdefault("edit_unit_type", "voxel_merge")
            debug.setdefault("selection_unit_type", "high_bit_node")
        return cand, mask, debug
    if candidate == "high_bit_voxel_prune_veto_fill":
        return _high_bit_voxel_prune_veto_fill(coords, base_stats, budget=budget, pool=pool, block_size=block_size)
    if candidate == "high_bit_relocate_swap":
        return _relocate_swap(coords, base_stats, budget=budget, pool=pool, block_size=block_size, radius_mode="sqrt2")
    if candidate == "parent_pattern_projection_v2":
        return _relocate_swap(
            coords,
            base_stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            same_parent_only=True,
            radius_mode="sqrt2",
            name="parent_pattern_projection_v2",
        )
    if candidate == "hybrid_voxel_context_beam":
        return _hybrid_voxel_context_beam(coords, base_stats, budget=budget, pool=pool, block_size=block_size)
    raise ValueError(f"unknown Phase2N candidate: {candidate}")


def _add_phase2n_columns(row: Dict[str, object], debug: Mapping[str, object], *, coords: torch.Tensor, cand_coords: torch.Tensor) -> Dict[str, object]:
    row["budget_ratio"] = row.get("requested_budget_ratio", "")
    row["D1_PSNR"] = row.get("processed_decoded_d1_psnr", "")
    row["Chamfer"] = row.get("processed_decoded_chamfer", "")
    row["point_to_plane_PSNR"] = row.get("processed_decoded_d2_psnr", "")
    for key in (
        "hole_risk",
        "boundary_removed_ratio",
        "density_drop_mean",
        "edit_unit_type",
        "selection_unit_type",
        "actual_edited_voxel_count",
        "affected_parent_count",
        "affected_block_count",
        "same_parent_edit_max",
        "same_block_edit_ratio_max",
        "selected_bit_sum",
        "selected_depth_hist_json",
        "pattern_simplification_gain",
        "move_distance_mean",
        "target_low_bit_score_mean",
        "operation_counts_json",
    ):
        row[key] = debug.get(key, row.get(key, ""))
    raw = _safe_float(row.get("actual_raw_percent"), float("nan"))
    edited = max(_safe_float(debug.get("actual_edited_voxel_count"), 0.0), 1.0)
    row["single_voxel_raw_gain"] = float(raw) / float(edited) if math.isfinite(raw) else ""
    row["prune_count"] = int(_safe_float(debug.get("prune_count"), max(int(coords.shape[0]) - int(cand_coords.shape[0]), 0)))
    row["add_count"] = int(_safe_float(debug.get("add_count"), 0.0))
    row["move_count"] = int(_safe_float(debug.get("move_count"), 0.0))
    row["swap_count"] = int(_safe_float(debug.get("swap_count"), 0.0))
    return row


def _read_rows(path: str) -> List[Dict[str, object]]:
    if not path or not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_phase2n(cli: argparse.Namespace) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        base_args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2N voxel context optimizer"),
            time.strftime("%Y%m%d"),
            time.strftime("%H%M%S"),
        )
    finally:
        sys.argv = old_argv
    for attr, value in {
        "compress": "SparsePCGC",
        "compression_loss_backend": "sparsepcgc_surrogate",
        "sparsepcgc_worker_gpu_stats": False,
        "enable_sparsepcgc_occupancy_debug": True,
        "enable_sparsepcgc_exact_occupancy_teacher": False,
        "sparsepcgc_exact_occupancy_interval": 1,
    }.items():
        setattr(base_args, attr, value)
    debug_args = base_args
    debug_args.sparsepcgc_skip_decode = True
    max_pool = max(int(float(x)) for x in _parse_csv_text(cli.pools))
    debug_args.sparsepcgc_occupancy_debug_topk_final = int(max_pool)
    debug_args.sparsepcgc_occupancy_debug_topk_per_layer = max(1024, min(int(max_pool), 8192))
    decode_args = copy.copy(base_args)
    decode_args.sparsepcgc_skip_decode = False
    decode_args.enable_sparsepcgc_occupancy_debug = False
    Path(cli.decoded_dir).mkdir(parents=True, exist_ok=True)
    decode_args.sparsepcgc_decoded_copy_dir = str(cli.decoded_dir)
    rows = _read_rows(cli.output_csv) if bool(cli.append_output) else []
    eval_cache: Dict[str, Dict[str, object]] = {}
    actual_eval_count = 0
    cache_hit_count = 0
    duplicate_skip = 0
    debug_encoder = build_actual_encoder(debug_args)
    decode_encoder = build_actual_encoder(decode_args)
    try:
        for file_idx, file_path in enumerate(cli.files):
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, base_args)
            sequence = Path(file_path).parent.name
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, base_args)
            base_stats = debug_encoder.encode_bits(base_xyz)
            baseline_stats = decode_encoder.encode_bits(base_xyz)
            base_bits = float(baseline_stats.get("bit", base_stats.get("bit", 0.0)))
            decoded_gt_path = str(baseline_stats.get("decoded_copy_path", ""))
            from tools.phase2m_multi_operator_context_rewriter import _eval_decoded_row
            from tools.phase2_rdo_beam_probe import _coord_match_ratio_from_paths, _quality_from_paths
            baseline_count, baseline_match, baseline_lossless = _coord_match_ratio_from_paths(file_path, decoded_gt_path) if decoded_gt_path else (0, float("nan"), False)
            baseline_quality = _quality_from_paths(
                file_path,
                decoded_gt_path,
                formal_max_points=int(cli.quality_max_points),
                normal_max_points=int(cli.normal_max_points),
                pc_error_path=str(cli.pc_error_path),
                use_pc_error=bool(cli.use_pc_error),
            ) if decoded_gt_path else {}
            for pool in [int(float(x)) for x in _parse_csv_text(cli.pools)]:
                for budget in [float(x) for x in _parse_csv_text(cli.budgets)]:
                    seen = set()
                    for candidate in _parse_csv_text(cli.candidates):
                        t0 = time.time()
                        cand_coords, _mask, debug = _candidate_to_coords_n(
                            candidate=candidate,
                            coords=coords,
                            base_stats=base_stats,
                            budget=float(budget),
                            pool=int(pool),
                            block_size=int(cli.block_size),
                            seed=int(cli.seed) + file_idx,
                        )
                        gen_time = time.time() - t0
                        sig = _coords_signature(cand_coords)
                        if sig in seen:
                            duplicate_skip += 1
                            continue
                        seen.add(sig)
                        row, hit, te, tq = _eval_decoded_row(
                            file_path=str(file_path),
                            sequence=sequence,
                            frame_id=frame_id,
                            candidate_name=candidate,
                            edit_sequence=candidate,
                            budget=float(budget),
                            pool=int(pool),
                            coords=coords,
                            cand_coords=cand_coords,
                            meta=meta,
                            args=base_args,
                            base_bits=base_bits,
                            baseline_stats=baseline_stats,
                            baseline_quality=baseline_quality,
                            decoded_gt_path=decoded_gt_path,
                            baseline_count=baseline_count,
                            baseline_match=baseline_match,
                            baseline_lossless=baseline_lossless,
                            decode_encoder=decode_encoder,
                            debug=debug,
                            cli=cli,
                            eval_cache=eval_cache,
                        )
                        actual_eval_count += 0 if hit else 1
                        cache_hit_count += int(hit)
                        row = _add_phase2n_columns(row, debug, coords=coords, cand_coords=cand_coords)
                        row["candidate_name"] = candidate
                        row["requested_budget_ratio"] = float(budget)
                        row["budget_ratio"] = float(budget)
                        row["candidate_pool_size"] = int(pool)
                        row["actual_eval_count"] = actual_eval_count
                        row["cache_hit_count"] = cache_hit_count
                        row["skipped_duplicate_count"] = duplicate_skip
                        row["candidate_generate_time"] = gen_time
                        row["encode_decode_time"] = te
                        row["quality_eval_time"] = tq
                        rows.append(row)
                        _write_csv(cli.output_csv, rows)
                    print(json.dumps({
                        "phase2n": True,
                        "sequence": sequence,
                        "frame": frame_id,
                        "budget": budget,
                        "pool": pool,
                        "rows": len(rows),
                        "actual_eval_count": actual_eval_count,
                        "cache_hit_count": cache_hit_count,
                        "duplicate_skip": duplicate_skip,
                    }, sort_keys=True), flush=True)
    finally:
        for enc in (debug_encoder, decode_encoder):
            close = getattr(enc, "close", None)
            if callable(close):
                close()
    _write_csv(cli.output_csv, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2N voxel-level context-aware RD optimizer")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--budgets", default="0.030,0.050")
    parser.add_argument("--pools", default="131072")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--quality-max-points", type=int, default=800)
    parser.add_argument("--normal-max-points", type=int, default=800)
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--pc-error-path", default="/home/maejima/MasterEx/compress/octree/SparsePCGC/extension/pc_error_d")
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2n_decoded")
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2n(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
