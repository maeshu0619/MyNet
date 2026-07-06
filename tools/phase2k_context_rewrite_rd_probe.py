#!/usr/bin/env python
"""Phase2K research probe: geometry-preserving context rewriting.

This is a research-only script.  It intentionally does not import train.py or
modify the training policy.  It reuses the Phase2J end-to-end decoded RD helpers
and adds candidate generators that try to differentiate from block-only pruning.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.utils.config.args import parse_pugan_args
from models.utils.data.dataset import load_ply
from models.utils.loss.actual_encoder import build_actual_encoder
from utils.compress.ply_io import write_ascii_ply_xyz

from tools.context_aware_where_probe import (
    _coord_key_setup,
    _coords_to_xyz,
    _lookup_occupied,
    _neighbor_count,
    _parent_info,
    _parse_csv_text,
    _quota_select,
    _safe_float,
    _unique_coords,
    build_candidate_coords,
)
from tools.phase2_rdo_beam_probe import (
    _coord_match_ratio_from_paths,
    _drop_mask_against_original,
    _high_bit_voxel_scores,
    _hist_json_from_tensor,
    _phase2j_row,
    _quality_from_paths,
    _select_aggressive_high_bit_mask,
    _write_csv,
)


def _prepare_args(cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2K context rewrite probe"),
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


def _bool_budget_reached(count: int, coords: torch.Tensor, budget: float) -> bool:
    target = max(int(math.ceil(float(coords.shape[0]) * float(budget))), 1)
    return bool(int(count) >= max(int(target * 0.98), 1))


def _mask_debug(
    *,
    name: str,
    family: str,
    coords: torch.Tensor,
    drop_mask: torch.Tensor,
    base_stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    extra: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    _score, bit_score, depth_score = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    selected_bits = bit_score[drop_mask]
    selected_depth = depth_score[drop_mask]
    parent = torch.div(coords, 2, rounding_mode="floor")
    block = torch.div(coords, int(block_size), rounding_mode="floor")
    unique_parent = torch.unique(parent[drop_mask], dim=0, sorted=True) if bool(drop_mask.any().item()) else torch.empty((0, 3), device=coords.device, dtype=torch.long)
    unique_block = torch.unique(block[drop_mask], dim=0, sorted=True) if bool(drop_mask.any().item()) else torch.empty((0, 3), device=coords.device, dtype=torch.long)
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    selected_neigh = neigh[drop_mask]
    drop_count = int(drop_mask.sum().item())
    debug = {
        "candidate_family": family,
        "candidate_variant": name,
        "canonical_method": name,
        "operation_type": "context_rewrite",
        "actual_prune_ratio": float(drop_count) / max(float(coords.shape[0]), 1.0),
        "budget_reached": _bool_budget_reached(drop_count, coords, budget),
        "candidate_pool_size": int(pool),
        "prune_count": int(drop_count),
        "add_count": 0,
        "move_count": 0,
        "merge_count": 0,
        "selected_bit_sum": float(selected_bits.sum().item()) if drop_count > 0 else 0.0,
        "selected_bit_mean": float(selected_bits.mean().item()) if drop_count > 0 else 0.0,
        "selected_bit_max": float(selected_bits.max().item()) if drop_count > 0 else 0.0,
        "selected_depth_hist_json": _hist_json_from_tensor(selected_depth[selected_depth >= 0]),
        "selected_parent_count": int(unique_parent.shape[0]),
        "selected_block_count": int(unique_block.shape[0]),
        "density_drop_mean": float((1.0 / (selected_neigh + 1.0)).mean().item()) if drop_count > 0 else 0.0,
        "hole_risk": float((selected_neigh <= 2).to(dtype=torch.float32).mean().item()) if drop_count > 0 else 0.0,
        "boundary_removed_ratio": float((selected_neigh <= 2).to(dtype=torch.float32).mean().item()) if drop_count > 0 else 0.0,
    }
    if extra:
        debug.update(dict(extra))
    return debug


def _restore_risky_drops(
    coords: torch.Tensor,
    drop_mask: torch.Tensor,
    *,
    repair_fraction: float,
    base_stats: Mapping[str, object],
    pool: int,
) -> Tuple[torch.Tensor, int]:
    if repair_fraction <= 0.0 or int(drop_mask.sum().item()) <= 0:
        return drop_mask, 0
    _score, bit_score, _depth = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    idx = drop_mask.nonzero(as_tuple=False).reshape(-1)
    repair_count = min(int(math.ceil(int(idx.numel()) * float(repair_fraction))), int(idx.numel()))
    # Repair the riskiest removed points.  This is a conservative add/repair
    # proxy: final coordinates keep those surface-sensitive samples.
    risk = (1.0 / (neigh.index_select(0, idx) + 1.0)) - 0.01 * bit_score.index_select(0, idx)
    restore = idx.index_select(0, torch.argsort(risk, descending=True)[:repair_count])
    out = drop_mask.clone()
    out[restore] = False
    return out, int(repair_count)


def _context_rewrite_prune_add(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    raw_drop, raw_debug = _select_aggressive_high_bit_mask(
        coords,
        base_stats,
        budget_ratio=float(budget),
        mode="high_bit_rate_first_quality_veto",
        max_pool=int(pool),
        block_size=int(block_size),
        quality_weight=0.0,
    )
    repaired_drop, repair_count = _restore_risky_drops(
        coords,
        raw_drop,
        repair_fraction=0.12,
        base_stats=base_stats,
        pool=int(pool),
    )
    debug = _mask_debug(
        name="high_bit_context_rewrite_prune_add",
        family="context_rewrite_prune_add",
        coords=coords,
        drop_mask=repaired_drop,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "component_operations": json.dumps(["high_bit_quality_veto_prune", "local_add_repair_proxy"]),
            "component_order": "prune_then_repair",
            "add_count": int(repair_count),
            "repair_add_count": int(repair_count),
            "repair_add_ratio": float(repair_count) / max(float(coords.shape[0]), 1.0),
            "parent_pattern_target": "high_bit_context_with_surface_repair",
            "same_parent_prune_max": raw_debug.get("same_parent_prune_max", ""),
            "same_block_prune_ratio_max": raw_debug.get("same_block_prune_ratio_max", ""),
        },
    )
    return repaired_drop, debug


def _surface_preserving_decimation(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    drop, debug0 = _select_aggressive_high_bit_mask(
        coords,
        base_stats,
        budget_ratio=float(budget),
        mode="high_bit_parent_block_cap_prune",
        max_pool=int(pool),
        block_size=int(block_size),
        quality_weight=0.0,
    )
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    # Protect obvious boundary/thin samples inside the high-bit pool, then fill
    # from the lower-risk high-bit tail if possible.
    risky = (neigh <= 2) & drop
    if bool(risky.any().item()):
        repaired = drop.clone()
        repaired[risky] = False
        need = int(drop.sum().item()) - int(repaired.sum().item())
        score, bit_score, _depth = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
        valid = torch.isfinite(score) & (bit_score > 0) & (~repaired) & (~risky) & (neigh >= 3)
        order = torch.argsort(torch.where(valid, score + 0.05 * neigh, score.new_full(score.shape, -1e12)), descending=True)
        fill = order[:need]
        fill = fill[valid.index_select(0, fill)]
        repaired[fill] = True
        drop = repaired
    debug = _mask_debug(
        name="high_bit_surface_preserving_decimation",
        family="surface_preserving_decimation",
        coords=coords,
        drop_mask=drop,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "component_operations": json.dumps(["high_bit_prune", "parent_block_cap", "boundary_veto"]),
            "surface_score_mean": debug0.get("surface_score_mean", ""),
            "same_parent_prune_max": debug0.get("same_parent_prune_max", ""),
            "same_block_prune_ratio_max": debug0.get("same_block_prune_ratio_max", ""),
        },
    )
    return drop, debug


def _move_snap_projection(
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
    keys, _occ = _coord_key_setup(coords)
    all_keys = torch.unique(keys(coords), sorted=True)
    neigh_offsets = torch.tensor(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        device=coords.device,
        dtype=torch.long,
    )
    selected = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    order = torch.argsort(torch.where(valid, score, score.new_full(score.shape, -1e12)), descending=True)
    # Move/snap here means high-bit samples are snapped into already occupied
    # 1-neighbors, producing a merge instead of an isolated deletion.
    for idx in order[: min(int(order.numel()), max(int(pool), 1))].tolist():
        if int(selected.sum().item()) >= target:
            break
        if not bool(valid[idx].item()):
            continue
        c = coords[idx].reshape(1, 3)
        nk = keys(c + neigh_offsets)
        if bool(_lookup_occupied(nk, all_keys).any().item()):
            selected[idx] = True
    cand_coords = torch.unique(coords[~selected].to(dtype=torch.long), dim=0, sorted=True)
    debug = _mask_debug(
        name="move_snap_context_projection",
        family="move_snap_context_projection",
        coords=coords,
        drop_mask=selected,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "operation_type": "move_snap_merge",
            "move_count": int(selected.sum().item()),
            "merge_count": int(selected.sum().item()),
            "move_distance_mean": 1.0 if int(selected.sum().item()) > 0 else 0.0,
            "component_operations": json.dumps(["high_bit_select", "snap_to_occupied_neighbor"]),
        },
    )
    return cand_coords, selected, debug


def _block_decomposition_with_repair(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    block_coords, _mask, block_debug = build_candidate_coords(
        "block_only",
        coords,
        float(budget),
        block_size=int(block_size),
        seed=int(seed),
        max_operation_edits=max(int(coords.shape[0]), 1),
    )
    block_drop = _drop_mask_against_original(coords, torch.unique(block_coords.to(dtype=torch.long), dim=0, sorted=True))
    score, bit_score, _depth = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    valid = torch.isfinite(score) & (bit_score > 0) & block_drop
    target = max(int(math.ceil(int(block_drop.sum().item()) * 0.70)), 1)
    adjusted = torch.where(valid, score, score.new_full(score.shape, -1e12))
    order = torch.argsort(adjusted, descending=True)
    pick = order[:target]
    pick = pick[valid.index_select(0, pick)]
    drop = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    drop[pick] = True
    drop, repair_count = _restore_risky_drops(coords, drop, repair_fraction=0.10, base_stats=base_stats, pool=int(pool))
    debug = _mask_debug(
        name="block_only_decomposition_with_repair",
        family="block_decomposition_with_repair",
        coords=coords,
        drop_mask=drop,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "component_operations": json.dumps(["block_only_target_blocks", "high_bit_group_subset", "local_repair_proxy"]),
            "block_only_drop_count": int(block_drop.sum().item()),
            "group_to_block_drop_ratio": float(drop.sum().item()) / max(float(block_drop.sum().item()), 1.0),
            "add_count": int(repair_count),
            "repair_add_count": int(repair_count),
            "diagnostic_block_only_budget_reached": block_debug.get("budget_reached", ""),
        },
    )
    return drop, debug


def _single_child_chain_simplify(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    target = max(int(math.ceil(float(coords.shape[0]) * float(budget) * 0.60)), 1)
    score, bit_score, _depth = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    _up, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    parent_pop_per_voxel = parent_pop.index_select(0, inverse_parent).to(dtype=torch.float32)
    valid = torch.isfinite(score) & (bit_score > 0) & (parent_pop_per_voxel <= 2)
    adjusted = torch.where(valid, score + (3.0 - parent_pop_per_voxel), score.new_full(score.shape, -1e12))
    order = torch.argsort(adjusted, descending=True)
    pick = order[:target]
    pick = pick[valid.index_select(0, pick)]
    drop = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    drop[pick] = True
    drop, repair_count = _restore_risky_drops(coords, drop, repair_fraction=0.15, base_stats=base_stats, pool=int(pool))
    debug = _mask_debug(
        name="single_child_chain_simplify_or_repair",
        family="single_child_chain_simplify_or_repair",
        coords=coords,
        drop_mask=drop,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "component_operations": json.dumps(["high_bit_low_parent_pop_prune", "chain_repair_proxy"]),
            "repair_add_count": int(repair_count),
            "add_count": int(repair_count),
        },
    )
    return drop, debug


def _candidate_to_coords(
    *,
    candidate: str,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    if candidate == "block_only":
        block_coords, _mask, debug = build_candidate_coords(
            "block_only",
            coords,
            float(budget),
            block_size=int(block_size),
            seed=int(seed),
            max_operation_edits=max(int(coords.shape[0]), 1),
        )
        cand = torch.unique(block_coords.to(dtype=torch.long), dim=0, sorted=True)
        drop = _drop_mask_against_original(coords, cand)
        debug = _mask_debug(
            name="block_only",
            family="block_only_baseline",
            coords=coords,
            drop_mask=drop,
            base_stats=base_stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            extra=dict(debug),
        )
        return cand, drop, debug
    if candidate == "high_bit_raw_prune":
        drop, debug = _select_aggressive_high_bit_mask(
            coords,
            base_stats,
            budget_ratio=float(budget),
            mode="high_bit_raw_prune",
            max_pool=int(pool),
            block_size=int(block_size),
        )
        cand = torch.unique(coords[~drop].to(dtype=torch.long), dim=0, sorted=True)
        debug = {**dict(debug), "candidate_variant": candidate, "canonical_method": candidate}
        return cand, drop, debug
    if candidate == "high_bit_context_rewrite_prune_add":
        drop, debug = _context_rewrite_prune_add(coords, base_stats, budget=budget, pool=pool, block_size=block_size)
        return torch.unique(coords[~drop].to(dtype=torch.long), dim=0, sorted=True), drop, debug
    if candidate == "high_bit_surface_preserving_decimation":
        drop, debug = _surface_preserving_decimation(coords, base_stats, budget=budget, pool=pool, block_size=block_size)
        return torch.unique(coords[~drop].to(dtype=torch.long), dim=0, sorted=True), drop, debug
    if candidate == "move_snap_context_projection":
        return _move_snap_projection(coords, base_stats, budget=budget, pool=pool, block_size=block_size)
    if candidate == "block_only_decomposition_with_repair":
        drop, debug = _block_decomposition_with_repair(coords, base_stats, budget=budget, pool=pool, block_size=block_size, seed=seed)
        return torch.unique(coords[~drop].to(dtype=torch.long), dim=0, sorted=True), drop, debug
    if candidate == "single_child_chain_simplify_or_repair":
        drop, debug = _single_child_chain_simplify(coords, base_stats, budget=budget, pool=pool, block_size=block_size)
        return torch.unique(coords[~drop].to(dtype=torch.long), dim=0, sorted=True), drop, debug
    raise ValueError(f"unknown Phase2K candidate: {candidate}")


def _row_extra(debug: Mapping[str, object]) -> Dict[str, object]:
    keys = [
        "candidate_family",
        "candidate_variant",
        "operation_type",
        "component_operations",
        "component_order",
        "move_count",
        "merge_count",
        "repair_add_count",
        "repair_add_ratio",
        "density_drop_mean",
        "hole_risk",
        "boundary_removed_ratio",
        "same_parent_prune_max",
        "same_block_prune_ratio_max",
        "block_only_drop_count",
        "group_to_block_drop_ratio",
        "parent_pattern_target",
    ]
    return {key: debug.get(key, "") for key in keys}


def run_phase2k(cli: argparse.Namespace) -> int:
    base_args = _prepare_args(cli)
    debug_args = copy.copy(base_args)
    debug_args.sparsepcgc_skip_decode = True
    max_pool = max(int(float(x)) for x in _parse_csv_text(cli.pools))
    debug_args.sparsepcgc_occupancy_debug_topk_final = int(max_pool)
    debug_args.sparsepcgc_occupancy_debug_topk_per_layer = max(1024, min(int(max_pool), 8192))
    decode_args = copy.copy(base_args)
    decode_args.sparsepcgc_skip_decode = False
    decode_args.enable_sparsepcgc_occupancy_debug = False
    Path(cli.decoded_dir).mkdir(parents=True, exist_ok=True)
    decode_args.sparsepcgc_decoded_copy_dir = str(cli.decoded_dir)

    budgets = [float(x) for x in _parse_csv_text(cli.budgets)]
    pools = [int(float(x)) for x in _parse_csv_text(cli.pools)]
    candidates = list(_parse_csv_text(cli.candidates))
    rows: List[Dict[str, object]] = []
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
            baseline_count, baseline_match, baseline_lossless = _coord_match_ratio_from_paths(file_path, decoded_gt_path) if decoded_gt_path else (0, float("nan"), False)
            baseline_quality = _quality_from_paths(
                file_path,
                decoded_gt_path,
                formal_max_points=int(cli.quality_max_points),
                normal_max_points=int(cli.normal_max_points),
                pc_error_path=str(cli.pc_error_path),
                use_pc_error=bool(cli.use_pc_error),
            ) if decoded_gt_path else {}
            for pool in pools:
                for budget in budgets:
                    for candidate in candidates:
                        cand_coords, drop_mask, debug = _candidate_to_coords(
                            candidate=candidate,
                            coords=coords,
                            base_stats=base_stats,
                            budget=float(budget),
                            pool=int(pool),
                            block_size=int(cli.block_size),
                            seed=int(cli.seed) + file_idx,
                        )
                        cand_xyz = _coords_to_xyz(cand_coords, meta, base_args)
                        processed_stats = decode_encoder.encode_bits(cand_xyz)
                        processed_path = str(processed_stats.get("decoded_copy_path", ""))
                        processed_count, processed_match, processed_lossless = _coord_match_ratio_from_paths(file_path, processed_path) if processed_path else (0, float("nan"), False)
                        with tempfile.TemporaryDirectory(prefix="phase2k_pre_") as tmp:
                            pre_path = Path(tmp) / "processed_pre.ply"
                            write_ascii_ply_xyz(pre_path, cand_xyz.detach().to("cpu").numpy().astype(np.float64, copy=False))
                            pre_quality = _quality_from_paths(
                                file_path,
                                pre_path,
                                formal_max_points=int(cli.quality_max_points),
                                normal_max_points=int(cli.normal_max_points),
                                pc_error_path=str(cli.pc_error_path),
                                use_pc_error=bool(cli.use_pc_error),
                            )
                        processed_quality = _quality_from_paths(
                            file_path,
                            processed_path,
                            formal_max_points=int(cli.quality_max_points),
                            normal_max_points=int(cli.normal_max_points),
                            pc_error_path=str(cli.pc_error_path),
                            use_pc_error=bool(cli.use_pc_error),
                        ) if processed_path else {}
                        processed_stats = dict(processed_stats)
                        processed_stats["prune_count"] = int(drop_mask.sum().item())
                        row = _phase2j_row(
                            file_path=str(file_path),
                            sequence=sequence,
                            frame_id=frame_id,
                            candidate_name=candidate,
                            budget=float(budget),
                            pool=int(pool),
                            base_bits=base_bits,
                            baseline_stats=baseline_stats,
                            processed_stats=processed_stats,
                            pre_quality=pre_quality,
                            baseline_quality=baseline_quality,
                            processed_quality=processed_quality,
                            debug=debug,
                            decoded_gt_path=decoded_gt_path,
                            decoded_processed_path=processed_path,
                            baseline_decode_count=baseline_count,
                            processed_decode_count=processed_count,
                            baseline_match_ratio=baseline_match,
                            processed_match_ratio=processed_match,
                            baseline_lossless=baseline_lossless,
                            processed_lossless=processed_lossless,
                        )
                        row.update(_row_extra(debug))
                        rows.append(row)
                        _write_csv(cli.output_csv, rows)
                    print(json.dumps({
                        "phase2k": True,
                        "sequence": sequence,
                        "frame": frame_id,
                        "budget": budget,
                        "pool": pool,
                        "rows": len(rows),
                    }, sort_keys=True), flush=True)
    finally:
        for enc in (debug_encoder, decode_encoder):
            close = getattr(enc, "close", None)
            if callable(close):
                close()
    _write_csv(cli.output_csv, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2K geometry-preserving context rewriting RD probe")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--budgets", default="0.030,0.050")
    parser.add_argument("--pools", default="131072")
    parser.add_argument(
        "--candidates",
        default="block_only,high_bit_raw_prune,high_bit_context_rewrite_prune_add,high_bit_surface_preserving_decimation,move_snap_context_projection,block_only_decomposition_with_repair",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quality-max-points", type=int, default=1500)
    parser.add_argument("--normal-max-points", type=int, default=1500)
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--pc-error-path", default="/home/maejima/MasterEx/compress/octree/SparsePCGC/extension/pc_error_d")
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2k_decoded")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2k(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
