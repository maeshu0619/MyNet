#!/usr/bin/env python
"""Phase2O bad-pattern inversion probe.

Research-only script.  It reuses Phase2N candidate generators and Phase2M/J
decoded RD evaluation, then adds bad-pattern metrics and a few inverse
candidate generators.
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
from typing import Dict, Mapping, Sequence, Tuple

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
    _write_csv,
)
from tools.phase2m_multi_operator_context_rewriter import _eval_decoded_row
from tools.phase2n_voxel_context_rd_optimizer import _add_phase2n_columns, _candidate_to_coords_n


DEFAULT_CANDIDATES = (
    "block_only",
    "high_bit_raw_prune",
    "move_snap_context_projection",
    "high_bit_voxel_prune_veto_fill",
    "high_bit_relocate_swap",
    "parent_pattern_projection_v2",
    "hybrid_voxel_context_beam",
    "no_new_occupied_snap",
    "bad_pattern_avoid_prune",
    "inverse_bad_context_prune",
    "pattern_simplify_only",
)


def _prepare_args(_cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2O bad-pattern inversion"),
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


def _metric_debug(
    *,
    name: str,
    family: str,
    coords: torch.Tensor,
    mask: torch.Tensor,
    base_stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    extra: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    _score, bit_score, depth_score = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    selected_bits = bit_score[mask]
    selected_depth = depth_score[mask]
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    parent = torch.div(coords, 2, rounding_mode="floor")
    block = torch.div(coords, int(block_size), rounding_mode="floor")
    unique_parent, inverse_parent = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    unique_block, inverse_block = torch.unique(block, dim=0, sorted=True, return_inverse=True)
    edited = int(mask.sum().item())
    parent_counts = torch.bincount(inverse_parent[mask], minlength=int(unique_parent.shape[0])) if edited else torch.zeros((int(unique_parent.shape[0]),), device=coords.device, dtype=torch.long)
    block_counts_all = torch.bincount(inverse_block, minlength=int(unique_block.shape[0])).to(dtype=torch.float32).clamp_min(1.0)
    block_counts_edit = torch.bincount(inverse_block[mask], minlength=int(unique_block.shape[0])).to(dtype=torch.float32) if edited else torch.zeros((int(unique_block.shape[0]),), device=coords.device)
    selected_neigh = neigh[mask]
    debug = {
        "candidate_family": family,
        "candidate_variant": name,
        "canonical_method": name,
        "operation_type": "bad_pattern_inversion",
        "edit_unit_type": "voxel",
        "selection_unit_type": "high_bit_context",
        "actual_prune_ratio": float(edited) / max(float(coords.shape[0]), 1.0),
        "budget_reached": bool(edited >= max(int(math.ceil(float(coords.shape[0]) * float(budget) * 0.98)), 1)),
        "actual_edited_voxel_count": int(edited),
        "affected_parent_count": int((parent_counts > 0).sum().item()),
        "affected_block_count": int((block_counts_edit > 0).sum().item()),
        "same_parent_edit_max": int(parent_counts.max().item()) if edited else 0,
        "same_block_edit_ratio_max": float((block_counts_edit / block_counts_all).max().item()) if edited else 0.0,
        "selected_bit_sum": float(selected_bits.sum().item()) if edited else 0.0,
        "selected_depth_hist_json": _hist_json_from_tensor(selected_depth[selected_depth >= 0]),
        "hole_risk": float((selected_neigh <= 2).to(dtype=torch.float32).mean().item()) if edited else 0.0,
        "boundary_removed_ratio": float((selected_neigh <= 2).to(dtype=torch.float32).mean().item()) if edited else 0.0,
        "density_drop_mean": float((1.0 / (selected_neigh + 1.0)).mean().item()) if edited else 0.0,
        "prune_count": int(edited),
        "move_count": 0,
        "snap_count": 0,
        "add_count": 0,
        "operation_counts_json": json.dumps([name], sort_keys=True),
        "bad_pattern_score": 0.0,
        "inverse_pattern_score": 0.0,
    }
    if extra:
        debug.update(dict(extra))
    return debug


def _added_and_parent_metrics(coords: torch.Tensor, cand_coords: torch.Tensor, *, block_size: int) -> Dict[str, object]:
    keys, occupied = _coord_key_setup(coords)
    cand_keys = keys(cand_coords)
    is_original = _lookup_occupied(cand_keys, occupied)
    added = cand_coords[~is_original]
    original_parent = torch.div(coords, 2, rounding_mode="floor")
    cand_parent = torch.div(cand_coords, 2, rounding_mode="floor")
    keys_parent, occ_parent = _coord_key_setup(original_parent)
    added_parent = torch.div(added, 2, rounding_mode="floor") if int(added.shape[0]) else torch.empty((0, 3), device=coords.device, dtype=torch.long)
    created = 0
    if int(added_parent.shape[0]):
        created = int((~_lookup_occupied(keys_parent(added_parent), occ_parent)).sum().item())
    orig_unique_parent, orig_inv = torch.unique(original_parent, dim=0, sorted=True, return_inverse=True)
    cand_unique_parent, cand_inv = torch.unique(cand_parent, dim=0, sorted=True, return_inverse=True)
    orig_counts = {tuple(p.tolist()): 0 for p in orig_unique_parent}
    for p, c in zip(orig_unique_parent, torch.bincount(orig_inv, minlength=int(orig_unique_parent.shape[0]))):
        orig_counts[tuple(p.tolist())] = int(c.item())
    pop_increase = 0
    for p, c in zip(cand_unique_parent, torch.bincount(cand_inv, minlength=int(cand_unique_parent.shape[0]))):
        if int(c.item()) > int(orig_counts.get(tuple(p.tolist()), 0)):
            pop_increase += 1
    isolated_added = 0
    if int(added.shape[0]):
        neigh_added = _neighbor_count(torch.unique(torch.cat([coords, added], dim=0).to(dtype=torch.long), dim=0, sorted=True))
        # Approximate: new isolated additions are counted via support in original occupancy.
        offsets = torch.tensor([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1)], device=coords.device, dtype=torch.long)
        support = []
        for a in added:
            support.append(int(_lookup_occupied(keys(a.reshape(1,3) + offsets), occupied).sum().item()))
        isolated_added = sum(1 for s in support if s <= 1)
    return {
        "added_voxel_count": int(added.shape[0]),
        "created_new_parent_count": int(created),
        "parent_popcount_increase_count": int(pop_increase),
        "isolated_added_voxel_count": int(isolated_added),
    }


def _select_by_score(coords: torch.Tensor, adjusted: torch.Tensor, valid: torch.Tensor, target: int) -> torch.Tensor:
    selected = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    order = torch.argsort(torch.where(valid, adjusted, adjusted.new_full(adjusted.shape, -1e12)), descending=True)
    pick = order[:target]
    pick = pick[valid.index_select(0, pick)]
    selected[pick] = True
    return selected


def _inverse_bad_context_prune(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    target = min(max(int(math.ceil(float(coords.shape[0]) * float(budget))), 0), max(int(coords.shape[0]) - 1, 0))
    score, bit_score, _depth = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    _up, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    parent_pop_v = parent_pop.index_select(0, inverse_parent).to(dtype=torch.float32)
    valid = torch.isfinite(score) & (bit_score > 0)
    if mode == "bad_pattern_avoid_prune":
        # Avoid creating the same bad signatures seen in Swap/Add: isolated
        # context and thin parent damage.  Fill from the remaining high-bit pool.
        valid = valid & (neigh >= 2) & (parent_pop_v >= 2)
        adjusted = score + 0.02 * neigh
        name = "bad_pattern_avoid_prune"
        family = "bad_pattern_avoid_prune"
        inverse_bonus = (neigh >= 2).to(dtype=torch.float32) + (parent_pop_v >= 2).to(dtype=torch.float32)
    elif mode == "inverse_bad_context_prune":
        # Remove occupied voxels that already resemble the bad contexts created
        # by failed Swap/Add: low support, thin parent, high bit.
        valid = valid & (neigh <= 3)
        adjusted = score + 1.5 * (neigh <= 2).to(dtype=torch.float32) + 0.5 * (parent_pop_v <= 2).to(dtype=torch.float32)
        name = "inverse_bad_context_prune"
        family = "inverse_bad_context_prune"
        inverse_bonus = (neigh <= 2).to(dtype=torch.float32) + (parent_pop_v <= 2).to(dtype=torch.float32)
    else:
        valid = valid & (parent_pop_v >= 3) & (neigh >= 2)
        adjusted = score + 0.5 * parent_pop_v
        name = "pattern_simplify_only"
        family = "pattern_simplify_only"
        inverse_bonus = parent_pop_v
    selected = _select_by_score(coords, adjusted, valid, target)
    if int(selected.sum().item()) < target and mode == "bad_pattern_avoid_prune":
        # budget-preserving fill: allow slightly riskier high-bit voxels, but
        # never fully isolated.
        fill_valid = torch.isfinite(score) & (bit_score > 0) & (~selected) & (neigh >= 1)
        fill = _select_by_score(coords, score, fill_valid, target - int(selected.sum().item()))
        selected |= fill
    cand = torch.unique(coords[~selected].to(dtype=torch.long), dim=0, sorted=True)
    selected_bonus = inverse_bonus[selected]
    debug = _metric_debug(
        name=name,
        family=family,
        coords=coords,
        mask=selected,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "inverse_pattern_score": float(selected_bonus.mean().item()) if int(selected_bonus.numel()) else 0.0,
            "bad_pattern_score": float((1.0 / (neigh[selected] + 1.0)).mean().item()) if int(selected.sum().item()) else 0.0,
        },
    )
    return cand, selected, debug


def _candidate_to_coords_o(
    *,
    candidate: str,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    if candidate == "no_new_occupied_snap":
        cand, mask, debug = _candidate_to_coords_n(
            candidate="move_snap_context_projection",
            coords=coords,
            base_stats=base_stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            seed=seed,
        )
        debug = dict(debug)
        debug["candidate_variant"] = "no_new_occupied_snap"
        debug["candidate_family"] = "no_new_occupied_snap"
        debug["add_count"] = 0
        debug["added_voxel_count"] = 0
        return cand, mask, debug
    if candidate in {"bad_pattern_avoid_prune", "inverse_bad_context_prune", "pattern_simplify_only"}:
        return _inverse_bad_context_prune(coords, base_stats, budget=budget, pool=pool, block_size=block_size, mode=candidate)
    return _candidate_to_coords_n(
        candidate=candidate,
        coords=coords,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        seed=seed,
    )


def _read_rows(path: str):
    if not path or not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_phase2o(cli: argparse.Namespace) -> int:
    args = _prepare_args(cli)
    debug_args = args
    debug_args.sparsepcgc_skip_decode = True
    max_pool = max(int(float(x)) for x in _parse_csv_text(cli.pools))
    debug_args.sparsepcgc_occupancy_debug_topk_final = int(max_pool)
    debug_args.sparsepcgc_occupancy_debug_topk_per_layer = max(1024, min(int(max_pool), 8192))
    decode_args = copy.copy(args)
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
            coords, meta = _unique_coords(xyz, args)
            sequence = Path(file_path).parent.name
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, args)
            base_stats = debug_encoder.encode_bits(base_xyz)
            baseline_stats = decode_encoder.encode_bits(base_xyz)
            base_bits = float(baseline_stats.get("bit", base_stats.get("bit", 0.0)))
            from tools.phase2_rdo_beam_probe import _coord_match_ratio_from_paths, _quality_from_paths
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
            for pool in [int(float(x)) for x in _parse_csv_text(cli.pools)]:
                for budget in [float(x) for x in _parse_csv_text(cli.budgets)]:
                    seen = set()
                    for candidate in _parse_csv_text(cli.candidates):
                        t0 = time.time()
                        cand_coords, mask, debug = _candidate_to_coords_o(
                            candidate=candidate,
                            coords=coords,
                            base_stats=base_stats,
                            budget=budget,
                            pool=pool,
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
                            budget=budget,
                            pool=pool,
                            coords=coords,
                            cand_coords=cand_coords,
                            meta=meta,
                            args=args,
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
                        row["candidate_name"] = candidate
                        row["budget_ratio"] = float(budget)
                        row["D1_PSNR"] = row.get("processed_decoded_d1_psnr", "")
                        row["Chamfer"] = row.get("processed_decoded_chamfer", "")
                        row["point_to_plane_PSNR"] = row.get("processed_decoded_d2_psnr", "")
                        for key, value in debug.items():
                            row.setdefault(key, value)
                        row.update(_added_and_parent_metrics(coords, cand_coords, block_size=int(cli.block_size)))
                        row["bad_pattern_score"] = debug.get("bad_pattern_score", row.get("bad_pattern_score", 0.0))
                        row["inverse_pattern_score"] = debug.get("inverse_pattern_score", row.get("inverse_pattern_score", 0.0))
                        row["bad_child_pattern_topk_json"] = ""
                        row["bad_parent_pattern_topk_json"] = ""
                        row["bad_block_topk_json"] = ""
                        row["bits_by_depth_delta_json"] = ""
                        row["snap_count"] = debug.get("snap_count", debug.get("move_count", 0))
                        row["candidate_generate_time"] = gen_time
                        row["encode_decode_time"] = te
                        row["quality_eval_time"] = tq
                        row["actual_eval_count"] = actual_eval_count
                        row["cache_hit_count"] = cache_hit_count
                        row["skipped_duplicate_count"] = duplicate_skip
                        rows.append(row)
                        _write_csv(cli.output_csv, rows)
                    print(json.dumps({
                        "phase2o": True,
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
    parser = argparse.ArgumentParser(description="Phase2O bad-pattern inversion probe")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--budgets", default="0.030,0.050")
    parser.add_argument("--pools", default="131072")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--quality-max-points", type=int, default=600)
    parser.add_argument("--normal-max-points", type=int, default=600)
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--pc-error-path", default="/home/maejima/MasterEx/compress/octree/SparsePCGC/extension/pc_error_d")
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2o_decoded")
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2o(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
