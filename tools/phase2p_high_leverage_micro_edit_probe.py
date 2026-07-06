#!/usr/bin/env python
"""Phase2P high-leverage micro context edit probe.

Research-only script.  It reuses the Phase2N/O candidate generators and the
Phase2M end-to-end decoded RD evaluator, then adds small-budget candidates that
avoid the bad Add patterns found in Phase2O.
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
    _coords_to_xyz,
    _parse_csv_text,
    _parent_info,
    _safe_float,
    _unique_coords,
)
from tools.phase2_rdo_beam_probe import (
    _coord_match_ratio_from_paths,
    _coords_signature,
    _high_bit_voxel_scores,
    _hist_json_from_tensor,
    _quality_from_paths,
    _write_csv,
)
from tools.phase2m_multi_operator_context_rewriter import _eval_decoded_row
from tools.phase2n_voxel_context_rd_optimizer import _candidate_to_coords_n
from tools.phase2o_bad_pattern_inversion_probe import (
    _added_and_parent_metrics,
    _candidate_to_coords_o,
    _metric_debug,
)


DEFAULT_CANDIDATES = (
    "codec_only",
    "single_voxel_high_leverage_prune",
    "parent_popcount_reduce",
    "single_child_chain_collapse",
    "snap_to_existing_only",
    "bad_pattern_inverse_prune",
    "high_leverage_micro_beam",
)


def _prepare_args(_cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2P high-leverage micro edit"),
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


def _select_top_valid(
    score: torch.Tensor,
    valid: torch.Tensor,
    target: int,
    *,
    inverse_parent: torch.Tensor | None = None,
    parent_cap: int | None = None,
) -> torch.Tensor:
    selected = torch.zeros_like(valid, dtype=torch.bool)
    order = torch.argsort(torch.where(valid, score, score.new_full(score.shape, -1e12)), descending=True)
    parent_counts = None
    if inverse_parent is not None and parent_cap is not None:
        parent_counts = torch.zeros((int(inverse_parent.max().item()) + 1,), device=score.device, dtype=torch.long)
    for idx in order.tolist():
        if int(selected.sum().item()) >= target:
            break
        if not bool(valid[idx].item()):
            continue
        if parent_counts is not None:
            p = int(inverse_parent[idx].item())
            if int(parent_counts[p].item()) >= int(parent_cap):
                continue
            parent_counts[p] += 1
        selected[idx] = True
    return selected


def _pattern_debug(
    coords: torch.Tensor,
    selected: torch.Tensor,
) -> Dict[str, object]:
    _unique_parent, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    if int(selected.sum().item()) <= 0:
        return {
            "parent_pattern_before_after_json": "{}",
            "popcount_before_after_json": "{}",
            "pattern_simplification_gain": 0.0,
        }
    selected_parent = torch.bincount(
        inverse_parent[selected],
        minlength=int(parent_pop.shape[0]),
    ).to(device=coords.device, dtype=torch.long)
    touched = selected_parent > 0
    before = parent_pop[touched].to(dtype=torch.long)
    after = (parent_pop[touched] - selected_parent[touched]).clamp_min(0).to(dtype=torch.long)
    transitions: Dict[str, int] = {}
    for b, a in zip(before.tolist(), after.tolist()):
        key = f"{int(b)}->{int(a)}"
        transitions[key] = transitions.get(key, 0) + 1
    gain = float((before.to(dtype=torch.float32) - after.to(dtype=torch.float32)).mean().item()) if int(before.numel()) else 0.0
    return {
        "parent_pattern_before_after_json": json.dumps(transitions, sort_keys=True),
        "popcount_before_after_json": json.dumps(transitions, sort_keys=True),
        "pattern_simplification_gain": gain,
    }


def _micro_prune_candidate(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    target = min(max(int(math.ceil(float(coords.shape[0]) * float(budget))), 0), max(int(coords.shape[0]) - 1, 0))
    score, bit_score, depth_score = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    neigh = torch.zeros((int(coords.shape[0]),), device=coords.device, dtype=torch.float32)
    # Reuse Phase2O/Phase2N neighbor semantics through local adjacency count.
    from tools.context_aware_where_probe import _neighbor_count

    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    unique_parent, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    parent_pop_v = parent_pop.index_select(0, inverse_parent).to(dtype=torch.float32)
    valid = torch.isfinite(score) & (bit_score > 0)
    parent_cap = None
    selection_unit = "high_bit_node"

    if mode == "single_voxel_high_leverage_prune":
        # High bit first, but prefer edits that either empty a tiny parent or
        # reduce popcount without creating a lonely/hole-like removal.
        adjusted = score + 0.35 * (parent_pop_v <= 2).to(dtype=torch.float32) + 0.03 * neigh
        valid = valid & (neigh >= 1) & (parent_pop_v >= 1)
        parent_cap = 1
    elif mode == "parent_popcount_reduce":
        # One edit per parent: force many small parent-level simplifications
        # rather than a hidden subtree/block deletion.
        adjusted = score + 0.20 * ((parent_pop_v >= 2) & (parent_pop_v <= 4)).to(dtype=torch.float32)
        valid = valid & (neigh >= 2) & (parent_pop_v >= 2) & (parent_pop_v <= 5)
        parent_cap = 1
        selection_unit = "parent_popcount"
    elif mode == "single_child_chain_collapse":
        # Approximate single-child chains by high-bit voxels in popcount-1
        # parents with low support.  This is intentionally measured because it
        # may be rate-good but quality-risky.
        adjusted = score + 0.80 * (parent_pop_v <= 1).to(dtype=torch.float32) + 0.30 * (neigh <= 2).to(dtype=torch.float32)
        valid = valid & (parent_pop_v <= 2) & (neigh >= 1)
        parent_cap = 2
        selection_unit = "single_child_chain_proxy"
    else:
        # Invert bad Add contexts by pruning already-unsupported high-bit
        # occupied voxels.  This is deliberately more conservative than the
        # Phase2O inverse candidate.
        adjusted = score + 0.50 * (neigh <= 2).to(dtype=torch.float32) + 0.25 * (parent_pop_v <= 2).to(dtype=torch.float32)
        valid = valid & (neigh >= 1) & (neigh <= 4)
        parent_cap = 1
        selection_unit = "bad_pattern_inverse"

    selected = _select_top_valid(adjusted, valid, target, inverse_parent=inverse_parent, parent_cap=parent_cap)
    if int(selected.sum().item()) < target and mode in {"single_voxel_high_leverage_prune", "bad_pattern_inverse_prune"}:
        # Budget-preserving fill from the same high-bit pool, still avoiding
        # true zero-support removals.
        fill_valid = torch.isfinite(score) & (bit_score > 0) & (~selected) & (neigh >= 1)
        fill = _select_top_valid(score, fill_valid, target - int(selected.sum().item()))
        selected |= fill
    cand = torch.unique(coords[~selected].to(dtype=torch.long), dim=0, sorted=True)
    debug = _metric_debug(
        name=mode,
        family="high_leverage_micro",
        coords=coords,
        mask=selected,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "selection_unit_type": selection_unit,
            "operation_counts_json": json.dumps([mode], sort_keys=True),
        },
    )
    debug.update(_pattern_debug(coords, selected))
    debug["budget_reached"] = bool(int(selected.sum().item()) >= max(int(math.ceil(float(coords.shape[0]) * float(budget) * 0.98)), 1))
    debug["saturation_reason"] = "" if bool(debug["budget_reached"]) else "insufficient_valid_high_leverage_voxels"
    return cand, selected, debug


def _candidate_to_coords_p(
    *,
    candidate: str,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    if candidate == "codec_only":
        mask = torch.zeros((int(coords.shape[0]),), device=coords.device, dtype=torch.bool)
        debug = _metric_debug(
            name="codec_only",
            family="codec_only",
            coords=coords,
            mask=mask,
            base_stats=base_stats,
            budget=0.0,
            pool=pool,
            block_size=block_size,
            extra={
                "edit_unit_type": "none",
                "selection_unit_type": "none",
                "operation_counts_json": json.dumps(["codec_only"]),
                "budget_reached": True,
                "saturation_reason": "",
            },
        )
        debug.update(_pattern_debug(coords, mask))
        return coords.clone(), mask, debug
    if candidate in {
        "single_voxel_high_leverage_prune",
        "parent_popcount_reduce",
        "single_child_chain_collapse",
        "bad_pattern_inverse_prune",
    }:
        return _micro_prune_candidate(
            coords,
            base_stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            mode=candidate,
        )
    if candidate == "snap_to_existing_only":
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
        debug["candidate_variant"] = "snap_to_existing_only"
        debug["candidate_family"] = "snap_to_existing_only"
        debug["add_count"] = 0
        debug["saturation_reason"] = "" if bool(debug.get("budget_reached", False)) else "no_existing_snap_targets"
        debug.update(_pattern_debug(coords, mask))
        return cand, mask, debug
    if candidate == "high_leverage_micro_beam":
        # Cheap two-step beam without extra actual evaluations: compare three
        # no-Add micro candidates by bit gain per risk and return the best.
        options = []
        for name in ("single_voxel_high_leverage_prune", "parent_popcount_reduce", "snap_to_existing_only"):
            cand, mask, debug = _candidate_to_coords_p(
                candidate=name,
                coords=coords,
                base_stats=base_stats,
                budget=budget,
                pool=pool,
                block_size=block_size,
                seed=seed,
            )
            risk = 1.0 + 100.0 * _safe_float(debug.get("hole_risk"), 0.0) + 10.0 * _safe_float(debug.get("same_block_edit_ratio_max"), 0.0)
            cheap = _safe_float(debug.get("selected_bit_sum"), 0.0) / max(risk, 1e-6)
            options.append((cheap, cand, mask, dict(debug), name))
        options.sort(key=lambda item: item[0], reverse=True)
        _cheap, cand, mask, debug, chosen = options[0]
        debug["candidate_variant"] = "high_leverage_micro_beam"
        debug["candidate_family"] = "high_leverage_micro_beam"
        debug["operation_counts_json"] = json.dumps([chosen], sort_keys=True)
        debug["edit_sequence"] = f"cheap_beam:{chosen}"
        return cand, mask, debug
    if candidate in {
        "block_only",
        "high_bit_raw_prune",
        "move_snap_context_projection",
        "high_bit_voxel_prune_veto_fill",
        "high_bit_relocate_swap",
        "parent_pattern_projection_v2",
        "hybrid_voxel_context_beam",
        "bad_pattern_avoid_prune",
        "inverse_bad_context_prune",
        "pattern_simplify_only",
    }:
        return _candidate_to_coords_o(
            candidate=candidate,
            coords=coords,
            base_stats=base_stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            seed=seed,
        )
    raise ValueError(f"unknown Phase2P candidate: {candidate}")


def _read_rows(path: str) -> list[Dict[str, object]]:
    if not path or not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_phase2p(cli: argparse.Namespace) -> int:
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
            decoded_gt_path = str(baseline_stats.get("decoded_copy_path", ""))
            baseline_count, baseline_match, baseline_lossless = (
                _coord_match_ratio_from_paths(file_path, decoded_gt_path) if decoded_gt_path else (0, float("nan"), False)
            )
            baseline_quality = (
                _quality_from_paths(
                    file_path,
                    decoded_gt_path,
                    formal_max_points=int(cli.quality_max_points),
                    normal_max_points=int(cli.normal_max_points),
                    pc_error_path=str(cli.pc_error_path),
                    use_pc_error=bool(cli.use_pc_error),
                )
                if decoded_gt_path
                else {}
            )
            for pool in [int(float(x)) for x in _parse_csv_text(cli.pools)]:
                for budget in [float(x) for x in _parse_csv_text(cli.budgets)]:
                    seen = set()
                    for candidate in _parse_csv_text(cli.candidates):
                        t0 = time.time()
                        cand_coords, mask, debug = _candidate_to_coords_p(
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
                            edit_sequence=str(debug.get("edit_sequence", candidate)),
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
                        row["edit_budget_ratio"] = float(budget)
                        row["budget_ratio"] = float(budget)
                        row["D1_PSNR"] = row.get("processed_decoded_d1_psnr", "")
                        row["Chamfer"] = row.get("processed_decoded_chamfer", "")
                        row["point_to_plane_PSNR"] = row.get("processed_decoded_d2_psnr", "")
                        for key, value in debug.items():
                            row.setdefault(key, value)
                        row.update(_added_and_parent_metrics(coords, cand_coords, block_size=int(cli.block_size)))
                        edited = max(int(_safe_float(row.get("actual_edited_voxel_count"), int(mask.sum().item()))), 0)
                        raw = _safe_float(row.get("actual_raw_percent"), 0.0)
                        d1_drop = max(_safe_float(row.get("d1_psnr_drop"), 0.0), 0.0)
                        chamfer_delta = max(_safe_float(row.get("delta_chamfer"), 0.0), 0.0)
                        row["actual_edited_voxel_ratio"] = float(edited) / max(float(coords.shape[0]), 1.0)
                        row["raw_gain_per_edited_voxel"] = (-raw) / max(float(edited), 1.0)
                        row["raw_gain_per_quality_loss"] = (-raw) / max(float(d1_drop + 0.001 * chamfer_delta), 1e-6)
                        row["parent_pattern_before_after_json"] = row.get("parent_pattern_before_after_json", "{}")
                        row["popcount_before_after_json"] = row.get("popcount_before_after_json", "{}")
                        row["budget_reached"] = row.get("budget_reached", bool(edited >= int(math.ceil(float(coords.shape[0]) * float(budget) * 0.98))))
                        row["saturation_reason"] = row.get("saturation_reason", "")
                        row["candidate_generate_time"] = gen_time
                        row["encode_decode_time"] = te
                        row["quality_eval_time"] = tq
                        row["actual_eval_count"] = actual_eval_count
                        row["cache_hit_count"] = cache_hit_count
                        row["skipped_duplicate_count"] = duplicate_skip
                        rows.append(row)
                        _write_csv(cli.output_csv, rows)
                    print(
                        json.dumps(
                            {
                                "phase2p": True,
                                "sequence": sequence,
                                "frame": frame_id,
                                "budget": budget,
                                "pool": pool,
                                "rows": len(rows),
                                "actual_eval_count": actual_eval_count,
                                "cache_hit_count": cache_hit_count,
                                "duplicate_skip": duplicate_skip,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        for enc in (debug_encoder, decode_encoder):
            close = getattr(enc, "close", None)
            if callable(close):
                close()
    _write_csv(cli.output_csv, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2P high-leverage micro edit probe")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--budgets", default="0.005,0.010,0.015,0.020")
    parser.add_argument("--pools", default="131072")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--quality-max-points", type=int, default=600)
    parser.add_argument("--normal-max-points", type=int, default=600)
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--pc-error-path", default="/home/maejima/MasterEx/compress/octree/SparsePCGC/extension/pc_error_d")
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2p_decoded")
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2p(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
