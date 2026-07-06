#!/usr/bin/env python
"""Phase2Q probability-guided SparsePCGC context editing probe.

Research-only script.  It reuses existing Phase2 decoded RD evaluation and
candidate builders, then adds probability-guided candidates based on
SparsePCGC occupancy debug (`prob_true`, `bits`, depth/context summaries).
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
    _coord_match_ratio_from_paths,
    _coords_signature,
    _hist_json_from_tensor,
    _mask_for_scaled_node,
    _parse_json_payload,
    _quality_from_paths,
    _write_csv,
)
from tools.phase2m_multi_operator_context_rewriter import _eval_decoded_row
from tools.phase2p_high_leverage_micro_edit_probe import _candidate_to_coords_p, _pattern_debug
from tools.phase2o_bad_pattern_inversion_probe import _added_and_parent_metrics, _metric_debug


DEFAULT_CANDIDATES = (
    "codec_only",
    "high_bit_raw_prune",
    "snap_to_existing_only",
    "low_prob_occupied_prune",
    "low_prob_snap_to_existing",
    "entropy_reducing_add",
    "parent_pattern_entropy_projection",
    "probability_guided_micro_beam",
)


def _prepare_args(_cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2Q probability-guided context edit"),
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


def _json_dict(value: object) -> Dict[str, float]:
    if isinstance(value, Mapping):
        out: Dict[str, float] = {}
        for k, v in value.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                pass
        return out
    try:
        parsed = json.loads(str(value or "{}"))
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {}


def _json_delta(before: object, after: object) -> str:
    b = _json_dict(before)
    a = _json_dict(after)
    keys = sorted(set(b) | set(a), key=str)
    return json.dumps({k: float(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys}, sort_keys=True)


def _top_node_headroom(stats: Mapping[str, object], *, edit_ratios=(0.01, 0.02, 0.03)) -> Dict[str, object]:
    nodes = [
        r
        for r in _parse_json_payload(stats.get("sparsepcgc_top_high_bit_nodes_json", ""), [])
        if bool(r.get("occupied", False))
    ]
    total_bits = max(_safe_float(stats.get("sparsepcgc_estimated_occupancy_bits"), 0.0), 1e-6)
    bits = sorted((float(r.get("bits", 0.0)) for r in nodes), reverse=True)
    out: Dict[str, object] = {"top_high_bit_occupied_node_count": int(len(bits))}
    for ratio in edit_ratios:
        k = min(max(int(math.ceil(len(bits) * float(ratio))), 0), len(bits))
        s = float(sum(bits[:k]))
        out[f"top_{int(ratio * 100)}p_high_bit_sum"] = s
        out[f"top_{int(ratio * 100)}p_high_bit_headroom_percent"] = s / total_bits * 100.0
    return out


def _prob_guided_voxel_scores(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    max_pool: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(coords.shape[0])
    score = torch.full((n,), float("-inf"), device=coords.device, dtype=torch.float32)
    bit_score = torch.zeros((n,), device=coords.device, dtype=torch.float32)
    prob_true_score = torch.ones((n,), device=coords.device, dtype=torch.float32)
    depth_score = torch.full((n,), -1, device=coords.device, dtype=torch.long)
    rows = [
        r
        for r in _parse_json_payload(base_stats.get("sparsepcgc_top_high_bit_nodes_json", ""), [])
        if bool(r.get("occupied", False))
    ][: max(int(max_pool), 0)]
    for r in rows:
        bits = float(r.get("bits", 0.0))
        prob_true = float(r.get("prob_true", 1.0))
        depth = int(r.get("depth", 0))
        mask = _mask_for_scaled_node(coords, r.get("coord", []), depth)
        if int(mask.sum().item()) <= 0:
            continue
        node_score = bits * max(1.0 - prob_true, 1e-3)
        update = mask & (node_score > score)
        score[update] = float(node_score)
        bit_score[update] = float(bits)
        prob_true_score[update] = float(prob_true)
        depth_score[update] = int(depth)
    return score, bit_score, prob_true_score, depth_score


def _select_with_parent_cap(
    score: torch.Tensor,
    valid: torch.Tensor,
    target: int,
    inverse_parent: torch.Tensor,
    *,
    parent_cap: int,
) -> torch.Tensor:
    selected = torch.zeros_like(valid, dtype=torch.bool)
    parent_counts = torch.zeros((int(inverse_parent.max().item()) + 1,), device=score.device, dtype=torch.long)
    order = torch.argsort(torch.where(valid, score, score.new_full(score.shape, -1e12)), descending=True)
    for idx in order.tolist():
        if int(selected.sum().item()) >= target:
            break
        if not bool(valid[idx].item()):
            continue
        p = int(inverse_parent[idx].item())
        if int(parent_counts[p].item()) >= int(parent_cap):
            continue
        selected[idx] = True
        parent_counts[p] += 1
    return selected


def _prob_prune_candidate(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    target = min(max(int(math.ceil(float(coords.shape[0]) * float(budget))), 0), max(int(coords.shape[0]) - 1, 0))
    score, bit_score, prob_true, depth = _prob_guided_voxel_scores(coords, base_stats, max_pool=int(pool))
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    _unique_parent, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    parent_pop_v = parent_pop.index_select(0, inverse_parent).to(dtype=torch.float32)
    valid = torch.isfinite(score) & (bit_score > 0)
    if mode == "low_prob_occupied_prune":
        valid = valid & (prob_true <= 0.25) & (neigh >= 1) & (parent_pop_v >= 1)
        adjusted = score + 0.02 * neigh
        parent_cap = 2
    elif mode == "bad_pattern_safe_prune":
        valid = valid & (prob_true <= 0.40) & (neigh >= 2) & (parent_pop_v >= 2)
        adjusted = score + 0.05 * neigh + 0.10 * (parent_pop_v <= 3).to(dtype=torch.float32)
        parent_cap = 1
    else:
        valid = valid & (prob_true <= 0.45) & (neigh >= 2) & (parent_pop_v >= 2)
        adjusted = score + 0.15 * ((parent_pop_v >= 2) & (parent_pop_v <= 5)).to(dtype=torch.float32)
        parent_cap = 1
    selected = _select_with_parent_cap(adjusted, valid, target, inverse_parent, parent_cap=parent_cap)
    if int(selected.sum().item()) < target and mode == "low_prob_occupied_prune":
        fill_valid = torch.isfinite(score) & (bit_score > 0) & (~selected) & (neigh >= 1)
        selected |= _select_with_parent_cap(score, fill_valid, target - int(selected.sum().item()), inverse_parent, parent_cap=4)
    cand = torch.unique(coords[~selected].to(dtype=torch.long), dim=0, sorted=True)
    selected_prob = prob_true[selected]
    debug = _metric_debug(
        name=mode,
        family="probability_guided_prune",
        coords=coords,
        mask=selected,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "selection_unit_type": "low_prob_high_bit_node",
            "operation_counts_json": json.dumps([mode], sort_keys=True),
            "low_prob_selected_mean": float(selected_prob.mean().item()) if int(selected_prob.numel()) else 1.0,
        },
    )
    debug.update(_pattern_debug(coords, selected))
    debug["budget_reached"] = bool(int(selected.sum().item()) >= max(int(math.ceil(float(coords.shape[0]) * float(budget) * 0.98)), 1))
    debug["saturation_reason"] = "" if bool(debug["budget_reached"]) else "insufficient_low_prob_high_bit_voxels"
    return cand, selected, debug


def _low_prob_snap_to_existing(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    target = min(max(int(math.ceil(float(coords.shape[0]) * float(budget))), 0), max(int(coords.shape[0]) - 1, 0))
    score, bit_score, prob_true, _depth = _prob_guided_voxel_scores(coords, base_stats, max_pool=int(pool))
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    keys, occupied = _coord_key_setup(coords)
    offsets = torch.tensor(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        device=coords.device,
        dtype=torch.long,
    )
    valid = torch.isfinite(score) & (bit_score > 0) & (prob_true <= 0.45) & (neigh >= 1)
    order = torch.argsort(torch.where(valid, score, score.new_full(score.shape, -1e12)), descending=True)
    selected = torch.zeros((int(coords.shape[0]),), device=coords.device, dtype=torch.bool)
    move_count = 0
    for idx in order.tolist():
        if int(selected.sum().item()) >= target:
            break
        if not bool(valid[idx].item()):
            continue
        src = coords[idx].reshape(1, 3)
        targets = src + offsets
        occ = _lookup_occupied(keys(targets), occupied)
        if not bool(occ.any().item()):
            continue
        selected[idx] = True
        move_count += 1
    cand = torch.unique(coords[~selected].to(dtype=torch.long), dim=0, sorted=True)
    debug = _metric_debug(
        name="low_prob_snap_to_existing",
        family="probability_guided_snap",
        coords=coords,
        mask=selected,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "edit_unit_type": "voxel_merge",
            "selection_unit_type": "low_prob_high_bit_node",
            "move_count": int(move_count),
            "snap_count": int(move_count),
            "operation_counts_json": json.dumps(["low_prob_snap_to_existing"], sort_keys=True),
            "low_prob_selected_mean": float(prob_true[selected].mean().item()) if int(selected.sum().item()) else 1.0,
        },
    )
    debug.update(_pattern_debug(coords, selected))
    debug["budget_reached"] = bool(int(selected.sum().item()) >= max(int(math.ceil(float(coords.shape[0]) * float(budget) * 0.98)), 1))
    debug["saturation_reason"] = "" if bool(debug["budget_reached"]) else "insufficient_existing_snap_targets"
    return cand, selected, debug


def _entropy_reducing_add_candidate(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    # Conservative Add: only fill empty sibling slots in existing parents with
    # strong local support.  This tests whether Add can help without the bad
    # Phase2O signatures (new parent / isolated Add).
    target = min(max(int(math.ceil(float(coords.shape[0]) * float(budget) * 0.25)), 0), 512)
    _score, bit_score, prob_true, _depth = _prob_guided_voxel_scores(coords, base_stats, max_pool=int(pool))
    _unique_parent, _inverse_parent, _slots, occ, _patterns, parent_pop = _parent_info(coords)
    keys, occupied = _coord_key_setup(coords)
    add_targets = []
    if target > 0:
        # Focus on parents with several occupied children: Add fills a local
        # sibling gap, not a new context.
        parent_order = torch.argsort(parent_pop.to(dtype=torch.float32), descending=True)
        offsets = torch.tensor(
            [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)],
            device=coords.device,
            dtype=torch.long,
        )
        for pidx in parent_order.tolist():
            if len(add_targets) >= target:
                break
            if int(parent_pop[pidx].item()) < 3 or int(parent_pop[pidx].item()) >= 8:
                continue
            parent_coord = _unique_parent[pidx]
            for slot in range(8):
                if bool(occ[pidx, slot].item()):
                    continue
                child = torch.tensor([slot // 4, (slot // 2) % 2, slot % 2], device=coords.device, dtype=torch.long)
                cand = parent_coord * 2 + child
                if bool(_lookup_occupied(keys(cand.reshape(1, 3)), occupied)[0].item()):
                    continue
                support = int(_lookup_occupied(keys(cand.reshape(1, 3) + offsets), occupied).sum().item())
                if support < 2:
                    continue
                add_targets.append(cand.reshape(1, 3))
                if len(add_targets) >= target:
                    break
    if add_targets:
        add_coords = torch.cat(add_targets, dim=0).to(dtype=torch.long)
        cand = torch.unique(torch.cat([coords, add_coords], dim=0).to(dtype=torch.long), dim=0, sorted=True)
    else:
        add_coords = torch.empty((0, 3), device=coords.device, dtype=torch.long)
        cand = coords.clone()
    mask = torch.zeros((int(coords.shape[0]),), device=coords.device, dtype=torch.bool)
    debug = _metric_debug(
        name="entropy_reducing_add",
        family="entropy_reducing_add",
        coords=coords,
        mask=mask,
        base_stats=base_stats,
        budget=budget,
        pool=pool,
        block_size=block_size,
        extra={
            "edit_unit_type": "voxel_add",
            "selection_unit_type": "existing_parent_sibling_gap",
            "add_count": int(add_coords.shape[0]),
            "prune_count": 0,
            "operation_counts_json": json.dumps(["entropy_reducing_add"], sort_keys=True),
            "budget_reached": bool(int(add_coords.shape[0]) > 0),
            "saturation_reason": "" if int(add_coords.shape[0]) > 0 else "no_safe_entropy_add_targets",
        },
    )
    debug.update(_pattern_debug(coords, mask))
    return cand, mask, debug


def _candidate_to_coords_q(
    *,
    candidate: str,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    if candidate in {"codec_only", "high_bit_raw_prune", "snap_to_existing_only"}:
        return _candidate_to_coords_p(
            candidate=candidate,
            coords=coords,
            base_stats=base_stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            seed=seed,
        )
    if candidate == "low_prob_occupied_prune":
        return _prob_prune_candidate(coords, base_stats, budget=budget, pool=pool, block_size=block_size, mode=candidate)
    if candidate == "low_prob_snap_to_existing":
        return _low_prob_snap_to_existing(coords, base_stats, budget=budget, pool=pool, block_size=block_size)
    if candidate == "entropy_reducing_add":
        return _entropy_reducing_add_candidate(coords, base_stats, budget=budget, pool=pool, block_size=block_size)
    if candidate == "parent_pattern_entropy_projection":
        return _prob_prune_candidate(coords, base_stats, budget=budget, pool=pool, block_size=block_size, mode=candidate)
    if candidate == "probability_guided_micro_beam":
        options = []
        for name in ("low_prob_occupied_prune", "low_prob_snap_to_existing", "parent_pattern_entropy_projection"):
            cand, mask, debug = _candidate_to_coords_q(
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
        debug["candidate_variant"] = "probability_guided_micro_beam"
        debug["candidate_family"] = "probability_guided_micro_beam"
        debug["operation_counts_json"] = json.dumps([chosen], sort_keys=True)
        debug["edit_sequence"] = f"cheap_prob_beam:{chosen}"
        return cand, mask, debug
    raise ValueError(f"unknown Phase2Q candidate: {candidate}")


def _read_rows(path: str) -> list[Dict[str, object]]:
    if not path or not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _probability_row_updates(
    *,
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> Dict[str, object]:
    before_bits = _safe_float(before.get("sparsepcgc_estimated_occupancy_bits"), float("nan"))
    after_bits = _safe_float(after.get("sparsepcgc_estimated_occupancy_bits"), float("nan"))
    before_low = _safe_float(before.get("sparsepcgc_prob_true_low_count"), 0.0)
    after_low = _safe_float(after.get("sparsepcgc_prob_true_low_count"), 0.0)
    before_high = sum(_json_dict(before.get("sparsepcgc_high_bit_nodes_by_depth_json")).values())
    after_high = sum(_json_dict(after.get("sparsepcgc_high_bit_nodes_by_depth_json")).values())
    return {
        "p_true_mean_before": before.get("sparsepcgc_prob_true_mean", ""),
        "p_true_mean_after": after.get("sparsepcgc_prob_true_mean", ""),
        "p_true_delta": _safe_float(after.get("sparsepcgc_prob_true_mean"), float("nan")) - _safe_float(before.get("sparsepcgc_prob_true_mean"), float("nan")),
        "p_true_quantiles_before_json": before.get("sparsepcgc_prob_true_quantiles_json", ""),
        "p_true_quantiles_after_json": after.get("sparsepcgc_prob_true_quantiles_json", ""),
        "bit_each_quantiles_before_json": before.get("sparsepcgc_bit_each_quantiles_json", ""),
        "bit_each_quantiles_after_json": after.get("sparsepcgc_bit_each_quantiles_json", ""),
        "occupancy_accuracy_before": before.get("sparsepcgc_occupancy_accuracy_at_0p5", ""),
        "occupancy_accuracy_after": after.get("sparsepcgc_occupancy_accuracy_at_0p5", ""),
        "occupied_recall_before": before.get("sparsepcgc_occupied_recall_at_0p5", ""),
        "occupied_recall_after": after.get("sparsepcgc_occupied_recall_at_0p5", ""),
        "empty_accuracy_before": before.get("sparsepcgc_empty_accuracy_at_0p5", ""),
        "empty_accuracy_after": after.get("sparsepcgc_empty_accuracy_at_0p5", ""),
        "total_estimated_bits_before": before_bits,
        "total_estimated_bits_after": after_bits,
        "estimated_bits_delta": after_bits - before_bits,
        "low_p_true_count_before": before_low,
        "low_p_true_count_after": after_low,
        "low_p_true_count_delta": after_low - before_low,
        "high_bit_symbol_count_before": before_high,
        "high_bit_symbol_count_after": after_high,
        "high_bit_symbol_count_delta": after_high - before_high,
        "bits_by_depth_delta_json": _json_delta(before.get("sparsepcgc_bits_by_depth_json"), after.get("sparsepcgc_bits_by_depth_json")),
        "child_pattern_delta_json": _json_delta(
            {str(r.get("pattern")): float(r.get("bits", 0.0)) for r in _parse_json_payload(before.get("sparsepcgc_bits_by_child_pattern_topk_json", ""), [])},
            {str(r.get("pattern")): float(r.get("bits", 0.0)) for r in _parse_json_payload(after.get("sparsepcgc_bits_by_child_pattern_topk_json", ""), [])},
        ),
        "bits_by_parent_popcount_before_json": before.get("sparsepcgc_bits_by_parent_popcount_json", ""),
        "bits_by_parent_popcount_after_json": after.get("sparsepcgc_bits_by_parent_popcount_json", ""),
        "bits_by_child_pattern_topk_before_json": before.get("sparsepcgc_bits_by_child_pattern_topk_json", ""),
        "bits_by_child_pattern_topk_after_json": after.get("sparsepcgc_bits_by_child_pattern_topk_json", ""),
        "bits_by_block_topk_before_json": before.get("sparsepcgc_bits_by_block_topk_json", ""),
        "bits_by_block_topk_after_json": after.get("sparsepcgc_bits_by_block_topk_json", ""),
    }


def run_phase2q(cli: argparse.Namespace) -> int:
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
    prob_cache: Dict[str, Dict[str, object]] = {}
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
            headroom = _top_node_headroom(base_stats)
            for pool in [int(float(x)) for x in _parse_csv_text(cli.pools)]:
                for budget in [float(x) for x in _parse_csv_text(cli.budgets)]:
                    seen = set()
                    for candidate in _parse_csv_text(cli.candidates):
                        t0 = time.time()
                        cand_coords, mask, debug = _candidate_to_coords_q(
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
                        prob_t0 = time.time()
                        if sig in prob_cache:
                            cand_prob_stats = prob_cache[sig]
                        else:
                            cand_prob_stats = debug_encoder.encode_bits(_coords_to_xyz(cand_coords, meta, args))
                            prob_cache[sig] = dict(cand_prob_stats)
                        prob_time = time.time() - prob_t0
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
                        row["actual_edit_ratio"] = _safe_float(row.get("actual_prune_ratio"), 0.0) + (
                            _safe_float(debug.get("add_count"), 0.0) / max(float(coords.shape[0]), 1.0)
                        )
                        row["raw_gain_per_edit"] = (-_safe_float(row.get("actual_raw_percent"), 0.0)) / max(row["actual_edit_ratio"] * float(coords.shape[0]), 1.0)
                        row["D1_PSNR"] = row.get("processed_decoded_d1_psnr", "")
                        row["Chamfer"] = row.get("processed_decoded_chamfer", "")
                        row["point_to_plane_PSNR"] = row.get("processed_decoded_d2_psnr", "")
                        for key, value in debug.items():
                            row.setdefault(key, value)
                        row.update(_added_and_parent_metrics(coords, cand_coords, block_size=int(cli.block_size)))
                        row["new_parent_count"] = row.get("created_new_parent_count", 0)
                        row["isolated_add_count"] = row.get("isolated_added_voxel_count", 0)
                        row["unsupported_add_count"] = row.get("isolated_added_voxel_count", 0)
                        row.update(_probability_row_updates(before=base_stats, after=cand_prob_stats))
                        row.update(headroom)
                        row["candidate_generate_time"] = gen_time
                        row["probability_debug_time"] = prob_time
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
                                "phase2q": True,
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
    parser = argparse.ArgumentParser(description="Phase2Q probability-guided context edit probe")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--budgets", default="0.010,0.020,0.030")
    parser.add_argument("--pools", default="131072")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--quality-max-points", type=int, default=600)
    parser.add_argument("--normal-max-points", type=int, default=600)
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--pc-error-path", default="/home/maejima/MasterEx/compress/octree/SparsePCGC/extension/pc_error_d")
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2q_decoded")
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2q(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
