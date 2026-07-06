#!/usr/bin/env python
"""Phase2S counterfactual context sensitivity probe.

Research-only script.  It extracts SparsePCGC false-negative / false-positive
high-bit nodes, applies tiny local counterfactual edits, evaluates
probability/estimated-bit deltas for every generated candidate, and measures
actual SparsePCGC raw bits only for the top estimated-bit improvements.
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
from typing import Dict, Iterable, Mapping, Sequence, Tuple

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
from tools.phase2_rdo_beam_probe import _coords_signature, _parse_json_payload, _write_csv
from tools.phase2q_probability_guided_context_edit import _probability_row_updates
from tools.phase2r_probability_flip_relocation_oracle import (
    _source_coord_for_node,
    _target_coord_for_node,
)


EDIT_TYPES = (
    "prune_mispredicted_occupied",
    "add_support_near_false_negative",
    "remove_context_near_false_positive",
    "move_within_same_parent",
    "move_within_same_grandparent",
    "snap_or_merge",
    "child_pattern_one_flip",
)


def _prepare_args(_cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2S counterfactual context sensitivity"),
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


def _read_rows(path: str) -> list[Dict[str, object]]:
    if not path or not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _node_rows(stats: Mapping[str, object], key: str, *, top_n: int) -> list[Dict[str, object]]:
    rows = _parse_json_payload(stats.get(key, ""), [])
    rows = [dict(r) for r in rows]
    rows.sort(key=lambda r: float(r.get("bits", 0.0)), reverse=True)
    return rows[: max(int(top_n), 0)]


def _coord_tuple(t: torch.Tensor) -> Tuple[int, int, int]:
    return tuple(int(v) for v in t.reshape(-1)[:3].tolist())


def _parent_pattern_maps(coords: torch.Tensor):
    unique_parent, inverse_parent, slots, occ, patterns, parent_pop = _parent_info(coords)
    pattern_map = {_coord_tuple(p): int(patterns[i].item()) for i, p in enumerate(unique_parent)}
    pop_map = {_coord_tuple(p): int(parent_pop[i].item()) for i, p in enumerate(unique_parent)}
    return pattern_map, pop_map


def _pattern_for_coord(coord: torch.Tensor, pattern_map: Mapping[Tuple[int, int, int], int]) -> Tuple[str, str]:
    parent = torch.div(coord.reshape(3), 2, rounding_mode="floor")
    pkey = _coord_tuple(parent)
    return ",".join(str(v) for v in pkey), str(int(pattern_map.get(pkey, 0)))


def _safe_add_target_near(
    source: torch.Tensor,
    coords: torch.Tensor,
    *,
    same_parent: bool = True,
    radius: int = 1,
) -> torch.Tensor | None:
    keys, occupied = _coord_key_setup(coords)
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    offsets.append((dx, dy, dz))
    off_t = torch.tensor(offsets, device=coords.device, dtype=torch.long)
    candidates = source.reshape(1, 3) + off_t
    if same_parent:
        parent = torch.div(source.reshape(1, 3), 2, rounding_mode="floor")
        candidates = candidates[(torch.div(candidates, 2, rounding_mode="floor") == parent).all(dim=1)]
    if candidates.numel() <= 0:
        return None
    occ = _lookup_occupied(keys(candidates), occupied)
    candidates = candidates[~occ]
    if candidates.numel() <= 0:
        return None
    support_offsets = torch.tensor(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        device=coords.device,
        dtype=torch.long,
    )
    best = None
    best_support = -1
    for cand in candidates:
        support = int(_lookup_occupied(keys(cand.reshape(1, 3) + support_offsets), occupied).sum().item())
        if support > best_support:
            best = cand.clone()
            best_support = support
    if best is None or best_support < 2:
        return None
    return best


def _occupied_neighbor_for_snap(source: torch.Tensor, coords: torch.Tensor, *, radius: int = 1) -> torch.Tensor | None:
    keys, occupied = _coord_key_setup(coords)
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    offsets.append((dx, dy, dz))
    off_t = torch.tensor(offsets, device=coords.device, dtype=torch.long)
    candidates = source.reshape(1, 3) + off_t
    occ = _lookup_occupied(keys(candidates), occupied)
    candidates = candidates[occ]
    if candidates.numel() <= 0:
        return None
    dist = torch.norm((candidates - source.reshape(1, 3)).to(dtype=torch.float32), dim=1)
    return candidates[int(torch.argmin(dist).item())].clone()


def _remove_coord(coords: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    keep = ~((coords == coord.reshape(1, 3)).all(dim=1))
    return torch.unique(coords[keep].to(dtype=torch.long), dim=0, sorted=True)


def _add_coord(coords: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
    return torch.unique(torch.cat([coords, coord.reshape(1, 3)], dim=0).to(dtype=torch.long), dim=0, sorted=True)


def _move_coord(coords: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.unique(torch.cat([_remove_coord(coords, source), target.reshape(1, 3)], dim=0).to(dtype=torch.long), dim=0, sorted=True)


def _count_add_risk(coords: torch.Tensor, cand: torch.Tensor) -> Tuple[int, int]:
    keys, occupied = _coord_key_setup(coords)
    cand_keys = keys(cand)
    is_original = _lookup_occupied(cand_keys, occupied)
    added = cand[~is_original]
    if int(added.shape[0]) <= 0:
        return 0, 0
    original_parent = torch.div(coords, 2, rounding_mode="floor")
    keys_parent, occ_parent = _coord_key_setup(original_parent)
    added_parent = torch.div(added, 2, rounding_mode="floor")
    new_parent = int((~_lookup_occupied(keys_parent(added_parent), occ_parent)).sum().item())
    offsets = torch.tensor([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)], device=coords.device, dtype=torch.long)
    isolated = 0
    for a in added:
        support = int(_lookup_occupied(keys(a.reshape(1, 3) + offsets), occupied).sum().item())
        isolated += int(support <= 1)
    return new_parent, isolated


def _cheap_counterfactual_delta(edit_type: str, node_type: str, node_row: Mapping[str, object]) -> float:
    """No-op-map local estimate used to avoid SparsePCGC forward for every candidate.

    Negative is better.  This is intentionally conservative for Add/Move because
    previous phases showed empty-target Add can create new expensive context.
    """
    bit = float(node_row.get("bits", 0.0))
    if edit_type in {"prune_mispredicted_occupied", "snap_or_merge"}:
        return -bit
    if edit_type in {"move_within_same_parent", "child_pattern_one_flip"}:
        return -0.55 * bit
    if edit_type == "move_within_same_grandparent":
        return -0.35 * bit
    if edit_type == "remove_context_near_false_positive":
        return -0.25 * bit
    if edit_type == "add_support_near_false_negative":
        return 0.50 * bit
    return 0.0


def _candidate_rows_for_frame(
    *,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    top_n: int,
    max_candidates: int,
) -> list[Dict[str, object]]:
    pattern_before, _pop_before = _parent_pattern_maps(coords)
    fn_rows = _node_rows(base_stats, "sparsepcgc_top_low_prob_occupied_nodes_json", top_n=top_n)
    fp_rows = _node_rows(base_stats, "sparsepcgc_top_high_prob_empty_nodes_json", top_n=top_n)
    occupied_keys, occupied = _coord_key_setup(coords)
    candidates: list[Dict[str, object]] = []
    used_names = set()

    def add_candidate(edit_type: str, node_type: str, source, target, node_row, cand_coords, extra=None):
        if cand_coords is None:
            return
        sig = _coords_signature(cand_coords)
        key = (edit_type, sig)
        if key in used_names:
            return
        used_names.add(key)
        pattern_after, _pop_after = _parent_pattern_maps(cand_coords)
        ref = source if source is not None else target
        parent_key, before_pattern = _pattern_for_coord(ref, pattern_before)
        _parent_key_after, after_pattern = _pattern_for_coord(ref, pattern_after)
        new_parent, isolated = _count_add_risk(coords, cand_coords)
        neigh = _neighbor_count(coords)
        local_support = 0
        if source is not None:
            mask = (coords == source.reshape(1, 3)).all(dim=1)
            if bool(mask.any().item()):
                local_support = int(neigh[mask.nonzero(as_tuple=False).reshape(-1)[0]].item())
        row = {
            "node_type": node_type,
            "edit_type": edit_type,
            "source_coord": json.dumps([int(v) for v in source.tolist()]) if source is not None else "",
            "target_coord": json.dumps([int(v) for v in target.tolist()]) if target is not None else "",
            "depth": int(node_row.get("depth", -1)),
            "parent_key": parent_key,
            "child_pattern_before": before_pattern,
            "child_pattern_after": after_pattern,
            "p_occ_before": float(node_row.get("prob", 0.0)),
            "p_true_before": float(node_row.get("prob_true", 0.0)),
            "bit_each_before": float(node_row.get("bits", 0.0)),
            "cheap_estimated_bits_delta": _cheap_counterfactual_delta(edit_type, node_type, node_row),
            "edit_count": int(max(abs(int(cand_coords.shape[0]) - int(coords.shape[0])), 1 if sig != _coords_signature(coords) else 0)),
            "add_count": max(int(cand_coords.shape[0]) - int(coords.shape[0]), 0),
            "prune_count": max(int(coords.shape[0]) - int(cand_coords.shape[0]), 0),
            "move_count": 1 if source is not None and target is not None and int(cand_coords.shape[0]) == int(coords.shape[0]) else 0,
            "snap_count": 1 if edit_type == "snap_or_merge" else 0,
            "new_parent_count": int(new_parent),
            "isolated_add_count": int(isolated),
            "neighbor_density": int(local_support),
            "local_support_count": int(local_support),
            "cand_coords": cand_coords,
        }
        if extra:
            row.update(extra)
        candidates.append(row)

    for node in fn_rows:
        source, _idx = _source_coord_for_node(node, coords, set())
        if source is None:
            continue
        add_candidate("prune_mispredicted_occupied", "false_negative", source, None, node, _remove_coord(coords, source))
        add_target = _safe_add_target_near(source, coords, same_parent=True, radius=1)
        if add_target is not None:
            add_candidate("add_support_near_false_negative", "false_negative", None, add_target, node, _add_coord(coords, add_target))
            add_candidate("move_within_same_parent", "false_negative", source, add_target, node, _move_coord(coords, source, add_target))
            add_candidate("child_pattern_one_flip", "false_negative", source, add_target, node, _move_coord(coords, source, add_target))
        gp_target = _safe_add_target_near(source, coords, same_parent=False, radius=2)
        if gp_target is not None:
            add_candidate("move_within_same_grandparent", "false_negative", source, gp_target, node, _move_coord(coords, source, gp_target))
        snap_target = _occupied_neighbor_for_snap(source, coords, radius=1)
        if snap_target is not None:
            add_candidate("snap_or_merge", "false_negative", source, snap_target, node, _remove_coord(coords, source))
        if len(candidates) >= int(max_candidates):
            break

    for node in fp_rows:
        target = _target_coord_for_node(node, coords, occupied_keys, occupied, set())
        if target is None:
            continue
        offsets = torch.tensor([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)], device=coords.device, dtype=torch.long)
        neigh = target.reshape(1, 3) + offsets
        occ = _lookup_occupied(occupied_keys(neigh), occupied)
        occupied_neigh = neigh[occ]
        if int(occupied_neigh.shape[0]) > 0:
            source = occupied_neigh[0].clone()
            add_candidate("remove_context_near_false_positive", "false_positive", source, target, node, _remove_coord(coords, source))
        if len(candidates) >= int(max_candidates):
            break
    return candidates[: int(max_candidates)]


def _actual_bit_percent(bit_encoder, args, coords, meta, base_bits: float, cache: Dict[str, Mapping[str, object]]):
    sig = _coords_signature(coords)
    if sig in cache:
        stats = dict(cache[sig])
        hit = True
    else:
        stats = bit_encoder.encode_bits(_coords_to_xyz(coords, meta, args))
        cache[sig] = dict(stats)
        hit = False
    bit = float(stats.get("bit", 0.0))
    return (bit - float(base_bits)) / max(float(base_bits), 1e-9) * 100.0, hit


def run_phase2s(cli: argparse.Namespace) -> int:
    args = _prepare_args(cli)
    debug_args = args
    debug_args.sparsepcgc_skip_decode = True
    debug_args.sparsepcgc_occupancy_debug_topk_final = int(cli.debug_topk)
    debug_args.sparsepcgc_occupancy_debug_topk_per_layer = max(1024, min(int(cli.debug_topk), 8192))
    bit_args = copy.copy(args)
    bit_args.sparsepcgc_skip_decode = True
    bit_args.enable_sparsepcgc_occupancy_debug = False

    rows = _read_rows(cli.output_csv) if bool(cli.append_output) else []
    prob_cache: Dict[str, Mapping[str, object]] = {}
    actual_cache: Dict[str, Mapping[str, object]] = {}
    duplicate_skip = 0
    actual_eval_count = 0
    cache_hit_count = 0
    debug_encoder = build_actual_encoder(debug_args)
    bit_encoder = build_actual_encoder(bit_args)
    try:
        for file_path in cli.files:
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            sequence = Path(file_path).parent.name
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, args)
            base_stats = debug_encoder.encode_bits(base_xyz)
            base_bit_stats = bit_encoder.encode_bits(base_xyz)
            base_bits = float(base_bit_stats.get("bit", base_stats.get("bit", 0.0)))
            candidates = _candidate_rows_for_frame(
                coords=coords,
                base_stats=base_stats,
                top_n=int(cli.top_nodes),
                max_candidates=int(cli.max_candidates),
            )
            seen = set()
            evaluated = []
            for cand in candidates:
                cand_coords = cand.pop("cand_coords")
                sig = _coords_signature(cand_coords)
                if sig in seen:
                    duplicate_skip += 1
                    continue
                seen.add(sig)
                cand["cand_sig"] = sig
                cand["cand_coords"] = cand_coords
                evaluated.append(cand)
            evaluated.sort(key=lambda r: _safe_float(r.get("cheap_estimated_bits_delta"), float("inf")))
            prob_eval_sigs = {r["cand_sig"] for r in evaluated[: max(int(cli.prob_eval_topk), 0)]}
            for cand in evaluated:
                if cand["cand_sig"] not in prob_eval_sigs:
                    cand["estimated_bits_delta"] = cand.get("cheap_estimated_bits_delta", "")
                    continue
                cand_coords = cand["cand_coords"]
                sig = cand["cand_sig"]
                if sig in prob_cache:
                    after_stats = dict(prob_cache[sig])
                    cache_hit_count += 1
                else:
                    after_stats = debug_encoder.encode_bits(_coords_to_xyz(cand_coords, meta, args))
                    prob_cache[sig] = dict(after_stats)
                prob_update = _probability_row_updates(before=base_stats, after=after_stats)
                est_delta = _safe_float(prob_update.get("estimated_bits_delta"), float("nan"))
                cand.update(prob_update)
                cand["estimated_bits_delta"] = est_delta
            evaluated.sort(key=lambda r: _safe_float(r.get("estimated_bits_delta"), float("inf")))
            actual_sigs = {r["cand_sig"] for r in evaluated[: max(int(cli.actual_topk), 0)]}
            if bool(cli.actual_per_edit_type):
                best_by_type: Dict[str, Dict[str, object]] = {}
                for cand in evaluated:
                    edit_type = str(cand.get("edit_type", ""))
                    if edit_type not in best_by_type:
                        best_by_type[edit_type] = cand
                actual_sigs.update(r["cand_sig"] for r in best_by_type.values())
            for cand in evaluated:
                cand_coords = cand.pop("cand_coords")
                actual_raw = ""
                d1 = ""
                chamfer = ""
                if cand["cand_sig"] in actual_sigs:
                    actual_raw, hit = _actual_bit_percent(bit_encoder, args, cand_coords, meta, base_bits, actual_cache)
                    actual_eval_count += 0 if hit else 1
                    cache_hit_count += int(hit)
                p_true_after = _safe_float(cand.get("p_true_mean_after"), float("nan"))
                bit_after = _safe_float(cand.get("total_estimated_bits_after"), float("nan"))
                bit_before = _safe_float(cand.get("total_estimated_bits_before"), float("nan"))
                pred_flip = bool(_safe_float(cand.get("estimated_bits_delta"), 0.0) < 0.0)
                row = {
                    "file": str(file_path),
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "p_occ_after": "",
                    "p_true_after": p_true_after,
                    "p_true_delta": cand.get("p_true_delta", ""),
                    "bit_each_after": "",
                    "bit_each_delta": "",
                    "estimated_bits_delta": cand.get("estimated_bits_delta", ""),
                    "prediction_flip_success": pred_flip,
                    "actual_raw_percent": actual_raw,
                    "D1_PSNR": d1,
                    "Chamfer": chamfer,
                    "cache_hit_count": cache_hit_count,
                    "duplicate_skip_count": duplicate_skip,
                    "actual_eval_count": actual_eval_count,
                }
                row.update(cand)
                rows.append(row)
                _write_csv(cli.output_csv, rows)
            print(
                json.dumps(
                    {
                        "phase2s": True,
                        "sequence": sequence,
                        "frame": frame_id,
                        "candidates": len(evaluated),
                        "actual_eval_count": actual_eval_count,
                        "cache_hit_count": cache_hit_count,
                        "duplicate_skip_count": duplicate_skip,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        for enc in (debug_encoder, bit_encoder):
            close = getattr(enc, "close", None)
            if callable(close):
                close()
    _write_csv(cli.output_csv, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2S counterfactual context sensitivity probe")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--top-nodes", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--prob-eval-topk", type=int, default=6)
    parser.add_argument("--actual-topk", type=int, default=8)
    parser.add_argument("--actual-per-edit-type", action="store_true")
    parser.add_argument("--debug-topk", type=int, default=4096)
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2s(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
