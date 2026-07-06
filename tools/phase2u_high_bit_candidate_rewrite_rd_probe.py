#!/usr/bin/env python
"""Phase2U high-bit candidate rewriting with local patch RD analysis.

Research-only script.  High-bit SparsePCGC nodes are used as seeds, then each
candidate voxel is classified by local patch risk and rewritten via keep /
prune / snap-merge / same-parent move / minimal repair templates.
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
from tools.phase2q_probability_guided_context_edit import _candidate_to_coords_q, _probability_row_updates
from tools.phase2t_multi_rule_context_edit_headroom import _base_headroom_row


DEFAULT_CANDIDATES = (
    "codec_only",
    "block_only",
    "high_bit_raw_prune",
    "low_prob_snap_to_existing",
    "snap_to_existing_only",
    "patch_snap_first_rewrite",
    "patch_move_first_rewrite",
    "patch_quality_veto_prune",
    "patch_prune_minimal_repair",
    "hybrid_rewrite_beam",
)


def _prepare_args(_cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2U high-bit patch rewrite"),
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


def _dataset_name(path: str) -> str:
    parts = Path(path).parts
    for name in ("8i", "UVG", "MVUB"):
        if name in parts:
            return name
    return "unknown"


def _sequence_name(path: str) -> str:
    return Path(path).parent.name


def _coord_tuple(coord: torch.Tensor) -> Tuple[int, int, int]:
    return tuple(int(v) for v in coord.reshape(-1)[:3].tolist())


def _same_parent_empty_target(source: torch.Tensor, coords: torch.Tensor, keys, occupied) -> Tuple[torch.Tensor | None, int]:
    parent = torch.div(source.reshape(3), 2, rounding_mode="floor")
    offsets6 = torch.tensor([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)], device=coords.device, dtype=torch.long)
    best = None
    best_support = -1
    for slot in range(8):
        child = torch.tensor([slot // 4, (slot // 2) % 2, slot % 2], device=coords.device, dtype=torch.long)
        cand = parent * 2 + child
        if bool((cand == source.reshape(3)).all().item()):
            continue
        if bool(_lookup_occupied(keys(cand.reshape(1, 3)), occupied)[0].item()):
            continue
        support = int(_lookup_occupied(keys(cand.reshape(1, 3) + offsets6), occupied).sum().item())
        if support > best_support:
            best = cand.clone()
            best_support = support
    return best, int(best_support)


def _occupied_neighbor(source: torch.Tensor, coords: torch.Tensor, keys, occupied) -> torch.Tensor | None:
    offsets = torch.tensor([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)], device=coords.device, dtype=torch.long)
    cand = source.reshape(1, 3) + offsets
    occ = _lookup_occupied(keys(cand), occupied)
    if not bool(occ.any().item()):
        return None
    targets = cand[occ]
    dist = torch.norm((targets - source.reshape(1, 3)).to(dtype=torch.float32), dim=1)
    return targets[int(torch.argmin(dist).item())].clone()


def _parent_slot_info(coords: torch.Tensor):
    unique_parent, inverse_parent, _slots, _occ, patterns, parent_pop = _parent_info(coords)
    pattern_by_parent = {_coord_tuple(p): int(patterns[i].item()) for i, p in enumerate(unique_parent)}
    pop_by_parent = {_coord_tuple(p): int(parent_pop[i].item()) for i, p in enumerate(unique_parent)}
    return inverse_parent, pattern_by_parent, pop_by_parent


def _patch_type(local_support: int, parent_pop: int, has_snap: bool, move_support: int, bit: float) -> str:
    if local_support <= 1 or parent_pop <= 1:
        return "geometry_critical"
    if has_snap and local_support >= 2:
        return "snap_safe"
    if move_support >= 2 and parent_pop >= 2:
        return "move_safe"
    if local_support >= 4 and parent_pop >= 2:
        return "delete_safe"
    if bit >= 10.0 and local_support >= 2:
        return "delete_safe"
    return "keep_recommended"


def _apply_rewrite(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget: float,
    pool: int,
    block_size: int,
    mode: str,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    score, bit_score, depth_score = _high_bit_voxel_scores(coords, base_stats, max_pool=int(pool))
    neigh = _neighbor_count(coords)
    inverse_parent, pattern_by_parent, pop_by_parent = _parent_slot_info(coords)
    keys, occupied = _coord_key_setup(coords)
    target_edits = max(1, int(math.ceil(float(coords.shape[0]) * float(budget))))
    valid = torch.isfinite(score) & (bit_score > 0)
    order = torch.argsort(torch.where(valid, score, score.new_full(score.shape, -1e12)), descending=True)
    source_remove: list[Tuple[int, int, int]] = []
    targets_add: list[torch.Tensor] = []
    selected_idx: list[int] = []
    samples = []
    counts = {
        "keep": 0,
        "prune": 0,
        "snap": 0,
        "move": 0,
        "add": 0,
        "delete_safe": 0,
        "snap_safe": 0,
        "move_safe": 0,
        "geometry_critical": 0,
        "keep_recommended": 0,
    }
    used_source = set()
    used_target = set()
    for idx in order.tolist():
        if len(selected_idx) >= target_edits:
            break
        if not bool(valid[idx].item()):
            continue
        src = coords[int(idx)].clone()
        skey = _coord_tuple(src)
        if skey in used_source:
            continue
        parent = torch.div(src.reshape(3), 2, rounding_mode="floor")
        pkey = _coord_tuple(parent)
        parent_pop = int(pop_by_parent.get(pkey, 0))
        support = int(neigh[int(idx)].item())
        bit = float(bit_score[int(idx)].item())
        snap_target = _occupied_neighbor(src, coords, keys, occupied)
        move_target, move_support = _same_parent_empty_target(src, coords, keys, occupied)
        ptype = _patch_type(support, parent_pop, snap_target is not None, move_support, bit)
        counts[ptype] = counts.get(ptype, 0) + 1
        op = "keep"
        tgt = None
        if mode == "patch_quality_veto_prune":
            if ptype in {"delete_safe", "snap_safe", "move_safe"}:
                op = "prune"
        elif mode == "patch_move_first_rewrite":
            if ptype == "move_safe" and move_target is not None:
                op, tgt = "move", move_target
            elif ptype == "snap_safe" and snap_target is not None:
                op = "snap"
            elif ptype == "delete_safe":
                op = "prune"
        elif mode == "patch_prune_minimal_repair":
            if ptype in {"delete_safe", "snap_safe"}:
                op = "prune"
                if move_target is not None and move_support >= 3 and len(targets_add) < max(1, int(target_edits * 0.10)):
                    # Repair is a sparse supported sibling add, never a new parent.
                    op, tgt = "prune_repair", move_target
            elif ptype == "move_safe" and move_target is not None:
                op, tgt = "move", move_target
        else:  # patch_snap_first_rewrite
            if ptype == "snap_safe" and snap_target is not None:
                op = "snap"
            elif ptype == "move_safe" and move_target is not None:
                op, tgt = "move", move_target
            elif ptype == "delete_safe":
                op = "prune"
        if op == "keep":
            counts["keep"] += 1
            continue
        tkey = _coord_tuple(tgt) if tgt is not None else None
        if tkey is not None and tkey in used_target:
            continue
        used_source.add(skey)
        if tkey is not None:
            used_target.add(tkey)
        selected_idx.append(int(idx))
        source_remove.append(skey)
        if op in {"move", "prune_repair"} and tgt is not None:
            targets_add.append(tgt.reshape(1, 3).to(device=coords.device, dtype=torch.long))
        counts["prune" if op in {"prune", "snap"} else op] = counts.get("prune" if op in {"prune", "snap"} else op, 0) + 1
        if op == "snap":
            counts["snap"] += 1
        if op == "move":
            counts["move"] += 1
        if op == "prune_repair":
            counts["add"] += 1
        if len(samples) < 48:
            samples.append({
                "source": list(skey),
                "target": list(tkey) if tkey is not None else None,
                "operation_type": op,
                "local_patch_type": ptype,
                "bit_each": bit,
                "depth": int(depth_score[int(idx)].item()),
                "local_support": support,
                "parent_pop": parent_pop,
                "child_pattern": int(pattern_by_parent.get(pkey, 0)),
            })
    source_remove_set = set(source_remove)
    keep_mask = torch.tensor([_coord_tuple(c) not in source_remove_set for c in coords], device=coords.device, dtype=torch.bool)
    pieces = [coords[keep_mask]]
    if targets_add:
        pieces.append(torch.cat(targets_add, dim=0))
    cand = torch.unique(torch.cat(pieces, dim=0).to(dtype=torch.long), dim=0, sorted=True)
    selected = torch.zeros((int(coords.shape[0]),), device=coords.device, dtype=torch.bool)
    if selected_idx:
        selected[torch.tensor(selected_idx, device=coords.device, dtype=torch.long)] = True
    selected_bits = bit_score[selected]
    selected_depth = depth_score[selected]
    selected_neigh = neigh[selected].to(dtype=torch.float32)
    parent_counts = torch.bincount(inverse_parent[selected], minlength=int(inverse_parent.max().item()) + 1) if int(selected.sum().item()) else torch.zeros((1,), device=coords.device, dtype=torch.long)
    debug = {
        "candidate_family": "phase2u_patch_rewrite",
        "candidate_variant": mode,
        "operation_type": "patch_rewrite",
        "actual_edit_ratio": float(len(selected_idx)) / max(float(coords.shape[0]), 1.0),
        "selected_bit_sum": float(selected_bits.sum().item()) if int(selected_bits.numel()) else 0.0,
        "selected_bit_mean": float(selected_bits.mean().item()) if int(selected_bits.numel()) else 0.0,
        "selected_depth_hist_json": _hist_json_from_tensor(selected_depth[selected_depth >= 0]),
        "hole_risk": float((selected_neigh <= 2).to(dtype=torch.float32).mean().item()) if int(selected_neigh.numel()) else 0.0,
        "boundary_removed_ratio": float((selected_neigh <= 2).to(dtype=torch.float32).mean().item()) if int(selected_neigh.numel()) else 0.0,
        "geometry_proxy": float((1.0 / (selected_neigh + 1.0)).mean().item()) if int(selected_neigh.numel()) else 0.0,
        "delete_safe_count": int(counts.get("delete_safe", 0)),
        "snap_safe_count": int(counts.get("snap_safe", 0)),
        "move_safe_count": int(counts.get("move_safe", 0)),
        "geometry_critical_count": int(counts.get("geometry_critical", 0)),
        "keep_count": int(counts.get("keep", 0)),
        "prune_count": int(counts.get("prune", 0)),
        "snap_count": int(counts.get("snap", 0)),
        "move_count": int(counts.get("move", 0)),
        "add_count": int(counts.get("add", 0)),
        "new_parent_count": 0,
        "isolated_add_count": 0,
        "affected_parent_count": int((parent_counts > 0).sum().item()) if int(selected.sum().item()) else 0,
        "same_parent_edit_max": int(parent_counts.max().item()) if int(selected.sum().item()) else 0,
        "local_patch_type": "mixed_patch_rewrite",
        "operation_counts_json": json.dumps(counts, sort_keys=True),
        "source_target_sample_json": json.dumps(samples, sort_keys=True),
        "cheap_estimated_bits_delta": -float(selected_bits.sum().item()) if int(selected_bits.numel()) else 0.0,
        "budget_reached": bool(len(selected_idx) >= max(1, int(target_edits * 0.98))),
    }
    return cand, debug


def _candidate_to_coords_u(
    *,
    candidate: str,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if candidate in {"codec_only", "high_bit_raw_prune", "snap_to_existing_only", "low_prob_snap_to_existing"}:
        cand, _mask, debug = _candidate_to_coords_q(
            candidate=candidate,
            coords=coords,
            base_stats=base_stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            seed=seed,
        )
        debug = dict(debug)
        debug.setdefault("operation_type", candidate)
        return cand, debug
    if candidate == "block_only":
        cand, _mask, debug = _candidate_to_coords_n(
            candidate=candidate,
            coords=coords,
            base_stats=base_stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            seed=seed,
        )
        debug = dict(debug)
        debug.setdefault("operation_type", "block_only")
        return cand, debug
    if candidate == "hybrid_rewrite_beam":
        options = []
        for mode in ("patch_snap_first_rewrite", "patch_move_first_rewrite", "patch_quality_veto_prune"):
            cand, dbg = _apply_rewrite(coords, base_stats, budget=budget, pool=pool, block_size=block_size, mode=mode)
            risk = 1.0 + 10.0 * _safe_float(dbg.get("geometry_proxy"), 0.0) + 3.0 * _safe_float(dbg.get("hole_risk"), 0.0)
            score = _safe_float(dbg.get("selected_bit_sum"), 0.0) / max(risk, 1e-6)
            options.append((score, cand, dbg, mode))
        options.sort(key=lambda x: x[0], reverse=True)
        _score, cand, dbg, chosen = options[0]
        dbg = dict(dbg)
        dbg["candidate_variant"] = "hybrid_rewrite_beam"
        dbg["operation_type"] = f"hybrid:{chosen}"
        return cand, dbg
    return _apply_rewrite(coords, base_stats, budget=budget, pool=pool, block_size=block_size, mode=candidate)


def _save_processed_ply(path: Path, coords: torch.Tensor, meta, args) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = _coords_to_xyz(coords, meta, args)
    write_ascii_ply_xyz(path, xyz.detach().to("cpu").numpy().astype(np.float64, copy=False))
    return str(path)


def _sanitize_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))


def run_phase2u(cli: argparse.Namespace) -> int:
    args = _prepare_args(cli)
    debug_args = args
    debug_args.sparsepcgc_skip_decode = True
    max_pool = max(int(float(x)) for x in _parse_csv_text(cli.pools))
    debug_args.sparsepcgc_occupancy_debug_topk_final = int(max_pool)
    debug_args.sparsepcgc_occupancy_debug_topk_per_layer = max(512, min(int(max_pool), 8192))
    decode_args = copy.copy(args)
    decode_args.sparsepcgc_skip_decode = not bool(cli.decode_quality)
    decode_args.enable_sparsepcgc_occupancy_debug = False
    Path(cli.decoded_dir).mkdir(parents=True, exist_ok=True)
    Path(cli.processed_dir).mkdir(parents=True, exist_ok=True)
    decode_args.sparsepcgc_decoded_copy_dir = str(cli.decoded_dir)

    rows = _read_rows(cli.output_csv) if bool(cli.append_output) else []
    debug_encoder = build_actual_encoder(debug_args)
    decode_encoder = build_actual_encoder(decode_args)
    eval_cache: Dict[str, Dict[str, object]] = {}
    actual_eval_count = 0
    cache_hit_count = 0
    duplicate_skip = 0
    try:
        for file_idx, file_path in enumerate(cli.files):
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            dataset = _dataset_name(file_path)
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
                ) if bool(cli.decode_quality) and decoded_gt_path else {}
            )
            if bool(cli.emit_headroom):
                head = _base_headroom_row(file_path=file_path, base_stats=base_stats)
                head["codec_setting_id"] = "SparsePCGC_default"
                rows.append(head)
                _write_csv(cli.output_csv, rows)
            generated = []
            for budget in [float(x) for x in _parse_csv_text(cli.budgets)]:
                for candidate in _parse_csv_text(cli.candidates):
                    try:
                        cand_coords, debug = _candidate_to_coords_u(
                            candidate=candidate,
                            coords=coords,
                            base_stats=base_stats,
                            budget=budget,
                            pool=max_pool,
                            block_size=int(cli.block_size),
                            seed=int(cli.seed) + file_idx,
                        )
                    except Exception as exc:
                        rows.append({
                            "dataset": dataset,
                            "sequence": sequence,
                            "frame_id": frame_id,
                            "codec_setting_id": "SparsePCGC_default",
                            "candidate_name": candidate,
                            "budget_ratio": budget,
                            "error": repr(exc),
                        })
                        continue
                    sig = _coords_signature(cand_coords)
                    generated.append((_safe_float(debug.get("cheap_estimated_bits_delta"), -_safe_float(debug.get("selected_bit_sum"), 0.0)), candidate, budget, cand_coords, dict(debug), sig))
            generated.sort(key=lambda x: x[0])
            selected = []
            seen = set()
            for item in generated:
                if item[-1] in seen:
                    duplicate_skip += 1
                    continue
                seen.add(item[-1])
                selected.append(item)
                if len(selected) >= int(cli.actual_topk):
                    break
            for cheap, candidate, budget, cand_coords, debug, sig in selected:
                processed_path = ""
                if candidate in set(_parse_csv_text(cli.save_processed_candidates)) or bool(cli.save_all_processed):
                    out_name = f"{dataset}_{sequence}_{frame_id}_{_sanitize_name(candidate)}_{int(float(budget)*1000):03d}permil.ply"
                    processed_path = _save_processed_ply(Path(cli.processed_dir) / out_name, cand_coords, meta, args)
                if bool(cli.decode_quality):
                    row, hit, _te, _tq = _eval_decoded_row(
                        file_path=str(file_path),
                        sequence=sequence,
                        frame_id=frame_id,
                        candidate_name=candidate,
                        edit_sequence=str(debug.get("operation_type", candidate)),
                        budget=float(budget),
                        pool=int(max_pool),
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
                    cache_hit_count += int(hit)
                    actual_eval_count += 0 if hit else 1
                else:
                    hit = sig in eval_cache
                    if hit:
                        row = dict(eval_cache[sig])
                    else:
                        stats = decode_encoder.encode_bits(_coords_to_xyz(cand_coords, meta, args))
                        raw = (float(stats.get("bit", 0.0)) - base_bits) / max(base_bits, 1e-9) * 100.0
                        row = {
                            "actual_raw_percent": raw,
                            "raw_bit": stats.get("bit", ""),
                            "base_bit": base_bits,
                            "D1_PSNR": "",
                            "Chamfer": "",
                            "point_to_plane_PSNR": "",
                        }
                        eval_cache[sig] = dict(row)
                    cache_hit_count += int(hit)
                    actual_eval_count += 0 if hit else 1
                point_delta = int(cand_coords.shape[0]) - int(coords.shape[0])
                selected_bit_sum = _safe_float(debug.get("selected_bit_sum"), 0.0)
                row.update({
                    "dataset": dataset,
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "codec_setting_id": "SparsePCGC_default",
                    "candidate_name": candidate,
                    "operation_type": debug.get("operation_type", candidate),
                    "budget_ratio": float(budget),
                    "actual_edit_ratio": debug.get("actual_edit_ratio", abs(point_delta) / max(float(coords.shape[0]), 1.0)),
                    "point_count_delta": point_delta,
                    "estimated_bits_delta": cheap,
                    "geometry_proxy": debug.get("geometry_proxy", ""),
                    "hole_risk": debug.get("hole_risk", ""),
                    "center_bit_each_mean": debug.get("selected_bit_mean", ""),
                    "selected_bit_sum": selected_bit_sum,
                    "local_patch_type": debug.get("local_patch_type", debug.get("selection_unit_type", "")),
                    "delete_safe_count": debug.get("delete_safe_count", ""),
                    "snap_safe_count": debug.get("snap_safe_count", ""),
                    "move_safe_count": debug.get("move_safe_count", ""),
                    "geometry_critical_count": debug.get("geometry_critical_count", ""),
                    "prune_count": debug.get("prune_count", max(int(coords.shape[0]) - int(cand_coords.shape[0]), 0)),
                    "snap_count": debug.get("snap_count", ""),
                    "move_count": debug.get("move_count", ""),
                    "add_count": debug.get("add_count", ""),
                    "keep_count": debug.get("keep_count", ""),
                    "new_parent_count": debug.get("new_parent_count", ""),
                    "isolated_add_count": debug.get("isolated_add_count", ""),
                    "affected_parent_count": debug.get("affected_parent_count", ""),
                    "source_target_sample_json": debug.get("source_target_sample_json", ""),
                    "processed_ply_path": processed_path,
                    "render_path": "",
                    "cache_hit_count": cache_hit_count,
                    "duplicate_skip_count": duplicate_skip,
                    "actual_eval_count": actual_eval_count,
                })
                row.update(debug)
                rows.append(row)
                _write_csv(cli.output_csv, rows)
            print(json.dumps({"phase2u": True, "sequence": sequence, "frame": frame_id, "generated": len(generated), "actual_eval_count": actual_eval_count, "cache_hit_count": cache_hit_count, "duplicate_skip_count": duplicate_skip}, sort_keys=True), flush=True)
    finally:
        for enc in (debug_encoder, decode_encoder):
            close = getattr(enc, "close", None)
            if callable(close):
                close()
    _write_csv(cli.output_csv, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2U high-bit candidate rewrite RD probe")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--budgets", default="0.010,0.020,0.030")
    parser.add_argument("--pools", default="8192")
    parser.add_argument("--actual-topk", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode-quality", action="store_true")
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2u_decoded")
    parser.add_argument("--quality-max-points", type=int, default=3000)
    parser.add_argument("--normal-max-points", type=int, default=3000)
    parser.add_argument("--pc-error-path", default="")
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--processed-dir", default="/data/maejima/log/phase2u_processed_pointclouds")
    parser.add_argument("--save-processed-candidates", default="high_bit_raw_prune,low_prob_snap_to_existing,hybrid_rewrite_beam,patch_snap_first_rewrite,patch_move_first_rewrite,patch_quality_veto_prune")
    parser.add_argument("--save-all-processed", action="store_true")
    parser.add_argument("--emit-headroom", action="store_true")
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2u(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
