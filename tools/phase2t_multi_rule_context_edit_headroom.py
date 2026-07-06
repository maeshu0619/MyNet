#!/usr/bin/env python
"""Phase2T multi-rule context editing and headroom probe.

Research-only script.  It reuses Phase2Q/R/S probability debug and candidate
builders, clusters low-p / high-bit contexts, applies a small set of local
multi-node edit templates, and actual-encodes only the best cheap candidates.
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
    _parse_json_payload,
    _quality_from_paths,
    _write_csv,
)
from tools.phase2m_multi_operator_context_rewriter import _eval_decoded_row
from tools.phase2n_voxel_context_rd_optimizer import _candidate_to_coords_n
from tools.phase2q_probability_guided_context_edit import _candidate_to_coords_q, _probability_row_updates
from tools.phase2r_probability_flip_relocation_oracle import _source_coord_for_node, _target_coord_for_node
from tools.phase2s_counterfactual_context_sensitivity import _occupied_neighbor_for_snap, _safe_add_target_near


DEFAULT_CANDIDATES = (
    "codec_only",
    "block_only",
    "high_bit_raw_prune",
    "snap_to_existing_only",
    "low_prob_snap_to_existing",
    "group_prune_low_support_FN",
    "group_same_parent_move",
    "group_same_grandparent_move",
    "group_snap_or_merge",
    "child_pattern_1to2_flip",
    "suppress_FP_context",
    "safe_support_add",
    "hybrid_group_rule_beam",
)


def _prepare_args(_cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2T multi-rule context edit"),
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


def _json_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _base_headroom_row(*, file_path: str, base_stats: Mapping[str, object]) -> Dict[str, object]:
    total_bits = max(_safe_float(base_stats.get("sparsepcgc_estimated_occupancy_bits"), 0.0), 1e-9)
    high_nodes = _parse_json_payload(base_stats.get("sparsepcgc_top_high_bit_nodes_json", ""), [])
    bits = sorted([float(r.get("bits", 0.0)) for r in high_nodes], reverse=True)
    def share(ratio: float) -> float:
        if not bits:
            return 0.0
        k = max(1, min(int(math.ceil(len(bits) * ratio)), len(bits)))
        return float(sum(bits[:k]) / total_bits * 100.0)
    low_counts = _json_dict(base_stats.get("sparsepcgc_occupied_low_prob_threshold_counts_json"))
    high_empty = _json_dict(base_stats.get("sparsepcgc_empty_high_prob_threshold_counts_json"))
    return {
        "dataset": _dataset_name(file_path),
        "sequence": _sequence_name(file_path),
        "frame_id": Path(file_path).stem,
        "codec_setting_id": "SparsePCGC_default",
        "candidate_name": "headroom_noop",
        "cluster_id": "all",
        "actual_raw_percent": 0.0,
        "estimated_bits_delta": 0.0,
        "occupancy_acc_at_0p5": base_stats.get("sparsepcgc_occupancy_accuracy_at_0p5", ""),
        "occupancy_acc_at_0p7": "",
        "occupancy_acc_at_0p8": "",
        "occupancy_acc_at_0p9": "",
        "occupied_recall": base_stats.get("sparsepcgc_occupied_recall_at_0p5", ""),
        "empty_accuracy": base_stats.get("sparsepcgc_empty_accuracy_at_0p5", ""),
        "p_true_quantiles_json": base_stats.get("sparsepcgc_prob_true_quantiles_json", ""),
        "bit_each_quantiles_json": base_stats.get("sparsepcgc_bit_each_quantiles_json", ""),
        "total_estimated_bits": total_bits,
        "top1p_high_bit_symbol_bit_share": share(0.01),
        "top3p_high_bit_symbol_bit_share": share(0.03),
        "top5p_high_bit_symbol_bit_share": share(0.05),
        "low_p_occupied_count": low_counts.get("0.5", low_counts.get("0p5", "")),
        "high_p_empty_count": high_empty.get("0.7", high_empty.get("0p7", "")),
        "improvement_headroom_estimate": share(0.03),
        "bits_by_depth_json": base_stats.get("sparsepcgc_bits_by_depth_json", ""),
        "bits_by_parent_popcount_json": base_stats.get("sparsepcgc_bits_by_parent_popcount_json", ""),
        "bits_by_child_pattern_topk_json": base_stats.get("sparsepcgc_bits_by_child_pattern_topk_json", ""),
    }


def _parse_coord(value: object) -> torch.Tensor | None:
    if value is None or value == "":
        return None
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list) and len(parsed) >= 3:
            return torch.tensor(parsed[:3], dtype=torch.long)
    except Exception:
        return None
    return None


def _coord_tuple(coord: torch.Tensor) -> Tuple[int, int, int]:
    return tuple(int(v) for v in coord.reshape(-1)[:3].tolist())


def _base_parent_maps(coords: torch.Tensor):
    unique_parent, _inverse_parent, _slots, _occ, patterns, parent_pop = _parent_info(coords)
    pattern_map = {_coord_tuple(p): int(patterns[i].item()) for i, p in enumerate(unique_parent)}
    pop_map = {_coord_tuple(p): int(parent_pop[i].item()) for i, p in enumerate(unique_parent)}
    return pattern_map, pop_map


def _parent_key_and_slot(coord: torch.Tensor) -> Tuple[Tuple[int, int, int], int]:
    c = coord.reshape(3).to(dtype=torch.long)
    parent = torch.div(c, 2, rounding_mode="floor")
    child = c - parent * 2
    slot = int(child[0].item()) * 4 + int(child[1].item()) * 2 + int(child[2].item())
    return _coord_tuple(parent), slot


def _pattern_after_edit(pattern: int, source: torch.Tensor | None, target: torch.Tensor | None) -> int:
    out = int(pattern)
    parent_key = None
    if source is not None:
        parent_key, slot = _parent_key_and_slot(source)
        out &= ~(1 << int(slot))
    if target is not None:
        t_parent, t_slot = _parent_key_and_slot(target)
        if parent_key is None or t_parent == parent_key:
            out |= 1 << int(t_slot)
    return int(out)


def _fast_candidate_rows_for_frame(
    *,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    top_n: int,
    max_candidates: int,
) -> list[Dict[str, object]]:
    """Generate Phase2T local counterfactual rows without per-candidate unique()."""
    pattern_map, _pop_map = _base_parent_maps(coords)
    neigh = _neighbor_count(coords)
    occupied_keys, occupied = _coord_key_setup(coords)
    rows: list[Dict[str, object]] = []
    used = set()

    def add(edit_type: str, node_type: str, node: Mapping[str, object], source, target):
        if len(rows) >= int(max_candidates):
            return
        key = (
            edit_type,
            tuple(source.tolist()) if source is not None else None,
            tuple(target.tolist()) if target is not None else None,
        )
        if key in used:
            return
        used.add(key)
        ref = source if source is not None else target
        if ref is None:
            return
        pkey, _slot = _parent_key_and_slot(ref)
        before = int(pattern_map.get(pkey, 0))
        after = _pattern_after_edit(before, source, target)
        local_support = 0
        if source is not None:
            mask = (coords == source.reshape(1, 3).to(device=coords.device)).all(dim=1)
            if bool(mask.any().item()):
                local_support = int(neigh[mask.nonzero(as_tuple=False).reshape(-1)[0]].item())
        new_parent = 0
        isolated = 0
        if source is None and target is not None:
            target_parent, _ = _parent_key_and_slot(target)
            new_parent = int(target_parent not in pattern_map)
            offsets = torch.tensor([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)], device=coords.device, dtype=torch.long)
            support = int(_lookup_occupied(occupied_keys(target.reshape(1, 3).to(device=coords.device) + offsets), occupied).sum().item())
            isolated = int(support <= 1)
        rows.append({
            "node_type": node_type,
            "edit_type": edit_type,
            "source_coord": json.dumps([int(v) for v in source.tolist()]) if source is not None else "",
            "target_coord": json.dumps([int(v) for v in target.tolist()]) if target is not None else "",
            "depth": int(node.get("depth", -1)),
            "parent_key": ",".join(str(v) for v in pkey),
            "child_pattern_before": str(before),
            "child_pattern_after": str(after),
            "p_occ_before": float(node.get("prob", 0.0)),
            "p_true_before": float(node.get("prob_true", 0.0)),
            "bit_each_before": float(node.get("bits", 0.0)),
            "local_support_count": int(local_support),
            "neighbor_density": int(local_support),
            "new_parent_count": int(new_parent),
            "isolated_add_count": int(isolated),
        })

    fn_rows = [
        dict(r)
        for r in _parse_json_payload(base_stats.get("sparsepcgc_top_low_prob_occupied_nodes_json", ""), [])
        if bool(r.get("occupied", True))
    ][: max(int(top_n), 0)]
    fp_rows = [
        dict(r)
        for r in _parse_json_payload(base_stats.get("sparsepcgc_top_high_prob_empty_nodes_json", ""), [])
        if not bool(r.get("occupied", False))
    ][: max(int(top_n), 0)]
    for node in fn_rows:
        source, _idx = _source_coord_for_node(node, coords, set())
        if source is None:
            continue
        add("prune_mispredicted_occupied", "false_negative", node, source, None)
        same = _safe_add_target_near(source, coords, same_parent=True, radius=1)
        if same is not None:
            add("add_support_near_false_negative", "false_negative", node, None, same)
            add("move_within_same_parent", "false_negative", node, source, same)
            add("child_pattern_one_flip", "false_negative", node, source, same)
        gp = _safe_add_target_near(source, coords, same_parent=False, radius=2)
        if gp is not None:
            add("move_within_same_grandparent", "false_negative", node, source, gp)
        snap = _occupied_neighbor_for_snap(source, coords, radius=1)
        if snap is not None:
            add("snap_or_merge", "false_negative", node, source, snap)
        if len(rows) >= int(max_candidates):
            break
    for node in fp_rows:
        if len(rows) >= int(max_candidates):
            break
        target = _target_coord_for_node(node, coords, occupied_keys, occupied, set())
        if target is None:
            continue
        offsets = torch.tensor([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1), (0,0,-1)], device=coords.device, dtype=torch.long)
        neigh_coords = target.reshape(1, 3).to(device=coords.device) + offsets
        occ = _lookup_occupied(occupied_keys(neigh_coords), occupied)
        if bool(occ.any().item()):
            add("remove_context_near_false_positive", "false_positive", node, neigh_coords[occ][0].clone(), target)
    return rows[: int(max_candidates)]


def _context_cluster_id(row: Mapping[str, object]) -> str:
    node = str(row.get("node_type", "node"))
    depth = str(row.get("depth", "-1"))
    pattern = str(row.get("child_pattern_before", ""))
    try:
        support = int(float(row.get("local_support_count", 0)))
    except Exception:
        support = 0
    support_bucket = "s0" if support <= 0 else "s1" if support <= 1 else "s2_3" if support <= 3 else "s4p"
    try:
        p = float(row.get("p_occ_before", 1.0))
    except Exception:
        p = 1.0
    p_bucket = "p_lt_1e-4" if p < 1e-4 else "p_lt_1e-3" if p < 1e-3 else "p_hi"
    return f"{node}_d{depth}_pat{pattern}_{support_bucket}_{p_bucket}"


def _cluster_summary_rows(
    *,
    file_path: str,
    base_stats: Mapping[str, object],
    candidate_rows: Sequence[Mapping[str, object]],
) -> list[Dict[str, object]]:
    total_bits = max(_safe_float(base_stats.get("sparsepcgc_estimated_occupancy_bits"), 0.0), 1e-9)
    clusters: Dict[str, list[Mapping[str, object]]] = {}
    for row in candidate_rows:
        clusters.setdefault(_context_cluster_id(row), []).append(row)
    out = []
    for cid, rows in sorted(clusters.items(), key=lambda kv: -sum(_safe_float(r.get("bit_each_before"), 0.0) for r in kv[1])):
        bits = [_safe_float(r.get("bit_each_before"), 0.0) for r in rows]
        edits = sorted(set(str(r.get("edit_type", "")) for r in rows))
        rep = rows[0] if rows else {}
        out.append({
            "dataset": _dataset_name(file_path),
            "sequence": _sequence_name(file_path),
            "frame_id": Path(file_path).stem,
            "codec_setting_id": "SparsePCGC_default",
            "candidate_name": "cluster_summary",
            "cluster_id": cid,
            "cluster_size": len(rows),
            "cluster_bit_share": float(sum(bits) / total_bits * 100.0),
            "mean_bit_each": _json_mean(bits),
            "representative_pattern": rep.get("child_pattern_before", ""),
            "recommended_edit_templates": json.dumps(edits, sort_keys=True),
            "context_rule_json": json.dumps({
                "depth": rep.get("depth", ""),
                "node_type": rep.get("node_type", ""),
                "support": rep.get("local_support_count", ""),
                "p_occ": rep.get("p_occ_before", ""),
                "bit_each": rep.get("bit_each_before", ""),
            }, sort_keys=True),
        })
    return out


def _remove_sources_add_targets(coords: torch.Tensor, sources: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]) -> torch.Tensor:
    device = coords.device
    source_set = {tuple(int(v) for v in s.to("cpu").tolist()) for s in sources if s is not None}
    keep = [tuple(int(v) for v in c.to("cpu").tolist()) not in source_set for c in coords]
    keep_mask = torch.tensor(keep, device=device, dtype=torch.bool)
    pieces = [coords[keep_mask]]
    valid_targets = [t.reshape(1, 3).to(device=device, dtype=torch.long) for t in targets if t is not None]
    if valid_targets:
        pieces.append(torch.cat(valid_targets, dim=0))
    return torch.unique(torch.cat(pieces, dim=0).to(dtype=torch.long), dim=0, sorted=True)


def _filter_rows_for_rule(rows: Sequence[Mapping[str, object]], rule: str) -> list[Mapping[str, object]]:
    if rule == "group_prune_low_support_FN":
        return [r for r in rows if r.get("edit_type") == "prune_mispredicted_occupied" and int(float(r.get("local_support_count", 0))) <= 2]
    if rule == "group_same_parent_move":
        return [r for r in rows if r.get("edit_type") == "move_within_same_parent"]
    if rule == "group_same_grandparent_move":
        return [r for r in rows if r.get("edit_type") == "move_within_same_grandparent"]
    if rule == "group_snap_or_merge":
        return [r for r in rows if r.get("edit_type") == "snap_or_merge"]
    if rule == "child_pattern_1to2_flip":
        return [r for r in rows if r.get("edit_type") == "child_pattern_one_flip"]
    if rule == "suppress_FP_context":
        return [r for r in rows if r.get("edit_type") == "remove_context_near_false_positive"]
    if rule == "safe_support_add":
        return [
            r for r in rows
            if r.get("edit_type") == "add_support_near_false_negative"
            and int(float(r.get("new_parent_count", 0))) == 0
            and int(float(r.get("isolated_add_count", 0))) == 0
        ]
    return []


def _group_rule_candidate(
    *,
    rule: str,
    coords: torch.Tensor,
    candidate_rows: Sequence[Mapping[str, object]],
    budget: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    rows = _filter_rows_for_rule(candidate_rows, rule)
    rows = sorted(rows, key=lambda r: _safe_float(r.get("bit_each_before"), 0.0), reverse=True)
    target = max(1, int(math.ceil(float(coords.shape[0]) * float(budget))))
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    sample = []
    used_source = set()
    used_target = set()
    for row in rows:
        if len(sources) + len(targets) >= target:
            break
        src = _parse_coord(row.get("source_coord"))
        tgt = _parse_coord(row.get("target_coord"))
        src_key = tuple(src.tolist()) if src is not None else None
        tgt_key = tuple(tgt.tolist()) if tgt is not None else None
        if src_key is not None and src_key in used_source:
            continue
        if tgt_key is not None and tgt_key in used_target:
            continue
        if src is not None:
            sources.append(src.to(device=coords.device))
            used_source.add(src_key)
        if tgt is not None and rule in {
            "group_same_parent_move",
            "group_same_grandparent_move",
            "child_pattern_1to2_flip",
            "safe_support_add",
        }:
            targets.append(tgt.to(device=coords.device))
            used_target.add(tgt_key)
        sample.append({
            "source": list(src_key) if src_key is not None else None,
            "target": list(tgt_key) if tgt_key is not None else None,
            "edit_type": row.get("edit_type", ""),
            "cluster_id": _context_cluster_id(row),
            "bit_each": row.get("bit_each_before", ""),
            "p_occ": row.get("p_occ_before", ""),
        })
    cand = _remove_sources_add_targets(coords, sources, targets)
    edit_count = len(sources) + len(targets)
    bits = [_safe_float(r.get("bit_each_before"), 0.0) for r in rows[:max(edit_count, 1)]]
    cluster_id = _context_cluster_id(rows[0]) if rows else f"{rule}_empty"
    debug = {
        "candidate_family": "phase2t_multi_rule",
        "candidate_variant": rule,
        "cluster_id": cluster_id,
        "cluster_size": len(rows),
        "cluster_bit_share": float(sum(bits)),
        "edit_type_counts_json": json.dumps({rule: int(edit_count)}, sort_keys=True),
        "context_rule_json": json.dumps({"rule": rule, "budget": budget, "source_count": len(sources), "target_count": len(targets)}, sort_keys=True),
        "source_target_sample_json": json.dumps(sample[:32], sort_keys=True),
        "actual_edit_ratio": float(abs(int(cand.shape[0]) - int(coords.shape[0])) + min(len(sources), len(targets))) / max(float(coords.shape[0]), 1.0),
        "point_count_delta": int(cand.shape[0]) - int(coords.shape[0]),
        "new_parent_count": int(sum(int(float(r.get("new_parent_count", 0))) for r in rows[:max(edit_count, 1)])),
        "isolated_add_count": int(sum(int(float(r.get("isolated_add_count", 0))) for r in rows[:max(edit_count, 1)])),
        "affected_parent_count": len(set(str(r.get("parent_key", "")) for r in rows[:max(edit_count, 1)])),
        "affected_block_count": "",
        "same_parent_edit_max": "",
        "selected_bit_sum": float(sum(bits)),
        "cheap_estimated_bits_delta": -float(sum(bits)) if rule != "safe_support_add" else float(sum(bits) * 0.5),
    }
    return cand, debug


def _multi_rule_candidate(
    *,
    candidate: str,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    candidate_rows: Sequence[Mapping[str, object]],
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
        debug.setdefault("cheap_estimated_bits_delta", -_safe_float(debug.get("selected_bit_sum"), 0.0))
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
        debug.setdefault("cheap_estimated_bits_delta", -_safe_float(debug.get("selected_bit_sum"), 0.0))
        return cand, debug
    if candidate == "hybrid_group_rule_beam":
        options = []
        for rule in (
            "group_prune_low_support_FN",
            "group_same_parent_move",
            "group_same_grandparent_move",
            "group_snap_or_merge",
            "child_pattern_1to2_flip",
            "suppress_FP_context",
        ):
            cand, dbg = _group_rule_candidate(rule=rule, coords=coords, candidate_rows=candidate_rows, budget=budget)
            risk = 1.0 + 4.0 * max(float(dbg.get("new_parent_count", 0) or 0), 0.0) + 2.0 * max(float(dbg.get("isolated_add_count", 0) or 0), 0.0)
            score = -_safe_float(dbg.get("cheap_estimated_bits_delta"), 0.0) / max(risk, 1e-6)
            options.append((score, cand, dbg, rule))
        options.sort(key=lambda x: x[0], reverse=True)
        _score, cand, dbg, chosen = options[0]
        dbg = dict(dbg)
        dbg["candidate_variant"] = "hybrid_group_rule_beam"
        dbg["candidate_family"] = "phase2t_hybrid_group_rule_beam"
        dbg["edit_type_counts_json"] = json.dumps({"hybrid_chosen": chosen}, sort_keys=True)
        return cand, dbg
    return _group_rule_candidate(rule=candidate, coords=coords, candidate_rows=candidate_rows, budget=budget)


def _actual_bit_percent(bit_encoder, args, coords, meta, base_bits: float, cache: Dict[str, Mapping[str, object]]):
    sig = _coords_signature(coords)
    if sig in cache:
        stats = dict(cache[sig])
        return (float(stats.get("bit", 0.0)) - float(base_bits)) / max(float(base_bits), 1e-9) * 100.0, stats, True
    stats = bit_encoder.encode_bits(_coords_to_xyz(coords, meta, args))
    cache[sig] = dict(stats)
    return (float(stats.get("bit", 0.0)) - float(base_bits)) / max(float(base_bits), 1e-9) * 100.0, stats, False


def run_phase2t(cli: argparse.Namespace) -> int:
    args = _prepare_args(cli)
    debug_args = args
    debug_args.sparsepcgc_skip_decode = True
    max_pool = max(int(float(x)) for x in _parse_csv_text(cli.pools))
    debug_args.sparsepcgc_occupancy_debug_topk_final = int(max_pool)
    debug_args.sparsepcgc_occupancy_debug_topk_per_layer = max(512, min(int(max_pool), 8192))
    bit_args = copy.copy(args)
    bit_args.sparsepcgc_skip_decode = not bool(cli.decode_quality)
    bit_args.enable_sparsepcgc_occupancy_debug = False
    if bool(cli.decode_quality):
        Path(cli.decoded_dir).mkdir(parents=True, exist_ok=True)
        bit_args.sparsepcgc_decoded_copy_dir = str(cli.decoded_dir)

    rows = _read_rows(cli.output_csv) if bool(cli.append_output) else []
    debug_encoder = build_actual_encoder(debug_args)
    bit_encoder = build_actual_encoder(bit_args)
    actual_cache: Dict[str, Mapping[str, object]] = {}
    eval_cache: Dict[str, Dict[str, object]] = {}
    actual_eval_count = 0
    cache_hit_count = 0
    duplicate_skip = 0
    try:
        for file_idx, file_path in enumerate(cli.files):
            t_file = time.time()
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            sequence = _sequence_name(file_path)
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, args)
            base_stats = debug_encoder.encode_bits(base_xyz)
            base_bit_stats = bit_encoder.encode_bits(base_xyz)
            base_bits = float(base_bit_stats.get("bit", base_stats.get("bit", 0.0)))
            decoded_gt_path = str(base_bit_stats.get("decoded_copy_path", ""))
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
                rows.append(_base_headroom_row(file_path=file_path, base_stats=base_stats))
                _write_csv(cli.output_csv, rows)

            candidate_rows = _fast_candidate_rows_for_frame(
                coords=coords,
                base_stats=base_stats,
                top_n=int(cli.top_nodes),
                max_candidates=int(cli.max_counterfactual_candidates),
            )
            if bool(cli.emit_clusters):
                rows.extend(_cluster_summary_rows(file_path=file_path, base_stats=base_stats, candidate_rows=candidate_rows))
                _write_csv(cli.output_csv, rows)

            generated = []
            for budget in [float(x) for x in _parse_csv_text(cli.budgets)]:
                for candidate in _parse_csv_text(cli.candidates):
                    t_gen = time.time()
                    try:
                        cand_coords, debug = _multi_rule_candidate(
                            candidate=candidate,
                            coords=coords,
                            base_stats=base_stats,
                            candidate_rows=candidate_rows,
                            budget=budget,
                            pool=max_pool,
                            block_size=int(cli.block_size),
                            seed=int(cli.seed) + int(file_idx),
                        )
                    except Exception as exc:
                        rows.append({
                            "dataset": _dataset_name(file_path),
                            "sequence": sequence,
                            "frame_id": frame_id,
                            "codec_setting_id": "SparsePCGC_default",
                            "candidate_name": candidate,
                            "edit_budget_ratio": budget,
                            "error": repr(exc),
                        })
                        continue
                    sig = _coords_signature(cand_coords)
                    cheap_delta = _safe_float(debug.get("cheap_estimated_bits_delta"), 0.0)
                    generated.append((cheap_delta, candidate, budget, cand_coords, dict(debug), time.time() - t_gen, sig))
            generated.sort(key=lambda x: x[0])
            selected = []
            seen = set()
            for item in generated:
                if int(cli.actual_topk) <= 0:
                    break
                if item[-1] in seen:
                    duplicate_skip += 1
                    continue
                seen.add(item[-1])
                selected.append(item)
                if len(selected) >= int(cli.actual_topk):
                    break

            for rank, (cheap_delta, candidate, budget, cand_coords, debug, t_gen, sig) in enumerate(selected):
                prob_update: Dict[str, object] = {}
                if rank < int(cli.prob_eval_topk):
                    after_stats = debug_encoder.encode_bits(_coords_to_xyz(cand_coords, meta, args))
                    prob_update = _probability_row_updates(before=base_stats, after=after_stats)
                if bool(cli.decode_quality):
                    row, hit, t_encode, t_quality = _eval_decoded_row(
                        file_path=str(file_path),
                        sequence=sequence,
                        frame_id=frame_id,
                        candidate_name=candidate,
                        edit_sequence=str(debug.get("candidate_variant", candidate)),
                        budget=float(budget),
                        pool=int(max_pool),
                        coords=coords,
                        cand_coords=cand_coords,
                        meta=meta,
                        args=args,
                        base_bits=base_bits,
                        baseline_stats=base_bit_stats,
                        baseline_quality=baseline_quality,
                        decoded_gt_path=decoded_gt_path,
                        baseline_count=baseline_count,
                        baseline_match=baseline_match,
                        baseline_lossless=baseline_lossless,
                        decode_encoder=bit_encoder,
                        debug=debug,
                        cli=cli,
                        eval_cache=eval_cache,
                    )
                    cache_hit_count += int(hit)
                    actual_eval_count += 0 if hit else 1
                else:
                    actual_raw, stats, hit = _actual_bit_percent(bit_encoder, args, cand_coords, meta, base_bits, actual_cache)
                    cache_hit_count += int(hit)
                    actual_eval_count += 0 if hit else 1
                    row = {
                        "file": str(file_path),
                        "dataset": _dataset_name(file_path),
                        "sequence": sequence,
                        "frame_id": frame_id,
                        "codec_setting_id": "SparsePCGC_default",
                        "candidate_name": candidate,
                        "edit_budget_ratio": float(budget),
                        "actual_raw_percent": actual_raw,
                        "raw_bit": stats.get("bit", ""),
                        "base_bit": base_bits,
                        "D1_PSNR": "",
                        "Chamfer": "",
                    }
                point_delta = int(cand_coords.shape[0]) - int(coords.shape[0])
                edit_ratio = float(abs(point_delta)) / max(float(coords.shape[0]), 1.0)
                est_delta = _safe_float(prob_update.get("estimated_bits_delta"), cheap_delta)
                row.update({
                    "dataset": _dataset_name(file_path),
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "codec_setting_id": "SparsePCGC_default",
                    "candidate_name": candidate,
                    "cluster_id": debug.get("cluster_id", debug.get("selection_unit_type", "")),
                    "actual_edit_ratio": debug.get("actual_edit_ratio", edit_ratio),
                    "point_count_delta": point_delta,
                    "estimated_bits_delta": est_delta,
                    "raw_gain_per_edit": -_safe_float(row.get("actual_raw_percent"), 0.0) / max(float(debug.get("actual_edit_ratio", edit_ratio) or 0.0), 1e-9),
                    "p_true_delta": prob_update.get("p_true_delta", ""),
                    "low_p_count_delta": prob_update.get("low_p_true_count_delta", ""),
                    "high_bit_count_delta": prob_update.get("high_bit_symbol_count_delta", ""),
                    "cluster_size": debug.get("cluster_size", ""),
                    "cluster_bit_share": debug.get("cluster_bit_share", ""),
                    "edit_type_counts_json": debug.get("edit_type_counts_json", ""),
                    "context_rule_json": debug.get("context_rule_json", ""),
                    "source_target_sample_json": debug.get("source_target_sample_json", ""),
                    "new_parent_count": debug.get("new_parent_count", debug.get("created_new_parent_count", "")),
                    "isolated_add_count": debug.get("isolated_add_count", debug.get("isolated_added_voxel_count", "")),
                    "affected_parent_count": debug.get("affected_parent_count", ""),
                    "affected_block_count": debug.get("affected_block_count", ""),
                    "same_parent_edit_max": debug.get("same_parent_edit_max", ""),
                    "cache_hit_count": cache_hit_count,
                    "duplicate_skip_count": duplicate_skip,
                    "actual_eval_count": actual_eval_count,
                    "candidate_generate_time": t_gen,
                    "frame_elapsed_time": time.time() - t_file,
                })
                row.update(debug)
                row.update(prob_update)
                rows.append(row)
                _write_csv(cli.output_csv, rows)
            print(json.dumps({
                "phase2t": True,
                "sequence": sequence,
                "frame": frame_id,
                "generated": len(generated),
                "actual_eval_count": actual_eval_count,
                "cache_hit_count": cache_hit_count,
                "duplicate_skip_count": duplicate_skip,
            }, sort_keys=True), flush=True)
    finally:
        for enc in (debug_encoder, bit_encoder):
            close = getattr(enc, "close", None)
            if callable(close):
                close()
    _write_csv(cli.output_csv, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2T multi-rule context edit and headroom probe")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--budgets", default="0.005,0.010,0.020,0.030")
    parser.add_argument("--pools", default="8192")
    parser.add_argument("--top-nodes", type=int, default=96)
    parser.add_argument("--max-counterfactual-candidates", type=int, default=256)
    parser.add_argument("--actual-topk", type=int, default=8)
    parser.add_argument("--prob-eval-topk", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emit-headroom", action="store_true")
    parser.add_argument("--emit-clusters", action="store_true")
    parser.add_argument("--decode-quality", action="store_true")
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2t_decoded")
    parser.add_argument("--quality-max-points", type=int, default=3000)
    parser.add_argument("--normal-max-points", type=int, default=3000)
    parser.add_argument("--pc-error-path", default="")
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2t(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
