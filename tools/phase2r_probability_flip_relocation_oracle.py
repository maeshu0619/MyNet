#!/usr/bin/env python
"""Phase2R probability-flip relocation oracle.

Research-only script.  It pairs unexpected occupied symbols (low p_occ,
occupied) with expected occupied-but-empty symbols (high p_occ, empty), applies
Prune+Add relocation, and measures actual SparsePCGC raw bit size.
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
    _parse_csv_text,
    _safe_float,
    _unique_coords,
)
from tools.phase2_rdo_beam_probe import _coords_signature, _mask_for_scaled_node, _parse_json_payload, _write_csv
from tools.phase2q_probability_guided_context_edit import _candidate_to_coords_q


DEFAULT_CANDIDATES = (
    "codec_only",
    "high_bit_raw_prune",
    "snap_to_existing_only",
    "low_prob_snap_to_existing",
    "prob_flip_global_oracle",
    "prob_flip_local_r1",
    "prob_flip_local_r2",
    "prob_flip_local_r4",
    "prob_flip_same_parent",
    "delta_guided_flip_beam",
)


def _prepare_args(_cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2R probability flip relocation"),
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


def _json_counts(value: object) -> Dict[str, float]:
    try:
        parsed = json.loads(str(value or "{}"))
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {}


def _gain_source(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log2((1.0 - p) / p)


def _gain_target(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log2(p / (1.0 - p))


def _node_base_coord(row: Mapping[str, object], device) -> torch.Tensor:
    depth = int(row.get("depth", 0))
    coord = torch.tensor(list(row.get("coord", []))[:3], device=device, dtype=torch.long)
    if int(coord.numel()) < 3:
        coord = torch.zeros((3,), device=device, dtype=torch.long)
    scale = int(2 ** max(depth, 0))
    return coord * scale


def _source_coord_for_node(
    row: Mapping[str, object],
    coords: torch.Tensor,
    used: set[Tuple[int, int, int]],
) -> Tuple[torch.Tensor | None, int]:
    depth = int(row.get("depth", 0))
    mask = _mask_for_scaled_node(coords, row.get("coord", []), depth)
    idxs = mask.nonzero(as_tuple=False).reshape(-1)
    for idx in idxs.tolist():
        c = tuple(int(v) for v in coords[int(idx)].tolist())
        if c not in used:
            return coords[int(idx)].clone(), int(idx)
    return None, -1


def _target_coord_for_node(
    row: Mapping[str, object],
    coords: torch.Tensor,
    occupied_keys,
    occupied,
    used: set[Tuple[int, int, int]],
) -> torch.Tensor | None:
    depth = int(row.get("depth", 0))
    base = _node_base_coord(row, coords.device)
    scale = int(2 ** max(depth, 0))
    candidates = [base]
    if scale > 1:
        candidates.extend(
            [
                base + torch.tensor([scale // 2, scale // 2, scale // 2], device=coords.device, dtype=torch.long),
                base + torch.tensor([scale - 1, 0, 0], device=coords.device, dtype=torch.long),
                base + torch.tensor([0, scale - 1, 0], device=coords.device, dtype=torch.long),
                base + torch.tensor([0, 0, scale - 1], device=coords.device, dtype=torch.long),
            ]
        )
    for cand in candidates:
        key = tuple(int(v) for v in cand.tolist())
        if key in used:
            continue
        if bool(_lookup_occupied(occupied_keys(cand.reshape(1, 3)), occupied)[0].item()):
            continue
        return cand.clone()
    return None


def _source_rows(stats: Mapping[str, object]) -> list[Dict[str, object]]:
    rows = _parse_json_payload(stats.get("sparsepcgc_top_low_prob_occupied_nodes_json", ""), [])
    if not rows:
        rows = [r for r in _parse_json_payload(stats.get("sparsepcgc_top_high_bit_nodes_json", ""), []) if bool(r.get("occupied", False))]
    out = []
    for r in rows:
        p = float(r.get("prob", 1.0))
        g = _gain_source(p)
        if g <= 0:
            continue
        d = dict(r)
        d["source_gain"] = float(g)
        out.append(d)
    out.sort(key=lambda r: float(r.get("source_gain", 0.0)) + float(r.get("bits", 0.0)), reverse=True)
    return out


def _target_rows(stats: Mapping[str, object]) -> list[Dict[str, object]]:
    rows = _parse_json_payload(stats.get("sparsepcgc_top_high_prob_empty_nodes_json", ""), [])
    if not rows:
        rows = [r for r in _parse_json_payload(stats.get("sparsepcgc_top_high_bit_nodes_json", ""), []) if not bool(r.get("occupied", False))]
    out = []
    for r in rows:
        p = float(r.get("prob", 0.0))
        g = _gain_target(p)
        if g <= 0:
            continue
        d = dict(r)
        d["target_gain"] = float(g)
        out.append(d)
    out.sort(key=lambda r: float(r.get("target_gain", 0.0)) + float(r.get("bits", 0.0)), reverse=True)
    return out


def _build_pairs(
    *,
    coords: torch.Tensor,
    stats: Mapping[str, object],
    budget: float,
    mode: str,
    radius: float = 0.0,
) -> Tuple[list[Tuple[torch.Tensor, torch.Tensor, Mapping[str, object], Mapping[str, object], float]], Dict[str, object]]:
    target_count = min(max(int(math.ceil(float(coords.shape[0]) * float(budget))), 0), max(int(coords.shape[0]) - 1, 0))
    sources = _source_rows(stats)
    targets = _target_rows(stats)
    occupied_keys, occupied = _coord_key_setup(coords)
    used_sources: set[Tuple[int, int, int]] = set()
    used_targets: set[Tuple[int, int, int]] = set()
    prepared_targets = []
    for tr in targets:
        tc = _target_coord_for_node(tr, coords, occupied_keys, occupied, used_targets)
        if tc is None:
            continue
        prepared_targets.append((tc, tr, float(tr.get("target_gain", 0.0))))
    target_by_coord: Dict[Tuple[int, int, int], Tuple[torch.Tensor, Mapping[str, object], float]] = {}
    targets_by_parent: Dict[Tuple[int, int, int], list[Tuple[torch.Tensor, Mapping[str, object], float]]] = {}
    for item in prepared_targets:
        tc, tr, tg = item
        key = tuple(int(v) for v in tc.tolist())
        if key not in target_by_coord or float(tg) > float(target_by_coord[key][2]):
            target_by_coord[key] = item
        pkey = tuple(int(v) for v in torch.div(tc, 2, rounding_mode="floor").tolist())
        targets_by_parent.setdefault(pkey, []).append(item)
    for plist in targets_by_parent.values():
        plist.sort(key=lambda item: float(item[2]), reverse=True)
    local_offsets = []
    if str(mode) == "local":
        r = int(math.ceil(float(radius)))
        r2 = float(radius) * float(radius)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx * dx + dy * dy + dz * dz <= r2:
                        local_offsets.append((dx, dy, dz))
    pairs = []
    same_parent_hits = 0
    for source_index, sr in enumerate(sources):
        if len(pairs) >= target_count:
            break
        sc, _idx = _source_coord_for_node(sr, coords, used_sources)
        if sc is None:
            continue
        sp = torch.div(sc, 2, rounding_mode="floor")
        picked = None
        if str(mode) == "global":
            # Both lists are already sorted by gain, so greedy pairing is the
            # intended oracle upper bound without O(N^2) search.
            for tc, tr, tg in prepared_targets[source_index:]:
                tkey = tuple(int(v) for v in tc.tolist())
                if tkey not in used_targets:
                    picked = (tc, tr, tg)
                    break
        elif str(mode) == "same_parent":
            pkey = tuple(int(v) for v in sp.tolist())
            for item in targets_by_parent.get(pkey, []):
                tkey = tuple(int(v) for v in item[0].tolist())
                if tkey not in used_targets:
                    picked = item
                    break
        else:
            best_score = -1e18
            for off in local_offsets:
                key = (
                    int(sc[0].item()) + int(off[0]),
                    int(sc[1].item()) + int(off[1]),
                    int(sc[2].item()) + int(off[2]),
                )
                item = target_by_coord.get(key)
                if item is None or key in used_targets:
                    continue
                score = float(sr.get("source_gain", 0.0)) + float(item[2])
                if score > best_score:
                    best_score = score
                    picked = item
        if picked is None:
            continue
        tc, tr, tg = picked
        skey = tuple(int(v) for v in sc.tolist())
        tkey = tuple(int(v) for v in tc.tolist())
        if skey in used_sources or tkey in used_targets:
            continue
        used_sources.add(skey)
        used_targets.add(tkey)
        same_parent = bool((torch.div(sc, 2, rounding_mode="floor") == torch.div(tc, 2, rounding_mode="floor")).all().item())
        same_parent_hits += int(same_parent)
        gain = float(sr.get("source_gain", 0.0)) + float(tr.get("target_gain", 0.0))
        pairs.append((sc, tc, sr, tr, gain))
    debug = {
        "source_candidate_count": int(len(sources)),
        "target_candidate_count": int(len(targets)),
        "same_parent_ratio": float(same_parent_hits) / max(float(len(pairs)), 1.0),
    }
    return pairs, debug


def _apply_pairs(coords: torch.Tensor, pairs) -> torch.Tensor:
    if not pairs:
        return coords.clone()
    source_set = {tuple(int(v) for v in sc.tolist()) for sc, _tc, _sr, _tr, _g in pairs}
    keep = []
    for c in coords:
        keep.append(tuple(int(v) for v in c.tolist()) not in source_set)
    keep_mask = torch.tensor(keep, device=coords.device, dtype=torch.bool)
    targets = torch.stack([tc for _sc, tc, _sr, _tr, _g in pairs], dim=0).to(dtype=torch.long)
    return torch.unique(torch.cat([coords[keep_mask], targets], dim=0).to(dtype=torch.long), dim=0, sorted=True)


def _relocation_candidate(
    *,
    candidate: str,
    coords: torch.Tensor,
    stats: Mapping[str, object],
    budget: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if candidate == "prob_flip_global_oracle":
        pairs, dbg = _build_pairs(coords=coords, stats=stats, budget=budget, mode="global")
        radius = ""
    elif candidate.startswith("prob_flip_local_r"):
        radius = float(candidate.rsplit("r", 1)[1])
        pairs, dbg = _build_pairs(coords=coords, stats=stats, budget=budget, mode="local", radius=radius)
    elif candidate == "prob_flip_same_parent":
        pairs, dbg = _build_pairs(coords=coords, stats=stats, budget=budget, mode="same_parent")
        radius = ""
    else:
        # Cheap oracle beam: choose whichever constraint has the largest
        # theoretical pair gain before actual evaluation.
        choices = []
        for name, mode, rad in [
            ("global", "global", 0.0),
            ("local2", "local", 2.0),
            ("same_parent", "same_parent", 0.0),
        ]:
            ps, d = _build_pairs(coords=coords, stats=stats, budget=budget, mode=mode, radius=rad)
            choices.append((sum(float(p[-1]) for p in ps), name, ps, d, rad))
        choices.sort(key=lambda item: item[0], reverse=True)
        _gain, chosen, pairs, dbg, radius = choices[0]
        dbg["beam_chosen"] = chosen
    cand = _apply_pairs(coords, pairs)
    source_probs = [float(sr.get("prob", 0.0)) for _sc, _tc, sr, _tr, _g in pairs]
    target_probs = [float(tr.get("prob", 0.0)) for _sc, _tc, _sr, tr, _g in pairs]
    source_bits = [float(sr.get("bits", 0.0)) for _sc, _tc, sr, _tr, _g in pairs]
    target_empty_costs = [float(tr.get("bits", 0.0)) for _sc, _tc, _sr, tr, _g in pairs]
    sample = []
    for sc, tc, sr, tr, gain in pairs[:32]:
        sample.append(
            {
                "source": [int(v) for v in sc.tolist()],
                "target": [int(v) for v in tc.tolist()],
                "source_p_occ": float(sr.get("prob", 0.0)),
                "target_p_occ": float(tr.get("prob", 0.0)),
                "source_bits": float(sr.get("bits", 0.0)),
                "target_bits": float(tr.get("bits", 0.0)),
                "pair_gain": float(gain),
            }
        )
    out = {
        **dbg,
        "source_count": int(len(pairs)),
        "target_count": int(len(pairs)),
        "theoretical_flip_gain_sum": float(sum(float(p[-1]) for p in pairs)),
        "source_p_occ_mean": float(sum(source_probs) / len(source_probs)) if source_probs else 0.0,
        "target_p_occ_mean": float(sum(target_probs) / len(target_probs)) if target_probs else 0.0,
        "source_bit_each_mean": float(sum(source_bits) / len(source_bits)) if source_bits else 0.0,
        "target_empty_cost_mean": float(sum(target_empty_costs) / len(target_empty_costs)) if target_empty_costs else 0.0,
        "radius": radius,
        "source_target_pairs_json_sample": json.dumps(sample, sort_keys=True),
    }
    return cand, out


def _candidate_to_coords_r(
    *,
    candidate: str,
    coords: torch.Tensor,
    stats: Mapping[str, object],
    budget: float,
    pool: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if candidate == "codec_only":
        return coords.clone(), {"source_count": 0, "target_count": 0}
    if candidate in {"high_bit_raw_prune", "snap_to_existing_only", "low_prob_snap_to_existing"}:
        cand, _mask, debug = _candidate_to_coords_q(
            candidate=candidate,
            coords=coords,
            base_stats=stats,
            budget=budget,
            pool=pool,
            block_size=block_size,
            seed=seed,
        )
        return cand, dict(debug)
    return _relocation_candidate(candidate=candidate, coords=coords, stats=stats, budget=budget)


def _count_add_metrics(coords: torch.Tensor, cand: torch.Tensor) -> Dict[str, object]:
    keys, occupied = _coord_key_setup(coords)
    cand_keys = keys(cand)
    is_original = _lookup_occupied(cand_keys, occupied)
    added = cand[~is_original]
    original_parent = torch.div(coords, 2, rounding_mode="floor")
    keys_parent, occ_parent = _coord_key_setup(original_parent)
    created = 0
    if int(added.shape[0]):
        added_parent = torch.div(added, 2, rounding_mode="floor")
        created = int((~_lookup_occupied(keys_parent(added_parent), occ_parent)).sum().item())
    isolated = 0
    if int(added.shape[0]):
        offsets = torch.tensor([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)], device=coords.device, dtype=torch.long)
        for a in added:
            support = int(_lookup_occupied(keys(a.reshape(1, 3) + offsets), occupied).sum().item())
            isolated += int(support <= 1)
    return {"new_parent_count": int(created), "isolated_add_count": int(isolated)}


def run_phase2r(cli: argparse.Namespace) -> int:
    args = _prepare_args(cli)
    debug_args = args
    debug_args.sparsepcgc_skip_decode = True
    max_pool = max(int(float(x)) for x in _parse_csv_text(cli.pools))
    debug_args.sparsepcgc_occupancy_debug_topk_final = int(max_pool)
    debug_args.sparsepcgc_occupancy_debug_topk_per_layer = max(1024, min(int(max_pool), 8192))
    bit_args = copy.copy(args)
    bit_args.sparsepcgc_skip_decode = True
    bit_args.enable_sparsepcgc_occupancy_debug = False

    rows = _read_rows(cli.output_csv) if bool(cli.append_output) else []
    eval_cache: Dict[str, Dict[str, object]] = {}
    actual_eval_count = 0
    cache_hit_count = 0
    duplicate_skip_count = 0
    debug_encoder = build_actual_encoder(debug_args)
    bit_encoder = build_actual_encoder(bit_args)
    try:
        for file_idx, file_path in enumerate(cli.files):
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            sequence = Path(file_path).parent.name
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, args)
            base_stats = debug_encoder.encode_bits(base_xyz)
            base_bit_stats = bit_encoder.encode_bits(base_xyz)
            base_bits = float(base_bit_stats.get("bit", base_stats.get("bit", 0.0)))
            empty_counts = _json_counts(base_stats.get("sparsepcgc_empty_high_prob_threshold_counts_json"))
            occupied_counts = _json_counts(base_stats.get("sparsepcgc_occupied_low_prob_threshold_counts_json"))
            for pool in [int(float(x)) for x in _parse_csv_text(cli.pools)]:
                for budget in [float(x) for x in _parse_csv_text(cli.budgets)]:
                    seen = set()
                    for candidate in _parse_csv_text(cli.candidates):
                        t0 = time.time()
                        cand_coords, debug = _candidate_to_coords_r(
                            candidate=candidate,
                            coords=coords,
                            stats=base_stats,
                            budget=budget,
                            pool=pool,
                            block_size=int(cli.block_size),
                            seed=int(cli.seed) + file_idx,
                        )
                        gen_time = time.time() - t0
                        sig = _coords_signature(cand_coords)
                        if sig in seen:
                            duplicate_skip_count += 1
                            continue
                        seen.add(sig)
                        encode_t0 = time.time()
                        if sig in eval_cache:
                            edited_stats = dict(eval_cache[sig])
                            cache_hit_count += 1
                            hit = True
                        else:
                            edited_stats = bit_encoder.encode_bits(_coords_to_xyz(cand_coords, meta, args))
                            eval_cache[sig] = dict(edited_stats)
                            actual_eval_count += 1
                            hit = False
                        encode_time = time.time() - encode_t0
                        edited_bits = float(edited_stats.get("bit", 0.0))
                        raw_percent = (edited_bits - base_bits) / max(base_bits, 1e-9) * 100.0
                        add_metrics = _count_add_metrics(coords, cand_coords)
                        edit_count = max(int(coords.shape[0]), int(cand_coords.shape[0])) - min(int(coords.shape[0]), int(cand_coords.shape[0]))
                        if int(debug.get("source_count", 0) or 0):
                            edit_count = int(debug.get("source_count", 0))
                        row = {
                            "file": str(file_path),
                            "sequence": sequence,
                            "frame_id": frame_id,
                            "candidate_name": candidate,
                            "budget_ratio": float(budget),
                            "actual_edit_ratio": float(edit_count) / max(float(coords.shape[0]), 1.0),
                            "source_count": int(debug.get("source_count", 0) or 0),
                            "target_count": int(debug.get("target_count", 0) or 0),
                            "actual_raw_percent": float(raw_percent),
                            "original_bit_size": float(base_bits),
                            "edited_bit_size": float(edited_bits),
                            "estimated_bits_delta": _safe_float(edited_stats.get("sparsepcgc_estimated_occupancy_bits"), float("nan")) - _safe_float(base_stats.get("sparsepcgc_estimated_occupancy_bits"), float("nan")),
                            "candidate_generate_time": float(gen_time),
                            "encode_time": float(encode_time),
                            "duplicate_skip_count": int(duplicate_skip_count),
                            "cache_hit_count": int(cache_hit_count),
                            "actual_eval_count": int(actual_eval_count),
                            "p_empty_ge_0p5_count": int(empty_counts.get("0.5", 0)),
                            "p_empty_ge_0p7_count": int(empty_counts.get("0.7", 0)),
                            "p_empty_ge_0p8_count": int(empty_counts.get("0.8", 0)),
                            "p_empty_ge_0p9_count": int(empty_counts.get("0.9", 0)),
                            "p_occ_lt_0p5_occupied_count": int(occupied_counts.get("0.5", 0)),
                            "p_occ_lt_0p3_occupied_count": int(occupied_counts.get("0.3", 0)),
                            "p_occ_lt_0p2_occupied_count": int(occupied_counts.get("0.2", 0)),
                            "high_p_empty_topk": base_stats.get("sparsepcgc_top_high_prob_empty_nodes_json", ""),
                            "low_p_occupied_topk": base_stats.get("sparsepcgc_top_low_prob_occupied_nodes_json", ""),
                        }
                        row.update(debug)
                        row.update(add_metrics)
                        rows.append(row)
                        _write_csv(cli.output_csv, rows)
                    print(
                        json.dumps(
                            {
                                "phase2r": True,
                                "sequence": sequence,
                                "frame": frame_id,
                                "budget": budget,
                                "pool": pool,
                                "rows": len(rows),
                                "actual_eval_count": actual_eval_count,
                                "cache_hit_count": cache_hit_count,
                                "duplicate_skip_count": duplicate_skip_count,
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
    parser = argparse.ArgumentParser(description="Phase2R probability-flip relocation oracle")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--budgets", default="0.005,0.010,0.020,0.030")
    parser.add_argument("--pools", default="131072")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2r(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
