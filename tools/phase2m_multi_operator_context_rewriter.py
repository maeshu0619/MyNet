#!/usr/bin/env python
"""Phase2M multi-operator RD context rewriter probe.

Research-only script.  It reuses Phase2J/K helpers and keeps all edits outside
the training code.  The beam search uses cheap context/geometry scores to limit
actual SparsePCGC encode/decode calls to a small final candidate set.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
    _coords_to_xyz,
    _parse_csv_text,
    _safe_float,
    _unique_coords,
)
from tools.phase2_rdo_beam_probe import (
    _coord_match_ratio_from_paths,
    _coords_signature,
    _phase2j_row,
    _quality_from_paths,
    _write_csv,
)
from tools.phase2k_context_rewrite_rd_probe import (
    _candidate_to_coords,
    _prepare_args,
    _row_extra,
)


BASELINE_CANDIDATES = ("block_only", "high_bit_raw_prune", "move_snap_context_projection")
DEFAULT_BEAM_OPS = (
    "high_bit_raw_prune",
    "move_snap_context_projection",
    "high_bit_context_rewrite_prune_add",
    "high_bit_surface_preserving_decimation",
    "block_only_decomposition_with_repair",
)


@dataclass
class RewriteState:
    coords: torch.Tensor
    edit_sequence: str
    cheap_score: float
    total_prune_count: int = 0
    total_add_count: int = 0
    total_move_count: int = 0
    total_merge_count: int = 0
    debug: Dict[str, object] = field(default_factory=dict)


def _op_counts_from_debug(debug: Mapping[str, object]) -> Tuple[int, int, int, int]:
    return (
        int(_safe_float(debug.get("prune_count"), 0.0)),
        int(_safe_float(debug.get("add_count"), 0.0)),
        int(_safe_float(debug.get("move_count"), 0.0)),
        int(_safe_float(debug.get("merge_count"), 0.0)),
    )


def _cheap_score(debug: Mapping[str, object], *, state: RewriteState, budget_ratio: float) -> float:
    bit_sum = _safe_float(debug.get("selected_bit_sum"), 0.0)
    bit_mean = _safe_float(debug.get("selected_bit_mean"), 0.0)
    hole = _safe_float(debug.get("hole_risk"), 0.0)
    boundary = _safe_float(debug.get("boundary_removed_ratio"), 0.0)
    add_count = _safe_float(debug.get("add_count"), 0.0)
    move_count = _safe_float(debug.get("move_count"), 0.0)
    actual_prune = _safe_float(debug.get("actual_prune_ratio"), 0.0)
    # Keep this cheap score intentionally simple and interpretable: rate first,
    # then veto obvious quality damage and excessive repair.
    return (
        float(state.cheap_score)
        + bit_sum * 2.5e-4
        + bit_mean * 2.0e-2
        + min(actual_prune, float(budget_ratio)) * 350.0
        - hole * 18.0
        - boundary * 12.0
        - add_count * 3.0e-4
        - move_count * 5.0e-5
    )


def _merge_debug(prev: Mapping[str, object], op_debug: Mapping[str, object], *, op_name: str) -> Dict[str, object]:
    out = dict(op_debug)
    prev_ops = []
    try:
        prev_ops = list(json.loads(str(prev.get("operation_counts_json", "[]"))))
    except Exception:
        prev_ops = []
    prev_ops.append(op_name)
    out["operation_counts_json"] = json.dumps(prev_ops, sort_keys=True)
    out["component_operations"] = json.dumps(prev_ops, sort_keys=True)
    out["component_order"] = "->".join(prev_ops)
    return out


def _build_multi_operator_beam(
    *,
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    budget_ratio: float,
    pool: int,
    block_size: int,
    seed: int,
    beam_width: int,
    iterations: int,
    ops: Sequence[str],
) -> Tuple[List[RewriteState], Dict[str, int]]:
    beam = [RewriteState(coords=coords, edit_sequence="start", cheap_score=0.0)]
    cache: Dict[str, RewriteState] = {_coords_signature(coords): beam[0]}
    stats = {
        "generated": 0,
        "duplicate_skip": 0,
        "cache_hit": 0,
    }
    for iteration in range(int(iterations)):
        next_states: List[RewriteState] = []
        for state_rank, state in enumerate(beam):
            # Stop/no-op keeps current state in the beam.
            next_states.append(state)
            used_ratio = float(max(int(coords.shape[0]) - int(state.coords.shape[0]), 0)) / max(float(coords.shape[0]), 1.0)
            remaining = max(float(budget_ratio) - used_ratio, 0.0)
            if remaining <= 1e-6:
                continue
            step_budget = min(max(float(budget_ratio) / max(float(iterations), 1.0), 0.0025), remaining)
            for op_idx, op_name in enumerate(ops):
                stats["generated"] += 1
                try:
                    cand_coords, _drop, op_debug = _candidate_to_coords(
                        candidate=op_name,
                        coords=state.coords,
                        base_stats=base_stats,
                        budget=step_budget,
                        pool=int(pool),
                        block_size=int(block_size),
                        seed=int(seed) + iteration * 100 + state_rank * 17 + op_idx,
                    )
                except Exception as exc:
                    continue
                sig = _coords_signature(cand_coords)
                if sig in cache:
                    stats["duplicate_skip"] += 1
                    continue
                op_prune, op_add, op_move, op_merge = _op_counts_from_debug(op_debug)
                debug = _merge_debug(state.debug, op_debug, op_name=op_name)
                new_state = RewriteState(
                    coords=cand_coords,
                    edit_sequence=(state.edit_sequence + "->" + op_name) if state.edit_sequence != "start" else op_name,
                    cheap_score=_cheap_score(op_debug, state=state, budget_ratio=budget_ratio),
                    total_prune_count=int(state.total_prune_count) + op_prune,
                    total_add_count=int(state.total_add_count) + op_add,
                    total_move_count=int(state.total_move_count) + op_move,
                    total_merge_count=int(state.total_merge_count) + op_merge,
                    debug=debug,
                )
                cache[sig] = new_state
                next_states.append(new_state)
        next_states.sort(key=lambda s: float(s.cheap_score), reverse=True)
        beam = next_states[: max(int(beam_width), 1)]
    return beam, stats


def _read_done_keys(path: str) -> set[Tuple[str, str, str, str, str, str]]:
    if not path or not Path(path).exists():
        return set()
    done = set()
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((
                    str(row.get("sequence", "")),
                    str(row.get("frame_id", "")),
                    str(row.get("candidate_name", "")),
                    str(row.get("budget_ratio", row.get("requested_budget_ratio", ""))),
                    str(row.get("candidate_pool_size", "")),
                    str(row.get("edit_sequence", "")),
                ))
    except Exception:
        return set()
    return done


def _eval_decoded_row(
    *,
    file_path: str,
    sequence: str,
    frame_id: str,
    candidate_name: str,
    edit_sequence: str,
    budget: float,
    pool: int,
    coords: torch.Tensor,
    cand_coords: torch.Tensor,
    meta,
    args,
    base_bits: float,
    baseline_stats: Mapping[str, object],
    baseline_quality: Mapping[str, object],
    decoded_gt_path: str,
    baseline_count: int,
    baseline_match: float,
    baseline_lossless: bool,
    decode_encoder,
    debug: Mapping[str, object],
    cli: argparse.Namespace,
    eval_cache: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, object], bool, float, float]:
    sig = _coords_signature(cand_coords)
    t_encode = 0.0
    t_quality = 0.0
    cache_hit = sig in eval_cache
    if cache_hit:
        row_base = dict(eval_cache[sig])
    else:
        cand_xyz = _coords_to_xyz(cand_coords, meta, args)
        t0 = time.time()
        processed_stats = decode_encoder.encode_bits(cand_xyz)
        t_encode = time.time() - t0
        processed_path = str(processed_stats.get("decoded_copy_path", ""))
        processed_count, processed_match, processed_lossless = _coord_match_ratio_from_paths(file_path, processed_path) if processed_path else (0, float("nan"), False)
        with tempfile.TemporaryDirectory(prefix="phase2m_pre_") as tmp:
            pre_path = Path(tmp) / "processed_pre.ply"
            write_ascii_ply_xyz(pre_path, cand_xyz.detach().to("cpu").numpy().astype(np.float64, copy=False))
            tq = time.time()
            pre_quality = _quality_from_paths(
                file_path,
                pre_path,
                formal_max_points=int(cli.quality_max_points),
                normal_max_points=int(cli.normal_max_points),
                pc_error_path=str(cli.pc_error_path),
                use_pc_error=bool(cli.use_pc_error),
            )
            t_quality += time.time() - tq
        tq = time.time()
        processed_quality = _quality_from_paths(
            file_path,
            processed_path,
            formal_max_points=int(cli.quality_max_points),
            normal_max_points=int(cli.normal_max_points),
            pc_error_path=str(cli.pc_error_path),
            use_pc_error=bool(cli.use_pc_error),
        ) if processed_path else {}
        t_quality += time.time() - tq
        processed_stats = dict(processed_stats)
        processed_stats["prune_count"] = int(max(int(coords.shape[0]) - int(cand_coords.shape[0]), 0))
        row_base = _phase2j_row(
            file_path=str(file_path),
            sequence=sequence,
            frame_id=frame_id,
            candidate_name=candidate_name,
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
        eval_cache[sig] = dict(row_base)
    row = dict(row_base)
    row["candidate_name"] = candidate_name
    row["edit_sequence"] = edit_sequence
    row["budget_ratio"] = float(budget)
    row["D1_PSNR"] = row.get("processed_decoded_d1_psnr", "")
    row["Chamfer"] = row.get("processed_decoded_chamfer", "")
    row["point_to_plane_PSNR"] = row.get("processed_decoded_d2_psnr", "")
    row["hole_risk"] = debug.get("hole_risk", "")
    row["boundary_removed_ratio"] = debug.get("boundary_removed_ratio", "")
    row["density_drop"] = debug.get("density_drop_mean", "")
    row["prune_count"] = int(max(int(coords.shape[0]) - int(cand_coords.shape[0]), 0))
    row["add_count"] = int(_safe_float(debug.get("add_count"), 0.0))
    row["move_count"] = int(_safe_float(debug.get("move_count"), 0.0))
    row["repair_count"] = int(_safe_float(debug.get("repair_add_count"), 0.0))
    row["operation_counts_json"] = debug.get("operation_counts_json", json.dumps([candidate_name]))
    row["cache_hit_count"] = 1 if cache_hit else 0
    row["actual_eval_count"] = 0 if cache_hit else 1
    row["skipped_duplicate_count"] = int(_safe_float(debug.get("skipped_duplicate_count"), 0.0))
    row["candidate_generate_time"] = debug.get("candidate_generate_time", "")
    row["encode_decode_time"] = t_encode
    row["quality_eval_time"] = t_quality
    return row, cache_hit, t_encode, t_quality


def _prepare_args(cli: argparse.Namespace):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2M multi operator context rewriter"),
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


def run_phase2m(cli: argparse.Namespace) -> int:
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

    rows: List[Dict[str, object]] = []
    if bool(cli.append_output) and Path(cli.output_csv).exists():
        with open(cli.output_csv, newline="") as f:
            rows = list(csv.DictReader(f))
    done = _read_done_keys(cli.output_csv) if bool(cli.append_output) else set()
    budgets = [float(x) for x in _parse_csv_text(cli.budgets)]
    pools = [int(float(x)) for x in _parse_csv_text(cli.pools)]
    beam_ops = list(_parse_csv_text(cli.beam_ops))
    baselines = list(_parse_csv_text(cli.baselines))

    debug_encoder = build_actual_encoder(debug_args)
    decode_encoder = build_actual_encoder(decode_args)
    eval_cache: Dict[str, Dict[str, object]] = {}
    total_actual_eval = 0
    total_cache_hit = 0
    total_duplicate_skip = 0
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
                    generated: List[Tuple[str, str, torch.Tensor, Dict[str, object]]] = []
                    for baseline in baselines:
                        t0 = time.time()
                        cand_coords, _drop, debug = _candidate_to_coords(
                            candidate=baseline,
                            coords=coords,
                            base_stats=base_stats,
                            budget=float(budget),
                            pool=int(pool),
                            block_size=int(cli.block_size),
                            seed=int(cli.seed) + file_idx,
                        )
                        debug = dict(debug)
                        debug["candidate_generate_time"] = float(time.time() - t0)
                        generated.append((baseline, baseline, cand_coords, debug))
                    t0 = time.time()
                    beam, beam_stats = _build_multi_operator_beam(
                        coords=coords,
                        base_stats=base_stats,
                        budget_ratio=float(budget),
                        pool=int(pool),
                        block_size=int(cli.block_size),
                        seed=int(cli.seed) + file_idx,
                        beam_width=int(cli.beam_width),
                        iterations=int(cli.iterations),
                        ops=beam_ops,
                    )
                    total_duplicate_skip += int(beam_stats.get("duplicate_skip", 0))
                    for rank, state in enumerate(beam[: max(int(cli.beam_width), 1)], start=1):
                        if state.edit_sequence == "start":
                            continue
                        debug = dict(state.debug)
                        debug["candidate_generate_time"] = float(time.time() - t0)
                        debug["skipped_duplicate_count"] = int(beam_stats.get("duplicate_skip", 0))
                        debug["add_count"] = int(state.total_add_count)
                        debug["move_count"] = int(state.total_move_count)
                        debug["merge_count"] = int(state.total_merge_count)
                        debug["repair_add_count"] = int(_safe_float(debug.get("repair_add_count"), 0.0))
                        generated.append((f"multi_operator_rewriter_rank{rank}", state.edit_sequence, state.coords, debug))
                    for candidate_name, edit_sequence, cand_coords, debug in generated:
                        key = (sequence, frame_id, candidate_name, str(float(budget)), str(int(pool)), edit_sequence)
                        if key in done:
                            total_cache_hit += 1
                            continue
                        row, hit, _te, _tq = _eval_decoded_row(
                            file_path=str(file_path),
                            sequence=sequence,
                            frame_id=frame_id,
                            candidate_name=candidate_name,
                            edit_sequence=edit_sequence,
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
                        total_cache_hit += int(hit)
                        total_actual_eval += 0 if hit else 1
                        row["cache_hit_count"] = int(row.get("cache_hit_count", 0)) + total_cache_hit
                        row["actual_eval_count"] = total_actual_eval
                        row["skipped_duplicate_count"] = int(row.get("skipped_duplicate_count", 0)) + total_duplicate_skip
                        rows.append(row)
                        _write_csv(cli.output_csv, rows)
                    print(json.dumps({
                        "phase2m": True,
                        "sequence": sequence,
                        "frame": frame_id,
                        "budget": budget,
                        "pool": pool,
                        "rows": len(rows),
                        "actual_eval_count": total_actual_eval,
                        "cache_hit_count": total_cache_hit,
                        "duplicate_skip": total_duplicate_skip,
                    }, sort_keys=True), flush=True)
    finally:
        for enc in (debug_encoder, decode_encoder):
            close = getattr(enc, "close", None)
            if callable(close):
                close()
    _write_csv(cli.output_csv, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2M multi-operator context rewriter")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--budgets", default="0.030,0.050")
    parser.add_argument("--pools", default="131072")
    parser.add_argument("--baselines", default=",".join(BASELINE_CANDIDATES))
    parser.add_argument("--beam-ops", default=",".join(DEFAULT_BEAM_OPS))
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quality-max-points", type=int, default=1000)
    parser.add_argument("--normal-max-points", type=int, default=1000)
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--pc-error-path", default="/home/maejima/MasterEx/compress/octree/SparsePCGC/extension/pc_error_d")
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2m_decoded")
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2m(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
