#!/usr/bin/env python
"""Phase 2A high-cost context atomic/micro oracle.

Phase 1.5のsymbol-to-voxel mappingを使い、最大2 entropy classに限定して
Prune/Merge/Adjust/Addの少数候補だけactual SparsePCGC decodeまで評価します。
SparsePCGC本体・myNet本体は変更しません。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
csv.field_size_limit(sys.maxsize)

from models.utils.data.dataset import load_ply
from tools.phase2_rdo_beam_probe import _quality_from_paths


PHASE1_DIR = Path("/data/maejima/log/phase1_sparsepcgc_baseline_inventory")
PHASE15_DIR = Path("/data/maejima/log/phase1_5_sparsepcgc_context_reachability")
DEFAULT_OUT_DIR = Path("/data/maejima/log/phase2a_sparsepcgc_context_micro_oracle")
DEFAULT_INPUT = Path("/data/maejima/data/ground/8i/loot/loot_vox10_1000.ply")
WORKER_SCRIPT = REPO_ROOT / "tools" / "phase3_sparsepcgc_phase1_worker.py"


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    tmp.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _coords_from_ply(path: Path) -> torch.Tensor:
    pts = torch.as_tensor(load_ply(str(path), return_color=False), dtype=torch.float32)
    coords = torch.unique(torch.round(pts).to(torch.long), dim=0, sorted=True)
    if coords.numel() and int(coords.min().item()) < 0:
        coords = coords - coords.min(dim=0, keepdim=True).values
    return coords


def _write_ascii_ply(path: Path, coords: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = coords.detach().cpu().to(torch.long).numpy()
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {arr.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in arr:
            f.write(f"{int(x)} {int(y)} {int(z)}\n")


def _sparsepcgc_python() -> List[str]:
    candidates = [
        Path.home() / "miniconda3/envs/sparsepcgc/bin/python",
        Path.home() / "anaconda3/envs/sparsepcgc/bin/python",
    ]
    for c in candidates:
        if c.exists():
            return [str(c)]
    return ["conda", "run", "-n", "sparsepcgc", "python"]


def _run_worker_decode(
    *,
    input_file: Path,
    output_dir: Path,
    setting: Mapping[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    request = {
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "sparsepcgc_root": "/home/maejima/MasterEx/compress/octree/SparsePCGC",
        "ckptdir": "/home/maejima/MasterEx/compress/octree/SparsePCGC/ckpts/dense/epoch_last.pth",
        "ckptdir_sr": "/home/maejima/MasterEx/compress/octree/SparsePCGC/ckpts/dense_1stage/epoch_last.pth",
        "ckptdir_ae": "/home/maejima/MasterEx/compress/octree/SparsePCGC/ckpts/dense_slne/epoch_last.pth",
        "ckptdir_low": "/home/maejima/MasterEx/compress/octree/SparsePCGC/ckpts/sparse_low/epoch_last.pth",
        "ckptdir_high": "/home/maejima/MasterEx/compress/octree/SparsePCGC/ckpts/sparse_high/epoch_last.pth",
        "ckptdir_offset": "/home/maejima/MasterEx/compress/octree/SparsePCGC/ckpts/sparse_offset/epoch_last.pth",
        "scale_AE": int(setting["scale_AE"]),
        "scale_SR": int(setting["scale_SR"]),
        "voxel_size": 1.0,
        "pos_quantscale": 1,
        "psnr_resolution": 1023,
        "decode": True,
        "topk_final": 1024,
        "topk_per_layer": 1024,
    }
    proc = subprocess.run(
        _sparsepcgc_python() + [str(WORKER_SCRIPT), "--mode", "decode"],
        input=json.dumps(request, sort_keys=True),
        cwd=str(MASTER_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(timeout),
        check=False,
    )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(f"worker non-json rc={proc.returncode}: {exc}; stderr={proc.stderr[-1000:]}")
    if proc.returncode != 0 or payload.get("status") != "ok":
        raise RuntimeError(f"worker failed rc={proc.returncode}: {payload}; stderr={proc.stderr[-1000:]}")
    result = payload["result"]
    result["worker_stderr_tail"] = proc.stderr[-1000:]
    return result


def _cell_indices(coords: torch.Tensor, factor: int, cell: Tuple[int, int, int]) -> torch.Tensor:
    cells = torch.div(coords, int(factor), rounding_mode="floor")
    target = torch.tensor(cell, dtype=torch.long, device=coords.device)
    return torch.nonzero((cells == target).all(dim=1), as_tuple=False).reshape(-1)


def _parse_cell_key(key: str) -> Tuple[int, Tuple[int, int, int]]:
    # f8:37,57,19
    left, right = key.split(":", 1)
    factor = int(left[1:])
    cell = tuple(int(v) for v in right.split(",")[:3])
    return factor, cell  # type: ignore[return-value]


def _make_candidate_coords(
    coords: torch.Tensor,
    row: Mapping[str, Any],
    operation: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    factor, cell = _parse_cell_key(str(row["mapped_cell_key"]))
    indices = _cell_indices(coords, factor, cell)
    occupied = str(row.get("occupied")) == "True"
    debug: Dict[str, Any] = {
        "source_indices": indices.tolist(),
        "mapping_factor": factor,
        "mapped_cell": cell,
        "operation": operation,
    }
    existing = {tuple(int(v) for v in c.tolist()) for c in coords}
    keep = torch.ones((coords.shape[0],), dtype=torch.bool)
    add_points: List[Tuple[int, int, int]] = []
    move_pairs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []

    if operation in {"prune", "merge"}:
        if not occupied or indices.numel() <= 0 or indices.numel() > 4:
            raise ValueError("not_prunable_micro_context")
        keep[indices] = False
        debug["prune_count"] = int(indices.numel())
        debug["merge_count"] = int(indices.numel()) if operation == "merge" else 0
    elif operation == "adjust":
        if not occupied or indices.numel() != 1:
            raise ValueError("adjust_requires_single_source")
        src = tuple(int(v) for v in coords[int(indices[0])].tolist())
        parent = tuple(v // 2 for v in src)
        target = None
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            cand = (src[0] + dx, src[1] + dy, src[2] + dz)
            if cand not in existing and tuple(v // 2 for v in cand) == parent:
                target = cand
                break
        if target is None:
            raise ValueError("no_same_parent_empty_target")
        keep[indices] = False
        add_points.append(target)
        move_pairs.append((src, target))
        debug["move_count"] = 1
    elif operation == "add":
        if occupied:
            raise ValueError("add_requires_empty_symbol")
        base = tuple(int(v) for v in (torch.tensor(cell) * factor).tolist())
        parent = tuple(v // 2 for v in base)
        parent_exists = any(tuple(v // 2 for v in p) == parent for p in existing)
        if base in existing or not parent_exists:
            raise ValueError("add_target_existing_or_new_parent")
        add_points.append(base)
        debug["add_count"] = 1
    else:
        raise ValueError(f"unknown_operation:{operation}")

    edited = coords[keep].clone()
    if add_points:
        edited = torch.cat([edited, torch.tensor(add_points, dtype=torch.long)], dim=0)
    edited = torch.unique(edited, dim=0, sorted=True)
    debug.update({
        "source_voxel_count": int(indices.numel()) if occupied else 0,
        "add_count": int(len(add_points)),
        "move_pairs_json": json.dumps(move_pairs),
        "symmetric_difference": int(len(existing.symmetric_difference({tuple(int(v) for v in c.tolist()) for c in edited}))),
    })
    return edited, debug


def _select_settings(entropy_rows: Sequence[Mapping[str, Any]], budget_rows: Sequence[Mapping[str, Any]], max_classes: int) -> List[str]:
    by_setting = {row["setting_id"]: row for row in entropy_rows}
    scores: Dict[str, float] = {}
    for row in budget_rows:
        if row.get("mode") == "realistic_proxy" and row.get("budget_label") == "1.0000%":
            scores[row["setting_id"]] = _safe_float(row.get("decoder_complete_percent"), 0.0)
    chosen: List[str] = []
    used_classes = set()
    for setting, _score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        cls = by_setting.get(setting, {}).get("entropy_class", setting)
        if cls in used_classes:
            continue
        chosen.append(setting)
        used_classes.add(cls)
        if len(chosen) >= max_classes:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2A context micro oracle")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--phase1-dir", type=Path, default=PHASE1_DIR)
    parser.add_argument("--phase15-dir", type=Path, default=PHASE15_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-entropy-classes", type=int, default=2)
    parser.add_argument("--candidates-per-operation", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--quality-max-points", type=int, default=3000)
    args = parser.parse_args()

    gate = json.loads((args.phase15_dir / "phase1_5_gate_result.json").read_text(encoding="utf-8"))
    if not bool(gate.get("pass")):
        print("[Phase2A] Phase1.5 Gate is not PASS; stop.", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entropy_rows = _read_csv(args.phase15_dir / "phase1_5_entropy_equivalent_classes.csv")
    budget_rows = _read_csv(args.phase15_dir / "phase1_5_reachable_upper_bound.csv")
    symbol_rows = _read_csv(args.phase15_dir / "phase1_5_symbol_to_voxel_mapping.csv")
    rate_rows = {r["setting_id"]: r for r in _read_csv(args.phase15_dir / "phase1_5_rate_accounting.csv")}
    phase1_rows = {r["setting_id"]: r for r in _read_csv(args.phase1_dir / "phase1b_official_pair_baselines.csv")}
    selected_settings = _select_settings(entropy_rows, budget_rows, args.max_entropy_classes)
    if args.dry_run:
        print(json.dumps({"selected_settings": selected_settings, "out_dir": str(args.out_dir)}, indent=2))
        return 0

    coords = _coords_from_ply(args.input)
    atomic_rows: List[Dict[str, Any]] = []
    actual_rows: List[Dict[str, Any]] = []
    cache: Dict[str, Dict[str, Any]] = {}
    cache_path = args.out_dir / "phase2a_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    for setting in selected_settings:
        setting_info = phase1_rows[setting]
        baseline_bits = _safe_float(rate_rows[setting]["decoder_complete_bits"])
        rows = [r for r in symbol_rows if r["setting_id"] == setting]
        op_candidates: List[Tuple[str, Mapping[str, Any]]] = []
        for op in ("prune", "merge", "adjust", "add"):
            if op == "add":
                pool = [r for r in rows if r["occupied"] == "False" and r["editable"] == "True"]
            elif op == "adjust":
                pool = [r for r in rows if r["occupied"] == "True" and _safe_int(r["descendant_input_voxel_count"]) == 1]
            else:
                pool = [r for r in rows if r["occupied"] == "True" and 0 < _safe_int(r["descendant_input_voxel_count"]) <= 4]
            pool.sort(key=lambda r: _safe_float(r["bit_each"]) / max(_safe_float(r["optimistic_cost_edits"], 1), 1), reverse=True)
            for r in pool[: args.candidates_per_operation]:
                op_candidates.append((op, r))

        for cand_idx, (op, sym) in enumerate(op_candidates):
            try:
                edited, dbg = _make_candidate_coords(coords, sym, op)
            except Exception as exc:
                atomic_rows.append({
                    "setting_id": setting,
                    "operation": op,
                    "symbol_id": sym.get("symbol_id"),
                    "candidate_status": "filtered",
                    "filter_reason": f"{type(exc).__name__}:{exc}",
                })
                continue
            edit_hash = _sha256_text(setting + op + json.dumps(dbg, sort_keys=True))
            ply_path = args.out_dir / "edited_ply" / f"{setting}_{op}_{cand_idx}_{edit_hash[:8]}.ply"
            _write_ascii_ply(ply_path, edited)
            out_codec = args.out_dir / "codec_outputs" / setting / f"{op}_{cand_idx}_{edit_hash[:8]}"
            cache_key = _sha256_text(str(ply_path) + setting + op + edit_hash)
            cache_hit = cache_key in cache
            if cache_hit:
                stats = cache[cache_key]
            else:
                stats = _run_worker_decode(input_file=ply_path, output_dir=out_codec, setting=setting_info, timeout=args.timeout)
                cache[cache_key] = stats
                cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
            bits = _safe_float(stats.get("file_size"))
            raw_percent = (bits - baseline_bits) / baseline_bits * 100.0 if baseline_bits > 0 else float("nan")
            decoded_path = str(stats.get("decoded_path", ""))
            quality: Dict[str, Any] = {}
            if decoded_path and Path(decoded_path).exists():
                quality = _quality_from_paths(
                    args.input,
                    decoded_path,
                    formal_max_points=args.quality_max_points,
                    normal_max_points=args.quality_max_points,
                    use_pc_error=False,
                )
            actual = {
                "setting_id": setting,
                "entropy_class": next((r["entropy_class"] for r in entropy_rows if r["setting_id"] == setting), ""),
                "operation": op,
                "symbol_id": sym.get("symbol_id"),
                "source_coord_or_cell": sym.get("mapped_cell_key"),
                "bit_each_seed": sym.get("bit_each"),
                "baseline_decoder_complete_bits": baseline_bits,
                "edited_decoder_complete_bits": bits,
                "actual_raw_percent": raw_percent,
                "actual_edit_ratio": _safe_float(dbg.get("symmetric_difference"), 0.0) / max(float(coords.shape[0]), 1.0),
                "symmetric_difference": dbg.get("symmetric_difference", ""),
                "source_voxel_count": dbg.get("source_voxel_count", ""),
                "pruned_count": dbg.get("prune_count", 0),
                "added_count": dbg.get("add_count", 0),
                "moved_count": dbg.get("move_count", 0),
                "merged_count": dbg.get("merge_count", 0),
                "D1_PSNR": quality.get("d1_psnr", ""),
                "D2_PSNR": quality.get("d2_psnr", ""),
                "Chamfer": quality.get("chamfer", ""),
                "decoded_path": decoded_path,
                "edited_ply_path": str(ply_path),
                "cache_hit": cache_hit,
                "debug_json": json.dumps(dbg, sort_keys=True),
                "status": "ok",
            }
            atomic_rows.append(actual)
            actual_rows.append(actual)
            print(f"[Phase2A] {setting} {op} raw={raw_percent:+.4f}% edit={actual['actual_edit_ratio']:.6f} cache={cache_hit}", flush=True)

    # Micro budget: greedy prune-only from best actual-improving prune/merge cells is diagnostic.
    micro_rows: List[Dict[str, Any]] = []
    for setting in selected_settings:
        best = [r for r in actual_rows if r["setting_id"] == setting and r["operation"] in {"prune", "merge"} and _safe_float(r["actual_raw_percent"]) < 0]
        best.sort(key=lambda r: _safe_float(r["actual_raw_percent"]))
        for ratio in (0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01):
            micro_rows.append({
                "setting_id": setting,
                "budget_ratio": ratio,
                "diagnostic": "micro-budget actual combination not run; atomic candidates only",
                "best_atomic_raw_percent": best[0]["actual_raw_percent"] if best else "",
                "best_atomic_operation": best[0]["operation"] if best else "",
                "phase2b_required": bool(best),
            })

    psnr5_rows = []
    psnr10_rows = []
    for row in actual_rows:
        for metric, value in (("D1", _safe_float(row.get("D1_PSNR"))), ("D2", _safe_float(row.get("D2_PSNR")))):
            for step, target_rows in ((5, psnr5_rows), (10, psnr10_rows)):
                band = "unavailable"
                if math.isfinite(value):
                    for target in range(40, 91, step):
                        if target - step / 2 <= value < target + step / 2:
                            band = f"{target}dB"
                            break
                target_rows.append({
                    "setting_id": row["setting_id"],
                    "operation": row["operation"],
                    "metric": metric,
                    "psnr": value,
                    "psnr_band": band,
                    "actual_raw_percent": row["actual_raw_percent"],
                    "actual_edit_ratio": row["actual_edit_ratio"],
                })

    pareto = sorted(actual_rows, key=lambda r: (_safe_float(r["actual_raw_percent"], 1e9), -_safe_float(r.get("D1_PSNR"), -1e9)))
    gate2 = {
        "phase": "Phase2A",
        "status": "PASS" if any(_safe_float(r["actual_raw_percent"]) < 0 for r in actual_rows) else "FAIL",
        "actual_eval_count": len(actual_rows),
        "best_raw_percent": min([_safe_float(r["actual_raw_percent"]) for r in actual_rows] or [float("nan")]),
        "phase2b_executed": False,
        "note": "Atomic-only limited oracle; micro-budget combinations are deferred.",
    }
    _write_csv(args.out_dir / "phase2a_atomic_candidates.csv", atomic_rows)
    _write_csv(args.out_dir / "phase2a_atomic_actual_results.csv", actual_rows)
    _write_csv(args.out_dir / "phase2a_micro_budget_results.csv", micro_rows)
    _write_csv(args.out_dir / "phase2a_context_delta.csv", actual_rows)
    _write_csv(args.out_dir / "phase2a_psnr_band_5db.csv", psnr5_rows)
    _write_csv(args.out_dir / "phase2a_psnr_band_10db.csv", psnr10_rows)
    _write_csv(args.out_dir / "phase2a_pareto_front.csv", pareto[:20])
    _write_json(args.out_dir / "phase2a_gate_result.json", gate2)
    report = [
        "# Phase2A context micro oracle",
        "",
        f"- selected_settings: {', '.join(selected_settings)}",
        f"- actual_eval_count: {len(actual_rows)}",
        f"- best_raw_percent: {gate2['best_raw_percent']}",
        "- Phase2B was not executed.",
    ]
    (args.out_dir / "phase2a_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[Phase2A] outputs: {args.out_dir}")
    print(f"[Phase2A] Gate status={gate2['status']} best={gate2['best_raw_percent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
