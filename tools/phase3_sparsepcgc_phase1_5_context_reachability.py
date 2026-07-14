#!/usr/bin/env python
"""Phase 1.5 context reachability audit for SparsePCGC native dense settings.

Phase 1のbaseline/occupancy dumpを再利用し、再圧縮なしで:
- D1/D2 evaluator差分
- decoder-complete rate accounting
- entropy-equivalent class
- high-bit symbol -> input voxel reachable upper bound
を整理します。
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
csv.field_size_limit(sys.maxsize)

from models.utils.data.dataset import load_ply
from tools.phase2_rdo_beam_probe import _coord_match_ratio_from_paths, _quality_from_paths


PHASE1_DIR = Path("/data/maejima/log/phase1_sparsepcgc_baseline_inventory")
DEFAULT_OUT_DIR = Path("/data/maejima/log/phase1_5_sparsepcgc_context_reachability")
DEFAULT_INPUT = Path("/data/maejima/data/ground/8i/loot/loot_vox10_1000.ply")
RATIOS = (0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01)


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


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _coords_from_ply(path: Path) -> torch.Tensor:
    pts = torch.as_tensor(load_ply(str(path), return_color=False), dtype=torch.float32)
    coords = torch.round(pts).to(torch.long)
    coords = torch.unique(coords, dim=0, sorted=True)
    if coords.numel() and int(coords.min().item()) < 0:
        coords = coords - coords.min(dim=0, keepdim=True).values
    return coords


def _coord_stats(path: Path) -> Dict[str, Any]:
    coords = _coords_from_ply(path)
    if coords.numel() <= 0:
        return {
            "point_count": 0,
            "unique_voxel_count": 0,
            "duplicate_count": 0,
            "coord_min_json": "[]",
            "coord_max_json": "[]",
        }
    raw = torch.round(torch.as_tensor(load_ply(str(path), return_color=False), dtype=torch.float32)).to(torch.long)
    return {
        "point_count": int(raw.shape[0]),
        "unique_voxel_count": int(coords.shape[0]),
        "duplicate_count": int(raw.shape[0] - coords.shape[0]),
        "coord_min_json": json.dumps([int(v) for v in coords.min(dim=0).values.tolist()]),
        "coord_max_json": json.dumps([int(v) for v in coords.max(dim=0).values.tolist()]),
    }


def _pc_error_rows(phase1_rows: Sequence[Mapping[str, Any]], *, out_dir: Path, pc_error_path: str, max_points: int, normal_points: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in phase1_rows:
        ref = Path(str(row.get("input_path") or DEFAULT_INPUT))
        dec = Path(str(row.get("decoded_path") or ""))
        base = {
            "setting_id": row.get("setting_id", ""),
            "scale_AE": row.get("scale_AE", ""),
            "scale_SR": row.get("scale_SR", ""),
            "reference_path": str(ref),
            "decoded_path": str(dec),
            "reference_hash": _file_hash(ref),
            "decoded_hash": _file_hash(dec),
            "status": "missing_decoded" if not dec.exists() else "ok",
        }
        ref_stats = _coord_stats(ref) if ref.exists() else {}
        dec_stats = _coord_stats(dec) if dec.exists() else {}
        for prefix, stats in (("reference", ref_stats), ("decoded", dec_stats)):
            for key, value in stats.items():
                base[f"{prefix}_{key}"] = value
        if dec.exists():
            quality = _quality_from_paths(
                ref,
                dec,
                formal_max_points=max_points,
                normal_max_points=normal_points,
                pc_error_path=pc_error_path,
                use_pc_error=False,
            )
            pc_quality = _pc_error_direct(ref, dec, pc_error_path=pc_error_path, timeout_seconds=25)
            _decoded_count, match_ratio, lossless = _coord_match_ratio_from_paths(ref, dec)
            phase1_d1 = _safe_float(row.get("d1_psnr"))
            phase1_d2 = _safe_float(row.get("d2_psnr"))
            base.update({
                "evaluator": "pc_error_d+myNet_path_eval",
                "resolution": quality.get("pc_error_resolution", ""),
                "metric_direction": "A=reference_GT,B=decoded",
                "symmetric": "pc_error_d_hausdorff_enabled; D1/D2 parsed from mseF",
                "D1": quality.get("d1_psnr", ""),
                "D2": quality.get("d2_psnr", ""),
                "phase1_wrapper_D1": phase1_d1,
                "phase1_wrapper_D2": phase1_d2,
                "delta_D1_vs_phase1": _safe_float(quality.get("d1_psnr")) - phase1_d1,
                "delta_D2_vs_phase1": _safe_float(quality.get("d2_psnr")) - phase1_d2,
                "mynet_D1": quality.get("mynet_d1_psnr", ""),
                "mynet_D2": quality.get("mynet_d2_psnr", ""),
                "pc_error_D1": quality.get("pc_error_d1_psnr", ""),
                "pc_error_D2": quality.get("pc_error_d2_psnr", ""),
                "pc_error_direct_success": pc_quality.get("pc_error_success", ""),
                "pc_error_direct_D1": pc_quality.get("pc_error_d1_psnr", ""),
                "pc_error_direct_D2": pc_quality.get("pc_error_d2_psnr", ""),
                "pc_error_direct_status": pc_quality.get("status", ""),
                "pc_error_direct_command": pc_quality.get("command", ""),
                "coord_match_ratio": match_ratio,
                "decode_lossless": lossless,
                "source_file": "tools.phase2_rdo_beam_probe",
                "source_function": "_quality_from_paths",
                "command": "reused decoded PLY; no recompression",
                "note": "Phase1 D1/D2 were path metrics; this row recomputes with pc_error enabled.",
            })
        rows.append(base)
    return rows


def _pc_error_direct(ref: Path, dec: Path, *, pc_error_path: str, timeout_seconds: int) -> Dict[str, Any]:
    pc = Path(pc_error_path)
    if not pc.exists():
        return {"pc_error_success": False, "status": "pc_error_missing"}
    try:
        coords = _coords_from_ply(ref)
        peak = int(max((coords.max(dim=0).values - coords.min(dim=0).values).max().item(), 1)) if coords.numel() else 1
        cmd = [
            str(pc),
            f"--fileA={ref}",
            f"--fileB={dec}",
            "--hausdorff=1",
            f"--resolution={peak}",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=int(timeout_seconds), check=False)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")

        def grab(label: str) -> float:
            for line in text.splitlines():
                if label in line:
                    vals = []
                    for token in line.replace(":", " ").replace(",", " ").split():
                        try:
                            vals.append(float(token))
                        except ValueError:
                            pass
                    if vals:
                        return float(vals[-1])
            return float("nan")

        return {
            "pc_error_success": proc.returncode == 0,
            "status": "ok" if proc.returncode == 0 else f"returncode_{proc.returncode}",
            "pc_error_d1_psnr": grab("mseF,PSNR (p2point)"),
            "pc_error_d2_psnr": grab("mseF,PSNR (p2plane)"),
            "command": " ".join(cmd),
        }
    except subprocess.TimeoutExpired:
        return {"pc_error_success": False, "status": f"timeout_{timeout_seconds}s", "command": str(pc)}
    except Exception as exc:
        return {"pc_error_success": False, "status": f"{type(exc).__name__}:{exc}", "command": str(pc)}


def _rate_rows(phase1_rows: Sequence[Mapping[str, Any]], debug_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    debug_by_setting = {str(row.get("setting_id")): row for row in debug_rows}
    rows: List[Dict[str, Any]] = []
    for row in phase1_rows:
        setting = str(row.get("setting_id"))
        debug = debug_by_setting.get(setting, {})
        main_bin = _safe_float(row.get("actual_bin_file_bits"))
        logical = _safe_float(row.get("actual_total_bits"))
        side = logical - main_bin if math.isfinite(logical) and math.isfinite(main_bin) else float("nan")
        ae_bytes = _safe_float(debug.get("debug_ae_side_stream_bytes"), float("nan"))
        ae_bits = ae_bytes * 8.0 if math.isfinite(ae_bytes) else float("nan")
        nplist_len = _safe_float(debug.get("debug_num_points_list_len"), float("nan"))
        metadata_bits = nplist_len * 4.0 if math.isfinite(nplist_len) else float("nan")
        rows.append({
            "setting_id": setting,
            "scale_AE": row.get("scale_AE", ""),
            "scale_SR": row.get("scale_SR", ""),
            "main_bin_bits": main_bin,
            "ae_side_stream_bits": ae_bits,
            "num_points_list_bits": metadata_bits,
            "side_stream_bits": side,
            "metadata_bits": metadata_bits,
            "filesystem_total_bits": main_bin,
            "returned_logical_bits": logical,
            "decoder_complete_bits": logical,
            "authoritative_rate_source": "LossyCoderDense.test file_size logical bits",
            "decode_required_files": "main bin + in-memory AE side stream + num_points_list in official coder path",
            "difference_reason": "actual_bin_file_bits excludes AE side stream and num_points_list side information",
            "decode_reproduction_status": "decoded PLY generated in Phase1",
            "main_bin_alone_is_authoritative": False,
        })
    return rows


def _entropy_rows(context_rows: Sequence[Mapping[str, Any]], rate_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rate_by_setting = {str(row.get("setting_id")): row for row in rate_rows}
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    for row in context_rows:
        payload = {
            "bits_by_depth_json": row.get("bits_by_depth_json", ""),
            "candidates_by_depth_json": row.get("candidates_by_depth_json", ""),
            "occupied_by_depth_json": row.get("occupied_by_depth_json", ""),
            "bits_by_parent_popcount_json": row.get("bits_by_parent_popcount_json", ""),
            "bits_by_child_pattern_topk_json": row.get("bits_by_child_pattern_topk_json", ""),
            "estimated_bits": row.get("total_estimated_occupancy_bits", ""),
            "nll": row.get("mean_nll", ""),
            "p_true_mean": row.get("p_true_mean", ""),
        }
        fingerprint = _sha256_text(json.dumps(payload, sort_keys=True))
        groups.setdefault(fingerprint, []).append(row)
    class_index = {fp: f"entropy_class_{idx:02d}" for idx, fp in enumerate(sorted(groups), start=1)}
    for fp, members in groups.items():
        class_id = class_index[fp]
        for member in members:
            setting = str(member.get("setting_id"))
            rate = rate_by_setting.get(setting, {})
            rows.append({
                "setting_id": setting,
                "entropy_class": class_id,
                "fingerprint_kind": "debug_metric_hash",
                "exact_hash": fp,
                "tolerance_comparison": "not_needed_for_json_metric_identity",
                "tensor_shape": member.get("sparsepcgc_candidate_count", ""),
                "node_count": member.get("sparsepcgc_candidate_count", ""),
                "producer_function": "SparsePCGCEncoder._estimate_lossless_occupancy_debug via Phase1 wrapper",
                "members_json": json.dumps([m.get("setting_id") for m in members], sort_keys=True),
                "decoder_complete_bits": rate.get("decoder_complete_bits", ""),
                "note": "Equivalent class is based on emitted debug metrics, not raw probability tensor dump.",
            })
    return rows


def _factor_for_setting(row: Mapping[str, Any], symbol_depth: int) -> int:
    scale_ae = _safe_int(row.get("scale_AE"))
    scale_sr = _safe_int(row.get("scale_SR"))
    return int(2 ** max(scale_sr + scale_ae + int(symbol_depth), 0))


def _build_cell_index(coords: torch.Tensor, max_factor: int) -> Dict[int, Dict[Tuple[int, int, int], List[int]]]:
    out: Dict[int, Dict[Tuple[int, int, int], List[int]]] = {}
    for factor in sorted({2 ** i for i in range(0, int(math.log2(max(max_factor, 1))) + 1)}):
        cells = torch.div(coords, factor, rounding_mode="floor")
        mapping: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, cell in enumerate(cells.tolist()):
            mapping.setdefault(tuple(int(v) for v in cell), []).append(idx)
        out[factor] = mapping
    return out


def _symbol_mapping_rows(
    context_rows: Sequence[Mapping[str, Any]],
    rate_rows: Sequence[Mapping[str, Any]],
    *,
    input_path: Path,
    max_symbols_per_setting: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    coords = _coords_from_ply(input_path)
    input_count = int(coords.shape[0])
    max_factor = 2 ** 8
    cell_index = _build_cell_index(coords, max_factor)
    rate_by_setting = {str(row.get("setting_id")): row for row in rate_rows}
    symbol_rows: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    for row in context_rows:
        setting = str(row.get("setting_id"))
        rate = rate_by_setting.get(setting, {})
        total_bits = _safe_float(rate.get("decoder_complete_bits"))
        top_nodes = _json_loads(row.get("top_high_bit_nodes_json"), [])
        if not isinstance(top_nodes, list):
            top_nodes = []
        candidates: List[Dict[str, Any]] = []
        for sid, node in enumerate(top_nodes[:max_symbols_per_setting]):
            if not isinstance(node, Mapping):
                continue
            depth = _safe_int(node.get("depth"))
            factor = _factor_for_setting(row, depth)
            coord = tuple(int(v) for v in node.get("coord", [0, 0, 0])[:3])
            desc = cell_index.get(factor, {}).get(coord, [])
            desc_count = len(desc)
            occ = bool(node.get("occupied"))
            bit = _safe_float(node.get("bits"))
            mean_neigh = float("nan")
            min_prune = desc_count if occ else ""
            min_add = "" if occ else 1
            min_adjust = desc_count + 1 if occ else 1
            min_merge = desc_count if occ else ""
            editable = bool((occ and 0 < desc_count <= 4) or ((not occ) and factor <= 16))
            geometry_safe = bool((occ and 0 < desc_count <= 2) or ((not occ) and factor <= 8))
            cost_opt = max(1, desc_count) if occ else 1
            cost_real = cost_opt if editable else 10**9
            cost_lower = cost_opt if geometry_safe else 10**9
            mapped_key = f"f{factor}:{coord[0]},{coord[1]},{coord[2]}"
            out = {
                "setting_id": setting,
                "entropy_class": row.get("entropy_class", ""),
                "symbol_id": sid,
                "node_coordinate": json.dumps(list(coord)),
                "depth": depth,
                "parent_mask": "",
                "parent_popcount": "",
                "p_true": node.get("prob_true", ""),
                "bit_each": bit,
                "occupied": occ,
                "descendant_input_voxel_count": desc_count,
                "descendant_decoded_voxel_count": "",
                "ancestor_symbol_ids": "",
                "affected_neighbor_context_ids": mapped_key,
                "minimum_prune_edits": min_prune,
                "minimum_add_edits": min_add,
                "minimum_adjust_edits": min_adjust,
                "minimum_merge_edits": min_merge,
                "geometry_risk_proxy": float(desc_count > 4),
                "local_neighbor_mean": mean_neigh,
                "editable": editable,
                "geometry_safe": geometry_safe,
                "exact_or_approximation": "realistic_proxy_from_downscale_cell_mapping",
                "mapping_factor": factor,
                "mapped_cell_key": mapped_key,
                "optimistic_cost_edits": cost_opt,
                "realistic_cost_edits": cost_real,
                "lower_bound_cost_edits": cost_lower,
                "decoder_complete_bits": total_bits,
            }
            symbol_rows.append(out)
            candidates.append(out)

        # overlap correction: keep one symbol per mapped cell for budget cover.
        unique: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            key = str(cand["mapped_cell_key"])
            if key not in unique or _safe_float(cand["bit_each"]) > _safe_float(unique[key]["bit_each"]):
                unique[key] = cand
        for mode, cost_key in (
            ("optimistic_upper_bound", "optimistic_cost_edits"),
            ("realistic_proxy", "realistic_cost_edits"),
            ("geometry_safe_lower_bound", "lower_bound_cost_edits"),
        ):
            pool = [c for c in unique.values() if _safe_float(c.get(cost_key), 1e30) < 1e20]
            pool.sort(key=lambda c: _safe_float(c["bit_each"]) / max(_safe_float(c[cost_key]), 1.0), reverse=True)
            for ratio in (0.0, *RATIOS):
                budget = 1 if ratio == 0.0 else max(1, int(math.floor(input_count * ratio)))
                used = 0
                gain = 0.0
                picked = 0
                for cand in pool:
                    cost = int(max(1, _safe_float(cand[cost_key], 1)))
                    if used + cost > budget:
                        continue
                    used += cost
                    gain += _safe_float(cand["bit_each"], 0.0)
                    picked += 1
                pct = gain / total_bits * 100.0 if math.isfinite(total_bits) and total_bits > 0 else float("nan")
                budget_rows.append({
                    "setting_id": setting,
                    "mode": mode,
                    "input_voxel_count": input_count,
                    "budget_label": "atomic_1_voxel" if ratio == 0.0 else f"{ratio*100:.4f}%",
                    "budget_ratio": ratio,
                    "budget_voxels": budget,
                    "used_edits": used,
                    "picked_symbols": picked,
                    "maximum_coverable_estimated_occupancy_bits": gain,
                    "decoder_complete_bits": total_bits,
                    "decoder_complete_percent": pct,
                    "gain_per_edit": gain / max(used, 1),
                    "overlap_correction": "one best symbol per mapped downscale cell",
                    "required_edits": used,
                })
    return symbol_rows, budget_rows


def _psnr_band(value: float, *, step: int) -> str:
    if not math.isfinite(value):
        return "unavailable"
    targets = list(range(40, 91, step))
    half = step / 2.0
    for target in targets:
        if target - half <= value < target + half:
            return f"{target}dB"
    return "out_of_40_90"


def _psnr_band_rows(phase1_rows: Sequence[Mapping[str, Any]], rate_rows: Sequence[Mapping[str, Any]], *, step: int) -> List[Dict[str, Any]]:
    rate_by_setting = {str(row.get("setting_id")): row for row in rate_rows}
    rows: List[Dict[str, Any]] = []
    for row in phase1_rows:
        setting = str(row.get("setting_id"))
        rate = rate_by_setting.get(setting, {})
        d1 = _safe_float(row.get("d1_psnr"))
        d2 = _safe_float(row.get("d2_psnr"))
        for metric, value in (("D1", d1), ("D2", d2)):
            rows.append({
                "setting_id": setting,
                "metric": metric,
                "psnr_band_step_db": step,
                "psnr_band": _psnr_band(value, step=step),
                "baseline_psnr": value,
                "baseline_decoder_complete_bits": rate.get("decoder_complete_bits", ""),
                "operation": "baseline_no_edit",
                "actual_edit_ratio": 0.0,
                "note": "Phase1.5 baseline coverage only; edited clouds are Phase2A.",
            })
    return rows


def _gate_result(budget_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    realistic_1 = [
        _safe_float(r.get("decoder_complete_percent"))
        for r in budget_rows
        if r.get("mode") == "realistic_proxy" and abs(_safe_float(r.get("budget_ratio")) - 0.01) < 1e-9
    ]
    realistic_05 = [
        _safe_float(r.get("decoder_complete_percent"))
        for r in budget_rows
        if r.get("mode") == "realistic_proxy" and abs(_safe_float(r.get("budget_ratio")) - 0.005) < 1e-9
    ]
    optimistic_1 = [
        _safe_float(r.get("decoder_complete_percent"))
        for r in budget_rows
        if r.get("mode") == "optimistic_upper_bound" and abs(_safe_float(r.get("budget_ratio")) - 0.01) < 1e-9
    ]
    max_realistic_1 = max([v for v in realistic_1 if math.isfinite(v)] or [float("nan")])
    max_realistic_05 = max([v for v in realistic_05 if math.isfinite(v)] or [float("nan")])
    max_optimistic_1 = max([v for v in optimistic_1 if math.isfinite(v)] or [float("nan")])
    if math.isfinite(max_realistic_1) and max_realistic_1 >= 5.0:
        status = "PASS"
    elif math.isfinite(max_realistic_05) and max_realistic_05 >= 2.0:
        status = "PASS"
    elif math.isfinite(max_optimistic_1) and max_optimistic_1 >= 5.0:
        status = "CONDITIONAL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "pass": status == "PASS",
        "conditional": status == "CONDITIONAL",
        "max_realistic_proxy_1pct_decoder_complete_percent": max_realistic_1,
        "max_realistic_proxy_0p5pct_decoder_complete_percent": max_realistic_05,
        "max_optimistic_upper_1pct_decoder_complete_percent": max_optimistic_1,
        "phase2a_allowed": status == "PASS",
        "reason": "PASS requires realistic proxy >=5% at 1% edits or >=2% at 0.5% edits.",
    }


def _report(out_dir: Path, gate: Mapping[str, Any], rate_rows: Sequence[Mapping[str, Any]], entropy_rows: Sequence[Mapping[str, Any]], budget_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Phase 1.5 SparsePCGC context reachability",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- no recompression; Phase 1 decoded PLY/debug dumps reused",
        "",
        "## Rate accounting",
    ]
    for row in rate_rows:
        lines.append(
            f"- {row['setting_id']}: decoder_complete={row['decoder_complete_bits']} "
            f"main_bin={row['main_bin_bits']} side={row['side_stream_bits']}"
        )
    lines.extend(["", "## Entropy classes"])
    classes: Dict[str, List[str]] = {}
    for row in entropy_rows:
        classes.setdefault(str(row["entropy_class"]), []).append(str(row["setting_id"]))
    for cls, members in sorted(classes.items()):
        lines.append(f"- {cls}: {', '.join(members)}")
    lines.extend(["", "## Reachability highlights"])
    for row in budget_rows:
        if row.get("mode") == "realistic_proxy" and row.get("budget_label") in {"0.5000%", "1.0000%"}:
            lines.append(
                f"- {row['setting_id']} {row['budget_label']}: "
                f"{float(row['decoder_complete_percent']):.3f}% decoder-complete upper proxy "
                f"used={row['used_edits']}"
            )
    lines.extend([
        "",
        "## Gate",
        "```json",
        json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "- top symbol share is not treated as input voxel edit share.",
        "- symbol-to-voxel mapping uses downscale-cell descendants and is labelled as proxy, not exact influence.",
        "- decoder-complete bits use logical bits returned by LossyCoderDense.test, not main .bin alone.",
    ])
    (out_dir / "phase1_5_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1.5 context reachability")
    parser.add_argument("--phase1-dir", type=Path, default=PHASE1_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pc-error-path", default="/home/maejima/MasterEx/compress/octree/SparsePCGC/extension/pc_error_d")
    parser.add_argument("--max-symbols-per-setting", type=int, default=32768)
    parser.add_argument("--formal-max-points", type=int, default=3000)
    parser.add_argument("--normal-max-points", type=int, default=3000)
    cli = parser.parse_args()

    required = [
        "phase1b_official_pair_baselines.csv",
        "phase1c_context_inventory.csv",
        "phase1_gate_result.json",
    ]
    missing = [name for name in required if not (cli.phase1_dir / name).exists()]
    if missing:
        print(f"Missing Phase1 outputs: {missing}", file=sys.stderr)
        return 2
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    if cli.dry_run:
        print(json.dumps({
            "phase1_dir": str(cli.phase1_dir),
            "out_dir": str(cli.out_dir),
            "input": str(cli.input),
            "required": required,
        }, indent=2))
        return 0

    phase1_rows = _read_csv(cli.phase1_dir / "phase1b_official_pair_baselines.csv")
    context_rows = _read_csv(cli.phase1_dir / "phase1c_context_inventory.csv")
    quality_rows = _pc_error_rows(
        phase1_rows,
        out_dir=cli.out_dir,
        pc_error_path=cli.pc_error_path,
        max_points=cli.formal_max_points,
        normal_points=cli.normal_max_points,
    )
    rate_rows = _rate_rows(phase1_rows, context_rows)
    entropy_rows = _entropy_rows(context_rows, rate_rows)
    entropy_by_setting = {row["setting_id"]: row["entropy_class"] for row in entropy_rows}
    for row in context_rows:
        row["entropy_class"] = entropy_by_setting.get(row.get("setting_id", ""), "")
    symbol_rows, budget_rows = _symbol_mapping_rows(
        context_rows,
        rate_rows,
        input_path=cli.input,
        max_symbols_per_setting=cli.max_symbols_per_setting,
    )
    psnr5 = _psnr_band_rows(phase1_rows, rate_rows, step=5)
    psnr10 = _psnr_band_rows(phase1_rows, rate_rows, step=10)
    gate = _gate_result(budget_rows)

    _write_csv(cli.out_dir / "phase1_5_quality_evaluator_audit.csv", quality_rows)
    _write_csv(cli.out_dir / "phase1_5_rate_accounting.csv", rate_rows)
    _write_csv(cli.out_dir / "phase1_5_entropy_equivalent_classes.csv", entropy_rows)
    _write_csv(cli.out_dir / "phase1_5_symbol_to_voxel_mapping.csv", symbol_rows)
    _write_csv(cli.out_dir / "phase1_5_reachable_upper_bound.csv", budget_rows)
    _write_csv(cli.out_dir / "phase1_5_psnr_band_5db_baseline.csv", psnr5)
    _write_csv(cli.out_dir / "phase1_5_psnr_band_10db_baseline.csv", psnr10)
    _write_json(cli.out_dir / "phase1_5_gate_result.json", gate)
    _report(cli.out_dir, gate, rate_rows, entropy_rows, budget_rows)
    print(f"[Phase1.5] outputs: {cli.out_dir}")
    print(f"[Phase1.5] Gate status={gate['status']} pass={gate['pass']}")
    return 0 if gate["status"] in {"PASS", "CONDITIONAL", "FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
