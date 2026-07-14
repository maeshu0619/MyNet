#!/usr/bin/env python
"""Phase 1 SparsePCGC native dense baseline inventory.

研究用の薄いwrapperです。SparsePCGC本体、myNet本体、学習コードは変更せず、
公式dense AE/SR pairのbaseline RD点とoccupancy/debug取得可否を段階的に確認します。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
csv.field_size_limit(sys.maxsize)

from models.utils.config.args import parse_pugan_args
from models.utils.data.dataset import load_ply
from tools.context_aware_where_probe import _parent_info, _unique_coords
from tools.phase2_rdo_beam_probe import _coord_match_ratio_from_paths, _quality_from_paths


OFFICIAL_PAIRS: Tuple[Tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (1, 1),
    (0, 2),
    (1, 2),
    (0, 3),
)
PHASE0_DIR = Path("/data/maejima/log/phase0_sparsepcgc_official_audit")
DEFAULT_OUT_DIR = Path("/data/maejima/log/phase1_sparsepcgc_baseline_inventory")
DEFAULT_INPUT = Path("/data/maejima/data/ground/8i/loot/loot_vox10_1000.ply")
WORKER_SCRIPT = REPO_ROOT / "tools" / "phase3_sparsepcgc_phase1_worker.py"
DEFAULT_DATASET = "8i"
DEFAULT_SEQUENCE = "loot"
DEFAULT_FRAME = "1000"
SCHEMA_VERSION = "phase1_baseline_inventory_v2"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _sha256_file(path: Path, *, max_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while True:
            if remaining is None:
                chunk = f.read(1024 * 1024)
            elif remaining <= 0:
                break
            else:
                chunk = f.read(min(1024 * 1024, remaining))
                remaining -= len(chunk)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return _sha256_file(path)


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(MASTER_ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def _setting_id(scale_ae: int, scale_sr: int) -> str:
    return f"native_vs1_pq1_ae{int(scale_ae)}_sr{int(scale_sr)}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _base_args():
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase1 sparsepcgc baseline inventory base args"),
            time.strftime("%Y%m%d"),
            time.strftime("%H%M%S"),
        )
    finally:
        sys.argv = old_argv
    args.compress = "SparsePCGC"
    args.compression_loss_backend = "sparsepcgc_surrogate"
    return args


def _phase_args(
    base_args: Any,
    *,
    scale_ae: int,
    scale_sr: int,
    decode: bool,
    occupancy_debug: bool,
    decoded_dir: Path,
    tmp_dir: Path,
    topk: int,
    timeout: float,
) -> Any:
    args = copy.copy(base_args)
    args.sparsepcgc_mode = "dense_lossy"
    args.sparsepcgc_voxel_size = 1.0
    args.sparsepcgc_pos_quantscale = 1
    args.sparsepcgc_psnr_resolution = 1023
    args.sparsepcgc_dense_scale_ae_list = str(int(scale_ae))
    args.sparsepcgc_dense_scale_sr_list = str(int(scale_sr))
    args.sparsepcgc_skip_decode = not bool(decode)
    args.sparsepcgc_decoded_copy_dir = str(decoded_dir)
    args.sparsepcgc_tmp_dir = str(tmp_dir)
    args.sparsepcgc_timeout = float(timeout)
    args.enable_sparsepcgc_occupancy_debug = bool(occupancy_debug)
    args.sparsepcgc_occupancy_debug_topk_final = int(topk)
    args.sparsepcgc_occupancy_debug_topk_per_layer = max(512, min(int(topk), 8192))
    args.sparsepcgc_worker_gpu_stats = True
    return args


def _encoder_args_summary(args: Any) -> Dict[str, Any]:
    root = Path(getattr(args, "sparsepcgc_root", MASTER_ROOT / "compress/octree/SparsePCGC"))
    if not root.is_absolute():
        root = (MASTER_ROOT / root).resolve()
    ckpt = Path(getattr(args, "sparsepcgc_ckptdir", root / "ckpts/dense/epoch_last.pth"))
    ckpt_sr = Path(getattr(args, "sparsepcgc_ckptdir_sr", root / "ckpts/dense_1stage/epoch_last.pth"))
    ckpt_ae = Path(getattr(args, "sparsepcgc_ckptdir_ae", root / "ckpts/dense_slne/epoch_last.pth"))
    ckpt_low = Path(getattr(args, "sparsepcgc_ckptdir_low", root / "ckpts/sparse_low/epoch_last.pth"))
    ckpt_high = Path(getattr(args, "sparsepcgc_ckptdir_high", root / "ckpts/sparse_high/epoch_last.pth"))
    ckpt_offset = Path(getattr(args, "sparsepcgc_ckptdir_offset", root / "ckpts/sparse_offset/epoch_last.pth"))
    return {
        "sparsepcgc_root": str(root),
        "checkpoint_path": str(ckpt),
        "checkpoint_hash": _hash_path(ckpt),
        "checkpoint_sr_path": str(ckpt_sr),
        "checkpoint_sr_hash": _hash_path(ckpt_sr),
        "checkpoint_ae_path": str(ckpt_ae),
        "checkpoint_ae_hash": _hash_path(ckpt_ae),
        "checkpoint_low_path": str(ckpt_low),
        "checkpoint_low_hash": _hash_path(ckpt_low),
        "checkpoint_high_path": str(ckpt_high),
        "checkpoint_high_hash": _hash_path(ckpt_high),
        "checkpoint_offset_path": str(ckpt_offset),
        "checkpoint_offset_hash": _hash_path(ckpt_offset),
    }


def _cache_key(
    *,
    phase: str,
    file_path: Path,
    setting_id: str,
    scale_ae: int,
    scale_sr: int,
    debug_mode: str,
    args_summary: Mapping[str, Any],
) -> str:
    stat = file_path.stat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "dataset": DEFAULT_DATASET,
        "sequence": DEFAULT_SEQUENCE,
        "frame": DEFAULT_FRAME,
        "input_absolute_path": str(file_path.resolve()),
        "input_size": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "input_content_hash": _sha256_file(file_path),
        "pc_type": "dense",
        "mode": "dense_lossy",
        "scale_AE": int(scale_ae),
        "scale_SR": int(scale_sr),
        "voxel_size": 1.0,
        "posQuantscale": 1,
        "debug_mode": debug_mode,
        "codec_path": args_summary.get("sparsepcgc_root", ""),
        "checkpoint_path": args_summary.get("checkpoint_path", ""),
        "checkpoint_hash": args_summary.get("checkpoint_hash", ""),
        "checkpoint_sr_path": args_summary.get("checkpoint_sr_path", ""),
        "checkpoint_sr_hash": args_summary.get("checkpoint_sr_hash", ""),
        "checkpoint_ae_path": args_summary.get("checkpoint_ae_path", ""),
        "checkpoint_ae_hash": args_summary.get("checkpoint_ae_hash", ""),
        "code_version": _git_commit(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_cache(path: Path, cache: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _input_points_n3(file_path: Path) -> torch.Tensor:
    return torch.as_tensor(load_ply(str(file_path), return_color=False), dtype=torch.float32)


def _input_xyz_3n(file_path: Path) -> torch.Tensor:
    points = _input_points_n3(file_path)
    return points.transpose(0, 1).contiguous()


def _sparsepcgc_python_command(base_args: Any) -> List[str]:
    explicit = str(getattr(base_args, "sparsepcgc_python", "") or "").strip()
    if explicit:
        return [explicit]
    env_name = str(getattr(base_args, "sparsepcgc_env", "sparsepcgc") or "sparsepcgc").strip()
    candidates: List[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix).parent / env_name / "bin" / "python")
    candidates.extend([
        Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python",
        Path.home() / "anaconda3" / "envs" / env_name / "bin" / "python",
    ])
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return [str(candidate)]
    return ["conda", "run", "-n", env_name, "python"]


def _worker_request(
    *,
    file_path: Path,
    output_dir: Path,
    args_summary: Mapping[str, Any],
    scale_ae: int,
    scale_sr: int,
    decode: bool,
    topk: int,
) -> Dict[str, Any]:
    return {
        "input_file": str(file_path),
        "output_dir": str(output_dir),
        "sparsepcgc_root": args_summary.get("sparsepcgc_root", ""),
        "ckptdir": args_summary.get("checkpoint_path", ""),
        "ckptdir_sr": args_summary.get("checkpoint_sr_path", ""),
        "ckptdir_ae": args_summary.get("checkpoint_ae_path", ""),
        "ckptdir_low": args_summary.get("checkpoint_low_path", ""),
        "ckptdir_high": args_summary.get("checkpoint_high_path", ""),
        "ckptdir_offset": args_summary.get("checkpoint_offset_path", ""),
        "scale_AE": int(scale_ae),
        "scale_SR": int(scale_sr),
        "voxel_size": 1.0,
        "pos_quantscale": 1,
        "psnr_resolution": 1023,
        "decode": bool(decode),
        "topk_final": int(topk),
        "topk_per_layer": max(512, min(int(topk), 8192)),
        "occupancy_low_prob_threshold": 0.1,
    }


def _run_worker(
    *,
    base_args: Any,
    mode: str,
    request: Mapping[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    cmd = _sparsepcgc_python_command(base_args) + [str(WORKER_SCRIPT), "--mode", mode]
    proc = subprocess.run(
        cmd,
        input=json.dumps(request, sort_keys=True),
        cwd=str(MASTER_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(timeout),
        check=False,
    )
    stdout = proc.stdout.strip().splitlines()
    payload_text = stdout[-1] if stdout else ""
    try:
        payload = json.loads(payload_text)
    except Exception as exc:
        raise RuntimeError(
            f"phase1 worker returned non-json output rc={proc.returncode}: {exc}; "
            f"stdout={proc.stdout[-1000:]!r}; stderr={proc.stderr[-2000:]!r}"
        )
    if proc.returncode != 0 or payload.get("status") != "ok":
        raise RuntimeError(
            f"phase1 worker failed rc={proc.returncode}: {payload.get('message', payload)}; "
            f"stderr={proc.stderr[-2000:]}"
        )
    result = payload.get("result", {})
    if not isinstance(result, dict):
        raise RuntimeError(f"phase1 worker result is not dict: {type(result)}")
    result["phase1_worker_stderr_tail"] = proc.stderr[-2000:]
    return result


def _input_structure(file_path: Path, base_args: Any) -> Dict[str, Any]:
    xyz = _input_points_n3(file_path)
    coords, _meta = _unique_coords(xyz, base_args)
    _parents, _inverse, _slots, occ, patterns, pop = _parent_info(coords)
    pop_cpu = pop.detach().cpu()
    pop_hist = torch.bincount(pop_cpu, minlength=9)
    single_child = int((pop_cpu == 1).sum().item())
    active_parent = int(pop_cpu.numel())
    structure = {
        "structure_metric_basis": "input_voxels_before_codec",
        "baseline_voxel_count": int(coords.shape[0]),
        "active_node_count_proxy": active_parent,
        "leaf_node_count_proxy": int(coords.shape[0]),
        "single_child_node_count_proxy": single_child,
        "single_child_node_ratio_proxy": float(single_child) / max(float(active_parent), 1.0),
        "parent_popcount_hist_json": json.dumps({str(i): int(pop_hist[i].item()) for i in range(int(pop_hist.numel())) if int(pop_hist[i].item()) > 0}, sort_keys=True),
        "parent_pattern_unique_count": int(torch.unique(patterns).numel()),
        "chain_metric_status": "not_computed_in_phase1_light_structure_proxy",
        "chain_count": "",
        "chain_mean_length": "",
        "chain_max_length": "",
        "chain_total_length": "",
    }
    return structure


def _decoded_paths(stats: Mapping[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for key in ("decoded_copy_path", "decoded_path"):
        value = stats.get(key, "")
        if not value:
            continue
        for part in str(value).split(","):
            part = part.strip()
            if part and part != "(decode skipped)":
                paths.append(Path(part))
    return paths


def _extract_component_bits(stats: Mapping[str, Any]) -> Dict[str, Any]:
    bitstream_path = str(stats.get("bitstream_path", ""))
    bit_paths = [Path(part.strip()) for part in bitstream_path.split(",") if part.strip()]
    actual_bin_bits = 0
    actual_bin_json: List[Dict[str, Any]] = []
    for path in bit_paths:
        exists = path.exists()
        bits = int(path.stat().st_size * 8) if exists else 0
        actual_bin_bits += bits
        actual_bin_json.append({"path": str(path), "exists": bool(exists), "bits": bits})
    logical_file_size = _safe_float(stats.get("file_size"), float("nan"))
    component = {
        "logical_total_bits_from_coder": logical_file_size,
        "actual_bin_file_bits": actual_bin_bits,
        "component_bits_json": json.dumps(actual_bin_json, sort_keys=True),
        "component_bits_status": "bin_total_only; coder logical file_size may include AE_side_bits_and_num_points_list",
        "ae_related_bits": "",
        "sr_related_bits": "",
        "occupancy_related_bits": "",
        "header_or_side_bits": "",
    }
    return component


def _occupancy_fields(stats: Mapping[str, Any]) -> Dict[str, Any]:
    p_quant = _json_loads(stats.get("sparsepcgc_prob_true_quantiles_json"), {})
    b_quant = _json_loads(stats.get("sparsepcgc_bit_each_quantiles_json"), {})
    fields = {
        "occupancy_debug_available": stats.get("sparsepcgc_occupancy_debug_available", ""),
        "occupancy_accuracy": stats.get("sparsepcgc_occupancy_accuracy_at_0p5", ""),
        "occupied_recall": stats.get("sparsepcgc_occupied_recall_at_0p5", ""),
        "empty_accuracy": stats.get("sparsepcgc_empty_accuracy_at_0p5", ""),
        "balanced_accuracy": stats.get("sparsepcgc_balanced_accuracy_at_0p5", ""),
        "total_estimated_occupancy_bits": stats.get("sparsepcgc_estimated_occupancy_bits", ""),
        "mean_nll": stats.get("sparsepcgc_pred_occupancy_nll", ""),
        "p_true_mean": stats.get("sparsepcgc_prob_true_mean", ""),
        "p_true_median": p_quant.get("q50", ""),
        "p_true_q01": p_quant.get("q01", ""),
        "p_true_q05": p_quant.get("q05", ""),
        "p_true_q10": p_quant.get("q10", ""),
        "bit_each_mean": stats.get("sparsepcgc_pred_occupancy_nll", ""),
        "bit_each_q90": b_quant.get("q90", ""),
        "bit_each_q95": b_quant.get("q95", ""),
        "bit_each_q99": b_quant.get("q99", ""),
        "bit_each_max": b_quant.get("max", ""),
        "p_true_quantiles_json": stats.get("sparsepcgc_prob_true_quantiles_json", ""),
        "bit_each_quantiles_json": stats.get("sparsepcgc_bit_each_quantiles_json", ""),
        "bits_by_depth_json": stats.get("sparsepcgc_bits_by_depth_json", ""),
        "candidates_by_depth_json": stats.get("sparsepcgc_candidates_by_depth_json", ""),
        "occupied_by_depth_json": stats.get("sparsepcgc_occupied_by_depth_json", ""),
        "low_prob_occupied_by_depth_json": stats.get("sparsepcgc_low_prob_occupied_by_depth_json", ""),
        "high_bit_nodes_by_depth_json": stats.get("sparsepcgc_high_bit_nodes_by_depth_json", ""),
        "bits_by_parent_popcount_json": stats.get("sparsepcgc_bits_by_parent_popcount_json", ""),
        "bits_by_child_pattern_topk_json": stats.get("sparsepcgc_bits_by_child_pattern_topk_json", ""),
        "bits_by_block_topk_json": stats.get("sparsepcgc_bits_by_block_topk_json", ""),
        "top_high_bit_nodes_json": stats.get("sparsepcgc_top_high_bit_nodes_json", ""),
    }
    return fields


def _bit_concentration_rows(row: Mapping[str, Any], ratios: Sequence[float]) -> List[Dict[str, Any]]:
    top_nodes = _json_loads(row.get("top_high_bit_nodes_json", ""), [])
    if not isinstance(top_nodes, list):
        top_nodes = []
    bits = sorted([_safe_float(item.get("bits")) for item in top_nodes if isinstance(item, Mapping)], reverse=True)
    bits = [value for value in bits if math.isfinite(value)]
    total_est = _safe_float(row.get("total_estimated_occupancy_bits"), float("nan"))
    candidate_count = _safe_float(row.get("sparsepcgc_candidate_count"), float("nan"))
    rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        if math.isfinite(candidate_count) and candidate_count > 0:
            k = max(1, int(math.ceil(candidate_count * float(ratio))))
            basis = "occupancy_symbol_count"
            reached = len(bits) >= k
        else:
            k = max(1, int(math.ceil(len(bits) * float(ratio)))) if bits else 0
            basis = "debug_top_high_bit_nodes_json_available_only"
            reached = False
        selected = bits[: min(k, len(bits))]
        bit_sum = float(sum(selected)) if selected else float("nan")
        share = bit_sum / total_est if selected and math.isfinite(total_est) and total_est > 0 else float("nan")
        rows.append({
            "setting_id": row.get("setting_id", ""),
            "dataset": row.get("dataset", ""),
            "sequence": row.get("sequence", ""),
            "frame": row.get("frame", ""),
            "ratio": float(ratio),
            "ratio_percent": float(ratio) * 100.0,
            "basis": basis,
            "candidate_count": candidate_count if math.isfinite(candidate_count) else "",
            "debug_top_nodes_available": len(bits),
            "k_requested": k,
            "budget_reached_by_debug_topk": bool(reached),
            "top_bit_sum": bit_sum,
            "top_bit_share_ratio": share,
            "top_bit_share_percent": share * 100.0 if math.isfinite(share) else float("nan"),
            "total_estimated_occupancy_bits": total_est,
        })
    return rows


@dataclass(frozen=True)
class EvalResult:
    row: Dict[str, Any]
    debug_row: Dict[str, Any]
    cache_hit_decode: bool
    cache_hit_debug: bool
    status: str
    error: str = ""


def _run_decode_or_cache(
    *,
    cache: MutableMapping[str, Dict[str, Any]],
    cache_path: Path,
    base_args: Any,
    file_path: Path,
    out_dir: Path,
    args_summary: Mapping[str, Any],
    scale_ae: int,
    scale_sr: int,
    timeout: float,
    formal_max_points: int,
    normal_max_points: int,
    use_pc_error: bool,
    pc_error_path: str,
) -> Tuple[Dict[str, Any], bool]:
    setting = _setting_id(scale_ae, scale_sr)
    key = _cache_key(
        phase="phase1_decode",
        file_path=file_path,
        setting_id=setting,
        scale_ae=scale_ae,
        scale_sr=scale_sr,
        debug_mode="decode",
        args_summary=args_summary,
    )
    if key in cache:
        row = dict(cache[key])
        row["cache_hit_decode"] = True
        return row, True
    phase_out = out_dir / "codec_outputs" / setting / "decode"
    decoded_dir = out_dir / "decoded" / setting
    request = _worker_request(
        file_path=file_path,
        output_dir=phase_out,
        args_summary=args_summary,
        scale_ae=scale_ae,
        scale_sr=scale_sr,
        decode=True,
        topk=1024,
    )
    start = time.time()
    stats = _run_worker(base_args=base_args, mode="decode", request=request, timeout=timeout)
    elapsed = time.time() - start
    component = _extract_component_bits(stats)
    decoded_paths = _decoded_paths(stats)
    decoded_path = str(decoded_paths[0]) if decoded_paths else ""
    quality: Dict[str, Any] = {}
    match_count = ""
    match_ratio = ""
    decode_lossless = ""
    if decoded_path:
        match_count, match_ratio, decode_lossless = _coord_match_ratio_from_paths(file_path, decoded_path)
        quality = _quality_from_paths(
            file_path,
            decoded_path,
            formal_max_points=formal_max_points,
            normal_max_points=normal_max_points,
            pc_error_path=pc_error_path,
            use_pc_error=use_pc_error,
        )
    row = {
        "status": "ok",
        "dataset": DEFAULT_DATASET,
        "sequence": DEFAULT_SEQUENCE,
        "frame": DEFAULT_FRAME,
        "setting_id": setting,
        "scale_AE": int(scale_ae),
        "scale_SR": int(scale_sr),
        "voxel_size": 1.0,
        "posQuantscale": 1,
        "pc_type": "dense",
        "mode": "dense_lossy",
        "input_path": str(file_path),
        "input_hash": _sha256_file(file_path),
        **args_summary,
        "actual_total_bits": component["logical_total_bits_from_coder"],
        "actual_bin_file_bits": component["actual_bin_file_bits"],
        "bpp": stats.get("bpp", ""),
        "num_points_raw": stats.get("num_points_raw", stats.get("input_point_count", "")),
        "decoded_voxel_count": stats.get("num_points", stats.get("decoded_codec_point_count", "")),
        "decoded_path": decoded_path,
        "baseline_decode_coord_match_count": match_count,
        "baseline_decode_coord_match_ratio": match_ratio,
        "baseline_decode_lossless": decode_lossless,
        "encode_time": stats.get("enc_time", ""),
        "decode_time": stats.get("dec_time", ""),
        "all_encode_time": stats.get("all_enc_time", ""),
        "all_decode_time": stats.get("all_dec_time", ""),
        "phase1_wrapper_elapsed": elapsed,
        "cache_hit_decode": False,
        "raw_stats_json": json.dumps({k: v for k, v in stats.items() if isinstance(v, (str, int, float, bool))}, sort_keys=True, default=str),
        **component,
        "d1_psnr": quality.get("d1_psnr", ""),
        "d2_psnr": quality.get("d2_psnr", ""),
        "d1_mse": quality.get("d1_mse", ""),
        "d2_mse": quality.get("d2_mse", ""),
        "chamfer": quality.get("chamfer", ""),
        "quality_eval_mode": quality.get("quality_eval_mode", ""),
        "mynet_eval_success": quality.get("mynet_eval_success", ""),
        "pc_error_d_success": quality.get("pc_error_d_success", ""),
    }
    cache[key] = dict(row)
    _save_cache(cache_path, cache)
    return row, False


def _run_debug_or_cache(
    *,
    cache: MutableMapping[str, Dict[str, Any]],
    cache_path: Path,
    base_args: Any,
    file_path: Path,
    out_dir: Path,
    args_summary: Mapping[str, Any],
    scale_ae: int,
    scale_sr: int,
    timeout: float,
    topk: int,
    exact_occupancy: bool,
) -> Tuple[Dict[str, Any], bool]:
    setting = _setting_id(scale_ae, scale_sr)
    key = _cache_key(
        phase="phase1_occupancy_debug",
        file_path=file_path,
        setting_id=setting,
        scale_ae=scale_ae,
        scale_sr=scale_sr,
        debug_mode=f"occupancy_topk{topk}_exact{int(exact_occupancy)}",
        args_summary=args_summary,
    )
    if key in cache:
        row = dict(cache[key])
        row["cache_hit_debug"] = True
        return row, True
    phase_out = out_dir / "codec_outputs" / setting / "debug"
    request = _worker_request(
        file_path=file_path,
        output_dir=phase_out,
        args_summary=args_summary,
        scale_ae=scale_ae,
        scale_sr=scale_sr,
        decode=False,
        topk=topk,
    )
    start = time.time()
    stats = _run_worker(base_args=base_args, mode="debug", request=request, timeout=timeout)
    elapsed = time.time() - start
    occ = _occupancy_fields(stats)
    row = {
        "status": "ok",
        "dataset": DEFAULT_DATASET,
        "sequence": DEFAULT_SEQUENCE,
        "frame": DEFAULT_FRAME,
        "setting_id": setting,
        "scale_AE": int(scale_ae),
        "scale_SR": int(scale_sr),
        "voxel_size": 1.0,
        "posQuantscale": 1,
        "pc_type": "dense",
        "mode": "dense_lossy",
        "input_path": str(file_path),
        "input_hash": _sha256_file(file_path),
        **args_summary,
        "debug_topk_final": int(topk),
        "debug_exact_occupancy_requested": bool(exact_occupancy),
        "debug_setting_args_propagated_to_worker": True,
        "debug_known_limitation": "Phase1 wrapper applies existing occupancy debug to the tensor after dense AE/SR downscale; SparsePCGC body is unchanged.",
        "debug_bit_size": stats.get("bit", stats.get("file_size", "")),
        "debug_bpp": stats.get("bpp", ""),
        "debug_elapsed": elapsed,
        "cache_hit_debug": False,
        "raw_debug_stats_json": json.dumps({k: v for k, v in stats.items() if isinstance(v, (str, int, float, bool))}, sort_keys=True, default=str),
        **occ,
    }
    # Keep any extra sparsepcgc fields that Phase 2 may need.
    for key_name, value in stats.items():
        if str(key_name).startswith("sparsepcgc_") and key_name not in row:
            row[key_name] = value
    cache[key] = dict(row)
    _save_cache(cache_path, cache)
    return row, False


def _evaluate_setting(
    *,
    phase_label: str,
    index: int,
    total: int,
    base_args: Any,
    file_path: Path,
    out_dir: Path,
    cache: MutableMapping[str, Dict[str, Any]],
    cache_path: Path,
    args_summary: Mapping[str, Any],
    scale_ae: int,
    scale_sr: int,
    timeout: float,
    topk: int,
    formal_max_points: int,
    normal_max_points: int,
    use_pc_error: bool,
    pc_error_path: str,
) -> EvalResult:
    setting = _setting_id(scale_ae, scale_sr)
    start = time.time()
    cache_hit_decode = False
    cache_hit_debug = False
    try:
        decode_row, cache_hit_decode = _run_decode_or_cache(
            cache=cache,
            cache_path=cache_path,
            base_args=base_args,
            file_path=file_path,
            out_dir=out_dir,
            args_summary=args_summary,
            scale_ae=scale_ae,
            scale_sr=scale_sr,
            timeout=timeout,
            formal_max_points=formal_max_points,
            normal_max_points=normal_max_points,
            use_pc_error=use_pc_error,
            pc_error_path=pc_error_path,
        )
        debug_row, cache_hit_debug = _run_debug_or_cache(
            cache=cache,
            cache_path=cache_path,
            base_args=base_args,
            file_path=file_path,
            out_dir=out_dir,
            args_summary=args_summary,
            scale_ae=scale_ae,
            scale_sr=scale_sr,
            timeout=timeout,
            topk=topk,
            exact_occupancy=False,
        )
        merged = dict(decode_row)
        for key, value in debug_row.items():
            if key not in merged or merged.get(key, "") == "":
                merged[key] = value
        elapsed = time.time() - start
        print(
            f"[{phase_label} {index}/{total}] {setting} "
            f"cache=decode:{'hit' if cache_hit_decode else 'miss'},debug:{'hit' if cache_hit_debug else 'miss'} "
            f"encode=ok occupancy={debug_row.get('occupancy_debug_available', '')} elapsed={elapsed:.1f}s",
            flush=True,
        )
        return EvalResult(merged, debug_row, cache_hit_decode, cache_hit_debug, "ok")
    except Exception as exc:
        elapsed = time.time() - start
        err = f"{type(exc).__name__}:{exc}"
        print(f"[{phase_label} {index}/{total}] {setting} status=error elapsed={elapsed:.1f}s error={err}", flush=True)
        base = {
            "status": "error",
            "error": err,
            "dataset": DEFAULT_DATASET,
            "sequence": DEFAULT_SEQUENCE,
            "frame": DEFAULT_FRAME,
            "setting_id": setting,
            "scale_AE": int(scale_ae),
            "scale_SR": int(scale_sr),
            "voxel_size": 1.0,
            "posQuantscale": 1,
            "input_path": str(file_path),
            "input_hash": _sha256_file(file_path) if file_path.exists() else "",
        }
        return EvalResult(base, dict(base), cache_hit_decode, cache_hit_debug, "error", err)


def _choose_phase1c(rows: Sequence[Mapping[str, Any]], *, max_count: int = 4) -> List[Tuple[int, int, str]]:
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return []
    selected: Dict[str, Tuple[int, int, str]] = {}

    def add(row: Mapping[str, Any], role: str) -> None:
        sid = str(row.get("setting_id"))
        selected.setdefault(sid, (int(row.get("scale_AE", 0)), int(row.get("scale_SR", 0)), role))

    d1_sorted = sorted(ok, key=lambda r: _safe_float(r.get("d1_psnr"), -1e9), reverse=True)
    bits_sorted = sorted(ok, key=lambda r: _safe_float(r.get("actual_total_bits"), 1e30))
    occ_sorted = sorted(ok, key=lambda r: _safe_float(r.get("bit_each_q99"), -1e9), reverse=True)
    add(d1_sorted[0], "high_quality_anchor")
    add(d1_sorted[len(d1_sorted) // 2], "middle_quality")
    add(bits_sorted[0], "low_rate")
    if occ_sorted:
        add(occ_sorted[0], "occupancy_tail_or_context_headroom")
    for row in d1_sorted:
        if len(selected) >= max_count:
            break
        add(row, "fill_psnr_coverage")
    return list(selected.values())[:max_count]


def _depth_rows(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    bits_by_depth = _json_loads(row.get("bits_by_depth_json"), {})
    candidates_by_depth = _json_loads(row.get("candidates_by_depth_json"), {})
    occupied_by_depth = _json_loads(row.get("occupied_by_depth_json"), {})
    high_bit_by_depth = _json_loads(row.get("high_bit_nodes_by_depth_json"), {})
    keys = set()
    for payload in (bits_by_depth, candidates_by_depth, occupied_by_depth, high_bit_by_depth):
        if isinstance(payload, Mapping):
            keys.update(str(k) for k in payload.keys())
    out = []
    for depth in sorted(keys, key=lambda x: int(x) if str(x).lstrip("-").isdigit() else str(x)):
        out.append({
            "setting_id": row.get("setting_id", ""),
            "dataset": row.get("dataset", ""),
            "sequence": row.get("sequence", ""),
            "frame": row.get("frame", ""),
            "depth": depth,
            "estimated_bits": bits_by_depth.get(depth, "") if isinstance(bits_by_depth, Mapping) else "",
            "candidate_count": candidates_by_depth.get(depth, "") if isinstance(candidates_by_depth, Mapping) else "",
            "occupied_count": occupied_by_depth.get(depth, "") if isinstance(occupied_by_depth, Mapping) else "",
            "high_bit_node_count": high_bit_by_depth.get(depth, "") if isinstance(high_bit_by_depth, Mapping) else "",
        })
    return out


def _make_report(
    *,
    out_dir: Path,
    phase1a_rows: Sequence[Mapping[str, Any]],
    phase1b_rows: Sequence[Mapping[str, Any]],
    phase1c_rows: Sequence[Mapping[str, Any]],
    selected_1c: Sequence[Tuple[int, int, str]],
    phase2_candidates: Sequence[Tuple[int, int, str]],
    gate: Mapping[str, Any],
) -> None:
    lines = [
        "# Phase 1 SparsePCGC native dense baseline inventory",
        "",
        f"- generated_at: {_now()}",
        f"- input: {DEFAULT_INPUT}",
        "- scope: baseline only; no Add/Prune/Merge/Adjust",
        "- codec setting: native dense, voxel_size=1, posQuantscale=1, official AE/SR pairs only",
        "",
        "## Phase 1A",
    ]
    for row in phase1a_rows:
        lines.append(
            f"- {row.get('setting_id')}: status={row.get('status')} bits={row.get('actual_total_bits')} "
            f"D1={row.get('d1_psnr')} D2={row.get('d2_psnr')} occupancy={row.get('occupancy_debug_available')}"
        )
    lines.extend(["", "## Phase 1B official pair summary"])
    for row in phase1b_rows:
        lines.append(
            f"- {row.get('setting_id')}: bits={row.get('actual_total_bits')} bpp={row.get('bpp')} "
            f"D1={row.get('d1_psnr')} D2={row.get('d2_psnr')} q99={row.get('bit_each_q99')}"
        )
    lines.extend(["", "## Phase 1C selected settings"])
    for ae, sr, role in selected_1c:
        lines.append(f"- {_setting_id(ae, sr)}: {role}")
    lines.extend(["", "## Phase 1C detailed occupancy/context"])
    for row in phase1c_rows:
        lines.append(
            f"- {row.get('setting_id')}: estimated_bits={row.get('total_estimated_occupancy_bits')} "
            f"NLL={row.get('mean_nll')} p_true_mean={row.get('p_true_mean')} top debug nodes={len(_json_loads(row.get('top_high_bit_nodes_json'), []))}"
        )
    lines.extend(["", "## Phase 2 candidate settings"])
    for ae, sr, role in phase2_candidates:
        lines.append(f"- {_setting_id(ae, sr)}: {role}")
    lines.extend([
        "",
        "## Gate",
        "```json",
        json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## Notes",
        "- estimated occupancy bits and actual total bits are intentionally separated.",
        "- occupancy debug is produced by a Phase1 wrapper hook on the dense AE/SR downscaled tensor; identical downscaled tensors can yield identical debug metrics across different AE/SR pairs.",
        "- bdXX external requantization, qlevel, voxel_size!=1, posQuantscale!=1 were not used.",
    ])
    (out_dir / "phase1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _phase0_ok() -> bool:
    required = [
        "phase0_gate_result.json",
        "phase0_dense_official_pairs.csv",
        "phase0_candidate_settings.csv",
        "phase0_cache_schema.json",
    ]
    if not all((PHASE0_DIR / name).exists() for name in required):
        return False
    try:
        payload = json.loads((PHASE0_DIR / "phase0_gate_result.json").read_text(encoding="utf-8"))
        return bool(payload.get("pass") or payload.get("gate_pass") or str(payload.get("status", "")).upper() == "PASS")
    except Exception:
        return False


def _pairs_for_phase(phase: str, selected_file: Path) -> List[Tuple[int, int]]:
    if phase == "1A":
        return [(1, 1)]
    if phase == "1B":
        return list(OFFICIAL_PAIRS)
    if selected_file.exists():
        rows = _read_csv(selected_file)
        pairs = []
        for row in rows:
            try:
                pairs.append((int(row["scale_AE"]), int(row["scale_SR"])))
            except Exception:
                pass
        if pairs:
            return pairs
    return [(1, 0), (1, 1), (1, 2)]


def _phase2_candidates(phase1c_rows: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int, str]]:
    ok = [row for row in phase1c_rows if row.get("status") == "ok"]
    if not ok:
        return []
    scored = []
    for row in ok:
        d1 = _safe_float(row.get("d1_psnr"), float("nan"))
        q99 = _safe_float(row.get("bit_each_q99"), float("nan"))
        bits = _safe_float(row.get("actual_total_bits"), float("nan"))
        if not math.isfinite(d1) or not math.isfinite(bits):
            continue
        # 極端な最低rate/最高rateだけを避け、high-bit tailを重視する軽い選定スコア。
        score = (q99 if math.isfinite(q99) else 0.0) + 0.01 * d1
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    out = []
    for _score, row in scored[:3]:
        out.append((int(row.get("scale_AE", 0)), int(row.get("scale_SR", 0)), "phase2_atomic_candidate_by_tail_and_quality"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 native dense SparsePCGC baseline inventory")
    parser.add_argument("--phase", choices=("1A", "1B", "1C", "all"), default="all")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--topk-light", type=int, default=4096)
    parser.add_argument("--topk-detail", type=int, default=32768)
    parser.add_argument("--formal-max-points", type=int, default=3000)
    parser.add_argument("--normal-max-points", type=int, default=3000)
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--pc-error-path", default="")
    cli = parser.parse_args()

    out_dir = cli.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cli.input.exists():
        print(f"[Phase1] input missing: {cli.input}", file=sys.stderr)
        return 2
    if not _phase0_ok():
        print(f"[Phase1] Phase0 output/Gate not found or not PASS: {PHASE0_DIR}", file=sys.stderr)
        return 3

    base_args = _base_args()
    args_summary = _encoder_args_summary(base_args)
    cache_path = out_dir / "phase1_cache.json"
    cache = _load_cache(cache_path)
    structure = _input_structure(cli.input, base_args)

    selected_file = out_dir / "phase1c_selected_settings.csv"
    run_phases = ["1A", "1B", "1C"] if cli.phase == "all" else [cli.phase]
    manifest = {
        "generated_at": _now(),
        "schema_version": SCHEMA_VERSION,
        "input": str(cli.input),
        "input_hash": _sha256_file(cli.input),
        "phase0_dir": str(PHASE0_DIR),
        "official_pairs": [{"scale_AE": ae, "scale_SR": sr, "setting_id": _setting_id(ae, sr)} for ae, sr in OFFICIAL_PAIRS],
        "args_summary": args_summary,
        "dry_run": bool(cli.dry_run),
        "run_phases": run_phases,
    }
    _write_json(out_dir / "phase1_run_manifest.json", manifest)

    if cli.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    phase1a_rows: List[Dict[str, Any]] = _read_csv(out_dir / "phase1a_smoke.csv")
    phase1b_rows: List[Dict[str, Any]] = _read_csv(out_dir / "phase1b_official_pair_baselines.csv")
    phase1c_rows: List[Dict[str, Any]] = _read_csv(out_dir / "phase1c_setting_occupancy_detail.csv")
    phase1c_debug_rows: List[Dict[str, Any]] = _read_csv(out_dir / "phase1c_context_inventory.csv")

    for phase in run_phases:
        pairs = _pairs_for_phase(phase, selected_file)
        results: List[EvalResult] = []
        for idx, (ae, sr) in enumerate(pairs, start=1):
            result = _evaluate_setting(
                phase_label=f"Phase{phase}",
                index=idx,
                total=len(pairs),
                base_args=base_args,
                file_path=cli.input,
                out_dir=out_dir,
                cache=cache,
                cache_path=cache_path,
                args_summary=args_summary,
                scale_ae=ae,
                scale_sr=sr,
                timeout=cli.timeout,
                topk=cli.topk_detail if phase == "1C" else cli.topk_light,
                formal_max_points=cli.formal_max_points,
                normal_max_points=cli.normal_max_points,
                use_pc_error=cli.use_pc_error,
                pc_error_path=cli.pc_error_path,
            )
            result.row.update(structure)
            results.append(result)

        if phase == "1A":
            phase1a_rows = [item.row for item in results]
            _write_csv(out_dir / "phase1a_smoke.csv", phase1a_rows)
            gate_1a = {
                "phase": "1A",
                "pass": bool(
                    phase1a_rows
                    and phase1a_rows[0].get("status") == "ok"
                    and str(phase1a_rows[0].get("actual_total_bits", "")) not in {"", "nan"}
                    and str(phase1a_rows[0].get("d1_psnr", "")) not in {"", "nan"}
                ),
                "occupancy_debug_available": phase1a_rows[0].get("occupancy_debug_available", "") if phase1a_rows else "",
                "note": "occupancy debug may be available but dense_lossy semantic setting-specificity is checked in Phase1B/1C.",
            }
            _write_json(out_dir / "phase1a_gate_result.json", gate_1a)
            if not gate_1a["pass"]:
                print("[Phase1] Phase1A Gate FAIL; stopping before Phase1B.", flush=True)
                break

        if phase == "1B":
            phase1b_rows = [item.row for item in results]
            _write_csv(out_dir / "phase1b_official_pair_baselines.csv", phase1b_rows)
            _write_csv(out_dir / "phase1b_component_bits.csv", [{k: row.get(k, "") for k in (
                "setting_id", "scale_AE", "scale_SR", "actual_total_bits", "actual_bin_file_bits",
                "component_bits_json", "component_bits_status", "ae_related_bits", "sr_related_bits",
                "occupancy_related_bits", "header_or_side_bits",
            )} for row in phase1b_rows])
            _write_csv(out_dir / "phase1b_structure_summary.csv", [{**{k: row.get(k, "") for k in (
                "setting_id", "scale_AE", "scale_SR", "structure_metric_basis", "baseline_voxel_count",
                "active_node_count_proxy", "leaf_node_count_proxy", "single_child_node_count_proxy",
                "single_child_node_ratio_proxy", "parent_popcount_hist_json", "parent_pattern_unique_count",
                "chain_metric_status", "chain_count", "chain_mean_length", "chain_max_length", "chain_total_length",
            )}} for row in phase1b_rows])
            _write_csv(out_dir / "phase1b_light_occupancy_summary.csv", [{k: row.get(k, "") for k in (
                "setting_id", "scale_AE", "scale_SR", "occupancy_debug_available", "occupancy_accuracy",
                "occupied_recall", "empty_accuracy", "balanced_accuracy", "total_estimated_occupancy_bits",
                "mean_nll", "p_true_mean", "p_true_median", "bit_each_mean", "bit_each_q95", "bit_each_q99",
                "debug_known_limitation",
            )} for row in phase1b_rows])
            selected = _choose_phase1c(phase1b_rows, max_count=4)
            selected_rows = [
                {
                    "setting_id": _setting_id(ae, sr),
                    "scale_AE": ae,
                    "scale_SR": sr,
                    "selection_role": role,
                    "selection_reason": "selected from Phase1B by quality/rate/occupancy-tail coverage",
                }
                for ae, sr, role in selected
            ]
            _write_csv(selected_file, selected_rows)
            lines = ["# Phase 1B RD summary", ""]
            for row in phase1b_rows:
                lines.append(
                    f"- {row.get('setting_id')}: bits={row.get('actual_total_bits')} D1={row.get('d1_psnr')} "
                    f"D2={row.get('d2_psnr')} occupancy={row.get('occupancy_debug_available')}"
                )
            lines.extend(["", "## Phase 1C selection"] + [f"- {r['setting_id']}: {r['selection_role']}" for r in selected_rows])
            (out_dir / "phase1b_rd_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        if phase == "1C":
            phase1c_rows = [item.row for item in results]
            phase1c_debug_rows = [item.debug_row for item in results]
            _write_csv(out_dir / "phase1c_setting_occupancy_detail.csv", phase1c_rows)
            depth_rows: List[Dict[str, Any]] = []
            concentration_rows: List[Dict[str, Any]] = []
            for row in phase1c_rows:
                depth_rows.extend(_depth_rows(row))
                concentration_rows.extend(_bit_concentration_rows(row, ratios=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.03)))
            _write_csv(out_dir / "phase1c_depth_metrics.csv", depth_rows)
            _write_csv(out_dir / "phase1c_bit_concentration.csv", concentration_rows)
            _write_csv(out_dir / "phase1c_context_inventory.csv", phase1c_debug_rows)
            _write_csv(out_dir / "phase1c_setting_comparison.csv", [{k: row.get(k, "") for k in (
                "setting_id", "scale_AE", "scale_SR", "actual_total_bits", "actual_bin_file_bits",
                "bpp", "d1_psnr", "d2_psnr", "chamfer", "occupancy_accuracy", "occupied_recall",
                "empty_accuracy", "balanced_accuracy", "total_estimated_occupancy_bits", "mean_nll",
                "p_true_mean", "bit_each_q95", "bit_each_q99", "debug_known_limitation",
            )} for row in phase1c_rows])

    selected_1c = []
    for row in _read_csv(selected_file):
        try:
            selected_1c.append((int(row["scale_AE"]), int(row["scale_SR"]), row.get("selection_role", "")))
        except Exception:
            pass
    phase2 = _phase2_candidates(phase1c_rows)
    gate = {
        "phase": "Phase1",
        "pass": bool(
            len(phase1b_rows) == 6
            and all(row.get("status") == "ok" for row in phase1b_rows)
            and len(phase1c_rows) >= 3
            and all(str(row.get("total_estimated_occupancy_bits", "")) not in {"", "nan"} for row in phase1c_rows)
            and len(phase2) >= 2
        ),
        "official_6_pair_baselines": len(phase1b_rows),
        "phase1c_detail_settings": len(phase1c_rows),
        "phase2_candidate_count": len(phase2),
        "estimated_bits_and_actual_bits_separated": True,
        "bdxx_used": False,
        "voxel_size_posquantscale_fixed": True,
        "body_code_modified_by_this_script": False,
        "limitation": "occupancy debug is produced by a Phase1 wrapper hook on the dense AE/SR downscaled tensor; identical entropy-coding tensors can yield identical debug metrics across different AE/SR pairs.",
    }
    _write_json(out_dir / "phase1_gate_result.json", gate)
    _make_report(
        out_dir=out_dir,
        phase1a_rows=phase1a_rows,
        phase1b_rows=phase1b_rows,
        phase1c_rows=phase1c_rows,
        selected_1c=selected_1c,
        phase2_candidates=phase2,
        gate=gate,
    )
    print(f"[Phase1] outputs: {out_dir}", flush=True)
    print(f"[Phase1] Gate pass={gate['pass']}", flush=True)
    return 0 if (cli.phase != "all" or gate["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
