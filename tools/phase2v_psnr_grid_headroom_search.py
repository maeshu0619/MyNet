#!/usr/bin/env python
"""Phase2V PSNR-grid SparsePCGC setting headroom search.

研究用probe。
SparsePCGCのmodel/checkpointは壊さず、既存actual encoderへ渡せる
voxel_size / pos_quantscale を広く探索する。

主な目的:
1. GTを同じSparsePCGC設定で圧縮・復号したときのD1 PSNRが、
   10〜100dBのどの帯域に分布するかを網羅的に調べる。
2. 各PSNR帯域の代表設定に対して high_bit_raw / low_prob_snap を適用し、
   occupancy予測精度・high-bit share・actual raw改善余地を診断する。
3. baselineとproposedは必ず同一codec設定で比較する。

注意:
- train.py / network.py / structure_actuator.py / args.py は編集しない。
- checkpointやnetwork出力probabilityは変更しない。
- psnr_resolutionは原則固定し、codec挙動探索は voxel_size / pos_quantscale 中心にする。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.utils.data.dataset import load_ply
from models.utils.loss.actual_encoder import build_actual_encoder

from tools.context_aware_where_probe import _coords_to_xyz, _parse_csv_text, _safe_float, _unique_coords
from tools.phase2_rdo_beam_probe import _coord_match_ratio_from_paths, _quality_from_paths, _write_csv
from tools.phase2t_multi_rule_context_edit_headroom import _base_headroom_row
from tools.phase2u_high_bit_candidate_rewrite_rd_probe import (
    _candidate_to_coords_u,
    _dataset_name,
    _prepare_args,
    _sequence_name,
)

DEFAULT_FILE = "/data/maejima/data/ground/8i/loot/loot_vox10_1000.ply"
DEFAULT_OUTPUT = "/data/maejima/log/PHASE2V_psnr_grid_headroom.csv"
DEFAULT_SELECTED_OUTPUT = "/data/maejima/log/PHASE2V_psnr_grid_selected_candidates.csv"
DEFAULT_INVENTORY = "/data/maejima/log/PHASE2V_psnr_grid_setting_inventory.csv"

# 10〜100dBの範囲を埋めるため、1.0近傍を細かく、粗い領域を広く探索する。
DEFAULT_VOXEL_VALUES = (
    "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.825,0.85,0.875,0.90,0.925,0.95,0.96,0.97,0.98,"
    "0.985,0.99,0.995,0.9975,1.0,1.0025,1.005,1.01,1.015,1.02,1.03,1.04,1.05,"
    "1.06,1.075,1.09,1.10,1.11,1.125,1.15,1.175,1.20,1.225,1.25,1.275,1.30,"
    "1.35,1.40,1.45,1.50,1.60,1.70,1.80,1.90,2.0,2.25,2.50,2.75,3.0,3.5,4.0"
)
DEFAULT_POSQ_VALUES = "1,2,3,4,5,6"
DEFAULT_CANDIDATES = "high_bit_raw_prune,low_prob_snap_to_existing"
DEFAULT_BUDGETS = "0.005,0.01,0.02,0.03"


def _read_rows(path: str) -> list[Dict[str, object]]:
    if not path or not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for raw in _parse_csv_text(text):
        if str(raw).strip():
            out.append(float(raw))
    return out


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for raw in _parse_csv_text(text):
        if str(raw).strip():
            out.append(int(float(raw)))
    return out


def _parse_settings(text: str) -> list[Tuple[str, float, int]]:
    settings: list[Tuple[str, float, int]] = []
    for raw in _parse_csv_text(text):
        parts = str(raw).split(":")
        if len(parts) != 3:
            raise ValueError(f"setting must be id:voxel_size:pos_quantscale, got {raw!r}")
        setting_id, voxel_size, pos_q = parts
        voxel = float(voxel_size)
        quant = int(float(pos_q))
        if voxel <= 0:
            raise ValueError(f"voxel_size must be > 0 in setting {raw!r}")
        if quant <= 0:
            raise ValueError(f"pos_quantscale must be > 0 in setting {raw!r}")
        settings.append((setting_id, voxel, quant))
    return settings


def _dedupe_settings(settings: Sequence[Tuple[str, float, int]]) -> list[Tuple[str, float, int]]:
    seen: set[tuple[float, int]] = set()
    out: list[Tuple[str, float, int]] = []
    for setting_id, voxel, posq in settings:
        key = (round(float(voxel), 8), int(posq))
        if key in seen:
            continue
        seen.add(key)
        out.append((str(setting_id), float(voxel), int(posq)))
    return out


def _setting_id(voxel: float, posq: int) -> str:
    return f"vox{float(voxel):g}_pq{int(posq)}"


def _generate_settings(
    *,
    voxel_values: Sequence[float],
    posq_values: Sequence[int],
    full_pair_grid: bool,
    include_default: bool,
) -> list[Tuple[str, float, int]]:
    settings: list[Tuple[str, float, int]] = []
    if include_default:
        settings.append(("default", 1.0, 1))

    if full_pair_grid:
        for voxel in voxel_values:
            for posq in posq_values:
                settings.append((_setting_id(float(voxel), int(posq)), float(voxel), int(posq)))
    else:
        # 軽量探索: 1変数sweep + 近傍の小さいpairのみ。
        for voxel in voxel_values:
            settings.append((_setting_id(float(voxel), 1), float(voxel), 1))
        for posq in posq_values:
            settings.append((_setting_id(1.0, int(posq)), 1.0, int(posq)))
        for voxel in voxel_values:
            if 0.90 <= float(voxel) <= 1.50:
                for posq in posq_values:
                    if int(posq) in (1, 2):
                        settings.append((_setting_id(float(voxel), int(posq)), float(voxel), int(posq)))
    return _dedupe_settings(settings)


def _safe_num(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _json_quantile(value: object, key: str) -> float:
    try:
        parsed = json.loads(str(value or "{}"))
        if isinstance(parsed, Mapping):
            return float(parsed.get(key, float("nan")))
    except Exception:
        pass
    return float("nan")


def _parse_candidate_budgets(text: str) -> list[float]:
    out: list[float] = []
    for raw in _parse_csv_text(text):
        value = float(raw)
        if value <= 0:
            raise ValueError(f"budget must be > 0, got {raw!r}")
        if value >= 1.0:
            value = value / 100.0
        out.append(float(value))
    return out


def _candidate_prefix(candidate: str, budget: float) -> str:
    budget_pct = float(budget) * 100.0
    label = f"{budget_pct:.3f}".replace(".", "p").rstrip("0").rstrip("p")
    return f"{candidate}_b{label}pct"


def _repo_candidate_paths() -> list[Path]:
    return [
        REPO_ROOT.parent / "compress/octree/SparsePCGC/encoder_multiple.py",
        REPO_ROOT / "models/utils/loss/actual_encoder.py",
        REPO_ROOT / "tools/phase2v_sparsepcgc_setting_sweep.py",
        Path(__file__).resolve(),
    ]


def _scan_setting_inventory() -> list[Dict[str, object]]:
    patterns = [
        "voxel", "voxel_size", "pos_quantscale", "pos_quantscale_list",
        "psnr_resolution", "resolution", "depth", "max_depth", "octree",
        "quant", "quantization", "scale", "dense_scale",
        "dense_scale_ae_list", "dense_scale_sr_list", "block", "block_size",
        "cube", "root", "normalize", "checkpoint", "ckpt", "model",
        "entropy", "coder", "bitstream", "lossless", "lossy",
    ]
    regex = re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)
    rows: list[Dict[str, object]] = []
    for path in _repo_candidate_paths():
        if not path.exists():
            rows.append({"file": str(path), "line": "", "setting_name": "", "snippet": "file_not_found", "kind_guess": "", "risk_guess": "unknown"})
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for idx, line in enumerate(lines, start=1):
            if not regex.search(line):
                continue
            snippet = line.strip()
            name_match = re.search(r"--([A-Za-z0-9_\-]+)|args\.([A-Za-z0-9_]+)|sparsepcgc_([A-Za-z0-9_]+)", snippet)
            setting_name = next((g for g in (name_match.groups() if name_match else []) if g), "")
            lower = snippet.lower()
            if "psnr" in lower:
                kind = "evaluation_or_resolution"
            elif "checkpoint" in lower or "ckpt" in lower or "model" in lower:
                kind = "model_compatibility"
            elif "voxel" in lower or "quant" in lower or "scale" in lower or "depth" in lower:
                kind = "codec_or_quantization_candidate"
            else:
                kind = "unknown"
            risk = "low"
            if kind == "model_compatibility":
                risk = "high"
            elif "dense_scale" in lower:
                risk = "medium_high"
            elif "psnr" in lower:
                risk = "medium"
            rows.append({
                "file": str(path),
                "line": idx,
                "setting_name": setting_name,
                "snippet": snippet[:240],
                "kind_guess": kind,
                "risk_guess": risk,
            })
    return rows


def _write_inventory_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    if path:
        _write_csv(path, [dict(r) for r in rows])


def _setting_args(base_args, *, voxel_size: float, pos_quantscale: int, psnr_resolution: int, decode: bool, topk: int):
    args = copy.copy(base_args)
    args.sparsepcgc_voxel_size = float(voxel_size)
    args.sparsepcgc_pos_quantscale = int(pos_quantscale)
    args.sparsepcgc_psnr_resolution = int(psnr_resolution)
    args.sparsepcgc_skip_decode = not bool(decode)
    args.enable_sparsepcgc_occupancy_debug = bool(not decode)
    if not decode:
        args.sparsepcgc_occupancy_debug_topk_final = int(topk)
        args.sparsepcgc_occupancy_debug_topk_per_layer = max(512, min(int(topk), 8192))
    else:
        args.enable_sparsepcgc_occupancy_debug = False
    return args


def _headroom_fields(file_path: str, base_stats: Mapping[str, object]) -> Dict[str, object]:
    head = _base_headroom_row(file_path=file_path, base_stats=base_stats)
    return {
        "no_op_occupancy_acc_at_0p5": head.get("occupancy_acc_at_0p5", ""),
        "occupied_recall": head.get("occupied_recall", ""),
        "empty_accuracy": head.get("empty_accuracy", ""),
        "p_true_quantiles_json": head.get("p_true_quantiles_json", ""),
        "bit_each_quantiles_json": head.get("bit_each_quantiles_json", ""),
        "top1p_high_bit_symbol_bit_share": head.get("top1p_high_bit_symbol_bit_share", ""),
        "top3p_high_bit_symbol_bit_share": head.get("top3p_high_bit_symbol_bit_share", ""),
        "top5p_high_bit_symbol_bit_share": head.get("top5p_high_bit_symbol_bit_share", ""),
        "low_p_occupied_count": head.get("low_p_occupied_count", ""),
        "high_p_empty_count": head.get("high_p_empty_count", ""),
        "total_estimated_bits": head.get("total_estimated_bits", ""),
        "bits_by_depth_json": head.get("bits_by_depth_json", ""),
    }


def _encode_candidate(*, encoder, coords: torch.Tensor, meta, args) -> Mapping[str, object]:
    return encoder.encode_bits(_coords_to_xyz(coords, meta, args))


def _candidate_raw_percent(
    *,
    candidate_name: str,
    coords: torch.Tensor,
    meta,
    base_stats: Mapping[str, object],
    base_bits: float,
    bit_encoder,
    args,
    budget: float,
    pool: int,
    block_size: int,
    seed: int,
) -> Tuple[float, Dict[str, object]]:
    cand_coords, debug = _candidate_to_coords_u(
        candidate=candidate_name,
        coords=coords,
        base_stats=base_stats,
        budget=float(budget),
        pool=int(pool),
        block_size=int(block_size),
        seed=int(seed),
    )
    stats = _encode_candidate(encoder=bit_encoder, coords=cand_coords, meta=meta, args=args)
    bit = _safe_float(stats.get("bit", stats.get("raw_bit", 0.0)), 0.0)
    raw_percent = (bit - float(base_bits)) / max(float(base_bits), 1e-9) * 100.0
    out = dict(debug)
    out.update({
        f"{candidate_name}_bit_size": bit,
        f"{candidate_name}_actual_edit_ratio": abs(int(cand_coords.shape[0]) - int(coords.shape[0])) / max(int(coords.shape[0]), 1),
        f"{candidate_name}_point_count": int(cand_coords.shape[0]),
    })
    return float(raw_percent), out


def _evaluate_noop_setting(
    *,
    file_path: str,
    coords: torch.Tensor,
    meta,
    base_xyz,
    base_args,
    setting_id: str,
    voxel_size: float,
    pos_quantscale: int,
    cli: argparse.Namespace,
) -> Dict[str, object]:
    t0 = time.time()
    debug_args = _setting_args(
        base_args,
        voxel_size=voxel_size,
        pos_quantscale=pos_quantscale,
        psnr_resolution=int(cli.psnr_resolution),
        decode=False,
        topk=int(cli.pool),
    )
    decode_args = _setting_args(
        base_args,
        voxel_size=voxel_size,
        pos_quantscale=pos_quantscale,
        psnr_resolution=int(cli.psnr_resolution),
        decode=True,
        topk=int(cli.pool),
    )
    decode_args.sparsepcgc_decoded_copy_dir = str(cli.decoded_dir)

    dataset = _dataset_name(file_path)
    sequence = _sequence_name(file_path)
    frame_id = Path(file_path).stem
    row: Dict[str, object] = {
        "stage": "noop",
        "dataset": dataset,
        "sequence": sequence,
        "frame_id": frame_id,
        "setting_id": setting_id,
        "codec_setting_id": f"SparsePCGC_vs{float(voxel_size):g}_pq{int(pos_quantscale)}_psnr{int(cli.psnr_resolution)}",
        "voxel_size": float(voxel_size),
        "pos_quantscale": int(pos_quantscale),
        "psnr_resolution": int(cli.psnr_resolution),
        "status": "ok",
        "error_message": "",
    }

    debug_encoder = build_actual_encoder(debug_args)
    decode_encoder = build_actual_encoder(decode_args)
    try:
        base_stats = debug_encoder.encode_bits(base_xyz)
        baseline_stats = decode_encoder.encode_bits(base_xyz)
        base_bits = _safe_float(baseline_stats.get("bit", base_stats.get("bit", 0.0)), 0.0)
        row.update(_headroom_fields(file_path, base_stats))
        row.update({
            "no_op_actual_bit_size": base_bits,
            "no_op_debug_bit_size": _safe_float(base_stats.get("bit", 0.0), 0.0),
            "no_op_point_count_after_codec_quant": base_stats.get("input_unique_point_count", ""),
            "p_true_q01": _json_quantile(row.get("p_true_quantiles_json"), "q01"),
            "p_true_q05": _json_quantile(row.get("p_true_quantiles_json"), "q05"),
            "p_true_q10": _json_quantile(row.get("p_true_quantiles_json"), "q10"),
            "p_true_median": _json_quantile(row.get("p_true_quantiles_json"), "q50"),
            "bit_each_q90": _json_quantile(row.get("bit_each_quantiles_json"), "q90"),
            "bit_each_q95": _json_quantile(row.get("bit_each_quantiles_json"), "q95"),
            "bit_each_q99": _json_quantile(row.get("bit_each_quantiles_json"), "q99"),
        })
        decoded_path = str(baseline_stats.get("decoded_copy_path", ""))
        if decoded_path:
            decoded_count, match_ratio, lossless = _coord_match_ratio_from_paths(file_path, decoded_path)
            row.update({
                "decoded_gt_path": decoded_path,
                "baseline_decoded_point_count": decoded_count,
                "baseline_decode_coord_match_ratio": match_ratio,
                "lossless": lossless,
            })
            quality = _quality_from_paths(
                file_path,
                decoded_path,
                formal_max_points=int(cli.quality_max_points),
                normal_max_points=int(cli.normal_max_points),
                pc_error_path=str(cli.pc_error_path),
                use_pc_error=bool(cli.use_pc_error),
            )
            row.update({
                "no_op_decoded_D1_PSNR": quality.get("d1_psnr", ""),
                "no_op_decoded_D2_PSNR": quality.get("d2_psnr", ""),
                "no_op_decoded_Chamfer": quality.get("chamfer", ""),
                "quality_eval_mode": quality.get("quality_eval_mode", ""),
            })
        else:
            row.update({"lossless": "", "status": "decode_path_missing"})
    except Exception as exc:
        row.update({"status": "error", "error_message": f"{type(exc).__name__}:{exc}"})
    finally:
        for encoder in (debug_encoder, decode_encoder):
            close = getattr(encoder, "close", None)
            if callable(close):
                close()
    row["elapsed_sec"] = float(time.time() - t0)
    return row


def _evaluate_candidate_setting(
    *,
    file_path: str,
    coords: torch.Tensor,
    meta,
    base_xyz,
    base_args,
    setting_id: str,
    voxel_size: float,
    pos_quantscale: int,
    target_psnr: object,
    cli: argparse.Namespace,
) -> Dict[str, object]:
    t0 = time.time()
    debug_args = _setting_args(
        base_args,
        voxel_size=voxel_size,
        pos_quantscale=pos_quantscale,
        psnr_resolution=int(cli.psnr_resolution),
        decode=False,
        topk=int(cli.pool),
    )
    bit_args = _setting_args(
        base_args,
        voxel_size=voxel_size,
        pos_quantscale=pos_quantscale,
        psnr_resolution=int(cli.psnr_resolution),
        decode=False,
        topk=int(cli.pool),
    )

    dataset = _dataset_name(file_path)
    sequence = _sequence_name(file_path)
    frame_id = Path(file_path).stem
    row: Dict[str, object] = {
        "stage": "candidate",
        "dataset": dataset,
        "sequence": sequence,
        "frame_id": frame_id,
        "setting_id": setting_id,
        "target_psnr": target_psnr,
        "codec_setting_id": f"SparsePCGC_vs{float(voxel_size):g}_pq{int(pos_quantscale)}_psnr{int(cli.psnr_resolution)}",
        "voxel_size": float(voxel_size),
        "pos_quantscale": int(pos_quantscale),
        "psnr_resolution": int(cli.psnr_resolution),
        "status": "ok",
        "error_message": "",
    }

    debug_encoder = build_actual_encoder(debug_args)
    bit_encoder = build_actual_encoder(bit_args)
    try:
        base_stats = debug_encoder.encode_bits(base_xyz)
        base_bits = _safe_float(base_stats.get("bit", 0.0), 0.0)
        row.update(_headroom_fields(file_path, base_stats))
        row.update({
            "no_op_debug_bit_size": base_bits,
            "no_op_actual_bit_size": base_bits,
            "no_op_point_count_after_codec_quant": base_stats.get("input_unique_point_count", ""),
        })
        for budget in _parse_candidate_budgets(cli.candidate_budgets):
            for candidate in [str(x) for x in _parse_csv_text(cli.candidates)]:
                prefix = _candidate_prefix(candidate, budget)
                try:
                    raw_percent, debug = _candidate_raw_percent(
                        candidate_name=candidate,
                        coords=coords,
                        meta=meta,
                        base_stats=base_stats,
                        base_bits=base_bits,
                        bit_encoder=bit_encoder,
                        args=base_args,
                        budget=float(budget),
                        pool=int(cli.pool),
                        block_size=int(cli.block_size),
                        seed=int(cli.seed) + abs(hash((setting_id, budget, candidate))) % 100000,
                    )
                    row[f"{prefix}_raw_percent"] = raw_percent
                    row[f"{prefix}_bit_size"] = debug.get(f"{candidate}_bit_size", "")
                    row[f"{prefix}_actual_edit_ratio"] = debug.get(f"{candidate}_actual_edit_ratio", "")
                    row[f"{prefix}_selected_bit_sum"] = debug.get("selected_bit_sum", "")
                except Exception as exc:
                    row[f"{prefix}_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
    except Exception as exc:
        row.update({"status": "error", "error_message": f"{type(exc).__name__}:{exc}"})
    finally:
        for encoder in (debug_encoder, bit_encoder):
            close = getattr(encoder, "close", None)
            if callable(close):
                close()
    row["elapsed_sec"] = float(time.time() - t0)
    return row


def _target_values(text: str) -> list[float]:
    if ":" in str(text):
        start, end, step = [float(x) for x in str(text).split(":")]
        out: list[float] = []
        value = start
        while value <= end + 1e-9:
            out.append(round(value, 6))
            value += step
        return out
    return _parse_float_list(text)


def _select_representative_settings(
    rows: Sequence[Mapping[str, object]],
    *,
    targets: Sequence[float],
    tolerance: float,
    per_target: int,
) -> list[Dict[str, object]]:
    ok_rows = [r for r in rows if str(r.get("stage", "noop")) == "noop" and str(r.get("status", "ok")) in ("ok", "")]
    selected: list[Dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for target in targets:
        scored: list[tuple[float, float, Mapping[str, object]]] = []
        for row in ok_rows:
            d1 = _safe_num(row.get("no_op_decoded_D1_PSNR"))
            if math.isnan(d1):
                continue
            diff = abs(d1 - float(target))
            if diff <= float(tolerance):
                # 同じtarget内では、近いPSNRかつtop3が大きいsettingを優先する。
                top3 = _safe_num(row.get("top3p_high_bit_symbol_bit_share"), 0.0)
                scored.append((diff, -top3, row))
        scored.sort(key=lambda x: (x[0], x[1]))
        for _, _, row in scored[: int(per_target)]:
            key = (str(row.get("setting_id")), str(target))
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["target_psnr"] = target
            out["target_psnr_abs_error"] = abs(_safe_num(row.get("no_op_decoded_D1_PSNR")) - float(target))
            selected.append(out)
    return selected


def _print_coverage(noop_csv: str, selected_rows: Sequence[Mapping[str, object]], *, targets: Sequence[float]) -> None:
    rows = _read_rows(noop_csv)
    d1s = [_safe_num(r.get("no_op_decoded_D1_PSNR")) for r in rows]
    d1s = [v for v in d1s if not math.isnan(v)]
    print("\n[Phase2V/PSNR-grid] coverage summary")
    print(f"  noop rows: {len(rows)}")
    if d1s:
        print(f"  D1 min/max: {min(d1s):.3f} / {max(d1s):.3f}")
    print(f"  selected target rows: {len(selected_rows)}")

    covered_targets = sorted({float(r.get("target_psnr")) for r in selected_rows if str(r.get("target_psnr", "")) != ""})
    missing = [t for t in targets if float(t) not in set(covered_targets)]
    print(f"  covered targets: {len(covered_targets)} / {len(targets)}")
    if missing:
        print("  missing targets sample:", ",".join(str(x) for x in missing[:40]))

    # 品質帯別に候補を表示する。
    for lo, hi in [(90, 100), (80, 90), (70, 80), (60, 70), (50, 60), (40, 50), (30, 40), (20, 30), (10, 20)]:
        band = []
        for row in rows:
            d1 = _safe_num(row.get("no_op_decoded_D1_PSNR"))
            if lo <= d1 < hi:
                band.append(row)
        if not band:
            print(f"  D1 [{lo},{hi}): no setting")
            continue
        band.sort(key=lambda r: -_safe_num(r.get("top3p_high_bit_symbol_bit_share"), 0.0))
        best = band[0]
        print(
            f"  D1 [{lo},{hi}): count={len(band)} best_top3 setting={best.get('setting_id')} "
            f"D1={_safe_num(best.get('no_op_decoded_D1_PSNR')):.3f} "
            f"acc={best.get('no_op_occupancy_acc_at_0p5')} "
            f"top3={best.get('top3p_high_bit_symbol_bit_share')} bits={best.get('no_op_actual_bit_size')}"
        )


def _print_candidate_summary(candidate_csv: str, *, topk: int) -> None:
    rows = _read_rows(candidate_csv)
    if not rows:
        print("[Phase2V/candidate-summary] no rows")
        return
    raw_keys = sorted({k for row in rows for k in row.keys() if k.endswith("_raw_percent")})
    scored: list[tuple[float, Mapping[str, object], str]] = []
    for row in rows:
        for key in raw_keys:
            raw = _safe_num(row.get(key))
            if not math.isnan(raw):
                scored.append((raw, row, key))
    scored.sort(key=lambda x: x[0])
    print("\n[Phase2V/candidate-summary] top raw-improvement selected settings")
    for raw, row, key in scored[: int(topk)]:
        print(
            f"  raw={raw:+.4f}% target={row.get('target_psnr')} setting={row.get('setting_id')} {key} "
            f"acc={row.get('no_op_occupancy_acc_at_0p5')} top3={row.get('top3p_high_bit_symbol_bit_share')} "
            f"bits={row.get('no_op_actual_bit_size')}"
        )


def run_phase2v(cli: argparse.Namespace) -> int:
    base_args = _prepare_args(cli)
    base_args.sparsepcgc_worker_gpu_stats = False
    base_args.enable_sparsepcgc_occupancy_debug = True
    Path(cli.decoded_dir).mkdir(parents=True, exist_ok=True)

    inventory_rows = _scan_setting_inventory()
    _write_csv(cli.inventory_csv, inventory_rows)
    print(f"[Phase2V] setting inventory rows: {len(inventory_rows)} -> {cli.inventory_csv}")

    file_path = str(cli.file)
    xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
    coords, meta = _unique_coords(xyz, base_args)
    base_xyz = _coords_to_xyz(coords, meta, base_args)

    settings = _parse_settings(cli.settings)
    if bool(cli.auto_settings):
        settings.extend(_generate_settings(
            voxel_values=_parse_float_list(cli.voxel_size_values),
            posq_values=_parse_int_list(cli.pos_quantscale_values),
            full_pair_grid=bool(cli.full_pair_grid),
            include_default=True,
        ))
    settings = _dedupe_settings(settings)
    if int(cli.max_settings) > 0:
        settings = settings[: int(cli.max_settings)]

    print(f"[Phase2V] noop settings to evaluate: {len(settings)}")
    noop_rows = _read_rows(cli.output_csv) if bool(cli.append_output) else []
    existing_noop = {
        (str(r.get("setting_id")), str(r.get("frame_id")))
        for r in noop_rows
        if str(r.get("stage", "noop")) == "noop"
    }
    frame_id = Path(file_path).stem

    # Stage 1: no-op PSNR grid探索。
    for idx, (setting_id, voxel, posq) in enumerate(settings):
        key = (str(setting_id), str(frame_id))
        if key in existing_noop:
            continue
        print(f"[Phase2V/noop] {idx+1}/{len(settings)} setting={setting_id} voxel={voxel:g} posq={posq}", flush=True)
        row = _evaluate_noop_setting(
            file_path=file_path,
            coords=coords,
            meta=meta,
            base_xyz=base_xyz,
            base_args=base_args,
            setting_id=setting_id,
            voxel_size=voxel,
            pos_quantscale=posq,
            cli=cli,
        )
        noop_rows.append(row)
        _write_csv(cli.output_csv, noop_rows)
        print(
            f"[Phase2V/noop] done setting={setting_id} D1={row.get('no_op_decoded_D1_PSNR')} "
            f"acc={row.get('no_op_occupancy_acc_at_0p5')} top3={row.get('top3p_high_bit_symbol_bit_share')} "
            f"bits={row.get('no_op_actual_bit_size')} elapsed={row.get('elapsed_sec')}",
            flush=True,
        )

    # Stage 2: target PSNRごとの代表settingを選ぶ。
    targets = _target_values(cli.psnr_targets)
    noop_rows = _read_rows(cli.output_csv)
    selected = _select_representative_settings(
        noop_rows,
        targets=targets,
        tolerance=float(cli.psnr_tolerance),
        per_target=int(cli.per_target),
    )
    _write_csv(cli.selected_settings_csv, selected)
    _print_coverage(cli.output_csv, selected, targets=targets)

    if bool(cli.no_candidate_stage):
        print("[Phase2V] candidate stage skipped by --no-candidate-stage")
        return 0

    # Stage 3: 各PSNR帯域代表settingで候補加工を評価。
    candidate_rows = _read_rows(cli.candidate_csv) if bool(cli.append_output) else []
    existing_candidate = {
        (str(r.get("setting_id")), str(r.get("target_psnr")), str(r.get("frame_id")))
        for r in candidate_rows
        if str(r.get("stage", "")) == "candidate"
    }
    if int(cli.max_selected_candidates) > 0:
        selected = selected[: int(cli.max_selected_candidates)]
    print(f"[Phase2V/candidate] selected settings to evaluate: {len(selected)}")
    for idx, row0 in enumerate(selected):
        setting_id = str(row0.get("setting_id"))
        target = row0.get("target_psnr")
        key = (setting_id, str(target), str(frame_id))
        if key in existing_candidate:
            continue
        voxel = _safe_num(row0.get("voxel_size"))
        posq = int(_safe_num(row0.get("pos_quantscale")))
        print(f"[Phase2V/candidate] {idx+1}/{len(selected)} target={target} setting={setting_id}", flush=True)
        crow = _evaluate_candidate_setting(
            file_path=file_path,
            coords=coords,
            meta=meta,
            base_xyz=base_xyz,
            base_args=base_args,
            setting_id=setting_id,
            voxel_size=voxel,
            pos_quantscale=posq,
            target_psnr=target,
            cli=cli,
        )
        # no-op側のPSNR情報も候補行にコピーして、後処理を簡単にする。
        for name in [
            "no_op_decoded_D1_PSNR", "no_op_decoded_D2_PSNR", "no_op_decoded_Chamfer",
            "target_psnr_abs_error", "lossless", "baseline_decode_coord_match_ratio",
        ]:
            if name in row0:
                crow[name] = row0.get(name)
        candidate_rows.append(crow)
        _write_csv(cli.candidate_csv, candidate_rows)
    _print_candidate_summary(cli.candidate_csv, topk=int(cli.summary_topk))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2V PSNR-grid SparsePCGC headroom search")
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--settings", default="default:1.0:1", help="Comma list of id:voxel_size:pos_quantscale")
    parser.add_argument("--auto-settings", action="store_true", default=True)
    parser.add_argument("--no-auto-settings", dest="auto_settings", action="store_false")
    parser.add_argument("--voxel-size-values", default=DEFAULT_VOXEL_VALUES)
    parser.add_argument("--pos-quantscale-values", default=DEFAULT_POSQ_VALUES)
    parser.add_argument("--full-pair-grid", action="store_true", default=True)
    parser.add_argument("--no-full-pair-grid", dest="full_pair_grid", action="store_false")
    parser.add_argument("--max-settings", type=int, default=0)
    parser.add_argument("--psnr-targets", default="10:100:1", help="start:end:step or comma list")
    parser.add_argument("--psnr-tolerance", type=float, default=0.75)
    parser.add_argument("--per-target", type=int, default=2)
    parser.add_argument("--max-selected-candidates", type=int, default=0)
    parser.add_argument("--no-candidate-stage", action="store_true", default=False)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--candidate-budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--pool", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--psnr-resolution", type=int, default=1023)
    parser.add_argument("--quality-max-points", type=int, default=1500)
    parser.add_argument("--normal-max-points", type=int, default=1500)
    parser.add_argument("--pc-error-path", default="")
    parser.add_argument("--use-pc-error", action="store_true", default=False)
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2v_psnr_grid_decoded")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--selected-settings-csv", default="/data/maejima/log/PHASE2V_psnr_grid_selected_settings.csv")
    parser.add_argument("--candidate-csv", default=DEFAULT_SELECTED_OUTPUT)
    parser.add_argument("--inventory-csv", default=DEFAULT_INVENTORY)
    parser.add_argument("--summary-topk", type=int, default=20)
    parser.add_argument("--append-output", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2v(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
