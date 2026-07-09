#!/usr/bin/env python
"""Phase2V minimal SparsePCGC setting headroom sweep.

Research-only probe.  It keeps the SparsePCGC model/checkpoints unchanged and
only varies codec input quantization settings that are already threaded through
the actual encoder.  Each setting is evaluated as baseline and proposed under
the same codec setting.
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
from typing import Dict, Mapping, Sequence, Tuple

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


DEFAULT_SETTINGS = "default:1.0:1"
DEFAULT_AUTO_VOXEL_SIZES = "0.75,0.875,1.0,1.125,1.25,1.375,1.5,1.75,2.0"
DEFAULT_AUTO_POS_QUANTSCALES = "1,2,3"
DEFAULT_FILE = "/data/maejima/data/ground/8i/loot/loot_vox10_1000.ply"
DEFAULT_OUTPUT = "/data/maejima/log/PHASE2V_sparsepcgc_setting_sweep_smoke.csv"


def _read_rows(path: str) -> list[Dict[str, object]]:
    if not path or not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_settings(text: str) -> list[Tuple[str, float, int]]:
    settings: list[Tuple[str, float, int]] = []
    for raw in _parse_csv_text(text):
        parts = str(raw).split(":")
        if len(parts) != 3:
            raise ValueError(f"setting must be id:voxel_size:pos_quantscale, got {raw!r}")
        setting_id, voxel_size, pos_q = parts
        voxel = float(voxel_size)
        quant = int(pos_q)
        if voxel <= 0:
            raise ValueError(f"voxel_size must be > 0 in setting {raw!r}")
        if quant <= 0:
            raise ValueError(f"pos_quantscale must be > 0 in setting {raw!r}")
        settings.append((setting_id, voxel, quant))
    return settings


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw in _parse_csv_text(text):
        if str(raw).strip() == "":
            continue
        values.append(float(raw))
    return values


def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for raw in _parse_csv_text(text):
        if str(raw).strip() == "":
            continue
        values.append(int(float(raw)))
    return values


def _dedupe_settings(settings: Sequence[Tuple[str, float, int]]) -> list[Tuple[str, float, int]]:
    """同じvoxel_size/pos_quantscaleの重複を除く。"""
    seen: set[tuple[float, int]] = set()
    out: list[Tuple[str, float, int]] = []
    for setting_id, voxel, quant in settings:
        key = (round(float(voxel), 8), int(quant))
        if key in seen:
            continue
        seen.add(key)
        out.append((str(setting_id), float(voxel), int(quant)))
    return out


def _generate_auto_settings(
    *,
    voxel_sizes: Sequence[float],
    pos_quantscales: Sequence[int],
    include_pair_grid: bool,
) -> list[Tuple[str, float, int]]:
    """default近傍を中心に、単変数sweepと軽い2変数gridを作る。"""
    settings: list[Tuple[str, float, int]] = []
    for voxel in voxel_sizes:
        settings.append((f"vox{float(voxel):g}_pq1", float(voxel), 1))
    for quant in pos_quantscales:
        settings.append((f"vox1_pq{int(quant)}", 1.0, int(quant)))
    if include_pair_grid:
        # 組み合わせ爆発を防ぐため、実用的な近傍だけに絞る。
        pair_voxels = [v for v in voxel_sizes if 0.875 <= float(v) <= 1.5]
        pair_quants = [q for q in pos_quantscales if int(q) in (1, 2)]
        for voxel in pair_voxels:
            for quant in pair_quants:
                settings.append((f"vox{float(voxel):g}_pq{int(quant)}", float(voxel), int(quant)))
    return _dedupe_settings(settings)


def _repo_candidate_paths() -> list[Path]:
    """SparsePCGCとactual encoder周辺の設定確認対象を返す。"""
    return [
        REPO_ROOT.parent / "compress/octree/SparsePCGC/encoder_multiple.py",
        REPO_ROOT / "models/utils/loss/actual_encoder.py",
        REPO_ROOT / "tools/phase2v_sparsepcgc_setting_sweep.py",
    ]


def _scan_setting_inventory() -> list[Dict[str, object]]:
    """設定らしき語を軽くgrepしてCSV化する。研究用の在庫表。"""
    patterns = [
        "voxel", "voxel_size", "pos_quantscale", "pos_quantscale_list",
        "psnr_resolution", "resolution", "depth", "max_depth", "octree",
        "quant", "quantization", "scale", "dense_scale",
        "dense_scale_ae_list", "dense_scale_sr_list", "block", "block_size",
        "cube", "root", "normalize", "checkpoint", "ckpt", "model",
        "entropy", "coder", "bitstream", "lossless", "lossy",
    ]
    rows: list[Dict[str, object]] = []
    regex = re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)
    for path in _repo_candidate_paths():
        if not path.exists():
            rows.append({
                "file": str(path),
                "line": "",
                "setting_name": "",
                "snippet": "file_not_found",
                "kind_guess": "",
                "risk_guess": "unknown",
            })
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            rows.append({
                "file": str(path),
                "line": "",
                "setting_name": "",
                "snippet": f"read_error:{type(exc).__name__}:{exc}",
                "kind_guess": "",
                "risk_guess": "unknown",
            })
            continue
        for idx, line in enumerate(lines, start=1):
            if not regex.search(line):
                continue
            snippet = line.strip()
            name_match = re.search(r"--([A-Za-z0-9_\\-]+)|args\\.([A-Za-z0-9_]+)|sparsepcgc_([A-Za-z0-9_]+)", snippet)
            setting_name = ""
            if name_match:
                setting_name = next((g for g in name_match.groups() if g), "")
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
    if not path:
        return
    _write_csv(path, [dict(r) for r in rows])


def _candidate_prefix(candidate: str, budget: float) -> str:
    """CSV列名に使う安全なprefixを作る。"""
    budget_pct = float(budget) * 100.0
    label = f"{budget_pct:.3f}".replace(".", "p").rstrip("0").rstrip("p")
    return f"{candidate}_b{label}pct"


def _parse_candidate_budgets(text: str) -> list[float]:
    budgets: list[float] = []
    for raw in _parse_csv_text(text):
        value = float(raw)
        if value <= 0:
            raise ValueError(f"budget must be > 0, got {raw!r}")
        # 1.0以上を百分率指定とみなし、1なら1%に変換する。
        if value >= 1.0:
            value = value / 100.0
        budgets.append(float(value))
    return budgets


def _safe_num(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _print_summary(output_csv: str, *, topk: int = 12) -> None:
    rows = _read_rows(output_csv)
    if not rows:
        print("[Phase2V/summary] no rows")
        return
    raw_keys = sorted({k for row in rows for k in row.keys() if k.endswith("_raw_percent")})
    print("\\n[Phase2V/summary] rows:", len(rows))
    print("[Phase2V/summary] raw columns:", ", ".join(raw_keys) if raw_keys else "(none)")
    scored: list[tuple[float, str, str, str, float, float, float]] = []
    for row in rows:
        setting_id = str(row.get("setting_id", ""))
        codec_id = str(row.get("codec_setting_id", ""))
        d1 = _safe_num(row.get("no_op_decoded_D1_PSNR"))
        top3 = _safe_num(row.get("top3p_high_bit_symbol_bit_share"))
        for key in raw_keys:
            raw = _safe_num(row.get(key))
            if math.isnan(raw):
                continue
            # 小さいほどよいraw改善、ただしno-op品質が壊れすぎた設定は後で人間が除外できるようD1も出す。
            scored.append((raw, setting_id, codec_id, key, d1, top3, _safe_num(row.get("no_op_actual_bit_size"))))
    scored.sort(key=lambda x: x[0])
    print("[Phase2V/summary] top raw-improvement candidates:")
    for raw, setting_id, codec_id, key, d1, top3, bits in scored[:topk]:
        print(f"  raw={raw:+.4f}%  setting={setting_id}  {key}  D1={d1:.3f}  top3={top3:.4f}  bits={bits:.0f}  {codec_id}")

    default_d1 = None
    for row in rows:
        if str(row.get("setting_id", "")).startswith("default"):
            default_d1 = _safe_num(row.get("no_op_decoded_D1_PSNR"))
            break
    if default_d1 is not None and not math.isnan(default_d1):
        print("[Phase2V/summary] quality-safe settings within default D1 - 3dB:")
        safe_rows = []
        for row in rows:
            d1 = _safe_num(row.get("no_op_decoded_D1_PSNR"))
            if not math.isnan(d1) and d1 >= default_d1 - 3.0:
                safe_rows.append(row)
        for row in safe_rows[:topk]:
            print(
                f"  setting={row.get('setting_id')} D1={_safe_num(row.get('no_op_decoded_D1_PSNR')):.3f} "
                f"acc={row.get('no_op_occupancy_acc_at_0p5')} top3={row.get('top3p_high_bit_symbol_bit_share')} "
                f"bits={row.get('no_op_actual_bit_size')}"
            )


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


def _json_quantile(value: object, key: str) -> float:
    try:
        parsed = json.loads(str(value or "{}"))
        if isinstance(parsed, Mapping):
            return float(parsed.get(key, float("nan")))
    except Exception:
        pass
    return float("nan")


def _encode_candidate(
    *,
    encoder,
    coords: torch.Tensor,
    meta,
    args,
) -> Mapping[str, object]:
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


def run_phase2v(cli: argparse.Namespace) -> int:
    base_args = _prepare_args(cli)
    base_args.sparsepcgc_worker_gpu_stats = False
    base_args.enable_sparsepcgc_occupancy_debug = True
    Path(cli.decoded_dir).mkdir(parents=True, exist_ok=True)
    rows = _read_rows(cli.output_csv) if bool(cli.append_output) else []

    settings = _parse_settings(cli.settings)
    if bool(cli.auto_settings):
        settings.extend(_generate_auto_settings(
            voxel_sizes=_parse_float_list(cli.voxel_size_values),
            pos_quantscales=_parse_int_list(cli.pos_quantscale_values),
            include_pair_grid=bool(cli.include_pair_grid),
        ))
    settings = _dedupe_settings(settings)
    if int(cli.max_settings) > 0:
        settings = settings[: int(cli.max_settings)]

    inventory_rows = _scan_setting_inventory()
    _write_inventory_csv(cli.inventory_csv, inventory_rows)
    print(f"[Phase2V] setting inventory rows: {len(inventory_rows)} -> {cli.inventory_csv}")
    print(f"[Phase2V] settings to evaluate: {len(settings)}")
    for setting_id, voxel_size, pos_q in settings:
        print(f"  - {setting_id}: voxel_size={voxel_size:g}, pos_quantscale={pos_q}")

    candidate_names = [str(x) for x in _parse_csv_text(cli.candidates)]
    candidate_budgets = _parse_candidate_budgets(cli.candidate_budgets)

    file_path = str(cli.file)
    xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
    coords, meta = _unique_coords(xyz, base_args)
    base_xyz = _coords_to_xyz(coords, meta, base_args)
    dataset = _dataset_name(file_path)
    sequence = _sequence_name(file_path)
    frame_id = Path(file_path).stem

    for setting_index, (setting_id, voxel_size, pos_q) in enumerate(settings):
        if any(
            str(row.get("setting_id")) == str(setting_id)
            and str(row.get("frame_id")) == frame_id
            for row in rows
        ):
            continue
        t0 = time.time()
        row: Dict[str, object] = {
            "dataset": dataset,
            "sequence": sequence,
            "frame_id": frame_id,
            "setting_id": setting_id,
            "codec_setting_id": f"SparsePCGC_vs{voxel_size:g}_pq{pos_q}_psnr{int(cli.psnr_resolution)}",
            "voxel_size": float(voxel_size),
            "pos_quantscale": int(pos_q),
            "psnr_resolution": int(cli.psnr_resolution),
            "dense_scale_ae_list_exists": True,
            "dense_scale_sr_list_exists": True,
            "dense_scale_ae_list": getattr(base_args, "sparsepcgc_dense_scale_ae_list", "1,0,1,0,1,0"),
            "dense_scale_sr_list": getattr(base_args, "sparsepcgc_dense_scale_sr_list", "0,1,1,2,2,3"),
            "budget_ratio": float(cli.budget),
        }
        debug_args = _setting_args(
            base_args,
            voxel_size=voxel_size,
            pos_quantscale=pos_q,
            psnr_resolution=int(cli.psnr_resolution),
            decode=False,
            topk=int(cli.pool),
        )
        decode_args = _setting_args(
            base_args,
            voxel_size=voxel_size,
            pos_quantscale=pos_q,
            psnr_resolution=int(cli.psnr_resolution),
            decode=bool(cli.decode_quality),
            topk=int(cli.pool),
        )
        decode_args.sparsepcgc_decoded_copy_dir = str(cli.decoded_dir)
        bit_args = _setting_args(
            base_args,
            voxel_size=voxel_size,
            pos_quantscale=pos_q,
            psnr_resolution=int(cli.psnr_resolution),
            decode=False,
            topk=int(cli.pool),
        )
        debug_encoder = build_actual_encoder(debug_args)
        decode_encoder = build_actual_encoder(decode_args)
        bit_encoder = build_actual_encoder(bit_args)
        try:
            base_stats = debug_encoder.encode_bits(base_xyz)
            baseline_stats = decode_encoder.encode_bits(base_xyz)
            base_bits = _safe_float(baseline_stats.get("bit", base_stats.get("bit", 0.0)), 0.0)
            row.update(_headroom_fields(file_path, base_stats))
            row.update({
                "no_op_actual_bit_size": base_bits,
                "no_op_debug_bit_size": _safe_float(base_stats.get("bit", 0.0), 0.0),
                "no_op_point_count_after_codec_quant": base_stats.get("input_unique_point_count", ""),
                "p_true_q10": _json_quantile(row.get("p_true_quantiles_json"), "q10"),
                "p_true_median": _json_quantile(row.get("p_true_quantiles_json"), "q50"),
                "bit_each_q90": _json_quantile(row.get("bit_each_quantiles_json"), "q90"),
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
                if bool(cli.decode_quality):
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
                row.update({"lossless": "", "proposed_unavailable_reason": "decode_path_missing"})

            for budget in candidate_budgets:
                for candidate in candidate_names:
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
                            seed=int(cli.seed) + int(setting_index),
                        )
                        prefix = _candidate_prefix(candidate, float(budget))
                        row[f"{prefix}_raw_percent"] = raw_percent
                        row[f"{prefix}_bit_size"] = debug.get(f"{candidate}_bit_size", "")
                        row[f"{prefix}_actual_edit_ratio"] = debug.get(f"{candidate}_actual_edit_ratio", "")
                        row[f"{prefix}_selected_bit_sum"] = debug.get("selected_bit_sum", "")
                    except Exception as exc:
                        prefix = _candidate_prefix(candidate, float(budget))
                        row[f"{prefix}_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
            row["elapsed_sec"] = float(time.time() - t0)
            print(
                f"[Phase2V] done setting={setting_id} bits={row.get('no_op_actual_bit_size')} "
                f"acc={row.get('no_op_occupancy_acc_at_0p5')} D1={row.get('no_op_decoded_D1_PSNR')} "
                f"elapsed={row['elapsed_sec']:.1f}s"
            )
        finally:
            for encoder in (debug_encoder, decode_encoder, bit_encoder):
                close = getattr(encoder, "close", None)
                if callable(close):
                    close()
        rows.append(row)
        _write_csv(cli.output_csv, rows)
    _print_summary(cli.output_csv, topk=int(cli.summary_topk))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2V minimal SparsePCGC setting headroom sweep")
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--settings", default=DEFAULT_SETTINGS, help="Comma list of id:voxel_size:pos_quantscale")
    parser.add_argument("--auto-settings", action="store_true", default=True, help="Generate detailed voxel_size/pos_quantscale sweep")
    parser.add_argument("--no-auto-settings", dest="auto_settings", action="store_false")
    parser.add_argument("--voxel-size-values", default=DEFAULT_AUTO_VOXEL_SIZES)
    parser.add_argument("--pos-quantscale-values", default=DEFAULT_AUTO_POS_QUANTSCALES)
    parser.add_argument("--include-pair-grid", action="store_true", default=True)
    parser.add_argument("--no-include-pair-grid", dest="include_pair_grid", action="store_false")
    parser.add_argument("--max-settings", type=int, default=0, help="0 means no limit")
    parser.add_argument("--candidates", default="high_bit_raw_prune,low_prob_snap_to_existing")
    parser.add_argument("--candidate-budgets", default="0.01", help="Comma list. Use 0.01 or 1 for 1 percent")
    parser.add_argument("--budget", type=float, default=0.010, help="Backward compatible; use --candidate-budgets for new runs")
    parser.add_argument("--pool", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--psnr-resolution", type=int, default=1023)
    parser.add_argument("--decode-quality", action="store_true", default=True)
    parser.add_argument("--no-decode-quality", dest="decode_quality", action="store_false")
    parser.add_argument("--quality-max-points", type=int, default=1500)
    parser.add_argument("--normal-max-points", type=int, default=1500)
    parser.add_argument("--pc-error-path", default="")
    parser.add_argument("--use-pc-error", action="store_true", default=False)
    parser.add_argument("--decoded-dir", default="/data/maejima/log/phase2v_decoded")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT)
    parser.add_argument("--inventory-csv", default="/data/maejima/log/PHASE2V_sparsepcgc_setting_inventory.csv")
    parser.add_argument("--summary-topk", type=int, default=12)
    parser.add_argument("--append-output", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2v(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
