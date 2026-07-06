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


DEFAULT_SETTINGS = "default:1.0:1,vox2:2.0:1,posq2:1.0:2"
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

            for candidate in ("high_bit_raw_prune", "low_prob_snap_to_existing"):
                try:
                    raw_percent, debug = _candidate_raw_percent(
                        candidate_name=candidate,
                        coords=coords,
                        meta=meta,
                        base_stats=base_stats,
                        base_bits=base_bits,
                        bit_encoder=bit_encoder,
                        args=base_args,
                        budget=float(cli.budget),
                        pool=int(cli.pool),
                        block_size=int(cli.block_size),
                        seed=int(cli.seed) + int(setting_index),
                    )
                    prefix = "high_bit_raw_1p" if candidate == "high_bit_raw_prune" else "low_prob_snap_to_existing_1p"
                    row[f"{prefix}_raw_percent"] = raw_percent
                    row[f"{prefix}_bit_size"] = debug.get(f"{candidate}_bit_size", "")
                    row[f"{prefix}_actual_edit_ratio"] = debug.get(f"{candidate}_actual_edit_ratio", "")
                    row[f"{prefix}_selected_bit_sum"] = debug.get("selected_bit_sum", "")
                except Exception as exc:
                    row[f"{candidate}_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
            row["elapsed_sec"] = float(time.time() - t0)
        finally:
            for encoder in (debug_encoder, decode_encoder, bit_encoder):
                close = getattr(encoder, "close", None)
                if callable(close):
                    close()
        rows.append(row)
        _write_csv(cli.output_csv, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2V minimal SparsePCGC setting headroom sweep")
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--settings", default=DEFAULT_SETTINGS, help="Comma list of id:voxel_size:pos_quantscale")
    parser.add_argument("--budget", type=float, default=0.010)
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
    parser.add_argument("--append-output", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_phase2v(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
