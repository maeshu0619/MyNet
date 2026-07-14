#!/usr/bin/env python
"""Minimal SparsePCGC Phase 1 worker.

SparsePCGC本体は変更せず、dense_lossyの公式AE/SR pairを1つだけ実行し、
JSONでbaseline/debug結果を返すためのwrapperです。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _build_encoder(req: Mapping[str, Any]):
    sparse_root = Path(str(req["sparsepcgc_root"])).expanduser().resolve()
    sys.path.insert(0, str(sparse_root))
    sys.path.insert(0, str(sparse_root / "test"))
    from encoder_multiple import SparsePCGCEncoder

    args = SimpleNamespace(
        mode="dense_lossy",
        device=str(req.get("device", "auto")),
        ckptdir=Path(str(req["ckptdir"])).expanduser().resolve(),
        ckptdir_sr=Path(str(req["ckptdir_sr"])).expanduser().resolve(),
        ckptdir_ae=Path(str(req["ckptdir_ae"])).expanduser().resolve(),
        ckptdir_low=Path(str(req["ckptdir_low"])).expanduser().resolve(),
        ckptdir_high=Path(str(req["ckptdir_high"])).expanduser().resolve(),
        ckptdir_offset=Path(str(req["ckptdir_offset"])).expanduser().resolve(),
        offset=False,
        voxel_size=float(req.get("voxel_size", 1.0)),
        pos_quantscale=int(req.get("pos_quantscale", 1)),
        psnr_resolution=int(req.get("psnr_resolution", 1023)),
        test_d2=False,
        inner_psnr=False,
        dense_scale_ae_list=[int(req["scale_AE"])],
        dense_scale_sr_list=[int(req["scale_SR"])],
        pos_quantscale_list=[4],
        skip_decode=not bool(req.get("decode", True)),
        sparsepcgc_occupancy_debug_topk_per_layer=int(req.get("topk_per_layer", 4096)),
        sparsepcgc_occupancy_debug_topk_final=int(req.get("topk_final", 4096)),
    )
    return SparsePCGCEncoder(args), args


def _pointcloud_stats(input_file: Path, output_dir: Path, bin_path: Path, dec_path: Path, args: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    from encoder_multiple import _decoded_point_count_summary, _pointcloud_stats as pc_stats

    normalized: dict[str, Any] = dict(result)
    normalized.update(pc_stats(input_file, args.voxel_size, args.pos_quantscale))
    normalized["input_path"] = str(input_file)
    normalized["output_dir"] = str(output_dir)
    normalized["bitstream_path"] = str(bin_path)
    normalized["decoded_path"] = str(dec_path) if dec_path.exists() else ""
    normalized["input_point_count"] = int(normalized["header_point_count"])
    normalized["codec_input_point_count"] = int(normalized["point_count"])
    normalized.update(_decoded_point_count_summary([dec_path] if dec_path.exists() else [], args.voxel_size, args.pos_quantscale))
    normalized["logical_file_size_bits"] = float(result.get("file_size", 0.0))
    normalized["actual_bin_file_bits"] = int(bin_path.stat().st_size * 8) if bin_path.exists() else 0
    normalized["file_size"] = float(result.get("file_size", 0.0))
    normalized["bpp"] = float(result.get("bpp", 0.0))
    return normalized


def _run_decode(req: Mapping[str, Any]) -> Mapping[str, Any]:
    encoder, args = _build_encoder(req)
    input_file = Path(str(req["input_file"])).expanduser().resolve()
    output_dir = Path(str(req["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    bin_path = output_dir / f"{stem}_R0.bin"
    dec_path = output_dir / f"{stem}_R0.ply"
    t0 = time.time()
    result = encoder.coder.test(
        str(input_file),
        str(bin_path),
        str(dec_path),
        voxel_size=args.voxel_size,
        posQuantscale=args.pos_quantscale,
        scale_AE=int(req["scale_AE"]),
        scale_SR=int(req["scale_SR"]),
        psnr_resolution=args.psnr_resolution,
        test_psnr=False,
    )
    normalized = _pointcloud_stats(input_file, output_dir, bin_path, dec_path, args, result)
    normalized["phase1_worker_elapsed"] = time.time() - t0
    normalized["phase1_worker_mode"] = "decode"
    return normalized


def _run_debug(req: Mapping[str, Any]) -> Mapping[str, Any]:
    encoder, args = _build_encoder(req)
    input_file = Path(str(req["input_file"])).expanduser().resolve()
    t0 = time.time()
    x_raw = encoder.coder.basic_coder.load_data(str(input_file), args.voxel_size, args.pos_quantscale)
    x_down, bitstream_ae, num_points_list = encoder.coder.downscale(
        x_raw,
        scale_AE=int(req["scale_AE"]),
        scale_SR=int(req["scale_SR"]),
    )
    if not hasattr(encoder.coder, "model") and hasattr(encoder.coder, "basic_coder"):
        # 既存debug関数はlossless coder互換の self.coder.model を参照する。
        # dense lossyでは同じentropy modelが basic_coder 側にあるため、
        # Phase1 wrapper内だけ参照を補う。本体ファイルは変更しない。
        encoder.coder.model = encoder.coder.basic_coder.model
    stats = encoder._estimate_lossless_occupancy_debug(
        x_down,
        low_prob_threshold=float(req.get("occupancy_low_prob_threshold", 0.1)),
        exact_occupancy=False,
    )
    stats = dict(stats)
    stats.update({
        "phase1_worker_mode": "occupancy_debug_after_dense_downscale",
        "phase1_worker_elapsed": time.time() - t0,
        "debug_input_point_count": int(x_raw.C.shape[0]) if hasattr(x_raw, "C") else "",
        "debug_downscaled_point_count": int(x_down.C.shape[0]) if hasattr(x_down, "C") else "",
        "debug_scale_AE": int(req["scale_AE"]),
        "debug_scale_SR": int(req["scale_SR"]),
        "debug_ae_side_stream_bytes": len(bitstream_ae) if bitstream_ae is not None else 0,
        "debug_num_points_list_len": len(num_points_list) if num_points_list is not None else 0,
    })
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase1 SparsePCGC worker")
    parser.add_argument("--mode", choices=("decode", "debug"), required=True)
    args = parser.parse_args()
    try:
        req = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            result = _run_decode(req) if args.mode == "decode" else _run_debug(req)
        print(json.dumps({"status": "ok", "result": _jsonable(result)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "error", "message": str(exc), "traceback": traceback.format_exc()}, sort_keys=True),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
