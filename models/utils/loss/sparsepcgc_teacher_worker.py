from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

_PROTOCOL_OUT = None


def _csv_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _debug(message: str) -> None:
    print(f"[SparsePCGCWorker] {message}", file=sys.stderr, flush=True)

def _collect_cuda_stats(prefix: str = "sparsepcgc_worker") -> dict[str, Any]:
    """
    SparsePCGC workerプロセス側のCUDA使用量を数値dictで返す。
    encoder_multiple.py側のstatsが返らない場合でも、worker wrapper側で最低限の実測値を取る。
    """
    out: dict[str, Any] = {
        f"{prefix}_cuda_available": False,
        f"{prefix}_cuda_device": "",
        f"{prefix}_cuda_allocated_mb": 0.0,
        f"{prefix}_cuda_reserved_mb": 0.0,
        f"{prefix}_cuda_max_allocated_mb": 0.0,
        f"{prefix}_cuda_max_reserved_mb": 0.0,
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return out

        device_index = torch.cuda.current_device()
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.synchronize(device)

        out[f"{prefix}_cuda_available"] = True
        out[f"{prefix}_cuda_device"] = str(device)
        out[f"{prefix}_cuda_allocated_mb"] = float(torch.cuda.memory_allocated(device)) / (1024.0 ** 2)
        out[f"{prefix}_cuda_reserved_mb"] = float(torch.cuda.memory_reserved(device)) / (1024.0 ** 2)
        out[f"{prefix}_cuda_max_allocated_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        out[f"{prefix}_cuda_max_reserved_mb"] = float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2)
    except Exception as exc:
        out[f"{prefix}_cuda_stats_error"] = str(exc)

    return out

def _setup_protocol_stdout() -> None:
    global _PROTOCOL_OUT
    if _PROTOCOL_OUT is not None:
        return
    try:
        protocol_fd = os.dup(sys.stdout.fileno())
        _PROTOCOL_OUT = os.fdopen(protocol_fd, "w", buffering=1)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except Exception:
        _PROTOCOL_OUT = sys.stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent SparsePCGC teacher worker for myNet_new.")
    parser.add_argument("--sparsepcgc-root", required=True)
    parser.add_argument("--mode", default="dense_lossless")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ckptdir", required=True)
    parser.add_argument("--ckptdir-sr", required=True)
    parser.add_argument("--ckptdir-ae", required=True)
    parser.add_argument("--ckptdir-low", required=True)
    parser.add_argument("--ckptdir-high", required=True)
    parser.add_argument("--ckptdir-offset", required=True)
    parser.add_argument("--offset", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--pos-quantscale", type=int, default=1)
    parser.add_argument("--psnr-resolution", type=int, default=1023)
    parser.add_argument("--test-d2", action="store_true")
    parser.add_argument("--dense-scale-ae-list", default="1,0,1,0,1,0")
    parser.add_argument("--dense-scale-sr-list", default="0,1,1,2,2,3")
    parser.add_argument("--pos-quantscale-list", default="4")
    parser.add_argument("--scale-m", type=int, default=8)
    parser.add_argument("--scale-ae", type=int, default=0)
    parser.add_argument("--scale-sr", type=int, default=2)
    parser.add_argument(
        "--inner-psnr",
        action="store_true",
        help="SparsePCGC内部でdecode/PSNRを計算する。学習時のbit教師では既定False",
    )
    parser.add_argument("--decode", action="store_true")
    parser.add_argument(
        "--gpu-stats",
        action="store_true",
        help="encode_one前後のCUDA/GPU使用量をresultへ含める",
    )
    parser.add_argument(
        "--gpu-stats-print",
        action="store_true",
        help="encode_one前後のCUDA/GPU使用量をstderrへ出す",
    )
    return parser.parse_args()


def _build_encoder(args: argparse.Namespace):
    sparse_root = Path(args.sparsepcgc_root).expanduser().resolve()
    _debug(f"using root={sparse_root}")
    sys.path.insert(0, str(sparse_root))
    sys.path.insert(0, str(sparse_root / "test"))

    _debug("importing encoder_multiple.SparsePCGCEncoder")
    import encoder_multiple

    if str(args.mode).strip().lower() == "dense_lossy":
        # ana_den6はLossyCoderDense.testが返すdecoder-complete logical bitsを
        # rateとして使う。一方encoder_multipleは表示用にfile_sizeを物理bin
        # サイズへ上書きするため、worker境界でlogical値を復元して両者を分離する。
        original_normalize = encoder_multiple._normalize_result

        def normalize_with_den6_logical_rate(
            result,
            input_file,
            output_dir,
            bin_paths,
            dec_paths,
            normalize_args,
            input_stats=None,
        ):
            normalized = original_normalize(
                result,
                input_file,
                output_dir,
                bin_paths,
                dec_paths,
                normalize_args,
                input_stats=input_stats,
            )
            main_bits = float(normalized.get("file_size", 0.0))
            # LossyCoderDense.test adds four decoder-side bits per downscale
            # stage.  Some encoder_multiple revisions expose only the packed
            # main-bin count in result["file_size"], so retain the den6 lower
            # bound even in that revision.
            decoder_side_bits = 4.0 * float(
                max(int(args.scale_sr), 0) + max(int(args.scale_ae), 0)
            )
            logical_bits = max(
                float(result.get("file_size", normalized.get("file_size", 0.0))),
                main_bits + decoder_side_bits,
            )
            normalized["main_bin_bits"] = main_bits
            normalized["decoder_side_stage_bits"] = decoder_side_bits
            normalized["logical_file_size"] = logical_bits
            normalized["side_information_bits"] = max(logical_bits - main_bits, 0.0)
            normalized["file_size"] = logical_bits
            point_count = max(int(normalized.get("point_count", 0)), 1)
            normalized["bpp"] = logical_bits / point_count
            normalized["rate_definition"] = "decoder_complete_logical_bits_from_LossyCoderDense.test"
            return normalized

        encoder_multiple._normalize_result = normalize_with_den6_logical_rate

    SparsePCGCEncoder = encoder_multiple.SparsePCGCEncoder
    _debug("imported SparsePCGCEncoder")

    encoder_args = SimpleNamespace(
        mode=args.mode,
        device=args.device,
        ckptdir=Path(args.ckptdir).expanduser().resolve(),
        ckptdir_sr=Path(args.ckptdir_sr).expanduser().resolve(),
        ckptdir_ae=Path(args.ckptdir_ae).expanduser().resolve(),
        ckptdir_low=Path(args.ckptdir_low).expanduser().resolve(),
        ckptdir_high=Path(args.ckptdir_high).expanduser().resolve(),
        ckptdir_offset=Path(args.ckptdir_offset).expanduser().resolve(),
        offset=bool(args.offset),
        voxel_size=float(args.voxel_size),
        pos_quantscale=int(args.pos_quantscale),
        psnr_resolution=int(args.psnr_resolution),
        test_d2=bool(args.test_d2),
        # encoder_multiple.py互換。bit教師では復号・PSNRを行わない。
        inner_psnr=bool(args.inner_psnr),
        test_psnr=bool(args.inner_psnr),
        decode=bool(args.decode),
        scale_m=int(args.scale_m),
        scale_ae=int(args.scale_ae),
        scale_sr=int(args.scale_sr),
        # encoder_multipleの版差に備えた互換alias。値は上記と同一である。
        scale_AE=int(args.scale_ae),
        scale_SR=int(args.scale_sr),
        dense_scale_ae_list=(
            [int(args.scale_ae)]
            if str(args.mode).strip().lower() == "dense_lossy"
            else _csv_int_list(args.dense_scale_ae_list)
        ),
        dense_scale_sr_list=(
            [int(args.scale_sr)]
            if str(args.mode).strip().lower() == "dense_lossy"
            else _csv_int_list(args.dense_scale_sr_list)
        ),
        pos_quantscale_list=_csv_int_list(args.pos_quantscale_list),
        skip_decode=not bool(args.decode),
    )
    _debug(f"initializing SparsePCGCEncoder mode={encoder_args.mode} device={encoder_args.device}")
    encoder = SparsePCGCEncoder(encoder_args)
    _debug("SparsePCGCEncoder ready")
    return encoder


def _emit(payload: Mapping[str, Any]) -> None:
    stream = _PROTOCOL_OUT if _PROTOCOL_OUT is not None else sys.stdout
    print(json.dumps(_jsonable(payload), sort_keys=True), file=stream, flush=True)


def main() -> int:
    _setup_protocol_stdout()
    args = _parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            encoder = _build_encoder(args)
    except Exception as exc:
        _emit({"status": "init_error", "message": str(exc), "traceback": traceback.format_exc()})
        return 2

    ready_payload = {
        "status": "ready",
        "mode": args.mode,
        "device": args.device,
        "scale_m": int(args.scale_m),
        "scale_ae": int(args.scale_ae),
        "scale_sr": int(args.scale_sr),
        "inner_psnr": bool(args.inner_psnr),
        "decode": bool(args.decode),
    }
    if bool(args.gpu_stats):
        ready_payload.update(_collect_cuda_stats("sparsepcgc_worker_init"))
    _emit(ready_payload)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"status": "error", "message": f"invalid json: {exc}"})
            continue

        request_id = request.get("request_id")
        if request.get("command") == "shutdown":
            _emit({"status": "bye", "request_id": request_id})
            return 0

        try:
            input_file = Path(str(request["input_file"])).expanduser().resolve()
            output_dir = Path(str(request["output_dir"])).expanduser().resolve()
            gpu_before = {}
            if bool(args.gpu_stats):
                gpu_before = _collect_cuda_stats("sparsepcgc_worker_before")

            with contextlib.redirect_stdout(sys.stderr):
                result = encoder.encode_one(
                    input_file,
                    output_dir,
                    occupancy_debug=bool(request.get("occupancy_debug", False)),
                    occupancy_low_prob_threshold=float(request.get("occupancy_low_prob_threshold", 0.1)),
                    exact_occupancy=bool(request.get("exact_occupancy", False)),
                    exact_teacher_mode=str(request.get("exact_teacher_mode", "auto")),
                    exact_teacher_uses_full_context=bool(request.get("exact_teacher_uses_full_context", False)),
                    exact_teacher_fallback_reason=str(request.get("exact_teacher_fallback_reason", "")),
                )

            # The encoder model remains resident for the next request, but its
            # convolution workspaces do not.  Return only those unused blocks
            # before the main process starts backward; values and model state
            # are unchanged and worker startup is still paid only once.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            gpu_after = {}
            if bool(args.gpu_stats):
                gpu_after = _collect_cuda_stats("sparsepcgc_worker_after")

            if not isinstance(result, dict):
                result = {"sparsepcgc_worker_raw_result": result}

            result["sparsepcgc_scale_m"] = int(args.scale_m)
            result["sparsepcgc_scale_ae"] = int(args.scale_ae)
            result["sparsepcgc_scale_sr"] = int(args.scale_sr)
            result["sparsepcgc_mode_effective"] = str(args.mode)
            result.setdefault("sparsepcgc_rate_definition", result.get("rate_definition", "physical_bin_bits"))

            if bool(args.gpu_stats):
                result.update(gpu_before)
                result.update(gpu_after)

                # encoder_multiple.py 側が別名でCUDA statsを返している場合に備え、
                # train側で拾いやすい代表keyへ正規化する。
                result["sparsepcgc_worker_cuda_available"] = bool(
                    result.get("sparsepcgc_worker_after_cuda_available", False)
                )
                result["sparsepcgc_worker_cuda_device"] = str(
                    result.get("sparsepcgc_worker_after_cuda_device", "")
                )
                result["sparsepcgc_worker_cuda_allocated_mb"] = float(
                    result.get("sparsepcgc_worker_after_cuda_allocated_mb", 0.0) or 0.0
                )
                result["sparsepcgc_worker_cuda_reserved_mb"] = float(
                    result.get("sparsepcgc_worker_after_cuda_reserved_mb", 0.0) or 0.0
                )
                result["sparsepcgc_worker_cuda_max_allocated_mb"] = float(
                    result.get("sparsepcgc_worker_after_cuda_max_allocated_mb", 0.0) or 0.0
                )
                result["sparsepcgc_worker_cuda_max_reserved_mb"] = float(
                    result.get("sparsepcgc_worker_after_cuda_max_reserved_mb", 0.0) or 0.0
                )
                result["sparsepcgc_worker_cuda_allocated_delta_mb"] = float(
                    result.get("sparsepcgc_worker_after_cuda_allocated_mb", 0.0) or 0.0
                ) - float(
                    result.get("sparsepcgc_worker_before_cuda_allocated_mb", 0.0) or 0.0
                )
                result["sparsepcgc_worker_cuda_reserved_delta_mb"] = float(
                    result.get("sparsepcgc_worker_after_cuda_reserved_mb", 0.0) or 0.0
                ) - float(
                    result.get("sparsepcgc_worker_before_cuda_reserved_mb", 0.0) or 0.0
                )

                if bool(args.gpu_stats_print):
                    _debug(
                        "GPUStats: "
                        f"device={result.get('sparsepcgc_worker_cuda_device', '')}, "
                        f"allocated={float(result.get('sparsepcgc_worker_cuda_allocated_mb', 0.0)):.2f}MB, "
                        f"reserved={float(result.get('sparsepcgc_worker_cuda_reserved_mb', 0.0)):.2f}MB, "
                        f"max_allocated={float(result.get('sparsepcgc_worker_cuda_max_allocated_mb', 0.0)):.2f}MB, "
                        f"max_reserved={float(result.get('sparsepcgc_worker_cuda_max_reserved_mb', 0.0)):.2f}MB"
                    )

            _emit({"status": "ok", "request_id": request_id, "result": result})
            if bool(request.get("exit_after_response", False)):
                return 0
        except Exception as exc:
            _emit(
                {
                    "status": "error",
                    "request_id": request_id,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
