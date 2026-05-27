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
    parser.add_argument("--decode", action="store_true")
    return parser.parse_args()


def _build_encoder(args: argparse.Namespace):
    sparse_root = Path(args.sparsepcgc_root).expanduser().resolve()
    _debug(f"using root={sparse_root}")
    sys.path.insert(0, str(sparse_root))
    sys.path.insert(0, str(sparse_root / "test"))

    _debug("importing encoder_multiple.SparsePCGCEncoder")
    from encoder_multiple import SparsePCGCEncoder
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
        dense_scale_ae_list=_csv_int_list(args.dense_scale_ae_list),
        dense_scale_sr_list=_csv_int_list(args.dense_scale_sr_list),
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

    _emit({"status": "ready", "mode": args.mode, "device": args.device})

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
            _emit({"status": "ok", "request_id": request_id, "result": result})
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
