#!/usr/bin/env python3
"""保存済みana_den6 actual PLYとmyNet workerのcodec bitを照合する。"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MYNET_ROOT = REPO_ROOT / "myNet"
SPARSEPCGC_ROOT = REPO_ROOT / "compress" / "octree" / "SparsePCGC"

REFERENCE_SHARES = {
    ("8I", 8): (0.40, 0.40, 0.20),
    ("MVUB", 8): (0.50, 0.40, 0.10),
    ("UVG", 7): (0.40, 0.50, 0.10),
}


def _analysis_root(dataset: str) -> Path:
    suffix = {"8I": "8i", "MVUB": "MVUB", "UVG": "UVG"}[dataset]
    return Path(f"/data/maejima/log/SparsePCGC_dense_mixed_edit_analysis_den6_{suffix}")


def _reference_row(dataset: str, scale_m: int, analysis_root: Path) -> dict[str, Any]:
    shares = REFERENCE_SHARES[(dataset, scale_m)]
    rows = json.loads((analysis_root / "state" / "run_rows.json").read_text(encoding="utf-8")).get("actual_rows", [])
    matches = [
        row
        for row in rows
        if str(row.get("dataset", "")).upper() == dataset
        and int(float(row.get("scale_m", -1))) == scale_m
        and math.isclose(float(row.get("total_ratio_percent", -1.0)), 0.25, abs_tol=1e-12)
        and all(
            math.isclose(float(row.get(key, -1.0)), expected, abs_tol=1e-12)
            for key, expected in zip(("add_share", "prune_share", "adjust_share"), shares)
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"den6参照actual行を一意に特定できない: {len(matches)}")
    return dict(matches[0])


def _worker_command(python: str, scale_m: int, scale_sr: int, psnr_resolution: int, device: str) -> list[str]:
    ckpt = SPARSEPCGC_ROOT / "ckpts"
    return [
        python,
        "models/utils/loss/sparsepcgc_teacher_worker.py",
        "--sparsepcgc-root", str(SPARSEPCGC_ROOT),
        "--mode", "dense_lossy",
        "--device", device,
        "--ckptdir", str(ckpt / "dense" / "epoch_last.pth"),
        "--ckptdir-sr", str(ckpt / "dense_1stage" / "epoch_last.pth"),
        "--ckptdir-ae", str(ckpt / "dense_slne" / "epoch_last.pth"),
        "--ckptdir-low", str(ckpt / "sparse_low" / "epoch_last.pth"),
        "--ckptdir-high", str(ckpt / "sparse_high" / "epoch_last.pth"),
        "--ckptdir-offset", str(ckpt / "sparse_offset" / "epoch_last.pth"),
        "--voxel-size", "1",
        "--pos-quantscale", "1",
        "--psnr-resolution", str(psnr_resolution),
        "--scale-m", str(scale_m),
        "--scale-ae", "0",
        "--scale-sr", str(scale_sr),
    ]


def _request(proc: subprocess.Popen[str], request_id: int, input_file: str, output_dir: Path) -> dict[str, Any]:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps({"request_id": request_id, "input_file": input_file, "output_dir": str(output_dir)}) + "\n")
    proc.stdin.flush()
    response = json.loads(proc.stdout.readline())
    if response.get("status") != "ok":
        raise RuntimeError(json.dumps(response, ensure_ascii=False))
    return dict(response["result"])


def main() -> int:
    parser = argparse.ArgumentParser(description="den6 actual PLYとmyNet SparsePCGC workerのlogical bit照合")
    parser.add_argument("--data", required=True, choices=("8i", "MVUB", "UVG"))
    parser.add_argument("--scale-m", type=int, required=True)
    parser.add_argument("--analysis-root", default="")
    parser.add_argument("--python", default="/home/maejima/miniconda3/envs/sparsepcgc/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="/tmp/mynet_den6_worker_codec_verify")
    cli = parser.parse_args()

    dataset = str(cli.data).upper()
    key = (dataset, int(cli.scale_m))
    if key not in REFERENCE_SHARES:
        raise ValueError(f"den6 0.25%参照profileが無い: {key}")
    row = _reference_row(dataset, int(cli.scale_m), Path(cli.analysis_root) if cli.analysis_root else _analysis_root(dataset))
    native_depth = 9 if dataset == "UVG" else 10
    scale_sr = native_depth - int(cli.scale_m)
    psnr_resolution = 511 if dataset == "UVG" else 1023
    output_dir = Path(cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        _worker_command(str(cli.python), int(cli.scale_m), scale_sr, psnr_resolution, str(cli.device)),
        cwd=str(MYNET_ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        assert proc.stdout is not None
        ready = json.loads(proc.stdout.readline())
        if ready.get("status") != "ready":
            raise RuntimeError(json.dumps(ready, ensure_ascii=False))
        baseline = _request(proc, 1, str(row["input_file"]), output_dir)
        edited = _request(proc, 2, str(row["edited_ply"]), output_dir)
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"request_id": 3, "command": "shutdown"}) + "\n")
        proc.stdin.flush()
        proc.stdout.readline()
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    baseline_bits = float(baseline["file_size"])
    edited_bits = float(edited["file_size"])
    saved_percent = 100.0 * (baseline_bits - edited_bits) / baseline_bits
    report = {
        "dataset": dataset,
        "setting_id": row["setting_id"],
        "reference_baseline_bits": float(row["baseline_decoder_complete_bits"]),
        "reference_edited_bits": float(row["edited_decoder_complete_bits"]),
        "reference_saved_percent": float(row["actual_saved_percent"]),
        "worker_baseline_bits": baseline_bits,
        "worker_edited_bits": edited_bits,
        "worker_saved_percent": saved_percent,
        "worker_rate_definition": baseline.get("sparsepcgc_rate_definition", ""),
    }
    report["passed"] = bool(
        math.isclose(report["reference_baseline_bits"], baseline_bits, abs_tol=1e-9)
        and math.isclose(report["reference_edited_bits"], edited_bits, abs_tol=1e-9)
        and math.isclose(report["reference_saved_percent"], saved_percent, abs_tol=1e-12)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
