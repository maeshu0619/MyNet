#!/usr/bin/env python3
"""学習対象PLYごとにana_den6 v2 candidate manifestを事前生成する。

学習中にden6候補生成を毎Step再実行すると時間・メモリが大きいため、
各frameの全順位付きEditCandidate poolを一度だけ生成して再利用する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_name(dataset: str, input_file: Path, scale_m: int) -> str:
    tag = _sha256_file(input_file)[:12]
    safe_stem = input_file.stem.replace(" ", "_")
    return f"{dataset}_{safe_stem}_m{int(scale_m)}_{tag}.json"


def _valid_existing_manifest(path: Path, input_file: Path, scale_m: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        payload.get("schema_version") == "ana_den6_ranked_candidate_manifest_v2"
        and str(payload.get("input_sha256", "")) == _sha256_file(input_file)
        and int(payload.get("scale_m", -1)) == int(scale_m)
        and all(payload.get("ranked_candidate_pools", {}).get(name) for name in ("Add", "Prune", "Adjust"))
    )


def _sparsepcgc_python_command(args: argparse.Namespace) -> list[str]:
    """den6本体をMinkowskiEngineを持つSparsePCGC環境で起動する。"""
    explicit_python = str(args.sparsepcgc_python or "").strip()
    if explicit_python:
        executable = Path(explicit_python).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"SparsePCGC Pythonが存在しない: {executable}")
        return [str(executable)]

    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "--no-capture-output", "-n", str(args.sparsepcgc_env), "python"]

    # condaが無い環境では、呼び出し元Pythonに依存することを明示して継続する。
    return [sys.executable]


def main() -> int:
    parser = argparse.ArgumentParser(description="ana_den6 v2 manifest一括生成")
    parser.add_argument("--data", required=True, choices=("8i", "MVUB", "UVG"))
    parser.add_argument("--input-root", required=True, help="学習対象PLYを含むディレクトリ")
    parser.add_argument("--scale-m", type=int, default=8)
    parser.add_argument(
        "--total-ratio-percent",
        type=float,
        default=-1.0,
        help="負ならdataset/m別den6 anchor値。debug時だけ明示して実行可能な操作率を検証する",
    )
    parser.add_argument(
        "--fallback-total-ratio-percent",
        default="0.20,0.15,0.10,0.05",
        help="target ratioでden6 mixed planが不成立のframeだけに順次試す小さい操作率",
    )
    parser.add_argument("--manifest-dir", default="/data/maejima/log/mynet_den6_manifests")
    parser.add_argument("--output-root", default="/data/maejima/log/mynet_den6_manifest_build")
    parser.add_argument("--pattern", default="*.ply")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--plan-variants",
        type=int,
        default=6,
        help="den6の衝突回避候補順序数。ana_den6.pyの既定値と同じ6",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-actual", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--sparsepcgc-env",
        default="sparsepcgc",
        help="den6/SparsePCGCを実行するconda環境名",
    )
    parser.add_argument(
        "--sparsepcgc-python",
        default="",
        help="den6/SparsePCGCを実行するPython絶対パス。指定時は--sparsepcgc-envより優先する",
    )
    args = parser.parse_args()

    root = Path(args.input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"入力ディレクトリが存在しない: {root}")
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tool = Path(__file__).resolve().with_name("ana_den6_reproduce.py")
    if not tool.is_file():
        raise FileNotFoundError(f"ana_den6_reproduce.pyが存在しない: {tool}")

    iterator = root.rglob(args.pattern) if args.recursive else root.glob(args.pattern)
    files = sorted(path.resolve() for path in iterator if path.is_file())
    if int(args.limit) > 0:
        files = files[: int(args.limit)]
    if not files:
        raise RuntimeError(f"対象PLYが見つからない: root={root}, pattern={args.pattern}")
    python_command = _sparsepcgc_python_command(args)

    failures = []
    for index, input_file in enumerate(files, start=1):
        manifest = manifest_dir / _manifest_name(args.data, input_file, args.scale_m)
        if args.resume and _valid_existing_manifest(manifest, input_file, args.scale_m):
            print(f"[{index}/{len(files)}] resume: {manifest}", flush=True)
            continue
        frame_output = output_root / manifest.stem
        command = [
            *python_command,
            str(tool),
            "--data",
            str(args.data),
            "--scale-m",
            str(int(args.scale_m)),
            "--plan-variants",
            str(int(args.plan_variants)),
            "--input-file",
            str(input_file),
            "--manifest-out",
            str(manifest),
            "--output-root",
            str(frame_output),
        ]
        if float(args.total_ratio_percent) >= 0.0:
            command.extend(("--total-ratio-percent", str(float(args.total_ratio_percent))))
        if str(args.fallback_total_ratio_percent).strip():
            command.extend(("--fallback-total-ratio-percent", str(args.fallback_total_ratio_percent)))
        if args.run_actual:
            command.append("--run-actual")
        if args.cpu:
            command.append("--cpu")
        print(f"[{index}/{len(files)}] build: {input_file}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0 or not _valid_existing_manifest(manifest, input_file, args.scale_m):
            failures.append({
                "input_file": str(input_file),
                "manifest": str(manifest),
                "returncode": int(completed.returncode),
            })
    summary = {
        "dataset": args.data,
        "scale_m": int(args.scale_m),
        "fallback_total_ratio_percent": str(args.fallback_total_ratio_percent),
        "sparsepcgc_command": python_command,
        "input_count": len(files),
        "failure_count": len(failures),
        "failures": failures,
    }
    summary_path = output_root / "manifest_batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
