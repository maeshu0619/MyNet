#!/usr/bin/env python3
"""ana_den6のhard voxel編集をmyNet導入前に再現・照合するツール。

myNetのproxy特徴を使って候補を作り直さず、ana_den6/ana_den5が公開済みの
候補生成、順位、衝突回避、actual plan適用を直接再利用する。既存のden6分析
成果物に保存された編集PLYと集合比較するため、学習時のWhere/Amount/Action
統合前にcanonical voxel座標系の不一致を検出できる。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SPARSEPCGC_ROOT = REPO_ROOT / "compress" / "octree" / "SparsePCGC"

# Excel実測行と同じ0.25% mixed patternを明示し、screening順位の偶然に依存しない。
REFERENCE_SHARES = {
    ("8I", 8): (0.40, 0.40, 0.20),
    # 保存済みden6 actual行を正とする。依頼文の参考値ではなく、state/Excelの
    # Add=50%%, Prune=40%%, Adjust=10%% の実測mixed planを再現する。
    ("MVUB", 8): (0.50, 0.40, 0.10),
    ("UVG", 7): (0.40, 0.50, 0.10),
}


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"moduleをimportできない: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _coord_set(coords: Any) -> set[tuple[int, int, int]]:
    values = np.asarray(coords)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"座標配列shapeが不正: {values.shape}")
    return {tuple(int(item) for item in row[-3:]) for row in np.rint(values).astype(np.int64)}


def _coord_hash(coords: Any) -> str:
    values = np.asarray(sorted(_coord_set(coords)), dtype=np.int64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _load_reference(
    root: Path,
    dataset: str,
    scale_m: int,
    input_file: Path | None = None,
) -> Mapping[str, Any]:
    state_path = root / "state" / "run_rows.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    shares = REFERENCE_SHARES[(dataset, scale_m)]
    rows = state.get("actual_rows") or []
    matches = [
        row
        for row in rows
        if str(row.get("dataset", "")).upper() == dataset
        and int(float(row.get("scale_m", -1))) == scale_m
        and (input_file is None or Path(str(row.get("input_file", ""))).resolve() == input_file)
        and math.isclose(float(row.get("total_ratio_percent", -1.0)), 0.25, abs_tol=1e-12)
        and all(
            math.isclose(float(row.get(key, -1.0)), expected, abs_tol=1e-12)
            for key, expected in zip(("add_share", "prune_share", "adjust_share"), shares)
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "参照actual行を一意に特定できない: "
            + json.dumps({"root": str(root), "dataset": dataset, "m": scale_m, "matches": len(matches)})
        )
    return matches[0]


def _default_analysis_root(dataset: str) -> Path:
    suffix = {"8I": "8i", "MVUB": "MVUB", "UVG": "UVG"}[dataset]
    return Path(f"/data/maejima/log/SparsePCGC_dense_mixed_edit_analysis_den6_{suffix}")


def _mark(root: Path, phase: str, **details: Any) -> None:
    """外部codecがkillされた場合にも最後の到達地点を残す。"""
    rss_kib = 0
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
                break
    except Exception:
        pass
    payload = {"phase": phase, "rss_kib": rss_kib, **details}
    path = root / "last_phase.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_den6_args(den6: ModuleType, den5: ModuleType, cli: argparse.Namespace) -> argparse.Namespace:
    argv = ["--data", cli.data, "--m-values", str(cli.scale_m), "--output-root", str(cli.output_root)]
    if cli.input_file:
        argv.extend(("--dataset-input", str(Path(cli.input_file).expanduser().resolve())))
    if cli.cpu:
        argv.append("--cpu")
    args = den6.build_parser(den5).parse_args(argv)
    return den6._prepare_args(den5, args)


def _parse_shares(text: str, dataset: str, scale_m: int) -> tuple[float, float, float]:
    if not str(text).strip():
        return REFERENCE_SHARES[(dataset, scale_m)]
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("--operation-sharesは正のAdd,Prune,Adjustの3値で指定する")
    total = sum(values)
    return tuple(value / total for value in values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_manifest_row(candidate: Any) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.candidate_id),
        "operation": str(candidate.operation),
        "remove_coords": [list(map(int, coord)) for coord in getattr(candidate, "remove_coords", ())],
        "add_coords": [list(map(int, coord)) for coord in getattr(candidate, "add_coords", ())],
        "candidate_edit_key": hashlib.sha256(
            json.dumps(
                {
                    "remove": sorted(list(map(int, coord)) for coord in getattr(candidate, "remove_coords", ())),
                    "add": sorted(list(map(int, coord)) for coord in getattr(candidate, "add_coords", ())),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _write_manifest(
    path: Path,
    *,
    dataset: str,
    input_file: Path,
    setting: Any,
    total_ratio_percent: float,
    shares: tuple[float, float, float],
    heuristics: Mapping[str, str],
    requested_counts: Mapping[str, int],
    plan: Sequence[Any],
    plan_meta: Mapping[str, Any],
    edited_coords: Any,
    reference: Mapping[str, Any] | None,
) -> None:
    operation_counts = {
        operation: sum(str(item.operation) == operation for item in plan)
        for operation in ("Add", "Prune", "Adjust")
    }
    payload = {
        "schema_version": "ana_den6_mixed_plan_manifest_v1",
        "den6_script": str((SPARSEPCGC_ROOT / "ana_den6.py").resolve()),
        "den6_sha256": _sha256_file(SPARSEPCGC_ROOT / "ana_den6.py"),
        "dataset": dataset,
        "input_file": str(input_file),
        "input_sha256": _sha256_file(input_file),
        "setting_id": str(setting.setting_id),
        "scale_m": int(setting.scale_m),
        "scale_ae": int(setting.scale_ae),
        "scale_sr": int(setting.scale_sr),
        "total_ratio_percent": float(total_ratio_percent),
        "operation_shares": {"Add": shares[0], "Prune": shares[1], "Adjust": shares[2]},
        "operation_heuristics": dict(heuristics),
        "requested_operation_counts": {name: int(value) for name, value in requested_counts.items()},
        "selected_operation_counts": operation_counts,
        "plan_metadata": dict(plan_meta),
        "selected_candidates": [_candidate_manifest_row(item) for item in plan],
        "final_voxel_hash": _coord_hash(edited_coords),
        "reference_actual": {
            key: reference.get(key)
            for key in (
                "baseline_decoder_complete_bits",
                "edited_decoder_complete_bits",
                "actual_saved_percent",
                "D1_loss_db",
                "D2_loss_db",
            )
        } if reference is not None else {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="ana_den6 mixed hard voxel再現・照合")
    parser.add_argument("--data", required=True, choices=("8i", "MVUB", "UVG"))
    parser.add_argument("--scale-m", type=int, default=8)
    parser.add_argument("--analysis-root", default="")
    parser.add_argument("--output-root", default="/tmp/mynet_ana_den6_reproduce")
    parser.add_argument("--input-file", default="", help="den6計算を行う特定PLY。空なら保存済み参照行の入力を使用する")
    parser.add_argument("--total-ratio-percent", type=float, default=0.25)
    parser.add_argument("--operation-shares", default="", help="Add,Prune,Adjust。空ならden6参照profile")
    parser.add_argument("--manifest-out", default="", help="同じden6 planをmyNetが読むJSON manifest出力先")
    parser.add_argument("--run-actual", action="store_true", help="再構築したplanをactual SparsePCGCで再評価する")
    parser.add_argument("--cpu", action="store_true")
    cli = parser.parse_args()

    dataset = str(cli.data).upper()
    key = (dataset, int(cli.scale_m))
    if key not in REFERENCE_SHARES:
        raise ValueError(f"0.25%の固定参照profileが無い: {key}")
    analysis_root = Path(cli.analysis_root).expanduser().resolve() if cli.analysis_root else _default_analysis_root(dataset)
    requested_input = Path(cli.input_file).expanduser().resolve() if cli.input_file else None
    reference = _load_reference(analysis_root, *key, requested_input) if requested_input is None else None
    output_root = Path(cli.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _mark(output_root, "reference_loaded", dataset=dataset, scale_m=int(cli.scale_m), input_file=str(requested_input or ""))

    den6 = _load_module(SPARSEPCGC_ROOT / "ana_den6.py", "mynet_ana_den6_reproduce_den6")
    den5, _ = den6._load_den5()
    args = _build_den6_args(den6, den5, cli)
    _mark(output_root, "den6_modules_loaded")
    engine_path, _ = den5._resolve_base_script(args)
    engine = den5._load_module(engine_path)
    codec_base_path = den5._resolve_codec_base_script(engine_path, args)
    codec_base = engine._load_base_module(codec_base_path) if hasattr(engine, "_load_base_module") else den5._load_module(codec_base_path)
    den5._require_base_api(engine, codec_base)
    input_items = list(engine._find_input_items_strict(codec_base, args))
    settings = list(engine._build_m_settings(codec_base, args))
    input_file = requested_input or Path(str(reference["input_file"])).resolve()
    input_item = next((item for item in input_items if Path(str(item.path)).resolve() == input_file), None)
    setting = next(
        (
            item
            for item in settings
            if reference is None or str(item.setting_id) == str(reference["setting_id"])
        ),
        None,
    )
    if input_item is None or setting is None:
        raise RuntimeError("den6参照行と現在のinput/settingを対応付けられない")

    _mark(output_root, "input_and_setting_resolved", input_file=str(input_file), setting_id=str(setting.setting_id))
    codec_base._resolve_checkpoints(args)
    import torch

    device = torch.device("cpu" if cli.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    _mark(output_root, "before_dense_coder", device=str(device))
    coder = codec_base._load_dense_coder(args, device)
    _mark(output_root, "dense_coder_loaded")
    try:
        raw_count, unique_count, _, _ = den5._count_raw_and_unique_voxels(codec_base, input_file)
        _mark(output_root, "before_baseline_codec", unique_voxel_count=int(unique_count))
        baseline_codec = engine._baseline_codec(codec_base, coder, input_file, setting, args, output_root)
        codec_base._cleanup()
        _mark(output_root, "before_baseline_probe")
        baseline_probe = engine._probe_state(codec_base, coder, input_file, setting, args, tag="mynet_den6_reproduce")
        _mark(output_root, "baseline_probe_loaded")

        shares = _parse_shares(cli.operation_shares, *key)
        total_ratio_percent = float(reference["total_ratio_percent"]) if reference else float(cli.total_ratio_percent)
        requested_counts = den6._allocate_counts(
            max(3, int(math.ceil(unique_count * total_ratio_percent / 100.0))),
            dict(zip(den6.OPERATIONS, shares)),
        )
        maximum_counts = dict(requested_counts)
        priors = den6._deep_copy_priors()
        pools, heuristics, _ = den6._prepare_operation_pools(
            den5, engine, baseline_probe, baseline_codec, setting, args, priors, dataset, maximum_counts
        )
        _mark(output_root, "operation_pools_built", maximum_counts=maximum_counts)
        priority = den6._operation_priority(priors, dataset, int(setting.scale_m), 0.10)
        built = den6._build_mixed_plan(
            pools,
            den5._state_feature_context(baseline_probe).occupied,
            requested_counts,
            int(args.plan_variants),
            priority,
        )
        if built is None:
            raise RuntimeError("ana_den6 mixed planを再構築できない")
        plan, plan_meta = built
        _mark(output_root, "mixed_plan_built", candidate_count=len(plan))
        edited_coords, removes, adds = den5._apply_actual_plan_fast(baseline_probe.original_coords, plan)
        reference_coords = codec_base._read_coords(Path(str(reference["edited_ply"]))) if reference else None
        operation_counts = {name: sum(str(item.operation) == name for item in plan) for name in den6.OPERATIONS}
        expected_ids = str(reference.get("selected_candidate_ids_sample", "")).split(",") if reference else []
        actual_ids = [str(item.candidate_id) for item in plan]
        report = {
            "dataset": dataset,
            "input_file": str(input_file),
            "setting_id": str(setting.setting_id),
            "raw_point_count": raw_count,
            "unique_voxel_count": unique_count,
            "heuristics": heuristics,
            "requested_counts": requested_counts,
            "reproduced_operation_counts": operation_counts,
            "reference_operation_counts": {
                "Add": int(reference["actual_add_operation_count"]),
                "Prune": int(reference["actual_prune_operation_count"]),
                "Adjust": int(reference["actual_adjust_operation_count"]),
            } if reference else {},
            "candidate_prefix_match": actual_ids[:len(expected_ids)] == expected_ids if reference else None,
            "reproduced_candidate_count": len(plan),
            "reference_candidate_count": int(reference["selected_candidate_count"]) if reference else None,
            "reproduced_final_voxel_hash": _coord_hash(edited_coords),
            "reference_final_voxel_hash": _coord_hash(reference_coords) if reference is not None else None,
            "final_voxel_set_match": _coord_set(edited_coords) == _coord_set(reference_coords) if reference is not None else None,
            "reproduced_removed_voxels": len(removes),
            "reproduced_added_voxels": len(adds),
            "reference_saved_percent": float(reference["actual_saved_percent"]) if reference else None,
            "reference_baseline_bits": float(reference["baseline_decoder_complete_bits"]) if reference else None,
            "reference_edited_bits": float(reference["edited_decoder_complete_bits"]) if reference else None,
            "plan_metadata": plan_meta,
        }
        actual_row: Mapping[str, Any] | None = None
        if cli.run_actual:
            _mark(output_root, "before_actual_codec", candidate_count=len(plan))
            # 過去Excelに無いframeでも、同じden6 planをactual codecで評価する。
            pattern_row: Mapping[str, Any] = reference or {
                "pattern_key": _coord_hash(edited_coords)[:20],
                "total_ratio_percent": total_ratio_percent,
                "add_share": shares[0],
                "prune_share": shares[1],
                "adjust_share": shares[2],
            }
            actual_row = den6._evaluate_mixed_plan(
                den5, engine, codec_base, coder, input_item, input_file, setting,
                baseline_codec, baseline_probe, pattern_row, unique_count, plan, plan_meta, args, output_root,
            )
            _mark(output_root, "actual_codec_completed")
            report["actual"] = {
                key: actual_row.get(key)
                for key in (
                    "baseline_decoder_complete_bits", "edited_decoder_complete_bits", "actual_saved_percent",
                    "D1_loss_db", "D2_loss_db", "actual_status",
                )
            }
        if cli.manifest_out:
            _write_manifest(
                Path(cli.manifest_out).expanduser().resolve(),
                dataset=dataset,
                input_file=input_file,
                setting=setting,
                total_ratio_percent=total_ratio_percent,
                shares=shares,
                heuristics=heuristics,
                requested_counts=requested_counts,
                plan=plan,
                plan_meta=plan_meta,
                edited_coords=edited_coords,
                reference=actual_row or reference,
            )
            report["manifest_out"] = str(Path(cli.manifest_out).expanduser().resolve())
        report["passed"] = bool(
            report["candidate_prefix_match"]
            and report["reproduced_candidate_count"] == report["reference_candidate_count"]
            and report["reproduced_operation_counts"] == report["reference_operation_counts"]
            and report["final_voxel_set_match"]
        ) if reference else True
        report_path = output_root / "reproduction_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["passed"] else 2
    finally:
        codec_base._cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
