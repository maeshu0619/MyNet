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
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SPARSEPCGC_ROOT = REPO_ROOT / "compress" / "octree" / "SparsePCGC"

# den6 actual分析で採用したdataset/m別anchor patternである。
# total_ratio_percentとAdd/Prune/Adjust shareを一体で保持し、0.25%へ固定しない。
REFERENCE_PROFILES = {
    ("8I", 8): {"total_ratio_percent": 0.25, "shares": (0.40, 0.40, 0.20)},
    ("8I", 7): {"total_ratio_percent": 0.05, "shares": (0.35, 0.30, 0.35)},
    # 保存済みden6 state/Excelのactual行を正とする。
    ("MVUB", 8): {"total_ratio_percent": 0.25, "shares": (0.50, 0.40, 0.10)},
    ("MVUB", 7): {"total_ratio_percent": 0.10, "shares": (0.35, 0.30, 0.35)},
    ("UVG", 8): {"total_ratio_percent": 0.50, "shares": (0.25, 0.70, 0.05)},
    ("UVG", 7): {"total_ratio_percent": 0.25, "shares": (0.40, 0.50, 0.10)},
}

# 既存の検証コードが参照する構成比だけの旧公開名を維持する。
REFERENCE_SHARES = {
    key: tuple(profile["shares"])
    for key, profile in REFERENCE_PROFILES.items()
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
    profile = REFERENCE_PROFILES[(dataset, scale_m)]
    shares = profile["shares"]
    rows = state.get("actual_rows") or []
    matches = [
        row
        for row in rows
        if str(row.get("dataset", "")).upper() == dataset
        and int(float(row.get("scale_m", -1))) == scale_m
        and (input_file is None or Path(str(row.get("input_file", ""))).resolve() == input_file)
        and math.isclose(
            float(row.get("total_ratio_percent", -1.0)),
            float(profile["total_ratio_percent"]),
            abs_tol=1e-12,
        )
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
    argv = [
        "--data", cli.data,
        "--m-values", str(cli.scale_m),
        "--output-root", str(cli.output_root),
        "--plan-variants", str(int(cli.plan_variants)),
    ]
    if cli.input_file:
        argv.extend(("--dataset-input", str(Path(cli.input_file).expanduser().resolve())))
    if cli.cpu:
        argv.append("--cpu")
    args = den6.build_parser(den5).parse_args(argv)
    return den6._prepare_args(den5, args)


def _parse_shares(text: str, dataset: str, scale_m: int) -> tuple[float, float, float]:
    if not str(text).strip():
        return tuple(REFERENCE_PROFILES[(dataset, scale_m)]["shares"])
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("--operation-sharesは正のAdd,Prune,Adjustの3値で指定する")
    total = sum(values)
    return tuple(value / total for value in values)


def _parse_fallback_ratios(text: str, target_ratio: float) -> tuple[float, ...]:
    """den6衝突回避planが作れないframeだけで試す小さい操作率を正規化する。"""
    values = []
    for token in str(text or "").split(","):
        if not token.strip():
            continue
        value = float(token.strip())
        if value <= 0.0 or value >= float(target_ratio):
            continue
        if value not in values:
            values.append(value)
    return tuple(sorted(values, reverse=True))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline_codec_for_manifest(
    engine: ModuleType,
    codec_base: ModuleType,
    coder: Any,
    input_file: Path,
    setting: Any,
    args: argparse.Namespace,
    output_root: Path,
    *,
    require_quality: bool,
) -> dict[str, Any]:
    """候補manifest生成に不要なformal D1/D2を避けつつ、den6と同じ実bitを得る。"""
    if require_quality:
        return dict(engine._baseline_codec(codec_base, coder, input_file, setting, args, output_root))

    baseline_dir = output_root / "baseline_codec_outputs" / str(setting.setting_id)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    file_tag = input_file.stem
    bin_path = baseline_dir / f"{file_tag}.bin"
    decoded_path = baseline_dir / f"{file_tag}_dec.ply"
    raw = coder.test(
        str(input_file),
        str(bin_path),
        str(decoded_path),
        voxel_size=1.0,
        posQuantscale=1,
        scale_AE=int(setting.scale_ae),
        scale_SR=int(setting.scale_sr),
        psnr_resolution=int(setting.psnr_resolution),
        test_psnr=False,
    )
    if not isinstance(raw, Mapping) or "file_size" not in raw:
        raise RuntimeError("SparsePCGC baseline bit-only codecがdecoder complete bitを返さない")
    return {
        "status": "ok",
        "decoder_complete_bits": raw["file_size"],
        "baseline_codec_mode": "bit_only_no_formal_quality",
    }


def _candidate_manifest_row(
    candidate: Any,
    *,
    pool_rank: int | None = None,
    pool_size: int | None = None,
    anchor_selected: bool = False,
) -> dict[str, Any]:
    """den5/den6が順位付けした候補を、学習で再利用可能な形へ保存する。"""
    remove_coords = [list(map(int, coord)) for coord in getattr(candidate, "remove_coords", ())]
    add_coords = [list(map(int, coord)) for coord in getattr(candidate, "add_coords", ())]
    row = {
        "candidate_id": str(candidate.candidate_id),
        "operation": str(candidate.operation),
        "remove_coords": remove_coords,
        "add_coords": add_coords,
        "candidate_edit_key": hashlib.sha256(
            json.dumps(
                {"remove": sorted(remove_coords), "add": sorted(add_coords)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "pool_rank": int(pool_rank) if pool_rank is not None else -1,
        "pool_size": int(pool_size) if pool_size is not None else -1,
        "anchor_selected": bool(anchor_selected),
        # den5のEditCandidate値を保存し、myNet側で別の近似式を再計算しない。
        "heuristic_score": float(getattr(candidate, "heuristic_score", 0.0) or 0.0),
        "fixed_context_gain_bits": float(getattr(candidate, "fixed_context_gain_bits", 0.0) or 0.0),
        "optimistic_gain_bits": float(getattr(candidate, "optimistic_gain_bits", 0.0) or 0.0),
        "subtree_bit_mass": float(getattr(candidate, "subtree_bit_mass", 0.0) or 0.0),
        "expected_new_descendant_bits": float(getattr(candidate, "expected_new_descendant_bits", 0.0) or 0.0),
        "mask_gain_bits": float(getattr(candidate, "mask_gain_bits", 0.0) or 0.0),
        "neighbor_bit_risk": float(getattr(candidate, "neighbor_bit_risk", 0.0) or 0.0),
        "geometry_cost": float(getattr(candidate, "geometry_cost", 0.0) or 0.0),
        "affected_voxel_cells": int(getattr(candidate, "affected_voxel_cells", 0) or 0),
        "operation_count": int(getattr(candidate, "operation_count", 1) or 1),
        "symbol_index": int(getattr(candidate, "symbol_index", -1) or -1),
        "partner_symbol_index": int(getattr(candidate, "partner_symbol_index", -1) or -1),
        "depth": int(getattr(candidate, "depth", -1) or -1),
        "region_shift": int(getattr(candidate, "region_shift", -1) or -1),
        "region_prefix": [int(value) for value in getattr(candidate, "region_prefix", ())],
        "note": str(getattr(candidate, "note", "") or ""),
    }
    return row


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
    operation_pools: Mapping[str, Sequence[Any]],
    operation_priority: Sequence[str],
    plan_variant_count: int,
    plan: Sequence[Any],
    plan_meta: Mapping[str, Any],
    edited_coords: Any,
    input_coords: Any,
    reference: Mapping[str, Any] | None,
) -> None:
    operation_counts = {
        operation: sum(str(item.operation) == operation for item in plan)
        for operation in ("Add", "Prune", "Adjust")
    }
    selected_keys = {
        hashlib.sha256(
            json.dumps(
                {
                    "remove": sorted([list(map(int, coord)) for coord in getattr(item, "remove_coords", ())]),
                    "add": sorted([list(map(int, coord)) for coord in getattr(item, "add_coords", ())]),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for item in plan
    }
    ranked_pools = {}
    for operation in ("Add", "Prune", "Adjust"):
        pool = list(operation_pools.get(operation, ()))
        ranked_pools[operation] = [
            _candidate_manifest_row(
                item,
                pool_rank=rank,
                pool_size=len(pool),
                anchor_selected=(
                    hashlib.sha256(
                        json.dumps(
                            {
                                "remove": sorted([list(map(int, coord)) for coord in getattr(item, "remove_coords", ())]),
                                "add": sorted([list(map(int, coord)) for coord in getattr(item, "add_coords", ())]),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    in selected_keys
                ),
            )
            for rank, item in enumerate(pool)
        ]

    payload = {
        "schema_version": "ana_den6_ranked_candidate_manifest_v2",
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
        "operation_priority": [str(value) for value in operation_priority],
        # ana_den6が探索したvariant総数を保持する。選択variant番号だけへ縮小しない。
        "plan_variants": max(int(plan_variant_count), 6),
        "plan_metadata": dict(plan_meta),
        "ranked_candidate_pools": ranked_pools,
        "selected_candidates": [
            _candidate_manifest_row(item, anchor_selected=True) for item in plan
        ],
        "input_voxel_hash": _coord_hash(input_coords),
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
    parser.add_argument("--total-ratio-percent", type=float, default=-1.0, help="負ならdataset/m別den6 anchor値")
    parser.add_argument("--operation-shares", default="", help="Add,Prune,Adjust。空ならden6参照profile")
    parser.add_argument(
        "--plan-variants",
        type=int,
        default=6,
        help="den6の衝突回避候補順序数。ana_den6.pyの既定値と同じ6",
    )
    parser.add_argument(
        "--fallback-total-ratio-percent",
        default="",
        help="target ratioでden6 mixed planが不成立のframeだけに試す小さい操作率。例: 0.20,0.15,0.10,0.05",
    )
    parser.add_argument("--manifest-out", default="", help="同じden6 planをmyNetが読むJSON manifest出力先")
    parser.add_argument("--run-actual", action="store_true", help="再構築したplanをactual SparsePCGCで再評価する")
    parser.add_argument("--cpu", action="store_true")
    cli = parser.parse_args()
    if int(cli.plan_variants) <= 0:
        raise ValueError("--plan-variantsは1以上で指定する")

    dataset = str(cli.data).upper()
    key = (dataset, int(cli.scale_m))
    if key not in REFERENCE_PROFILES:
        raise ValueError(f"den6参照profileが無い: {key}")
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
        baseline_codec = _baseline_codec_for_manifest(
            engine,
            codec_base,
            coder,
            input_file,
            setting,
            args,
            output_root,
            require_quality=bool(cli.run_actual),
        )
        codec_base._cleanup()
        _mark(output_root, "before_baseline_probe")
        baseline_probe = engine._probe_state(codec_base, coder, input_file, setting, args, tag="mynet_den6_reproduce")
        _mark(output_root, "baseline_probe_loaded")

        shares = _parse_shares(cli.operation_shares, *key)
        total_ratio_percent = (
            float(reference["total_ratio_percent"])
            if reference is not None
            else float(cli.total_ratio_percent)
            if float(cli.total_ratio_percent) >= 0.0
            else float(REFERENCE_PROFILES[key]["total_ratio_percent"])
        )
        requested_total_ratio_percent = float(total_ratio_percent)
        requested_counts = den6._allocate_counts(
            max(3, int(math.ceil(unique_count * total_ratio_percent / 100.0))),
            dict(zip(den6.OPERATIONS, shares)),
        )
        # 元のana_den6実行が同一frame/mで実際に評価した全patternを参照し、
        # 候補poolの必要最大数も元実行と同じにする。0.25% anchorだけでpoolを
        # 切り詰めると、Networkが学習時に探索できるden6候補範囲が失われる。
        maximum_counts = dict(requested_counts)
        state_path = analysis_root / "state" / "run_rows.json"
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                for row in state.get("actual_rows", []) or []:
                    try:
                        if str(row.get("dataset", "")).upper() != dataset:
                            continue
                        if int(float(row.get("scale_m", -1))) != int(setting.scale_m):
                            continue
                        if Path(str(row.get("input_file", ""))).resolve() != input_file:
                            continue
                        row_budget = max(
                            3,
                            int(math.ceil(unique_count * float(row["total_ratio_percent"]) / 100.0)),
                        )
                        row_counts = den6._allocate_counts(
                            row_budget,
                            {
                                "Add": float(row["add_share"]),
                                "Prune": float(row["prune_share"]),
                                "Adjust": float(row["adjust_share"]),
                            },
                        )
                    except (KeyError, TypeError, ValueError, OverflowError):
                        continue
                    for operation in den6.OPERATIONS:
                        maximum_counts[operation] = max(
                            int(maximum_counts.get(operation, 0)),
                            int(row_counts.get(operation, 0)),
                        )
            except (OSError, json.JSONDecodeError):
                pass
        priors = den6._deep_copy_priors()
        pools, heuristics, _ = den6._prepare_operation_pools(
            den5, engine, baseline_probe, baseline_codec, setting, args, priors, dataset, maximum_counts
        )
        pool_counts = {operation: len(pools.get(operation, ())) for operation in den6.OPERATIONS}
        _mark(
            output_root,
            "operation_pools_built",
            maximum_counts=maximum_counts,
            requested_counts=requested_counts,
            pool_counts=pool_counts,
            plan_variants=int(args.plan_variants),
        )
        priority = den6._operation_priority(priors, dataset, int(setting.scale_m), 0.10)
        occupied = den5._state_feature_context(baseline_probe).occupied
        fallback_ratios = _parse_fallback_ratios(
            cli.fallback_total_ratio_percent,
            requested_total_ratio_percent,
        )
        attempted_ratios = (requested_total_ratio_percent, *fallback_ratios)
        built = None
        selected_counts = requested_counts
        for candidate_ratio_percent in attempted_ratios:
            candidate_counts = den6._allocate_counts(
                max(3, int(math.ceil(unique_count * candidate_ratio_percent / 100.0))),
                dict(zip(den6.OPERATIONS, shares)),
            )
            candidate_built = den6._build_mixed_plan(
                pools,
                occupied,
                candidate_counts,
                int(args.plan_variants),
                priority,
            )
            if candidate_built is not None:
                total_ratio_percent = float(candidate_ratio_percent)
                selected_counts = candidate_counts
                built = candidate_built
                break
        if built is None:
            _mark(
                output_root,
                "mixed_plan_unavailable",
                requested_counts=requested_counts,
                pool_counts=pool_counts,
                plan_variants=int(args.plan_variants),
                attempted_total_ratio_percents=attempted_ratios,
                operation_priority=priority,
            )
            raise RuntimeError("ana_den6 mixed planを再構築できない")
        plan, plan_meta = built
        plan_meta = {
            **dict(plan_meta),
            "requested_total_ratio_percent": requested_total_ratio_percent,
            "selected_total_ratio_percent": float(total_ratio_percent),
            "ratio_fallback_applied": bool(total_ratio_percent != requested_total_ratio_percent),
        }
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
            "baseline_codec_mode": str(baseline_codec.get("baseline_codec_mode", "formal_quality")),
            "raw_point_count": raw_count,
            "unique_voxel_count": unique_count,
            "heuristics": heuristics,
            "requested_counts": selected_counts,
            "requested_total_ratio_percent": requested_total_ratio_percent,
            "selected_total_ratio_percent": float(total_ratio_percent),
            "ratio_fallback_applied": bool(total_ratio_percent != requested_total_ratio_percent),
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
                requested_counts=selected_counts,
                operation_pools=pools,
                operation_priority=priority,
                plan_variant_count=int(args.plan_variants),
                plan=plan,
                plan_meta=plan_meta,
                edited_coords=edited_coords,
                input_coords=baseline_probe.original_coords,
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
