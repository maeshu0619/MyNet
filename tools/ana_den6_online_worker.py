#!/usr/bin/env python3
"""ana_den6と同一の候補生成・Heuristic順位をonline学習用cacheへ保存するworker。

このworkerはmyNet本体とは別Python processで動かす。SparsePCGCとmyNetが同じ
``models`` package名を持つため、同一processへimportするとmodule衝突が起きるからである。
actualで多数planを探索せず、baseline probeと順位付き候補poolだけを一度生成する。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "ana_den6_online_candidate_cache_v2"
OPERATIONS = ("Add", "Prune", "Adjust")
DEFAULT_PROFILES: dict[tuple[str, int], tuple[float, tuple[float, float, float]]] = {
    ("8I", 8): (0.0025, (0.40, 0.40, 0.20)),
    ("8I", 7): (0.0005, (0.35, 0.30, 0.35)),
    ("MVUB", 8): (0.0025, (0.50, 0.40, 0.10)),
    ("MVUB", 7): (0.0010, (0.35, 0.30, 0.35)),
    ("UVG", 8): (0.0050, (0.25, 0.70, 0.05)),
    ("UVG", 7): (0.0025, (0.40, 0.50, 0.10)),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coord_hash(coords: Any) -> str:
    import numpy as np

    values = np.asarray(coords)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"座標shapeが不正である: {values.shape}")
    values = np.unique(np.rint(values[:, -3:]).astype(np.int64, copy=False), axis=0)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"moduleを読み込めない: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _dataset_key(raw: str) -> str:
    text = str(raw).strip().upper()
    return "8I" if text in {"8I", "8IVSLF"} else text


def _parse_shares(raw: str, dataset: str, scale_m: int) -> tuple[float, float, float]:
    if str(raw).strip():
        values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
        if len(values) != 3 or any(value <= 0.0 for value in values):
            raise ValueError("--operation-sharesは正のAdd,Prune,Adjustの3値で指定する")
        total = sum(values)
        return tuple(float(value / total) for value in values)
    profile = DEFAULT_PROFILES.get((dataset, int(scale_m)))
    if profile is None:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return profile[1]


def _default_total_ratio(dataset: str, scale_m: int) -> float:
    profile = DEFAULT_PROFILES.get((dataset, int(scale_m)))
    return float(profile[0]) if profile is not None else 0.0025


def _build_den6_args(
    den6: ModuleType,
    den5: ModuleType,
    *,
    data: str,
    scale_m: int,
    input_file: Path,
    output_root: Path,
    use_cpu: bool,
) -> argparse.Namespace:
    argv = [
        "--data", str(data),
        "--m-values", str(int(scale_m)),
        "--output-root", str(output_root),
        "--dataset-input", str(input_file),
    ]
    if use_cpu:
        argv.append("--cpu")
    args = den6.build_parser(den5).parse_args(argv)
    return den6._prepare_args(den5, args)


def _candidate_payload(
    den5: ModuleType,
    candidate: Any,
    *,
    rank: int,
    pool_size: int,
    heuristic: str,
    state: Any,
) -> dict[str, Any]:
    score = float(den5._candidate_score(candidate, heuristic, state, {"status": "actual_only_no_surrogate"}))
    payload = {
        "candidate_id": str(getattr(candidate, "candidate_id", "")),
        "operation": str(getattr(candidate, "operation", "")),
        "pool_rank": int(rank),
        "rank_score": float(1.0 if pool_size <= 1 else 1.0 - rank / max(pool_size - 1, 1)),
        "heuristic_score": score,
        "remove_coords": [
            [int(value) for value in coord]
            for coord in getattr(candidate, "remove_coords", ())
        ],
        "add_coords": [
            [int(value) for value in coord]
            for coord in getattr(candidate, "add_coords", ())
        ],
        "affected_voxel_cells": int(getattr(candidate, "affected_voxel_cells", 1)),
        "operation_count": int(getattr(candidate, "operation_count", 1)),
    }
    # den6と同じ候補特徴を診断・将来拡張に再利用できるよう有限scalarだけ保持する。
    for name in (
        "symbol_index", "partner_symbol_index", "depth", "region_shift",
        "fixed_context_gain_bits", "optimistic_gain_bits", "subtree_bit_mass",
        "expected_new_descendant_bits", "mask_gain_bits", "neighbor_bit_risk",
        "geometry_cost",
    ):
        value = getattr(candidate, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            payload[name] = value
    return payload


def _initial_plan_payload(
    den6: ModuleType,
    den5: ModuleType,
    pools: Mapping[str, Sequence[Any]],
    occupied: set[tuple[int, int, int]],
    counts: Mapping[str, int],
    plan_variants: int,
    priority: tuple[str, str, str],
    original_coords: Any,
) -> dict[str, Any]:
    """Network residual=0時と同じ、den6順位先頭から1planだけ構築する。"""
    del plan_variants
    selected = []
    removes: set[tuple[int, int, int]] = set()
    adds: set[tuple[int, int, int]] = set()
    selected_counts = {name: 0 for name in OPERATIONS}
    for operation in priority:
        for candidate in pools.get(operation, ()):
            if selected_counts[operation] >= int(counts[operation]):
                break
            if not den6._candidate_compatible(candidate, occupied, removes, adds):
                continue
            candidate_removes, candidate_adds = den6._candidate_coords(candidate)
            removes.update(candidate_removes)
            adds.update(candidate_adds)
            selected.append(candidate)
            selected_counts[operation] += 1
    if any(selected_counts[name] <= 0 for name in OPERATIONS):
        return {
            "available": False,
            "reason": "single_online_plan_missing_operation",
            "operation_counts": selected_counts,
        }
    edited_coords, removes, adds = den5._apply_actual_plan_fast(original_coords, selected)
    return {
        "available": True,
        "candidate_ids": [str(getattr(item, "candidate_id", "")) for item in selected],
        "operation_counts": selected_counts,
        "metadata": {
            "operation_order": ">".join(priority),
            "variant_index": 0,
            "one_pattern_only": True,
        },
        "final_voxel_hash": _coord_hash(edited_coords),
        "removed_voxel_count": int(len(removes)),
        "added_voxel_count": int(len(adds)),
    }


def _compact_online_shortlist(
    pool: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Step 0の選択候補を必ず残し、rank近傍だけをonline residual用に保持する。"""
    selected = [item for item in pool if str(item.get("candidate_id", "")) in selected_ids]
    if len(selected) != len(selected_ids) or len(selected) > int(limit):
        raise RuntimeError("online shortlistが初期den6 planを完全に保持できない")
    keep_ids = {str(item.get("candidate_id", "")) for item in selected}
    for item in pool:
        if len(keep_ids) >= int(limit):
            break
        keep_ids.add(str(item.get("candidate_id", "")))
    return [dict(item) for item in pool if str(item.get("candidate_id", "")) in keep_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="ana_den6 online candidate cache worker")
    parser.add_argument("--sparsepcgc-root", required=True)
    parser.add_argument("--data", required=True, choices=("8i", "MVUB", "UVG"))
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--scale-m", required=True, type=int)
    parser.add_argument("--scale-ae", default=0, type=int)
    parser.add_argument("--scale-sr", required=True, type=int)
    parser.add_argument("--total-ratio", default=-1.0, type=float, help="0-1表記")
    parser.add_argument("--operation-shares", default="")
    parser.add_argument("--max-total-ratio", default=0.0099, type=float, help="候補poolを用意する最大総操作率")
    parser.add_argument("--full-pool-limit-per-operation", default=0, type=int)
    parser.add_argument("--compact-reserve-factor", default=1.0, type=float)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cpu", action="store_true")
    cli = parser.parse_args()

    sparse_root = Path(cli.sparsepcgc_root).expanduser().resolve()
    input_file = Path(cli.input_file).expanduser().resolve()
    output_json = Path(cli.output_json).expanduser().resolve()
    output_root = Path(cli.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not input_file.is_file():
        raise FileNotFoundError(f"入力PLYが存在しない: {input_file}")
    den6_path = sparse_root / "ana_den6.py"
    den5_path = sparse_root / "ana_den5_v8.py"
    if not den6_path.is_file() or not den5_path.is_file():
        raise FileNotFoundError(
            f"ana_den6.py/ana_den5_v8.pyが不足している: root={sparse_root}"
        )

    started = time.time()
    den6 = _load_module(den6_path, "mynet_den6_online_den6")
    den5 = _load_module(den5_path, "mynet_den6_online_den5")
    # den6側が選ぶden5版と明示pathが一致することを確認する。
    loaded_den5, loaded_den5_path = den6._load_den5()
    if Path(loaded_den5_path).resolve() != den5_path:
        den5 = loaded_den5
        den5_path = Path(loaded_den5_path).resolve()

    args = _build_den6_args(
        den6,
        den5,
        data=cli.data,
        scale_m=int(cli.scale_m),
        input_file=input_file,
        output_root=output_root,
        use_cpu=bool(cli.cpu),
    )
    engine_path, _ = den5._resolve_base_script(args)
    engine = den5._load_module(engine_path)
    codec_base_path = den5._resolve_codec_base_script(engine_path, args)
    codec_base = (
        engine._load_base_module(codec_base_path)
        if hasattr(engine, "_load_base_module")
        else den5._load_module(codec_base_path)
    )
    den5._require_base_api(engine, codec_base)

    input_items = list(engine._find_input_items_strict(codec_base, args))
    input_item = next(
        (item for item in input_items if Path(str(item.path)).resolve() == input_file),
        None,
    )
    if input_item is None:
        raise RuntimeError(f"den6 input itemを特定できない: {input_file}")
    settings = list(engine._build_m_settings(codec_base, args))
    setting = next(
        (
            item for item in settings
            if int(item.scale_m) == int(cli.scale_m)
            and int(item.scale_ae) == int(cli.scale_ae)
            and int(item.scale_sr) == int(cli.scale_sr)
        ),
        None,
    )
    if setting is None:
        # den5の既定m設定はAE=0代表点だけだが、train側がAE=1等を明示した場合も
        # 同じSparsePCGC DenseSettingとしてexact probeへ渡せるようにする。
        native_resolution = int(getattr(args, "native_resolution", 1023))
        native_bits = int(math.ceil(math.log2(native_resolution + 1)))
        if native_bits - int(cli.scale_ae) - int(cli.scale_sr) != int(cli.scale_m):
            available = [
                (int(item.scale_ae), int(item.scale_sr), int(item.scale_m))
                for item in settings
            ]
            raise RuntimeError(
                "ana_den6の設定とtrain設定が一致しない: "
                f"requested={(cli.scale_ae, cli.scale_sr, cli.scale_m)}, available={available}"
            )
        setting = codec_base.DenseSetting(
            setting_id=(
                "native_vs1_pq1"
                f"_ae{int(cli.scale_ae)}_sr{int(cli.scale_sr)}_m{int(cli.scale_m)}"
            ),
            scale_ae=int(cli.scale_ae),
            scale_sr=int(cli.scale_sr),
            scale_m=int(cli.scale_m),
            pair_source="mynet_online_requested_pair",
            voxel_size=1.0,
            pos_quantscale=1,
            psnr_resolution=native_resolution,
        )

    codec_base._resolve_checkpoints(args)
    import torch

    device = torch.device("cpu" if cli.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    coder = codec_base._load_dense_coder(args, device)
    try:
        raw_count, unique_count, _, _ = den5._count_raw_and_unique_voxels(codec_base, input_file)
        baseline_codec = engine._baseline_codec(
            codec_base, coder, input_file, setting, args, output_root
        )
        codec_base._cleanup()
        baseline_probe = engine._probe_state(
            codec_base,
            coder,
            input_file,
            setting,
            args,
            tag="mynet_den6_online",
        )

        dataset = _dataset_key(cli.data)
        total_ratio = (
            float(cli.total_ratio)
            if float(cli.total_ratio) > 0.0
            else _default_total_ratio(dataset, int(cli.scale_m))
        )
        total_ratio = min(max(total_ratio, 3.0 / max(float(unique_count), 1.0)), 0.0099)
        shares_tuple = _parse_shares(cli.operation_shares, dataset, int(cli.scale_m))
        shares = dict(zip(OPERATIONS, shares_tuple))

        initial_total_count = max(3, int(math.ceil(unique_count * total_ratio)))
        initial_counts = den6._allocate_counts(initial_total_count, shares)
        max_total_ratio = min(max(float(cli.max_total_ratio), total_ratio), 0.0099)
        maximum_total_count = max(3, int(math.ceil(unique_count * max_total_ratio)))
        # 1つのplanを作るために必要な候補だけを生成する。旧既定8192件×3操作は、
        # 約1900 actionの一意planに対して過剰だった。衝突回避用に25%だけ内部余裕を持つ。
        configured_limit = max(int(cli.full_pool_limit_per_operation), 0)
        build_reserve = max(float(cli.compact_reserve_factor), 1.25)
        # 先行操作との衝突により、件数の少ないAdjustでも元順位を深く読む。
        # そのため操作別countではなく3操作中の最大countを共通基準にする。
        common_required = max(
            int(math.ceil(float(max(initial_counts.values())) * build_reserve)),
            max(initial_counts.values()),
            3,
        )
        per_operation_limits = {}
        for name in OPERATIONS:
            required = int(common_required)
            if configured_limit > 0:
                required = min(required, configured_limit)
                required = max(required, int(initial_counts[name]))
            per_operation_limits[name] = min(required, maximum_total_count)
        maximum_counts = dict(per_operation_limits)
        priors = den6._deep_copy_priors()
        pools, heuristics, diagnostics = den6._prepare_operation_pools(
            den5,
            engine,
            baseline_probe,
            baseline_codec,
            setting,
            args,
            priors,
            dataset,
            maximum_counts,
        )
        # 順位はden6._prepare_operation_poolsが返した順序をそのまま保持する。
        full_serialized_pools: dict[str, list[dict[str, Any]]] = {}
        for operation in OPERATIONS:
            pool = list(pools.get(operation, ()))[:per_operation_limits[operation]]
            full_serialized_pools[operation] = [
                _candidate_payload(
                    den5,
                    candidate,
                    rank=rank,
                    pool_size=len(pool),
                    heuristic=str(heuristics[operation]),
                    state=baseline_probe,
                )
                for rank, candidate in enumerate(pool)
            ]
            pools[operation] = pool
            if not pool:
                raise RuntimeError(f"ana_den6 online候補poolが空である: {operation}")

        for name in OPERATIONS:
            initial_counts[name] = min(int(initial_counts[name]), len(pools[name]))
        priority = den6._operation_priority(
            priors,
            dataset,
            int(setting.scale_m),
            total_ratio * 100.0,
        )
        occupied = den5._state_feature_context(baseline_probe).occupied
        initial_plan = _initial_plan_payload(
            den6,
            den5,
            pools,
            occupied,
            initial_counts,
            int(args.plan_variants),
            priority,
            baseline_probe.original_coords,
        )
        if not bool(initial_plan.get("available", False)):
            raise RuntimeError(f"den6 initial planを構築できない: {initial_plan.get('reason', '')}")
        selected_ids = set(str(value) for value in initial_plan.get("candidate_ids", ()))
        reserve_factor = max(float(cli.compact_reserve_factor), 1.0)
        shortlist_limits = {
            operation: int(initial_counts[operation]) + int(
                math.ceil(float(initial_counts[operation]) * (reserve_factor - 1.0))
            )
            for operation in OPERATIONS
        }
        serialized_pools = {
            operation: _compact_online_shortlist(
                full_serialized_pools[operation],
                {
                    candidate_id
                    for candidate_id in selected_ids
                    if candidate_id in {
                        str(item.get("candidate_id", "")) for item in full_serialized_pools[operation]
                    }
                },
                shortlist_limits[operation],
            )
            for operation in OPERATIONS
        }

        payload = {
            "schema_version": SCHEMA_VERSION,
            "source": "ana_den6_exact_unique_plan_online_v6",
            "created_at_unix": time.time(),
            "elapsed_sec": float(time.time() - started),
            "dataset": dataset,
            "input_file": str(input_file),
            "input_sha256": _sha256_file(input_file),
            "input_voxel_hash": _coord_hash(baseline_probe.original_coords),
            "raw_point_count": int(raw_count),
            "unique_voxel_count": int(unique_count),
            "setting_id": str(setting.setting_id),
            "scale_m": int(setting.scale_m),
            "scale_ae": int(setting.scale_ae),
            "scale_sr": int(setting.scale_sr),
            "total_ratio": float(total_ratio),
            "operation_shares": {name: float(shares[name]) for name in OPERATIONS},
            "operation_heuristics": dict(heuristics),
            "operation_priority": list(priority),
            "plan_variants": int(args.plan_variants),
            "anchor_operation_counts": dict(initial_counts),
            "operation_candidate_shortlists": serialized_pools,
            "shortlist_limits": shortlist_limits,
            "full_pool_counts": {name: len(full_serialized_pools[name]) for name in OPERATIONS},
            "pool_diagnostics": diagnostics,
            "initial_heuristic_plan": initial_plan,
            "baseline_decoder_complete_bits": float(
                baseline_codec.get("decoder_complete_bits", baseline_codec.get("bit", 0.0))
            ),
            "den6_path": str(den6_path),
            "den6_sha256": _sha256_file(den6_path),
            "den5_path": str(den5_path),
            "den5_sha256": _sha256_file(den5_path),
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_json.with_suffix(output_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_json)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output_json": str(output_json),
                    "input_file": str(input_file),
                    "setting_id": str(setting.setting_id),
                    "pool_counts": {
                        name: len(serialized_pools[name]) for name in OPERATIONS
                    },
                    "initial_plan_available": bool(initial_plan.get("available", False)),
                    "elapsed_sec": float(time.time() - started),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        codec_base._cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
