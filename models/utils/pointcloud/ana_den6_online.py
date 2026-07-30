"""063943互換のden6 Exact順位をNetwork residualへ渡すonline入口。

cache内の候補は複数の完成planではなく、1つのAdd/Prune/Adjust複合planを組む
Voxel edit-unit順位である。Step 0は保存Exact planを再生し、その後は同じ順位へ
Network residualを加える。Actual encodeは最終的に選ばれた1 planだけに1回行う。
Network-only推論は別フラグで測定し、Exact anchorの成績と混同しない。
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


SCHEMA_VERSION = "ana_den6_gt_terms_single_proposal_cache_v7"
SOURCE_NAME = "ana_den6_gt_terms_single_proposal_online_v7"
FIXED_FEATURE_SCHEMA_VERSION = "ana_den6_gt_fixed_symbol_features_v2"
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_GLOBAL_PAYLOAD_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_FIXED_FEATURE_CACHE: "OrderedDict[str, dict[str, np.ndarray]]" = OrderedDict()
_CACHE_STATS = {
    "build": 0, "memory_hit": 0, "disk_hit": 0,
    "exact_teacher_hit": 0, "exact_teacher_missing": 0,
    "exact_teacher_build": 0, "exact_teacher_build_failed": 0,
    "exact_teacher_fresh_build_load": 0,
}


def reset_ana_den6_online_runtime_cache() -> None:
    """test2のcold計測前に、同一process内の再利用状態だけを破棄する。"""
    _GLOBAL_PAYLOAD_CACHE.clear()
    _FIXED_FEATURE_CACHE.clear()
    for key in tuple(_CACHE_STATS):
        _CACHE_STATS[key] = 0


def _sha256_file(path: Path) -> str:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    for old_key in [item for item in _FILE_HASH_CACHE if item[0] == str(resolved) and item != key]:
        _FILE_HASH_CACHE.pop(old_key, None)
    _FILE_HASH_CACHE[key] = value
    return value


def _dataset_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"8i", "8ivslf"}:
        return "8i"
    if text == "mvub":
        return "MVUB"
    if text == "uvg":
        return "UVG"
    raise RuntimeError(f"ana_den6 onlineが未対応のdatasetである: {value}")


def _setting_id(args: Any) -> str:
    return (
        f"vs{float(getattr(args, 'sparsepcgc_voxel_size', 1.0)):.12g}"
        f"_pq{int(getattr(args, 'sparsepcgc_pos_quantscale', 1))}"
        f"_ae{int(getattr(args, 'sparsepcgc_scale_ae', 0))}"
        f"_sr{int(getattr(args, 'sparsepcgc_scale_sr', 0))}"
        f"_m{int(getattr(args, 'sparsepcgc_scale_m', 8))}"
    )


def _checkpoint_identifier(args: Any) -> str:
    """巨大checkpoint全体を再hashせず、圧縮設定を識別できるfingerprintを返す。"""
    for name in (
        "sparsepcgc_ckpt_dense", "sparsepcgc_ckptdir",
        "sparsepcgc_ckpt", "sparsepcgc_ckptdir_ae", "sparsepcgc_ckptdir_sr",
    ):
        raw = str(getattr(args, name, "") or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            stat = path.stat()
            return f"{path.resolve()}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}"
        return raw
    return "default_sparsepcgc_dense_checkpoint"


def _identity(args: Any, input_file: Path, input_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "input_file": str(input_file),
        "input_sha256": str(input_sha256),
        "dataset": _dataset_name(getattr(args, "dataname", "8i")),
        "setting_id": _setting_id(args),
        "voxel_size": float(getattr(args, "sparsepcgc_voxel_size", 1.0)),
        "pos_quantscale": int(getattr(args, "sparsepcgc_pos_quantscale", 1)),
        "scale_ae": int(getattr(args, "sparsepcgc_scale_ae", 0)),
        "scale_sr": int(getattr(args, "sparsepcgc_scale_sr", 0)),
        "scale_m": int(getattr(args, "sparsepcgc_scale_m", 8)),
        "codec_mode": str(getattr(args, "sparsepcgc_mode", "dense_lossy")),
        "checkpoint_identifier": _checkpoint_identifier(args),
    }


def _cache_file(args: Any, identity: Mapping[str, Any]) -> Path:
    root = Path(str(getattr(
        args,
        "heuristic_guidance_online_cache_dir",
        "/data/maejima/log/mynet_den6_online_cache",
    ))).expanduser().resolve()
    stem = Path(str(identity["input_file"])).stem.replace(" ", "_")
    signature = hashlib.sha256(
        json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return root / f"single_proposal_{identity['dataset']}_{stem}_{identity['setting_id']}_{signature}.json"


def _fixed_feature_file(args: Any, identity: Mapping[str, Any]) -> Path:
    root = Path(str(getattr(
        args,
        "heuristic_guidance_online_cache_dir",
        "/data/maejima/log/mynet_den6_online_cache",
    ))).expanduser().resolve()
    stem = Path(str(identity["input_file"])).stem.replace(" ", "_")
    signature = hashlib.sha256(
        json.dumps(
            {**dict(identity), "fixed_feature_schema": FIXED_FEATURE_SCHEMA_VERSION},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return root / (
        f"fixed_features_{identity['dataset']}_{stem}_{identity['setting_id']}_{signature}.npz"
    )


def _build_fixed_features(args: Any, identity: Mapping[str, Any], output_path: Path) -> None:
    explicit_python = str(
        getattr(args, "heuristic_guidance_online_python", "")
        or getattr(args, "sparsepcgc_python", "")
    ).strip()
    if explicit_python:
        python_executable = Path(explicit_python).expanduser().resolve()
    else:
        env_name = str(
            getattr(args, "heuristic_guidance_online_conda_env", "sparsepcgc")
        ).strip() or "sparsepcgc"
        python_executable = (
            Path(sys.executable).resolve().parents[1].parent / env_name / "bin" / "python"
        )
    tool = Path(__file__).resolve().parents[3] / "tools" / "ana_den6_online_worker.py"
    sparsepcgc_root = Path(str(getattr(
        args, "sparsepcgc_root", Path(__file__).resolve().parents[4] / "compress/octree/SparsePCGC"
    ))).expanduser().resolve()
    build_root = output_path.parent / "fixed_feature_build" / output_path.stem
    command = [
        str(python_executable), str(tool),
        "--sparsepcgc-root", str(sparsepcgc_root),
        "--data", str(identity["dataset"]),
        "--input-file", str(identity["input_file"]),
        "--scale-m", str(int(identity["scale_m"])),
        "--scale-ae", str(int(identity["scale_ae"])),
        "--scale-sr", str(int(identity["scale_sr"])),
        "--output-json", str(output_path.with_suffix(".unused.json")),
        "--output-root", str(build_root),
        "--fixed-features-output", str(output_path),
    ]
    # SparsePCGC/torchacのextension初期化ログをtrain.pyのStepログへ混ぜない。
    # 失敗時だけ末尾を例外へ含め、正常時は固定特徴の生成待ちであることを
    # 呼び出し側のauditから判別できるようにする。
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0 or not output_path.is_file():
        diagnostic = "\n".join(
            (completed.stderr or completed.stdout or "").splitlines()[-20:]
        )
        raise RuntimeError(
            "SparsePCGC GT固定symbol feature生成に失敗した: "
            f"returncode={completed.returncode}, output={output_path}, "
            f"diagnostic={diagnostic}"
        )


def _load_fixed_features(
    path: Path,
    identity: Mapping[str, Any],
    coords: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    memory_key = str(path.resolve())
    cached = _FIXED_FEATURE_CACHE.get(memory_key)
    if cached is None:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
        metadata = json.loads(str(arrays.pop("metadata_json").reshape(()).item()))
        if str(metadata.get("schema_version", "")) != FIXED_FEATURE_SCHEMA_VERSION:
            raise RuntimeError("GT固定symbol feature cache schemaが不正である")
        if str(metadata.get("source", "")) != "sparsepcgc_gt_symbol_features_no_plan_no_pool_cache":
            raise RuntimeError("GT固定symbol feature cache sourceが不正である")
        if str(metadata.get("input_sha256", "")) != str(identity["input_sha256"]):
            raise RuntimeError("GT固定symbol feature cacheの入力SHA256が一致しない")
        if (
            bool(metadata.get("contains_selected_action", True))
            or bool(metadata.get("contains_candidate_pool", True))
            or bool(metadata.get("contains_actual_edited_result", True))
        ):
            raise RuntimeError(
                "GT固定symbol feature cacheへ行動教師・Pool・加工後actualが混入した"
            )
        arrays["metadata"] = metadata
        _FIXED_FEATURE_CACHE[memory_key] = arrays
        _FIXED_FEATURE_CACHE.move_to_end(memory_key)
        while len(_FIXED_FEATURE_CACHE) > 2:
            _FIXED_FEATURE_CACHE.popitem(last=False)
        cached = arrays

    current = np.rint(
        coords[0].transpose(0, 1).detach().cpu().numpy()
    ).astype(np.int64, copy=False)
    stored = np.asarray(cached["coords"], dtype=np.int64)
    reorder = cached.get("_network_reorder")
    if current.shape != stored.shape:
        raise RuntimeError(
            "GT固定symbol featureとNetwork canonical voxelのshapeが一致しない: "
            f"fixed={stored.shape}, network={current.shape}"
        )
    if reorder is None:
        if np.array_equal(current, stored):
            reorder = np.arange(stored.shape[0], dtype=np.int64)
        else:
            # SparsePCGCとNetworkは同じGT集合を異なるcanonical順で保持し得る。
            # 座標を完全照合してから feature indexだけをNetwork順へ写す。
            stored_order = np.lexsort((stored[:, 2], stored[:, 1], stored[:, 0]))
            current_order = np.lexsort((current[:, 2], current[:, 1], current[:, 0]))
            if not np.array_equal(stored[stored_order], current[current_order]):
                raise RuntimeError(
                    "GT固定symbol featureとNetwork canonical voxelの座標集合が一致しない"
                )
            reorder = np.empty((stored.shape[0],), dtype=np.int64)
            reorder[current_order] = stored_order
        cached["_network_reorder"] = reorder
    elif not np.array_equal(stored[np.asarray(reorder, dtype=np.int64)], current):
        raise RuntimeError("GT固定symbol featureのcached座標対応が現在入力と一致しない")
    result = {
        "source": "sparsepcgc_gt_symbol_features_no_plan_no_pool_cache",
        "metadata": dict(cached["metadata"]),
        "score": {},
        "valid": {},
        "direction_index": {},
    }
    for operation, prefix in (("Add", "add"), ("Prune", "prune"), ("Adjust", "adjust")):
        result["score"][operation] = torch.from_numpy(
            cached[f"{prefix}_score"][reorder]
        ).to(
            device=device, dtype=torch.float32
        ).view(1, 1, -1)
        result["valid"][operation] = torch.from_numpy(
            cached[f"{prefix}_valid"][reorder]
        ).to(
            device=device, dtype=torch.bool
        ).view(1, 1, -1)
    for operation, prefix in (("Add", "add"), ("Adjust", "adjust")):
        result["direction_index"][operation] = torch.from_numpy(
            cached[f"{prefix}_direction"][reorder].astype(np.int64, copy=False)
        ).to(device=device, dtype=torch.long).view(1, -1)
    result["direction_bits"] = {}
    for operation, prefix in (("Add", "add"), ("Adjust", "adjust")):
        result["direction_bits"][operation] = torch.from_numpy(
            cached[f"{prefix}_direction_bits"][reorder].astype(np.int64, copy=False)
        ).to(device=device, dtype=torch.long).view(1, -1)
    return result


def _canonical_geometry_terms(coords: torch.Tensor) -> dict[str, Any]:
    if coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[1] != 3:
        raise RuntimeError(
            "ana_den6 onlineはbatch=1のfull-cloud canonical voxelだけを受け付ける: "
            f"shape={tuple(coords.shape)}"
        )
    rows = coords[0].transpose(0, 1).detach().to(device="cpu", dtype=torch.int64).contiguous()
    if rows.numel() == 0:
        raise RuntimeError("ana_den6 onlineのGT canonical voxelが空である")
    # Python tupleの全点sortは大規模点群で遅いため、NumPyのlexsortを1回だけ使う。
    values = rows.numpy()
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    packed = np.ascontiguousarray(values[order], dtype=np.int64)
    voxel_hash = hashlib.sha256(packed.tobytes(order="C")).hexdigest()
    minimum = packed.min(axis=0)
    maximum = packed.max(axis=0)
    centroid = packed.mean(axis=0, dtype=np.float64)
    return {
        "input_voxel_hash": voxel_hash,
        "unique_voxel_count": int(packed.shape[0]),
        "bbox_min": [int(value) for value in minimum.tolist()],
        "bbox_max": [int(value) for value in maximum.tolist()],
        "centroid": [float(value) for value in centroid.tolist()],
    }


def _validate_payload(payload: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if str(payload.get("source", "")) != SOURCE_NAME:
        raise RuntimeError("single-proposal online cache sourceが不正である")
    for name, expected in identity.items():
        if str(payload.get(name, "")) != str(expected):
            raise RuntimeError(f"single-proposal online cache identity不一致: {name}")
    geometry = payload.get("gt_geometry_terms")
    if not isinstance(geometry, Mapping) or not str(geometry.get("input_voxel_hash", "")):
        raise RuntimeError("single-proposal online cacheにGT幾何項が無い")
    forbidden = {
        "operation_candidate_shortlists", "ranked_candidate_pools", "initial_heuristic_plan",
        "selected_candidate_ids", "processed_point_cloud", "generated_point_cloud",
        "actual_generated_loss", "surrogate_target",
    }
    present = sorted(name for name in forbidden if name in payload)
    if present:
        raise RuntimeError(f"single-proposal online cacheに加工候補/結果が混入した: {present}")


def _load_or_build_payload(context: Mapping[str, Any], args: Any) -> dict[str, Any]:
    input_file = Path(str(getattr(args, "_current_input_file", ""))).expanduser().resolve()
    if not input_file.is_file():
        raise RuntimeError("ana_den6 onlineではargs._current_input_fileに実在PLYが必要である")
    input_sha256 = _sha256_file(input_file)
    identity = _identity(args, input_file, input_sha256)
    cache_path = _cache_file(args, identity)
    memory_key = str(cache_path)

    cached = _GLOBAL_PAYLOAD_CACHE.get(memory_key)
    if isinstance(cached, dict):
        _validate_payload(cached, identity)
        _GLOBAL_PAYLOAD_CACHE.move_to_end(memory_key)
        _CACHE_STATS["memory_hit"] += 1
        setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
        return dict(cached)

    payload = None
    if cache_path.is_file():
        try:
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            _validate_payload(candidate, identity)
            payload = dict(candidate)
            _CACHE_STATS["disk_hit"] += 1
        except (OSError, ValueError, TypeError, RuntimeError):
            # v7はGT固定metadataだけなので、安全かつ即時に再構築できる。
            payload = None

    if payload is None:
        coords = context.get("full_global_voxel_coords", context.get("global_voxel_coords"))
        if not torch.is_tensor(coords):
            raise RuntimeError("ana_den6 online contextにglobal_voxel_coordsが無い")
        geometry_terms = _canonical_geometry_terms(coords)
        payload = {
            **identity,
            "source": SOURCE_NAME,
            "gt_geometry_terms": geometry_terms,
            # 実圧縮GT値はLoss.actual_gt_cacheへ1回だけ保存し、ここでは責務だけを記録する。
            "gt_compression_terms": {
                "owner": "Loss.actual_gt_cache",
                "scope": "unmodified_full_cloud_gt_only",
            },
            "proposal_policy": "one_where_amount_action_per_step",
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
        _CACHE_STATS["build"] += 1

    payload["cache_path"] = str(cache_path)
    payload["cache_signature"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if bool(getattr(args, "heuristic_guidance_fixed_symbol_features", False)):
        fixed_path = _fixed_feature_file(args, identity)
        if not fixed_path.is_file():
            _build_fixed_features(args, identity, fixed_path)
        payload["gt_fixed_feature_path"] = str(fixed_path)
    _GLOBAL_PAYLOAD_CACHE[memory_key] = dict(payload)
    _GLOBAL_PAYLOAD_CACHE.move_to_end(memory_key)
    max_entries = max(int(getattr(args, "heuristic_guidance_online_memory_entries", 4)), 1)
    while len(_GLOBAL_PAYLOAD_CACHE) > max_entries:
        _GLOBAL_PAYLOAD_CACHE.popitem(last=False)
    setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
    return payload


def _load_exact_single_plan_teacher(
    args: Any,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """den6と同じGT由来edit-unit順位を読み、完成planは1つだけに固定する。

    旧cacheの ``operation_candidate_shortlists`` は複数の完成planではなく、
    1つのplanを構成するVoxel単位のedit候補である。Network出力、加工後点群、
    加工後loss、複数planのActual結果は含まない。名称をedit_unitsへ変換してから
    Networkへ渡し、誤ってmulti-plan経路として扱われないようにする。
    """
    root = Path(str(getattr(
        args,
        "heuristic_guidance_online_cache_dir",
        "/data/maejima/log/mynet_den6_online_cache",
    ))).expanduser().resolve()
    input_file = Path(str(identity["input_file"]))
    stem = input_file.stem.replace(" ", "_")
    dataset = str(identity["dataset"])
    setting_id = str(identity["setting_id"])
    allowed_sources = {
        "ana_den6_exact_unique_plan_online_v6",
        "ana_den6_exact_one_pattern_anchor_online_v6",
    }
    reserve_factor = min(max(float(getattr(
        args, "heuristic_guidance_online_compact_reserve_factor", 4.0
    )), 1.0), 4.0)

    exact_identity_key = "|".join((
            "exact_single_plan_teacher_v8",
            str(identity["input_sha256"]),
            str(identity["dataset"]),
            str(identity["setting_id"]),
            str(identity["codec_mode"]),
            str(identity["checkpoint_identifier"]),
            str(reserve_factor),
    ))

    def load_memory() -> dict[str, Any] | None:
        cached = _GLOBAL_PAYLOAD_CACHE.get(exact_identity_key)
        if not isinstance(cached, Mapping):
            return None
        _GLOBAL_PAYLOAD_CACHE.move_to_end(exact_identity_key)
        _CACHE_STATS["memory_hit"] += 1
        if bool(getattr(args, "_ana_den6_online_fresh_build_load", False)):
            _CACHE_STATS["exact_teacher_fresh_build_load"] += 1
        else:
            _CACHE_STATS["exact_teacher_hit"] += 1
        setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
        return dict(cached)

    def store_memory(path: Path, teacher: Mapping[str, Any]) -> None:
        del path
        _GLOBAL_PAYLOAD_CACHE[exact_identity_key] = dict(teacher)
        _GLOBAL_PAYLOAD_CACHE.move_to_end(exact_identity_key)
        # Exact JSONは1件約8MBでも、座標整数をPython objectへ展開すると
        # file size以上のRAMを使う。64 frame保持は今回のmain process
        # OS OOMを悪化させるため、同一Step内の再利用に足りる直近数件へ
        # 制限する。順位・plan・Actual値は一切変更しない。
        max_entries = min(max(
            int(getattr(args, "heuristic_guidance_online_memory_entries", 4)), 1
        ), 4)
        while len(_GLOBAL_PAYLOAD_CACHE) > max_entries:
            _GLOBAL_PAYLOAD_CACHE.popitem(last=False)

    memory_teacher = load_memory()
    if memory_teacher is not None:
        return memory_teacher
    exact_cache_paths = list(root.glob(f"{dataset}_{stem}_{setting_id}_*.pt"))
    exact_cache_paths += list(
        root.glob(f"exact_single_plan_{dataset}_{stem}_{setting_id}_*.json")
    )
    for path in sorted(exact_cache_paths):
        try:
            candidate = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.suffix.lower() == ".json"
                else torch.load(str(path), map_location="cpu")
            )
        except Exception:
            continue
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get("source", "")) not in allowed_sources:
            continue
        if str(candidate.get("input_sha256", "")) != str(identity["input_sha256"]):
            continue
        candidate_setting = str(candidate.get("setting_id", ""))
        if candidate_setting != setting_id and not candidate_setting.endswith("_" + setting_id):
            continue
        edit_units = candidate.get("operation_candidate_shortlists")
        if not isinstance(edit_units, Mapping) or any(
            not isinstance(edit_units.get(name), list) or not edit_units.get(name)
            for name in ("Add", "Prune", "Adjust")
        ):
            continue
        anchor_plan = candidate.get("initial_heuristic_plan")
        if not isinstance(anchor_plan, Mapping) or not bool(anchor_plan.get("available", False)):
            continue
        anchor_counts = dict(
            candidate.get("anchor_operation_counts")
            or anchor_plan.get("operation_counts")
            or {}
        )
        full_counts = dict(candidate.get("full_pool_counts") or {})
        # reserve>1では、Exact planだけを保持した旧cacheを候補学習へ流用しない。
        # 旧cacheは削除せず、必要な順位を持つ別cacheを初回だけ生成する。
        if reserve_factor > 1.0 and any(
            len(edit_units[name]) < min(
                int(full_counts.get(name, len(edit_units[name]))),
                int(np.ceil(max(int(anchor_counts.get(name, 1)), 1) * reserve_factor)),
            )
            for name in ("Add", "Prune", "Adjust")
        ):
            continue

        teacher = dict(candidate)
        teacher.pop("operation_candidate_shortlists", None)
        # 完全Poolはoffline蒸留Cache構築専用であり、通常のHeuristic実行planへ
        # 混入させない。必要なcompact edit unitsだけを下で明示的に渡す。
        teacher.pop("operation_candidate_pools", None)
        teacher.pop("ranked_candidate_pools", None)
        teacher.pop("initial_heuristic_plan", None)
        teacher["operation_edit_units"] = dict(edit_units)
        teacher["heuristic_anchor_plan"] = dict(anchor_plan)
        teacher["legacy_den6_source"] = str(candidate.get("source", ""))
        teacher["source"] = "ana_den6_exact_single_plan_teacher_online_v8"
        teacher["proposal_policy"] = "one_where_amount_action_per_step"
        teacher["full_plan_candidate_count"] = 1
        teacher["actual_candidate_encode_count"] = 0
        teacher["cache_path"] = str(path)
        teacher["cache_signature"] = hashlib.sha256(
            (str(path.resolve()) + "|" + str(identity["input_sha256"])).encode("utf-8")
        ).hexdigest()
        store_memory(path, teacher)
        if bool(getattr(args, "_ana_den6_online_fresh_build_load", False)):
            _CACHE_STATS["exact_teacher_fresh_build_load"] += 1
        else:
            _CACHE_STATS["exact_teacher_hit"] += 1
        setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
        return teacher

    # 一部frameは旧v6 compact cacheではなく、同じden6計算のv2 manifestだけを
    # 持つ。そこから「実際に選ばれた1planのedit units」だけを取り出す。
    manifest_root = Path(str(getattr(
        args,
        "heuristic_guidance_den6_manifest_dir",
        "/data/maejima/log/mynet_den6_manifests",
    ))).expanduser().resolve()
    scale_m = int(identity["scale_m"])
    manifest_paths = (
        []
        if (
            reserve_factor > 1.0
            or not bool(getattr(
                args, "heuristic_guidance_allow_den6_manifest_fallback", True
            ))
        )
        else sorted(manifest_root.glob(f"{dataset}_{stem}_m{scale_m}_*.json"))
    )
    for path in manifest_paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if str(manifest.get("schema_version", "")) != "ana_den6_ranked_candidate_manifest_v2":
            continue
        if str(manifest.get("input_sha256", "")) != str(identity["input_sha256"]):
            continue
        if any(
            int(manifest.get(name, -1)) != int(identity[name])
            for name in ("scale_m", "scale_ae", "scale_sr")
        ):
            continue
        selected = manifest.get("selected_candidates")
        if not isinstance(selected, list) or not selected:
            continue
        edit_units = {name: [] for name in ("Add", "Prune", "Adjust")}
        for item in selected:
            if not isinstance(item, Mapping):
                continue
            operation = str(item.get("operation", ""))
            if operation not in edit_units:
                continue
            unit = dict(item)
            unit["pool_rank"] = len(edit_units[operation])
            edit_units[operation].append(unit)
        if any(not edit_units[name] for name in edit_units):
            continue
        for units in edit_units.values():
            size = len(units)
            for rank, unit in enumerate(units):
                unit["rank_score"] = float(
                    1.0 if size <= 1 else 1.0 - rank / float(size - 1)
                )
        selected_counts = {
            name: len(edit_units[name]) for name in ("Add", "Prune", "Adjust")
        }
        metadata = dict(manifest.get("plan_metadata") or {})
        operation_order = str(metadata.get("operation_order", ""))
        priority = [name for name in operation_order.split(">") if name in edit_units]
        if set(priority) != set(edit_units):
            priority = list(manifest.get("operation_priority") or ("Prune", "Add", "Adjust"))
        total_count = max(sum(selected_counts.values()), 1)
        total_ratio = float(
            metadata.get(
                "selected_total_ratio_percent",
                manifest.get("total_ratio_percent", 0.25),
            )
        ) / 100.0
        teacher = {
            "schema_version": "ana_den6_exact_single_plan_teacher_cache_v8",
            "source": "ana_den6_exact_single_plan_teacher_online_v8",
            "legacy_den6_source": "ana_den6_ranked_candidate_manifest_v2:selected_plan",
            "input_file": str(identity["input_file"]),
            "input_sha256": str(identity["input_sha256"]),
            "input_voxel_hash": str(manifest.get("input_voxel_hash", "")),
            "dataset": dataset,
            "setting_id": setting_id,
            "scale_m": int(identity["scale_m"]),
            "scale_ae": int(identity["scale_ae"]),
            "scale_sr": int(identity["scale_sr"]),
            "total_ratio": total_ratio,
            "operation_shares": {
                name: selected_counts[name] / float(total_count) for name in selected_counts
            },
            "operation_heuristics": dict(manifest.get("operation_heuristics") or {}),
            "operation_priority": priority,
            "operation_edit_units": edit_units,
            "heuristic_anchor_plan": {
                "available": True,
                "candidate_ids": [str(item.get("candidate_id", "")) for item in selected],
                "operation_counts": selected_counts,
                "metadata": {**metadata, "operation_order": ">".join(priority)},
                "final_voxel_hash": str(manifest.get("final_voxel_hash", "")),
            },
            "proposal_policy": "one_where_amount_action_per_step",
            "full_plan_candidate_count": 1,
            "actual_candidate_encode_count": 0,
            "cache_path": str(path),
            "cache_signature": hashlib.sha256(
                (str(path.resolve()) + "|" + str(identity["input_sha256"])).encode("utf-8")
            ).hexdigest(),
        }
        store_memory(path, teacher)
        _CACHE_STATS["exact_teacher_hit"] += 1
        setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
        return teacher

    if bool(getattr(args, "heuristic_guidance_auto_build_exact_single_plan_teacher", False)):
        explicit_python = str(
            getattr(args, "heuristic_guidance_online_python", "")
            or getattr(args, "sparsepcgc_python", "")
        ).strip()
        if explicit_python:
            python_executable = Path(explicit_python).expanduser().resolve()
        else:
            env_name = str(
                getattr(args, "heuristic_guidance_online_conda_env", "sparsepcgc")
            ).strip() or "sparsepcgc"
            python_executable = (
                Path(sys.executable).resolve().parents[1].parent / env_name / "bin" / "python"
            )
        tool = Path(__file__).resolve().parents[3] / "tools" / "ana_den6_online_worker.py"
        sparsepcgc_root = Path(str(getattr(
            args, "sparsepcgc_root", Path(__file__).resolve().parents[4] / "compress/octree/SparsePCGC"
        ))).expanduser().resolve()
        signature = str(identity["input_sha256"])[:16]
        reserve_tag = str(reserve_factor).replace(".", "p")
        output_json = root / (
            f"exact_single_plan_{dataset}_{stem}_{setting_id}_{signature}_rf{reserve_tag}.json"
        )
        output_root = root / "build" / f"{dataset}_{stem}_{setting_id}_{signature}"
        command = [
            str(python_executable), str(tool),
            "--sparsepcgc-root", str(sparsepcgc_root),
            "--data", dataset,
            "--input-file", str(identity["input_file"]),
            "--scale-m", str(int(identity["scale_m"])),
            "--scale-ae", str(int(identity["scale_ae"])),
            "--scale-sr", str(int(identity["scale_sr"])),
            "--max-total-ratio", str(float(getattr(
                args, "heuristic_guidance_online_max_total_ratio", 0.0099
            ))),
            "--compact-reserve-factor", str(float(getattr(
                args, "heuristic_guidance_online_compact_reserve_factor", 1.0
            ))),
            "--output-json", str(output_json),
            "--output-root", str(output_root),
        ]
        if bool(getattr(args, "cpu", False)):
            command.append("--cpu")
        _CACHE_STATS["exact_teacher_build"] += 1
        setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0 and output_json.is_file():
            # 同じ関数を1回だけ再試行する。再帰先ではbuildを無効にして、
            # worker異常時の無限再起動を防ぐ。
            setattr(args, "heuristic_guidance_auto_build_exact_single_plan_teacher", False)
            setattr(args, "_ana_den6_online_fresh_build_load", True)
            try:
                return _load_exact_single_plan_teacher(args, identity)
            finally:
                setattr(args, "_ana_den6_online_fresh_build_load", False)
                setattr(args, "heuristic_guidance_auto_build_exact_single_plan_teacher", True)
        _CACHE_STATS["exact_teacher_build_failed"] += 1
        setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
    _CACHE_STATS["exact_teacher_missing"] += 1
    setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
    return None


def prefetch_ana_den6_online_guidance(args: Any, input_files: Any) -> dict[str, int]:
    """既存exact teacherは直接再利用し、欠落分だけ使用時に1回生成する。"""
    del args, input_files
    return {"submitted": 0, "pending": 0, "done": 0}


def shutdown_ana_den6_online_prefetch(*, wait: bool = True) -> None:
    del wait


def attach_ana_den6_online_guidance(
    context: Mapping[str, Any],
    args: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """den6 Exact edit-unit poolを訓練・評価用contextへ付与する。"""
    mode = str(getattr(args, "heuristic_guidance_mode", "proxy_prior")).strip().lower()
    if mode != "ana_den6_online":
        return dict(context)
    if not isinstance(context, Mapping):
        raise RuntimeError("ana_den6 onlineにはfull-cloud canonical contextが必要である")
    # 063943経路と同様に、den6が順位付けしたedit-unit集合とExact anchorを
    # 最優先する。欠落時にGT proxyへ静かに劣化させない。
    timing_start = time.perf_counter()
    stats_before = dict(_CACHE_STATS)
    identity = None
    if context.get("input_file") and context.get("input_sha256"):
        identity = _identity(
            args,
            Path(str(context["input_file"])),
            str(context["input_sha256"]),
        )
    else:
        current_input = str(getattr(args, "_current_input_file", "") or "").strip()
        current_path = Path(current_input).expanduser().resolve() if current_input else None
        if current_path is not None and current_path.is_file():
            identity = _identity(args, current_path, _sha256_file(current_path))
    payload = _load_exact_single_plan_teacher(args, identity) if identity is not None else None
    if payload is None:
        if bool(getattr(args, "heuristic_guidance_require_exact_single_plan_teacher", True)):
            raise RuntimeError(
                "ana_den6 Exact cacheが見つからない。proxy/全Voxel分類へfallbackせず、"
                "初回だけden6 workerでcacheを構築すること"
            )
        payload = _load_or_build_payload(context, args)
    timing_end = time.perf_counter()
    stats_after = dict(_CACHE_STATS)
    setattr(args, "_ana_den6_online_last_timing", {
        "heuristic_pool_wall_time": float(timing_end - timing_start),
        "worker_reported_time": float(payload.get("elapsed_sec", 0.0) or 0.0),
        "exact_teacher_build_delta": int(
            stats_after.get("exact_teacher_build", 0)
            - stats_before.get("exact_teacher_build", 0)
        ),
        "memory_hit_delta": int(
            stats_after.get("memory_hit", 0) - stats_before.get("memory_hit", 0)
        ),
        "disk_hit_delta": int(
            stats_after.get("disk_hit", 0) - stats_before.get("disk_hit", 0)
        ),
        "exact_teacher_hit_delta": int(
            stats_after.get("exact_teacher_hit", 0)
            - stats_before.get("exact_teacher_hit", 0)
        ),
        "fresh_build_load_delta": int(
            stats_after.get("exact_teacher_fresh_build_load", 0)
            - stats_before.get("exact_teacher_fresh_build_load", 0)
        ),
        "cache_root": str(getattr(args, "heuristic_guidance_online_cache_dir", "")),
        "input_file": str(getattr(args, "_current_input_file", "")),
    })
    payload["teacher_bootstrap_active"] = False
    payload["teacher_bootstrap_steps"] = 0
    out = dict(context)
    # key名は既存Networkとの互換性を保つ。中身はden6候補順位と1つのanchor planである。
    out["ana_den6_ranked_candidate_guidance"] = payload
    fixed_path = str(payload.get("gt_fixed_feature_path", "")).strip()
    if fixed_path:
        coords = out.get("global_voxel_coords")
        if not torch.is_tensor(coords):
            raise RuntimeError("GT固定symbol featureにはglobal_voxel_coordsが必要である")
        identity = _identity(
            args,
            Path(str(payload["input_file"])),
            str(payload["input_sha256"]),
        )
        out["ana_den6_fixed_feature_guidance"] = _load_fixed_features(
            Path(fixed_path), identity, coords, device
        )
    out["ana_den6_online_cache_path"] = str(payload.get("cache_path", ""))
    out["ana_den6_online_cache_used"] = True
    if torch.is_tensor(out.get("global_voxel_coords")):
        out["global_voxel_coords"] = out["global_voxel_coords"].to(device=device)
    return out
