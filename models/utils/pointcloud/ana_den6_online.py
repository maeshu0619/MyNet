"""GT固定情報だけを保持する、単一提案方式のonline Heuristic入口。

旧実装は初見frameごとにana_den6 workerを起動し、Add/Prune/Adjustの候補poolと
初期planを生成・永続化していた。現在は候補や加工結果をcacheしない。GT由来の
識別情報と幾何統計だけを軽量cacheへ保存し、Where/Amount/Actionは各Stepで
Heuristic priorとNetwork residualから一度だけ決定する。

GTの実圧縮項はLoss.actual_gt_cache、GT点群はdataset._PLY_CACHEがそれぞれ既存の
責務として保持する。このmoduleへ重複保存せず、加工後点群・加工後loss・Network
の選択結果は一切保存しない。
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCHEMA_VERSION = "ana_den6_gt_terms_single_proposal_cache_v7"
SOURCE_NAME = "ana_den6_gt_terms_single_proposal_online_v7"
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_GLOBAL_PAYLOAD_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_CACHE_STATS = {"build": 0, "memory_hit": 0, "disk_hit": 0}


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
    _GLOBAL_PAYLOAD_CACHE[memory_key] = dict(payload)
    _GLOBAL_PAYLOAD_CACHE.move_to_end(memory_key)
    max_entries = max(int(getattr(args, "heuristic_guidance_online_memory_entries", 4)), 1)
    while len(_GLOBAL_PAYLOAD_CACHE) > max_entries:
        _GLOBAL_PAYLOAD_CACHE.popitem(last=False)
    setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
    return payload


def prefetch_ana_den6_online_guidance(args: Any, input_files: Any) -> dict[str, int]:
    """候補workerは廃止済み。軽量metadataは使用時に同期なしで生成する。"""
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
    """GT metadataを付け、単一提案priorの識別情報をNetworkへ渡す。"""
    mode = str(getattr(args, "heuristic_guidance_mode", "proxy_prior")).strip().lower()
    if mode != "ana_den6_online":
        return dict(context)
    if not isinstance(context, Mapping):
        raise RuntimeError("ana_den6 onlineにはfull-cloud canonical contextが必要である")
    payload = _load_or_build_payload(context, args)
    out = dict(context)
    # 互換key名は維持するが、中身は候補poolではなくGT固定metadataだけである。
    out["ana_den6_ranked_candidate_guidance"] = payload
    out["ana_den6_online_cache_path"] = str(payload.get("cache_path", ""))
    out["ana_den6_online_cache_used"] = True
    if torch.is_tensor(out.get("global_voxel_coords")):
        out["global_voxel_coords"] = out["global_voxel_coords"].to(device=device)
    return out
