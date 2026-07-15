"""ana_den6候補をtrain中にlazy生成し、full-cloud contextへ接続する。

手動manifestは不要である。各frameを初めて見た時だけ別processでana_den6と
ana_den5_v8を実行し、確定anchorと限定候補をbinary cacheへ保存する。以後のStepは
cacheを読み、Networkが1つのplanだけを選ぶ。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA_VERSION = "ana_den6_online_one_pattern_cache_v4"
SOURCE_NAME = "ana_den6_exact_one_pattern_anchor_online_v4"
OPERATIONS = ("Add", "Prune", "Adjust")
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_GLOBAL_PAYLOAD_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_CACHE_STATS = {"build": 0, "memory_hit": 0, "disk_hit": 0, "invalid": 0}


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


def _dataset_cli_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"8i", "8ivslf"}:
        return "8i"
    if text == "mvub":
        return "MVUB"
    if text == "uvg":
        return "UVG"
    raise RuntimeError(f"ana_den6 onlineが未対応のdatasetである: {value}")


def _checkpoint_identifier(args: Any) -> str:
    candidates = (
        getattr(args, "sparsepcgc_ckpt_dense", ""),
        getattr(args, "sparsepcgc_ckptdir", ""),
        getattr(args, "sparsepcgc_ckpt", ""),
    )
    for raw in candidates:
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.is_file():
            resolved = path.resolve()
            return f"{resolved}:{_sha256_file(resolved)}"
        return text
    return "default_sparsepcgc_dense_checkpoint"


def _heuristic_version(args: Any) -> str:
    return str(getattr(args, "heuristic_guidance_online_heuristic_version", SOURCE_NAME)).strip() or SOURCE_NAME


def _setting_id(args: Any) -> str:
    return (
        f"vs{float(getattr(args, 'sparsepcgc_voxel_size', 1.0)):.12g}"
        f"_pq{int(getattr(args, 'sparsepcgc_pos_quantscale', 1))}"
        f"_ae{int(getattr(args, 'sparsepcgc_scale_ae', 0))}"
        f"_sr{int(getattr(args, 'sparsepcgc_scale_sr', 0))}"
        f"_m{int(getattr(args, 'sparsepcgc_scale_m', 8))}"
    )


def _cache_identity(args: Any, input_file: Path, input_sha256: str) -> dict[str, Any]:
    return {
        "input_file": str(input_file),
        "input_sha256": str(input_sha256),
        "dataset": _dataset_cli_name(getattr(args, "dataname", "8i")),
        "setting_id": _setting_id(args),
        "voxel_size": float(getattr(args, "sparsepcgc_voxel_size", 1.0)),
        "pos_quantscale": int(getattr(args, "sparsepcgc_pos_quantscale", 1)),
        "heuristic_version": _heuristic_version(args),
        "checkpoint_identifier": _checkpoint_identifier(args),
    }


@contextmanager
def _cache_lock(path: Path):
    """process間で同じcacheを同時生成しないための排他lockである。"""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield


def _repo_paths() -> tuple[Path, Path, Path]:
    # .../myNet/models/utils/pointcloud/ana_den6_online.py
    mynet_root = Path(__file__).resolve().parents[3]
    # 訓練経路からSparsePCGCディレクトリを直接参照しない。
    # den5/den6参照版はmyNet/reference/den6へ固定してcheckpoint互換性を管理する。
    analysis_root = mynet_root / "reference" / "den6"
    worker = mynet_root / "tools" / "ana_den6_online_worker.py"
    return mynet_root, analysis_root, worker


def _cache_file(args: Any, input_file: Path, input_sha256: str) -> Path:
    root = Path(
        str(
            getattr(
                args,
                "heuristic_guidance_online_cache_dir",
                "/data/maejima/log/mynet_den6_online_cache",
            )
        )
    ).expanduser().resolve()
    dataset = _dataset_cli_name(getattr(args, "dataname", "8i"))
    stem = input_file.stem.replace(" ", "_")
    name = (
        f"{dataset}_{stem}_{_setting_id(args)}_"
        f"{hashlib.sha256(json.dumps(_cache_identity(args, input_file, input_sha256), sort_keys=True).encode()).hexdigest()[:16]}.pt"
    )
    return root / name


def _worker_prefix(args: Any) -> list[str]:
    explicit_python = str(getattr(args, "heuristic_guidance_online_python", "")).strip()
    if explicit_python:
        python_path = Path(explicit_python).expanduser().resolve()
        if not python_path.is_file():
            raise RuntimeError(
                "--heuristic_guidance_online_pythonが存在しない: "
                f"{python_path}"
            )
        return [str(python_path)]

    conda_env = str(getattr(args, "heuristic_guidance_online_conda_env", "sparsepcgc")).strip()
    if conda_env:
        conda = shutil.which("conda")
        if conda is None:
            raise RuntimeError(
                "ana_den6 online worker用condaが見つからない。"
                "--heuristic_guidance_online_pythonでMinkowskiEngine環境のPythonを指定すること"
            )
        return [conda, "run", "--no-capture-output", "-n", conda_env, "python"]
    return [sys.executable]


def _worker_command(
    args: Any,
    *,
    input_file: Path,
    output_cache: Path,
    analysis_root: Path,
    worker: Path,
) -> list[str]:
    data = _dataset_cli_name(getattr(args, "dataname", "8i"))
    build_root = output_cache.parent / "build" / output_cache.stem
    command = _worker_prefix(args) + [
        str(worker),
        "--analysis-root", str(analysis_root),
        "--data", data,
        "--input-file", str(input_file),
        "--scale-m", str(int(getattr(args, "sparsepcgc_scale_m", 8))),
        "--scale-ae", str(int(getattr(args, "sparsepcgc_scale_ae", 0))),
        "--scale-sr", str(int(getattr(args, "sparsepcgc_scale_sr", 0))),
        "--setting-id", _setting_id(args),
        "--voxel-size", str(float(getattr(args, "sparsepcgc_voxel_size", 1.0))),
        "--pos-quantscale", str(int(getattr(args, "sparsepcgc_pos_quantscale", 1))),
        "--heuristic-version", _heuristic_version(args),
        "--checkpoint-identifier", _checkpoint_identifier(args),
        "--max-total-ratio", str(float(getattr(args, "heuristic_guidance_online_max_total_ratio", 0.0099))),
        "--pool-limit-per-operation", str(int(getattr(args, "heuristic_guidance_online_pool_limit", 512))),
        "--compact-reserve-factor", str(int(getattr(args, "heuristic_guidance_online_compact_reserve_factor", 8))),
        "--output-cache", str(output_cache),
        "--output-root", str(build_root),
    ]
    ratio_percent = float(getattr(args, "heuristic_guidance_total_ratio_percent", -1.0))
    if ratio_percent >= 0.0:
        command.extend(("--total-ratio", str(ratio_percent / 100.0)))
    shares = str(getattr(args, "heuristic_guidance_operation_shares", "")).strip()
    if shares:
        command.extend(("--operation-shares", shares))
    device_mode = str(getattr(args, "heuristic_guidance_online_worker_device", "cpu")).strip().lower()
    if device_mode == "cpu":
        command.append("--cpu")
    elif device_mode not in {"auto", "cuda"}:
        raise RuntimeError(
            "heuristic_guidance_online_worker_deviceはcpu/auto/cudaのいずれかである"
        )
    return command


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    args: Any,
    input_file: Path,
    input_sha256: str,
    analysis_root: Path,
) -> None:
    if str(payload.get("schema_version", "")) != SCHEMA_VERSION:
        raise RuntimeError("ana_den6 online cache schemaが不正である")
    if str(payload.get("source", "")) != SOURCE_NAME:
        raise RuntimeError("ana_den6 online cache sourceが不正である")
    if Path(str(payload.get("input_file", ""))).resolve() != input_file:
        raise RuntimeError("ana_den6 online cacheと現在の入力PLYが一致しない")
    if str(payload.get("input_sha256", "")) != input_sha256:
        raise RuntimeError("ana_den6 online cacheと入力PLYのSHA256が一致しない")
    if str(payload.get("setting_id", "")) != _setting_id(args):
        raise RuntimeError("ana_den6 online cacheとAE/SR/m設定が一致しない")
    identity = _cache_identity(args, input_file, input_sha256)
    for name in ("dataset", "voxel_size", "pos_quantscale", "heuristic_version", "checkpoint_identifier"):
        if str(payload.get(name, "")) != str(identity[name]):
            raise RuntimeError(f"ana_den6 online cache identity不一致: {name}")
    den6_path = analysis_root / "ana_den6.py"
    den5_path = analysis_root / "ana_den5_v8.py"
    if not den6_path.is_file() or not den5_path.is_file():
        raise RuntimeError("現在のana_den6.py/ana_den5_v8.pyを検証できない")
    if str(payload.get("den6_sha256", "")) != _sha256_file(den6_path):
        raise RuntimeError("cache生成時と現在のana_den6.pyが一致しない")
    if str(payload.get("den5_sha256", "")) != _sha256_file(den5_path):
        raise RuntimeError("cache生成時と現在のana_den5_v8.pyが一致しない")
    shortlists = payload.get("operation_candidate_shortlists")
    if not isinstance(shortlists, Mapping):
        raise RuntimeError("ana_den6 online cacheに局所候補shortlistが無い")
    anchor = payload.get("initial_heuristic_plan")
    if not isinstance(anchor, Mapping) or not bool(anchor.get("available", False)):
        raise RuntimeError("ana_den6 online cacheに確定anchor planが無い")
    for operation in OPERATIONS:
        candidates = shortlists.get(operation)
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(f"ana_den6 online cacheの{operation}局所候補が空である")



def _merge_timing_sidecar(payload: dict[str, Any], cache_path: Path) -> dict[str, Any]:
    """大きなcacheを再保存せず記録した実測保存時間をpayloadへ統合する。"""
    sidecar = cache_path.with_suffix(cache_path.suffix + ".timing.json")
    if not sidecar.is_file():
        return payload
    try:
        timing = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if isinstance(timing, Mapping):
        payload["runtime_breakdown_sec"] = {
            **dict(payload.get("runtime_breakdown_sec") or {}),
            **{str(key): float(value) for key, value in timing.items()},
        }
    return payload

def _load_or_build_payload(args: Any) -> dict[str, Any]:
    input_file = Path(str(getattr(args, "_current_input_file", ""))).expanduser().resolve()
    if not input_file.is_file():
        raise RuntimeError("ana_den6 onlineでは各Stepのargs._current_input_fileに実在PLYが必要である")
    input_sha256 = _sha256_file(input_file)
    _, analysis_root, worker = _repo_paths()
    if not worker.is_file():
        raise RuntimeError(f"ana_den6 online workerが存在しない: {worker}")
    output_cache = _cache_file(args, input_file, input_sha256)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    memory_key = str(output_cache)

    cached = _GLOBAL_PAYLOAD_CACHE.get(memory_key)
    if isinstance(cached, dict):
        _validate_payload(cached, args=args, input_file=input_file, input_sha256=input_sha256, analysis_root=analysis_root)
        _GLOBAL_PAYLOAD_CACHE.move_to_end(memory_key)
        _CACHE_STATS["memory_hit"] += 1
        setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
        print(f"[ana_den6_online] cache hit: {input_file.name} (memory)", flush=True)
        return cached

    with _cache_lock(output_cache):
        payload = None
        if output_cache.is_file():
            try:
                candidate = torch.load(output_cache, map_location="cpu", weights_only=False)
                _validate_payload(candidate, args=args, input_file=input_file, input_sha256=input_sha256, analysis_root=analysis_root)
                payload = _merge_timing_sidecar(dict(candidate), output_cache)
                _CACHE_STATS["disk_hit"] += 1
                print(f"[ana_den6_online] cache hit: {input_file.name} (disk)", flush=True)
            except Exception as exc:
                _CACHE_STATS["invalid"] += 1
                invalid = output_cache.with_suffix(output_cache.suffix + f".invalid.{int(time.time())}")
                output_cache.replace(invalid)
                print(f"[ana_den6_online] cache invalid: {input_file.name}: {type(exc).__name__}: {exc}; moved={invalid.name}", flush=True)

        if payload is None:
            command = _worker_command(args, input_file=input_file, output_cache=output_cache, analysis_root=analysis_root, worker=worker)
            print(f"[ana_den6_online] 初回frameのden6候補cacheを生成する: {input_file.name}", flush=True)
            completed = subprocess.run(command, cwd=str(worker.parent.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=dict(os.environ))
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()[-8000:]
                raise RuntimeError(f"ana_den6 online candidate生成に失敗した。 returncode={completed.returncode}\n{detail}")
            if not output_cache.is_file():
                raise RuntimeError("ana_den6 online workerがbinary cacheを出力しなかった")
            payload = torch.load(output_cache, map_location="cpu", weights_only=False)
            payload = _merge_timing_sidecar(dict(payload), output_cache)
            _validate_payload(payload, args=args, input_file=input_file, input_sha256=input_sha256, analysis_root=analysis_root)
            _CACHE_STATS["build"] += 1

    payload["cache_path"] = str(output_cache)
    payload["cache_signature"] = hashlib.sha256(json.dumps(_cache_identity(args, input_file, input_sha256), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    _GLOBAL_PAYLOAD_CACHE[memory_key] = payload
    _GLOBAL_PAYLOAD_CACHE.move_to_end(memory_key)
    max_entries = max(int(getattr(args, "heuristic_guidance_online_memory_entries", 4)), 1)
    while len(_GLOBAL_PAYLOAD_CACHE) > max_entries:
        _GLOBAL_PAYLOAD_CACHE.popitem(last=False)
    setattr(args, "_ana_den6_online_cache_stats", dict(_CACHE_STATS))
    return payload


def attach_ana_den6_online_guidance(
    context: Mapping[str, Any],
    args: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """online候補cacheをfull-cloud contextへ付与する。"""
    mode = str(getattr(args, "heuristic_guidance_mode", "proxy_prior")).strip().lower()
    if mode != "ana_den6_online":
        return dict(context)
    if not isinstance(context, Mapping):
        raise RuntimeError("ana_den6 onlineにはfull-cloud canonical contextが必要である")
    coords = context.get("full_global_voxel_coords", context.get("global_voxel_coords"))
    if not torch.is_tensor(coords):
        raise RuntimeError("ana_den6 online contextにglobal_voxel_coordsが無い")
    if coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[1] != 3:
        raise RuntimeError(
            "ana_den6 onlineはbatch=1のfull-cloud canonical voxelだけを受け付ける: "
            f"shape={tuple(coords.shape)}"
        )
    payload = _load_or_build_payload(args)
    out = dict(context)
    out["ana_den6_ranked_candidate_guidance"] = payload
    out["ana_den6_online_cache_path"] = str(payload.get("cache_path", ""))
    out["ana_den6_online_cache_used"] = True
    # cacheはCPU objectとして保持し、座標Tensorだけ既存contextのdeviceを維持する。
    if torch.is_tensor(out.get("global_voxel_coords")):
        out["global_voxel_coords"] = out["global_voxel_coords"].to(device=device)
    return out
