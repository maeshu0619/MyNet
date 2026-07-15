"""ana_den6のcandidate planをmyNetのstrict hard voxel anchorへ接続する。

``ana_den6_reproduce`` はden6本体が出力したplan manifestだけを受け付ける。
保存済み編集PLYを直接使う旧経路は ``ana_den6_reference_ply`` と明示的に分離し、
候補生成・順位・衝突回避を再現したものとして扱わない。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from models.utils.io.utils_ply import read_ply


_REFERENCE_SHARES = {
    ("8i", 8): (0.40, 0.40, 0.20),
    ("mvub", 8): (0.50, 0.40, 0.10),
    ("uvg", 7): (0.40, 0.50, 0.10),
}


def _dataset_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "8i" if text in {"8i", "8ivslf"} else text


def _analysis_root(dataset: str) -> Path:
    suffix = {"8i": "8i", "mvub": "MVUB", "uvg": "UVG"}[dataset]
    return Path(f"/data/maejima/log/SparsePCGC_dense_mixed_edit_analysis_den6_{suffix}")


def _setting_id(args: Any) -> str:
    return (
        "native_vs1_pq1"
        f"_ae{int(getattr(args, 'sparsepcgc_scale_ae', 0))}"
        f"_sr{int(getattr(args, 'sparsepcgc_scale_sr', 0))}"
        f"_m{int(getattr(args, 'sparsepcgc_scale_m', 8))}"
    )


def _load_reference(args: Any) -> dict[str, Any]:
    """同一入力・設定のden6 0.25% actual行だけを返す。"""
    dataset = _dataset_key(getattr(args, "dataname", ""))
    scale_m = int(getattr(args, "sparsepcgc_scale_m", 8))
    shares = _REFERENCE_SHARES.get((dataset, scale_m))
    if shares is None:
        raise RuntimeError(f"ana_den6厳密anchorの参照profileが無い: dataset={dataset}, m={scale_m}")

    source_file = Path(str(getattr(args, "_current_input_file", ""))).resolve()
    if not source_file.is_file():
        raise RuntimeError("ana_den6厳密anchorで現在の入力PLYを特定できない")
    state_path = _analysis_root(dataset) / "state" / "run_rows.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    matches = []
    for row in state.get("actual_rows", []):
        if Path(str(row.get("input_file", ""))).resolve() != source_file:
            continue
        if str(row.get("setting_id", "")) != _setting_id(args):
            continue
        if not math.isclose(float(row.get("total_ratio_percent", -1.0)), 0.25, abs_tol=1e-12):
            continue
        if not all(
            math.isclose(float(row.get(key, -1.0)), expected, abs_tol=1e-12)
            for key, expected in zip(("add_share", "prune_share", "adjust_share"), shares)
        ):
            continue
        matches.append(dict(row))
    if len(matches) != 1:
        raise RuntimeError(
            "ana_den6厳密anchorの参照actual行を一意に特定できない: "
            f"input={source_file}, setting={_setting_id(args)}, matches={len(matches)}"
        )
    return matches[0]


def _read_ply_coords(path: Path) -> torch.Tensor:
    with path.open("rb") as stream:
        header_lines = 0
        format_name = ""
        for raw_line in stream:
            header_lines += 1
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("format "):
                format_name = line.split(maxsplit=2)[1]
            if line == "end_header":
                break
        else:
            raise RuntimeError(f"ana_den6編集PLYのheaderが閉じていない: {path}")
    try:
        if format_name == "ascii":
            values = np.loadtxt(path, dtype=np.float64, skiprows=header_lines, usecols=(0, 1, 2))
        else:
            data = read_ply(path)
            values = np.column_stack((data["x"], data["y"], data["z"]))
    except (KeyError, ValueError, OSError) as exc:
        raise RuntimeError(f"ana_den6編集PLYのxyzを読めない: {path}") from exc
    values = np.atleast_2d(values)
    coords = np.unique(np.rint(values).astype(np.int64, copy=False), axis=0)
    return torch.from_numpy(coords.T.copy()).unsqueeze(0).contiguous()


def _coord_hash(coords_b3n: torch.Tensor) -> str:
    coords = coords_b3n.detach().cpu().to(dtype=torch.long)
    if coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[1] != 3:
        raise RuntimeError(f"canonical voxel座標shapeが不正: {tuple(coords.shape)}")
    values = torch.unique(coords[0].transpose(0, 1), dim=0, sorted=True).numpy()
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_den6_sha256() -> str:
    """manifest生成時と同じana_den6.pyであることを必須にする。"""
    den6_path = Path(__file__).resolve().parents[4] / "compress" / "octree" / "SparsePCGC" / "ana_den6.py"
    if not den6_path.is_file():
        raise RuntimeError(f"ana_den6.pyが存在しないため厳密planを検証できない: {den6_path}")
    return _sha256_file(den6_path)


def _manifest_final_coords(
    manifest_path: Path,
    *,
    args: Any,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """den6が出力したcandidate planを初期canonical集合へ厳密に適用する。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("schema_version", "")) != "ana_den6_mixed_plan_manifest_v1":
        raise RuntimeError(f"ana_den6 plan manifest schemaが不正: {manifest_path}")
    if str(manifest.get("den6_sha256", "")) != _current_den6_sha256():
        raise RuntimeError("ana_den6 plan manifest生成時と現在のana_den6.pyのSHA256が一致しない")
    input_file = Path(str(getattr(args, "_current_input_file", ""))).resolve()
    if Path(str(manifest.get("input_file", ""))).resolve() != input_file:
        raise RuntimeError("ana_den6 plan manifestとtrain入力PLYが一致しない")
    if str(manifest.get("setting_id", "")) != _setting_id(args):
        raise RuntimeError("ana_den6 plan manifestとAE/SR/m設定が一致しない")
    if str(manifest.get("input_sha256", "")) != _sha256_file(input_file):
        raise RuntimeError("ana_den6 plan manifestとtrain入力PLYのSHA256が一致しない")

    initial_rows = torch.unique(
        initial.detach().cpu().to(dtype=torch.long)[0].transpose(0, 1), dim=0, sorted=True
    ).numpy()
    occupied = {tuple(int(value) for value in row) for row in initial_rows}
    removes: set[tuple[int, int, int]] = set()
    adds: set[tuple[int, int, int]] = set()
    operation_counts = {"Add": 0, "Prune": 0, "Adjust": 0}
    for candidate in manifest.get("selected_candidates", []):
        operation = str(candidate.get("operation", ""))
        if operation not in operation_counts:
            raise RuntimeError(f"ana_den6 plan manifestに未知Actionがある: {operation}")
        candidate_removes = {
            tuple(int(value) for value in coord) for coord in candidate.get("remove_coords", [])
        }
        candidate_adds = {
            tuple(int(value) for value in coord) for coord in candidate.get("add_coords", [])
        }
        if operation == "Add" and (candidate_removes or not candidate_adds):
            raise RuntimeError("ana_den6 plan manifestのAdd候補定義が不正")
        if operation == "Prune" and (not candidate_removes or candidate_adds):
            raise RuntimeError("ana_den6 plan manifestのPrune候補定義が不正")
        if operation == "Adjust" and (not candidate_removes or not candidate_adds):
            raise RuntimeError("ana_den6 plan manifestのAdjust候補定義が不正")
        if not candidate_removes.issubset(occupied) or candidate_adds & occupied:
            raise RuntimeError("ana_den6 plan manifestの候補が入力canonical voxel集合に適用不能")
        if removes & candidate_removes or adds & candidate_adds or removes & candidate_adds or adds & candidate_removes:
            raise RuntimeError("ana_den6 plan manifestにden6衝突回避違反がある")
        removes.update(candidate_removes)
        adds.update(candidate_adds)
        operation_counts[operation] += 1
    expected_counts = manifest.get("selected_operation_counts", {})
    if {name: int(expected_counts.get(name, -1)) for name in operation_counts} != operation_counts:
        raise RuntimeError("ana_den6 plan manifestのAction件数がcandidate列と一致しない")
    final_rows = np.asarray(sorted((occupied - removes) | adds), dtype=np.int64)
    final = torch.from_numpy(final_rows.T.copy()).unsqueeze(0).contiguous()
    expected_hash = str(manifest.get("final_voxel_hash", ""))
    if _coord_hash(final) != expected_hash:
        raise RuntimeError("ana_den6 plan manifestの最終voxel hashが再構築結果と一致しない")
    return final, manifest


def attach_ana_den6_reference_anchor(
    context: Mapping[str, Any], args: Any, *, device: torch.device
) -> dict[str, Any]:
    """検証済みden6 hard voxel集合をfull-cloud actual経路へ安全に付与する。"""
    mode = str(getattr(args, "heuristic_guidance_mode", "proxy_prior")).strip().lower()
    if mode not in {"ana_den6_reproduce", "ana_den6_reference_ply"}:
        return dict(context)
    if not isinstance(context, Mapping):
        raise RuntimeError("ana_den6厳密anchorにはfull cloud canonical contextが必要")

    initial = context.get("full_global_voxel_coords", context.get("global_voxel_coords"))
    if not torch.is_tensor(initial):
        raise RuntimeError("ana_den6厳密anchorで入力canonical voxel座標が無い")
    initial = initial.detach().to(dtype=torch.long)
    if initial.ndim != 3 or initial.shape[0] != 1 or initial.shape[1] != 3:
        raise RuntimeError(f"ana_den6厳密anchorはbatch=1のfull cloudだけを受け付ける: {tuple(initial.shape)}")
    if int(torch.unique(initial[0].transpose(0, 1), dim=0).shape[0]) <= 0:
        raise RuntimeError("ana_den6厳密anchorの入力voxel集合が空である")

    if mode == "ana_den6_reproduce":
        manifest_path = Path(str(getattr(args, "heuristic_guidance_den6_plan_manifest", ""))).expanduser()
        if not str(manifest_path) or not manifest_path.is_file():
            raise RuntimeError(
                "ana_den6_reproduceには--heuristic_guidance_den6_plan_manifestで、"
                "tools/ana_den6_reproduce.pyが出力したplan manifestを指定する必要がある"
            )
        # 同じargsを複数frameで共有しても、別入力へplanを誤適用しない。
        cache_key = "|".join((
            str(manifest_path.resolve()),
            str(Path(str(getattr(args, "_current_input_file", ""))).resolve()),
            _coord_hash(initial),
        ))
        cache = getattr(args, "_ana_den6_reference_anchor_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(args, "_ana_den6_reference_anchor_cache", cache)
        cached = cache.get(cache_key)
        if not isinstance(cached, dict):
            final, manifest = _manifest_final_coords(manifest_path.resolve(), args=args, initial=initial)
            cached = {"coords": final.cpu(), "manifest": manifest}
            cache[cache_key] = cached
        edited = cached["coords"].to(device=device, dtype=torch.long, non_blocking=True)
        manifest = cached["manifest"]
        actual_reference = manifest.get("reference_actual", {})
        if not isinstance(actual_reference, Mapping):
            actual_reference = {}
        reference = {
            "input_file": manifest["input_file"],
            "edited_ply": "",
            "actual_saved_percent": float(actual_reference.get("actual_saved_percent", float("nan"))),
            "baseline_decoder_complete_bits": float(actual_reference.get("baseline_decoder_complete_bits", float("nan"))),
            "edited_decoder_complete_bits": float(actual_reference.get("edited_decoder_complete_bits", float("nan"))),
            "actual_add_operation_count": int(manifest["selected_operation_counts"]["Add"]),
            "actual_prune_operation_count": int(manifest["selected_operation_counts"]["Prune"]),
            "actual_adjust_operation_count": int(manifest["selected_operation_counts"]["Adjust"]),
        }
        anchor_source = "ana_den6_candidate_plan_manifest"
        final_hash = str(manifest["final_voxel_hash"])
    else:
        reference = _load_reference(args)
        cache_key = str(Path(str(reference["edited_ply"])).resolve())
        cache = getattr(args, "_ana_den6_reference_anchor_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(args, "_ana_den6_reference_anchor_cache", cache)
        cached = cache.get(cache_key)
        if not isinstance(cached, dict):
            edited_file = Path(str(reference["edited_ply"])).resolve()
            if not edited_file.is_file():
                raise RuntimeError(f"ana_den6参照PLYが存在しない: {edited_file}")
            cached = {"coords": _read_ply_coords(edited_file).cpu(), "reference": reference}
            cache[cache_key] = cached
        edited = cached["coords"].to(device=device, dtype=torch.long, non_blocking=True)
        anchor_source = "saved_den6_final_ply_not_candidate_plan"
        final_hash = _coord_hash(edited)
    out = dict(context)
    out.update(
        {
            "actual_oracle_enabled": True,
            "actual_oracle_override_final_voxel_coords": edited,
            "actual_oracle_override_add_count": int(reference["actual_add_operation_count"]),
            "actual_oracle_override_drop_count": int(reference["actual_prune_operation_count"]),
            "actual_oracle_override_move_count": int(reference["actual_adjust_operation_count"]),
            "actual_oracle_override_subtree_prune_count": 0,
            "actual_oracle_override_scope": "full_cloud",
            "ana_den6_reference_anchor_used": True,
            "ana_den6_reference_input_file": str(reference["input_file"]),
            "ana_den6_reference_edited_ply": str(reference["edited_ply"]),
            "ana_den6_reference_final_voxel_hash": final_hash,
            "ana_den6_reference_anchor_source": anchor_source,
            "ana_den6_reference_saved_percent": float(reference["actual_saved_percent"]),
            "ana_den6_reference_baseline_bits": float(reference["baseline_decoder_complete_bits"]),
            "ana_den6_reference_edited_bits": float(reference["edited_decoder_complete_bits"]),
        }
    )
    setattr(args, "_ana_den6_reference_anchor_active", True)
    setattr(args, "_ana_den6_reference_anchor_source", anchor_source)
    setattr(args, "_ana_den6_reference_expected_saved_percent", float(reference["actual_saved_percent"]))
    setattr(args, "_ana_den6_reference_input_file", str(reference["input_file"]))
    setattr(
        args,
        "_ana_den6_reference_operation_counts",
        {
            "Add": int(reference["actual_add_operation_count"]),
            "Prune": int(reference["actual_prune_operation_count"]),
            "Adjust": int(reference["actual_adjust_operation_count"]),
        },
    )
    return out
