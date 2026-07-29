#!/usr/bin/env python3
"""保存済みden6/Actual結果から訓練専用Layer A cacheを構築する。"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import zlib
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.utils.cache.exact_teacher_cache import ExactTeacherCache, build_fingerprint
from models.utils.data.dataset import load_ply


def _voxel_payload(input_file, voxel_size, pos_quantscale):
    points = np.asarray(load_ply(str(input_file), return_color=False), dtype=np.float64)[:, :3]
    voxels = np.rint(points / max(float(voxel_size), 1e-12))
    voxels = np.rint(voxels / max(float(pos_quantscale), 1e-12)).astype(np.int32)
    voxels = np.unique(voxels, axis=0)
    raw = np.ascontiguousarray(voxels).tobytes(order="C")
    return {
        "dtype": "int32",
        "shape": list(voxels.shape),
        "encoding": "zlib+base64",
        "data": base64.b64encode(zlib.compress(raw, level=6)).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bbox_min": voxels.min(0).tolist(),
        "bbox_max": voxels.max(0).tolist(),
    }


def run(args):
    records_by_state = {}
    source_by_state = {}
    for source_path in args.input:
        with gzip.open(source_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("contains_virtual_actual_labels", True):
            raise RuntimeError("推定ActualをLayer Aへ保存できない")
        for record in payload.get("records", ()):
            state = dict(record["state_key"])
            state_id = "{}|{}".format(state["input_sha256"], state["setting_id"])
            records_by_state.setdefault(state_id, []).append(record)
            source_by_state.setdefault(state_id, set()).add(str(Path(source_path).resolve()))

    cache = ExactTeacherCache(args.output_root)
    results = []
    sparse_root = Path(args.sparsepcgc_root).resolve()
    den5 = sparse_root / "ana_den5_v8.py"
    den6 = sparse_root / "ana_den6.py"
    checkpoint = Path(args.codec_checkpoint).resolve()
    online_root = Path(args.den6_online_cache_root).expanduser().resolve()
    state_items = sorted(records_by_state.items())
    if int(args.max_states) > 0:
        state_items = state_items[: int(args.max_states)]
    for state_id, records in state_items:
        state = dict(records[0]["state_key"])
        input_file = str(Path(state["input_file"]).resolve())
        online_payload = None
        online_path = None
        if online_root.is_dir():
            for candidate_path in online_root.glob("exact_single_plan_*.json"):
                try:
                    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if (
                    str(candidate_payload.get("input_sha256", "")) == str(state["input_sha256"])
                    and str(candidate_payload.get("setting_id", "")) == str(state["setting_id"])
                ):
                    online_payload = candidate_payload
                    online_path = candidate_path
                    break
        sources = [str(den5), str(den6), str(checkpoint), *sorted(source_by_state[state_id])]
        if online_path is not None:
            sources.append(str(online_path))
        candidate_pools = dict(
            (online_payload or {}).get("operation_candidate_pools") or {}
        )
        if not candidate_pools:
            candidate_pools = dict(
                (online_payload or {}).get("operation_candidate_shortlists") or {}
            )
        full_pool_counts = dict(
            (online_payload or {}).get("full_pool_counts") or {}
        )
        candidate_pool_complete = bool(candidate_pools) and all(
            len(candidate_pools.get(name) or ())
            == int(full_pool_counts.get(name, -1))
            and len(candidate_pools.get(name) or ()) > 0
            for name in ("Add", "Prune", "Adjust")
        )
        codec = {
            key: state.get(key) for key in (
                "codec_mode", "setting_id", "scale_m", "scale_ae", "scale_sr",
                "voxel_size", "pos_quantscale", "native_resolution",
            )
        }
        geometry = {
            "label": "formal_D1_D2_loss_db",
            "actual_only": True,
            "missing_values_are_not_imputed": True,
        }
        fingerprint = build_fingerprint(
            input_path=input_file,
            codec=codec,
            source_files=sources,
            geometry=geometry,
        )
        try:
            loaded = cache.load(fingerprint)
        except FileNotFoundError:
            loaded = None
        if loaded is not None:
            results.append({
                "state_id": state_id,
                "path": str(cache.path_for(fingerprint["fingerprint_sha256"])),
                "cache_hit_verified": True,
                "cold_build": False,
                "octree_rebuild_count": 0,
                "fixed_feature_rebuild_count": 0,
                "teacher_actual_encode_count": 0,
                "gt_actual_encode_count": 0,
                "actual_plan_count": int(loaded["content"]["actual_plan_count"]),
                "voxel_count": int(loaded["content"]["canonical_occupied_voxels"]["shape"][0]),
            })
            continue
        content = {
            "state_id": state_id,
            "state": state,
            "canonical_occupied_voxels": _voxel_payload(
                input_file, state.get("voxel_size", 1.0), state.get("pos_quantscale", 1)
            ),
            "baseline_bits": sorted({float(row["baseline_bits"]) for row in records}),
            "actual_plans": records,
            "actual_plan_count": len(records),
            "contains_only_measured_actual": True,
            "actual_label_interpolation": False,
            "teacher_hard_apply_allowed": False,
            "den6_exact_candidate_cache": online_payload,
            "operation_candidate_pools": candidate_pools,
            "candidate_pool_complete": candidate_pool_complete,
            "candidate_shortlist_available": online_payload is not None,
            "missing_fields": {
                "add_source": True,
                "add_direction": True,
                "reject_reason": True,
                "full_candidate_pool": not candidate_pool_complete,
                "octree_fixed_features": True,
            },
        }
        path = cache.write(fingerprint, content)
        loaded = cache.load(fingerprint)
        results.append({
            "state_id": state_id,
            "path": str(path),
            "cache_hit_verified": True,
            "cold_build": True,
            "actual_plan_count": int(loaded["content"]["actual_plan_count"]),
            "voxel_count": int(loaded["content"]["canonical_occupied_voxels"]["shape"][0]),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=(
        "/data/maejima/log/mynet_kproposal_offline/8i_actual_plans.json.gz",
        "/data/maejima/log/mynet_kproposal_offline/mvub_actual_plans.json.gz",
        "/data/maejima/log/mynet_kproposal_offline/uvg_actual_plans.json.gz",
    ))
    parser.add_argument("--output-root", default="/data/maejima/log/mynet_exact_teacher_cache")
    parser.add_argument("--max-states", default=0, type=int)
    parser.add_argument(
        "--den6-online-cache-root",
        default="/data/maejima/log/mynet_den6_online_cache",
    )
    parser.add_argument(
        "--sparsepcgc-root",
        default="/home/maejima/MasterEx/compress/octree/SparsePCGC",
    )
    parser.add_argument(
        "--codec-checkpoint",
        default="/home/maejima/MasterEx/compress/octree/SparsePCGC/ckpts/dense/epoch_last.pth",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
