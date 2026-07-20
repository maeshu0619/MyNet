#!/usr/bin/env python3
"""Reconstruct den6 executed plans without repeating candidate Actual encodes.

This is an *offline diagnostic/conversion* tool.  It imports the immutable
SparsePCGC analysis module, rebuilds the candidate pools once, and replaces
only den6's final Actual-evaluation callback with a validator which joins the
already measured rows from ``state/run_rows.json``.  It is never imported by
train.py, test.py, or a policy module.

The conversion fails closed when the rebuilt member list does not reproduce
the recorded den6 plan key.  Consequently a truncated ``candidate_ids_sample``
is never treated as a complete plan and estimated/virtual rows are never
promoted to Actual teachers.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


OPERATIONS = ("Add", "Prune", "Adjust")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    # Python 3.8 dataclasses resolves postponed annotations through
    # ``sys.modules[cls.__module__]`` while the class decorator runs.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _candidate_row(candidate):
    def value(name, default=0.0):
        raw = getattr(candidate, name, default)
        if isinstance(raw, np.generic):
            raw = raw.item()
        return raw

    return {
        "candidate_id": str(value("candidate_id", "")),
        "operation": str(value("operation", "")),
        "remove_coords": [list(map(int, row)) for row in value("remove_coords", ())],
        "add_coords": [list(map(int, row)) for row in value("add_coords", ())],
        "symbol_index": int(value("symbol_index", -1)),
        "partner_symbol_index": int(value("partner_symbol_index", -1)),
        "depth": int(value("depth", -1)),
        "region_shift": int(value("region_shift", 0)),
        "fixed_context_gain_bits": float(value("fixed_context_gain_bits", 0.0)),
        "subtree_bit_mass": float(value("subtree_bit_mass", 0.0)),
        "expected_new_descendant_bits": float(value("expected_new_descendant_bits", 0.0)),
        "mask_gain_bits": float(value("mask_gain_bits", 0.0)),
        "neighbor_bit_risk": float(value("neighbor_bit_risk", 0.0)),
        "geometry_cost": float(value("geometry_cost", 0.0)),
        "optimistic_gain_bits": float(value("optimistic_gain_bits", 0.0)),
        "heuristic_score": float(value("heuristic_score", 0.0)),
    }


def _coord_hash(coords):
    packed = np.asarray(coords, dtype=np.int64)
    if packed.ndim != 2 or packed.shape[1] != 3:
        raise RuntimeError("invalid final coordinates: {}".format(packed.shape))
    order = np.lexsort((packed[:, 2], packed[:, 1], packed[:, 0]))
    return hashlib.sha256(
        np.ascontiguousarray(packed[order], dtype=np.int64).tobytes(order="C")
    ).hexdigest()


def _actual_lookup(run_rows_path):
    payload = json.loads(Path(run_rows_path).read_text(encoding="utf-8"))
    lookup = {}
    for row in payload.get("actual_rows", ()):
        try:
            gain = float(row.get("actual_saved_percent", float("nan")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(gain):
            continue
        key = (
            str(Path(str(row["input_file"])).resolve()),
            str(row["setting_id"]),
            str(row["pattern_key"]),
        )
        if key in lookup:
            raise RuntimeError("duplicate Actual row: {}".format(key))
        lookup[key] = dict(row)
    if not lookup:
        raise RuntimeError("no finite Actual rows: {}".format(run_rows_path))
    return payload, lookup


def reconstruct(args):
    sparse_root = Path(args.sparsepcgc_root).expanduser().resolve()
    den6_path = sparse_root / "ana_den6.py"
    if not den6_path.is_file():
        raise RuntimeError("ana_den6.py is missing: {}".format(den6_path))
    source_payload, actual_rows = _actual_lookup(args.run_rows)
    preflight = dict(source_payload.get("preflight") or {})
    expected_den6 = str(preflight.get("ana_den6_sha256", ""))
    current_den6 = _sha256_file(den6_path)
    if expected_den6 and expected_den6 != current_den6:
        raise RuntimeError(
            "den6 SHA256 mismatch: recorded={} current={}".format(
                expected_den6, current_den6
            )
        )

    den6 = _load_module(den6_path, "mynet_offline_den6")
    reconstructed = []
    seen = set()

    def evaluate_without_actual(
        den5,
        engine,
        codec_base,
        coder,
        input_item,
        input_file,
        setting,
        baseline_codec,
        baseline_probe,
        pattern_row,
        unique_voxel_count,
        plan,
        plan_meta,
        den6_args,
        root,
    ):
        key = (
            str(Path(input_file).resolve()),
            str(setting.setting_id),
            str(pattern_row["pattern_key"]),
        )
        measured = actual_rows.get(key)
        if measured is None:
            raise RuntimeError("rebuilt pattern has no recorded Actual row: {}".format(key))

        members = sorted(den5._candidate_edit_key(item) for item in plan)
        plan_key = den6._stable_hash(
            {"pattern_key": pattern_row["pattern_key"], "members": members}
        )
        if plan_key != str(measured.get("plan_key", "")):
            raise RuntimeError(
                "plan-key mismatch for {}: rebuilt={} recorded={}".format(
                    key, plan_key, measured.get("plan_key")
                )
            )

        final_coords, removes, adds = den5._apply_actual_plan_fast(
            baseline_probe.original_coords, plan
        )
        operation_counts = {name: 0 for name in OPERATIONS}
        candidates = []
        for item in plan:
            row = _candidate_row(item)
            if row["operation"] not in operation_counts:
                raise RuntimeError("unknown operation: {}".format(row["operation"]))
            operation_counts[row["operation"]] += 1
            candidates.append(row)
        recorded_counts = {
            name: int(dict(measured.get("selected_counts") or {}).get(name, -1))
            for name in OPERATIONS
        }
        if operation_counts != recorded_counts:
            raise RuntimeError(
                "operation-count mismatch for {}: rebuilt={} recorded={}".format(
                    key, operation_counts, recorded_counts
                )
            )

        record = {
            "state_key": {
                "input_file": key[0],
                "input_sha256": _sha256_file(input_file),
                "setting_id": key[1],
                "scale_m": int(setting.scale_m),
                "scale_ae": int(setting.scale_ae),
                "scale_sr": int(setting.scale_sr),
                "voxel_size": float(getattr(den6_args, "voxel_size", 1.0)),
                "pos_quantscale": int(getattr(den6_args, "posQuantscale", 1)),
                "native_resolution": int(den6_args.native_resolution),
                "codec_mode": "dense_lossy",
            },
            "pattern_key": key[2],
            "plan_key": plan_key,
            "operation_order": str(plan_meta.get("operation_order", "")),
            "operation_counts": operation_counts,
            "requested_counts": {
                name: int(measured.get("requested_{}_count".format(name.lower()), 0))
                for name in OPERATIONS
            },
            "total_ratio_percent": float(measured["total_ratio_percent"]),
            "shares": {
                "Add": float(measured["add_share"]),
                "Prune": float(measured["prune_share"]),
                "Adjust": float(measured["adjust_share"]),
            },
            "actual_saved_percent": float(measured["actual_saved_percent"]),
            "actual_gain_percent": float(measured["actual_saved_percent"]),
            "actual_compression_loss_percent": -float(measured["actual_saved_percent"]),
            "baseline_bits": float(measured["baseline_decoder_complete_bits"]),
            "edited_bits": float(measured["edited_decoder_complete_bits"]),
            "geometry": {
                "D1_loss_db": float(measured["D1_loss_db"]),
                "D2_loss_db": float(measured["D2_loss_db"]),
            },
            "estimated_gain_percent": float(measured["estimated_saved_percent"]),
            "interaction_gain_percent": float(measured["interaction_saved_percent"]),
            "removed_voxel_count": int(len(removes)),
            "added_voxel_count": int(len(adds)),
            "final_voxel_hash": _coord_hash(final_coords),
            "candidates": candidates,
        }
        reconstructed.append(record)
        seen.add(key)
        return dict(measured)

    # Do not rewrite the historical den6 state/workbooks.  A private temporary
    # root is still used because the immutable driver writes its manifest and
    # baseline codec artifacts there.
    den6._evaluate_mixed_plan = evaluate_without_actual
    den6._save_state = lambda *unused_args, **unused_kwargs: None
    den6._write_workbooks = lambda *unused_args, **unused_kwargs: (
        "offline_dataset_only",
        "offline_dataset_only",
    )

    temporary_root = Path(
        tempfile.mkdtemp(prefix="mynet_kproposal_reconstruct_", dir=args.tmp_root)
    )
    try:
        argv = [
            "--data", str(preflight["dataset"]),
            "--dataset-input", str(preflight["dataset_input"]),
            "--m-values", ",".join(map(str, preflight["m_values"])),
            "--total-operation-ratios",
            ",".join(map(str, preflight["total_operation_ratios"])),
            "--mix-share-step", str(preflight["mix_share_step"]),
            "--min-operation-share", str(preflight["min_operation_share"]),
            "--actual-patterns-per-total", str(preflight["actual_patterns_per_total"]),
            "--maximum-D1-loss", str(preflight["quality_limit_db"]),
            "--maximum-D2-loss", str(preflight["quality_limit_db"]),
            "--plan-variants", str(args.plan_variants),
            "--output-root", str(temporary_root / "run"),
            "--tmp-dir", str(temporary_root / "tmp"),
        ]
        result = int(den6.main(argv))
        if result != 0:
            raise RuntimeError("den6 offline reconstruction exited {}".format(result))
    finally:
        shutil.rmtree(str(temporary_root), ignore_errors=True)

    missing = sorted(set(actual_rows) - seen)
    if missing:
        raise RuntimeError(
            "{} recorded Actual plans were not reconstructed; first={}".format(
                len(missing), missing[0]
            )
        )
    reconstructed.sort(
        key=lambda row: (
            row["state_key"]["input_sha256"],
            row["state_key"]["setting_id"],
            row["pattern_key"],
        )
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mynet_kproposal_actual_plan_dataset_v1",
        "offline_only": True,
        "contains_virtual_actual_labels": False,
        "den6_sha256": current_den6,
        "source_run_rows": str(Path(args.run_rows).resolve()),
        "source_run_rows_sha256": _sha256_file(args.run_rows),
        "records": reconstructed,
    }
    with gzip.open(str(output), "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    return {
        "output": str(output),
        "records": len(reconstructed),
        "states": len(
            {
                (row["state_key"]["input_sha256"], row["state_key"]["setting_id"])
                for row in reconstructed
            }
        ),
        "output_sha256": _sha256_file(output),
        "output_bytes": output.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert recorded den6 Actual rows into exact executed-plan data"
    )
    parser.add_argument("--run-rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sparsepcgc-root",
        default="/home/maejima/MasterEx/compress/octree/SparsePCGC",
    )
    parser.add_argument("--tmp-root", default="/dev/shm")
    parser.add_argument("--plan-variants", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args), sort_keys=True))


if __name__ == "__main__":
    main()
