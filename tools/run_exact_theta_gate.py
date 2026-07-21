#!/usr/bin/env python3
"""保存ActualだけでExact Generator + Network theta仮説を段階検証する。"""

import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modules.exact_teacher_theta_adapter import (  # noqa: E402
    CatalogThetaSelector,
    ExactTeacherThetaCatalog,
)


OPERATIONS = ("Add", "Prune", "Adjust")
ORDERS = tuple(itertools.permutations(OPERATIONS))
RATIO_VALUES = (0.05, 0.10, 0.25, 0.50, 1.00)
REASON_ORDER = {
    "screening_top1": 0,
    "screening_top2": 1,
    "balanced_anchor": 2,
    "primary_heavy_anchor": 3,
    "adjust_sensitivity_anchor": 4,
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("moduleを読めない: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_ply_xyz(path):
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    coords = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1)
    return np.unique(np.rint(coords).astype(np.int64), axis=0)


def _coord_hash(coords):
    values = np.asarray(coords, dtype=np.int64)
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    return hashlib.sha256(
        np.ascontiguousarray(values[order], dtype=np.int64).tobytes(order="C")
    ).hexdigest()


def _theta_descriptor(item):
    theta = item.theta
    generator_input = theta["generator_input"]
    outcome = theta["generator_outcome"]
    ratio = float(generator_input["total_ratio_percent"])
    ratio_index = RATIO_VALUES.index(ratio)
    order = tuple(outcome["operation_order"])
    order_index = ORDERS.index(order)
    variant = int(outcome["variant_index"])
    result = np.zeros(5 + 3 + 6 + 6, dtype=np.float64)
    result[ratio_index] = 1.0
    result[5:8] = [float(generator_input["share"][name]) for name in OPERATIONS]
    result[8 + order_index] = 1.0
    if 0 <= variant < 6:
        result[14 + variant] = 1.0
    return result


def _distance(records):
    values = np.stack([_theta_descriptor(item) for item in records], axis=0)
    scale = np.std(values, axis=0)
    scale[scale < 1e-8] = 1.0
    return np.sqrt(np.mean(
        ((values[:, None, :] - values[None, :, :]) / scale) ** 2, axis=2
    ))


def _medoids(records, count):
    distance = _distance(records)
    medoids = [int(distance.sum(1).argmin())]
    while len(medoids) < min(count, len(records)):
        nearest = distance[:, medoids].min(1)
        nearest[medoids] = -1.0
        medoids.append(int(nearest.argmax()))
    for _ in range(30):
        assignment = distance[:, medoids].argmin(1)
        changed = False
        for mode in range(len(medoids)):
            members = np.flatnonzero(assignment == mode)
            if not len(members):
                continue
            replacement = int(members[
                distance[np.ix_(members, members)].sum(1).argmin()
            ])
            if replacement != medoids[mode]:
                medoids[mode] = replacement
                changed = True
        if not changed:
            break
    return medoids


def _coverage(records, count):
    distance = _distance(records)
    selected = [max(
        range(len(records)),
        key=lambda index: (
            float(records[index].source_actual_row["screening_score"]), -index
        ),
    )]
    while len(selected) < min(count, len(records)):
        nearest = distance[:, selected].min(1)
        nearest[selected] = -1.0
        selected.append(int(nearest.argmax()))
    return selected


def _select_indices(records, count, method):
    if method == "estimated_score":
        return sorted(range(len(records)), key=lambda index: (
            -float(records[index].source_actual_row["screening_score"]),
            int(records[index].source_actual_row["screening_rank_within_total"]),
            records[index].theta_id,
        ))[:count]
    if method == "theta_kmedoids":
        return _medoids(records, count)
    if method == "theta_coverage":
        return _coverage(records, count)
    if method == "den6_saved_rule":
        return sorted(range(len(records)), key=lambda index: (
            REASON_ORDER.get(
                str(records[index].source_actual_row.get("selection_reason", "")), 99
            ),
            int(records[index].source_actual_row["screening_rank_within_total"]),
            float(records[index].record["total_ratio_percent"]),
            records[index].theta_id,
        ))[:count]
    raise ValueError(method)


def _catalog_payload(catalog, source_paths):
    states = {}
    for state_id in catalog.state_ids:
        rows = []
        for item in catalog.records(state_id):
            record = item.record
            source = item.source_actual_row
            rows.append({
                "state_id": state_id,
                "theta_id": item.theta_id,
                "theta": item.theta,
                "plan_key": str(record["plan_key"]),
                "pattern_key": str(record["pattern_key"]),
                "requested_count": dict(record["requested_counts"]),
                "accepted_count": dict(record["operation_counts"]),
                "operation_order": str(source["operation_order"]),
                "executable_hash": None,
                "final_voxel_hash": str(record.get("final_voxel_hash", "")) or None,
                "actual_gain_percent": float(record["actual_gain_percent"]),
                "geometry": dict(record["geometry"]),
                "estimated_gain_percent": float(record["estimated_gain_percent"]),
                "screening_score": float(source["screening_score"]),
                "screening_rank_within_total": int(source["screening_rank_within_total"]),
                "selection_reason": str(source.get("selection_reason", "")),
                "source_dataset_row": {
                    "dataset_path": next(
                        path for path in source_paths
                        if str(record["state_key"]["input_file"]).lower().split("/ground/")[1].split("/")[0]
                        in Path(path).name.lower()
                    ),
                    "pattern_key": str(record["pattern_key"]),
                },
                "field_masks": {
                    "add_source_available": False,
                    "add_direction_available": False,
                    "variant_available": int(source.get("variant_index", -1)) >= 0,
                    "final_voxel_hash_available": bool(record.get("final_voxel_hash")),
                },
            })
        states[state_id] = {
            "state_key": dict(catalog.records(state_id)[0].record["state_key"]),
            "records": rows,
        }
    return {
        "schema_version": "mynet_exact_theta_catalog_v1",
        "offline_only": True,
        "contains_only_measured_actual": True,
        "actual_label_interpolation": False,
        "theta_semantics": {
            "generator_input": ["total_ratio_percent", "share"],
            "generator_outcome_not_free_input": ["operation_order", "variant_index"],
            "fixed_not_learned": ["den5_score_formula", "den5_score_coefficients", "plan_variant_count"],
        },
        "states": states,
    }


def _analyse_gate_a(catalog):
    methods = ("estimated_score", "theta_kmedoids", "theta_coverage", "den6_saved_rule")
    counts = (1, 2, 4, 8, 16, "all")
    states = {}
    for state_id in catalog.state_ids:
        records = catalog.records(state_id)
        gains = np.asarray([float(item.record["actual_gain_percent"]) for item in records])
        all_best = float(gains.max())
        high_value = set(np.flatnonzero(gains >= 0.90 * all_best).tolist())
        result = {
            "candidate_count": len(records),
            "all_best_gain_percent": all_best,
            "high_value_mode_threshold_percent": 0.90 * all_best,
            "high_value_mode_count": len(high_value),
            "high_value_theta_ids": [records[index].theta_id for index in sorted(high_value)],
        }
        for method in methods:
            values = {}
            for requested in counts:
                count = len(records) if requested == "all" else min(int(requested), len(records))
                selected = _select_indices(records, count, method)
                best = float(gains[selected].max())
                values[str(requested)] = {
                    "best_gain_percent": best,
                    "all_best_recovery": best / max(all_best, 1e-12),
                    "theta_ids": [records[index].theta_id for index in selected],
                    "selected_high_value_mode_count": len(high_value & set(selected)),
                    "dropped_high_value_theta_ids": [
                        records[index].theta_id for index in sorted(high_value - set(selected))
                    ],
                    "theta_coverage": {
                        "ratio": sorted({
                            float(records[index].theta["generator_input"]["total_ratio_percent"])
                            for index in selected
                        }),
                        "order": sorted({
                            ">".join(records[index].theta["generator_outcome"]["operation_order"])
                            for index in selected
                        }),
                        "variant": sorted({
                            int(records[index].theta["generator_outcome"]["variant_index"])
                            for index in selected
                        }),
                        "unique_share_count": len({
                            tuple(
                                float(records[index].theta["generator_input"]["share"][name])
                                for name in OPERATIONS
                            )
                            for index in selected
                        }),
                    },
                }
            result[method] = values
        random_result = {}
        for count in (8, 16):
            generator = random.Random(20260721)
            best_values = []
            for _ in range(1000):
                selected = generator.sample(range(len(records)), min(count, len(records)))
                best_values.append(float(gains[selected].max()))
            random_result[str(count)] = {
                "mean_best_gain_percent": float(np.mean(best_values)),
                "p10_best_gain_percent": float(np.quantile(best_values, 0.10)),
                "p90_best_gain_percent": float(np.quantile(best_values, 0.90)),
            }
        result["random_reference"] = random_result
        states[state_id] = result
    minimum = {}
    for method in methods:
        for count in (8, 16):
            key = "{}_k{}".format(method, count)
            values = [states[state][method][str(count)]["all_best_recovery"] for state in states]
            minimum[key] = {"minimum": min(values), "mean": float(np.mean(values))}
    gate_pass = any(
        minimum["{}_k{}".format(method, count)]["minimum"] >= 0.90
        for method in methods for count in (8, 16)
    )
    return {"pass": gate_pass, "states": states, "summary": minimum}


def _replay_gate_b(catalog, sparse_root):
    den5_path = Path(sparse_root) / "ana_den5_v8.py"
    den5 = _load_module(den5_path, "mynet_exact_theta_den5_readonly")
    total = 0
    plan_key_match = 0
    count_match = 0
    order_match = 0
    packed_match = 0
    final_hash_match = 0
    missing_add_source = 0
    missing_add_direction = 0
    by_state = {}
    for state_id in catalog.state_ids:
        records = catalog.records(state_id)
        input_file = str(records[0].record["state_key"]["input_file"])
        coords = _read_ply_xyz(input_file)
        generated = catalog.generate(
            state_id, [item.theta_id for item in records], torch.from_numpy(coords)
        )
        state_result = {"records": len(records), "final_hash_mismatch": []}
        for index, item in enumerate(records):
            total += 1
            source = item.source_actual_row
            record = item.record
            plan_key_match += int(str(source["plan_key"]) == str(record["plan_key"]))
            expected_count = [int(record["operation_counts"][name]) for name in ("Prune", "Add", "Adjust")]
            packed_count = generated.executable.accepted_count[0, index].tolist()
            count_ok = expected_count == packed_count
            count_match += int(count_ok)
            order = [name for name in str(source["operation_order"]).split(">") if name in OPERATIONS]
            packed_order = [
                ("Prune", "Add", "Adjust")[value]
                for value in generated.executable.operation_order[0, index].tolist()
            ]
            order_match += int(order == packed_order)
            packed_match += int(count_ok and order == packed_order)
            plan = [
                SimpleNamespace(
                    remove_coords=tuple(tuple(map(int, value)) for value in member.get("remove_coords", ())),
                    add_coords=tuple(tuple(map(int, value)) for value in member.get("add_coords", ())),
                )
                for member in record["candidates"]
            ]
            final_coords, _, _ = den5._apply_actual_plan_fast(coords, plan)
            actual_hash = _coord_hash(final_coords)
            hash_ok = actual_hash == str(record["final_voxel_hash"])
            final_hash_match += int(hash_ok)
            if not hash_ok:
                state_result["final_hash_mismatch"].append(str(record["plan_key"]))
            missing_add_source += int(generated.missing_add_source_count[index])
            missing_add_direction += int(generated.missing_add_direction_count[index])
        by_state[state_id] = state_result
    return {
        "pass": all(value == total for value in (
            plan_key_match, count_match, order_match, packed_match, final_hash_match
        )),
        "total": total,
        "plan_key_match": plan_key_match,
        "count_match": count_match,
        "order_match": order_match,
        "common_executable_pack_match": packed_match,
        "final_voxel_hash_match": final_hash_match,
        "missing_add_source_count": missing_add_source,
        "missing_add_direction_count": missing_add_direction,
        "den5_path": str(den5_path),
        "den5_sha256": _sha256_file(den5_path),
        "states": by_state,
    }


def _state_features(record, coords):
    values = np.asarray(coords, dtype=np.float64)
    minimum = values.min(0)
    maximum = values.max(0)
    extent = np.maximum(maximum - minimum, 1.0)
    scale = max(float(extent.max()), 1.0)
    state = record.record["state_key"]
    return np.asarray([
        math.log1p(values.shape[0]) / 20.0,
        *(extent / scale).tolist(),
        *((values.mean(0) - minimum) / extent).tolist(),
        *(values.std(0) / extent).tolist(),
        float(state.get("scale_ae", 0)) / 8.0,
        float(state.get("scale_sr", 0)) / 8.0,
        float(state.get("scale_m", 0)) / 16.0,
        math.log1p(float(state.get("voxel_size", 1.0))) / 4.0,
        math.log1p(float(state.get("pos_quantscale", 1.0))) / 4.0,
        math.log1p(float(state.get("native_resolution", 1023))) / 10.0,
    ], dtype=np.float32)


def _diverse_topk(logits, descriptors, count, penalty=0.05):
    score = logits.detach().cpu().numpy().astype(np.float64)
    distance = np.sqrt(np.mean(
        (descriptors[:, None, :] - descriptors[None, :, :]) ** 2, axis=2
    ))
    selected = [int(score.argmax())]
    while len(selected) < min(count, len(score)):
        diversity = distance[:, selected].min(1)
        utility = score + float(penalty) * diversity
        utility[selected] = -np.inf
        selected.append(int(utility.argmax()))
    return selected


def _selector_gate_c(catalog, steps):
    preferred = [
        state for state in catalog.state_ids
        if "native_vs1_pq1_ae0_sr2_m8" in state
        and len(catalog.records(state)) == 22
    ]
    state_id = preferred[0] if preferred else catalog.state_ids[0]
    records = catalog.records(state_id)
    coords = _read_ply_xyz(records[0].record["state_key"]["input_file"])
    state = torch.from_numpy(_state_features(records[0], coords)).view(1, -1)
    theta_values = np.stack([_theta_descriptor(item) for item in records]).astype(np.float32)
    theta = torch.from_numpy(theta_values)
    gains = torch.tensor(
        [float(item.record["actual_gain_percent"]) for item in records], dtype=torch.float32
    )
    normalized = (gains - gains.mean()) / gains.std().clamp_min(1e-6)
    target = torch.softmax(normalized / 0.25, dim=0)
    torch.manual_seed(20260721)
    model = CatalogThetaSelector(state.shape[1], theta.shape[1], hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    history = []
    for step in range(int(steps)):
        logits = model(state, theta)
        loss = -(target * torch.log_softmax(logits, dim=0)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, int(steps) - 1):
            history.append({"step": step + 1, "loss": float(loss.detach())})
    with torch.no_grad():
        logits = model(state, theta)
    all_best = float(gains.max())
    results = {}
    for count in (8, 16):
        selected = _diverse_topk(logits, theta_values, count)
        selected_best = float(gains[selected].max())
        generated = catalog.generate(
            state_id,
            [records[index].theta_id for index in selected],
            torch.from_numpy(coords),
        )
        plan_keys_match = all(
            generated.catalog_plan_keys[position] == str(records[index].record["plan_key"])
            for position, index in enumerate(selected)
        )
        results[str(count)] = {
            "selected_theta_ids": [records[index].theta_id for index in selected],
            "unique_theta_count": len(set(records[index].theta_id for index in selected)),
            "best_gain_percent": selected_best,
            "all_best_recovery": selected_best / max(all_best, 1e-12),
            "exact_generator_plan_key_match": plan_keys_match,
        }
    gate_pass = any(
        result["all_best_recovery"] >= 0.90
        and result["unique_theta_count"] >= math.ceil(int(count) * 0.75)
        and result["exact_generator_plan_key_match"]
        for count, result in results.items()
    )
    return {
        "pass": gate_pass,
        "state_id": state_id,
        "optimizer_steps": int(steps),
        "all_best_gain_percent": all_best,
        "history": history,
        "results": results,
        "selector_input": "shape/codec state features + discrete catalog theta descriptor",
        "actual_source": "saved measured Actual only",
    }


def run(args):
    catalog = ExactTeacherThetaCatalog.from_actual_plan_datasets(args.input)
    payload = _catalog_payload(catalog, args.input)
    output = Path(args.catalog_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(output), "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    report = {
        "catalog": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "states": len(catalog.state_ids),
            "records": sum(len(catalog.records(state)) for state in catalog.state_ids),
        },
        "gate_a": _analyse_gate_a(catalog),
    }
    if report["gate_a"]["pass"]:
        report["gate_b"] = _replay_gate_b(catalog, args.sparsepcgc_root)
    else:
        report["gate_b"] = {"pass": False, "skipped": "Gate A不合格"}
    if report["gate_b"].get("pass"):
        report["gate_c"] = _selector_gate_c(catalog, args.selector_steps)
    else:
        report["gate_c"] = {"pass": False, "skipped": "Gate B不合格"}
    report_path = Path(args.report_output).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=(
        "/data/maejima/log/mynet_kproposal_offline/8i_actual_plans.json.gz",
        "/data/maejima/log/mynet_kproposal_offline/mvub_actual_plans.json.gz",
        "/data/maejima/log/mynet_kproposal_offline/uvg_actual_plans.json.gz",
    ))
    parser.add_argument(
        "--catalog-output",
        default="/data/maejima/log/mynet_kproposal_offline/exact_theta_catalog_v1.json.gz",
    )
    parser.add_argument(
        "--report-output",
        default="/data/maejima/log/mynet_kproposal_offline/exact_theta_gate_report.json",
    )
    parser.add_argument(
        "--sparsepcgc-root",
        default="/home/maejima/MasterEx/compress/octree/SparsePCGC",
    )
    parser.add_argument("--selector-steps", type=int, default=40)
    args = parser.parse_args()
    report, report_path = run(args)
    print(json.dumps({
        "report": str(report_path),
        "catalog": report["catalog"],
        "gate_a": report["gate_a"]["pass"],
        "gate_b": report["gate_b"].get("pass", False),
        "gate_c": report["gate_c"].get("pass", False),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
