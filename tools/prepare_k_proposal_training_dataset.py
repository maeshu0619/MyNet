#!/usr/bin/env python3
"""Prepare leakage-audited K-slot teacher modes from reconstructed Actual plans.

Offline only.  Coordinates are sparse supervision keys which must be joined to
the current input's local/Octree features by the training loader; they are not
candidate IDs or inference lookups.  Voxel values are rank-weighted relative
values, not causal marginal codec gains.
"""

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


OPERATIONS = ("Prune", "Add", "Adjust")
OP_INDEX = {name: index for index, name in enumerate(OPERATIONS)}


def _load(paths):
    rows = []
    for path in paths:
        with gzip.open(str(path), "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("contains_virtual_actual_labels", True):
            raise RuntimeError("virtual/estimated labels are forbidden: {}".format(path))
        rows.extend(payload.get("records", ()))
    return rows


def _state_id(row):
    state = row["state_key"]
    return "{}|{}".format(state["input_sha256"], state["setting_id"])


def _footprint(row):
    result = set()
    for candidate in row["candidates"]:
        operation = candidate["operation"]
        for coord in candidate.get("remove_coords", ()):
            result.add((operation, "remove") + tuple(map(int, coord)))
        for coord in candidate.get("add_coords", ()):
            result.add((operation, "add") + tuple(map(int, coord)))
    return result


def _priority_vector(text):
    order = [item for item in str(text).split(">") if item in OP_INDEX]
    ranks = [0.0] * 3
    for rank, operation in enumerate(order):
        ranks[OP_INDEX[operation]] = 1.0 - rank / 2.0
    return ranks


def _direction_histogram(row):
    # Exact direction exists only for Adjust (remove source + add target).
    # Historical Add candidates stored the target and symbol ID, not source;
    # therefore Add-direction supervision is explicitly marked unavailable.
    histogram = np.zeros(6, dtype=np.float64)
    count = 0
    for candidate in row["candidates"]:
        if candidate["operation"] != "Adjust":
            continue
        removes = candidate.get("remove_coords", ())
        adds = candidate.get("add_coords", ())
        if len(removes) != 1 or len(adds) != 1:
            continue
        delta = np.asarray(adds[0], dtype=np.int64) - np.asarray(removes[0], dtype=np.int64)
        axis = int(np.abs(delta).argmax())
        sign = int(delta[axis] >= 0)
        histogram[2 * axis + sign] += 1.0
        count += 1
    if count:
        histogram /= float(count)
    return histogram.tolist()


def _spatial_descriptor(row):
    coords = []
    parents = []
    for candidate in row["candidates"]:
        values = candidate.get("add_coords", ()) or candidate.get("remove_coords", ())
        for coord in values:
            array = np.asarray(coord, dtype=np.float64)
            coords.append(array)
            parents.append(tuple((array.astype(np.int64) // 2).tolist()))
    if not coords:
        return [0.0] * 5
    matrix = np.stack(coords, axis=0)
    extent = np.maximum(matrix.max(0) - matrix.min(0), 1.0)
    concentration = 1.0 / (1.0 + extent / max(float(len(coords)) ** (1.0 / 3.0), 1.0))
    unique_parent_ratio = len(set(parents)) / float(len(parents))
    return concentration.tolist() + [unique_parent_ratio, float(len(coords))]


def _operation_score_statistics(row):
    means = []
    maxima = []
    for operation in OPERATIONS:
        scores = [
            float(candidate["heuristic_score"])
            for candidate in row["candidates"]
            if candidate["operation"] == operation
        ]
        means.append(float(np.mean(scores)) if scores else 0.0)
        maxima.append(float(np.max(scores)) if scores else 0.0)
    return means + maxima


def _operation_basis_statistics(row):
    fields = (
        "fixed_context_gain_bits",
        "subtree_bit_mass",
        "expected_new_descendant_bits",
        "mask_gain_bits",
        "neighbor_bit_risk",
        "optimistic_gain_bits",
        "geometry_cost",
    )
    operation_means = []
    for operation in OPERATIONS:
        candidates = [
            candidate for candidate in row["candidates"]
            if candidate["operation"] == operation
        ]
        operation_means.append([
            float(np.mean([float(candidate[field]) for candidate in candidates]))
            if candidates else 0.0
            for field in fields
        ])
    return np.asarray(operation_means, dtype=np.float64).mean(axis=0).tolist()


def _operation_overlap(row):
    sets = {name: set() for name in OPERATIONS}
    for candidate in row["candidates"]:
        operation = candidate["operation"]
        values = candidate.get("remove_coords", ()) or candidate.get("add_coords", ())
        sets[operation].update(tuple(map(int, coord)) for coord in values)
    result = []
    for first, second in (("Prune", "Add"), ("Prune", "Adjust"), ("Add", "Adjust")):
        union = sets[first] | sets[second]
        result.append(len(sets[first] & sets[second]) / float(max(len(union), 1)))
    return result


def _descriptor(row):
    counts = row["operation_counts"]
    total_count = max(float(sum(counts.values())), 1.0)
    enable = [float(counts[name] > 0) for name in OPERATIONS]
    shares = [float(counts[name]) / total_count for name in OPERATIONS]
    requested = row["requested_counts"]
    requested_total = max(float(sum(requested.values())), 1.0)
    rejected_ratio = max(requested_total - total_count, 0.0) / requested_total
    total_ratio_fraction = float(row["total_ratio_percent"]) / 100.0
    values = (
        [total_ratio_fraction * share for share in shares]
        + shares
        + enable
        + _priority_vector(row["operation_order"])
        + _operation_score_statistics(row)
        + _operation_basis_statistics(row)
        + _operation_overlap(row)
        + [rejected_ratio]
    )
    return np.asarray(values, dtype=np.float64)


def _distance_matrix(rows):
    descriptors = np.stack([_descriptor(row) for row in rows], axis=0)
    clustering_descriptors = np.stack([
        np.concatenate((
            _descriptor(row),
            np.asarray(_direction_histogram(row), dtype=np.float64),
            np.asarray(_spatial_descriptor(row), dtype=np.float64),
            np.asarray((
                float(row["geometry"]["D1_loss_db"]),
                float(row["geometry"]["D2_loss_db"]),
                float(row["actual_gain_percent"]),
                float(row["interaction_gain_percent"]),
            ), dtype=np.float64),
        ))
        for row in rows
    ], axis=0)
    scale = np.std(clustering_descriptors, axis=0)
    scale[scale < 1e-6] = 1.0
    numeric = np.mean(np.abs(
        clustering_descriptors[:, None, :] - clustering_descriptors[None, :, :]
    ) / scale[None, None, :], axis=2)
    footprints = [_footprint(row) for row in rows]
    jaccard = np.zeros_like(numeric)
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            union = footprints[first] | footprints[second]
            overlap = footprints[first] & footprints[second]
            distance = 1.0 - len(overlap) / float(max(len(union), 1))
            jaccard[first, second] = distance
            jaccard[second, first] = distance
    return 0.65 * numeric + 0.35 * jaccard, descriptors


def _k_medoids(rows, maximum_modes):
    count = len(rows)
    mode_count = min(maximum_modes, count)
    distance, descriptors = _distance_matrix(rows)
    # Deterministic gain-aware farthest-first seed, followed by medoid updates.
    first = int(np.argmax([float(row["actual_gain_percent"]) for row in rows]))
    medoids = [first]
    while len(medoids) < mode_count:
        nearest = distance[:, medoids].min(axis=1)
        nearest[medoids] = -1.0
        medoids.append(int(nearest.argmax()))
    for _ in range(25):
        assignment = distance[:, medoids].argmin(axis=1)
        changed = False
        for mode in range(mode_count):
            members = np.nonzero(assignment == mode)[0]
            if not len(members):
                continue
            costs = distance[np.ix_(members, members)].sum(axis=1)
            replacement = int(members[int(costs.argmin())])
            if replacement != medoids[mode]:
                medoids[mode] = replacement
                changed = True
        if not changed:
            break
    assignment = distance[:, medoids].argmin(axis=1)
    return medoids, assignment, descriptors, distance


def _rank_weighted_voxels(rows):
    gains = np.asarray([float(row["actual_gain_percent"]) for row in rows])
    order = gains.argsort().argsort().astype(np.float64)
    rank = order / max(float(len(rows) - 1), 1.0)
    accum = {}
    for row_index, row in enumerate(rows):
        # Positive top plans get +1, worsening/bottom plans get -1.
        weight = 2.0 * rank[row_index] - 1.0
        if gains[row_index] <= 0.0:
            weight = min(weight, -0.5)
        for candidate in row["candidates"]:
            operation = candidate["operation"]
            if operation not in OP_INDEX:
                continue
            coords = (
                candidate.get("remove_coords", ())
                if operation != "Add" else candidate.get("add_coords", ())
            )
            direction = None
            if operation == "Adjust":
                remove = candidate.get("remove_coords", ())
                add = candidate.get("add_coords", ())
                if len(remove) == 1 and len(add) == 1:
                    direction = [int(add[0][axis]) - int(remove[0][axis]) for axis in range(3)]
            for coord in coords:
                key = (operation,) + tuple(map(int, coord))
                cell = accum.setdefault(key, [0.0, 0, 0, 0, direction])
                cell[0] += weight * max(gains[row_index], 0.05)
                cell[1] += 1
                cell[2] += int(rank[row_index] >= 0.75)
                cell[3] += int(rank[row_index] <= 0.25 or gains[row_index] <= 0.0)
    result = []
    for key, (weighted, frequency, top, bad, direction) in accum.items():
        value = 1.0 / (1.0 + math.exp(-weighted / max(float(frequency), 1.0)))
        result.append({
            "operation": key[0],
            "coord": list(key[1:]),
            "rank_weighted_relative_value": value,
            "selected_frequency": frequency,
            "top_frequency": top,
            "bad_frequency": bad,
            "direction": direction,
            "direction_available": direction is not None,
        })
    result.sort(key=lambda item: (item["operation"], item["coord"]))
    return result


def _strict_split(states):
    """Choose three state edges with disjoint frame *and* codec setting.

    Repeated settings connect all available frames, so assigning every state
    without leakage is impossible in the current corpus.  Conflicting states
    are excluded and reported rather than silently leaked.
    """
    state_items = list(states.items())
    best = None
    for chosen in itertools.permutations(state_items, 3):
        frames = [item[1][0]["state_key"]["input_sha256"] for item in chosen]
        settings = [item[1][0]["state_key"]["setting_id"] for item in chosen]
        if len(set(frames)) != 3 or len(set(settings)) != 3:
            continue
        score = sum(len(item[1]) for item in chosen)
        if best is None or score > best[0]:
            best = (score, chosen)
    if best is None:
        raise RuntimeError("no frame+codec-disjoint train/validation/test split exists")
    names = ("train", "validation", "test")
    split = {name: chosen[0] for name, chosen in zip(names, best[1])}
    used = set(split.values())
    return split, sorted(set(states) - used)


def prepare(args):
    rows = _load(args.input)
    states = {}
    for row in rows:
        states.setdefault(_state_id(row), []).append(row)
    split, excluded = _strict_split(states)
    prepared = {}
    for state_id, state_rows in states.items():
        medoids, assignment, descriptors, distance = _k_medoids(
            state_rows, args.maximum_k
        )
        prepared[state_id] = {
            "state_key": state_rows[0]["state_key"],
            "candidate_count": len(state_rows),
            "mode_count": len(medoids),
            "mode_medoids": [
                {
                    "plan_key": state_rows[index]["plan_key"],
                    "descriptor": descriptors[index].tolist(),
                    "actual_gain_percent": float(state_rows[index]["actual_gain_percent"]),
                    "geometry": state_rows[index]["geometry"],
                    "member_count": int((assignment == mode).sum()),
                }
                for mode, index in enumerate(medoids)
            ],
            "voxel_relative_values": _rank_weighted_voxels(state_rows),
            "actual_gain_summary": {
                "minimum": float(min(row["actual_gain_percent"] for row in state_rows)),
                "median": float(np.median([row["actual_gain_percent"] for row in state_rows])),
                "maximum": float(max(row["actual_gain_percent"] for row in state_rows)),
            },
        }
    payload = {
        "schema_version": "mynet_kproposal_mode_dataset_v1",
        "offline_only": True,
        "voxel_target_semantics": "rank_weighted_relative_value_not_causal_gain",
        "add_direction_teacher_available": False,
        "maximum_k": int(args.maximum_k),
        "split": split,
        "excluded_state_ids_due_to_frame_or_codec_leakage": excluded,
        "states": prepared,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(output), "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "output": str(output),
        "sha256": digest,
        "state_count": len(states),
        "candidate_count": len(rows),
        "split": split,
        "excluded_state_count": len(excluded),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-k", type=int, default=8)
    args = parser.parse_args()
    if not 2 <= args.maximum_k <= 16:
        raise ValueError("maximum-k must be in [2, 16]")
    print(json.dumps(prepare(args), sort_keys=True))


if __name__ == "__main__":
    main()
