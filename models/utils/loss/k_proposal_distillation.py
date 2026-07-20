"""Offline-only losses for fitting specialized K proposal slots.

This module consumes compact tensors produced from previously executed
Heuristic candidates and their recorded Actual scalars.  It is not imported by
the inference policy and never invokes den, a codec probe, or an Actual encoder.

Voxel targets produced by the companion offline builder are *rank-weighted
relative values*.  They are observational comparisons inside one state/mode,
not causal per-voxel Actual gains.
"""

import gzip
import itertools
import json
from pathlib import Path

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


OPERATIONS = ("Prune", "Add", "Adjust")
OPERATION_INDEX = {name: index for index, name in enumerate(OPERATIONS)}
ORDER_PERMUTATIONS = list(itertools.permutations(OPERATIONS))
SUPPORTED_MODE_SCHEMAS = {
    "mynet_kproposal_mode_dataset_v1",
    "mynet_kproposal_mode_dataset_v2",
}


class OfflineKProposalTeacherStore:
    """Explicit training-only reader for the compact Actual mode dataset.

    Constructing this object is the only file read.  The inference policy does
    not instantiate or import it.  Add target coordinates cannot be joined to
    occupied-source logits because the historical den6 schema omitted the Add
    source; Add Where masks therefore stay false instead of inventing labels.
    """

    def __init__(self, path):
        with gzip.open(str(path), "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        self.schema_version = str(payload.get("schema_version", ""))
        if self.schema_version not in SUPPORTED_MODE_SCHEMAS:
            raise RuntimeError("unsupported K proposal offline dataset schema")
        if not bool(payload.get("offline_only", False)):
            raise RuntimeError("K proposal teacher dataset must be offline-only")
        if payload.get("voxel_target_semantics") != "rank_weighted_relative_value_not_causal_gain":
            raise RuntimeError("voxel target semantics are missing or unsafe")
        self.path = str(path)
        self.payload = payload
        self.states = dict(payload.get("states") or {})
        self.split = dict(payload.get("split") or {})
        self._runtime_index = {}
        for state_id, state in self.states.items():
            key = state.get("state_key", {})
            index_key = (
                str(Path(str(key.get("input_file", ""))).expanduser().resolve()),
                int(key.get("scale_m", -1)),
                int(key.get("scale_ae", -1)),
                int(key.get("scale_sr", -1)),
                float(key.get("voxel_size", -1.0)),
                int(key.get("pos_quantscale", -1)),
                int(key.get("native_resolution", -1)),
            )
            self._runtime_index[index_key] = state_id

    @staticmethod
    def state_id(input_sha256, setting_id):
        return "{}|{}".format(str(input_sha256), str(setting_id))

    def find_state_for_input(self, input_file, args, split=None):
        index_key = (
            str(Path(str(input_file)).expanduser().resolve()),
            int(getattr(args, "sparsepcgc_scale_m", -1)),
            int(getattr(args, "sparsepcgc_scale_ae", -1)),
            int(getattr(args, "sparsepcgc_scale_sr", -1)),
            float(getattr(args, "sparsepcgc_voxel_size", -1.0)),
            int(getattr(args, "sparsepcgc_pos_quantscale", -1)),
            int(getattr(args, "sparsepcgc_psnr_resolution", -1)),
        )
        state_id = self._runtime_index.get(index_key)
        if state_id is None:
            return None
        if split is not None and self.split.get(str(split)) != state_id:
            return None
        return state_id

    def training_source_coordinates(self, state_id):
        """v2に実在するsource教師だけを返す。欠落したAdd sourceは生成しない。"""
        state = self.states.get(str(state_id))
        if not isinstance(state, dict):
            return []
        coordinates = set()
        for mode in state.get("mode_medoids") or ():
            for operation in OPERATIONS:
                rows = dict((mode.get("voxel_targets") or {}).get(operation) or {})
                source_coords = list(rows.get("source_coords") or ())
                source_available = list(rows.get("source_available") or ())
                for index, coord in enumerate(source_coords):
                    if (
                        coord is not None
                        and index < len(source_available)
                        and bool(source_available[index])
                    ):
                        coordinates.add(tuple(map(int, coord)))
        return [list(coord) for coord in sorted(coordinates)]

    def training_target_coordinates(self, state_id):
        """mode別Add/Adjust targetの和集合を返し、source教師とは扱わない。"""
        state = self.states.get(str(state_id))
        if not isinstance(state, dict):
            return []
        coordinates = set()
        for mode in state.get("mode_medoids") or ():
            for operation in ("Add", "Adjust"):
                rows = dict((mode.get("voxel_targets") or {}).get(operation) or {})
                targets = list(rows.get("target_coords") or ())
                available = list(rows.get("target_available") or ())
                for index, coord in enumerate(targets):
                    if (
                        coord is not None
                        and index < len(available)
                        and bool(available[index])
                    ):
                        coordinates.add(tuple(map(int, coord)))
        return [list(coord) for coord in sorted(coordinates)]

    @staticmethod
    def _hash_coordinates(coords, minimum, span_y, span_z):
        shifted = coords.long() - minimum.view(1, 3)
        return (shifted[:, 0] * span_y + shifted[:, 1]) * span_z + shifted[:, 2]

    @classmethod
    def _join_to_shortlist(cls, shortlist_coords, rows, reference):
        """疎座標をshortlistへexact joinし、値とmaskを返す。"""
        values = reference.new_zeros((shortlist_coords.shape[0],))
        mask = torch.zeros_like(values, dtype=torch.bool)
        valid_rows = [row for row in rows if row.get("coord") is not None]
        if not valid_rows or shortlist_coords.numel() == 0:
            return values, mask
        teacher_coords = torch.tensor(
            [row["coord"] for row in valid_rows],
            device=reference.device,
            dtype=torch.long,
        )
        combined_min = torch.minimum(shortlist_coords.amin(0), teacher_coords.amin(0))
        combined_max = torch.maximum(shortlist_coords.amax(0), teacher_coords.amax(0))
        spans = combined_max - combined_min + 1
        if int(spans[0].item()) * int(spans[1].item()) * int(spans[2].item()) >= 2 ** 62:
            raise RuntimeError("coordinate range is too large for exact sparse join")
        shortlist_hash = cls._hash_coordinates(
            shortlist_coords, combined_min, spans[1], spans[2]
        )
        teacher_hash = cls._hash_coordinates(
            teacher_coords, combined_min, spans[1], spans[2]
        )
        sorted_hash, order = teacher_hash.sort()
        position = torch.searchsorted(sorted_hash, shortlist_hash)
        in_range = position < sorted_hash.numel()
        safe = position.clamp_max(max(int(sorted_hash.numel()) - 1, 0))
        matches = in_range & (sorted_hash[safe] == shortlist_hash)
        if matches.any():
            teacher_values = reference.new_tensor([
                float(row.get("value", 1.0)) for row in valid_rows
            ])
            values[matches] = teacher_values[order[safe[matches]]]
            mask[matches] = True
        return values, mask

    @staticmethod
    def _theta_from_mode(mode):
        theta = dict(mode.get("explicit_theta") or {})
        share = theta.get("share", (0.0, 0.0, 0.0))
        if len(share) != 3:
            share = (0.0, 0.0, 0.0)
        return {
            "ratio_class": int(theta.get("ratio_class", -1)),
            "total_ratio_fraction": float(theta.get("total_ratio_fraction", 0.0)),
            "share": list(map(float, share)),
            "order_class": int(theta.get("order_class", -1)),
            "variant": int(theta.get("variant", -1)),
        }

    def teacher_for_output(
        self, state_id, proposal_output, voxel_coords, split=None
    ):
        if split is not None and self.split.get(str(split)) != state_id:
            raise RuntimeError("state is not in requested leakage-audited split")
        state = self.states.get(str(state_id))
        if not isinstance(state, dict):
            raise KeyError("offline K proposal state is unavailable: {}".format(state_id))
        proposal_count = int(proposal_output["proposal_count"])
        modes = list(state.get("mode_medoids") or ())[:proposal_count]
        if not modes:
            raise RuntimeError("offline state has no teacher modes")
        reference = proposal_output["slot_logits"]
        descriptor_dim = int(proposal_output["compact_plans"]["descriptor"].shape[2])
        descriptors = reference.new_zeros((1, proposal_count, descriptor_dim))
        gains = reference.new_zeros((1, proposal_count))
        geometries = reference.new_zeros((1, proposal_count))
        mode_mask = torch.zeros((1, proposal_count), device=reference.device, dtype=torch.bool)
        interactions = reference.new_zeros((1, proposal_count))
        mode_ranks = reference.new_zeros((1, proposal_count))
        high_value_mask = torch.zeros(
            (1, proposal_count), device=reference.device, dtype=torch.bool
        )
        ratio_class = torch.full(
            (1, proposal_count), -1, device=reference.device, dtype=torch.long
        )
        total_ratio = reference.new_zeros((1, proposal_count))
        shares = reference.new_zeros((1, proposal_count, 3))
        order_class = torch.full_like(ratio_class, -1)
        variant_class = torch.full_like(ratio_class, -1)
        theta_mask = torch.zeros_like(mode_mask)
        for index, mode in enumerate(modes):
            descriptor = mode.get("descriptor", ())
            if len(descriptor) != descriptor_dim:
                raise RuntimeError("offline/current plan descriptor mismatch")
            descriptors[0, index] = reference.new_tensor(descriptor)
            gains[0, index] = float(mode["actual_gain_percent"])
            geometries[0, index] = float(mode["geometry"]["D1_loss_db"])
            interactions[0, index] = float(mode.get("interaction_gain_percent", 0.0))
            mode_ranks[0, index] = float(mode.get("actual_rank", index))
            high_value_mask[0, index] = bool(mode.get("high_value", False))
            theta = self._theta_from_mode(mode)
            ratio_class[0, index] = theta["ratio_class"]
            total_ratio[0, index] = theta["total_ratio_fraction"]
            shares[0, index] = reference.new_tensor(theta["share"])
            order_class[0, index] = theta["order_class"]
            variant_class[0, index] = theta["variant"]
            theta_mask[0, index] = theta["ratio_class"] >= 0
            mode_mask[0, index] = True
        if mode_mask.any() and not high_value_mask.any():
            valid_gain = gains[mode_mask]
            threshold = torch.quantile(valid_gain, 0.75)
            high_value_mask = mode_mask & (gains >= threshold)

        shortlist = proposal_output["shortlist_indices"]
        natural_shortlist = proposal_output.get("natural_shortlist_indices", shortlist)
        if shortlist.shape[0] != 1 or voxel_coords.shape[0] != 1:
            raise RuntimeError("offline sparse voxel join currently requires batch=1")
        shortlist_coords = torch.gather(
            voxel_coords.long(), 2,
            shortlist.unsqueeze(1).expand(-1, 3, -1),
        )[0].transpose(0, 1).contiguous()
        natural_shortlist_coords = torch.gather(
            voxel_coords.long(), 2,
            natural_shortlist.unsqueeze(1).expand(-1, 3, -1),
        )[0].transpose(0, 1).contiguous()
        mode_source_value = reference.new_zeros(
            (1, proposal_count, 3, shortlist.shape[1])
        )
        mode_source_mask = torch.zeros_like(mode_source_value, dtype=torch.bool)
        mode_direction_index = torch.full(
            (1, proposal_count, 2, shortlist.shape[1]),
            -1,
            device=reference.device,
            dtype=torch.long,
        )
        mode_direction_mask = torch.zeros_like(mode_direction_index, dtype=torch.bool)
        shortlist_recall = reference.new_zeros((1, proposal_count, 3))
        shortlist_recall_mask = torch.zeros_like(shortlist_recall, dtype=torch.bool)
        training_shortlist_recall = reference.new_zeros((1, proposal_count, 3))
        training_shortlist_recall_mask = torch.zeros_like(
            training_shortlist_recall, dtype=torch.bool
        )
        target_rows = []
        target_owner = []
        aggregate_rows = list(state.get("voxel_relative_values", ()))
        for mode_index, mode in enumerate(modes):
            voxel_targets = dict(mode.get("voxel_targets") or {})
            if voxel_targets:
                for operation_name, operation in OPERATION_INDEX.items():
                    operation_rows = dict(voxel_targets.get(operation_name) or {})
                    source_coords = list(operation_rows.get("source_coords") or ())
                    source_available = list(operation_rows.get("source_available") or ())
                    source_join_rows = []
                    for row_index, coord in enumerate(source_coords):
                        available = row_index < len(source_available) and bool(source_available[row_index])
                        if available and coord is not None:
                            source_join_rows.append({"coord": coord, "value": 1.0})
                    joined, joined_mask = self._join_to_shortlist(
                        shortlist_coords, source_join_rows, reference
                    )
                    _, natural_join_mask = self._join_to_shortlist(
                        natural_shortlist_coords, source_join_rows, reference
                    )
                    mode_source_value[0, mode_index, operation] = joined
                    if source_join_rows:
                        training_shortlist_recall[0, mode_index, operation] = (
                            joined_mask.sum().to(reference.dtype) / float(len(source_join_rows))
                        )
                        training_shortlist_recall_mask[0, mode_index, operation] = True
                        shortlist_recall[0, mode_index, operation] = (
                            natural_join_mask.sum().to(reference.dtype) / float(len(source_join_rows))
                        )
                        shortlist_recall_mask[0, mode_index, operation] = True
                        # exact mode教師は未選択shortlistをnegativeとして扱う。
                        # teacherがshortlist外だけの場合は偽のall-negative教師を避ける。
                        if joined_mask.any():
                            mode_source_mask[0, mode_index, operation] = True
                    targets = list(operation_rows.get("target_coords") or ())
                    target_available = list(operation_rows.get("target_available") or ())
                    directions = list(operation_rows.get("direction_index") or ())
                    direction_available = list(operation_rows.get("direction_available") or ())
                    for row_index, coord in enumerate(targets):
                        if row_index < len(target_available) and target_available[row_index] and coord is not None:
                            target_rows.append(list(map(int, coord)))
                            target_owner.append((mode_index, operation))
                    if operation_name not in ("Add", "Adjust"):
                        continue
                    direction_operation = 0 if operation_name == "Add" else 1
                    direction_rows = []
                    for row_index, coord in enumerate(source_coords):
                        valid = (
                            row_index < len(direction_available)
                            and direction_available[row_index]
                            and row_index < len(directions)
                            and coord is not None
                        )
                        if valid:
                            direction_rows.append({
                                "coord": coord,
                                "value": float(directions[row_index]),
                            })
                    direction_values, direction_join_mask = self._join_to_shortlist(
                        shortlist_coords, direction_rows, reference
                    )
                    mode_direction_index[0, mode_index, direction_operation] = direction_values.long()
                    mode_direction_mask[0, mode_index, direction_operation] = direction_join_mask
            else:
                # v1はmode別sourceを持たないため、集約相対価値を各modeへ
                # 複製する。Add targetをoccupied sourceへ変換してはいけない。
                for row in aggregate_rows:
                    operation = OPERATION_INDEX.get(str(row.get("operation", "")), -1)
                    if operation < 0 or operation == OPERATION_INDEX["Add"]:
                        continue
                    joined, joined_mask = self._join_to_shortlist(
                        shortlist_coords,
                        [{
                            "coord": row.get("coord"),
                            "value": float(row.get("rank_weighted_relative_value", 0.5)),
                        }],
                        reference,
                    )
                    mode_source_value[0, mode_index, operation][joined_mask] = joined[joined_mask]
                    mode_source_mask[0, mode_index, operation] |= joined_mask

        maximum_targets = max(
            [sum(1 for owner in target_owner if owner == (mode, operation))
             for mode in range(proposal_count) for operation in range(3)] + [1]
        )
        mode_target_coord = torch.zeros(
            (1, proposal_count, 3, maximum_targets, 3),
            device=reference.device,
            dtype=torch.long,
        )
        mode_target_mask = torch.zeros(
            (1, proposal_count, 3, maximum_targets),
            device=reference.device,
            dtype=torch.bool,
        )
        offsets = {(mode, operation): 0 for mode in range(proposal_count) for operation in range(3)}
        for coord, owner in zip(target_rows, target_owner):
            offset = offsets[owner]
            mode_target_coord[0, owner[0], owner[1], offset] = torch.tensor(
                coord, device=reference.device, dtype=torch.long
            )
            mode_target_mask[0, owner[0], owner[1], offset] = True
            offsets[owner] += 1

        mode_target_value = None
        mode_target_value_mask = None
        target_reachable_recall = reference.new_zeros((1, proposal_count, 3))
        target_reachable_recall_mask = torch.zeros_like(
            target_reachable_recall, dtype=torch.bool
        )
        target_candidate_coords = proposal_output.get("target_candidate_coords")
        if torch.is_tensor(target_candidate_coords) and target_candidate_coords.ndim == 3:
            if target_candidate_coords.shape[0] != 1 or target_candidate_coords.shape[1] != 3:
                raise RuntimeError("shared target candidate coordinates must be [1,3,T]")
            target_domain = target_candidate_coords[0].transpose(0, 1).long().contiguous()
            mode_target_value = reference.new_zeros(
                (1, proposal_count, 3, target_domain.shape[0])
            )
            mode_target_value_mask = torch.zeros_like(mode_target_value, dtype=torch.bool)
            for mode_index in range(proposal_count):
                for operation in range(3):
                    rows = [
                        {"coord": mode_target_coord[0, mode_index, operation, index].tolist(), "value": 1.0}
                        for index in range(maximum_targets)
                        if mode_target_mask[0, mode_index, operation, index]
                    ]
                    joined, joined_mask = self._join_to_shortlist(
                        target_domain, rows, reference
                    )
                    mode_target_value[0, mode_index, operation] = joined
                    if rows:
                        target_reachable_recall[0, mode_index, operation] = (
                            joined_mask.sum().to(reference.dtype) / float(len(rows))
                        )
                        target_reachable_recall_mask[0, mode_index, operation] = True
                    if rows and joined_mask.any():
                        mode_target_value_mask[0, mode_index, operation] = True

        # 集約aliasはv1呼出側の互換用。lossはmode別Tensorを使う。
        voxel_target = mode_source_value[:, 0]
        voxel_mask = mode_source_mask[:, 0]
        actual_replay_gain = proposal_output.get("actual_replay_gain")
        actual_replay_mask = proposal_output.get("actual_replay_mask")
        if not torch.is_tensor(actual_replay_gain):
            actual_replay_gain = reference.new_zeros((1, proposal_count))
        else:
            actual_replay_gain = actual_replay_gain.to(reference)
        if not torch.is_tensor(actual_replay_mask):
            actual_replay_mask = torch.zeros(
                (1, proposal_count), device=reference.device, dtype=torch.bool
            )
        else:
            actual_replay_mask = actual_replay_mask.to(
                device=reference.device, dtype=torch.bool
            )
        if actual_replay_gain.shape != (1, proposal_count) or actual_replay_mask.shape != (1, proposal_count):
            raise RuntimeError("Actual replay tensors must be [1,K]")
        return {
            "mode_descriptor": descriptors,
            "actual_gain": gains,
            "geometry": geometries,
            "interaction": interactions,
            "mode_rank": mode_ranks,
            "high_value_mask": high_value_mask,
            "mode_mask": mode_mask,
            "state_ids": [str(state_id)],
            "theta": {
                "ratio_class": ratio_class,
                "total_ratio": total_ratio,
                "share": shares,
                "order_class": order_class,
                "variant_class": variant_class,
                "mask": theta_mask,
            },
            "mode_source_value": mode_source_value,
            "mode_source_mask": mode_source_mask,
            "mode_target_coord": mode_target_coord,
            "mode_target_mask": mode_target_mask,
            "mode_target_value": mode_target_value,
            "mode_target_value_mask": mode_target_value_mask,
            "mode_direction_index": mode_direction_index,
            "mode_direction_mask": mode_direction_mask,
            "shortlist_natural_recall": shortlist_recall,
            "shortlist_natural_recall_mask": shortlist_recall_mask,
            "shortlist_training_recall": training_shortlist_recall,
            "shortlist_training_recall_mask": training_shortlist_recall_mask,
            "target_reachable_recall": target_reachable_recall,
            "target_reachable_recall_mask": target_reachable_recall_mask,
            "voxel_relative_value": voxel_target,
            "voxel_value_mask": voxel_mask,
            "voxel_target_semantics": "rank_weighted_relative_value_not_causal_gain",
            "add_where_teacher_available": bool(mode_source_mask[:, :, 1].any()),
            "actual_replay_gain": actual_replay_gain,
            "actual_replay_mask": actual_replay_mask,
            "schema_version": self.schema_version,
        }


def _hungarian_min_cost(cost):
    """Return row->column assignment for a small detached rectangular cost.

    Classical O(n^3) Hungarian implementation.  Matching itself is discrete;
    gradients flow through the selected entries of the original cost tensor.
    """
    matrix = cost.detach().float().cpu()
    rows, columns = matrix.shape
    transposed = False
    if rows > columns:
        matrix = matrix.t().contiguous()
        rows, columns = matrix.shape
        transposed = True
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                current = float(matrix[i0 - 1, j - 1]) - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current
                    way[j] = j0
                if minimum[j] < delta:
                    delta = minimum[j]
                    j1 = j
            for j in range(columns + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, columns + 1) if p[j] > 0]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    return pairs


class KProposalSetLoss(nn.Module):
    """Mode-aware proposal and batched-Critic distillation objective."""

    DEFAULT_WEIGHTS = {
        "mode_matching": 1.0,
        "theta_supervision": 1.0,
        "coverage": 1.0,
        "teacher_soft_best": 1.0,
        "voxel_relative_value": 1.0,
        "target_set": 1.0,
        "direction": 1.0,
        "candidate_value": 1.0,
        "ranking": 0.5,
        "hard_negative": 1.0,
        "critic_selection": 1.0,
        "high_value_diversity": 5.0,
        "geometry": 2.0,
        "interaction": 1.0,
        "uncertainty_calibration": 0.25,
        "actual_replay_value": 1.0,
        "actual_elite_imitation": 1.0,
    }

    def __init__(self, weights=None, dominance_ratio=100.0):
        super().__init__()
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({key: float(value) for key, value in weights.items()})
        self.dominance_ratio = max(float(dominance_ratio), 1.0)

    @staticmethod
    def _pairwise_descriptor_cost(proposal, teacher):
        detached = teacher.detach()
        scale = (
            detached.std(dim=1, keepdim=True)
            + 0.25 * detached.abs().mean(dim=1, keepdim=True)
        ).clamp_min(0.05)
        normalised = (
            (proposal.unsqueeze(2) - teacher.unsqueeze(1)) / scale.unsqueeze(1)
        ).abs().mean(3)
        # descriptorの単位差でmatching lossが無制限に増えないよう有界化する。
        return torch.tanh(normalised / 4.0)

    @staticmethod
    def _matched_indices(cost, teacher_mask):
        assignments = []
        for batch_index in range(cost.shape[0]):
            valid_teacher = torch.nonzero(teacher_mask[batch_index], as_tuple=False).flatten()
            if valid_teacher.numel() == 0:
                assignments.append([])
                continue
            local = cost[batch_index, :, valid_teacher]
            pairs = _hungarian_min_cost(local)
            assignments.append([
                (row, int(valid_teacher[column])) for row, column in pairs
            ])
        return assignments

    @staticmethod
    def _zero(reference):
        return reference.sum() * 0.0

    @staticmethod
    def _proposal_theta_cost(proposal_output, teacher, reference):
        """現在出力に存在する識別可能thetaだけでpair costを作る。"""
        teacher_theta = teacher.get("theta")
        if not isinstance(teacher_theta, dict):
            return None
        batch, slots = reference.shape[:2]
        teachers = int(teacher_theta["share"].shape[1])
        theta_mask = teacher_theta.get("mask")
        if torch.is_tensor(theta_mask):
            theta_mask = theta_mask.to(device=reference.device, dtype=torch.bool)
            if not theta_mask.any():
                return None
        costs = reference.new_zeros((batch, slots, teachers))
        terms = 0
        if torch.is_tensor(proposal_output.get("total_ratio")):
            predicted = proposal_output["total_ratio"].to(reference).squeeze(-1)
            target = teacher_theta["total_ratio"].to(reference)
            costs = costs + (predicted.unsqueeze(2) - target.unsqueeze(1)).abs() / 0.0025
            terms += 1
        if torch.is_tensor(proposal_output.get("shares")):
            predicted = proposal_output["shares"].to(reference)
            target = teacher_theta["share"].to(reference)
            costs = costs + (
                predicted.unsqueeze(2) - target.unsqueeze(1)
            ).abs().mean(3) / 0.20
            terms += 1
        predicted_order = proposal_output.get("operation_order")
        if not torch.is_tensor(predicted_order):
            executable = proposal_output.get("executable_plans")
            if isinstance(executable, dict):
                predicted_order = executable.get("operation_order")
            elif executable is not None:
                predicted_order = getattr(executable, "operation_order", None)
        target_order = teacher_theta.get("order_class")
        if torch.is_tensor(predicted_order) and torch.is_tensor(target_order):
            if predicted_order.ndim == 3 and predicted_order.shape[-1] == 3:
                predicted_classes = torch.full(
                    predicted_order.shape[:2], -1,
                    device=predicted_order.device, dtype=torch.long,
                )
                for order_class, permutation in enumerate(ORDER_PERMUTATIONS):
                    pattern = torch.tensor(
                        [OPERATION_INDEX[name] for name in permutation],
                        device=predicted_order.device,
                    )
                    predicted_classes = torch.where(
                        (predicted_order.long() == pattern).all(2),
                        predicted_classes.new_full((), order_class),
                        predicted_classes,
                    )
                costs = costs + (
                    predicted_classes.unsqueeze(2) != target_order.to(predicted_classes).unsqueeze(1)
                ).to(reference.dtype)
                terms += 1
        predicted_variant = proposal_output.get("variant_class")
        target_variant = teacher_theta.get("variant_class")
        if torch.is_tensor(predicted_variant) and torch.is_tensor(target_variant):
            valid_variant = target_variant >= 0
            variant_cost = (
                predicted_variant.unsqueeze(2)
                != target_variant.to(predicted_variant).unsqueeze(1)
            ).to(reference.dtype)
            costs = costs + variant_cost * valid_variant.unsqueeze(1).to(reference.dtype)
            terms += 1
        if not terms:
            return None
        costs = costs / float(terms)
        if torch.is_tensor(theta_mask):
            costs = costs * theta_mask.unsqueeze(1).to(costs.dtype)
        return costs

    @staticmethod
    def _source_pair_cost(slot_logits, teacher_value, teacher_mask):
        """slotとmode別source教師のpair BCEをmatching costへ加える。"""
        if not torch.is_tensor(teacher_value) or not torch.is_tensor(teacher_mask):
            return None
        centred = slot_logits - slot_logits.mean(dim=3, keepdim=True).detach()
        scaled = centred / centred.std(dim=3, keepdim=True).detach().clamp_min(0.25)
        probabilities = torch.sigmoid(scaled.clamp(-8.0, 8.0)).unsqueeze(2)
        target = teacher_value.to(slot_logits).unsqueeze(1)
        mask = teacher_mask.to(device=slot_logits.device, dtype=torch.bool).unsqueeze(1)
        error = (probabilities - target).abs() * mask.to(slot_logits.dtype)
        denominator = mask.sum(dim=(3, 4)).clamp_min(1).to(slot_logits.dtype)
        return error.sum(dim=(3, 4)) / denominator

    @staticmethod
    def _target_pair_cost(proposal_output, teacher, reference):
        """実行後target集合のJaccard距離をmatching専用に計算する。"""
        executable = proposal_output.get("executable_plans")
        if executable is None:
            executable = proposal_output.get("executable_plan")
        if executable is None:
            return None
        if isinstance(executable, dict):
            predicted_coord = executable.get("target_coord")
            predicted_mask = executable.get("accepted_mask")
        else:
            predicted_coord = getattr(executable, "target_coord", None)
            predicted_mask = getattr(executable, "accepted_mask", None)
        teacher_coord = teacher.get("mode_target_coord")
        teacher_mask = teacher.get("mode_target_mask")
        if not all(torch.is_tensor(value) for value in (
            predicted_coord, predicted_mask, teacher_coord, teacher_mask
        )):
            return None
        batch, slots = predicted_coord.shape[:2]
        teacher_count = teacher_coord.shape[1]
        cost = reference.new_zeros((batch, slots, teacher_count))
        operation_count = reference.new_zeros((batch, slots, teacher_count))
        # operationごとに座標universeを一度作り、K×T intersectionを行列積する。
        for batch_index in range(batch):
            for operation in (1, 2):
                predicted_rows = [
                    predicted_coord[batch_index, slot_index, operation][
                        predicted_mask[batch_index, slot_index, operation].bool()
                    ].long()
                    for slot_index in range(slots)
                ]
                teacher_rows = [
                    teacher_coord[batch_index, teacher_index, operation][
                        teacher_mask[batch_index, teacher_index, operation].bool()
                    ].long()
                    for teacher_index in range(teacher_count)
                ]
                nonempty = [rows for rows in predicted_rows + teacher_rows if rows.numel()]
                if not nonempty:
                    continue
                combined = torch.cat(nonempty, dim=0)
                minimum = combined.amin(0)
                maximum = combined.amax(0)
                spans = maximum - minimum + 1
                if int(spans[0].item()) * int(spans[1].item()) * int(spans[2].item()) >= 2 ** 62:
                    raise RuntimeError("target coordinate range is too large for exact sparse join")

                def hashes(rows):
                    shifted = rows - minimum.view(1, 3)
                    return (shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2] + shifted[:, 2]

                universe = torch.unique(hashes(combined), sorted=True)
                predicted_incidence = reference.new_zeros((slots, universe.numel()))
                teacher_incidence = reference.new_zeros((teacher_count, universe.numel()))
                for slot_index, rows in enumerate(predicted_rows):
                    if rows.numel():
                        predicted_incidence[slot_index, torch.searchsorted(universe, hashes(rows))] = 1.0
                for teacher_index, rows in enumerate(teacher_rows):
                    if rows.numel():
                        teacher_incidence[teacher_index, torch.searchsorted(universe, hashes(rows))] = 1.0
                intersection = predicted_incidence @ teacher_incidence.transpose(0, 1)
                union = (
                    predicted_incidence.sum(1, keepdim=True)
                    + teacher_incidence.sum(1).unsqueeze(0)
                    - intersection
                )
                present = union > 0
                distance = 1.0 - intersection / union.clamp_min(1.0)
                cost[batch_index] += distance * present.to(distance.dtype)
                operation_count[batch_index] += present.to(distance.dtype)
        return cost / operation_count.clamp_min(1.0)

    @staticmethod
    def _direction_logits(proposal_output):
        candidates = (
            proposal_output.get("slot_direction_logits"),
            proposal_output.get("direction_logits"),
        )
        raw = proposal_output.get("raw_proposals")
        if isinstance(raw, dict):
            candidates += (raw.get("direction_logits"),)
        for value in candidates:
            if torch.is_tensor(value) and value.ndim == 5:
                return value
        return None

    @staticmethod
    def _target_logits(proposal_output):
        for key in ("slot_target_logits", "target_logits"):
            value = proposal_output.get(key)
            if torch.is_tensor(value) and value.ndim == 4:
                return value
        raw = proposal_output.get("raw_proposals")
        if isinstance(raw, dict):
            value = raw.get("target_logits")
            if torch.is_tensor(value) and value.ndim == 4:
                return value
        return None

    @staticmethod
    def _symmetric_chamfer_chunked(predicted, target, chunk_size=1024):
        """編集点集合だけをchunk化し、全組距離Tensorの常駐を避ける。"""
        predicted_min = []
        for start in range(0, predicted.shape[0], chunk_size):
            distance = torch.cdist(
                predicted[start : start + chunk_size].float().unsqueeze(0),
                target.float().unsqueeze(0),
            )[0]
            predicted_min.append(distance.amin(1))
        target_min = []
        for start in range(0, target.shape[0], chunk_size):
            distance = torch.cdist(
                target[start : start + chunk_size].float().unsqueeze(0),
                predicted.float().unsqueeze(0),
            )[0]
            target_min.append(distance.amin(1))
        return 0.5 * (
            torch.cat(predicted_min).mean() + torch.cat(target_min).mean()
        )

    def forward(self, proposal_output, teacher):
        proposal_descriptor = proposal_output["compact_plans"]["descriptor"]
        critic_gain = proposal_output.get(
            "predicted_plan_gain", proposal_output["predicted_gain"]
        ).squeeze(-1)
        critic_geometry = proposal_output["predicted_geometry"].squeeze(-1)
        critic_interaction = proposal_output.get("predicted_interaction")
        if torch.is_tensor(critic_interaction):
            critic_interaction = critic_interaction.squeeze(-1)
        critic_uncertainty = proposal_output.get("uncertainty")
        if torch.is_tensor(critic_uncertainty):
            critic_uncertainty = critic_uncertainty.squeeze(-1).clamp_min(1e-4)
        critic_score = proposal_output["critic_score"].squeeze(-1)
        slot_logits = proposal_output["slot_logits"]
        teacher_descriptor = teacher["mode_descriptor"].to(proposal_descriptor)
        teacher_gain = teacher["actual_gain"].to(critic_gain)
        teacher_geometry = teacher["geometry"].to(critic_geometry)
        teacher_mask = teacher.get(
            "mode_mask", torch.ones_like(teacher_gain, dtype=torch.bool)
        ).to(device=critic_gain.device, dtype=torch.bool)

        pair_cost = self._pairwise_descriptor_cost(
            proposal_descriptor, teacher_descriptor
        )
        theta_cost = self._proposal_theta_cost(
            proposal_output, teacher, proposal_descriptor
        )
        if torch.is_tensor(theta_cost):
            pair_cost = pair_cost + 0.35 * theta_cost
        mode_source_value = teacher.get("mode_source_value")
        mode_source_mask = teacher.get("mode_source_mask")
        if not torch.is_tensor(mode_source_value):
            legacy_value = teacher.get("voxel_relative_value")
            legacy_mask = teacher.get("voxel_value_mask")
            if torch.is_tensor(legacy_value):
                teacher_count = teacher_descriptor.shape[1]
                mode_source_value = legacy_value.unsqueeze(1).expand(-1, teacher_count, -1, -1)
                if legacy_mask is None:
                    legacy_mask = torch.ones_like(legacy_value, dtype=torch.bool)
                mode_source_mask = legacy_mask.unsqueeze(1).expand(-1, teacher_count, -1, -1)
        source_pair_cost = self._source_pair_cost(
            slot_logits, mode_source_value, mode_source_mask
        )
        if torch.is_tensor(source_pair_cost):
            pair_cost = pair_cost + 0.35 * source_pair_cost.detach()
        target_pair_cost = self._target_pair_cost(
            proposal_output, teacher, proposal_descriptor
        )
        if torch.is_tensor(target_pair_cost):
            pair_cost = pair_cost + 0.25 * target_pair_cost
        assignments = self._matched_indices(pair_cost, teacher_mask)
        matched_cost = []
        predicted_matched = []
        target_matched = []
        predicted_geometry_matched = []
        target_geometry_matched = []
        predicted_interaction_matched = []
        target_interaction_matched = []
        predicted_uncertainty_matched = []
        matched_overlap_weight = []
        selection_targets = []
        for batch_index, pairs in enumerate(assignments):
            if not pairs:
                continue
            slot_index = torch.tensor(
                [row for row, _ in pairs], device=critic_gain.device, dtype=torch.long
            )
            teacher_index = torch.tensor(
                [column for _, column in pairs], device=critic_gain.device, dtype=torch.long
            )
            matched_cost.append(pair_cost[batch_index, slot_index, teacher_index])
            overlap_weight = torch.exp(
                -4.0 * pair_cost[batch_index, slot_index, teacher_index].detach()
            ).clamp(0.0, 1.0)
            overlap_weight = torch.where(
                overlap_weight >= 0.02, overlap_weight, torch.zeros_like(overlap_weight)
            )
            matched_overlap_weight.append(overlap_weight)
            predicted_matched.append(critic_gain[batch_index, slot_index])
            target_matched.append(teacher_gain[batch_index, teacher_index])
            predicted_geometry_matched.append(critic_geometry[batch_index, slot_index])
            target_geometry_matched.append(teacher_geometry[batch_index, teacher_index])
            if torch.is_tensor(critic_interaction) and torch.is_tensor(teacher.get("interaction")):
                predicted_interaction_matched.append(critic_interaction[batch_index, slot_index])
                target_interaction_matched.append(
                    teacher["interaction"].to(critic_interaction)[batch_index, teacher_index]
                )
            if torch.is_tensor(critic_uncertainty):
                predicted_uncertainty_matched.append(critic_uncertainty[batch_index, slot_index])
            best_pair = teacher_gain[batch_index, teacher_index].argmax()
            selection_targets.append((
                batch_index, slot_index[best_pair], overlap_weight[best_pair]
            ))

        zero = self._zero(proposal_descriptor)
        raw = {}
        raw["mode_matching"] = (
            torch.cat(matched_cost).mean() if matched_cost else zero
        )

        theta_losses = []
        teacher_theta = teacher.get("theta")
        if isinstance(teacher_theta, dict):
            ratio_logits = proposal_output.get("ratio_logits")
            share_output = proposal_output.get("shares")
            order_logits = proposal_output.get("order_logits")
            variant_logits = proposal_output.get("variant_logits")
            for batch_index, pairs in enumerate(assignments):
                for slot_index, teacher_index in pairs:
                    ratio_class = int(teacher_theta["ratio_class"][batch_index, teacher_index])
                    if torch.is_tensor(ratio_logits) and ratio_class >= 0:
                        theta_losses.append(F.cross_entropy(
                            ratio_logits[batch_index, slot_index].view(1, -1),
                            ratio_logits.new_tensor([ratio_class], dtype=torch.long),
                        ))
                    if torch.is_tensor(share_output):
                        theta_losses.append(F.smooth_l1_loss(
                            share_output[batch_index, slot_index],
                            teacher_theta["share"].to(share_output)[batch_index, teacher_index],
                            beta=0.05,
                        ))
                    order_class = int(teacher_theta["order_class"][batch_index, teacher_index])
                    if torch.is_tensor(order_logits) and order_class >= 0:
                        theta_losses.append(F.cross_entropy(
                            order_logits[batch_index, slot_index].view(1, -1),
                            order_logits.new_tensor([order_class], dtype=torch.long),
                        ))
                    variant_class = int(teacher_theta["variant_class"][batch_index, teacher_index])
                    if torch.is_tensor(variant_logits) and variant_class >= 0:
                        theta_losses.append(F.cross_entropy(
                            variant_logits[batch_index, slot_index].view(1, -1),
                            variant_logits.new_tensor([variant_class], dtype=torch.long),
                        ))
        raw["theta_supervision"] = (
            torch.stack(theta_losses).mean() if theta_losses else zero
        )

        positive_mask = teacher_mask & (teacher_gain > 0.0)
        positive_cost = pair_cost.masked_fill(~positive_mask.unsqueeze(1), float("inf"))
        coverage_distance = positive_cost.amin(dim=1)
        finite_coverage = torch.isfinite(coverage_distance)
        raw["coverage"] = (
            coverage_distance[finite_coverage].mean() if finite_coverage.any() else zero
        )

        teacher_best = teacher_gain.masked_fill(~teacher_mask, -torch.inf).amax(dim=1)
        proposal_soft_best = torch.logsumexp(critic_gain / 0.10, dim=1) * 0.10
        soft_best_error = F.smooth_l1_loss(
            proposal_soft_best, teacher_best, beta=0.25, reduction="none"
        )
        coverage_confidence = torch.exp(
            -4.0 * coverage_distance.masked_fill(
                ~finite_coverage, 10.0
            ).mean(dim=1).detach()
        )
        coverage_confidence = torch.where(
            coverage_confidence >= 0.02,
            coverage_confidence,
            torch.zeros_like(coverage_confidence),
        )
        raw["teacher_soft_best"] = (
            soft_best_error * coverage_confidence
        ).sum() / coverage_confidence.sum().clamp_min(1.0)

        # Hungarian対応後のslotへ個別にWhere教師を返す。slot最大値への
        # 集約はspecializationを壊すため使用しない。
        source_losses = []
        if torch.is_tensor(mode_source_value):
            mode_source_value = mode_source_value.to(slot_logits)
            mode_source_mask = mode_source_mask.to(
                device=slot_logits.device, dtype=torch.bool
            )
            centred_slot_logits = slot_logits - slot_logits.mean(
                dim=3, keepdim=True
            ).detach()
            normalised_slot_logits = centred_slot_logits / centred_slot_logits.std(
                dim=3, keepdim=True
            ).detach().clamp_min(0.25)
            normalised_slot_logits = normalised_slot_logits.clamp(-8.0, 8.0)
            for batch_index, pairs in enumerate(assignments):
                for slot_index, teacher_index in pairs:
                    mask = mode_source_mask[batch_index, teacher_index]
                    if mask.any():
                        target_value = mode_source_value[batch_index, teacher_index][mask]
                        positive = target_value > 0.5
                        positive_count = positive.sum().to(slot_logits.dtype)
                        negative_count = (~positive).sum().to(slot_logits.dtype)
                        positive_weight = (
                            negative_count / positive_count.clamp_min(1.0)
                        ).clamp(1.0, 50.0)
                        source_losses.append(F.binary_cross_entropy_with_logits(
                            normalised_slot_logits[batch_index, slot_index][mask],
                            target_value,
                            pos_weight=positive_weight,
                        ))
        raw["voxel_relative_value"] = (
            torch.stack(source_losses).mean() if source_losses else zero
        )

        target_losses = []
        target_logits = self._target_logits(proposal_output)
        mode_target_value = teacher.get("mode_target_value")
        mode_target_value_mask = teacher.get("mode_target_value_mask")
        if (
            torch.is_tensor(target_logits)
            and torch.is_tensor(mode_target_value)
            and target_logits.shape[-2:] == mode_target_value.shape[-2:]
        ):
            for batch_index, pairs in enumerate(assignments):
                for slot_index, teacher_index in pairs:
                    mask = mode_target_value_mask[batch_index, teacher_index].to(
                        device=target_logits.device, dtype=torch.bool
                    )
                    if mask.any():
                        target_value = mode_target_value.to(target_logits)[
                            batch_index, teacher_index
                        ][mask]
                        positive = target_value > 0.5
                        positive_count = positive.sum().to(target_logits.dtype)
                        negative_count = (~positive).sum().to(target_logits.dtype)
                        positive_weight = (
                            negative_count / positive_count.clamp_min(1.0)
                        ).clamp(1.0, 50.0)
                        target_losses.append(F.binary_cross_entropy_with_logits(
                            target_logits[batch_index, slot_index][mask],
                            target_value,
                            pos_weight=positive_weight,
                        ))
        executable = proposal_output.get("executable_plans")
        if executable is None:
            executable = proposal_output.get("executable_plan")
        if executable is not None:
            if isinstance(executable, dict):
                target_coord_ste = executable.get("target_coord_ste")
                accepted_mask = executable.get("accepted_mask")
            else:
                target_coord_ste = getattr(executable, "target_coord_ste", None)
                accepted_mask = getattr(executable, "accepted_mask", None)
            teacher_target_coord = teacher.get("mode_target_coord")
            teacher_target_mask = teacher.get("mode_target_mask")
            if all(torch.is_tensor(value) for value in (
                target_coord_ste, accepted_mask, teacher_target_coord, teacher_target_mask
            )):
                for batch_index, pairs in enumerate(assignments):
                    for slot_index, teacher_index in pairs:
                        for operation in (1, 2):
                            predicted = target_coord_ste[batch_index, slot_index, operation][
                                accepted_mask[batch_index, slot_index, operation].bool()
                            ]
                            target = teacher_target_coord[batch_index, teacher_index, operation][
                                teacher_target_mask[batch_index, teacher_index, operation].bool()
                            ].to(target_coord_ste)
                            if predicted.numel() and target.numel():
                                chamfer = self._symmetric_chamfer_chunked(
                                    predicted, target
                                )
                                target_losses.append(
                                    1.0 - torch.exp(
                                        -0.125 * chamfer
                                    )
                                )
        raw["target_set"] = torch.stack(target_losses).mean() if target_losses else zero

        direction_losses = []
        direction_logits = self._direction_logits(proposal_output)
        direction_target = teacher.get("mode_direction_index")
        direction_mask = teacher.get("mode_direction_mask")
        if torch.is_tensor(direction_logits) and torch.is_tensor(direction_target):
            # 対応shapeは[B,K,2,26,M]。全slot方向をCritic前に出した経路だけを教師化する。
            if direction_logits.shape[2] == 2 and direction_logits.shape[3] == 26:
                if direction_logits.shape[-1] != direction_target.shape[-1]:
                    shortlist = proposal_output.get("shortlist_indices")
                    if not torch.is_tensor(shortlist):
                        raise RuntimeError("dense direction logits require shortlist_indices")
                    gather_index = shortlist[:, None, None, None, :].expand(
                        -1, direction_logits.shape[1], 2, 26, -1
                    )
                    direction_logits = torch.gather(
                        direction_logits, 4, gather_index
                    )
                for batch_index, pairs in enumerate(assignments):
                    for slot_index, teacher_index in pairs:
                        mask = direction_mask[batch_index, teacher_index].to(
                            device=direction_logits.device, dtype=torch.bool
                        )
                        if mask.any():
                            logits = direction_logits[batch_index, slot_index].permute(0, 2, 1)
                            logits = logits - logits.mean(dim=2, keepdim=True).detach()
                            logits = logits / logits.std(
                                dim=2, keepdim=True
                            ).detach().clamp_min(0.25)
                            direction_losses.append(F.cross_entropy(
                                logits.clamp(-8.0, 8.0)[mask],
                                direction_target.to(direction_logits.device)[batch_index, teacher_index][mask],
                            ))
        raw["direction"] = (
            torch.stack(direction_losses).mean() if direction_losses else zero
        )

        if predicted_matched:
            predicted_values = torch.cat(predicted_matched)
            target_values = torch.cat(target_matched)
            overlap_weights = torch.cat(matched_overlap_weight).to(predicted_values)
            raw["candidate_value"] = (
                F.smooth_l1_loss(
                    predicted_values, target_values, beta=0.25, reduction="none"
                ) * overlap_weights
            ).sum() / overlap_weights.sum().clamp_min(1.0)
            predicted_geometries = torch.cat(predicted_geometry_matched)
            target_geometries = torch.cat(target_geometry_matched)
            raw["geometry"] = (
                F.smooth_l1_loss(
                    predicted_geometries, target_geometries,
                    beta=0.25, reduction="none",
                ) * overlap_weights
            ).sum() / overlap_weights.sum().clamp_min(1.0)
            ranking_losses = []
            hard_negative_losses = []
            # 異なるstate間のgainは比較せず、各batch/state内だけで順位を学習する。
            for batch_index, pairs in enumerate(assignments):
                if len(pairs) < 2:
                    continue
                slots = torch.tensor([row for row, _ in pairs], device=critic_gain.device)
                modes = torch.tensor([column for _, column in pairs], device=critic_gain.device)
                predicted_state = critic_gain[batch_index, slots]
                target_state = teacher_gain[batch_index, modes]
                state_overlap = torch.exp(
                    -4.0 * pair_cost[batch_index, slots, modes].detach()
                ).clamp(0.0, 1.0)
                state_overlap = torch.where(
                    state_overlap >= 0.02,
                    state_overlap,
                    torch.zeros_like(state_overlap),
                )
                target_difference = target_state[:, None] - target_state[None, :]
                predicted_difference = predicted_state[:, None] - predicted_state[None, :]
                ordered = target_difference.abs() > 1e-4
                if ordered.any():
                    pair_overlap = torch.minimum(
                        state_overlap[:, None], state_overlap[None, :]
                    )
                    ordered_weight = pair_overlap[ordered]
                    ranking_losses.append(F.softplus(
                        -target_difference.sign()[ordered] * predicted_difference[ordered]
                    ).mul(ordered_weight).sum() / ordered_weight.sum().clamp_min(1.0))
                lower_quantile = torch.quantile(target_state.detach(), 0.25)
                relative_negative = target_state <= lower_quantile
                false_positive = predicted_state >= predicted_state.detach().median()
                negative = relative_negative & false_positive
                if negative.any():
                    margin = target_state.detach().median()
                    negative_weight = state_overlap[negative]
                    hard_negative_losses.append(
                        (F.softplus(predicted_state[negative] - margin) * negative_weight).sum()
                        / negative_weight.sum().clamp_min(1.0)
                    )
            raw["ranking"] = torch.stack(ranking_losses).mean() if ranking_losses else zero
            raw["hard_negative"] = (
                torch.stack(hard_negative_losses).mean() if hard_negative_losses else zero
            )
        else:
            for name in ("candidate_value", "geometry", "ranking", "hard_negative"):
                raw[name] = zero

        raw["interaction"] = (
            (
                F.smooth_l1_loss(
                    torch.cat(predicted_interaction_matched),
                    torch.cat(target_interaction_matched),
                    beta=0.25, reduction="none",
                ) * torch.cat(matched_overlap_weight).to(
                    torch.cat(predicted_interaction_matched)
                )
            ).sum() / torch.cat(matched_overlap_weight).sum().clamp_min(1.0)
            if predicted_interaction_matched else zero
        )
        if predicted_uncertainty_matched and predicted_matched:
            sigma = torch.cat(predicted_uncertainty_matched).clamp_min(1e-4)
            residual = (
                torch.cat(predicted_matched) - torch.cat(target_matched)
            ).detach().abs()
            raw["uncertainty_calibration"] = (
                F.smooth_l1_loss(
                    sigma, residual.clamp_min(1e-4), beta=0.25, reduction="none"
                ) * torch.cat(matched_overlap_weight).to(sigma)
            ).sum() / torch.cat(matched_overlap_weight).sum().clamp_min(1.0)
        else:
            raw["uncertainty_calibration"] = zero

        selection_losses = []
        for batch_index, target_slot, overlap_weight in selection_targets:
            selection_losses.append(
                F.cross_entropy(
                    critic_score[batch_index:batch_index + 1], target_slot.view(1)
                ) * overlap_weight
            )
        raw["critic_selection"] = (
            torch.stack(selection_losses).mean() if selection_losses else zero
        )

        normalized_descriptor = F.normalize(proposal_descriptor, dim=2, eps=1e-6)
        similarity = torch.matmul(normalized_descriptor, normalized_descriptor.transpose(1, 2))
        off_diagonal = ~torch.eye(
            similarity.shape[1], device=similarity.device, dtype=torch.bool
        ).unsqueeze(0)
        high_value_slot = torch.zeros_like(critic_gain, dtype=torch.bool)
        explicit_high_value = teacher.get("high_value_mask")
        for batch_index, pairs in enumerate(assignments):
            if not pairs:
                continue
            valid_targets = teacher_gain[batch_index, teacher_mask[batch_index]]
            threshold = torch.quantile(valid_targets.detach(), 0.75)
            for slot_index, teacher_index in pairs:
                is_high = teacher_gain[batch_index, teacher_index] >= threshold
                if torch.is_tensor(explicit_high_value):
                    is_high = is_high | explicit_high_value[batch_index, teacher_index]
                high_value_slot[batch_index, slot_index] = bool(is_high)
        high_value_pair = high_value_slot.unsqueeze(2) & high_value_slot.unsqueeze(1)
        diversity_mask = off_diagonal.expand_as(similarity) & high_value_pair
        raw["high_value_diversity"] = (
            (F.softplus((similarity[diversity_mask] - 0.80) * 5.0) / 5.0).mean()
            if diversity_mask.any() else zero
        )

        replay_gain = teacher.get("actual_replay_gain")
        replay_mask = teacher.get("actual_replay_mask")
        actual_k_oracle = None
        actual_k_oracle_count = 0
        if torch.is_tensor(replay_gain) and torch.is_tensor(replay_mask):
            replay_gain = replay_gain.to(critic_gain)
            replay_mask = replay_mask.to(device=critic_gain.device, dtype=torch.bool)
            if replay_mask.any():
                raw["actual_replay_value"] = F.smooth_l1_loss(
                    critic_gain[replay_mask], replay_gain[replay_mask], beta=0.25
                )
                per_state = replay_gain.masked_fill(~replay_mask, -torch.inf).amax(1)
                valid_state = replay_mask.any(1)
                actual_k_oracle = per_state[valid_state].mean().detach()
                actual_k_oracle_count = int(valid_state.sum().item())
            else:
                raw["actual_replay_value"] = zero
        else:
            raw["actual_replay_value"] = zero
        elite_nll = proposal_output.get("actual_elite_imitation_nll")
        raw["actual_elite_imitation"] = (
            elite_nll.mean() if torch.is_tensor(elite_nll) and actual_k_oracle_count else zero
        )

        weighted = {
            name: value * float(self.weights.get(name, 0.0))
            for name, value in raw.items()
        }
        total = torch.stack(tuple(weighted.values())).sum()
        nonzero_magnitudes = [
            abs(float(value.detach().cpu()))
            for value in weighted.values()
            if abs(float(value.detach().cpu())) > 1e-12
        ]
        dominance_warning = False
        dominance_ratio = 0.0
        if len(nonzero_magnitudes) >= 2:
            dominance_ratio = max(nonzero_magnitudes) / max(min(nonzero_magnitudes), 1e-12)
            dominance_warning = dominance_ratio > self.dominance_ratio
        return {
            "total": total,
            "raw": raw,
            "weights": dict(self.weights),
            "weighted": weighted,
            "dominance_ratio": dominance_ratio,
            "dominance_warning": dominance_warning,
            "assignments": assignments,
            "metrics": {
                "teacher_soft_best": teacher_best.detach().mean(),
                "predicted_critic_soft_best": proposal_soft_best.detach().mean(),
                "actual_k_oracle": actual_k_oracle,
                "actual_k_oracle_state_count": actual_k_oracle_count,
            },
        }
