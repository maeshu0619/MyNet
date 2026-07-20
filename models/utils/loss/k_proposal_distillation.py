"""Offline-only losses for fitting specialized K proposal slots.

This module consumes compact tensors produced from previously executed
Heuristic candidates and their recorded Actual scalars.  It is not imported by
the inference policy and never invokes den, a codec probe, or an Actual encoder.

Voxel targets produced by the companion offline builder are *rank-weighted
relative values*.  They are observational comparisons inside one state/mode,
not causal per-voxel Actual gains.
"""

import gzip
import json
from pathlib import Path

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if payload.get("schema_version") != "mynet_kproposal_mode_dataset_v1":
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

    @staticmethod
    def _hash_coordinates(coords, minimum, span_y, span_z):
        shifted = coords.long() - minimum.view(1, 3)
        return (shifted[:, 0] * span_y + shifted[:, 1]) * span_z + shifted[:, 2]

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
        for index, mode in enumerate(modes):
            descriptor = mode.get("descriptor", ())
            if len(descriptor) != descriptor_dim:
                raise RuntimeError("offline/current plan descriptor mismatch")
            descriptors[0, index] = reference.new_tensor(descriptor)
            gains[0, index] = float(mode["actual_gain_percent"])
            geometries[0, index] = float(mode["geometry"]["D1_loss_db"])
            mode_mask[0, index] = True

        shortlist = proposal_output["shortlist_indices"]
        if shortlist.shape[0] != 1 or voxel_coords.shape[0] != 1:
            raise RuntimeError("offline sparse voxel join currently requires batch=1")
        shortlist_coords = torch.gather(
            voxel_coords.long(), 2,
            shortlist.unsqueeze(1).expand(-1, 3, -1),
        )[0].transpose(0, 1).contiguous()
        voxel_target = reference.new_zeros((1, 3, shortlist.shape[1]))
        voxel_mask = torch.zeros_like(voxel_target, dtype=torch.bool)
        rows_by_operation = {0: [], 1: [], 2: []}
        operation_index = {"Prune": 0, "Add": 1, "Adjust": 2}
        for row in state.get("voxel_relative_values", ()):
            operation = operation_index.get(str(row.get("operation", "")), -1)
            if operation < 0 or operation == 1:
                # Add target is not an occupied source label in the old schema.
                continue
            rows_by_operation[operation].append(row)
        for operation, rows in rows_by_operation.items():
            if not rows:
                continue
            teacher_coords = torch.tensor(
                [row["coord"] for row in rows], device=reference.device, dtype=torch.long
            )
            combined_min = torch.minimum(
                shortlist_coords.amin(dim=0), teacher_coords.amin(dim=0)
            )
            combined_max = torch.maximum(
                shortlist_coords.amax(dim=0), teacher_coords.amax(dim=0)
            )
            spans = combined_max - combined_min + 1
            if int(spans[0].item()) * int(spans[1].item()) * int(spans[2].item()) >= 2 ** 62:
                raise RuntimeError("coordinate range is too large for exact sparse join")
            shortlist_hash = self._hash_coordinates(
                shortlist_coords, combined_min, spans[1], spans[2]
            )
            teacher_hash = self._hash_coordinates(
                teacher_coords, combined_min, spans[1], spans[2]
            )
            sorted_hash, order = teacher_hash.sort()
            position = torch.searchsorted(sorted_hash, shortlist_hash)
            in_range = position < sorted_hash.numel()
            safe_position = position.clamp_max(max(int(sorted_hash.numel()) - 1, 0))
            matches = in_range & (sorted_hash[safe_position] == shortlist_hash)
            if matches.any():
                values = reference.new_tensor([
                    float(row["rank_weighted_relative_value"]) for row in rows
                ])
                matched_values = values[order[safe_position[matches]]]
                voxel_target[0, operation, matches] = matched_values
                voxel_mask[0, operation, matches] = True
        return {
            "mode_descriptor": descriptors,
            "actual_gain": gains,
            "geometry": geometries,
            "mode_mask": mode_mask,
            "voxel_relative_value": voxel_target,
            "voxel_value_mask": voxel_mask,
            "voxel_target_semantics": "rank_weighted_relative_value_not_causal_gain",
            "add_where_teacher_available": False,
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
        "coverage": 1.0,
        "oracle_best": 1.0,
        "voxel_relative_value": 1.0,
        "candidate_value": 1.0,
        "ranking": 0.5,
        "hard_negative": 0.5,
        "critic_selection": 1.0,
        "high_value_diversity": 5.0,
        "geometry": 2.0,
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
        ).clamp_min(1e-3)
        return ((proposal.unsqueeze(2) - teacher.unsqueeze(1)) / scale.unsqueeze(1)).abs().mean(3)

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

    def forward(self, proposal_output, teacher):
        proposal_descriptor = proposal_output["compact_plans"]["descriptor"]
        critic_gain = proposal_output.get(
            "predicted_plan_gain", proposal_output["predicted_gain"]
        ).squeeze(-1)
        critic_geometry = proposal_output["predicted_geometry"].squeeze(-1)
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
        assignments = self._matched_indices(pair_cost, teacher_mask)
        matched_cost = []
        predicted_matched = []
        target_matched = []
        predicted_geometry_matched = []
        target_geometry_matched = []
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
            predicted_matched.append(critic_gain[batch_index, slot_index])
            target_matched.append(teacher_gain[batch_index, teacher_index])
            predicted_geometry_matched.append(critic_geometry[batch_index, slot_index])
            target_geometry_matched.append(teacher_geometry[batch_index, teacher_index])
            best_pair = teacher_gain[batch_index, teacher_index].argmax()
            selection_targets.append((batch_index, slot_index[best_pair]))

        zero = self._zero(proposal_descriptor)
        raw = {}
        raw["mode_matching"] = (
            torch.cat(matched_cost).mean() if matched_cost else zero
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
        raw["oracle_best"] = F.smooth_l1_loss(
            proposal_soft_best, teacher_best, beta=0.25
        )

        voxel_target = teacher.get("voxel_relative_value", None)
        voxel_mask = teacher.get("voxel_value_mask", None)
        if torch.is_tensor(voxel_target):
            voxel_target = voxel_target.to(slot_logits)
            # Dense relative value supervises the best matching slot for every
            # operation/shortlist voxel; it is not called marginal Actual gain.
            proposal_voxel = slot_logits.amax(dim=1)
            if voxel_mask is None:
                voxel_mask = torch.ones_like(voxel_target, dtype=torch.bool)
            voxel_mask = voxel_mask.to(device=slot_logits.device, dtype=torch.bool)
            raw["voxel_relative_value"] = F.binary_cross_entropy_with_logits(
                proposal_voxel[voxel_mask], voxel_target[voxel_mask]
            ) if voxel_mask.any() else zero
        else:
            raw["voxel_relative_value"] = zero

        if predicted_matched:
            predicted_values = torch.cat(predicted_matched)
            target_values = torch.cat(target_matched)
            raw["candidate_value"] = F.smooth_l1_loss(
                predicted_values, target_values, beta=0.25
            )
            predicted_geometries = torch.cat(predicted_geometry_matched)
            target_geometries = torch.cat(target_geometry_matched)
            raw["geometry"] = F.smooth_l1_loss(
                predicted_geometries, target_geometries, beta=0.25
            )
            if predicted_values.numel() > 1:
                target_difference = target_values[:, None] - target_values[None, :]
                predicted_difference = predicted_values[:, None] - predicted_values[None, :]
                ordered = target_difference.abs() > 1e-4
                raw["ranking"] = F.softplus(
                    -target_difference.sign()[ordered] * predicted_difference[ordered]
                ).mean() if ordered.any() else zero
                hard_negative = (target_values <= 0.0) & (
                    predicted_values >= predicted_values.detach().median()
                )
                raw["hard_negative"] = F.softplus(
                    predicted_values[hard_negative]
                ).mean() if hard_negative.any() else zero
            else:
                raw["ranking"] = zero
                raw["hard_negative"] = zero
        else:
            for name in ("candidate_value", "geometry", "ranking", "hard_negative"):
                raw[name] = zero

        selection_losses = []
        for batch_index, target_slot in selection_targets:
            selection_losses.append(F.cross_entropy(
                critic_score[batch_index:batch_index + 1], target_slot.view(1)
            ))
        raw["critic_selection"] = (
            torch.stack(selection_losses).mean() if selection_losses else zero
        )

        normalized_descriptor = F.normalize(proposal_descriptor, dim=2, eps=1e-6)
        similarity = torch.matmul(normalized_descriptor, normalized_descriptor.transpose(1, 2))
        off_diagonal = ~torch.eye(
            similarity.shape[1], device=similarity.device, dtype=torch.bool
        ).unsqueeze(0)
        high_value = torch.sigmoid(critic_gain).unsqueeze(2) * torch.sigmoid(critic_gain).unsqueeze(1)
        raw["high_value_diversity"] = (
            F.relu(similarity - 0.80) * high_value
        )[off_diagonal.expand_as(similarity)].mean()

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
        }
