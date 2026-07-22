"""Exact Teacher CacheからSingle-Plan Studentへ与える訓練専用損失。"""

from __future__ import annotations

import gzip
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.executable_voxel_plan import coordinate_indices
from ..cache.exact_teacher_cache import SCHEMA_VERSION as EXACT_CACHE_SCHEMA, payload_checksum


OPERATIONS = ("Prune", "Add", "Adjust")
OPERATION_INDEX = {name: index for index, name in enumerate(OPERATIONS)}
ORDER_PERMUTATIONS = tuple(itertools.permutations(OPERATIONS))


class SinglePlanTeacherStore:
    """保存Actualが実在するplanだけをstate単位で読む。"""

    def __init__(self, paths: Iterable[str]):
        states: Dict[str, list] = {}
        for path in paths:
            with gzip.open(str(path), "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload.get("contains_virtual_actual_labels", True):
                raise RuntimeError("推定ActualをSingle-Plan教師へ使用できない")
            for record in payload.get("records", ()):
                state = dict(record["state_key"])
                state_id = "{}|{}".format(state["input_sha256"], state["setting_id"])
                value = dict(record)
                value["source_dataset"] = str(Path(path).resolve())
                states.setdefault(state_id, []).append(value)
        self.states = {
            key: tuple(sorted(values, key=lambda row: (
                -float(row["actual_gain_percent"]), str(row["plan_key"])
            )))
            for key, values in states.items()
        }

    @classmethod
    def from_exact_cache_root(cls, root: str):
        """Layer Aだけを検証して読み、元dataset pathへ依存しないstoreを作る。"""
        instance = cls.__new__(cls)
        states: Dict[str, list] = {}
        for path in sorted(Path(root).expanduser().resolve().glob("exact_teacher_*.json.gz")):
            with gzip.open(str(path), "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload.get("schema_version") != EXACT_CACHE_SCHEMA:
                continue
            if not payload.get("complete") or not payload.get("training_only"):
                raise RuntimeError("不完全または推論用のTeacher Cacheを拒否した")
            if str(payload.get("payload_sha256", "")) != payload_checksum(payload):
                raise RuntimeError("Teacher Cache checksum不一致: {}".format(path))
            content = dict(payload.get("content") or {})
            if not content.get("contains_only_measured_actual", False):
                raise RuntimeError("推定Actualを含むTeacher Cacheを拒否した")
            state_id = str(content.get("state_id", ""))
            if state_id:
                states.setdefault(state_id, []).extend(content.get("actual_plans", ()))
        instance.states = {}
        for key, values in states.items():
            unique = {str(row["plan_key"]): row for row in values}
            instance.states[key] = tuple(sorted(unique.values(), key=lambda row: (
                -float(row["actual_gain_percent"]), str(row["plan_key"])
            )))
        return instance

    def find(self, input_path: str, setting_id: str) -> Optional[str]:
        resolved = str(Path(input_path).resolve())
        for state_id, rows in self.states.items():
            state = rows[0]["state_key"]
            if str(Path(state["input_file"]).resolve()) == resolved and str(
                state["setting_id"]
            ) == str(setting_id):
                return state_id
        return None

    def supervision_record(self, state_id: str, coverage_step: int) -> Mapping[str, object]:
        """elite/near-best/hard-negativeを固定反復せず回す。"""
        rows = self.states[state_id]
        if not rows:
            raise RuntimeError("Teacher stateにplanがない")
        gains = [float(row["actual_gain_percent"]) for row in rows]
        best = max(gains)
        elite = [row for row in rows if float(row["actual_gain_percent"]) >= 0.90 * best]
        near = [row for row in rows if 0.60 * best <= float(row["actual_gain_percent"]) < 0.90 * best]
        # Actualで改善しているplanを「state内下位」という理由だけで負例化しない。
        # それを行うとeliteと重なるWhereを周期的に抑制し、蒸留を相殺する。
        hard = [row for row in rows if float(row["actual_gain_percent"]) <= 0.0]
        # Single-Planは複数modeを同時再現できないため、主教師はstate内Actual最良planへ
        # 固定する。4回に1回だけ下位Actualをhard negativeとして回し、悪化planの
        # Whereを抑える。near-bestの座標を正例として混ぜて最良Whereを相殺しない。
        if hard and int(coverage_step) % 4 == 3:
            role = "hard_negative"
            group = hard
            group_index = (int(coverage_step) // 4) % len(group)
        else:
            role = "elite"
            group = elite[:1]
            group_index = 0
        result = dict(group[group_index])
        result["_teacher_role"] = role
        return result


class SinglePlanDistillationLoss(nn.Module):
    """operation別Where/DirectionとAmount/share/orderを1 planへ蒸留する。"""

    def __init__(self):
        super().__init__()
        self._fixed_feature_oracle_cache = {}
        offsets = [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)
        ]
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)

    @staticmethod
    def _coords(voxel_coords: torch.Tensor) -> torch.Tensor:
        if voxel_coords.ndim == 3 and voxel_coords.shape[1] == 3:
            return voxel_coords[0].transpose(0, 1).to(torch.long)
        if voxel_coords.ndim == 3 and voxel_coords.shape[2] == 3:
            return voxel_coords[0].to(torch.long)
        if voxel_coords.ndim == 2 and voxel_coords.shape[1] == 3:
            return voxel_coords.to(torch.long)
        raise ValueError("voxel_coords shapeが不正である")

    @staticmethod
    def _positive_ranking_loss(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return logits.new_zeros(())
        unique_indices = indices.unique()
        targets = torch.zeros_like(logits)
        targets[unique_indices] = 1.0
        negative_count = max(int(logits.numel()) - int(unique_indices.numel()), 1)
        positive_weight = logits.new_tensor(
            float(negative_count) / float(max(int(unique_indices.numel()), 1))
        )
        # 全valid domainを欠落なく使い、正例数が数百でも各Voxelへ直接勾配を返す。
        dense_bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=positive_weight
        )
        # 全domain BCEだけでは約1000:1の負例にtop-k誤検出が埋もれるため、
        # 現在最上位の誤検出をTeacher正例より下へ押すlistwise補助を加える。
        negative_logits = logits.masked_fill(targets > 0.0, float("-inf"))
        hard_count = min(
            max(int(unique_indices.numel()) * 4, int(unique_indices.numel())),
            negative_count,
        )
        hard_negative = torch.topk(
            negative_logits, k=hard_count, largest=True, sorted=False
        ).values
        positive = logits.index_select(0, unique_indices)
        repeated_positive = positive.repeat(
            int(math.ceil(float(hard_count) / float(max(int(positive.numel()), 1))))
        )[:hard_count]
        hard_ranking = F.softplus(0.25 + hard_negative - repeated_positive).mean()
        return dense_bce + 2.0 * hard_ranking

    def _direction_scores(self, terms, operation_index: int, source_indices: torch.Tensor):
        vectors = terms["base_direction_vectors"][0, operation_index, :, source_indices].transpose(0, 1)
        concentration = terms["direction_concentration"][0, operation_index, 0, source_indices]
        unit_offsets = F.normalize(self.offsets.to(vectors), dim=1)
        return torch.matmul(vectors, unit_offsets.transpose(0, 1)) * concentration.unsqueeze(1)

    def forward(
        self,
        terms: Mapping[str, torch.Tensor],
        voxel_coords: torch.Tensor,
        teacher: Mapping[str, object],
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        coords = self._coords(voxel_coords)
        where = terms["base_where_logits"][0]
        candidates = list(teacher.get("candidates", ()))
        by_operation = {name: [] for name in OPERATIONS}
        for candidate in candidates:
            operation = str(candidate.get("operation", ""))
            if operation in by_operation:
                by_operation[operation].append(candidate)

        prune_sources = torch.tensor(
            [row["remove_coords"][0] for row in by_operation["Prune"] if row.get("remove_coords")],
            device=coords.device, dtype=torch.long,
        ).reshape(-1, 3)
        adjust_sources = torch.tensor(
            [row["remove_coords"][0] for row in by_operation["Adjust"] if row.get("remove_coords")],
            device=coords.device, dtype=torch.long,
        ).reshape(-1, 3)
        prune_indices = coordinate_indices(prune_sources, coords)
        adjust_indices = coordinate_indices(adjust_sources, coords)
        prune_indices = prune_indices[prune_indices >= 0]
        adjust_indices = adjust_indices[adjust_indices >= 0]
        prune_loss = self._positive_ranking_loss(where[0], prune_indices)
        adjust_loss = self._positive_ranking_loss(where[2], adjust_indices)
        teacher_role = str(teacher.get("_teacher_role", "elite"))
        positive_teacher = teacher_role != "hard_negative"
        if not positive_teacher:
            if prune_indices.numel():
                prune_loss = F.softplus(
                    where[0].index_select(0, prune_indices).mean() - where[0].median()
                )
            if adjust_indices.numel():
                adjust_loss = F.softplus(
                    where[2].index_select(0, adjust_indices).mean() - where[2].median()
                )

        # 旧schemaのAdd source/directionは捏造せず、target集合への到達確率だけを教師化する。
        add_targets = torch.tensor(
            [row["add_coords"][0] for row in by_operation["Add"] if row.get("add_coords")],
            device=coords.device, dtype=torch.long,
        ).reshape(-1, 3)
        add_target_loss = where.new_zeros(())
        reachable_add = 0
        if add_targets.numel():
            source_queries = add_targets[:, None, :] - self.offsets.to(coords).view(1, 26, 3)
            source_indices = coordinate_indices(source_queries.reshape(-1, 3), coords).view(-1, 26)
            valid = source_indices >= 0
            target_scores = []
            for target_index in range(add_targets.shape[0]):
                source = source_indices[target_index][valid[target_index]]
                if source.numel() == 0:
                    continue
                directions = torch.nonzero(valid[target_index], as_tuple=False).flatten()
                direction_score = self._direction_scores(terms, 0, source)
                pair = where[1].index_select(0, source) + direction_score.gather(
                    1, directions.view(-1, 1)
                ).squeeze(1)
                target_scores.append(torch.logsumexp(pair, dim=0))
            if target_scores:
                reachable_add = len(target_scores)
                # target到達pairを全source×26の近似partitionと比較し、logit全体の
                # 一様シフトやscale発散でlossを下げられない相対rankingにする。
                partition = torch.logsumexp(where[1], dim=0) + math.log(26.0)
                add_target_loss = F.softplus(
                    partition - torch.stack(target_scores).mean()
                )
                if not positive_teacher:
                    add_target_loss = F.softplus(
                        torch.stack(target_scores).mean() - partition
                    )

        direction_loss = where.new_zeros(())
        direction_count = 0
        if adjust_indices.numel():
            valid_rows = []
            valid_directions = []
            offset_lookup = {
                tuple(int(value) for value in row): index
                for index, row in enumerate(self.offsets.detach().cpu().tolist())
            }
            for row, source_index in zip(by_operation["Adjust"], coordinate_indices(adjust_sources, coords).tolist()):
                if source_index < 0 or not row.get("add_coords"):
                    continue
                source = row["remove_coords"][0]
                target = row["add_coords"][0]
                delta = tuple(int(target[axis]) - int(source[axis]) for axis in range(3))
                if delta in offset_lookup:
                    valid_rows.append(source_index)
                    valid_directions.append(offset_lookup[delta])
            if valid_rows:
                source_tensor = torch.tensor(valid_rows, device=coords.device, dtype=torch.long)
                target_direction = torch.tensor(valid_directions, device=coords.device, dtype=torch.long)
                direction_logits = self._direction_scores(terms, 1, source_tensor)
                if positive_teacher:
                    direction_loss = F.cross_entropy(direction_logits, target_direction)
                else:
                    selected_probability = torch.softmax(direction_logits, dim=1).gather(
                        1, target_direction.view(-1, 1)
                    ).squeeze(1)
                    direction_loss = -torch.log1p(-selected_probability.clamp(max=1.0 - 1e-6)).mean()
                direction_count = len(valid_rows)

        ratio_target = where.new_tensor(float(teacher["total_ratio_percent"]) / 100.0)
        ratio_prediction = terms["total_ratio_mean"].float().mean()
        ratio_loss = F.smooth_l1_loss(ratio_prediction, ratio_target, beta=1e-4)
        shares = teacher["shares"]
        share_target = where.new_tensor([
            float(shares["Prune"]), float(shares["Add"]), float(shares["Adjust"])
        ])
        share_loss = -(share_target * terms["shares_mean"][0, :, 0].clamp_min(1e-8).log()).sum()
        enabled_target = where.new_tensor([
            float(bool(by_operation[name])) for name in OPERATIONS
        ])
        gate_loss = F.binary_cross_entropy_with_logits(
            terms["gate_logits"][0, :, 0], enabled_target
        )
        order_names = tuple(str(teacher["operation_order"]).split(">"))
        priority = terms["priority_base_logits"][0, :, 0]
        order_loss = where.new_zeros(())
        remaining = list(range(3))
        for name in order_names:
            target_index = OPERATION_INDEX[name]
            local_target = remaining.index(target_index)
            order_loss = order_loss + F.cross_entropy(
                priority[remaining].view(1, -1),
                priority.new_tensor([local_target], dtype=torch.long),
            )
            remaining.remove(target_index)
        if not positive_teacher:
            # 悪化planのAmount/share/orderを正解として模倣しない。
            ratio_loss = ratio_loss * 0.0
            share_loss = share_loss * 0.0
            order_loss = order_loss * 0.0
            gate_loss = gate_loss * 0.0

        actual_gain = where.new_tensor(float(teacher["actual_gain_percent"]))
        geometry = teacher.get("geometry") or {}
        geometry_target = where.new_tensor(max(
            float(geometry.get("D1_loss_db", 0.0)),
            float(geometry.get("D2_loss_db", 0.0)),
        ))
        edit_target = where.new_tensor(float(teacher["total_ratio_percent"]) / 100.0)
        utility_loss = (
            F.smooth_l1_loss(terms["utility_absolute_gain"].float().mean(), actual_gain, beta=0.1)
            + F.smooth_l1_loss(terms["utility_geometry_cost"].float().mean(), geometry_target, beta=0.1)
            + F.smooth_l1_loss(terms["utility_edit_cost"].float().mean(), edit_target, beta=1e-4)
        )
        total = (
            prune_loss + adjust_loss + add_target_loss + direction_loss
            + ratio_loss + share_loss + gate_loss + order_loss + utility_loss
        )
        executable = terms.get("executable_plan")
        recalls = {
            "prune_source_reachable": float(prune_indices.numel()) / float(max(prune_sources.shape[0], 1)),
            "adjust_source_reachable": float(adjust_indices.numel()) / float(max(adjust_sources.shape[0], 1)),
            "prune_raw_topk_recall": 0.0,
            "adjust_raw_topk_recall": 0.0,
            "prune_source_recall": 0.0,
            "add_target_recall": 0.0,
            "adjust_source_recall": 0.0,
            "adjust_direction_recall": 0.0,
        }
        # Teacherをforwardへ注入せず、入力由来固定特徴だけで到達できる順位上限を
        # 一度だけ診断する。低ければoptimizerではなく表現不足である。
        oracle_key = str(teacher.get("plan_key", ""))
        fixed = terms.get("fixed_features")
        if oracle_key not in self._fixed_feature_oracle_cache and torch.is_tensor(fixed):
            oracle_metrics = {}
            for metric_prefix, expected_indices in (
                ("prune", prune_indices.unique()),
                ("adjust", adjust_indices.unique()),
            ):
                best = 0.0
                if expected_indices.numel():
                    topk_count = min(int(expected_indices.numel()), int(fixed.shape[-1]))
                    for channel in range(int(fixed.shape[1])):
                        values = fixed[0, channel]
                        for largest in (True, False):
                            selected = torch.topk(
                                values, k=topk_count, largest=largest, sorted=False
                            ).indices
                            recall = float(
                                torch.isin(expected_indices, selected).float().mean().detach().cpu()
                            )
                            best = max(best, recall)
                oracle_metrics[f"{metric_prefix}_fixed_feature_oracle_recall"] = best
            self._fixed_feature_oracle_cache[oracle_key] = oracle_metrics
        recalls.update(self._fixed_feature_oracle_cache.get(oracle_key, {}))
        # raw順位とExecutable Plan後を分けて記録し、Where学習不足と
        # valid/collision/orderによる脱落を混同しない。
        for metric_name, operation_index, expected_indices in (
            ("prune_raw_topk_recall", 0, prune_indices),
            ("adjust_raw_topk_recall", 2, adjust_indices),
        ):
            if expected_indices.numel():
                raw_topk = torch.topk(
                    where[operation_index],
                    k=min(int(expected_indices.unique().numel()), int(where.shape[-1])),
                    largest=True,
                    sorted=False,
                ).indices
                recalls[metric_name] = float(
                    torch.isin(expected_indices.unique(), raw_topk).float().mean().detach().cpu()
                )
        if executable is not None:
            teacher_sets = (prune_sources, add_targets, adjust_sources)
            plan_sets = (
                executable.source_coord[0, 0, 0][executable.accepted_mask[0, 0, 0]],
                executable.target_coord[0, 0, 1][executable.accepted_mask[0, 0, 1]],
                executable.source_coord[0, 0, 2][executable.accepted_mask[0, 0, 2]],
            )
            for name, expected, predicted in zip(
                ("prune_source_recall", "add_target_recall", "adjust_source_recall"),
                teacher_sets,
                plan_sets,
            ):
                if expected.numel():
                    recalls[name] = float(
                        (coordinate_indices(expected, predicted) >= 0).float().mean().detach().cpu()
                    )
            teacher_adjust_pairs = {
                (tuple(map(int, row["remove_coords"][0])), tuple(map(int, row["add_coords"][0])))
                for row in by_operation["Adjust"]
                if row.get("remove_coords") and row.get("add_coords")
            }
            adjust_mask = executable.accepted_mask[0, 0, 2]
            student_adjust_pairs = {
                (tuple(map(int, source)), tuple(map(int, target)))
                for source, target in zip(
                    executable.source_coord[0, 0, 2][adjust_mask].detach().cpu().tolist(),
                    executable.target_coord[0, 0, 2][adjust_mask].detach().cpu().tolist(),
                )
            }
            if teacher_adjust_pairs:
                recalls["adjust_direction_recall"] = float(
                    len(teacher_adjust_pairs & student_adjust_pairs)
                    / float(len(teacher_adjust_pairs))
                )
        metrics = {
            "prune_source_loss": float(prune_loss.detach()),
            "add_target_loss": float(add_target_loss.detach()),
            "adjust_source_loss": float(adjust_loss.detach()),
            "direction_loss": float(direction_loss.detach()),
            "ratio_loss": float(ratio_loss.detach()),
            "share_loss": float(share_loss.detach()),
            "gate_loss": float(gate_loss.detach()),
            "order_loss": float(order_loss.detach()),
            "utility_loss": float(utility_loss.detach()),
            "add_target_reachable": float(reachable_add),
            "adjust_direction_count": float(direction_count),
            "teacher_gain_percent": float(actual_gain.detach()),
            "teacher_role": teacher_role,
            **recalls,
        }
        return total, metrics
