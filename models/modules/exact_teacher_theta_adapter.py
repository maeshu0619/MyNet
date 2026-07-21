"""保存済みden6実行planをExact Teacher用Executable契約へ変換する。

このmoduleはoffline診断・訓練専用である。推論経路からimportせず、保存Actualのない
thetaへ値を補間しない。旧schemaで欠けているAdd source/directionは欠損のまま扱う。
"""

from dataclasses import dataclass
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from .executable_voxel_plan import (
    ADJUST,
    ADD,
    PRUNE,
    ExecutableVoxelPlanBatch,
    coordinate_indices,
    executable_plan_hashes,
)


DEN6_OPERATIONS = ("Add", "Prune", "Adjust")
EXECUTABLE_OPERATIONS = ("Prune", "Add", "Adjust")
OPERATION_INDEX = {name: index for index, name in enumerate(EXECUTABLE_OPERATIONS)}
ORDER_PERMUTATIONS = tuple(itertools.permutations(DEN6_OPERATIONS))
MISSING_COORDINATE = torch.iinfo(torch.long).min


def _stable_hash(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_id(state: Mapping[str, object]) -> str:
    return "{}|{}".format(state["input_sha256"], state["setting_id"])


def _actual_rows(path: str) -> Dict[Tuple[str, str, str], Mapping[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
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
        result[key] = row
    return result


@dataclass(frozen=True)
class ExactThetaRecord:
    """Actualが実在するstate-local離散theta IDと完全実行member。"""

    state_id: str
    theta_id: str
    theta: Mapping[str, object]
    record: Mapping[str, object]
    source_actual_row: Mapping[str, object]


@dataclass
class ExactGeneratedPlans:
    """共通Executable planと、旧schema欠損を含む照合metadata。"""

    executable: ExecutableVoxelPlanBatch
    theta_ids: List[str]
    catalog_plan_keys: List[str]
    final_voxel_hashes: List[str]
    missing_add_source_count: List[int]
    missing_add_direction_count: List[int]


class ExactTeacherThetaCatalog:
    """den6でActual評価済みの離散thetaだけをimmutable catalogとして読む。"""

    def __init__(self, records: Sequence[ExactThetaRecord]):
        grouped: Dict[str, List[ExactThetaRecord]] = {}
        for record in records:
            grouped.setdefault(record.state_id, []).append(record)
        self._records = {
            state: tuple(sorted(values, key=lambda item: item.theta_id))
            for state, values in grouped.items()
        }
        self._lookup = {
            (record.state_id, record.theta_id): record
            for values in self._records.values()
            for record in values
        }

    @classmethod
    def from_actual_plan_datasets(cls, paths: Iterable[str]):
        records = []
        for path in paths:
            with gzip.open(str(path), "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            schema = str(payload.get("schema_version", ""))
            if schema not in (
                "mynet_kproposal_actual_plan_dataset_v1",
                "mynet_kproposal_actual_plan_dataset_v2",
            ):
                raise RuntimeError("未対応Actual plan schema: {}".format(schema))
            if payload.get("contains_virtual_actual_labels", True):
                raise RuntimeError("推定Actualを含むdatasetはExact catalogへ使用できない")
            source_path = str(payload.get("source_run_rows", ""))
            source_lookup = _actual_rows(source_path)
            for raw in payload.get("records", ()):
                state = dict(raw["state_key"])
                source_key = (
                    str(Path(str(state["input_file"])).resolve()),
                    str(state["setting_id"]),
                    str(raw["pattern_key"]),
                )
                source = source_lookup.get(source_key)
                if source is None:
                    raise RuntimeError("source Actual rowがない: {}".format(source_key))
                if str(source.get("plan_key", "")) != str(raw.get("plan_key", "")):
                    raise RuntimeError("source/offline plan_key不一致: {}".format(source_key))
                shares = {
                    name: float(dict(raw["shares"])[name]) for name in DEN6_OPERATIONS
                }
                order = [
                    name for name in str(source.get("operation_order", "")).split(">")
                    if name in DEN6_OPERATIONS
                ]
                if len(order) != 3:
                    raise RuntimeError("operation order欠損: {}".format(source_key))
                theta = {
                    "generator_input": {
                        "total_ratio_percent": float(raw["total_ratio_percent"]),
                        "share": shares,
                    },
                    # order/variantは自由入力ではなくden6内部比較の結果である。
                    "generator_outcome": {
                        "operation_order": order,
                        "variant_index": int(source.get("variant_index", -1)),
                    },
                    "fixed_generator": {
                        "plan_variant_count": 6,
                        "score_coefficients_learned": False,
                    },
                }
                theta_id = _stable_hash({
                    "state_id": _state_id(state),
                    "pattern_key": str(raw["pattern_key"]),
                    "theta": theta,
                })[:24]
                records.append(ExactThetaRecord(
                    state_id=_state_id(state), theta_id=theta_id,
                    theta=theta, record=raw, source_actual_row=source,
                ))
        return cls(records)

    @property
    def state_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._records))

    def records(self, state_id: str) -> Tuple[ExactThetaRecord, ...]:
        if state_id not in self._records:
            raise KeyError("未知state_id: {}".format(state_id))
        return self._records[state_id]

    def get(self, state_id: str, theta_id: str) -> ExactThetaRecord:
        return self._lookup[(state_id, theta_id)]

    def generate(
        self,
        state_id: str,
        theta_ids: Sequence[str],
        voxel_coords: torch.Tensor,
        debug_hash: bool = False,
    ) -> ExactGeneratedPlans:
        """catalog内thetaを、再順位付けせず共通Executable planへ変換する。"""
        selected = [self.get(state_id, theta_id) for theta_id in theta_ids]
        executable, missing_source, missing_direction = _pack_executed_members(
            voxel_coords, selected, debug_hash=debug_hash
        )
        return ExactGeneratedPlans(
            executable=executable,
            theta_ids=list(theta_ids),
            catalog_plan_keys=[str(item.record["plan_key"]) for item in selected],
            final_voxel_hashes=[str(item.record["final_voxel_hash"]) for item in selected],
            missing_add_source_count=missing_source,
            missing_add_direction_count=missing_direction,
        )


def _pack_executed_members(
    voxel_coords: torch.Tensor,
    records: Sequence[ExactThetaRecord],
    debug_hash: bool,
) -> Tuple[ExecutableVoxelPlanBatch, List[int], List[int]]:
    """den6実行済みmemberを疎planへpackする。候補生成や衝突再解決はしない。"""
    if voxel_coords.ndim == 3:
        if voxel_coords.shape[0] != 1:
            raise ValueError("Exact Teacher adapterは1 stateずつ生成する")
        coords = voxel_coords[0].transpose(0, 1) if voxel_coords.shape[1] == 3 else voxel_coords[0]
    elif voxel_coords.ndim == 2 and voxel_coords.shape[1] == 3:
        coords = voxel_coords
    else:
        raise ValueError("voxel_coordsは[N,3]または[1,3,N]でなければならない")
    coords = coords.to(dtype=torch.long)
    slots = len(records)
    requested_values = []
    accepted_values = []
    orders = []
    grouped_members = []
    missing_sources = []
    missing_directions = []
    for item in records:
        requested_map = dict(item.record["requested_counts"])
        accepted_map = dict(item.record["operation_counts"])
        requested_values.append([int(requested_map[name]) for name in EXECUTABLE_OPERATIONS])
        accepted_values.append([int(accepted_map[name]) for name in EXECUTABLE_OPERATIONS])
        order_names = list(item.theta["generator_outcome"]["operation_order"])
        orders.append([OPERATION_INDEX[name] for name in order_names])
        groups = {name: [] for name in EXECUTABLE_OPERATIONS}
        for member in item.record.get("candidates", ()):
            groups[str(member["operation"])].append(member)
        grouped_members.append(groups)
        missing_sources.append(sum(
            not bool(member.get("remove_coords"))
            for member in groups["Add"]
        ))
        missing_directions.append(sum(
            not bool(member.get("remove_coords"))
            for member in groups["Add"]
        ))
    requested = torch.tensor(requested_values, device=coords.device, dtype=torch.long).unsqueeze(0)
    accepted_count = torch.tensor(accepted_values, device=coords.device, dtype=torch.long).unsqueeze(0)
    operation_order = torch.tensor(orders, device=coords.device, dtype=torch.long).unsqueeze(0)
    capacity = max(int(accepted_count.max().item()), int(requested.max().item()), 1)
    shape = (1, slots, 3, capacity)
    source_index = torch.full(shape, -1, device=coords.device, dtype=torch.long)
    source_coord = torch.full(
        shape + (3,), MISSING_COORDINATE, device=coords.device, dtype=torch.long
    )
    direction_index = torch.full(shape, -1, device=coords.device, dtype=torch.long)
    target_coord = torch.full_like(source_coord, MISSING_COORDINATE)
    accepted_mask = torch.zeros(shape, device=coords.device, dtype=torch.bool)
    score_mean = torch.zeros((1, slots, 3), device=coords.device, dtype=torch.float32)
    score_max = torch.zeros_like(score_mean)
    offsets = [
        (x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)
    ]
    offset_index = {value: index for index, value in enumerate(offsets)}
    for slot, groups in enumerate(grouped_members):
        for name, operation in OPERATION_INDEX.items():
            members = groups[name]
            if len(members) != int(accepted_count[0, slot, operation]):
                raise RuntimeError("catalog member/count不一致: {} {}".format(slot, name))
            scores = []
            for position, member in enumerate(members):
                removes = list(member.get("remove_coords") or ())
                adds = list(member.get("add_coords") or ())
                source = removes[0] if removes else None
                target = adds[0] if adds else (source if operation == PRUNE else None)
                if source is not None:
                    source_tensor = torch.tensor(source, device=coords.device, dtype=torch.long)
                    source_coord[0, slot, operation, position] = source_tensor
                if target is not None:
                    target_coord[0, slot, operation, position] = torch.tensor(
                        target, device=coords.device, dtype=torch.long
                    )
                if source is not None and target is not None and operation != PRUNE:
                    delta = tuple(int(target[axis]) - int(source[axis]) for axis in range(3))
                    direction_index[0, slot, operation, position] = offset_index.get(delta, -1)
                accepted_mask[0, slot, operation, position] = True
                scores.append(float(member.get("heuristic_score", 0.0)))
            if scores:
                values = torch.tensor(scores, device=coords.device, dtype=torch.float32)
                score_mean[0, slot, operation] = values.mean()
                score_max[0, slot, operation] = values.max()
    # 全sourceを一括疎joinし、memberごとの全点走査を避ける。
    source_known = accepted_mask & (source_coord[..., 0] != MISSING_COORDINATE)
    if bool(source_known.any()):
        source_index[source_known] = coordinate_indices(source_coord[source_known], coords)
    budget = torch.arange(capacity, device=coords.device).view(1, 1, 1, -1)
    reject_mask = (budget < requested.unsqueeze(-1)) & ~accepted_mask
    reject_reason_count = torch.zeros((1, slots, 3, 9), device=coords.device, dtype=torch.long)
    unique_count = int(torch.unique(coords, dim=0).shape[0])
    final_count = (
        unique_count - accepted_count[:, :, PRUNE] + accepted_count[:, :, ADD]
    )
    denominator = max(unique_count, 1)
    requested_ratio = requested.float() / denominator
    accepted_ratio = accepted_count.float() / denominator
    accepted_share = accepted_count.float() / accepted_count.sum(2, keepdim=True).clamp_min(1)
    rejected_ratio = (requested - accepted_count).float() / requested.clamp_min(1).float()
    order_feature = operation_order.float() / 2.0
    mean_offset = torch.zeros((1, slots, 6), device=coords.device, dtype=torch.float32)
    for operation, start in ((ADD, 0), (ADJUST, 3)):
        valid = accepted_mask[:, :, operation] & (
            source_coord[:, :, operation, :, 0] != MISSING_COORDINATE
        )
        displacement = (target_coord[:, :, operation] - source_coord[:, :, operation]).float()
        mean_offset[:, :, start:start + 3] = (
            (displacement * valid.unsqueeze(-1)).sum(2)
            / valid.sum(2, keepdim=True).clamp_min(1)
        )
    total_ratio = accepted_count.sum(2, keepdim=True).float() / denominator
    final_ratio = final_count.float().unsqueeze(-1) / denominator
    descriptor = torch.cat((
        requested_ratio, accepted_ratio, accepted_share, rejected_ratio,
        order_feature, score_mean, score_max, mean_offset, total_ratio, final_ratio,
    ), dim=2)
    executable = ExecutableVoxelPlanBatch(
        operation_order=operation_order, requested_count=requested,
        accepted_count=accepted_count, source_index=source_index,
        source_coord=source_coord, direction_index=direction_index,
        target_coord=target_coord, target_coord_ste=target_coord.float(),
        accepted_mask=accepted_mask, reject_mask=reject_mask,
        reject_reason_count=reject_reason_count, plan_descriptor=descriptor,
        final_count=final_count,
    )
    if debug_hash:
        executable.plan_hash = executable_plan_hashes(executable)
    return executable, missing_sources, missing_directions


class CatalogThetaSelector(nn.Module):
    """state特徴とcatalog theta記述からstate-local utilityを予測する小型selector。"""

    def __init__(self, state_dim: int, theta_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.theta_encoder = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.utility = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_features: torch.Tensor, theta_features: torch.Tensor):
        if state_features.ndim != 2 or state_features.shape[0] != 1:
            raise ValueError("診断selectorは1 state [1,S]を受け取る")
        state = self.state_encoder(state_features).expand(theta_features.shape[0], -1)
        theta = self.theta_encoder(theta_features)
        return self.utility(torch.cat((state, theta), dim=1)).squeeze(1)
