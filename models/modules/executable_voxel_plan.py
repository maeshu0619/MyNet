"""K個の提案を同一の実行可能Voxel plan契約へ正規化する。

このmoduleはNetwork、Actuator、codecをimportしない。入力点群全体をK個複製せず、
選ばれたsource/targetだけを疎Tensorとして保持する。
"""

from dataclasses import dataclass, fields
import hashlib
from typing import Callable, List, Optional, Tuple, Union

import torch


OPERATION_NAMES = ("Prune", "Add", "Adjust")
PRUNE, ADD, ADJUST = range(3)
REJECT_REASON_NAMES = (
    "source_duplicate",
    "source_already_removed",
    "target_occupied",
    "target_duplicate",
    "invalid_direction",
    "target_domain_invalid",
    "source_target_conflict",
    "operation_disabled",
    "budget_exceeded",
)


@dataclass
class ExecutableVoxelPlanBatch:
    """疎な実行plan。候補軸Lは最大requested countだけを保持する。"""

    operation_order: torch.Tensor       # [B,K,3]
    requested_count: torch.Tensor       # [B,K,3]
    accepted_count: torch.Tensor        # [B,K,3]
    source_index: torch.Tensor           # [B,K,3,L], padding=-1
    source_coord: torch.Tensor           # [B,K,3,L,3]
    direction_index: torch.Tensor        # [B,K,3,L], Prune=-1
    target_coord: torch.Tensor           # [B,K,3,L,3]
    target_coord_ste: torch.Tensor       # [B,K,3,L,3] hard座標・soft方向勾配
    accepted_mask: torch.Tensor          # [B,K,3,L]
    reject_mask: torch.Tensor            # [B,K,3,L] (要求budgetの未充足位置)
    reject_reason_count: torch.Tensor    # [B,K,3,R]
    plan_descriptor: torch.Tensor        # [B,K,29]
    final_count: torch.Tensor            # [B,K]
    plan_hash: Optional[List[List[str]]] = None


def _row_membership(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """[Q,3]の各rowが[R,3]に存在するかをtorch.uniqueだけで求める。"""
    if query.numel() == 0 or reference.numel() == 0:
        return torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
    both = torch.cat((reference.to(torch.long), query.to(torch.long)), dim=0)
    _, inverse = torch.unique(both, dim=0, sorted=True, return_inverse=True)
    ref_ids = torch.unique(inverse[: reference.shape[0]], sorted=True)
    query_ids = inverse[reference.shape[0] :]
    pos = torch.searchsorted(ref_ids, query_ids)
    inside = pos < ref_ids.numel()
    safe = pos.clamp(max=max(int(ref_ids.numel()) - 1, 0))
    return inside & (ref_ids[safe] == query_ids)


def coordinate_indices(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """[Q,3]を[R,3]へ疎joinし、一致しないrowは-1を返す。"""
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != 3 or reference.shape[1] != 3:
        raise ValueError("query/reference must be [N,3]")
    if query.numel() == 0 or reference.numel() == 0:
        return torch.full((query.shape[0],), -1, dtype=torch.long, device=query.device)
    reference = reference.to(device=query.device, dtype=torch.long)
    query = query.to(torch.long)
    _, inverse = torch.unique(
        torch.cat((reference, query), dim=0),
        dim=0, sorted=True, return_inverse=True,
    )
    reference_ids = inverse[: reference.shape[0]]
    query_ids = inverse[reference.shape[0] :]
    order = torch.argsort(reference_ids)
    sorted_ids = reference_ids.index_select(0, order)
    position = torch.searchsorted(sorted_ids, query_ids)
    inside = position < sorted_ids.numel()
    safe = position.clamp_max(max(int(sorted_ids.numel()) - 1, 0))
    matched = inside & (sorted_ids.index_select(0, safe) == query_ids)
    result = torch.full_like(query_ids, -1)
    result[matched] = order.index_select(0, safe[matched])
    return result


def _first_row_mask(rows: torch.Tensor) -> torch.Tensor:
    """score順Tensorに対し同一rowの最初だけを残す。"""
    if rows.numel() == 0:
        return torch.zeros(rows.shape[0], dtype=torch.bool, device=rows.device)
    _, inverse = torch.unique(rows.to(torch.long), dim=0, sorted=True, return_inverse=True)
    position = torch.arange(inverse.numel(), device=rows.device, dtype=torch.long)
    order = torch.argsort(inverse.to(torch.long) * (inverse.numel() + 1) + position)
    sorted_inverse = inverse.index_select(0, order)
    first = torch.ones_like(sorted_inverse, dtype=torch.bool)
    if first.numel() > 1:
        first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
    result = torch.zeros_like(first)
    result[order[first]] = True
    return result


def _remove_rows(base: torch.Tensor, removed: torch.Tensor) -> torch.Tensor:
    return base[~_row_membership(base, removed)] if removed.numel() else base


def _append_rows(base: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
    if added.numel() == 0:
        return base
    return torch.unique(torch.cat((base, added.to(base)), dim=0), dim=0, sorted=True)


DirectionProvider = Callable[[int, int, int, torch.Tensor], torch.Tensor]
TargetValidProvider = Callable[
    [int, int, int, torch.Tensor, torch.Tensor], torch.Tensor
]


class ExecutableVoxelPlanBuilder:
    """26近傍domain上でK個のraw proposalを実行可能planへ変換する。"""

    descriptor_dim = 29

    def __init__(self, source_window_multiplier: int = 4):
        self.source_window_multiplier = max(int(source_window_multiplier), 1)
        offsets = [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)
        ]
        self.neighbor_offsets = torch.tensor(offsets, dtype=torch.long)

    @staticmethod
    def _normalise_coords(voxel_coords: torch.Tensor) -> torch.Tensor:
        if voxel_coords.ndim != 3:
            raise ValueError("voxel_coords must be [B,3,N] or [B,N,3]")
        if voxel_coords.shape[1] == 3:
            return voxel_coords.transpose(1, 2).contiguous().to(torch.long)
        if voxel_coords.shape[2] == 3:
            return voxel_coords.contiguous().to(torch.long)
        raise ValueError("voxel_coords must contain a coordinate dimension of size 3")

    @staticmethod
    def _direction_values(
        direction_logits: Optional[torch.Tensor],
        provider: Optional[DirectionProvider],
        b: int,
        k: int,
        operation: int,
        source_index: torch.Tensor,
    ) -> torch.Tensor:
        direction_operation = 0 if operation == ADD else 1
        if provider is not None:
            values = provider(b, k, operation, source_index)
        elif direction_logits is not None:
            values = direction_logits[b, k, direction_operation]
            if values.ndim != 2:
                raise ValueError("direction logits slice must be [26,N] or [N,26]")
            if values.shape[0] == 26:
                values = values.index_select(1, source_index).transpose(0, 1)
            elif values.shape[1] == 26:
                values = values.index_select(0, source_index)
            else:
                raise ValueError("direction logits must use 26-neighbor domain")
        else:
            raise ValueError("Add/Adjust require direction_logits or direction_logit_provider")
        if tuple(values.shape) != (source_index.numel(), 26):
            raise ValueError("direction provider must return [source_count,26]")
        return values

    def build(
        self,
        voxel_coords: torch.Tensor,
        operation_scores: torch.Tensor,
        requested_count: torch.Tensor,
        operation_order: torch.Tensor,
        direction_logits: Optional[torch.Tensor] = None,
        direction_logit_provider: Optional[DirectionProvider] = None,
        direction_valid_mask: Optional[torch.Tensor] = None,
        target_coord_min: Optional[torch.Tensor] = None,
        target_coord_max: Optional[torch.Tensor] = None,
        target_valid_provider: Optional[TargetValidProvider] = None,
        operation_enabled: Optional[torch.Tensor] = None,
        source_indices: Optional[torch.Tensor] = None,
        debug_hash: bool = False,
    ) -> ExecutableVoxelPlanBatch:
        coords = self._normalise_coords(voxel_coords)
        if operation_scores.ndim != 4 or operation_scores.shape[2] != 3:
            raise ValueError("operation_scores must be [B,K,3,N]")
        batch, slots, _, score_points = operation_scores.shape
        if tuple(coords.shape[:1]) != (batch,):
            raise ValueError("coordinate and score batch dimensions differ")
        if source_indices is None:
            if coords.shape[1] != score_points:
                raise ValueError("source_indices is required for shortlist scores")
            candidate_source_index = torch.arange(
                score_points, device=operation_scores.device, dtype=torch.long
            ).view(1, -1).expand(batch, -1)
        else:
            candidate_source_index = source_indices.to(
                device=operation_scores.device, dtype=torch.long
            )
            if tuple(candidate_source_index.shape) != (batch, score_points):
                raise ValueError("source_indices must be [B,score_points]")
            invalid_source = (
                (candidate_source_index < 0)
                | (candidate_source_index >= int(coords.shape[1]))
            )
            if bool(invalid_source.any()):
                raise ValueError("source_indices contains an out-of-range index")
        if tuple(requested_count.shape) != (batch, slots, 3):
            raise ValueError("requested_count must be [B,K,3]")
        if tuple(operation_order.shape) != (batch, slots, 3):
            raise ValueError("operation_order must be [B,K,3]")
        requested = requested_count.to(device=operation_scores.device, dtype=torch.long).clamp_min(0)
        order = operation_order.to(device=operation_scores.device, dtype=torch.long)
        if not torch.all(torch.sort(order, dim=2).values == order.new_tensor((0, 1, 2))):
            raise ValueError("each operation_order row must be a permutation of 0,1,2")
        enabled = (
            torch.ones_like(requested, dtype=torch.bool)
            if operation_enabled is None
            else operation_enabled.to(device=requested.device, dtype=torch.bool)
        )
        capacity = max(int(requested.max().item()), 1)
        shape = (batch, slots, 3, capacity)
        source_index = torch.full(shape, -1, dtype=torch.long, device=operation_scores.device)
        source_coord = torch.zeros(shape + (3,), dtype=torch.long, device=operation_scores.device)
        direction_index = torch.full(shape, -1, dtype=torch.long, device=operation_scores.device)
        target_coord = torch.zeros(shape + (3,), dtype=torch.long, device=operation_scores.device)
        target_coord_ste = operation_scores.new_zeros(shape + (3,))
        accepted_mask = torch.zeros(shape, dtype=torch.bool, device=operation_scores.device)
        reject_reason = torch.zeros(
            (batch, slots, 3, len(REJECT_REASON_NAMES)), dtype=torch.long,
            device=operation_scores.device,
        )
        accepted_count = torch.zeros_like(requested)
        offsets = self.neighbor_offsets.to(device=operation_scores.device)
        score_mean = operation_scores.new_zeros((batch, slots, 3))
        score_max = operation_scores.new_zeros((batch, slots, 3))

        # B/K/operationだけを制御loopとし、点ごとのPython loopは作らない。
        for b in range(batch):
            initial = torch.unique(coords[b], dim=0, sorted=True)
            for k in range(slots):
                removed = initial.new_empty((0, 3))
                added = initial.new_empty((0, 3))
                for rank in range(3):
                    operation = int(order[b, k, rank].item())
                    wanted = int(requested[b, k, operation].item())
                    if wanted <= 0:
                        continue
                    if not bool(enabled[b, k, operation]):
                        reject_reason[b, k, operation, 7] += wanted
                        continue
                    scores = operation_scores[b, k, operation]
                    finite = torch.isfinite(scores)
                    # 全点sortを避け、実要求量の小さなsource windowだけをGPU top-kする。
                    source_window = min(
                        int(scores.numel()),
                        max(wanted, wanted * self.source_window_multiplier),
                    )
                    if source_window <= 0:
                        reject_reason[b, k, operation, 8] += wanted
                        continue
                    ranked_values, ranked = torch.topk(
                        scores.masked_fill(~finite, -torch.inf),
                        k=source_window,
                        largest=True,
                        sorted=True,
                    )
                    ranked_local = ranked[torch.isfinite(ranked_values)]
                    ranked = candidate_source_index[b].index_select(0, ranked_local)
                    ranked_coords = coords[b].index_select(0, ranked)
                    unique_source = _first_row_mask(ranked_coords)
                    reject_reason[b, k, operation, 0] += int((~unique_source).sum().item())
                    ranked_local = ranked_local[unique_source]
                    ranked = ranked[unique_source]
                    ranked_coords = ranked_coords[unique_source]
                    still_present = ~_row_membership(ranked_coords, removed)
                    reject_reason[b, k, operation, 1] += int((~still_present).sum().item())
                    ranked_local = ranked_local[still_present]
                    ranked = ranked[still_present]
                    ranked_coords = ranked_coords[still_present]

                    if operation == PRUNE:
                        take = min(wanted, int(ranked.numel()))
                        reject_reason[b, k, operation, 8] += max(
                            int(ranked.numel()) - wanted, 0
                        )
                        chosen_index = ranked[:take]
                        chosen_local_index = ranked_local[:take]
                        chosen_source = ranked_coords[:take]
                        chosen_direction = ranked.new_full((take,), -1)
                        chosen_target = chosen_source
                        chosen_target_ste = chosen_target.to(operation_scores.dtype)
                        removed = _append_rows(removed, chosen_source)
                    else:
                        window = int(ranked.numel())
                        ranked = ranked[:window]
                        ranked_local = ranked_local[:window]
                        ranked_coords = ranked_coords[:window]
                        direction = self._direction_values(
                            direction_logits, direction_logit_provider,
                            b, k, operation, ranked,
                        )
                        valid_direction = torch.isfinite(direction)
                        if direction_valid_mask is not None:
                            mask = direction_valid_mask[b, k, 0 if operation == ADD else 1]
                            if mask.shape[0] == 26:
                                mask = mask.index_select(1, ranked).transpose(0, 1)
                            else:
                                mask = mask.index_select(0, ranked)
                            valid_direction &= mask.to(torch.bool)
                        reject_reason[b, k, operation, 4] += int((~valid_direction).sum().item())
                        target = ranked_coords[:, None, :] + offsets.view(1, 26, 3)
                        domain_valid = torch.ones_like(valid_direction)
                        if target_coord_min is not None:
                            lower = target_coord_min.to(device=target.device, dtype=target.dtype)
                            lower = lower[b] if lower.ndim == 2 else lower
                            domain_valid &= (target >= lower.view(1, 1, 3)).all(2)
                        if target_coord_max is not None:
                            upper = target_coord_max.to(device=target.device, dtype=target.dtype)
                            upper = upper[b] if upper.ndim == 2 else upper
                            domain_valid &= (target <= upper.view(1, 1, 3)).all(2)
                        if target_valid_provider is not None:
                            supplied = target_valid_provider(
                                b, k, operation, ranked, target
                            ).to(device=target.device, dtype=torch.bool)
                            if supplied.shape != domain_valid.shape:
                                raise ValueError("target valid provider must return [source_count,26]")
                            domain_valid &= supplied
                        reject_reason[b, k, operation, 5] += int(
                            ((~domain_valid) & valid_direction).sum().item()
                        )
                        occupied = _row_membership(target.reshape(-1, 3), initial).view(window, 26)
                        reject_reason[b, k, operation, 2] += int((occupied & valid_direction).sum().item())
                        valid = valid_direction & domain_valid & ~occupied
                        # 既採用targetとの重複およびsource-target交差をoperation順に解決する。
                        duplicate_target = _row_membership(target.reshape(-1, 3), added).view(window, 26)
                        conflict = _row_membership(target.reshape(-1, 3), removed).view(window, 26)
                        reject_reason[b, k, operation, 3] += int((duplicate_target & valid).sum().item())
                        reject_reason[b, k, operation, 6] += int((conflict & valid).sum().item())
                        valid &= ~duplicate_target & ~conflict
                        pair_score = scores.index_select(0, ranked_local).unsqueeze(1) + direction
                        flat_order = torch.argsort(
                            pair_score.masked_fill(~valid, -torch.inf).reshape(-1), descending=True
                        )
                        flat_valid = valid.reshape(-1).index_select(0, flat_order)
                        flat_order = flat_order[flat_valid]
                        pair_source_pos = torch.div(flat_order, 26, rounding_mode="floor")
                        pair_direction = flat_order.remainder(26)
                        pair_source_index = ranked.index_select(0, pair_source_pos)
                        pair_source_local_index = ranked_local.index_select(0, pair_source_pos)
                        pair_source_coord = ranked_coords.index_select(0, pair_source_pos)
                        pair_target = target.reshape(-1, 3).index_select(0, flat_order)
                        # 同一source/targetの最高score候補だけを残す。26方向全体は一時Tensorのみ。
                        source_first = _first_row_mask(pair_source_coord)
                        target_first = _first_row_mask(pair_target)
                        admissible = source_first & target_first
                        reject_reason[b, k, operation, 0] += int((~source_first).sum().item())
                        reject_reason[b, k, operation, 3] += int((source_first & ~target_first).sum().item())
                        selected_pair = admissible.nonzero(as_tuple=False).flatten()[:wanted]
                        reject_reason[b, k, operation, 8] += max(
                            int(admissible.sum().item()) - wanted, 0
                        )
                        take = int(selected_pair.numel())
                        chosen_index = pair_source_index.index_select(0, selected_pair)
                        chosen_local_index = pair_source_local_index.index_select(0, selected_pair)
                        chosen_source = pair_source_coord.index_select(0, selected_pair)
                        chosen_direction = pair_direction.index_select(0, selected_pair)
                        chosen_target = pair_target.index_select(0, selected_pair)
                        selected_direction_logits = direction.index_select(
                            0, pair_source_pos.index_select(0, selected_pair)
                        )
                        expected_offset = torch.softmax(
                            selected_direction_logits, dim=1
                        ) @ offsets.to(selected_direction_logits.dtype)
                        hard_target_float = chosen_target.to(expected_offset.dtype)
                        chosen_target_ste = (
                            hard_target_float.detach()
                            + expected_offset
                            - expected_offset.detach()
                        )
                        added = _append_rows(added, chosen_target)
                        if operation == ADJUST:
                            removed = _append_rows(removed, chosen_source)

                    accepted_count[b, k, operation] = take
                    if take:
                        source_index[b, k, operation, :take] = chosen_index
                        source_coord[b, k, operation, :take] = chosen_source
                        direction_index[b, k, operation, :take] = chosen_direction
                        target_coord[b, k, operation, :take] = chosen_target
                        target_coord_ste[b, k, operation, :take] = chosen_target_ste
                        accepted_mask[b, k, operation, :take] = True
                        chosen_scores = scores.index_select(0, chosen_local_index)
                        score_mean[b, k, operation] = chosen_scores.mean()
                        score_max[b, k, operation] = chosen_scores.max()

        budget_mask = torch.arange(capacity, device=requested.device).view(1, 1, 1, -1) < requested.unsqueeze(-1)
        reject_mask = budget_mask & ~accepted_mask
        unique_count = torch.tensor(
            [torch.unique(coords[b], dim=0).shape[0] for b in range(batch)],
            device=requested.device, dtype=torch.long,
        ).view(batch, 1)
        final_count = unique_count - accepted_count[:, :, PRUNE] + accepted_count[:, :, ADD]
        denominator = unique_count.clamp_min(1).to(operation_scores.dtype).unsqueeze(-1)
        requested_ratio = requested.to(operation_scores.dtype) / denominator
        accepted_ratio = accepted_count.to(operation_scores.dtype) / denominator
        accepted_share = accepted_count.to(operation_scores.dtype) / accepted_count.sum(2, keepdim=True).clamp_min(1)
        rejected_ratio = (requested - accepted_count).to(operation_scores.dtype) / requested.clamp_min(1).to(operation_scores.dtype)
        order_feature = order.to(operation_scores.dtype) / 2.0
        mean_offset = operation_scores.new_zeros((batch, slots, 6))
        for operation, start in ((ADD, 0), (ADJUST, 3)):
            mask = accepted_mask[:, :, operation].unsqueeze(-1)
            displacement = (target_coord[:, :, operation] - source_coord[:, :, operation]).to(operation_scores.dtype)
            mean_offset[:, :, start : start + 3] = (
                (displacement * mask).sum(2) / mask.sum(2).clamp_min(1)
            )
        total_ratio = accepted_count.sum(2, keepdim=True).to(operation_scores.dtype) / denominator
        final_ratio = final_count.to(operation_scores.dtype).unsqueeze(-1) / denominator
        descriptor = torch.cat(
            (requested_ratio, accepted_ratio, accepted_share, rejected_ratio,
             order_feature, score_mean, score_max, mean_offset, total_ratio, final_ratio), dim=2
        )
        if descriptor.shape[2] != self.descriptor_dim:
            raise RuntimeError("executable plan descriptor size mismatch")
        result = ExecutableVoxelPlanBatch(
            operation_order=order, requested_count=requested,
            accepted_count=accepted_count, source_index=source_index,
            source_coord=source_coord, direction_index=direction_index,
            target_coord=target_coord, target_coord_ste=target_coord_ste,
            accepted_mask=accepted_mask,
            reject_mask=reject_mask, reject_reason_count=reject_reason,
            plan_descriptor=descriptor, final_count=final_count,
        )
        if debug_hash:
            result.plan_hash = executable_plan_hashes(result)
        return result


def select_executable_plan(
    plan: ExecutableVoxelPlanBatch, selected_slot: torch.Tensor
) -> ExecutableVoxelPlanBatch:
    """各batchの1 slotを選び、K=1を保った契約を返す。"""
    selected_slot = selected_slot.to(device=plan.operation_order.device, dtype=torch.long).reshape(-1)
    values = {}
    for field in fields(plan):
        value = getattr(plan, field.name)
        if torch.is_tensor(value):
            index = selected_slot.view(-1, 1, *([1] * (value.ndim - 2)))
            values[field.name] = torch.gather(
                value, 1, index.expand(value.shape[0], 1, *value.shape[2:])
            )
        elif field.name == "plan_hash" and value is not None:
            values[field.name] = [[value[b][int(selected_slot[b])]] for b in range(len(value))]
        else:
            values[field.name] = value
    return ExecutableVoxelPlanBatch(**values)


def executable_plan_hashes(plan: ExecutableVoxelPlanBatch) -> List[List[str]]:
    """debug時だけCPU同期し、実行順を含む決定論的SHA256を返す。"""
    hashes = []
    batch, slots, _ = plan.operation_order.shape
    for b in range(batch):
        row = []
        for k in range(slots):
            digest = hashlib.sha256()
            for operation in plan.operation_order[b, k].detach().cpu().tolist():
                mask = plan.accepted_mask[b, k, operation]
                tensors = (
                    plan.source_coord[b, k, operation][mask],
                    plan.direction_index[b, k, operation][mask].unsqueeze(1),
                    plan.target_coord[b, k, operation][mask],
                )
                digest.update(bytes((int(operation),)))
                for tensor in tensors:
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            row.append(digest.hexdigest())
        hashes.append(row)
    return hashes


def apply_selected_executable_plan(
    voxel_coords: torch.Tensor,
    plan: ExecutableVoxelPlanBatch,
    selected_slot: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """再探索せず選択済み1 planだけを適用し、padding済みVoxel集合を返す。"""
    coords = ExecutableVoxelPlanBuilder._normalise_coords(voxel_coords)
    selected = select_executable_plan(plan, selected_slot)
    outputs = []
    for b in range(coords.shape[0]):
        current = torch.unique(coords[b], dim=0, sorted=True)
        for operation in selected.operation_order[b, 0].tolist():
            mask = selected.accepted_mask[b, 0, operation]
            sources = selected.source_coord[b, 0, operation][mask]
            targets = selected.target_coord[b, 0, operation][mask]
            if operation == PRUNE:
                current = _remove_rows(current, sources)
            elif operation == ADD:
                current = _append_rows(current, targets)
            else:
                current = _append_rows(_remove_rows(current, sources), targets)
        outputs.append(torch.unique(current, dim=0, sorted=True))
    width = max([int(value.shape[0]) for value in outputs] or [0])
    padded = coords.new_zeros((coords.shape[0], 3, width))
    valid = torch.zeros((coords.shape[0], width), dtype=torch.bool, device=coords.device)
    for b, value in enumerate(outputs):
        padded[b, :, : value.shape[0]] = value.transpose(0, 1)
        valid[b, : value.shape[0]] = True
    return padded, valid
