"""Teacherを推論へ持ち込まないSparsePCGC Single-Plan Student。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .executable_voxel_plan import ExecutableVoxelPlanBuilder
from .network_only_codec_policy import NetworkOnlyCodecPolicy


class SinglePlanStudentPolicy(nn.Module):
    """1回のNetwork forwardから一意な1個のExecutable Planを生成する。"""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 48,
        max_total_ratio: float = 0.0099,
        fixed_feature_dim: int = 6,
    ):
        super().__init__()
        self.policy = NetworkOnlyCodecPolicy(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            max_total_ratio=max_total_ratio,
            fixed_feature_dim=fixed_feature_dim,
            direct_where_head=True,
        )
        self.plan_builder = ExecutableVoxelPlanBuilder(source_window_multiplier=4)
        descriptor_dim = self.plan_builder.descriptor_dim
        utility_hidden = max(int(hidden_dim), 32)
        self.utility_head = nn.Sequential(
            nn.Linear(descriptor_dim, utility_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(utility_hidden, 4),
        )
        # 未学習時のランダムUtilityで行動を変えない。出力は補助回帰専用である。
        nn.init.zeros_(self.utility_head[-1].weight)
        nn.init.zeros_(self.utility_head[-1].bias)

    @staticmethod
    def _requested_counts(terms, point_count: int) -> torch.Tensor:
        total_ratio = terms["total_ratio"].squeeze(-1).squeeze(-1)
        shares = terms["shares"].squeeze(-1)
        gates = terms["gate_hard"].squeeze(-1)
        total = torch.ceil(total_ratio * float(point_count)).to(torch.long).clamp_min(0)
        raw = shares * total.unsqueeze(1).to(shares.dtype)
        counts = torch.floor(raw).to(torch.long)
        remainder = total - counts.sum(dim=1)
        # 端数は最大fractionへ決定論的に配分する。Teacher Amountのhard overrideはしない。
        fraction_order = (raw - counts.to(raw.dtype)).argsort(dim=1, descending=True)
        for rank in range(3):
            add = (remainder > rank).to(torch.long)
            counts.scatter_add_(1, fraction_order[:, rank : rank + 1], add.unsqueeze(1))
        return counts * gates.to(torch.long)

    @staticmethod
    def _direction_provider(terms, offsets):
        vectors = terms["base_direction_vectors"]
        concentration = terms["direction_concentration"]
        unit_offsets = F.normalize(offsets.to(vectors), dim=1)

        def provider(batch_index, slot_index, operation, source_index):
            del slot_index
            operation_index = 0 if int(operation) == 1 else 1
            selected_vectors = vectors[
                batch_index, operation_index, :, source_index
            ].transpose(0, 1)
            selected_scale = concentration[
                batch_index, operation_index, 0, source_index
            ].unsqueeze(1)
            return torch.matmul(selected_vectors, unit_offsets.transpose(0, 1)) * selected_scale

        return provider

    def forward(self, features, voxel_coords, args, *, training=None, fixed_features=None):
        if training is None:
            training = self.training
        stage = str(getattr(
            args, "single_plan_training_stage", "actual_calibration"
        )).strip().lower()
        policy_training = bool(training and stage == "actual_calibration")
        terms = self.policy(
            features,
            args,
            training=policy_training,
            fixed_features=fixed_features,
        )
        if voxel_coords is None:
            raise RuntimeError("Single-Plan Studentにはcanonical voxel座標が必要である")
        point_count = int(features.shape[-1])
        requested = self._requested_counts(terms, point_count).unsqueeze(1)
        operation_order = terms["priority_order"].unsqueeze(1)
        operation_scores = terms["where_logits"].unsqueeze(1)
        enabled = terms["gate_hard"].squeeze(-1).to(torch.bool).unsqueeze(1)
        offsets = self.plan_builder.neighbor_offsets.to(device=features.device)
        executable = self.plan_builder.build(
            voxel_coords=voxel_coords,
            operation_scores=operation_scores,
            requested_count=requested,
            operation_order=operation_order,
            direction_logit_provider=self._direction_provider(terms, offsets),
            operation_enabled=enabled,
            debug_hash=bool(getattr(args, "single_plan_debug_hash", False)),
            # 推論時はplanを変えないreject詳細の集計によるGPU同期を避ける。
            collect_reject_reasons=bool(
                getattr(
                    args,
                    "single_plan_collect_reject_reasons",
                    training or bool(getattr(args, "single_plan_debug_hash", False)),
                )
            ),
            # global_voxel_coordsはOctreeが生成したcanonical occupied voxelである。
            assume_unique_coords=True,
        )
        utility_raw = self.utility_head(executable.plan_descriptor[:, 0].float())
        absolute_gain, geometry_cost, edit_cost, log_uncertainty = utility_raw.unbind(dim=1)
        uncertainty = F.softplus(log_uncertainty)
        terms.update({
            "executable_plan": executable,
            "utility_absolute_gain": absolute_gain,
            "utility_geometry_cost": geometry_cost,
            "utility_edit_cost": edit_cost,
            "utility_uncertainty": uncertainty,
            "utility_plan": absolute_gain - geometry_cost - edit_cost,
            "single_plan_count": 1,
            "selected_plan_count": 1,
            "proposal_count": 1,
            "critic_batch_count": 0,
            "selected_slot": None,
            "teacher_reference_count": 0,
            "cache_reference_count": 0,
            "den6_call_count": 0,
            "candidate_actual_encode_count": 0,
        })
        return terms
