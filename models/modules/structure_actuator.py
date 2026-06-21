import math
import time

import torch
import torch.nn as nn
from models.utils.pointcloud.sparsepcgc_voxel import (
    canonical_sparsepcgc_voxel_coords,
    restore_points_from_voxel_coords,
    sparsepcgc_effective_qs_value,
    sparsepcgc_effective_qs_tensor,
)

class StructureRepairActuator(nn.Module):
    """Apply small geometry-preserving movements that realize repair policies.

    The actuator predicts both a small displacement and a point-wise keep/drop
    gate.  This lets the downstream compression loss reach actual deletion
    decisions instead of only moving every point.
    """

    def __init__(self, in_channels, hidden_dim=64, args=None):
        super().__init__()
        self.args = args
        self.last_runtime_timing = {}
        neighbor_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        self.register_buffer(
            "neighbor_offsets",
            torch.tensor(neighbor_offsets, dtype=torch.float32),
            persistent=False,
        )
        self.move_voxel_head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, len(neighbor_offsets), 1),
        )
        self.drop_head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, 1),
        )
        self.add_head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, 1),
        )
        self.add_voxel_head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, len(neighbor_offsets), 1),
        )
        self.operation_gate_head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 3, 1),
        )
        self.subtree_move_source_head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, 1),
        )
        # Pruneの実行量をActuator特徴から推定し、削除割合も学習対象にする。
        self.drop_amount_head = nn.Conv1d(in_channels, 1, 1)
        # Addの実行量をActuator特徴から推定し、固定比率に張り付かないようにする。
        self.add_amount_head = nn.Conv1d(in_channels, 1, 1)
        # Adjustの実行量をActuator特徴から推定し、source選択数も学習対象にする。
        self.move_amount_head = nn.Conv1d(in_channels, 1, 1)
        nn.init.zeros_(self.move_voxel_head[-1].weight)
        nn.init.zeros_(self.move_voxel_head[-1].bias)
        nn.init.zeros_(self.drop_head[-1].weight)
        nn.init.zeros_(self.add_head[-1].weight)
        nn.init.zeros_(self.add_voxel_head[-1].weight)
        nn.init.zeros_(self.add_voxel_head[-1].bias)
        nn.init.zeros_(self.operation_gate_head[-1].weight)
        nn.init.zeros_(self.subtree_move_source_head[-1].weight)
        nn.init.normal_(self.drop_amount_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.add_amount_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.move_amount_head.weight, mean=0.0, std=1e-3)
        target_repair_ratio = float(
            getattr(self.args, "target_repair_ratio", getattr(self.args, "target_disp_ratio", 0.20))
        )
        init_drop_ratio = float(getattr(self.args, "repair_init_drop_ratio", 0.05))
        init_drop = init_drop_ratio / max(target_repair_ratio, 1e-6)
        init_drop = min(max(init_drop, 1e-4), 0.95)
        init_drop_bias = math.log(init_drop / max(1.0 - init_drop, 1e-6))
        nn.init.constant_(self.drop_head[-1].bias, init_drop_bias)
        init_add_ratio = min(max(float(getattr(self.args, "repair_init_add_ratio", 0.03)), 1e-4), 0.95)
        init_add_bias = math.log(init_add_ratio / max(1.0 - init_add_ratio, 1e-6))
        nn.init.constant_(self.add_head[-1].bias, init_add_bias)
        init_gate_values = (
            float(getattr(self.args, "repair_operation_gate_init_drop", 0.50)),
            float(getattr(self.args, "repair_operation_gate_init_add", 0.50)),
            float(getattr(self.args, "repair_operation_gate_init_move", 0.50)),
        )
        init_gate_bias = [
            math.log(min(max(v, 1e-4), 1.0 - 1e-4) / max(1.0 - min(max(v, 1e-4), 1.0 - 1e-4), 1e-6))
            for v in init_gate_values
        ]
        nn.init.constant_(self.operation_gate_head[-1].bias[0], init_gate_bias[0])
        nn.init.constant_(self.operation_gate_head[-1].bias[1], init_gate_bias[1])
        nn.init.constant_(self.operation_gate_head[-1].bias[2], init_gate_bias[2])
        init_subtree_move = min(
            max(float(getattr(self.args, "repair_subtree_move_source_init_prob", 0.02)), 1e-4),
            1.0 - 1e-4,
        )
        nn.init.constant_(
            self.subtree_move_source_head[-1].bias,
            math.log(init_subtree_move / max(1.0 - init_subtree_move, 1e-6)),
        )
        nn.init.constant_(self.drop_amount_head.bias, 0.0)
        nn.init.constant_(self.add_amount_head.bias, 0.0)
        nn.init.constant_(self.move_amount_head.bias, 0.0)
        self.debug_tensors = {}

    def _child_slot_target_mask(self, voxel_coords, octree_context):
        # 1. octree_cubtree.pyで作った厳密maskがあればそれを使う
        if isinstance(octree_context, dict):
            mask = octree_context.get("point_valid_empty_child_mask", None)
            if mask is not None:
                if not torch.is_tensor(mask):
                    mask = torch.as_tensor(mask)
                mask = mask.to(device=voxel_coords.device, dtype=torch.bool)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                if mask.ndim == 3 and mask.shape[1] == self.neighbor_offsets.shape[0] and mask.shape[2] == voxel_coords.shape[2]:
                    mask = mask.permute(0, 2, 1).contiguous()
                if mask.shape[0] == 1 and voxel_coords.shape[0] > 1:
                    mask = mask.expand(voxel_coords.shape[0], -1, -1)
                if mask.ndim == 3 and mask.shape[0] == voxel_coords.shape[0] and mask.shape[1] == voxel_coords.shape[2]:
                    return mask

        # 2. なければ簡易版として、voxel座標/2による同一parent制約を使う
        B, _, N = voxel_coords.shape
        offsets = self.neighbor_offsets.to(device=voxel_coords.device, dtype=torch.long)
        K = int(offsets.shape[0])

        current = voxel_coords.transpose(1, 2).contiguous()
        targets = current[:, :, None, :] + offsets.view(1, 1, K, 3)

        current_parent = torch.div(current, 2, rounding_mode="floor")
        target_parent = torch.div(targets, 2, rounding_mode="floor")
        same_parent_mask = (target_parent == current_parent[:, :, None, :]).all(dim=-1)

        return same_parent_mask

    def _fit_leaf_pattern_map(self, value, like_tensor):
        """
        Section4:
        leaf_pattern_diag内の [B,N] / [B,1,N] Tensorを
        Actuator内の [B,1,N] に揃える。
        """
        B, _, N = like_tensor.shape
        device = like_tensor.device
        dtype = like_tensor.dtype

        if not torch.is_tensor(value):
            return like_tensor.new_zeros((B, 1, N))

        out = value.to(device=device, dtype=dtype)

        if out.ndim == 1:
            out = out.view(1, 1, -1)
        elif out.ndim == 2:
            out = out.unsqueeze(1)
        elif out.ndim == 3:
            if out.shape[1] != 1:
                out = out[:, :1, :]
        else:
            return like_tensor.new_zeros((B, 1, N))

        if out.shape[0] == 1 and B > 1:
            out = out.expand(B, -1, -1).contiguous()

        if out.shape[0] != B:
            return like_tensor.new_zeros((B, 1, N))

        current_n = int(out.shape[-1])
        if current_n == N:
            pass
        elif current_n > N:
            out = out[:, :, :N].contiguous()
        elif current_n > 0:
            pad = out[:, :, -1:].expand(B, 1, N - current_n)
            out = torch.cat([out, pad], dim=2).contiguous()
        else:
            out = like_tensor.new_zeros((B, 1, N))

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _leaf_pattern_actuator_priors(self, structure, like_tensor):
        """
        Section4:
        OctreeStructureAnalysisが作ったleaf pattern候補gainを
        ActuatorのPrune/Add/Move source biasへ変換する。

        ここではforward値を変えるためのpriorとして使うだけで、
        leaf_pattern_diag自体へ勾配を戻す目的ではない。
        """
        B, _, N = like_tensor.shape
        zero = like_tensor.new_zeros((B, 1, N))

        out = {
            "enabled": False,
            "delete_prior": zero,
            "add_prior": zero,
            "move_prior": zero,
            "best_prior": zero,
            "delete_prior_mean": 0.0,
            "add_prior_mean": 0.0,
            "move_prior_mean": 0.0,
            "best_prior_mean": 0.0,
            "best_prior_max": 0.0,
        }

        if not bool(getattr(self.args, "leaf_pattern_actuator_prior", True)):
            return out

        if not isinstance(structure, dict):
            return out

        leaf_diag = structure.get("leaf_pattern_diag", None)
        if not isinstance(leaf_diag, dict):
            return out

        if not bool(leaf_diag.get("available", False)):
            return out

        scale = max(float(getattr(self.args, "leaf_pattern_actuator_prior_scale", 2.0)), 1e-6)

        if bool(leaf_diag.get("actual_oracle_enabled", False)):
            delete_prior = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_drop_score", None),
                like_tensor,
            ).clamp_min(0.0).detach()
            add_prior = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_add_score", None),
                like_tensor,
            ).clamp_min(0.0).detach()
            move_prior = zero
            best_prior = torch.maximum(delete_prior, add_prior)
            out.update(
                {
                    "enabled": True,
                    "delete_prior": delete_prior,
                    "add_prior": add_prior,
                    "move_prior": move_prior,
                    "best_prior": best_prior,
                    "delete_prior_mean": float(delete_prior.detach().float().mean().cpu()),
                    "add_prior_mean": float(add_prior.detach().float().mean().cpu()),
                    "move_prior_mean": 0.0,
                    "best_prior_mean": float(best_prior.detach().float().mean().cpu()),
                    "best_prior_max": float(best_prior.detach().float().max().cpu()),
                }
            )
            return out

        delete_gain = self._fit_leaf_pattern_map(
            leaf_diag.get("delete_nll_gain", None),
            like_tensor,
        ).clamp_min(0.0)
        add_gain = self._fit_leaf_pattern_map(
            leaf_diag.get("add_nll_gain", None),
            like_tensor,
        ).clamp_min(0.0)
        move_gain = self._fit_leaf_pattern_map(
            leaf_diag.get("move_nll_gain", None),
            like_tensor,
        ).clamp_min(0.0)

        delete_prior = torch.tanh(delete_gain * scale).clamp(0.0, 1.0).detach()
        add_prior = torch.tanh(add_gain * scale).clamp(0.0, 1.0).detach()
        move_prior = torch.tanh(move_gain * scale).clamp(0.0, 1.0).detach()
        best_prior = torch.maximum(torch.maximum(delete_prior, add_prior), move_prior)

        out.update(
            {
                "enabled": True,
                "delete_prior": delete_prior,
                "add_prior": add_prior,
                "move_prior": move_prior,
                "best_prior": best_prior,
                "delete_prior_mean": float(delete_prior.detach().float().mean().cpu()),
                "add_prior_mean": float(add_prior.detach().float().mean().cpu()),
                "move_prior_mean": float(move_prior.detach().float().mean().cpu()),
                "best_prior_mean": float(best_prior.detach().float().mean().cpu()),
                "best_prior_max": float(best_prior.detach().float().max().cpu()),
            }
        )
        return out

    def _fit_leaf_pattern_long_map(self, value, batch_size, point_count, device):
        """
        Section5:
        leaf_pattern_diag内のchild slot Tensorを [B, N] のlong Tensorに揃える。
        """
        if not torch.is_tensor(value):
            return torch.full(
                (batch_size, point_count),
                -1,
                device=device,
                dtype=torch.long,
            )

        out = value.to(device=device, dtype=torch.long)

        if out.ndim == 1:
            out = out.view(1, -1)
        elif out.ndim == 2:
            pass
        elif out.ndim == 3:
            if out.shape[1] == 1:
                out = out.squeeze(1)
            elif out.shape[2] == 1:
                out = out.squeeze(2)
            else:
                out = out[:, 0, :]
        else:
            return torch.full(
                (batch_size, point_count),
                -1,
                device=device,
                dtype=torch.long,
            )

        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, -1).contiguous()

        if out.shape[0] != batch_size:
            return torch.full(
                (batch_size, point_count),
                -1,
                device=device,
                dtype=torch.long,
            )

        current_n = int(out.shape[1])
        if current_n == point_count:
            return out.contiguous()

        if current_n > point_count:
            return out[:, :point_count].contiguous()

        if current_n > 0:
            pad = out[:, -1:].expand(batch_size, point_count - current_n)
            return torch.cat([out, pad], dim=1).contiguous()

        return torch.full(
            (batch_size, point_count),
            -1,
            device=device,
            dtype=torch.long,
        )

    def _leaf_pattern_target_direction_priors(
        self,
        structure,
        target_child_slots,
        like_tensor,
        leaf_actuator_prior=None,
    ):
        """
        Section5:
        Add/Move候補target voxelのchild slotと、
        OctreeStructureAnalysisが推奨したbest child slotを照合し、
        一致するtarget方向へlogit biasを作る。

        target_child_slots は [B, N, K] である。
        戻り値の add_target_bias / move_target_bias も [B, N, K] である。
        """
        B, N, K = target_child_slots.shape
        device = target_child_slots.device
        dtype = like_tensor.dtype

        zero_bnk = like_tensor.new_zeros((B, N, K))

        out = {
            "enabled": False,
            "add_target_bias": zero_bnk,
            "move_target_bias": zero_bnk,
            "add_target_match_ratio": 0.0,
            "move_target_match_ratio": 0.0,
            "add_target_bias_mean": 0.0,
            "move_target_bias_mean": 0.0,
        }

        if not bool(getattr(self.args, "leaf_pattern_target_direction_prior", True)):
            return out

        if not isinstance(structure, dict):
            return out

        leaf_diag = structure.get("leaf_pattern_diag", None)
        if not isinstance(leaf_diag, dict):
            return out

        if not bool(leaf_diag.get("available", False)):
            return out

        if leaf_actuator_prior is None or not isinstance(leaf_actuator_prior, dict):
            return out

        best_add_slot = self._fit_leaf_pattern_long_map(
            leaf_diag.get("best_add_child_slot", None),
            batch_size=B,
            point_count=N,
            device=device,
        )
        best_move_slot = self._fit_leaf_pattern_long_map(
            leaf_diag.get("best_move_target_child_slot", None),
            batch_size=B,
            point_count=N,
            device=device,
        )

        add_valid = best_add_slot.ge(0) & best_add_slot.le(7)
        move_valid = best_move_slot.ge(0) & best_move_slot.le(7)

        add_match = (
            target_child_slots == best_add_slot.unsqueeze(2)
        ) & add_valid.unsqueeze(2)

        move_match = (
            target_child_slots == best_move_slot.unsqueeze(2)
        ) & move_valid.unsqueeze(2)

        add_source_prior = leaf_actuator_prior.get(
            "add_prior",
            like_tensor.new_zeros((B, 1, N)),
        ).to(device=device, dtype=dtype)

        move_source_prior = leaf_actuator_prior.get(
            "move_prior",
            like_tensor.new_zeros((B, 1, N)),
        ).to(device=device, dtype=dtype)

        if add_source_prior.ndim == 3:
            add_source_prior = add_source_prior.squeeze(1)
        if move_source_prior.ndim == 3:
            move_source_prior = move_source_prior.squeeze(1)

        add_target_bias = add_match.to(dtype=dtype) * add_source_prior.unsqueeze(2)
        move_target_bias = move_match.to(dtype=dtype) * move_source_prior.unsqueeze(2)

        out.update(
            {
                "enabled": True,
                "add_target_bias": add_target_bias.detach(),
                "move_target_bias": move_target_bias.detach(),
                "add_target_match_ratio": float(add_match.to(torch.float32).mean().detach().cpu()),
                "move_target_match_ratio": float(move_match.to(torch.float32).mean().detach().cpu()),
                "add_target_bias_mean": float(add_target_bias.detach().float().mean().cpu()),
                "move_target_bias_mean": float(move_target_bias.detach().float().mean().cpu()),
            }
        )
        return out

    def _leaf_pattern_operation_masks(self, structure, like_tensor):
        """
        leaf pattern診断でNLL改善が見込めるsource候補だけを残す。

        現在のleaf pattern scoreはSparsePCGCの完全なoracleではないが、
        無関係な大量Voxel編集を許すとexact occupancy NLLが序盤から悪化する。
        そのためhard maskは「正の改善候補だけに絞る」軽い安全弁として使う。
        """
        B, _, N = like_tensor.shape
        device = like_tensor.device

        false_mask = torch.zeros((B, 1, N), device=device, dtype=torch.bool)
        out = {
            "enabled": False,
            "delete_mask": false_mask,
            "add_mask": false_mask,
            "move_mask": false_mask,
            "actual_oracle_enabled": False,
            "actual_oracle_drop_used": False,
            "actual_oracle_add_used": False,
            "actual_oracle_move_used": False,
            "actual_oracle_drop_best_percent": 0.0,
            "actual_oracle_drop_tested_count": 0,
            "actual_oracle_bad_candidate_count": 0,
            "actual_oracle_improving_candidate_count": 0,
            "actual_oracle_combo_extra_count": 0,
            "actual_oracle_generated_candidate_count": 0,
            "actual_oracle_accepted_candidate_count": 0,
            "actual_oracle_accepted_prune_count": 0,
            "actual_oracle_accepted_add_count": 0,
            "actual_oracle_accepted_adjust_count": 0,
            "actual_oracle_accepted_subtree_move_count": 0,
            "actual_oracle_accepted_parent_collapse_count": 0,
            "actual_oracle_accepted_pattern_canonicalize_count": 0,
            "actual_oracle_noop_label_count": 0,
            "actual_oracle_noop_label_weight": 0.0,
            "actual_oracle_high_rate_mppov_count": 0,
            "actual_oracle_low_prob_occupied_count": 0,
            "actual_oracle_single_child_chain_count": 0,
            "actual_oracle_context_pattern_candidate_count": 0,
            "actual_oracle_eval_count": 0,
            "actual_oracle_eval_max": 0,
            "actual_oracle_time": 0.0,
            "actual_oracle_delta_actual_percent": 0.0,
            "actual_oracle_proxy_percent": 0.0,
            "actual_oracle_geometry_percent": 0.0,
            "actual_oracle_original_actual_bits": 0.0,
            "actual_oracle_edited_actual_bits": 0.0,
            "actual_oracle_fast_diagnostic_used": False,
            "actual_oracle_fast_diagnostic_full_drop_count": 0,
            "actual_oracle_fast_diagnostic_local_drop_count": 0,
            "actual_oracle_fast_diagnostic_full_drop_ratio": 0.0,
            "actual_oracle_fast_diagnostic_local_drop_ratio": 0.0,
            "actual_oracle_drop_reason": "",
            "actual_oracle_operation": "",
            "actual_oracle_scheduled_operation": "",
            "actual_oracle_drop_bad_mask": false_mask,
            "actual_oracle_add_bad_mask": false_mask,
            "actual_oracle_move_bad_mask": false_mask,
            "actual_oracle_drop_bad_score": false_mask.to(dtype=like_tensor.dtype),
            "actual_oracle_add_bad_score": false_mask.to(dtype=like_tensor.dtype),
            "actual_oracle_move_bad_score": false_mask.to(dtype=like_tensor.dtype),
            "actual_oracle_best_add_direction_index": torch.full_like(false_mask, -1, dtype=torch.long),
            "actual_oracle_bad_add_direction_index": torch.full_like(false_mask, -1, dtype=torch.long),
            "actual_oracle_move_direction_index": torch.full_like(false_mask, -1, dtype=torch.long),
            "actual_oracle_move_bad_direction_index": torch.full_like(false_mask, -1, dtype=torch.long),
        }

        if not isinstance(structure, dict):
            return out
        leaf_diag = structure.get("leaf_pattern_diag", None)
        if not isinstance(leaf_diag, dict):
            return out

        # actual SparsePCGCで改善確認済みの候補があるstepでは、
        # empirical leaf-pattern maskではなくoracle maskを優先する。
        # force_no_edit時はdrop maskが全falseになり、全操作を明示的に止める。
        if bool(leaf_diag.get("actual_oracle_enabled", False)):
            oracle_drop_mask = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_drop_mask", None),
                like_tensor,
            ).to(dtype=torch.bool)
            oracle_add_mask = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_add_mask", None),
                like_tensor,
            ).to(dtype=torch.bool)
            oracle_move_mask = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_move_mask", None),
                like_tensor,
            ).to(dtype=torch.bool)
            oracle_drop_bad_mask = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_drop_bad_mask", None),
                like_tensor,
            ).to(dtype=torch.bool)
            oracle_add_bad_mask = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_add_bad_mask", None),
                like_tensor,
            ).to(dtype=torch.bool)
            oracle_move_bad_mask = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_move_bad_mask", None),
                like_tensor,
            ).to(dtype=torch.bool)
            oracle_drop_bad_score = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_drop_bad_score", None),
                like_tensor,
            )
            oracle_add_bad_score = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_add_bad_score", None),
                like_tensor,
            )
            oracle_move_bad_score = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_move_bad_score", None),
                like_tensor,
            )
            oracle_add_direction_index = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_best_add_direction_index", None),
                like_tensor,
            ).to(dtype=torch.long)
            oracle_bad_add_direction_index = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_bad_add_direction_index", None),
                like_tensor,
            ).to(dtype=torch.long)
            oracle_move_direction_index = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_move_direction_index", None),
                like_tensor,
            ).to(dtype=torch.long)
            oracle_bad_move_direction_index = self._fit_leaf_pattern_map(
                leaf_diag.get("actual_oracle_move_bad_direction_index", None),
                like_tensor,
            ).to(dtype=torch.long)
            out.update(
                {
                    "enabled": True,
                    "delete_mask": oracle_drop_mask,
                    "add_mask": oracle_add_mask,
                    "move_mask": oracle_move_mask,
                    "actual_oracle_enabled": True,
                    "actual_oracle_drop_used": bool(leaf_diag.get("actual_oracle_drop_used", False)),
                    "actual_oracle_add_used": bool(leaf_diag.get("actual_oracle_add_used", False)),
                    "actual_oracle_move_used": bool(leaf_diag.get("actual_oracle_move_used", False)),
                    "actual_oracle_drop_best_percent": float(
                        leaf_diag.get("actual_oracle_drop_best_percent", 0.0) or 0.0
                    ),
                    "actual_oracle_drop_tested_count": int(
                        leaf_diag.get("actual_oracle_drop_tested_count", 0) or 0
                    ),
                    "actual_oracle_bad_candidate_count": int(
                        leaf_diag.get("actual_oracle_bad_candidate_count", 0) or 0
                    ),
                    "actual_oracle_improving_candidate_count": int(
                        leaf_diag.get("actual_oracle_improving_candidate_count", 0) or 0
                    ),
                    "actual_oracle_combo_extra_count": int(
                        leaf_diag.get("actual_oracle_combo_extra_count", 0) or 0
                    ),
                    "actual_oracle_generated_candidate_count": int(
                        leaf_diag.get("actual_oracle_generated_candidate_count", 0) or 0
                    ),
                    "actual_oracle_accepted_candidate_count": int(
                        leaf_diag.get("actual_oracle_accepted_candidate_count", 0) or 0
                    ),
                    "actual_oracle_accepted_prune_count": int(
                        leaf_diag.get("actual_oracle_accepted_prune_count", 0) or 0
                    ),
                    "actual_oracle_accepted_add_count": int(
                        leaf_diag.get("actual_oracle_accepted_add_count", 0) or 0
                    ),
                    "actual_oracle_accepted_adjust_count": int(
                        leaf_diag.get("actual_oracle_accepted_adjust_count", 0) or 0
                    ),
                    "actual_oracle_accepted_subtree_move_count": int(
                        leaf_diag.get("actual_oracle_accepted_subtree_move_count", 0) or 0
                    ),
                    "actual_oracle_accepted_parent_collapse_count": int(
                        leaf_diag.get("actual_oracle_accepted_parent_collapse_count", 0) or 0
                    ),
                    "actual_oracle_accepted_pattern_canonicalize_count": int(
                        leaf_diag.get("actual_oracle_accepted_pattern_canonicalize_count", 0) or 0
                    ),
                    "actual_oracle_noop_label_count": int(
                        leaf_diag.get("actual_oracle_noop_label_count", 0) or 0
                    ),
                    "actual_oracle_noop_label_weight": float(
                        leaf_diag.get("actual_oracle_noop_label_weight", 0.0) or 0.0
                    ),
                    "actual_oracle_high_rate_mppov_count": int(
                        leaf_diag.get("actual_oracle_high_rate_mppov_count", 0) or 0
                    ),
                    "actual_oracle_low_prob_occupied_count": int(
                        leaf_diag.get("actual_oracle_low_prob_occupied_count", 0) or 0
                    ),
                    "actual_oracle_single_child_chain_count": int(
                        leaf_diag.get("actual_oracle_single_child_chain_count", 0) or 0
                    ),
                    "actual_oracle_context_pattern_candidate_count": int(
                        leaf_diag.get("actual_oracle_context_pattern_candidate_count", 0) or 0
                    ),
                    "actual_oracle_eval_count": int(
                        leaf_diag.get("actual_oracle_eval_count", 0) or 0
                    ),
                    "actual_oracle_eval_max": int(
                        leaf_diag.get("actual_oracle_eval_max", 0) or 0
                    ),
                    "actual_oracle_time": float(
                        leaf_diag.get("actual_oracle_time", 0.0) or 0.0
                    ),
                    "actual_oracle_delta_actual_percent": float(
                        leaf_diag.get("actual_oracle_delta_actual_percent", 0.0) or 0.0
                    ),
                    "actual_oracle_proxy_percent": float(
                        leaf_diag.get("actual_oracle_proxy_percent", 0.0) or 0.0
                    ),
                    "actual_oracle_geometry_percent": float(
                        leaf_diag.get("actual_oracle_geometry_percent", 0.0) or 0.0
                    ),
                    "actual_oracle_original_actual_bits": float(
                        leaf_diag.get("actual_oracle_original_actual_bits", 0.0) or 0.0
                    ),
                    "actual_oracle_edited_actual_bits": float(
                        leaf_diag.get("actual_oracle_edited_actual_bits", 0.0) or 0.0
                    ),
                    "actual_oracle_fast_diagnostic_used": bool(
                        leaf_diag.get("actual_oracle_fast_diagnostic_used", False)
                    ),
                    "actual_oracle_fast_diagnostic_full_drop_count": int(
                        leaf_diag.get("actual_oracle_fast_diagnostic_full_drop_count", 0) or 0
                    ),
                    "actual_oracle_fast_diagnostic_local_drop_count": int(
                        leaf_diag.get("actual_oracle_fast_diagnostic_local_drop_count", 0) or 0
                    ),
                    "actual_oracle_fast_diagnostic_full_drop_ratio": float(
                        leaf_diag.get("actual_oracle_fast_diagnostic_full_drop_ratio", 0.0) or 0.0
                    ),
                    "actual_oracle_fast_diagnostic_local_drop_ratio": float(
                        leaf_diag.get("actual_oracle_fast_diagnostic_local_drop_ratio", 0.0) or 0.0
                    ),
                    "actual_oracle_fast_diagnostic_full_add_count": int(
                        leaf_diag.get("actual_oracle_fast_diagnostic_full_add_count", 0) or 0
                    ),
                    "actual_oracle_fast_diagnostic_local_add_count": int(
                        leaf_diag.get("actual_oracle_fast_diagnostic_local_add_count", 0) or 0
                    ),
                    "actual_oracle_fast_diagnostic_full_add_ratio": float(
                        leaf_diag.get("actual_oracle_fast_diagnostic_full_add_ratio", 0.0) or 0.0
                    ),
                    "actual_oracle_fast_diagnostic_local_add_ratio": float(
                        leaf_diag.get("actual_oracle_fast_diagnostic_local_add_ratio", 0.0) or 0.0
                    ),
                    "actual_oracle_drop_reason": str(
                        leaf_diag.get("actual_oracle_drop_reason", "")
                    ),
                    "actual_oracle_operation": str(
                        leaf_diag.get("actual_oracle_operation", "")
                    ),
                    "actual_oracle_scheduled_operation": str(
                        leaf_diag.get("actual_oracle_scheduled_operation", "")
                    ),
                    "actual_oracle_drop_bad_mask": oracle_drop_bad_mask & (~oracle_drop_mask),
                    "actual_oracle_add_bad_mask": oracle_add_bad_mask & (~oracle_add_mask),
                    "actual_oracle_move_bad_mask": oracle_move_bad_mask & (~oracle_move_mask),
                    "actual_oracle_drop_bad_score": oracle_drop_bad_score,
                    "actual_oracle_add_bad_score": oracle_add_bad_score,
                    "actual_oracle_move_bad_score": oracle_move_bad_score,
                    "actual_oracle_best_add_direction_index": oracle_add_direction_index,
                    "actual_oracle_bad_add_direction_index": oracle_bad_add_direction_index,
                    "actual_oracle_move_direction_index": oracle_move_direction_index,
                    "actual_oracle_move_bad_direction_index": oracle_bad_move_direction_index,
                }
            )
            return out

        if not bool(getattr(self.args, "leaf_pattern_operation_mask", False)):
            return out
        if not bool(leaf_diag.get("available", False)):
            return out

        threshold = max(float(getattr(self.args, "leaf_pattern_operation_mask_gain_threshold", 0.02)), 0.0)
        delete_gain = self._fit_leaf_pattern_map(leaf_diag.get("delete_nll_gain", None), like_tensor)
        add_gain = self._fit_leaf_pattern_map(leaf_diag.get("add_nll_gain", None), like_tensor)
        move_gain = self._fit_leaf_pattern_map(leaf_diag.get("move_nll_gain", None), like_tensor)

        delete_valid = self._fit_leaf_pattern_map(
            leaf_diag.get("delete_valid_mask", None),
            like_tensor,
        ).to(dtype=torch.bool)
        add_valid = self._fit_leaf_pattern_map(
            leaf_diag.get("add_valid_mask", None),
            like_tensor,
        ).to(dtype=torch.bool)
        move_valid = self._fit_leaf_pattern_map(
            leaf_diag.get("move_valid_mask", None),
            like_tensor,
        ).to(dtype=torch.bool)

        # DeleteはSparsePCGC actual改善とleaf empirical gainの相関が弱い。
        # 実改善stepでは delete_nll_gain<=0 の候補も多いため、正gainだけで
        # hard maskすると改善候補を消してしまう。親nodeを空にしないvalid条件だけ残す。
        delete_mask = delete_valid
        add_mask = add_valid & (add_gain > threshold)
        move_mask = move_valid & (move_gain > threshold)

        if not bool((delete_mask | add_mask | move_mask).any().detach().item()):
            return out

        out.update(
            {
                "enabled": True,
                "delete_mask": delete_mask,
                "add_mask": add_mask,
                "move_mask": move_mask,
            }
        )
        return out
    
    def _context_voxel_step_and_offset(self, pts_xyz, coord_scale, octree_context):
        # 初期Octree/Subtreeメタデータがある場合は、そのglobal_qs/global_offsetを優先する。
        # これにより、Network入力前に作ったVoxel座標系をActuator内でも維持する。
        if isinstance(octree_context, dict):
            qs_raw = octree_context.get("global_qs", self._effective_qs())

            if torch.is_tensor(qs_raw):
                step = qs_raw.to(device=pts_xyz.device, dtype=pts_xyz.dtype).reshape(-1, 1, 1)
                if step.shape[0] == 1 and pts_xyz.shape[0] > 1:
                    step = step.expand(pts_xyz.shape[0], -1, -1)
                if step.shape[0] != pts_xyz.shape[0]:
                    raise ValueError("octree_context['global_qs'] batch size does not match pts_xyz.")
                step = step.clamp_min(1e-9)
            else:
                qs_value = max(float(qs_raw), 1e-9)
                step = pts_xyz.new_full((pts_xyz.shape[0], 1, 1), qs_value)

            if "global_offset" in octree_context:
                offset = octree_context["global_offset"]
                if not torch.is_tensor(offset):
                    offset = torch.as_tensor(offset)
                offset = offset.to(device=pts_xyz.device, dtype=pts_xyz.dtype)

                if offset.ndim == 1 and offset.numel() == 3:
                    offset = offset.view(1, 3, 1)
                elif offset.ndim == 2 and offset.shape[-1] == 3:
                    offset = offset.view(-1, 3, 1)
                elif offset.ndim == 2 and offset.shape[0] == 3:
                    offset = offset.unsqueeze(0)
                elif offset.ndim == 3 and offset.shape[1] == 3:
                    offset = offset[:, :, :1]
                else:
                    raise ValueError("octree_context['global_offset'] must have shape [3], [B,3], [3,1], or [B,3,1].")

                if offset.shape[0] == 1 and pts_xyz.shape[0] > 1:
                    offset = offset.expand(pts_xyz.shape[0], -1, -1)
                if offset.shape[0] != pts_xyz.shape[0]:
                    raise ValueError("octree_context['global_offset'] batch size does not match pts_xyz.")
            else:
                offset = pts_xyz.new_zeros((pts_xyz.shape[0], 3, 1))

            return step.contiguous(), offset.contiguous(), True

        # fallback時だけ従来のvoxel_stepを使う。
        step = self._voxel_step(pts_xyz, coord_scale)
        offset = pts_xyz.new_zeros((pts_xyz.shape[0], 3, 1))
        return step, offset, False


    def _voxel_centers_from_global_coords(self, voxel_coords, voxel_step, voxel_offset, dtype):
        # global voxel座標を、同じglobal_offset/global_qsに基づいて点座標へ戻す。
        return voxel_offset.to(dtype=dtype) + voxel_coords.to(dtype=dtype) * voxel_step.to(dtype=dtype)

    def _effective_qs(self):
        compress_key = (
            str(getattr(self.args, "compress", ""))
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )
        if compress_key == "sparsepcgc":
            return max(float(sparsepcgc_effective_qs_value(self.args)), 1e-9)
        if compress_key in {"gpcc", "gpcctmc3"}:
            return max(float(getattr(self.args, "gpcc_effective_qs", getattr(self.args, "qs", 1.0))), 1e-9)
        return max(float(getattr(self.args, "qs", 2.0)), 1e-9)

    def _add_enabled(self):
        return bool(getattr(self.args, "add", True))

    def _prune_enabled(self):
        return bool(getattr(self.args, "prune", True))

    def _disp_enabled(self):
        return bool(getattr(self.args, "disp", True))

    def _threshold_cap_mode(self):
        mode = str(getattr(self.args, "repair_selection_mode", "target")).strip().lower().replace("-", "_")
        return mode in {"threshold_cap", "cap", "optional", "threshold"}

    def _max_offset(self, pts_xyz, coord_scale):
        raw_max = float(getattr(self.args, "max_repair_offset", getattr(self.args, "max_disp_offset", 0.002)))
        qstep_max = float(getattr(self.args, "max_repair_qstep", 0.25)) * self._effective_qs()
        raw_max = max(raw_max, qstep_max)
        if coord_scale is None:
            return pts_xyz.new_full((pts_xyz.shape[0], 1, 1), raw_max)
        if torch.is_tensor(coord_scale):
            scale = coord_scale.to(device=pts_xyz.device, dtype=pts_xyz.dtype).reshape(-1, 1, 1)
            if scale.shape[0] == 1 and pts_xyz.shape[0] > 1:
                scale = scale.expand(pts_xyz.shape[0], -1, -1)
            return raw_max / scale.clamp_min(1e-9)
        return pts_xyz.new_full((pts_xyz.shape[0], 1, 1), raw_max / max(float(coord_scale), 1e-9))

    def _voxel_step(self, pts_xyz, coord_scale):
        compress_key = (
            str(getattr(self.args, "compress", ""))
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )
        if compress_key == "sparsepcgc":
            return sparsepcgc_effective_qs_tensor(
                pts_xyz,
                args=self.args,
                coord_scale=coord_scale,
            ).clamp_min(1e-9)
        
        qstep = self._effective_qs()
        if coord_scale is None:
            return pts_xyz.new_full((pts_xyz.shape[0], 1, 1), qstep)
        if torch.is_tensor(coord_scale):
            scale = coord_scale.to(device=pts_xyz.device, dtype=pts_xyz.dtype).reshape(-1, 1, 1)
            if scale.shape[0] == 1 and pts_xyz.shape[0] > 1:
                scale = scale.expand(pts_xyz.shape[0], -1, -1)
            return qstep / scale.clamp_min(1e-9)
        return pts_xyz.new_full((pts_xyz.shape[0], 1, 1), qstep / max(float(coord_scale), 1e-9))

    @staticmethod
    def _voxel_coords(pts_xyz, voxel_step):
        return torch.round(pts_xyz / voxel_step.clamp_min(1e-9)).to(torch.long)

    def _sparsepcgc_quantized_coords(self, pts_xyz, coord_scale, fallback_voxel_step=None, global_offset=None):
        compress_key = (
            str(getattr(self.args, "compress", ""))
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )

        if compress_key != "sparsepcgc":
            step = fallback_voxel_step if fallback_voxel_step is not None else self._voxel_step(pts_xyz, coord_scale)
            if global_offset is None:
                return self._voxel_coords(pts_xyz, step)
            return self._voxel_coords(pts_xyz - global_offset, step)

        return canonical_sparsepcgc_voxel_coords(
            pts_xyz,
            args=self.args,
            coord_scale=coord_scale,
            global_offset=global_offset,
        ).to(torch.long)
    @classmethod
    def _coords_membership_mask(cls, query_coords, reference_coords):
        B, _, N = query_coords.shape
        out = torch.zeros((B, N), device=query_coords.device, dtype=torch.bool)
        for b in range(B):
            query = query_coords[b].transpose(0, 1).contiguous()
            reference = reference_coords[b].transpose(0, 1).contiguous()
            out[b] = cls._coords_membership(query, reference)
        return out

    def _first_unique_selected_mask(self, target_coords, selected_mask):
        if selected_mask.ndim == 3:
            selected_mask = selected_mask.squeeze(1)
        selected_mask = selected_mask.to(device=target_coords.device, dtype=torch.bool)
        keep = torch.zeros_like(selected_mask, dtype=torch.bool)
        for b in range(target_coords.shape[0]):
            selected_idx = selected_mask[b].nonzero(as_tuple=False).flatten()
            if selected_idx.numel() == 0:
                continue
            coords_b = target_coords[b : b + 1].index_select(2, selected_idx)
            keep_b = self._first_unique_coord_mask(coords_b).squeeze(0)
            keep[b, selected_idx[keep_b]] = True
        return keep

    @staticmethod
    def _coord_keys(coords, mins, spans):
        shifted = coords - mins.view(1, 3)
        return (
            shifted[:, 0] * spans[1].clamp_min(1) * spans[2].clamp_min(1)
            + shifted[:, 1] * spans[2].clamp_min(1)
            + shifted[:, 2]
        )

    @classmethod
    def _coords_membership(cls, query_coords, reference_coords):
        if query_coords.numel() == 0:
            return torch.zeros((query_coords.shape[0],), device=query_coords.device, dtype=torch.bool)
        if reference_coords.numel() == 0:
            return torch.zeros((query_coords.shape[0],), device=query_coords.device, dtype=torch.bool)
        combined = torch.cat([query_coords, reference_coords], dim=0)
        mins = combined.amin(dim=0)
        spans = (combined.amax(dim=0) - mins + 1).to(torch.long)
        query_keys = cls._coord_keys(query_coords.to(torch.long), mins, spans)
        reference_keys = torch.unique(cls._coord_keys(reference_coords.to(torch.long), mins, spans), sorted=True)
        pos = torch.searchsorted(reference_keys, query_keys)
        in_bounds = pos < reference_keys.numel()
        safe_pos = pos.clamp(max=max(int(reference_keys.numel()) - 1, 0))
        return in_bounds & (reference_keys[safe_pos] == query_keys)

    def _build_voxel_cache(self, voxel_coords):
        cache = []
        B, _, _ = voxel_coords.shape
        for b in range(B):
            coords = voxel_coords[b].transpose(0, 1).contiguous()
            if coords.numel() == 0:
                empty = coords.new_empty((0,), dtype=torch.long)
                cache.append(
                    {
                        "coords": coords,
                        "unique_coords": coords.new_empty((0, 3), dtype=torch.long),
                        "inverse": empty,
                        "counts": empty.to(dtype=torch.float32),
                        "occupied_keys": empty,
                        "key_mins": coords.new_zeros((3,), dtype=torch.long),
                        "key_spans": coords.new_ones((3,), dtype=torch.long),
                        "voxel_count": 0,
                    }
                )
                continue
            unique_coords, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
            voxel_count = int(unique_coords.shape[0])
            counts = torch.bincount(inverse, minlength=voxel_count).to(
                device=voxel_coords.device,
                dtype=torch.float32,
            )
            key_min = unique_coords.amin(dim=0) - 1
            key_span = (unique_coords.amax(dim=0) - unique_coords.amin(dim=0) + 3).to(torch.long).clamp_min(1)
            occupied_keys = torch.sort(self._coord_keys(unique_coords, key_min, key_span)).values
            cache.append(
                {
                    "coords": coords,
                    "unique_coords": unique_coords,
                    "inverse": inverse,
                    "counts": counts,
                    "occupied_keys": occupied_keys,
                    "key_mins": key_min,
                    "key_spans": key_span,
                    "voxel_count": voxel_count,
                }
            )
        return cache

    @staticmethod
    def _child_slot_from_coords_lastdim(coords):
        coords = coords.to(dtype=torch.long)
        return (
            coords[..., 0].remainder(2)
            + 2 * coords[..., 1].remainder(2)
            + 4 * coords[..., 2].remainder(2)
        ).to(dtype=torch.long)

    def _point_parent_codes_and_child_slots(self, voxel_coords):
        B, _, N = voxel_coords.shape
        parent_codes = torch.zeros((B, N), device=voxel_coords.device, dtype=torch.long)
        child_slots = torch.zeros((B, N), device=voxel_coords.device, dtype=torch.long)
        for b in range(B):
            coords = voxel_coords[b].transpose(0, 1).contiguous().to(dtype=torch.long)
            if coords.numel() == 0:
                continue
            parents = torch.div(coords, 2, rounding_mode="floor")
            slots = self._child_slot_from_coords_lastdim(coords)
            child_slots[b] = slots
            unique_parents, inverse = torch.unique(parents, dim=0, sorted=True, return_inverse=True)
            codes = torch.zeros((unique_parents.shape[0],), device=voxel_coords.device, dtype=torch.long)
            for slot in range(8):
                mask = slots == slot
                if bool(mask.any().item()):
                    parent_ids = torch.unique(inverse[mask], sorted=False)
                    bit = codes.new_full((parent_ids.numel(),), 1 << slot)
                    codes[parent_ids] = torch.bitwise_or(codes[parent_ids], bit)
            parent_codes[b] = codes.index_select(0, inverse)
        return parent_codes, child_slots

    def _occupancy_code_popularity(self, octree_context, full_octree_context, like_tensor):
        parts = []

        def _add_codes(ctx, key, weight=1.0):
            if not isinstance(ctx, dict) or key not in ctx:
                return
            value = ctx.get(key, None)
            if value is None:
                return
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.to(device=like_tensor.device, dtype=torch.long).reshape(-1)
            value = value[(value >= 0) & (value < 256)]
            if value.numel() <= 0:
                return
            if float(weight) != 1.0:
                repeat = max(int(round(float(weight))), 1)
                value = value.repeat(repeat)
            parts.append(value)

        for ctx in (full_octree_context, octree_context):
            _add_codes(ctx, "occupancy_codes", weight=3.0)
            _add_codes(ctx, "ancestor_occupancy_codes", weight=2.0)
            _add_codes(ctx, "sibling_occupancy_codes", weight=2.0)
            _add_codes(ctx, "parent_occupancy_code", weight=2.0)

        counts = like_tensor.new_ones((256,), dtype=torch.float32) * float(
            getattr(self.args, "repair_pattern_prior_smoothing", 1.0)
        )
        if parts:
            codes = torch.cat(parts, dim=0)
            counts = counts + torch.bincount(codes, minlength=256).to(device=like_tensor.device, dtype=counts.dtype)
        score = torch.log1p(counts)
        score = score / score.max().clamp_min(1e-6)
        return score.to(dtype=like_tensor.dtype)

    @classmethod
    def _coords_membership_cached(cls, query_coords, reference_keys, key_mins, key_spans):
        if query_coords.numel() == 0 or reference_keys.numel() == 0:
            return torch.zeros((query_coords.shape[0],), device=query_coords.device, dtype=torch.bool)
        query_keys = cls._coord_keys(query_coords.to(torch.long), key_mins, key_spans)
        pos = torch.searchsorted(reference_keys, query_keys)
        in_bounds = pos < reference_keys.numel()
        safe_pos = pos.clamp(max=max(int(reference_keys.numel()) - 1, 0))
        return in_bounds & (reference_keys[safe_pos] == query_keys)

    @staticmethod
    def _isin_voxel_ids(inverse, selected_voxel_idx):
        if selected_voxel_idx.numel() == 0:
            return torch.zeros_like(inverse, dtype=torch.bool)
        if selected_voxel_idx.numel() == 1:
            return inverse == selected_voxel_idx.reshape(()).to(device=inverse.device, dtype=inverse.dtype)
        return torch.isin(inverse, selected_voxel_idx.to(device=inverse.device, dtype=inverse.dtype))

    @staticmethod
    def _top_unique_voxels_from_point_scores(scores, inverse, count):
        if int(count) <= 0 or scores.numel() == 0:
            empty = inverse.new_empty((0,), dtype=torch.long)
            return empty, scores.new_empty((0,))

        order = torch.argsort(scores.detach(), descending=True)
        sorted_voxels = inverse.index_select(0, order).to(dtype=torch.long)
        sorted_scores = scores.index_select(0, order)
        positions = torch.arange(sorted_voxels.numel(), device=sorted_voxels.device, dtype=torch.long)

        # First occurrence in score-sorted order is the per-voxel max score.
        # This avoids the old Python loop fallback on PyTorch versions without scatter_reduce_.
        stride = int(sorted_voxels.numel()) + 1
        voxel_then_pos = sorted_voxels * stride + positions
        by_voxel = torch.argsort(voxel_then_pos)
        voxels_by_id = sorted_voxels.index_select(0, by_voxel)
        pos_by_id = positions.index_select(0, by_voxel)
        first = torch.ones_like(pos_by_id, dtype=torch.bool)
        if pos_by_id.numel() > 1:
            first[1:] = voxels_by_id[1:] != voxels_by_id[:-1]
        first_pos = pos_by_id[first]
        if first_pos.numel() == 0:
            empty = inverse.new_empty((0,), dtype=torch.long)
            return empty, scores.new_empty((0,))

        k = min(int(count), int(first_pos.numel()))
        selected_pos = torch.topk(first_pos, k=k, largest=False, sorted=False).values
        return sorted_voxels.index_select(0, selected_pos), sorted_scores.index_select(0, selected_pos)

    def _empty_neighbor_target_mask(self, voxel_coords, voxel_cache=None):
        B, _, N = voxel_coords.shape
        offsets = self.neighbor_offsets.to(device=voxel_coords.device, dtype=torch.long)
        voxel_cache = self._build_voxel_cache(voxel_coords) if voxel_cache is None else voxel_cache
        masks = []
        for b in range(B):
            item = voxel_cache[b]
            current = item["coords"]
            targets = current[:, None, :] + offsets.view(1, -1, 3)
            occupied = self._coords_membership_cached(
                targets.reshape(-1, 3),
                item["occupied_keys"],
                item["key_mins"],
                item["key_spans"],
            ).view(N, -1)
            masks.append(~occupied)
        return torch.stack(masks, dim=0)

    @staticmethod
    def _voxel_point_counts(voxel_coords, voxel_cache=None):
        B, _, N = voxel_coords.shape
        counts = torch.zeros((B, 1, N), device=voxel_coords.device, dtype=torch.float32)
        if voxel_cache is not None:
            for b, item in enumerate(voxel_cache):
                if item["voxel_count"] <= 0:
                    continue
                counts[b, 0] = item["counts"].index_select(0, item["inverse"])
            return counts
        for b in range(B):
            coords = voxel_coords[b].transpose(0, 1).contiguous()
            if coords.numel() == 0:
                continue
            _, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
            voxel_counts = torch.bincount(inverse, minlength=int(inverse.max().item()) + 1).to(
                device=voxel_coords.device,
                dtype=torch.float32,
            )
            counts[b, 0] = voxel_counts.index_select(0, inverse)
        return counts

    @classmethod
    def _unique_voxel_count(cls, voxel_coords, point_mask=None):
        B = voxel_coords.shape[0]
        total = 0
        if point_mask is not None:
            if point_mask.ndim == 3:
                point_mask = point_mask.squeeze(1)
            point_mask = point_mask.to(device=voxel_coords.device, dtype=torch.bool)
        for b in range(B):
            coords = voxel_coords[b].transpose(0, 1).contiguous()
            if point_mask is not None:
                coords = coords[point_mask[b]]
            if coords.numel() == 0:
                continue
            total += int(torch.unique(coords, dim=0).shape[0])
        return total

    @staticmethod
    def _unique_voxel_count_from_cache(voxel_cache, point_mask=None):
        total = 0
        if point_mask is not None:
            if point_mask.ndim == 3:
                point_mask = point_mask.squeeze(1)
            point_mask = point_mask.to(dtype=torch.bool)
        for b, item in enumerate(voxel_cache):
            voxel_count = int(item["voxel_count"])
            if voxel_count <= 0:
                continue
            if point_mask is None:
                total += voxel_count
                continue
            mask_b = point_mask[b].to(device=item["inverse"].device, dtype=torch.bool)
            if not bool(mask_b.any().item()):
                continue
            selected_inverse = item["inverse"][mask_b]
            total += int(torch.unique(selected_inverse, sorted=False).numel())
        return total

    def _append_unique_voxel_coords(self, base_coords, add_coords):
        """
        base occupied coordsにadd coordsを追加し、重複voxelを1つにまとめる。
        入出力は[N, 3]のlong tensorである。
        """
        if base_coords is None:
            base_coords = add_coords.new_empty((0, 3), dtype=torch.long)
        base_coords = base_coords.to(dtype=torch.long).reshape(-1, 3)

        if add_coords is None:
            return torch.unique(base_coords, dim=0, sorted=True) if base_coords.numel() > 0 else base_coords

        add_coords = add_coords.to(device=base_coords.device, dtype=torch.long).reshape(-1, 3)
        if add_coords.numel() == 0:
            return torch.unique(base_coords, dim=0, sorted=True) if base_coords.numel() > 0 else base_coords
        if base_coords.numel() == 0:
            return torch.unique(add_coords, dim=0, sorted=True)

        return torch.unique(torch.cat([base_coords, add_coords], dim=0), dim=0, sorted=True)

    def _remove_voxel_coords(self, base_coords, remove_coords):
        """
        base occupied coordsからremove coordsを削除する。
        入出力は[N, 3]のlong tensorである。
        """
        base_coords = base_coords.to(dtype=torch.long).reshape(-1, 3)
        if base_coords.numel() == 0:
            return base_coords
        if remove_coords is None:
            return base_coords

        remove_coords = remove_coords.to(device=base_coords.device, dtype=torch.long).reshape(-1, 3)
        if remove_coords.numel() == 0:
            return base_coords

        remove_unique = torch.unique(remove_coords, dim=0, sorted=True)
        remove_mask = self._coords_membership(base_coords, remove_unique)
        return base_coords[~remove_mask].contiguous()

    def _first_unique_rows_mask(self, coords_n3):
        """
        [N, 3]座標について、同一座標の最初の出現だけTrueにする。
        duplicate targetを1つだけ残すために使う。
        """
        coords_n3 = coords_n3.to(dtype=torch.long).reshape(-1, 3)
        if coords_n3.numel() == 0:
            return torch.zeros((0,), device=coords_n3.device, dtype=torch.bool)

        _, inverse = torch.unique(coords_n3, dim=0, sorted=True, return_inverse=True)
        idx = torch.arange(inverse.numel(), device=inverse.device, dtype=inverse.dtype)
        sort_key = inverse * inverse.numel() + idx
        order = torch.argsort(sort_key)
        sorted_inverse = inverse.index_select(0, order)

        first_sorted = torch.ones_like(sorted_inverse, dtype=torch.bool)
        if sorted_inverse.numel() > 1:
            first_sorted[1:] = sorted_inverse[1:] != sorted_inverse[:-1]

        mask = torch.zeros_like(inverse, dtype=torch.bool)
        mask[order[first_sorted]] = True
        return mask

    def _build_voxel_edit_state_single(
        self,
        voxel_coords_b,
        hard_drop_mask_b,
        add_target_coords_b,
        add_target_mask_b,
        move_source_mask_b,
        move_target_coords_b,
        move_valid_mask_b,
    ):
        """
        1batch分のPrune/Add/Moveをoccupied voxel集合上に反映する。
        Moveは source occupied voxel削除 + target empty voxel追加として扱う。
        """
        coords_n3 = voxel_coords_b.transpose(0, 1).contiguous().to(dtype=torch.long)
        initial_unique = torch.unique(coords_n3, dim=0, sorted=True)

        current = initial_unique
        debug = {
            "initial_count": int(initial_unique.shape[0]),
            "drop_count": 0,
            "add_count": 0,
            "move_count": 0,
            "same_voxel_move_rejected": 0,
            "existing_target_rejected": 0,
            "duplicate_target_rejected": 0,
            "final_count": int(initial_unique.shape[0]),
        }

        # ------------------------------------------------------------
        # Prune：点ではなく、選ばれた点が属するoccupied voxelを削除する。
        # ------------------------------------------------------------
        if hard_drop_mask_b is not None:
            drop_mask = hard_drop_mask_b.to(device=coords_n3.device, dtype=torch.bool).reshape(-1)
            if drop_mask.numel() == coords_n3.shape[0] and bool(drop_mask.any().item()):
                drop_coords = torch.unique(coords_n3[drop_mask], dim=0, sorted=True)
                current = self._remove_voxel_coords(current, drop_coords)
                debug["drop_count"] = int(drop_coords.shape[0])

        # ------------------------------------------------------------
        # Add：empty child-slot / empty neighbor voxelをoccupiedに追加する。
        # ------------------------------------------------------------
        if add_target_coords_b is not None and add_target_mask_b is not None:
            add_coords_n3 = add_target_coords_b.transpose(0, 1).contiguous().to(device=coords_n3.device, dtype=torch.long)
            add_mask = add_target_mask_b.to(device=coords_n3.device, dtype=torch.bool).reshape(-1)
            if add_mask.numel() == add_coords_n3.shape[0] and bool(add_mask.any().item()):
                selected_add = add_coords_n3[add_mask]
                unique_add_mask = self._first_unique_rows_mask(selected_add)
                debug["duplicate_target_rejected"] += int((~unique_add_mask).sum().item())
                selected_add = selected_add[unique_add_mask]

                # 念のため、既存occupiedへAddする候補は除外する。
                already_occupied = self._coords_membership(selected_add, current)
                debug["existing_target_rejected"] += int(already_occupied.sum().item())
                selected_add = selected_add[~already_occupied]

                if selected_add.numel() > 0:
                    current = self._append_unique_voxel_coords(current, selected_add)
                    debug["add_count"] = int(selected_add.shape[0])

        # ------------------------------------------------------------
        # Move：source voxelを削除し、target voxelを追加する。
        # ------------------------------------------------------------
        if move_source_mask_b is not None and move_target_coords_b is not None:
            move_mask = move_source_mask_b.to(device=coords_n3.device, dtype=torch.bool).reshape(-1)
            if move_valid_mask_b is not None:
                move_mask = move_mask & move_valid_mask_b.to(device=coords_n3.device, dtype=torch.bool).reshape(-1)

            move_targets_n3 = move_target_coords_b.transpose(0, 1).contiguous().to(device=coords_n3.device, dtype=torch.long)

            if move_mask.numel() == coords_n3.shape[0] and move_targets_n3.shape[0] == coords_n3.shape[0] and bool(move_mask.any().item()):
                source_coords = coords_n3[move_mask]
                target_coords = move_targets_n3[move_mask]

                same_voxel = (source_coords == target_coords).all(dim=1)
                debug["same_voxel_move_rejected"] = int(same_voxel.sum().item())
                source_coords = source_coords[~same_voxel]
                target_coords = target_coords[~same_voxel]

                if target_coords.numel() > 0:
                    # Move targetは初期occupiedおよびPrune/Add反映後のcurrentに存在しないものだけ許可する。
                    target_existing = self._coords_membership(target_coords, current)
                    debug["existing_target_rejected"] += int(target_existing.sum().item())
                    source_coords = source_coords[~target_existing]
                    target_coords = target_coords[~target_existing]

                if target_coords.numel() > 0:
                    unique_target_mask = self._first_unique_rows_mask(target_coords)
                    debug["duplicate_target_rejected"] += int((~unique_target_mask).sum().item())
                    source_coords = source_coords[unique_target_mask]
                    target_coords = target_coords[unique_target_mask]

                if target_coords.numel() > 0:
                    source_unique = torch.unique(source_coords, dim=0, sorted=True)
                    target_unique = torch.unique(target_coords, dim=0, sorted=True)
                    current = self._remove_voxel_coords(current, source_unique)
                    current = self._append_unique_voxel_coords(current, target_unique)
                    debug["move_count"] = int(target_unique.shape[0])

        current = torch.unique(current, dim=0, sorted=True) if current.numel() > 0 else current
        debug["final_count"] = int(current.shape[0])
        weights = torch.ones((1, int(current.shape[0])), device=current.device, dtype=torch.float32)
        return current.transpose(0, 1).contiguous(), weights, debug

    def _pad_voxel_edit_state(self, coords_list, weights_list, device, dtype):
        """
        batchごとに長さが違うfinal voxel coordsを[B, 3, M]へpaddingする。
        final_voxel_valid_maskも同時に返す。
        """
        batch_size = len(coords_list)
        max_count = max([int(coords.shape[-1]) for coords in coords_list] or [0])

        if max_count <= 0:
            coords = torch.empty((batch_size, 3, 0), device=device, dtype=torch.long)
            weights = torch.empty((batch_size, 1, 0), device=device, dtype=dtype)
            valid_mask = torch.empty((batch_size, 0), device=device, dtype=torch.bool)
            return coords, weights, valid_mask

        padded_coords = torch.zeros((batch_size, 3, max_count), device=device, dtype=torch.long)
        padded_weights = torch.zeros((batch_size, 1, max_count), device=device, dtype=dtype)
        valid_mask = torch.zeros((batch_size, max_count), device=device, dtype=torch.bool)

        for b, coords_b in enumerate(coords_list):
            count = int(coords_b.shape[-1])
            if count <= 0:
                continue
            padded_coords[b, :, :count] = coords_b.to(device=device, dtype=torch.long)
            padded_weights[b, :, :count] = weights_list[b].to(device=device, dtype=dtype)
            valid_mask[b, :count] = True

        return padded_coords, padded_weights, valid_mask

    @classmethod
    def _selected_voxels_absent_count(cls, before_coords, selected_mask, after_coords, after_mask):
        if selected_mask.ndim == 3:
            selected_mask = selected_mask.squeeze(1)
        if after_mask.ndim == 3:
            after_mask = after_mask.squeeze(1)
        selected_mask = selected_mask.to(device=before_coords.device, dtype=torch.bool)
        after_mask = after_mask.to(device=after_coords.device, dtype=torch.bool)
        total = 0
        for b in range(before_coords.shape[0]):
            selected_coords = before_coords[b].transpose(0, 1).contiguous()[selected_mask[b]]
            if selected_coords.numel() == 0:
                continue
            selected_coords = torch.unique(selected_coords, dim=0)
            kept_after = after_coords[b].transpose(0, 1).contiguous()[after_mask[b]]
            present = cls._coords_membership(selected_coords, kept_after)
            total += int((~present).sum().item())
        return total

    def _neighbor_target_membership_mask(self, voxel_coords, reference_mask, voxel_cache=None):
        B, _, N = voxel_coords.shape
        offsets = self.neighbor_offsets.to(device=voxel_coords.device, dtype=torch.long)
        voxel_cache = self._build_voxel_cache(voxel_coords) if voxel_cache is None else voxel_cache
        if reference_mask.ndim == 3:
            reference_mask = reference_mask.squeeze(1)
        reference_mask = reference_mask.to(device=voxel_coords.device, dtype=torch.bool)
        masks = []
        for b in range(B):
            item = voxel_cache[b]
            current = item["coords"]
            reference = current[reference_mask[b]]
            targets = current[:, None, :] + offsets.view(1, -1, 3)
            if reference.numel() == 0:
                masks.append(torch.zeros((N, offsets.shape[0]), device=voxel_coords.device, dtype=torch.bool))
                continue
            reference_keys = torch.sort(self._coord_keys(reference, item["key_mins"], item["key_spans"])).values
            masks.append(
                self._coords_membership_cached(
                    targets.reshape(-1, 3),
                    reference_keys,
                    item["key_mins"],
                    item["key_spans"],
                ).view(N, -1)
            )
        return torch.stack(masks, dim=0)

    @staticmethod
    def _voxel_mean_logits(logits, voxel_coords, voxel_cache=None):
        B, K, N = logits.shape
        out = torch.empty_like(logits)
        if voxel_cache is not None:
            for b, item in enumerate(voxel_cache):
                voxel_count = int(item["voxel_count"])
                if voxel_count <= 0:
                    out[b] = logits[b]
                    continue
                inverse = item["inverse"]
                index = inverse.view(1, N).expand(K, N)
                sums = logits.new_zeros((K, voxel_count))
                sums.scatter_add_(1, index, logits[b])
                counts = item["counts"].to(device=logits.device, dtype=logits.dtype).clamp_min(1.0)
                means = sums / counts.view(1, voxel_count)
                out[b] = means.index_select(1, inverse)
            return out
        for b in range(B):
            coords = voxel_coords[b].transpose(0, 1).contiguous()
            if coords.numel() == 0:
                out[b] = logits[b]
                continue
            _, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
            voxel_count = int(inverse.max().item()) + 1
            index = inverse.view(1, N).expand(K, N)
            sums = logits.new_zeros((K, voxel_count))
            sums.scatter_add_(1, index, logits[b])
            counts = torch.bincount(inverse, minlength=voxel_count).to(
                device=logits.device,
                dtype=logits.dtype,
            ).clamp_min(1.0)
            means = sums / counts.view(1, voxel_count)
            out[b] = means.index_select(1, inverse)
        return out

    @staticmethod
    def _first_unique_coord_mask(voxel_coords):
        B, _, N = voxel_coords.shape
        unique_mask = torch.zeros((B, N), device=voxel_coords.device, dtype=torch.bool)
        for b in range(B):
            coords = voxel_coords[b].transpose(0, 1).contiguous()
            if coords.numel() == 0:
                continue

            _, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)

            idx = torch.arange(inverse.numel(), device=inverse.device, dtype=inverse.dtype)
            sort_key = inverse * inverse.numel() + idx
            order = torch.argsort(sort_key)

            sorted_inverse = inverse.index_select(0, order)
            first = torch.ones_like(sorted_inverse, dtype=torch.bool)
            if sorted_inverse.numel() > 1:
                first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
            unique_mask[b, order[first]] = True

        return unique_mask

    def _hard_voxel_drop_mask(
        self,
        voxel_coords,
        drop_scores,
        target_drop_ratio,
        max_drop_ratio,
        selection_mask,
        hard_threshold=0.0,
        voxel_cache=None,
        force_min_count=False,
        max_hard_count=0,
        allow_single_candidate=False,
    ):
        B, _, N = drop_scores.shape
        hard_drop = torch.zeros_like(drop_scores, dtype=torch.bool)
        threshold_cap_mode = self._threshold_cap_mode()
        if N <= 0 or float(max_drop_ratio) <= 0.0:
            return hard_drop
        if not threshold_cap_mode and float(target_drop_ratio) <= 0.0:
            return hard_drop
        if selection_mask is None:
            valid_all = torch.ones((B, N), device=drop_scores.device, dtype=torch.bool)
        else:
            valid_all = selection_mask.squeeze(1) if selection_mask.ndim == 3 else selection_mask
            valid_all = valid_all.to(device=drop_scores.device, dtype=torch.bool)
        voxel_cache = self._build_voxel_cache(voxel_coords) if voxel_cache is None else voxel_cache
        for b in range(B):
            valid = valid_all[b]
            if not bool(valid.any().item()):
                continue
            item = voxel_cache[b]
            inverse_all = item["inverse"]
            voxel_count_all = int(item["voxel_count"])
            if voxel_count_all <= 1:
                continue
            score_values = drop_scores[b, 0].detach()
            finite_valid = valid & torch.isfinite(score_values)
            if not bool(finite_valid.any().item()):
                continue
            score_floor = torch.finfo(score_values.dtype).min
            invalid_threshold = score_floor * 0.5
            voxel_scores = score_values.new_full((voxel_count_all,), score_floor)
            scatter_reduce = getattr(voxel_scores, "scatter_reduce_", None)
            masked_scores = score_values.masked_fill(~finite_valid, score_floor)
            if callable(scatter_reduce):
                voxel_scores.scatter_reduce_(0, inverse_all, masked_scores, reduce="amax", include_self=True)
                valid_voxels = voxel_scores > invalid_threshold
                voxel_count = int(valid_voxels.sum().item())
            else:
                valid_inverse = inverse_all[finite_valid]
                valid_scores = score_values[finite_valid]
                voxel_count = int(torch.unique(valid_inverse, sorted=False).numel())
            if voxel_count <= 1 and not bool(allow_single_candidate):
                continue
            if threshold_cap_mode:
                cap_count = int(math.ceil(float(max_drop_ratio) * float(voxel_count)))
                reserve = 0 if bool(allow_single_candidate) else 1
                drop_count = min(max(cap_count, 0), voxel_count - reserve)
            else:
                cap_count = int(round(float(max_drop_ratio) * float(voxel_count)))
                target_count = int(round(float(target_drop_ratio) * float(voxel_count)))
                if (force_min_count or bool(allow_single_candidate)) and target_drop_ratio > 0.0:
                    target_count = max(target_count, 1)
                if (force_min_count or bool(allow_single_candidate)) and max_drop_ratio > 0.0:
                    cap_count = max(cap_count, 1)
                reserve = 0 if bool(allow_single_candidate) else 1
                drop_count = min(target_count, cap_count, voxel_count - reserve)
            if int(max_hard_count) > 0:
                drop_count = min(drop_count, int(max_hard_count))
            if drop_count <= 0:
                continue
            if callable(scatter_reduce):
                candidate_scores = voxel_scores.masked_fill(~valid_voxels, score_floor)
                selected_voxel_idx = torch.topk(
                    candidate_scores,
                    k=min(int(drop_count), int(voxel_count)),
                    largest=True,
                    sorted=False,
                ).indices
                if threshold_cap_mode:
                    selected_scores = voxel_scores.index_select(0, selected_voxel_idx)
                    selected_voxel_idx = selected_voxel_idx[selected_scores >= float(hard_threshold)]
                    if selected_voxel_idx.numel() <= 0:
                        continue
            else:
                selected_voxel_idx, selected_scores = self._top_unique_voxels_from_point_scores(
                    valid_scores,
                    valid_inverse,
                    drop_count,
                )
                if threshold_cap_mode:
                    selected_voxel_idx = selected_voxel_idx[selected_scores >= float(hard_threshold)]
                    if selected_voxel_idx.numel() <= 0:
                        continue
            selected_points = self._isin_voxel_ids(inverse_all, selected_voxel_idx)
            hard_drop[b, 0] = selected_points
        return hard_drop

    @staticmethod
    def _priority_topk_gate(priority, target_ratio, tau):
        B, _, N = priority.shape
        if N <= 0:
            return priority
        keep = max(1, min(int(round(float(target_ratio) * float(N))), N))
        flat = priority.detach().reshape(B, N)
        threshold = torch.topk(flat, k=keep, dim=1, largest=True).values[:, -1].view(B, 1, 1)
        return torch.sigmoid((priority - threshold) / max(float(tau), 1e-6))

    @staticmethod
    def _masked_mean(values, point_mask):
        if point_mask is None:
            return values.mean()
        if point_mask.ndim == 2:
            point_mask = point_mask.unsqueeze(1)
        if point_mask.ndim != 3 or point_mask.shape[0] != values.shape[0] or point_mask.shape[2] != values.shape[2]:
            raise ValueError("point_mask must broadcast to [B, 1, N].")
        mask = point_mask.to(device=values.device, dtype=values.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    def _numeric_floor(self, values, arg_name="repair_soft_normalizer_floor", default=1e-4):
        # AMP fp16では1e-12が0へ丸まり、0 * inf のNaN勾配を作りやすい。
        floor = max(float(getattr(self.args, arg_name, default)), 0.0)
        if torch.is_tensor(values) and values.is_floating_point():
            if values.dtype in (torch.float16, torch.bfloat16):
                floor = max(floor, 1e-4)
            else:
                floor = max(floor, float(torch.finfo(values.dtype).tiny))
        return floor

    def _safe_budget_scale(self, raw_sum_detached, budget, arg_name="repair_soft_normalizer_floor", default=1e-4):
        # raw_sumが0/極小のときに budget / raw_sum がinfにならないよう、scale自体を0へ落とす。
        # 操作量headへの勾配は直接量損失で残すため、ここでは非有限勾配を作らないことを優先する。
        floor = self._numeric_floor(raw_sum_detached, arg_name=arg_name, default=default)
        raw_sum = torch.nan_to_num(
            raw_sum_detached.detach(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        clean_budget = torch.nan_to_num(
            budget,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        safe_sum = raw_sum.clamp_min(floor)
        scale = torch.where(raw_sum > floor, clean_budget / safe_sum, torch.zeros_like(clean_budget))
        return torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    def _exploration_phase(self):
        if not self.training:
            return 1.0
        fraction = min(max(float(getattr(self.args, "repair_exploration_fraction", 0.0)), 0.0), 1.0)
        if fraction <= 0.0:
            return 1.0
        total_steps = max(int(getattr(self.args, "_total_train_steps_estimate", 0)), 1)
        step = min(max(int(getattr(self.args, "_global_train_step", 0)), 0), total_steps)
        progress = min(float(step) / float(max(total_steps, 1)), 1.0)
        return min(progress / fraction, 1.0)

    def _annealed_value(self, start_name, end_name, default_start=0.0, default_end=0.0):
        start = float(getattr(self.args, start_name, default_start))
        end = float(getattr(self.args, end_name, default_end))
        phase = self._exploration_phase()
        return start + (end - start) * phase

    @staticmethod
    def _safe_logit(prob):
        # 0/1付近の確率を安全にlogitへ戻し、比率biasや探索ノイズをlogit空間で足す。
        prob = prob.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(prob / (1.0 - prob))

    def _learned_operation_ratio(self, actuator_features, head, max_ratio, random_mix_start, random_mix_end):
        # 全点特徴を集約して、このStepでAdd/Adjust/Pruneする割合を作る。
        # repair_learn_operation_amounts=False の場合でも、
        # amount_head への勾配確認用に gradient-only 経路を残す。
        if max_ratio <= 0.0:
            return actuator_features.new_zeros((actuator_features.shape[0], 1, 1))

        # ============================================================
        # Amount head 用の集約特徴
        # ============================================================
        # 既存実装では mean だけを使っていたため、
        # 点群内の局所的な強い異常やばらつきがAmount headへ入りにくかった。
        #
        # headの入力チャネル数は変えず、mean + std + max を同じC次元に合成する。
        # これにより既存の drop/add/move_amount_head の構造を壊さず、
        # Stepごとの操作割合が入力状態に応じて変化しやすくなる。
        # ============================================================
        mean_feat = actuator_features.mean(dim=2, keepdim=True)
        max_feat = actuator_features.amax(dim=2, keepdim=True)

        if actuator_features.shape[2] > 1:
            std_feat = actuator_features.std(dim=2, keepdim=True, unbiased=False)
        else:
            std_feat = torch.zeros_like(mean_feat)

        pooled = (
            mean_feat
            + float(getattr(self.args, "repair_amount_pool_std_weight", 0.50)) * std_feat
            + float(getattr(self.args, "repair_amount_pool_max_weight", 0.25)) * max_feat
        )

        # まず必ず amount head を通す。
        # raw logit を軽く正規化し、sigmoid 飽和による amount_head 勾配消失を抑える。
        raw_logit = torch.nan_to_num(
            head(pooled),
            nan=0.0,
            posinf=float(getattr(self.args, "repair_operation_amount_logit_scale", 6.0)),
            neginf=-float(getattr(self.args, "repair_operation_amount_logit_scale", 6.0)),
        )
        ratio_logit_scale = max(float(getattr(self.args, "repair_operation_amount_logit_scale", 6.0)), 1e-6)
        bounded_logit = ratio_logit_scale * torch.tanh(raw_logit / ratio_logit_scale)
        learned_ratio = torch.sigmoid(bounded_logit) * float(max_ratio)

        if bool(getattr(self.args, "repair_learn_operation_amounts", True)):
            ratio = learned_ratio
        else:
            # forward値は固定比率に近いままにし、backwardだけ learned_ratio へ流す。
            # これにより、固定操作量モードでも amount_head の勾配が完全には切れない。
            fixed_ratio = pooled.new_full((pooled.shape[0], 1, 1), float(max_ratio))
            eps = float(getattr(self.args, "repair_amount_fixed_mode_grad_eps", 1e-3))
            ratio = fixed_ratio + eps * (learned_ratio - learned_ratio.detach())

        # 学習初期だけランダム比率を混ぜ、Add/Adjust量の探索範囲を広げる。
        random_mix = min(max(self._annealed_value(random_mix_start, random_mix_end), 0.0), 1.0)
        if self.training and random_mix > 0.0:
            random_ratio = torch.rand_like(ratio) * float(max_ratio)
            ratio = (1.0 - random_mix) * ratio + random_mix * random_ratio

        return ratio.clamp(0.0, float(max_ratio))

    def _learned_operation_gates(self, actuator_features, prune_enabled, add_enabled, move_enabled):
        B = actuator_features.shape[0]
        enabled_mask = actuator_features.new_tensor(
            [
                1.0 if prune_enabled else 0.0,
                1.0 if add_enabled else 0.0,
                1.0 if move_enabled else 0.0,
            ]
        ).view(1, 3, 1)
        if not bool(getattr(self.args, "repair_operation_gate_enabled", True)):
            gate_prob = enabled_mask.expand(B, -1, -1).contiguous()
            gate_hard = gate_prob
            gate = gate_prob
            gate_logit = torch.where(
                gate_prob > 0.0,
                gate_prob.new_full(gate_prob.shape, 8.0),
                gate_prob.new_full(gate_prob.shape, -8.0),
            )
            return gate[:, 0:1], gate[:, 1:2], gate[:, 2:3], gate_prob, gate_hard, gate_logit

        mean_feat = actuator_features.mean(dim=2, keepdim=True)
        max_feat = actuator_features.amax(dim=2, keepdim=True)
        if actuator_features.shape[2] > 1:
            std_feat = actuator_features.std(dim=2, keepdim=True, unbiased=False)
        else:
            std_feat = torch.zeros_like(mean_feat)
        pooled = (
            mean_feat
            + float(getattr(self.args, "repair_operation_gate_pool_std_weight", 0.50)) * std_feat
            + float(getattr(self.args, "repair_operation_gate_pool_max_weight", 0.25)) * max_feat
        )
        logit_scale = max(float(getattr(self.args, "repair_operation_gate_logit_scale", 6.0)), 1e-6)
        raw_logit = torch.nan_to_num(
            self.operation_gate_head(pooled),
            nan=0.0,
            posinf=logit_scale,
            neginf=-logit_scale,
        )
        gate_logit = logit_scale * torch.tanh(raw_logit / logit_scale)
        temperature = max(float(getattr(self.args, "repair_operation_gate_temperature", 1.0)), 1e-6)
        gate_prob = torch.sigmoid(gate_logit / temperature) * enabled_mask

        random_mix = min(
            max(
                self._annealed_value(
                    "repair_operation_gate_random_mix_start",
                    "repair_operation_gate_random_mix_end",
                ),
                0.0,
            ),
            1.0,
        )
        if self.training and random_mix > 0.0:
            random_gate = torch.rand_like(gate_prob) * enabled_mask
            gate_prob = (1.0 - random_mix) * gate_prob + random_mix * random_gate

        min_forward = min(
            max(float(getattr(self.args, "repair_operation_gate_min_forward", 0.0)), 0.0),
            0.49,
        )
        if self.training and min_forward > 0.0:
            gate_prob = gate_prob * (1.0 - min_forward) + min_forward * enabled_mask

        hard_threshold = min(max(float(getattr(self.args, "repair_operation_gate_hard_threshold", 0.5)), 0.0), 1.0)
        gate_hard = (gate_prob >= hard_threshold).to(dtype=gate_prob.dtype) * enabled_mask
        if bool(getattr(self.args, "repair_operation_gate_hard_forward", False)):
            gate = gate_hard.detach() + gate_prob - gate_prob.detach()
        else:
            gate = gate_prob
        return gate[:, 0:1], gate[:, 1:2], gate[:, 2:3], gate_prob, gate_hard, gate_logit
    
    def _scale_amount_downstream_grad(self, ratio, op_name=""):
        # Amount ratio のforward値は変えず、backwardだけ強める。
        # 操作ごとに圧縮損失への感度が違うため、Prune/Add/Moveで別々の倍率を使う。
        # ratio.detach() + scale * (ratio - ratio.detach()) は、
        # forwardではratioと同じ値になり、backwardではscale倍の勾配を返す。
        op_name = str(op_name).strip().lower()

        if op_name in {"drop", "prune", "delete"}:
            scale = float(
                getattr(
                    self.args,
                    "repair_drop_amount_downstream_grad_scale",
                    getattr(self.args, "repair_amount_downstream_grad_scale", 6.0),
                )
            )
        elif op_name in {"add", "insert"}:
            scale = float(
                getattr(
                    self.args,
                    "repair_add_amount_downstream_grad_scale",
                    getattr(self.args, "repair_amount_downstream_grad_scale", 6.0),
                )
            )
        elif op_name in {"move", "adjust", "disp"}:
            scale = float(
                getattr(
                    self.args,
                    "repair_move_amount_downstream_grad_scale",
                    getattr(self.args, "repair_amount_downstream_grad_scale", 6.0),
                )
            )
        else:
            scale = float(getattr(self.args, "repair_amount_downstream_grad_scale", 6.0))

        max_scale = max(float(getattr(self.args, "repair_amount_downstream_grad_max_scale", 8.0)), 1.0)
        scale = min(max(scale, 1.0), max_scale)
        return ratio.detach() + scale * (ratio - ratio.detach())

    def _scale_where_downstream_grad(self, value, op_name=""):
        # Where score / logit のforward値は変えず、backwardだけ強める。
        # これにより圧縮損失 L_com から drop_head / add_head / move_voxel_head へ戻る勾配を操作別に調整する。
        op_name = str(op_name).strip().lower()

        if op_name in {"drop", "prune", "delete"}:
            scale = float(
                getattr(
                    self.args,
                    "repair_drop_where_downstream_grad_scale",
                    getattr(self.args, "repair_where_downstream_grad_scale", 1.0),
                )
            )
        elif op_name in {"add", "insert"}:
            scale = float(
                getattr(
                    self.args,
                    "repair_add_where_downstream_grad_scale",
                    getattr(self.args, "repair_where_downstream_grad_scale", 1.0),
                )
            )
        elif op_name in {"move", "adjust", "disp"}:
            scale = float(
                getattr(
                    self.args,
                    "repair_move_where_downstream_grad_scale",
                    getattr(self.args, "repair_where_downstream_grad_scale", 1.0),
                )
            )
        else:
            scale = float(getattr(self.args, "repair_where_downstream_grad_scale", 1.0))

        # Where勾配倍率を安全範囲に制限する。
        # 既存実装では max(scale, 1.0) により1倍未満へ下げられなかった。
        # しかし Add Where だけが突出している場合は、forward値を変えずに
        # backwardだけ弱める必要がある。
        min_scale = max(
            float(getattr(self.args, "repair_where_downstream_grad_min_scale", 0.05)),
            1e-4,
        )
        max_scale = max(
            float(getattr(self.args, "repair_where_downstream_grad_max_scale", 8.0)),
            min_scale,
        )
        scale = min(max(scale, min_scale), max_scale)

        return value.detach() + scale * (value - value.detach())

    def _ratio_bias(self, ratio, max_ratio):
        # 学習した操作量を位置scoreへ戻し、何個選ぶかとどこを選ぶかの勾配をつなぐ。
        if max_ratio <= 0.0:
            return ratio.new_zeros(ratio.shape)
        normalized = (ratio / float(max_ratio)).clamp(1e-4, 1.0 - 1e-4)
        return self._safe_logit(normalized) * float(getattr(self.args, "repair_operation_amount_bias_scale", 2.0))

    def _max_add_ratio(self):
        # targetなしAmount学習では、target_add_ratioを上限決定に使わない。
        # max_add_ratioだけをAdd Amountの探索上限として使う。
        if self._sparsepcgc_add_experiment_active():
            max_ratio = float(getattr(self.args, "sparsepcgc_add_max_ratio", getattr(self.args, "max_add_ratio", 0.30)))
            max_ratio = max_ratio * self._sparsepcgc_add_warmup()
        else:
            max_ratio = float(getattr(self.args, "max_add_ratio", 0.30))
        return min(max(max_ratio, 0.0), 0.30)

    def _sparsepcgc_add_warmup(self):
        steps = max(int(getattr(self.args, "sparsepcgc_add_warmup_steps", 0)), 0)
        if steps <= 0:
            return 1.0
        step = int(getattr(self.args, "_global_train_step", 0)) + 1
        return min(1.0, max(0.0, float(step) / float(steps)))

    def _repair_move_warmup(self):
        # Move is much more destructive for SparsePCGC than add/drop, so let its cap ramp in slowly.
        if not self.training:
            return 1.0
        steps = max(int(getattr(self.args, "repair_move_warmup_steps", 0)), 0)
        if steps <= 0:
            return 1.0
        step = int(getattr(self.args, "_global_train_step", 0)) + 1
        return min(1.0, max(0.0, float(step) / float(steps)))

    def _sparsepcgc_add_experiment_active(self):
        compress_key = (
            str(getattr(self.args, "compress", ""))
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )
        if compress_key != "sparsepcgc":
            return False
        if not bool(getattr(self.args, "sparsepcgc_enable_add_experiment", False)):
            return False
        if bool(getattr(self.args, "sparsepcgc_add_only_when_compression_primary", True)):
            return str(getattr(self.args, "loss_mode", "legacy_total")).strip().lower() == "compression_primary"
        return True

    def _target_add_count(self, point_count, candidate_ratio_override=None, force_min_count=None):
        if point_count <= 0 or not self._add_enabled():
            return 0, 0.0
        max_ratio = self._max_add_ratio()
        if max_ratio <= 0.0:
            return 0, 0.0
        if candidate_ratio_override is None:
            start = float(getattr(self.args, "repair_add_candidate_ratio_start", 0.0)) or max_ratio
            end = float(getattr(self.args, "repair_add_candidate_ratio_end", 0.0)) or max_ratio
            phase = self._exploration_phase()
            candidate_ratio = start + (end - start) * phase
        else:
            candidate_ratio = float(candidate_ratio_override)
        candidate_ratio = min(max(candidate_ratio, 0.0), max_ratio)
        max_add_points = int(math.ceil(max_ratio * float(point_count))) if max_ratio > 0.0 else 0
        expected_points = candidate_ratio * float(point_count)
        if force_min_count is None:
            force_min_count = bool(getattr(self.args, "repair_force_min_add_voxels", False))
        min_expected = max(float(getattr(self.args, "repair_add_min_expected_voxels", 1.0)), 0.0)
        if candidate_ratio <= 0.0:
            add_points = 0
        elif force_min_count or expected_points >= min_expected:
            add_points = int(math.ceil(expected_points))
        else:
            add_points = 0
        return min(add_points, max_add_points, point_count), float(candidate_ratio)

    @staticmethod
    def _gumbel_like(values):
        eps = torch.finfo(values.dtype).eps
        uniform = torch.rand_like(values).clamp(eps, 1.0 - eps)
        return -torch.log(-torch.log(uniform))

    @staticmethod
    def _random_ratio_mask_like(values, max_ratio, point_mask=None):
        max_ratio = max(float(max_ratio), 0.0)
        if max_ratio <= 0.0:
            return torch.zeros_like(values)
        B = values.shape[0]
        ratios = torch.rand((B, 1, 1), device=values.device, dtype=values.dtype) * min(max_ratio, 1.0)
        mask = (torch.rand_like(values) < ratios).to(dtype=values.dtype)
        if point_mask is not None:
            mask = mask * point_mask.to(device=values.device, dtype=values.dtype)
        return mask

    def _operation_amount_logit(self, actuator_features, head):
        # amount head の raw logit を返す。
        # ratio化後の sigmoid が飽和しても、logit側には直接勾配を残す。
        mean_feat = actuator_features.mean(dim=2, keepdim=True)
        max_feat = actuator_features.amax(dim=2, keepdim=True)
        if actuator_features.shape[2] > 1:
            std_feat = actuator_features.std(dim=2, keepdim=True, unbiased=False)
        else:
            std_feat = torch.zeros_like(mean_feat)
        pooled = (
            mean_feat
            + float(getattr(self.args, "repair_amount_pool_std_weight", 0.50)) * std_feat
            + float(getattr(self.args, "repair_amount_pool_max_weight", 0.25)) * max_feat
        )
        scale = float(getattr(self.args, "repair_operation_amount_logit_scale", 6.0))
        raw_logit = torch.nan_to_num(head(pooled), nan=0.0, posinf=scale, neginf=-scale)
        bounded_logit = scale * torch.tanh(raw_logit / max(scale, 1e-6))
        return raw_logit + (bounded_logit - raw_logit).detach()

    def _target_ratio_logit(self, target_ratio, max_ratio, like_tensor):
        # target_ratio / max_ratio を logit 空間へ写像する。
        # ratio損失だけでは sigmoid 飽和時に勾配が消えるため、
        # amount head の raw logit を直接目標へ寄せる補助教師として使う。
        max_ratio = max(float(max_ratio), 1e-9)
        target_prob = target_ratio / max_ratio
        if not torch.is_tensor(target_prob):
            target_prob = like_tensor.new_tensor(float(target_prob))
        target_prob = target_prob.to(device=like_tensor.device, dtype=like_tensor.dtype)
        target_prob_max = min(
            max(float(getattr(self.args, "repair_operation_amount_target_prob_max", 0.98)), 0.50),
            1.0 - 1e-4,
        )
        target_prob = target_prob.clamp(1e-4, target_prob_max)
        return torch.log(target_prob / (1.0 - target_prob))

    def _actual_oracle_amount_bce_loss(self, learned_ratio, target_ratio, max_ratio):
        # actual oracleが採択した編集量をAmount headへ返すための損失。
        # 小さいratio同士のMSEは勾配が弱すぎるため、max_ratioで正規化したBCEを使う。
        if max_ratio <= 0.0:
            return learned_ratio.new_zeros(())
        pred_prob = (learned_ratio / float(max_ratio)).clamp(1e-4, 1.0 - 1e-4)
        if not torch.is_tensor(target_ratio):
            target_ratio = learned_ratio.new_tensor(float(target_ratio))
        target_prob = target_ratio.to(device=learned_ratio.device, dtype=learned_ratio.dtype) / float(max_ratio)
        target_prob = target_prob.clamp(0.0, 1.0 - 1e-4)
        target_prob = target_prob.expand_as(pred_prob)
        loss = -(
            target_prob.detach() * pred_prob.log()
            + (1.0 - target_prob.detach()) * torch.log1p(-pred_prob)
        )
        return torch.nan_to_num(loss.mean(), nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        pts_xyz,
        structure,
        cause_scores,
        policy_probs,
        actuator_features,
        repair_priority=None,
        coord_scale=None,
        selection_mask=None,
        octree_context=None,
        full_octree_context=None,
    ):
        timing_enabled = bool(getattr(self.args, "debug_timing", False))
        runtime_timing = {}
        if timing_enabled:
            if pts_xyz.is_cuda:
                torch.cuda.synchronize(pts_xyz.device)
            runtime_start = time.perf_counter()
            runtime_cursor = runtime_start

            def _mark_runtime(name):
                nonlocal runtime_cursor
                if pts_xyz.is_cuda:
                    torch.cuda.synchronize(pts_xyz.device)
                now = time.perf_counter()
                runtime_timing[name] = float(now - runtime_cursor)
                runtime_cursor = now
        else:
            runtime_start = None

        snap_strength = float(getattr(self.args, "repair_snap_strength", getattr(self.args, "disp_snap_strength", 0.35)))
        max_offset = self._max_offset(pts_xyz, coord_scale)
        stage_raw = str(getattr(self.args, "training_stage", "joint")).strip().lower()
        force_joint_actuator = (
            str(getattr(self.args, "loss_mode", "legacy_total")).strip().lower() == "compression_primary"
            and bool(getattr(self.args, "cp_force_joint_actuator", True))
        )
        # compression_primaryではloss構造を固定するため、actuator強度もjoint相当に固定する。
        # legacy_totalでは既存のdiagnosis/joint差をそのまま残す。
        stage = "joint" if force_joint_actuator else stage_raw
        if stage == "diagnosis":
            actuator_strength = float(getattr(self.args, "diagnosis_actuator_strength", 0.1))
        else:
            actuator_strength = float(getattr(self.args, "repair_actuator_strength", 1.0))
        compress_key = (
            str(getattr(self.args, "compress", ""))
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )
        sparsepcgc_context = compress_key == "sparsepcgc"
        add_enabled = self._add_enabled()
        prune_enabled = self._prune_enabled()
        disp_enabled = self._disp_enabled()
        # SparsePCGCはSparse Tensorのactive coordinate数がbit数に直結しやすい。
        # 新規empty voxelへのaddはactive coordinateを増やすため、既定ではSparsePCGC時だけ止める。
        sparsepcgc_add_experiment_active = self._sparsepcgc_add_experiment_active()
        if sparsepcgc_context and bool(getattr(self.args, "sparsepcgc_disable_add", True)) and not sparsepcgc_add_experiment_active:
            add_enabled = False
        operation_enabled = add_enabled or prune_enabled or disp_enabled
        threshold_cap_mode = self._threshold_cap_mode()

        preserve = policy_probs[:, 0:1, :]
        p_chain = policy_probs[:, 1:2, :]
        p_sibling = policy_probs[:, 2:3, :]
        p_parent = policy_probs[:, 3:4, :]
        p_context = policy_probs[:, 4:5, :]
        p_comp = policy_probs[:, 5:6, :]
        p_outlier = policy_probs[:, 6:7, :] if policy_probs.shape[1] > 6 else policy_probs.new_zeros(preserve.shape)
        base_repair_gate = (1.0 - preserve).clamp(0.0, 1.0)
        if selection_mask is not None:
            if selection_mask.ndim == 2:
                selection_mask = selection_mask.unsqueeze(1)
            if selection_mask.ndim != 3 or selection_mask.shape[0] != pts_xyz.shape[0] or selection_mask.shape[2] != pts_xyz.shape[2]:
                raise ValueError("selection_mask must broadcast to [B, 1, N].")
            selection_mask = selection_mask.to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            base_repair_gate = base_repair_gate * selection_mask
        target_ratio = float(getattr(self.args, "target_repair_ratio", getattr(self.args, "target_disp_ratio", 0.20)))
        if not operation_enabled:
            target_ratio = 0.0
        max_repair_ratio = max(float(getattr(self.args, "max_repair_ratio", target_ratio)), target_ratio)
        if not operation_enabled:
            max_repair_ratio = 0.0
        gate_cap_ratio = max_repair_ratio if bool(getattr(self.args, "repair_learn_operation_amounts", True)) else target_ratio
        if bool(getattr(self.args, "repair_priority_gate", True)) and repair_priority is not None:
            priority = repair_priority.to(device=pts_xyz.device, dtype=pts_xyz.dtype).clamp(0.0, 1.0)
            priority_gate = self._priority_topk_gate(
                priority,
                target_ratio=max(gate_cap_ratio, 1e-4),
                tau=float(getattr(self.args, "repair_priority_gate_tau", 0.08)),
            )
            repair_gate = base_repair_gate * priority_gate
        else:
            repair_gate = base_repair_gate
        if bool(getattr(self.args, "repair_gate_mean_cap", True)):
            gate_mean = self._masked_mean(repair_gate, selection_mask).detach().clamp_min(1e-6)
            # 操作量を学習する場合は固定targetではなく広めの候補上限でrepair候補を残す。
            gate_scale = (gate_cap_ratio / gate_mean).clamp_max(1.0)
            repair_gate = repair_gate * gate_scale

        node_score = cause_scores[:, 0:1, :]
        single_score = cause_scores[:, 1:2, :]
        lowprob_score = cause_scores[:, 2:3, :] if cause_scores.shape[1] > 2 else preserve.new_zeros(preserve.shape)
        if cause_scores.shape[1] >= 8:
            quant_score = cause_scores[:, 4:5, :]
            sparse_score = cause_scores[:, 5:6, :]
            local_outlier_score = cause_scores[:, 6:7, :]
        else:
            quant_score = preserve.new_zeros(preserve.shape)
            sparse_score = cause_scores[:, 4:5, :] if cause_scores.shape[1] > 4 else preserve.new_zeros(preserve.shape)
            local_outlier_score = cause_scores[:, 5:6, :] if cause_scores.shape[1] > 5 else preserve.new_zeros(preserve.shape)
        shape_score = cause_scores[:, -1:, :]
        leaf_actuator_prior = self._leaf_pattern_actuator_priors(
            structure,
            preserve,
        )
        leaf_drop_prior = leaf_actuator_prior["delete_prior"]
        leaf_add_prior = leaf_actuator_prior["add_prior"]
        leaf_move_prior = leaf_actuator_prior["move_prior"]
        full_context_available = isinstance(full_octree_context, dict) and bool(full_octree_context)

        actuator_parent_occupancy_code = 0
        actuator_sibling_count = 0
        actuator_ancestor_count = 0
        full_context_bonus = preserve.new_zeros(preserve.shape)

        if full_context_available:
            actuator_parent_occupancy_code = int(full_octree_context.get("parent_occupancy_code", 0) or 0)

            sibling_ids = full_octree_context.get("sibling_node_ids", None)
            if sibling_ids is None:
                sibling_ids = full_octree_context.get("sibling_paths", [])
            actuator_sibling_count = len(sibling_ids) if hasattr(sibling_ids, "__len__") else 0

            ancestor_ids = full_octree_context.get("ancestor_node_ids", [])
            actuator_ancestor_count = len(ancestor_ids) if hasattr(ancestor_ids, "__len__") else 0

            # 親・兄弟・祖先文脈が存在するSubtreeほど、構造操作候補として少しだけ強める。
            # まずは小さい補正に留め、既存policyを壊さない。
            context_strength = min(
                (float(actuator_sibling_count) + 0.5 * float(actuator_ancestor_count)) / 8.0,
                1.0,
            )
            full_context_bonus = preserve.new_full(preserve.shape, context_strength)

        # 初期Octree/Subtreeのglobal voxel座標系を優先して使う。
        voxel_step, voxel_offset, uses_context_voxel_frame = self._context_voxel_step_and_offset(
            pts_xyz,
            coord_scale,
            octree_context,
        )
        voxel_norm = (voxel_step * math.sqrt(3.0)).clamp_min(1e-9)

        context_voxel_coords = None
        if isinstance(octree_context, dict) and octree_context.get("global_voxel_coords", None) is not None:
            context_voxel_coords = octree_context["global_voxel_coords"]
        else:
            if bool(getattr(self.args, "warn_missing_input_voxel", True)):
                global_step = getattr(self.args, "_global_train_step", "NA")
                sample_name = getattr(self.args, "_current_sample_name", "NA")
                print(
                    f"[Warning] Voxel is NOT found! "
                    f"Network input prebuilt Octree/Voxel was not passed to Actuator. "
                    f"global_step={global_step}, sample={sample_name}, "
                    f"fallback=local_recomputed"
                )
        if context_voxel_coords is not None:
            if not torch.is_tensor(context_voxel_coords):
                context_voxel_coords = torch.as_tensor(context_voxel_coords)
            context_voxel_coords = context_voxel_coords.to(device=pts_xyz.device, dtype=torch.long)

        if context_voxel_coords is not None:
            if context_voxel_coords.ndim == 2 and context_voxel_coords.shape[-1] == 3:
                context_voxel_coords = context_voxel_coords.transpose(0, 1).unsqueeze(0)
            elif context_voxel_coords.ndim == 3 and context_voxel_coords.shape[-1] == 3:
                context_voxel_coords = context_voxel_coords.permute(0, 2, 1).contiguous()

            if context_voxel_coords.ndim != 3 or context_voxel_coords.shape[1] != 3:
                raise ValueError("octree_context['global_voxel_coords'] must have shape [N, 3], [B, N, 3], or [B, 3, N].")

            if context_voxel_coords.shape[0] == 1 and pts_xyz.shape[0] > 1:
                context_voxel_coords = context_voxel_coords.expand(pts_xyz.shape[0], -1, -1)

            if context_voxel_coords.shape[0] != pts_xyz.shape[0]:
                raise ValueError("octree_context global_voxel_coords batch size does not match pts_xyz.")

            if int(context_voxel_coords.shape[2]) != int(pts_xyz.shape[2]):
                raise ValueError(
                    "octree_context['global_voxel_coords'] point count must match pts_xyz. "
                    "This usually means subtree_tree was not sliced as a subset of full cloud global coords. "
                    "Build canonical_subtree_tree from full_octree_context using selected_subtree_keys before calling Actuator."
                )

        if context_voxel_coords is not None:
            # Network入力前に作った初期Octree/Voxel座標をそのまま使う。
            voxel_coords = context_voxel_coords.contiguous()
            actuator_voxel_mode = "prebuilt_global_voxel_coords"
            actuator_local_recomputed = False
        else:
            if bool(getattr(self.args, "forbid_local_voxel_recompute", False)):
                raise ValueError(
                    "StructureRepairActuator requires octree_context['global_voxel_coords'] when "
                    "forbid_local_voxel_recompute=True."
                )
            # prebuilt voxelがない場合だけfallbackとして再Voxel化する。
            # SparsePCGC時はcanonical共通関数を通し、round(xyz/voxel_size) → round(/posQuantscale) に統一する。
            voxel_coords = self._sparsepcgc_quantized_coords(
                pts_xyz,
                coord_scale,
                fallback_voxel_step=voxel_step,
                global_offset=voxel_offset,
            )
            actuator_voxel_mode = "local_recomputed"
            actuator_local_recomputed = True

        point_parent_node_ids = None
        point_child_slots = None
        point_valid_empty_child_mask = None

        if isinstance(octree_context, dict):
            point_parent_node_ids = octree_context.get("point_parent_node_ids", None)
            point_child_slots = octree_context.get("point_child_slots", None)
            point_valid_empty_child_mask = octree_context.get("point_valid_empty_child_mask", None)
        # ここから先はprebuilt/localどちらでも共通で必要である。
        voxel_cache = self._build_voxel_cache(voxel_coords)
        # Phase3: 点操作とは別に、occupied voxel集合としての編集状態を作る。
        # 既存のpts_out/final_wはこの時点では変更しない。
        voxel_edit_state_enabled = bool(getattr(self.args, "repair_voxel_edit_state", True))
        voxel_edit_mode = "prune_add_move_voxel_state" if voxel_edit_state_enabled else "disabled"

        voxel_edit_initial_coords = voxel_coords.detach()
        voxel_edit_initial_count = self._unique_voxel_count_from_cache(voxel_cache)

        neighbor_offsets = self.neighbor_offsets.to(device=pts_xyz.device, dtype=pts_xyz.dtype)
        neighbor_offsets_long = self.neighbor_offsets.to(device=pts_xyz.device, dtype=torch.long)

        empty_target_mask = self._empty_neighbor_target_mask(voxel_coords, voxel_cache=voxel_cache)

        B, _, N = pts_xyz.shape

        child_slot_mask = self._child_slot_target_mask(voxel_coords, octree_context)
        if child_slot_mask is not None:
            empty_target_mask = empty_target_mask & child_slot_mask
            actuator_target_mode = "octree_child_slot_masked"
        else:
            actuator_target_mode = "neighbor_empty_voxel"

        parent_occupancy_codes, source_child_slots = self._point_parent_codes_and_child_slots(voxel_coords)
        occupancy_code_popularity = self._occupancy_code_popularity(
            octree_context,
            full_octree_context,
            pts_xyz,
        )
        source_child_bits = torch.bitwise_left_shift(
            torch.ones_like(source_child_slots, dtype=torch.long),
            source_child_slots,
        )
        parent_codes_without_source = torch.bitwise_and(
            parent_occupancy_codes,
            255 - source_child_bits,
        )
        base_pattern_popularity = occupancy_code_popularity.index_select(
            0,
            parent_occupancy_codes.reshape(-1).clamp(0, 255),
        ).view(B, N)
        drop_pattern_gain = occupancy_code_popularity.index_select(
            0,
            parent_codes_without_source.reshape(-1).clamp(0, 255),
        ).view(B, N) - base_pattern_popularity

        current_voxels_n3 = voxel_coords.transpose(1, 2).contiguous()
        candidate_neighbor_voxels = (
            current_voxels_n3[:, :, None, :]
            + neighbor_offsets_long.view(1, 1, -1, 3)
        )
        target_child_slots = self._child_slot_from_coords_lastdim(candidate_neighbor_voxels)
        target_child_bits = torch.bitwise_left_shift(
            torch.ones_like(target_child_slots, dtype=torch.long),
            target_child_slots,
        )
        parent_code_expanded = parent_occupancy_codes[:, :, None].expand_as(target_child_bits)
        add_pattern_codes = torch.bitwise_or(parent_code_expanded, target_child_bits)
        move_pattern_codes = torch.bitwise_or(
            torch.bitwise_and(parent_code_expanded, 255 - source_child_bits[:, :, None]),
            target_child_bits,
        )

        add_pattern_gain = occupancy_code_popularity.index_select(
            0,
            add_pattern_codes.reshape(-1).clamp(0, 255),
        ).view(B, N, -1) - base_pattern_popularity[:, :, None]
        move_pattern_gain = occupancy_code_popularity.index_select(
            0,
            move_pattern_codes.reshape(-1).clamp(0, 255),
        ).view(B, N, -1) - base_pattern_popularity[:, :, None]

        pattern_gain_scale = max(float(getattr(self.args, "repair_pattern_prior_scale", 6.0)), 0.0)
        drop_pattern_prior = torch.tanh(drop_pattern_gain * pattern_gain_scale).unsqueeze(1)
        add_pattern_prior = torch.tanh(add_pattern_gain * pattern_gain_scale)
        move_pattern_prior = torch.tanh(move_pattern_gain * pattern_gain_scale)

        leaf_target_direction_prior = self._leaf_pattern_target_direction_priors(
            structure,
            target_child_slots,
            preserve,
            leaf_actuator_prior=leaf_actuator_prior,
        )
        leaf_add_target_bias = leaf_target_direction_prior["add_target_bias"]
        leaf_move_target_bias = leaf_target_direction_prior["move_target_bias"]

        # leaf pattern診断を「参考bias」ではなく、操作候補集合の制限にも使う。
        leaf_operation_masks = self._leaf_pattern_operation_masks(
            structure,
            preserve,
        )
        leaf_delete_op_mask = leaf_operation_masks["delete_mask"]
        leaf_add_op_mask = leaf_operation_masks["add_mask"]
        leaf_move_op_mask = leaf_operation_masks["move_mask"]
        actual_oracle_enabled = bool(leaf_operation_masks.get("actual_oracle_enabled", False))
        actual_oracle_has_drop = bool(leaf_delete_op_mask.detach().any().item()) if actual_oracle_enabled else False
        actual_oracle_has_add = bool(leaf_add_op_mask.detach().any().item()) if actual_oracle_enabled else False
        actual_oracle_has_move = bool(leaf_move_op_mask.detach().any().item()) if actual_oracle_enabled else False
        actual_oracle_drop_bad_mask = leaf_operation_masks.get(
            "actual_oracle_drop_bad_mask",
            torch.zeros_like(leaf_delete_op_mask, dtype=torch.bool),
        ).to(device=pts_xyz.device, dtype=torch.bool)
        actual_oracle_add_bad_mask = leaf_operation_masks.get(
            "actual_oracle_add_bad_mask",
            torch.zeros_like(leaf_add_op_mask, dtype=torch.bool),
        ).to(device=pts_xyz.device, dtype=torch.bool)
        actual_oracle_move_bad_mask = leaf_operation_masks.get(
            "actual_oracle_move_bad_mask",
            torch.zeros_like(leaf_move_op_mask, dtype=torch.bool),
        ).to(device=pts_xyz.device, dtype=torch.bool)
        actual_oracle_drop_bad_score = leaf_operation_masks.get(
            "actual_oracle_drop_bad_score",
            torch.zeros_like(leaf_delete_op_mask, dtype=pts_xyz.dtype),
        ).to(device=pts_xyz.device, dtype=pts_xyz.dtype)
        actual_oracle_add_bad_score = leaf_operation_masks.get(
            "actual_oracle_add_bad_score",
            torch.zeros_like(leaf_add_op_mask, dtype=pts_xyz.dtype),
        ).to(device=pts_xyz.device, dtype=pts_xyz.dtype)
        actual_oracle_move_bad_score = leaf_operation_masks.get(
            "actual_oracle_move_bad_score",
            torch.zeros_like(leaf_move_op_mask, dtype=pts_xyz.dtype),
        ).to(device=pts_xyz.device, dtype=pts_xyz.dtype)
        actual_oracle_has_bad_drop = bool(actual_oracle_drop_bad_mask.detach().any().item()) if actual_oracle_enabled else False
        actual_oracle_has_bad_add = bool(actual_oracle_add_bad_mask.detach().any().item()) if actual_oracle_enabled else False
        actual_oracle_has_bad_move = bool(actual_oracle_move_bad_mask.detach().any().item()) if actual_oracle_enabled else False
        actual_oracle_add_direction_index = leaf_operation_masks.get(
            "actual_oracle_best_add_direction_index",
            torch.full_like(leaf_add_op_mask, -1, dtype=torch.long),
        ).to(device=pts_xyz.device, dtype=torch.long)
        actual_oracle_bad_add_direction_index = leaf_operation_masks.get(
            "actual_oracle_bad_add_direction_index",
            torch.full_like(leaf_add_op_mask, -1, dtype=torch.long),
        ).to(device=pts_xyz.device, dtype=torch.long)
        actual_oracle_move_direction_index = leaf_operation_masks.get(
            "actual_oracle_move_direction_index",
            torch.full_like(leaf_move_op_mask, -1, dtype=torch.long),
        ).to(device=pts_xyz.device, dtype=torch.long)
        actual_oracle_bad_move_direction_index = leaf_operation_masks.get(
            "actual_oracle_move_bad_direction_index",
            torch.full_like(leaf_move_op_mask, -1, dtype=torch.long),
        ).to(device=pts_xyz.device, dtype=torch.long)

        child_slot_candidate_ratio = None
        if child_slot_mask is not None:
            child_slot_candidate_ratio = child_slot_mask.to(dtype=pts_xyz.dtype).mean()
        else:
            child_slot_candidate_ratio = pts_xyz.new_zeros(())

        if selection_mask is None:
            selection_bool = torch.ones((B, N), device=pts_xyz.device, dtype=torch.bool)
        else:
            selection_bool = selection_mask.squeeze(1) if selection_mask.ndim == 3 else selection_mask
            selection_bool = selection_bool.to(device=pts_xyz.device, dtype=torch.bool)

        voxel_point_counts = self._voxel_point_counts(voxel_coords, voxel_cache=voxel_cache).to(device=pts_xyz.device)
        before_occupied_voxels = self._unique_voxel_count_from_cache(voxel_cache, selection_bool)
        (
            drop_operation_gate,
            add_operation_gate,
            move_operation_gate,
            operation_gate_prob,
            operation_gate_hard,
            operation_gate_logit,
        ) = self._learned_operation_gates(
            actuator_features,
            prune_enabled=prune_enabled,
            add_enabled=add_enabled,
            move_enabled=disp_enabled,
        )
        if actual_oracle_enabled:
            if actual_oracle_has_drop and prune_enabled:
                drop_operation_gate = torch.ones_like(drop_operation_gate)
            else:
                drop_operation_gate = torch.zeros_like(drop_operation_gate)
            if actual_oracle_has_add and add_enabled:
                add_operation_gate = torch.ones_like(add_operation_gate)
            else:
                add_operation_gate = torch.zeros_like(add_operation_gate)
            if actual_oracle_has_move and disp_enabled:
                move_operation_gate = torch.ones_like(move_operation_gate)
            else:
                move_operation_gate = torch.zeros_like(move_operation_gate)

        if timing_enabled:
            _mark_runtime("setup")

        delete_prior = (
            0.95 * p_outlier
            + 0.75 * p_chain
            + 0.55 * p_sibling
            + 0.45 * p_parent
            + 0.20 * p_context
            + 0.25 * node_score
            + 0.25 * single_score
            + 0.15 * lowprob_score
            + 0.45 * quant_score
            + 0.25 * sparse_score
            + 0.35 * local_outlier_score
            - 0.85 * preserve
            - 0.75 * shape_score
        )
        if full_context_available:
            delete_prior = delete_prior + 0.05 * full_context_bonus
        if bool(leaf_actuator_prior.get("enabled", False)):
            delete_prior = delete_prior + float(
                getattr(self.args, "leaf_pattern_actuator_drop_weight", 0.75)
            ) * leaf_drop_prior
        delete_prior = delete_prior + float(
            getattr(self.args, "repair_drop_pattern_prior_weight", 1.5)
        ) * drop_pattern_prior
        # targetなしAmount学習では、Pruneの実行量をtarget_drop_ratioへ寄せない。
        # max_drop_ratioだけを0〜30%の探索上限として使う。
        target_drop_ratio = 0.0
        max_drop_ratio = min(
            max(float(getattr(self.args, "max_drop_ratio", 0.30)), 0.0),
            0.30,
        ) if prune_enabled else 0.0
        if not prune_enabled:
            max_drop_ratio = 0.0
        # Pruneする割合を特徴から学習し、固定target_drop_ratioだけに依存しない削除数にする。
        learned_drop_ratio = self._learned_operation_ratio(
            actuator_features,
            self.drop_amount_head,
            max_drop_ratio if prune_enabled else 0.0,
            "repair_drop_amount_random_mix_start",
            "repair_drop_amount_random_mix_end",
        )
        raw_learned_drop_ratio = learned_drop_ratio
        drop_ratio_floor = min(
            max(float(getattr(self.args, "repair_drop_ratio_floor", 0.0)), 0.0),
            float(max_drop_ratio),
        )
        if self.training and prune_enabled and drop_ratio_floor > 0.0:
            learned_drop_ratio_floored = torch.maximum(
                learned_drop_ratio,
                learned_drop_ratio.new_tensor(drop_ratio_floor),
            )
            learned_drop_ratio = (
                learned_drop_ratio_floored.detach()
                + learned_drop_ratio
                - learned_drop_ratio.detach()
            )
        learned_drop_ratio = learned_drop_ratio * drop_operation_gate
        learned_drop_ratio_for_ops = self._scale_amount_downstream_grad(
            learned_drop_ratio,
            op_name="drop",
        )
        # Prune量head用のSoft/Hard比較値を初期化する。
        # PruneはVoxel単位で行うため、量の比較も点数ではなくVoxel数基準で行う。
        drop_ratio_soft = pts_xyz.new_zeros(())
        drop_ratio_hard = pts_xyz.new_zeros(())
        drop_ratio_soft_batch = pts_xyz.new_zeros((B, 1, 1))
        drop_ratio_hard_batch = pts_xyz.new_zeros((B, 1, 1))

        drop_amount_supervision_loss = pts_xyz.new_zeros(())
        drop_amount_soft_consistency_loss = pts_xyz.new_zeros(())
        actual_oracle_drop_amount_loss = pts_xyz.new_zeros(())
        actual_oracle_add_amount_loss = pts_xyz.new_zeros(())
        actual_oracle_move_amount_loss = pts_xyz.new_zeros(())
        actual_oracle_drop_amount_logit_loss = pts_xyz.new_zeros(())
        actual_oracle_add_amount_logit_loss = pts_xyz.new_zeros(())
        actual_oracle_move_amount_logit_loss = pts_xyz.new_zeros(())

        soft_drop_budget = pts_xyz.new_zeros((B, 1, 1))
        valid_delete_voxel_count = pts_xyz.new_ones((B, 1, 1))

        soft_drop_sum = pts_xyz.new_zeros(())
        hard_drop_sum = pts_xyz.new_zeros(())
        soft_drop_voxel_sum = pts_xyz.new_zeros(())
        hard_drop_voxel_sum = pts_xyz.new_zeros(())
        # hard削除数は整数なので、学習比率の値だけを使ってVoxel選択数へ変換する。
        learned_drop_ratio_value = float(learned_drop_ratio.detach().mean().cpu()) if prune_enabled else 0.0
        delete_prior = torch.sigmoid(delete_prior.clamp(-8.0, 8.0))
        delete_prior = delete_prior * drop_operation_gate
        drop_score_noise = max(
            self._annealed_value("repair_drop_score_noise_start", "repair_drop_score_noise_end"),
            0.0,
        )
        # ============================================================
        # Prune Where logit の飽和対策
        # ============================================================
        # drop_head の raw logit が数十〜数百になると、
        # sigmoidが完全飽和して drop_prob_proxy から drop_head へ勾配が戻らなくなる。
        #
        # そのため、forward値として使うlogitをtanhで有界化する。
        # _scale_where_downstream_grad はその後に適用し、forward値は有界のまま、
        # backwardだけ操作別倍率で調整する。
        # ============================================================
        raw_drop_logit = self.drop_head(actuator_features)
        raw_drop_logit_for_forward = torch.nan_to_num(
            raw_drop_logit,
            nan=0.0,
            posinf=float(getattr(self.args, "repair_drop_where_logit_scale", 6.0)),
            neginf=-float(getattr(self.args, "repair_drop_where_logit_scale", 6.0)),
        )

        drop_logit_scale = max(
            float(getattr(self.args, "repair_drop_where_logit_scale", 6.0)),
            1e-6,
        )
        learned_drop_logit = drop_logit_scale * torch.tanh(raw_drop_logit_for_forward / drop_logit_scale)

        learned_drop_logit = self._scale_where_downstream_grad(
            learned_drop_logit,
            op_name="drop",
        )

        if self.training and drop_score_noise > 0.0:
            learned_drop_logit = learned_drop_logit + torch.randn_like(learned_drop_logit) * drop_score_noise
            # ノイズ追加後もsigmoid飽和を避けるため、もう一度有界化する。
            learned_drop_logit = drop_logit_scale * torch.tanh(learned_drop_logit / drop_logit_scale)

        learned_drop = torch.sigmoid(learned_drop_logit)
        learned_drop_prob = learned_drop.mean()
        drop_proxy_tau = max(float(getattr(self.args, "repair_drop_soft_proxy_tau", 8.0)), 1e-6)
        drop_prob_proxy = torch.sigmoid(learned_drop_logit / drop_proxy_tau)
        raw_proxy_grad_eps = min(
            max(
                float(
                    getattr(
                        self.args,
                        "repair_drop_where_proxy_raw_grad_eps",
                        0.001,
                    )
                ),
                0.0,
            ),
            0.20,
        )
        if raw_proxy_grad_eps > 0.0:
            raw_drop_logit_for_grad = torch.where(
                torch.isfinite(raw_drop_logit),
                raw_drop_logit,
                raw_drop_logit.detach().new_zeros(raw_drop_logit.shape),
            )
            drop_prob_proxy = (
                drop_prob_proxy
                + raw_proxy_grad_eps
                * (raw_drop_logit_for_grad - raw_drop_logit_for_grad.detach())
            )
        drop_prob_proxy = drop_prob_proxy * drop_operation_gate
        drop_prob = (repair_gate * delete_prior * learned_drop).clamp(0.0, 1.0)
        if prune_enabled and max_drop_ratio > 0.0:
            # ============================================================
            # Prune Amount由来のratio biasを弱める
            # ============================================================
            # Amountは「何個削るか」を決める役割であり、
            # Whereは「どこを削るか」のランキングを決める役割である。
            #
            # ここでratio biasを強く足しすぎると、drop_probが一気に1.0へ寄り、
            # drop_headのWhere勾配が再び消える。
            # そのためPruneだけbias倍率とclipを別に持たせる。
            # ============================================================
            prune_ratio_bias = self._ratio_bias(learned_drop_ratio_for_ops, max_drop_ratio)
            prune_ratio_bias = prune_ratio_bias * float(
                getattr(self.args, "repair_prune_ratio_bias_scale", 0.10)
            )
            prune_ratio_bias_clip = max(
                float(getattr(self.args, "repair_prune_ratio_bias_clip", 1.50)),
                0.0,
            )
            if prune_ratio_bias_clip > 0.0:
                prune_ratio_bias = prune_ratio_bias.clamp(
                    -prune_ratio_bias_clip,
                    prune_ratio_bias_clip,
                )

            drop_prob = torch.sigmoid(self._safe_logit(drop_prob) + prune_ratio_bias)

        drop_prob_direct = drop_prob
        drop_random_mix = min(
            max(self._annealed_value("repair_drop_random_mix_start", "repair_drop_random_mix_end"), 0.0),
            1.0,
        )
        if self.training and prune_enabled and drop_random_mix > 0.0:
            random_drop = self._random_ratio_mask_like(drop_prob, max_drop_ratio, selection_mask)
            drop_prob = ((1.0 - drop_random_mix) * drop_prob + drop_random_mix * random_drop).clamp(0.0, 1.0)
        if not prune_enabled:
            drop_prob = torch.zeros_like(drop_prob)
            drop_prob_direct = torch.zeros_like(drop_prob_direct)
            drop_prob_proxy = torch.zeros_like(drop_prob_proxy)
        drop_prob = self._voxel_mean_logits(drop_prob, voxel_coords, voxel_cache=voxel_cache).clamp(0.0, 1.0)
        if actual_oracle_enabled:
            oracle_drop_forward = leaf_delete_op_mask.to(device=drop_prob.device, dtype=drop_prob.dtype)
            oracle_grad_eps = min(
                max(float(getattr(self.args, "sparsepcgc_actual_oracle_where_grad_eps", 0.05)), 0.0),
                0.20,
            )
            drop_prob = (
                oracle_drop_forward.detach()
                + oracle_grad_eps * (drop_prob - drop_prob.detach())
            ).clamp(0.0, 1.0)
            drop_prob_direct = (
                oracle_drop_forward.detach()
                + oracle_grad_eps * (drop_prob_direct - drop_prob_direct.detach())
            ).clamp(0.0, 1.0)
            drop_prob_proxy = (
                oracle_drop_forward.detach()
                + oracle_grad_eps * (drop_prob_proxy - drop_prob_proxy.detach())
            )
        drop_prob_raw_for_amount = drop_prob
        delete_candidate_mask = selection_bool.clone()
        delete_max_points = int(getattr(self.args, "repair_delete_max_points_per_voxel", 8))
        if delete_max_points > 0:
            delete_candidate_mask = delete_candidate_mask & (
                voxel_point_counts.squeeze(1) <= float(delete_max_points)
            )

        # leaf pattern診断がDeleteを推奨したnode/voxelだけをDelete source候補にする。
        # これにより、圧縮率改善と無関係なDeleteを候補集合から除外する。
        if bool(leaf_operation_masks.get("enabled", False)):
            delete_candidate_mask = delete_candidate_mask & leaf_delete_op_mask.squeeze(1)
        delete_candidate_weight = delete_candidate_mask.unsqueeze(1).to(dtype=drop_prob.dtype)

        # voxel_point_counts は同一Voxel内の全点に同じ点数が入っている。
        # 1 / voxel_point_counts を掛けてsumすると、点数ではなくVoxel数として数えられる。
        voxel_count_weight = voxel_point_counts.to(device=drop_prob.device, dtype=drop_prob.dtype).clamp_min(1.0)
        delete_candidate_voxel_weight = delete_candidate_weight / voxel_count_weight

        # Soft削除score。
        # drop_prob_raw_for_amount はVoxel平均後なので、同一Voxel内では同じ値になる。
        soft_drop_prob_raw = drop_prob_raw_for_amount * delete_candidate_weight

        # learned_drop_ratio をTensorのまま使い、削除候補Voxel数に対するSoft削除予算を作る。
        # ここをdetach / float / itemにしないことが重要である。
        learned_drop_ratio_for_budget = learned_drop_ratio_for_ops.reshape(B, 1, 1).clamp(
            0.0,
            float(max_drop_ratio),
        )

        # 有効な削除候補Voxel数。
        # 点数ではなくVoxel数基準で数える。
        valid_delete_voxel_count = delete_candidate_voxel_weight.sum(dim=2, keepdim=True).clamp_min(1.0)

        # Soft削除予算。
        # 例：削除候補Voxelが1000個、learned_drop_ratio=0.1なら、100 voxel分を削除するSoft予算になる。
        soft_drop_budget = learned_drop_ratio_for_budget * valid_delete_voxel_count
        soft_drop_budget = torch.minimum(soft_drop_budget, valid_delete_voxel_count)

        # raw softのVoxel質量をdetachして正規化係数にする。
        # 分母をdetachすることで、総量方向の勾配を主に learned_drop_ratio / drop_amount_head に返す。
        soft_drop_raw_voxel_sum_det = (
            soft_drop_prob_raw.detach() * delete_candidate_voxel_weight
        ).sum(dim=2, keepdim=True)
        soft_drop_budget_scale = self._safe_budget_scale(soft_drop_raw_voxel_sum_det, soft_drop_budget)

        # Soft削除量を learned_drop_ratio が決めるVoxel予算に合わせる。
        # これにより「どのくらい削除するか」がVoxel数基準で学習対象になる。
        soft_drop_prob = torch.nan_to_num(
            soft_drop_prob_raw * soft_drop_budget_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        soft_drop_prob_clamped = soft_drop_prob.clamp(0.0, 1.0)

        saturated_grad_eps = max(
            float(getattr(self.args, "repair_drop_where_saturated_grad_eps", 0.05)),
            0.0,
        )
        saturated_grad_eps = min(saturated_grad_eps, 0.20)

        # ============================================================
        # Prune guard用のsoft値
        # ============================================================
        # forward値は0〜1にclampした値を使う。
        # backwardではsoft_drop_prob側へ微小勾配を残す。
        # ============================================================
        soft_drop_prob_for_guard = (
            soft_drop_prob_clamped
            + saturated_grad_eps * (soft_drop_prob - soft_drop_prob.detach())
        )

        # ============================================================
        # Prune Where用のSTE勾配経路
        # ============================================================
        # soft_drop_prob_for_guard はforward安全性のため0〜1へclampした値を使う。
        # ただしclamp飽和時にdrop_head勾配が消えないよう、
        # backwardだけ drop_prob_proxy 側へ戻す。
        # ============================================================
        prune_where_ste_grad_scale = max(
            float(getattr(self.args, "repair_prune_where_ste_grad_scale", 1.0)),
            0.0,
        )

        # ============================================================
        # Prune Where用のSTE勾配経路
        # ============================================================
        # forwardの削除候補制限は delete_candidate_weight に従う。
        # ただし backward では、delete_candidate_weight によって
        # drop_head 勾配が完全に0化されるのを避ける。
        #
        # soft_drop_where_grad_masked
        #   実際の削除候補制限を反映した従来のproxy
        #
        # soft_drop_where_grad_direct
        #   drop_headへ直接戻すための保険proxy
        #   forward値はmasked側に合わせ、backwardだけdrop_prob_proxyへ流す。
        # ============================================================

        soft_drop_where_grad_masked = (
            drop_prob_proxy * delete_candidate_weight
        ).clamp(0.0, 1.0)

        soft_drop_where_direct_grad_scale = max(
            float(getattr(self.args, "repair_prune_where_direct_grad_scale", 0.10)),
            0.0,
        )

        # ============================================================
        # 重要：
        # direct経路では clamp をかけない。
        # forward値は masked 側に合わせるが、backwardだけ drop_prob_proxy へ直接返す。
        # ここで clamp すると、forward が 0/1 境界にいる場合に再び勾配が消える。
        # ============================================================
        soft_drop_where_grad_direct = (
            soft_drop_where_grad_masked.detach()
            + soft_drop_where_direct_grad_scale
            * (drop_prob_proxy - drop_prob_proxy.detach())
        )

        # forward値は soft_drop_where_grad_masked と同じ。
        # backwardでは masked 経路と direct 経路の両方を使う。
        soft_drop_where_grad_base = (
            soft_drop_where_grad_masked.detach()
            + (soft_drop_where_grad_masked - soft_drop_where_grad_masked.detach())
            + (soft_drop_where_grad_direct - soft_drop_where_grad_direct.detach())
        )

        soft_drop_prob_for_ste = (
            soft_drop_prob_for_guard.detach()
            + prune_where_ste_grad_scale
            * (soft_drop_where_grad_base - soft_drop_where_grad_base.detach())
        )

        if actual_oracle_enabled and actual_oracle_has_drop:
            # The full-cloud teacher may intersect one selected subtree almost
            # completely. Cap the local hard application so a useful global
            # teacher cannot erase 90-100% of the shadow geometry.
            oracle_local_cap = min(
                float(max_drop_ratio),
                max(
                    float(
                        getattr(
                            self.args,
                            "sparsepcgc_actual_oracle_local_max_drop_ratio",
                            0.05,
                        )
                    ),
                    0.0,
                ),
            )
            hard_drop_mask = self._hard_voxel_drop_mask(
                voxel_coords,
                drop_prob,
                target_drop_ratio=oracle_local_cap,
                max_drop_ratio=oracle_local_cap,
                selection_mask=(
                    leaf_delete_op_mask.to(device=pts_xyz.device, dtype=torch.bool)
                    & delete_candidate_mask.unsqueeze(1)
                ),
                hard_threshold=float(getattr(self.args, "repair_drop_hard_threshold", 0.5)),
                voxel_cache=voxel_cache,
                force_min_count=False,
                max_hard_count=int(getattr(self.args, "repair_max_hard_drop_voxels", 0)),
                allow_single_candidate=False,
            )
        else:
            hard_drop_mask = self._hard_voxel_drop_mask(
                voxel_coords,
                drop_prob,
                target_drop_ratio=learned_drop_ratio_value,
                max_drop_ratio=learned_drop_ratio_value,
                selection_mask=delete_candidate_mask.unsqueeze(1),
                hard_threshold=float(getattr(self.args, "repair_drop_hard_threshold", 0.5)),
                voxel_cache=voxel_cache,
                force_min_count=bool(getattr(self.args, "repair_force_min_drop_voxels", False)),
                max_hard_count=int(getattr(self.args, "repair_max_hard_drop_voxels", 0)),
                allow_single_candidate=False,
            )
        hard_drop = hard_drop_mask.to(dtype=pts_xyz.dtype)
        # Phase3: Pruneは点削除ではなく、対象点が属するoccupied voxelの削除候補として記録する。
        voxel_edit_drop_mask = hard_drop_mask.detach().squeeze(1).to(dtype=torch.bool)


        # Hard forward + Soft backward のSTE。
        # forwardではVoxel単位のhard_drop、backwardではPrune Where専用proxyを使う。
        drop_prob_st = hard_drop - soft_drop_prob_for_ste.detach() + soft_drop_prob_for_ste

        # ============================================================
        # keep_prob も hard forward + soft backward にする。
        # forward値は 1 - hard_drop のまま。
        # backwardだけ Prune Where proxy へ戻す。
        #
        # ここで clamp(0, 1) を直接かけると、
        # keep_prob が 0 または 1 の境界に張り付き、Prune Where 勾配が消えやすい。
        # ============================================================
        keep_prob_hard = (1.0 - hard_drop).clamp(0.0, 1.0)
        keep_prob_soft_for_grad = 1.0 - soft_drop_where_grad_base

        keep_prob = (
            keep_prob_hard.detach()
            + (keep_prob_soft_for_grad - keep_prob_soft_for_grad.detach())
        )

        # Soft/Hard削除量の監視値。
        # point sum は実際に何点消えるかの確認用。
        soft_drop_sum = soft_drop_prob_for_guard.detach().sum()
        hard_drop_sum = hard_drop.detach().sum()

        # voxel sum は、何Voxel分を削除するかの確認用。
        # Hardは0/1なので、比較対象のSoftも0〜1に収めた値を使う。
        # soft_drop_prob本体は予算正規化で1を超える場合があり、
        # そのまま使うとdrop_ratio_softだけが過大になる。
        soft_drop_voxel_mass_per_batch = (
            soft_drop_prob_for_guard * delete_candidate_voxel_weight
        ).sum(dim=2, keepdim=True)

        hard_drop_voxel_mass_per_batch = (
            hard_drop.detach() * delete_candidate_voxel_weight
        ).sum(dim=2, keepdim=True)

        soft_drop_voxel_sum = soft_drop_voxel_mass_per_batch.detach().sum()
        hard_drop_voxel_sum = hard_drop_voxel_mass_per_batch.detach().sum()

        # learned_drop_ratio は「削除候補Voxelのうち何割を削除するか」を表すため、
        # drop_ratio_soft / drop_ratio_hard もVoxel数基準で計算する。
        drop_ratio_soft_batch = soft_drop_voxel_mass_per_batch / valid_delete_voxel_count
        drop_ratio_hard_batch = hard_drop_voxel_mass_per_batch / valid_delete_voxel_count

        drop_ratio_soft = drop_ratio_soft_batch.mean()
        drop_ratio_hard = drop_ratio_hard_batch.mean()

        # drop_amount_head 専用の量一致損失。
        # Hard削除量は教師値としてdetachし、learned_drop_ratio側だけに勾配を流す。
        drop_amount_supervision_loss = (
            drop_ratio_hard_batch.detach() - learned_drop_ratio
        ).pow(2).mean()
        if (actual_oracle_has_drop or actual_oracle_has_bad_drop) and max_drop_ratio > 0.0:
            drop_oracle_target_ratio = (
                drop_ratio_hard_batch.detach()
                if actual_oracle_has_drop
                else torch.zeros_like(drop_ratio_hard_batch)
            )
            actual_oracle_drop_amount_loss = self._actual_oracle_amount_bce_loss(
                raw_learned_drop_ratio,
                drop_oracle_target_ratio,
                max_drop_ratio,
            )
            drop_amount_logit_for_oracle = self._operation_amount_logit(
                actuator_features,
                self.drop_amount_head,
            ).mean()
            drop_amount_target_logit_for_oracle = self._target_ratio_logit(
                drop_oracle_target_ratio.mean(),
                max_drop_ratio,
                raw_learned_drop_ratio,
            ).detach()
            actual_oracle_drop_amount_logit_loss = (
                drop_amount_logit_for_oracle - drop_amount_target_logit_for_oracle
            ).pow(2)
            actual_oracle_drop_amount_loss = (
                actual_oracle_drop_amount_loss
                + float(getattr(self.args, "sparsepcgc_actual_oracle_amount_logit_weight", 0.25))
                * actual_oracle_drop_amount_logit_loss
            )

        # Soft削除量と learned_drop_ratio の整合性も補助的に見る。
        drop_amount_soft_consistency_loss = (
            drop_ratio_soft_batch.detach() - learned_drop_ratio
        ).pow(2).mean()

        if timing_enabled:
            _mark_runtime("delete")

        move_score = (repair_gate * (1.0 - hard_drop)).clamp(0.0, 1.0)
        subtree_move_source_logit = self.subtree_move_source_head(actuator_features)
        subtree_move_source_logit = self._scale_where_downstream_grad(
            subtree_move_source_logit,
            op_name="move",
        )
        subtree_move_source_logit = self._voxel_mean_logits(
            subtree_move_source_logit,
            voxel_coords,
            voxel_cache=voxel_cache,
        )
        subtree_move_source_prob = torch.sigmoid(subtree_move_source_logit).clamp(0.0, 1.0)
        move_source_prior = torch.sigmoid(
            (
                0.70 * p_comp
                + 0.55 * quant_score
                + 0.45 * sparse_score
                + 0.35 * p_chain
                + 0.25 * p_sibling
                + 0.20 * local_outlier_score
                - 0.45 * preserve
                - 0.65 * shape_score
            ).clamp(-8.0, 8.0)
        )
        if full_context_available:
            move_source_prior = (move_source_prior + 0.05 * full_context_bonus).clamp(0.0, 1.0)
        if bool(leaf_actuator_prior.get("enabled", False)):
            move_source_prior = torch.maximum(
                move_source_prior,
                float(getattr(self.args, "leaf_pattern_actuator_move_weight", 0.75)) * leaf_move_prior,
            ).clamp(0.0, 1.0)
        prior_weight = float(getattr(self.args, "repair_move_source_prior_weight", 0.35))
        if sparsepcgc_context:
            prior_weight = max(
                prior_weight,
                float(getattr(self.args, "sparsepcgc_move_source_prior_weight", 0.55)),
            )
        if prior_weight > 0.0:
            source_prior = (move_source_prior * prior_weight).clamp(0.0, 1.0)
            if selection_mask is not None:
                source_prior = source_prior * selection_mask.to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            move_score = torch.maximum(move_score, source_prior * (1.0 - hard_drop))
        move_pattern_source_prior = (
            move_pattern_prior.masked_fill(~empty_target_mask, -1.0)
            .amax(dim=2)
            .clamp_min(0.0)
            .unsqueeze(1)
        )
        move_score = torch.maximum(
            move_score,
            (
                float(getattr(self.args, "repair_move_pattern_prior_weight", 1.25))
                * move_pattern_source_prior
            ).clamp(0.0, 1.0) * (1.0 - hard_drop),
        )
        subtree_move_source_weight = max(
            float(getattr(self.args, "repair_subtree_move_source_prior_weight", 1.0)),
            0.0,
        )
        if subtree_move_source_weight > 0.0:
            move_score = torch.maximum(
                move_score,
                (
                    subtree_move_source_weight
                    * subtree_move_source_prob
                    * (1.0 - hard_drop)
                ).clamp(0.0, 1.0),
            )
        # targetなしAmount学習では、Moveの実行量をtarget_move_ratioへ寄せない。
        # max_move_ratioだけを0〜30%の探索上限として使う。
        target_move_ratio = 0.0
        raw_max_move_ratio = min(
            max(float(getattr(self.args, "max_move_ratio", 0.30)), 0.0),
            0.30,
        ) if disp_enabled else 0.0
        move_warmup = self._repair_move_warmup()
        max_move_ratio = raw_max_move_ratio * move_warmup if disp_enabled else 0.0
        # Adjustする割合を特徴から学習し、固定target_ratioだけに依存しないsource数にする。
        learned_move_ratio = self._learned_operation_ratio(
            actuator_features,
            self.move_amount_head,
            max_move_ratio,
            "repair_move_amount_random_mix_start",
            "repair_move_amount_random_mix_end",
        )
        raw_learned_move_ratio = learned_move_ratio
        # AdjustがHard実行0%に潰れるのを防ぐ。
        # forward値だけ下限を持たせ、backwardは元のlearned_move_ratioへ流す。
        move_ratio_floor = min(
            max(float(getattr(self.args, "repair_move_ratio_floor", 0.03)), 0.0),
            float(max_move_ratio),
        )
        if self.training and disp_enabled and move_ratio_floor > 0.0:
            learned_move_ratio_floored = torch.maximum(
                learned_move_ratio,
                learned_move_ratio.new_tensor(move_ratio_floor),
            )
            learned_move_ratio = (
                learned_move_ratio_floored.detach()
                + learned_move_ratio
                - learned_move_ratio.detach()
            )
        learned_move_ratio = learned_move_ratio * move_operation_gate
        move_score = move_score * move_operation_gate
        learned_move_ratio_for_ops = self._scale_amount_downstream_grad(
            learned_move_ratio,
            op_name="move",
        )
        # Adjust量head用のSoft/Hard比較値を初期化する。
        # Adjustはsource voxel単位で行うため、量の比較もVoxel数基準で行う。
        move_ratio_soft = pts_xyz.new_zeros(())
        move_ratio_hard = pts_xyz.new_zeros(())
        move_ratio_soft_batch = pts_xyz.new_zeros((B, 1, 1))
        move_ratio_hard_batch = pts_xyz.new_zeros((B, 1, 1))

        move_amount_supervision_loss = pts_xyz.new_zeros(())
        move_amount_soft_consistency_loss = pts_xyz.new_zeros(())

        soft_move_budget = pts_xyz.new_zeros((B, 1, 1))
        valid_move_source_voxel_count = pts_xyz.new_ones((B, 1, 1))

        soft_move_score = pts_xyz.new_zeros((B, 1, N))
        soft_move_score_for_guard = pts_xyz.new_zeros((B, 1, N))
        move_candidate_voxel_weight = pts_xyz.new_zeros((B, 1, N))

        soft_move_sum = pts_xyz.new_zeros(())
        hard_move_sum = pts_xyz.new_zeros(())
        soft_move_voxel_sum = pts_xyz.new_zeros(())
        hard_move_voxel_sum = pts_xyz.new_zeros(())
        # hard選択個数は整数なので、学習比率の値だけを使って選択数へ変換する。
        move_target_ratio = float(learned_move_ratio.detach().mean().cpu()) if disp_enabled else 0.0
        require_empty_move = bool(getattr(self.args, "repair_move_require_empty_target", True))
        prefer_occupied_move = bool(getattr(self.args, "repair_move_prefer_occupied_target", False)) and not require_empty_move
        sparse_empty_guard = bool(
            sparsepcgc_context and getattr(self.args, "enable_sparsepcgc_empty_target_guard", False)
        )
        # Legacy SparsePCGC tuning could prefer occupied targets for merge-like
        # behavior, but the empty-target guard is the stronger operation
        # invariant when enabled.
        if sparsepcgc_context and bool(getattr(self.args, "sparsepcgc_move_existing_target_only", True)) and not sparse_empty_guard:
            require_empty_move = False
            prefer_occupied_move = True
        elif sparse_empty_guard:
            # The SparsePCGC empty-target guard is the stronger invariant: if it is
            # enabled, Adjust is only allowed to move an occupied source voxel into
            # an empty target voxel, avoiding the old occupied-target preference.
            require_empty_move = True
            prefer_occupied_move = False
        dropped_target_mask = self._neighbor_target_membership_mask(
            voxel_coords,
            hard_drop_mask,
            voxel_cache=voxel_cache,
        )
        move_target_valid = torch.ones_like(move_score)
        if not disp_enabled:
            move_score = torch.zeros_like(move_score)
        elif max_move_ratio > 0.0:
            # 学習したAdjust量をsource scoreへ反映し、量と位置を同じlogit上で調整する。
            move_score = torch.sigmoid(self._safe_logit(move_score) + self._ratio_bias(learned_move_ratio_for_ops, max_move_ratio))
        move_score_noise = max(
            self._annealed_value("repair_move_score_noise_start", "repair_move_score_noise_end"),
            0.0,
        )
        if self.training and disp_enabled and move_score_noise > 0.0:
            # 学習初期のsource探索を広げるため、Adjust scoreへannealされるノイズを入れる。
            move_score = torch.sigmoid(self._safe_logit(move_score) + torch.randn_like(move_score) * move_score_noise)
        move_score = self._voxel_mean_logits(move_score, voxel_coords, voxel_cache=voxel_cache).clamp(0.0, 1.0)
        if require_empty_move:
            valid_move_points = empty_target_mask & (~dropped_target_mask)
        elif prefer_occupied_move:
            valid_move_points = (~empty_target_mask) & (~dropped_target_mask)
        else:
            valid_move_points = torch.ones_like(empty_target_mask, dtype=torch.bool) & (~dropped_target_mask)

        if (
            bool(getattr(self.args, "leaf_pattern_target_direction_mask", False))
            and bool(leaf_target_direction_prior.get("enabled", False))
        ):
            move_target_allowed = leaf_move_target_bias > 0
            valid_move_points = valid_move_points & move_target_allowed

        has_valid_move_target = valid_move_points.any(dim=2).unsqueeze(1).to(dtype=move_score.dtype)
        if require_empty_move:
            move_target_valid = has_valid_move_target
            move_score = move_score * has_valid_move_target
        elif prefer_occupied_move:
            move_target_valid = has_valid_move_target
            move_score = move_score * has_valid_move_target
        base_move_candidate_mask = selection_bool & (~hard_drop_mask.squeeze(1))
        base_move_candidate_mask = base_move_candidate_mask & has_valid_move_target.squeeze(1).to(dtype=torch.bool)

        # leaf pattern診断がMoveを推奨したnode/voxelだけをMove source候補にする。
        if bool(leaf_operation_masks.get("enabled", False)):
            base_move_candidate_mask = base_move_candidate_mask & leaf_move_op_mask.squeeze(1)

        move_candidate_mask = base_move_candidate_mask
        move_max_points = int(getattr(self.args, "repair_move_max_points_per_voxel", 8))

        if move_max_points > 0:
            limited_move_candidate_mask = base_move_candidate_mask & (
                voxel_point_counts.squeeze(1) <= float(move_max_points)
            )

            # 候補が少なすぎる場合だけ、Voxel内点数制限を緩める。
            # bool maskの切替なのでHard/Soft近似の勾配経路は壊さない。
            if self.training and bool(getattr(self.args, "repair_move_relax_voxel_count_when_starved", False)):
                min_ratio = min(
                    max(float(getattr(self.args, "repair_move_candidate_min_ratio", 0.05)), 0.0),
                    1.0,
                )
                base_count = base_move_candidate_mask.sum(dim=1)
                limited_count = limited_move_candidate_mask.sum(dim=1)
                min_count = torch.ceil(base_count.to(dtype=pts_xyz.dtype) * min_ratio).to(dtype=torch.long)

                starved = (base_count > 0) & (limited_count < min_count)
                move_candidate_mask = torch.where(
                    starved.view(B, 1),
                    base_move_candidate_mask,
                    limited_move_candidate_mask,
                )
            else:
                move_candidate_mask = limited_move_candidate_mask
        # Soft移動候補をHard移動候補と同じ候補集合に制限する。
        # Adjustはsource voxel単位で行うため、Soft側もVoxel単位の量として正規化する。
        move_candidate_weight = move_candidate_mask.unsqueeze(1).to(dtype=move_score.dtype)

        # voxel_point_counts は同一Voxel内の全点に同じ点数が入っている。
        # 1 / voxel_point_counts を掛けてsumすると、点数ではなくVoxel数として数えられる。
        move_voxel_count_weight = voxel_point_counts.to(
            device=move_score.device,
            dtype=move_score.dtype,
        ).clamp_min(1.0)

        move_candidate_voxel_weight = move_candidate_weight / move_voxel_count_weight

        # Soft移動score。
        # move_score はVoxel平均後なので、同一Voxel内では同じ値になる。
        soft_move_score_raw = move_score * move_candidate_weight

        # learned_move_ratio をTensorのまま使い、移動候補Voxel数に対するSoft移動予算を作る。
        # ここをdetach / float / itemにしないことが重要である。
        learned_move_ratio_for_budget = learned_move_ratio_for_ops.reshape(B, 1, 1).clamp(
            0.0,
            float(max_move_ratio),
        )

        # 有効な移動候補source voxel数。
        valid_move_source_voxel_count = move_candidate_voxel_weight.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(1.0)

        # Soft移動予算。
        # 例：移動候補Voxelが1000個、learned_move_ratio=0.1なら、100 voxel分をMoveするSoft予算になる。
        soft_move_budget = learned_move_ratio_for_budget * valid_move_source_voxel_count
        soft_move_budget = torch.minimum(soft_move_budget, valid_move_source_voxel_count)

        # raw softのVoxel質量をdetachして正規化係数にする。
        # 分母をdetachすることで、総量方向の勾配を主に learned_move_ratio / move_amount_head に返す。
        soft_move_raw_voxel_sum_det = (
            soft_move_score_raw.detach() * move_candidate_voxel_weight
        ).sum(dim=2, keepdim=True)
        soft_move_budget_scale = self._safe_budget_scale(soft_move_raw_voxel_sum_det, soft_move_budget)

        # Soft移動量を learned_move_ratio が決めるVoxel予算に合わせる。
        # これにより「どのくらいAdjustするか」がVoxel数基準で学習対象になる。
        soft_move_score = torch.nan_to_num(
            soft_move_score_raw * soft_move_budget_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ============================================================
        # Hard Moveは、方向guardを計算した後で作る。
        # ここでは一旦ゼロで初期化する。
        # guard前候補でHardを作ると、Soft側のguard後予算とズレる。
        # ============================================================
        hard_move_mask = torch.zeros((B, 1, N), device=pts_xyz.device, dtype=torch.bool)
        hard_move = hard_move_mask.to(dtype=pts_xyz.dtype)


        raw_hard_move_bool = torch.zeros((B, N), device=pts_xyz.device, dtype=torch.bool)

        move_mask = hard_move
        move_mask_for_guard = move_mask

        move_logits = self.move_voxel_head(actuator_features)
        move_logits = self._scale_where_downstream_grad(
            move_logits,
            op_name="move",
        )
        move_logits = self._voxel_mean_logits(move_logits, voxel_coords, voxel_cache=voxel_cache)

        # Section5:
        # best_move_target_child_slotと一致するtarget方向を強める。
        # 方向そのものは上流の valid_move_points でmask済みである。
        if bool(leaf_target_direction_prior.get("enabled", False)):
            move_logits = move_logits + float(
                getattr(self.args, "leaf_pattern_move_target_direction_weight", 1.25)
            ) * leaf_move_target_bias.permute(0, 2, 1).to(
                device=move_logits.device,
                dtype=move_logits.dtype,
            )

        move_valid_target = valid_move_points.transpose(1, 2)

        no_valid_move = ~move_valid_target.any(dim=1, keepdim=True)
        safe_valid_move = torch.where(no_valid_move, torch.ones_like(move_valid_target), move_valid_target)
        # float16でもoverflowしない負値を使う
        mask_value = torch.finfo(move_logits.dtype).min
        move_logits = move_logits.masked_fill(~safe_valid_move, mask_value)

        move_probs = torch.softmax(move_logits, dim=1)
        offset_norm = torch.linalg.norm(neighbor_offsets, dim=1).to(device=pts_xyz.device, dtype=move_logits.dtype)
        move_dir_target_logits = (-offset_norm.view(1, -1, 1)).expand_as(move_logits)
        move_dir_target_logits = move_dir_target_logits.masked_fill(~safe_valid_move, mask_value)
        move_dir_target = torch.softmax(move_dir_target_logits, dim=1).detach()
        move_direction_ce_per_point = -(
            move_dir_target * torch.log_softmax(move_logits, dim=1)
        ).sum(dim=1, keepdim=True)
        move_direction_ce = (
            move_direction_ce_per_point * has_valid_move_target.to(dtype=move_direction_ce_per_point.dtype)
        ).sum() / has_valid_move_target.sum().clamp_min(1.0)
        move_idx = move_probs.detach().argmax(dim=1, keepdim=True)
        hard_move_dir = torch.zeros_like(move_probs)
        hard_move_dir.scatter_(1, move_idx, 1.0)
        move_dir = hard_move_dir - move_probs.detach() + move_probs
        move_selected_valid = (
            move_dir * move_valid_target.to(dtype=move_dir.dtype)
        ).sum(dim=1, keepdim=True)
        # conflict系のguardは0〜1に収めたMove強度で計算する。
        quant_move_conflict_loss = self._masked_mean(
            move_mask_for_guard * (1.0 - move_selected_valid).clamp(0.0, 1.0),
            selection_mask,
        )
        selected_offsets = torch.einsum("bkn,kc->bcn", move_dir, neighbor_offsets)

        # soft方向用の連続target voxel座標
        target_voxels_soft = voxel_coords.to(dtype=pts_xyz.dtype) + selected_offsets

        # 点座標へ戻すときも、初期Octree/Voxelのglobal offset/global qsを使う。
        target_centers = self._voxel_centers_from_global_coords(
            target_voxels_soft,
            voxel_step,
            voxel_offset,
            dtype=pts_xyz.dtype,
        )
        move_idx_flat = move_idx.squeeze(1)
        selected_offsets_long = neighbor_offsets_long.index_select(0, move_idx_flat.reshape(-1))
        selected_offsets_long = selected_offsets_long.view(B, N, 3).transpose(1, 2).contiguous()
        source_sparsepcgc_coords = voxel_coords
        target_sparsepcgc_coords = voxel_coords + selected_offsets_long
        target_existing_occupied_mask = self._coords_membership_mask(
            target_sparsepcgc_coords,
            source_sparsepcgc_coords,
        )
        # ============================================================
        # Adjust Hard/Softを同じguard後候補集合で作る
        # ============================================================

        # まず、guard前Hardをデバッグ用に作る。
        # これは実行には使わず、raw_hard_move_countのログ専用である。
        raw_hard_move_mask = self._hard_voxel_drop_mask(
            voxel_coords,
            move_score,
            target_drop_ratio=move_target_ratio,
            max_drop_ratio=move_target_ratio,
            selection_mask=move_candidate_mask.unsqueeze(1),
            hard_threshold=float(getattr(self.args, "repair_move_hard_threshold", 0.5)),
            voxel_cache=voxel_cache,
            force_min_count=bool(getattr(self.args, "repair_force_min_move_voxels", False)),
            max_hard_count=int(getattr(self.args, "repair_max_hard_move_voxels", 0)),
        )
        raw_hard_move_bool = raw_hard_move_mask.squeeze(1).detach().to(dtype=torch.bool)

        # target重複の確認用。
        target_first_unique_raw_mask = self._first_unique_selected_mask(
            target_sparsepcgc_coords,
            raw_hard_move_bool,
        )
        target_duplicate_reject_mask = raw_hard_move_bool & (~target_first_unique_raw_mask)

        # penalty計算用には、guard前Softを使う。
        raw_move_mask_for_penalty = soft_move_score_for_guard

        empty_target_violation_loss = self._masked_mean(
            raw_move_mask_for_penalty
            * target_existing_occupied_mask.unsqueeze(1).to(dtype=pts_xyz.dtype),
            selection_mask,
        )
        target_duplicate_voxel_loss = self._masked_mean(
            raw_move_mask_for_penalty
            * target_duplicate_reject_mask.unsqueeze(1).to(dtype=pts_xyz.dtype),
            selection_mask,
        )

        empty_target_guard_enabled = bool(
            sparsepcgc_context and getattr(self.args, "enable_sparsepcgc_empty_target_guard", False)
        )
        target_duplicate_guard_enabled = bool(
            sparsepcgc_context and getattr(self.args, "enable_sparsepcgc_target_duplicate_guard", False)
        )

        # ------------------------------------------------------------
        # ここでHard/Soft共通のguard後候補集合を作る。
        # ------------------------------------------------------------
        guarded_move_candidate_mask = move_candidate_mask.clone()

        empty_guard_reject_bool = torch.zeros_like(raw_hard_move_bool, dtype=torch.bool)
        duplicate_guard_reject_bool = torch.zeros_like(raw_hard_move_bool, dtype=torch.bool)

        if empty_target_guard_enabled:
            empty_guard_reject_bool = guarded_move_candidate_mask & target_existing_occupied_mask
            guarded_move_candidate_mask = guarded_move_candidate_mask & (~target_existing_occupied_mask)

        # duplicate guardをかける前の候補を保存する。
        # empty target guardは維持したまま、duplicate guardだけ緩和できるようにする。
        guarded_move_candidate_mask_before_duplicate = guarded_move_candidate_mask.clone()

        if target_duplicate_guard_enabled:
            # Soft/Hard共通で、同じtarget voxelへ向かう候補は1つに絞る。
            target_first_unique_candidate_mask = self._first_unique_selected_mask(
                target_sparsepcgc_coords,
                guarded_move_candidate_mask,
            )
            duplicate_guard_reject_bool = guarded_move_candidate_mask & (~target_first_unique_candidate_mask)
            guarded_move_candidate_mask = guarded_move_candidate_mask & target_first_unique_candidate_mask

        # duplicate guardでAdjust候補がほぼ消える場合だけ、duplicate guardを緩める。
        # empty target guard後の候補へ戻すため、既存occupied targetへの移動は許可しない。
        if (
            self.training
            and target_duplicate_guard_enabled
            and bool(getattr(self.args, "repair_move_relax_duplicate_guard_when_starved", False))
        ):
            min_ratio = min(
                max(float(getattr(self.args, "repair_move_candidate_min_ratio", 0.05)), 0.0),
                1.0,
            )
            before_dup_count = guarded_move_candidate_mask_before_duplicate.sum(dim=1)
            after_dup_count = guarded_move_candidate_mask.sum(dim=1)
            min_count = torch.ceil(before_dup_count.to(dtype=pts_xyz.dtype) * min_ratio).to(dtype=torch.long)

            starved = (before_dup_count > 0) & (after_dup_count < min_count)
            guarded_move_candidate_mask = torch.where(
                starved.view(B, 1),
                guarded_move_candidate_mask_before_duplicate,
                guarded_move_candidate_mask,
            )

        # guardで落ちた候補のログ用。
        guard_rejected_bool = move_candidate_mask & (~guarded_move_candidate_mask)

        # ------------------------------------------------------------
        # Hard Moveをguard後候補から作り直す。
        # これが最重要である。
        # ------------------------------------------------------------
        valid_move_source_voxel_count_effective = (
            move_candidate_voxel_weight
            * guarded_move_candidate_mask.unsqueeze(1).to(dtype=move_candidate_voxel_weight.dtype)
        ).sum(dim=2, keepdim=True).clamp_min(1.0)
        # guard後にMove候補が完全に消えると、move_amount_head への下流勾配も消える。
        # その場合だけ、forward値はほぼ変えず、backward用に極小の候補重みを残す。
        if self.training and disp_enabled:
            guarded_candidate_count = guarded_move_candidate_mask.sum(dim=1, keepdim=True)
            no_guarded_candidate = guarded_candidate_count <= 0

            if bool(no_guarded_candidate.any().detach().cpu().item()):
                # Phase1:
                # SparsePCGCではMove候補が消えた場合にbase候補へ戻さない。
                # 候補復活はoccupancy pattern破壊を再発させるため、
                # forward/backwardともguard後候補だけを使う。
                move_allowed_weight = guarded_move_candidate_mask.unsqueeze(1).to(
                    dtype=move_candidate_voxel_weight.dtype
                )
            else:
                move_allowed_weight = guarded_move_candidate_mask.unsqueeze(1).to(
                    dtype=move_candidate_voxel_weight.dtype
                )
        else:
            move_allowed_weight = guarded_move_candidate_mask.unsqueeze(1).to(dtype=move_candidate_voxel_weight.dtype)

        move_target_ratio_for_hard = move_target_ratio
        min_move_expected_voxels = max(
            float(getattr(self.args, "repair_move_min_hard_expected_voxels", 1.0)),
            0.0,
        )
        if min_move_expected_voxels > 0.0 and disp_enabled:
            expected_move_voxels = (
                learned_move_ratio.detach().reshape(B, 1, 1)
                * valid_move_source_voxel_count_effective.detach()
            ).mean()
            if float(expected_move_voxels.cpu()) < min_move_expected_voxels:
                move_target_ratio_for_hard = 0.0

        hard_move_mask = self._hard_voxel_drop_mask(
            voxel_coords,
            move_score,
            target_drop_ratio=move_target_ratio_for_hard,
            max_drop_ratio=move_target_ratio_for_hard,
            selection_mask=guarded_move_candidate_mask.unsqueeze(1),
            hard_threshold=float(getattr(self.args, "repair_move_hard_threshold", 0.5)),
            voxel_cache=voxel_cache,
            force_min_count=bool(getattr(self.args, "repair_force_min_move_voxels", False)),
            max_hard_count=int(getattr(self.args, "repair_max_hard_move_voxels", 0)),
        )
        hard_move = hard_move_mask.to(dtype=pts_xyz.dtype)
        # Phase3: guard後に確定したHard Move sourceをVoxel編集状態用に保存する。
        voxel_edit_move_source_mask = hard_move_mask.detach().squeeze(1).to(dtype=torch.bool)

        # ------------------------------------------------------------
        # Soft Moveも同じguard後候補から作る。
        # ------------------------------------------------------------
        soft_move_score_effective_raw = soft_move_score * move_allowed_weight

        soft_move_budget_effective = learned_move_ratio_for_budget * valid_move_source_voxel_count_effective
        soft_move_budget_effective = torch.minimum(
            soft_move_budget_effective,
            valid_move_source_voxel_count_effective,
        )

        soft_move_effective_voxel_sum_det = (
            soft_move_score_effective_raw.detach() * move_candidate_voxel_weight
        ).sum(dim=2, keepdim=True)
        soft_move_effective_budget_scale = self._safe_budget_scale(
            soft_move_effective_voxel_sum_det,
            soft_move_budget_effective,
        )

        soft_move_score_effective = torch.nan_to_num(
            soft_move_score_effective_raw * soft_move_effective_budget_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ------------------------------------------------------------
        # Adjust = source Soft Prune + target Soft Add
        # ------------------------------------------------------------
        move_source_soft_delete = soft_move_score_effective.clamp(0.0, 1.0)
        move_source_soft_keep = (1.0 - move_source_soft_delete).clamp(0.0, 1.0)
        move_target_soft_add = move_source_soft_delete

        # ------------------------------------------------------------
        # Hard forward + Soft backward
        # forwardはguard後Hard、backwardはguard後Soft。
        # ------------------------------------------------------------
        move_mask = hard_move - soft_move_score_effective.detach() + soft_move_score_effective

        # Pruneで削除された点はAdjustしない。
        move_mask = move_mask * keep_prob

        soft_move_score_for_guard = soft_move_score_effective.clamp(0.0, 1.0)
        move_mask_for_guard = move_mask.clamp(0.0, 1.0)

        # ------------------------------------------------------------
        # Soft/Hard量の統計をguard後基準で再計算する。
        # ------------------------------------------------------------
        soft_move_sum = soft_move_score_for_guard.detach().sum()
        hard_move_sum = hard_move.detach().sum()

        soft_move_voxel_mass_per_batch = (
            soft_move_score_effective * move_candidate_voxel_weight
        ).sum(dim=2, keepdim=True)

        hard_move_voxel_mass_per_batch = (
            hard_move.detach() * move_candidate_voxel_weight
        ).sum(dim=2, keepdim=True)

        soft_move_voxel_sum = soft_move_voxel_mass_per_batch.detach().sum()
        hard_move_voxel_sum = hard_move_voxel_mass_per_batch.detach().sum()

        move_ratio_soft_batch = soft_move_voxel_mass_per_batch / valid_move_source_voxel_count_effective
        move_ratio_hard_batch = hard_move_voxel_mass_per_batch / valid_move_source_voxel_count_effective

        move_ratio_soft = move_ratio_soft_batch.mean()
        move_ratio_hard = move_ratio_hard_batch.mean()

        move_amount_supervision_loss = (
            move_ratio_hard_batch.detach() - learned_move_ratio
        ).pow(2).mean()

        move_amount_soft_consistency_loss = (
            move_ratio_soft_batch.detach() - learned_move_ratio
        ).pow(2).mean()

        primitive_delta = target_centers - pts_xyz
        delta = move_mask * primitive_delta
        pts_out = pts_xyz + delta
        move_target_voxel_coords = voxel_coords + selected_offsets_long
        # Phase3: Moveを source voxel削除 + target voxel追加 として扱うためのtarget制約。
        voxel_edit_same_move_mask = (move_target_voxel_coords == voxel_coords).all(dim=1)

        voxel_edit_move_target_existing_mask = self._coords_membership_mask(
            move_target_voxel_coords,
            voxel_coords,
        )

        voxel_edit_move_target_empty_mask = torch.gather(
            empty_target_mask,
            2,
            move_idx_flat.unsqueeze(-1),
        ).squeeze(-1)

        if child_slot_mask is not None:
            voxel_edit_move_child_slot_mask = torch.gather(
                child_slot_mask,
                2,
                move_idx_flat.unsqueeze(-1),
            ).squeeze(-1)
        else:
            voxel_edit_move_child_slot_mask = torch.ones_like(voxel_edit_move_target_empty_mask, dtype=torch.bool)

        voxel_edit_move_valid_mask = voxel_edit_move_source_mask
        voxel_edit_move_valid_mask = voxel_edit_move_valid_mask & (~voxel_edit_same_move_mask)

        if bool(getattr(self.args, "repair_voxel_edit_require_empty_move_target", True)):
            voxel_edit_move_valid_mask = voxel_edit_move_valid_mask & voxel_edit_move_target_empty_mask
            voxel_edit_move_valid_mask = voxel_edit_move_valid_mask & (~voxel_edit_move_target_existing_mask)

        voxel_edit_move_valid_mask = voxel_edit_move_valid_mask & voxel_edit_move_child_slot_mask

        final_voxel_coords = torch.where(
            hard_move_mask.to(device=voxel_coords.device, dtype=torch.bool).expand_as(voxel_coords),
            move_target_voxel_coords,
            voxel_coords,
        )
        same_voxel_move_mask = (
            hard_move_mask.squeeze(1)
            & (move_target_voxel_coords == voxel_coords).all(dim=1)
        )
        moved_different_voxel_mask = hard_move_mask.squeeze(1) & (~same_voxel_move_mask)
        # Actuator内部のVoxel状態を更新する。
        # pts_outから再Voxel化せず、初期Voxelに対するMove結果として保持する。
        final_voxel_coords_state = torch.where(
            hard_move_mask.to(device=voxel_coords.device, dtype=torch.bool).expand_as(voxel_coords),
            move_target_voxel_coords,
            voxel_coords,
        )
        if timing_enabled:
            _mark_runtime("adjust_move")

        # BCE系損失へ渡る可能性があるため、final_wを確率範囲へ安全に収める。
        # NaN/Infもここで除去する。
        # forward値は keep_prob_hard によって 0/1 範囲内にある。
        # ここで clamp すると backward が再び境界で潰れるため、nan_to_num のみにする。
        final_w = torch.nan_to_num(
            keep_prob,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        max_add_ratio_value = self._max_add_ratio()
        # Addする割合を特徴から学習し、固定10%のような張り付きから外す。
        learned_add_ratio = self._learned_operation_ratio(
            actuator_features,
            self.add_amount_head,
            max_add_ratio_value if add_enabled else 0.0,
            "repair_add_amount_random_mix_start",
            "repair_add_amount_random_mix_end",
        )
        raw_learned_add_ratio = learned_add_ratio
        add_ratio_floor_applied = False
        add_ratio_floor = min(max(float(getattr(self.args, "repair_add_ratio_floor", 0.0)), 0.0), max_add_ratio_value)
        if self.training and add_enabled and add_ratio_floor > 0.0:
            learned_add_ratio_before_floor = learned_add_ratio

            learned_add_ratio_floored = torch.maximum(
                learned_add_ratio,
                learned_add_ratio.new_tensor(add_ratio_floor),
            ).clamp(0.0, max_add_ratio_value)

            # forward値だけfloorを効かせ、backwardは元のlearned_add_ratioへ流す
            learned_add_ratio = (
                learned_add_ratio_floored.detach()
                + learned_add_ratio
                - learned_add_ratio.detach()
            )

            add_ratio_floor_applied = bool(
                (learned_add_ratio_floored > learned_add_ratio_before_floor + 1e-12)
                .detach()
                .any()
                .cpu()
                .item()
            )
        learned_add_ratio = learned_add_ratio * add_operation_gate
        learned_add_ratio_for_ops = self._scale_amount_downstream_grad(
            learned_add_ratio,
            op_name="add",
        )
        # hardなtop-k個数は整数なので、学習比率の値だけを候補数計算へ渡す。
        learned_add_ratio_value = float(learned_add_ratio.detach().mean().cpu()) if add_enabled else 0.0
        add_k, add_candidate_ratio = self._target_add_count(
            N,
            candidate_ratio_override=learned_add_ratio_value,
            force_min_count=(
                bool(getattr(self.args, "repair_force_min_add_voxels", False))
                or bool(actual_oracle_enabled and actual_oracle_has_add)
            ),
        )
        add_ratio = pts_xyz.new_zeros(())
        add_ratio_loss = pts_xyz.new_zeros(())
        add_shape_guard = pts_xyz.new_zeros(())
        add_offset_reg = pts_xyz.new_zeros(())
        add_drop_conflict_loss = pts_xyz.new_zeros(())
        added_keep_loss = pts_xyz.new_zeros(())
        add_min_offset_loss = pts_xyz.new_zeros(())
        quant_add_guard = pts_xyz.new_zeros(())
        add_direction_ce = pts_xyz.new_zeros(())
        add_prob = pts_xyz.new_zeros((B, 1, N))
        add_priority = add_prob
        K_add = int(neighbor_offsets.shape[0])

        add_ratio_soft = pts_xyz.new_zeros(())
        add_ratio_hard = pts_xyz.new_zeros(())
        add_amount_supervision_loss = pts_xyz.new_zeros(())
        add_amount_soft_consistency_loss = pts_xyz.new_zeros(())

        soft_add_budget = pts_xyz.new_zeros((B, 1))
        soft_add_pair = pts_xyz.new_zeros((B, N * K_add))
        hard_add_pair = pts_xyz.new_zeros((B, N * K_add))
        add_count_value = 0
        add_effective_count_value = 0
        add_target_voxel_count_value = 0
        # ============================================================
        # Addをtarget voxel単位で扱うためのSoft/Hard状態。
        # 既存のadd_probはsource点単位に畳まれるため、Add量の主指標には使わない。
        # ============================================================
        add_target_soft_add = pts_xyz.new_zeros((B, 1, 0))
        add_target_hard_add = pts_xyz.new_zeros((B, 1, 0))
        add_target_add_st = pts_xyz.new_zeros((B, 1, 0))
        add_target_voxel_coords = voxel_coords.new_empty((B, 3, 0))
        # Phase3: Addが実行されない場合でも、Voxel編集状態構築で参照できる空のAdd候補を用意する。
        voxel_edit_add_target_coords = add_target_voxel_coords.detach()
        voxel_edit_add_target_mask = torch.zeros((B, 0), device=pts_xyz.device, dtype=torch.bool)

        add_score_noise = max(
            self._annealed_value("repair_add_score_noise_start", "repair_add_score_noise_end"),
            0.0,
        )
        add_weight_random_mix = min(
            max(self._annealed_value("repair_add_weight_random_mix_start", "repair_add_weight_random_mix_end"), 0.0),
            1.0,
        )
        if add_k > 0:
            learned_add_logit = self.add_head(actuator_features)
            learned_add_logit = self._scale_where_downstream_grad(
                learned_add_logit,
                op_name="add",
            )
            add_prior = (
                0.25 * p_sibling
                + 0.20 * p_parent
                + 0.35 * p_context
                + 1.00 * p_comp
                + 0.15 * lowprob_score
                - 0.90 * preserve
                - 0.75 * p_outlier
                - 0.85 * quant_score
                - 0.55 * sparse_score
                - 0.45 * local_outlier_score
                - 0.65 * shape_score
            )
            if full_context_available:
                add_prior = add_prior + 0.05 * full_context_bonus
            if bool(leaf_actuator_prior.get("enabled", False)):
                add_prior = add_prior + float(
                    getattr(self.args, "leaf_pattern_actuator_add_weight", 0.50)
                ) * leaf_add_prior
            add_prior = add_prior + float(
                getattr(self.args, "repair_add_pattern_prior_weight", 1.25)
            ) * add_pattern_prior.clamp_min(0.0).amax(dim=2, keepdim=True).transpose(1, 2)
            if sparsepcgc_add_experiment_active and not bool(getattr(self.args, "sparsepcgc_add_use_candidate_score", True)):
                add_prior = torch.zeros_like(add_prior)
            # Add量の学習結果を位置logitに足し、どのVoxelへ追加するかの勾配も残す。
            add_logit = learned_add_logit + add_prior + self._ratio_bias(learned_add_ratio_for_ops, max_add_ratio_value)
            if self.training and add_score_noise > 0.0:
                add_logit = add_logit + self._gumbel_like(add_logit) * add_score_noise
            add_voxel_logits = self.add_voxel_head(actuator_features)
            add_voxel_logits = self._scale_where_downstream_grad(
                add_voxel_logits,
                op_name="add",
            )
            add_logit = self._voxel_mean_logits(add_logit, voxel_coords, voxel_cache=voxel_cache)
            add_voxel_logits = self._voxel_mean_logits(add_voxel_logits, voxel_coords, voxel_cache=voxel_cache)

            if bool(leaf_target_direction_prior.get("enabled", False)):
                add_voxel_logits = add_voxel_logits + float(
                    getattr(self.args, "leaf_pattern_add_target_direction_weight", 1.25)
                ) * leaf_add_target_bias.permute(0, 2, 1).to(
                    device=add_voxel_logits.device,
                    dtype=add_voxel_logits.dtype,
                )

            pair_logits = (add_voxel_logits + add_logit).permute(0, 2, 1).contiguous()

            pair_logits = pair_logits + float(
                getattr(self.args, "repair_add_pair_pattern_prior_weight", 2.0)
            ) * add_pattern_prior
            if selection_mask is None:
                base_valid = torch.ones((B, N), device=pts_xyz.device, dtype=torch.bool)
            else:
                base_valid = selection_mask.squeeze(1) if selection_mask.ndim == 3 else selection_mask
                base_valid = base_valid.to(device=pts_xyz.device, dtype=torch.bool)
            keep_threshold = float(getattr(self.args, "add_noop_keep_threshold", 0.5))
            base_valid = base_valid & (~hard_drop_mask.squeeze(1))
            if keep_threshold > 0.0:
                base_valid = base_valid & (keep_prob.detach().squeeze(1) >= keep_threshold)

            # leaf pattern診断がAddを推奨したnode/voxelだけをAdd source候補にする。
            if bool(leaf_operation_masks.get("enabled", False)):
                base_valid = base_valid & leaf_add_op_mask.squeeze(1)
            valid_pair = empty_target_mask & base_valid.unsqueeze(2)
            if (
                bool(getattr(self.args, "leaf_pattern_target_direction_mask", False))
                and bool(leaf_target_direction_prior.get("enabled", False))
            ):
                add_target_allowed = leaf_add_target_bias > 0
                valid_pair = valid_pair & add_target_allowed

            candidate_base_voxels_long = voxel_coords.transpose(1, 2).contiguous().unsqueeze(2)  # [B, N, 1, 3]
            candidate_offsets_long = neighbor_offsets_long.view(1, 1, -1, 3)                    # [1, 1, K, 3]
            candidate_target_voxels_long = candidate_base_voxels_long + candidate_offsets_long  # [B, N, K, 3]

            candidate_target_voxels_flat = (
                candidate_target_voxels_long
                .reshape(B, -1, 3)
                .transpose(1, 2)
                .contiguous()
            )  # [B, 3, N*K]

            unique_target_pair_mask = (
                self._first_unique_coord_mask(candidate_target_voxels_flat)
                .view(B, N, -1)
            )

            valid_pair = valid_pair & unique_target_pair_mask

            # ============================================================
            # Addは必ず「現在のHard状態で空のtarget voxel」だけを候補にする。
            # empty_target_maskは初期voxel_coords基準なので、
            # Prune/Adjust後のfinal_voxel_coords_stateに対しても空であるか確認する。
            # ============================================================
            add_hardening_threshold = float(
                getattr(
                    self.args,
                    "operation_count_drop_threshold",
                    getattr(self.args, "test_drop_threshold", 0.5),
                )
            )
            current_keep_for_add = final_w.detach().squeeze(1) >= add_hardening_threshold

            current_empty_target_pair_mask = torch.zeros_like(valid_pair, dtype=torch.bool)
            for b in range(B):
                current_occ_coords = final_voxel_coords_state[b].transpose(0, 1).contiguous()
                current_occ_coords = current_occ_coords[current_keep_for_add[b]]

                candidate_targets_b = candidate_target_voxels_long[b].reshape(-1, 3).contiguous()

                if current_occ_coords.numel() == 0:
                    current_empty_target_pair_mask[b] = True
                else:
                    occupied_now = self._coords_membership(
                        candidate_targets_b,
                        current_occ_coords,
                    ).view(N, K_add)
                    current_empty_target_pair_mask[b] = ~occupied_now

            valid_pair = valid_pair & current_empty_target_pair_mask

            # AMP/float16環境では、float32の最小値(-3e38)をhalf Tensorへ入れるとoverflowする。
            # そのため、masked_fillに使う負値は、必ず対象Tensor自身のdtypeから作る。
            add_dir_logits = add_voxel_logits.permute(0, 2, 1).contiguous()
            add_dir_mask_value = torch.finfo(add_dir_logits.dtype).min
            add_dir_logits = add_dir_logits.masked_fill(~valid_pair, add_dir_mask_value)

            add_dir_target_logits = (
                -offset_norm.to(dtype=add_dir_logits.dtype).view(1, 1, -1)
            ).expand_as(add_dir_logits)
            add_dir_target_logits = add_dir_target_logits.masked_fill(
                ~valid_pair,
                add_dir_mask_value,
            )
            add_dir_target = torch.softmax(add_dir_target_logits, dim=2).detach()
            add_direction_ce_per_point = -(
                add_dir_target * torch.log_softmax(add_dir_logits, dim=2)
            ).sum(dim=2, keepdim=True)
            add_valid_point = valid_pair.any(dim=2, keepdim=True).to(dtype=add_direction_ce_per_point.dtype)
            add_direction_ce = (
                add_direction_ce_per_point * add_valid_point
            ).sum() / add_valid_point.sum().clamp_min(1.0)
            valid_counts = valid_pair.reshape(B, -1).sum(dim=1)
            effective_add_k = min(int(add_k), int(valid_counts.min().detach().cpu().item()))
            if effective_add_k > 0:
                pair_mask_value = torch.finfo(pair_logits.dtype).min
                pair_scores = pair_logits.masked_fill(~valid_pair, pair_mask_value).reshape(B, -1)
                top_pair_values, top_pair_idx = torch.topk(
                    pair_scores.detach(),
                    k=effective_add_k,
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                selected_pair_logits = torch.gather(pair_scores, 1, top_pair_idx)
                add_strength_tau = max(float(getattr(self.args, "repair_add_strength_tau", 1.0)), 1e-6)
                top_threshold = top_pair_values.min(dim=1, keepdim=True).values.detach()
                selected_pair_strength = torch.sigmoid((selected_pair_logits - top_threshold) / add_strength_tau)
                selected_pair_strength = torch.nan_to_num(
                    selected_pair_strength,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                add_base_idx = torch.div(top_pair_idx, neighbor_offsets.shape[0], rounding_mode="floor")
                add_dir_idx = top_pair_idx.remainder(neighbor_offsets.shape[0])
                idx_expand_xyz = add_base_idx.unsqueeze(1).expand(-1, 3, -1)
                selected_base_voxels_long = torch.gather(voxel_coords, 2, idx_expand_xyz)
                selected_offsets_add_long = neighbor_offsets_long.index_select(0, add_dir_idx.reshape(-1))
                selected_offsets_add_long = selected_offsets_add_long.view(B, effective_add_k, 3).transpose(1, 2)
                selected_add_voxels_long = selected_base_voxels_long + selected_offsets_add_long
                unique_add_target_mask = self._first_unique_coord_mask(selected_add_voxels_long).to(
                    dtype=pair_scores.dtype
                )
                if threshold_cap_mode:
                    hard_top_add = (
                        selected_pair_strength >= float(getattr(self.args, "repair_add_hard_threshold", 0.5))
                    ).to(dtype=pair_scores.dtype)
                else:
                    hard_top_add = torch.ones_like(selected_pair_strength)
                hard_top_add = hard_top_add * unique_add_target_mask
                if self.training and add_weight_random_mix > 0.0:
                    random_hard_add = (
                        torch.rand_like(hard_top_add) < float(add_weight_random_mix)
                    ).to(dtype=hard_top_add.dtype)
                    hard_top_add = torch.maximum(hard_top_add, random_hard_add * unique_add_target_mask)
                hard_add_pair = torch.zeros_like(pair_scores)
                hard_add_pair.scatter_(1, top_pair_idx, hard_top_add)

                tau = max(float(getattr(self.args, "add_soft_match_tau", 0.05)), 1e-6)
                threshold = top_threshold

                valid_pair_flat = valid_pair.reshape(B, -1).to(dtype=pair_scores.dtype)

                # pairごとのSoft追加確率。
                # これは「どのVoxelへ追加するか」のSoft表現であり、add_head / add_voxel_headへ勾配を返す。
                soft_add_pair_raw = torch.sigmoid((pair_scores - threshold) / tau)
                soft_add_pair_raw = soft_add_pair_raw * valid_pair_flat

                # add_amount_head が出した learned_add_ratio をTensorのまま使い、
                # このStepで全体のうち何割をAddするかのSoft予算にする。
                learned_add_ratio_for_budget = learned_add_ratio_for_ops.reshape(B, 1).clamp(
                    0.0,
                    float(max_add_ratio_value),
                )

                # N点に対する追加数のSoft予算。
                # 例：N=29066, learned_add_ratio=0.0115 なら約334点分のSoft予算。
                soft_add_budget = learned_add_ratio_for_budget * float(N)

                # 有効候補数を超えないようにする。
                valid_pair_count = valid_pair_flat.sum(dim=1, keepdim=True).clamp_min(1.0)
                soft_add_budget = torch.minimum(soft_add_budget, valid_pair_count)

                # raw Soft値の合計をdetachして正規化係数にする。
                # 分母をdetachすることで、Soft総量の主な勾配を learned_add_ratio に返す。
                soft_add_raw_sum_det = soft_add_pair_raw.detach().sum(dim=1, keepdim=True)
                soft_add_budget_scale = self._safe_budget_scale(soft_add_raw_sum_det, soft_add_budget)

                # Soft追加量を learned_add_ratio が決める予算に合わせる。
                # これにより add_amount_head が「どのくらい追加するか」を学習対象にできる。
                soft_add_pair = torch.nan_to_num(
                    soft_add_pair_raw * soft_add_budget_scale,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                soft_add_pair = soft_add_pair.clamp(0.0, 1.0)

                # ============================================================
                # pair全体のSTEは補助的なsource側可視化にだけ使う。
                # Addの実操作量は、後でtop-k selected target voxel単位で作る。
                # ============================================================
                add_pair_st = hard_add_pair - soft_add_pair.detach() + soft_add_pair
                add_pair_st_view = add_pair_st.view(B, N, -1)

                # add_probはsource点単位に畳まれるため、Add量の主指標にはしない。
                # shape guard / quant guard用の補助信号としてだけ使う。
                add_prob = add_pair_st_view.sum(dim=2, keepdim=True).transpose(1, 2).clamp(0.0, 1.0)

                pair_priority = torch.sigmoid(pair_logits.clamp(-8.0, 8.0))
                add_priority = pair_priority.max(dim=2, keepdim=True).values.transpose(1, 2)

                selected_base_voxels = selected_base_voxels_long.to(dtype=pts_xyz.dtype)
                selected_offsets_add = selected_offsets_add_long.to(dtype=pts_xyz.dtype)
                # 追加先Voxelを、初期Octree/Voxelのglobal座標系から点座標へ戻す。
                added_pts = self._voxel_centers_from_global_coords(
                    selected_add_voxels_long,
                    voxel_step,
                    voxel_offset,
                    dtype=pts_xyz.dtype,
                )
                added_base = torch.gather(pts_out, 2, idx_expand_xyz)
                added_delta = added_pts - added_base
                selected_add_strength = torch.gather(
                    pair_priority.reshape(B, -1),
                    1,
                    top_pair_idx,
                ).unsqueeze(1)

                selected_hard_add = torch.gather(
                    hard_add_pair,
                    1,
                    top_pair_idx,
                ).unsqueeze(1)

                # Soft追加量をtop-kで選ばれたtarget voxel候補に対応させる。
                # ここには learned_add_ratio / add_head / add_voxel_head への勾配が含まれる。
                selected_soft_add = torch.gather(
                    soft_add_pair,
                    1,
                    top_pair_idx,
                ).unsqueeze(1)

                # ============================================================
                # Addをtarget voxel単位のHard/Soft近似として定義する。
                # forwardではHardなtarget voxel追加、
                # backwardではSoftなtarget voxel追加量を使う。
                # ============================================================
                add_target_soft_add = selected_soft_add.clamp(0.0, 1.0)
                add_target_hard_add = selected_hard_add.to(dtype=add_target_soft_add.dtype)

                add_target_add_st = (
                    add_target_hard_add
                    - add_target_soft_add.detach()
                    + add_target_soft_add
                )

                # 追加先target voxel座標もtarget単位で保持する。
                add_target_voxel_coords = selected_add_voxels_long

                # Phase3: Addは点追加ではなく、target occupied voxel追加候補として記録する。
                voxel_edit_add_target_coords = add_target_voxel_coords.detach()
                voxel_edit_add_target_mask = (
                    add_target_hard_add.detach().squeeze(1).to(dtype=torch.bool)
                )

                # final_wへ入れる追加点の重みはtarget voxel単位STEを使う。
                added_w = add_target_add_st

                pts_out = torch.cat([pts_out, added_pts], dim=2)
                final_w = torch.cat([final_w, added_w], dim=2)
                final_voxel_coords_state = torch.cat(
                    [final_voxel_coords_state, selected_add_voxels_long],
                    dim=2,
                )
                final_voxel_coords = torch.cat([final_voxel_coords, selected_add_voxels_long], dim=2)

                # Soft上の追加率。
                # Addはtarget voxel単位で扱うため、selected target voxelのSoft量を使う。
                add_ratio_soft = add_target_soft_add.sum() / max(float(B * N), 1.0)

                # Hard上の実追加率。
                # 実際に追加したtarget voxel数を使う。
                add_ratio_hard = add_target_hard_add.detach().sum() / max(float(B * N), 1.0)

                max_add_ratio_t = pts_xyz.new_tensor(float(max_add_ratio_value))

                # Addの学習用実行率はtarget voxel単位のsoft量を使う。
                # add_ratio_hardはhard mask由来なので、学習用lossには直接使わない。
                add_ratio_loss = torch.relu(add_ratio_soft - max_add_ratio_t).pow(2)

                # add_amount_head 専用の量一致損失。
                # Hard実行量は教師値としてdetachし、learned_add_ratio側だけに勾配を流す。
                add_amount_supervision_loss = (
                    add_ratio_hard.detach() - learned_add_ratio.mean()
                ).pow(2)

                # Soft追加量と learned_add_ratio の整合性も見る。
                # soft_add_pair は learned_add_ratio から作られるため、この項は補助的に小さく使う。
                add_amount_soft_consistency_loss = (
                    add_ratio_soft.detach() - learned_add_ratio.mean()
                ).pow(2)
                add_shape_guard = self._masked_mean(add_prob * shape_score, selection_mask)
                quant_add_guard = self._masked_mean(
                    add_prob * (quant_score + sparse_score).clamp(0.0, 1.0),
                    selection_mask,
                )
                add_drop_conflict_loss = add_prob.new_zeros(())
                selected_hard_add_det = selected_hard_add.detach()
                added_keep_loss = (
                    (1.0 - selected_add_strength).pow(2) * selected_hard_add_det
                ).sum() / selected_hard_add_det.sum().clamp_min(1.0)
                voxel_norm_safe = voxel_norm.clamp_min(self._numeric_floor(voxel_norm, default=1e-6))
                added_delta_norm = torch.linalg.norm(added_delta, dim=1, keepdim=True) / voxel_norm_safe
                add_offset_reg = (
                    added_delta_norm.pow(2) * add_target_add_st.detach()
                ).sum() / add_target_add_st.detach().sum().clamp_min(1.0)
                add_min_offset_loss = add_prob.new_zeros(())
                add_count_value = int(add_target_hard_add.detach().sum().item())
                hardening_threshold = float(
                    getattr(
                        self.args,
                        "operation_count_drop_threshold",
                        getattr(self.args, "test_drop_threshold", 0.5),
                    )
                )
                add_effective_count_value = int(
                    (add_target_add_st.detach() >= hardening_threshold).sum().item()
                )
                add_target_voxel_count_value = self._unique_voxel_count(
                    add_target_voxel_coords,
                    (add_target_hard_add.detach() >= hardening_threshold),
                )
        # 後段互換用の add_ratio は、古いゼロ初期値ではなく学習用のsoft実行率にする。
        # hard実行率は add_ratio_hard として別keyで保持する。
        add_ratio = add_ratio_soft
        if actual_oracle_enabled and actual_oracle_has_add:
            leaf_diag_for_oracle_add = structure.get("leaf_pattern_diag", {}) if isinstance(structure, dict) else {}
            oracle_add_slots = self._fit_leaf_pattern_long_map(
                leaf_diag_for_oracle_add.get("best_add_child_slot", None),
                batch_size=B,
                point_count=N,
                device=pts_xyz.device,
            )
            oracle_add_source_mask = leaf_add_op_mask.squeeze(1).to(device=pts_xyz.device, dtype=torch.bool)
            oracle_add_source_mask = oracle_add_source_mask & oracle_add_slots.ge(0) & oracle_add_slots.le(7)
            if bool(oracle_add_source_mask.detach().any().item()):
                max_oracle_add = max(
                    int(getattr(self.args, "sparsepcgc_actual_oracle_max_selected_voxels", 4)),
                    1,
                )
                oracle_target_lists = []
                oracle_mask_lists = []
                for b in range(B):
                    source_idx_b = oracle_add_source_mask[b].nonzero(as_tuple=False).reshape(-1)
                    if source_idx_b.numel() <= 0:
                        oracle_target_lists.append(voxel_coords.new_empty((3, 0)))
                        oracle_mask_lists.append(torch.zeros((0,), device=pts_xyz.device, dtype=torch.bool))
                        continue
                    source_idx_b = source_idx_b[:max_oracle_add]
                    source_voxels_b = voxel_coords[b].index_select(1, source_idx_b).transpose(0, 1).contiguous()
                    slots_b = oracle_add_slots[b].index_select(0, source_idx_b).to(dtype=torch.long)
                    slot_bits_b = torch.stack(
                        [
                            slots_b & 1,
                            (slots_b >> 1) & 1,
                            (slots_b >> 2) & 1,
                        ],
                        dim=1,
                    ).to(device=voxel_coords.device, dtype=torch.long)
                    target_b = torch.div(source_voxels_b, 2, rounding_mode="floor") * 2 + slot_bits_b
                    target_b = torch.unique(target_b, dim=0, sorted=True)
                    if int(target_b.shape[0]) > max_oracle_add:
                        target_b = target_b[:max_oracle_add]
                    oracle_target_lists.append(target_b.transpose(0, 1).contiguous())
                    oracle_mask_lists.append(torch.ones((target_b.shape[0],), device=pts_xyz.device, dtype=torch.bool))

                max_targets = max((int(item.shape[1]) for item in oracle_target_lists), default=0)
                if max_targets > 0:
                    oracle_coords = voxel_coords.new_zeros((B, 3, max_targets))
                    oracle_mask = torch.zeros((B, max_targets), device=pts_xyz.device, dtype=torch.bool)
                    for b, coords_b in enumerate(oracle_target_lists):
                        count_b = int(coords_b.shape[1])
                        if count_b <= 0:
                            continue
                        oracle_coords[b, :, :count_b] = coords_b
                        oracle_mask[b, :count_b] = True
                    if voxel_edit_add_target_coords.shape[2] > 0:
                        voxel_edit_add_target_coords = torch.cat(
                            [voxel_edit_add_target_coords, oracle_coords.detach()],
                            dim=2,
                        )
                        voxel_edit_add_target_mask = torch.cat(
                            [voxel_edit_add_target_mask, oracle_mask.detach()],
                            dim=1,
                        )
                    else:
                        voxel_edit_add_target_coords = oracle_coords.detach()
                        voxel_edit_add_target_mask = oracle_mask.detach()
                    oracle_add_count = int(oracle_mask.detach().sum().item())
                    add_count_value = max(add_count_value, oracle_add_count)
                    add_effective_count_value = max(add_effective_count_value, oracle_add_count)
                    add_target_voxel_count_value = max(add_target_voxel_count_value, oracle_add_count)
                    add_ratio_hard = pts_xyz.new_tensor(
                        float(oracle_add_count) / max(float(B * N), 1.0)
                    )
                    add_ratio_soft = torch.maximum(add_ratio_soft, add_ratio_hard.detach())
                    add_ratio = add_ratio_soft
        if (actual_oracle_has_add or actual_oracle_has_bad_add) and max_add_ratio_value > 0.0:
            add_oracle_target_ratio = (
                add_ratio_hard.detach()
                if actual_oracle_has_add
                else torch.zeros_like(add_ratio_hard)
            )
            actual_oracle_add_amount_loss = self._actual_oracle_amount_bce_loss(
                raw_learned_add_ratio,
                add_oracle_target_ratio,
                max_add_ratio_value,
            )
            add_amount_logit_for_oracle = self._operation_amount_logit(
                actuator_features,
                self.add_amount_head,
            ).mean()
            add_amount_target_logit_for_oracle = self._target_ratio_logit(
                add_oracle_target_ratio,
                max_add_ratio_value,
                raw_learned_add_ratio,
            ).detach()
            actual_oracle_add_amount_logit_loss = (
                add_amount_logit_for_oracle - add_amount_target_logit_for_oracle
            ).pow(2)
            actual_oracle_add_amount_loss = (
                actual_oracle_add_amount_loss
                + float(getattr(self.args, "sparsepcgc_actual_oracle_amount_logit_weight", 0.25))
                * actual_oracle_add_amount_logit_loss
            )
        if (actual_oracle_has_move or actual_oracle_has_bad_move) and raw_max_move_ratio > 0.0:
            move_oracle_target_ratio = (
                leaf_move_op_mask.to(dtype=pts_xyz.dtype).mean()
                if actual_oracle_has_move
                else pts_xyz.new_zeros(())
            ).clamp(0.0, raw_max_move_ratio)
            actual_oracle_move_amount_loss = self._actual_oracle_amount_bce_loss(
                raw_learned_move_ratio,
                move_oracle_target_ratio,
                raw_max_move_ratio,
            )
            move_amount_logit_for_oracle = self._operation_amount_logit(
                actuator_features,
                self.move_amount_head,
            ).mean()
            move_amount_target_logit_for_oracle = self._target_ratio_logit(
                move_oracle_target_ratio,
                raw_max_move_ratio,
                raw_learned_move_ratio,
            ).detach()
            actual_oracle_move_amount_logit_loss = (
                move_amount_logit_for_oracle - move_amount_target_logit_for_oracle
            ).pow(2)
            actual_oracle_move_amount_loss = (
                actual_oracle_move_amount_loss
                + float(getattr(self.args, "sparsepcgc_actual_oracle_amount_logit_weight", 0.25))
                * actual_oracle_move_amount_logit_loss
            )
        if timing_enabled:
            _mark_runtime("add")

        hardening_threshold = float(
            getattr(self.args, "operation_count_drop_threshold", getattr(self.args, "test_drop_threshold", 0.5))
        )
        hard_keep_mask = final_w.detach() >= hardening_threshold
        point_aligned_after_occupied_voxels = self._unique_voxel_count(final_voxel_coords, hard_keep_mask)
        # Phase3: Prune/Add/Moveをoccupied voxel集合へ反映した最終Voxel状態を作る。
        # 既存のpts_out/final_wは変更しない。
        voxel_edit_debug_list = []
        voxel_edit_coords_list = []
        voxel_edit_weights_list = []

        if voxel_edit_state_enabled:
            for b in range(B):
                coords_b, weights_b, debug_b = self._build_voxel_edit_state_single(
                    voxel_coords_b=voxel_coords[b],
                    hard_drop_mask_b=voxel_edit_drop_mask[b],
                    add_target_coords_b=voxel_edit_add_target_coords[b] if voxel_edit_add_target_coords.shape[2] > 0 else None,
                    add_target_mask_b=voxel_edit_add_target_mask[b] if voxel_edit_add_target_mask.numel() > 0 else None,
                    move_source_mask_b=voxel_edit_move_source_mask[b],
                    move_target_coords_b=move_target_voxel_coords[b],
                    move_valid_mask_b=voxel_edit_move_valid_mask[b],
                )
                voxel_edit_coords_list.append(coords_b)
                voxel_edit_weights_list.append(weights_b.to(device=pts_xyz.device, dtype=pts_xyz.dtype))
                voxel_edit_debug_list.append(debug_b)

            voxel_edit_final_coords, voxel_edit_final_weights, voxel_edit_valid_mask = self._pad_voxel_edit_state(
                voxel_edit_coords_list,
                voxel_edit_weights_list,
                device=pts_xyz.device,
                dtype=pts_xyz.dtype,
            )
        else:
            voxel_edit_final_coords = final_voxel_coords.detach().to(dtype=torch.long)
            voxel_edit_final_weights = final_w.detach()
            voxel_edit_valid_mask = torch.ones(
                (B, int(voxel_edit_final_coords.shape[-1])),
                device=pts_xyz.device,
                dtype=torch.bool,
            )
            voxel_edit_debug_list = [
                {
                    "initial_count": int(voxel_edit_final_coords.shape[-1]),
                    "drop_count": 0,
                    "add_count": 0,
                    "move_count": 0,
                    "same_voxel_move_rejected": 0,
                    "existing_target_rejected": 0,
                    "duplicate_target_rejected": 0,
                    "final_count": int(voxel_edit_final_coords.shape[-1]),
                }
                for _ in range(B)
            ]

        actual_oracle_override_move_count_value = 0
        actual_oracle_override_drop_count_value = 0
        actual_oracle_override_subtree_prune_count_value = 0
        leaf_diag_for_override = structure.get("leaf_pattern_diag", {}) if isinstance(structure, dict) else {}
        actual_oracle_edit_record_bits_value = 0.0
        actual_oracle_raw_percent_value = 0.0
        if isinstance(leaf_diag_for_override, dict):
            actual_oracle_edit_record_bits_value = max(
                float(leaf_diag_for_override.get("actual_oracle_edit_record_bits", 0.0) or 0.0),
                0.0,
            )
            actual_oracle_raw_percent_value = float(
                leaf_diag_for_override.get("actual_oracle_raw_percent", 0.0) or 0.0
            )
        override_final_voxel_coords = (
            leaf_diag_for_override.get("actual_oracle_override_final_voxel_coords", None)
            if isinstance(leaf_diag_for_override, dict)
            else None
        )
        if actual_oracle_enabled and torch.is_tensor(override_final_voxel_coords):
            override_coords = override_final_voxel_coords.detach().to(device=pts_xyz.device, dtype=torch.long)
            if override_coords.ndim == 2:
                override_coords = (
                    override_coords.transpose(0, 1).contiguous().unsqueeze(0)
                    if override_coords.shape[-1] == 3
                    else override_coords.unsqueeze(0)
                )
            elif override_coords.ndim == 3 and override_coords.shape[1] != 3 and override_coords.shape[-1] == 3:
                override_coords = override_coords.permute(0, 2, 1).contiguous()
            if override_coords.ndim == 3 and override_coords.shape[1] == 3 and override_coords.shape[-1] > 0:
                if override_coords.shape[0] == 1 and B > 1:
                    override_coords = override_coords.expand(B, -1, -1).contiguous()
                if override_coords.shape[0] == B:
                    actual_oracle_override_move_count_value = max(
                        int(leaf_diag_for_override.get("actual_oracle_override_move_count", 0) or 0),
                        0,
                    )
                    actual_oracle_override_drop_count_value = max(
                        int(leaf_diag_for_override.get("actual_oracle_override_drop_count", 0) or 0),
                        0,
                    )
                    actual_oracle_override_subtree_prune_count_value = max(
                        int(
                            leaf_diag_for_override.get(
                                "actual_oracle_override_subtree_prune_count",
                                0,
                            )
                            or 0
                        ),
                        0,
                    )
                    voxel_edit_final_coords = override_coords
                    voxel_edit_final_weights = pts_xyz.new_ones((B, 1, int(override_coords.shape[-1])))
                    voxel_edit_valid_mask = torch.ones(
                        (B, int(override_coords.shape[-1])),
                        device=pts_xyz.device,
                        dtype=torch.bool,
                    )
                    voxel_edit_debug_list = [
                        {
                            "initial_count": int(voxel_coords.shape[-1]),
                            "drop_count": int(actual_oracle_override_drop_count_value),
                            "add_count": 0,
                            "move_count": int(actual_oracle_override_move_count_value),
                            "same_voxel_move_rejected": 0,
                            "existing_target_rejected": 0,
                            "duplicate_target_rejected": 0,
                            "final_count": int(override_coords.shape[-1]),
                            "subtree_prune_count": int(
                                actual_oracle_override_subtree_prune_count_value
                            ),
                        }
                        for _ in range(B)
                    ]

        voxel_edit_initial_count_value = int(sum(item.get("initial_count", 0) for item in voxel_edit_debug_list))
        voxel_edit_final_count_value = int(sum(item.get("final_count", 0) for item in voxel_edit_debug_list))
        voxel_edit_drop_count_value = int(sum(item.get("drop_count", 0) for item in voxel_edit_debug_list))
        voxel_edit_add_count_value = int(sum(item.get("add_count", 0) for item in voxel_edit_debug_list))
        voxel_edit_move_count_value = int(sum(item.get("move_count", 0) for item in voxel_edit_debug_list))
        voxel_edit_same_voxel_move_rejected_value = int(sum(item.get("same_voxel_move_rejected", 0) for item in voxel_edit_debug_list))
        voxel_edit_existing_target_rejected_value = int(sum(item.get("existing_target_rejected", 0) for item in voxel_edit_debug_list))
        voxel_edit_duplicate_target_rejected_value = int(sum(item.get("duplicate_target_rejected", 0) for item in voxel_edit_debug_list))

        voxel_edit_child_slot_rejected_value = int(
            (voxel_edit_move_source_mask & (~voxel_edit_move_child_slot_mask)).detach().sum().item()
        )
        voxel_edit_empty_target_rejected_value = int(
            (voxel_edit_move_source_mask & (~voxel_edit_move_target_empty_mask)).detach().sum().item()
        )

        delete_target_voxel_count_value = self._unique_voxel_count_from_cache(voxel_cache, hard_drop_mask)
        delete_removed_point_count_value = int(hard_drop_mask.detach().sum().item())
        delete_emptied_voxel_count_value = self._selected_voxels_absent_count(
            voxel_coords,
            hard_drop_mask,
            final_voxel_coords,
            hard_keep_mask,
        )
        move_source_voxel_count_value = self._unique_voxel_count_from_cache(voxel_cache, hard_move_mask)
        move_target_voxel_count_value = self._unique_voxel_count(move_target_voxel_coords, hard_move_mask)
        move_source_emptied_voxel_count_value = self._selected_voxels_absent_count(
            voxel_coords,
            hard_move_mask,
            final_voxel_coords,
            hard_keep_mask,
        )
        move_target_new_voxel_count_value = self._selected_voxels_absent_count(
            move_target_voxel_coords,
            hard_move_mask,
            voxel_coords,
            selection_bool,
        )
        move_source_not_emptied_count_value = max(
            int(move_source_voxel_count_value) - int(move_source_emptied_voxel_count_value),
            0,
        )
        same_voxel_adjust_count_value = int(same_voxel_move_mask.detach().sum().item())
        moved_different_voxel_count_value = int(moved_different_voxel_mask.detach().sum().item())
        if actual_oracle_override_move_count_value > 0:
            move_source_voxel_count_value = max(
                int(move_source_voxel_count_value),
                int(actual_oracle_override_move_count_value),
            )
            move_target_voxel_count_value = max(
                int(move_target_voxel_count_value),
                int(actual_oracle_override_move_count_value),
            )
            move_source_emptied_voxel_count_value = max(
                int(move_source_emptied_voxel_count_value),
                int(actual_oracle_override_move_count_value),
            )
            move_target_new_voxel_count_value = max(
                int(move_target_new_voxel_count_value),
                int(actual_oracle_override_move_count_value),
            )
            moved_different_voxel_count_value = max(
                int(moved_different_voxel_count_value),
                int(actual_oracle_override_move_count_value),
            )
        hard_drop_count_value = int(hard_drop.detach().sum().item())
        hard_move_count_value = int(hard_move.detach().sum().item())
        raw_hard_move_count_value = int(raw_hard_move_bool.detach().sum().item())
        adjusted_point_rate_value = float(hard_move_count_value) / max(float(B * N), 1.0)
        sparsepcgc_source_unique_voxel_count_value = self._unique_voxel_count(
            source_sparsepcgc_coords,
            hard_move_mask,
        )
        sparsepcgc_target_unique_voxel_count_value = self._unique_voxel_count(
            target_sparsepcgc_coords,
            hard_move_mask,
        )
        sparsepcgc_target_duplicate_voxel_count_value = max(
            int(hard_move_count_value) - int(sparsepcgc_target_unique_voxel_count_value),
            0,
        )
        sparsepcgc_target_duplicate_rate_value = (
            float(sparsepcgc_target_duplicate_voxel_count_value) / max(float(hard_move_count_value), 1.0)
        )
        final_move_bool = hard_move_mask.squeeze(1).detach().to(dtype=torch.bool)
        target_existing_occupied_count_value = int(
            (final_move_bool & target_existing_occupied_mask).detach().sum().item()
        )
        target_empty_voxel_count_value = max(
            int(hard_move_count_value) - int(target_existing_occupied_count_value),
            0,
        )
        target_existing_occupied_rate_value = (
            float(target_existing_occupied_count_value) / max(float(hard_move_count_value), 1.0)
        )
        target_empty_voxel_rate_value = (
            float(target_empty_voxel_count_value) / max(float(hard_move_count_value), 1.0)
        )
        empty_guard_rejected_count_value = int(empty_guard_reject_bool.detach().sum().item())
        target_duplicate_guard_rejected_count_value = int(duplicate_guard_reject_bool.detach().sum().item())
        sparsepcgc_guard_rejected_count_value = int(guard_rejected_bool.detach().sum().item())
        preserve_hard = (~hard_drop_mask) & (~hard_move_mask)
        preserve_ratio = preserve_hard.to(dtype=pts_xyz.dtype).mean()
        after_occupied_voxels = voxel_edit_final_count_value if voxel_edit_state_enabled else point_aligned_after_occupied_voxels

        delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
        voxel_norm_safe = voxel_norm.clamp_min(self._numeric_floor(voxel_norm, default=1e-6))
        normalized_delta = delta_norm / voxel_norm_safe
        edit_reg = self._masked_mean(normalized_delta.pow(2) * hard_move, selection_mask)
        moved_points = hard_move.sum().clamp_min(1.0)
        moved_delta_mean = (delta_norm * hard_move).sum() / moved_points

        repair_gate_mean = self._masked_mean(repair_gate, selection_mask)

        # targetなしAmount学習では、repair対象率もtargetへ寄せない。
        # max_repair_ratioを超えた場合だけ罰する。
        ratio_loss = torch.relu(
            repair_gate_mean - repair_gate_mean.new_tensor(float(max_repair_ratio))
        ).pow(2)

        shape_guard = self._masked_mean(repair_gate * shape_score, selection_mask)
        drop_ratio = drop_ratio_soft
        if threshold_cap_mode or bool(getattr(self.args, "repair_learn_operation_amounts", True)):
            # 操作量を学習する場合は固定削除率へ引っ張らず、静的上限だけを守る。
            drop_ratio_loss = torch.relu(drop_ratio - drop_ratio.new_tensor(float(max_drop_ratio))) ** 2
        else:
            drop_ratio_loss = (drop_ratio - target_drop_ratio) ** 2
        drop_cap_loss = torch.relu(drop_ratio - max_drop_ratio) ** 2
        # shape guardは0〜1に収めたSoft削除強度で計算する。
        # STE本体のsoft_drop_probは量勾配用、guardは過大値を避けるためsoft_drop_prob_for_guardを使う。
        drop_shape_guard = self._masked_mean(soft_drop_prob_for_guard * shape_score, selection_mask)
        # Prune Where勾配をdrop_headへ戻すため、Prune soft proxyはdrop_prob_proxyを主に使う。
        # target量ではなく、どの点を削るとrate/structureが変わるかを学習させる。
        prune_drop_signal = (
            0.10 * soft_drop_prob_for_guard.detach()
            + 0.90 * drop_prob_proxy
        ).clamp(0.0, 1.0)
        prune_keep_signal = (1.0 - prune_drop_signal).clamp(0.0, 1.0)
        leaf_delete_gain = (
            voxel_point_counts <= float(max(delete_max_points, 1))
        ).to(device=pts_xyz.device, dtype=pts_xyz.dtype)
        prune_geom_importance = (
            0.50
            + 2.00 * shape_score
            + 0.75 * preserve
            + 0.25 * (1.0 - local_outlier_score.clamp(0.0, 1.0))
        ).detach().clamp(0.0, 4.0)
        prune_rate_importance = (
            0.50
            + 1.00 * p_comp
            + 0.75 * p_chain
            + 0.50 * p_sibling
            + 0.50 * node_score
            + 0.75 * single_score
            + 0.50 * quant_score
            + 0.50 * sparse_score
            + 0.25 * lowprob_score
            + 0.50 * local_outlier_score
            + 0.75 * leaf_delete_gain
        ).detach().clamp(0.0, 6.0)
        prune_node_importance = (0.50 + 1.50 * node_score + 0.75 * leaf_delete_gain).detach().clamp(0.0, 4.0)
        prune_single_importance = (0.50 + 2.00 * single_score + 0.75 * p_chain + 0.50 * p_sibling).detach().clamp(0.0, 4.0)
        prune_bit_importance = (
            prune_rate_importance + 0.50 * prune_node_importance + 0.25 * prune_single_importance
        ).detach().clamp(0.0, 8.0)
        prune_soft_geom = self._masked_mean(prune_drop_signal * prune_geom_importance, selection_mask)
        prune_soft_rate = self._masked_mean(prune_keep_signal * prune_rate_importance, selection_mask)
        prune_soft_node = self._masked_mean(prune_keep_signal * prune_node_importance, selection_mask)
        prune_soft_single = self._masked_mean(prune_keep_signal * prune_single_importance, selection_mask)
        prune_soft_bit = self._masked_mean(prune_keep_signal * prune_bit_importance, selection_mask)
        drop_prob_direct_mean = self._masked_mean(drop_prob_direct, selection_mask)
        drop_prob_proxy_mean = self._masked_mean(drop_prob_proxy, selection_mask)
        amount_target_mode = str(
            getattr(self.args, "repair_amount_target_mode", "none")
        ).strip().lower()
        if amount_target_mode == "target":
            drop_direct_target = drop_prob_proxy_mean.new_tensor(float(target_drop_ratio))
            drop_direct_target_loss = (drop_prob_proxy_mean - drop_direct_target).pow(2)
        else:
            drop_direct_target_loss = drop_prob_proxy_mean.new_zeros(())
        drop_entropy_point = -(
            drop_prob_proxy.clamp(1e-6, 1.0 - 1e-6) * drop_prob_proxy.clamp(1e-6, 1.0).log()
            + (1.0 - drop_prob_proxy).clamp(1e-6, 1.0) * (1.0 - drop_prob_proxy).clamp(1e-6, 1.0).log()
        )
        drop_entropy = self._masked_mean(drop_entropy_point, selection_mask)
        soft_drop_mass = drop_prob_proxy.sum()
        # Local guardは0〜1に収めたSoft削除強度とMove強度を使う。
        local_edit_guard = self._masked_mean(
            (soft_drop_prob_for_guard + move_mask_for_guard).clamp(0.0, 1.0) * shape_score,
            selection_mask,
        )
        # ============================================================
        # targetなしAmount学習
        # ============================================================
        # Amount headを固定targetへ寄せると、L_comではなくtarget方向へ学習が進む。
        # そのため、repair_amount_target_mode='none' では
        # direct/logit target損失を0にする。
        #
        # Amountの学習は以下で行う。
        #   1. L_comから実操作へ戻る下流勾配
        #   2. Soft/Hard実行量と learned_*_ratio の整合性
        #   3. 上限cap超過ペナルティ
        # ============================================================
        if amount_target_mode == "target":
            drop_amount_target = learned_drop_ratio.new_tensor(
                float(getattr(self.args, "target_drop_ratio", 0.0)) if prune_enabled else 0.0
            ).clamp(0.0, float(max_drop_ratio))

            move_amount_target = learned_move_ratio.new_tensor(
                float(getattr(self.args, "target_move_ratio", 0.0)) if disp_enabled else 0.0
            ).clamp(0.0, float(max_move_ratio))

            add_amount_target = learned_add_ratio.new_tensor(
                float(getattr(self.args, "target_add_ratio", 0.0)) if add_enabled else 0.0
            ).clamp(0.0, float(max_add_ratio_value))

            drop_amount_logit = self._operation_amount_logit(
                actuator_features,
                self.drop_amount_head,
            ).mean()

            add_amount_logit = self._operation_amount_logit(
                actuator_features,
                self.add_amount_head,
            ).mean()

            move_amount_logit = self._operation_amount_logit(
                actuator_features,
                self.move_amount_head,
            ).mean()

            drop_amount_target_logit = self._target_ratio_logit(
                drop_amount_target.detach(),
                max_drop_ratio,
                learned_drop_ratio,
            )

            add_amount_target_logit = self._target_ratio_logit(
                add_amount_target.detach(),
                max_add_ratio_value,
                learned_add_ratio,
            )

            move_amount_target_logit = self._target_ratio_logit(
                move_amount_target.detach(),
                max_move_ratio,
                learned_move_ratio,
            )

            operation_amount_logit_loss = (
                (drop_amount_logit - drop_amount_target_logit.detach()).pow(2)
                + (add_amount_logit - add_amount_target_logit.detach()).pow(2)
                + (move_amount_logit - move_amount_target_logit.detach()).pow(2)
            )

            operation_amount_direct_loss = (
                (learned_drop_ratio.mean() - drop_amount_target).pow(2)
                + (learned_move_ratio.mean() - move_amount_target).pow(2)
                + (learned_add_ratio.mean() - add_amount_target).pow(2)
            )
        else:
            # targetなしAmount学習では、固定targetへ寄せる損失は使わない。
            # ただし後段のdebug_tensors / returnで参照されるため、
            # target系Tensorは0として必ず定義しておく。
            drop_amount_target = learned_drop_ratio.new_zeros(())
            move_amount_target = learned_move_ratio.new_zeros(())
            add_amount_target = learned_add_ratio.new_zeros(())

            drop_amount_target_logit = learned_drop_ratio.new_zeros(())
            move_amount_target_logit = learned_move_ratio.new_zeros(())
            add_amount_target_logit = learned_add_ratio.new_zeros(())

            drop_amount_logit = learned_drop_ratio.new_zeros(())
            move_amount_logit = learned_move_ratio.new_zeros(())
            add_amount_logit = learned_add_ratio.new_zeros(())

            operation_amount_logit_loss = learned_drop_ratio.new_zeros(())
            operation_amount_direct_loss = learned_drop_ratio.new_zeros(())


        # Addは古い add_ratio ではなく、target voxel単位のsoft実行率を使う。
        # hard-soft consistency は hard側をdetachし、soft側だけを学習対象にする。
        add_hard_soft_consistency_loss = (
            add_ratio_soft - add_ratio_hard.detach()
        ).abs()

        operation_amount_consistency_loss = (
            (drop_ratio - learned_drop_ratio.mean()).pow(2)
            + (move_ratio_soft - learned_move_ratio.mean()).pow(2)
            + (add_ratio_soft - learned_add_ratio.mean()).pow(2)
            + add_hard_soft_consistency_loss
        )

        if torch.is_tensor(operation_gate_prob) and operation_gate_prob.ndim >= 3:
            operation_ratio_vec = operation_gate_prob.float().mean(dim=(0, 2)).clamp_min(0.0)
        else:
            operation_ratio_vec = torch.stack(
                [drop_ratio, move_ratio_soft, add_ratio_soft]
            ).clamp_min(0.0)
        operation_ratio_prob = operation_ratio_vec / operation_ratio_vec.sum().clamp_min(1e-6)
        operation_entropy = -(operation_ratio_prob * operation_ratio_prob.clamp_min(1e-6).log()).sum()
        operation_entropy_weight_raw = max(
            float(getattr(self.args, "repair_operation_entropy_weight", 0.0)),
            0.0,
        )
        operation_entropy_warmup_steps = max(
            int(getattr(self.args, "repair_operation_entropy_warmup_steps", 0)),
            0,
        )
        if operation_entropy_weight_raw > 0.0 and operation_entropy_warmup_steps > 0:
            train_step_for_entropy = max(int(getattr(self.args, "_global_train_step", 0)), 0)
            operation_entropy_phase = max(
                1.0 - float(train_step_for_entropy) / float(operation_entropy_warmup_steps),
                0.0,
            )
        elif operation_entropy_weight_raw > 0.0:
            operation_entropy_phase = 1.0
        else:
            operation_entropy_phase = 0.0
        operation_entropy_weight_effective = operation_entropy_weight_raw * operation_entropy_phase
        operation_entropy_loss = -operation_entropy.new_tensor(
            float(operation_entropy_weight_effective)
        ) * operation_entropy
        soft_activity_loss = (
            drop_prob.mean()
            + learned_drop_prob
            + drop_prob_proxy.mean()
            + move_score.mean()
            + add_prob.mean()
            + learned_drop_ratio.mean()
            + learned_move_ratio.mean()
            + learned_add_ratio.mean()
        )
        exploration_noise = max(float(drop_score_noise), float(move_score_noise), float(add_score_noise))
        operation_temperature = float(getattr(self.args, "repair_priority_gate_tau", getattr(self.args, "repair_policy_temperature", 1.0)))
        # ============================================================
        # Where headへ直接かけるActuator補助損失
        # ============================================================
        # 目的
        # ・圧縮損失ではなく、L_actuator側からWhere headへ勾配を流す
        # ・Amount headではなく、drop_head / add_head / move source score側を更新する
        # ・hard選択結果は教師としてdetachし、soft score側だけを学習させる
        # ============================================================

        eps = torch.finfo(pts_xyz.dtype).eps

        # ------------------------------------------------------------
        # Prune Where補助損失
        # ------------------------------------------------------------
        # hard_dropはdetachして教師にする。
        # soft_drop_prob_for_guard側だけに勾配を流す。
        # これにより drop_head にActuator損失由来の勾配が出る。
        drop_where_actuator_loss = pts_xyz.new_zeros(())
        if prune_enabled:
            # ========================================================
            # Prune Where補助損失は drop_prob_proxy に直接かける。
            # soft_drop_prob_for_ste は hard/guard/予算/clamp を経由しており、
            # drop_head への勾配確認用としては遠すぎる。
            # ========================================================
            drop_where_pred = torch.nan_to_num(
                drop_prob_proxy,
                nan=0.5,
                posinf=1.0,
                neginf=0.0,
            ).clamp(eps, 1.0 - eps)

            drop_where_target = torch.nan_to_num(
                hard_drop.detach(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

            with torch.cuda.amp.autocast(enabled=False):
                drop_where_loss_raw = torch.nn.functional.binary_cross_entropy(
                    drop_where_pred.float(),
                    drop_where_target.float(),
                    reduction="none",
                )

            # selection_mask がある場合だけ対象点に制限する。
            # ただし delete_candidate_weight では重み付けしない。
            # それを使うと、候補枯渇時にまた drop_head 勾配が0になる。
            if selection_mask is not None:
                where_loss_weight = selection_mask.to(
                    device=drop_where_loss_raw.device,
                    dtype=drop_where_loss_raw.dtype,
                )
                if where_loss_weight.ndim == 2:
                    where_loss_weight = where_loss_weight.unsqueeze(1)
                denom = where_loss_weight.sum().clamp_min(1.0)
                drop_where_actuator_loss = (drop_where_loss_raw * where_loss_weight.float()).sum() / denom
            else:
                drop_where_actuator_loss = drop_where_loss_raw.mean()

        # ------------------------------------------------------------
        # Move Where補助損失
        # ------------------------------------------------------------
        # hard_moveはdetachして教師にする。
        # soft_move_score_for_guard側だけに勾配を流す。
        # これによりMove source選択側にActuator損失由来の勾配が出る。
        move_where_actuator_loss = pts_xyz.new_zeros(())
        if disp_enabled:
            move_where_pred = torch.nan_to_num(
                soft_move_score_for_guard,
                nan=0.5,
                posinf=1.0,
                neginf=0.0,
            ).clamp(eps, 1.0 - eps)

            move_where_target = torch.nan_to_num(
                hard_move.detach(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            with torch.cuda.amp.autocast(enabled=False):
                move_where_actuator_loss = torch.nn.functional.binary_cross_entropy(
                    move_where_pred.float(),
                    move_where_target.float(),
                )

        # ------------------------------------------------------------
        # Add Where補助損失
        # ------------------------------------------------------------
        # Addはsource点単位ではなく、target voxel pair単位で選んでいる。
        # そのため soft_add_pair / hard_add_pair を使う。
        # これにより add_head と add_voxel_head にActuator損失由来の勾配が出る。
        add_where_actuator_loss = pts_xyz.new_zeros(())
        if add_enabled and soft_add_pair.numel() > 0 and hard_add_pair.numel() > 0:
            add_where_pred = torch.nan_to_num(
                soft_add_pair,
                nan=0.5,
                posinf=1.0,
                neginf=0.0,
            ).clamp(eps, 1.0 - eps)

            add_where_target = torch.nan_to_num(
                hard_add_pair.detach(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            with torch.cuda.amp.autocast(enabled=False):
                add_where_actuator_loss = torch.nn.functional.binary_cross_entropy(
                    add_where_pred.float(),
                    add_where_target.float(),
                )

        # ------------------------------------------------------------
        # Actual SparsePCGC oracle candidate supervision
        # ------------------------------------------------------------
        # Forward編集は「actualで改善確認済み」の候補だけに限定する。
        # ただし、実Codecで悪化した候補もwhere/gateの負例として学習へ入れる。
        # これにより悪化編集を実際に出力してL_comを壊さず、
        # Actuatorは「選ぶべき候補」と「避けるべき候補」の両方を覚えられる。
        operation_gate_oracle_loss = pts_xyz.new_zeros(())
        actual_oracle_drop_bad_count_value = 0
        actual_oracle_add_bad_count_value = 0
        actual_oracle_move_bad_count_value = 0
        actual_oracle_candidate_where_loss = pts_xyz.new_zeros(())
        actual_oracle_direction_supervision_loss = pts_xyz.new_zeros(())
        if actual_oracle_enabled:
            oracle_candidate_weight = max(
                float(getattr(self.args, "sparsepcgc_actual_oracle_candidate_where_weight", 1.0)),
                0.0,
            )

            def _oracle_where_bce(pred, good_mask, bad_mask, bad_score):
                good_mask = good_mask.to(device=pred.device, dtype=torch.bool)
                bad_mask = bad_mask.to(device=pred.device, dtype=torch.bool) & (~good_mask)
                valid = good_mask | bad_mask
                if not bool(valid.detach().any().item()):
                    return pred.new_zeros(()), 0
                target = good_mask.to(device=pred.device, dtype=pred.dtype)
                weight = torch.ones_like(pred, dtype=pred.dtype)
                if torch.is_tensor(bad_score):
                    weight = weight + bad_mask.to(dtype=pred.dtype) * bad_score.to(
                        device=pred.device,
                        dtype=pred.dtype,
                    ).clamp(0.0, 5.0)
                pred_safe = torch.nan_to_num(
                    pred,
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(eps, 1.0 - eps)
                with torch.cuda.amp.autocast(enabled=False):
                    raw = torch.nn.functional.binary_cross_entropy(
                        pred_safe.float(),
                        target.float(),
                        reduction="none",
                    )
                valid_f = valid.to(device=raw.device, dtype=raw.dtype)
                denom = (valid_f * weight.float()).sum().clamp_min(1.0)
                loss_value = (raw * valid_f * weight.float()).sum() / denom
                return loss_value.to(dtype=pred.dtype), int(bad_mask.detach().sum().item())

            def _oracle_where_bce_logits(logit, good_mask, bad_mask, bad_score):
                good_mask = good_mask.to(device=logit.device, dtype=torch.bool)
                bad_mask = bad_mask.to(device=logit.device, dtype=torch.bool) & (~good_mask)
                valid = good_mask | bad_mask
                if not bool(valid.detach().any().item()):
                    return logit.new_zeros(()), 0
                target = good_mask.to(device=logit.device, dtype=logit.dtype)
                weight = torch.ones_like(logit, dtype=logit.dtype)
                if torch.is_tensor(bad_score):
                    weight = weight + bad_mask.to(dtype=logit.dtype) * bad_score.to(
                        device=logit.device,
                        dtype=logit.dtype,
                    ).clamp(0.0, 5.0)
                safe_logit = torch.nan_to_num(
                    logit,
                    nan=0.0,
                    posinf=30.0,
                    neginf=-30.0,
                )
                logit_clip = max(
                    float(getattr(self.args, "sparsepcgc_actual_oracle_candidate_logit_clip", 20.0)),
                    1.0,
                )
                clipped_logit = safe_logit.clamp(-float(logit_clip), float(logit_clip))
                safe_logit = safe_logit + (clipped_logit - safe_logit).detach()
                with torch.cuda.amp.autocast(enabled=False):
                    raw = torch.nn.functional.binary_cross_entropy_with_logits(
                        safe_logit.float(),
                        target.float(),
                        reduction="none",
                    )
                valid_f = valid.to(device=raw.device, dtype=raw.dtype)
                denom = (valid_f * weight.float()).sum().clamp_min(1.0)
                loss_value = (raw * valid_f * weight.float()).sum() / denom
                return loss_value.to(dtype=logit.dtype), int(bad_mask.detach().sum().item())

            def _oracle_direction_loss(logits, good_mask, good_index, bad_mask, bad_index):
                class_count = int(logits.shape[1])
                good_index = good_index.to(device=logits.device, dtype=torch.long)
                bad_mask = bad_mask.to(device=logits.device, dtype=torch.bool)
                good_valid = (
                    good_mask.to(device=logits.device, dtype=torch.bool)
                    & (good_index >= 0)
                    & (good_index < class_count)
                )
                bad_valid = bad_mask & (~good_valid)
                if not bool((good_valid | bad_valid).detach().any().item()):
                    return logits.new_zeros(())
                safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                good_logit = safe_logits.gather(1, good_index.clamp(0, class_count - 1))
                # A rejected direction suppresses the currently preferred direction,
                # leaving the remaining empty-neighbor directions available to explore.
                bad_logit = safe_logits.amax(dim=1, keepdim=True)
                with torch.cuda.amp.autocast(enabled=False):
                    good_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                        good_logit.float(),
                        torch.ones_like(good_logit, dtype=torch.float32),
                        reduction="none",
                    )
                    bad_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                        bad_logit.float(),
                        torch.zeros_like(bad_logit, dtype=torch.float32),
                        reduction="none",
                    )
                valid = good_valid | bad_valid
                raw = good_raw * good_valid.float() + bad_raw * bad_valid.float()
                return (raw.sum() / valid.float().sum().clamp_min(1.0)).to(dtype=logits.dtype)

            if prune_enabled and (actual_oracle_has_drop or actual_oracle_has_bad_drop):
                drop_oracle_loss, actual_oracle_drop_bad_count_value = _oracle_where_bce_logits(
                    learned_drop_logit / drop_proxy_tau,
                    leaf_delete_op_mask,
                    actual_oracle_drop_bad_mask,
                    actual_oracle_drop_bad_score,
                )
                drop_where_actuator_loss = drop_where_actuator_loss + oracle_candidate_weight * drop_oracle_loss
                actual_oracle_candidate_where_loss = actual_oracle_candidate_where_loss + drop_oracle_loss

            if add_enabled and (actual_oracle_has_add or actual_oracle_has_bad_add):
                oracle_add_logit = self.add_head(actuator_features)
                oracle_add_logit = self._scale_where_downstream_grad(
                    oracle_add_logit,
                    op_name="add",
                )
                oracle_add_logit = self._voxel_mean_logits(
                    oracle_add_logit,
                    voxel_coords,
                    voxel_cache=voxel_cache,
                )
                add_oracle_loss, actual_oracle_add_bad_count_value = _oracle_where_bce_logits(
                    oracle_add_logit,
                    leaf_add_op_mask,
                    actual_oracle_add_bad_mask,
                    actual_oracle_add_bad_score,
                )
                add_where_actuator_loss = add_where_actuator_loss + oracle_candidate_weight * add_oracle_loss
                actual_oracle_candidate_where_loss = actual_oracle_candidate_where_loss + add_oracle_loss
                oracle_add_direction_logits = self.add_voxel_head(actuator_features)
                oracle_add_direction_logits = self._scale_where_downstream_grad(
                    oracle_add_direction_logits,
                    op_name="add",
                )
                oracle_add_direction_logits = self._voxel_mean_logits(
                    oracle_add_direction_logits,
                    voxel_coords,
                    voxel_cache=voxel_cache,
                )
                if actual_oracle_has_bad_add and not actual_oracle_has_add:
                    bad_direction_mask = actual_oracle_add_bad_mask.to(
                        device=oracle_add_direction_logits.device,
                        dtype=oracle_add_direction_logits.dtype,
                    )
                    bad_direction_logit = oracle_add_direction_logits.amax(dim=1, keepdim=True)
                    add_direction_oracle_loss = (
                        torch.nn.functional.softplus(bad_direction_logit.float())
                        * bad_direction_mask.float()
                    ).sum() / bad_direction_mask.float().sum().clamp_min(1.0)
                    add_direction_oracle_loss = add_direction_oracle_loss.to(
                        dtype=oracle_add_direction_logits.dtype
                    )
                else:
                    add_direction_oracle_loss = _oracle_direction_loss(
                        oracle_add_direction_logits,
                        leaf_add_op_mask,
                        actual_oracle_add_direction_index,
                        actual_oracle_add_bad_mask,
                        actual_oracle_bad_add_direction_index,
                    )
                add_direction_oracle_loss = (
                    float(getattr(self.args, "sparsepcgc_actual_oracle_direction_weight", 1.0))
                    * add_direction_oracle_loss
                )
                add_where_actuator_loss = (
                    add_where_actuator_loss + oracle_candidate_weight * add_direction_oracle_loss
                )
                actual_oracle_candidate_where_loss = (
                    actual_oracle_candidate_where_loss + add_direction_oracle_loss
                )
                actual_oracle_direction_supervision_loss = (
                    actual_oracle_direction_supervision_loss + add_direction_oracle_loss
                )

            if disp_enabled and (actual_oracle_has_move or actual_oracle_has_bad_move):
                subtree_move_oracle_loss, actual_oracle_move_bad_count_value = _oracle_where_bce_logits(
                    subtree_move_source_logit,
                    leaf_move_op_mask,
                    actual_oracle_move_bad_mask,
                    actual_oracle_move_bad_score,
                )
                move_where_actuator_loss = (
                    move_where_actuator_loss + oracle_candidate_weight * subtree_move_oracle_loss
                )
                oracle_move_direction_logits = self.move_voxel_head(actuator_features)
                oracle_move_direction_logits = self._scale_where_downstream_grad(
                    oracle_move_direction_logits,
                    op_name="move",
                )
                oracle_move_direction_logits = self._voxel_mean_logits(
                    oracle_move_direction_logits,
                    voxel_coords,
                    voxel_cache=voxel_cache,
                )
                move_direction_oracle_loss = _oracle_direction_loss(
                    oracle_move_direction_logits,
                    leaf_move_op_mask,
                    actual_oracle_move_direction_index,
                    actual_oracle_move_bad_mask,
                    actual_oracle_bad_move_direction_index,
                )
                move_direction_oracle_loss = (
                    float(getattr(self.args, "sparsepcgc_actual_oracle_direction_weight", 1.0))
                    * move_direction_oracle_loss
                )
                move_where_actuator_loss = (
                    move_where_actuator_loss + oracle_candidate_weight * move_direction_oracle_loss
                )
                actual_oracle_candidate_where_loss = (
                    actual_oracle_candidate_where_loss
                    + subtree_move_oracle_loss
                    + move_direction_oracle_loss
                )
                actual_oracle_direction_supervision_loss = (
                    actual_oracle_direction_supervision_loss + move_direction_oracle_loss
                )

            gate_known = operation_gate_prob.new_tensor(
                [
                    1.0 if (actual_oracle_has_drop or actual_oracle_has_bad_drop) else 0.0,
                    1.0 if (actual_oracle_has_add or actual_oracle_has_bad_add) else 0.0,
                    1.0 if (actual_oracle_has_move or actual_oracle_has_bad_move) else 0.0,
                ]
            ).view(1, 3, 1)
            gate_target = operation_gate_prob.new_tensor(
                [
                    1.0 if actual_oracle_has_drop else 0.0,
                    1.0 if actual_oracle_has_add else 0.0,
                    1.0 if actual_oracle_has_move else 0.0,
                ]
            ).view(1, 3, 1)
            gate_pred = torch.sigmoid(operation_gate_logit).clamp(eps, 1.0 - eps)
            with torch.cuda.amp.autocast(enabled=False):
                gate_loss_raw = torch.nn.functional.binary_cross_entropy(
                    gate_pred.float(),
                    gate_target.expand_as(gate_pred).float(),
                    reduction="none",
                )
            gate_known = gate_known.expand_as(gate_loss_raw).float()
            operation_gate_oracle_loss = (
                gate_loss_raw * gate_known
            ).sum() / gate_known.sum().clamp_min(1.0)
        # ============================================================
        # L_actuator に入る補助損失の finite 化
        # ============================================================
        # 目的:
        # ・Actuator内部の一部損失が inf / nan になっても L_total 全体を壊さない
        # ・BCE, CE, 正規化除算, logit補助損失の異常値をここで止める
        # ============================================================
        def _finite_actuator_loss(x):
            if not torch.is_tensor(x):
                return pts_xyz.new_tensor(float(x))
            return torch.nan_to_num(
                x,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        edit_reg = _finite_actuator_loss(edit_reg)
        ratio_loss = _finite_actuator_loss(ratio_loss)
        shape_guard = _finite_actuator_loss(shape_guard)

        drop_ratio_loss = _finite_actuator_loss(drop_ratio_loss)
        drop_cap_loss = _finite_actuator_loss(drop_cap_loss)
        drop_shape_guard = _finite_actuator_loss(drop_shape_guard)
        drop_direct_target_loss = _finite_actuator_loss(drop_direct_target_loss)
        drop_entropy = _finite_actuator_loss(drop_entropy)

        add_ratio_loss = _finite_actuator_loss(add_ratio_loss)
        add_shape_guard = _finite_actuator_loss(add_shape_guard)
        add_offset_reg = _finite_actuator_loss(add_offset_reg)
        add_drop_conflict_loss = _finite_actuator_loss(add_drop_conflict_loss)
        added_keep_loss = _finite_actuator_loss(added_keep_loss)
        add_min_offset_loss = _finite_actuator_loss(add_min_offset_loss)

        quant_move_conflict_loss = _finite_actuator_loss(quant_move_conflict_loss)
        quant_add_guard = _finite_actuator_loss(quant_add_guard)
        empty_target_violation_loss = _finite_actuator_loss(empty_target_violation_loss)
        target_duplicate_voxel_loss = _finite_actuator_loss(target_duplicate_voxel_loss)
        local_edit_guard = _finite_actuator_loss(local_edit_guard)

        operation_amount_consistency_loss = _finite_actuator_loss(operation_amount_consistency_loss)
        operation_amount_direct_loss = _finite_actuator_loss(operation_amount_direct_loss)
        operation_amount_logit_loss = _finite_actuator_loss(operation_amount_logit_loss)
        operation_entropy_loss = _finite_actuator_loss(operation_entropy_loss)
        soft_activity_loss = _finite_actuator_loss(soft_activity_loss)

        move_direction_ce = _finite_actuator_loss(move_direction_ce)
        add_direction_ce = _finite_actuator_loss(add_direction_ce)

        drop_where_actuator_loss = _finite_actuator_loss(drop_where_actuator_loss)
        add_where_actuator_loss = _finite_actuator_loss(add_where_actuator_loss)
        move_where_actuator_loss = _finite_actuator_loss(move_where_actuator_loss)
        operation_gate_oracle_loss = _finite_actuator_loss(operation_gate_oracle_loss)
        actual_oracle_candidate_where_loss = _finite_actuator_loss(actual_oracle_candidate_where_loss)
        actual_oracle_direction_supervision_loss = _finite_actuator_loss(
            actual_oracle_direction_supervision_loss
        )

        drop_amount_supervision_loss = _finite_actuator_loss(drop_amount_supervision_loss)
        drop_amount_soft_consistency_loss = _finite_actuator_loss(drop_amount_soft_consistency_loss)
        move_amount_supervision_loss = _finite_actuator_loss(move_amount_supervision_loss)
        move_amount_soft_consistency_loss = _finite_actuator_loss(move_amount_soft_consistency_loss)
        add_amount_supervision_loss = _finite_actuator_loss(add_amount_supervision_loss)
        add_amount_soft_consistency_loss = _finite_actuator_loss(add_amount_soft_consistency_loss)
        actual_oracle_drop_amount_loss = _finite_actuator_loss(actual_oracle_drop_amount_loss)
        actual_oracle_add_amount_loss = _finite_actuator_loss(actual_oracle_add_amount_loss)
        actual_oracle_move_amount_loss = _finite_actuator_loss(actual_oracle_move_amount_loss)
        actual_oracle_drop_amount_logit_loss = _finite_actuator_loss(actual_oracle_drop_amount_logit_loss)
        actual_oracle_add_amount_logit_loss = _finite_actuator_loss(actual_oracle_add_amount_logit_loss)
        actual_oracle_move_amount_logit_loss = _finite_actuator_loss(actual_oracle_move_amount_logit_loss)
        actual_oracle_amount_supervision_loss = (
            actual_oracle_drop_amount_loss
            + actual_oracle_add_amount_loss
            + actual_oracle_move_amount_loss
        )

        loss = (
            edit_reg
            + float(getattr(self.args, "repair_ratio_weight", 0.1)) * ratio_loss
            + float(getattr(self.args, "repair_shape_guard_weight", 0.05)) * shape_guard
            + float(getattr(self.args, "repair_drop_ratio_weight", 1.0)) * (drop_ratio_loss + drop_cap_loss)
            + float(getattr(self.args, "repair_drop_shape_guard_weight", 0.5)) * drop_shape_guard
            + float(getattr(self.args, "repair_drop_direct_target_weight", 5.0)) * drop_direct_target_loss
            + float(getattr(self.args, "repair_drop_entropy_weight", 0.01)) * drop_entropy
            + float(getattr(self.args, "repair_add_ratio_weight", 4.0)) * add_ratio_loss
            + float(getattr(self.args, "repair_add_shape_guard_weight", 0.5)) * add_shape_guard
            + float(getattr(self.args, "repair_add_offset_weight", 0.25)) * add_offset_reg
            + float(getattr(self.args, "repair_add_drop_conflict_weight", 2.0)) * add_drop_conflict_loss
            + float(getattr(self.args, "repair_add_keep_weight", 1.0)) * added_keep_loss
            + float(getattr(self.args, "repair_add_min_offset_weight", 0.5)) * add_min_offset_loss
            + float(getattr(self.args, "repair_quant_guard_weight", 1.0)) * (quant_move_conflict_loss + quant_add_guard)
            + float(getattr(self.args, "sparsepcgc_empty_target_penalty_weight", 0.0)) * empty_target_violation_loss
            + float(getattr(self.args, "sparsepcgc_target_duplicate_penalty_weight", 0.0)) * target_duplicate_voxel_loss
            + float(getattr(self.args, "repair_local_guard_weight", 0.25)) * local_edit_guard
            + float(getattr(self.args, "repair_operation_amount_consistency_weight", 0.01)) * operation_amount_consistency_loss
            + float(getattr(self.args, "repair_operation_amount_direct_weight", 0.01)) * operation_amount_direct_loss
            + float(getattr(self.args, "repair_operation_amount_logit_weight", 1e-4)) * operation_amount_logit_loss
            + operation_entropy_loss
            + float(getattr(self.args, "repair_soft_activity_weight", 1e-3)) * soft_activity_loss
            + float(getattr(self.args, "repair_move_direction_ce_weight", 1e-3)) * move_direction_ce
            + float(getattr(self.args, "repair_add_direction_ce_weight", 1e-3)) * add_direction_ce
            + float(getattr(self.args, "repair_drop_where_actuator_weight", 0.1)) * drop_where_actuator_loss
            + float(getattr(self.args, "repair_add_where_actuator_weight", 0.1)) * add_where_actuator_loss
            + float(getattr(self.args, "repair_move_where_actuator_weight", 0.1)) * move_where_actuator_loss
            + float(getattr(self.args, "repair_operation_gate_oracle_weight", 0.1)) * operation_gate_oracle_loss
            # 量headは補助損失ではなく、主に圧縮損失で操作量を学習させる。
            # ここは勾配を完全に死なせないための弱い足場に留める。
            + float(getattr(self.args, "repair_drop_amount_supervision_weight", 0.001)) * drop_amount_supervision_loss
            + float(getattr(self.args, "repair_drop_amount_soft_consistency_weight", 0.0005)) * drop_amount_soft_consistency_loss
            + float(getattr(self.args, "repair_move_amount_supervision_weight", 0.001)) * move_amount_supervision_loss
            + float(getattr(self.args, "repair_move_amount_soft_consistency_weight", 0.0005)) * move_amount_soft_consistency_loss
            + float(getattr(self.args, "repair_add_amount_supervision_weight", 0.001)) * add_amount_supervision_loss
            + float(getattr(self.args, "repair_add_amount_soft_consistency_weight", 0.0005)) * add_amount_soft_consistency_loss
            + float(getattr(self.args, "sparsepcgc_actual_oracle_amount_weight", 0.05))
            * actual_oracle_amount_supervision_loss
            + float(getattr(self.args, "sparsepcgc_actual_oracle_direction_loss_weight", 0.01))
            * actual_oracle_direction_supervision_loss
        )
        loss = torch.nan_to_num(
            loss,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        # if not hasattr(self, "_printed_amount_grad_debug_loss"):
        #     amount_params = [
        #         self.drop_amount_head.weight,
        #         self.drop_amount_head.bias,
        #         self.add_amount_head.weight,
        #         self.add_amount_head.bias,
        #         self.move_amount_head.weight,
        #         self.move_amount_head.bias,
        #     ]

        #     amount_grads = torch.autograd.grad(
        #         loss,
        #         amount_params,
        #         retain_graph=True,
        #         create_graph=False,
        #         allow_unused=True,
        #     )

        #     def _grad_norm(g):
        #         if g is None:
        #             return None
        #         return float(g.detach().float().norm().cpu())

        #     print(
        #         "[AmountGradDebug:actuator_loss]",
        #         "drop_w=", _grad_norm(amount_grads[0]),
        #         "drop_b=", _grad_norm(amount_grads[1]),
        #         "add_w=", _grad_norm(amount_grads[2]),
        #         "add_b=", _grad_norm(amount_grads[3]),
        #         "move_w=", _grad_norm(amount_grads[4]),
        #         "move_b=", _grad_norm(amount_grads[5]),
        #     )
        #     self._printed_amount_grad_debug_loss = True
        
        if timing_enabled:
            _mark_runtime("postprocess")
            runtime_timing["total"] = float(time.perf_counter() - runtime_start)
            self.last_runtime_timing = runtime_timing
        else:
            self.last_runtime_timing = {}

        # Actuatorのhard/soft出力の統計を取る比較モード。学習中の挙動確認や、学習比率headの効果確認に。
        if bool(getattr(self.args, "print_actuator_hard_soft_compare", False)):
            with torch.no_grad():
                def _mf(x):
                    if torch.is_tensor(x):
                        return float(x.detach().float().mean().cpu())
                    return float(x)

                def _sf(x):
                    if torch.is_tensor(x):
                        return float(x.detach().float().sum().cpu())
                    return float(x)

                def _corr(a, b):
                    a = a.detach().float().reshape(-1)
                    b = b.detach().float().reshape(-1)
                    if a.numel() <= 1 or b.numel() <= 1:
                        return 0.0
                    a = a - a.mean()
                    b = b - b.mean()
                    denom = a.norm() * b.norm()
                    if float(denom.cpu()) <= 1e-12:
                        return 0.0
                    return float((a * b).sum().div(denom).cpu())

                drop_hard_mean = _mf(hard_drop)

                # 最終的なSoft削除量はsoft_drop_probだが、表示用平均は0〜1に収めた値を見る。
                # drop_prob は位置score寄り、soft_drop_prob_for_guard は量headで予算化された削除強度である。
                drop_soft_mean = _mf(soft_drop_prob_for_guard)
                drop_score_soft_mean = _mf(drop_prob)

                drop_proxy_mean = _mf(drop_prob_proxy)
                drop_direct_mean = _mf(drop_prob_direct)

                drop_abs_diff = _mf((hard_drop - soft_drop_prob_for_guard).abs())
                drop_corr = _corr(hard_drop, soft_drop_prob_for_guard)

                drop_ratio_soft_value = _mf(drop_ratio_soft)
                drop_ratio_hard_value = _mf(drop_ratio_hard)

                # 点数基準の差。
                drop_amount_abs_diff = abs(_sf(soft_drop_prob_for_guard) - _sf(hard_drop))

                # Voxel数基準の差。
                drop_amount_voxel_abs_diff = abs(_sf(soft_drop_voxel_mass_per_batch) - _sf(hard_drop_voxel_mass_per_batch))
                drop_amount_ratio_abs_diff = abs(drop_ratio_soft_value - drop_ratio_hard_value)

                soft_drop_budget_mean = _mf(soft_drop_budget)

                # 点数基準のsum。
                soft_drop_sum_value = _sf(soft_drop_prob_for_guard)
                hard_drop_sum_value = _sf(hard_drop)

                # Voxel数基準のsum。
                soft_drop_voxel_sum_value = _sf(soft_drop_voxel_mass_per_batch)
                hard_drop_voxel_sum_value = _sf(hard_drop_voxel_mass_per_batch)

                valid_delete_voxel_count_value = _sf(valid_delete_voxel_count)

                # move_score は位置score寄り、soft_move_score_for_guard は量headで予算化されたMove強度である。
                move_score_soft_mean = _mf(move_score)
                move_soft_mean = _mf(soft_move_score_for_guard)
                move_hard_mean = _mf(hard_move)
                move_mask_mean = _mf(move_mask_for_guard)

                move_abs_diff = _mf((hard_move - soft_move_score_for_guard).abs())
                move_corr = _corr(hard_move, soft_move_score_for_guard)

                move_ratio_soft_value = _mf(move_ratio_soft)
                move_ratio_hard_value = _mf(move_ratio_hard)

                # 点数基準の差。
                move_amount_abs_diff = abs(_sf(soft_move_score_for_guard) - _sf(hard_move))

                # Voxel数基準の差。
                move_amount_voxel_abs_diff = abs(
                    _sf(soft_move_voxel_mass_per_batch) - _sf(hard_move_voxel_mass_per_batch)
                )
                move_amount_ratio_abs_diff = abs(move_ratio_soft_value - move_ratio_hard_value)

                soft_move_budget_mean = _mf(soft_move_budget)

                # 点数基準のsum。
                soft_move_sum_value = _sf(soft_move_score_for_guard)
                hard_move_sum_value = _sf(hard_move)

                # Voxel数基準のsum。
                soft_move_voxel_sum_value = _sf(soft_move_voxel_mass_per_batch)
                hard_move_voxel_sum_value = _sf(hard_move_voxel_mass_per_batch)

                valid_move_source_voxel_count_value = _sf(valid_move_source_voxel_count)
                valid_move_source_voxel_count_effective_value = _sf(valid_move_source_voxel_count_effective)

                move_dir_conf = _mf(move_probs.max(dim=1, keepdim=True).values)
                move_dir_entropy = _mf(
                    -(move_probs.clamp_min(1e-8).log() * move_probs).sum(dim=1, keepdim=True)
                )

                # add_probはsource点単位に畳んだ補助信号。
                # Add量の主比較はtarget voxel単位で見る。
                add_soft_mean = _mf(add_target_soft_add)
                add_soft_sum = _sf(add_target_soft_add)
                add_ratio_soft_value = _mf(add_ratio_soft)
                add_ratio_hard_value = _mf(add_ratio_hard)

                add_amount_abs_diff = abs(
                    _sf(add_target_soft_add) - _sf(add_target_hard_add)
                )
                add_amount_ratio_abs_diff = abs(add_ratio_soft_value - add_ratio_hard_value)
                add_soft_budget_mean = _mf(soft_add_budget)

                # pair単位は補助ログとして残す。
                add_soft_pair_sum = _sf(soft_add_pair)
                add_hard_pair_sum = _sf(hard_add_pair)

                # target voxel単位の主ログ。
                add_target_soft_add_sum = _sf(add_target_soft_add)
                add_target_hard_add_sum = _sf(add_target_hard_add)
                add_target_soft_hard_sum_abs_diff = abs(
                    add_target_soft_add_sum - add_target_hard_add_sum
                )
                final_w_mean = _mf(final_w)
                final_w_min = float(final_w.detach().float().amin().cpu()) if final_w.numel() > 0 else 0.0
                final_w_max = float(final_w.detach().float().amax().cpu()) if final_w.numel() > 0 else 0.0

                # ============================================================
                # Adjustのguard後Soft/Hard比較
                # ============================================================
                # 既存のmove_soft_meanはguard前Softを見ている可能性がある。
                # そのため、Hard実行に対応するguard後Softで比較する。
                move_soft_effective_mean = _mf(soft_move_score_effective)
                move_source_soft_keep_mean = _mf(move_source_soft_keep)
                move_target_soft_add_mean = _mf(move_target_soft_add)

                move_abs_diff_effective = _mf(
                    (hard_move - soft_move_score_effective).abs()
                )
                move_corr_effective = _corr(
                    hard_move,
                    soft_move_score_effective,
                )

                raw_hard_move_mean = _mf(
                    raw_hard_move_bool.to(dtype=pts_xyz.dtype).unsqueeze(1)
                )

                move_guard_reject_rate = (
                    float(raw_hard_move_count_value - hard_move_count_value)
                    / max(float(raw_hard_move_count_value), 1.0)
                )

                print(
                    "[ActuatorHardSoftCompare] "
                    f"drop_soft_mean={drop_soft_mean:.6f}, "
                    f"drop_score_soft_mean={drop_score_soft_mean:.6f}, "
                    f"drop_hard_mean={drop_hard_mean:.6f}, "
                    f"drop_abs_diff={drop_abs_diff:.6f}, "
                    f"drop_corr={drop_corr:.6f}, "
                    f"drop_proxy_mean={drop_proxy_mean:.6f}, "
                    f"drop_direct_mean={drop_direct_mean:.6f}, "
                    f"drop_ratio_soft={drop_ratio_soft_value:.6f}, "
                    f"drop_ratio_hard={drop_ratio_hard_value:.6f}, "
                    f"drop_amount_abs_diff={drop_amount_abs_diff:.6f}, "
                    f"drop_amount_voxel_abs_diff={drop_amount_voxel_abs_diff:.6f}, "
                    f"drop_amount_ratio_abs_diff={drop_amount_ratio_abs_diff:.6f}, "
                    f"soft_drop_budget_mean={soft_drop_budget_mean:.6f}, "
                    f"soft_drop_sum={soft_drop_sum_value:.6f}, "
                    f"hard_drop_sum={hard_drop_sum_value:.6f}, "
                    f"soft_drop_voxel_sum={soft_drop_voxel_sum_value:.6f}, "
                    f"hard_drop_voxel_sum={hard_drop_voxel_sum_value:.6f}, "
                    f"valid_delete_voxel_count={valid_delete_voxel_count_value:.6f}, "
                    f"hard_drop_count={hard_drop_count_value}, "
                    f"delete_target_voxel_count={delete_target_voxel_count_value}, "
                    f"delete_emptied_voxel_count={delete_emptied_voxel_count_value}, "
                    f"move_soft_mean={move_soft_mean:.6f}, "
                    f"move_score_soft_mean={move_score_soft_mean:.6f}, "
                    f"move_hard_mean={move_hard_mean:.6f}, "
                    f"move_mask_mean={move_mask_mean:.6f}, "
                    f"move_abs_diff={move_abs_diff:.6f}, "
                    f"move_corr={move_corr:.6f}, "
                    f"move_soft_effective_mean={move_soft_effective_mean:.6f}, "
                    f"move_abs_diff_effective={move_abs_diff_effective:.6f}, "
                    f"move_corr_effective={move_corr_effective:.6f}, "
                    f"move_source_soft_keep_mean={move_source_soft_keep_mean:.6f}, "
                    f"move_target_soft_add_mean={move_target_soft_add_mean:.6f}, "
                    f"raw_hard_move_mean={raw_hard_move_mean:.6f}, "
                    f"move_guard_reject_rate={move_guard_reject_rate:.6f}, "
                    f"move_ratio_soft={move_ratio_soft_value:.6f}, "
                    f"move_ratio_hard={move_ratio_hard_value:.6f}, "
                    f"move_amount_abs_diff={move_amount_abs_diff:.6f}, "
                    f"move_amount_voxel_abs_diff={move_amount_voxel_abs_diff:.6f}, "
                    f"move_amount_ratio_abs_diff={move_amount_ratio_abs_diff:.6f}, "
                    f"soft_move_budget_mean={soft_move_budget_mean:.6f}, "
                    f"soft_move_sum={soft_move_sum_value:.6f}, "
                    f"hard_move_sum={hard_move_sum_value:.6f}, "
                    f"soft_move_voxel_sum={soft_move_voxel_sum_value:.6f}, "
                    f"hard_move_voxel_sum={hard_move_voxel_sum_value:.6f}, "
                    f"valid_move_source_voxel_count={valid_move_source_voxel_count_value:.6f}, "
                    f"valid_move_source_voxel_count_effective={valid_move_source_voxel_count_effective_value:.6f}, "
                    f"hard_move_count={hard_move_count_value}, "
                    f"raw_hard_move_count={raw_hard_move_count_value}, "
                    f"move_dir_conf={move_dir_conf:.6f}, "
                    f"move_dir_entropy={move_dir_entropy:.6f}, "
                    f"add_soft_mean={add_soft_mean:.6f}, "
                    f"add_soft_sum={add_soft_sum:.6f}, "
                    f"add_count={add_count_value}, "
                    f"add_effective_count={add_effective_count_value}, "
                    f"add_target_voxel_count={add_target_voxel_count_value}, "
                    f"add_ratio_soft={add_ratio_soft_value:.6f}, "
                    f"add_ratio_hard={add_ratio_hard_value:.6f}, "
                    f"add_amount_abs_diff={add_amount_abs_diff:.6f}, "
                    f"add_amount_ratio_abs_diff={add_amount_ratio_abs_diff:.6f}, "
                    f"add_soft_budget_mean={add_soft_budget_mean:.6f}, "
                    f"add_soft_pair_sum={add_soft_pair_sum:.6f}, "
                    f"add_hard_pair_sum={add_hard_pair_sum:.6f}, "
                    f"add_target_soft_add_sum={add_target_soft_add_sum:.6f}, "
                    f"add_target_hard_add_sum={add_target_hard_add_sum:.6f}, "
                    f"add_target_soft_hard_sum_abs_diff={add_target_soft_hard_sum_abs_diff:.6f}, "
                    f"learned_drop_ratio={_mf(learned_drop_ratio):.6f}, "
                    f"learned_move_ratio={_mf(learned_move_ratio):.6f}, "
                    f"learned_add_ratio={_mf(learned_add_ratio):.6f}, "
                    f"final_w_mean={final_w_mean:.6f}, "
                    f"final_w_min={final_w_min:.6f}, "
                    f"final_w_max={final_w_max:.6f}, "
                    f"before_voxels={before_occupied_voxels}, "
                    f"after_voxels={after_occupied_voxels}, "
                    f"occupied_voxel_delta={after_occupied_voxels - before_occupied_voxels}, "
                    f"actuator_voxel_mode={actuator_voxel_mode}, "
                    f"local_recomputed={bool(actuator_local_recomputed)}, "
                    f"uses_context_voxel_frame={bool(uses_context_voxel_frame)}, "
                    f"final_voxel_update_mode=state_update_from_initial_voxels, "
                    f"final_voxel_recomputed_from_pts_out=False, "
                    f"initial_voxel_point_count={int(voxel_coords.shape[2])}, "
                    f"final_voxel_point_count={int(voxel_edit_final_coords.shape[2])}, "
                    f"final_voxel_added_slots={int(voxel_edit_final_coords.shape[2] - voxel_coords.shape[2])}"
                )
        # SparsePCGC/Voxel modeでは、Actuatorの公開出力もoccupied voxel状態から復元する。
        # 点ごとのsoft座標差分はproxy勾配用に内部で使うだけで、出力点群にはしない。
        voxel_restored_output_enabled = bool(
            getattr(
                self.args,
                "repair_output_voxel_restored_points",
                sparsepcgc_context and voxel_edit_state_enabled,
            )
        )
        point_soft_delta_debug = delta
        if voxel_restored_output_enabled and voxel_edit_state_enabled:
            pts_out = self._voxel_centers_from_global_coords(
                voxel_edit_final_coords,
                voxel_step,
                voxel_offset,
                dtype=pts_xyz.dtype,
            )
            final_w = voxel_edit_final_weights.to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            delta = pts_xyz.new_zeros(pts_xyz.shape)
            delta_norm = torch.zeros_like(delta_norm)
            moved_delta_mean = pts_xyz.new_zeros(())

        # Phase2: Voxel/Octree出力を点群xyzへ戻すためのdebug経路。
        canonical_voxel_coords_before = voxel_coords.detach()
        canonical_voxel_coords_after = (
            voxel_edit_final_coords.detach()
            if voxel_edit_state_enabled
            else final_voxel_coords.detach()
        )

        voxel_restore_meta = {
            "global_qs": voxel_step.detach(),
            "global_offset": voxel_offset.detach(),
            "effective_qs_tensor": voxel_step.detach(),
            "global_offset_tensor": voxel_offset.detach(),
            "voxel_size": float(getattr(self.args, "sparsepcgc_voxel_size", 1.0)),
            "pos_quantscale": int(getattr(self.args, "sparsepcgc_pos_quantscale", 1)),
            "quant_mode": str(getattr(self.args, "sparsepcgc_quant_mode", "round_voxel_then_pos")),
        }

        restored_xyz_debug = None
        restore_info = {
            "restore_input_points": int(canonical_voxel_coords_after.shape[-1]),
            "restore_output_points": int(canonical_voxel_coords_after.shape[-1]),
            "restore_unique": bool(getattr(self.args, "sparsepcgc_restore_unique_voxels", True)),
            "restore_center": bool(getattr(self.args, "sparsepcgc_dequantize_center", False)),
            "restore_has_meta": True,
        }

        if bool(getattr(self.args, "sparsepcgc_restore_points_debug", False)):
            with torch.no_grad():
                restored_xyz_debug, restore_info = restore_points_from_voxel_coords(
                    canonical_voxel_coords_after,
                    meta=voxel_restore_meta,
                    args=self.args,
                    center=bool(getattr(self.args, "sparsepcgc_dequantize_center", False)),
                    unique=bool(getattr(self.args, "sparsepcgc_restore_unique_voxels", True)),
                    dtype=pts_xyz.dtype,
                    device=pts_xyz.device,
                )
                restored_xyz_debug = restored_xyz_debug.detach()

        self.debug_tensors = {
            # AdjustをSoft Prune + Soft Addとして扱うための情報
            "operation_amount_direct_loss": operation_amount_direct_loss,
            "drop_amount_target": drop_amount_target.detach(),
            "move_amount_target": move_amount_target.detach(),
            "add_amount_target": add_amount_target.detach(),
            "move_source_soft_delete": move_source_soft_delete,
            "move_source_soft_keep": move_source_soft_keep,
            "move_target_soft_add": move_target_soft_add,
            "move_source_voxel_coords": voxel_coords,
            "move_target_voxel_coords": move_target_voxel_coords,
            "move_soft_score_effective": soft_move_score_effective,
            "move_soft_value_mode": "soft_prune_source_and_soft_add_target",
            "full_octree_context_available": pts_xyz.new_tensor(float(full_context_available)).detach(),
            "actuator_parent_occupancy_code": pts_xyz.new_tensor(float(actuator_parent_occupancy_code)).detach(),
            "actuator_sibling_count": pts_xyz.new_tensor(float(actuator_sibling_count)).detach(),
            "actuator_ancestor_count": pts_xyz.new_tensor(float(actuator_ancestor_count)).detach(),
            "full_context_bonus_mean": full_context_bonus.mean().detach(),
            "child_slot_candidate_ratio": child_slot_candidate_ratio.detach(),
            "full_context_bonus_mean": full_context_bonus.mean().detach(),

            # Section4:
            # leaf pattern priorがActuatorへ入っているか確認するdebug。
            "leaf_actuator_prior_enabled": pts_xyz.new_tensor(
                float(bool(leaf_actuator_prior.get("enabled", False)))
            ).detach(),
            "leaf_actuator_drop_prior_mean": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("delete_prior_mean", 0.0))
            ).detach(),
            "leaf_actuator_add_prior_mean": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("add_prior_mean", 0.0))
            ).detach(),
            "leaf_actuator_move_prior_mean": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("move_prior_mean", 0.0))
            ).detach(),
            "leaf_actuator_best_prior_mean": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("best_prior_mean", 0.0))
            ).detach(),
            "leaf_actuator_best_prior_max": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("best_prior_max", 0.0))
            ).detach(),

            # Section5:
            # leaf pattern診断がAdd/Move target方向へ反映されたか確認する。
            "leaf_target_direction_prior_enabled": pts_xyz.new_tensor(
                float(bool(leaf_target_direction_prior.get("enabled", False)))
            ).detach(),
            "leaf_add_target_match_ratio": pts_xyz.new_tensor(
                float(leaf_target_direction_prior.get("add_target_match_ratio", 0.0))
            ).detach(),
            "leaf_move_target_match_ratio": pts_xyz.new_tensor(
                float(leaf_target_direction_prior.get("move_target_match_ratio", 0.0))
            ).detach(),
            "leaf_add_target_bias_mean": pts_xyz.new_tensor(
                float(leaf_target_direction_prior.get("add_target_bias_mean", 0.0))
            ).detach(),
            "leaf_move_target_bias_mean": pts_xyz.new_tensor(
                float(leaf_target_direction_prior.get("move_target_bias_mean", 0.0))
            ).detach(),

            "child_slot_candidate_ratio": child_slot_candidate_ratio.detach(),
            "repair_gate": repair_gate.mean().detach(),
            # 既存keyの add_ratio は互換用に残すが、中身は学習用のsoft実行率にする。
            "add_ratio": add_ratio_soft.detach(),
            "add_prob_mean": add_prob.mean().detach(),
            "add_prob_max": add_prob.max().detach() if add_prob.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "add_priority_mean": add_priority.mean().detach(),
            "add_priority_max": add_priority.max().detach() if add_priority.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "add_candidate_ratio": pts_xyz.new_tensor(float(add_candidate_ratio)).detach(),
            "add_candidate_count": pts_xyz.new_tensor(float(add_k)).detach(),
            "add_ratio_soft": add_ratio_soft.detach(),
            "add_ratio_hard": add_ratio_hard.detach(),
            "add_amount_supervision_loss": add_amount_supervision_loss.detach(),
            "add_amount_soft_consistency_loss": add_amount_soft_consistency_loss.detach(),
            "actual_oracle_drop_amount_loss": actual_oracle_drop_amount_loss.detach(),
            "actual_oracle_add_amount_loss": actual_oracle_add_amount_loss.detach(),
            "actual_oracle_move_amount_loss": actual_oracle_move_amount_loss.detach(),
            "actual_oracle_drop_amount_logit_loss": actual_oracle_drop_amount_logit_loss.detach(),
            "actual_oracle_add_amount_logit_loss": actual_oracle_add_amount_logit_loss.detach(),
            "actual_oracle_amount_supervision_loss": actual_oracle_amount_supervision_loss.detach(),
            "add_soft_budget_mean": soft_add_budget.detach().mean(),
            "add_soft_pair_sum": soft_add_pair.detach().sum(),
            "add_hard_pair_sum": hard_add_pair.detach().sum(),
            "add_soft_hard_sum_abs_diff": (soft_add_pair.detach().sum() - hard_add_pair.detach().sum()).abs(),
            "add_hard_soft_consistency_loss": add_hard_soft_consistency_loss.detach(),
            # target voxel単位のAdd Hard/Soft状態。
            # Addの主指標はこちらを使う。
            "add_target_soft_add": add_target_soft_add.detach(),
            "add_target_hard_add": add_target_hard_add.detach(),
            "add_target_add_st": add_target_add_st.detach(),
            "add_target_voxel_coords": add_target_voxel_coords.detach(),
            "add_target_soft_add_sum": add_target_soft_add.detach().sum(),
            "add_target_hard_add_sum": add_target_hard_add.detach().sum(),
            "add_target_soft_hard_sum_abs_diff": (
                add_target_soft_add.detach().sum()
                - add_target_hard_add.detach().sum()
            ).abs(),
            "operation_gate_prob": operation_gate_prob.detach(),
            "operation_gate_hard": operation_gate_hard.detach(),
            "operation_gate_logit": operation_gate_logit.detach(),
            "drop_operation_gate": drop_operation_gate.detach().mean(),
            "add_operation_gate": add_operation_gate.detach().mean(),
            "move_operation_gate": move_operation_gate.detach().mean(),
            "operation_gate_oracle_loss": operation_gate_oracle_loss.detach(),
            "actual_oracle_candidate_where_loss": actual_oracle_candidate_where_loss.detach(),
            "actual_oracle_direction_supervision_loss": actual_oracle_direction_supervision_loss.detach(),
            "actual_oracle_bad_candidate_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_bad_candidate_count", 0))
            ).detach(),
            "actual_oracle_improving_candidate_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_improving_candidate_count", 0))
            ).detach(),
            "actual_oracle_combo_extra_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_combo_extra_count", 0))
            ).detach(),
            "actual_oracle_generated_candidate_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_generated_candidate_count", 0))
            ).detach(),
            "actual_oracle_accepted_candidate_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_candidate_count", 0))
            ).detach(),
            "actual_oracle_accepted_prune_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_prune_count", 0))
            ).detach(),
            "actual_oracle_accepted_add_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_add_count", 0))
            ).detach(),
            "actual_oracle_accepted_adjust_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_adjust_count", 0))
            ).detach(),
            "actual_oracle_accepted_subtree_move_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_subtree_move_count", 0))
            ).detach(),
            "actual_oracle_accepted_parent_collapse_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_parent_collapse_count", 0))
            ).detach(),
            "actual_oracle_accepted_pattern_canonicalize_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_pattern_canonicalize_count", 0))
            ).detach(),
            "actual_oracle_noop_label_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_noop_label_count", 0))
            ).detach(),
            "actual_oracle_noop_label_weight": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_noop_label_weight", 0.0))
            ).detach(),
            "actual_oracle_high_rate_mppov_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_high_rate_mppov_count", 0))
            ).detach(),
            "actual_oracle_low_prob_occupied_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_low_prob_occupied_count", 0))
            ).detach(),
            "actual_oracle_single_child_chain_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_single_child_chain_count", 0))
            ).detach(),
            "actual_oracle_context_pattern_candidate_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_context_pattern_candidate_count", 0))
            ).detach(),
            "actual_oracle_eval_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_eval_count", 0))
            ).detach(),
            "actual_oracle_eval_max": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_eval_max", 0))
            ).detach(),
            "actual_oracle_time": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_time", 0.0))
            ).detach(),
            "actual_oracle_drop_bad_count": pts_xyz.new_tensor(float(actual_oracle_drop_bad_count_value)).detach(),
            "actual_oracle_add_bad_count": pts_xyz.new_tensor(float(actual_oracle_add_bad_count_value)).detach(),
            "actual_oracle_move_bad_count": pts_xyz.new_tensor(float(actual_oracle_move_bad_count_value)).detach(),
            "actual_oracle_drop_reason": str(leaf_operation_masks.get("actual_oracle_drop_reason", "")),
            "actual_oracle_operation": str(leaf_operation_masks.get("actual_oracle_operation", "")),
            "actual_oracle_scheduled_operation": str(
                leaf_operation_masks.get("actual_oracle_scheduled_operation", "")
            ),
            "actual_oracle_edit_record_bits": pts_xyz.new_tensor(
                float(actual_oracle_edit_record_bits_value)
            ).detach(),
            "actual_oracle_raw_percent": pts_xyz.new_tensor(
                float(actual_oracle_raw_percent_value)
            ).detach(),
            "actual_oracle_delta_actual_percent": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_delta_actual_percent", 0.0))
            ).detach(),
            "actual_oracle_proxy_percent": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_proxy_percent", 0.0))
            ).detach(),
            "actual_oracle_geometry_percent": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_geometry_percent", 0.0))
            ).detach(),
            "actual_oracle_original_actual_bits": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_original_actual_bits", 0.0))
            ).detach(),
            "actual_oracle_edited_actual_bits": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_edited_actual_bits", 0.0))
            ).detach(),
            "actual_oracle_fast_diagnostic_used": pts_xyz.new_tensor(
                float(bool(leaf_operation_masks.get("actual_oracle_fast_diagnostic_used", False)))
            ).detach(),
            "actual_oracle_fast_diagnostic_full_drop_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_fast_diagnostic_full_drop_count", 0))
            ).detach(),
            "actual_oracle_fast_diagnostic_local_drop_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_fast_diagnostic_local_drop_count", 0))
            ).detach(),
            "actual_oracle_fast_diagnostic_full_drop_ratio": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_fast_diagnostic_full_drop_ratio", 0.0))
            ).detach(),
            "actual_oracle_fast_diagnostic_local_drop_ratio": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_fast_diagnostic_local_drop_ratio", 0.0))
            ).detach(),
            "raw_learned_drop_ratio": raw_learned_drop_ratio.mean().detach(),
            "raw_learned_add_ratio": raw_learned_add_ratio.mean().detach(),
            "raw_learned_move_ratio": raw_learned_move_ratio.mean().detach(),
            "learned_drop_ratio": learned_drop_ratio.mean().detach(),
            "learned_drop_prob": learned_drop_prob.detach(),
            "drop_prob_mean": drop_prob.mean().detach(),
            "drop_prob_min": drop_prob.amin().detach() if drop_prob.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "drop_prob_max": drop_prob.amax().detach() if drop_prob.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "drop_prob_direct_mean": drop_prob_direct.mean().detach(),
            "drop_prob_direct_min": drop_prob_direct.amin().detach() if drop_prob_direct.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "drop_prob_direct_max": drop_prob_direct.amax().detach() if drop_prob_direct.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "drop_prob_proxy_mean": drop_prob_proxy.mean().detach(),
            "drop_prob_proxy_min": drop_prob_proxy.amin().detach() if drop_prob_proxy.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "drop_prob_proxy_max": drop_prob_proxy.amax().detach() if drop_prob_proxy.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "drop_logit_mean": learned_drop_logit.mean().detach(),
            "drop_logit_min": learned_drop_logit.amin().detach() if learned_drop_logit.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "drop_logit_max": learned_drop_logit.amax().detach() if learned_drop_logit.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "keep_prob_mean": keep_prob.mean().detach(),
            "keep_prob_min": keep_prob.amin().detach() if keep_prob.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "keep_prob_max": keep_prob.amax().detach() if keep_prob.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "drop_entropy": drop_entropy.detach(),
            "soft_drop_mass": soft_drop_mass.detach(),
            "prune_soft_geom": prune_soft_geom.detach(),
            "prune_soft_rate": prune_soft_rate.detach(),
            "prune_soft_node": prune_soft_node.detach(),
            "prune_soft_single": prune_soft_single.detach(),
            "prune_soft_bit": prune_soft_bit.detach(),
            "learned_drop_ratio_std": learned_drop_ratio.detach().float().std(unbiased=False),
            "learned_add_ratio": learned_add_ratio.mean().detach(),
            "learned_add_ratio_std": learned_add_ratio.detach().float().std(unbiased=False),
            "learned_move_ratio": learned_move_ratio.mean().detach(),
            "learned_move_ratio_std": learned_move_ratio.detach().float().std(unbiased=False),
            "operation_amount_consistency_loss": operation_amount_consistency_loss.detach(),
            "operation_entropy": operation_entropy.detach(),
            "operation_entropy_loss": operation_entropy_loss.detach(),
            "operation_entropy_weight_effective": pts_xyz.new_tensor(
                float(operation_entropy_weight_effective)
            ).detach(),
            "soft_activity_loss": soft_activity_loss.detach(),
            "move_direction_ce": move_direction_ce.detach(),
            "add_direction_ce": add_direction_ce.detach(),
            "temperature": pts_xyz.new_tensor(float(operation_temperature)).detach(),
            "exploration_noise": pts_xyz.new_tensor(float(exploration_noise)).detach(),
            "operation_prob_floor_applied": pts_xyz.new_tensor(float(add_ratio_floor_applied)).detach(),
            "move_score_noise": pts_xyz.new_tensor(float(move_score_noise)).detach(),
            "sparsepcgc_add_experiment_enabled": pts_xyz.new_tensor(float(sparsepcgc_add_experiment_active)).detach(),
            "sparsepcgc_add_warmup": pts_xyz.new_tensor(float(self._sparsepcgc_add_warmup())).detach(),
            "add_score_noise": pts_xyz.new_tensor(float(add_score_noise)).detach(),
            "add_weight_random_mix": pts_xyz.new_tensor(float(add_weight_random_mix)).detach(),
            "drop_score_noise": pts_xyz.new_tensor(float(drop_score_noise)).detach(),
            "drop_random_mix": pts_xyz.new_tensor(float(drop_random_mix)).detach(),
            "add_enabled": pts_xyz.new_tensor(float(add_enabled)).detach(),
            "prune_enabled": pts_xyz.new_tensor(float(prune_enabled)).detach(),
            "disp_enabled": pts_xyz.new_tensor(float(disp_enabled)).detach(),
            "actuator_strength": pts_xyz.new_tensor(float(actuator_strength)).detach(),
            "force_joint_actuator": pts_xyz.new_tensor(float(force_joint_actuator)).detach(),
            "threshold_cap_mode": pts_xyz.new_tensor(float(threshold_cap_mode)).detach(),
            "actuator_voxel_mode": actuator_voxel_mode,
            "local_recomputed": pts_xyz.new_tensor(float(actuator_local_recomputed)).detach(),
            "add_drop_conflict_loss": add_drop_conflict_loss.detach(),
            "added_keep_loss": added_keep_loss.detach(),
            "add_min_offset_loss": add_min_offset_loss.detach(),
            "quant_move_conflict_loss": quant_move_conflict_loss.detach(),
            "quant_add_guard": quant_add_guard.detach(),
            "local_edit_guard": local_edit_guard.detach(),
            "quant_score_mean": quant_score.mean().detach(),
            "delta_norm": delta_norm.mean().detach(),
            "moved_delta_mean": moved_delta_mean.detach(),
            "move_ratio": hard_move.mean().detach(),
            "hard_move_count": pts_xyz.new_tensor(float(hard_move_count_value)).detach(),
            "move_score_mean": move_score.mean().detach(),
            "move_ratio_soft": move_ratio_soft.detach(),
            "move_ratio_hard": move_ratio_hard.detach(),
            "move_ratio_soft_batch_mean": move_ratio_soft_batch.detach().mean(),
            "move_ratio_hard_batch_mean": move_ratio_hard_batch.detach().mean(),
            "move_amount_supervision_loss": move_amount_supervision_loss.detach(),
            "move_amount_soft_consistency_loss": move_amount_soft_consistency_loss.detach(),

            "soft_move_budget_mean": soft_move_budget.detach().mean(),
            "valid_move_source_voxel_count_mean": valid_move_source_voxel_count.detach().mean(),
            "valid_move_source_voxel_count_effective_mean": valid_move_source_voxel_count_effective.detach().mean(),

            # 点数基準のAdjust量。
            "soft_move_sum": soft_move_score_for_guard.detach().sum(),
            "hard_move_sum": hard_move.detach().sum(),
            "move_soft_hard_sum_abs_diff": (
                soft_move_score_for_guard.detach().sum() - hard_move.detach().sum()
            ).abs(),

            # Voxel数基準のAdjust量。
            "soft_move_voxel_sum": soft_move_voxel_mass_per_batch.detach().sum(),
            "hard_move_voxel_sum": hard_move_voxel_mass_per_batch.detach().sum(),
            "move_soft_hard_voxel_sum_abs_diff": (
                soft_move_voxel_mass_per_batch.detach().sum()
                - hard_move_voxel_mass_per_batch.detach().sum()
            ).abs(),

            "move_soft_hard_ratio_abs_diff": (
                move_ratio_soft.detach() - move_ratio_hard.detach()
            ).abs(),
            "move_source_prior_mean": move_source_prior.mean().detach(),
            "subtree_move_source_prob_mean": subtree_move_source_prob.mean().detach(),
            "subtree_move_source_prob_max": subtree_move_source_prob.amax().detach() if subtree_move_source_prob.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "adjusted_point_count": pts_xyz.new_tensor(float(hard_move_count_value)).detach(),
            "adjusted_point_rate": pts_xyz.new_tensor(float(adjusted_point_rate_value)).detach(),
            "raw_hard_move_count_before_sparsepcgc_guard": pts_xyz.new_tensor(float(raw_hard_move_count_value)).detach(),
            "source_unique_voxel_count": pts_xyz.new_tensor(float(sparsepcgc_source_unique_voxel_count_value)).detach(),
            "target_unique_voxel_count": pts_xyz.new_tensor(float(sparsepcgc_target_unique_voxel_count_value)).detach(),
            "target_duplicate_voxel_count": pts_xyz.new_tensor(float(sparsepcgc_target_duplicate_voxel_count_value)).detach(),
            "target_voxel_duplicate_rate": pts_xyz.new_tensor(float(sparsepcgc_target_duplicate_rate_value)).detach(),
            "target_existing_occupied_count": pts_xyz.new_tensor(float(target_existing_occupied_count_value)).detach(),
            "target_existing_occupied_rate": pts_xyz.new_tensor(float(target_existing_occupied_rate_value)).detach(),
            "target_empty_voxel_count": pts_xyz.new_tensor(float(target_empty_voxel_count_value)).detach(),
            "target_empty_voxel_rate": pts_xyz.new_tensor(float(target_empty_voxel_rate_value)).detach(),
            "empty_target_violation_loss": empty_target_violation_loss.detach(),
            "target_duplicate_voxel_loss": target_duplicate_voxel_loss.detach(),
            "enable_sparsepcgc_empty_target_guard": pts_xyz.new_tensor(float(empty_target_guard_enabled)).detach(),
            "enable_sparsepcgc_target_duplicate_guard": pts_xyz.new_tensor(float(target_duplicate_guard_enabled)).detach(),
            "sparsepcgc_empty_target_guard_rejected_count": pts_xyz.new_tensor(float(empty_guard_rejected_count_value)).detach(),
            "sparsepcgc_target_duplicate_guard_rejected_count": pts_xyz.new_tensor(float(target_duplicate_guard_rejected_count_value)).detach(),
            "sparsepcgc_guard_rejected_count": pts_xyz.new_tensor(float(sparsepcgc_guard_rejected_count_value)).detach(),
            "sparsepcgc_move_existing_target_only": pts_xyz.new_tensor(float(getattr(self.args, "sparsepcgc_move_existing_target_only", False))).detach(),
            "repair_move_require_empty_target": pts_xyz.new_tensor(float(getattr(self.args, "repair_move_require_empty_target", True))).detach(),
            "repair_move_require_empty_target_effective": pts_xyz.new_tensor(float(require_empty_move)).detach(),
            "repair_move_max_points_per_voxel": pts_xyz.new_tensor(float(getattr(self.args, "repair_move_max_points_per_voxel", 8))).detach(),
            "repair_move_warmup": pts_xyz.new_tensor(float(move_warmup)).detach(),
            "target_move_ratio": pts_xyz.new_tensor(float(target_move_ratio)).detach(),
            "max_move_ratio": pts_xyz.new_tensor(float(max_move_ratio)).detach(),
            "repair_move_hard_threshold": pts_xyz.new_tensor(float(getattr(self.args, "repair_move_hard_threshold", 0.5))).detach(),
            "move_target_valid_ratio": move_target_valid.mean().detach(),
            "before_occupied_voxel_count": pts_xyz.new_tensor(float(before_occupied_voxels)).detach(),
            "after_occupied_voxel_count": pts_xyz.new_tensor(float(after_occupied_voxels)).detach(),
            "occupied_voxel_delta": pts_xyz.new_tensor(float(after_occupied_voxels - before_occupied_voxels)).detach(),
            "delete_target_voxel_count": pts_xyz.new_tensor(float(delete_target_voxel_count_value)).detach(),
            "delete_emptied_voxel_count": pts_xyz.new_tensor(float(delete_emptied_voxel_count_value)).detach(),
            "delete_removed_point_count": pts_xyz.new_tensor(float(delete_removed_point_count_value)).detach(),
            "hard_drop_ratio": hard_drop.mean().detach(),
            "hard_drop_count": pts_xyz.new_tensor(float(hard_drop_count_value)).detach(),
            "add_target_voxel_count": pts_xyz.new_tensor(float(add_target_voxel_count_value)).detach(),
            "add_actual_point_count": pts_xyz.new_tensor(float(add_effective_count_value)).detach(),
            "move_source_voxel_count": pts_xyz.new_tensor(float(move_source_voxel_count_value)).detach(),
            "move_target_voxel_count": pts_xyz.new_tensor(float(move_target_voxel_count_value)).detach(),
            "move_source_emptied_voxel_count": pts_xyz.new_tensor(float(move_source_emptied_voxel_count_value)).detach(),
            "move_target_new_voxel_count": pts_xyz.new_tensor(float(move_target_new_voxel_count_value)).detach(),
            "move_source_not_emptied_count": pts_xyz.new_tensor(float(move_source_not_emptied_count_value)).detach(),
            "moved_different_voxel_count": pts_xyz.new_tensor(float(moved_different_voxel_count_value)).detach(),
            "same_voxel_adjust_count": pts_xyz.new_tensor(float(same_voxel_adjust_count_value)).detach(),
            "preserve_ratio": preserve_ratio.detach(),
            "edit_reg": edit_reg.detach(),
            "drop_ratio": drop_ratio.detach(),
            "drop_ratio_soft": drop_ratio_soft.detach(),
            "drop_ratio_hard": drop_ratio_hard.detach(),
            "drop_ratio_soft_batch_mean": drop_ratio_soft_batch.detach().mean(),
            "drop_ratio_hard_batch_mean": drop_ratio_hard_batch.detach().mean(),
            "drop_amount_supervision_loss": drop_amount_supervision_loss.detach(),
            "drop_amount_soft_consistency_loss": drop_amount_soft_consistency_loss.detach(),

            "soft_drop_budget_mean": soft_drop_budget.detach().mean(),
            "valid_delete_voxel_count_mean": valid_delete_voxel_count.detach().mean(),

            # 点数基準のPrune量。
            "soft_drop_sum": soft_drop_prob_for_guard.detach().sum(),
            "hard_drop_sum": hard_drop.detach().sum(),
            "drop_soft_hard_sum_abs_diff": (
                soft_drop_prob_for_guard.detach().sum() - hard_drop.detach().sum()
            ).abs(),

            # Voxel数基準のPrune量。
            "soft_drop_voxel_sum": soft_drop_voxel_mass_per_batch.detach().sum(),
            "hard_drop_voxel_sum": hard_drop_voxel_mass_per_batch.detach().sum(),
            "drop_soft_hard_voxel_sum_abs_diff": (
                soft_drop_voxel_mass_per_batch.detach().sum()
                - hard_drop_voxel_mass_per_batch.detach().sum()
            ).abs(),

            "drop_soft_hard_ratio_abs_diff": (drop_ratio_soft.detach() - drop_ratio_hard.detach()).abs(),
            "keep_ratio": keep_prob.mean().detach(),
            "policy_chain_mean": p_chain.mean().detach(),
            "policy_sibling_mean": p_sibling.mean().detach(),
            "policy_parent_mean": p_parent.mean().detach(),
            "policy_context_mean": p_context.mean().detach(),
            "policy_comp_mean": p_comp.mean().detach(),
            "policy_outlier_mean": p_outlier.mean().detach(),
            "actuator_target_mode": actuator_target_mode, 
            "voxel_edit_state_enabled": pts_xyz.new_tensor(float(voxel_edit_state_enabled)).detach(),
            "voxel_edit_mode": voxel_edit_mode,
            "voxel_edit_initial_coords": voxel_edit_initial_coords,
            "voxel_edit_final_coords": voxel_edit_final_coords.detach(),
            "voxel_edit_final_weights": voxel_edit_final_weights.detach(),
            "voxel_edit_valid_mask": voxel_edit_valid_mask.detach(),
            "voxel_edit_initial_count": pts_xyz.new_tensor(float(voxel_edit_initial_count_value)).detach(),
            "voxel_edit_final_count": pts_xyz.new_tensor(float(voxel_edit_final_count_value)).detach(),
            "voxel_edit_drop_count": pts_xyz.new_tensor(float(voxel_edit_drop_count_value)).detach(),
            "voxel_edit_add_count": pts_xyz.new_tensor(float(voxel_edit_add_count_value)).detach(),
            "voxel_edit_move_count": pts_xyz.new_tensor(float(voxel_edit_move_count_value)).detach(),
            "voxel_edit_same_voxel_move_rejected": pts_xyz.new_tensor(float(voxel_edit_same_voxel_move_rejected_value)).detach(),
            "voxel_edit_existing_target_rejected": pts_xyz.new_tensor(float(voxel_edit_existing_target_rejected_value)).detach(),
            "voxel_edit_duplicate_target_rejected": pts_xyz.new_tensor(float(voxel_edit_duplicate_target_rejected_value)).detach(),
            "voxel_edit_child_slot_rejected": pts_xyz.new_tensor(float(voxel_edit_child_slot_rejected_value)).detach(),
            "voxel_edit_empty_target_rejected": pts_xyz.new_tensor(float(voxel_edit_empty_target_rejected_value)).detach(),
            "canonical_voxel_coords_before": canonical_voxel_coords_before,
            "canonical_voxel_coords_after": canonical_voxel_coords_after,
            "voxel_restore_meta": voxel_restore_meta,
            "restored_xyz_debug": restored_xyz_debug,
            "restore_info": restore_info,
        }
        if not bool(getattr(self.args, "retain_debug_tensors", False)):
            self.debug_tensors = {
                key: (value.detach() if torch.is_tensor(value) else value)
                for key, value in self.debug_tensors.items()
            }
        return pts_out, final_w, loss, {
            # Adjust Soft状態
            "learned_drop_ratio_requires_grad": pts_xyz.new_tensor(float(learned_drop_ratio.requires_grad)),
            "learned_add_ratio_requires_grad": pts_xyz.new_tensor(float(learned_add_ratio.requires_grad)),
            "learned_move_ratio_requires_grad": pts_xyz.new_tensor(float(learned_move_ratio.requires_grad)),
            "operation_amount_logit_loss": operation_amount_logit_loss.detach(),
            "drop_amount_logit_mean": drop_amount_logit.detach(),
            "add_amount_logit_mean": add_amount_logit.detach(),
            "move_amount_logit_mean": move_amount_logit.detach(),
            "drop_amount_target_logit": drop_amount_target_logit.detach(),
            "add_amount_target_logit": add_amount_target_logit.detach(),
            "move_amount_target_logit": move_amount_target_logit.detach(),

            "drop_amount_head_weight_requires_grad": pts_xyz.new_tensor(float(self.drop_amount_head.weight.requires_grad)),
            "add_amount_head_weight_requires_grad": pts_xyz.new_tensor(float(self.add_amount_head.weight.requires_grad)),
            "move_amount_head_weight_requires_grad": pts_xyz.new_tensor(float(self.move_amount_head.weight.requires_grad)),
            "move_source_soft_delete": move_source_soft_delete,
            "move_source_soft_keep": move_source_soft_keep,
            "move_target_soft_add": move_target_soft_add,
            "move_source_voxel_coords": voxel_coords,
            "move_target_voxel_coords": move_target_voxel_coords,
            "move_soft_score_effective": soft_move_score_effective,
            "move_soft_value_mode": "soft_prune_source_and_soft_add_target",
            "child_slot_candidate_ratio": child_slot_candidate_ratio,
            "repair_gate": repair_gate,
            "child_slot_candidate_ratio": child_slot_candidate_ratio,

            # Section4:
            # Network / train.py / CSVへ渡すためのleaf pattern actuator prior debug。
            "leaf_actuator_prior_enabled": pts_xyz.new_tensor(
                float(bool(leaf_actuator_prior.get("enabled", False)))
            ),
            "leaf_actuator_drop_prior_mean": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("delete_prior_mean", 0.0))
            ),
            "leaf_actuator_add_prior_mean": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("add_prior_mean", 0.0))
            ),
            "leaf_actuator_move_prior_mean": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("move_prior_mean", 0.0))
            ),
            "leaf_actuator_best_prior_mean": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("best_prior_mean", 0.0))
            ),
            "leaf_actuator_best_prior_max": pts_xyz.new_tensor(
                float(leaf_actuator_prior.get("best_prior_max", 0.0))
            ),

            # Section5:
            "leaf_target_direction_prior_enabled": pts_xyz.new_tensor(
                float(bool(leaf_target_direction_prior.get("enabled", False)))
            ),
            "leaf_add_target_match_ratio": pts_xyz.new_tensor(
                float(leaf_target_direction_prior.get("add_target_match_ratio", 0.0))
            ),
            "leaf_move_target_match_ratio": pts_xyz.new_tensor(
                float(leaf_target_direction_prior.get("move_target_match_ratio", 0.0))
            ),
            "leaf_add_target_bias_mean": pts_xyz.new_tensor(
                float(leaf_target_direction_prior.get("add_target_bias_mean", 0.0))
            ),
            "leaf_move_target_bias_mean": pts_xyz.new_tensor(
                float(leaf_target_direction_prior.get("move_target_bias_mean", 0.0))
            ),

            "repair_gate": repair_gate,
            "drop_prob": drop_prob,
            "keep_prob": keep_prob,
            "drop_prob_direct": drop_prob_direct,
            "drop_prob_proxy": drop_prob_proxy,
            "drop_logit": learned_drop_logit,
            "learned_drop_logit": learned_drop_logit,
            "soft_drop_where_grad_base": soft_drop_where_grad_base,
            "prune_where_proxy": soft_drop_where_grad_base,
            "soft_drop_prob_for_ste": soft_drop_prob_for_ste,
            "soft_drop_where_grad_masked": soft_drop_where_grad_masked,
            "soft_drop_where_grad_direct": soft_drop_where_grad_direct,
            "add_prob": add_prob,
            "add_priority": add_priority,
            "add_ratio": add_ratio,

            # Phase7-3: soft ratioも既に計算済みのratio値を使う。
            # mean(score)とratioがズレるとdebug解釈が混乱するため、ratio系はratio変数に統一する。
            "drop_ratio_soft": drop_ratio_soft.detach() if torch.is_tensor(drop_ratio_soft) else pts_xyz.new_tensor(0.0),
            "add_ratio_soft": add_ratio_soft.detach() if torch.is_tensor(add_ratio_soft) else pts_xyz.new_tensor(0.0),
            "move_ratio_soft": move_ratio_soft.detach() if torch.is_tensor(move_ratio_soft) else pts_xyz.new_tensor(0.0),

            # Phase7-3: hard ratioは既に計算済みの値を使う。
            # add_target_mask / move_source_mask という変数はこの実装には存在しないため使わない。
            "drop_ratio_hard": drop_ratio_hard.detach() if torch.is_tensor(drop_ratio_hard) else pts_xyz.new_tensor(0.0),
            "add_ratio_hard": add_ratio_hard.detach() if torch.is_tensor(add_ratio_hard) else pts_xyz.new_tensor(0.0),
            "move_ratio_hard": move_ratio_hard.detach() if torch.is_tensor(move_ratio_hard) else pts_xyz.new_tensor(0.0),

            "voxel_soft_drop_mean": torch.nan_to_num(drop_prob_proxy.float().mean(), nan=0.0, posinf=0.0, neginf=0.0).detach(),
            "voxel_soft_add_mean": torch.nan_to_num(add_target_soft_add.float().mean(), nan=0.0, posinf=0.0, neginf=0.0).detach(),
            "voxel_soft_move_mean": torch.nan_to_num(soft_move_score_effective.float().mean(), nan=0.0, posinf=0.0, neginf=0.0).detach(),

            "add_ratio_loss_value": add_ratio_loss.detach() if torch.is_tensor(add_ratio_loss) else pts_xyz.new_tensor(0.0),
            # Phase7-3: Add単体のhard/soft整合性debug。
            # add_consistency_loss という変数は存在しないため、既存の add_hard_soft_consistency_loss を使う。
            "add_consistency_loss_value": add_hard_soft_consistency_loss.detach() if torch.is_tensor(add_hard_soft_consistency_loss) else pts_xyz.new_tensor(0.0),
            # Phase7-3: Add Amount head側のsoft ratio整合性debug。
            "add_amount_consistency_loss_value": add_amount_soft_consistency_loss.detach() if torch.is_tensor(add_amount_soft_consistency_loss) else pts_xyz.new_tensor(0.0),
            "add_prob_mean": add_prob.mean(),
            "add_prob_max": add_prob.max() if add_prob.numel() > 0 else pts_xyz.new_zeros(()),
            "add_priority_mean": add_priority.mean(),
            "add_priority_max": add_priority.max() if add_priority.numel() > 0 else pts_xyz.new_zeros(()),
            "add_count": add_count_value,
            "add_effective_count": add_effective_count_value,
            "add_candidate_ratio": float(add_candidate_ratio),
            "add_candidate_count": int(add_k),
            "add_ratio_soft": add_ratio_soft.detach(),
            "add_ratio_hard": add_ratio_hard.detach(),
            "add_amount_supervision_loss": add_amount_supervision_loss.detach(),
            "add_amount_soft_consistency_loss": add_amount_soft_consistency_loss.detach(),
            "actual_oracle_drop_amount_loss": actual_oracle_drop_amount_loss.detach(),
            "actual_oracle_add_amount_loss": actual_oracle_add_amount_loss.detach(),
            "actual_oracle_move_amount_loss": actual_oracle_move_amount_loss.detach(),
            "actual_oracle_drop_amount_logit_loss": actual_oracle_drop_amount_logit_loss.detach(),
            "actual_oracle_add_amount_logit_loss": actual_oracle_add_amount_logit_loss.detach(),
            "actual_oracle_amount_supervision_loss": actual_oracle_amount_supervision_loss.detach(),
            "add_soft_budget_mean": soft_add_budget.detach().mean(),
            "add_soft_pair_sum": soft_add_pair.detach().sum(),
            "add_hard_pair_sum": hard_add_pair.detach().sum(),
            "add_soft_hard_sum_abs_diff": (soft_add_pair.detach().sum() - hard_add_pair.detach().sum()).abs(),
            "add_soft_hard_ratio_abs_diff": (add_ratio_soft.detach() - add_ratio_hard.detach()).abs(),
            # target voxel単位のAdd Hard/Soft状態。
            "add_target_soft_add": add_target_soft_add,
            "add_target_hard_add": add_target_hard_add,
            "add_target_add_st": add_target_add_st,
            "add_target_voxel_coords": add_target_voxel_coords,
            "add_target_soft_add_sum": add_target_soft_add.detach().sum(),
            "add_target_hard_add_sum": add_target_hard_add.detach().sum(),
            "add_target_soft_hard_sum_abs_diff": (
                add_target_soft_add.detach().sum()
                - add_target_hard_add.detach().sum()
            ).abs(),
            "operation_gate_prob": operation_gate_prob,
            "operation_gate_hard": operation_gate_hard.detach(),
            "operation_gate_logit": operation_gate_logit,
            "drop_operation_gate": drop_operation_gate.mean(),
            "add_operation_gate": add_operation_gate.mean(),
            "move_operation_gate": move_operation_gate.mean(),
            "operation_gate_oracle_loss": operation_gate_oracle_loss.detach(),
            "actual_oracle_candidate_where_loss": actual_oracle_candidate_where_loss.detach(),
            "actual_oracle_direction_supervision_loss": actual_oracle_direction_supervision_loss.detach(),
            "actual_oracle_bad_candidate_count": int(leaf_operation_masks.get("actual_oracle_bad_candidate_count", 0)),
            "actual_oracle_improving_candidate_count": int(leaf_operation_masks.get("actual_oracle_improving_candidate_count", 0)),
            "actual_oracle_combo_extra_count": int(leaf_operation_masks.get("actual_oracle_combo_extra_count", 0)),
            "actual_oracle_generated_candidate_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_generated_candidate_count", 0))
            ).detach(),
            "actual_oracle_accepted_candidate_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_candidate_count", 0))
            ).detach(),
            "actual_oracle_accepted_prune_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_prune_count", 0))
            ).detach(),
            "actual_oracle_accepted_add_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_add_count", 0))
            ).detach(),
            "actual_oracle_accepted_adjust_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_adjust_count", 0))
            ).detach(),
            "actual_oracle_accepted_subtree_move_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_subtree_move_count", 0))
            ).detach(),
            "actual_oracle_accepted_parent_collapse_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_parent_collapse_count", 0))
            ).detach(),
            "actual_oracle_accepted_pattern_canonicalize_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_accepted_pattern_canonicalize_count", 0))
            ).detach(),
            "actual_oracle_noop_label_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_noop_label_count", 0))
            ).detach(),
            "actual_oracle_noop_label_weight": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_noop_label_weight", 0.0))
            ).detach(),
            "actual_oracle_high_rate_mppov_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_high_rate_mppov_count", 0))
            ).detach(),
            "actual_oracle_low_prob_occupied_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_low_prob_occupied_count", 0))
            ).detach(),
            "actual_oracle_single_child_chain_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_single_child_chain_count", 0))
            ).detach(),
            "actual_oracle_context_pattern_candidate_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_context_pattern_candidate_count", 0))
            ).detach(),
            "actual_oracle_eval_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_eval_count", 0))
            ).detach(),
            "actual_oracle_eval_max": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_eval_max", 0))
            ).detach(),
            "actual_oracle_time": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_time", 0.0))
            ).detach(),
            "actual_oracle_drop_bad_count": actual_oracle_drop_bad_count_value,
            "actual_oracle_add_bad_count": actual_oracle_add_bad_count_value,
            "actual_oracle_move_bad_count": actual_oracle_move_bad_count_value,
            "actual_oracle_drop_reason": str(leaf_operation_masks.get("actual_oracle_drop_reason", "")),
            "actual_oracle_operation": str(leaf_operation_masks.get("actual_oracle_operation", "")),
            "actual_oracle_scheduled_operation": str(
                leaf_operation_masks.get("actual_oracle_scheduled_operation", "")
            ),
            "actual_oracle_edit_record_bits": float(actual_oracle_edit_record_bits_value),
            "actual_oracle_raw_percent": float(actual_oracle_raw_percent_value),
            "actual_oracle_delta_actual_percent": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_delta_actual_percent", 0.0))
            ).detach(),
            "actual_oracle_proxy_percent": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_proxy_percent", 0.0))
            ).detach(),
            "actual_oracle_geometry_percent": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_geometry_percent", 0.0))
            ).detach(),
            "actual_oracle_original_actual_bits": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_original_actual_bits", 0.0))
            ).detach(),
            "actual_oracle_edited_actual_bits": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_edited_actual_bits", 0.0))
            ).detach(),
            "actual_oracle_fast_diagnostic_used": pts_xyz.new_tensor(
                float(bool(leaf_operation_masks.get("actual_oracle_fast_diagnostic_used", False)))
            ).detach(),
            "actual_oracle_fast_diagnostic_full_drop_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_fast_diagnostic_full_drop_count", 0))
            ).detach(),
            "actual_oracle_fast_diagnostic_local_drop_count": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_fast_diagnostic_local_drop_count", 0))
            ).detach(),
            "actual_oracle_fast_diagnostic_full_drop_ratio": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_fast_diagnostic_full_drop_ratio", 0.0))
            ).detach(),
            "actual_oracle_fast_diagnostic_local_drop_ratio": pts_xyz.new_tensor(
                float(leaf_operation_masks.get("actual_oracle_fast_diagnostic_local_drop_ratio", 0.0))
            ).detach(),
            "raw_learned_drop_ratio": raw_learned_drop_ratio.mean(),
            "raw_learned_add_ratio": raw_learned_add_ratio.mean(),
            "raw_learned_move_ratio": raw_learned_move_ratio.mean(),
            "learned_drop_ratio": learned_drop_ratio.mean(),
            "learned_drop_prob": learned_drop_prob,
            "drop_prob_mean": drop_prob.mean(),
            "drop_ratio_soft": drop_ratio_soft.detach(),
            "drop_ratio_hard": drop_ratio_hard.detach(),
            "drop_ratio_soft_batch_mean": drop_ratio_soft_batch.detach().mean(),
            "drop_ratio_hard_batch_mean": drop_ratio_hard_batch.detach().mean(),
            "drop_amount_supervision_loss": drop_amount_supervision_loss.detach(),
            "drop_amount_soft_consistency_loss": drop_amount_soft_consistency_loss.detach(),

            "soft_drop_budget_mean": soft_drop_budget.detach().mean(),
            "valid_delete_voxel_count_mean": valid_delete_voxel_count.detach().mean(),

            # 点数基準のPrune量。
            "soft_drop_sum": soft_drop_prob_for_guard.detach().sum(),
            "hard_drop_sum": hard_drop.detach().sum(),
            "drop_soft_hard_sum_abs_diff": (
                soft_drop_prob_for_guard.detach().sum() - hard_drop.detach().sum()
            ).abs(),

            # Voxel数基準のPrune量。
            "soft_drop_voxel_sum": soft_drop_voxel_mass_per_batch.detach().sum(),
            "hard_drop_voxel_sum": hard_drop_voxel_mass_per_batch.detach().sum(),
            "drop_soft_hard_voxel_sum_abs_diff": (
                soft_drop_voxel_mass_per_batch.detach().sum()
                - hard_drop_voxel_mass_per_batch.detach().sum()
            ).abs(),

            "drop_soft_hard_ratio_abs_diff": (drop_ratio_soft.detach() - drop_ratio_hard.detach()).abs(),
            "drop_prob_min": drop_prob.amin() if drop_prob.numel() > 0 else pts_xyz.new_zeros(()),
            "drop_prob_max": drop_prob.amax() if drop_prob.numel() > 0 else pts_xyz.new_zeros(()),
            "drop_prob_direct_mean": drop_prob_direct.mean(),
            "drop_prob_direct_min": drop_prob_direct.amin() if drop_prob_direct.numel() > 0 else pts_xyz.new_zeros(()),
            "drop_prob_direct_max": drop_prob_direct.amax() if drop_prob_direct.numel() > 0 else pts_xyz.new_zeros(()),
            "drop_prob_proxy_mean": drop_prob_proxy.mean(),
            "drop_prob_proxy_min": drop_prob_proxy.amin() if drop_prob_proxy.numel() > 0 else pts_xyz.new_zeros(()),
            "drop_prob_proxy_max": drop_prob_proxy.amax() if drop_prob_proxy.numel() > 0 else pts_xyz.new_zeros(()),
            "drop_logit_mean": learned_drop_logit.mean(),
            "drop_logit_min": learned_drop_logit.amin() if learned_drop_logit.numel() > 0 else pts_xyz.new_zeros(()),
            "drop_logit_max": learned_drop_logit.amax() if learned_drop_logit.numel() > 0 else pts_xyz.new_zeros(()),
            "keep_prob_mean": keep_prob.mean(),
            "keep_prob_min": keep_prob.amin() if keep_prob.numel() > 0 else pts_xyz.new_zeros(()),
            "keep_prob_max": keep_prob.amax() if keep_prob.numel() > 0 else pts_xyz.new_zeros(()),
            "drop_entropy": drop_entropy,
            "soft_drop_mass": soft_drop_mass,
            "selected_drop_count_hard": pts_xyz.new_tensor(float(hard_drop_count_value)),
            "prune_soft_geom": prune_soft_geom,
            "prune_soft_rate": prune_soft_rate,
            "prune_soft_node": prune_soft_node,
            "prune_soft_single": prune_soft_single,
            "prune_soft_bit": prune_soft_bit,
            "drop_direct_target_loss": drop_direct_target_loss,
            "learned_drop_ratio_std": learned_drop_ratio.float().std(unbiased=False),
            "learned_add_ratio": learned_add_ratio.mean(),
            "learned_add_ratio_std": learned_add_ratio.float().std(unbiased=False),
            "learned_move_ratio": learned_move_ratio.mean(),
            "learned_move_ratio_std": learned_move_ratio.float().std(unbiased=False),
            "operation_amount_consistency_loss": operation_amount_consistency_loss,
            "operation_entropy": operation_entropy,
            "operation_entropy_loss": operation_entropy_loss,
            "operation_entropy_weight_effective": pts_xyz.new_tensor(
                float(operation_entropy_weight_effective)
            ),
            "move_ratio_soft": move_ratio_soft.detach(),
            "move_ratio_hard": move_ratio_hard.detach(),
            "move_ratio_soft_batch_mean": move_ratio_soft_batch.detach().mean(),
            "move_ratio_hard_batch_mean": move_ratio_hard_batch.detach().mean(),
            "move_amount_supervision_loss": move_amount_supervision_loss.detach(),
            "move_amount_soft_consistency_loss": move_amount_soft_consistency_loss.detach(),

            "soft_move_budget_mean": soft_move_budget.detach().mean(),
            "valid_move_source_voxel_count_mean": valid_move_source_voxel_count.detach().mean(),
            "valid_move_source_voxel_count_effective_mean": valid_move_source_voxel_count_effective.detach().mean(),

            # 点数基準のAdjust量。
            "soft_move_sum": soft_move_score_for_guard.detach().sum(),
            "hard_move_sum": hard_move.detach().sum(),
            "move_soft_hard_sum_abs_diff": (
                soft_move_score_for_guard.detach().sum() - hard_move.detach().sum()
            ).abs(),

            # Voxel数基準のAdjust量。
            "soft_move_voxel_sum": soft_move_voxel_mass_per_batch.detach().sum(),
            "hard_move_voxel_sum": hard_move_voxel_mass_per_batch.detach().sum(),
            "move_soft_hard_voxel_sum_abs_diff": (
                soft_move_voxel_mass_per_batch.detach().sum()
                - hard_move_voxel_mass_per_batch.detach().sum()
            ).abs(),

            "move_soft_hard_ratio_abs_diff": (
                move_ratio_soft.detach() - move_ratio_hard.detach()
            ).abs(),
            "soft_activity_loss": soft_activity_loss,
            "subtree_move_source_prob_mean": subtree_move_source_prob.mean(),
            "subtree_move_source_prob_max": subtree_move_source_prob.amax() if subtree_move_source_prob.numel() > 0 else pts_xyz.new_zeros(()),
            "move_direction_ce": move_direction_ce,
            "add_direction_ce": add_direction_ce,
            "temperature": float(operation_temperature),
            "actuator_voxel_mode": actuator_voxel_mode,
            "local_recomputed": bool(actuator_local_recomputed),
            "exploration_noise": float(exploration_noise),
            "operation_prob_floor_applied": bool(add_ratio_floor_applied),
            "move_score_noise": float(move_score_noise),
            "sparsepcgc_add_experiment_enabled": bool(sparsepcgc_add_experiment_active),
            "sparsepcgc_add_warmup": float(self._sparsepcgc_add_warmup()),
            "add_score_noise": float(add_score_noise),
            "add_weight_random_mix": float(add_weight_random_mix),
            "drop_score_noise": float(drop_score_noise),
            "drop_random_mix": float(drop_random_mix),
            "add_enabled": bool(add_enabled),
            "prune_enabled": bool(prune_enabled),
            "disp_enabled": bool(disp_enabled),
            "actuator_stage": stage,
            "actuator_stage_raw": stage_raw,
            "actuator_strength": float(actuator_strength),
            "force_joint_actuator": bool(force_joint_actuator),
            "threshold_cap_mode": bool(threshold_cap_mode),
            "delta": delta,
            "point_soft_delta_debug": point_soft_delta_debug,
            "primitive_delta": primitive_delta,
            "move_ratio": hard_move.mean(),
            "hard_move_count": hard_move_count_value,
            "move_score_mean": move_score.mean(),
            "move_source_prior_mean": move_source_prior.mean(),
            "adjusted_point_count": hard_move_count_value,
            "adjusted_point_rate": adjusted_point_rate_value,
            "raw_hard_move_count_before_sparsepcgc_guard": raw_hard_move_count_value,
            "source_unique_voxel_count": sparsepcgc_source_unique_voxel_count_value,
            "target_unique_voxel_count": sparsepcgc_target_unique_voxel_count_value,
            "target_duplicate_voxel_count": sparsepcgc_target_duplicate_voxel_count_value,
            "target_voxel_duplicate_rate": sparsepcgc_target_duplicate_rate_value,
            "target_existing_occupied_count": target_existing_occupied_count_value,
            "target_existing_occupied_rate": target_existing_occupied_rate_value,
            "target_empty_voxel_count": target_empty_voxel_count_value,
            "target_empty_voxel_rate": target_empty_voxel_rate_value,
            "empty_target_violation_loss": empty_target_violation_loss,
            "target_duplicate_voxel_loss": target_duplicate_voxel_loss,
            "enable_sparsepcgc_empty_target_guard": empty_target_guard_enabled,
            "enable_sparsepcgc_target_duplicate_guard": target_duplicate_guard_enabled,
            "sparsepcgc_empty_target_guard_rejected_count": empty_guard_rejected_count_value,
            "sparsepcgc_target_duplicate_guard_rejected_count": target_duplicate_guard_rejected_count_value,
            "sparsepcgc_guard_rejected_count": sparsepcgc_guard_rejected_count_value,
            "sparsepcgc_move_existing_target_only": bool(getattr(self.args, "sparsepcgc_move_existing_target_only", False)),
            "repair_move_require_empty_target": bool(getattr(self.args, "repair_move_require_empty_target", True)),
            "repair_move_require_empty_target_effective": bool(require_empty_move),
            "repair_move_max_points_per_voxel": int(getattr(self.args, "repair_move_max_points_per_voxel", 8)),
            "repair_move_warmup": float(move_warmup),
            "target_move_ratio": float(target_move_ratio),
            "max_move_ratio": float(max_move_ratio),
            "repair_move_hard_threshold": float(getattr(self.args, "repair_move_hard_threshold", 0.5)),
            "move_target_valid_ratio": move_target_valid.mean(),
            "moved_delta_mean": moved_delta_mean,
            "before_occupied_voxel_count": before_occupied_voxels,
            "after_occupied_voxel_count": after_occupied_voxels,
            "occupied_voxel_delta": after_occupied_voxels - before_occupied_voxels,
            "delete_target_voxel_count": delete_target_voxel_count_value,
            "delete_emptied_voxel_count": delete_emptied_voxel_count_value,
            "delete_removed_point_count": delete_removed_point_count_value,
            "hard_drop_ratio": hard_drop.mean(),
            "hard_drop_count": hard_drop_count_value,
            "add_target_voxel_count": add_target_voxel_count_value,
            "add_actual_point_count": add_effective_count_value,
            "move_source_voxel_count": move_source_voxel_count_value,
            "move_target_voxel_count": move_target_voxel_count_value,
            "move_source_emptied_voxel_count": move_source_emptied_voxel_count_value,
            "move_target_new_voxel_count": move_target_new_voxel_count_value,
            "move_source_not_emptied_count": move_source_not_emptied_count_value,
            "moved_different_voxel_count": moved_different_voxel_count_value,
            "same_voxel_adjust_count": same_voxel_adjust_count_value,
            "preserve_ratio": preserve_ratio,
            "edit_reg": edit_reg,
            "ratio_loss": ratio_loss,
            "shape_guard": shape_guard,
            "drop_ratio_loss": drop_ratio_loss,
            "drop_cap_loss": drop_cap_loss,
            "drop_shape_guard": drop_shape_guard,
            "add_ratio_loss": add_ratio_loss,
            "add_shape_guard": add_shape_guard,
            "add_offset_reg": add_offset_reg,
            "add_drop_conflict_loss": add_drop_conflict_loss,
            "added_keep_loss": added_keep_loss,
            "add_min_offset_loss": add_min_offset_loss,
            "quant_move_conflict_loss": quant_move_conflict_loss,
            "quant_add_guard": quant_add_guard,
            "local_edit_guard": local_edit_guard,
            "point_parent_node_ids": point_parent_node_ids,

            # Phase3: occupied voxel集合としての編集状態。
            # final_voxel_coords / final_voxel_weights は点対応ではなくoccupied voxel対応へ切り替える。
            "voxel_edit_state_enabled": bool(voxel_edit_state_enabled),
            "voxel_edit_mode": voxel_edit_mode,
            "initial_voxel_coords": voxel_edit_initial_coords,
            "final_voxel_coords": voxel_edit_final_coords,
            "final_voxel_weights": voxel_edit_final_weights,
            "final_voxel_valid_mask": voxel_edit_valid_mask,
            "voxel_step": voxel_step,
            "voxel_offset": voxel_offset,

            # 既存の点対応Voxel状態も必要な場合に参照できるように残す。
            "point_aligned_initial_voxel_coords": voxel_coords,
            "point_aligned_final_voxel_coords": final_voxel_coords,
            "point_aligned_final_voxel_weights": final_w,

            "voxel_edit_initial_count": voxel_edit_initial_count_value,
            "voxel_edit_final_count": voxel_edit_final_count_value,
            "voxel_edit_drop_count": voxel_edit_drop_count_value,
            "voxel_edit_add_count": voxel_edit_add_count_value,
            "voxel_edit_move_count": voxel_edit_move_count_value,
            "input_voxel_count": voxel_edit_initial_count_value,
            "final_voxel_count": voxel_edit_final_count_value,
            "before_occupied_voxel_count_for_stats": before_occupied_voxels,
            "point_aligned_after_occupied_voxel_count": point_aligned_after_occupied_voxels,
            "voxel_edit_same_voxel_move_rejected": voxel_edit_same_voxel_move_rejected_value,
            "voxel_edit_existing_target_rejected": voxel_edit_existing_target_rejected_value,
            "voxel_edit_duplicate_target_rejected": voxel_edit_duplicate_target_rejected_value,
            "voxel_edit_child_slot_rejected": voxel_edit_child_slot_rejected_value,
            "voxel_edit_empty_target_rejected": voxel_edit_empty_target_rejected_value,
            "estimated_edit_record_bits": float(actual_oracle_edit_record_bits_value),
            "estimated_edit_record_raw_percent": float(actual_oracle_raw_percent_value),
            "canonical_voxel_coords_before": canonical_voxel_coords_before,
            # Phase7-2: full-context / full-cloud correctionへ渡す微分可能なsoft編集量。
            # hard mask / final_voxel_coords.long() ではなく、既存のsoft probability / score / amountを使う。
            # ここはloss接続用なのでdetachしない。
            "voxel_soft_drop_score": torch.nan_to_num(drop_prob_proxy.float().mean(), nan=0.0, posinf=0.0, neginf=0.0),
            "voxel_soft_add_score": torch.nan_to_num(add_target_soft_add.float().mean(), nan=0.0, posinf=0.0, neginf=0.0),
            "voxel_soft_move_score": torch.nan_to_num(soft_move_score_effective.float().mean(), nan=0.0, posinf=0.0, neginf=0.0),

            "voxel_soft_drop_amount": torch.nan_to_num(learned_drop_ratio.float().mean(), nan=0.0, posinf=0.0, neginf=0.0),
            "voxel_soft_add_amount": torch.nan_to_num(learned_add_ratio.float().mean(), nan=0.0, posinf=0.0, neginf=0.0),
            "voxel_soft_move_amount": torch.nan_to_num(learned_move_ratio.float().mean(), nan=0.0, posinf=0.0, neginf=0.0),

            "voxel_soft_edit_score": torch.nan_to_num(
                (
                    drop_prob_proxy.float().mean()
                    + add_target_soft_add.float().mean()
                    + soft_move_score_effective.float().mean()
                ) / 3.0,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),

            "voxel_soft_edit_count_proxy": torch.nan_to_num(
                soft_drop_mass
                + add_target_soft_add.float().sum()
                + soft_move_score_effective.float().sum(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "canonical_voxel_coords_after": canonical_voxel_coords_after,
            "voxel_restore_meta": voxel_restore_meta,
            "restored_xyz_debug": restored_xyz_debug,
            "restore_info": restore_info,
            "repair_output_voxel_restored_points": bool(voxel_restored_output_enabled),
            "final_voxel_update_mode": "occupied_voxel_edit_state_phase3",
            "final_voxel_recomputed_from_pts_out": False,
            "point_child_slots": point_child_slots,
            "point_valid_empty_child_mask": point_valid_empty_child_mask, 
            "full_octree_context_available": bool(full_context_available),
            "actuator_parent_occupancy_code": int(actuator_parent_occupancy_code),
            "actuator_sibling_count": int(actuator_sibling_count),
            "actuator_ancestor_count": int(actuator_ancestor_count),
            "full_context_bonus_mean": full_context_bonus.mean(),
        }
