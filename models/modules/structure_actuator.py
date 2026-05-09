import math

import torch
import torch.nn as nn


class StructureRepairActuator(nn.Module):
    """Apply small geometry-preserving movements that realize repair policies.

    The actuator predicts both a small displacement and a point-wise keep/drop
    gate.  This lets the downstream compression loss reach actual deletion
    decisions instead of only moving every point.
    """

    def __init__(self, in_channels, hidden_dim=64, args=None):
        super().__init__()
        self.args = args
        self.offset_head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 3, 1),
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
        self.add_dir_head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 3, 1),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        nn.init.zeros_(self.drop_head[-1].weight)
        nn.init.zeros_(self.add_head[-1].weight)
        nn.init.zeros_(self.add_dir_head[-1].weight)
        nn.init.zeros_(self.add_dir_head[-1].bias)
        target_repair_ratio = float(
            getattr(self.args, "target_repair_ratio", getattr(self.args, "target_disp_ratio", 0.20))
        )
        target_drop_ratio = float(getattr(self.args, "target_drop_ratio", 0.01))
        init_drop = target_drop_ratio / max(target_repair_ratio, 1e-6)
        init_drop = min(max(init_drop, 1e-4), 0.95)
        init_drop_bias = math.log(init_drop / max(1.0 - init_drop, 1e-6))
        nn.init.constant_(self.drop_head[-1].bias, init_drop_bias)
        target_add_ratio = min(max(float(getattr(self.args, "target_add_ratio", 0.01)), 1e-4), 0.95)
        init_add_bias = math.log(target_add_ratio / max(1.0 - target_add_ratio, 1e-6))
        nn.init.constant_(self.add_head[-1].bias, init_add_bias)
        self.debug_tensors = {}

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
            return max(
                float(getattr(self.args, "sparsepcgc_effective_qs", 0.0))
                or float(getattr(self.args, "sparsepcgc_voxel_size", 1.0)) * float(getattr(self.args, "sparsepcgc_pos_quantscale", 1)),
                1e-9,
            )
        if compress_key in {"gpcc", "gpcctmc3"}:
            return max(float(getattr(self.args, "gpcc_effective_qs", getattr(self.args, "qs", 1.0))), 1e-9)
        return max(float(getattr(self.args, "qs", 2.0)), 1e-9)

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

    @staticmethod
    def _clip_vector(delta, max_norm):
        norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1e-12)
        scale = (max_norm / norm).clamp_max(1.0)
        return delta * scale

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

    def _target_add_count(self, point_count):
        if point_count <= 0 or not bool(getattr(self.args, "add", True)):
            return 0
        target_ratio = max(float(getattr(self.args, "target_add_ratio", 0.01)), 0.0)
        max_ratio = max(float(getattr(self.args, "max_add_ratio", max(target_ratio, 0.0))), 0.0)
        if target_ratio <= 0.0 or max_ratio <= 0.0:
            return 0
        max_add_points = max(1, int(max_ratio * float(point_count)))
        add_points = max(1, int(target_ratio * float(point_count)))
        return min(add_points, max_add_points, point_count)

    @staticmethod
    def _mask_add_scores(add_scores, selection_mask):
        if selection_mask is None:
            return add_scores
        valid = selection_mask.squeeze(1) if selection_mask.ndim == 3 else selection_mask
        valid = valid.to(device=add_scores.device, dtype=torch.bool)
        masked = add_scores.masked_fill(~valid, -1.0e6)
        all_invalid = ~valid.any(dim=1)
        if bool(all_invalid.any().item()):
            masked = torch.where(all_invalid[:, None], add_scores, masked)
        return masked

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
    ):
        snap_strength = float(getattr(self.args, "repair_snap_strength", getattr(self.args, "disp_snap_strength", 0.35)))
        max_offset = self._max_offset(pts_xyz, coord_scale)
        stage = str(getattr(self.args, "training_stage", "joint")).strip().lower()
        if stage == "diagnosis":
            actuator_strength = float(getattr(self.args, "diagnosis_actuator_strength", 0.1))
        else:
            actuator_strength = float(getattr(self.args, "repair_actuator_strength", 1.0))

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
        if bool(getattr(self.args, "repair_priority_gate", True)) and repair_priority is not None:
            priority = repair_priority.to(device=pts_xyz.device, dtype=pts_xyz.dtype).clamp(0.0, 1.0)
            priority_gate = self._priority_topk_gate(
                priority,
                target_ratio=max(target_ratio, 1e-4),
                tau=float(getattr(self.args, "repair_priority_gate_tau", 0.08)),
            )
            repair_gate = base_repair_gate * priority_gate
        else:
            repair_gate = base_repair_gate
        if bool(getattr(self.args, "repair_gate_mean_cap", True)):
            gate_mean = self._masked_mean(repair_gate, selection_mask).detach().clamp_min(1e-6)
            gate_scale = (target_ratio / gate_mean).clamp_max(1.0)
            repair_gate = repair_gate * gate_scale

        node_score = cause_scores[:, 0:1, :]
        single_score = cause_scores[:, 1:2, :]
        lowprob_score = cause_scores[:, 2:3, :] if cause_scores.shape[1] > 2 else preserve.new_zeros(preserve.shape)
        shape_idx = 6 if cause_scores.shape[1] > 6 else 5
        shape_score = cause_scores[:, shape_idx:shape_idx+1, :]

        delete_prior = (
            0.95 * p_outlier
            + 0.75 * p_chain
            + 0.55 * p_sibling
            + 0.45 * p_parent
            + 0.20 * p_context
            + 0.25 * node_score
            + 0.25 * single_score
            + 0.15 * lowprob_score
            - 0.85 * preserve
            - 0.75 * shape_score
        )
        delete_prior = torch.sigmoid(delete_prior.clamp(-8.0, 8.0))
        learned_drop = torch.sigmoid(self.drop_head(actuator_features))
        drop_prob = (repair_gate * delete_prior * learned_drop).clamp(0.0, 1.0)
        keep_prob = (1.0 - drop_prob).clamp(1e-4, 1.0)

        snap_delta = structure["snap_delta"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
        learned_delta = torch.tanh(self.offset_head(actuator_features)) * max_offset
        motion_gate = repair_gate * keep_prob

        # Give each named policy a genuinely different geometric primitive so
        # the diagnosis/policy path is not only cosmetic.  All paths remain
        # small and geometry-preserving, but they bias edits differently.
        delta_chain = snap_strength * 1.20 * snap_delta
        delta_sibling = snap_strength * 0.60 * snap_delta + 0.40 * learned_delta
        delta_parent = snap_strength * 0.90 * snap_delta + 0.20 * learned_delta
        delta_context = 0.85 * learned_delta
        delta_comp = 0.55 * learned_delta - 0.20 * snap_strength * snap_delta
        delta_outlier = 0.15 * learned_delta

        primitive_delta = (
            p_chain * delta_chain
            + p_sibling * delta_sibling
            + p_parent * delta_parent
            + p_context * delta_context
            + p_comp * delta_comp
            + p_outlier * delta_outlier
        )
        raw_delta = actuator_strength * motion_gate * primitive_delta
        delta = self._clip_vector(raw_delta, max_offset)
        pts_out = pts_xyz + delta

        final_w = keep_prob
        B, _, N = pts_xyz.shape
        add_k = self._target_add_count(N)
        add_ratio = pts_xyz.new_zeros(())
        add_ratio_loss = pts_xyz.new_zeros(())
        add_shape_guard = pts_xyz.new_zeros(())
        add_offset_reg = pts_xyz.new_zeros(())
        add_prob = pts_xyz.new_zeros((B, 1, N))
        add_priority = add_prob
        add_count_value = 0
        if add_k > 0:
            learned_add_logit = self.add_head(actuator_features)
            add_prior = (
                0.80 * p_sibling
                + 0.70 * p_parent
                + 0.65 * p_context
                + 0.70 * p_comp
                + 0.55 * node_score
                + 0.45 * lowprob_score
                + 0.35 * single_score
                - 0.90 * preserve
                - 0.60 * p_outlier
                - 0.65 * shape_score
            )
            add_priority = torch.sigmoid((learned_add_logit + add_prior).clamp(-8.0, 8.0))
            add_scores = self._mask_add_scores((learned_add_logit + add_prior).squeeze(1), selection_mask)
            add_idx = torch.topk(add_scores.detach(), k=add_k, dim=1, largest=True, sorted=False).indices
            add_idx = torch.sort(add_idx, dim=1).values
            hard_add_mask = torch.zeros_like(add_scores)
            hard_add_mask.scatter_(1, add_idx, 1.0)

            tau = max(float(getattr(self.args, "add_soft_match_tau", 0.05)), 1e-6)
            threshold = torch.gather(add_scores, 1, add_idx[:, -1:].detach())
            soft_add_mask = torch.sigmoid((add_scores - threshold) / tau)
            hard_ratio = hard_add_mask.mean(dim=1, keepdim=True)
            soft_mean = soft_add_mask.mean(dim=1, keepdim=True).detach().clamp_min(1e-12)
            soft_add_mask = (soft_add_mask * (hard_ratio / soft_mean)).clamp(0.0, 1.0)
            add_mask_st = hard_add_mask - soft_add_mask.detach() + soft_add_mask
            add_prob = add_mask_st.unsqueeze(1)

            learned_add_delta = torch.tanh(self.add_dir_head(actuator_features)) * max_offset
            center_dir = pts_out - pts_out.mean(dim=2, keepdim=True)
            center_norm = torch.linalg.norm(center_dir, dim=1, keepdim=True).clamp_min(1e-12)
            center_delta = 0.25 * max_offset * center_dir / center_norm
            add_delta_all = actuator_strength * (
                0.50 * primitive_delta
                + 0.35 * learned_add_delta
                + 0.15 * center_delta
            )
            add_delta_all = self._clip_vector(add_delta_all, max_offset)

            idx_expand_xyz = add_idx.unsqueeze(1).expand(-1, 3, -1)
            idx_expand_w = add_idx.unsqueeze(1)
            added_base = torch.gather(pts_out, 2, idx_expand_xyz)
            added_delta = torch.gather(add_delta_all, 2, idx_expand_xyz)
            added_pts = added_base + added_delta
            added_w = torch.gather(add_prob, 2, idx_expand_w)

            pts_out = torch.cat([pts_out, added_pts], dim=2)
            final_w = torch.cat([final_w, added_w], dim=2)

            add_ratio = add_prob.mean()
            target_add_ratio = pts_xyz.new_tensor(float(getattr(self.args, "target_add_ratio", 0.01)))
            add_ratio_loss = (add_prob.mean(dim=2).squeeze(1) - target_add_ratio).pow(2).mean()
            add_shape_guard = self._masked_mean(add_prob * shape_score, selection_mask)
            added_delta_norm = torch.linalg.norm(added_delta, dim=1, keepdim=True) / max_offset.clamp_min(1e-12)
            add_offset_reg = (added_delta_norm.pow(2) * added_w.detach()).sum() / added_w.detach().sum().clamp_min(1.0)
            add_count_value = int(add_k)

        delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
        normalized_delta = delta_norm / max_offset.clamp_min(1e-12)
        edit_reg = self._masked_mean(normalized_delta.pow(2) * motion_gate.detach(), selection_mask)

        ratio_loss = (self._masked_mean(repair_gate, selection_mask) - target_ratio) ** 2
        shape_guard = self._masked_mean(repair_gate * cause_scores[:, shape_idx:shape_idx+1, :], selection_mask)
        target_drop_ratio = float(getattr(self.args, "target_drop_ratio", 0.01))
        max_drop_ratio = max(float(getattr(self.args, "max_drop_ratio", max(target_drop_ratio, 0.01))), target_drop_ratio)
        drop_ratio = self._masked_mean(drop_prob, selection_mask)
        drop_ratio_loss = (drop_ratio - target_drop_ratio) ** 2
        drop_cap_loss = torch.relu(drop_ratio - max_drop_ratio) ** 2
        drop_shape_guard = self._masked_mean(drop_prob * shape_score, selection_mask)
        loss = (
            edit_reg
            + float(getattr(self.args, "repair_ratio_weight", 0.1)) * ratio_loss
            + float(getattr(self.args, "repair_shape_guard_weight", 0.05)) * shape_guard
            + float(getattr(self.args, "repair_drop_ratio_weight", 1.0)) * (drop_ratio_loss + drop_cap_loss)
            + float(getattr(self.args, "repair_drop_shape_guard_weight", 0.5)) * drop_shape_guard
            + float(getattr(self.args, "repair_add_ratio_weight", 4.0)) * add_ratio_loss
            + float(getattr(self.args, "repair_add_shape_guard_weight", 0.5)) * add_shape_guard
            + float(getattr(self.args, "repair_add_offset_weight", 0.25)) * add_offset_reg
        )

        self.debug_tensors = {
            "repair_gate": repair_gate.mean().detach(),
            "add_ratio": add_ratio.detach(),
            "delta_norm": delta_norm.mean().detach(),
            "edit_reg": edit_reg.detach(),
            "drop_ratio": drop_ratio.detach(),
            "keep_ratio": keep_prob.mean().detach(),
            "policy_chain_mean": p_chain.mean().detach(),
            "policy_sibling_mean": p_sibling.mean().detach(),
            "policy_parent_mean": p_parent.mean().detach(),
            "policy_context_mean": p_context.mean().detach(),
            "policy_comp_mean": p_comp.mean().detach(),
            "policy_outlier_mean": p_outlier.mean().detach(),
        }
        return pts_out, final_w, loss, {
            "repair_gate": repair_gate,
            "drop_prob": drop_prob,
            "keep_prob": keep_prob,
            "add_prob": add_prob,
            "add_priority": add_priority,
            "add_ratio": add_ratio,
            "add_count": add_count_value,
            "delta": delta,
            "primitive_delta": primitive_delta,
            "edit_reg": edit_reg,
            "ratio_loss": ratio_loss,
            "shape_guard": shape_guard,
            "drop_ratio_loss": drop_ratio_loss,
            "drop_cap_loss": drop_cap_loss,
            "drop_shape_guard": drop_shape_guard,
            "add_ratio_loss": add_ratio_loss,
            "add_shape_guard": add_shape_guard,
            "add_offset_reg": add_offset_reg,
        }
