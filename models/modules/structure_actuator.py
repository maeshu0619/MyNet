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

    def _max_add_ratio(self):
        target_ratio = max(float(getattr(self.args, "target_add_ratio", 0.01)), 0.0)
        max_ratio = max(float(getattr(self.args, "max_add_ratio", max(target_ratio, 0.0))), target_ratio)
        return max(max_ratio, 0.0)

    def _target_add_count(self, point_count):
        if point_count <= 0 or not bool(getattr(self.args, "add", True)):
            return 0, 0.0
        max_ratio = self._max_add_ratio()
        if max_ratio <= 0.0:
            return 0, 0.0
        start = float(getattr(self.args, "repair_add_candidate_ratio_start", 0.0)) or max_ratio
        end = float(getattr(self.args, "repair_add_candidate_ratio_end", 0.0)) or max_ratio
        phase = self._exploration_phase()
        candidate_ratio = start + (end - start) * phase
        candidate_ratio = min(max(candidate_ratio, 0.0), max_ratio)
        max_add_points = max(1, int(round(max_ratio * float(point_count))))
        add_points = int(round(candidate_ratio * float(point_count)))
        if candidate_ratio > 0.0:
            add_points = max(add_points, 1)
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

    @staticmethod
    def _mask_add_scores(add_scores, selection_mask, keep_prob=None, keep_threshold=0.0):
        if selection_mask is None:
            valid = torch.ones_like(add_scores, dtype=torch.bool)
        else:
            valid = selection_mask.squeeze(1) if selection_mask.ndim == 3 else selection_mask
            valid = valid.to(device=add_scores.device, dtype=torch.bool)
        if keep_prob is not None and float(keep_threshold) > 0.0:
            keep = keep_prob.squeeze(1) if keep_prob.ndim == 3 else keep_prob
            keep = keep.to(device=add_scores.device, dtype=add_scores.dtype)
            valid = valid & (keep.detach() >= float(keep_threshold))
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
        target_drop_ratio = float(getattr(self.args, "target_drop_ratio", 0.01))
        max_drop_ratio = max(float(getattr(self.args, "max_drop_ratio", max(target_drop_ratio, 0.01))), target_drop_ratio)
        delete_prior = torch.sigmoid(delete_prior.clamp(-8.0, 8.0))
        drop_score_noise = max(
            self._annealed_value("repair_drop_score_noise_start", "repair_drop_score_noise_end"),
            0.0,
        )
        learned_drop_logit = self.drop_head(actuator_features)
        if self.training and drop_score_noise > 0.0:
            learned_drop_logit = learned_drop_logit + torch.randn_like(learned_drop_logit) * drop_score_noise
        learned_drop = torch.sigmoid(learned_drop_logit)
        drop_prob = (repair_gate * delete_prior * learned_drop).clamp(0.0, 1.0)
        drop_random_mix = min(
            max(self._annealed_value("repair_drop_random_mix_start", "repair_drop_random_mix_end"), 0.0),
            1.0,
        )
        if self.training and drop_random_mix > 0.0:
            random_drop = self._random_ratio_mask_like(drop_prob, max_drop_ratio, selection_mask)
            drop_prob = ((1.0 - drop_random_mix) * drop_prob + drop_random_mix * random_drop).clamp(0.0, 1.0)
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
        add_k, add_candidate_ratio = self._target_add_count(N)
        add_ratio = pts_xyz.new_zeros(())
        add_ratio_loss = pts_xyz.new_zeros(())
        add_shape_guard = pts_xyz.new_zeros(())
        add_offset_reg = pts_xyz.new_zeros(())
        add_drop_conflict_loss = pts_xyz.new_zeros(())
        added_keep_loss = pts_xyz.new_zeros(())
        add_min_offset_loss = pts_xyz.new_zeros(())
        add_prob = pts_xyz.new_zeros((B, 1, N))
        add_priority = add_prob
        add_count_value = 0
        add_effective_count_value = 0
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
            add_logit = learned_add_logit + add_prior
            if self.training and add_score_noise > 0.0:
                add_logit = add_logit + self._gumbel_like(add_logit) * add_score_noise
            add_priority = torch.sigmoid(add_logit.clamp(-8.0, 8.0))
            add_scores = self._mask_add_scores(
                add_logit.squeeze(1),
                selection_mask,
                keep_prob=keep_prob,
                keep_threshold=float(getattr(self.args, "add_noop_keep_threshold", 0.5)),
            )
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
            add_weight_mode = str(getattr(self.args, "repair_add_weight_mode", "hard")).strip().lower()
            if add_weight_mode == "soft":
                add_prob = add_mask_st.unsqueeze(1) * add_priority
            else:
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
            selected_add_strength = torch.gather(add_priority, 2, idx_expand_w)
            added_w = torch.gather(add_prob, 2, idx_expand_w)
            if add_weight_mode == "soft":
                added_w = selected_add_strength
            if self.training and add_weight_random_mix > 0.0:
                random_w = torch.rand_like(added_w)
                added_w = ((1.0 - add_weight_random_mix) * added_w + add_weight_random_mix * random_w).clamp(0.0, 1.0)

            pts_out = torch.cat([pts_out, added_pts], dim=2)
            final_w = torch.cat([final_w, added_w], dim=2)

            add_ratio = added_w.sum() / max(float(B * N), 1.0)
            target_add_ratio = pts_xyz.new_tensor(float(getattr(self.args, "target_add_ratio", 0.01)))
            add_ratio_loss = (add_ratio - target_add_ratio).pow(2)
            add_shape_guard = self._masked_mean(add_prob * shape_score, selection_mask)
            add_drop_conflict_loss = self._masked_mean(add_prob * drop_prob.detach(), selection_mask)
            if str(getattr(self.args, "repair_add_weight_mode", "hard")).strip().lower() == "soft":
                added_keep_loss = (added_w * (1.0 - added_w)).mean()
            else:
                added_keep_loss = (1.0 - selected_add_strength).pow(2).mean()
            added_delta_norm = torch.linalg.norm(added_delta, dim=1, keepdim=True) / max_offset.clamp_min(1e-12)
            add_offset_reg = (added_delta_norm.pow(2) * added_w.detach()).sum() / added_w.detach().sum().clamp_min(1.0)
            min_offset_qstep = max(float(getattr(self.args, "repair_add_min_offset_qstep", 0.20)), 0.0)
            max_offset_qstep = max(float(getattr(self.args, "max_repair_qstep", 0.25)), min_offset_qstep, 1e-6)
            min_offset_norm = min(min_offset_qstep / max_offset_qstep, 1.0)
            if min_offset_norm > 0.0:
                min_offset_shortfall = torch.relu(added_delta_norm.new_tensor(min_offset_norm) - added_delta_norm).pow(2)
                add_min_offset_loss = (
                    min_offset_shortfall * selected_add_strength.detach()
                ).sum() / selected_add_strength.detach().sum().clamp_min(1.0)
            add_count_value = int(add_k)
            hardening_threshold = float(
                getattr(self.args, "operation_count_drop_threshold", getattr(self.args, "test_drop_threshold", 0.5))
            )
            add_effective_count_value = int((added_w.detach() >= hardening_threshold).sum().item())

        delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
        normalized_delta = delta_norm / max_offset.clamp_min(1e-12)
        edit_reg = self._masked_mean(normalized_delta.pow(2) * motion_gate.detach(), selection_mask)

        ratio_loss = (self._masked_mean(repair_gate, selection_mask) - target_ratio) ** 2
        shape_guard = self._masked_mean(repair_gate * cause_scores[:, shape_idx:shape_idx+1, :], selection_mask)
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
            + float(getattr(self.args, "repair_add_drop_conflict_weight", 2.0)) * add_drop_conflict_loss
            + float(getattr(self.args, "repair_add_keep_weight", 1.0)) * added_keep_loss
            + float(getattr(self.args, "repair_add_min_offset_weight", 0.5)) * add_min_offset_loss
        )

        self.debug_tensors = {
            "repair_gate": repair_gate.mean().detach(),
            "add_ratio": add_ratio.detach(),
            "add_candidate_ratio": pts_xyz.new_tensor(float(add_candidate_ratio)).detach(),
            "add_score_noise": pts_xyz.new_tensor(float(add_score_noise)).detach(),
            "add_weight_random_mix": pts_xyz.new_tensor(float(add_weight_random_mix)).detach(),
            "drop_score_noise": pts_xyz.new_tensor(float(drop_score_noise)).detach(),
            "drop_random_mix": pts_xyz.new_tensor(float(drop_random_mix)).detach(),
            "add_drop_conflict_loss": add_drop_conflict_loss.detach(),
            "added_keep_loss": added_keep_loss.detach(),
            "add_min_offset_loss": add_min_offset_loss.detach(),
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
            "add_effective_count": add_effective_count_value,
            "add_candidate_ratio": float(add_candidate_ratio),
            "add_score_noise": float(add_score_noise),
            "add_weight_random_mix": float(add_weight_random_mix),
            "drop_score_noise": float(drop_score_noise),
            "drop_random_mix": float(drop_random_mix),
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
            "add_drop_conflict_loss": add_drop_conflict_loss,
            "added_keep_loss": added_keep_loss,
            "add_min_offset_loss": add_min_offset_loss,
        }
