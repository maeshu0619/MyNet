"""Inference-safe K-slot SparsePCGC proposal policy.

The runtime module intentionally imports no den implementation, cache loader,
teacher plan, codec probe, or Actual encoder.  One shared pointwise trunk and
one shared codec-cost basis feed K independently learned plan tokens.  The
tokens deterministically rerank a shared shortlist; they are not K stochastic
draws from one logit tensor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LOCAL_COST_NAMES = (
    "direct_codec_gain",
    "descendant_gain",
    "expected_new_bits",
    "mask_gain",
    "context_risk",
    "hotspot_proxy",
    "geometry_risk",
)
OPERATION_NAMES = ("Prune", "Add", "Adjust")


class NetworkKProposalPolicy(nn.Module):
    """Create K specialized plans from one shared encoder/basis forward.

    K is deliberately a token dimension.  No sampling is needed at inference,
    and changing one token changes only its own proposal mode.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim=48,
        proposal_count=8,
        max_total_ratio=0.0099,
        shortlist_size=32768,
    ):
        super().__init__()
        self.proposal_count = int(proposal_count)
        if self.proposal_count < 2 or self.proposal_count > 16:
            raise ValueError("proposal_count must be in [2, 16]")
        self.max_total_ratio = float(max_total_ratio)
        self.shortlist_size = max(int(shortlist_size), 256)
        self.fixed_feature_dim = 6
        local_hidden = max(int(hidden_dim), 16)
        global_hidden = max(int(hidden_dim), 32)
        policy_channels = int(in_channels) + self.fixed_feature_dim

        self.shared_local_trunk = nn.Sequential(
            nn.Conv1d(policy_channels, local_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv1d(local_hidden, local_hidden, 1),
            nn.SiLU(inplace=True),
        )
        self.shared_basis_head = nn.Conv1d(
            local_hidden, 3 * len(LOCAL_COST_NAMES), 1
        )
        self.shared_direction_head = nn.Conv1d(local_hidden, 2 * 4, 1)

        codec_dim = 7
        self.shared_global_trunk = nn.Sequential(
            nn.Linear(2 * policy_channels + codec_dim, global_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(global_hidden, global_hidden),
            nn.SiLU(inplace=True),
            nn.LayerNorm(global_hidden),
        )
        self.plan_tokens = nn.Parameter(
            torch.randn(self.proposal_count, global_hidden) * 0.02
        )
        self.token_mixer = nn.Sequential(
            nn.Linear(2 * global_hidden, global_hidden),
            nn.SiLU(inplace=True),
            nn.LayerNorm(global_hidden),
        )
        self.coefficient_head = nn.Linear(
            global_hidden, 3 * len(LOCAL_COST_NAMES)
        )
        self.amount_head = nn.Linear(global_hidden, 1)
        self.share_head = nn.Linear(global_hidden, 3)
        self.enable_head = nn.Linear(global_hidden, 3)
        self.priority_head = nn.Linear(global_hidden, 3)
        self.threshold_head = nn.Linear(global_hidden, 3)
        self.temperature_head = nn.Linear(global_hidden, 3)
        self.direction_delta_head = nn.Linear(global_hidden, 2 * 3)
        self.confidence_head = nn.Linear(global_hidden, 1)

        # The critic sees post-shortlist, post-validity/collision compact plan
        # statistics.  It evaluates all K plans in one tensor batch.
        self.critic = nn.Sequential(
            nn.Linear(global_hidden + 29, global_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(global_hidden, global_hidden // 2),
            nn.SiLU(inplace=True),
        )
        self.critic_gain_head = nn.Linear(global_hidden // 2, 1)
        self.critic_geometry_head = nn.Linear(global_hidden // 2, 1)
        self.critic_interaction_head = nn.Linear(global_hidden // 2, 1)
        self.critic_uncertainty_head = nn.Linear(global_hidden // 2, 1)

        offsets = [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)
        ]
        unit_offsets = F.normalize(torch.tensor(offsets, dtype=torch.float32), dim=1)
        self.register_buffer("unit_neighbor_offsets", unit_offsets, persistent=False)
        self.register_buffer(
            "neighbor_offsets_long", torch.tensor(offsets, dtype=torch.long), persistent=False
        )
        self._initialize_mode_biases()

    def _initialize_mode_biases(self):
        # Eight deliberately different initial modes.  They are initialization
        # only and remain fully learnable; they are not teacher labels/floors.
        ratios = torch.tensor((0.0005, 0.0010, 0.0025, 0.0050, 0.0100))
        ratios = ratios.clamp_max(self.max_total_ratio * (1.0 - 1e-4))
        ratio_fraction = (ratios / max(self.max_total_ratio, 1e-8)).clamp(1e-4, 1 - 1e-4)
        ratio_logits = torch.logit(ratio_fraction)
        share_modes = torch.tensor(
            (
                (0.40, 0.40, 0.20),
                (0.20, 0.60, 0.20),
                (0.60, 0.20, 0.20),
                (0.30, 0.30, 0.40),
                (0.45, 0.35, 0.20),
                (0.25, 0.50, 0.25),
                (0.50, 0.20, 0.30),
                (0.34, 0.33, 0.33),
            ),
            dtype=torch.float32,
        )
        with torch.no_grad():
            self.amount_head.weight.zero_()
            self.share_head.weight.zero_()
            self.enable_head.weight.zero_()
            self.priority_head.weight.zero_()
            self.threshold_head.weight.zero_()
            self.temperature_head.weight.zero_()
            self.amount_head.bias.fill_(float(ratio_logits[2]))
            self.share_head.bias.copy_(torch.log(share_modes[0]))
            self.enable_head.bias.fill_(math.log(0.9 / 0.1))
            self.priority_head.bias.zero_()
            self.threshold_head.bias.zero_()
            self.temperature_head.bias.fill_(math.log(math.exp(0.65) - 1.0))
            self.critic_gain_head.weight.zero_()
            self.critic_gain_head.bias.zero_()
            self.critic_interaction_head.weight.zero_()
            self.critic_interaction_head.bias.zero_()
        # Token-specific biases must be represented by tokens rather than K
        # separate heavy heads.  Seed token coordinates with amount/share modes.
        with torch.no_grad():
            for slot in range(self.proposal_count):
                ratio_index = slot % int(ratio_logits.numel())
                self.plan_tokens[slot, 0] = ratio_logits[ratio_index]
                mode = share_modes[slot % int(share_modes.shape[0])]
                self.plan_tokens[slot, 1:4] = torch.log(mode)
                self.plan_tokens[slot, 4:7] = torch.roll(
                    torch.tensor((1.0, 0.0, -1.0)), slot % 3
                )

    @staticmethod
    def codec_tensor(args, like):
        resolution = float(getattr(args, "sparsepcgc_psnr_resolution", 1023) or 1023)
        values = (
            float(getattr(args, "sparsepcgc_scale_ae", 0)),
            float(getattr(args, "sparsepcgc_scale_sr", 2)),
            float(getattr(args, "sparsepcgc_scale_m", 8)) / 16.0,
            math.log1p(max(float(getattr(args, "sparsepcgc_voxel_size", 1.0)), 0.0)),
            math.log1p(max(float(getattr(args, "sparsepcgc_pos_quantscale", 1.0)), 0.0)),
            math.log1p(max(resolution, 1.0)) / 10.0,
            float(getattr(args, "sparsepcgc_native_bit_depth", 0)) / 16.0,
        )
        return like.new_tensor(values).view(1, -1).expand(like.shape[0], -1)

    @staticmethod
    def _signed_basis(raw_basis):
        signed = raw_basis.clone()
        for index in (2, 4, 6):
            signed[:, :, index] = -F.softplus(raw_basis[:, :, index])
        return signed

    @staticmethod
    def _gather_points(values, indices):
        # values [B,...,N], indices [B,M] -> [B,...,M]
        view_shape = (indices.shape[0],) + (1,) * (values.ndim - 2) + (indices.shape[1],)
        expand_shape = values.shape[:-1] + (indices.shape[1],)
        return torch.gather(values, values.ndim - 1, indices.view(view_shape).expand(expand_shape))

    def _shared_shortlist(self, basis, coefficients, points):
        # A union proxy is produced by the Network basis itself, never by a
        # heuristic/cache.  It is deliberately token-independent: modifying
        # one specialist must not silently change every other specialist's
        # candidate domain through a moving shortlist.
        union_score = basis.abs().amax(dim=2).amax(dim=1)
        size = min(int(points), int(self.shortlist_size))
        return union_score.topk(size, dim=1, largest=True, sorted=True).indices

    def _compact_plan_statistics(
        self, slot_logits, shortlist_indices, ratios, shares, enables,
        enable_probability,
        priorities, temperatures, shortlist_basis, voxel_coords,
    ):
        """Build differentiable compact descriptors after shortlist validity.

        The exact Actuator remains the final authority.  This stage removes
        duplicate source choices across operations in learned priority order
        and records rejected mass, so Critic input describes the executable
        shortlist plan rather than the unconstrained dense proposal.
        """
        batch, slots, operations, shortlist = slot_logits.shape
        point_count = int(voxel_coords.shape[-1]) if torch.is_tensor(voxel_coords) else shortlist
        requested_continuous = ratios * shares * enable_probability * float(point_count)
        requested = (ratios * shares * enables * float(point_count)).round().long()
        requested = requested.clamp(min=0, max=shortlist)
        selected = torch.zeros_like(slot_logits, dtype=torch.bool)
        accepted_count = torch.zeros(
            (batch, slots, operations), device=slot_logits.device, dtype=slot_logits.dtype
        )
        collision_count = torch.zeros_like(accepted_count)
        score_mean = torch.zeros_like(accepted_count)
        score_max = torch.zeros_like(accepted_count)

        # K and three operations are small fixed control dimensions.  There is
        # no point/candidate Python loop and no candidate object construction.
        for b in range(batch):
            for k in range(slots):
                occupied_source = torch.zeros(shortlist, device=slot_logits.device, dtype=torch.bool)
                order = priorities[b, k].argsort(descending=True)
                for rank in range(operations):
                    operation = int(order[rank].item())
                    count = int(requested[b, k, operation].item())
                    if count <= 0:
                        continue
                    scores = slot_logits[b, k, operation] / temperatures[b, k, operation].clamp_min(0.05)
                    scores = scores.masked_fill(occupied_source, -torch.inf)
                    available = int((~occupied_source).sum().item())
                    take = min(count, available)
                    if take <= 0:
                        collision_count[b, k, operation] = float(count)
                        continue
                    chosen = scores.topk(take, largest=True, sorted=False).indices
                    selected[b, k, operation, chosen] = True
                    occupied_source[chosen] = True
                    accepted_count[b, k, operation] = float(take)
                    collision_count[b, k, operation] = float(count - take)
                    chosen_scores = slot_logits[b, k, operation, chosen]
                    score_mean[b, k, operation] = chosen_scores.mean()
                    score_max[b, k, operation] = chosen_scores.max()

        hard_ratio_forward = accepted_count / max(float(point_count), 1.0)
        soft_ratio = requested_continuous / max(float(point_count), 1.0)
        hard_ratio = hard_ratio_forward.detach() + soft_ratio - soft_ratio.detach()
        total_accepted = accepted_count.sum(dim=2, keepdim=True).clamp_min(1.0)
        hard_share_forward = accepted_count / total_accepted
        hard_share = hard_share_forward.detach() + shares - shares.detach()
        enable_ste = enables.detach() + enable_probability - enable_probability.detach()
        selected_float = selected.to(slot_logits.dtype)
        selected_mass = selected_float.sum(dim=3).clamp_min(1.0)
        selected_basis = torch.einsum(
            "bkom,bocm->bkoc", selected_float, shortlist_basis
        ) / selected_mass.unsqueeze(-1)
        selected_basis = selected_basis.mean(dim=2)
        overlap = torch.stack(
            (
                (selected_float[:, :, 0] * selected_float[:, :, 1]).sum(dim=2),
                (selected_float[:, :, 0] * selected_float[:, :, 2]).sum(dim=2),
                (selected_float[:, :, 1] * selected_float[:, :, 2]).sum(dim=2),
            ),
            dim=2,
        ) / float(max(shortlist, 1))
        rejected_ratio = collision_count / requested.to(slot_logits.dtype).clamp_min(1.0)
        descriptor = torch.cat(
            (
                hard_ratio,
                hard_share,
                enable_ste,
                priorities,
                score_mean,
                score_max,
                selected_basis,
                overlap,
                rejected_ratio.mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
        if descriptor.shape[2] != 29:
            raise RuntimeError("internal K-plan descriptor size mismatch")
        return {
            "selected_shortlist_mask": selected,
            "requested_count": requested,
            "accepted_count": accepted_count,
            "collision_count": collision_count,
            "hard_ratio": hard_ratio_forward,
            "hard_share": hard_share_forward,
            "descriptor": descriptor,
        }

    @staticmethod
    def _select_slot_tensor(value, selected_slot):
        if not torch.is_tensor(value) or value.ndim < 2:
            return value
        batch = value.shape[0]
        index = selected_slot.view(batch, 1, *([1] * (value.ndim - 2)))
        index = index.expand(batch, 1, *value.shape[2:])
        return torch.gather(value, 1, index).squeeze(1)

    def forward(self, features, args, training=None, fixed_features=None, voxel_coords=None):
        if features.ndim != 3:
            raise ValueError("NetworkKProposalPolicy expects [B,C,N] features")
        if training is None:
            training = self.training
        batch, _, points = features.shape
        if fixed_features is None:
            fixed_features = features.new_zeros((batch, self.fixed_feature_dim, points))
        if tuple(fixed_features.shape) != (batch, self.fixed_feature_dim, points):
            raise ValueError("K-policy fixed feature shape mismatch")

        policy_features = torch.cat((features, fixed_features.to(features)), dim=1)
        local = self.shared_local_trunk(policy_features)
        basis = self._signed_basis(
            self.shared_basis_head(local).view(batch, 3, len(LOCAL_COST_NAMES), points)
        )
        direction_field = self.shared_direction_head(local).view(batch, 2, 4, points)
        pooled = torch.cat((policy_features.mean(2), policy_features.amax(2)), dim=1)
        global_feature = self.shared_global_trunk(
            torch.cat((pooled, self.codec_tensor(args, pooled)), dim=1)
        )
        tokens = self.plan_tokens.view(1, self.proposal_count, -1).expand(batch, -1, -1)
        state = global_feature.unsqueeze(1).expand(-1, self.proposal_count, -1)
        slot_feature = self.token_mixer(torch.cat((state, tokens), dim=2))

        coefficients = torch.tanh(self.coefficient_head(slot_feature)).view(
            batch, self.proposal_count, 3, len(LOCAL_COST_NAMES)
        )
        amount_logit = self.amount_head(slot_feature)
        # Token coordinates seed genuinely distinct modes before offline fit.
        amount_logit = amount_logit + 0.20 * tokens[:, :, 0:1]
        total_ratio = torch.sigmoid(amount_logit) * self.max_total_ratio
        share_logits = self.share_head(slot_feature) + 0.20 * tokens[:, :, 1:4]
        shares = torch.softmax(share_logits, dim=2)
        enable_logits = self.enable_head(slot_feature)
        enable_probability = torch.sigmoid(enable_logits)
        enables = (enable_probability >= 0.5).to(features.dtype)
        priorities = self.priority_head(slot_feature) + 0.25 * tokens[:, :, 4:7]
        thresholds = torch.tanh(self.threshold_head(slot_feature))
        temperatures = F.softplus(self.temperature_head(slot_feature)) + 0.10
        confidence = torch.sigmoid(self.confidence_head(slot_feature))

        shortlist_indices = self._shared_shortlist(basis, coefficients, points)
        shortlist_basis = self._gather_points(basis, shortlist_indices)
        slot_logits = torch.einsum(
            "bocm,bkoc->bkom", shortlist_basis, coefficients
        )
        slot_logits = slot_logits - thresholds.unsqueeze(-1) + priorities.unsqueeze(-1)

        compact = self._compact_plan_statistics(
            slot_logits, shortlist_indices, total_ratio, shares, enables,
            enable_probability,
            priorities, temperatures, shortlist_basis, voxel_coords,
        )
        critic_input = torch.cat((slot_feature, compact["descriptor"]), dim=2)
        critic_hidden = self.critic(critic_input)
        predicted_gain = self.critic_gain_head(critic_hidden)
        predicted_geometry = F.softplus(self.critic_geometry_head(critic_hidden))
        predicted_interaction = self.critic_interaction_head(critic_hidden)
        uncertainty = F.softplus(self.critic_uncertainty_head(critic_hidden)) + 1e-4
        predicted_plan_gain = predicted_gain + predicted_interaction
        critic_score = (
            predicted_plan_gain - predicted_geometry
            - uncertainty * (1.0 - confidence)
        )
        selected_slot = critic_score.squeeze(-1).argmax(dim=1)
        critic_log_probability = torch.log_softmax(
            critic_score.squeeze(-1), dim=1
        )
        selected_critic_log_probability = critic_log_probability.gather(
            1, selected_slot.view(batch, 1)
        ).mean()
        critic_probability = torch.softmax(critic_score.squeeze(-1), dim=1)
        critic_selection_entropy = -(
            critic_probability * critic_log_probability
        ).sum(dim=1).mean()

        selected_short_mask = self._select_slot_tensor(
            compact["selected_shortlist_mask"], selected_slot
        )
        selected_slot_logits = self._select_slot_tensor(slot_logits, selected_slot)
        # Preserve the Critic-evaluated source plan at the Actuator boundary.
        # The Actuator can still reject an invalid Add/Adjust target, but it
        # cannot silently replace a selected source with an unscored shortlist
        # member.  Any target-level rejection is audited after execution.
        selected_slot_logits = torch.where(
            selected_short_mask,
            selected_slot_logits + 1.0e4,
            selected_slot_logits.new_full((), -1.0e4),
        )
        full_where_logits = features.new_full((batch, 3, points), -1.0e4)
        full_where_logits.scatter_(
            2,
            shortlist_indices.unsqueeze(1).expand(-1, 3, -1),
            selected_slot_logits,
        )
        selected_coefficients = self._select_slot_tensor(coefficients, selected_slot)
        selected_direction_delta = self._select_slot_tensor(
            self.direction_delta_head(slot_feature).view(batch, self.proposal_count, 2, 3),
            selected_slot,
        )
        base_vectors = F.normalize(
            direction_field[:, :, :3] + selected_direction_delta.unsqueeze(-1),
            dim=2,
            eps=1e-6,
        )
        concentration = F.softplus(direction_field[:, :, 3:4]) + 0.1
        direction_logits = torch.einsum(
            "bodn,qd->boqn", base_vectors, self.unit_neighbor_offsets.to(base_vectors)
        ) * concentration

        selected_total_ratio_raw = self._select_slot_tensor(total_ratio, selected_slot).unsqueeze(-1)
        selected_shares_raw = self._select_slot_tensor(shares, selected_slot).unsqueeze(-1)
        selected_accepted_count = self._select_slot_tensor(
            compact["accepted_count"], selected_slot
        )
        accepted_total = selected_accepted_count.sum(dim=1, keepdim=True)
        accepted_total_ratio = (
            (accepted_total - 0.125).clamp_min(0.0) / max(float(points), 1.0)
        ).unsqueeze(-1)
        accepted_shares = (
            selected_accepted_count / accepted_total.clamp_min(1.0)
        ).unsqueeze(-1)
        # Hard forward is exactly the collision-resolved compact count while
        # backward remains connected to the raw Amount/share distributions.
        selected_total_ratio = (
            accepted_total_ratio.detach()
            + selected_total_ratio_raw
            - selected_total_ratio_raw.detach()
        )
        selected_shares = (
            accepted_shares.detach()
            + selected_shares_raw
            - selected_shares_raw.detach()
        )
        selected_enable_probability = self._select_slot_tensor(
            enable_probability, selected_slot
        ).unsqueeze(-1)
        selected_enables_hard = (selected_accepted_count > 0).to(features.dtype).unsqueeze(-1)
        selected_enables = (
            selected_enables_hard.detach()
            + selected_enable_probability
            - selected_enable_probability.detach()
        )
        selected_priorities = self._select_slot_tensor(priorities, selected_slot).unsqueeze(-1)
        selected_temperature = self._select_slot_tensor(
            temperatures.mean(dim=2, keepdim=True), selected_slot
        ).unsqueeze(-1)
        selected_threshold = self._select_slot_tensor(thresholds, selected_slot).unsqueeze(-1)

        selected = {
            "local_cost_maps": basis,
            "fixed_features": fixed_features,
            "local_cost_names": LOCAL_COST_NAMES,
            "coefficients": selected_coefficients,
            "coefficient_mean": selected_coefficients,
            "coefficient_scale": torch.zeros_like(selected_coefficients),
            "base_where_logits": full_where_logits,
            "where_logits": full_where_logits,
            "where_sampling_temperature": selected_temperature,
            "where_threshold": selected_threshold,
            "base_direction_vectors": base_vectors,
            "direction_concentration": concentration,
            "direction_logits": direction_logits,
            "direction_sampling_temperature": features.new_tensor(1.0),
            "total_ratio_raw": selected_total_ratio_raw / max(self.max_total_ratio, 1e-8),
            "total_ratio_unconstrained": selected_total_ratio_raw,
            "total_ratio": selected_total_ratio,
            "total_ratio_mean": selected_total_ratio_raw,
            "amount_sample_log_prob": selected_total_ratio.sum() * 0.0,
            "amount_distribution_entropy": selected_total_ratio.sum() * 0.0,
            "share_logits": self._select_slot_tensor(share_logits, selected_slot),
            "shares": selected_shares,
            "shares_raw": selected_shares_raw,
            "shares_mean": selected_shares_raw,
            "share_sample_log_prob": selected_shares.sum() * 0.0,
            "share_distribution_entropy": selected_shares.sum() * 0.0,
            "gate_logits": self._select_slot_tensor(enable_logits, selected_slot).unsqueeze(-1),
            "gates": selected_enables,
            "gate_probability": selected_enable_probability,
            "gate_hard": selected_enables_hard,
            "gate_base_probability": selected_enable_probability,
            "priority_logits": selected_priorities,
            "priority_base_logits": selected_priorities,
            "priorities": torch.softmax(selected_priorities, dim=1),
            "priority_order": selected_priorities.squeeze(-1).argsort(dim=1, descending=True),
            "priority_sample_log_prob": selected_priorities.sum() * 0.0,
            "priority_entropy": -(torch.softmax(selected_priorities, 1) * torch.log_softmax(selected_priorities, 1)).sum(1).mean(),
            "temperature": selected_temperature,
            "operation_ratios": selected_total_ratio * selected_shares * selected_enables,
            "predicted_local_gain_sum": self._select_slot_tensor(
                compact["descriptor"][:, :, 12:15].sum(dim=2, keepdim=True), selected_slot
            ),
            "interaction_correction": self._select_slot_tensor(predicted_interaction, selected_slot),
            "predicted_plan_gain": self._select_slot_tensor(
                predicted_plan_gain, selected_slot
            ),
            "selected_spatial_statistics": self._select_slot_tensor(compact["descriptor"][:, :, 18:28], selected_slot),
            "where_entropy": -(torch.softmax(full_where_logits, 2) * torch.log_softmax(full_where_logits, 2)).sum(2).mean(),
            "amount_entropy": selected_total_ratio.sum() * 0.0,
            "share_entropy": -(selected_shares.clamp_min(1e-8) * selected_shares.clamp_min(1e-8).log()).sum(1).mean(),
            "ratio_mean_entropy": selected_total_ratio.sum() * 0.0,
            "action_entropy": -(selected_enable_probability.clamp_min(1e-8) * selected_enable_probability.clamp_min(1e-8).log()).mean(),
            "exploration_active": False,
            "exploration_fraction": features.new_tensor(0.0),
            "composite_policy_log_prob": selected_critic_log_probability,
            "composite_policy_entropy": critic_selection_entropy,
        }
        return {
            "selected_policy_terms": selected,
            "shared_basis": basis,
            "shared_direction_field": direction_field,
            "shortlist_indices": shortlist_indices,
            "slot_logits": slot_logits,
            "slot_features": slot_feature,
            "total_ratio": total_ratio,
            "shares": shares,
            "enable_probability": enable_probability,
            "enables": enables,
            "priorities": priorities,
            "thresholds": thresholds,
            "temperatures": temperatures,
            "confidence": confidence,
            "compact_plans": compact,
            "predicted_gain": predicted_gain,
            "predicted_plan_gain": predicted_plan_gain,
            "predicted_geometry": predicted_geometry,
            "predicted_interaction": predicted_interaction,
            "uncertainty": uncertainty,
            "critic_score": critic_score,
            "critic_selection_log_prob": selected_critic_log_probability,
            "critic_selection_entropy": critic_selection_entropy,
            "selected_slot": selected_slot,
            "shared_encoder_forward_count": 1,
            "shared_basis_forward_count": 1,
            "proposal_count": self.proposal_count,
            "critic_batch_count": 1,
            "selected_plan_count": 1,
            "den6_call_count": 0,
            "cache_reference_count": 0,
            "teacher_reference_count": 0,
            "sparsepcgc_probe_count": 0,
            "candidate_actual_encode_count": 0,
            "selected_shortlist_mask": selected_short_mask,
        }
