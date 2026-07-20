"""Network-only SparsePCGC edit policy.

This module deliberately has no dependency on den4/den5/den6, candidate pools,
teacher plans, or codec probes.  It consumes only the feature tensor produced by
the normal point-cloud encoder/Octree analysis and explicit codec settings.
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


class NetworkOnlyCodecPolicy(nn.Module):
    """Predict one composite Add/Prune/Adjust plan from one forward pass."""

    def __init__(self, in_channels, hidden_dim=48, max_total_ratio=0.0099):
        super().__init__()
        self.max_total_ratio = float(max_total_ratio)
        local_hidden = max(int(hidden_dim), 16)
        global_hidden = max(int(hidden_dim), 32)

        self.fixed_feature_dim = 6
        policy_in_channels = int(in_channels) + self.fixed_feature_dim
        self.local_trunk = nn.Sequential(
            nn.Conv1d(policy_in_channels, local_hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv1d(local_hidden, local_hidden, 1),
            nn.SiLU(inplace=True),
        )
        self.local_cost_head = nn.Conv1d(local_hidden, 3 * len(LOCAL_COST_NAMES), 1)
        # Two local 3-D direction fields (Add and Adjust) plus concentration.
        self.direction_field_head = nn.Conv1d(local_hidden, 8, 1)

        codec_dim = 7
        self.global_trunk = nn.Sequential(
            nn.Linear(2 * policy_in_channels + codec_dim, global_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(global_hidden, global_hidden),
            nn.SiLU(inplace=True),
            nn.LayerNorm(global_hidden),
        )
        self.coefficient_head = nn.Linear(global_hidden, 3 * len(LOCAL_COST_NAMES))
        self.coefficient_scale_head = nn.Linear(global_hidden, 3 * len(LOCAL_COST_NAMES))
        self.amount_head = nn.Linear(global_hidden, 1)
        self.amount_scale_head = nn.Linear(global_hidden, 1)
        self.share_head = nn.Linear(global_hidden, 3)
        self.share_scale_head = nn.Linear(global_hidden, 3)
        self.gate_head = nn.Linear(global_hidden, 3)
        self.priority_head = nn.Linear(global_hidden, 3)
        self.threshold_head = nn.Linear(global_hidden, 3)
        self.temperature_head = nn.Linear(global_hidden, 1)
        self.interaction_head = nn.Sequential(
            # global state + shares/gates/priorities/ratio + selected spatial
            # statistics (mass, 6/18/26-neighbour concentration, overlap).
            nn.Linear(global_hidden + 3 + 3 + 3 + 1 + 15, global_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(global_hidden, 1),
        )

        offsets = [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)
        ]
        offset_tensor = torch.tensor(offsets, dtype=torch.float32)
        offset_tensor = F.normalize(offset_tensor, dim=1)
        self.register_buffer("unit_neighbor_offsets", offset_tensor, persistent=False)

        # Initial total ratio is 0.25%, with a learnable 0.4/0.4/0.2 split.
        # These are biases only; no target loss or hard fixed amount is used.
        init_ratio = min(0.0025 / max(self.max_total_ratio, 1e-8), 1.0 - 1e-4)
        nn.init.constant_(self.amount_head.bias, math.log(init_ratio / (1.0 - init_ratio)))
        nn.init.zeros_(self.amount_head.weight)
        nn.init.zeros_(self.amount_scale_head.weight)
        nn.init.constant_(self.amount_scale_head.bias, math.log(math.exp(0.55) - 1.0))
        nn.init.zeros_(self.share_head.weight)
        with torch.no_grad():
            self.share_head.bias.copy_(torch.log(torch.tensor([0.4, 0.4, 0.2])))
        nn.init.zeros_(self.share_scale_head.weight)
        nn.init.constant_(self.share_scale_head.bias, math.log(math.exp(0.35) - 1.0))
        nn.init.zeros_(self.coefficient_scale_head.weight)
        nn.init.constant_(self.coefficient_scale_head.bias, math.log(math.exp(0.10) - 1.0))
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, math.log(0.9 / 0.1))
        nn.init.zeros_(self.priority_head.weight)
        nn.init.zeros_(self.priority_head.bias)
        nn.init.zeros_(self.threshold_head.weight)
        nn.init.zeros_(self.threshold_head.bias)
        nn.init.zeros_(self.temperature_head.weight)
        nn.init.constant_(self.temperature_head.bias, math.log(math.exp(0.60) - 1.0))
        # Start as an exact local-sum model.  A random plan-level correction
        # was several percentage points at Step 1 while the actual scalar was
        # near zero, and its Huber gradient immediately collapsed Amount/share.
        nn.init.zeros_(self.interaction_head[-1].weight)
        nn.init.zeros_(self.interaction_head[-1].bias)

    @staticmethod
    def codec_tensor(args, like):
        """Return dataset-agnostic codec conditioning values in stable ranges."""
        native_resolution = float(
            getattr(args, "sparsepcgc_psnr_resolution", 1023) or 1023
        )
        values = (
            float(getattr(args, "sparsepcgc_scale_ae", 0)),
            float(getattr(args, "sparsepcgc_scale_sr", 2)),
            float(getattr(args, "sparsepcgc_scale_m", 8)) / 16.0,
            math.log1p(max(float(getattr(args, "sparsepcgc_voxel_size", 1.0)), 0.0)),
            math.log1p(max(float(getattr(args, "sparsepcgc_pos_quantscale", 1.0)), 0.0)),
            math.log1p(max(native_resolution, 1.0)) / 10.0,
            float(getattr(args, "sparsepcgc_native_bit_depth", 0)) / 16.0,
        )
        return like.new_tensor(values).view(1, -1).expand(like.shape[0], -1)

    @staticmethod
    def _binary_concrete(logits, temperature, training, exploration_floor):
        temperature = temperature.clamp_min(0.05)
        if training:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            logistic = torch.log(uniform) - torch.log1p(-uniform)
            probability = torch.sigmoid((logits + logistic) / temperature)
        else:
            probability = torch.sigmoid(logits / temperature)
        if training and exploration_floor > 0.0:
            probability = probability * (1.0 - exploration_floor) + exploration_floor
        hard = (probability >= 0.5).to(probability.dtype)
        straight_through = hard.detach() + probability - probability.detach()
        return straight_through, probability, hard

    def forward(self, features, args, training=None, fixed_features=None):
        if features.ndim != 3:
            raise ValueError("NetworkOnlyCodecPolicy expects [B,C,N] features")
        if training is None:
            training = self.training
        batch, _, points = features.shape
        if fixed_features is None:
            fixed_features = features.new_zeros((batch, self.fixed_feature_dim, points))
        if fixed_features.shape != (batch, self.fixed_feature_dim, points):
            raise ValueError(
                "network-only fixed features must have shape "
                f"{(batch, self.fixed_feature_dim, points)}, got {tuple(fixed_features.shape)}"
            )
        policy_features = torch.cat((features, fixed_features.to(features)), dim=1)
        local = self.local_trunk(policy_features)
        local_cost = self.local_cost_head(local).view(
            batch, 3, len(LOCAL_COST_NAMES), points
        )
        # Risks/new bits should subtract from gain; the signs are architectural,
        # while their magnitudes and all operation coefficients remain learned.
        signed_local_cost = local_cost.clone()
        signed_local_cost[:, :, 2] = -F.softplus(local_cost[:, :, 2])
        signed_local_cost[:, :, 4] = -F.softplus(local_cost[:, :, 4])
        signed_local_cost[:, :, 6] = -F.softplus(local_cost[:, :, 6])

        pooled = torch.cat(
            (policy_features.mean(dim=2), policy_features.amax(dim=2)), dim=1
        )
        codec = self.codec_tensor(args, pooled)
        global_feature = self.global_trunk(torch.cat((pooled, codec), dim=1))
        current_step = max(int(getattr(args, "_global_train_step", 0)), 0)
        anneal_steps = max(
            int(getattr(args, "network_only_exploration_anneal_steps", 200)), 1
        )
        exploration_fraction = (
            max(0.25, 1.0 - float(current_step) / float(anneal_steps))
            if training else 0.0
        )
        exploration_active = bool(training and exploration_fraction > 0.0)

        coefficient_mean_logits = self.coefficient_head(global_feature).view(
            batch, 3, len(LOCAL_COST_NAMES)
        )
        coefficient_scale = (
            F.softplus(self.coefficient_scale_head(global_feature)) + 1e-3
        ).view(batch, 3, len(LOCAL_COST_NAMES))
        coefficient_latent = coefficient_mean_logits
        if training:
            coefficient_latent = coefficient_latent + (
                coefficient_scale * float(exploration_fraction)
                * torch.randn_like(coefficient_latent)
            )
        coefficients = torch.tanh(coefficient_latent)
        base_where_logits = (signed_local_cost * coefficients.unsqueeze(-1)).sum(dim=2)

        amount_mean_logit = self.amount_head(global_feature).view(batch, 1, 1)
        amount_scale = F.softplus(self.amount_scale_head(global_feature)).view(batch, 1, 1) + 1e-3
        amount_latent = amount_mean_logit
        if training:
            amount_latent = amount_latent + (
                amount_scale * float(exploration_fraction) * torch.randn_like(amount_latent)
            )
        total_ratio_raw = torch.sigmoid(amount_latent)
        total_ratio_unconstrained = total_ratio_raw * self.max_total_ratio
        total_ratio = total_ratio_unconstrained
        total_ratio_mean = torch.sigmoid(amount_mean_logit) * self.max_total_ratio
        effective_amount_scale = (
            amount_scale * max(float(exploration_fraction), 1e-3)
        ).clamp_min(0.10 if training else 1e-3)
        amount_distribution = torch.distributions.Normal(amount_mean_logit, effective_amount_scale)
        amount_sample_log_prob = amount_distribution.log_prob(amount_latent.detach()).mean()
        amount_distribution_entropy = amount_distribution.entropy().mean()

        share_mean_logits = self.share_head(global_feature)
        share_scale = F.softplus(self.share_scale_head(global_feature)) + 1e-3
        share_logits = share_mean_logits
        if training:
            share_logits = share_logits + (
                share_scale * float(exploration_fraction) * torch.randn_like(share_logits)
            )
        shares_raw = torch.softmax(share_logits, dim=1).view(batch, 3, 1)
        shares = shares_raw
        shares_mean = torch.softmax(share_mean_logits, dim=1).view(batch, 3, 1)
        effective_share_scale = (
            share_scale * max(float(exploration_fraction), 1e-3)
        ).clamp_min(0.10 if training else 1e-3)
        share_distribution = torch.distributions.Normal(share_mean_logits, effective_share_scale)
        share_sample_log_prob = share_distribution.log_prob(share_logits.detach()).mean()
        share_distribution_entropy = share_distribution.entropy().mean()
        gate_logits = self.gate_head(global_feature).view(batch, 3, 1)
        temperature = F.softplus(self.temperature_head(global_feature)).view(batch, 1, 1) + 0.15
        exploration_floor = (
            max(float(getattr(args, "network_only_action_exploration_floor", 0.05)), 0.0)
            if training else 0.0
        )
        gates, gate_probability, gate_hard = self._binary_concrete(
            gate_logits,
            temperature,
            bool(training),
            min(exploration_floor, 0.25),
        )
        gate_base_probability = torch.sigmoid(gate_logits / temperature.clamp_min(0.05))
        priority_base_logits = self.priority_head(global_feature).view(batch, 3, 1)
        priority_logits = priority_base_logits
        if training:
            priority_uniform = torch.rand_like(priority_logits).clamp_(1e-6, 1.0 - 1e-6)
            priority_logits = priority_logits - (
                torch.log(-torch.log(priority_uniform)) * float(exploration_fraction)
            )
        priorities = torch.softmax(priority_logits, dim=1)
        priority_order = priority_logits.squeeze(-1).argsort(dim=1, descending=True)
        priority_log_prob_terms = []
        for rank in range(3):
            remaining = priority_order[:, rank:]
            remaining_logits = torch.gather(priority_base_logits.squeeze(-1), 1, remaining)
            # The selected operation is first in ``remaining`` by construction.
            priority_log_prob_terms.append(torch.log_softmax(remaining_logits, dim=1)[:, 0].mean())
        priority_sample_log_prob = torch.stack(priority_log_prob_terms).sum()
        priority_entropy = -(
            torch.softmax(priority_base_logits, dim=1)
            * torch.log_softmax(priority_base_logits, dim=1)
        ).sum(dim=1).mean()
        where_threshold = torch.tanh(self.threshold_head(global_feature)).view(batch, 3, 1)
        # Priority is a learned tie-breaker, not a heuristic base score.
        base_where_logits = base_where_logits - where_threshold + priority_base_logits
        where_logits = base_where_logits + (priority_logits - priority_base_logits)
        where_sampling_temperature = (
            temperature
            * max(float(exploration_fraction), 0.25)
            * max(float(getattr(args, "network_only_where_gumbel_scale", 1.0)), 0.0)
        ).clamp_min(0.05)
        if training:
            if float(getattr(args, "network_only_where_gumbel_scale", 1.0)) > 0.0:
                uniform = torch.rand_like(where_logits).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(uniform))
                # A Gumbel perturbation with scale T samples the categorical
                # distribution softmax(base/T).  Keep T so the likelihood of
                # the actually executed top-k can use the same distribution.
                where_logits = where_logits + gumbel * where_sampling_temperature.detach()
        operation_ratios = total_ratio * shares * gates

        direction_field = self.direction_field_head(local).view(batch, 2, 4, points)
        base_vectors = F.normalize(direction_field[:, :, :3], dim=2, eps=1e-6)
        concentration = F.softplus(direction_field[:, :, 3:4]) + 0.1
        base_direction_logits = torch.einsum(
            "bodn,kd->bokn", base_vectors, self.unit_neighbor_offsets.to(base_vectors)
        ) * concentration
        direction_sampling_temperature = features.new_tensor(
            max(float(exploration_fraction), 0.25) if training else 1.0
        )
        if training:
            # One iid Gumbel value per direction, shared spatially within this
            # one sampled plan.  Every point keeps the correct categorical
            # marginal while avoiding an additional [B,2,26,N] random tensor
            # (about 160 MiB and >1 s at one million points).
            uniform = torch.rand(
                (batch, 2, int(self.unit_neighbor_offsets.shape[0]), 1),
                device=base_direction_logits.device,
                dtype=base_direction_logits.dtype,
            ).clamp_(1e-6, 1.0 - 1e-6)
            direction_gumbel = -torch.log(-torch.log(uniform))
            direction_logits = (
                base_direction_logits
                + direction_gumbel * direction_sampling_temperature.detach()
            )
        else:
            direction_logits = base_direction_logits

        soft_selection = torch.sigmoid(where_logits / temperature.clamp_min(0.05))
        composed_local_gain = (signed_local_cost * coefficients.unsqueeze(-1)).sum(dim=2)
        local_gain_per_operation = (
            soft_selection * composed_local_gain
        ).mean(dim=2) * operation_ratios.squeeze(-1)
        predicted_local_gain_sum = local_gain_per_operation.sum(dim=1, keepdim=True)
        selected_mass = soft_selection.mean(dim=2)
        selected_denominator = soft_selection.sum(dim=2, keepdim=True).clamp_min(1e-6)
        neighbor_concentration = (
            soft_selection.unsqueeze(2) * fixed_features[:, None, 0:3, :]
        ).sum(dim=3) / selected_denominator
        selection_overlap = torch.stack(
            (
                (soft_selection[:, 0] * soft_selection[:, 1]).mean(dim=1),
                (soft_selection[:, 0] * soft_selection[:, 2]).mean(dim=1),
                (soft_selection[:, 1] * soft_selection[:, 2]).mean(dim=1),
            ),
            dim=1,
        )
        selected_spatial_statistics = torch.cat(
            (selected_mass, neighbor_concentration.reshape(batch, -1), selection_overlap),
            dim=1,
        )
        interaction_input = torch.cat(
            (
                global_feature,
                shares.squeeze(-1),
                gate_probability.squeeze(-1),
                priorities.squeeze(-1),
                total_ratio.squeeze(-1),
                selected_spatial_statistics,
            ),
            dim=1,
        )
        interaction_correction = self.interaction_head(interaction_input)
        predicted_plan_gain = predicted_local_gain_sum + interaction_correction

        eps = 1e-6
        action_entropy = -(
            gate_base_probability.clamp(eps, 1.0 - eps) * gate_base_probability.clamp_min(eps).log()
            + (1.0 - gate_base_probability).clamp(eps, 1.0) * (1.0 - gate_base_probability).clamp_min(eps).log()
        ).mean()
        share_entropy = -(
            shares_mean.clamp_min(eps) * shares_mean.clamp_min(eps).log()
        ).sum(dim=1).mean()
        ratio_mean_fraction = (total_ratio_mean / self.max_total_ratio).clamp(eps, 1.0 - eps)
        ratio_mean_entropy = -(
            ratio_mean_fraction * ratio_mean_fraction.log()
            + (1.0 - ratio_mean_fraction) * (1.0 - ratio_mean_fraction).log()
        ).mean()
        amount_policy_entropy = 0.5 * (share_entropy + ratio_mean_entropy)
        where_probability = torch.softmax(base_where_logits / temperature.clamp_min(0.05), dim=2)
        where_entropy = -(where_probability.clamp_min(eps) * where_probability.clamp_min(eps).log()).sum(dim=2).mean()

        return {
            "local_cost_maps": signed_local_cost,
            "fixed_features": fixed_features,
            "local_cost_names": LOCAL_COST_NAMES,
            "coefficients": coefficients,
            "coefficient_mean": torch.tanh(coefficient_mean_logits),
            "coefficient_scale": coefficient_scale,
            "base_where_logits": base_where_logits,
            "where_logits": where_logits,
            "where_sampling_temperature": where_sampling_temperature,
            "where_threshold": where_threshold,
            # Keep the compact vector field.  Base categorical probabilities
            # are reconstructed only for executed sources in the Actuator;
            # materialising a second [B,2,26,N] tensor costs several seconds
            # of saved-tensor offload at one million points.
            "base_direction_vectors": base_vectors,
            "direction_concentration": concentration,
            "direction_logits": direction_logits,
            "direction_sampling_temperature": direction_sampling_temperature,
            "total_ratio_raw": total_ratio_raw,
            "total_ratio_unconstrained": total_ratio_unconstrained,
            "total_ratio": total_ratio,
            "total_ratio_mean": total_ratio_mean,
            "amount_sample_log_prob": amount_sample_log_prob,
            "amount_distribution_entropy": amount_distribution_entropy,
            "share_logits": share_logits,
            "shares": shares,
            "shares_raw": shares_raw,
            "shares_mean": shares_mean,
            "share_sample_log_prob": share_sample_log_prob,
            "share_distribution_entropy": share_distribution_entropy,
            "gate_logits": gate_logits,
            "gates": gates,
            "gate_probability": gate_probability,
            "gate_hard": gate_hard,
            "gate_base_probability": gate_base_probability,
            "priority_logits": priority_logits,
            "priority_base_logits": priority_base_logits,
            "priorities": priorities,
            "priority_order": priority_order,
            "priority_sample_log_prob": priority_sample_log_prob,
            "priority_entropy": priority_entropy,
            "temperature": temperature,
            "operation_ratios": operation_ratios,
            "predicted_local_gain_sum": predicted_local_gain_sum,
            "interaction_correction": interaction_correction,
            "predicted_plan_gain": predicted_plan_gain,
            "selected_spatial_statistics": selected_spatial_statistics,
            "where_entropy": where_entropy,
            "amount_entropy": amount_policy_entropy,
            "share_entropy": share_entropy,
            "ratio_mean_entropy": ratio_mean_entropy,
            "action_entropy": action_entropy,
            "exploration_active": exploration_active,
            "exploration_fraction": features.new_tensor(float(exploration_fraction)),
        }
