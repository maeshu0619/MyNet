"""Inference-safe K-slot SparsePCGC proposal policy.

The runtime module intentionally imports no den implementation, cache loader,
teacher plan, codec probe, or Actual encoder.  One shared pointwise trunk and
one shared codec-cost basis feed K independently learned plan tokens.  The
tokens deterministically rerank a shared shortlist; they are not K stochastic
draws from one logit tensor.
"""

import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .executable_voxel_plan import (
    ExecutableVoxelPlanBuilder,
    coordinate_indices,
    select_executable_plan,
)


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
        self.amount_head = nn.Linear(global_hidden, 5)
        self.share_head = nn.Linear(global_hidden, 3)
        self.order_head = nn.Linear(global_hidden, 6)
        self.variant_head = nn.Linear(global_hidden, 6)
        self.direction_delta_head = nn.Linear(global_hidden, 2 * 3)
        self.confidence_head = nn.Linear(global_hidden, 1)
        self.slot_ratio_bias = nn.Parameter(torch.zeros(self.proposal_count, 5))
        self.slot_share_bias = nn.Parameter(torch.zeros(self.proposal_count, 3))
        self.slot_order_bias = nn.Parameter(torch.zeros(self.proposal_count, 6))
        self.slot_variant_bias = nn.Parameter(torch.zeros(self.proposal_count, 6))
        self.register_buffer(
            "ratio_values",
            torch.tensor((0.0005, 0.0010, 0.0025, 0.0050, 0.0100)),
            persistent=False,
        )
        self.register_buffer(
            "order_permutations",
            torch.tensor(tuple(itertools.permutations(range(3))), dtype=torch.long),
            persistent=False,
        )

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
        self.executable_plan_builder = ExecutableVoxelPlanBuilder()
        self._initialize_mode_biases()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # 旧K checkpointの連続Amount/dead headは新しい離散thetaとshapeが異なる。
        amount_weight = prefix + "amount_head.weight"
        amount_bias = prefix + "amount_head.bias"
        if amount_weight in state_dict and tuple(state_dict[amount_weight].shape) != tuple(self.amount_head.weight.shape):
            state_dict.pop(amount_weight)
        if amount_bias in state_dict and tuple(state_dict[amount_bias].shape) != tuple(self.amount_head.bias.shape):
            state_dict.pop(amount_bias)
        for head_name in ("enable_head", "priority_head", "threshold_head", "temperature_head"):
            state_dict.pop(prefix + head_name + ".weight", None)
            state_dict.pop(prefix + head_name + ".bias", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _initialize_mode_biases(self):
        # Eight deliberately different initial modes.  They are initialization
        # only and remain fully learnable; they are not teacher labels/floors.
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
            # 小さい非零初期値で入力条件付けを初Stepから有効にする。
            nn.init.xavier_uniform_(self.amount_head.weight, gain=0.01)
            nn.init.xavier_uniform_(self.share_head.weight, gain=0.01)
            nn.init.xavier_uniform_(self.order_head.weight, gain=0.01)
            nn.init.xavier_uniform_(self.variant_head.weight, gain=0.01)
            self.amount_head.bias.zero_()
            self.share_head.bias.zero_()
            self.order_head.bias.zero_()
            self.variant_head.bias.zero_()
            self.slot_ratio_bias.fill_(-2.0)
            self.slot_order_bias.fill_(-2.0)
            self.slot_variant_bias.fill_(-2.0)
            for slot in range(self.proposal_count):
                self.slot_ratio_bias[slot, slot % 5] = 2.0
                self.slot_share_bias[slot].copy_(
                    torch.log(share_modes[slot % int(share_modes.shape[0])])
                )
                self.slot_order_bias[slot, slot % 6] = 2.0
                self.slot_variant_bias[slot, (slot * 5) % 6] = 2.0
            self.critic_gain_head.weight.zero_()
            self.critic_gain_head.bias.zero_()
            self.critic_interaction_head.weight.zero_()
            self.critic_interaction_head.bias.zero_()
            # 未学習時にgeometry/uncertaintyだけでslotが固定されない中立初期値。
            self.critic_geometry_head.weight.zero_()
            self.critic_geometry_head.bias.fill_(-6.0)
            self.critic_uncertainty_head.weight.zero_()
            self.critic_uncertainty_head.bias.fill_(-6.0)
        # Token-specific biases must be represented by tokens rather than K
        # separate heavy heads.  Seed token coordinates with amount/share modes.
        with torch.no_grad():
            for slot in range(self.proposal_count):
                ratio_index = slot % int(self.ratio_values.numel())
                self.plan_tokens[slot, 0] = float(ratio_index) / 4.0
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
        shared_weight = coefficients.detach().abs().mean(dim=1).unsqueeze(-1)
        union_score = (basis.abs() * shared_weight).amax(dim=2).amax(dim=1)
        size = min(int(points), int(self.shortlist_size))
        return union_score.topk(size, dim=1, largest=True, sorted=True).indices

    def _augment_training_shortlist(
        self, natural_indices, voxel_coords, teacher_coords, teacher_target_coords
    ):
        """訓練時だけ既知teacher sourceを追加し、自然recallは別Tensorで保持する。"""
        if teacher_coords is None and teacher_target_coords is None:
            return natural_indices
        if natural_indices.shape[0] != 1:
            raise RuntimeError("teacher shortlist augmentation currently requires batch=1")
        coords = voxel_coords[0].transpose(0, 1) if voxel_coords.shape[1] == 3 else voxel_coords[0]
        query_parts = []
        if teacher_coords:
            query_parts.append(torch.as_tensor(
                teacher_coords, device=coords.device, dtype=torch.long
            ).reshape(-1, 3))
        # target集合は実行shortlistを膨らませず、専用loss domainで扱う。
        if not query_parts:
            return natural_indices
        query = torch.cat(query_parts, dim=0)
        matched = coordinate_indices(query, coords.long())
        matched = matched[matched >= 0]
        if not matched.numel():
            return natural_indices
        return torch.unique(
            torch.cat((natural_indices[0], matched), dim=0), sorted=True
        ).view(1, -1)

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
        all_actual_exploration = bool(
            training and getattr(args, "network_k_all_actual_enabled", False)
        )
        exploration_temperature_start = max(float(getattr(
            args, "network_k_all_actual_temperature", 1.0
        )), 0.05)
        exploration_temperature_min = min(max(float(getattr(
            args, "network_k_all_actual_temperature_min", 0.25
        )), 0.01), exploration_temperature_start)
        anneal_steps = max(int(getattr(
            args, "network_k_all_actual_anneal_steps", 500
        )), 1)
        anneal_progress = min(max(
            float(getattr(args, "_global_train_step", 0)) / float(anneal_steps), 0.0
        ), 1.0)
        exploration_temperature = (
            exploration_temperature_start
            + (exploration_temperature_min - exploration_temperature_start)
            * anneal_progress
        )

        def sample_categorical(logits):
            """探索量を実際のhard標本へ反映するCategorical STE。"""
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            noisy = logits + exploration_temperature * gumbel
            hard = F.one_hot(noisy.argmax(dim=2), num_classes=logits.shape[2]).to(logits)
            soft = torch.softmax(noisy / max(exploration_temperature, 0.05), dim=2)
            return hard.detach() + soft - soft.detach()
        if fixed_features is None:
            raise RuntimeError(
                "K proposalには入力由来Octree/local fixed featuresが必須である"
            )
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

        coefficient_mean_raw = self.coefficient_head(slot_feature).view(
            batch, self.proposal_count, 3, len(LOCAL_COST_NAMES)
        )
        coefficient_std = max(float(getattr(
            args, "network_k_all_actual_coefficient_std", 0.15
        )), 1e-4)
        if all_actual_exploration:
            coefficient_sample_raw = coefficient_mean_raw + coefficient_std * torch.randn_like(
                coefficient_mean_raw
            )
        else:
            coefficient_sample_raw = coefficient_mean_raw
        coefficients = torch.tanh(coefficient_sample_raw)
        coefficient_log_prob = -0.5 * (
            (coefficient_sample_raw.detach() - coefficient_mean_raw) / coefficient_std
        ).pow(2).mean(dim=(2, 3))
        ratio_values = tuple(float(value) for value in getattr(
            args, "network_k_ratio_values", tuple(self.ratio_values.tolist())
        ))
        if len(ratio_values) != 5:
            raise ValueError("network_k_ratio_valuesは5値でなければならない")
        ratio_values_tensor = features.new_tensor(ratio_values)
        ratio_logits = self.amount_head(slot_feature) + self.slot_ratio_bias.unsqueeze(0)
        ratio_probability = torch.softmax(ratio_logits, dim=2)
        if all_actual_exploration:
            ratio_ste = sample_categorical(ratio_logits)
            ratio_class = ratio_ste.argmax(dim=2)
        else:
            ratio_class = ratio_logits.argmax(dim=2)
            ratio_hard = F.one_hot(ratio_class, num_classes=5).to(features.dtype)
            ratio_ste = ratio_hard.detach() + ratio_probability - ratio_probability.detach()
        ratio_log_prob = (
            ratio_ste.detach() * torch.log_softmax(ratio_logits, dim=2)
        ).sum(dim=2)
        total_ratio = (ratio_ste * ratio_values_tensor.view(1, 1, 5)).sum(2, keepdim=True)

        share_logits = self.share_head(slot_feature) + self.slot_share_bias.unsqueeze(0)
        if all_actual_exploration:
            uniform = torch.rand_like(share_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            shares_soft = torch.softmax(
                (share_logits + exploration_temperature * gumbel)
                / max(exploration_temperature, 0.05), dim=2
            )
        else:
            shares_soft = torch.softmax(share_logits, dim=2)
        lattice_step = float(getattr(args, "network_k_share_lattice_step", 0.05))
        if lattice_step > 0.0:
            shares_hard = torch.round(shares_soft / lattice_step) * lattice_step
            shares_hard = shares_hard.clamp_min(lattice_step)
            shares_hard = shares_hard / shares_hard.sum(2, keepdim=True).clamp_min(1e-8)
            shares = shares_hard.detach() + shares_soft - shares_soft.detach()
        else:
            shares_hard = shares_soft.detach()
            shares = shares_soft
        share_log_prob = (
            shares_hard.detach() * torch.log_softmax(share_logits, dim=2)
        ).sum(dim=2)

        # v1/v2の既存163 planは3操作すべてONであり、enableは識別不能なので固定する。
        enable_probability = features.new_ones((batch, self.proposal_count, 3))
        enables = features.new_ones((batch, self.proposal_count, 3))
        enable_logits = features.new_full((batch, self.proposal_count, 3), 12.0)

        order_logits = self.order_head(slot_feature) + self.slot_order_bias.unsqueeze(0)
        order_probability = torch.softmax(order_logits, dim=2)
        if all_actual_exploration:
            order_sample = sample_categorical(order_logits)
            order_class = order_sample.argmax(dim=2)
        else:
            order_class = order_logits.argmax(dim=2)
            order_sample = F.one_hot(order_class, num_classes=6).to(features.dtype)
        order_log_prob = (
            order_sample.detach() * torch.log_softmax(order_logits, dim=2)
        ).sum(dim=2)
        operation_order = self.order_permutations.to(order_class).index_select(
            0, order_class.reshape(-1)
        ).view(batch, self.proposal_count, 3)
        priorities = features.new_zeros((batch, self.proposal_count, 3))
        rank_values = features.new_tensor((3.0, 2.0, 1.0)).view(1, 1, 3).expand_as(priorities)
        priorities.scatter_(2, operation_order, rank_values)
        variant_logits = self.variant_head(slot_feature) + self.slot_variant_bias.unsqueeze(0)
        if all_actual_exploration:
            variant_sample = sample_categorical(variant_logits)
            variant_class = variant_sample.argmax(dim=2)
        else:
            variant_class = variant_logits.argmax(dim=2)
            variant_sample = F.one_hot(variant_class, num_classes=6).to(features.dtype)
        variant_log_prob = (
            variant_sample.detach() * torch.log_softmax(variant_logits, dim=2)
        ).sum(dim=2)
        confidence = torch.sigmoid(self.confidence_head(slot_feature))

        natural_shortlist_indices = self._shared_shortlist(basis, coefficients, points)
        teacher_coords = getattr(args, "_network_k_training_teacher_coords", None)
        teacher_target_coords = getattr(
            args, "_network_k_training_teacher_target_coords", None
        )
        if (teacher_coords is not None or teacher_target_coords is not None) and not training:
            raise RuntimeError("teacher shortlistは訓練時にだけ使用できる")
        shortlist_indices = self._augment_training_shortlist(
            natural_shortlist_indices,
            voxel_coords,
            teacher_coords if training else None,
            teacher_target_coords if training else None,
        )
        shortlist_basis = self._gather_points(basis, shortlist_indices)
        base_slot_logits = torch.einsum(
            "bocm,bkoc->bkom", shortlist_basis, coefficients
        )
        base_slot_logits = base_slot_logits + priorities.unsqueeze(-1)
        # den6のpool走査variantを、共有shortlist順位への小さな決定論的傾きで表す。
        variant_slopes = features.new_tensor((
            (0.0, 0.0, 0.0),
            (0.08, 0.0, 0.0),
            (0.0, 0.08, 0.0),
            (0.0, 0.0, 0.08),
            (-0.05, 0.05, 0.0),
            (0.05, 0.0, -0.05),
        ))
        selected_slopes = variant_slopes.index_select(
            0, variant_class.reshape(-1)
        ).view(batch, self.proposal_count, 3, 1)
        shortlist_rank = torch.linspace(
            1.0, 0.0, shortlist_indices.shape[1],
            device=features.device, dtype=features.dtype,
        ).view(1, 1, 1, -1)
        slot_scale = base_slot_logits.detach().std(dim=3, keepdim=True).clamp_min(1e-4)
        base_slot_logits = base_slot_logits + selected_slopes * shortlist_rank * slot_scale
        if all_actual_exploration:
            uniform = torch.rand_like(base_slot_logits).clamp_(1e-6, 1.0 - 1e-6)
            where_gumbel = -torch.log(-torch.log(uniform))
            slot_logits = base_slot_logits + where_gumbel * exploration_temperature
        else:
            slot_logits = base_slot_logits

        if not torch.is_tensor(voxel_coords):
            raise RuntimeError("K proposalにはglobal voxel座標が必須である")
        target_domain = str(getattr(
            args, "network_k_target_domain", "neighbor26_empty"
        )).strip().lower()
        if target_domain != "neighbor26_empty":
            raise RuntimeError(
                "K policyのchild_slot domainには明示的direction_valid_maskが必要であり未接続である"
            )
        requested_continuous = total_ratio * shares * enable_probability * float(points)
        requested_count = (
            total_ratio * shares * enables * float(points)
        ).round().to(dtype=torch.long)
        requested_count = requested_count.clamp(min=0, max=int(shortlist_indices.shape[1]))
        # builderへshortlist indexを渡し、K×全点score Tensorも作らない。
        direction_delta_mean = self.direction_delta_head(slot_feature).view(
            batch, self.proposal_count, 2, 3
        )
        direction_std = max(float(getattr(
            args, "network_k_all_actual_direction_std", 0.10
        )), 1e-4)
        if all_actual_exploration:
            direction_delta_all = direction_delta_mean + direction_std * torch.randn_like(
                direction_delta_mean
            )
        else:
            direction_delta_all = direction_delta_mean
        direction_log_prob = -0.5 * (
            (direction_delta_all.detach() - direction_delta_mean) / direction_std
        ).pow(2).mean(dim=(2, 3))
        slot_direction_logits = None
        if (
            bool(training)
            and not all_actual_exploration
            and bool(str(getattr(args, "network_k_offline_dataset", "")).strip())
        ):
            shortlist_direction = self._gather_points(direction_field, shortlist_indices)
            shortlist_vectors = F.normalize(
                shortlist_direction[:, None, :, :3]
                + direction_delta_all.unsqueeze(-1),
                dim=3,
                eps=1e-6,
            )
            shortlist_concentration = F.softplus(
                shortlist_direction[:, None, :, 3:4]
            ) + 0.1
            slot_direction_logits = torch.einsum(
                "bkodm,qd->bkoqm",
                shortlist_vectors,
                self.unit_neighbor_offsets.to(shortlist_vectors),
            ) * shortlist_concentration

        slot_target_logits = None
        target_candidate_coords = None
        if teacher_target_coords is not None:
            if batch != 1:
                raise RuntimeError("target-set教師はbatch=1のoffline訓練専用である")
            target_rows = torch.as_tensor(
                teacher_target_coords, device=features.device, dtype=torch.long
            ).reshape(-1, 3)
            target_candidate_coords = target_rows.transpose(0, 1).unsqueeze(0)
            offsets = self.executable_plan_builder.neighbor_offsets.to(features.device)
            all_coords = (
                voxel_coords[0].transpose(0, 1)
                if voxel_coords.shape[1] == 3 else voxel_coords[0]
            ).long()
            target_count = int(target_rows.shape[0])
            target_chunks = []
            for start in range(0, target_count, 512):
                chunk = target_rows[start : start + 512]
                source_rows = (
                    chunk[:, None, :] - offsets[None, :, :]
                ).reshape(-1, 3)
                source_global = coordinate_indices(source_rows, all_coords)
                reachable = source_global >= 0
                safe_global = source_global.clamp_min(0)
                source_basis = self._gather_points(
                    basis, safe_global.view(1, -1)
                )
                source_score = torch.einsum(
                    "bocp,bkoc->bkop", source_basis, coefficients
                ) + priorities.unsqueeze(-1)
                source_direction = self._gather_points(
                    direction_field, safe_global.view(1, -1)
                )
                source_vectors = F.normalize(
                    source_direction[:, None, :, :3]
                    + direction_delta_all.unsqueeze(-1),
                    dim=3, eps=1e-6,
                )
                pair_offsets = offsets.repeat(chunk.shape[0], 1).to(
                    source_vectors.dtype
                )
                direction_score = torch.einsum(
                    "bkodp,pd->bkop", source_vectors, pair_offsets
                ) * (F.softplus(source_direction[:, None, :, 3]) + 0.1)
                chunk_values = features.new_full(
                    (1, self.proposal_count, 3, chunk.shape[0]), -20.0
                )
                for operation, direction_operation in ((1, 0), (2, 1)):
                    pair_score = (
                        source_score[:, :, operation]
                        + direction_score[:, :, direction_operation]
                    ).view(1, self.proposal_count, chunk.shape[0], 26)
                    pair_score = pair_score.masked_fill(
                        ~reachable.view(1, 1, chunk.shape[0], 26), -torch.inf
                    )
                    value = torch.logsumexp(pair_score, dim=3)
                    chunk_values[:, :, operation] = torch.where(
                        torch.isfinite(value), value, value.new_full((), -20.0)
                    )
                target_chunks.append(chunk_values)
            slot_target_logits = torch.cat(target_chunks, dim=3)

        def _direction_provider(batch_index, slot_index, operation, source_indices):
            direction_operation = 0 if int(operation) == 1 else 1
            vectors = direction_field[
                batch_index, direction_operation, :3
            ].index_select(1, source_indices).transpose(0, 1)
            vectors = F.normalize(
                vectors + direction_delta_all[batch_index, slot_index, direction_operation],
                dim=1,
                eps=1e-6,
            )
            concentration_value = F.softplus(
                direction_field[batch_index, direction_operation, 3]
                .index_select(0, source_indices)
            ).unsqueeze(1) + 0.1
            return (
                vectors @ self.unit_neighbor_offsets.to(vectors).transpose(0, 1)
            ) * concentration_value

        executable_plans = self.executable_plan_builder.build(
            voxel_coords=voxel_coords,
            operation_scores=slot_logits,
            requested_count=requested_count,
            operation_order=operation_order,
            direction_logit_provider=_direction_provider,
            operation_enabled=enables.to(dtype=torch.bool),
            source_indices=shortlist_indices,
            target_coord_min=voxel_coords.new_tensor((0, 0, 0)),
            target_coord_max=voxel_coords.new_tensor((
                int(getattr(args, "sparsepcgc_psnr_resolution", 1023) or 1023),
            ) * 3),
            # 全K Actual診断ではslot collapseをplan単位で必ず検出する。
            debug_hash=bool(
                all_actual_exploration
                or getattr(args, "network_k_debug_plan_hash", False)
            ),
        )
        accepted_count = executable_plans.accepted_count.to(dtype=features.dtype)
        collision_count = (requested_count - executable_plans.accepted_count).clamp_min(0).to(
            dtype=features.dtype
        )

        # hard plan descriptorをforward値に保ちつつ、Amount/shareへSTE勾配を返す。
        descriptor = executable_plans.plan_descriptor
        soft_requested_ratio = requested_continuous / max(float(points), 1.0)
        soft_accepted_ratio = soft_requested_ratio
        soft_share = shares
        descriptor = descriptor.clone()
        descriptor[:, :, 0:3] = (
            descriptor[:, :, 0:3].detach()
            + soft_requested_ratio - soft_requested_ratio.detach()
        )
        descriptor[:, :, 3:6] = (
            descriptor[:, :, 3:6].detach()
            + soft_accepted_ratio - soft_accepted_ratio.detach()
        )
        descriptor[:, :, 6:9] = (
            descriptor[:, :, 6:9].detach() + soft_share - soft_share.detach()
        )

        # 実行planだけをWhere教師・ログへ公開する。
        selected_shortlist_mask_all = torch.zeros_like(slot_logits, dtype=torch.bool)
        shortlist_inverse = torch.full(
            (batch, points), -1, device=features.device, dtype=torch.long
        )
        shortlist_inverse.scatter_(
            1,
            shortlist_indices,
            torch.arange(shortlist_indices.shape[1], device=features.device)
            .view(1, -1).expand(batch, -1),
        )
        for batch_index in range(batch):
            for slot_index in range(self.proposal_count):
                for operation in range(3):
                    mask = executable_plans.accepted_mask[batch_index, slot_index, operation]
                    source = executable_plans.source_index[
                        batch_index, slot_index, operation
                    ][mask]
                    if source.numel() == 0:
                        continue
                    position = shortlist_inverse[batch_index].index_select(0, source)
                    position = position[position >= 0]
                    selected_shortlist_mask_all[
                        batch_index, slot_index, operation, position
                    ] = True
        where_log_probability = torch.log_softmax(base_slot_logits, dim=3)
        selected_where_count = selected_shortlist_mask_all.sum(dim=(2, 3)).clamp_min(1)
        where_log_prob = (
            where_log_probability * selected_shortlist_mask_all.to(where_log_probability.dtype)
        ).sum(dim=(2, 3)) / selected_where_count.to(where_log_probability.dtype)
        slot_policy_log_prob = (
            ratio_log_prob + share_log_prob + order_log_prob + variant_log_prob
            + where_log_prob + coefficient_log_prob + direction_log_prob
        )
        slot_policy_entropy = (
            -(ratio_probability.clamp_min(1e-8) * ratio_probability.clamp_min(1e-8).log()).sum(2)
            -(order_probability.clamp_min(1e-8) * order_probability.clamp_min(1e-8).log()).sum(2)
            -(torch.softmax(variant_logits, 2).clamp_min(1e-8)
              * torch.log_softmax(variant_logits, 2)).sum(2)
            -(torch.softmax(share_logits, 2).clamp_min(1e-8)
              * torch.log_softmax(share_logits, 2)).sum(2)
            -(torch.softmax(base_slot_logits, 3).clamp_min(1e-8)
              * torch.log_softmax(base_slot_logits, 3)).sum(3).mean(2)
        )
        compact = {
            "selected_shortlist_mask": selected_shortlist_mask_all,
            "requested_count": requested_count,
            "accepted_count": accepted_count,
            "collision_count": collision_count,
            "hard_ratio": accepted_count / max(float(points), 1.0),
            "hard_share": accepted_count / accepted_count.sum(2, keepdim=True).clamp_min(1.0),
            "descriptor": descriptor,
        }
        critic_input = torch.cat((slot_feature, compact["descriptor"]), dim=2)
        critic_hidden = self.critic(critic_input)
        predicted_gain = self.critic_gain_head(critic_hidden)
        predicted_geometry = F.softplus(self.critic_geometry_head(critic_hidden))
        predicted_interaction = self.critic_interaction_head(critic_hidden)
        uncertainty = F.softplus(self.critic_uncertainty_head(critic_hidden)) + 1e-4
        predicted_plan_gain = predicted_gain + predicted_interaction
        gain_scale = max(float(getattr(args, "network_k_critic_gain_scale", 5.0)), 1e-6)
        geometry_scale = max(float(getattr(args, "network_k_critic_geometry_scale", 1.0)), 1e-6)
        lambda_geometry = max(float(
            getattr(args, "network_k_critic_lambda_geometry", 1.0)
        ), 0.0)
        lambda_uncertainty = max(float(
            getattr(args, "network_k_critic_lambda_uncertainty", 0.05)
        ), 0.0)
        critic_score = (
            predicted_plan_gain / gain_scale
            - lambda_geometry * predicted_geometry / geometry_scale
            - lambda_uncertainty * uncertainty * (1.0 - confidence)
        )
        selected_slot = critic_score.squeeze(-1).argmax(dim=1)
        selected_executable_plan = select_executable_plan(executable_plans, selected_slot)
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
            direction_delta_all,
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
            accepted_total / max(float(points), 1.0)
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
        selected_enables_hard = torch.ones_like(
            selected_enable_probability, dtype=features.dtype
        )
        selected_enables = (
            selected_enables_hard.detach()
            + selected_enable_probability
            - selected_enable_probability.detach()
        )
        selected_priorities = self._select_slot_tensor(priorities, selected_slot).unsqueeze(-1)
        selected_temperature = features.new_ones((batch, 1, 1))
        selected_threshold = features.new_zeros((batch, 3, 1))

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
            "total_ratio_raw": selected_total_ratio_raw / max(
                float(ratio_values_tensor.max().item()), 1e-8
            ),
            "total_ratio_unconstrained": selected_total_ratio_raw,
            "total_ratio": selected_total_ratio,
            "total_ratio_mean": selected_total_ratio_raw,
            "amount_sample_log_prob": selected_total_ratio.sum() * 0.0,
            "amount_distribution_entropy": -(
                ratio_probability.clamp_min(1e-8) * ratio_probability.clamp_min(1e-8).log()
            ).sum(2).mean(),
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
            "amount_entropy": -(
                ratio_probability.clamp_min(1e-8) * ratio_probability.clamp_min(1e-8).log()
            ).sum(2).mean(),
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
            "executable_plans": vars(executable_plans),
            "executable_plan_batch": executable_plans,
            "selected_executable_plan": selected_executable_plan,
            "shared_basis": basis,
            "shared_direction_field": direction_field,
            "shortlist_indices": shortlist_indices,
            "natural_shortlist_indices": natural_shortlist_indices,
            "unique_executable_plan_count": (
                min(len(set(row)) for row in executable_plans.plan_hash)
                if executable_plans.plan_hash is not None else -1
            ),
            "slot_logits": slot_logits,
            "slot_direction_logits": slot_direction_logits,
            "slot_target_logits": slot_target_logits,
            "target_candidate_coords": target_candidate_coords,
            "slot_features": slot_feature,
            "total_ratio": total_ratio,
            "ratio_logits": ratio_logits,
            "ratio_class": ratio_class,
            "shares": shares,
            "enable_probability": enable_probability,
            "enables": enables,
            "priorities": priorities,
            "operation_order": operation_order,
            "order_logits": order_logits,
            "order_class": order_class,
            "variant_logits": variant_logits,
            "variant_class": variant_class,
            "slot_policy_log_prob": slot_policy_log_prob,
            "slot_policy_entropy": slot_policy_entropy,
            "all_actual_exploration": all_actual_exploration,
            "exploration_temperature": features.new_tensor(exploration_temperature),
            "exploration_anneal_progress": features.new_tensor(anneal_progress),
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
