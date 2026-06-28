import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_surrogate_pred_clip(args):
    clip = float(getattr(args, "surrogate_pred_clip_percent", -1.0))
    if clip < 0.0:
        clip = float(getattr(args, "compression_surrogate_pred_clip", 0.0))
    return max(float(clip), 0.0)


def resolve_surrogate_target_clip(args):
    # Target clipping changes the teacher value itself.  Keep the actual codec
    # percent raw unless the caller explicitly opts in.
    clip = float(getattr(args, "surrogate_target_clip_percent", 0.0))
    return max(float(clip), 0.0)


ACTUAL_OCCUPANCY_DEBUG_KEYS = (
    "actual_occupancy_pattern_before",
    "actual_occupancy_pattern_after",
    "actual_occupancy_pattern_delta",
    "actual_occupancy_entropy_before",
    "actual_occupancy_entropy_after",
    "actual_occupancy_entropy_delta",
    "actual_occupancy_nll_before",
    "actual_occupancy_nll_after",
    "actual_occupancy_nll_delta",
    "actual_lowprob_occupancy_count_before",
    "actual_lowprob_occupancy_count_after",
    "actual_lowprob_occupancy_count_delta",
    "actual_lowprob_occupancy_ratio_before",
    "actual_lowprob_occupancy_ratio_after",
    "actual_lowprob_occupancy_ratio_delta",
    "actual_occupancy_predictability_before",
    "actual_occupancy_predictability_after",
    "actual_occupancy_predictability_delta",
)


class _CompressionSurrogateNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, pred_clip=2.0):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 16)
        self.pred_clip = float(pred_clip)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_raw(self, x):
        return self.net(x)

    def forward(self, x):
        raw = self.forward_raw(x)
        if self.pred_clip > 0:
            return self.pred_clip * torch.tanh(raw / self.pred_clip)
        return raw


class SurrogateCompressionLossMixin:
    @staticmethod
    def _parse_surrogate_levels(args):
        raw = getattr(args, "compression_surrogate_levels", "4,6,8")
        if isinstance(raw, (list, tuple)):
            vals = raw
        else:
            vals = str(raw).replace(" ", "").split(",")
        levels = []
        for val in vals:
            if val == "":
                continue
            level = int(val)
            if level > 0:
                levels.append(level)
        return levels or [4, 6, 8]

    def _ensure_surrogate_device(self, device):
        first_param = next(self.compression_surrogate.parameters())
        if first_param.device != device:
            self.compression_surrogate = self.compression_surrogate.to(device)
            state = self.surrogate_optimizer.state
            for value in state.values():
                for key, item in value.items():
                    if torch.is_tensor(item):
                        value[key] = item.to(device)

    @staticmethod
    def _all_finite(*values):
        for value in values:
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                return False
        return True

    @staticmethod
    def _safe_log_bit_ratio(before_bits, after_bits):
        if before_bits is None or after_bits is None:
            return float("nan")
        before_bits = float(before_bits)
        after_bits = float(after_bits)
        if not (math.isfinite(before_bits) and math.isfinite(after_bits)) or before_bits <= 0.0 or after_bits <= 0.0:
            return float("nan")
        return float(math.log(after_bits / before_bits))

    def _prepare_surrogate_target(self, args, raw_percent, device, *, before_bits=None, after_bits=None):
        raw_percent = float(raw_percent)
        clip = resolve_surrogate_target_clip(args)
        if clip > 0.0:
            clamped_percent = min(max(raw_percent, -clip), clip)
            clip_min = -clip
            clip_max = clip
        else:
            clamped_percent = raw_percent
            clip_min = float("nan")
            clip_max = float("nan")
        log_ratio = self._safe_log_bit_ratio(before_bits, after_bits)
        log_scale = max(float(getattr(args, "surrogate_log_bit_ratio_scale", 100.0)), 1e-9)
        use_log_ratio = bool(getattr(args, "surrogate_use_log_bit_ratio_target", False)) and math.isfinite(log_ratio)
        train_value = log_ratio * log_scale if use_log_ratio else clamped_percent
        target_mode = "log_bit_ratio_scaled" if use_log_ratio else ("percent_clamped" if clip > 0.0 else "percent_raw")
        target = torch.tensor([[float(train_value)]], device=device, dtype=torch.float32)
        return {
            "target": target,
            "raw_percent": float(raw_percent),
            "clamped_percent": float(clamped_percent),
            "train_value": float(train_value),
            "target_clamped": abs(float(clamped_percent) - float(raw_percent)) > 1e-6,
            "target_mode": target_mode,
            "log_ratio": float(log_ratio),
            "clip_min": float(clip_min),
            "clip_max": float(clip_max),
            "log_scale": float(log_scale),
        }

    def _surrogate_feature_names(self, args):
        names = [
            "log_input_points",
            "log_weight_sum",
            "weight_mean",
            "weight_std",
            "bbox_norm_mean_x",
            "bbox_norm_mean_y",
            "bbox_norm_mean_z",
            "bbox_norm_std_x",
            "bbox_norm_std_y",
            "bbox_norm_std_z",
            "bbox_volume_log",
        ]
        stat_names = ["log_node", "log_single", "occupancy_entropy_per_node", "log_mass", "node_density"]
        for level in self.surrogate_levels:
            names.extend([f"level{int(level)}_{name}" for name in stat_names])
        names.extend([f"qlevel_{name}" for name in stat_names])
        names.extend(["log_q_norm", "log_inv_q_norm", "q_level_norm", "codec_sparsepcgc", "codec_gpcc", "codec_draco"])
        return names

    def _update_teacher_gap_debug(self, args, cache_key, teacher_type, actual_percent, teacher_is_actual):
        if not hasattr(self, "_teacher_gap_cache"):
            self._teacher_gap_cache = {}
        if not teacher_is_actual or not math.isfinite(float(actual_percent)):
            return {"teacher_gap_percent": None, "teacher_gap_status": "not_actual_teacher"}
        sample_key = str(cache_key or getattr(args, "_current_sample_name", "unknown"))
        entry = self._teacher_gap_cache.setdefault(sample_key, {})
        teacher_type = str(teacher_type)
        if teacher_type == "full_cloud_actual":
            previous = entry.get("subtree_local_actual")
            entry["full_cloud_actual"] = float(actual_percent)
            if previous is None:
                return {"teacher_gap_percent": None, "teacher_gap_status": "no_previous_subtree_teacher"}
            return {"teacher_gap_percent": float(actual_percent) - float(previous), "teacher_gap_status": "full_cloud_minus_last_subtree"}
        if teacher_type == "subtree_local_actual":
            previous = entry.get("full_cloud_actual")
            entry["subtree_local_actual"] = float(actual_percent)
            if previous is None:
                return {"teacher_gap_percent": None, "teacher_gap_status": "no_previous_full_cloud_teacher"}
            return {"teacher_gap_percent": float(actual_percent) - float(previous), "teacher_gap_status": "subtree_minus_last_full_cloud"}
        return {"teacher_gap_percent": None, "teacher_gap_status": "unsupported_teacher_type"}

    def _log_surrogate_event(self, message):
        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(f"[CompressionSurrogate] {message}")

    def _update_sparsepcgc_aux_gate(self, args, aux_value, actual_value, teacher_is_actual):
        if not hasattr(self, "_sparsepcgc_aux_gate_pairs"):
            self._sparsepcgc_aux_gate_pairs = []
        if teacher_is_actual and math.isfinite(float(aux_value)) and math.isfinite(float(actual_value)):
            self._sparsepcgc_aux_gate_pairs.append((float(aux_value), float(actual_value)))
        window = max(int(getattr(args, "sparsepcgc_aux_gating_window", 100)), 2)
        self._sparsepcgc_aux_gate_pairs = self._sparsepcgc_aux_gate_pairs[-window:]
        pairs = list(self._sparsepcgc_aux_gate_pairs)
        if len(pairs) < 2:
            return None, None, len(pairs)
        aux = np.asarray([item[0] for item in pairs], dtype=np.float64)
        actual = np.asarray([item[1] for item in pairs], dtype=np.float64)
        if float(np.std(aux)) <= 1e-12 or float(np.std(actual)) <= 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(aux, actual)[0, 1])
            corr = corr if math.isfinite(corr) else 0.0
        sign_match = float(np.mean(np.sign(aux) == np.sign(actual)))
        return corr, sign_match, len(pairs)

    def _surrogate_state_is_finite(self):
        for param in self.compression_surrogate.parameters():
            if not torch.isfinite(param.detach()).all():
                return False
        for state in self.surrogate_optimizer.state.values():
            for item in state.values():
                if torch.is_tensor(item) and not torch.isfinite(item.detach()).all():
                    return False
        return True

    def _reset_compression_surrogate(self, reason):
        device = next(self.compression_surrogate.parameters()).device
        self.compression_surrogate = _CompressionSurrogateNet(
            in_dim=self.surrogate_feature_dim,
            hidden_dim=int(getattr(self.args, "compression_surrogate_hidden_dim", 128)),
            pred_clip=resolve_surrogate_pred_clip(self.args),
        ).to(device)
        self.surrogate_optimizer = torch.optim.Adam(
            self.compression_surrogate.parameters(),
            lr=max(float(getattr(self.args, "compression_surrogate_lr", 1e-3)), float(getattr(self.args, "min_surrogate_lr", 1e-6))),
            weight_decay=float(getattr(self.args, "compression_surrogate_weight_decay", 1e-5)),
        )
        self.compression_surrogate.eval()
        self._set_surrogate_trainable(False)
        self._log_surrogate_event(f"reset network ({reason}).")
        for_better_path = getattr(self.args, "_for_better_path", None)
        if for_better_path:
            try:
                from models.utils.training.for_better_logging import log_for_better_event

                log_for_better_event(
                    for_better_path,
                    "compression_surrogate_reset",
                    reason=reason,
                    global_step=int(getattr(self.args, "_global_train_step", 0)),
                )
            except Exception:
                pass

    def _surrogate_target_from_actual(self, args, stats_gen, stats_ref, device):
        target_percent = self._relative_percent(float(stats_gen["bit"]), float(stats_ref["bit"]))
        return self._prepare_surrogate_target(
            args,
            target_percent,
            device,
            before_bits=float(stats_ref.get("bit", float("nan"))),
            after_bits=float(stats_gen.get("bit", float("nan"))),
        )

    @staticmethod
    def _decode_keys(keys, grid):
        grid_t = torch.as_tensor(grid, device=keys.device, dtype=torch.long)
        xy = grid_t * grid_t
        z = torch.div(keys, xy, rounding_mode="floor")
        rem = keys - z * xy
        y = torch.div(rem, grid_t, rounding_mode="floor")
        x = rem - y * grid_t
        return torch.stack([x, y, z], dim=1)

    def _soft_level_stats(self, coords_norm, weights, level, args):
        grid = 2 ** int(level)
        scaled = (coords_norm * float(grid - 1)).clamp(0.0, float(grid - 1))
        base = torch.floor(scaled).to(torch.long)
        frac = scaled - base.to(scaled.dtype)

        masses = []
        keys = []
        for bx in (0, 1):
            wx = frac[0] if bx else (1.0 - frac[0])
            ix = (base[0] + bx).clamp(0, grid - 1)
            for by in (0, 1):
                wy = frac[1] if by else (1.0 - frac[1])
                iy = (base[1] + by).clamp(0, grid - 1)
                for bz in (0, 1):
                    wz = frac[2] if bz else (1.0 - frac[2])
                    iz = (base[2] + bz).clamp(0, grid - 1)
                    corner_weight = (wx * wy * wz).clamp_min(0.0)
                    masses.append(weights * corner_weight)
                    keys.append(ix + grid * (iy + grid * iz))

        mass = torch.cat(masses, dim=0)
        key = torch.cat(keys, dim=0)
        order = torch.argsort(key)
        key = key[order]
        mass = mass[order]
        unique_key, inverse = torch.unique_consecutive(key, return_inverse=True)
        voxel_mass = mass.new_zeros(unique_key.shape[0]).scatter_add(0, inverse, mass)

        gain = float(getattr(args, "compression_surrogate_occ_gain", 1.0))
        occ = 1.0 - torch.exp(-gain * voxel_mass.clamp_min(0.0))
        occ = occ.clamp(1e-6, 1.0 - 1e-6)

        node = occ.sum()
        entropy = -(occ * torch.log2(occ) + (1.0 - occ) * torch.log2(1.0 - occ)).sum()
        mass_total = voxel_mass.sum()

        if level <= 1 or unique_key.numel() == 0:
            single = node.new_zeros(())
        else:
            coords = self._decode_keys(unique_key, grid)
            parent_grid = grid // 2
            parent = torch.div(coords, 2, rounding_mode="floor")
            child_bits = coords - parent * 2
            child_id = child_bits[:, 0] * 4 + child_bits[:, 1] * 2 + child_bits[:, 2]
            parent_key = parent[:, 0] + parent_grid * (parent[:, 1] + parent_grid * parent[:, 2])
            parent_unique, parent_inv = torch.unique(parent_key, sorted=True, return_inverse=True)
            child_occ = occ.new_zeros(parent_unique.numel() * 8)
            flat_child_idx = parent_inv * 8 + child_id
            child_occ = child_occ.scatter_add(0, flat_child_idx, occ).view(-1, 8).clamp(1e-6, 1.0 - 1e-6)
            not_occ = (1.0 - child_occ).clamp(1e-6, 1.0)
            prod_not = not_occ.prod(dim=1, keepdim=True)
            single_prob = (child_occ * prod_not / not_occ).sum(dim=1)
            single = single_prob.sum()

        node_safe = node.clamp_min(1e-6)
        grid_total = float(grid ** 3)
        return torch.stack(
            [
                torch.log1p(node),
                torch.log1p(single),
                entropy / node_safe,
                torch.log1p(mass_total),
                node / max(grid_total, 1.0),
            ]
        )

    def _build_soft_compression_features(self, args, gen_xyz, gt_xyz, final_w):
        gen_xyz = torch.nan_to_num(gen_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        gt_xyz = torch.nan_to_num(gt_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        B, _, N = gen_xyz.shape
        if final_w is None:
            weights_all = gen_xyz.new_ones(B, N)
        else:
            weights_all = final_w.squeeze(1).to(device=gen_xyz.device, dtype=gen_xyz.dtype)
            if weights_all.shape[-1] > N:
                weights_all = weights_all[..., :N]
            elif weights_all.shape[-1] < N:
                pad = weights_all.new_ones(*weights_all.shape[:-1], N - weights_all.shape[-1])
                weights_all = torch.cat([weights_all, pad], dim=-1)
            weights_all = torch.nan_to_num(weights_all, nan=0.0, posinf=1.0, neginf=0.0)
            weights_all = weights_all.clamp(0.0, 1.0)

        features = []
        ref_min = gt_xyz.detach().amin(dim=2)
        ref_max = gt_xyz.detach().amax(dim=2)
        ref_span = (ref_max - ref_min).clamp_min(1e-6)
        codec_key = self._surrogate_backend_label(args).replace("_surrogate", "").replace("_actual", "")
        is_sparsepcgc = 1.0 if codec_key == "sparsepcgc" else 0.0
        is_gpcc = 1.0 if codec_key == "gpcc" else 0.0
        is_draco = 1.0 if codec_key == "draco" else 0.0
        effective_qs = self._surrogate_effective_qs(args, codec_key)

        for b in range(B):
            pts = gen_xyz[b].to(torch.float32)
            weights = weights_all[b].to(torch.float32)
            coords_norm = ((pts - ref_min[b].to(pts.dtype).unsqueeze(1)) / ref_span[b].to(pts.dtype).unsqueeze(1)).clamp(0.0, 1.0)
            w_sum = weights.sum().clamp_min(1e-6)
            w_mean = weights.mean()
            w_std = weights.std(unbiased=False)
            mean_xyz = (coords_norm * weights.unsqueeze(0)).sum(dim=1) / w_sum
            centered = coords_norm - mean_xyz.unsqueeze(1)
            std_xyz = torch.sqrt((centered.pow(2) * weights.unsqueeze(0)).sum(dim=1) / w_sum + 1e-8)
            bbox = (coords_norm.amax(dim=1) - coords_norm.amin(dim=1)).clamp_min(1e-6)
            global_feat = [
                torch.log1p(gen_xyz.new_tensor(float(N), dtype=torch.float32)),
                torch.log1p(w_sum),
                w_mean,
                w_std,
                mean_xyz[0], mean_xyz[1], mean_xyz[2],
                std_xyz[0], std_xyz[1], std_xyz[2],
                torch.log1p(bbox.prod()),
            ]
            level_feat = [
                self._soft_level_stats(coords_norm, weights, level, args)
                for level in self.surrogate_levels
            ]
            span_max = float(ref_span[b].detach().max().cpu())
            span_mean = ref_span[b].detach().mean().clamp_min(1e-6)
            q_level = max(int(math.ceil(math.log2(max(span_max / max(effective_qs, 1e-9) + 1.0, 2.0)))), 1)
            q_level = min(q_level, max(int(getattr(args, "proxy_max_depth", 12)), 1))
            qstep_feat = self._soft_level_stats(coords_norm, weights, q_level, args)
            q_norm = (gen_xyz.new_tensor(float(effective_qs), dtype=torch.float32) / span_mean.to(device=gen_xyz.device, dtype=torch.float32)).clamp_min(1e-9)
            codec_feat = torch.stack(
                [
                    torch.log1p(q_norm),
                    torch.log1p(1.0 / q_norm),
                    gen_xyz.new_tensor(float(q_level) / max(float(getattr(args, "proxy_max_depth", 12)), 1.0), dtype=torch.float32),
                    gen_xyz.new_tensor(is_sparsepcgc, dtype=torch.float32),
                    gen_xyz.new_tensor(is_gpcc, dtype=torch.float32),
                    gen_xyz.new_tensor(is_draco, dtype=torch.float32),
                ]
            ).to(device=gen_xyz.device, dtype=torch.float32)
            features.append(torch.cat([torch.stack(global_feat), *level_feat, qstep_feat, codec_feat], dim=0))

        return torch.stack(features, dim=0).to(device=gen_xyz.device, dtype=torch.float32)

    @staticmethod
    def _surrogate_effective_qs(args, codec_key):
        if str(codec_key).strip().lower() == "sparsepcgc":
            return max(
                float(getattr(args, "sparsepcgc_effective_qs", 0.0))
                or float(getattr(args, "sparsepcgc_voxel_size", 1.0)) * float(getattr(args, "sparsepcgc_pos_quantscale", 1)),
                1e-9,
            )
        if str(codec_key).strip().lower() == "gpcc":
            return max(float(getattr(args, "gpcc_effective_qs", getattr(args, "qs", 1.0))), 1e-9)
        if str(codec_key).strip().lower() == "draco":
            return max(float(getattr(args, "draco_effective_qs", getattr(args, "qs", 1.0))), 1e-9)
        return max(float(getattr(args, "qs", 1.0)), 1e-9)

    def _soft_aux_percent_from_features(self, x_soft, x_ref):
        level_dim = 5 * len(self.surrogate_levels)
        if x_soft.shape[1] < 11 + level_dim or x_ref.shape[1] < 11 + level_dim:
            zero = x_soft.new_zeros(())
            return zero, zero
        gen_levels = x_soft[:, 11:11 + level_dim].reshape(x_soft.shape[0], len(self.surrogate_levels), 5)
        ref_levels = x_ref[:, 11:11 + level_dim].reshape(x_ref.shape[0], len(self.surrogate_levels), 5)
        gen_node = gen_levels[:, :, 0].mean()
        gen_single = gen_levels[:, :, 1].mean()
        ref_node = ref_levels[:, :, 0].detach().mean().clamp_min(1e-6)
        ref_single = ref_levels[:, :, 1].detach().mean().clamp_min(1e-6)
        node_percent = 100.0 * (gen_node - ref_node) / ref_node.abs().clamp_min(1e-6)
        single_percent = 100.0 * (gen_single - ref_single) / ref_single.abs().clamp_min(1e-6)
        return node_percent, single_percent

    @staticmethod
    def _as_proxy_scalar(value, reference):
        if not torch.is_tensor(value):
            return reference.new_zeros(())
        value = value.to(device=reference.device, dtype=reference.dtype)
        return value.reshape(()) if value.numel() == 1 else value.mean()

    @staticmethod
    def _proxy_debug_scalar(value):
        if not torch.is_tensor(value):
            return None
        try:
            return float(value.detach().float().mean().cpu())
        except Exception:
            return None

    @staticmethod
    def _proxy_debug_requires_grad(value):
        return bool(torch.is_tensor(value) and value.requires_grad)

    def _actuator_soft_rate_proxy(self, args, reference):
        terms = getattr(args, "_last_actuator_soft_terms", {}) or {}
        if not isinstance(terms, dict):
            return reference.new_zeros(())

        add_proxy = (
            self._as_proxy_scalar(terms.get("add_ratio"), reference)
            + self._as_proxy_scalar(terms.get("learned_add_ratio"), reference)
            + self._as_proxy_scalar(terms.get("add_prob_mean"), reference)
            + 0.1 * self._as_proxy_scalar(terms.get("add_direction_ce"), reference)
        )
        prune_proxy = (
            2.0 * self._as_proxy_scalar(terms.get("drop_prob"), reference)
            + 2.0 * self._as_proxy_scalar(terms.get("drop_prob_proxy"), reference)
            + 2.0 * self._as_proxy_scalar(terms.get("learned_drop_prob"), reference)
            + self._as_proxy_scalar(terms.get("learned_drop_ratio"), reference)
            + 5.0 * self._as_proxy_scalar(terms.get("prune_soft_rate"), reference)
            + 2.0 * self._as_proxy_scalar(terms.get("prune_soft_node"), reference)
            + 2.0 * self._as_proxy_scalar(terms.get("prune_soft_single"), reference)
            + 3.0 * self._as_proxy_scalar(terms.get("prune_soft_bit"), reference)
        )
        move_proxy = (
            self._as_proxy_scalar(terms.get("move_score_mean"), reference)
            + self._as_proxy_scalar(terms.get("learned_move_ratio"), reference)
            + 0.1 * self._as_proxy_scalar(terms.get("move_direction_ce"), reference)
        )
        proxy = (
            float(getattr(args, "compression_soft_rate_add_weight", 2.0)) * add_proxy
            + float(getattr(args, "compression_soft_rate_prune_weight", 10.0)) * prune_proxy
            + float(getattr(args, "compression_soft_rate_move_weight", 0.5)) * move_proxy
        )
        return torch.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0)

    def _set_surrogate_trainable(self, trainable):
        for param in self.compression_surrogate.parameters():
            param.requires_grad_(trainable)

    @staticmethod
    def _can_update_compression_surrogate(args):
        if str(getattr(args, "trainORtest", "train")).strip().lower() != "train":
            return False
        if not torch.is_grad_enabled():
            return False
        inference_enabled = getattr(torch, "is_inference_mode_enabled", None)
        if callable(inference_enabled) and inference_enabled():
            return False
        return True

    @staticmethod
    def _surrogate_update_allowed_by_schedule(args):
        if bool(getattr(args, "_surrogate_pretrain_active", False)):
            return True
        if bool(getattr(args, "_surrogate_auto_frozen", False)):
            return False
        if not bool(getattr(args, "surrogate_update_during_training", True)):
            return False
        interval = max(int(getattr(args, "surrogate_update_interval", 1)), 1)
        step = int(getattr(args, "_global_train_step", 0))
        return (step % interval) == 0

    def _update_surrogate_auto_freeze_state(self, args, *, abs_error, train_loss, teacher_is_actual):
        if bool(getattr(args, "_surrogate_pretrain_active", False)) or not bool(
            getattr(args, "surrogate_auto_freeze", True)
        ):
            setattr(args, "_surrogate_auto_frozen", False)
            return {"frozen": False, "streak": 0, "event": "disabled"}
        frozen = bool(getattr(args, "_surrogate_auto_frozen", False))
        streak = int(getattr(args, "_surrogate_auto_freeze_streak", 0))
        if not teacher_is_actual:
            return {"frozen": frozen, "streak": streak, "event": "no_actual_teacher"}
        abs_error = float(abs_error)
        train_loss = float(train_loss)
        if not (math.isfinite(abs_error) and math.isfinite(train_loss)):
            setattr(args, "_surrogate_auto_frozen", False)
            setattr(args, "_surrogate_auto_freeze_streak", 0)
            return {"frozen": False, "streak": 0, "event": "non_finite_resume"}

        freeze_ok = (
            abs_error <= float(getattr(args, "surrogate_freeze_abs_error", 1.0))
            and train_loss <= float(getattr(args, "surrogate_freeze_train_loss", 1.0))
        )
        if frozen:
            resume = (
                abs_error >= float(getattr(args, "surrogate_resume_abs_error", 2.0))
                or train_loss >= float(getattr(args, "surrogate_resume_train_loss", 2.0))
            )
            if resume:
                frozen = False
                streak = 0
                event = "resume"
            else:
                event = "frozen"
        else:
            streak = streak + 1 if freeze_ok else 0
            if streak >= max(int(getattr(args, "surrogate_freeze_patience", 8)), 1):
                frozen = True
                event = "freeze"
            else:
                event = "warmup"
        setattr(args, "_surrogate_auto_frozen", bool(frozen))
        setattr(args, "_surrogate_auto_freeze_streak", int(streak))
        return {"frozen": bool(frozen), "streak": int(streak), "event": event}

    @staticmethod
    def _compression_main_grad_scale(args, *, actual_bit_percent, abs_error, train_loss=0.0):
        if not bool(getattr(args, "compression_good_step_boost", True)):
            return 1.0, "disabled"
        abs_error = float(abs_error)
        if not math.isfinite(abs_error) or abs_error > float(getattr(args, "compression_boost_max_abs_error", 1.0)):
            return 1.0, "surrogate_error_high"
        actual_bit_percent = float(actual_bit_percent)
        if not math.isfinite(actual_bit_percent):
            return 1.0, "actual_non_finite"
        frozen = bool(getattr(args, "_surrogate_auto_frozen", False))
        if bool(getattr(args, "compression_boost_requires_surrogate_frozen", True)) and not frozen:
            train_loss = float(train_loss)
            if (
                actual_bit_percent < 0.0
                and math.isfinite(train_loss)
                and train_loss <= float(getattr(args, "compression_good_step_prefreeze_max_train_loss", 4.0))
            ):
                return float(getattr(args, "compression_good_step_prefreeze_scale", 1.15)), "good_actual_delta_prefreeze"
            return 1.0, "surrogate_not_frozen"
        if actual_bit_percent < 0.0:
            return float(getattr(args, "compression_good_step_boost_scale", 1.5)), "good_actual_delta"
        if actual_bit_percent > 0.0:
            return float(getattr(args, "compression_bad_step_penalty_scale", 1.25)), "bad_actual_delta"
        return 1.0, "neutral_actual_delta"

    def _surrogate_target_cache_key(self, args, cache_key):
        backend = self._surrogate_backend_label(args)
        return f"{backend}|{cache_key or ''}"

    def _surrogate_target_is_current_step(self, args, entry):
        if entry is None:
            return False
        current_step = int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0)))
        return int(entry.get("global_step", -1)) == current_step

    def _get_cached_surrogate_target(self, args, cache_key):
        entry = None
        key = self._surrogate_target_cache_key(args, cache_key)
        backend = self._surrogate_backend_label(args)
        cache = getattr(self, "surrogate_target_cache", None)
        if cache is not None and key in cache:
            cache.move_to_end(key)
            cached_entry = dict(cache[key])
            entry = cached_entry
            entry["cache_hit"] = "exact" if self._surrogate_target_is_current_step(args, cached_entry) else "stale"
        if entry is not None and bool(getattr(args, "_surrogate_pretrain_active", False)):
            current_step = int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0)))
            entry_step = int(entry.get("global_step", current_step))
            entry_age = max(current_step - entry_step, 0)
            max_age = max(int(getattr(args, "surrogate_pretrain_max_target_age", 20)), 0)
            if entry_age > 0 and (
                not bool(getattr(args, "surrogate_pretrain_allow_stale_target", True))
                or entry_age > max_age
            ):
                entry = None
        if entry is None and not cache_key and bool(getattr(args, "compression_surrogate_reuse_last_target", False)):
            last_entry = getattr(self, "last_surrogate_target_entry", None)
            if (
                last_entry is not None
                and str(last_entry.get("backend_label", backend)) == backend
            ):
                entry = dict(last_entry)
                entry["cache_hit"] = "last" if self._surrogate_target_is_current_step(args, last_entry) else "last_stale"
        if (
            entry is None
            and bool(getattr(args, "_surrogate_pretrain_active", False))
            and bool(getattr(args, "surrogate_pretrain_allow_stale_target", True))
        ):
            last_entry = getattr(self, "last_surrogate_target_entry", None)
            if (
                last_entry is not None
                and str(last_entry.get("backend_label", backend)) == backend
            ):
                current_step = int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0)))
                last_step = int(last_entry.get("global_step", -10**12))
                max_age = max(int(getattr(args, "surrogate_pretrain_max_target_age", 20)), 0)
                if 0 <= current_step - last_step <= max_age:
                    entry = dict(last_entry)
                    entry["cache_hit"] = "pretrain_last_stale"
        return entry

    def _store_cached_surrogate_target(self, args, cache_key, entry):
        stored = dict(entry)
        stored.pop("cache_hit", None)
        self.last_surrogate_target_entry = dict(stored)
        key = self._surrogate_target_cache_key(args, cache_key)
        cache = getattr(self, "surrogate_target_cache", None)
        max_entries = max(int(getattr(self, "surrogate_target_cache_max_entries", 0)), 0)
        if cache is None or not cache_key or max_entries <= 0:
            return
        cache[key] = dict(stored)
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)

    def _store_surrogate_replay(self, args, x_soft, target):
        if bool(getattr(args, "_surrogate_pretrain_active", False)):
            max_entries = max(
                int(getattr(args, "surrogate_pretrain_replay_buffer_size", getattr(self, "surrogate_replay_max_entries", 0))),
                0,
            )
        else:
            max_entries = max(int(getattr(self, "surrogate_replay_max_entries", 0)), 0)
        if max_entries <= 0 or not self._all_finite(x_soft, target):
            return
        x_cpu = x_soft.detach().to(device="cpu", dtype=torch.float32)
        y_cpu = target.detach().to(device="cpu", dtype=torch.float32)
        if y_cpu.shape[0] == 1 and x_cpu.shape[0] > 1:
            y_cpu = y_cpu.expand(x_cpu.shape[0], -1).contiguous()
        meta = {
            "global_step": int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0))),
            "surrogate_step": int(getattr(self, "_surrogate_step", 0)),
            "stored_at": time.time(),
            "teacher_scope": str(getattr(args, "_current_teacher_scope", "")),
            "sample_name": str(getattr(args, "_current_sample_name", "")),
            "replay_is_full_cloud": str(getattr(args, "_current_teacher_scope", "")) == "full_cloud",
        }
        for idx in range(x_cpu.shape[0]):
            entry = (x_cpu[idx].clone(), y_cpu[min(idx, y_cpu.shape[0] - 1)].clone(), dict(meta))
            if len(self.surrogate_replay) < max_entries:
                self.surrogate_replay.append(entry)
            else:
                self.surrogate_replay[self.surrogate_replay_next] = entry
                self.surrogate_replay_next = (self.surrogate_replay_next + 1) % max_entries
            while len(self.surrogate_replay) > max_entries:
                self.surrogate_replay.pop(0)
                self.surrogate_replay_next = min(self.surrogate_replay_next, max(len(self.surrogate_replay) - 1, 0))

    def _sample_surrogate_replay(self, args, device):
        replay = getattr(self, "surrogate_replay", None)
        self._last_surrogate_replay_mean_age = 0.0
        self._last_surrogate_replay_full_cloud_count = 0
        if not replay:
            return None, None
        if bool(getattr(args, "_surrogate_pretrain_active", False)):
            min_size = max(int(getattr(args, "surrogate_pretrain_replay_min_size", 0)), 0)
            if len(replay) < min_size:
                return None, None
        batch = min(max(int(getattr(args, "compression_surrogate_replay_batch", 8)), 1), len(replay))
        start = int(getattr(self, "_surrogate_call_count", 0)) % len(replay)
        indices = [(start + offset) % len(replay) for offset in range(batch)]
        x = torch.stack([replay[idx][0] for idx in indices], dim=0).to(device=device, dtype=torch.float32, non_blocking=True) # CPU保存Replayを学習時だけGPUへ転送する
        y = torch.stack([replay[idx][1] for idx in indices], dim=0).to(device=device, dtype=torch.float32, non_blocking=True) # Replay教師targetを学習時だけGPUへ転送する
        current_step = int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0)))
        ages = [max(current_step - int(replay[idx][2].get("global_step", current_step)), 0) for idx in indices]
        self._last_surrogate_replay_mean_age = float(sum(ages) / float(max(len(ages), 1)))
        self._last_surrogate_replay_full_cloud_count = int(
            sum(1 for idx in indices if bool(replay[idx][2].get("replay_is_full_cloud", False)))
        )
        return x, y

    def _train_surrogate_replay(self, args, device):
        self._last_surrogate_replay_sample_count = 0
        self._last_surrogate_replay_steps = 0
        self._last_surrogate_replay_mean_age = 0.0
        self._last_surrogate_replay_full_cloud_count = 0
        if bool(getattr(args, "_surrogate_pretrain_active", False)) and not bool(
            getattr(args, "surrogate_pretrain_use_replay", True)
        ):
            return None
        replay_steps = max(int(getattr(args, "compression_surrogate_replay_steps", 0)), 0)
        if replay_steps <= 0:
            return None
        x_replay, y_replay = self._sample_surrogate_replay(args, device)
        if x_replay is None:
            return None
        self._last_surrogate_replay_sample_count = int(x_replay.shape[0])
        self._last_surrogate_replay_steps = int(replay_steps)
        replay_loss = self._train_compression_surrogate(args, x_replay, y_replay, train_steps=replay_steps) # Replay minibatchでSurrogateを追加更新する
        del x_replay, y_replay # Replay minibatchのGPU参照をStep内で解放する
        return replay_loss # Replay学習損失を呼び出し元へ返す

    @staticmethod
    def _surrogate_cache_stats_from_entry(entry):
        return {
            "bit": float(entry.get("gt_actual_bit", 0.0)),
            "bpp": float(entry.get("gt_bpp", 0.0)),
            "bpn": float(entry.get("gt_bpn", 0.0)),
            "single": float(entry.get("gt_single", 0.0)),
            "node": float(entry.get("gt_node", 0.0)),
            "octree_single": float(entry.get("gt_octree_single", entry.get("gt_single", 0.0))),
            "octree_node": float(entry.get("gt_octree_node", entry.get("gt_node", 0.0))),
            "octree_depth": int(entry.get("gt_octree_depth", 0)),
            "encode_time": float(entry.get("gt_actual_encode_time", 0.0)),
            "unique_coord_count": int(entry.get("gt_unique_coord_count", entry.get("gt_points", 0))),
            "point_count": int(entry.get("gt_points", 0)),
            "codec": str(entry.get("teacher_codec", "octattention")),
        }

    def _should_refresh_surrogate_teacher(self, args, entry, refresh_actual_gen=True):
        if isinstance(refresh_actual_gen, str) and refresh_actual_gen.strip().lower() == "always":
            return True
        if not bool(refresh_actual_gen):
            return False
        interval = max(int(getattr(args, "compression_surrogate_refresh_interval", 0)), 0)
        step = int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0)))
        if entry is None:
            # 新しいsampleでも毎回actual codecを呼ばず、global step間隔で教師を更新する。
            # これにより外部codecの重複実行を避け、GPU/CPU時間がstepごとに膨らむのを防ぐ。
            return step == 0 or (interval > 0 and step % interval == 0)
        if interval <= 0:
            return False
        last_step = int(entry.get("global_step", -10**12))
        return (step - last_step) >= interval

    def _surrogate_refresh_policy_label(self, args):
        return (
            f"periodic_interval={int(getattr(args, 'compression_surrogate_refresh_interval', 0))},"
            f"warmup_steps={int(getattr(args, 'compression_surrogate_warmup_steps', 0))},"
            f"replay_steps={int(getattr(args, 'compression_surrogate_replay_steps', 0))},"
            f"replay_batch={int(getattr(args, 'compression_surrogate_replay_batch', 0))},"
            f"forward={getattr(args, 'compression_surrogate_forward_mode', 'surrogate')},"
            f"reuse_last_target={bool(getattr(args, 'compression_surrogate_reuse_last_target', True))},"
            f"pretrain_stale={bool(getattr(args, 'surrogate_pretrain_allow_stale_target', False))}"
        )

    def _train_compression_surrogate(self, args, x_soft, target, train_steps=None):
        if train_steps is None:
            train_steps = max(int(getattr(args, "compression_surrogate_train_steps", 2)), 0)
        else:
            train_steps = max(int(train_steps), 0)
        if train_steps <= 0:
            return x_soft.new_zeros(())
        if not self._can_update_compression_surrogate(args):
            return x_soft.new_zeros(())
        if not self._surrogate_update_allowed_by_schedule(args):
            return x_soft.new_zeros(())
        if not self._all_finite(x_soft, target):
            self._log_surrogate_event("skipped train step because x_soft/target was non-finite.")
            if not self._surrogate_state_is_finite():
                self._reset_compression_surrogate("non-finite state while skipping train")
            return x_soft.new_zeros(())

        self._ensure_surrogate_device(x_soft.device)
        if not self._surrogate_state_is_finite():
            self._reset_compression_surrogate("non-finite params before train")
        self.compression_surrogate.train()
        self._set_surrogate_trainable(True)
        x_det = x_soft.detach()
        y_det = target.detach().expand(x_det.shape[0], -1)
        last_loss = x_soft.new_zeros(())
        weight = float(getattr(args, "compression_surrogate_bit_weight", 1.0))

        with self._compression_autocast_ctx(x_soft.device):
            for _ in range(train_steps):
                self.surrogate_optimizer.zero_grad(set_to_none=True)
                pred = self.compression_surrogate(x_det)
                if not self._all_finite(pred):
                    self.surrogate_optimizer.zero_grad(set_to_none=True)
                    self._reset_compression_surrogate("non-finite prediction during train")
                    return x_soft.new_zeros(())
                loss = float(weight) * F.smooth_l1_loss(pred, y_det, reduction="mean")
                if not torch.isfinite(loss):
                    self.surrogate_optimizer.zero_grad(set_to_none=True)
                    self._reset_compression_surrogate("non-finite loss during train")
                    return x_soft.new_zeros(())
                loss.backward()
                grad_finite = True
                for param in self.compression_surrogate.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        grad_finite = False
                        break
                if not grad_finite:
                    self.surrogate_optimizer.zero_grad(set_to_none=True)
                    self._reset_compression_surrogate("non-finite gradient during train")
                    return x_soft.new_zeros(())
                torch.nn.utils.clip_grad_norm_(
                    self.compression_surrogate.parameters(),
                    float(getattr(args, "compression_surrogate_grad_clip", 10.0)),
                )
                self.surrogate_optimizer.step()
                if not self._surrogate_state_is_finite():
                    self._reset_compression_surrogate("non-finite params after optimizer step")
                    return x_soft.new_zeros(())
                last_loss = loss.detach()

        self.compression_surrogate.eval()
        self._set_surrogate_trainable(False)
        self.surrogate_optimizer.zero_grad(set_to_none=True) # Surrogate更新後にgrad bufferをNone化してGPU保持を減らす
        cleanup_cuda_cache = bool(getattr(args, "compression_surrogate_empty_cache_after_update", True)) and x_soft.is_cuda # CUDA cache解放を行うか判定する
        if cleanup_cuda_cache and torch.cuda.is_available(): # CUDAが使える場合だけreserved memoryを確認する
            threshold_mb = float(getattr(args, "compression_surrogate_empty_cache_threshold_mb", 12288.0)) # cache解放のreserved memory閾値を取得する
            reserved_mb = float(torch.cuda.memory_reserved(x_soft.device)) / (1024.0 * 1024.0) # 現在のreserved memoryをMB単位で測る
            cleanup_cuda_cache = bool(threshold_mb <= 0.0 or reserved_mb >= threshold_mb) # 閾値超過時だけempty_cacheを走らせる
        if cleanup_cuda_cache: # GPU cache解放が必要ならallocator cacheを返す
            torch.cuda.empty_cache() # 精度を変えずにnvidia-smi上の一時的な確保量を下げる
        self._surrogate_step += train_steps
        return last_loss

    def _surrogate_backend_label(self, args, codec_name=None):
        backend = self._compression_loss_backend(args)
        if backend in {"surrogate", "soft_surrogate"}:
            codec = codec_name or str(getattr(args, "compress", "OctAttention")).strip().lower()
            codec = codec.replace("_", "").replace("-", "")
            if codec == "sparsepcgc":
                return "sparsepcgc_surrogate"
            if codec == "gpcc":
                return "gpcc_surrogate"
            if codec == "draco":
                return "draco_surrogate"
            return "octattention_surrogate"
        return backend

    def _get_compression_loss_surrogate(
        self,
        args,
        gen_xyz,
        gt_xyz,
        final_w,
        cache_key=None,
        refresh_actual_gen=True,
        actual_gen_xyz=None,
        subtree_tree=None,
        full_octree_context=None,
        octree_input_mode="auto",
    ):
        timing_enabled = bool(getattr(args, "debug_timing", False) or getattr(args, "_surrogate_pretrain_timing_enabled", False))
        timing = {}

        def _sync_timing():
            if timing_enabled and torch.is_tensor(gen_xyz) and gen_xyz.is_cuda:
                torch.cuda.synchronize(gen_xyz.device)

        def _mark_timing(name, start_time):
            if not timing_enabled:
                return start_time
            _sync_timing()
            now = time.time()
            timing[name] = now - start_time
            return now

        if timing_enabled:
            _sync_timing()
            timing_cursor = time.time()
        else:
            timing_cursor = 0.0
        self._surrogate_call_count = int(getattr(self, "_surrogate_call_count", 0)) + 1
        requested_octree_mode = str(octree_input_mode or "auto").strip().lower()
        uses_subtree_tree = isinstance(subtree_tree, dict)
        if requested_octree_mode == "prebuilt_subtree_tree" and not uses_subtree_tree:
            raise ValueError("octree_input_mode=prebuilt_subtree_tree requires subtree_tree in surrogate compression loss.")
        prebuilt_codes = subtree_tree.get("occupancy_codes", None) if uses_subtree_tree else None
        if prebuilt_codes is not None:
            prebuilt_code_t = torch.as_tensor(prebuilt_codes, dtype=torch.long)
            prebuilt_node_count = float(prebuilt_code_t.numel())
            if prebuilt_code_t.numel() > 0:
                child_counts = (
                    (
                        prebuilt_code_t.reshape(-1, 1)
                        >> torch.arange(8, dtype=torch.long, device=prebuilt_code_t.device).view(1, -1)
                    )
                    & 1
                ).sum(dim=1)
                prebuilt_single_count = float((child_counts == 1).sum().item())
            else:
                prebuilt_single_count = 0.0
        else:
            prebuilt_node_count = 0.0
            prebuilt_single_count = 0.0
        compression_proxy_input_mode = "prebuilt_subtree_tree" if uses_subtree_tree else ("full_cloud" if requested_octree_mode == "full_cloud" else "local_recomputed")
        x_soft = self._build_soft_compression_features(args, gen_xyz, gt_xyz, final_w)
        timing_cursor = _mark_timing("feature_gen", timing_cursor)
        aux_node_weight = float(getattr(args, "compression_surrogate_aux_node_weight", 0.0))
        aux_single_weight = float(getattr(args, "compression_surrogate_aux_single_weight", 0.0))
        log_soft_aux = bool(getattr(args, "compression_surrogate_log_soft_aux", True))
        need_soft_aux = bool(log_soft_aux or aux_node_weight > 0.0 or aux_single_weight > 0.0)
        need_sparse_aux = bool(getattr(args, "sparsepcgc_aux_loss", True) and self._is_sparsepcgc_context(args))
        x_ref = None
        if need_soft_aux or need_sparse_aux:
            x_ref = self._build_soft_compression_features(args, gt_xyz, gt_xyz, None)
        if need_soft_aux:
            soft_node_percent, soft_single_percent = self._soft_aux_percent_from_features(x_soft, x_ref)
        else:
            soft_node_percent = x_soft.new_zeros(())
            soft_single_percent = x_soft.new_zeros(())
        timing_cursor = _mark_timing("feature_ref_aux", timing_cursor)
        sparse_terms = self._sparsepcgc_aux_feature_terms(
            args,
            gen_xyz,
            gt_xyz,
            final_w,
            x_gen=x_soft,
            x_ref=x_ref,
        )
        timing_cursor = _mark_timing("sparsepcgc_aux_proxy", timing_cursor)
        inputs_finite = self._all_finite(gen_xyz, gt_xyz, x_soft)
        target = None
        stats_gen = None
        target_entry = self._get_cached_surrogate_target(args, cache_key)
        target_cache_hit = str(target_entry.get("cache_hit", "miss")) if target_entry is not None else "miss"
        pretrain_mode = str(getattr(args, "_surrogate_pretrain_mode", getattr(args, "surrogate_pretrain_mode", ""))).strip().lower()
        pretrain_teacher_type = str(
            getattr(args, "_surrogate_pretrain_teacher_type", getattr(args, "surrogate_pretrain_subtree_teacher_type", ""))
        ).strip().lower()
        pretrain_local_proxy_teacher = bool(
            getattr(args, "_surrogate_pretrain_active", False)
            and pretrain_mode in {"subtree", "hybrid"}
            and pretrain_teacher_type == "local_proxy"
            and not bool(refresh_actual_gen)
        )
        if pretrain_local_proxy_teacher:
            target_entry = None
            target_cache_hit = "local_proxy"
        actual_bit_percent = 0.0
        actual_bpp_percent = 0.0
        actual_single_percent = 0.0
        actual_node_percent = 0.0
        target_percent_value = 0.0
        target_raw_percent_value = 0.0
        target_train_percent_value = 0.0
        target_clamped_percent_value = 0.0
        target_log_ratio_value = float("nan")
        target_clip_min_value = float("nan")
        target_clip_max_value = float("nan")
        target_mode_value = "none"
        target_was_clamped = False
        target_scale = "none"
        target_teacher_source = "none"
        local_proxy_replay_stored = False
        gen_points = int(gen_xyz.shape[-1])
        gen_actual_bit = float("nan")
        gen_total_bit_with_edit_record = float("nan")
        actual_edit_record_bits = 0.0
        actual_raw_percent_value = 0.0
        policy_actual_noop_guard_used = False
        policy_actual_noop_guard_margin = max(
            float(getattr(args, "sparsepcgc_policy_actual_noop_guard_margin", 0.0)),
            0.0,
        )
        policy_actual_noop_guard_percent = float("nan")
        policy_actual_noop_guard_raw_percent = float("nan")
        policy_actual_noop_guard_raw_bit = float("nan")
        policy_actual_noop_guard_raw_total_bit = float("nan")
        policy_actual_noop_guard_raw_edit_record_bits = float("nan")
        cached_gt = None
        local_proxy_rate_target_value = float("nan")
        local_proxy_aux_target_value = float("nan")
        local_proxy_rate_error = ""
        if not self._surrogate_state_is_finite():
            self._reset_compression_surrogate("non-finite params before inference")

        teacher_refreshed = bool(inputs_finite and self._should_refresh_surrogate_teacher(args, target_entry, refresh_actual_gen))
        local_proxy_teacher = bool(
            pretrain_local_proxy_teacher
            or (
                inputs_finite
                and need_sparse_aux
                and not teacher_refreshed
                and target_entry is None
                and bool(getattr(args, "sparsepcgc_surrogate_local_proxy_on_target_miss", True))
            )
        )
        if local_proxy_teacher and target_cache_hit == "miss":
            target_cache_hit = "local_proxy"
        if teacher_refreshed:
            actual_t0 = time.time() if timing_enabled else 0.0
            cached_gt = self._get_cached_actual_gt(cache_key)
            if cached_gt is None:
                cached_gt = self._encode_actual_batch(args, gt_xyz)
                self._store_cached_actual_gt(cache_key, cached_gt)
            # actual codec教師は評価指標なので、train用ノイズなしの編集点群で測る。
            actual_xyz = gen_xyz if actual_gen_xyz is None else actual_gen_xyz
            cached_oracle_stats = (
                full_octree_context.get("actual_oracle_cached_edited_actual_stats", None)
                if isinstance(full_octree_context, dict)
                and str(requested_octree_mode).strip().lower() == "full_cloud"
                and str(full_octree_context.get("actual_oracle_override_scope", "")) == "full_cloud"
                else None
            )
            expected_actual_points = int(actual_xyz.shape[-1]) if torch.is_tensor(actual_xyz) else -1
            actual_gen_cache_hit = bool(
                isinstance(cached_oracle_stats, dict)
                and int(cached_oracle_stats.get("point_count", -2)) == expected_actual_points
                and float(cached_oracle_stats.get("bit", 0.0)) > 0.0
            )
            if actual_gen_cache_hit:
                stats_gen = dict(cached_oracle_stats)
            else:
                stats_gen = self._encode_actual_batch(args, actual_xyz, final_w=final_w)
            if timing_enabled:
                timing["actual_encode"] = time.time() - actual_t0
                timing_cursor = time.time()
            teacher_codec = str(stats_gen.get("codec", cached_gt.get("codec", "octattention"))).strip().lower()
            actual_edit_record_bits = 0.0
            voxel_state = getattr(args, "_last_actuator_voxel_state", None)
            if (
                str(teacher_codec).strip().lower() == "sparsepcgc"
                and bool(getattr(args, "sparsepcgc_edit_record_bits_enabled", True))
                and isinstance(voxel_state, dict)
            ):
                try:
                    actual_edit_record_bits = float(voxel_state.get("estimated_edit_record_bits", 0.0) or 0.0)
                except Exception:
                    actual_edit_record_bits = 0.0
                if not math.isfinite(actual_edit_record_bits):
                    actual_edit_record_bits = 0.0
                actual_edit_record_bits = max(actual_edit_record_bits, 0.0)
            raw_gen_bit = float(stats_gen["bit"])
            gen_total_bit_with_edit_record = raw_gen_bit + actual_edit_record_bits
            stats_for_target = dict(stats_gen)
            stats_for_target["bit"] = float(gen_total_bit_with_edit_record)
            actual_raw_percent_value = self._relative_percent(raw_gen_bit, float(cached_gt["bit"]))
            actual_bit_percent = self._relative_percent(gen_total_bit_with_edit_record, float(cached_gt["bit"]))
            policy_actual_noop_guard_used = False
            policy_actual_noop_guard_margin = max(
                float(getattr(args, "sparsepcgc_policy_actual_noop_guard_margin", 0.0)),
                0.0,
            )
            policy_actual_noop_guard_percent = float(actual_bit_percent)
            policy_actual_noop_guard_raw_percent = float(actual_bit_percent)
            policy_actual_noop_guard_raw_bit = float(raw_gen_bit)
            policy_actual_noop_guard_raw_total_bit = float(gen_total_bit_with_edit_record)
            policy_actual_noop_guard_raw_edit_record_bits = float(actual_edit_record_bits)
            if (
                str(teacher_codec).strip().lower() == "sparsepcgc"
                and bool(getattr(args, "sparsepcgc_policy_actual_noop_guard", True))
                and not bool(getattr(args, "direct_network_prune", False))
                and float(actual_bit_percent) > float(policy_actual_noop_guard_margin)
            ):
                policy_actual_noop_guard_used = True
                policy_actual_noop_guard_percent = 0.0
            actual_bpp_percent = self._relative_percent(float(stats_gen["bpp"]), float(cached_gt["bpp"]))
            actual_single_percent = self._relative_percent(
                float(stats_gen["single"]),
                float(cached_gt["single"]),
                ref_min=1.0,
            )
            actual_node_percent = self._relative_percent(
                float(stats_gen["node"]),
                float(cached_gt["node"]),
                ref_min=1.0,
            )
            target_debug = self._surrogate_target_from_actual(args, stats_for_target, cached_gt, gen_xyz.device)
            target = target_debug["target"]
            target_raw_percent_value = float(actual_bit_percent)
            target_train_percent_value = float(target_debug["train_value"])
            target_clamped_percent_value = float(target_debug["clamped_percent"])
            target_log_ratio_value = float(target_debug["log_ratio"])
            target_clip_min_value = float(target_debug["clip_min"])
            target_clip_max_value = float(target_debug["clip_max"])
            target_mode_value = str(target_debug["target_mode"])
            target_was_clamped = bool(target_debug["target_clamped"])
            target_scale = str(target_debug["target_mode"])
            target_teacher_source = "fresh_actual"
            if actual_gen_cache_hit:
                target_teacher_source = "oracle_cached_actual"
            warmup_steps = max(
                int(getattr(args, "compression_surrogate_warmup_steps", getattr(args, "compression_surrogate_train_steps", 2))),
                0,
            )
            L_sur = self._train_compression_surrogate(args, x_soft, target, train_steps=warmup_steps)
            extra_good_steps = max(int(getattr(args, "compression_good_step_extra_surrogate_steps", 0)), 0)
            if (
                extra_good_steps > 0
                and float(actual_bit_percent) < 0.0
                and not bool(getattr(args, "_surrogate_pretrain_active", False))
            ):
                L_sur_extra = self._train_compression_surrogate(args, x_soft, target, train_steps=extra_good_steps)
                if self._all_finite(L_sur_extra):
                    L_sur = L_sur_extra
            timing_cursor = _mark_timing("surrogate_fit", timing_cursor)
            self._store_surrogate_replay(args, x_soft, target)
            actual_value_source = "fresh_teacher"
            target_percent_value = float(actual_bit_percent)
            gen_points = int(stats_gen["point_count"])
            gen_actual_bit = float(stats_gen["bit"])
            current_step = int(getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0)))
            self._store_cached_surrogate_target(
                args,
                cache_key,
                {
                    "backend_label": self._surrogate_backend_label(args, teacher_codec),
                    "teacher_codec": teacher_codec,
                    "actual_bit_percent": float(actual_bit_percent),
                    "actual_bit_percent_raw": float(actual_bit_percent),
                    "actual_raw_percent": float(actual_raw_percent_value),
                    "actual_edit_record_bits": float(actual_edit_record_bits),
                    "policy_actual_noop_guard_used": bool(policy_actual_noop_guard_used),
                    "policy_actual_noop_guard_margin": float(policy_actual_noop_guard_margin),
                    "policy_actual_noop_guard_percent": float(policy_actual_noop_guard_percent),
                    "policy_actual_noop_guard_raw_percent": float(policy_actual_noop_guard_raw_percent),
                    "policy_actual_noop_guard_raw_bit": float(policy_actual_noop_guard_raw_bit),
                    "policy_actual_noop_guard_raw_total_bit": float(policy_actual_noop_guard_raw_total_bit),
                    "policy_actual_noop_guard_raw_edit_record_bits": float(policy_actual_noop_guard_raw_edit_record_bits),
                    "actual_bpp_percent": float(actual_bpp_percent),
                    "actual_single_percent": float(actual_single_percent),
                    "actual_node_percent": float(actual_node_percent),
                    "target_bit_percent": float(target_percent_value),
                    "target_train_value": float(target_train_percent_value),
                    "target_clamped_percent": float(target_clamped_percent_value),
                    "target_log_ratio": float(target_log_ratio_value),
                    "target_mode": str(target_mode_value),
                    "target_clamped": bool(target_was_clamped),
                    "gt_points": int(cached_gt["point_count"]),
                    "gen_points": int(gen_points),
                    "gt_actual_bit": float(cached_gt["bit"]),
                    "gen_actual_bit": float(gen_actual_bit),
                    "gen_total_bit_with_edit_record": float(gen_total_bit_with_edit_record),
                    "gt_actual_encode_time": float(cached_gt.get("encode_time", 0.0)),
                    "gen_actual_encode_time": float(stats_gen.get("encode_time", 0.0)),
                    "gt_unique_coord_count": int(cached_gt.get("unique_coord_count", cached_gt.get("point_count", 0))),
                    "gen_unique_coord_count": int(stats_gen.get("unique_coord_count", stats_gen.get("point_count", gen_points))),
                    "gt_bpp": float(cached_gt["bpp"]),
                    "gen_bpp": float(stats_gen["bpp"]),
                    "gt_bpn": float(cached_gt["bpn"]),
                    "gen_bpn": float(stats_gen["bpn"]),
                    "gt_single": float(cached_gt["single"]),
                    "gen_single": float(stats_gen["single"]),
                    "gt_node": float(cached_gt["node"]),
                    "gen_node": float(stats_gen["node"]),
                    "gt_octree_single": float(cached_gt.get("octree_single", cached_gt["single"])),
                    "gen_octree_single": float(stats_gen.get("octree_single", stats_gen["single"])),
                    "gt_octree_node": float(cached_gt.get("octree_node", cached_gt["node"])),
                    "gen_octree_node": float(stats_gen.get("octree_node", stats_gen["node"])),
                    "gt_octree_depth": int(cached_gt.get("octree_depth", 0)),
                    "gen_octree_depth": int(stats_gen.get("octree_depth", 0)),
                    **self._actual_occupancy_debug_from_stats(cached_gt, stats_gen),
                    "global_step": int(current_step),
                    "surrogate_step": int(getattr(self, "_surrogate_step", 0)),
                },
            )
        elif local_proxy_teacher and inputs_finite:
            teacher_codec = self._surrogate_backend_label(args).replace("_surrogate", "")
            cached_gt = self._get_cached_actual_gt(cache_key)
            if cached_gt is None:
                try:
                    cached_gt = self._encode_actual_batch(args, gt_xyz)
                    self._store_cached_actual_gt(cache_key, cached_gt)
                except Exception:
                    cached_gt = None
            if cached_gt is None:
                cached_gt = {
                    "bit": 0.0,
                    "bpp": 0.0,
                    "bpn": 0.0,
                    "single": 0.0,
                    "node": 0.0,
                    "octree_single": 0.0,
                    "octree_node": 0.0,
                    "octree_depth": 0,
                    "point_count": int(gt_xyz.shape[-1]),
                    "codec": teacher_codec,
                }
            local_aux_target = (
                aux_node_weight * soft_node_percent.to(device=gen_xyz.device, dtype=torch.float32)
                + aux_single_weight * soft_single_percent.to(device=gen_xyz.device, dtype=torch.float32)
                + float(getattr(args, "com_sparsepcgc", 0.0))
                * sparse_terms["loss"].to(device=gen_xyz.device, dtype=torch.float32)
            ).detach()
            local_aux_target = local_aux_target.reshape(-1).mean().reshape(())
            local_proxy_aux_target_value = self._scalar(local_aux_target)
            local_rate_target = None
            try:
                _, proxy_loss_bit, _, _, _, _ = self._get_compression_loss_proxy(
                    args,
                    gen_xyz=gen_xyz,
                    gt_xyz=gt_xyz,
                    final_w=final_w,
                    cache_key=cache_key,
                    run_grad_probe=False,
                    actual_gen_xyz=actual_gen_xyz,
                    subtree_tree=subtree_tree,
                    full_octree_context=full_octree_context,
                    octree_input_mode=octree_input_mode,
                )
                if torch.is_tensor(proxy_loss_bit):
                    local_rate_target = proxy_loss_bit.to(device=gen_xyz.device, dtype=torch.float32).reshape(-1).mean().detach()
                    local_proxy_rate_target_value = self._scalar(local_rate_target)
            except Exception as exc:
                local_proxy_rate_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                self._log_surrogate_event(f"local proxy rate target failed; using aux-only target. error={local_proxy_rate_error}")
            if local_rate_target is not None and self._all_finite(local_rate_target):
                local_proxy_target = (
                    float(getattr(args, "sparsepcgc_surrogate_local_proxy_rate_weight", 1.0)) * local_rate_target
                    + float(getattr(args, "sparsepcgc_surrogate_local_proxy_aux_weight", 0.25)) * local_aux_target
                )
            else:
                local_proxy_target = local_aux_target
            local_proxy_target = local_proxy_target.reshape(1, 1)
            if not self._all_finite(local_proxy_target):
                local_proxy_target = gen_xyz.new_zeros((1, 1), dtype=torch.float32)
            target = local_proxy_target
            target_percent_value = self._scalar(local_proxy_target.reshape(()))
            target_raw_percent_value = float(target_percent_value)
            target_debug = self._prepare_surrogate_target(args, target_raw_percent_value, gen_xyz.device)
            target = target_debug["target"]
            target_train_percent_value = float(target_debug["train_value"])
            target_clamped_percent_value = float(target_debug["clamped_percent"])
            target_log_ratio_value = float(target_debug["log_ratio"])
            target_clip_min_value = float(target_debug["clip_min"])
            target_clip_max_value = float(target_debug["clip_max"])
            target_mode_value = str(target_debug["target_mode"])
            target_was_clamped = bool(target_debug["target_clamped"])
            target_scale = f"local_proxy_aux_{target_mode_value}"
            target_teacher_source = "local_proxy"
            L_sur = self._train_compression_surrogate(args, x_soft, target)
            timing_cursor = _mark_timing("surrogate_fit", timing_cursor)
            local_proxy_replay_stored = bool(getattr(args, "surrogate_pretrain_store_local_proxy_replay", False))
            if local_proxy_replay_stored:
                self._store_surrogate_replay(args, x_soft, target)
            actual_value_source = "local_proxy"
            gen_points = int(gen_xyz.shape[-1])
        elif target_entry is not None:
            teacher_codec = str(target_entry.get("teacher_codec", "octattention")).strip().lower()
            cached_gt = self._surrogate_cache_stats_from_entry(target_entry)
            actual_bit_percent = float(target_entry.get("actual_bit_percent", 0.0))
            actual_bpp_percent = float(target_entry.get("actual_bpp_percent", 0.0))
            actual_single_percent = float(target_entry.get("actual_single_percent", 0.0))
            actual_node_percent = float(target_entry.get("actual_node_percent", 0.0))
            target_percent_value = float(target_entry.get("target_bit_percent", actual_bit_percent))
            gen_points = int(target_entry.get("gen_points", gen_points))
            gen_actual_bit = float(target_entry.get("gen_actual_bit", float("nan")))
            gen_total_bit_with_edit_record = float(
                target_entry.get("gen_total_bit_with_edit_record", gen_actual_bit)
            )
            actual_edit_record_bits = float(target_entry.get("actual_edit_record_bits", 0.0) or 0.0)
            actual_raw_percent_value = float(target_entry.get("actual_raw_percent", actual_bit_percent))
            policy_actual_noop_guard_used = bool(target_entry.get("policy_actual_noop_guard_used", False))
            policy_actual_noop_guard_margin = float(
                target_entry.get("policy_actual_noop_guard_margin", policy_actual_noop_guard_margin)
            )
            # policy_actual_noop_guard_percent = float(
            #     target_entry.get("policy_actual_noop_guard_percent", policy_actual_noop_guard_percent)
            # )
            policy_actual_noop_guard_raw_percent = float(
                target_entry.get("policy_actual_noop_guard_raw_percent", policy_actual_noop_guard_raw_percent)
            )
            policy_actual_noop_guard_raw_bit = float(
                target_entry.get("policy_actual_noop_guard_raw_bit", policy_actual_noop_guard_raw_bit)
            )
            policy_actual_noop_guard_raw_total_bit = float(
                target_entry.get("policy_actual_noop_guard_raw_total_bit", policy_actual_noop_guard_raw_total_bit)
            )
            policy_actual_noop_guard_raw_edit_record_bits = float(
                target_entry.get(
                    "policy_actual_noop_guard_raw_edit_record_bits",
                    policy_actual_noop_guard_raw_edit_record_bits,
                )
            )
            stale_hit = "stale" in str(target_cache_hit).lower()
            actual_value_source = "stale_target" if stale_hit else "target_cache"
            if bool(getattr(args, "_surrogate_pretrain_active", False)) and not bool(
                getattr(args, "surrogate_pretrain_skip_on_target_miss", False)
            ):
                target_raw_percent_value = float(target_percent_value)
                target_debug = self._prepare_surrogate_target(
                    args,
                    target_raw_percent_value,
                    gen_xyz.device,
                    before_bits=float(target_entry.get("gt_actual_bit", float("nan"))),
                    after_bits=float(target_entry.get("gen_actual_bit", float("nan"))),
                )
                target = target_debug["target"]
                target_train_percent_value = float(target_debug["train_value"])
                target_clamped_percent_value = float(target_debug["clamped_percent"])
                target_log_ratio_value = float(target_debug["log_ratio"])
                target_clip_min_value = float(target_debug["clip_min"])
                target_clip_max_value = float(target_debug["clip_max"])
                target_mode_value = str(target_debug["target_mode"])
                target_was_clamped = bool(target_debug["target_clamped"])
                target_scale = f"actual_bit_percent_cache_{target_mode_value}"
                target_teacher_source = actual_value_source
                L_sur = self._train_compression_surrogate(args, x_soft, target)
            else:
                L_sur = x_soft.new_zeros(())
                target_raw_percent_value = float(target_percent_value)
                target_train_percent_value = float(target_entry.get("target_train_value", target_percent_value))
                target_clamped_percent_value = float(target_entry.get("target_clamped_percent", target_train_percent_value))
                target_log_ratio_value = float(target_entry.get("target_log_ratio", float("nan")))
                target_clip_min_value = -resolve_surrogate_target_clip(args) if resolve_surrogate_target_clip(args) > 0.0 else float("nan")
                target_clip_max_value = resolve_surrogate_target_clip(args) if resolve_surrogate_target_clip(args) > 0.0 else float("nan")
                target_mode_value = str(target_entry.get("target_mode", "actual_bit_percent_cache_no_update"))
                target_was_clamped = bool(target_entry.get("target_clamped", abs(target_train_percent_value - target_raw_percent_value) > 1e-6))
                target_scale = "actual_bit_percent_cache_no_update"
                target_teacher_source = actual_value_source
            timing_cursor = _mark_timing("target_cache", timing_cursor)
        else:
            teacher_codec = self._surrogate_backend_label(args).replace("_surrogate", "")
            cached_gt = {
                "bit": 0.0,
                "bpp": 0.0,
                "bpn": 0.0,
                "single": 0.0,
                "node": 0.0,
                "point_count": int(gt_xyz.shape[-1]),
                "codec": teacher_codec,
            }
            if not inputs_finite:
                self._log_surrogate_event("skipped teacher refresh because generator features were non-finite.")
            L_sur = x_soft.new_zeros(())
            actual_value_source = "target_missing_skip" if bool(
                getattr(args, "surrogate_pretrain_skip_on_target_miss", False)
            ) else "missing"
            timing_cursor = _mark_timing("target_missing", timing_cursor)

        update_on_teacher_only = bool(getattr(args, "surrogate_update_on_teacher_refresh_only", False))
        replay_loss = None if (update_on_teacher_only and not teacher_refreshed) else self._train_surrogate_replay(args, gen_xyz.device)
        replay_sample_count = int(getattr(self, "_last_surrogate_replay_sample_count", 0))
        replay_steps = int(getattr(self, "_last_surrogate_replay_steps", 0))
        replay_age = float(getattr(self, "_last_surrogate_replay_mean_age", 0.0))
        replay_full_cloud_count = int(getattr(self, "_last_surrogate_replay_full_cloud_count", 0))
        if replay_loss is not None:
            L_sur = 0.5 * (L_sur + replay_loss) if (teacher_refreshed or actual_value_source == "local_proxy") else replay_loss
        timing_cursor = _mark_timing("surrogate_replay", timing_cursor)

        self._surrogate_target_count = int(getattr(self, "_surrogate_target_count", 0)) + 1
        self._surrogate_target_clamp_count = int(getattr(self, "_surrogate_target_clamp_count", 0)) + int(bool(target_was_clamped))
        target_clamp_rate = float(self._surrogate_target_clamp_count) / float(max(self._surrogate_target_count, 1))

        self._ensure_surrogate_device(gen_xyz.device)
        self.compression_surrogate.eval()
        self._set_surrogate_trainable(False)
        # ============================================================
        # Surrogate予測からNetworkへの勾配だけを切る
        # ============================================================
        detach_surrogate_from_network = bool(
            getattr(args, "detach_surrogate_from_network", True)
        )

        # x_soft自体はsoft圧縮proxyに使うため壊さない。
        # Surrogate予測に入れる入力だけdetachする。
        x_surrogate_pred = x_soft.detach() if detach_surrogate_from_network else x_soft
        x_pred = None
        if inputs_finite:
            # ============================================================
            # Surrogate予測入力をNetworkから切り離す
            # ============================================================
            x_pred = x_surrogate_pred
            if detach_surrogate_from_network:
                with torch.no_grad():
                    pred_raw = (
                        self.compression_surrogate.forward_raw(x_pred)
                        if hasattr(self.compression_surrogate, "forward_raw")
                        else self.compression_surrogate(x_pred)
                    )
                    pred = self.compression_surrogate(x_pred)
            else:
                pred_raw = (
                    self.compression_surrogate.forward_raw(x_pred)
                    if hasattr(self.compression_surrogate, "forward_raw")
                    else self.compression_surrogate(x_pred)
                )
                pred = self.compression_surrogate(x_pred)
            if not self._all_finite(pred):
                if detach_surrogate_from_network:
                    with torch.no_grad():
                        pred_raw = (
                            self.compression_surrogate.forward_raw(x_pred)
                            if hasattr(self.compression_surrogate, "forward_raw")
                            else self.compression_surrogate(x_pred)
                        )
                        pred = self.compression_surrogate(x_pred)
                else:
                    pred_raw = (
                        self.compression_surrogate.forward_raw(x_pred)
                        if hasattr(self.compression_surrogate, "forward_raw")
                        else self.compression_surrogate(x_pred)
                    )
                    pred = self.compression_surrogate(x_pred)
            if not self._all_finite(pred):
                self._log_surrogate_event("using zero prediction because inference stayed non-finite after reset.")
                pred_raw = x_soft.new_zeros((x_soft.shape[0], 1), dtype=torch.float32)
                pred = x_soft.new_zeros((x_soft.shape[0], 1), dtype=torch.float32)
        else:
            pred_raw = x_soft.new_zeros((x_soft.shape[0], 1), dtype=torch.float32)
            pred = x_soft.new_zeros((x_soft.shape[0], 1), dtype=torch.float32)
        timing_cursor = _mark_timing("surrogate_predict", timing_cursor)

        surrogate_bit_percent = pred.reshape(-1).mean() if pred.numel() > 0 else x_soft.new_zeros(())
        surrogate_raw_percent = pred_raw.reshape(-1).mean() if pred_raw.numel() > 0 else x_soft.new_zeros(())
        forward_teacher_raw_percent_value = float(actual_bit_percent)
        # Surrogateへ渡す教師値は、target生成時のclamp/transformと揃えて学習を安定化する。
        # raw actual percent は別途ログに残し、loss側だけを安全な値へ寄せる。
        forward_teacher_percent_value = float(actual_bit_percent)
        forward_teacher_source = str(actual_value_source)
        # ============================================================
        # Direct Network Prune:
        # no-op置換後・local proxy置換後ではなく、raw actual percentを
        # compression lossのforward教師値として使う。
        # ============================================================
        if bool(getattr(args, "direct_network_prune", False)) and bool(
            getattr(args, "direct_prune_use_raw_compression_loss", True)
        ):
            forward_teacher_percent_value = float(actual_bit_percent)
            forward_teacher_source = "direct_network_raw_actual"
        if not math.isfinite(forward_teacher_percent_value):
            forward_teacher_percent_value = 0.0
            forward_teacher_source = f"{forward_teacher_source}_nonfinite_zero"
        forward_teacher_clamped = bool(
            math.isfinite(forward_teacher_raw_percent_value)
            and abs(float(forward_teacher_percent_value) - forward_teacher_raw_percent_value) > 1e-6
        )

        actual_bit_percent_t = gen_xyz.new_tensor(float(actual_bit_percent), dtype=torch.float32)
        forward_teacher_percent_t = gen_xyz.new_tensor(float(forward_teacher_percent_value), dtype=torch.float32)
        pred_percent = surrogate_bit_percent.detach().reshape(())
        pred_raw_percent = surrogate_raw_percent.detach().reshape(())
        target_percent = pred_percent.new_tensor(float(target_percent_value)).detach().reshape(())
        surrogate_signed_bit_error = pred_percent - target_percent
        surrogate_abs_bit_error = surrogate_signed_bit_error.abs()
        train_target_t = pred_percent.new_tensor(float(target_train_percent_value)).reshape(())
        raw_actual_t = pred_percent.new_tensor(float(target_raw_percent_value)).reshape(())
        surrogate_loss_against_train_target = F.smooth_l1_loss(pred_percent.reshape(1), train_target_t.detach().reshape(1), reduction="mean")
        surrogate_loss_against_raw_actual = F.smooth_l1_loss(pred_percent.reshape(1), raw_actual_t.detach().reshape(1), reduction="mean")
        teacher_is_actual_for_control = bool(actual_value_source in {"fresh_teacher", "target_cache", "stale_target"})
        sparse_aux_actual_teacher_allowed = bool(
            (not teacher_is_actual_for_control)
            or bool(getattr(args, "sparsepcgc_aux_with_actual_teacher", False))
        )
        auto_freeze_debug = self._update_surrogate_auto_freeze_state(
            args,
            abs_error=self._scalar(surrogate_abs_bit_error),
            train_loss=self._scalar(L_sur),
            teacher_is_actual=teacher_is_actual_for_control,
        )
        main_grad_scale, main_grad_scale_reason = self._compression_main_grad_scale(
            args,
            actual_bit_percent=float(actual_bit_percent),
            abs_error=self._scalar(surrogate_abs_bit_error),
            train_loss=self._scalar(L_sur),
        )
        loss_single = soft_single_percent.to(device=gen_xyz.device, dtype=torch.float32)
        loss_nodes = soft_node_percent.to(device=gen_xyz.device, dtype=torch.float32)
        actuator_rate_proxy = self._actuator_soft_rate_proxy(args, loss_nodes)
        actuator_terms = getattr(args, "_last_actuator_soft_terms", {}) or {}
        prune_soft_rate = self._as_proxy_scalar(actuator_terms.get("prune_soft_rate"), loss_nodes)
        prune_soft_node = self._as_proxy_scalar(actuator_terms.get("prune_soft_node"), loss_nodes)
        prune_soft_single = self._as_proxy_scalar(actuator_terms.get("prune_soft_single"), loss_nodes)
        prune_soft_bit = self._as_proxy_scalar(actuator_terms.get("prune_soft_bit"), loss_nodes)
        prune_com_proxy = torch.nan_to_num(
            prune_soft_rate + 0.50 * prune_soft_node + 0.50 * prune_soft_single + prune_soft_bit,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if actuator_rate_proxy.requires_grad:
            loss_nodes = loss_nodes + float(getattr(args, "compression_soft_node_actuator_grad_weight", 10.0)) * (
                actuator_rate_proxy - actuator_rate_proxy.detach()
            )
            loss_single = loss_single + float(getattr(args, "compression_soft_single_actuator_grad_weight", 5.0)) * (
                actuator_rate_proxy - actuator_rate_proxy.detach()
            )
        if prune_com_proxy.requires_grad:
            loss_nodes = loss_nodes + float(getattr(args, "compression_soft_prune_node_grad_weight", 25.0)) * (
                prune_soft_node - prune_soft_node.detach()
            )
            loss_single = loss_single + float(getattr(args, "compression_soft_prune_single_grad_weight", 20.0)) * (
                prune_soft_single - prune_soft_single.detach()
            )
        loss_bit_proxy = surrogate_bit_percent
        if actuator_rate_proxy.requires_grad:
            loss_bit_proxy = loss_bit_proxy + float(getattr(args, "compression_soft_bit_actuator_grad_weight", 10.0)) * (
                actuator_rate_proxy - actuator_rate_proxy.detach()
            )
        if prune_com_proxy.requires_grad:
            loss_bit_proxy = loss_bit_proxy + float(getattr(args, "compression_soft_prune_bit_grad_weight", 30.0)) * (
                prune_soft_bit - prune_soft_bit.detach()
            )
        if final_w is None:
            effective_point_count = gen_xyz.new_tensor(float(gen_xyz.shape[-1]), dtype=torch.float32)
        else:
            point_w = final_w.to(device=gen_xyz.device, dtype=torch.float32)
            if point_w.ndim == 3:
                point_w = point_w.squeeze(1)
            effective_point_count = point_w.clamp(0.0, 1.0).sum(dim=1).mean()
        ref_point_count = gen_xyz.new_tensor(float(max(int(gt_xyz.shape[-1]), 1)), dtype=torch.float32)
        soft_point_percent = 100.0 * (effective_point_count - ref_point_count) / ref_point_count.abs().clamp_min(1.0)
        sparsepcgc_soft_rate_weight = float(getattr(args, "compression_soft_rate_sparsepcgc_weight", 0.05))
        if teacher_is_actual_for_control and not sparse_aux_actual_teacher_allowed:
            sparsepcgc_soft_rate_weight = 0.0
        soft_rate_proxy_for_grad = (
            float(getattr(args, "compression_soft_rate_point_weight", 0.25)) * soft_point_percent
            + float(getattr(args, "compression_soft_rate_node_weight", 0.10)) * loss_nodes
            + float(getattr(args, "compression_soft_rate_single_weight", 0.05)) * loss_single
            + sparsepcgc_soft_rate_weight * sparse_terms["loss"].to(device=gen_xyz.device, dtype=torch.float32)
            + actuator_rate_proxy
        )
        soft_rate_proxy_for_grad = torch.nan_to_num(
            soft_rate_proxy_for_grad,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        soft_rate_proxy_grad_weight = max(
            float(getattr(args, "compression_soft_rate_proxy_grad_weight", 0.05)),
            0.0,
        ) * float(main_grad_scale)
        soft_rate_proxy_ste = (
            soft_rate_proxy_grad_weight
            * (soft_rate_proxy_for_grad - soft_rate_proxy_for_grad.detach())
            if inputs_finite and soft_rate_proxy_grad_weight > 0.0
            else gen_xyz.new_zeros(())
        )
        prune_rate_proxy_grad_weight = max(
            float(getattr(args, "compression_soft_prune_rate_proxy_grad_weight", 10.0)),
            0.0,
        ) * float(main_grad_scale)
        soft_prune_rate_ste = (
            prune_rate_proxy_grad_weight * (prune_com_proxy - prune_com_proxy.detach())
            if inputs_finite and prune_com_proxy.requires_grad and prune_rate_proxy_grad_weight > 0.0
            else gen_xyz.new_zeros(())
        )
        forward_mode = str(
            getattr(args, "compression_surrogate_forward_mode", "teacher_ste")
        ).strip().lower()

        detach_surrogate_from_network = bool(
            getattr(args, "detach_surrogate_from_network", True)
        )

        if detach_surrogate_from_network:
            # ============================================================
            # Surrogate由来の勾配をNetworkへ流さないモード
            # ============================================================
            surrogate_weight = 0.0

            # ログ用・確認用の値は作るが、Networkへ勾配を返さない
            surrogate_loss_for_grad = (
                loss_bit_proxy.detach()
                if inputs_finite and torch.is_tensor(loss_bit_proxy)
                else gen_xyz.new_zeros(())
            )
            surrogate_loss_for_grad_weighted = gen_xyz.new_zeros(())

            # forward教師値は実Codec値またはmissing時のlocal proxy値。Networkへ直接勾配は返さない。
            main_loss = forward_teacher_percent_t.detach()
        else:
            # ============================================================
            # 従来モード：Surrogate/soft proxyの勾配をNetworkへ返す
            # ============================================================
            surrogate_weight = self._surrogate_weight(args) * float(main_grad_scale)
            surrogate_loss_for_grad = loss_bit_proxy if inputs_finite else gen_xyz.new_zeros(())
            surrogate_loss_for_grad_weighted = surrogate_weight * surrogate_loss_for_grad

            if forward_mode == "teacher_ste":
                surrogate_loss = surrogate_bit_percent if inputs_finite else None
                if surrogate_loss is None:
                    main_loss = forward_teacher_percent_t
                else:
                    main_loss = forward_teacher_percent_t + surrogate_weight * (
                        surrogate_loss - surrogate_loss.detach()
                    )
            else:
                main_loss = (
                    float(main_grad_scale) * surrogate_bit_percent
                    if inputs_finite
                    else forward_teacher_percent_t
                )

        main_loss = main_loss + soft_rate_proxy_ste + soft_prune_rate_ste
        if not detach_surrogate_from_network:
            surrogate_loss_for_grad_weighted = surrogate_loss_for_grad_weighted + (
                soft_rate_proxy_grad_weight * soft_rate_proxy_for_grad
                if inputs_finite and soft_rate_proxy_grad_weight > 0.0
                else gen_xyz.new_zeros(())
            )
            surrogate_loss_for_grad_weighted = surrogate_loss_for_grad_weighted + (
                prune_rate_proxy_grad_weight * prune_com_proxy
                if inputs_finite and prune_com_proxy.requires_grad and prune_rate_proxy_grad_weight > 0.0
                else gen_xyz.new_zeros(())
            )
        else:
            surrogate_loss_for_grad_weighted = gen_xyz.new_zeros(())
        try:
            setattr(
                args,
                "_soft_proxy_com_debug",
                {
                    "soft_proxy_com_requires_grad": self._proxy_debug_requires_grad(soft_rate_proxy_for_grad),
                    "soft_proxy_prune_com_requires_grad": self._proxy_debug_requires_grad(prune_com_proxy),
                    "drop_prob_requires_grad": self._proxy_debug_requires_grad(actuator_terms.get("drop_prob")),
                    "keep_prob_requires_grad": self._proxy_debug_requires_grad(actuator_terms.get("keep_prob")),
                    "drop_prob_mean": self._proxy_debug_scalar(actuator_terms.get("drop_prob_mean")),
                    "drop_prob_min": self._proxy_debug_scalar(actuator_terms.get("drop_prob_min")),
                    "drop_prob_max": self._proxy_debug_scalar(actuator_terms.get("drop_prob_max")),
                    "drop_prob_proxy_mean": self._proxy_debug_scalar(actuator_terms.get("drop_prob_proxy_mean")),
                    "drop_prob_proxy_min": self._proxy_debug_scalar(actuator_terms.get("drop_prob_proxy_min")),
                    "drop_prob_proxy_max": self._proxy_debug_scalar(actuator_terms.get("drop_prob_proxy_max")),
                    "keep_prob_mean": self._proxy_debug_scalar(actuator_terms.get("keep_prob_mean")),
                    "keep_prob_min": self._proxy_debug_scalar(actuator_terms.get("keep_prob_min")),
                    "keep_prob_max": self._proxy_debug_scalar(actuator_terms.get("keep_prob_max")),
                    "drop_logit_mean": self._proxy_debug_scalar(actuator_terms.get("drop_logit_mean")),
                    "drop_logit_min": self._proxy_debug_scalar(actuator_terms.get("drop_logit_min")),
                    "drop_logit_max": self._proxy_debug_scalar(actuator_terms.get("drop_logit_max")),
                    "drop_entropy": self._proxy_debug_scalar(actuator_terms.get("drop_entropy")),
                    "selected_drop_count_hard": self._proxy_debug_scalar(actuator_terms.get("selected_drop_count_hard")),
                    "soft_drop_mass": self._proxy_debug_scalar(actuator_terms.get("soft_drop_mass")),
                    "prune_soft_rate_value": self._proxy_debug_scalar(prune_soft_rate),
                    "prune_soft_node_value": self._proxy_debug_scalar(prune_soft_node),
                    "prune_soft_single_value": self._proxy_debug_scalar(prune_soft_single),
                    "prune_soft_bit_value": self._proxy_debug_scalar(prune_soft_bit),
                },
            )
        except Exception:
            pass
        aux_loss = aux_node_weight * loss_nodes + aux_single_weight * loss_single
        aux_objective = aux_loss if bool(getattr(args, "compression_surrogate_aux_in_objective", False)) else aux_loss.new_zeros(())
        sparsepcgc_aux_weight = float(getattr(args, "com_sparsepcgc", 0.0))
        sparse_aux_raw = sparse_terms["loss"]
        sparse_aux_loss = sparsepcgc_aux_weight * sparse_aux_raw
        sparsepcgc_aux_backprop_requested = bool(getattr(args, "sparsepcgc_aux_backprop", False))
        sparse_corr, sparse_sign_match, sparse_gate_count = self._update_sparsepcgc_aux_gate(
            args,
            self._scalar(sparse_aux_loss.detach()),
            float(actual_bit_percent),
            bool(actual_value_source == "fresh_teacher"),
        )
        sparsepcgc_aux_gating = bool(getattr(args, "sparsepcgc_aux_gating", True))
        sparsepcgc_aux_gate_mode = str(getattr(args, "sparsepcgc_aux_gate_mode", "hard")).strip().lower()
        sparsepcgc_aux_min_corr = float(getattr(args, "sparsepcgc_aux_min_corr", 0.30))
        sparsepcgc_aux_min_sign = float(getattr(args, "sparsepcgc_aux_min_sign_match", 0.50))
        sparsepcgc_aux_used_for_backprop = bool(
            sparsepcgc_aux_backprop_requested and sparse_aux_actual_teacher_allowed
        )
        sparsepcgc_aux_gating_reason = (
            "disabled_by_arg"
            if not sparsepcgc_aux_backprop_requested
            else ("blocked_by_actual_teacher" if not sparse_aux_actual_teacher_allowed else "enabled")
        )
        sparsepcgc_aux_gate_multiplier = 1.0 if sparsepcgc_aux_used_for_backprop else 0.0
        if sparsepcgc_aux_used_for_backprop and sparsepcgc_aux_gating:
            if sparsepcgc_aux_gate_mode == "soft":
                min_mult = min(max(float(getattr(args, "sparsepcgc_aux_soft_min_weight", 0.05)), 0.0), 1.0)
                if sparse_corr is None or sparse_sign_match is None:
                    sparsepcgc_aux_gate_multiplier = min_mult
                    sparsepcgc_aux_gating_reason = "soft_insufficient_rolling_pairs"
                else:
                    corr_denom = max(1.0 - sparsepcgc_aux_min_corr, 1e-6)
                    sign_denom = max(1.0 - sparsepcgc_aux_min_sign, 1e-6)
                    corr_quality = min(max((float(sparse_corr) - sparsepcgc_aux_min_corr) / corr_denom, 0.0), 1.0)
                    sign_quality = min(max((float(sparse_sign_match) - sparsepcgc_aux_min_sign) / sign_denom, 0.0), 1.0)
                    quality = min(corr_quality, sign_quality)
                    sparsepcgc_aux_gate_multiplier = min_mult + (1.0 - min_mult) * quality
                    if quality >= 1.0:
                        sparsepcgc_aux_gating_reason = "soft_passed"
                    elif sparse_corr < sparsepcgc_aux_min_corr:
                        sparsepcgc_aux_gating_reason = "soft_corr_scaled"
                    elif sparse_sign_match < sparsepcgc_aux_min_sign:
                        sparsepcgc_aux_gating_reason = "soft_sign_match_scaled"
                    else:
                        sparsepcgc_aux_gating_reason = "soft_partial"
                sparsepcgc_aux_used_for_backprop = bool(sparsepcgc_aux_gate_multiplier > 0.0)
            else:
                if sparse_corr is None or sparse_sign_match is None:
                    sparsepcgc_aux_used_for_backprop = False
                    sparsepcgc_aux_gate_multiplier = 0.0
                    sparsepcgc_aux_gating_reason = "insufficient_rolling_pairs"
                elif sparse_corr < sparsepcgc_aux_min_corr:
                    sparsepcgc_aux_used_for_backprop = False
                    sparsepcgc_aux_gate_multiplier = 0.0
                    sparsepcgc_aux_gating_reason = "corr_below_threshold"
                elif sparse_sign_match < sparsepcgc_aux_min_sign:
                    sparsepcgc_aux_used_for_backprop = False
                    sparsepcgc_aux_gate_multiplier = 0.0
                    sparsepcgc_aux_gating_reason = "sign_match_below_threshold"
                else:
                    sparsepcgc_aux_gating_reason = "passed"
        elif sparsepcgc_aux_backprop_requested and sparse_aux_actual_teacher_allowed:
            sparsepcgc_aux_gating_reason = "gating_disabled"
        sparsepcgc_aux_weight_effective = sparsepcgc_aux_weight * sparsepcgc_aux_gate_multiplier if sparsepcgc_aux_used_for_backprop else 0.0
        sparse_aux_objective = sparsepcgc_aux_weight_effective * sparse_aux_raw if sparsepcgc_aux_used_for_backprop else sparse_aux_loss.new_zeros(())
        sparse_aux_term = sparse_terms["loss"] if sparsepcgc_aux_used_for_backprop else sparse_terms["loss"].detach().new_zeros(())
        proxy_aux_for_grad = sparse_aux_objective if sparsepcgc_aux_used_for_backprop else sparse_aux_loss.new_zeros(())
        lcom_without_sparse = main_loss + aux_objective
        lcom_with_sparse = main_loss + aux_loss + sparse_aux_loss
        L_com = lcom_without_sparse + sparse_aux_objective

        def _has_grad(value):
            return bool(torch.is_tensor(value) and value.requires_grad)

        surrogate_pred_grad_active = bool(
            (not detach_surrogate_from_network)
            and inputs_finite
            and _has_grad(surrogate_bit_percent)
            and abs(float(surrogate_weight)) > 0.0
        )
        soft_rate_proxy_grad_active = _has_grad(soft_rate_proxy_ste)
        soft_prune_proxy_grad_active = _has_grad(soft_prune_rate_ste)
        sparse_aux_grad_active = bool(sparsepcgc_aux_used_for_backprop and _has_grad(sparse_aux_objective))
        network_grad_components = []
        if surrogate_pred_grad_active:
            network_grad_components.append("surrogate_pred_ste")
        if soft_rate_proxy_grad_active:
            network_grad_components.append("soft_rate_proxy_ste")
        if soft_prune_proxy_grad_active:
            network_grad_components.append("soft_prune_proxy_ste")
        if sparse_aux_grad_active:
            network_grad_components.append("sparsepcgc_aux")

        if network_grad_components:
            grad_source = "+".join(network_grad_components)
        else:
            grad_source = "actual_only_no_grad"
        backend_label = self._surrogate_backend_label(args, teacher_codec)

        # ============================================================
        # last_compression_terms にはNetwork学習に必要な圧縮勾配を残す。
        # detachするのは、実Codec教師値 hard と Surrogate確認用項だけである。
        # ============================================================
        if detach_surrogate_from_network:
            stored_hard = forward_teacher_percent_t.detach()
            stored_surrogate = gen_xyz.new_zeros(())
        else:
            stored_hard = forward_teacher_percent_t
            stored_surrogate = surrogate_loss_for_grad_weighted

        stored_main = main_loss
        stored_bit = loss_bit_proxy
        stored_node = loss_nodes
        stored_single = loss_single
        stored_objective = L_com
        stored_forward = L_com
        stored_aux = aux_loss
        stored_sparsepcgc = sparse_aux_term
        stored_op = soft_rate_proxy_for_grad
        compression_grad_component_summary = ",".join(network_grad_components) if network_grad_components else "none"
        
        self._store_compression_terms(
            main=stored_main,
            bit=stored_bit,
            node=stored_node,
            single=stored_single,
            bpn=gen_xyz.new_zeros(()),
            objective=stored_objective,
            forward=stored_forward,
            hard=stored_hard,
            surrogate=stored_surrogate,
            aux=stored_aux,
            sparsepcgc=stored_sparsepcgc,
            op=stored_op,
            backend=backend_label,
        )

        current_step_for_debug = int(
            getattr(args, "_global_train_step", getattr(self, "_surrogate_call_count", 0))
        )
        if teacher_refreshed:
            target_step_for_debug = current_step_for_debug
        elif target_entry is not None:
            target_step_for_debug = int(target_entry.get("global_step", current_step_for_debug))
        else:
            target_step_for_debug = current_step_for_debug
        teacher_target_age = max(current_step_for_debug - target_step_for_debug, 0)
        rate_proxy_before_value = float(cached_gt["bit"])
        rate_proxy_after_value = rate_proxy_before_value * (1.0 + self._scalar(pred_percent) / 100.0)
        gt_octree_node_value = float(cached_gt.get("octree_node", cached_gt.get("node", 0.0)))
        gt_octree_single_value = float(cached_gt.get("octree_single", cached_gt.get("single", 0.0)))
        if target_entry is not None and not teacher_refreshed:
            gen_octree_node_value = float(target_entry.get("gen_octree_node", 0.0))
            gen_octree_single_value = float(target_entry.get("gen_octree_single", 0.0))
            gen_octree_depth_value = int(target_entry.get("gen_octree_depth", 0))
        elif stats_gen is not None:
            gen_octree_node_value = float(stats_gen.get("octree_node", 0.0))
            gen_octree_single_value = float(stats_gen.get("octree_single", 0.0))
            gen_octree_depth_value = int(stats_gen.get("octree_depth", 0))
        else:
            gen_octree_node_value = 0.0
            gen_octree_single_value = 0.0
            gen_octree_depth_value = 0
        teacher_stale = bool((target_entry is not None) and (not teacher_refreshed) and ("stale" in str(target_cache_hit).lower()))
        teacher_replayed = bool(replay_loss is not None)
        teacher_skipped = bool(
            (not teacher_refreshed)
            and target_entry is None
            and replay_loss is None
            and actual_value_source != "local_proxy"
        )
        if teacher_refreshed:
            teacher_mode = "refresh"
        elif actual_value_source == "local_proxy":
            teacher_mode = "local_proxy"
        elif teacher_stale:
            teacher_mode = "stale"
        elif target_entry is not None:
            teacher_mode = "cache"
        elif teacher_replayed:
            teacher_mode = "replay"
        else:
            teacher_mode = "skip"
        if target_teacher_source == "none":
            target_teacher_source = actual_value_source
        teacher_is_actual = bool(actual_value_source in {"fresh_teacher", "target_cache", "stale_target"})
        teacher_is_local_proxy = bool(actual_value_source == "local_proxy")
        pred_clip_value = resolve_surrogate_pred_clip(args)
        pred_clipped = bool(abs(self._scalar(pred_raw_percent) - self._scalar(pred_percent)) > 1e-6)
        pred_clip_min_value = -pred_clip_value if pred_clip_value > 0.0 else float("nan")
        pred_clip_max_value = pred_clip_value if pred_clip_value > 0.0 else float("nan")
        pred_ratio = 1.0 + self._scalar(pred_percent) / 100.0
        if bool(getattr(args, "surrogate_use_log_bit_ratio_target", False)):
            surrogate_pred_log_ratio = self._scalar(pred_percent) / max(float(getattr(args, "surrogate_log_bit_ratio_scale", 100.0)), 1e-9)
        else:
            surrogate_pred_log_ratio = math.log(max(pred_ratio, 1e-12)) if math.isfinite(pred_ratio) else float("nan")
        teacher_type_label = "full_cloud_actual" if str(getattr(args, "_current_teacher_scope", "")) == "full_cloud" and teacher_is_actual else ("subtree_local_actual" if teacher_is_actual else str(actual_value_source))
        teacher_gap_debug = self._update_teacher_gap_debug(args, cache_key, teacher_type_label, actual_bit_percent, teacher_is_actual)
        replay_ratio = float(replay_full_cloud_count) / float(max(replay_sample_count, 1))
        feature_names = self._surrogate_feature_names(args)
        if teacher_refreshed and stats_gen is not None:
            actual_occupancy_debug = self._actual_occupancy_debug_from_stats(cached_gt, stats_gen)
        elif target_entry is not None:
            actual_occupancy_debug = {
                key: target_entry.get(key, float("nan"))
                for key in ACTUAL_OCCUPANCY_DEBUG_KEYS
            }
        else:
            actual_occupancy_debug = {}

        full_cloud_corr_state = getattr(args, "_full_cloud_actual_correction_state", {}) or {}
        if not isinstance(full_cloud_corr_state, dict):
            full_cloud_corr_state = {}

        self.last_compression_debug = {
            "metric": "actual_total_bit_percent",
            "teacher_codec": teacher_codec,
            "total_bit": self._scalar(surrogate_bit_percent),
            "compression_objective": self._scalar(L_com),
            "compression_main_loss": self._scalar(main_loss),
            "compression_aux_loss": self._scalar(aux_loss),
            "compression_soft_rate_proxy_for_grad": self._scalar(soft_rate_proxy_for_grad.detach()),
            "compression_soft_rate_proxy_grad_weight": float(soft_rate_proxy_grad_weight),
            "compression_soft_point_percent": self._scalar(soft_point_percent.detach()),
            "compression_aux_in_objective": bool(getattr(args, "compression_surrogate_aux_in_objective", False)),
            "compression_main_grad_scale": float(main_grad_scale),
            "compression_main_grad_scale_reason": str(main_grad_scale_reason),
            "compression_proxy_input_mode": compression_proxy_input_mode,
            "compression_proxy_uses_subtree_tree": bool(uses_subtree_tree),
            "compression_proxy_uses_full_context": bool(isinstance(full_octree_context, dict)),
            "compression_proxy_fallback_reason": "" if uses_subtree_tree or requested_octree_mode == "full_cloud" else "missing_subtree_tree",
            "prebuilt_node_count_used": float(prebuilt_node_count),
            "prebuilt_single_child_count_used": float(prebuilt_single_count),
            "sparsepcgc_aux_loss": self._scalar(sparse_terms["loss"].detach()),
            "sparsepcgc_aux_value": self._scalar(sparse_aux_loss.detach()),
            "sparsepcgc_aux_raw": self._scalar(sparse_aux_raw.detach()),
            "sparsepcgc_aux_weighted": self._scalar(sparse_aux_loss.detach()),
            "sparsepcgc_aux_weight": sparsepcgc_aux_weight,
            "sparsepcgc_aux_weight_raw": float(sparsepcgc_aux_weight),
            "sparsepcgc_aux_weight_effective": float(sparsepcgc_aux_weight_effective),
            "sparsepcgc_aux_backprop": bool(sparsepcgc_aux_backprop_requested),
            "sparsepcgc_aux_used_for_backprop": bool(sparsepcgc_aux_used_for_backprop),
            "sparsepcgc_aux_actual_teacher_allowed": bool(sparse_aux_actual_teacher_allowed),
            "sparsepcgc_aux_gating_enabled": bool(sparsepcgc_aux_gating),
            "sparsepcgc_aux_gate_mode": str(sparsepcgc_aux_gate_mode),
            "sparsepcgc_aux_gate_multiplier": float(sparsepcgc_aux_gate_multiplier),
            "sparsepcgc_aux_gating_reason": str(sparsepcgc_aux_gating_reason),
            "corr_sparsepcgc_aux_actual_rolling": None if sparse_corr is None else float(sparse_corr),
            "sign_match_sparsepcgc_aux_actual_rolling": None if sparse_sign_match is None else float(sparse_sign_match),
            "sparsepcgc_aux_gating_count": int(sparse_gate_count),
            "com_sparsepcgc_weight": sparsepcgc_aux_weight,
            "lcom_without_sparsepcgc_aux": self._scalar(lcom_without_sparse.detach()),
            "lcom_with_sparsepcgc_aux": self._scalar(lcom_with_sparse.detach()),
            "sparsepcgc_active_coord_loss": self._scalar(sparse_terms["active"].detach()),
            "sparsepcgc_isolated_proxy_loss": self._scalar(sparse_terms["single"].detach()),
            "sparsepcgc_entropy_proxy_loss": self._scalar(sparse_terms["entropy"].detach()),
            "sparsepcgc_density_proxy_loss": self._scalar(sparse_terms["density"].detach()),
            "occupancy_entropy": self._scalar(sparse_terms["entropy"].detach()),
            "nll_delta": self._scalar(sparse_terms["entropy"].detach()),
            "occupancy_pattern_before": self._scalar(sparse_terms.get("occupancy_pattern_before", sparse_aux_loss.new_tensor(float("nan")))),
            "occupancy_pattern_after": self._scalar(sparse_terms.get("occupancy_pattern_after", sparse_aux_loss.new_tensor(float("nan")))),
            "occupancy_pattern_delta": self._scalar(sparse_terms.get("occupancy_pattern_delta", sparse_aux_loss.new_tensor(float("nan")))),
            "lowprob_occupancy_count_before": self._scalar(sparse_terms.get("lowprob_occupancy_count_before", sparse_aux_loss.new_tensor(float("nan")))),
            "lowprob_occupancy_count_after": self._scalar(sparse_terms.get("lowprob_occupancy_count_after", sparse_aux_loss.new_tensor(float("nan")))),
            "lowprob_occupancy_ratio": self._scalar(sparse_terms.get("lowprob_occupancy_ratio", sparse_aux_loss.new_tensor(float("nan")))),
            "occupancy_entropy_before": self._scalar(sparse_terms.get("occupancy_entropy_before", sparse_aux_loss.new_tensor(float("nan")))),
            "occupancy_entropy_after": self._scalar(sparse_terms.get("occupancy_entropy_after", sparse_aux_loss.new_tensor(float("nan")))),
            "occupancy_entropy_delta": self._scalar(sparse_terms.get("occupancy_entropy_delta", sparse_aux_loss.new_tensor(float("nan")))),
            "occupancy_nll_before": self._scalar(sparse_terms.get("occupancy_nll_before", sparse_aux_loss.new_tensor(float("nan")))),
            "occupancy_nll_after": self._scalar(sparse_terms.get("occupancy_nll_after", sparse_aux_loss.new_tensor(float("nan")))),
            "occupancy_nll_delta": self._scalar(sparse_terms.get("occupancy_nll_delta", sparse_aux_loss.new_tensor(float("nan")))),
            "single_child_chain_length_before": self._scalar(sparse_terms.get("single_child_chain_length_before", sparse_aux_loss.new_tensor(float("nan")))),
            "single_child_chain_length_after": self._scalar(sparse_terms.get("single_child_chain_length_after", sparse_aux_loss.new_tensor(float("nan")))),
            "sibling_occupancy_balance_before": self._scalar(sparse_terms.get("sibling_occupancy_balance_before", sparse_aux_loss.new_tensor(float("nan")))),
            "sibling_occupancy_balance_after": self._scalar(sparse_terms.get("sibling_occupancy_balance_after", sparse_aux_loss.new_tensor(float("nan")))),
            "bpp": float(actual_bpp_percent),
            "gt_points": int(cached_gt["point_count"]),
            "gen_points": gen_points,
                "gt_actual_bit": float(cached_gt["bit"]),
                "gen_actual_bit": gen_actual_bit,
                "gen_total_bit_with_edit_record": float(gen_total_bit_with_edit_record),
                "actual_edit_record_bits": float(actual_edit_record_bits),
                "gt_actual_encode_time": float(cached_gt.get("encode_time", 0.0)),
            "gen_actual_encode_time": float(stats_gen.get("encode_time", 0.0)) if stats_gen is not None else 0.0,
            "actual_encode_time_total": float(cached_gt.get("encode_time", 0.0))
            + (float(stats_gen.get("encode_time", 0.0)) if stats_gen is not None else 0.0),
            "gt_unique_coord_count": int(cached_gt.get("unique_coord_count", cached_gt.get("point_count", 0))),
            "gen_unique_coord_count": int(
                stats_gen.get("unique_coord_count", stats_gen.get("point_count", gen_points))
            ) if stats_gen is not None else int(target_entry.get("gen_unique_coord_count", gen_points)) if target_entry is not None else gen_points,
                "actual_total_bit_percent": self._scalar(actual_bit_percent_t),
                "actual_bit_percent_raw": float(actual_bit_percent),
                "actual_bit_percent_used_for_loss": float(forward_teacher_percent_value),
                "actual_bit_percent_used_for_loss_source": str(forward_teacher_source),
                "actual_target": self._scalar(actual_bit_percent_t),
                "actual_raw_percent": float(actual_raw_percent_value),
                "policy_actual_noop_guard_used": bool(policy_actual_noop_guard_used),
                "policy_actual_noop_guard_margin": float(policy_actual_noop_guard_margin),
                "policy_actual_noop_guard_percent": float(policy_actual_noop_guard_percent),
                "policy_actual_noop_guard_raw_percent": float(policy_actual_noop_guard_raw_percent),
                "policy_actual_noop_guard_reason": (
                    "raw_actual_above_margin" if bool(policy_actual_noop_guard_used) else ""
                ),
                "policy_actual_noop_guard_replaced_in_loss": bool(
                    policy_actual_noop_guard_used
                    and abs(float(forward_teacher_percent_value) - float(actual_bit_percent)) > 1e-6
                ),
                "policy_actual_noop_guard_raw_bit": float(policy_actual_noop_guard_raw_bit),
                "policy_actual_noop_guard_raw_total_bit": float(policy_actual_noop_guard_raw_total_bit),
                "policy_actual_noop_guard_raw_edit_record_bits": float(policy_actual_noop_guard_raw_edit_record_bits),
                "actual_target_percent_with_edit_record": float(target_raw_percent_value),
            "actual_clamped_percent": float(target_clamped_percent_value),
            "actual_forward_value": self._scalar(forward_teacher_percent_t),
            "actual_forward_raw_value": float(forward_teacher_raw_percent_value),
            "actual_forward_clamped": bool(forward_teacher_clamped),
            "actual_forward_source": str(forward_teacher_source),
            "compression_loss_raw": float(actual_bit_percent),
            "compression_loss_used": float(forward_teacher_percent_value),
            "compression_loss_noop_replaced": bool(
                policy_actual_noop_guard_used
                and abs(float(forward_teacher_percent_value) - float(actual_bit_percent)) > 1e-6
            ),
            "compression_forward_teacher_percent": float(forward_teacher_percent_value),
            "compression_forward_teacher_source": str(forward_teacher_source),
            "local_proxy_target_percent": float(target_percent_value) if actual_value_source == "local_proxy" else None,
            "local_proxy_rate_target_percent": None if not math.isfinite(local_proxy_rate_target_value) else float(local_proxy_rate_target_value),
            "local_proxy_aux_target_percent": None if not math.isfinite(local_proxy_aux_target_value) else float(local_proxy_aux_target_value),
            "local_proxy_rate_error": str(local_proxy_rate_error),
            "forward_display_value": self._scalar(main_loss.detach()),
            "final_L_com_value": self._scalar(L_com.detach()),
            "surrogate_pred": self._scalar(surrogate_bit_percent.detach()),
            "surrogate_pred_raw_percent": self._scalar(pred_raw_percent),
            "surrogate_pred_clipped_percent": self._scalar(pred_percent),
            "surrogate_pred_log_ratio": float(surrogate_pred_log_ratio),
            "pred_clipped": bool(pred_clipped),
            "pred_clip_min": float(pred_clip_min_value),
            "pred_clip_max": float(pred_clip_max_value),
            "surrogate_loss_for_grad": self._scalar(surrogate_loss_for_grad_weighted.detach()),
            "proxy_aux_for_grad": self._scalar(proxy_aux_for_grad.detach()),
            "grad_source": grad_source,
            "detach_surrogate_from_network": bool(detach_surrogate_from_network),
            "surrogate_weight_effective": float(surrogate_weight),
            "surrogate_pred_requires_grad": _has_grad(surrogate_bit_percent),
            "surrogate_input_requires_grad": _has_grad(x_pred),
            "loss_bit_proxy_requires_grad": _has_grad(loss_bit_proxy),
            "soft_rate_proxy_requires_grad": _has_grad(soft_rate_proxy_for_grad),
            "soft_rate_proxy_ste_requires_grad": _has_grad(soft_rate_proxy_ste),
            "soft_prune_proxy_requires_grad": _has_grad(prune_com_proxy),
            "soft_prune_rate_ste_requires_grad": _has_grad(soft_prune_rate_ste),
            "main_loss_requires_grad": _has_grad(main_loss),
            "lcom_requires_grad": _has_grad(L_com),
            "stored_main_requires_grad": _has_grad(stored_main),
            "stored_bit_requires_grad": _has_grad(stored_bit),
            "stored_node_requires_grad": _has_grad(stored_node),
            "stored_single_requires_grad": _has_grad(stored_single),
            "stored_op_requires_grad": _has_grad(stored_op),
            "stored_surrogate_requires_grad": _has_grad(stored_surrogate),
            "network_grad_from_surrogate_pred": bool(surrogate_pred_grad_active),
            "network_grad_from_soft_rate_proxy": bool(soft_rate_proxy_grad_active),
            "network_grad_from_soft_prune_proxy": bool(soft_prune_proxy_grad_active),
            "network_grad_from_sparsepcgc_aux": bool(sparse_aux_grad_active),
            "network_grad_component_summary": compression_grad_component_summary,
            "compression_soft_prune_rate_proxy_grad_weight": float(prune_rate_proxy_grad_weight),
            "actual_value_is_fresh": bool(teacher_refreshed),
            "actual_value_source": actual_value_source,
            "actual_value_source_detail": (
                "actual SparsePCGC/codec relative bit percent"
                if teacher_is_actual
                else "local differentiable proxy, not actual SparsePCGC bit"
                if teacher_is_local_proxy
                else str(actual_value_source)
            ),
            "surrogate_teacher_source": target_teacher_source,
            "surrogate_teacher_is_actual": bool(teacher_is_actual),
            "surrogate_teacher_is_local_proxy": bool(teacher_is_local_proxy),
            "surrogate_target_scale": target_scale,
            "surrogate_target_raw_bit": float(target_raw_percent_value),
            "surrogate_target_train_bit": float(target_train_percent_value),
            "surrogate_train_target_percent": float(target_clamped_percent_value),
            "surrogate_train_target_log_ratio": float(target_log_ratio_value),
            "surrogate_train_target_value": float(target_train_percent_value),
            "surrogate_target_mode": str(target_mode_value),
            "target_clamp_min": float(target_clip_min_value),
            "target_clamp_max": float(target_clip_max_value),
            "raw_target_gap_percent": float(target_raw_percent_value) - self._scalar(pred_percent),
            "raw_actual_vs_train_target_gap": float(target_raw_percent_value) - float(target_train_percent_value),
            "surrogate_loss_against_raw_actual": self._scalar(surrogate_loss_against_raw_actual),
            "surrogate_loss_against_train_target": self._scalar(surrogate_loss_against_train_target),
            "surrogate_target_clamped": bool(target_was_clamped),
            "target_clamped": bool(target_was_clamped),
            "target_clamp_rate": float(target_clamp_rate),
            "surrogate_pred_clip": pred_clip_value,
            "surrogate_local_proxy_replay_stored": bool(local_proxy_replay_stored),
            "relative_percent_direction": "positive=worse_bits_increase; negative=better_bits_decrease",
            "rate_proxy_before": rate_proxy_before_value,
            "rate_proxy_after": rate_proxy_after_value,
            "rate_proxy_delta": self._scalar(surrogate_bit_percent.detach()),
            "actual_single_percent": float(actual_single_percent),
            "actual_node_percent": float(actual_node_percent),
            "soft_single_percent": self._scalar(loss_single.detach()),
            "soft_node_percent": self._scalar(loss_nodes.detach()),
            "heuristic_cause_score_node": self._scalar(loss_nodes.detach()),
            "heuristic_cause_score_single": self._scalar(loss_single.detach()),
            "heuristic_cause_score_lowprob": self._scalar(sparse_terms["entropy"].detach()),
            "heuristic_sparse_proxy": self._scalar(sparse_aux_loss.detach()),
            "heuristic_quant_proxy": self._scalar(sparse_terms["active"].detach()),
            "heuristic_node_proxy": self._scalar(loss_nodes.detach()),
            "cause_score_used_for_backprop": bool(sparsepcgc_aux_used_for_backprop),
            "cause_score_is_actual_teacher": False,
            "cause_score_is_heuristic": True,
            "node_delta": gen_octree_node_value - gt_octree_node_value,
            "single_delta": gen_octree_single_value - gt_octree_single_value,
            "surrogate_pred_bit": self._scalar(pred_percent),
            "surrogate_objective_bit": self._scalar(surrogate_bit_percent.detach()),
            "surrogate_target_bit": self._scalar(target_percent),
            "surrogate_abs_bit_error": self._scalar(surrogate_abs_bit_error),
            "surrogate_rel_error": self._scalar(
                surrogate_abs_bit_error / (target_percent.abs() + pred_percent.new_tensor(1e-6))
            ),
            "surrogate_signed_bit_error": self._scalar(surrogate_signed_bit_error.detach()),
            "surrogate_abs_mean_error": self._scalar(surrogate_abs_bit_error),
            "surrogate_train_loss": self._scalar(L_sur),
            "surrogate_replay_size": int(len(getattr(self, "surrogate_replay", []))),
            "surrogate_replay_sample_count": replay_sample_count,
            "surrogate_replay_steps": replay_steps,
            "replay_age": float(replay_age),
            "replay_full_cloud_count": int(replay_full_cloud_count),
            "replay_is_full_cloud": bool(replay_full_cloud_count > 0),
            "surrogate_replay_used": teacher_replayed,
            "surrogate_forward_mode": forward_mode,
            "teacher_refresh": bool(teacher_refreshed),
            "teacher_refreshed": bool(teacher_refreshed),
            "teacher_replayed": bool(teacher_replayed),
            "teacher_stale": bool(teacher_stale),
            "teacher_skipped": bool(teacher_skipped),
            "teacher_mode": teacher_mode,
            "teacher_type": teacher_type_label,
            "full_cloud_teacher_used": bool(str(getattr(args, "_current_teacher_scope", "")) == "full_cloud" and teacher_is_actual),
            "full_cloud_actual_percent": float(actual_bit_percent) if str(getattr(args, "_current_teacher_scope", "")) == "full_cloud" and teacher_is_actual else None,
            "subtree_teacher_percent": float(actual_bit_percent) if str(getattr(args, "_current_teacher_scope", "")) == "subtree_local" and teacher_is_actual else None,
            "teacher_gap_percent": teacher_gap_debug.get("teacher_gap_percent"),
            "teacher_gap_status": str(teacher_gap_debug.get("teacher_gap_status", "")),
            "full_cloud_teacher_count": int(1 if teacher_type_label == "full_cloud_actual" else 0),
            "subtree_teacher_count": int(1 if teacher_type_label == "subtree_local_actual" else 0),
            "full_cloud_actual_correction_gap_ema": full_cloud_corr_state.get("ema_full_vs_subtree_gap"),
            "full_cloud_actual_correction_context_gap_ema": full_cloud_corr_state.get("ema_full_vs_context_gap"),
            "full_cloud_actual_correction_proxy_gap_ema": full_cloud_corr_state.get("ema_full_vs_proxy_gap"),
            "full_cloud_actual_correction_last_full_delta": full_cloud_corr_state.get("last_full_actual_delta"),
            "full_cloud_actual_correction_last_subtree_delta": full_cloud_corr_state.get("last_subtree_actual_delta"),
            "full_cloud_actual_correction_last_context_delta": full_cloud_corr_state.get("last_full_context_delta"),
            "teacher_type_counts": str(teacher_type_label),
            "full_cloud_replay_ratio": float(replay_ratio),
            "full_cloud_calib_interval": int(getattr(args, "surrogate_full_cloud_calib_interval", 0)),
            "full_cloud_calib_triggered": bool(str(getattr(args, "_current_teacher_anchor_reason", "")) == "surrogate_full_cloud_calib"),
            "teacher_anchor_reason": str(getattr(args, "_current_teacher_anchor_reason", "")),
            "subtree_id": str(getattr(args, "_current_subtree_id", "")),
            "sample_name": str(getattr(args, "_current_sample_name", "")),
            "teacher_cache_hit": target_cache_hit,
            "teacher_target_age": int(teacher_target_age),
            "teacher_refresh_interval": int(getattr(args, "compression_surrogate_refresh_interval", 0)),
            "actual_eval_interval": int(getattr(args, "actual_eval_interval", 0)),
            "reuse_last_target": bool(getattr(args, "compression_surrogate_reuse_last_target", True)),
            "teacher_refresh_policy": self._surrogate_refresh_policy_label(args),
            "surrogate_backward_enabled": bool(inputs_finite),
            "surrogate_update_during_training": bool(getattr(args, "surrogate_update_during_training", True)),
            "surrogate_update_interval": int(getattr(args, "surrogate_update_interval", 1)),
            "surrogate_update_on_teacher_refresh_only": bool(
                getattr(args, "surrogate_update_on_teacher_refresh_only", False)
            ),
            "surrogate_auto_frozen": bool(auto_freeze_debug.get("frozen", False)),
            "surrogate_auto_freeze_streak": int(auto_freeze_debug.get("streak", 0)),
            "surrogate_auto_freeze_event": str(auto_freeze_debug.get("event", "")),
            "surrogate_pretrain_active": bool(getattr(args, "_surrogate_pretrain_active", False)),
            "surrogate_pretrain_mode": pretrain_mode,
            "surrogate_pretrain_teacher_type": pretrain_teacher_type,
            "surrogate_pretrain_actual_scope": str(getattr(args, "_surrogate_pretrain_actual_scope", "")),
            "surrogate_pretrain_full_calibration": bool(getattr(args, "_surrogate_pretrain_full_calibration", False)),
            "surrogate_input_feature_names": ",".join(feature_names),
            "surrogate_input_feature_dim": int(x_soft.shape[1]),
            "surrogate_uses_operation_features": False,
            "surrogate_uses_codec_condition": True,
            "surrogate_uses_quant_condition": True,
            "surrogate_uses_occupancy_features": True,
            "surrogate_uses_before_after_delta": False,
            "surrogate_input_version": "soft_octree_global_v2",
            "inputs_finite": bool(inputs_finite),
            "before_bits": float(cached_gt["bit"]),
            "after_bits": float(gen_actual_bit),
            "log_bit_ratio": float(math.log(max(float(gen_actual_bit), 1e-9) / max(float(cached_gt["bit"]), 1e-9))) if math.isfinite(float(gen_actual_bit)) else None,
            "gt_octree_node": gt_octree_node_value,
            "gen_octree_node": gen_octree_node_value,
            "gt_octree_single": gt_octree_single_value,
            "gen_octree_single": gen_octree_single_value,
            "gt_octree_depth": int(cached_gt.get("octree_depth", target_entry.get("gt_octree_depth", 0) if target_entry else 0)),
            "gen_octree_depth": gen_octree_depth_value,
            "timing": {key: round(float(value), 6) for key, value in timing.items()},
        }
        self.last_compression_debug.update(actual_occupancy_debug)
        debug_gen_xyz = gen_xyz if actual_gen_xyz is None else actual_gen_xyz
        self._maybe_update_sparsepcgc_debug(
            args,
            self.last_compression_debug,
            gen_xyz=debug_gen_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
            codec_name=teacher_codec,
        )

        if self._should_verbose_step(args):
            msg = (
                f"L_com(actual_total_bit_percent):{self._scalar(actual_bit_percent_t):.6f}, "
                f"surrogate_bit_percent:{self._scalar(pred_percent):.6f}, "
                f"surrogate_abs_bit_error:{self._scalar(surrogate_abs_bit_error):.6f}, "
                f"L_sur:{self._scalar(L_sur):.6f}"
            )
            if teacher_refreshed and stats_gen is not None:
                msg += (
                    f", actual_bit:{float(cached_gt['bit']):.6f}->{float(stats_gen['bit']):.6f} "
                    f"({float(actual_bit_percent):.4f}%)"
                )
            elif target_entry is not None:
                msg += f", actual_bit:cached({target_cache_hit})"
            else:
                msg += ", actual_bit:skipped(non-finite)"
            self.writer.write(msg)

        self._log_compression_grad_probe(args, backend_label, L_com, gen_xyz)
        return (
            L_com,
            loss_bit_proxy,
            loss_single,
            loss_nodes,
            cached_gt,
            {
                "bit": float(cached_gt["bit"]),
                "bpp": float(cached_gt["bpp"]),
                "bpn": float(cached_gt["bpn"]),
                "single": float(cached_gt["single"]),
                "node": float(cached_gt["node"]),
            },
        )
