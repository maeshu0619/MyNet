import time

import numpy as np
import torch

from .actual_encoder import build_actual_encoder
from models.utils.compression.octree_stats import hard_octree_occupancy_stats


class CompressionLossMixin:
    def _store_compression_terms(self, **terms):
        self.last_compression_terms = dict(terms)

    @staticmethod
    def _compression_rate_metric(args):
        return str(getattr(args, "compression_rate_metric", "bits_per_point")).strip().lower()

    @staticmethod
    def _compression_loss_backend(args):
        return str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()

    @staticmethod
    def _codec_key(args):
        backend = str(getattr(args, "compression_loss_backend", "")).strip().lower()
        compress = str(getattr(args, "compress", "OctAttention")).strip().lower().replace("_", "").replace("-", "")
        if backend.startswith("sparsepcgc") or compress == "sparsepcgc":
            return "sparsepcgc"
        if backend.startswith("gpcc") or compress == "gpcc":
            return "gpcc"
        if backend.startswith("draco") or compress == "draco":
            return "draco"
        return "octattention"

    def _is_sparsepcgc_context(self, args, codec_name=None):
        if codec_name is not None and str(codec_name).strip().lower() == "sparsepcgc":
            return True
        return self._codec_key(args) == "sparsepcgc"

    @staticmethod
    def _positive_count(count):
        return max(float(count), 1.0)

    def _metric_value(self, value, own_point_count, ref_point_count, args):
        mode = self._compression_rate_metric(args)
        if mode == "total_bits":
            return value
        if mode == "bits_per_input_point":
            return value / self._positive_count(ref_point_count)
        return value / self._positive_count(own_point_count)

    @staticmethod
    def _relative_ratio(value, ref, ref_min=1e-12):
        if torch.is_tensor(value):
            ref_t = value.new_tensor(float(ref))
            denom = ref_t.abs().clamp_min(float(ref_min))
            return (value - ref_t) / denom
        denom = max(abs(float(ref)), float(ref_min))
        return (float(value) - float(ref)) / denom

    @staticmethod
    def _relative_percent(value, ref, ref_min=1e-12):
        return 100.0 * CompressionLossMixin._relative_ratio(value, ref, ref_min)

    def _sparsepcgc_quantized_coords(self, args, pts_3n):
        voxel_size = max(float(getattr(args, "sparsepcgc_voxel_size", 1.0)), 1e-9)
        pos_q = max(int(getattr(args, "sparsepcgc_pos_quantscale", 1)), 1)
        coords = torch.round(pts_3n.transpose(0, 1).contiguous().to(torch.float32) / voxel_size)
        if pos_q > 1:
            coords = torch.round(coords / float(pos_q))
        return coords.to(torch.long)

    @staticmethod
    def _coord_key_3d(coords):
        if coords.numel() == 0:
            return coords.new_zeros((0,), dtype=torch.long)
        mins = coords.amin(dim=0)
        shifted = coords - mins
        span = (shifted.amax(dim=0) + 1).clamp_min(1)
        return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]

    def _sparsepcgc_isolated_count(self, unique_coords):
        if unique_coords.numel() == 0:
            return 0, 0.0
        offsets = torch.tensor(
            [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ],
            device=unique_coords.device,
            dtype=torch.long,
        )
        query = (unique_coords[:, None, :] + offsets.view(1, -1, 3)).reshape(-1, 3)
        combined = torch.cat([unique_coords, query], dim=0)
        mins = combined.amin(dim=0)
        span = (combined.amax(dim=0) - mins + 1).clamp_min(1)
        def _keys(coords):
            shifted = coords - mins
            return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]
        occupied_keys = torch.unique(_keys(unique_coords), sorted=True)
        query_keys = _keys(query)
        pos = torch.searchsorted(occupied_keys, query_keys)
        in_bounds = pos < occupied_keys.numel()
        safe_pos = pos.clamp(max=max(int(occupied_keys.numel()) - 1, 0))
        found = in_bounds & (occupied_keys[safe_pos] == query_keys)
        neighbor_count = found.view(unique_coords.shape[0], -1).sum(dim=1)
        isolated = int((neighbor_count == 0).sum().item())
        mean_neighbors = float(neighbor_count.to(torch.float32).mean().item()) if neighbor_count.numel() > 0 else 0.0
        return isolated, mean_neighbors

    def _sparsepcgc_hard_stats_single(self, args, pts_3n):
        coords = self._sparsepcgc_quantized_coords(args, pts_3n)
        point_count = int(coords.shape[0])
        if point_count <= 0:
            return {
                "points": 0,
                "active": 0,
                "duplicates": 0,
                "isolated": 0,
                "sparse_density": 0.0,
                "local_density_var": 0.0,
                "mean_neighbors": 0.0,
            }
        unique_coords, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
        active = int(unique_coords.shape[0])
        counts = torch.bincount(inverse, minlength=active).to(torch.float32)
        density_var = float(counts.var(unbiased=False).item()) if counts.numel() > 1 else 0.0
        isolated, mean_neighbors = self._sparsepcgc_isolated_count(unique_coords)
        return {
            "points": point_count,
            "active": active,
            "duplicates": int(point_count - active),
            "isolated": isolated,
            "sparse_density": float(active) / max(float(point_count), 1.0),
            "local_density_var": density_var,
            "mean_neighbors": mean_neighbors,
        }

    def _sparsepcgc_hard_stats_batch(self, args, xyz, final_w=None):
        stats = {
            "points": 0,
            "active": 0,
            "duplicates": 0,
            "isolated": 0,
            "sparse_density": 0.0,
            "local_density_var": 0.0,
            "mean_neighbors": 0.0,
        }
        if xyz is None:
            return stats
        sparse_density_vals = []
        density_vals = []
        neighbor_vals = []
        with torch.no_grad():
            for b in range(xyz.shape[0]):
                pts_b = self._select_actual_points(xyz[b].to(torch.float32), final_w, args, b)
                item = self._sparsepcgc_hard_stats_single(args, pts_b)
                for key in ("points", "active", "duplicates", "isolated"):
                    stats[key] += int(item[key])
                sparse_density_vals.append(float(item["sparse_density"]))
                density_vals.append(float(item["local_density_var"]))
                neighbor_vals.append(float(item["mean_neighbors"]))
        stats["sparse_density"] = float(sum(sparse_density_vals) / max(len(sparse_density_vals), 1))
        stats["local_density_var"] = float(sum(density_vals) / max(len(density_vals), 1))
        stats["mean_neighbors"] = float(sum(neighbor_vals) / max(len(neighbor_vals), 1))
        return stats

    def _sparsepcgc_hard_stats_from_voxel_state(self, args, voxel_state):
        # Actuatorが作ったfinal_voxel_coords/final_voxel_weightsからSparsePCGC用hard統計を作る。
        stats = {
            "points": 0,
            "active": 0,
            "duplicates": 0,
            "isolated": 0,
            "sparse_density": 0.0,
            "local_density_var": 0.0,
            "mean_neighbors": 0.0,
        }

        if not isinstance(voxel_state, dict):
            return stats

        coords = voxel_state.get("final_voxel_coords", None)
        weights = voxel_state.get("final_voxel_weights", None)
        if coords is None or not torch.is_tensor(coords):
            return stats

        if coords.ndim != 3:
            return stats

        # [B, 3, N] に揃える。
        if coords.shape[1] != 3 and coords.shape[-1] == 3:
            coords = coords.permute(0, 2, 1).contiguous()

        if weights is not None and torch.is_tensor(weights):
            if weights.ndim == 3:
                weights = weights.squeeze(1)
            elif weights.ndim != 2:
                weights = weights.reshape(coords.shape[0], -1)
            weights = weights.to(device=coords.device, dtype=torch.float32)
        else:
            weights = None

        density_vals = []
        neighbor_vals = []
        sparse_density_vals = []

        with torch.no_grad():
            for b in range(coords.shape[0]):
                coords_b = coords[b].transpose(0, 1).contiguous().to(torch.long)

                if weights is not None:
                    w_b = weights[b].reshape(-1)
                    if w_b.numel() > coords_b.shape[0]:
                        w_b = w_b[:coords_b.shape[0]]
                    elif w_b.numel() < coords_b.shape[0]:
                        pad = w_b.new_ones(coords_b.shape[0] - w_b.numel())
                        w_b = torch.cat([w_b, pad], dim=0)
                    keep = self._effective_keep_mask_from_weights(w_b, args)
                    coords_b = coords_b[keep]

                point_count = int(coords_b.shape[0])
                if point_count <= 0:
                    sparse_density_vals.append(0.0)
                    density_vals.append(0.0)
                    neighbor_vals.append(0.0)
                    continue

                unique_coords, inverse = torch.unique(coords_b, dim=0, sorted=True, return_inverse=True)
                active = int(unique_coords.shape[0])
                counts = torch.bincount(inverse, minlength=active).to(torch.float32)
                isolated, mean_neighbors = self._sparsepcgc_isolated_count(unique_coords)

                stats["points"] += point_count
                stats["active"] += active
                stats["duplicates"] += int(point_count - active)
                stats["isolated"] += int(isolated)
                sparse_density_vals.append(float(active) / max(float(point_count), 1.0))
                density_vals.append(float(counts.var(unbiased=False).item()) if counts.numel() > 1 else 0.0)
                neighbor_vals.append(float(mean_neighbors))

        stats["sparse_density"] = float(sum(sparse_density_vals) / max(len(sparse_density_vals), 1))
        stats["local_density_var"] = float(sum(density_vals) / max(len(density_vals), 1))
        stats["mean_neighbors"] = float(sum(neighbor_vals) / max(len(neighbor_vals), 1))
        return stats

    def _sparsepcgc_debug_metrics(self, args, gen_xyz, gt_xyz, final_w=None):
        before = self._sparsepcgc_hard_stats_batch(args, gt_xyz, final_w=None)

        voxel_state = self._get_actuator_voxel_state(args, gen_xyz.device)
        if voxel_state is not None:
            after = self._sparsepcgc_hard_stats_from_voxel_state(args, voxel_state)
            uses_actuator_voxel_state = True
        else:
            after = self._sparsepcgc_hard_stats_batch(args, gen_xyz, final_w=final_w)
            uses_actuator_voxel_state = False
        return {
            "sparsepcgc_debug_uses_actuator_voxel_state": bool(uses_actuator_voxel_state),
            "sparsepcgc_before_active_coords": int(before["active"]),
            "sparsepcgc_after_active_coords": int(after["active"]),
            "sparsepcgc_active_coord_delta": int(after["active"] - before["active"]),
            "sparsepcgc_before_occupied_voxels": int(before["active"]),
            "sparsepcgc_after_occupied_voxels": int(after["active"]),
            "sparsepcgc_occupied_voxel_delta": int(after["active"] - before["active"]),
            "sparsepcgc_before_duplicate_points": int(before["duplicates"]),
            "sparsepcgc_after_duplicate_points": int(after["duplicates"]),
            "sparsepcgc_duplicate_delta": int(after["duplicates"] - before["duplicates"]),
            "sparsepcgc_before_isolated_voxels": int(before["isolated"]),
            "sparsepcgc_after_isolated_voxels": int(after["isolated"]),
            "sparsepcgc_isolated_delta": int(after["isolated"] - before["isolated"]),
            "sparsepcgc_before_sparse_density": float(before["sparse_density"]),
            "sparsepcgc_after_sparse_density": float(after["sparse_density"]),
            "sparsepcgc_sparse_density_delta": float(after["sparse_density"] - before["sparse_density"]),
            "sparsepcgc_before_local_density_var": float(before["local_density_var"]),
            "sparsepcgc_after_local_density_var": float(after["local_density_var"]),
            "sparsepcgc_local_density_var_delta": float(after["local_density_var"] - before["local_density_var"]),
            "sparsepcgc_before_mean_neighbors": float(before["mean_neighbors"]),
            "sparsepcgc_after_mean_neighbors": float(after["mean_neighbors"]),
            "sparsepcgc_mean_neighbors_delta": float(after["mean_neighbors"] - before["mean_neighbors"]),
        }

    def _maybe_update_sparsepcgc_debug(self, args, debug, gen_xyz, gt_xyz, final_w=None, codec_name=None):
        if not self._is_sparsepcgc_context(args, codec_name=codec_name):
            return debug
        debug["sparsepcgc_debug_collected"] = False
        debug["sparsepcgc_debug_time"] = 0.0
        debug["sparsepcgc_condition_voxel_size"] = float(getattr(args, "sparsepcgc_voxel_size", getattr(args, "octree_voxel", 1e-3)))
        debug["sparsepcgc_condition_pos_quantscale"] = int(getattr(args, "sparsepcgc_pos_quantscale", 1))
        debug["sparsepcgc_condition_actual_quant_mode"] = "round_xyz_div_voxel_then_div_posquantscale"
        debug["sparsepcgc_condition_proxy_quant_mode"] = "soft_features_gt_bbox_normalized"
        debug["sparsepcgc_condition_rounding"] = "torch_round"
        debug["sparsepcgc_condition_dedup"] = "torch_unique_quantized_coords"
        debug["sparsepcgc_condition_teacher_scope"] = str(getattr(args, "_current_teacher_scope", ""))
        def _bbox_text(tensor, reduce_name):
            values = (tensor.amin(dim=(0, 2)) if reduce_name == "min" else tensor.amax(dim=(0, 2))).detach().float().cpu().tolist()
            return ",".join(f"{float(value):.6g}" for value in values[:3])
        debug["sparsepcgc_condition_gt_bbox_min"] = _bbox_text(gt_xyz[:, :3, :], "min")
        debug["sparsepcgc_condition_gt_bbox_max"] = _bbox_text(gt_xyz[:, :3, :], "max")
        debug["sparsepcgc_condition_gen_bbox_min"] = _bbox_text(gen_xyz[:, :3, :], "min")
        debug["sparsepcgc_condition_gen_bbox_max"] = _bbox_text(gen_xyz[:, :3, :], "max")
        debug["sparsepcgc_condition_local_min_offset"] = "subtree_local_min" if str(getattr(args, "_current_teacher_scope", "")) == "subtree_local" else "global_coords"
        debug["sparsepcgc_condition_warning"] = (
            "subtree_local_teacher_differs_from_actual_full_cloud_context"
            if str(getattr(args, "_current_teacher_scope", "")) == "subtree_local"
            else ""
        )
        # SparsePCGCのhard統計はactive coordinate集合を実際に作るため重い。
        # 学習信号はsoft proxy側から流し、hard統計はログ/診断対象stepだけ計算する。
        if not bool(getattr(args, "_collect_sparsepcgc_debug", False)):
            return debug
        start = time.time()
        debug.update(self._sparsepcgc_debug_metrics(args, gen_xyz=gen_xyz, gt_xyz=gt_xyz, final_w=final_w))
        debug["sparsepcgc_debug_collected"] = True
        debug["sparsepcgc_debug_time"] = float(time.time() - start)
        return debug

    def _sparsepcgc_aux_feature_terms(self, args, gen_xyz, gt_xyz, final_w, x_gen=None, x_ref=None):
        if not bool(getattr(args, "sparsepcgc_aux_loss", True)):
            zero = gen_xyz.new_zeros(())
            return {"loss": zero, "active": zero, "single": zero, "entropy": zero, "density": zero}
        if not self._is_sparsepcgc_context(args):
            zero = gen_xyz.new_zeros(())
            return {"loss": zero, "active": zero, "single": zero, "entropy": zero, "density": zero}
        voxel_state = self._get_actuator_voxel_state(args, gen_xyz.device)
        aux_uses_actuator_voxel_state = voxel_state is not None
        if x_gen is None:
            x_gen = self._build_soft_compression_features(args, gen_xyz, gt_xyz, final_w)
        if x_ref is None:
            x_ref = self._build_soft_compression_features(args, gt_xyz, gt_xyz, None)
        level_dim = 5 * len(self.surrogate_levels)
        q_start = 11 + level_dim
        if x_gen.shape[1] < q_start + 5 or x_ref.shape[1] < q_start + 5:
            zero = gen_xyz.new_zeros(())
            return {"loss": zero, "active": zero, "single": zero, "entropy": zero, "density": zero}
        q_gen = x_gen[:, q_start:q_start + 5].mean(dim=0)
        q_ref = x_ref[:, q_start:q_start + 5].detach().mean(dim=0)
        active_gen = torch.expm1(q_gen[0]).clamp_min(0.0)
        active_ref = torch.expm1(q_ref[0]).clamp_min(1e-6)
        single_gen = torch.expm1(q_gen[1]).clamp_min(0.0)
        single_ref = torch.expm1(q_ref[1]).clamp_min(1e-6)
        entropy_gen = q_gen[2]
        entropy_ref = q_ref[2].abs().clamp_min(1e-6)
        density_gen = q_gen[4]
        density_ref = q_ref[4].abs().clamp_min(1e-6)
        lowprob_gen = (active_gen * entropy_gen.clamp_min(0.0)).clamp_min(0.0)
        lowprob_ref = (active_ref * q_ref[2].clamp_min(0.0)).clamp_min(0.0)
        active_term = 100.0 * (active_gen - active_ref) / active_ref
        single_term = 100.0 * (single_gen - single_ref) / single_ref
        entropy_term = 100.0 * (entropy_gen - q_ref[2]) / entropy_ref
        density_term = 100.0 * (density_gen - q_ref[4]) / density_ref
        clip = float(getattr(args, "sparsepcgc_aux_reward_clip", 50.0))
        if clip > 0.0:
            active_term = active_term.clamp(-clip, clip)
            single_term = single_term.clamp(-clip, clip)
            entropy_term = entropy_term.clamp(-clip, clip)
            density_term = density_term.clamp(-clip, clip)
        loss = (
            float(getattr(args, "sparsepcgc_active_coord_weight", 0.60)) * active_term
            + float(getattr(args, "sparsepcgc_isolated_proxy_weight", 0.25)) * single_term
            + float(getattr(args, "sparsepcgc_entropy_proxy_weight", 0.15)) * entropy_term
            + float(getattr(args, "sparsepcgc_density_proxy_weight", 0.05)) * density_term
        )
        return {
            "loss": loss,
            "active": active_term,
            "single": single_term,
            "entropy": entropy_term,
            "density": density_term,
            "occupancy_pattern_before": active_ref.detach(),
            "occupancy_pattern_after": active_gen.detach(),
            "occupancy_pattern_delta": (active_gen - active_ref).detach(),
            "lowprob_occupancy_count_before": lowprob_ref.detach(),
            "lowprob_occupancy_count_after": lowprob_gen.detach(),
            "lowprob_occupancy_ratio": (lowprob_gen / active_gen.clamp_min(1e-6)).detach(),
            "occupancy_entropy_before": q_ref[2].detach(),
            "occupancy_entropy_after": entropy_gen.detach(),
            "occupancy_entropy_delta": (entropy_gen - q_ref[2]).detach(),
            "occupancy_nll_before": q_ref[2].detach(),
            "occupancy_nll_after": entropy_gen.detach(),
            "occupancy_nll_delta": (entropy_gen - q_ref[2]).detach(),
            "single_child_chain_length_before": single_ref.detach(),
            "single_child_chain_length_after": single_gen.detach(),
            "sibling_occupancy_balance_before": density_ref.detach(),
            "sibling_occupancy_balance_after": density_gen.detach(),
            "octree_pattern_entropy_before": q_ref[2].detach(),
            "octree_pattern_entropy_after": entropy_gen.detach(),
            "octree_pattern_entropy_delta": (entropy_gen - q_ref[2]).detach(),
            "octree_pattern_nll_before": q_ref[2].detach(),
            "octree_pattern_nll_after": entropy_gen.detach(),
            "octree_pattern_nll_delta": (entropy_gen - q_ref[2]).detach(),
            "octree_pattern_lowprob_ratio": (lowprob_gen / active_gen.clamp_min(1e-6)).detach(),
            "occupancy_proxy_definition": "mynet_soft_octree_aux_not_sparsepcgc_candidate_probability",
            "sparsepcgc_aux_uses_actuator_voxel_state": gen_xyz.new_tensor(
                float(aux_uses_actuator_voxel_state)
            ).detach(),
            "sparsepcgc_aux_final_voxel_recomputed_from_pts_out": gen_xyz.new_tensor(
                float(bool(voxel_state.get("final_voxel_recomputed_from_pts_out", True))) if voxel_state is not None else 1.0
            ).detach(),
        }

    def _get_cached_actual_gt(self, cache_key):
        if not self.gt_cache_enabled or not cache_key:
            return None
        cache_entry = self.actual_gt_cache.get(cache_key)
        if cache_entry is None:
            return None
        self.actual_gt_cache.move_to_end(cache_key)
        return dict(cache_entry)

    def _store_cached_actual_gt(self, cache_key, cache_entry):
        if not self.gt_cache_enabled or not cache_key or self.gt_cache_max_entries <= 0:
            return
        self.actual_gt_cache[cache_key] = dict(cache_entry)
        self.actual_gt_cache.move_to_end(cache_key)
        while len(self.actual_gt_cache) > self.gt_cache_max_entries:
            self.actual_gt_cache.popitem(last=False)

    def _get_actual_encoder(self, args):
        backend = self._compression_loss_backend(args)
        compress_key = str(getattr(args, "compress", "OctAttention")).strip().lower().replace("_", "").replace("-", "")
        if backend.startswith("sparsepcgc") or compress_key == "sparsepcgc":
            codec_key = "sparsepcgc"
        elif backend.startswith("gpcc") or compress_key == "gpcc":
            codec_key = "gpcc"
        elif backend.startswith("draco") or compress_key == "draco":
            codec_key = "draco"
        else:
            codec_key = "octattention"
        if self.actual_encoder is None or getattr(self, "actual_encoder_codec_key", None) != codec_key:
            old_encoder = self.actual_encoder
            if old_encoder is not None and hasattr(old_encoder, "close"):
                old_encoder.close()
            self.actual_encoder = build_actual_encoder(args, writer=self.writer)
            self.actual_encoder_codec_key = codec_key
            if hasattr(self, "actual_gt_cache"):
                self.actual_gt_cache.clear()
            if hasattr(self, "surrogate_target_cache"):
                self.surrogate_target_cache.clear()
            if hasattr(self, "last_surrogate_target_entry"):
                self.last_surrogate_target_entry = None
        return self.actual_encoder

    def _reset_actual_encoder_after_error(self):
        old_encoder = getattr(self, "actual_encoder", None)
        if old_encoder is not None and hasattr(old_encoder, "close"):
            try:
                old_encoder.close()
            except Exception:
                pass
        self.actual_encoder = None
        self.actual_encoder_codec_key = None
        for name in ("actual_gt_cache", "surrogate_target_cache"):
            cache = getattr(self, name, None)
            if hasattr(cache, "clear"):
                cache.clear()
        if hasattr(self, "last_surrogate_target_entry"):
            self.last_surrogate_target_entry = None

    @staticmethod
    def _effective_keep_mask_from_weights(weights, args):
        weights = torch.nan_to_num(weights.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        point_count = int(weights.numel())
        if point_count <= 0:
            return weights.new_zeros((0,), dtype=torch.bool)
        threshold = float(
            getattr(args, "operation_count_drop_threshold", getattr(args, "test_drop_threshold", 0.5))
        )
        keep_mask = weights >= threshold
        keep_count = int(keep_mask.sum().item())
        if keep_count <= 0 or keep_count >= point_count:
            expected_keep = int(round(float(weights.sum().item())))
            expected_keep = min(max(expected_keep, 1), point_count)
            if 0 < expected_keep < point_count:
                topk_idx = torch.topk(weights, k=expected_keep, largest=True, sorted=False).indices
                keep_mask = torch.zeros_like(weights, dtype=torch.bool)
                keep_mask.scatter_(0, topk_idx, True)
        if not bool(keep_mask.any().item()):
            keep_mask[torch.argmax(weights)] = True
        return keep_mask

    def _select_actual_points(self, xyz_3n, final_w, args, batch_idx):
        if final_w is None:
            return xyz_3n
        if final_w.ndim == 3:
            weights = final_w[batch_idx].squeeze(0)
        elif final_w.ndim == 2:
            weights = final_w[batch_idx]
        else:
            weights = final_w.reshape(-1)
        point_count = int(xyz_3n.shape[-1])
        weights = weights.to(device=xyz_3n.device, dtype=torch.float32).reshape(-1)
        if weights.numel() > point_count:
            weights = weights[:point_count]
        elif weights.numel() < point_count:
            pad = weights.new_ones((point_count - int(weights.numel()),))
            weights = torch.cat([weights, pad], dim=0)
        keep_mask = self._effective_keep_mask_from_weights(weights, args)
        return xyz_3n[:, keep_mask]

    def _encode_actual_batch(self, args, xyz, final_w=None):
        encoder = self._get_actual_encoder(args)
        stats_list = []
        for b in range(xyz.shape[0]):
            pts_b = self._select_actual_points(xyz[b].to(torch.float32), final_w, args, b)
            # actual codec評価は教師/ログ用で微分しない。ここをinference_modeにして、
            # 外部codecやhard quantize周辺に不要な計算グラフを残さない。
            with torch.inference_mode():
                stats = dict(encoder.encode_bits(pts_b))
                stats = self._attach_octree_aux_stats(args, pts_b, stats)
            stats_list.append(stats)
        total_bit = sum(s["bit"] for s in stats_list)
        total_single = sum(s["single"] for s in stats_list)
        total_node = sum(s["node"] for s in stats_list)
        total_points = sum(s["point_count"] for s in stats_list)
        total_octree_single = sum(float(s.get("octree_single", s.get("single", 0.0))) for s in stats_list)
        total_octree_node = sum(float(s.get("octree_node", s.get("node", 0.0))) for s in stats_list)
        total_encode_time = sum(float(s.get("encode_time", 0.0)) for s in stats_list)
        total_unique_coord = sum(int(s.get("unique_coord_count", s.get("point_count", 0))) for s in stats_list)
        max_octree_depth = max((int(s.get("octree_depth", 0)) for s in stats_list), default=0)
        total_leaf = sum(int(s.get("octree_leaf_count", s.get("point_count", 0))) for s in stats_list)
        total_occupancy_patterns = sum(int(s.get("octree_occupancy_pattern_count", 0)) for s in stats_list)
        total_lowprob_occupancy = sum(float(s.get("octree_lowprob_occupancy_count", 0.0)) for s in stats_list)
        octree_node_denom = max(float(total_octree_node), 1.0)
        sparse_candidate_count = sum(int(s.get("sparsepcgc_candidate_count", 0)) for s in stats_list)
        sparse_occupied_count = sum(int(s.get("sparsepcgc_occupied_candidate_count", 0)) for s in stats_list)
        sparse_estimated_bits = sum(float(s.get("sparsepcgc_estimated_occupancy_bits", 0.0)) for s in stats_list)
        sparse_prob_true_low_count = sum(float(s.get("sparsepcgc_prob_true_low_count", 0.0)) for s in stats_list)
        sparse_occupied_low_count = sum(float(s.get("sparsepcgc_occupied_low_prob_count", 0.0)) for s in stats_list)
        sparse_debug_available = any(bool(s.get("sparsepcgc_occupancy_debug_available", False)) for s in stats_list)
        sparse_threshold_values = [
            float(s.get("sparsepcgc_low_prob_threshold", 0.0))
            for s in stats_list
            if "sparsepcgc_low_prob_threshold" in s
        ]
        exact_candidate_count = sum(int(s.get("sparsepcgc_exact_candidate_count", 0)) for s in stats_list)
        exact_occupied_count = sum(int(s.get("sparsepcgc_exact_occupied_count", 0)) for s in stats_list)
        exact_estimated_bits = sum(float(s.get("sparsepcgc_exact_estimated_bits", 0.0)) for s in stats_list)
        exact_bce_bits = sum(float(s.get("sparsepcgc_exact_bce_bits", 0.0)) for s in stats_list)
        exact_actual_bits = sum(float(s.get("sparsepcgc_exact_actual_bitstream_bits", 0.0)) for s in stats_list)
        exact_impl_bits = sum(float(s.get("exact_bits_impl", 0.0)) for s in stats_list)
        exact_sparsepcgc_bits = sum(float(s.get("exact_bits_sparsepcgc_estimate_bitrate", 0.0)) for s in stats_list)
        exact_enabled = any("sparsepcgc_exact_estimated_bits" in s for s in stats_list)
        exact_last = next((s for s in reversed(stats_list) if "sparsepcgc_exact_estimated_bits" in s), {})

        def _weighted_octree_stat(key):
            weighted = 0.0
            for item in stats_list:
                weight = float(item.get("octree_node", item.get("node", 0.0)))
                weighted += float(item.get(key, 0.0)) * weight
            return weighted / octree_node_denom

        occupancy_entropy = _weighted_octree_stat("octree_occupancy_entropy")
        occupancy_nll = _weighted_octree_stat("octree_occupancy_nll")
        lowprob_ratio = total_lowprob_occupancy / octree_node_denom
        occupancy_predictability = max(0.0, min(1.0, 1.0 - occupancy_entropy / 8.0))

        def _weighted_sparsepcgc_stat(key):
            denom = max(float(sparse_candidate_count), 1.0)
            weighted = 0.0
            for item in stats_list:
                weight = float(item.get("sparsepcgc_candidate_count", 0.0))
                weighted += float(item.get(key, 0.0)) * weight
            return weighted / denom

        def _weighted_exact_stat(key):
            denom = max(float(exact_candidate_count), 1.0)
            weighted = 0.0
            for item in stats_list:
                weight = float(item.get("sparsepcgc_exact_candidate_count", 0.0))
                weighted += float(item.get(key, 0.0)) * weight
            return weighted / denom

        result = {
            "bit": float(total_bit),
            "bpp": float(total_bit) / max(float(total_points), 1.0),
            "bpn": float(total_bit) / max(float(total_node), 1.0),
            "single": float(total_single),
            "node": float(total_node),
            "octree_single": float(total_octree_single),
            "octree_node": float(total_octree_node),
            "octree_depth": int(max_octree_depth),
            "octree_leaf_count": int(total_leaf),
            "encode_time": float(total_encode_time),
            "unique_coord_count": int(total_unique_coord),
            "octree_occupancy_pattern_count": int(total_occupancy_patterns),
            "octree_occupancy_entropy": float(occupancy_entropy),
            "octree_occupancy_nll": float(occupancy_nll),
            "octree_lowprob_occupancy_count": float(total_lowprob_occupancy),
            "octree_lowprob_occupancy_ratio": float(lowprob_ratio),
            "octree_occupancy_predictability": float(occupancy_predictability),
            "sparsepcgc_occupancy_debug_available": bool(sparse_debug_available),
            "sparsepcgc_candidate_count": int(sparse_candidate_count),
            "sparsepcgc_occupied_candidate_count": int(sparse_occupied_count),
            "sparsepcgc_actual_occupancy_label_ratio": float(sparse_occupied_count) / max(float(sparse_candidate_count), 1.0),
            "sparsepcgc_pred_prob_entropy": float(_weighted_sparsepcgc_stat("sparsepcgc_pred_prob_entropy")),
            "sparsepcgc_pred_occupancy_nll": float(_weighted_sparsepcgc_stat("sparsepcgc_pred_occupancy_nll")),
            "sparsepcgc_estimated_occupancy_bits": float(sparse_estimated_bits),
            "sparsepcgc_estimated_occupancy_bpp": float(sparse_estimated_bits) / max(float(total_points), 1.0),
            "sparsepcgc_prob_true_mean": float(_weighted_sparsepcgc_stat("sparsepcgc_prob_true_mean")),
            "sparsepcgc_prob_true_low_count": float(sparse_prob_true_low_count),
            "sparsepcgc_prob_true_low_ratio": float(sparse_prob_true_low_count) / max(float(sparse_candidate_count), 1.0),
            "sparsepcgc_occupied_low_prob_count": float(sparse_occupied_low_count),
            "sparsepcgc_occupied_low_prob_ratio": float(sparse_occupied_low_count) / max(float(sparse_occupied_count), 1.0),
            "sparsepcgc_low_prob_threshold": float(sparse_threshold_values[-1]) if sparse_threshold_values else float(getattr(args, "sparsepcgc_occupancy_low_prob_threshold", 0.1)),
            "point_count": int(total_points),
            "codec": str(getattr(encoder, "codec_name", "octattention")),
            "per_batch": stats_list,
        }
        if exact_enabled:
            exact_bits_abs_diff = abs(float(exact_impl_bits) - float(exact_sparsepcgc_bits))
            exact_bits_rel_diff = exact_bits_abs_diff / max(abs(float(exact_sparsepcgc_bits)), 1e-6)
            exact_actual_gap = float(exact_estimated_bits) - float(exact_actual_bits)
            exact_actual_gap_percent = exact_actual_gap / max(float(exact_actual_bits), 1e-6) * 100.0
            result.update(
                {
                    "sparsepcgc_exact_candidate_count": int(exact_candidate_count),
                    "sparsepcgc_exact_occupied_count": int(exact_occupied_count),
                    "sparsepcgc_exact_occupancy_label_ratio": float(exact_occupied_count) / max(float(exact_candidate_count), 1.0),
                    "sparsepcgc_exact_prob_mean": float(_weighted_exact_stat("sparsepcgc_exact_prob_mean")),
                    "sparsepcgc_exact_prob_entropy": float(_weighted_exact_stat("sparsepcgc_exact_prob_entropy")),
                    "sparsepcgc_exact_prob_true_mean": float(_weighted_exact_stat("sparsepcgc_exact_prob_true_mean")),
                    "sparsepcgc_exact_occupancy_nll": float(_weighted_exact_stat("sparsepcgc_exact_occupancy_nll")),
                    "sparsepcgc_exact_estimated_bits": float(exact_estimated_bits),
                    "sparsepcgc_exact_estimated_bpp": float(exact_estimated_bits) / max(float(total_points), 1.0),
                    "sparsepcgc_exact_low_prob_ratio": float(_weighted_exact_stat("sparsepcgc_exact_low_prob_ratio")),
                    "sparsepcgc_exact_bce_bits": float(exact_bce_bits),
                    "sparsepcgc_exact_actual_bitstream_bits": float(exact_actual_bits),
                    "sparsepcgc_exact_teacher_mode": str(exact_last.get("sparsepcgc_exact_teacher_mode", "")),
                    "exact_teacher_uses_full_context": bool(exact_last.get("exact_teacher_uses_full_context", False)),
                    "exact_teacher_fallback_reason": str(exact_last.get("exact_teacher_fallback_reason", "")),
                    "exact_teacher_candidate_source": str(exact_last.get("exact_teacher_candidate_source", "")),
                    "exact_teacher_label_source": str(exact_last.get("exact_teacher_label_source", "")),
                    "exact_bits_impl": float(exact_impl_bits),
                    "exact_bits_sparsepcgc_estimate_bitrate": float(exact_sparsepcgc_bits),
                    "exact_bits_abs_diff": float(exact_bits_abs_diff),
                    "exact_bits_rel_diff": float(exact_bits_rel_diff),
                    "exact_bits_match": bool(exact_bits_abs_diff <= max(1e-5, abs(float(exact_sparsepcgc_bits)) * 1e-6)),
                    "exact_estimated_vs_actual_bit_gap": float(exact_actual_gap),
                    "exact_estimated_vs_actual_bit_gap_percent": float(exact_actual_gap_percent),
                }
            )
        return result

    def _actual_octree_stat_qs(self, args, codec_name):
        codec_key = str(codec_name).strip().lower()
        if codec_key == "sparsepcgc":
            return max(float(getattr(args, "sparsepcgc_voxel_size", 1.0)), 1e-9)
        if codec_key == "gpcc":
            return max(float(getattr(args, "gpcc_effective_qs", getattr(args, "qs", 1.0))), 1e-9)
        if codec_key == "draco":
            return max(float(getattr(args, "draco_effective_qs", getattr(args, "qs", 1.0))), 1e-9)
        return max(float(getattr(args, "qs", 1.0)), 1e-9)

    def _attach_octree_aux_stats(self, args, pts_3n, stats):
        codec_name = str(stats.get("codec", getattr(self, "actual_encoder_codec_key", "octattention"))).strip().lower()
        need_aux = (
            bool(getattr(args, "compression_octree_stat_force", True))
            or float(stats.get("node", 0.0)) <= 0.0
            or float(stats.get("single", 0.0)) <= 0.0
            or codec_name in {"sparsepcgc", "gpcc", "draco"}
        )
        if not need_aux:
            stats.setdefault("octree_node", float(stats.get("node", 0.0)))
            stats.setdefault("octree_single", float(stats.get("single", 0.0)))
            stats.setdefault("octree_depth", 0)
            stats.setdefault("octree_leaf_count", int(stats.get("point_count", 0)))
            return stats
        aux = hard_octree_occupancy_stats(
            pts_3n,
            qs=self._actual_octree_stat_qs(args, codec_name),
            max_depth=int(getattr(args, "compression_octree_stat_depth", 0)),
            quant_mode="sparsepcgc" if codec_name == "sparsepcgc" else "round",
            pos_quantscale=int(getattr(args, "sparsepcgc_pos_quantscale", 1)) if codec_name == "sparsepcgc" else 1,
        )
        aux_node = float(aux["node_count"])
        aux_single = float(aux["single_child_count"])
        stats["octree_node"] = aux_node
        stats["octree_single"] = aux_single
        stats["octree_depth"] = int(aux["max_depth"])
        stats["octree_leaf_count"] = int(aux["leaf_count"])
        stats["octree_single_ratio"] = float(aux["single_ratio"])
        stats["octree_mean_children"] = float(aux["mean_children"])
        stats["octree_occupancy_pattern_count"] = int(aux["occupancy_pattern_count"])
        stats["octree_occupancy_entropy"] = float(aux["occupancy_entropy"])
        stats["octree_occupancy_nll"] = float(aux["occupancy_nll"])
        stats["octree_lowprob_occupancy_count"] = float(aux["lowprob_occupancy_count"])
        stats["octree_lowprob_occupancy_ratio"] = float(aux["lowprob_occupancy_ratio"])
        stats["octree_occupancy_predictability"] = float(aux["occupancy_predictability"])
        if float(stats.get("node", 0.0)) <= 0.0 or codec_name in {"sparsepcgc", "gpcc", "draco"}:
            stats["node"] = aux_node
        if float(stats.get("single", 0.0)) <= 0.0 or codec_name in {"sparsepcgc", "gpcc", "draco"}:
            stats["single"] = aux_single
        stats["bpn"] = float(stats.get("bit", 0.0)) / max(float(stats.get("node", 0.0)), 1.0)
        return stats

    @staticmethod
    def _actual_occupancy_debug_from_stats(before_stats, after_stats):
        def _float(stats, key, default=0.0):
            try:
                return float(stats.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        def _int(stats, key, default=0):
            try:
                return int(stats.get(key, default))
            except (TypeError, ValueError):
                return int(default)

        before_pattern = _int(before_stats, "octree_occupancy_pattern_count")
        after_pattern = _int(after_stats, "octree_occupancy_pattern_count")
        before_entropy = _float(before_stats, "octree_occupancy_entropy")
        after_entropy = _float(after_stats, "octree_occupancy_entropy")
        before_nll = _float(before_stats, "octree_occupancy_nll", before_entropy)
        after_nll = _float(after_stats, "octree_occupancy_nll", after_entropy)
        before_lowprob_count = _float(before_stats, "octree_lowprob_occupancy_count")
        after_lowprob_count = _float(after_stats, "octree_lowprob_occupancy_count")
        before_lowprob_ratio = _float(before_stats, "octree_lowprob_occupancy_ratio")
        after_lowprob_ratio = _float(after_stats, "octree_lowprob_occupancy_ratio")
        before_predictability = _float(before_stats, "octree_occupancy_predictability")
        after_predictability = _float(after_stats, "octree_occupancy_predictability")
        debug = {
            "occupancy_proxy_definition": "myNet hard/proxy octree occupancy statistics, not SparsePCGC candidate occupancy",
            "actual_occupancy_definition": "actual codec input hard-octree occupancy statistics from quantized coordinates",
            "predicted_occupancy_definition": "SparsePCGC exact values use sigmoid(out_cls.F) only in sparsepcgc_exact_* logs",
            "actual_occupancy_pattern_before": before_pattern,
            "actual_occupancy_pattern_after": after_pattern,
            "actual_occupancy_pattern_delta": after_pattern - before_pattern,
            "actual_occupancy_entropy_before": before_entropy,
            "actual_occupancy_entropy_after": after_entropy,
            "actual_occupancy_entropy_delta": after_entropy - before_entropy,
            "actual_occupancy_nll_before": before_nll,
            "actual_occupancy_nll_after": after_nll,
            "actual_occupancy_nll_delta": after_nll - before_nll,
            "actual_lowprob_occupancy_count_before": before_lowprob_count,
            "actual_lowprob_occupancy_count_after": after_lowprob_count,
            "actual_lowprob_occupancy_count_delta": after_lowprob_count - before_lowprob_count,
            "actual_lowprob_occupancy_ratio_before": before_lowprob_ratio,
            "actual_lowprob_occupancy_ratio_after": after_lowprob_ratio,
            "actual_lowprob_occupancy_ratio_delta": after_lowprob_ratio - before_lowprob_ratio,
            "actual_occupancy_predictability_before": before_predictability,
            "actual_occupancy_predictability_after": after_predictability,
            "actual_occupancy_predictability_delta": after_predictability - before_predictability,
        }
        if bool(before_stats.get("sparsepcgc_occupancy_debug_available", False)) or bool(
            after_stats.get("sparsepcgc_occupancy_debug_available", False)
        ):
            sparse_float_keys = [
                "sparsepcgc_actual_occupancy_label_ratio",
                "sparsepcgc_pred_prob_entropy",
                "sparsepcgc_pred_occupancy_nll",
                "sparsepcgc_estimated_occupancy_bits",
                "sparsepcgc_estimated_occupancy_bpp",
                "sparsepcgc_prob_true_mean",
                "sparsepcgc_prob_true_low_count",
                "sparsepcgc_prob_true_low_ratio",
                "sparsepcgc_occupied_low_prob_count",
                "sparsepcgc_occupied_low_prob_ratio",
            ]
            sparse_int_keys = [
                "sparsepcgc_candidate_count",
                "sparsepcgc_occupied_candidate_count",
            ]
            for key in sparse_float_keys:
                before_value = _float(before_stats, key, float("nan"))
                after_value = _float(after_stats, key, float("nan"))
                debug[f"{key}_before"] = before_value
                debug[f"{key}_after"] = after_value
                debug[f"{key}_delta"] = after_value - before_value
                debug[key] = after_value
            for key in sparse_int_keys:
                before_value = _int(before_stats, key, 0)
                after_value = _int(after_stats, key, 0)
                debug[f"{key}_before"] = before_value
                debug[f"{key}_after"] = after_value
                debug[f"{key}_delta"] = after_value - before_value
                debug[key] = after_value
            debug["sparsepcgc_occupancy_debug_available"] = True
            debug["sparsepcgc_low_prob_threshold"] = _float(
                after_stats,
                "sparsepcgc_low_prob_threshold",
                _float(before_stats, "sparsepcgc_low_prob_threshold", 0.1),
            )
        if "sparsepcgc_exact_estimated_bits" in before_stats or "sparsepcgc_exact_estimated_bits" in after_stats:
            exact_float_keys = [
                "sparsepcgc_exact_occupancy_label_ratio",
                "sparsepcgc_exact_prob_mean",
                "sparsepcgc_exact_prob_entropy",
                "sparsepcgc_exact_prob_true_mean",
                "sparsepcgc_exact_occupancy_nll",
                "sparsepcgc_exact_estimated_bits",
                "sparsepcgc_exact_estimated_bpp",
                "sparsepcgc_exact_low_prob_ratio",
                "sparsepcgc_exact_bce_bits",
                "sparsepcgc_exact_actual_bitstream_bits",
                "exact_bits_impl",
                "exact_bits_sparsepcgc_estimate_bitrate",
                "exact_bits_abs_diff",
                "exact_bits_rel_diff",
                "exact_estimated_vs_actual_bit_gap",
                "exact_estimated_vs_actual_bit_gap_percent",
            ]
            exact_int_keys = [
                "sparsepcgc_exact_candidate_count",
                "sparsepcgc_exact_occupied_count",
            ]
            for key in exact_float_keys:
                before_value = _float(before_stats, key, float("nan"))
                after_value = _float(after_stats, key, float("nan"))
                debug[f"{key}_before"] = before_value
                debug[f"{key}_after"] = after_value
                debug[f"{key}_delta"] = after_value - before_value
                debug[key] = after_value
            for key in exact_int_keys:
                before_value = _int(before_stats, key, 0)
                after_value = _int(after_stats, key, 0)
                debug[f"{key}_before"] = before_value
                debug[f"{key}_after"] = after_value
                debug[f"{key}_delta"] = after_value - before_value
                debug[key] = after_value
            debug["exact_bits_match"] = bool(after_stats.get("exact_bits_match", False))
            for key in (
                "sparsepcgc_exact_teacher_mode",
                "exact_teacher_uses_full_context",
                "exact_teacher_fallback_reason",
                "exact_teacher_candidate_source",
                "exact_teacher_label_source",
            ):
                debug[key] = after_stats.get(key, before_stats.get(key, ""))
            debug["sparsepcgc_exact_occupancy_nll_delta"] = debug.get("sparsepcgc_exact_occupancy_nll_delta", float("nan"))
            debug["sparsepcgc_exact_estimated_bits_delta"] = debug.get("sparsepcgc_exact_estimated_bits_delta", float("nan"))
            debug["sparsepcgc_exact_bpp_delta"] = debug.get("sparsepcgc_exact_estimated_bpp_delta", float("nan"))
        return debug

    def _log_compression_grad_probe(self, args, label, L_com, gen_xyz):
        if not bool(getattr(args, "compression_grad_probe", True)):
            return
        every = max(int(getattr(args, "compression_grad_probe_every", 1)), 1)
        self._compression_grad_probe_count += 1
        if self._compression_grad_probe_count % every != 0:
            return

        requires_grad = bool(torch.is_tensor(L_com) and L_com.requires_grad)
        grad_fn = type(L_com.grad_fn).__name__ if requires_grad and L_com.grad_fn is not None else "None"
        grad_norm = None
        grad_ok = False
        err = None

        if requires_grad:
            try:
                grad = torch.autograd.grad(
                    L_com,
                    gen_xyz,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if grad is not None:
                    grad_norm = float(grad.detach().norm().cpu())
                    grad_ok = grad_norm > 0.0 and np.isfinite(grad_norm)
            except Exception as exc:
                err = str(exc)

        msg = (
            f"[GradCheck][L_com:{label}] "
            f"requires_grad={requires_grad}, grad_fn={grad_fn}, "
            f"grad_to_gen_xyz={'OK' if grad_ok else 'NG'}, "
            f"grad_norm={grad_norm if grad_norm is not None else 'None'}"
        )
        if err is not None:
            msg += f", err={err}"
        if self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(msg)

    def _actual_codec_disabled_for_train(self, args):
        return (
            bool(getattr(args, "disable_actual_codec_during_train", False))
            and str(getattr(args, "trainORtest", "train")).strip().lower() == "train"
            and not bool(getattr(args, "_surrogate_pretrain_active", False))
        )

    def _get_compression_loss_actual_codec(
        self,
        args,
        gen_xyz,
        gt_xyz,
        final_w,
        cache_key=None,
        use_proxy_surrogate=False,
        actual_gen_xyz=None,
        subtree_tree=None,
        full_octree_context=None,
        octree_input_mode="auto",
    ):
        actual_xyz = gen_xyz if actual_gen_xyz is None else actual_gen_xyz
        cached_gt = self._get_cached_actual_gt(cache_key)
        if cached_gt is None:
            cached_gt = self._encode_actual_batch(args, gt_xyz)
            self._store_cached_actual_gt(cache_key, cached_gt)

        # actual codec評価は評価指標なので、train用の量子化ノイズを入れないclean編集点群を使う。
        stats_gen = self._encode_actual_batch(args, actual_xyz, final_w=final_w)
        codec_name = str(stats_gen.get("codec", cached_gt.get("codec", "octattention"))).strip().lower()
        backend_label = f"{codec_name}_actual_ste" if use_proxy_surrogate else f"{codec_name}_actual"
        gt_bit = float(cached_gt["bit"])
        gen_bit = float(stats_gen["bit"])
        loss_bit_ratio = self._relative_ratio(gen_bit, gt_bit)
        loss_bit_percent = 100.0 * loss_bit_ratio

        L_com_hard = gen_xyz.new_tensor(loss_bit_percent)
        L_com = L_com_hard

        proxy_debug = None
        if use_proxy_surrogate:
            proxy_L_com, proxy_loss_bit, proxy_loss_single, proxy_loss_nodes, _, _ = self._get_compression_loss_proxy(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                run_grad_probe=False,
                actual_gen_xyz=actual_xyz,
                subtree_tree=subtree_tree,
                full_octree_context=full_octree_context,
                octree_input_mode=octree_input_mode,
            )
            proxy_terms = dict(getattr(self, "last_compression_terms", {}) or {})
            proxy_bit = proxy_terms.get("bit")
            surrogate_loss = proxy_bit if torch.is_tensor(proxy_bit) else None
            L_com = self._compose_discrete_loss(L_com_hard, surrogate_loss, args)
            proxy_debug = {
                "L_com": self._scalar(proxy_L_com),
                "loss_bit": self._scalar(proxy_loss_bit),
                "loss_single": self._scalar(proxy_loss_single),
                "loss_nodes": self._scalar(proxy_loss_nodes),
            }

        loss_bit = gen_xyz.new_tensor(loss_bit_percent)
        loss_single = gen_xyz.new_tensor(
            self._relative_percent(float(stats_gen["single"]), float(cached_gt["single"]), ref_min=1.0)
        )
        loss_nodes = gen_xyz.new_tensor(
            self._relative_percent(float(stats_gen["node"]), float(cached_gt["node"]), ref_min=1.0)
        )
        exact_available = "sparsepcgc_exact_estimated_bits" in stats_gen and "sparsepcgc_exact_estimated_bits" in cached_gt
        exact_nll_delta = float("nan")
        exact_bits_delta = float("nan")
        exact_bpp_delta = float("nan")
        exact_loss = gen_xyz.new_zeros(())
        if exact_available:
            exact_nll_delta = float(stats_gen.get("sparsepcgc_exact_occupancy_nll", 0.0)) - float(
                cached_gt.get("sparsepcgc_exact_occupancy_nll", 0.0)
            )
            exact_bits_delta = self._relative_percent(
                float(stats_gen.get("sparsepcgc_exact_estimated_bits", 0.0)),
                float(cached_gt.get("sparsepcgc_exact_estimated_bits", 0.0)),
                ref_min=1.0,
            )
            exact_bpp_delta = self._relative_percent(
                float(stats_gen.get("sparsepcgc_exact_estimated_bpp", 0.0)),
                float(cached_gt.get("sparsepcgc_exact_estimated_bpp", 0.0)),
                ref_min=1e-9,
            )
        if exact_available and bool(getattr(args, "enable_sparsepcgc_exact_occupancy_loss", False)):
            exact_loss = exact_loss + gen_xyz.new_tensor(
                float(getattr(args, "sparsepcgc_exact_occupancy_loss_weight", 0.0)) * exact_nll_delta
            )
            exact_loss = exact_loss + gen_xyz.new_tensor(
                float(getattr(args, "sparsepcgc_exact_bits_loss_weight", 0.0)) * exact_bits_delta
            )
            L_com = L_com + exact_loss
        self._store_compression_terms(
            main=L_com,
            bit=L_com_hard,
            single=gen_xyz.new_zeros(()),
            node=gen_xyz.new_zeros(()),
            bpn=gen_xyz.new_zeros(()),
            objective=L_com,
            sparsepcgc_exact=exact_loss,
            backend=backend_label,
        )
        self.last_compression_debug = {
            "metric": "actual_total_bit_percent",
            "teacher_codec": codec_name,
            "total_bit": loss_bit_percent,
            "bpp": self._relative_percent(float(stats_gen["bpp"]), float(cached_gt["bpp"])),
            "gt_points": int(cached_gt["point_count"]),
            "gen_points": int(stats_gen["point_count"]),
            "gt_unique_coord_count": int(cached_gt.get("unique_coord_count", cached_gt.get("point_count", 0))),
            "gen_unique_coord_count": int(stats_gen.get("unique_coord_count", stats_gen.get("point_count", 0))),
            "gt_actual_bit": gt_bit,
            "gen_actual_bit": gen_bit,
            "actual_total_bit_percent": loss_bit_percent,
            "actual_value_is_fresh": True,
            "actual_value_source": "actual_codec",
            "rate_proxy_before": gt_bit,
            "rate_proxy_after": gen_bit,
            "rate_proxy_delta": loss_bit_percent,
            "node_delta": float(stats_gen["node"]) - float(cached_gt["node"]),
            "single_delta": float(stats_gen["single"]) - float(cached_gt["single"]),
            "proxy_surrogate": proxy_debug,
            "sparsepcgc_exact_occupancy_nll_delta": exact_nll_delta,
            "sparsepcgc_exact_estimated_bits_delta": exact_bits_delta,
            "sparsepcgc_exact_bpp_delta": exact_bpp_delta,
            "sparsepcgc_exact_loss_candidate": self._scalar(exact_loss),
            "sparsepcgc_exact_loss_enabled": bool(
                exact_available and getattr(args, "enable_sparsepcgc_exact_occupancy_loss", False)
            ),
            "actuator_voxel_state_available": bool(
                self._get_actuator_voxel_state(args, gen_xyz.device) is not None
            ),
        }
        if codec_name == "sparsepcgc":
            proxy_bit_percent = float(proxy_debug["loss_bit"]) if proxy_debug is not None else float("nan")
            self.last_compression_debug.update(
                {
                    "actual_sparsepcgc_bit": gen_bit,
                    "actual_sparsepcgc_gt_bit": gt_bit,
                    "actual_sparsepcgc_bit_delta": gen_bit - gt_bit,
                    "proxy_sparsepcgc_bit": proxy_bit_percent,
                    "proxy_sparsepcgc_bit_percent": proxy_bit_percent,
                    "proxy_actual_bit_gap": float("nan"),
                    "proxy_actual_bit_gap_percent": proxy_bit_percent - loss_bit_percent if proxy_debug is not None else float("nan"),
                    "estimated_occupancy_bits": float(stats_gen.get("sparsepcgc_estimated_occupancy_bits", float("nan"))),
                    "mean_prob_true": float(stats_gen.get("sparsepcgc_prob_true_mean", float("nan"))),
                    "low_prob_true_count": float(stats_gen.get("sparsepcgc_prob_true_low_count", float("nan"))),
                    "low_prob_true_ratio": float(stats_gen.get("sparsepcgc_prob_true_low_ratio", float("nan"))),
                }
            )
        self.last_compression_debug.update(self._actual_occupancy_debug_from_stats(cached_gt, stats_gen))
        self._maybe_update_sparsepcgc_debug(
            args,
            self.last_compression_debug,
            gen_xyz=actual_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
            codec_name=codec_name,
        )

        if self._should_verbose_step(args):
            surrogate_msg = ""
            if proxy_debug is not None:
                surrogate_msg = (
                    f", proxy_surrogate_L:{proxy_debug['L_com']:.4f}, "
                    f"proxy_surrogate_bit:{proxy_debug['loss_bit']:.4f}"
                )
            self.writer.write(
                f"L_com(actual {codec_name}):{self._scalar(L_com):.6f}->"
                f"bit:{gt_bit:.1f}->{gen_bit:.1f}, "
                f"rel:{loss_bit_percent:.4f}%"
                f"{surrogate_msg}"
            )

        self._log_compression_grad_probe(args, backend_label, L_com, gen_xyz)

        stats_gt = {
            "bit": gt_bit,
            "bpp": float(cached_gt["bpp"]),
            "bpn": float(cached_gt["bpn"]),
            "single": float(cached_gt["single"]),
            "node": float(cached_gt["node"]),
        }
        return L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt
    
    def _get_actuator_voxel_state(self, args, device=None):
        # Networkが保存したActuator後のVoxel状態を安全に取得する。
        voxel_state = getattr(args, "_last_actuator_voxel_state", None)
        if not isinstance(voxel_state, dict):
            return None

        final_coords = voxel_state.get("final_voxel_coords", None)
        if final_coords is None or not torch.is_tensor(final_coords):
            return None

        out = dict(voxel_state)
        if device is not None:
            for key in ("initial_voxel_coords", "final_voxel_coords", "final_voxel_weights", "voxel_step", "voxel_offset"):
                value = out.get(key, None)
                if torch.is_tensor(value):
                    out[key] = value.to(device=device, non_blocking=True)
        return out

    def get_compression_loss(
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
        self._store_compression_terms()
        requested_mode = str(octree_input_mode or "auto").strip().lower()
        if requested_mode == "prebuilt_subtree_tree" and subtree_tree is None:
            raise ValueError("octree_input_mode=prebuilt_subtree_tree requires subtree_tree in get_compression_loss().")
        backend = self._compression_loss_backend(args)
        surrogate_backends = {"octattention_surrogate", "sparsepcgc_surrogate", "gpcc_surrogate", "draco_surrogate", "surrogate", "soft_surrogate"}
        if backend != "proxy" and self._actual_codec_disabled_for_train(args):
            has_surrogate_teacher = bool(getattr(self, "last_surrogate_target_entry", None) is not None)
            has_surrogate_teacher = has_surrogate_teacher or bool(getattr(self, "surrogate_replay", []))
            if backend in surrogate_backends and has_surrogate_teacher:
                L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt = self._get_compression_loss_surrogate(
                    args,
                    gen_xyz=gen_xyz,
                    gt_xyz=gt_xyz,
                    final_w=final_w,
                    cache_key=cache_key,
                    refresh_actual_gen=False,
                    actual_gen_xyz=actual_gen_xyz,
                    subtree_tree=subtree_tree,
                    full_octree_context=full_octree_context,
                    octree_input_mode=octree_input_mode,
                )
            else:
                L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt = self._get_compression_loss_proxy(
                    args,
                    gen_xyz=gen_xyz,
                    gt_xyz=gt_xyz,
                    final_w=final_w,
                    cache_key=cache_key,
                    run_grad_probe=True,
                    actual_gen_xyz=actual_gen_xyz,
                    subtree_tree=subtree_tree,
                    full_octree_context=full_octree_context,
                    octree_input_mode=octree_input_mode,
                )
            self.last_compression_debug["actual_codec_disabled_during_train"] = True
            self.last_compression_debug["requested_backend"] = backend
            return L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt
        actual_interval_backends = {
            "octattention_actual",
            "actual_octattention",
            "real_octattention",
            "sparsepcgc_actual",
            "gpcc_actual",
            "draco_actual",
            "octattention_actual_ste",
            "actual_octattention_ste",
            "real_octattention_ste",
            "sparsepcgc_actual_ste",
            "gpcc_actual_ste",
            "draco_actual_ste",
        }
        if (
            backend in actual_interval_backends
            and str(getattr(args, "trainORtest", "train")).strip().lower() == "train"
        ):
            interval = max(int(getattr(args, "actual_eval_interval", 1000)), 0)
            step = int(getattr(args, "_global_train_step", 0)) + 1
            if interval <= 0 or (interval > 1 and step % interval != 0):
                L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt = self._get_compression_loss_proxy(
                    args,
                    gen_xyz=gen_xyz,
                    gt_xyz=gt_xyz,
                    final_w=final_w,
                    cache_key=cache_key,
                    run_grad_probe=True,
                    actual_gen_xyz=actual_gen_xyz,
                    subtree_tree=subtree_tree,
                    full_octree_context=full_octree_context,
                    octree_input_mode=octree_input_mode,
                )
                self.last_compression_debug["actual_codec_skipped_by_interval"] = True
                self.last_compression_debug["requested_backend"] = backend
                self.last_compression_debug["actual_eval_interval"] = int(interval)
                return L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt
        if backend in surrogate_backends:
            try:
                return self._get_compression_loss_surrogate(
                    args,
                    gen_xyz=gen_xyz,
                    gt_xyz=gt_xyz,
                    final_w=final_w,
                    cache_key=cache_key,
                    refresh_actual_gen=refresh_actual_gen,
                    actual_gen_xyz=actual_gen_xyz,
                    subtree_tree=subtree_tree,
                    full_octree_context=full_octree_context,
                    octree_input_mode=octree_input_mode,
                )
            except Exception as exc:
                if not bool(getattr(args, "actual_codec_fallback_to_proxy_on_error", True)):
                    raise
                error_text = f"{type(exc).__name__}: {str(exc)}"
                self._reset_actual_encoder_after_error()
                log_fn = getattr(self, "_log_surrogate_event", None)
                if callable(log_fn):
                    log_fn(
                        "actual/surrogate teacher failed; falling back to proxy loss "
                        f"for this step. backend={backend}, error={error_text[:500]}"
                    )
                L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt = self._get_compression_loss_proxy(
                    args,
                    gen_xyz=gen_xyz,
                    gt_xyz=gt_xyz,
                    final_w=final_w,
                    cache_key=cache_key,
                    run_grad_probe=True,
                    actual_gen_xyz=actual_gen_xyz,
                    subtree_tree=subtree_tree,
                    full_octree_context=full_octree_context,
                    octree_input_mode=octree_input_mode,
                )
                self.last_compression_debug["actual_codec_error"] = error_text[:1000]
                self.last_compression_debug["actual_codec_fallback_to_proxy"] = True
                self.last_compression_debug["requested_backend"] = backend
                return L_com, loss_bit, loss_single, loss_nodes, cached_gt, stats_gt
        if backend in {"octattention_actual", "actual_octattention", "real_octattention", "sparsepcgc_actual", "gpcc_actual", "draco_actual"}:
            return self._get_compression_loss_actual_codec(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                use_proxy_surrogate=False,
                actual_gen_xyz=actual_gen_xyz,
                subtree_tree=subtree_tree,
                full_octree_context=full_octree_context,
                octree_input_mode=octree_input_mode,
            )
        if backend in {"octattention_actual_ste", "actual_octattention_ste", "real_octattention_ste", "sparsepcgc_actual_ste", "gpcc_actual_ste", "draco_actual_ste"}:
            return self._get_compression_loss_actual_codec(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=final_w,
                cache_key=cache_key,
                use_proxy_surrogate=True,
                actual_gen_xyz=actual_gen_xyz,
                subtree_tree=subtree_tree,
                full_octree_context=full_octree_context,
                octree_input_mode=octree_input_mode,
            )
        if backend != "proxy":
            raise ValueError(
                "--compression_loss_backend must be one of: proxy, "
                "octattention_actual, octattention_actual_ste, octattention_surrogate, "
                "sparsepcgc_actual, sparsepcgc_actual_ste, sparsepcgc_surrogate, "
                "gpcc_actual, gpcc_actual_ste, gpcc_surrogate, "
                "draco_actual, draco_actual_ste, draco_surrogate "
                f"(got {backend})"
            )
        return self._get_compression_loss_proxy(
            args,
            gen_xyz=gen_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
            cache_key=cache_key,
            run_grad_probe=True,
            actual_gen_xyz=actual_gen_xyz,
            subtree_tree=subtree_tree,
            full_octree_context=full_octree_context,
            octree_input_mode=octree_input_mode,
        )
