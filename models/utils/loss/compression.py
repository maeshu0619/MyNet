import hashlib
import json
import math
import os
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

        voxel_state = self._get_actuator_voxel_state(args, gen_xyz.device)

        debug["sparsepcgc_debug_uses_same_voxel_state_as_actual"] = bool(
            voxel_state is not None
            and bool(debug.get("actual_uses_actuator_voxel_state", False))
        )
        debug["sparsepcgc_debug_voxel_state_available"] = bool(voxel_state is not None)
        debug["sparsepcgc_debug_voxel_state_update_mode"] = (
            str(voxel_state.get("final_voxel_update_mode", ""))
            if voxel_state is not None
            else ""
        )
        debug["sparsepcgc_debug_final_voxel_recomputed_from_pts_out"] = (
            bool(voxel_state.get("final_voxel_recomputed_from_pts_out", True))
            if voxel_state is not None
            else True
        )

        debug["sparsepcgc_debug_collected"] = True
        debug["sparsepcgc_debug_time"] = float(time.time() - start)
        return debug

    def _build_sparsepcgc_exact_fallback_teacher_loss(
        self,
        args,
        gen_xyz,
        gt_xyz,
        final_w,
        stats_gen,
        cached_gt,
    ):
        """
        SparsePCGC exact occupancy candidate が無効な場合のfallback teacherを作る。

        注意：
        - SparsePCGC本体のactual bitは置き換えない。
        - これはbackward用の近似teacherである。
        - まずActuatorのfinal_voxel_coordsを使い、なければ既存soft auxへ落とす。
        """
        if not self._is_sparsepcgc_context(args, codec_name=stats_gen.get("codec", None)):
            return gen_xyz.new_zeros(()), {
                "sparsepcgc_exact_fallback_used": False,
                "sparsepcgc_exact_fallback_reason": "not_sparsepcgc",
                "sparsepcgc_exact_fallback_has_grad": False,
            }

        aux_terms = self._sparsepcgc_aux_feature_terms(
            args,
            gen_xyz=gen_xyz,
            gt_xyz=gt_xyz,
            final_w=final_w,
        )

        aux_loss = aux_terms.get("loss", None)
        if torch.is_tensor(aux_loss) and aux_loss.requires_grad:
            return aux_loss, {
                "sparsepcgc_exact_fallback_used": True,
                "sparsepcgc_exact_fallback_reason": "sparsepcgc_aux_feature_terms",
                "sparsepcgc_exact_fallback_has_grad": True,
            }

        # 最後の保険：勾配がない場合は0を返す。
        return gen_xyz.new_zeros(()), {
            "sparsepcgc_exact_fallback_used": False,
            "sparsepcgc_exact_fallback_reason": "no_grad_aux_available",
            "sparsepcgc_exact_fallback_has_grad": False,
        }

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
        aux_hard_value_uses_actuator_voxel_state = False
        hard_terms = None

        clip = float(getattr(args, "sparsepcgc_aux_reward_clip", 50.0))

        # まずsoft側をclipする。
        # このsoft値はbackward勾配を担うため、後段clampで勾配を潰さないようにする。
        if clip > 0.0:
            active_soft_term = active_term.clamp(-clip, clip)
            single_soft_term = single_term.clamp(-clip, clip)
            entropy_term = entropy_term.clamp(-clip, clip)
            density_soft_term = density_term.clamp(-clip, clip)
        else:
            active_soft_term = active_term
            single_soft_term = single_term
            density_soft_term = density_term

        if voxel_state is not None and bool(getattr(args, "sparsepcgc_aux_use_actuator_hard_value", True)):
            hard_terms = self._sparsepcgc_aux_hard_terms_from_voxel_state(
                args,
                voxel_state=voxel_state,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
            )

            # hard側もforward値として同じ範囲にclipする。
            if clip > 0.0:
                active_hard_term = hard_terms["active"].clamp(-clip, clip)
                single_hard_term = hard_terms["single"].clamp(-clip, clip)
                density_hard_term = hard_terms["density"].clamp(-clip, clip)
            else:
                active_hard_term = hard_terms["active"]
                single_hard_term = hard_terms["single"]
                density_hard_term = hard_terms["density"]

            # Forward値はActuator後hard voxel統計。
            # Backward勾配はsoft feature近似から借りる。
            active_term = active_hard_term + (active_soft_term - active_soft_term.detach())
            single_term = single_hard_term + (single_soft_term - single_soft_term.detach())
            density_term = density_hard_term + (density_soft_term - density_soft_term.detach())

            # entropy_termはfinal_voxel_coordsだけでは定義しにくいためsoft近似のまま残す。
            aux_hard_value_uses_actuator_voxel_state = True
        else:
            active_term = active_soft_term
            single_term = single_soft_term
            density_term = density_soft_term
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
            "sparsepcgc_aux_hard_value_uses_actuator_voxel_state": gen_xyz.new_tensor(
                float(aux_hard_value_uses_actuator_voxel_state)
            ).detach(),
            "sparsepcgc_aux_hard_value_active_before": (
                hard_terms["before_active"].detach()
                if hard_terms is not None
                else active_ref.detach()
            ),
            "sparsepcgc_aux_hard_value_active_after": (
                hard_terms["after_active"].detach()
                if hard_terms is not None
                else active_gen.detach()
            ),
            "sparsepcgc_aux_hard_value_isolated_before": (
                hard_terms["before_isolated"].detach()
                if hard_terms is not None
                else single_ref.detach()
            ),
            "sparsepcgc_aux_hard_value_isolated_after": (
                hard_terms["after_isolated"].detach()
                if hard_terms is not None
                else single_gen.detach()
            ),
            "sparsepcgc_aux_hard_value_density_before": (
                hard_terms["before_density"].detach()
                if hard_terms is not None
                else density_ref.detach()
            ),
            "sparsepcgc_aux_hard_value_density_after": (
                hard_terms["after_density"].detach()
                if hard_terms is not None
                else density_gen.detach()
            ),
            "sparsepcgc_aux_forward_uses_actuator_voxel_state": gen_xyz.new_tensor(
                float(aux_hard_value_uses_actuator_voxel_state)
            ).detach(),
            "sparsepcgc_aux_backward_uses_soft_gen_xyz": gen_xyz.new_tensor(1.0).detach(),
        }

    def _get_cached_actual_gt(self, cache_key):
        if not self.gt_cache_enabled or not cache_key:
            return None
        cache_entry = self.actual_gt_cache.get(cache_key)
        if cache_entry is not None:
            self.actual_gt_cache.move_to_end(cache_key)
            return dict(cache_entry)
        disk_path = self._actual_gt_disk_cache_path(cache_key)
        if disk_path:
            try:
                with open(disk_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                stats = payload.get("stats", None)
                if payload.get("fingerprint") == self._actual_gt_cache_fingerprint(cache_key) and isinstance(stats, dict):
                    self.actual_gt_cache[cache_key] = dict(stats)
                    self.actual_gt_cache.move_to_end(cache_key)
                    return dict(stats)
            except (OSError, ValueError, TypeError):
                pass
        return None

    def _store_cached_actual_gt(self, cache_key, cache_entry):
        if not self.gt_cache_enabled or not cache_key or self.gt_cache_max_entries <= 0:
            return
        self.actual_gt_cache[cache_key] = dict(cache_entry)
        self.actual_gt_cache.move_to_end(cache_key)
        while len(self.actual_gt_cache) > self.gt_cache_max_entries:
            self.actual_gt_cache.popitem(last=False)
        disk_path = self._actual_gt_disk_cache_path(cache_key)
        if disk_path:
            serializable_stats = {}
            for key, value in dict(cache_entry).items():
                if isinstance(value, float):
                    if math.isfinite(value):
                        serializable_stats[str(key)] = value
                elif isinstance(value, (str, bool, int)) or value is None:
                    serializable_stats[str(key)] = value
                elif isinstance(value, np.generic):
                    scalar = value.item()
                    if not isinstance(scalar, float) or math.isfinite(scalar):
                        serializable_stats[str(key)] = scalar
            payload = {
                "fingerprint": self._actual_gt_cache_fingerprint(cache_key),
                "stats": serializable_stats,
            }
            try:
                os.makedirs(os.path.dirname(disk_path), exist_ok=True)
                temp_path = f"{disk_path}.{os.getpid()}.tmp"
                with open(temp_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, sort_keys=True, allow_nan=False)
                os.replace(temp_path, disk_path)
            except (OSError, ValueError, TypeError):
                try:
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.unlink(temp_path)
                except OSError:
                    pass

    def _actual_gt_cache_fingerprint(self, cache_key):
        args = self.args
        source_path = str(cache_key).split("|", 1)[0]
        try:
            source_stat = os.stat(source_path)
            source_identity = f"{source_stat.st_size}:{source_stat.st_mtime_ns}"
        except OSError:
            source_identity = "missing"
        return "|".join(
            [
                str(cache_key),
                source_identity,
                str(getattr(args, "sparsepcgc_mode", "dense_lossless")),
                str(getattr(args, "sparsepcgc_ckptdir", "")),
                str(float(getattr(args, "sparsepcgc_voxel_size", 1.0))),
                str(int(getattr(args, "sparsepcgc_pos_quantscale", 1))),
            ]
        )

    def _actual_gt_disk_cache_path(self, cache_key):
        if not bool(getattr(self.args, "sparsepcgc_actual_gt_disk_cache", False)):
            return ""
        root = str(getattr(self.args, "sparsepcgc_actual_gt_disk_cache_dir", "")).strip()
        if not root:
            return ""
        digest = hashlib.sha1(self._actual_gt_cache_fingerprint(cache_key).encode("utf-8")).hexdigest()
        return os.path.join(os.path.expanduser(root), f"{digest}.json")

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
        # ============================================================
        # Phase2:
        # actual_encoder.py が返した exact teacher valid 判定を
        # batch集約後の result にも残す。
        # これがないと _get_compression_loss_actual_codec() 側で
        # exact_gen_valid / exact_gt_valid が常にFalseになりやすい。
        # ============================================================
        exact_valid_values = [
            bool(s.get("sparsepcgc_exact_teacher_valid", False))
            for s in stats_list
            if "sparsepcgc_exact_estimated_bits" in s
        ]

        exact_invalid_reasons = [
            str(s.get("sparsepcgc_exact_teacher_invalid_reason", ""))
            for s in stats_list
            if "sparsepcgc_exact_estimated_bits" in s
            and str(s.get("sparsepcgc_exact_teacher_invalid_reason", "")).strip()
        ]

        exact_teacher_valid = bool(
            exact_enabled
            and exact_candidate_count > 0
            and len(exact_valid_values) > 0
            and all(exact_valid_values)
            and np.isfinite(float(exact_estimated_bits))
        )

        exact_teacher_invalid_reason = ""
        if exact_enabled and not exact_teacher_valid:
            if exact_candidate_count <= 0:
                exact_teacher_invalid_reason = "candidate_count_zero"
            elif exact_invalid_reasons:
                exact_teacher_invalid_reason = ";".join(exact_invalid_reasons[:4])
            elif not np.isfinite(float(exact_estimated_bits)):
                exact_teacher_invalid_reason = "bits_non_finite"
            else:
                exact_teacher_invalid_reason = "unknown"

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
                    "sparsepcgc_exact_teacher_valid": bool(exact_teacher_valid),
                    "sparsepcgc_exact_teacher_invalid_reason": str(exact_teacher_invalid_reason),
                }
            )
        return result

    def _encode_actual_many(self, args, xyz_list):
        encoder = self._get_actual_encoder(args)
        candidate_payload = []
        point_tensors = []
        for candidate_idx, xyz in enumerate(list(xyz_list or [])):
            if not torch.is_tensor(xyz):
                candidate_payload.append({"candidate_id": int(candidate_idx)})
                point_tensors.append(None)
                continue
            pts_b = xyz.to(torch.float32)
            candidate_payload.append(
                {
                    "candidate_id": int(candidate_idx),
                    "pts": pts_b,
                }
            )
            point_tensors.append(pts_b)

        with torch.inference_mode():
            if hasattr(encoder, "encode_bits_many"):
                raw_stats_list = list(
                    encoder.encode_bits_many(
                        candidate_payload,
                        max_parallel=int(getattr(args, "sparsepcgc_actual_parallel_candidates", 1)),
                        mode=str(getattr(args, "sparsepcgc_actual_parallel_mode", "single")),
                        fallback_to_single=bool(
                            getattr(args, "sparsepcgc_actual_parallel_fallback_to_single", True)
                        ),
                    )
                )
            else:
                raw_stats_list = []
                for payload in candidate_payload:
                    pts_b = payload.get("pts", None)
                    if not torch.is_tensor(pts_b):
                        raw_stats_list.append(
                            {
                                "candidate_id": payload.get("candidate_id", -1),
                                "actual_requested": False,
                                "actual_finished": False,
                                "actual_worker_id": -1,
                                "actual_wall_time": 0.0,
                                "actual_error_reason": "candidate_tensor_missing",
                                "actual_parallel_mode_effective": "single",
                            }
                        )
                        continue
                    stats = dict(encoder.encode_bits(pts_b))
                    stats.update(
                        {
                            "candidate_id": payload.get("candidate_id", -1),
                            "actual_requested": True,
                            "actual_finished": True,
                            "actual_worker_id": -1,
                            "actual_wall_time": float(stats.get("encode_time", 0.0) or 0.0),
                            "actual_error_reason": "",
                            "actual_parallel_mode_effective": "single",
                        }
                    )
                    raw_stats_list.append(stats)

        stats_list = []
        for pts_b, raw_stats in zip(point_tensors, raw_stats_list):
            stats = dict(raw_stats or {})
            if torch.is_tensor(pts_b) and bool(stats.get("actual_finished", False)):
                stats = self._attach_octree_aux_stats(args, pts_b, stats)
            stats_list.append(stats)
        return stats_list

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
        node_value = float(stats.get("node", 0.0))
        single_value = float(stats.get("single", -1.0))
        # SparsePCGC/G-PCC/Draco wrappers already return codec-side node and
        # single-child counts. Rebuilding the complete octree here duplicated
        # the same work and dominated 8i step time. The local calculation is a
        # fallback only when those codec statistics are unavailable.
        need_aux = node_value <= 0.0 or single_value < 0.0
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
            # SparsePCGCの実符号化は、hard-octreeのpattern histogramではなく、
            # upsampler/classifierが出す候補voxelごとのoccupancy labelを算術符号化する。
            # そのためSparsePCGC候補occupancyが取れている時は、Predicted/Actualの
            # 主occupancy指標も同じ候補空間の値に揃える。
            before_candidate = _int(before_stats, "sparsepcgc_candidate_count", 0)
            after_candidate = _int(after_stats, "sparsepcgc_candidate_count", 0)
            before_entropy_sparse = _float(before_stats, "sparsepcgc_pred_prob_entropy", float("nan"))
            after_entropy_sparse = _float(after_stats, "sparsepcgc_pred_prob_entropy", float("nan"))
            before_nll_sparse = _float(before_stats, "sparsepcgc_pred_occupancy_nll", float("nan"))
            after_nll_sparse = _float(after_stats, "sparsepcgc_pred_occupancy_nll", float("nan"))
            before_predictability_sparse = _float(before_stats, "sparsepcgc_prob_true_mean", float("nan"))
            after_predictability_sparse = _float(after_stats, "sparsepcgc_prob_true_mean", float("nan"))
            before_low_count_sparse = _float(before_stats, "sparsepcgc_prob_true_low_count", float("nan"))
            after_low_count_sparse = _float(after_stats, "sparsepcgc_prob_true_low_count", float("nan"))
            before_low_sparse = _float(before_stats, "sparsepcgc_prob_true_low_ratio", float("nan"))
            after_low_sparse = _float(after_stats, "sparsepcgc_prob_true_low_ratio", float("nan"))

            debug["occupancy_proxy_definition"] = (
                "SparsePCGC candidate occupancy from sigmoid(out_cls.F) and isin(out_cls.C, x_high.C)"
            )
            debug["actual_occupancy_definition"] = (
                "SparsePCGC candidate occupancy labels used by BinaryArithmeticCoding"
            )
            debug["predicted_occupancy_definition"] = (
                "SparsePCGC candidate probability/NLL in the same candidate space as actual labels"
            )

            if before_candidate > 0 or after_candidate > 0:
                candidate_delta = after_candidate - before_candidate
                debug["occupancy_pattern_before"] = before_candidate
                debug["occupancy_pattern_after"] = after_candidate
                debug["occupancy_pattern_delta"] = candidate_delta
                debug["actual_occupancy_pattern_before"] = before_candidate
                debug["actual_occupancy_pattern_after"] = after_candidate
                debug["actual_occupancy_pattern_delta"] = candidate_delta

            if np.isfinite(before_entropy_sparse) and np.isfinite(after_entropy_sparse):
                entropy_delta_sparse = after_entropy_sparse - before_entropy_sparse
                debug["occupancy_entropy_before"] = before_entropy_sparse
                debug["occupancy_entropy_after"] = after_entropy_sparse
                debug["occupancy_entropy_delta"] = entropy_delta_sparse
                debug["actual_occupancy_entropy_before"] = before_entropy_sparse
                debug["actual_occupancy_entropy_after"] = after_entropy_sparse
                debug["actual_occupancy_entropy_delta"] = entropy_delta_sparse

            if np.isfinite(before_nll_sparse) and np.isfinite(after_nll_sparse):
                nll_delta_sparse = after_nll_sparse - before_nll_sparse
                debug["occupancy_nll_before"] = before_nll_sparse
                debug["occupancy_nll_after"] = after_nll_sparse
                debug["occupancy_nll_delta"] = nll_delta_sparse
                debug["actual_occupancy_nll_before"] = before_nll_sparse
                debug["actual_occupancy_nll_after"] = after_nll_sparse
                debug["actual_occupancy_nll_delta"] = nll_delta_sparse

            if np.isfinite(before_low_sparse) and np.isfinite(after_low_sparse):
                low_delta_sparse = after_low_sparse - before_low_sparse
                debug["lowprob_occupancy_ratio"] = after_low_sparse
                debug["actual_lowprob_occupancy_ratio_before"] = before_low_sparse
                debug["actual_lowprob_occupancy_ratio_after"] = after_low_sparse
                debug["actual_lowprob_occupancy_ratio_delta"] = low_delta_sparse

            if np.isfinite(before_low_count_sparse) and np.isfinite(after_low_count_sparse):
                low_count_delta_sparse = after_low_count_sparse - before_low_count_sparse
                debug["lowprob_occupancy_count_before"] = before_low_count_sparse
                debug["lowprob_occupancy_count_after"] = after_low_count_sparse
                debug["actual_lowprob_occupancy_count_before"] = before_low_count_sparse
                debug["actual_lowprob_occupancy_count_after"] = after_low_count_sparse
                debug["actual_lowprob_occupancy_count_delta"] = low_count_delta_sparse

            if np.isfinite(before_predictability_sparse) and np.isfinite(after_predictability_sparse):
                debug["actual_occupancy_predictability_before"] = before_predictability_sparse
                debug["actual_occupancy_predictability_after"] = after_predictability_sparse
                debug["actual_occupancy_predictability_delta"] = (
                    after_predictability_sparse - before_predictability_sparse
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
        if bool(getattr(args, "compact_step_text_log", False)):
            return
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
    
    def _sparsepcgc_aux_hard_terms_from_voxel_state(self, args, voxel_state, gen_xyz, gt_xyz):
        # Actuator後のfinal_voxel_coordsからSparsePCGC aux用のhard統計を作る。
        # これはforward値の基準に使う。Hard統計なので、この値自体には勾配を期待しない。
        before = self._sparsepcgc_hard_stats_batch(args, gt_xyz, final_w=None)
        after = self._sparsepcgc_hard_stats_from_voxel_state(args, voxel_state)

        device = gen_xyz.device
        dtype = gen_xyz.dtype

        active_ref_raw = float(before.get("active", 0))
        isolated_ref_raw = float(before.get("isolated", 0))
        density_ref_raw = float(before.get("local_density_var", 0.0))

        active_after = float(after.get("active", 0))
        isolated_after = float(after.get("isolated", 0))
        density_after = float(after.get("local_density_var", 0.0))

        active_denom = max(abs(active_ref_raw), 1.0)
        isolated_denom = max(abs(isolated_ref_raw), 1.0)
        density_denom = max(abs(density_ref_raw), 1e-6)

        active_term = torch.tensor(
            100.0 * (active_after - active_ref_raw) / active_denom,
            device=device,
            dtype=dtype,
        )
        single_term = torch.tensor(
            100.0 * (isolated_after - isolated_ref_raw) / isolated_denom,
            device=device,
            dtype=dtype,
        )
        density_term = torch.tensor(
            100.0 * (density_after - density_ref_raw) / density_denom,
            device=device,
            dtype=dtype,
        )

        return {
            "active": active_term,
            "single": single_term,
            "density": density_term,
            "before_active": torch.tensor(float(before.get("active", 0)), device=device, dtype=dtype),
            "after_active": torch.tensor(float(after.get("active", 0)), device=device, dtype=dtype),
            "before_isolated": torch.tensor(float(before.get("isolated", 0)), device=device, dtype=dtype),
            "after_isolated": torch.tensor(float(after.get("isolated", 0)), device=device, dtype=dtype),
            "before_density": torch.tensor(float(before.get("local_density_var", 0.0)), device=device, dtype=dtype),
            "after_density": torch.tensor(float(after.get("local_density_var", 0.0)), device=device, dtype=dtype),
        }

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
        voxel_state = self._get_actuator_voxel_state(args, gen_xyz.device)
        voxel_actual_debug = {}

        actual_xyz = gen_xyz if actual_gen_xyz is None else actual_gen_xyz
        actual_final_w = final_w

        if (
            self._is_sparsepcgc_context(args)
            and voxel_state is not None
            and bool(getattr(args, "sparsepcgc_actual_use_actuator_voxel_state", True))
        ):
            restored_xyz, voxel_actual_debug = self._voxel_state_to_codec_xyz(
                args,
                voxel_state=voxel_state,
                like_xyz=gen_xyz,
            )
            if restored_xyz is not None:
                actual_xyz = restored_xyz
                # final_voxel_coords はすでにoccupied voxel集合なので、
                # point-wise final_w をさらに適用しない。
                actual_final_w = None
                try:
                    setattr(args, "_current_actual_uses_voxel_restored", True)
                except Exception:
                    pass
            else:
                try:
                    setattr(args, "_current_actual_uses_voxel_restored", False)
                except Exception:
                    pass
        else:
            try:
                setattr(args, "_current_actual_uses_voxel_restored", False)
            except Exception:
                pass
        cached_gt = self._get_cached_actual_gt(cache_key)
        if cached_gt is None:
            cached_gt = self._encode_actual_batch(args, gt_xyz)
            self._store_cached_actual_gt(cache_key, cached_gt)

        # The full-cloud oracle has already encoded this exact accepted override.
        # Reuse that result once in the immediately following full-cloud loss;
        # every step still performs a real candidate encode and records actual bits.
        cached_oracle_stats = (
            full_octree_context.get("actual_oracle_cached_edited_actual_stats", None)
            if isinstance(full_octree_context, dict)
            and str(octree_input_mode).strip().lower() == "full_cloud"
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
            # actual codec評価は評価指標なので、train用の量子化ノイズを入れないclean編集点群を使う。
            stats_gen = self._encode_actual_batch(args, actual_xyz, final_w=actual_final_w)
        codec_name = str(stats_gen.get("codec", cached_gt.get("codec", "octattention"))).strip().lower()
        backend_label = f"{codec_name}_actual_ste" if use_proxy_surrogate else f"{codec_name}_actual"
        gt_bit = float(cached_gt["bit"])
        gen_bit = float(stats_gen["bit"])
        edit_record_bits = 0.0
        if (
            self._is_sparsepcgc_context(args)
            and bool(getattr(args, "sparsepcgc_edit_record_bits_enabled", True))
            and isinstance(voxel_state, dict)
        ):
            try:
                edit_record_bits = float(voxel_state.get("estimated_edit_record_bits", 0.0) or 0.0)
            except Exception:
                edit_record_bits = 0.0
            if not math.isfinite(edit_record_bits):
                edit_record_bits = 0.0
            edit_record_bits = max(edit_record_bits, 0.0)
        if (
            self._is_sparsepcgc_context(args)
            and bool(getattr(args, "sparsepcgc_edit_record_bits_enabled", True))
            and isinstance(full_octree_context, dict)
        ):
            try:
                context_edit_record_bits = float(
                    full_octree_context.get("actual_oracle_edit_record_bits", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                context_edit_record_bits = 0.0
            if math.isfinite(context_edit_record_bits):
                edit_record_bits = max(edit_record_bits, context_edit_record_bits, 0.0)
        gen_total_bit = gen_bit + edit_record_bits
        raw_loss_bit_percent = 100.0 * self._relative_ratio(gen_bit, gt_bit)
        loss_bit_ratio = self._relative_ratio(gen_total_bit, gt_bit)
        loss_bit_percent = 100.0 * loss_bit_ratio
        if not bool(getattr(args, "compression_loss_delta", True)):
            loss_bit_percent = 100.0 - loss_bit_percent
        policy_actual_noop_guard_used = False
        policy_actual_noop_guard_margin = max(
            float(getattr(args, "sparsepcgc_policy_actual_noop_guard_margin", 0.0)),
            0.0,
        )
        policy_actual_noop_guard_raw_percent = float(loss_bit_percent)
        policy_actual_noop_guard_raw_bit = float(gen_bit)
        policy_actual_noop_guard_raw_total_bit = float(gen_total_bit)
        policy_actual_noop_guard_raw_edit_record_bits = float(edit_record_bits)
        if (
            self._is_sparsepcgc_context(args)
            and bool(getattr(args, "sparsepcgc_policy_actual_noop_guard", True))
            and float(raw_loss_bit_percent) > float(policy_actual_noop_guard_margin)
        ):
            # If the measured policy edit is worse than no-op, the codec action
            # selected by training for this step is no-op.  The raw bad edit is
            # still logged below so this cannot masquerade as an improvement.
            policy_actual_noop_guard_used = True
            stats_gen = dict(cached_gt)
            gen_bit = float(gt_bit)
            edit_record_bits = 0.0
            gen_total_bit = float(gt_bit)
            raw_loss_bit_percent = 0.0
            loss_bit_ratio = 0.0
            loss_bit_percent = 0.0 if bool(getattr(args, "compression_loss_delta", True)) else 100.0

        L_com_hard = gen_xyz.new_tensor(loss_bit_percent)
        L_com = L_com_hard

        proxy_debug = None
        if use_proxy_surrogate:
            proxy_L_com, proxy_loss_bit, proxy_loss_single, proxy_loss_nodes, _, _ = self._get_compression_loss_proxy(
                args,
                gen_xyz=gen_xyz,
                gt_xyz=gt_xyz,
                final_w=actual_final_w,
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
        exact_available_raw = (
            "sparsepcgc_exact_estimated_bits" in stats_gen
            and "sparsepcgc_exact_estimated_bits" in cached_gt
        )

        exact_gen_valid = bool(stats_gen.get("sparsepcgc_exact_teacher_valid", False))
        exact_gt_valid = bool(cached_gt.get("sparsepcgc_exact_teacher_valid", False))

        exact_available = bool(exact_available_raw and exact_gen_valid and exact_gt_valid)

        exact_invalid_reason = ""
        if exact_available_raw and not exact_available:
            exact_invalid_reason = (
                f"gen={stats_gen.get('sparsepcgc_exact_teacher_invalid_reason', '')};"
                f"gt={cached_gt.get('sparsepcgc_exact_teacher_invalid_reason', '')}"
            )

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

            exact_loss_value = exact_bits_delta
            exact_loss = gen_xyz.new_tensor(float(exact_loss_value))
        else:
            exact_loss = gen_xyz.new_zeros(())
        if exact_available and bool(getattr(args, "enable_sparsepcgc_exact_occupancy_loss", False)):
            exact_loss = exact_loss + gen_xyz.new_tensor(
                float(getattr(args, "sparsepcgc_exact_occupancy_loss_weight", 0.0)) * exact_nll_delta
            )
            exact_loss = exact_loss + gen_xyz.new_tensor(
                float(getattr(args, "sparsepcgc_exact_bits_loss_weight", 0.0)) * exact_bits_delta
            )

        exact_teacher_weight = float(getattr(args, "sparsepcgc_exact_teacher_loss_weight", 0.0))
        exact_teacher_grad_weight = float(getattr(args, "sparsepcgc_exact_teacher_grad_weight", 1.0))
        exact_fallback_weight = float(getattr(args, "sparsepcgc_exact_fallback_weight", 0.2))

        exact_teacher_loss_for_backprop = gen_xyz.new_zeros(())
        exact_teacher_used_for_backprop = False
        exact_fallback_debug = {}

        if exact_teacher_weight > 0.0:
            if exact_available:
                fallback_proxy, exact_fallback_debug = self._build_sparsepcgc_exact_fallback_teacher_loss(
                    args=args,
                    gen_xyz=gen_xyz,
                    gt_xyz=gt_xyz,
                    final_w=final_w,
                    stats_gen=stats_gen,
                    cached_gt=cached_gt,
                )

                # forward値はSparsePCGC exact hard delta。
                # backwardはfallback_proxyへ流す。
                if torch.is_tensor(fallback_proxy) and fallback_proxy.requires_grad:
                    exact_teacher_loss_for_backprop = exact_teacher_weight * (
                        exact_loss.detach()
                        + exact_teacher_grad_weight * (fallback_proxy - fallback_proxy.detach())
                    )
                    exact_teacher_used_for_backprop = True
                else:
                    exact_teacher_loss_for_backprop = exact_teacher_weight * exact_loss.detach()
                    exact_teacher_used_for_backprop = False

            else:
                fallback_proxy, exact_fallback_debug = self._build_sparsepcgc_exact_fallback_teacher_loss(
                    args=args,
                    gen_xyz=gen_xyz,
                    gt_xyz=gt_xyz,
                    final_w=final_w,
                    stats_gen=stats_gen,
                    cached_gt=cached_gt,
                )

                if torch.is_tensor(fallback_proxy) and fallback_proxy.requires_grad:
                    exact_teacher_loss_for_backprop = exact_fallback_weight * fallback_proxy
                    exact_teacher_used_for_backprop = True
                else:
                    exact_teacher_loss_for_backprop = gen_xyz.new_zeros(())
                    exact_teacher_used_for_backprop = False

        # ============================================================
        # Phase2:
        # exact / fallback teacher を L_com へ接続する。
        # ============================================================
        if torch.is_tensor(exact_teacher_loss_for_backprop):
            L_com = L_com + exact_teacher_loss_for_backprop

        self._store_compression_terms(
            main=L_com,
            bit=L_com_hard,
            single=gen_xyz.new_zeros(()),
            node=gen_xyz.new_zeros(()),
            bpn=gen_xyz.new_zeros(()),
            objective=L_com,
            sparsepcgc_exact=exact_loss,
            sparsepcgc_exact_teacher=exact_teacher_loss_for_backprop,
            backend=backend_label,
        )

        self.last_compression_debug = {
            "actual_uses_actuator_voxel_state": bool(
                voxel_state is not None
                and voxel_actual_debug.get("voxel_state_codec_xyz_used", False)
            ),
            "actual_voxel_state_reason": str(
                voxel_actual_debug.get("voxel_state_codec_xyz_reason", "")
            ),
            "actual_voxel_state_points": int(
                voxel_actual_debug.get("voxel_state_codec_xyz_points", 0)
            ),
            "actual_voxel_state_final_voxel_coords_count": int(
                voxel_actual_debug.get("voxel_state_final_voxel_coords_count", 0)
            ),
            "actual_voxel_state_update_mode": str(
                voxel_actual_debug.get("voxel_state_final_voxel_update_mode", "")
            ),
            "actual_voxel_state_recomputed_from_pts_out": bool(
                voxel_actual_debug.get("voxel_state_final_voxel_recomputed_from_pts_out", True)
            ),
            "actual_final_w_source": "none_voxel_state_already_occupied" if actual_final_w is None else "point_final_w",
            "metric": "actual_total_bit_percent",
            "teacher_codec": codec_name,
            "total_bit": loss_bit_percent,
            "bpp": self._relative_percent(float(stats_gen["bpp"]), float(cached_gt["bpp"])),
            "actual_scope": str(getattr(args, "_current_teacher_scope", "")),
            "actual_input_source": (
                "actuator_final_voxel_coords"
                if bool(getattr(args, "_current_actual_uses_voxel_restored", False))
                else "gen_xyz_or_actual_gen_xyz"
            ),
            "actual_used_voxel_restored_points": bool(getattr(args, "_current_actual_uses_voxel_restored", False)),
            "actual_input_points": int(stats_gen.get("point_count", 0)),
            "actual_gen_oracle_cache_hit": bool(actual_gen_cache_hit),
                "actual_total_bits": gen_total_bit,
                "actual_raw_bits": gen_bit,
            "actual_edit_record_bits": edit_record_bits,
            "policy_actual_noop_guard_used": bool(policy_actual_noop_guard_used),
            "policy_actual_noop_guard_margin": float(policy_actual_noop_guard_margin),
            "policy_actual_noop_guard_raw_percent": float(policy_actual_noop_guard_raw_percent),
            "policy_actual_noop_guard_raw_bit": float(policy_actual_noop_guard_raw_bit),
            "policy_actual_noop_guard_raw_total_bit": float(policy_actual_noop_guard_raw_total_bit),
            "policy_actual_noop_guard_raw_edit_record_bits": float(policy_actual_noop_guard_raw_edit_record_bits),
                "actual_edit_record_percent": self._relative_percent(
                    gen_bit + edit_record_bits,
                    gen_bit,
                    ref_min=1.0,
                ),
                "actual_bpp": float(stats_gen.get("bpp", 0.0)),
                "actual_delta_percent": loss_bit_percent,
            "actual_occupancy_nll": float(stats_gen.get("octree_occupancy_nll", 0.0)),
            "actual_node_count": float(stats_gen.get("node", 0.0)),
            "actual_single_child_count": float(stats_gen.get("single", 0.0)),
            "actual_lowprob_count": float(stats_gen.get("octree_lowprob_occupancy_count", 0.0)),
            "gt_points": int(cached_gt["point_count"]),
            "gen_points": int(stats_gen["point_count"]),
            "gt_unique_coord_count": int(cached_gt.get("unique_coord_count", cached_gt.get("point_count", 0))),
            "gen_unique_coord_count": int(stats_gen.get("unique_coord_count", stats_gen.get("point_count", 0))),
            "gt_actual_bit": gt_bit,
                "gen_actual_bit": gen_bit,
                "gen_total_bit_with_edit_record": gen_total_bit,
                "actual_total_bit_percent": loss_bit_percent,
                "actual_raw_percent": raw_loss_bit_percent,
                "actual_value_is_fresh": True,
            "actual_value_source": "actual_codec",
            "rate_proxy_before": gt_bit,
                "rate_proxy_after": gen_total_bit,
                "rate_proxy_after_raw": gen_bit,
                "rate_proxy_delta": loss_bit_percent,
            "node_delta": float(stats_gen["node"]) - float(cached_gt["node"]),
            "single_delta": float(stats_gen["single"]) - float(cached_gt["single"]),
            "proxy_surrogate": proxy_debug,
            "sparsepcgc_exact_occupancy_nll_delta": exact_nll_delta,
            "sparsepcgc_exact_estimated_bits_delta": exact_bits_delta,
            "sparsepcgc_exact_bpp_delta": exact_bpp_delta,
            "sparsepcgc_exact_loss_candidate": self._scalar(exact_loss),
            # Phase2: SparsePCGC exact / fallback teacher の状態
            "sparsepcgc_exact_available_raw": bool(exact_available_raw),
            "sparsepcgc_exact_available": bool(exact_available),
            "sparsepcgc_exact_gen_valid": bool(exact_gen_valid),
            "sparsepcgc_exact_gt_valid": bool(exact_gt_valid),
            "sparsepcgc_exact_invalid_reason": str(exact_invalid_reason),
            "sparsepcgc_exact_teacher_loss_weight": float(exact_teacher_weight),
            "sparsepcgc_exact_teacher_grad_weight": float(exact_teacher_grad_weight),
            "sparsepcgc_exact_fallback_weight": float(exact_fallback_weight),
            "sparsepcgc_exact_teacher_used_for_backprop": bool(exact_teacher_used_for_backprop),
            "sparsepcgc_exact_teacher_loss_for_backprop": self._scalar(exact_teacher_loss_for_backprop),
            "sparsepcgc_exact_loss_enabled": bool(
                exact_available and getattr(args, "enable_sparsepcgc_exact_occupancy_loss", False)
            ),
            "actuator_voxel_state_available": bool(
                self._get_actuator_voxel_state(args, gen_xyz.device) is not None
            ),
        }
        self.last_compression_debug.update(exact_fallback_debug)

        if codec_name == "sparsepcgc":
            proxy_bit_percent = float(proxy_debug["loss_bit"]) if proxy_debug is not None else float("nan")
            self.last_compression_debug.update(
                {
                        "actual_sparsepcgc_bit": gen_bit,
                        "actual_sparsepcgc_total_bit_with_edit_record": gen_total_bit,
                        "actual_sparsepcgc_edit_record_bits": edit_record_bits,
                        "actual_sparsepcgc_gt_bit": gt_bit,
                        "actual_sparsepcgc_bit_delta": gen_total_bit - gt_bit,
                        "actual_sparsepcgc_raw_bit_delta": gen_bit - gt_bit,
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
            for key in (
                "initial_voxel_coords",
                "final_voxel_coords",
                "final_voxel_weights",
                "final_voxel_valid_mask",
                "voxel_step",
                "voxel_offset",
                "point_aligned_initial_voxel_coords",
                "point_aligned_final_voxel_coords",
                "point_aligned_final_voxel_weights",
                "voxel_soft_drop_score",
                "voxel_soft_add_score",
                "voxel_soft_move_score",
                "voxel_soft_drop_amount",
                "voxel_soft_add_amount",
                "voxel_soft_move_amount",
                "drop_ratio_soft",
                "add_ratio_soft",
                "move_ratio_soft",
            ):
                value = out.get(key, None)
                if torch.is_tensor(value):
                    out[key] = value.to(device=device, non_blocking=True)
        return out

    def _voxel_state_to_codec_xyz(self, args, voxel_state, like_xyz):
        """
        Actuatorが作ったoccupied voxel stateを、actual codecへ渡す点群に変換する。
        目的は、actual SparsePCGC / proxy / full-context loss / debug metric が
        同じ final_voxel_coords を見るようにすることである。

        注意：
        final_voxel_coords は [B, 3, N] のglobal voxel coordsである。
        voxel_step / voxel_offset があれば、それを使って点座標へ戻す。
        なければ coords 自体をfloat座標として使う。
        """
        if not isinstance(voxel_state, dict):
            return None, {
                "voxel_state_codec_xyz_used": False,
                "voxel_state_codec_xyz_reason": "missing_voxel_state",
            }

        coords = voxel_state.get("final_voxel_coords", None)
        valid_mask = voxel_state.get("final_voxel_valid_mask", None)
        voxel_step = voxel_state.get("voxel_step", None)
        voxel_offset = voxel_state.get("voxel_offset", None)

        if coords is None or not torch.is_tensor(coords):
            return None, {
                "voxel_state_codec_xyz_used": False,
                "voxel_state_codec_xyz_reason": "missing_final_voxel_coords",
            }

        coords = coords.to(device=like_xyz.device, dtype=torch.long)

        if coords.ndim == 2:
            if coords.shape[0] == 3:
                coords = coords.unsqueeze(0)
            elif coords.shape[1] == 3:
                coords = coords.transpose(0, 1).contiguous().unsqueeze(0)
            else:
                return None, {
                    "voxel_state_codec_xyz_used": False,
                    "voxel_state_codec_xyz_reason": "invalid_final_voxel_coords_shape",
                }
        elif coords.ndim == 3:
            if coords.shape[1] == 3:
                coords = coords.contiguous()
            elif coords.shape[2] == 3:
                coords = coords.permute(0, 2, 1).contiguous()
            else:
                return None, {
                    "voxel_state_codec_xyz_used": False,
                    "voxel_state_codec_xyz_reason": "invalid_final_voxel_coords_shape",
                }
        else:
            return None, {
                "voxel_state_codec_xyz_used": False,
                "voxel_state_codec_xyz_reason": "invalid_final_voxel_coords_ndim",
            }

        B = coords.shape[0]

        if valid_mask is not None and torch.is_tensor(valid_mask):
            valid_mask = valid_mask.to(device=coords.device, dtype=torch.bool)
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.squeeze(1)
            if valid_mask.ndim == 1:
                valid_mask = valid_mask.view(1, -1)
            if valid_mask.shape[0] == 1 and B > 1:
                valid_mask = valid_mask.expand(B, -1)
            if valid_mask.shape[0] != B or valid_mask.shape[1] != coords.shape[2]:
                valid_mask = None

        if voxel_step is not None and torch.is_tensor(voxel_step):
            step = voxel_step.to(device=like_xyz.device, dtype=like_xyz.dtype)
            if step.ndim == 0:
                step = step.view(1, 1, 1)
            elif step.ndim == 1:
                step = step.view(-1, 1, 1)
            elif step.ndim == 2:
                step = step.view(step.shape[0], 1, 1)
            elif step.ndim == 3:
                step = step[:, :1, :1]
            if step.shape[0] == 1 and B > 1:
                step = step.expand(B, -1, -1)
        else:
            step = like_xyz.new_ones((B, 1, 1))

        if voxel_offset is not None and torch.is_tensor(voxel_offset):
            offset = voxel_offset.to(device=like_xyz.device, dtype=like_xyz.dtype)
            if offset.ndim == 1 and offset.numel() == 3:
                offset = offset.view(1, 3, 1)
            elif offset.ndim == 2 and offset.shape[-1] == 3:
                offset = offset.view(-1, 3, 1)
            elif offset.ndim == 2 and offset.shape[0] == 3:
                offset = offset.unsqueeze(0)
            elif offset.ndim == 3 and offset.shape[1] == 3:
                offset = offset[:, :, :1]
            else:
                offset = like_xyz.new_zeros((B, 3, 1))
            if offset.shape[0] == 1 and B > 1:
                offset = offset.expand(B, -1, -1)
        else:
            offset = like_xyz.new_zeros((B, 3, 1))

        xyz = offset + coords.to(dtype=like_xyz.dtype) * step

        # actual encoderはbatch内で可変長を扱うため、paddingは避ける。
        # ただし既存 _encode_actual_batch は [B,3,N] を想定するので、
        # B=1ではmaskで切り、B>1ではvalid_maskをfinal_w扱いにする。
        if valid_mask is not None and B == 1:
            xyz = xyz[:, :, valid_mask[0]]

        debug = {
            "voxel_state_codec_xyz_used": True,
            "voxel_state_codec_xyz_reason": "ok",
            "voxel_state_codec_xyz_points": int(xyz.shape[-1]),
            "voxel_state_codec_xyz_batch": int(xyz.shape[0]),
            "voxel_state_final_voxel_coords_count": int(coords.shape[-1]),
            "voxel_state_has_valid_mask": bool(valid_mask is not None),
            "voxel_state_final_voxel_update_mode": str(voxel_state.get("final_voxel_update_mode", "")),
            "voxel_state_final_voxel_recomputed_from_pts_out": bool(
                voxel_state.get("final_voxel_recomputed_from_pts_out", True)
            ),
            "voxel_state_actuator_voxel_mode": str(voxel_state.get("actuator_voxel_mode", "")),
        }
        return xyz.contiguous(), debug

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
