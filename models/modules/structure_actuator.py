import math
import time

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
        nn.init.zeros_(self.drop_amount_head.weight)
        nn.init.zeros_(self.add_amount_head.weight)
        nn.init.zeros_(self.move_amount_head.weight)
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
        nn.init.constant_(self.drop_amount_head.bias, 0.0)
        nn.init.constant_(self.add_amount_head.bias, 0.0)
        nn.init.constant_(self.move_amount_head.bias, 0.0)
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

    @staticmethod
    def _voxel_max_scores(scores, inverse, voxel_count):
        voxel_scores = scores.new_full((voxel_count,), -1.0e6)
        scatter_reduce = getattr(voxel_scores, "scatter_reduce_", None)
        if callable(scatter_reduce):
            scatter_reduce(0, inverse, scores, reduce="amax", include_self=True)
            return voxel_scores
        for voxel_id in range(int(voxel_count)):
            mask = inverse == voxel_id
            if bool(mask.any().item()):
                voxel_scores[voxel_id] = scores[mask].max()
        return voxel_scores

    @classmethod
    def _top_voxel_indices_by_score(cls, scores, inverse, voxel_count, drop_count):
        voxel_scores = cls._voxel_max_scores(scores, inverse, voxel_count)
        if int(drop_count) <= 0:
            return torch.empty((0,), device=scores.device, dtype=torch.long)
        if bool((voxel_scores > -1.0e6).all().item()):
            return torch.topk(voxel_scores, k=drop_count, largest=True, sorted=False).indices
        order = torch.argsort(scores.detach(), descending=True)
        sorted_voxels = inverse.index_select(0, order).detach().cpu().tolist()
        selected = []
        seen = set()
        for voxel_id in sorted_voxels:
            voxel_id = int(voxel_id)
            if voxel_id in seen:
                continue
            seen.add(voxel_id)
            selected.append(voxel_id)
            if len(selected) >= drop_count:
                break
        if len(selected) < drop_count:
            for voxel_id in range(int(voxel_count)):
                if voxel_id in seen:
                    continue
                selected.append(voxel_id)
                if len(selected) >= drop_count:
                    break
        return torch.as_tensor(selected, device=scores.device, dtype=torch.long)

    def _hard_voxel_drop_mask(
        self,
        voxel_coords,
        drop_scores,
        target_drop_ratio,
        max_drop_ratio,
        selection_mask,
        hard_threshold=0.0,
        voxel_cache=None,
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
            if voxel_count <= 1:
                continue
            if threshold_cap_mode:
                cap_count = int(math.ceil(float(max_drop_ratio) * float(voxel_count)))
                drop_count = min(max(cap_count, 0), voxel_count - 1)
            else:
                cap_count = int(round(float(max_drop_ratio) * float(voxel_count)))
                target_count = int(round(float(target_drop_ratio) * float(voxel_count)))
                if target_drop_ratio > 0.0:
                    target_count = max(target_count, 1)
                if max_drop_ratio > 0.0:
                    cap_count = max(cap_count, 1)
                drop_count = min(target_count, cap_count, voxel_count - 1)
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

    def _hard_point_topk_mask(
        self,
        scores,
        target_ratio,
        selection_mask=None,
        exclude_mask=None,
        hard_threshold=0.0,
    ):
        B, _, N = scores.shape
        hard_mask = torch.zeros_like(scores, dtype=torch.bool)
        if N <= 0 or float(target_ratio) <= 0.0:
            return hard_mask
        threshold_cap_mode = self._threshold_cap_mode()
        if selection_mask is None:
            valid_all = torch.ones((B, N), device=scores.device, dtype=torch.bool)
        else:
            valid_all = selection_mask.squeeze(1) if selection_mask.ndim == 3 else selection_mask
            valid_all = valid_all.to(device=scores.device, dtype=torch.bool)
        if exclude_mask is not None:
            exclude = exclude_mask.squeeze(1) if exclude_mask.ndim == 3 else exclude_mask
            valid_all = valid_all & (~exclude.to(device=scores.device, dtype=torch.bool))
        for b in range(B):
            score_values = scores[b, 0].detach()
            valid = valid_all[b] & torch.isfinite(score_values) & (score_values > 0.0)
            valid_count = int(valid.sum().item())
            if valid_count <= 0:
                continue
            count = int(round(float(target_ratio) * float(valid_count)))
            if threshold_cap_mode:
                count = min(max(count, 0), valid_count)
            else:
                count = min(max(count, 1), valid_count)
            if count <= 0:
                continue
            mask_value = torch.finfo(scores.dtype).min
            masked_scores = score_values.masked_fill(~valid, mask_value)
            idx = torch.topk(masked_scores, k=count, largest=True, sorted=False).indices
            if threshold_cap_mode:
                idx_scores = score_values.index_select(0, idx)
                idx = idx[idx_scores >= float(hard_threshold)]
                if idx.numel() <= 0:
                    continue
            hard_mask[b, 0, idx] = True
        return hard_mask

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

    @staticmethod
    def _safe_logit(prob):
        # 0/1付近の確率を安全にlogitへ戻し、比率biasや探索ノイズをlogit空間で足す。
        prob = prob.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(prob / (1.0 - prob))

    def _learned_operation_ratio(self, actuator_features, head, max_ratio, random_mix_start, random_mix_end):
        # 全点特徴を集約して、このStepでAdd/Adjustする割合を学習可能なTensorとして作る。
        if max_ratio <= 0.0:
            return actuator_features.new_zeros((actuator_features.shape[0], 1, 1))
        pooled = actuator_features.mean(dim=2, keepdim=True)
        if bool(getattr(self.args, "repair_learn_operation_amounts", True)):
            ratio = torch.sigmoid(head(pooled)) * float(max_ratio)
        else:
            ratio = pooled.new_full((pooled.shape[0], 1, 1), float(max_ratio))
        # 学習初期だけランダム比率を混ぜ、Add/Adjust量の探索範囲を広げる。
        random_mix = min(max(self._annealed_value(random_mix_start, random_mix_end), 0.0), 1.0)
        if self.training and random_mix > 0.0:
            random_ratio = torch.rand_like(ratio) * float(max_ratio)
            ratio = (1.0 - random_mix) * ratio + random_mix * random_ratio
        return ratio.clamp(0.0, float(max_ratio))

    def _ratio_bias(self, ratio, max_ratio):
        # 学習した操作量を位置scoreへ戻し、何個選ぶかとどこを選ぶかの勾配をつなぐ。
        if max_ratio <= 0.0:
            return ratio.new_zeros(ratio.shape)
        normalized = (ratio / float(max_ratio)).clamp(1e-4, 1.0 - 1e-4)
        return self._safe_logit(normalized) * float(getattr(self.args, "repair_operation_amount_bias_scale", 2.0))

    def _max_add_ratio(self):
        target_ratio = self._target_add_ratio_value()
        if self._sparsepcgc_add_experiment_active():
            max_ratio = max(float(getattr(self.args, "sparsepcgc_add_max_ratio", 0.003)), target_ratio)
            max_ratio = max_ratio * self._sparsepcgc_add_warmup()
            max_ratio = max(max_ratio, target_ratio)
        else:
            max_ratio = max(float(getattr(self.args, "max_add_ratio", max(target_ratio, 0.0))), target_ratio)
        return max(max_ratio, 0.0)

    def _target_add_ratio_value(self):
        if self._sparsepcgc_add_experiment_active():
            ratio = max(float(getattr(self.args, "sparsepcgc_add_target_ratio", 0.001)), 0.0)
            return ratio * self._sparsepcgc_add_warmup()
        return max(float(getattr(self.args, "target_add_ratio", 0.01)), 0.0)

    def _sparsepcgc_add_warmup(self):
        steps = max(int(getattr(self.args, "sparsepcgc_add_warmup_steps", 0)), 0)
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

    def _target_add_count(self, point_count, candidate_ratio_override=None):
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
        add_points = int(math.ceil(candidate_ratio * float(point_count))) if candidate_ratio > 0.0 else 0
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
        mask_value = torch.finfo(add_scores.dtype).min
        masked = add_scores.masked_fill(~valid, mask_value)
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

        voxel_step = self._voxel_step(pts_xyz, coord_scale)
        voxel_norm = (voxel_step * math.sqrt(3.0)).clamp_min(1e-9)
        voxel_coords = self._voxel_coords(pts_xyz, voxel_step)
        voxel_cache = self._build_voxel_cache(voxel_coords)
        neighbor_offsets = self.neighbor_offsets.to(device=pts_xyz.device, dtype=pts_xyz.dtype)
        neighbor_offsets_long = self.neighbor_offsets.to(device=pts_xyz.device, dtype=torch.long)
        empty_target_mask = self._empty_neighbor_target_mask(voxel_coords, voxel_cache=voxel_cache)
        B, _, N = pts_xyz.shape
        if selection_mask is None:
            selection_bool = torch.ones((B, N), device=pts_xyz.device, dtype=torch.bool)
        else:
            selection_bool = selection_mask.squeeze(1) if selection_mask.ndim == 3 else selection_mask
            selection_bool = selection_bool.to(device=pts_xyz.device, dtype=torch.bool)
        voxel_point_counts = self._voxel_point_counts(voxel_coords, voxel_cache=voxel_cache).to(device=pts_xyz.device)
        before_occupied_voxels = self._unique_voxel_count_from_cache(voxel_cache, selection_bool)
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
        target_drop_ratio = float(getattr(self.args, "target_drop_ratio", 0.01)) if prune_enabled else 0.0
        max_drop_ratio = max(float(getattr(self.args, "max_drop_ratio", max(target_drop_ratio, 0.01))), target_drop_ratio)
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
        # hard削除数は整数なので、学習比率の値だけを使ってVoxel選択数へ変換する。
        learned_drop_ratio_value = float(learned_drop_ratio.detach().mean().cpu()) if prune_enabled else 0.0
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
        if prune_enabled and max_drop_ratio > 0.0:
            # 学習したPrune量を削除scoreへ反映し、量と位置を同じlogit上で調整する。
            drop_prob = torch.sigmoid(self._safe_logit(drop_prob) + self._ratio_bias(learned_drop_ratio, max_drop_ratio))
        drop_random_mix = min(
            max(self._annealed_value("repair_drop_random_mix_start", "repair_drop_random_mix_end"), 0.0),
            1.0,
        )
        if self.training and prune_enabled and drop_random_mix > 0.0:
            random_drop = self._random_ratio_mask_like(drop_prob, max_drop_ratio, selection_mask)
            drop_prob = ((1.0 - drop_random_mix) * drop_prob + drop_random_mix * random_drop).clamp(0.0, 1.0)
        if not prune_enabled:
            drop_prob = torch.zeros_like(drop_prob)
        drop_prob = self._voxel_mean_logits(drop_prob, voxel_coords, voxel_cache=voxel_cache).clamp(0.0, 1.0)
        # 削除は点単位ではなくcodec量子化step上のleaf voxel単位で決める。
        # Octree occupancyはVoxelが1点でも残ると変わらないため、選択Voxel内の点をまとめて削除する。
        delete_candidate_mask = selection_bool.clone()
        delete_max_points = int(getattr(self.args, "repair_delete_max_points_per_voxel", 8))
        if delete_max_points > 0:
            delete_candidate_mask = delete_candidate_mask & (
                voxel_point_counts.squeeze(1) <= float(delete_max_points)
            )
        hard_drop_mask = self._hard_voxel_drop_mask(
            voxel_coords,
            drop_prob,
            target_drop_ratio=learned_drop_ratio_value,
            max_drop_ratio=learned_drop_ratio_value,
            selection_mask=delete_candidate_mask.unsqueeze(1),
            hard_threshold=float(getattr(self.args, "repair_drop_hard_threshold", 0.5)),
            voxel_cache=voxel_cache,
        )
        hard_drop = hard_drop_mask.to(dtype=pts_xyz.dtype)
        drop_prob_st = hard_drop - drop_prob.detach() + drop_prob
        keep_prob = (1.0 - drop_prob_st).clamp(0.0, 1.0)
        if timing_enabled:
            _mark_runtime("delete")

        move_score = (repair_gate * (1.0 - hard_drop)).clamp(0.0, 1.0)
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
        max_move_ratio = max(float(getattr(self.args, "max_move_ratio", target_ratio)), target_ratio) if disp_enabled else 0.0
        # Adjustする割合を特徴から学習し、固定target_ratioだけに依存しないsource数にする。
        learned_move_ratio = self._learned_operation_ratio(
            actuator_features,
            self.move_amount_head,
            max_move_ratio,
            "repair_move_amount_random_mix_start",
            "repair_move_amount_random_mix_end",
        )
        # hard選択個数は整数なので、学習比率の値だけを使って選択数へ変換する。
        move_target_ratio = float(learned_move_ratio.detach().mean().cpu()) if disp_enabled else 0.0
        require_empty_move = bool(getattr(self.args, "repair_move_require_empty_target", True))
        prefer_occupied_move = bool(getattr(self.args, "repair_move_prefer_occupied_target", False)) and not require_empty_move
        # SparsePCGCではtargetを新規empty voxelにするとactive coordinateが増えやすい。
        # 既存occupied targetへのmergeを優先し、sourceが空になる操作だけがactive削減へ効くようにする。
        if sparsepcgc_context and bool(getattr(self.args, "sparsepcgc_move_existing_target_only", True)):
            require_empty_move = False
            prefer_occupied_move = True
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
            move_score = torch.sigmoid(self._safe_logit(move_score) + self._ratio_bias(learned_move_ratio, max_move_ratio))
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
        has_valid_move_target = valid_move_points.any(dim=2).unsqueeze(1).to(dtype=move_score.dtype)
        if require_empty_move:
            move_target_valid = has_valid_move_target
            move_score = move_score * has_valid_move_target
        elif prefer_occupied_move:
            move_target_valid = has_valid_move_target
            move_score = move_score * has_valid_move_target
        # 調整もsource voxelを先に選ぶ。Voxel内の一部だけを微小移動してもoccupancyが変わらないため、
        # 選択source voxel内の点を同じtarget voxel候補へ移す方針にする。
        move_candidate_mask = selection_bool & (~hard_drop_mask.squeeze(1))
        move_max_points = int(getattr(self.args, "repair_move_max_points_per_voxel", 8))
        if move_max_points > 0:
            move_candidate_mask = move_candidate_mask & (
                voxel_point_counts.squeeze(1) <= float(move_max_points)
            )
        move_candidate_mask = move_candidate_mask & has_valid_move_target.squeeze(1).to(dtype=torch.bool)
        hard_move_mask = self._hard_voxel_drop_mask(
            voxel_coords,
            move_score,
            target_drop_ratio=move_target_ratio,
            max_drop_ratio=move_target_ratio,
            selection_mask=move_candidate_mask.unsqueeze(1),
            hard_threshold=float(getattr(self.args, "repair_move_hard_threshold", 0.5)),
            voxel_cache=voxel_cache,
        )
        hard_move = hard_move_mask.to(dtype=pts_xyz.dtype)
        move_mask = hard_move - move_score.detach() + move_score
        move_mask = move_mask * keep_prob

        move_logits = self.move_voxel_head(actuator_features)
        move_logits = self._voxel_mean_logits(move_logits, voxel_coords, voxel_cache=voxel_cache)
        move_valid_target = valid_move_points.transpose(1, 2)
        no_valid_move = ~move_valid_target.any(dim=1, keepdim=True)
        safe_valid_move = torch.where(no_valid_move, torch.ones_like(move_valid_target), move_valid_target)
        # float16でもoverflowしない負値を使う
        mask_value = torch.finfo(move_logits.dtype).min
        move_logits = move_logits.masked_fill(~safe_valid_move, mask_value)

        move_probs = torch.softmax(move_logits, dim=1)
        move_idx = move_probs.detach().argmax(dim=1, keepdim=True)
        hard_move_dir = torch.zeros_like(move_probs)
        hard_move_dir.scatter_(1, move_idx, 1.0)
        move_dir = hard_move_dir - move_probs.detach() + move_probs
        move_selected_valid = (
            move_dir * move_valid_target.to(dtype=move_dir.dtype)
        ).sum(dim=1, keepdim=True)
        quant_move_conflict_loss = self._masked_mean(
            move_mask * (1.0 - move_selected_valid).clamp(0.0, 1.0),
            selection_mask,
        )
        selected_offsets = torch.einsum("bkn,kc->bcn", move_dir, neighbor_offsets)
        target_centers = (voxel_coords.to(dtype=pts_xyz.dtype) + selected_offsets) * voxel_step
        primitive_delta = target_centers - pts_xyz
        delta = move_mask * primitive_delta
        pts_out = pts_xyz + delta
        move_idx_flat = move_idx.squeeze(1)
        selected_offsets_long = neighbor_offsets_long.index_select(0, move_idx_flat.reshape(-1))
        selected_offsets_long = selected_offsets_long.view(B, N, 3).transpose(1, 2).contiguous()
        move_target_voxel_coords = voxel_coords + selected_offsets_long
        same_voxel_move_mask = (
            hard_move_mask.squeeze(1)
            & (move_target_voxel_coords == voxel_coords).all(dim=1)
        )
        moved_different_voxel_mask = hard_move_mask.squeeze(1) & (~same_voxel_move_mask)
        if timing_enabled:
            _mark_runtime("adjust_move")

        final_w = keep_prob
        max_add_ratio_value = self._max_add_ratio()
        # Addする割合を特徴から学習し、固定10%のような張り付きから外す。
        learned_add_ratio = self._learned_operation_ratio(
            actuator_features,
            self.add_amount_head,
            max_add_ratio_value if add_enabled else 0.0,
            "repair_add_amount_random_mix_start",
            "repair_add_amount_random_mix_end",
        )
        # hardなtop-k個数は整数なので、学習比率の値だけを候補数計算へ渡す。
        learned_add_ratio_value = float(learned_add_ratio.detach().mean().cpu()) if add_enabled else 0.0
        add_k, add_candidate_ratio = self._target_add_count(N, candidate_ratio_override=learned_add_ratio_value)
        add_ratio = pts_xyz.new_zeros(())
        add_ratio_loss = pts_xyz.new_zeros(())
        add_shape_guard = pts_xyz.new_zeros(())
        add_offset_reg = pts_xyz.new_zeros(())
        add_drop_conflict_loss = pts_xyz.new_zeros(())
        added_keep_loss = pts_xyz.new_zeros(())
        add_min_offset_loss = pts_xyz.new_zeros(())
        quant_add_guard = pts_xyz.new_zeros(())
        add_prob = pts_xyz.new_zeros((B, 1, N))
        add_priority = add_prob
        add_count_value = 0
        add_effective_count_value = 0
        add_target_voxel_count_value = 0
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
            if sparsepcgc_add_experiment_active and not bool(getattr(self.args, "sparsepcgc_add_use_candidate_score", True)):
                add_prior = torch.zeros_like(add_prior)
            # Add量の学習結果を位置logitに足し、どのVoxelへ追加するかの勾配も残す。
            add_logit = learned_add_logit + add_prior + self._ratio_bias(learned_add_ratio, max_add_ratio_value)
            if self.training and add_score_noise > 0.0:
                add_logit = add_logit + self._gumbel_like(add_logit) * add_score_noise
            add_voxel_logits = self.add_voxel_head(actuator_features)
            add_logit = self._voxel_mean_logits(add_logit, voxel_coords, voxel_cache=voxel_cache)
            add_voxel_logits = self._voxel_mean_logits(add_voxel_logits, voxel_coords, voxel_cache=voxel_cache)
            pair_logits = (add_voxel_logits + add_logit).permute(0, 2, 1).contiguous()
            if selection_mask is None:
                base_valid = torch.ones((B, N), device=pts_xyz.device, dtype=torch.bool)
            else:
                base_valid = selection_mask.squeeze(1) if selection_mask.ndim == 3 else selection_mask
                base_valid = base_valid.to(device=pts_xyz.device, dtype=torch.bool)
            keep_threshold = float(getattr(self.args, "add_noop_keep_threshold", 0.5))
            base_valid = base_valid & (~hard_drop_mask.squeeze(1))
            if keep_threshold > 0.0:
                base_valid = base_valid & (keep_prob.detach().squeeze(1) >= keep_threshold)
            # 追加は空Voxelを選んで、そのVoxel中心に点を置く。
            # 既存occupied voxelへ追加してもoccupancyが変わらず、Octree rateの改善信号が弱くなるため。
            valid_pair = empty_target_mask & base_valid.unsqueeze(2)
            valid_counts = valid_pair.reshape(B, -1).sum(dim=1)
            effective_add_k = min(int(add_k), int(valid_counts.min().detach().cpu().item()))
            if effective_add_k > 0:
                mask_value = torch.finfo(pair_logits.dtype).min
                pair_scores = pair_logits.masked_fill(~valid_pair, mask_value).reshape(B, -1)
                top_pair_values, top_pair_idx = torch.topk(
                    pair_scores.detach(),
                    k=effective_add_k,
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                pair_priority_flat = torch.sigmoid(pair_scores.clamp(-8.0, 8.0))
                selected_pair_strength = torch.gather(pair_priority_flat, 1, top_pair_idx)
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
                threshold = top_pair_values.min(dim=1, keepdim=True).values.detach()
                soft_add_pair = torch.sigmoid((pair_scores - threshold) / tau)
                soft_add_pair = soft_add_pair * valid_pair.reshape(B, -1).to(dtype=soft_add_pair.dtype)
                hard_ratio = hard_add_pair.mean(dim=1, keepdim=True)
                soft_mean = soft_add_pair.mean(dim=1, keepdim=True).detach().clamp_min(1e-12)
                soft_add_pair = (soft_add_pair * (hard_ratio / soft_mean)).clamp(0.0, 1.0)
                add_pair_st = hard_add_pair - soft_add_pair.detach() + soft_add_pair
                add_pair_st = add_pair_st.view(B, N, -1)
                add_prob = add_pair_st.sum(dim=2, keepdim=True).transpose(1, 2).clamp(0.0, 1.0)
                pair_priority = torch.sigmoid(pair_logits.clamp(-8.0, 8.0))
                add_priority = pair_priority.max(dim=2, keepdim=True).values.transpose(1, 2)

                selected_base_voxels = selected_base_voxels_long.to(dtype=pts_xyz.dtype)
                selected_offsets_add = selected_offsets_add_long.to(dtype=pts_xyz.dtype)
                added_pts = selected_add_voxels_long.to(dtype=pts_xyz.dtype) * voxel_step
                added_base = torch.gather(pts_out, 2, idx_expand_xyz)
                added_delta = added_pts - added_base
                selected_add_strength = torch.gather(pair_priority.reshape(B, -1), 1, top_pair_idx).unsqueeze(1)
                selected_hard_add = torch.gather(hard_add_pair, 1, top_pair_idx).unsqueeze(1)
                add_weight_mode = str(getattr(self.args, "repair_add_weight_mode", "hard")).strip().lower()
                if add_weight_mode == "soft":
                    added_w = selected_add_strength * selected_hard_add.detach()
                else:
                    added_w = selected_hard_add - selected_add_strength.detach() + selected_add_strength

                pts_out = torch.cat([pts_out, added_pts], dim=2)
                final_w = torch.cat([final_w, added_w], dim=2)

                add_ratio = added_w.sum() / max(float(B * N), 1.0)
                target_add_ratio = pts_xyz.new_tensor(self._target_add_ratio_value() if add_enabled else 0.0)
                if threshold_cap_mode:
                    max_add_ratio_t = pts_xyz.new_tensor(self._max_add_ratio())
                    add_ratio_loss = torch.relu(add_ratio - max_add_ratio_t).pow(2)
                else:
                    add_ratio_loss = (add_ratio - target_add_ratio).pow(2)
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
                added_delta_norm = torch.linalg.norm(added_delta, dim=1, keepdim=True) / voxel_norm.clamp_min(1e-12)
                add_offset_reg = (added_delta_norm.pow(2) * added_w.detach()).sum() / added_w.detach().sum().clamp_min(1.0)
                add_min_offset_loss = add_prob.new_zeros(())
                add_count_value = int(selected_hard_add.detach().sum().item())
                hardening_threshold = float(
                    getattr(self.args, "operation_count_drop_threshold", getattr(self.args, "test_drop_threshold", 0.5))
                )
                add_effective_count_value = int((added_w.detach() >= hardening_threshold).sum().item())
                add_target_voxel_count_value = self._unique_voxel_count(
                    selected_add_voxels_long,
                    (selected_hard_add.detach() >= hardening_threshold),
                )
        if timing_enabled:
            _mark_runtime("add")

        hardening_threshold = float(
            getattr(self.args, "operation_count_drop_threshold", getattr(self.args, "test_drop_threshold", 0.5))
        )
        final_voxel_coords = self._voxel_coords(pts_out, voxel_step)
        hard_keep_mask = final_w.detach() >= hardening_threshold
        after_occupied_voxels = self._unique_voxel_count(final_voxel_coords, hard_keep_mask)
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
        hard_drop_count_value = int(hard_drop.detach().sum().item())
        hard_move_count_value = int(hard_move.detach().sum().item())
        preserve_hard = (~hard_drop_mask) & (~hard_move_mask)
        preserve_ratio = preserve_hard.to(dtype=pts_xyz.dtype).mean()

        delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
        normalized_delta = delta_norm / voxel_norm.clamp_min(1e-12)
        edit_reg = self._masked_mean(normalized_delta.pow(2) * hard_move, selection_mask)
        moved_points = hard_move.sum().clamp_min(1.0)
        moved_delta_mean = (delta_norm * hard_move).sum() / moved_points

        repair_gate_mean = self._masked_mean(repair_gate, selection_mask)
        if threshold_cap_mode:
            ratio_loss = torch.relu(repair_gate_mean - gate_cap_ratio) ** 2
        else:
            ratio_loss = (repair_gate_mean - target_ratio) ** 2
        shape_guard = self._masked_mean(repair_gate * shape_score, selection_mask)
        drop_ratio = self._masked_mean(drop_prob, selection_mask)
        if threshold_cap_mode or bool(getattr(self.args, "repair_learn_operation_amounts", True)):
            # 操作量を学習する場合は固定削除率へ引っ張らず、静的上限だけを守る。
            drop_ratio_loss = torch.relu(drop_ratio - drop_ratio.new_tensor(float(max_drop_ratio))) ** 2
        else:
            drop_ratio_loss = (drop_ratio - target_drop_ratio) ** 2
        drop_cap_loss = torch.relu(drop_ratio - max_drop_ratio) ** 2
        drop_shape_guard = self._masked_mean(drop_prob * shape_score, selection_mask)
        local_edit_guard = self._masked_mean((drop_prob + move_mask).clamp(0.0, 1.0) * shape_score, selection_mask)
        move_ratio_soft = self._masked_mean(move_mask, selection_mask)
        # 各操作量headが実際のsoft操作率を追えるようにし、量headにも安定した勾配を渡す。
        operation_amount_consistency_loss = (
            (drop_ratio - learned_drop_ratio.mean()).pow(2)
            + (move_ratio_soft - learned_move_ratio.mean()).pow(2)
            + (add_ratio - learned_add_ratio.mean()).pow(2)
        )
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
            + float(getattr(self.args, "repair_quant_guard_weight", 1.0)) * (quant_move_conflict_loss + quant_add_guard)
            + float(getattr(self.args, "repair_local_guard_weight", 0.25)) * local_edit_guard
            + float(getattr(self.args, "repair_operation_amount_consistency_weight", 1.0)) * operation_amount_consistency_loss
        )
        if timing_enabled:
            _mark_runtime("postprocess")
            runtime_timing["total"] = float(time.perf_counter() - runtime_start)
            self.last_runtime_timing = runtime_timing
        else:
            self.last_runtime_timing = {}

        self.debug_tensors = {
                "repair_gate": repair_gate.mean().detach(),
            "add_ratio": add_ratio.detach(),
            "add_prob_mean": add_prob.mean().detach(),
            "add_prob_max": add_prob.max().detach() if add_prob.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "add_priority_mean": add_priority.mean().detach(),
            "add_priority_max": add_priority.max().detach() if add_priority.numel() > 0 else pts_xyz.new_zeros(()).detach(),
            "add_candidate_ratio": pts_xyz.new_tensor(float(add_candidate_ratio)).detach(),
            "add_candidate_count": pts_xyz.new_tensor(float(add_k)).detach(),
            "learned_drop_ratio": learned_drop_ratio.mean().detach(),
            "learned_add_ratio": learned_add_ratio.mean().detach(),
            "learned_move_ratio": learned_move_ratio.mean().detach(),
            "operation_amount_consistency_loss": operation_amount_consistency_loss.detach(),
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
                "move_source_prior_mean": move_source_prior.mean().detach(),
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
                "add_prob_mean": add_prob.mean(),
                "add_prob_max": add_prob.max() if add_prob.numel() > 0 else pts_xyz.new_zeros(()),
                "add_priority_mean": add_priority.mean(),
                "add_priority_max": add_priority.max() if add_priority.numel() > 0 else pts_xyz.new_zeros(()),
                "add_count": add_count_value,
            "add_effective_count": add_effective_count_value,
            "add_candidate_ratio": float(add_candidate_ratio),
            "add_candidate_count": int(add_k),
            "learned_drop_ratio": learned_drop_ratio.mean(),
            "learned_add_ratio": learned_add_ratio.mean(),
            "learned_move_ratio": learned_move_ratio.mean(),
            "operation_amount_consistency_loss": operation_amount_consistency_loss,
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
            "primitive_delta": primitive_delta,
                "move_ratio": hard_move.mean(),
                "hard_move_count": hard_move_count_value,
                "move_score_mean": move_score.mean(),
                "move_source_prior_mean": move_source_prior.mean(),
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
        }
