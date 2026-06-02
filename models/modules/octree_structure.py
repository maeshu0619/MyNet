import torch
import torch.nn as nn
from contextlib import nullcontext
from ..utils.pointcloud.utils_repkpu import get_knn_pts
from ..utils.compression.proxy_octree import ProxyOctreeConfig, SoftOctreeRateProxy


class OctreeStructureAnalysis(nn.Module):
    """Build point-wise octree inefficiency descriptors."""

    def __init__(self, args, writer=None):
        super().__init__()
        self.args = args
        self.writer = writer
        self.ctx_level = int(getattr(args, "octree_ctx_level", 5))
        self.ctx_dim = max(int(getattr(args, "octree_ctx_dim", 8)), 8)
        self.qs = self._effective_qs(args)
        self.k_geo = max(int(getattr(args, "structure_geo_k", 8)), 3)
        self.geo_max_points = max(int(getattr(args, "structure_geo_max_points", 4096)), 0)
        self.feature_dim = 40
        self.max_depth = int(getattr(args, "proxy_max_depth", 12))
        self.diag_levels = self._parse_diag_levels(getattr(args, "octree_diag_levels", "4,6,8,10,12"))
        neighbor_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        self.register_buffer(
            "neighbor_offsets",
            torch.tensor(neighbor_offsets, dtype=torch.long),
            persistent=False,
        )

        proxy_cfg = ProxyOctreeConfig(
            max_depth=self.max_depth,
            qs=self.qs,
            round_tau=float(getattr(args, "proxy_round_tau", 0.12)),
            mass_to_occ_gain=float(getattr(args, "proxy_mass_to_occ_gain", 1.0)),
            ctx_dim=self.ctx_dim,
        )
        self.proxy_octree_ctx = SoftOctreeRateProxy(proxy_cfg)

    @staticmethod
    def _effective_qs(args):
        compress_key = (
            str(getattr(args, "compress", ""))
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )
        if compress_key == "sparsepcgc":
            return max(
                float(getattr(args, "sparsepcgc_effective_qs", 0.0))
                or float(getattr(args, "sparsepcgc_voxel_size", 1.0)) * float(getattr(args, "sparsepcgc_pos_quantscale", 1)),
                1e-9,
            )
        if compress_key in {"gpcc", "gpcctmc3"}:
            return max(float(getattr(args, "gpcc_effective_qs", getattr(args, "qs", 2.0))), 1e-9)
        return max(float(getattr(args, "qs", 2.0)), 1e-9)

    def _parse_diag_levels(self, raw_levels):
        if isinstance(raw_levels, (list, tuple)):
            tokens = raw_levels
        else:
            tokens = str(raw_levels).replace(" ", "").split(",")
        levels = []
        for token in tokens:
            if token == "":
                continue
            level = max(min(int(token), self.max_depth), 1)
            if level not in levels:
                levels.append(level)
        return levels or [min(4, self.max_depth), min(6, self.max_depth), min(8, self.max_depth), self.max_depth]

    def _qs_override(self, pts_xyz, coord_scale):
        if coord_scale is None:
            return pts_xyz.new_full((pts_xyz.shape[0],), self.qs)
        if torch.is_tensor(coord_scale):
            scale = coord_scale.to(device=pts_xyz.device, dtype=pts_xyz.dtype).reshape(-1)
            if scale.numel() == 1 and pts_xyz.shape[0] > 1:
                scale = scale.expand(pts_xyz.shape[0])
            return self.qs / scale.clamp_min(1e-9)
        return pts_xyz.new_full((pts_xyz.shape[0],), self.qs / max(float(coord_scale), 1e-9))

    @staticmethod
    def _normalize_pointwise(x, eps=1e-6):
        return x / (x.mean(dim=2, keepdim=True).detach().clamp_min(eps))

    @staticmethod
    def _safe_context(ctx, min_dim):
        if ctx.shape[1] >= min_dim:
            return ctx
        pad = ctx.new_zeros((ctx.shape[0], min_dim - ctx.shape[1], ctx.shape[2]))
        return torch.cat([ctx, pad], dim=1)

    def _should_collect_level_debug(self):
        return bool(
            getattr(self.args, "verbose_step_logs", False)
            and getattr(self.args, "_log_this_step", True)
        ) or bool(getattr(self.args, "_collect_octree_level_debug", False))

    def _grid_phase(self, pts_xyz, qs_override):
        B, _, _ = pts_xyz.shape
        qs = qs_override.to(device=pts_xyz.device, dtype=pts_xyz.dtype).view(B, 1, 1).clamp_min(1e-9)
        q = pts_xyz / qs
        q_round = torch.round(q)
        phase = q - torch.floor(q)
        center_delta = (q_round - q) * qs
        center_delta_norm = center_delta / qs
        return phase, center_delta, center_delta_norm

    def _local_geometry_stats(self, pts_xyz):
        B, _, N = pts_xyz.shape
        if self.geo_max_points <= 0 or N > self.geo_max_points or N <= 1:
            return pts_xyz.new_zeros((B, 3, N))
        autocast_ctx = torch.cuda.amp.autocast(enabled=False) if pts_xyz.is_cuda else nullcontext()
        with autocast_ctx:
            pts_work = pts_xyz.to(torch.float32)
            k = min(self.k_geo + 1, N)
            if k <= 1:
                return pts_xyz.new_zeros((B, 3, N))

            # get_knn_pts は pointops CUDA が使える場合はCUDA KNNを使う
            knn_all = get_knn_pts(
                k,
                pts_work,
                pts_work,
                return_idx=False,
            )  # [B, 3, N, k]

            # 先頭は自分自身である想定なので除外
            if knn_all.shape[-1] > 1:
                knn_all = knn_all[..., 1:]
            else:
                return pts_xyz.new_zeros((B, 3, N))

            center = pts_work.unsqueeze(-1)  # [B, 3, N, 1]
            diff_ch = knn_all - center       # [B, 3, N, k-1]
            diff = diff_ch.permute(0, 2, 3, 1).contiguous()  # [B, N, k-1, 3]
            mean_dist = torch.linalg.norm(diff, dim=-1).mean(dim=-1, keepdim=True).transpose(1, 2)
            density = self._normalize_pointwise(1.0 / mean_dist.clamp_min(1e-6)).clamp(0.0, 4.0) / 4.0
            cov = torch.matmul(diff.transpose(-1, -2), diff) / max(float(k - 1), 1.0)
            eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
            eig_sum = eig.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            curvature = (eig[:, :, 0:1] / eig_sum).transpose(1, 2).clamp(0.0, 1.0)
            anisotropy = ((eig[:, :, -1:] - eig[:, :, 0:1]) / eig[:, :, -1:].clamp_min(1e-12)).transpose(1, 2).clamp(0.0, 1.0)
            stats = torch.nan_to_num(
                torch.cat([density, curvature, anisotropy], dim=1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        return stats.to(dtype=pts_xyz.dtype)

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
        if query_coords.numel() == 0 or reference_coords.numel() == 0:
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

    def _quantized_voxel_stats(self, pts_xyz, qs_override, snap_delta_norm):
        B, _, N = pts_xyz.shape
        if N <= 0:
            return pts_xyz.new_zeros((B, 4, N))
        qs = qs_override.to(device=pts_xyz.device, dtype=pts_xyz.dtype).view(B, 1, 1).clamp_min(1e-9)
        coords = torch.round(pts_xyz / qs).to(torch.long)
        offsets = self.neighbor_offsets.to(device=pts_xyz.device, dtype=torch.long)
        stats = []
        for b in range(B):
            coord_b = coords[b].transpose(0, 1).contiguous()
            unique_coords, inverse = torch.unique(coord_b, dim=0, sorted=True, return_inverse=True)
            voxel_count = int(unique_coords.shape[0])
            if voxel_count <= 0:
                stats.append(pts_xyz.new_zeros((4, N)))
                continue
            counts = torch.bincount(inverse, minlength=voxel_count).to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            point_counts = counts.index_select(0, inverse).clamp_min(1.0)
            merge_pressure = ((point_counts - 1.0) / point_counts).view(1, N)
            density = self._normalize_pointwise(point_counts.view(1, 1, N)).view(1, N).clamp(0.0, 4.0) / 4.0
            targets = coord_b[:, None, :] + offsets.view(1, -1, 3)
            key_mins = unique_coords.amin(dim=0) - 1
            key_spans = (unique_coords.amax(dim=0) - unique_coords.amin(dim=0) + 3).to(torch.long).clamp_min(1)
            occupied_keys = torch.sort(self._coord_keys(unique_coords.to(torch.long), key_mins, key_spans)).values
            query_keys = self._coord_keys(targets.reshape(-1, 3).to(torch.long), key_mins, key_spans)
            pos = torch.searchsorted(occupied_keys, query_keys)
            in_bounds = pos < occupied_keys.numel()
            safe_pos = pos.clamp(max=max(int(occupied_keys.numel()) - 1, 0))
            occupied = (in_bounds & (occupied_keys[safe_pos] == query_keys)).view(N, -1)
            neighbor_empty = (1.0 - occupied.to(dtype=pts_xyz.dtype).mean(dim=1)).view(1, N)
            quant_residual = (
                torch.linalg.norm(snap_delta_norm[b], dim=0, keepdim=True) / (3.0 ** 0.5)
            ).clamp(0.0, 1.0)
            stats.append(torch.cat([merge_pressure, density, neighbor_empty, quant_residual], dim=0))
        return torch.stack(stats, dim=0).to(dtype=pts_xyz.dtype)

    def _level_octree_stats_single(self, pts_xyz, qs_value):
        if pts_xyz.shape[-1] <= 0:
            return []
        qs_value = max(float(qs_value), 1e-9)
        q = torch.round(pts_xyz / qs_value).to(torch.int64).transpose(0, 1).contiguous()
        stats = []
        for level in self.diag_levels:
            shift = max(int(self.max_depth - level), 0)
            cell = torch.div(q, 1 << shift, rounding_mode="floor") if shift > 0 else q
            uniq = torch.unique(cell, dim=0, sorted=False)
            occupied = int(uniq.shape[0])
            if occupied <= 0:
                stats.append(
                    {
                        "level": int(level),
                        "occupied": 0,
                        "parents": 0,
                        "single": 0,
                        "single_ratio": 0.0,
                        "mean_children": 0.0,
                        "std_children": 0.0,
                        "grid_fill": 0.0,
                    }
                )
                continue
            if level > 1:
                parent = torch.div(uniq, 2, rounding_mode="floor")
                _, inv = torch.unique(parent, dim=0, sorted=False, return_inverse=True)
                child_counts = torch.bincount(inv, minlength=int(inv.max().item()) + 1).to(torch.float32)
                parent_count = int(child_counts.numel())
                single_count = int((child_counts == 1).sum().item())
                mean_children = float(child_counts.mean().item())
                std_children = float(child_counts.std(unbiased=False).item())
            else:
                parent_count = 1
                single_count = occupied if occupied == 1 else 0
                mean_children = float(occupied)
                std_children = 0.0
            stats.append(
                {
                    "level": int(level),
                    "occupied": occupied,
                    "parents": parent_count,
                    "single": single_count,
                    "single_ratio": float(single_count) / max(float(parent_count), 1.0),
                    "mean_children": mean_children,
                    "std_children": std_children,
                    "grid_fill": float(occupied) / max(float(8 ** int(level)), 1.0),
                }
            )
        return stats

    def _aggregate_level_debug(self, pts_xyz, qs_override):
        if not self._should_collect_level_debug():
            return None
        per_batch = []
        for b in range(pts_xyz.shape[0]):
            per_batch.append(self._level_octree_stats_single(pts_xyz[b], float(qs_override[b].detach().cpu())))
        if not per_batch:
            return None
        summary = []
        for idx, level in enumerate(self.diag_levels):
            entries = [batch[idx] for batch in per_batch if idx < len(batch)]
            if not entries:
                continue
            summary.append(
                {
                    "level": int(level),
                    "occupied_mean": float(sum(item["occupied"] for item in entries) / len(entries)),
                    "parents_mean": float(sum(item["parents"] for item in entries) / len(entries)),
                    "single_mean": float(sum(item["single"] for item in entries) / len(entries)),
                    "single_ratio_mean": float(sum(item["single_ratio"] for item in entries) / len(entries)),
                    "mean_children_mean": float(sum(item["mean_children"] for item in entries) / len(entries)),
                    "std_children_mean": float(sum(item["std_children"] for item in entries) / len(entries)),
                    "grid_fill_mean": float(sum(item["grid_fill"] for item in entries) / len(entries)),
                }
            )
        return summary

    @staticmethod
    def _tree_tensor(tree, key, device, dtype=None):
        if tree is None or key not in tree:
            return None
        value = tree.get(key)
        if torch.is_tensor(value):
            out = value.detach().to(device=device)
        else:
            out = torch.as_tensor(value, device=device)
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out

    @staticmethod
    def _fit_point_rows(values, point_count: int):
        if values is None:
            return None
        values = values.reshape(-1, values.shape[-1])
        current = int(values.shape[0])
        if current == point_count:
            return values
        if current <= 0:
            return values.new_zeros((point_count, values.shape[-1]))
        if current > point_count:
            return values[:point_count]
        pad = values[-1:].expand(point_count - current, -1)
        return torch.cat([values, pad], dim=0)

    @staticmethod
    def _popcount_codes(codes, dtype):
        codes = codes.to(dtype=torch.long).reshape(-1)
        counts = torch.zeros_like(codes, dtype=dtype)
        for child in range(8):
            counts = counts + (((codes >> child) & 1).to(dtype=dtype))
        return counts

    def _neighbor_occupancy_from_global_coords(self, coords):
        if coords.numel() == 0:
            return coords.new_zeros((0,), dtype=torch.float32)
        coords = coords.to(dtype=torch.long)
        unique_coords = torch.unique(coords, dim=0, sorted=True)
        offsets = self.neighbor_offsets.to(device=coords.device, dtype=torch.long)
        targets = coords[:, None, :] + offsets.view(1, -1, 3)
        occupied = self._coords_membership(targets.reshape(-1, 3), unique_coords).view(coords.shape[0], -1)
        return occupied.to(dtype=torch.float32).mean(dim=1)

    def _quantized_voxel_stats_from_tree(self, pts_xyz, global_coords, qs_override, snap_delta_norm):
        B, _, N = pts_xyz.shape
        stats = []
        for b in range(B):
            coords_b = self._fit_point_rows(global_coords[b], N).to(device=pts_xyz.device, dtype=torch.long)
            unique_coords, inverse = torch.unique(coords_b, dim=0, sorted=True, return_inverse=True)
            voxel_count = int(unique_coords.shape[0])
            if voxel_count <= 0:
                stats.append(pts_xyz.new_zeros((4, N)))
                continue
            counts = torch.bincount(inverse, minlength=voxel_count).to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            point_counts = counts.index_select(0, inverse).clamp_min(1.0)
            merge_pressure = ((point_counts - 1.0) / point_counts).view(1, N)
            density = self._normalize_pointwise(point_counts.view(1, 1, N)).view(1, N).clamp(0.0, 4.0) / 4.0
            neighbor_occ = self._neighbor_occupancy_from_global_coords(coords_b).to(device=pts_xyz.device, dtype=pts_xyz.dtype).view(1, N)
            neighbor_empty = 1.0 - neighbor_occ
            quant_residual = (
                torch.linalg.norm(snap_delta_norm[b], dim=0, keepdim=True) / (3.0 ** 0.5)
            ).clamp(0.0, 1.0)
            stats.append(torch.cat([merge_pressure, density, neighbor_empty, quant_residual], dim=0))
        return torch.stack(stats, dim=0).to(dtype=pts_xyz.dtype)

    def _point_feature_voxel_key(self, pts_xyz, qs_override):
        B, _, N = pts_xyz.shape
        keys = []
        for b in range(B):
            qs_b = qs_override[b].to(device=pts_xyz.device, dtype=pts_xyz.dtype).clamp_min(1e-9)
            offset = pts_xyz[b].amin(dim=1, keepdim=True)
            coords = torch.round((pts_xyz[b] - offset) / qs_b).to(dtype=torch.long).transpose(0, 1).contiguous()
            if coords.numel() <= 0:
                keys.append(torch.empty((0,), device=pts_xyz.device, dtype=torch.long))
                continue
            span = (coords.amax(dim=0) - coords.amin(dim=0) + 1).clamp_min(1)
            shifted = coords - coords.amin(dim=0, keepdim=True)
            keys.append(shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2])
        return torch.stack(keys, dim=0) if keys else torch.empty((B, N), device=pts_xyz.device, dtype=torch.long)

    def _prebuilt_octree_context(self, pts_xyz, subtree_tree=None, full_octree_context=None):
        if subtree_tree is None:
            return None
        B, _, N = pts_xyz.shape
        if B != 1:
            return None
        device = pts_xyz.device
        dtype = pts_xyz.dtype
        coords = self._tree_tensor(subtree_tree, "global_voxel_coords", device, dtype=torch.long)
        if coords is None or coords.numel() <= 0:
            return None
        coords = self._fit_point_rows(coords, N)
        parent_coords = torch.div(coords, 2, rounding_mode="floor")
        unique_parents, inverse = torch.unique(parent_coords, dim=0, sorted=True, return_inverse=True)
        child_index = ((coords[:, 0] & 1) * 4 + (coords[:, 1] & 1) * 2 + (coords[:, 2] & 1)).to(torch.long)
        occupancy = torch.zeros((unique_parents.shape[0], 8), device=device, dtype=torch.bool)
        occupancy[inverse, child_index] = True
        pattern_weights = (2 ** torch.arange(8, device=device, dtype=torch.long)).view(1, 8)
        parent_pattern_code = (occupancy.to(torch.long) * pattern_weights).sum(dim=1)

        pattern_hist = torch.bincount(parent_pattern_code, minlength=256).to(dtype=dtype)
        parent_pattern_prob = pattern_hist.index_select(0, parent_pattern_code).clamp_min(1.0)
        parent_pattern_prob = parent_pattern_prob / parent_pattern_prob.sum().clamp_min(1.0)

        point_pattern_prob = parent_pattern_prob.index_select(0, inverse).view(1, 1, N)
        pattern_nll = -torch.log2(point_pattern_prob.clamp_min(torch.finfo(dtype).eps))
        pattern_nll = pattern_nll / 8.0
        child_counts = occupancy.sum(dim=1).to(dtype=dtype).clamp_min(1.0)
        point_child_counts = child_counts.index_select(0, inverse).view(1, 1, N)
        row_exist = pts_xyz.new_ones((1, 1, N))
        mean_occ = (point_child_counts / 8.0).clamp(0.0, 1.0)
        single_proxy = (point_child_counts <= 1.0).to(dtype=dtype)
        self_occ = point_pattern_prob.clamp(0.0, 1.0)
        bit_entropy = (torch.log2(point_child_counts).view(1, 1, N) / 3.0).clamp(0.0, 1.0)
        bit_entropy = torch.maximum(bit_entropy, pattern_nll.clamp(0.0, 1.0))
        sibling_occ = ((point_child_counts - 1.0) / 7.0).clamp(0.0, 1.0)
        neighbor_occ = self._neighbor_occupancy_from_global_coords(coords).to(device=device, dtype=dtype).view(1, 1, N)
        child_id = (child_index.to(dtype=dtype).view(1, 1, N) / 7.0).clamp(0.0, 1.0)

        context_codes = []
        if full_octree_context is not None:
            ancestor = self._tree_tensor(full_octree_context, "ancestor_occupancy_codes", device, dtype=torch.long)
            sibling = self._tree_tensor(full_octree_context, "sibling_occupancy_codes", device, dtype=torch.long)
            parent_code = full_octree_context.get("parent_occupancy_code", None)
            if ancestor is not None and ancestor.numel() > 0:
                context_codes.append(ancestor)
            if sibling is not None and sibling.numel() > 0:
                context_codes.append(sibling)
            if parent_code is not None:
                context_codes.append(torch.as_tensor([int(parent_code)], device=device, dtype=torch.long))
        if context_codes:
            codes = torch.cat([item.reshape(-1) for item in context_codes], dim=0)
            context_occ = (self._popcount_codes(codes, dtype=dtype).mean() / 8.0).clamp(0.0, 1.0)
            mean_occ = (0.75 * mean_occ + 0.25 * context_occ).clamp(0.0, 1.0)
            sibling_occ = (0.75 * sibling_occ + 0.25 * context_occ).clamp(0.0, 1.0)

        oct_ctx = torch.cat(
            [row_exist, mean_occ, single_proxy, self_occ, bit_entropy, sibling_occ, neighbor_occ, child_id],
            dim=1,
        )
        return oct_ctx

    @staticmethod
    def _missing_prebuilt_keys(subtree_tree, required_keys):
        if not isinstance(subtree_tree, dict):
            return list(required_keys)
        return [key for key in required_keys if key not in subtree_tree or subtree_tree.get(key) is None]

    def _prebuilt_level_debug(self, subtree_tree):
        if subtree_tree is None or not self._should_collect_level_debug():
            return None
        codes = subtree_tree.get("occupancy_codes")
        depths = subtree_tree.get("node_depths")
        if codes is None or depths is None:
            return None
        codes = torch.as_tensor(codes, dtype=torch.long)
        depths = torch.as_tensor(depths, dtype=torch.long)
        if codes.numel() <= 0:
            return None
        summary = []
        for level in self.diag_levels:
            mask = depths == int(level)
            if not bool(mask.any().item()):
                continue
            child_counts = self._popcount_codes(codes[mask], dtype=torch.float32)
            summary.append(
                {
                    "level": int(level),
                    "occupied_mean": float(child_counts.sum().item()),
                    "parents_mean": float(child_counts.numel()),
                    "single_mean": float((child_counts == 1).sum().item()),
                    "single_ratio_mean": float((child_counts == 1).to(torch.float32).mean().item()),
                    "mean_children_mean": float(child_counts.mean().item()),
                    "std_children_mean": float(child_counts.std(unbiased=False).item()),
                    "grid_fill_mean": 0.0,
                }
            )
        return summary

    def forward(
        self,
        pts_xyz,
        final_w=None,
        coord_scale=None,
        subtree_tree=None,
        full_octree_context=None,
        octree_input_mode="auto",
    ):
        if pts_xyz.ndim != 3 or pts_xyz.shape[1] != 3:
            raise ValueError("pts_xyz must have shape [B, 3, N]")

        input_dtype = pts_xyz.dtype
        work_xyz = pts_xyz.float() if pts_xyz.dtype in (torch.float16, torch.bfloat16) else pts_xyz
        qs_override = self._qs_override(work_xyz, coord_scale)
        requested_mode = str(octree_input_mode or "auto").strip().lower()
        if requested_mode == "auto" and subtree_tree is None:
            requested_mode = "full_cloud"
        prebuilt_required = requested_mode == "prebuilt_subtree_tree"
        required_prebuilt_keys = (
            "global_voxel_coords",
            "occupancy_codes",
            "node_depths",
            "global_morton_keys",
        )
        missing_prebuilt = self._missing_prebuilt_keys(subtree_tree, required_prebuilt_keys)
        if prebuilt_required and missing_prebuilt:
            raise ValueError(
                "octree_input_mode=prebuilt_subtree_tree requires subtree_tree metadata keys: "
                + ", ".join(required_prebuilt_keys)
                + f"; missing: {', '.join(missing_prebuilt)}"
            )
        prebuilt_ctx = self._prebuilt_octree_context(
            work_xyz,
            subtree_tree=subtree_tree,
            full_octree_context=full_octree_context,
        )
        if prebuilt_required and prebuilt_ctx is None:
            raise ValueError("octree_input_mode=prebuilt_subtree_tree could not build a prebuilt octree context.")
        if prebuilt_ctx is None and requested_mode not in {"auto", "full_cloud", "local_recomputed", "debug_local_recomputed"}:
            raise ValueError(f"Unsupported octree_input_mode without prebuilt metadata: {octree_input_mode}")
        if (
            prebuilt_ctx is None
            and requested_mode not in {"full_cloud", "debug_local_recomputed"}
            and not bool(getattr(self.args, "allow_local_octree_recompute", False))
        ):
            raise ValueError(
                "Local Octree recompute is disabled. Use octree_input_mode=full_cloud/debug_local_recomputed "
                "or provide prebuilt subtree_tree metadata."
            )

        with torch.no_grad():
            if prebuilt_ctx is not None:
                oct_ctx = prebuilt_ctx
                level_debug = self._prebuilt_level_debug(subtree_tree)
                effective_octree_input_mode = "prebuilt_subtree_tree"
            else:
                oct_ctx = self.proxy_octree_ctx.build_point_context(
                    pts_xyz=work_xyz,
                    ctx_level=self.ctx_level,
                    final_w=final_w,
                    qs_override=qs_override,
                )
                level_debug = self._aggregate_level_debug(work_xyz, qs_override)
                effective_octree_input_mode = "local_recomputed"
        oct_ctx = self._safe_context(oct_ctx.to(dtype=work_xyz.dtype), 8)

        row_exist = oct_ctx[:, 0:1, :]
        mean_occ = oct_ctx[:, 1:2, :]
        single_proxy = oct_ctx[:, 2:3, :]
        self_occ = oct_ctx[:, 3:4, :]
        bit_entropy = oct_ctx[:, 4:5, :]
        sibling_occ = oct_ctx[:, 5:6, :]
        neighbor_occ = oct_ctx[:, 6:7, :]
        child_id = oct_ctx[:, 7:8, :]

        phase, snap_delta, snap_delta_norm = self._grid_phase(work_xyz, qs_override)
        point_feature_voxel_key = None
        geo_stats = self._local_geometry_stats(work_xyz)
        tree_coords = None
        if prebuilt_ctx is not None:
            raw_tree_coords = self._tree_tensor(subtree_tree, "global_voxel_coords", work_xyz.device, dtype=torch.long)
            if raw_tree_coords is not None:
                tree_coords = raw_tree_coords.view(1, -1, 3)
        if tree_coords is not None:
            quant_stats = self._quantized_voxel_stats_from_tree(work_xyz, tree_coords, qs_override, snap_delta_norm)
            point_feature_voxel_key = self._tree_tensor(subtree_tree, "global_morton_keys", pts_xyz.device, dtype=torch.long)
            if point_feature_voxel_key is not None:
                point_feature_voxel_key = self._fit_point_rows(point_feature_voxel_key.reshape(-1, 1), pts_xyz.shape[-1]).reshape(1, -1)
        else:
            if prebuilt_required:
                raise ValueError("prebuilt_subtree_tree mode requires _quantized_voxel_stats_from_tree().")
            quant_stats = self._quantized_voxel_stats(work_xyz, qs_override, snap_delta_norm)
            point_feature_voxel_key = self._point_feature_voxel_key(work_xyz, qs_override)
        local_density = geo_stats[:, 0:1, :]
        local_curvature = geo_stats[:, 1:2, :]
        local_anisotropy = geo_stats[:, 2:3, :]
        quant_merge = quant_stats[:, 0:1, :]
        quant_density = quant_stats[:, 1:2, :]
        quant_empty = quant_stats[:, 2:3, :]
        quant_residual = quant_stats[:, 3:4, :]
        phase_centered = phase - 0.5
        radius = torch.linalg.norm(work_xyz, dim=1, keepdim=True)
        abs_xyz = work_xyz.abs()

        sparse_proxy = ((1.0 - sibling_occ) * (1.0 - neighbor_occ)).clamp(0.0, 1.0)
        eps = torch.finfo(work_xyz.dtype).eps
        occupancy_nll = -torch.log2(self_occ.clamp_min(eps))
        occupancy_nll = self._normalize_pointwise(occupancy_nll).clamp(0.0, 4.0) / 4.0
        lowprob_proxy = self._normalize_pointwise(
            occupancy_nll * (0.5 + bit_entropy) * (0.5 + sparse_proxy)
        ).clamp(0.0, 4.0) / 4.0
        node_proxy = self._normalize_pointwise(
            row_exist + mean_occ + sparse_proxy + 0.25 * single_proxy
        ).clamp(0.0, 4.0) / 4.0
        context_proxy = self._normalize_pointwise(
            bit_entropy * (0.5 + torch.abs(child_id - 0.5)) * (1.0 + sparse_proxy)
        ).clamp(0.0, 4.0) / 4.0
        quant_proxy = self._normalize_pointwise(
            0.45 * quant_merge
            + 0.25 * quant_empty
            + 0.20 * quant_density
            + 0.10 * quant_residual
        ).clamp(0.0, 4.0) / 4.0
        density_outlier = (1.0 - local_density.clamp(0.0, 1.0))
        outlier_proxy = (sparse_proxy * (1.0 - self_occ).clamp(0.0, 1.0) * (1.0 - neighbor_occ).clamp(0.0, 1.0))
        outlier_proxy = outlier_proxy + 0.5 * density_outlier
        outlier_proxy = self._normalize_pointwise(
            outlier_proxy + 0.05 * radius / radius.mean(dim=2, keepdim=True).detach().clamp_min(eps)
        )
        outlier_proxy = outlier_proxy.clamp(0.0, 4.0) / 4.0
        shape_proxy = (
            neighbor_occ * (1.0 - bit_entropy.clamp(0.0, 1.0))
            + 0.25 * self_occ
            + 0.50 * local_curvature
            + 0.25 * local_anisotropy
        ).clamp(0.0, 1.0)

        cause_targets_raw = torch.cat(
            [
                node_proxy,
                single_proxy.clamp(0.0, 1.0),
                lowprob_proxy.clamp(0.0, 1.0),
                context_proxy.clamp(0.0, 1.0),
                quant_proxy.clamp(0.0, 1.0),
                sparse_proxy.clamp(0.0, 1.0),
                outlier_proxy.clamp(0.0, 1.0),
                shape_proxy.clamp(0.0, 1.0),
            ],
            dim=1,
        )
        cause_targets = (
            cause_targets_raw
            / cause_targets_raw.sum(dim=1, keepdim=True).clamp_min(1e-6)
        ).detach()

        feature = torch.cat(
            [
                work_xyz,
                abs_xyz,
                radius,
                phase,
                phase_centered,
                snap_delta_norm.clamp(-2.0, 2.0),
                geo_stats,
                quant_stats,
                oct_ctx[:, :8, :],
                cause_targets,
                sparse_proxy,
            ],
            dim=1,
        )

        if feature.shape[1] != self.feature_dim:
            raise RuntimeError(f"Octree feature dim mismatch: {feature.shape[1]} != {self.feature_dim}")

        return {
            "features": feature.to(dtype=input_dtype),
            "cause_targets": cause_targets.to(dtype=input_dtype),
            "oct_ctx": oct_ctx.to(dtype=input_dtype),
            "snap_delta": snap_delta.to(dtype=input_dtype),
            "geo_stats": geo_stats.to(dtype=input_dtype),
            "quant_stats": quant_stats.to(dtype=input_dtype),
            "qs_override": qs_override.to(dtype=input_dtype),
            "single_proxy": single_proxy.to(dtype=input_dtype),
            "node_proxy": node_proxy.to(dtype=input_dtype),
            "lowprob_proxy": lowprob_proxy.to(dtype=input_dtype),
            "context_proxy": context_proxy.to(dtype=input_dtype),
            "quant_proxy": quant_proxy.to(dtype=input_dtype),
            "sparse_proxy": sparse_proxy.to(dtype=input_dtype),
            "outlier_proxy": outlier_proxy.to(dtype=input_dtype),
            "occupancy_nll_proxy": occupancy_nll.to(dtype=input_dtype),
            "shape_proxy": shape_proxy.to(dtype=input_dtype),
            "level_debug": level_debug,
            "octree_input_mode": effective_octree_input_mode,
            "octree_input_mode_requested": requested_mode,
            "structural_voxel_mode": "global_context" if prebuilt_ctx is not None else "local_recomputed",
            "point_feature_voxel_mode": "global_context" if prebuilt_ctx is not None else "local_xyz",
            "local_recomputed": prebuilt_ctx is None,
            "structural_voxel_key": self._tree_tensor(subtree_tree, "global_morton_keys", pts_xyz.device, dtype=torch.long)
            if prebuilt_ctx is not None
            else None,
            "point_feature_voxel_key": point_feature_voxel_key,
        }
