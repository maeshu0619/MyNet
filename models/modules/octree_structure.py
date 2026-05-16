import torch
import torch.nn as nn
from contextlib import nullcontext

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
        ) or getattr(self.args, "trainORtest", "train") != "train"

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
            pts = pts_work.transpose(1, 2).contiguous()
            dist = torch.cdist(pts, pts)
            k = min(self.k_geo + 1, N)
            if k <= 1:
                return pts_xyz.new_zeros((B, 3, N))
            knn_idx = torch.topk(dist, k=k, largest=False, dim=-1).indices[:, :, 1:]
            gather_idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            pts_expand = pts.unsqueeze(1).expand(B, N, N, 3)
            knn = torch.gather(pts_expand, 2, gather_idx)
            center = pts.unsqueeze(2)
            diff = knn - center
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
            occupied = self._coords_membership(targets.reshape(-1, 3), unique_coords).view(N, -1)
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

    def forward(self, pts_xyz, final_w=None, coord_scale=None):
        if pts_xyz.ndim != 3 or pts_xyz.shape[1] != 3:
            raise ValueError("pts_xyz must have shape [B, 3, N]")

        input_dtype = pts_xyz.dtype
        work_xyz = pts_xyz.float() if pts_xyz.dtype in (torch.float16, torch.bfloat16) else pts_xyz
        qs_override = self._qs_override(work_xyz, coord_scale)

        with torch.no_grad():
            oct_ctx = self.proxy_octree_ctx.build_point_context(
                pts_xyz=work_xyz,
                ctx_level=self.ctx_level,
                final_w=final_w,
                qs_override=qs_override,
            )
            level_debug = self._aggregate_level_debug(work_xyz, qs_override)
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
        geo_stats = self._local_geometry_stats(work_xyz)
        quant_stats = self._quantized_voxel_stats(work_xyz, qs_override, snap_delta_norm)
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
        }
