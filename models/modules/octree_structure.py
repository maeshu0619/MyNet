import torch
import torch.nn as nn
from contextlib import nullcontext
from ..utils.pointcloud.utils_repkpu import get_knn_pts
from ..utils.compression.proxy_octree import ProxyOctreeConfig, SoftOctreeRateProxy
from .heuristic_guidance import build_heuristic_guidance


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

    def _grid_phase(self, pts_xyz, qs_override, global_offset=None):
        B, _, _ = pts_xyz.shape
        qs = qs_override.to(device=pts_xyz.device, dtype=pts_xyz.dtype).view(B, 1, 1).clamp_min(1e-9)

        if global_offset is None:
            offset = pts_xyz.new_zeros((B, 3, 1))
        else:
            if not torch.is_tensor(global_offset):
                global_offset = torch.as_tensor(global_offset, device=pts_xyz.device, dtype=pts_xyz.dtype)
            offset = global_offset.to(device=pts_xyz.device, dtype=pts_xyz.dtype)

            if offset.ndim == 1 and offset.numel() == 3:
                offset = offset.view(1, 3, 1)
            elif offset.ndim == 2 and offset.shape[-1] == 3:
                offset = offset.view(-1, 3, 1)
            elif offset.ndim == 2 and offset.shape[0] == 3:
                offset = offset.unsqueeze(0)
            elif offset.ndim == 3 and offset.shape[1] == 3:
                offset = offset[:, :, :1]
            else:
                raise ValueError(f"global_offset has invalid shape: {tuple(offset.shape)}")

            if offset.shape[0] == 1 and B > 1:
                offset = offset.expand(B, -1, -1)

        q = (pts_xyz - offset) / qs
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

    def _neighbor_occupancy_chunked(self, coords, unique_coords=None):
        """Compute the exact 26-neighbour occupancy with bounded workspace.

        The previous implementation materialised all ``N * 26 * 3`` int64
        coordinates and then copied them again while building membership keys.
        Chunking changes only evaluation order: every queried coordinate and
        the float32 mean over the same 26 booleans remain identical.
        """
        if coords.numel() == 0:
            return coords.new_zeros((0,), dtype=torch.float32)
        coords = coords.to(dtype=torch.long)
        if unique_coords is None:
            unique_coords = torch.unique(coords, dim=0, sorted=True)
        else:
            unique_coords = unique_coords.to(device=coords.device, dtype=torch.long)

        offsets = self.neighbor_offsets.to(device=coords.device, dtype=torch.long)
        key_mins = unique_coords.amin(dim=0) - 1
        key_spans = (
            unique_coords.amax(dim=0) - unique_coords.amin(dim=0) + 3
        ).to(torch.long).clamp_min(1)
        occupied_keys = torch.sort(
            self._coord_keys(unique_coords, key_mins, key_spans)
        ).values
        chunk_size = max(
            int(getattr(self.args, "structure_neighbor_query_chunk", 32768)),
            1,
        )
        result = torch.empty(
            (coords.shape[0],), device=coords.device, dtype=torch.float32
        )
        for start in range(0, int(coords.shape[0]), chunk_size):
            end = min(start + chunk_size, int(coords.shape[0]))
            targets = coords[start:end, None, :] + offsets.view(1, -1, 3)
            query_keys = self._coord_keys(
                targets.reshape(-1, 3), key_mins, key_spans
            )
            pos = torch.searchsorted(occupied_keys, query_keys)
            in_bounds = pos < occupied_keys.numel()
            safe_pos = pos.clamp(max=max(int(occupied_keys.numel()) - 1, 0))
            occupied = (
                in_bounds & (occupied_keys[safe_pos] == query_keys)
            ).view(end - start, -1)
            result[start:end] = occupied.to(torch.float32).mean(dim=1)
        return result

    def _quantized_voxel_stats(self, pts_xyz, qs_override, snap_delta_norm):
        B, _, N = pts_xyz.shape
        if N <= 0:
            return pts_xyz.new_zeros((B, 4, N))
        qs = qs_override.to(device=pts_xyz.device, dtype=pts_xyz.dtype).view(B, 1, 1).clamp_min(1e-9)
        coords = torch.round(pts_xyz / qs).to(torch.long)
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
            neighbor_occ = self._neighbor_occupancy_chunked(
                coord_b, unique_coords=unique_coords
            ).to(dtype=pts_xyz.dtype)
            neighbor_empty = (1.0 - neighbor_occ).view(1, N)
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

    def _build_node_voxel_descriptor(
        self,
        pts_xyz,
        feature,
        oct_ctx,
        subtree_tree=None,
        full_octree_context=None,
        point_feature_voxel_key=None,
        prebuilt_ctx=None,
        leaf_pattern_diag=None,
    ):
        if not bool(getattr(self.args, "octree_structure_node_descriptor", True)):
            return None

        desc = {
            "node_features": feature,
            "node_mask": torch.ones(
                (pts_xyz.shape[0], pts_xyz.shape[-1]),
                device=pts_xyz.device,
                dtype=torch.bool,
            ),
            "point_feature_voxel_key": point_feature_voxel_key,
            "global_qs": None,
            "global_offset": None,
            "source": "local_xyz",
        }

        context = subtree_tree if isinstance(subtree_tree, dict) else full_octree_context
        if isinstance(context, dict):
            if "global_qs" in context:
                desc["global_qs"] = context.get("global_qs")
            if "global_offset" in context:
                desc["global_offset"] = context.get("global_offset")

        if prebuilt_ctx is not None and isinstance(subtree_tree, dict):
            coords_raw = self._tree_tensor(subtree_tree, "global_voxel_coords", pts_xyz.device, dtype=torch.long)
            if coords_raw is not None:
                coords_n3 = self._normalize_global_coords_n3(
                    coords_raw,
                    point_count=pts_xyz.shape[-1],
                    device=pts_xyz.device,
                )
                if coords_n3 is not None:
                    # node descriptor側は [B,3,N] 形式で保持する。
                    desc["voxel_coords"] = coords_n3.transpose(0, 1).contiguous().unsqueeze(0)
                    desc["source"] = "prebuilt_subtree_tree"

            for tree_key, out_key in (
                ("node_depths", "node_depth"),
                ("occupancy_codes", "occupancy_code"),
                ("parent_ids", "parent_id"),
                ("child_indices", "child_index"),
                ("sibling_occupancy", "sibling_occupancy"),
                ("global_morton_keys", "global_morton_keys"),
            ):
                value = self._tree_tensor(subtree_tree, tree_key, pts_xyz.device)
                if value is not None:
                    desc[out_key] = value
        else:
            desc["voxel_coords"] = None

        if oct_ctx is not None and torch.is_tensor(oct_ctx):
            desc["oct_ctx"] = oct_ctx

        # Section1:
        # 後続のSectionでActuatorやCostAttributionへ渡せるよう、
        # leaf pattern診断をdescriptorにも保持する。
        if isinstance(leaf_pattern_diag, dict):
            desc["leaf_pattern_diag"] = leaf_pattern_diag
            desc["leaf_pattern_available"] = bool(leaf_pattern_diag.get("available", False))
            desc["leaf_parent_pattern_code"] = leaf_pattern_diag.get("parent_pattern_code", None)
            desc["leaf_child_slot"] = leaf_pattern_diag.get("child_slot", None)
            desc["leaf_parent_pattern_frequency"] = leaf_pattern_diag.get("parent_pattern_frequency", None)
            desc["leaf_parent_pattern_nll"] = leaf_pattern_diag.get("parent_pattern_nll", None)

        return desc

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
    def _normalize_global_coords_n3(coords, point_count=None, device=None):
        """
        global_voxel_coords を [N, 3] に正規化する。
        受け付ける形は [N,3], [3,N], [B,N,3], [B,3,N] である。
        現在のOctreeStructureAnalysisはB=1前提でprebuilt contextを作る。
        """
        if coords is None:
            return None

        if not torch.is_tensor(coords):
            coords = torch.as_tensor(coords)

        if device is not None:
            coords = coords.to(device=device)

        coords = coords.to(dtype=torch.long)

        if coords.ndim == 2:
            if coords.shape[1] == 3:
                out = coords.contiguous()
            elif coords.shape[0] == 3:
                out = coords.transpose(0, 1).contiguous()
            else:
                return None

        elif coords.ndim == 3:
            # B=1のみ対応。B>1は可変長subtreeと衝突しやすいため明示的に先頭を使う。
            if coords.shape[0] != 1:
                return None

            if coords.shape[2] == 3:
                # [1, N, 3]
                out = coords[0].contiguous()
            elif coords.shape[1] == 3:
                # [1, 3, N]
                out = coords[0].transpose(0, 1).contiguous()
            else:
                return None
        else:
            return None

        if point_count is not None:
            point_count = int(point_count)
            current = int(out.shape[0])

            if current == point_count:
                return out

            if current <= 0:
                return out.new_zeros((point_count, 3))

            if current > point_count:
                return out[:point_count].contiguous()

            pad = out[-1:].expand(point_count - current, 3)
            return torch.cat([out, pad], dim=0).contiguous()

        return out.contiguous()
    
    @staticmethod
    def _stable_voxel_keys_from_coords_n3(coords_n3):
        """
        Phase4:
        global_morton_keys が無い場合でも、
        full cloud canonical global_voxel_coords から安定したkeyを作る。
        """
        if coords_n3 is None:
            return None
        if not torch.is_tensor(coords_n3):
            coords_n3 = torch.as_tensor(coords_n3)
        if coords_n3.ndim != 2 or coords_n3.shape[1] != 3:
            return None

        coords_n3 = coords_n3.to(dtype=torch.long)
        return (
            coords_n3[:, 0] * 73856093
            + coords_n3[:, 1] * 19349663
            + coords_n3[:, 2] * 83492791
        ).view(1, -1).contiguous()

    def _empty_leaf_pattern_diagnosis(self, pts_xyz, reason="disabled"):
        """
        Section1:
        leaf pattern診断が使えない場合でも、後段が同じkeyで参照できるように空診断を返す。
        ここでは学習挙動を変えないため、すべてdebug/将来拡張用の値として保持する。
        """
        B, _, N = pts_xyz.shape
        device = pts_xyz.device
        dtype = pts_xyz.dtype

        return {
            "available": False,
            "reason": str(reason),
            "source": "none",
            "voxel_coords": None,
            "parent_coords": None,
            "child_slot": torch.full((B, N), -1, device=device, dtype=torch.long),
            "parent_pattern_code": torch.zeros((B, N), device=device, dtype=torch.long),
            "parent_child_count": torch.zeros((B, N), device=device, dtype=dtype),
            "parent_pattern_frequency": torch.zeros((B, N), device=device, dtype=dtype),
            "parent_pattern_nll": torch.zeros((B, N), device=device, dtype=dtype),
            "unique_parent_count": 0,
            "unique_pattern_count": 0,
            "mean_child_count": 0.0,
            "single_child_parent_ratio": 0.0,
            "max_pattern_frequency": 0.0,

            # Section2:
            # Delete/Add/Move候補ごとのpattern頻度改善・NLL改善。
            "delete_pattern_gain": torch.zeros((B, N), device=device, dtype=dtype),
            "add_pattern_gain": torch.zeros((B, N), device=device, dtype=dtype),
            "move_pattern_gain": torch.zeros((B, N), device=device, dtype=dtype),
            "delete_nll_gain": torch.zeros((B, N), device=device, dtype=dtype),
            "add_nll_gain": torch.zeros((B, N), device=device, dtype=dtype),
            "move_nll_gain": torch.zeros((B, N), device=device, dtype=dtype),
            "delete_valid_mask": torch.zeros((B, N), device=device, dtype=torch.bool),
            "add_valid_mask": torch.zeros((B, N), device=device, dtype=torch.bool),
            "move_valid_mask": torch.zeros((B, N), device=device, dtype=torch.bool),
            "best_add_child_slot": torch.full((B, N), -1, device=device, dtype=torch.long),
            "best_move_target_child_slot": torch.full((B, N), -1, device=device, dtype=torch.long),
            "best_operation_hint": torch.zeros((B, N), device=device, dtype=torch.long),
            "delete_gain_mean": 0.0,
            "add_gain_mean": 0.0,
            "move_gain_mean": 0.0,
            "high_gain_candidate_ratio": 0.0,
            "candidate_available": False,

            # Section3:
            "leaf_feature_integration_used": False,
            "leaf_feature_best_gain_mean": 0.0,
            "leaf_feature_best_gain_max": 0.0,
        }

    def _leaf_pattern_diagnosis_from_coords(
        self,
        pts_xyz,
        coords_b3n,
        *,
        source="global_voxel_coords",
    ):
        """
        Section1:
        canonical voxel coordsから、各occupied voxelが属するparent nodeとchild slotを求める。
        さらにparentごとの8-child occupancy patternを作る。

        ここで得る値:
        - parent_coords
        - child_slot
        - parent_pattern_code
        - parent_child_count
        - parent_pattern_frequency
        - parent_pattern_nll

        注意:
        Section1では損失や操作選択には使わない。
        まずdebug可能な診断情報として外へ出すだけである。
        """
        if not bool(getattr(self.args, "leaf_pattern_diagnosis", True)):
            return self._empty_leaf_pattern_diagnosis(
                pts_xyz,
                reason="disabled_by_args",
            )

        if coords_b3n is None or not torch.is_tensor(coords_b3n):
            return self._empty_leaf_pattern_diagnosis(
                pts_xyz,
                reason="coords_missing",
            )

        B, _, N = pts_xyz.shape
        device = pts_xyz.device
        dtype = pts_xyz.dtype
        collect_debug_scalars = bool(
            getattr(self.args, "leaf_pattern_diagnosis_debug", False)
            or (
                getattr(self.args, "verbose_step_logs", False)
                and getattr(self.args, "_log_this_step", True)
            )
            or getattr(self.args, "_collect_structure_debug", False)
        )

        if coords_b3n.ndim == 2:
            coords_n3 = self._normalize_global_coords_n3(
                coords_b3n,
                point_count=N,
                device=device,
            )
            if coords_n3 is None:
                return self._empty_leaf_pattern_diagnosis(
                    pts_xyz,
                    reason="coords_normalize_failed",
                )
            coords_b3n = coords_n3.transpose(0, 1).contiguous().unsqueeze(0)

        elif coords_b3n.ndim == 3:
            if coords_b3n.shape[1] == 3:
                coords_b3n = coords_b3n.to(device=device, dtype=torch.long).contiguous()
            elif coords_b3n.shape[2] == 3:
                coords_b3n = coords_b3n.to(device=device, dtype=torch.long).permute(0, 2, 1).contiguous()
            else:
                return self._empty_leaf_pattern_diagnosis(
                    pts_xyz,
                    reason=f"invalid_coords_shape={tuple(coords_b3n.shape)}",
                )
        else:
            return self._empty_leaf_pattern_diagnosis(
                pts_xyz,
                reason=f"invalid_coords_ndim={coords_b3n.ndim}",
            )

        if coords_b3n.shape[0] == 1 and B > 1:
            coords_b3n = coords_b3n.expand(B, -1, -1).contiguous()

        if coords_b3n.shape[0] != B:
            return self._empty_leaf_pattern_diagnosis(
                pts_xyz,
                reason=f"batch_mismatch={coords_b3n.shape[0]}!={B}",
            )

        if coords_b3n.shape[-1] != N:
            fixed = []
            for b in range(B):
                fixed_b = self._fit_point_rows(
                    coords_b3n[b].transpose(0, 1).contiguous(),
                    N,
                )
                fixed.append(fixed_b.transpose(0, 1).contiguous())
            coords_b3n = torch.stack(fixed, dim=0).to(device=device, dtype=torch.long)

        child_slot_list = []
        parent_code_list = []
        parent_count_list = []
        parent_freq_list = []
        parent_nll_list = []
        parent_coords_point_list = []
        delete_pattern_gain_list = []
        add_pattern_gain_list = []
        move_pattern_gain_list = []
        delete_nll_gain_list = []
        add_nll_gain_list = []
        move_nll_gain_list = []
        delete_valid_mask_list = []
        add_valid_mask_list = []
        move_valid_mask_list = []
        best_add_child_slot_list = []
        best_move_target_child_slot_list = []
        best_operation_hint_list = []

        delete_gain_mean_values = []
        add_gain_mean_values = []
        move_gain_mean_values = []
        high_gain_candidate_ratio_values = []

        unique_parent_count_max = 0
        unique_pattern_count_max = 0
        mean_child_count_values = []
        single_child_ratio_values = []
        max_pattern_frequency_values = []

        pattern_weights = (2 ** torch.arange(8, device=device, dtype=torch.long)).view(1, 8)

        for b in range(B):
            coords_n3 = coords_b3n[b].transpose(0, 1).contiguous().to(dtype=torch.long)

            if coords_n3.numel() <= 0:
                child_slot_list.append(torch.full((N,), -1, device=device, dtype=torch.long))
                parent_code_list.append(torch.zeros((N,), device=device, dtype=torch.long))
                parent_count_list.append(torch.zeros((N,), device=device, dtype=dtype))
                parent_freq_list.append(torch.zeros((N,), device=device, dtype=dtype))
                parent_nll_list.append(torch.zeros((N,), device=device, dtype=dtype))
                parent_coords_point_list.append(torch.zeros((3, N), device=device, dtype=torch.long))

                delete_pattern_gain_list.append(torch.zeros((N,), device=device, dtype=dtype))
                add_pattern_gain_list.append(torch.zeros((N,), device=device, dtype=dtype))
                move_pattern_gain_list.append(torch.zeros((N,), device=device, dtype=dtype))
                delete_nll_gain_list.append(torch.zeros((N,), device=device, dtype=dtype))
                add_nll_gain_list.append(torch.zeros((N,), device=device, dtype=dtype))
                move_nll_gain_list.append(torch.zeros((N,), device=device, dtype=dtype))
                delete_valid_mask_list.append(torch.zeros((N,), device=device, dtype=torch.bool))
                add_valid_mask_list.append(torch.zeros((N,), device=device, dtype=torch.bool))
                move_valid_mask_list.append(torch.zeros((N,), device=device, dtype=torch.bool))
                best_add_child_slot_list.append(torch.full((N,), -1, device=device, dtype=torch.long))
                best_move_target_child_slot_list.append(torch.full((N,), -1, device=device, dtype=torch.long))
                best_operation_hint_list.append(torch.zeros((N,), device=device, dtype=torch.long))
                continue

            parent_coords = torch.div(coords_n3, 2, rounding_mode="floor")
            parent_cache = getattr(self, "_active_canonical_parent_cache", None)
            can_reuse_parent_partition = bool(
                b == 0
                and B == 1
                and isinstance(parent_cache, dict)
                and torch.is_tensor(parent_cache.get("unique_parents"))
                and torch.is_tensor(parent_cache.get("inverse"))
                and int(parent_cache["inverse"].numel()) == int(coords_n3.shape[0])
            )
            if can_reuse_parent_partition:
                # The cache was produced earlier in this same forward from the
                # same canonical global_voxel_coords.  Only the partition is
                # shared; child-slot convention and all following arithmetic
                # remain exactly as before.
                unique_parents = parent_cache["unique_parents"]
                inverse = parent_cache["inverse"]
            else:
                unique_parents, inverse = torch.unique(
                    parent_coords,
                    dim=0,
                    sorted=True,
                    return_inverse=True,
                )

            child_slot = (
                (coords_n3[:, 0] & 1)
                + 2 * (coords_n3[:, 1] & 1)
                + 4 * (coords_n3[:, 2] & 1)
            ).to(dtype=torch.long)

            occupancy = torch.zeros(
                (unique_parents.shape[0], 8),
                device=device,
                dtype=torch.bool,
            )
            occupancy[inverse, child_slot] = True

            parent_pattern_code_unique = (
                occupancy.to(dtype=torch.long) * pattern_weights
            ).sum(dim=1)

            child_count_unique = occupancy.sum(dim=1).to(dtype=dtype)
            pattern_hist = torch.bincount(
                parent_pattern_code_unique,
                minlength=256,
            ).to(device=device, dtype=dtype)

            parent_count = max(int(unique_parents.shape[0]), 1)
            parent_pattern_frequency_unique = (
                pattern_hist.index_select(0, parent_pattern_code_unique)
                / float(parent_count)
            ).clamp(0.0, 1.0)

            parent_pattern_nll_unique = -torch.log2(
                parent_pattern_frequency_unique.clamp_min(torch.finfo(dtype).eps)
            )

            parent_coords_point = unique_parents.index_select(0, inverse).transpose(0, 1).contiguous()
            parent_code_point = parent_pattern_code_unique.index_select(0, inverse)
            child_count_point = child_count_unique.index_select(0, inverse)
            pattern_freq_point = parent_pattern_frequency_unique.index_select(0, inverse)
            pattern_nll_point = parent_pattern_nll_unique.index_select(0, inverse)
            candidate_scores = self._leaf_pattern_candidate_scores_single(
                parent_pattern_code_unique,
                child_count_unique,
                child_slot,
                inverse,
                device=device,
                dtype=dtype,
            )

            child_slot_list.append(child_slot)
            parent_code_list.append(parent_code_point)
            parent_count_list.append(child_count_point)
            parent_freq_list.append(pattern_freq_point)
            parent_nll_list.append(pattern_nll_point)
            parent_coords_point_list.append(parent_coords_point)
            delete_pattern_gain_list.append(candidate_scores["delete_pattern_gain"])
            add_pattern_gain_list.append(candidate_scores["add_pattern_gain"])
            move_pattern_gain_list.append(candidate_scores["move_pattern_gain"])
            delete_nll_gain_list.append(candidate_scores["delete_nll_gain"])
            add_nll_gain_list.append(candidate_scores["add_nll_gain"])
            move_nll_gain_list.append(candidate_scores["move_nll_gain"])
            delete_valid_mask_list.append(candidate_scores["delete_valid_mask"])
            add_valid_mask_list.append(candidate_scores["add_valid_mask"])
            move_valid_mask_list.append(candidate_scores["move_valid_mask"])
            best_add_child_slot_list.append(candidate_scores["best_add_child_slot"])
            best_move_target_child_slot_list.append(candidate_scores["best_move_target_child_slot"])
            best_operation_hint_list.append(candidate_scores["best_operation_hint"])

            unique_parent_count_max = max(unique_parent_count_max, int(unique_parents.shape[0]))
            if collect_debug_scalars:
                # 以下はログ専用であり、学習TensorやHeuristic順位には使わない。
                # 通常StepではGPU同期を避け、ログ対象Stepだけ集計する。
                unique_pattern_count_max = max(
                    unique_pattern_count_max,
                    int(torch.unique(parent_pattern_code_unique).numel()),
                )
                mean_child_count_values.append(float(child_count_unique.detach().float().mean().cpu()))
                single_child_ratio_values.append(float((child_count_unique <= 1.0).detach().float().mean().cpu()))
                max_pattern_frequency_values.append(float(parent_pattern_frequency_unique.detach().float().max().cpu()))
                gain_threshold = float(getattr(self.args, "leaf_pattern_candidate_gain_threshold", 0.05))
                delete_gain = candidate_scores["delete_nll_gain"].detach().float()
                add_gain = candidate_scores["add_nll_gain"].detach().float()
                move_gain = candidate_scores["move_nll_gain"].detach().float()
                best_gain = torch.maximum(torch.maximum(delete_gain, add_gain), move_gain)
                delete_gain_mean_values.append(float(delete_gain.mean().cpu()) if delete_gain.numel() > 0 else 0.0)
                add_gain_mean_values.append(float(add_gain.mean().cpu()) if add_gain.numel() > 0 else 0.0)
                move_gain_mean_values.append(float(move_gain.mean().cpu()) if move_gain.numel() > 0 else 0.0)
                high_gain_candidate_ratio_values.append(
                    float((best_gain > gain_threshold).to(torch.float32).mean().cpu())
                    if best_gain.numel() > 0
                    else 0.0
                )

        child_slot_out = torch.stack(child_slot_list, dim=0)
        parent_code_out = torch.stack(parent_code_list, dim=0)
        parent_count_out = torch.stack(parent_count_list, dim=0)
        parent_freq_out = torch.stack(parent_freq_list, dim=0)
        parent_nll_out = torch.stack(parent_nll_list, dim=0)
        parent_coords_out = torch.stack(parent_coords_point_list, dim=0)
        delete_pattern_gain_out = torch.stack(delete_pattern_gain_list, dim=0)
        add_pattern_gain_out = torch.stack(add_pattern_gain_list, dim=0)
        move_pattern_gain_out = torch.stack(move_pattern_gain_list, dim=0)
        delete_nll_gain_out = torch.stack(delete_nll_gain_list, dim=0)
        add_nll_gain_out = torch.stack(add_nll_gain_list, dim=0)
        move_nll_gain_out = torch.stack(move_nll_gain_list, dim=0)
        delete_valid_mask_out = torch.stack(delete_valid_mask_list, dim=0)
        add_valid_mask_out = torch.stack(add_valid_mask_list, dim=0)
        move_valid_mask_out = torch.stack(move_valid_mask_list, dim=0)
        best_add_child_slot_out = torch.stack(best_add_child_slot_list, dim=0)
        best_move_target_child_slot_out = torch.stack(best_move_target_child_slot_list, dim=0)
        best_operation_hint_out = torch.stack(best_operation_hint_list, dim=0)

        return {
            "available": True,
            "reason": "",
            "source": str(source),
            "diagnostic_scalars_collected": bool(collect_debug_scalars),
            "voxel_coords": coords_b3n.detach(),
            "parent_coords": parent_coords_out.detach(),
            "child_slot": child_slot_out.detach(),
            "parent_pattern_code": parent_code_out.detach(),
            "parent_child_count": parent_count_out.detach(),
            "parent_pattern_frequency": parent_freq_out.detach(),
            "parent_pattern_nll": parent_nll_out.detach(),
            "unique_parent_count": int(unique_parent_count_max),
            "unique_pattern_count": int(unique_pattern_count_max),
            "mean_child_count": float(sum(mean_child_count_values) / max(len(mean_child_count_values), 1)),
            "single_child_parent_ratio": float(sum(single_child_ratio_values) / max(len(single_child_ratio_values), 1)),
            "max_pattern_frequency": float(sum(max_pattern_frequency_values) / max(len(max_pattern_frequency_values), 1)),

            "delete_pattern_gain": delete_pattern_gain_out.detach(),
            "add_pattern_gain": add_pattern_gain_out.detach(),
            "move_pattern_gain": move_pattern_gain_out.detach(),
            "delete_nll_gain": delete_nll_gain_out.detach(),
            "add_nll_gain": add_nll_gain_out.detach(),
            "move_nll_gain": move_nll_gain_out.detach(),
            "delete_valid_mask": delete_valid_mask_out.detach(),
            "add_valid_mask": add_valid_mask_out.detach(),
            "move_valid_mask": move_valid_mask_out.detach(),
            "best_add_child_slot": best_add_child_slot_out.detach(),
            "best_move_target_child_slot": best_move_target_child_slot_out.detach(),
            "best_operation_hint": best_operation_hint_out.detach(),
            "delete_gain_mean": float(sum(delete_gain_mean_values) / max(len(delete_gain_mean_values), 1)),
            "add_gain_mean": float(sum(add_gain_mean_values) / max(len(add_gain_mean_values), 1)),
            "move_gain_mean": float(sum(move_gain_mean_values) / max(len(move_gain_mean_values), 1)),
            "high_gain_candidate_ratio": float(sum(high_gain_candidate_ratio_values) / max(len(high_gain_candidate_ratio_values), 1)),
            "candidate_available": bool(getattr(self.args, "leaf_pattern_candidate_diagnosis", True)),
        }

    def _merge_actual_oracle_into_leaf_pattern(self, leaf_pattern_diag, source_tree, like_tensor):
        """
        SparsePCGC actual oracle がtrain.py側で確認した編集候補を、
        leaf pattern診断と同じ経路でActuatorへ渡す。

        通常のleaf histogram gainはactual codecと完全には一致しないため、
        actualで改善を確認した候補があるstepではこちらを優先する。
        """
        if not isinstance(source_tree, dict):
            return leaf_pattern_diag
        if not bool(source_tree.get("actual_oracle_enabled", False)):
            return leaf_pattern_diag

        if not isinstance(leaf_pattern_diag, dict):
            leaf_pattern_diag = self._empty_leaf_pattern_diagnosis(
                like_tensor,
                reason="actual_oracle_only",
            )

        B, _, N = like_tensor.shape
        device = like_tensor.device
        dtype = like_tensor.dtype

        def _fit_map(value, *, as_bool=False):
            if not torch.is_tensor(value):
                if as_bool:
                    return torch.zeros((B, N), device=device, dtype=torch.bool)
                return torch.zeros((B, N), device=device, dtype=dtype)

            out = value.detach().to(device=device)
            if out.ndim == 1:
                out = out.view(1, -1)
            elif out.ndim == 2:
                pass
            elif out.ndim == 3:
                if out.shape[1] == 1:
                    out = out.squeeze(1)
                elif out.shape[2] == 1:
                    out = out.squeeze(2)
                else:
                    out = out[:, 0, :]
            else:
                if as_bool:
                    return torch.zeros((B, N), device=device, dtype=torch.bool)
                return torch.zeros((B, N), device=device, dtype=dtype)

            if out.shape[0] == 1 and B > 1:
                out = out.expand(B, -1).contiguous()
            if out.shape[0] != B:
                if as_bool:
                    return torch.zeros((B, N), device=device, dtype=torch.bool)
                return torch.zeros((B, N), device=device, dtype=dtype)

            current_n = int(out.shape[1])
            if current_n > N:
                out = out[:, :N].contiguous()
            elif current_n < N:
                if current_n > 0:
                    pad = out[:, -1:].expand(B, N - current_n)
                    out = torch.cat([out, pad], dim=1).contiguous()
                else:
                    if as_bool:
                        return torch.zeros((B, N), device=device, dtype=torch.bool)
                    return torch.zeros((B, N), device=device, dtype=dtype)

            if as_bool:
                return out.to(dtype=torch.bool)
            return torch.nan_to_num(
                out.to(dtype=dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        def _fit_long_map(value, default=-1):
            if not torch.is_tensor(value):
                return torch.full((B, N), int(default), device=device, dtype=torch.long)

            out = value.detach().to(device=device, dtype=torch.long)
            if out.ndim == 1:
                out = out.view(1, -1)
            elif out.ndim == 2:
                pass
            elif out.ndim == 3:
                if out.shape[1] == 1:
                    out = out.squeeze(1)
                elif out.shape[2] == 1:
                    out = out.squeeze(2)
                else:
                    out = out[:, 0, :]
            else:
                return torch.full((B, N), int(default), device=device, dtype=torch.long)

            if out.shape[0] == 1 and B > 1:
                out = out.expand(B, -1).contiguous()
            if out.shape[0] != B:
                return torch.full((B, N), int(default), device=device, dtype=torch.long)

            current_n = int(out.shape[1])
            if current_n > N:
                out = out[:, :N].contiguous()
            elif current_n < N:
                if current_n > 0:
                    pad = out[:, -1:].expand(B, N - current_n)
                    out = torch.cat([out, pad], dim=1).contiguous()
                else:
                    return torch.full((B, N), int(default), device=device, dtype=torch.long)
            return out

        out = dict(leaf_pattern_diag)
        out["available"] = True
        out["reason"] = ""
        out["source"] = f"{out.get('source', 'none')}+actual_oracle"
        out["actual_oracle_enabled"] = True
        out["actual_oracle_drop_mask"] = _fit_map(
            source_tree.get("actual_oracle_drop_mask", None),
            as_bool=True,
        )
        out["actual_oracle_drop_score"] = _fit_map(
            source_tree.get("actual_oracle_drop_score", None),
            as_bool=False,
        )
        out["actual_oracle_drop_bad_mask"] = _fit_map(
            source_tree.get("actual_oracle_drop_bad_mask", None),
            as_bool=True,
        )
        out["actual_oracle_drop_bad_score"] = _fit_map(
            source_tree.get("actual_oracle_drop_bad_score", None),
            as_bool=False,
        )
        out["actual_oracle_drop_used"] = bool(source_tree.get("actual_oracle_drop_used", False))
        out["actual_oracle_drop_best_percent"] = float(
            source_tree.get("actual_oracle_drop_best_percent", 0.0) or 0.0
        )
        out["actual_oracle_drop_tested_count"] = int(
            source_tree.get("actual_oracle_drop_tested_count", 0) or 0
        )
        out["actual_oracle_bad_candidate_count"] = int(
            source_tree.get("actual_oracle_bad_candidate_count", 0) or 0
        )
        out["actual_oracle_improving_candidate_count"] = int(
            source_tree.get("actual_oracle_improving_candidate_count", 0) or 0
        )
        out["actual_oracle_combo_extra_count"] = int(
            source_tree.get("actual_oracle_combo_extra_count", 0) or 0
        )
        for key in (
            "actual_oracle_generated_candidate_count",
            "actual_oracle_accepted_candidate_count",
            "actual_oracle_accepted_prune_count",
            "actual_oracle_accepted_add_count",
            "actual_oracle_accepted_adjust_count",
            "actual_oracle_accepted_subtree_move_count",
            "actual_oracle_accepted_parent_collapse_count",
            "actual_oracle_accepted_pattern_canonicalize_count",
            "actual_oracle_noop_label_count",
            "actual_oracle_high_rate_mppov_count",
            "actual_oracle_low_prob_occupied_count",
            "actual_oracle_single_child_chain_count",
            "actual_oracle_context_pattern_candidate_count",
            "actual_oracle_eval_count",
            "actual_oracle_eval_max",
            "actual_oracle_fast_diagnostic_full_add_count",
            "actual_oracle_fast_diagnostic_local_add_count",
        ):
            out[key] = int(source_tree.get(key, 0) or 0)
        for key in (
            "actual_oracle_noop_label_weight",
            "actual_oracle_time",
            "actual_oracle_delta_actual_percent",
            "actual_oracle_proxy_percent",
            "actual_oracle_geometry_percent",
            "actual_oracle_original_actual_bits",
            "actual_oracle_edited_actual_bits",
            "actual_oracle_fast_diagnostic_full_add_ratio",
            "actual_oracle_fast_diagnostic_local_add_ratio",
        ):
            out[key] = float(source_tree.get(key, 0.0) or 0.0)
        out["actual_oracle_drop_reason"] = str(
            source_tree.get("actual_oracle_drop_reason", "")
        )
        out["actual_oracle_scheduled_operation"] = str(
            source_tree.get("actual_oracle_scheduled_operation", "")
        )
        out["actual_oracle_add_mask"] = _fit_map(
            source_tree.get("actual_oracle_add_mask", None),
            as_bool=True,
        )
        out["actual_oracle_add_score"] = _fit_map(
            source_tree.get("actual_oracle_add_score", None),
            as_bool=False,
        )
        oracle_add_slot = _fit_long_map(
            source_tree.get("actual_oracle_best_add_child_slot", None),
            default=-1,
        )
        out["actual_oracle_best_add_child_slot"] = oracle_add_slot
        out["actual_oracle_best_add_direction_index"] = _fit_long_map(
            source_tree.get("actual_oracle_best_add_direction_index", None),
            default=-1,
        )
        out["actual_oracle_add_bad_mask"] = _fit_map(
            source_tree.get("actual_oracle_add_bad_mask", None),
            as_bool=True,
        )
        out["actual_oracle_add_bad_score"] = _fit_map(
            source_tree.get("actual_oracle_add_bad_score", None),
            as_bool=False,
        )
        out["actual_oracle_bad_add_child_slot"] = _fit_long_map(
            source_tree.get("actual_oracle_bad_add_child_slot", None),
            default=-1,
        )
        out["actual_oracle_bad_add_direction_index"] = _fit_long_map(
            source_tree.get("actual_oracle_bad_add_direction_index", None),
            default=-1,
        )
        if bool(out["actual_oracle_add_mask"].any().detach().cpu()):
            base_add_slot = out.get("best_add_child_slot", None)
            if torch.is_tensor(base_add_slot):
                base_add_slot = base_add_slot.detach().to(device=device, dtype=torch.long)
                if base_add_slot.ndim == 3:
                    base_add_slot = base_add_slot.squeeze(1) if base_add_slot.shape[1] == 1 else base_add_slot[:, 0, :]
                if base_add_slot.ndim == 1:
                    base_add_slot = base_add_slot.view(1, -1)
                if base_add_slot.shape[0] == 1 and B > 1:
                    base_add_slot = base_add_slot.expand(B, -1).contiguous()
                if base_add_slot.shape[0] == B and base_add_slot.shape[1] >= N:
                    base_add_slot = base_add_slot[:, :N].contiguous()
                else:
                    base_add_slot = torch.full((B, N), -1, device=device, dtype=torch.long)
            else:
                base_add_slot = torch.full((B, N), -1, device=device, dtype=torch.long)
            add_mask = out["actual_oracle_add_mask"].to(device=device, dtype=torch.bool)
            out["best_add_child_slot"] = torch.where(add_mask, oracle_add_slot, base_add_slot).detach()
        out["actual_oracle_add_used"] = bool(source_tree.get("actual_oracle_add_used", False))
        override_coords = source_tree.get("actual_oracle_override_final_voxel_coords", None)
        if torch.is_tensor(override_coords):
            override_coords = override_coords.detach().to(device=device, dtype=torch.long)
            if override_coords.ndim == 2:
                override_coords = (
                    override_coords.transpose(0, 1).contiguous().unsqueeze(0)
                    if override_coords.shape[-1] == 3
                    else override_coords.unsqueeze(0)
                )
            elif override_coords.ndim == 3 and override_coords.shape[1] != 3 and override_coords.shape[-1] == 3:
                override_coords = override_coords.permute(0, 2, 1).contiguous()
            if override_coords.ndim == 3 and override_coords.shape[1] == 3:
                out["actual_oracle_override_final_voxel_coords"] = override_coords
                out["actual_oracle_override_move_count"] = int(
                    source_tree.get("actual_oracle_override_move_count", 0) or 0
                )
                out["actual_oracle_override_add_count"] = int(
                    source_tree.get("actual_oracle_override_add_count", 0) or 0
                )
                out["actual_oracle_override_drop_count"] = int(
                    source_tree.get("actual_oracle_override_drop_count", 0) or 0
                )
                out["actual_oracle_override_subtree_prune_count"] = int(
                    source_tree.get("actual_oracle_override_subtree_prune_count", 0) or 0
                )
                out["actual_oracle_override_scope"] = str(
                    source_tree.get("actual_oracle_override_scope", "") or ""
                )
        move_mask = source_tree.get("actual_oracle_move_mask", None)
        if torch.is_tensor(move_mask):
            out["actual_oracle_move_mask"] = move_mask.detach().to(device=device)
            move_score = source_tree.get("actual_oracle_move_score", None)
            if torch.is_tensor(move_score):
                out["actual_oracle_move_score"] = move_score.detach().to(device=device)
            out["actual_oracle_move_used"] = bool(source_tree.get("actual_oracle_override_move_count", 0) or 0)
        out["actual_oracle_move_bad_mask"] = _fit_map(
            source_tree.get("actual_oracle_move_bad_mask", None),
            as_bool=True,
        )
        out["actual_oracle_move_bad_score"] = _fit_map(
            source_tree.get("actual_oracle_move_bad_score", None),
            as_bool=False,
        )
        out["actual_oracle_move_direction_index"] = _fit_long_map(
            source_tree.get("actual_oracle_move_direction_index", None),
            default=-1,
        )
        out["actual_oracle_move_bad_direction_index"] = _fit_long_map(
            source_tree.get("actual_oracle_move_bad_direction_index", None),
            default=-1,
        )
        out["actual_oracle_edit_record_bits"] = float(
            source_tree.get("actual_oracle_edit_record_bits", 0.0) or 0.0
        )
        out["actual_oracle_best_edit_record_bits"] = float(
            source_tree.get("actual_oracle_best_edit_record_bits", 0.0) or 0.0
        )
        out["actual_oracle_raw_percent"] = float(source_tree.get("actual_oracle_raw_percent", 0.0) or 0.0)
        out["actual_oracle_best_raw_percent"] = float(
            source_tree.get("actual_oracle_best_raw_percent", 0.0) or 0.0
        )
        out["actual_oracle_operation"] = str(source_tree.get("actual_oracle_operation", ""))
        return out

    def _leaf_pattern_feature_channels(self, leaf_pattern_diag, like_tensor):
        """
        Section3:
        leaf pattern候補gainを、既存proxyへ混ぜ込むための0から1特徴へ変換する。
        feature_dimは増やさず、既存40次元構造を維持する。
        """
        B, _, N = like_tensor.shape
        device = like_tensor.device
        dtype = like_tensor.dtype

        def _get_float_map(key):
            if not isinstance(leaf_pattern_diag, dict):
                return like_tensor.new_zeros((B, 1, N))
            value = leaf_pattern_diag.get(key, None)
            if not torch.is_tensor(value):
                return like_tensor.new_zeros((B, 1, N))
            value = value.to(device=device, dtype=dtype)
            if value.ndim == 2:
                value = value.unsqueeze(1)
            elif value.ndim == 3 and value.shape[1] != 1:
                value = value[:, :1, :]
            if value.shape[0] == 1 and B > 1:
                value = value.expand(B, -1, -1)
            if value.shape[-1] != N:
                if value.shape[-1] > N:
                    value = value[:, :, :N]
                elif value.shape[-1] > 0:
                    pad = value[:, :, -1:].expand(value.shape[0], value.shape[1], N - value.shape[-1])
                    value = torch.cat([value, pad], dim=2)
                else:
                    value = like_tensor.new_zeros((B, 1, N))
            return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        def _get_long_map(key):
            if not isinstance(leaf_pattern_diag, dict):
                return torch.zeros((B, 1, N), device=device, dtype=torch.long)
            value = leaf_pattern_diag.get(key, None)
            if not torch.is_tensor(value):
                return torch.zeros((B, 1, N), device=device, dtype=torch.long)
            value = value.to(device=device, dtype=torch.long)
            if value.ndim == 2:
                value = value.unsqueeze(1)
            elif value.ndim == 3 and value.shape[1] != 1:
                value = value[:, :1, :]
            if value.shape[0] == 1 and B > 1:
                value = value.expand(B, -1, -1)
            if value.shape[-1] != N:
                if value.shape[-1] > N:
                    value = value[:, :, :N]
                elif value.shape[-1] > 0:
                    pad = value[:, :, -1:].expand(value.shape[0], value.shape[1], N - value.shape[-1])
                    value = torch.cat([value, pad], dim=2)
                else:
                    value = torch.zeros((B, 1, N), device=device, dtype=torch.long)
            return value

        scale = max(float(getattr(self.args, "leaf_pattern_feature_gain_scale", 2.0)), 1e-6)

        delete_gain = torch.tanh(_get_float_map("delete_nll_gain").clamp_min(0.0) * scale).clamp(0.0, 1.0)
        add_gain = torch.tanh(_get_float_map("add_nll_gain").clamp_min(0.0) * scale).clamp(0.0, 1.0)
        move_gain = torch.tanh(_get_float_map("move_nll_gain").clamp_min(0.0) * scale).clamp(0.0, 1.0)
        best_gain = torch.maximum(torch.maximum(delete_gain, add_gain), move_gain)

        parent_nll = torch.tanh(_get_float_map("parent_pattern_nll") / 4.0).clamp(0.0, 1.0)
        parent_freq = _get_float_map("parent_pattern_frequency").clamp(0.0, 1.0)
        child_count = (_get_float_map("parent_child_count") / 8.0).clamp(0.0, 1.0)

        best_op = _get_long_map("best_operation_hint").to(dtype=dtype)
        best_op = (best_op / 3.0).clamp(0.0, 1.0)

        return {
            "delete_gain": delete_gain,
            "add_gain": add_gain,
            "move_gain": move_gain,
            "best_gain": best_gain,
            "parent_nll": parent_nll,
            "parent_freq": parent_freq,
            "child_count": child_count,
            "best_op": best_op,
        }
    
    def _leaf_pattern_candidate_scores_single(
        self,
        parent_pattern_code_unique,
        child_count_unique,
        child_slot,
        inverse,
        *,
        device,
        dtype,
    ):
        """
        Section2:
        parent occupancy codeごとに、Delete/Add/Move後のpattern頻度改善を計算する。
        ここでは操作を実行しない。候補診断だけを返す。

        操作定義:
        - Delete: 現在のchild slotを0へ変える
        - Add: 同じparent内のempty child slotを1へ変える
        - Move: 現在のchild slotを0にし、empty child slotを1へ変える
        """
        parent_pattern_code_unique = parent_pattern_code_unique.to(device=device, dtype=torch.long).reshape(-1)
        child_count_unique = child_count_unique.to(device=device, dtype=dtype).reshape(-1)
        child_slot = child_slot.to(device=device, dtype=torch.long).reshape(-1)
        inverse = inverse.to(device=device, dtype=torch.long).reshape(-1)

        parent_count = int(parent_pattern_code_unique.numel())
        point_count = int(child_slot.numel())

        if parent_count <= 0 or point_count <= 0:
            return {
                "delete_pattern_gain": torch.zeros((point_count,), device=device, dtype=dtype),
                "add_pattern_gain": torch.zeros((point_count,), device=device, dtype=dtype),
                "move_pattern_gain": torch.zeros((point_count,), device=device, dtype=dtype),
                "delete_nll_gain": torch.zeros((point_count,), device=device, dtype=dtype),
                "add_nll_gain": torch.zeros((point_count,), device=device, dtype=dtype),
                "move_nll_gain": torch.zeros((point_count,), device=device, dtype=dtype),
                "delete_valid_mask": torch.zeros((point_count,), device=device, dtype=torch.bool),
                "add_valid_mask": torch.zeros((point_count,), device=device, dtype=torch.bool),
                "move_valid_mask": torch.zeros((point_count,), device=device, dtype=torch.bool),
                "best_add_child_slot": torch.full((point_count,), -1, device=device, dtype=torch.long),
                "best_move_target_child_slot": torch.full((point_count,), -1, device=device, dtype=torch.long),
                "best_operation_hint": torch.zeros((point_count,), device=device, dtype=torch.long),
            }

        smoothing = max(float(getattr(self.args, "leaf_pattern_candidate_smoothing", 1.0)), 0.0)
        code_hist = torch.bincount(
            parent_pattern_code_unique.clamp(0, 255),
            minlength=256,
        ).to(device=device, dtype=dtype)

        code_prob = code_hist + float(smoothing)
        code_prob = code_prob / code_prob.sum().clamp_min(torch.finfo(dtype).eps)
        code_nll = -torch.log2(code_prob.clamp_min(torch.finfo(dtype).eps))

        current_code = parent_pattern_code_unique.index_select(0, inverse).clamp(0, 255)
        current_prob = code_prob.index_select(0, current_code)
        current_nll = code_nll.index_select(0, current_code)

        bit_current = (1 << child_slot.clamp(0, 7)).to(device=device, dtype=torch.long)
        delete_code = torch.bitwise_and(current_code, torch.bitwise_not(bit_current)).clamp(0, 255)

        delete_prob = code_prob.index_select(0, delete_code)
        delete_nll = code_nll.index_select(0, delete_code)

        delete_pattern_gain = torch.log2(delete_prob.clamp_min(torch.finfo(dtype).eps)) - torch.log2(
            current_prob.clamp_min(torch.finfo(dtype).eps)
        )
        delete_nll_gain = current_nll - delete_nll

        min_children_after = max(int(getattr(self.args, "leaf_pattern_delete_min_children_after", 1)), 0)
        point_child_count = child_count_unique.index_select(0, inverse)
        delete_valid_mask = (point_child_count - 1.0) >= float(min_children_after)

        slot_values = torch.arange(8, device=device, dtype=torch.long).view(1, 8)
        slot_bits = (1 << slot_values).to(dtype=torch.long)
        current_code_2d = current_code.view(-1, 1)
        current_prob_2d = current_prob.view(-1, 1)
        current_nll_2d = current_nll.view(-1, 1)

        occupied_slot_mask = (torch.bitwise_and(current_code_2d, slot_bits) != 0)
        empty_slot_mask = ~occupied_slot_mask

        add_code_all = torch.bitwise_or(current_code_2d, slot_bits).clamp(0, 255)
        add_prob_all = code_prob.index_select(0, add_code_all.reshape(-1)).view(point_count, 8)
        add_nll_all = code_nll.index_select(0, add_code_all.reshape(-1)).view(point_count, 8)

        add_pattern_gain_all = torch.log2(add_prob_all.clamp_min(torch.finfo(dtype).eps)) - torch.log2(
            current_prob_2d.clamp_min(torch.finfo(dtype).eps)
        )
        add_nll_gain_all = current_nll_2d - add_nll_all

        very_bad = torch.full_like(add_nll_gain_all, -1.0e6)
        add_score_all = torch.where(empty_slot_mask, add_nll_gain_all, very_bad)
        best_add_score, best_add_slot = add_score_all.max(dim=1)
        add_valid_mask = empty_slot_mask.any(dim=1)
        best_add_slot = torch.where(
            add_valid_mask,
            best_add_slot.to(dtype=torch.long),
            torch.full_like(best_add_slot, -1),
        )

        add_pattern_gain = torch.where(
            add_valid_mask,
            add_pattern_gain_all.gather(1, best_add_slot.clamp_min(0).view(-1, 1)).view(-1),
            torch.zeros((point_count,), device=device, dtype=dtype),
        )
        add_nll_gain = torch.where(
            add_valid_mask,
            best_add_score,
            torch.zeros((point_count,), device=device, dtype=dtype),
        )

        source_removed_code = torch.bitwise_and(current_code, torch.bitwise_not(bit_current)).view(-1, 1)
        move_code_all = torch.bitwise_or(source_removed_code, slot_bits).clamp(0, 255)
        move_prob_all = code_prob.index_select(0, move_code_all.reshape(-1)).view(point_count, 8)
        move_nll_all = code_nll.index_select(0, move_code_all.reshape(-1)).view(point_count, 8)

        move_pattern_gain_all = torch.log2(move_prob_all.clamp_min(torch.finfo(dtype).eps)) - torch.log2(
            current_prob_2d.clamp_min(torch.finfo(dtype).eps)
        )
        move_nll_gain_all = current_nll_2d - move_nll_all

        move_score_all = torch.where(empty_slot_mask, move_nll_gain_all, very_bad)
        best_move_score, best_move_slot = move_score_all.max(dim=1)
        move_valid_mask = empty_slot_mask.any(dim=1) & (point_child_count >= 1.0)
        best_move_slot = torch.where(
            move_valid_mask,
            best_move_slot.to(dtype=torch.long),
            torch.full_like(best_move_slot, -1),
        )

        move_pattern_gain = torch.where(
            move_valid_mask,
            move_pattern_gain_all.gather(1, best_move_slot.clamp_min(0).view(-1, 1)).view(-1),
            torch.zeros((point_count,), device=device, dtype=dtype),
        )
        move_nll_gain = torch.where(
            move_valid_mask,
            best_move_score,
            torch.zeros((point_count,), device=device, dtype=dtype),
        )

        delete_pattern_gain = torch.where(delete_valid_mask, delete_pattern_gain, torch.zeros_like(delete_pattern_gain))
        delete_nll_gain = torch.where(delete_valid_mask, delete_nll_gain, torch.zeros_like(delete_nll_gain))

        # 0 preserve, 1 delete, 2 add, 3 move
        stacked_gain = torch.stack(
            [
                torch.zeros_like(delete_nll_gain),
                delete_nll_gain,
                add_nll_gain,
                move_nll_gain,
            ],
            dim=0,
        )
        best_operation_hint = stacked_gain.argmax(dim=0).to(dtype=torch.long)

        return {
            "delete_pattern_gain": delete_pattern_gain,
            "add_pattern_gain": add_pattern_gain,
            "move_pattern_gain": move_pattern_gain,
            "delete_nll_gain": delete_nll_gain,
            "add_nll_gain": add_nll_gain,
            "move_nll_gain": move_nll_gain,
            "delete_valid_mask": delete_valid_mask,
            "add_valid_mask": add_valid_mask,
            "move_valid_mask": move_valid_mask,
            "best_add_child_slot": best_add_slot,
            "best_move_target_child_slot": best_move_slot,
            "best_operation_hint": best_operation_hint,
        }
    
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

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                "_neighbor_occupancy_from_global_coords expects coords with shape [N, 3], "
                f"but got {tuple(coords.shape)}. "
                "Normalize global_voxel_coords with _normalize_global_coords_n3() before calling this function."
            )
        # During a canonical full-cloud forward this exact 26-neighbour map is
        # consumed by both _prebuilt_octree_context and
        # _quantized_voxel_stats_from_tree.  Reuse the first result inside the
        # same forward; it is a deterministic, detached integer-coordinate
        # calculation and has no autograd graph.
        active = getattr(self, "_active_canonical_neighbor_cache", None)
        if isinstance(active, dict) and active.get("allow_read", False):
            cached = active.get("value")
            if torch.is_tensor(cached) and int(cached.numel()) == int(coords.shape[0]):
                return cached

        unique_coords = torch.unique(coords, dim=0, sorted=True)
        result = self._neighbor_occupancy_chunked(
            coords, unique_coords=unique_coords
        )
        active = getattr(self, "_active_canonical_neighbor_cache", None)
        if isinstance(active, dict) and active.get("allow_write", False):
            active["value"] = result
            active["allow_write"] = False
            active["allow_read"] = True
        return result

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
        # subtree_tree を優先する。
        # なければ full_octree_context の current input 用 global_voxel_coords を使う。
        source_tree = subtree_tree if isinstance(subtree_tree, dict) else full_octree_context

        if not isinstance(source_tree, dict):
            return None

        B, _, N = pts_xyz.shape
        if B != 1:
            return None

        device = pts_xyz.device
        dtype = pts_xyz.dtype

        coords_raw = self._tree_tensor(source_tree, "global_voxel_coords", device, dtype=torch.long)
        if coords_raw is None or coords_raw.numel() <= 0:
            return None

        coords = self._normalize_global_coords_n3(
            coords_raw,
            point_count=N,
            device=device,
        )
        if coords is None or coords.numel() <= 0:
            return None

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"global_voxel_coords must be normalized to [N, 3], got {tuple(coords.shape)}"
            )
        
        parent_coords = torch.div(coords, 2, rounding_mode="floor")
        unique_parents, inverse = torch.unique(parent_coords, dim=0, sorted=True, return_inverse=True)
        # _leaf_pattern_diagnosis_from_coords receives these same canonical
        # coordinates later in this forward.  Sharing the parent partition
        # removes a second large torch.unique without changing child slots,
        # occupancy codes, targets, or gradients.
        self._active_canonical_parent_cache = {
            "unique_parents": unique_parents,
            "inverse": inverse,
        }
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

        # Per-forward only.  Never reuse a neighbour map across frames.
        self._active_canonical_neighbor_cache = {
            "allow_write": True,
            "allow_read": False,
            "value": None,
        }
        self._active_canonical_parent_cache = None

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
        force_canonical = bool(getattr(self.args, "force_full_cloud_canonical_voxel_basis", True))

        if prebuilt_ctx is None and force_canonical:
            raise ValueError(
                "force_full_cloud_canonical_voxel_basis=True requires prebuilt global_voxel_coords. "
                f"octree_input_mode={octree_input_mode}, requested_mode={requested_mode}"
            )

        if (
            prebuilt_ctx is None
            and requested_mode not in {"full_cloud", "debug_local_recomputed"}
            and not bool(getattr(self.args, "allow_local_octree_recompute", False))
        ):
            raise ValueError(
                "Local Octree recompute is disabled. Use prebuilt full-cloud canonical metadata."
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

        context_source = subtree_tree if isinstance(subtree_tree, dict) else full_octree_context
        global_offset = None
        if isinstance(context_source, dict):
            global_offset = context_source.get("global_offset", None)
        phase, snap_delta, snap_delta_norm = self._grid_phase(
            work_xyz,
            qs_override,
            global_offset=global_offset,
        )
        point_feature_voxel_key = None
        geo_stats = self._local_geometry_stats(work_xyz)
        tree_coords = None

        # Section1/2:
        # leaf pattern診断の初期値。
        # prebuilt canonical voxel coordsが取れた場合だけ、実診断で上書きする。
        leaf_pattern_diag = self._empty_leaf_pattern_diagnosis(
            work_xyz,
            reason="prebuilt_coords_not_available",
        )

        # Section1:
        # full cloud canonical / prebuilt subtree のvoxel coordsから作るleaf pattern診断。
        # 初期値は空診断にしておき、prebuilt coordsが取れた場合だけ上書きする。
        leaf_pattern_diag = self._empty_leaf_pattern_diagnosis(
            work_xyz,
            reason="prebuilt_coords_not_available",
        )
        leaf_oracle_source_tree = subtree_tree if isinstance(subtree_tree, dict) else full_octree_context
        if prebuilt_ctx is not None:
            source_tree = leaf_oracle_source_tree
            raw_tree_coords = self._tree_tensor(source_tree, "global_voxel_coords", work_xyz.device, dtype=torch.long)

            if raw_tree_coords is not None:
                coords_n3 = self._normalize_global_coords_n3(
                    raw_tree_coords,
                    point_count=work_xyz.shape[-1],
                    device=work_xyz.device,
                )
                if coords_n3 is not None:
                    tree_coords = coords_n3.view(1, -1, 3).contiguous()
                    quant_stats = self._quantized_voxel_stats_from_tree(
                        work_xyz,
                        tree_coords,
                        qs_override,
                        snap_delta_norm,
                    )
                    # Section1:
                    # local xyz再量子化ではなく、prebuilt global_voxel_coordsから
                    # parent node / child slot / 8-child occupancy patternを診断する。
                    leaf_pattern_diag = self._leaf_pattern_diagnosis_from_coords(
                        work_xyz,
                        tree_coords,
                        source=str(source_tree.get("octree_context_scope", "prebuilt_global_voxel_coords"))
                        if isinstance(source_tree, dict)
                        else "prebuilt_global_voxel_coords",
                    )
                    source_tree_for_point_key = (
                        subtree_tree
                        if isinstance(subtree_tree, dict)
                        else full_octree_context
                    )

                    point_feature_voxel_key = self._tree_tensor(
                        source_tree_for_point_key,
                        "global_morton_keys",
                        work_xyz.device,
                        dtype=torch.long,
                    )

                    if point_feature_voxel_key is not None:
                        point_feature_voxel_key = self._fit_point_rows(
                            point_feature_voxel_key.reshape(-1, 1),
                            work_xyz.shape[-1],
                        ).reshape(1, -1).to(
                            device=work_xyz.device,
                            dtype=torch.long,
                        )
        source_tree_for_key = subtree_tree if isinstance(subtree_tree, dict) else full_octree_context
        structural_voxel_key = None
        phase4_structural_key_source = "missing"
        raw_structural_key = None

        if prebuilt_ctx is not None and isinstance(source_tree_for_key, dict):
            raw_structural_key = self._tree_tensor(
                source_tree_for_key,
                "global_morton_keys",
                pts_xyz.device,
                dtype=torch.long,
            )

            if raw_structural_key is not None and raw_structural_key.numel() > 0:
                structural_voxel_key = self._fit_point_rows(
                    raw_structural_key.reshape(-1, 1),
                    pts_xyz.shape[-1],
                ).reshape(1, -1).to(device=pts_xyz.device, dtype=torch.long)
                phase4_structural_key_source = "global_morton_keys"

            if structural_voxel_key is None:
                raw_coords_for_key = self._tree_tensor(
                    source_tree_for_key,
                    "global_voxel_coords",
                    pts_xyz.device,
                    dtype=torch.long,
                )

                coords_n3_for_key = self._normalize_global_coords_n3(
                    raw_coords_for_key,
                    point_count=pts_xyz.shape[-1],
                    device=pts_xyz.device,
                )

                structural_voxel_key = self._stable_voxel_keys_from_coords_n3(
                    coords_n3_for_key
                )

                if structural_voxel_key is not None:
                    structural_voxel_key = structural_voxel_key.to(
                        device=pts_xyz.device,
                        dtype=torch.long,
                    )
                    phase4_structural_key_source = "global_voxel_coords_hash"
        else:
            if prebuilt_required:
                raise ValueError("prebuilt_subtree_tree mode requires _quantized_voxel_stats_from_tree().")
            quant_stats = self._quantized_voxel_stats(work_xyz, qs_override, snap_delta_norm)
            point_feature_voxel_key = self._point_feature_voxel_key(work_xyz, qs_override)

            # Section1/2:
            # local再量子化はfull cloud canonical基準ではないため、candidate診断には使わない。
            leaf_pattern_diag = self._empty_leaf_pattern_diagnosis(
                work_xyz,
                reason="local_recomputed_path",
            )

            # Section1:
            # local再量子化経路では、full cloud canonical基準ではないため、
            # leaf pattern診断は使わない。
            leaf_pattern_diag = self._empty_leaf_pattern_diagnosis(
                work_xyz,
                reason="local_recomputed_path",
            )
        leaf_pattern_diag = self._merge_actual_oracle_into_leaf_pattern(
            leaf_pattern_diag,
            leaf_oracle_source_tree,
            work_xyz,
        )
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

        leaf_feature_integration_used = False
        leaf_feature_best_gain_mean = 0.0
        leaf_feature_best_gain_max = 0.0

        if (
            bool(getattr(self.args, "leaf_pattern_feature_integration", True))
            and isinstance(leaf_pattern_diag, dict)
            and bool(leaf_pattern_diag.get("available", False))
        ):
            leaf_feat = self._leaf_pattern_feature_channels(
                leaf_pattern_diag,
                shape_proxy,
            )
            blend = min(max(float(getattr(self.args, "leaf_pattern_feature_blend_weight", 0.35)), 0.0), 1.0)

            delete_gain_feat = leaf_feat["delete_gain"]
            add_gain_feat = leaf_feat["add_gain"]
            move_gain_feat = leaf_feat["move_gain"]
            best_gain_feat = leaf_feat["best_gain"]
            parent_nll_feat = leaf_feat["parent_nll"]

            # 圧縮上あやしい候補をlow_probability/context/sparse/quant proxyへ反映する。
            lowprob_proxy = ((1.0 - blend) * lowprob_proxy + blend * torch.maximum(lowprob_proxy, best_gain_feat)).clamp(0.0, 1.0)
            context_proxy = ((1.0 - blend) * context_proxy + blend * torch.maximum(context_proxy, parent_nll_feat)).clamp(0.0, 1.0)
            sparse_proxy = ((1.0 - blend) * sparse_proxy + blend * torch.maximum(sparse_proxy, delete_gain_feat)).clamp(0.0, 1.0)
            quant_proxy = ((1.0 - blend) * quant_proxy + blend * torch.maximum(quant_proxy, torch.maximum(add_gain_feat, move_gain_feat))).clamp(0.0, 1.0)

            # shape_proxyは幾何保持側なので、rate gainだけで過剰に下げない。
            # ここでは触らず、Section4のActuator側でgeometry guardと一緒に使う。
            leaf_feature_integration_used = True
            collect_leaf_feature_debug = bool(
                getattr(self.args, "leaf_pattern_diagnosis_debug", False)
                or (
                    getattr(self.args, "verbose_step_logs", False)
                    and getattr(self.args, "_log_this_step", True)
                )
                or getattr(self.args, "_collect_structure_debug", False)
            )
            if collect_leaf_feature_debug:
                leaf_feature_best_gain_mean = float(best_gain_feat.detach().float().mean().cpu())
                leaf_feature_best_gain_max = float(best_gain_feat.detach().float().max().cpu())

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
        node_voxel_desc = self._build_node_voxel_descriptor(
            pts_xyz=pts_xyz,
            feature=feature.to(dtype=input_dtype),
            oct_ctx=oct_ctx.to(dtype=input_dtype),
            subtree_tree=subtree_tree,
            full_octree_context=full_octree_context,
            point_feature_voxel_key=point_feature_voxel_key,
            prebuilt_ctx=prebuilt_ctx,
            leaf_pattern_diag=leaf_pattern_diag,
        )
        source_tree_for_key = subtree_tree if isinstance(subtree_tree, dict) else full_octree_context
        structural_voxel_key = None
        phase4_structural_key_source = "missing"
        raw_structural_key = None

        if prebuilt_ctx is not None and isinstance(source_tree_for_key, dict):
            raw_structural_key = self._tree_tensor(
                source_tree_for_key,
                "global_morton_keys",
                pts_xyz.device,
                dtype=torch.long,
            )

            if raw_structural_key is not None and raw_structural_key.numel() > 0:
                structural_voxel_key = self._fit_point_rows(
                    raw_structural_key.reshape(-1, 1),
                    pts_xyz.shape[-1],
                ).reshape(1, -1).to(device=pts_xyz.device, dtype=torch.long)
                phase4_structural_key_source = "global_morton_keys"

            if structural_voxel_key is None:
                raw_coords_for_key = self._tree_tensor(
                    source_tree_for_key,
                    "global_voxel_coords",
                    pts_xyz.device,
                    dtype=torch.long,
                )

                coords_n3_for_key = self._normalize_global_coords_n3(
                    raw_coords_for_key,
                    point_count=pts_xyz.shape[-1],
                    device=pts_xyz.device,
                )

                structural_voxel_key = self._stable_voxel_keys_from_coords_n3(
                    coords_n3_for_key
                )

                if structural_voxel_key is not None:
                    structural_voxel_key = structural_voxel_key.to(
                        device=pts_xyz.device,
                        dtype=torch.long,
                    )
                    phase4_structural_key_source = "global_voxel_coords_hash"
        result = {
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
            "phase4_structural_key_source": str(phase4_structural_key_source),
            "structural_voxel_key": structural_voxel_key,
            "point_feature_voxel_key": point_feature_voxel_key,
            "node_voxel_desc": node_voxel_desc,

            "leaf_pattern_diag": leaf_pattern_diag,
            "leaf_pattern_available": bool(
                isinstance(leaf_pattern_diag, dict)
                and leaf_pattern_diag.get("available", False)
            ),
            "leaf_pattern_source": str(
                leaf_pattern_diag.get("source", "none")
                if isinstance(leaf_pattern_diag, dict)
                else "none"
            ),
            "leaf_pattern_reason": str(
                leaf_pattern_diag.get("reason", "")
                if isinstance(leaf_pattern_diag, dict)
                else "missing"
            ),
            "leaf_unique_parent_count": int(
                leaf_pattern_diag.get("unique_parent_count", 0)
                if isinstance(leaf_pattern_diag, dict)
                else 0
            ),
            "leaf_unique_pattern_count": int(
                leaf_pattern_diag.get("unique_pattern_count", 0)
                if isinstance(leaf_pattern_diag, dict)
                else 0
            ),
            "leaf_mean_child_count": float(
                leaf_pattern_diag.get("mean_child_count", 0.0)
                if isinstance(leaf_pattern_diag, dict)
                else 0.0
            ),
            "leaf_single_child_parent_ratio": float(
                leaf_pattern_diag.get("single_child_parent_ratio", 0.0)
                if isinstance(leaf_pattern_diag, dict)
                else 0.0
            ),
            "leaf_max_pattern_frequency": float(
                leaf_pattern_diag.get("max_pattern_frequency", 0.0)
                if isinstance(leaf_pattern_diag, dict)
                else 0.0
            ),

            # Section2:
            "leaf_candidate_available": bool(
                leaf_pattern_diag.get("candidate_available", False)
                if isinstance(leaf_pattern_diag, dict)
                else False
            ),
            "leaf_delete_gain_mean": float(
                leaf_pattern_diag.get("delete_gain_mean", 0.0)
                if isinstance(leaf_pattern_diag, dict)
                else 0.0
            ),
            "leaf_add_gain_mean": float(
                leaf_pattern_diag.get("add_gain_mean", 0.0)
                if isinstance(leaf_pattern_diag, dict)
                else 0.0
            ),
            "leaf_move_gain_mean": float(
                leaf_pattern_diag.get("move_gain_mean", 0.0)
                if isinstance(leaf_pattern_diag, dict)
                else 0.0
            ),
            "leaf_high_gain_candidate_ratio": float(
                leaf_pattern_diag.get("high_gain_candidate_ratio", 0.0)
                if isinstance(leaf_pattern_diag, dict)
                else 0.0
            ),

            # Section3:
            "leaf_feature_integration_used": bool(leaf_feature_integration_used),
            "leaf_feature_best_gain_mean": float(leaf_feature_best_gain_mean),
            "leaf_feature_best_gain_max": float(leaf_feature_best_gain_max),
        }
        # ana_den6の式を既存proxyへ写像したpriorである。
        # 追加のactual codec呼出やKNNは行わないため、Step時間への影響を小さく保つ。
        # ana_den6_residualでは、den5/den6が生成した順位付き候補poolを
        # proxyへ置換せず、そのままHeuristic guidanceへ渡す。
        exact_den6_guidance = (
            source_tree_for_key.get("ana_den6_ranked_candidate_guidance")
            if isinstance(source_tree_for_key, dict)
            else None
        )
        if not isinstance(exact_den6_guidance, dict) and isinstance(full_octree_context, dict):
            exact_den6_guidance = full_octree_context.get("ana_den6_ranked_candidate_guidance")
        if isinstance(exact_den6_guidance, dict):
            result["ana_den6_ranked_candidate_guidance"] = exact_den6_guidance

        current_global_voxel_coords = (
            source_tree_for_key.get("global_voxel_coords")
            if isinstance(source_tree_for_key, dict)
            else None
        )
        if torch.is_tensor(current_global_voxel_coords):
            result["global_voxel_coords"] = current_global_voxel_coords

        result["heuristic_guidance"] = build_heuristic_guidance(result, self.args)
        return result
