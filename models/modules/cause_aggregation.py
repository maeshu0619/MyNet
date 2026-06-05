import torch
import torch.nn as nn


class CauseDiagnosisAggregation(nn.Module):
    """Aggregate point causes into local subtree-like repair units.

    The exact Octree node ids are discrete and expensive to keep in the main
    training graph.  This module builds deterministic repair units by grouping
    points in a coarse quantized grid.  It then gathers the unit-level cause
    vector back to each point, so the policy is driven by local structure rather
    than by isolated point decisions.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.unit_level = int(getattr(args, "repair_unit_level", max(1, int(getattr(args, "octree_ctx_level", 5)))))
        self.priority_dim = 1

    def _unit_keys(self, pts_xyz):
        B, _, N = pts_xyz.shape
        grid = max(2 ** max(int(self.unit_level), 1), 2)
        mins = pts_xyz.amin(dim=2, keepdim=True)
        maxs = pts_xyz.amax(dim=2, keepdim=True)
        span = (maxs - mins).amax(dim=1, keepdim=True).clamp_min(1e-9)
        coords = torch.floor((pts_xyz - mins) / span * float(grid)).long().clamp_(0, grid - 1)
        coords = coords.permute(0, 2, 1).contiguous()
        return coords[:, :, 0] * (grid * grid) + coords[:, :, 1] * grid + coords[:, :, 2]

    def _aggregate_single(self, values, keys):
        C, N = values.shape
        unique, inverse = torch.unique(keys, sorted=False, return_inverse=True)
        unit_count = unique.numel()
        sums = values.new_zeros((C, unit_count))
        counts = values.new_zeros((1, unit_count))
        sums.scatter_add_(1, inverse.view(1, N).expand(C, N), values)
        counts.scatter_add_(1, inverse.view(1, N), values.new_ones((1, N)))
        means = sums / counts.clamp_min(1.0)
        gathered = means.index_select(1, inverse)
        priority = gathered[:, :].amax(dim=0, keepdim=True)
        return gathered, priority

    def forward(self, pts_xyz, cause_scores, cause_targets, unit_keys=None):
        unit_mode = "prebuilt"
        if unit_keys is None:
            if not bool(getattr(self.args, "allow_local_repair_unit_recompute", False)):
                raise ValueError(
                    "CauseDiagnosisAggregation requires prebuilt unit_keys. "
                    "Set allow_local_repair_unit_recompute=True only for debug/local fallback runs."
                )
            keys = self._unit_keys(pts_xyz.detach())
            unit_mode = "local_recomputed"
        else:
            if unit_keys.ndim == 1:
                unit_keys = unit_keys.view(1, -1)
            if unit_keys.ndim == 3 and unit_keys.shape[1] == 1:
                unit_keys = unit_keys.squeeze(1)
            if unit_keys.ndim != 2 or unit_keys.shape[0] != pts_xyz.shape[0] or unit_keys.shape[1] != pts_xyz.shape[2]:
                raise ValueError(
                    "unit_keys must have shape [B, N] matching pts_xyz."
                )
            keys = unit_keys.to(device=pts_xyz.device, dtype=torch.long)
        agg_scores = []
        agg_targets = []
        priorities = []
        unit_counts = []
        min_unit_sizes = []
        max_unit_sizes = []
        for b in range(pts_xyz.shape[0]):
            score_b, priority_b = self._aggregate_single(cause_scores[b], keys[b])
            target_b, _ = self._aggregate_single(cause_targets[b], keys[b])
            agg_scores.append(score_b)
            agg_targets.append(target_b.detach())
            priorities.append(priority_b.detach())

            # ============================================================
            # Phase4:
            # unit key が粗すぎないかを確認する。
            # 例:
            # - unit_count=1 なら全点が1unitに潰れている可能性がある
            # - max_unit_size が極端に大きいと局所診断が粗すぎる
            # ============================================================
            unique_b, inverse_b = torch.unique(
                keys[b],
                sorted=False,
                return_inverse=True,
            )
            count_b = torch.bincount(
                inverse_b,
                minlength=int(unique_b.numel()),
            )

            unit_counts.append(int(unique_b.numel()))

            if count_b.numel() > 0:
                min_unit_sizes.append(int(count_b.min().detach().cpu()))
                max_unit_sizes.append(int(count_b.max().detach().cpu()))
            else:
                min_unit_sizes.append(0)
                max_unit_sizes.append(0)
        return {
            "scores": torch.stack(agg_scores, dim=0),
            "targets": torch.stack(agg_targets, dim=0),
            "priority": torch.stack(priorities, dim=0),
            "unit_keys": keys,
            "unit_mode": unit_mode,
            "local_recomputed": unit_mode == "local_recomputed",
            "unit_count": int(max(unit_counts) if unit_counts else 0),
            "min_unit_size": int(min(min_unit_sizes) if min_unit_sizes else 0),
            "max_unit_size": int(max(max_unit_sizes) if max_unit_sizes else 0),
        }
