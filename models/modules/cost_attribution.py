import torch
import torch.nn as nn
import torch.nn.functional as F


CAUSE_NAMES = (
    "node_count",
    "single_child_chain",
    "low_probability_occupancy",
    "context_difficulty",
    "quantization_waste",
    "sparse_fragmentation",
    "local_outlier",
    "shape_preservation",
)


class CostAttributionModule(nn.Module):
    """Predict the cause of octree coding inefficiency for each point."""

    def __init__(self, in_channels, hidden_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, len(CAUSE_NAMES), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.debug_tensors = {}

    def forward(self, features):
        if not torch.is_tensor(features):
            raise TypeError("features must be a Tensor.")
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B, C, N], got {tuple(features.shape)}")

        input_mode = "node_voxel" if bool(getattr(self, "node_voxel_mode", False)) else "point"

        logits = self.net(features)

        # ============================================================
        # Phase4:
        # CostAttribution の logits が NaN/Inf になると、
        # policy / actuator 側まで壊れるため、softmax前に安全化する。
        # ============================================================
        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        scores = torch.softmax(logits, dim=1)

        if torch.is_grad_enabled():
            score_entropy = -(
                scores.clamp_min(1e-8).log()
                * scores
            ).sum(dim=1).mean()

            self.debug_tensors = {
                "cause_mean": scores.mean().detach(),
                "cause_max": scores.max().detach(),
                "cause_entropy": score_entropy.detach(),
                "logits_abs_mean": logits.detach().abs().mean(),
                "logits_abs_max": logits.detach().abs().max(),
                "scores_requires_grad": bool(scores.requires_grad),
                "logits_requires_grad": bool(logits.requires_grad),
                "input_mode": input_mode,
                "input_shape": tuple(features.shape),
            }
        return scores, logits

    def attribution_loss(self, logits, targets, weights=None, point_mask=None):
        logits_f = logits.to(dtype=torch.float32)
        targets_f = targets.to(device=logits.device, dtype=torch.float32)
        if weights is not None:
            if torch.is_tensor(weights):
                w = weights.to(device=logits.device, dtype=torch.float32).view(1, -1, 1)
            else:
                w = logits_f.new_tensor(weights).view(1, -1, 1)
            targets_f = targets_f * w
        target_mass = targets_f.sum(dim=1, keepdim=True)
        finite_targets = torch.isfinite(targets_f).all(dim=1, keepdim=True)
        finite_logits = torch.isfinite(logits_f).all(dim=1, keepdim=True)
        valid_rows = finite_targets & finite_logits & (target_mass > 1e-6)

        norm_targets = torch.where(
            valid_rows.expand_as(targets_f),
            targets_f / target_mass.clamp_min(1e-6),
            torch.zeros_like(targets_f),
        )
        uniform = torch.full_like(norm_targets, 1.0 / max(int(norm_targets.shape[1]), 1))
        safe_targets = torch.where(
            valid_rows.expand_as(norm_targets),
            norm_targets.clamp_min(1e-8),
            uniform,
        )
        safe_targets = safe_targets / safe_targets.sum(dim=1, keepdim=True).clamp_min(1e-6)

        safe_logits = torch.where(
            valid_rows.expand_as(logits_f),
            logits_f,
            torch.zeros_like(logits_f),
        )
        log_probs = F.log_softmax(safe_logits, dim=1)
        loss = F.kl_div(log_probs, safe_targets, reduction="none").sum(dim=1, keepdim=True)
        valid_mask = valid_rows.to(device=loss.device, dtype=loss.dtype)
        if point_mask is None:
            denom = valid_mask.sum().clamp_min(1.0)
            return (loss * valid_mask).sum() / denom
        if point_mask.ndim == 2:
            point_mask = point_mask.unsqueeze(1)
        if point_mask.ndim != 3 or point_mask.shape[0] != loss.shape[0] or point_mask.shape[2] != loss.shape[2]:
            raise ValueError("point_mask must broadcast to [B, 1, N].")
        mask = point_mask.to(device=loss.device, dtype=loss.dtype) * valid_mask
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom
