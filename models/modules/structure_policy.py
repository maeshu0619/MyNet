import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


POLICY_NAMES = (
    "preserve",
    "chain_collapse",
    "sibling_merge",
    "parent_absorb",
    "context_smooth",
    "geometry_compensate",
    "outlier_suppression",
)


class StructureRepairPolicy(nn.Module):
    """Select structure repair primitives from attributed octree costs."""

    def __init__(self, in_channels, hidden_dim=96, temperature=1.0):
        super().__init__()
        self.temperature = max(float(temperature), 1e-3)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, len(POLICY_NAMES), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[0] = 3.0
            self.net[-1].bias[1:] = -1.0
        self.debug_tensors = {}

    @staticmethod
    def build_teacher(cause_targets):
        node = cause_targets[:, 0:1, :]
        single = cause_targets[:, 1:2, :]
        lowprob = cause_targets[:, 2:3, :]
        context = cause_targets[:, 3:4, :]
        if cause_targets.shape[1] >= 8:
            quant = cause_targets[:, 4:5, :]
            sparse = cause_targets[:, 5:6, :]
            outlier = cause_targets[:, 6:7, :]
        else:
            quant = torch.zeros_like(node)
            sparse = cause_targets[:, 4:5, :]
            outlier = cause_targets[:, 5:6, :]
        shape = cause_targets[:, -1:, :]
        max_cost = torch.maximum(
            torch.maximum(torch.maximum(node, single), torch.maximum(lowprob, context)),
            torch.maximum(quant, torch.maximum(sparse, outlier)),
        )

        preserve = (0.60 * shape + 0.40 * (1.0 - max_cost)).clamp_min(0.02)
        chain_collapse = single.clamp_min(0.02)
        sibling_merge = (0.35 * sparse + 0.30 * quant + 0.35 * node).clamp_min(0.02)
        parent_absorb = (0.60 * node + 0.25 * single + 0.15 * quant).clamp_min(0.02)
        context_smooth = (0.45 * lowprob + 0.35 * context + 0.20 * sparse).clamp_min(0.02)
        geometry_compensate = (shape * max_cost).clamp_min(0.02)
        outlier_suppression = (0.65 * outlier + 0.35 * quant).clamp_min(0.02)
        raw = torch.cat(
            [
                preserve,
                chain_collapse,
                sibling_merge,
                parent_absorb,
                context_smooth,
                geometry_compensate,
                outlier_suppression,
            ],
            dim=1,
        )
        return raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, features):
        if not torch.is_tensor(features):
            raise TypeError("features must be a Tensor.")
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B, C, N], got {tuple(features.shape)}")

        input_mode = "node_voxel" if bool(getattr(self, "node_voxel_mode", False)) else "point"
        if self.training and bool(getattr(self, "activation_checkpointing", False)):
            dummy = features.new_zeros((), requires_grad=True)
            logits = checkpoint(
                lambda value, _dummy: self.net(value),
                features,
                dummy,
                use_reentrant=True,
            )
        else:
            logits = self.net(features)
        probs = F.softmax(logits / self.temperature, dim=1)
        # Debug-only full-cloud reductions are intentionally disabled on the
        # normal hot path.  They do not contribute to any training loss.
        if torch.is_grad_enabled() and bool(getattr(self, "collect_runtime_debug", False)):
            self.debug_tensors = {
                "repair_ratio": (1.0 - probs[:, 0:1, :]).mean().detach(),
                "policy_entropy": (-(probs.clamp_min(1e-6).log() * probs).sum(dim=1)).mean().detach(),
                "input_mode": input_mode,
                "input_shape": tuple(features.shape),
            }
        elif not bool(getattr(self, "collect_runtime_debug", False)):
            self.debug_tensors = {
                "input_mode": input_mode,
                "input_shape": tuple(features.shape),
            }
        return probs, logits

    def policy_loss(self, logits, teacher, entropy_weight=0.0, point_mask=None):
        logits_f = logits.to(dtype=torch.float32)
        teacher_f = teacher.to(device=logits.device, dtype=torch.float32)
        teacher_mass = teacher_f.sum(dim=1, keepdim=True)
        finite_teacher = torch.isfinite(teacher_f).all(dim=1, keepdim=True)
        finite_logits = torch.isfinite(logits_f).all(dim=1, keepdim=True)
        valid_rows = finite_teacher & finite_logits & (teacher_mass > 1e-6)

        norm_teacher = torch.where(
            valid_rows.expand_as(teacher_f),
            teacher_f / teacher_mass.clamp_min(1e-6),
            torch.zeros_like(teacher_f),
        )
        uniform = torch.full_like(norm_teacher, 1.0 / max(int(norm_teacher.shape[1]), 1))
        safe_teacher = torch.where(
            valid_rows.expand_as(norm_teacher),
            norm_teacher.clamp_min(1e-8),
            uniform,
        )
        safe_teacher = safe_teacher / safe_teacher.sum(dim=1, keepdim=True).clamp_min(1e-6)
        safe_logits = torch.where(
            valid_rows.expand_as(logits_f),
            logits_f,
            torch.zeros_like(logits_f),
        )

        log_probs = F.log_softmax(safe_logits / self.temperature, dim=1)
        probs = log_probs.exp()
        ce_map = F.kl_div(log_probs, safe_teacher.detach(), reduction="none").sum(dim=1, keepdim=True)
        entropy_map = -(log_probs * probs).sum(dim=1, keepdim=True)
        valid_mask = valid_rows.to(device=ce_map.device, dtype=ce_map.dtype)
        if point_mask is None:
            denom = valid_mask.sum().clamp_min(1.0)
            ce = (ce_map * valid_mask).sum() / denom
            entropy = (entropy_map * valid_mask).sum() / denom
        else:
            if point_mask.ndim == 2:
                point_mask = point_mask.unsqueeze(1)
            if point_mask.ndim != 3 or point_mask.shape[0] != ce_map.shape[0] or point_mask.shape[2] != ce_map.shape[2]:
                raise ValueError("point_mask must broadcast to [B, 1, N].")
            mask = point_mask.to(device=ce_map.device, dtype=ce_map.dtype) * valid_mask
            denom = mask.sum().clamp_min(1.0)
            ce = (ce_map * mask).sum() / denom
            entropy = (entropy_map * mask).sum() / denom
        if entropy_weight == 0.0:
            return ce
        return ce - float(entropy_weight) * entropy
