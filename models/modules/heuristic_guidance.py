"""ana_den6の知見をmyNetへ軽量に接続するHeuristic guidance。

Heuristicは最終行動を固定せず、Where・Amount・Actionの初期priorと
Networkが探索する局所範囲だけを与える。最終的な選択は既存の学習headと
actual SparsePCGC損失へ残す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch


@dataclass(frozen=True)
class HeuristicProfile:
    dataset: str
    scale_m: int
    total_ratio: float
    add_share: float
    prune_share: float
    adjust_share: float
    add_heuristic: str
    prune_heuristic: str
    adjust_heuristic: str
    source: str


# ana_den5/ana_den6のactual結果から得た初期点である。
# total_ratioは百分率ではなく0-1の割合で保持する。
_PROFILES: Dict[tuple[str, int], HeuristicProfile] = {
    ("8i", 8): HeuristicProfile(
        "8i", 8, 0.0025, 0.40, 0.40, 0.20,
        "geometry_safe_rate", "subtree_collapse", "hotspot_cluster",
        "ana_den6_25pattern_best",
    ),
    ("8i", 7): HeuristicProfile(
        "8i", 7, 0.0005, 0.35, 0.30, 0.35,
        "subtree_collapse", "geometry_safe_rate", "hotspot_cluster",
        "ana_den6_25pattern_best",
    ),
    ("mvub", 8): HeuristicProfile(
        "mvub", 8, 0.0025, 0.35, 0.30, 0.35,
        "subtree_collapse", "geometry_safe_rate", "direct_bit_density",
        "ana_den6_25pattern_best",
    ),
    ("mvub", 7): HeuristicProfile(
        "mvub", 7, 0.0010, 0.35, 0.30, 0.35,
        "geometry_safe_rate", "direct_bit_density", "geometry_safe_rate",
        "ana_den6_safe_m7_initial",
    ),
    ("uvg", 8): HeuristicProfile(
        "uvg", 8, 0.0050, 0.25, 0.70, 0.05,
        "geometry_safe_rate", "context_repair", "context_repair",
        "ana_den6_25pattern_best",
    ),
    ("uvg", 7): HeuristicProfile(
        "uvg", 7, 0.0025, 0.40, 0.50, 0.10,
        "geometry_safe_rate", "hotspot_cluster", "geometry_safe_rate",
        "ana_den6_25pattern_best",
    ),
}


_HEURISTIC_NAMES = {
    "direct_bit_density",
    "subtree_collapse",
    "context_repair",
    "hotspot_cluster",
    "geometry_safe_rate",
}


def _dataset_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"8i", "8ivslf"}:
        return "8i"
    if text == "mvub":
        return "mvub"
    if text == "uvg":
        return "uvg"
    return text or "8i"


def _fit_b1n(value: Any, like: torch.Tensor) -> torch.Tensor:
    """任意の点対応値を[B,1,N]へ揃える。"""
    B, _, N = like.shape
    if not torch.is_tensor(value):
        return like.new_zeros((B, 1, N))
    # Heuristicは固定priorであり学習対象ではない。既存Networkのgraphへ接続しない。
    out = value.detach().to(device=like.device, dtype=like.dtype)
    if out.ndim == 1:
        out = out.view(1, 1, -1)
    elif out.ndim == 2:
        out = out.unsqueeze(1)
    elif out.ndim == 3:
        if out.shape[1] != 1:
            out = out[:, :1, :]
    else:
        return like.new_zeros((B, 1, N))
    if out.shape[0] == 1 and B > 1:
        out = out.expand(B, -1, -1)
    if out.shape[0] != B:
        return like.new_zeros((B, 1, N))
    if out.shape[-1] > N:
        out = out[..., :N]
    elif out.shape[-1] < N:
        if out.shape[-1] <= 0:
            return like.new_zeros((B, 1, N))
        out = torch.cat([out, out[..., -1:].expand(B, 1, N - out.shape[-1])], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_score(value: torch.Tensor) -> torch.Tensor:
    """候補順位だけを保つ軽量なbatch内0-1正規化。"""
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lower = value.amin(dim=2, keepdim=True)
    upper = value.amax(dim=2, keepdim=True)
    span = (upper - lower).clamp_min(1e-6)
    normalized = (value - lower) / span
    flat = (upper - lower) <= 1e-6
    normalized = torch.where(flat, torch.full_like(normalized, 0.5), normalized)
    return normalized.clamp(0.0, 1.0)


def _leaf_gain(structure: Mapping[str, Any], operation: str, like: torch.Tensor) -> torch.Tensor:
    diag = structure.get("leaf_pattern_diag") if isinstance(structure, Mapping) else None
    if not isinstance(diag, Mapping) or not bool(diag.get("available", False)):
        return like.new_zeros(like.shape)
    key = {
        "Add": "add_nll_gain",
        "Prune": "delete_nll_gain",
        "Adjust": "move_nll_gain",
    }[operation]
    return _fit_b1n(diag.get(key), like).clamp_min(0.0)


def _score_formula(
    name: str,
    *,
    direct: torch.Tensor,
    descendant: torch.Tensor,
    expected_new: torch.Tensor,
    mask_gain: torch.Tensor,
    context_risk: torch.Tensor,
    hotspot: torch.Tensor,
    geometry: torch.Tensor,
) -> torch.Tensor:
    """ana_den5の式をmyNet既存proxyへ写像した微分不要の候補score。"""
    if name == "direct_bit_density":
        score = direct
    elif name == "subtree_collapse":
        score = direct + descendant - expected_new
    elif name == "context_repair":
        score = direct + mask_gain - context_risk
    elif name == "hotspot_cluster":
        score = direct + descendant + hotspot - expected_new
    elif name == "geometry_safe_rate":
        score = (direct + descendant - expected_new) / (1.0 + geometry.clamp_min(0.0))
    else:
        score = direct
    return _normalize_score(score)


def resolve_profile(args: Any) -> HeuristicProfile:
    dataset = _dataset_key(getattr(args, "dataname", "8i"))
    scale_m = int(getattr(args, "sparsepcgc_scale_m", 8))
    profile = _PROFILES.get((dataset, scale_m))
    if profile is None:
        candidates = [item for (name, _), item in _PROFILES.items() if name == dataset]
        if candidates:
            profile = min(candidates, key=lambda item: abs(int(item.scale_m) - scale_m))
        else:
            profile = _PROFILES[("8i", 8)]

    total_override = float(getattr(args, "heuristic_guidance_total_ratio_percent", -1.0))
    shares_override = str(getattr(args, "heuristic_guidance_operation_shares", "")).strip()
    heuristic_override = str(getattr(args, "heuristic_guidance_operation_heuristics", "")).strip()

    total_ratio = profile.total_ratio if total_override < 0.0 else max(total_override, 0.0) / 100.0
    shares = [profile.add_share, profile.prune_share, profile.adjust_share]
    if shares_override:
        try:
            parsed = [max(float(item.strip()), 0.0) for item in shares_override.split(",")]
            if len(parsed) == 3 and sum(parsed) > 0.0:
                shares = [value / sum(parsed) for value in parsed]
        except Exception:
            pass
    heuristics = [profile.add_heuristic, profile.prune_heuristic, profile.adjust_heuristic]
    if heuristic_override:
        parsed = [item.strip() for item in heuristic_override.split(",")]
        if len(parsed) == 3 and all(item in _HEURISTIC_NAMES for item in parsed):
            heuristics = parsed

    return HeuristicProfile(
        dataset=dataset,
        scale_m=scale_m,
        total_ratio=float(total_ratio),
        add_share=float(shares[0]),
        prune_share=float(shares[1]),
        adjust_share=float(shares[2]),
        add_heuristic=heuristics[0],
        prune_heuristic=heuristics[1],
        adjust_heuristic=heuristics[2],
        source=profile.source + ("+cli_override" if total_override >= 0.0 or shares_override or heuristic_override else ""),
    )


def build_heuristic_guidance(structure: Mapping[str, Any], args: Any) -> Dict[str, Any]:
    """既存のOctree proxyからWhere・Amount・Action priorを作る。"""
    enabled = bool(getattr(args, "heuristic_guidance_enabled", True))
    like = structure.get("occupancy_nll_proxy") if isinstance(structure, Mapping) else None
    if not torch.is_tensor(like):
        return {"enabled": False, "reason": "occupancy_nll_proxy_missing"}

    profile = resolve_profile(args)
    if not enabled:
        return {
            "enabled": False,
            "reason": "disabled",
            "dataset": profile.dataset,
            "scale_m": int(profile.scale_m),
            "profile_source": profile.source,
        }

    direct = _normalize_score(_fit_b1n(structure.get("occupancy_nll_proxy"), like))
    node = _normalize_score(_fit_b1n(structure.get("node_proxy"), like))
    single = _normalize_score(_fit_b1n(structure.get("single_proxy"), like))
    lowprob = _normalize_score(_fit_b1n(structure.get("lowprob_proxy"), like))
    context = _normalize_score(_fit_b1n(structure.get("context_proxy"), like))
    quant = _normalize_score(_fit_b1n(structure.get("quant_proxy"), like))
    sparse = _normalize_score(_fit_b1n(structure.get("sparse_proxy"), like))
    outlier = _normalize_score(_fit_b1n(structure.get("outlier_proxy"), like))
    shape = _normalize_score(_fit_b1n(structure.get("shape_proxy"), like))

    descendant = 0.55 * node + 0.45 * single
    context_risk = 0.55 * context + 0.25 * shape + 0.20 * sparse
    hotspot = direct * (0.50 + lowprob) * (0.50 + context) * (0.75 + single)
    geometry = 0.35 * shape + 0.25 * quant + 0.20 * outlier + 0.20 * sparse

    where_prior: Dict[str, torch.Tensor] = {}
    selected_names = {
        "Add": profile.add_heuristic,
        "Prune": profile.prune_heuristic,
        "Adjust": profile.adjust_heuristic,
    }
    for operation, heuristic_name in selected_names.items():
        leaf = _normalize_score(_leaf_gain(structure, operation, like))
        expected_new = geometry if operation in {"Add", "Adjust"} else 0.20 * shape
        where_prior[operation] = _score_formula(
            heuristic_name,
            direct=direct,
            descendant=descendant,
            expected_new=expected_new,
            mask_gain=leaf + 0.25 * lowprob,
            context_risk=context_risk,
            hotspot=hotspot,
            geometry=geometry,
        ).to(dtype=like.dtype).detach()

    shares = {
        "Add": profile.add_share,
        "Prune": profile.prune_share,
        "Adjust": profile.adjust_share,
    }
    amount_prior = {
        name: float(profile.total_ratio * share)
        for name, share in shares.items()
    }
    max_share = max(shares.values()) if shares else 1.0
    # gateは全操作を候補に残しつつ、den6で有望だった構成比を初期Action priorにする。
    action_gate_prior = {
        name: float(0.50 + 0.50 * share / max(max_share, 1e-9))
        for name, share in shares.items()
    }

    collect_debug = bool(
        (getattr(args, "verbose_step_logs", False) and getattr(args, "_log_this_step", True))
        or getattr(args, "_collect_structure_debug", False)
    )
    where_prior_mean = {}
    if collect_debug:
        # 通常StepでGPU同期を起こさないよう、CPU scalar化はログ対象時だけ行う。
        where_prior_mean = {
            name: float(value.detach().float().mean().cpu())
            for name, value in where_prior.items()
        }

    return {
        "enabled": True,
        "profile_source": profile.source,
        "dataset": profile.dataset,
        "scale_m": int(profile.scale_m),
        "total_ratio": float(profile.total_ratio),
        "operation_shares": shares,
        "operation_heuristics": selected_names,
        "amount_prior": amount_prior,
        "action_gate_prior": action_gate_prior,
        "where_prior": where_prior,
        "where_prior_mean": where_prior_mean,
        "formula_basis": "ana_den5_formula_mapped_to_existing_mynet_octree_proxy",
    }
