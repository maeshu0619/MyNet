"""ana_den6のexact候補順位をmyNetのWhere・Amount・Actionへ接続する。

``ana_den6_residual`` ではmyNet proxyによる近似再計算を禁止し、
ana_den5/ana_den6が生成した順位付きEditCandidate poolを唯一の探索空間とする。
Step 0のhard行動はden6 anchor、以後は既存Network headのresidualだけで
候補順位・操作量・Action構成を微調整する。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping

import torch


_EXACT_GUIDANCE_CACHE: "OrderedDict[tuple[str, str, str, int], Dict[str, Any]]" = OrderedDict()


def release_exact_guidance_cache(args: Any = None) -> None:
    """Release mapped per-step guidance without changing its CPU manifest."""
    _EXACT_GUIDANCE_CACHE.clear()
    if args is not None:
        cache = getattr(args, "_ana_den6_exact_tensor_guidance_cache", None)
        if isinstance(cache, dict):
            cache.clear()


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
        "mvub", 8, 0.0025, 0.50, 0.40, 0.10,
        "subtree_collapse", "geometry_safe_rate", "direct_bit_density",
        "ana_den6_state_actual_0p25",
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


_NEIGHBOR_OFFSETS_26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
)
_OFFSET_TO_INDEX = {offset: index for index, offset in enumerate(_NEIGHBOR_OFFSETS_26)}


def _normalize_coords_b3n(value: Any, like: torch.Tensor) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    coords = value.detach().to(device=like.device, dtype=torch.long)
    if coords.ndim == 2:
        if coords.shape[0] == 3:
            coords = coords.unsqueeze(0)
        elif coords.shape[1] == 3:
            coords = coords.transpose(0, 1).contiguous().unsqueeze(0)
        else:
            return None
    elif coords.ndim == 3:
        if coords.shape[1] == 3:
            coords = coords.contiguous()
        elif coords.shape[2] == 3:
            coords = coords.permute(0, 2, 1).contiguous()
        else:
            return None
    else:
        return None
    if coords.shape[0] == 1 and like.shape[0] > 1:
        coords = coords.expand(like.shape[0], -1, -1).contiguous()
    if coords.shape[0] != like.shape[0] or coords.shape[2] != like.shape[2]:
        return None
    return coords


def _rank_value(rank: int, pool_size: int) -> float:
    if pool_size <= 1:
        return 1.0
    return float(1.0 - float(rank) / float(max(pool_size - 1, 1)))


def _exact_den6_guidance(
    structure: Mapping[str, Any],
    args: Any,
    like: torch.Tensor,
    exact: Mapping[str, Any],
) -> Dict[str, Any]:
    """den6 pool rankを現在のcanonical voxel行へ厳密に対応付ける。"""
    if str(exact.get("source", "")) not in {
        "ana_den6_exact_ranked_candidate_pool_v2",
        "ana_den6_exact_ranked_candidate_pool_online_v1",
        "ana_den6_exact_compact_candidate_shortlist_online_v3",
        "ana_den6_exact_compact_candidate_shortlist_online_v5",
        "ana_den6_exact_one_pattern_anchor_online_v4",
        "ana_den6_exact_one_pattern_anchor_online_v5",
        "ana_den6_exact_unique_plan_online_v6",
        "ana_den6_exact_one_pattern_anchor_online_v6",
    }:
        raise RuntimeError("ana_den6 exact candidate guidance sourceが不正である")
    coords = _normalize_coords_b3n(structure.get("global_voxel_coords"), like)
    if coords is None:
        raise RuntimeError(
            "ana_den6 exact guidanceにはActuatorと同じglobal_voxel_coordsが必要である"
        )

    # den6 Heuristicはframe固定であり学習graphを持たない。
    # 同一frame・同一deviceでは座標辞書作成と候補写像を再実行せず、そのまま再利用する。
    # Different manifests for the same voxel count must never share a mapped
    # candidate tensor.  The prior key used only input_voxel_hash and could
    # return a stale v2/v4 payload after a v5 cache refresh.
    cache_signature = str(exact.get("cache_signature", "")).strip()
    if not cache_signature:
        cache_signature = "|".join(
            str(exact.get(name, ""))
            for name in (
                "source",
                "schema_version",
                "input_sha256",
                "input_voxel_hash",
                "manifest_sha256",
                "den5_sha256",
                "den6_sha256",
            )
        )
    cache_key = (cache_signature, str(like.device), str(like.dtype), int(coords.shape[-1]))
    cached_guidance = _EXACT_GUIDANCE_CACHE.get(cache_key)
    if isinstance(cached_guidance, dict):
        _EXACT_GUIDANCE_CACHE.move_to_end(cache_key)
        return cached_guidance

    # online/residual training visits a different full-cloud frame every step.
    # Keeping mapped CUDA tensors for old frames cannot produce a cache hit in
    # the current step and used several GiB per frame.  Drop the previous frame
    # *before* allocating the next one, while retaining same-frame reuse above.
    mode = str(getattr(args, "heuristic_guidance_mode", "proxy_prior")).strip().lower()
    if mode in {"ana_den6_online", "ana_den6_residual"}:
        _EXACT_GUIDANCE_CACHE.clear()

    pools = exact.get("operation_candidate_shortlists")
    if not isinstance(pools, Mapping):
        # 既存manifestは同じden6順位をranked_candidate_pools名で保持している。
        pools = exact.get("ranked_candidate_pools")
    if not isinstance(pools, Mapping):
        raise RuntimeError("ana_den6 exact guidanceに局所候補shortlistが無い")
    B, _, N = coords.shape
    where_prior = {name: like.new_zeros((B, 1, N)) for name in ("Add", "Prune", "Adjust")}
    candidate_mask = {name: torch.zeros((B, 1, N), device=like.device, dtype=torch.bool) for name in where_prior}
    # target方向はdense [B,26,N]で常駐させると8iで100MB級になるため、
    # den6候補pairだけをCOO形式で保持し、Actuatorの既存方向logit上へ必要時にscatterする。
    direction_sparse_lists = {
        "Add": {"batch": [], "source": [], "direction": [], "score": []},
        "Adjust": {"batch": [], "source": [], "direction": [], "score": []},
    }
    # full-cloud hard planをden6と同じconflict ruleで再構築するための候補index写像。
    candidate_map_lists = {
        operation: {
            "pool_rank": [],
            "rank_score": [],
            "source_index": [],
            "direction_index": [],
            "pair_candidate_index": [],
            "pair_source_index": [],
            "pair_direction_index": [],
        }
        for operation in ("Add", "Prune", "Adjust")
    }
    mapped = {name: 0 for name in where_prior}
    unmapped = {name: 0 for name in where_prior}

    # manifestはCPU上の固定Heuristicである。1 frameにつき一度だけ座標辞書を作る。
    for b in range(B):
        rows = coords[b].transpose(0, 1).detach().cpu().tolist()
        lookup = {tuple(int(value) for value in row): index for index, row in enumerate(rows)}
        occupied = set(lookup)
        for operation in ("Add", "Prune", "Adjust"):
            pool = pools.get(operation)
            if not isinstance(pool, list):
                raise RuntimeError(f"ana_den6 exact guidanceの{operation} poolが不正である")
            pool_size = len(pool)
            for fallback_rank, candidate in enumerate(pool):
                if not isinstance(candidate, Mapping):
                    continue
                rank = int(candidate.get("pool_rank", fallback_rank))
                # compact cacheでは元のden6 pool順位を保持したscoreを使う。
                # shortlist長で再正規化するとStep 0の順位が変わってしまう。
                raw_rank_score = candidate.get("rank_score", None)
                try:
                    rank_score = float(raw_rank_score)
                except (TypeError, ValueError):
                    rank_score = _rank_value(rank, pool_size)
                if not math.isfinite(rank_score):
                    rank_score = _rank_value(rank, pool_size)
                remove_coords = [tuple(int(v) for v in coord) for coord in candidate.get("remove_coords", ())]
                add_coords = [tuple(int(v) for v in coord) for coord in candidate.get("add_coords", ())]
                mapped_this = False
                candidate_source_index = -1
                candidate_direction_index = -1
                candidate_add_pairs = []
                if operation == "Prune":
                    for source in remove_coords:
                        source_index = lookup.get(source)
                        if source_index is None:
                            continue
                        where_prior[operation][b, 0, source_index] = torch.maximum(
                            where_prior[operation][b, 0, source_index],
                            like.new_tensor(rank_score),
                        )
                        candidate_mask[operation][b, 0, source_index] = True
                        if candidate_source_index < 0:
                            candidate_source_index = int(source_index)
                        mapped_this = True
                elif operation == "Adjust":
                    if len(remove_coords) == 1 and len(add_coords) == 1:
                        source = remove_coords[0]
                        target = add_coords[0]
                        source_index = lookup.get(source)
                        offset = tuple(target[axis] - source[axis] for axis in range(3))
                        direction_index = _OFFSET_TO_INDEX.get(offset)
                        if source_index is not None and direction_index is not None:
                            where_prior[operation][b, 0, source_index] = torch.maximum(
                                where_prior[operation][b, 0, source_index],
                                like.new_tensor(rank_score),
                            )
                            candidate_mask[operation][b, 0, source_index] = True
                            pair = direction_sparse_lists[operation]
                            pair["batch"].append(b)
                            pair["source"].append(source_index)
                            pair["direction"].append(direction_index)
                            pair["score"].append(rank_score)
                            candidate_source_index = int(source_index)
                            candidate_direction_index = int(direction_index)
                            mapped_this = True
                else:  # Addはtargetに隣接するoccupied sourceへ同じrankを写像する。
                    for target in add_coords:
                        for direction_index, offset in enumerate(_NEIGHBOR_OFFSETS_26):
                            source = tuple(target[axis] - offset[axis] for axis in range(3))
                            if source not in occupied:
                                continue
                            source_index = lookup[source]
                            where_prior[operation][b, 0, source_index] = torch.maximum(
                                where_prior[operation][b, 0, source_index],
                                like.new_tensor(rank_score),
                            )
                            candidate_mask[operation][b, 0, source_index] = True
                            pair = direction_sparse_lists[operation]
                            pair["batch"].append(b)
                            pair["source"].append(source_index)
                            pair["direction"].append(direction_index)
                            pair["score"].append(rank_score)
                            candidate_add_pairs.append((int(source_index), int(direction_index)))
                            mapped_this = True
                if b == 0:
                    candidate_map = candidate_map_lists[operation]
                    candidate_map["pool_rank"].append(int(rank))
                    candidate_map["rank_score"].append(float(rank_score))
                    candidate_map["source_index"].append(int(candidate_source_index))
                    candidate_map["direction_index"].append(int(candidate_direction_index))
                    for source_index, direction_index in candidate_add_pairs:
                        candidate_map["pair_candidate_index"].append(int(fallback_rank))
                        candidate_map["pair_source_index"].append(int(source_index))
                        candidate_map["pair_direction_index"].append(int(direction_index))
                if mapped_this:
                    mapped[operation] += 1
                else:
                    unmapped[operation] += 1

    direction_sparse = {}
    for operation, values in direction_sparse_lists.items():
        direction_sparse[operation] = {
            "batch_index": torch.as_tensor(values["batch"], device=like.device, dtype=torch.long),
            "source_index": torch.as_tensor(values["source"], device=like.device, dtype=torch.long),
            "direction_index": torch.as_tensor(values["direction"], device=like.device, dtype=torch.long),
            "rank_score": torch.as_tensor(values["score"], device=like.device, dtype=like.dtype),
        }

    candidate_tensor_map = {}
    for operation, values in candidate_map_lists.items():
        candidate_tensor_map[operation] = {
            "pool_rank": torch.as_tensor(values["pool_rank"], device=like.device, dtype=torch.long),
            "rank_score": torch.as_tensor(values["rank_score"], device=like.device, dtype=like.dtype),
            "source_index": torch.as_tensor(values["source_index"], device=like.device, dtype=torch.long),
            "direction_index": torch.as_tensor(values["direction_index"], device=like.device, dtype=torch.long),
            "pair_candidate_index": torch.as_tensor(values["pair_candidate_index"], device=like.device, dtype=torch.long),
            "pair_source_index": torch.as_tensor(values["pair_source_index"], device=like.device, dtype=torch.long),
            "pair_direction_index": torch.as_tensor(values["pair_direction_index"], device=like.device, dtype=torch.long),
        }

    shares_raw = exact.get("operation_shares") or {}
    shares = {name: max(float(shares_raw.get(name, 0.0)), 0.0) for name in where_prior}
    share_sum = sum(shares.values())
    if share_sum <= 0.0:
        raise RuntimeError("ana_den6 exact guidanceのoperation share合計が0である")
    shares = {name: value / share_sum for name, value in shares.items()}
    total_ratio = max(float(exact.get("total_ratio", 0.0)), 0.0)
    amount_prior = {name: total_ratio * shares[name] for name in shares}
    max_share = max(shares.values())
    action_gate_prior = {
        name: float(0.50 + 0.50 * shares[name] / max(max_share, 1e-9))
        for name in shares
    }
    guidance = {
        "enabled": True,
        "profile_source": str(exact.get("source", "")),
        "dataset": _dataset_key(exact.get("dataset", getattr(args, "dataname", "8i"))),
        "scale_m": int(exact.get("scale_m", getattr(args, "sparsepcgc_scale_m", 8))),
        "total_ratio": total_ratio,
        "operation_shares": shares,
        "operation_heuristics": dict(exact.get("operation_heuristics") or {}),
        "amount_prior": amount_prior,
        "action_gate_prior": action_gate_prior,
        "where_prior": {name: value.detach() for name, value in where_prior.items()},
        "candidate_mask": {name: value.detach() for name, value in candidate_mask.items()},
        "target_direction_sparse": {
            name: {key: value.detach() for key, value in values.items()}
            for name, values in direction_sparse.items()
        },
        "candidate_tensor_map": {
            name: {key: value.detach() for key, value in values.items()}
            for name, values in candidate_tensor_map.items()
        },
        "where_prior_mean": {},
        "formula_basis": (
            str(exact.get("source", ""))
            if str(exact.get("source", ""))
            in {
                "ana_den6_exact_one_pattern_anchor_online_v4",
                "ana_den6_exact_one_pattern_anchor_online_v5",
                "ana_den6_exact_compact_candidate_shortlist_online_v5",
                "ana_den6_exact_unique_plan_online_v6",
                "ana_den6_exact_one_pattern_anchor_online_v6",
            }
            else "ana_den6_exact_ranked_editcandidate_pool_online_v1"
            if str(exact.get("source", "")) == "ana_den6_exact_ranked_candidate_pool_online_v1"
            else "ana_den6_exact_ranked_editcandidate_pool_v2"
        ),
        "exact_candidate_guidance": exact,
        "exact_candidate_mapped_count": mapped,
        "exact_candidate_unmapped_count": unmapped,
    }
    _EXACT_GUIDANCE_CACHE[cache_key] = guidance
    _EXACT_GUIDANCE_CACHE.move_to_end(cache_key)
    while len(_EXACT_GUIDANCE_CACHE) > 8:
        _EXACT_GUIDANCE_CACHE.popitem(last=False)
    return guidance


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

    mode = str(getattr(args, "heuristic_guidance_mode", "proxy_prior")).strip().lower()
    exact = structure.get("ana_den6_ranked_candidate_guidance") if isinstance(structure, Mapping) else None
    if mode in {"ana_den6_online", "ana_den6_residual"}:
        if not isinstance(exact, Mapping):
            raise RuntimeError(
                "ana_den6 online/residualでexact candidate guidanceがNetworkへ伝播していない。"
                "proxy_priorへ代替してはならない"
            )
        manifest_key = str(
            exact.get("cache_signature", exact.get("manifest_sha256", exact.get("manifest_path", "")))
        )
        subtree_key = str(getattr(args, "_current_subtree_id", ""))
        cache_key = (
            manifest_key,
            subtree_key,
            str(like.device),
            str(like.dtype),
            int(like.shape[0]),
            int(like.shape[-1]),
        )
        cache = getattr(args, "_ana_den6_exact_tensor_guidance_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            setattr(args, "_ana_den6_exact_tensor_guidance_cache", cache)
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            cache.move_to_end(cache_key)
            return cached
        # The next sequential frame cannot reuse the previous frame's mapped
        # CUDA guidance.  Clear it before constructing the replacement so peak
        # memory never contains both full-cloud mappings at once.
        cache.clear()
        guidance = _exact_den6_guidance(structure, args, like, exact)
        cache[cache_key] = guidance
        cache.move_to_end(cache_key)
        max_entries = max(int(getattr(args, "heuristic_guidance_tensor_cache_entries", 8)), 1)
        while len(cache) > max_entries:
            cache.popitem(last=False)
        return guidance

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
        "formula_basis": "mynet_proxy_approximation_not_den6_editcandidate_ranking",
    }
