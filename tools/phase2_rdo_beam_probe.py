#!/usr/bin/env python
"""Phase2 RDO beam-search probe for context-aware SparsePCGC edits.

Research-only script.  It does not import train.py and does not alter the
training policy.  It reuses candidate generators from context_aware_where_probe
and evaluates cumulative edit sequences with actual SparsePCGC raw bits.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.utils.config.args import parse_pugan_args
from models.utils.data.dataset import load_ply
from models.utils.loss.actual_encoder import build_actual_encoder
from utils.compress.evaluation import EvaluationConfig, evaluate_decoded_geometry
from utils.compress.ply_io import read_ply_xyz, write_ascii_ply_xyz

from tools.context_aware_where_probe import (
    _coord_key_setup,
    _coords_to_xyz,
    _hist,
    _lookup_occupied,
    _neighbor_count,
    _parent_info,
    _parse_csv_text,
    _phase1_rule_candidate_score,
    _quota_select,
    _safe_float,
    _unique_coords,
    build_candidate_coords,
    context_metrics,
    geometry_proxy,
)


DEFAULT_METHODS = (
    "addv2_n512_g0p3_nn1_densmedium",
    "addv2_n256_g0p3_nn1_densmedium",
    "high_nll_branch_prune",
    "high_nll_branch_prune_soft_score",
    "pattern_projection_add_n256",
    "pattern_projection_add_n512",
    "high_nll_parent_group_prune",
    "voxel_merge_snap",
)
DEFAULT_DIAGNOSTIC_METHODS = ("smacro_cap0p0025_maxb1_top10_geommedium",)
PHASE1_METHODS = (
    "noop",
    "block_only",
    "addv2_n512_g0p3_nn1_densmedium",
    "addv2_n256_g0p3_nn1_densmedium",
    "high_nll_branch_prune",
    "voxel_merge_snap",
    "smacro_cap0p0025_maxb1_top10_geommedium",
)
SPARSEPCGC_RATE_DEBUG_KEYS = (
    "sparsepcgc_estimated_occupancy_bits",
    "sparsepcgc_pred_occupancy_nll",
    "sparsepcgc_prob_true_mean",
    "sparsepcgc_prob_true_low_ratio",
    "sparsepcgc_occupied_low_prob_ratio",
    "sparsepcgc_bits_by_depth_json",
    "sparsepcgc_candidates_by_depth_json",
    "sparsepcgc_occupied_by_depth_json",
    "sparsepcgc_low_prob_occupied_by_depth_json",
    "sparsepcgc_high_bit_nodes_by_depth_json",
    "sparsepcgc_bits_by_parent_popcount_json",
    "sparsepcgc_bits_by_child_pattern_topk_json",
    "sparsepcgc_bits_by_block_topk_json",
    "sparsepcgc_top_high_bit_nodes_json",
)


@dataclass
class BeamState:
    state_id: str
    coords: torch.Tensor
    raw_bit: float
    actual_raw_percent: float
    objective_j: float
    edit_sequence: str
    total_add_count: int = 0
    total_prune_count: int = 0
    total_move_count: int = 0
    total_merge_count: int = 0
    cumulative_add_ratio: float = 0.0
    cumulative_prune_ratio: float = 0.0
    geometry_proxy_summary: float = 0.0
    add_density_delta_mean: float = 0.0


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _bool_text(value: bool) -> str:
    return "True" if bool(value) else "False"


def _high_nll_guard(row: Mapping[str, object], mode: str) -> Tuple[bool, float, str]:
    partial = _safe_float(row.get("partial_context_damage_ratio"), 1.0)
    parent = _safe_float(row.get("parent_emptying_ratio"), 0.0)
    score_proxy = _safe_float(row.get("score_proxy"), 0.0)
    if mode == "hard":
        ok = partial <= 0.60 and parent >= 0.45
        threshold = 0.45
    elif mode == "relaxed":
        ok = partial <= 0.60 and parent >= 0.40
        threshold = 0.40
    elif mode == "soft":
        ok = partial <= 0.65 and parent >= 0.35
        threshold = 0.35
    else:
        raise ValueError(f"unknown high_nll guard mode: {mode}")
    score = 1000.0 + parent * 120.0 - partial * 120.0 + min(score_proxy, 10000.0) * 1e-4
    reason = f"high_nll_{mode}_guard" if ok else f"high_nll_{mode}_reject_parent_ge_{threshold:.2f}"
    return bool(ok), float(score), reason


def _beam_candidate_allowed(
    method: str,
    row: Mapping[str, object],
    state: BeamState,
    *,
    high_nll_guard_mode: str,
    max_total_add_count: int,
    max_cumulative_add_ratio: float,
) -> Tuple[bool, float, str]:
    if method == "stop":
        return True, 0.0, "stop"
    if method in {"high_nll_branch_prune", "high_nll_branch_prune_soft_score", "high_nll_parent_group_prune"}:
        return _high_nll_guard(row, high_nll_guard_mode)
    if method == "voxel_merge_snap":
        partial = _safe_float(row.get("partial_context_damage_ratio"), 1.0)
        move_distance = _safe_float(row.get("move_distance_mean"), float("inf"))
        ok = move_distance <= 1.0 and partial <= 0.60
        score = 800.0 - partial * 100.0 + min(_safe_float(row.get("score_proxy"), 0.0), 10000.0) * 1e-4
        return bool(ok), float(score), "guarded_merge_snap" if ok else "merge_guard_reject"
    if method.startswith("addv2_") or str(row.get("operation_type", "")) == "add":
        add_count = int(_safe_float(row.get("add_count"), 0.0))
        add_nn_max = _safe_float(row.get("added_point_nn_distance_max"), float("inf"))
        density_guard = str(row.get("density_guard", ""))
        cumulative_add_ratio = float(state.cumulative_add_ratio) + _safe_float(row.get("add_ratio"), 0.0)
        ok = (
            add_count > 0
            and add_nn_max <= 1.0
            and density_guard in {"medium", "strict"}
            and int(state.total_add_count) + add_count <= int(max_total_add_count)
            and cumulative_add_ratio <= float(max_cumulative_add_ratio)
        )
        score = 900.0 + min(float(add_count), 512.0) / 512.0 * 50.0 + min(_safe_float(row.get("score_proxy"), 0.0), 10000.0) * 1e-4
        return bool(ok), float(score), "guarded_add" if ok else "add_guard_reject"
    if method.startswith("smacro_") or method.startswith("smacro"):
        return False, float("-inf"), "diagnostic_only"
    return True, 0.0, "unguarded"


def _objective_j(
    actual_raw_percent: float,
    *,
    cumulative_add_ratio: float,
    geometry_proxy_value: float,
    add_density_delta: float,
    lambda_add: float,
    lambda_geom: float,
    lambda_density: float,
) -> float:
    return (
        float(actual_raw_percent)
        + float(lambda_add) * float(cumulative_add_ratio)
        + float(lambda_geom) * float(geometry_proxy_value)
        + float(lambda_density) * float(add_density_delta)
    )


def _operation_family(method: str, operation_type: str) -> str:
    if method == "stop":
        return "stop"
    if method in {"add_then_high_nll_local", "high_nll_then_add_repair", "small_macro_then_add_repair"}:
        return "composite"
    if method.startswith("addv2_") or method.startswith("pattern_projection_add"):
        return "add_pattern_repair"
    if method.startswith("high_nll"):
        return "high_nll_prune"
    if method == "rate_anchor_group_prune":
        return "rate_anchor_group_prune"
    if method == "voxel_merge_snap":
        return "merge_snap"
    if method.startswith("smacro"):
        return "small_macro"
    return str(operation_type or "unknown")


def _canonical_method(method: str) -> str:
    if method == "pattern_projection_add_n256":
        return "addv2_n256_g0p3_nn1_densmedium"
    if method == "pattern_projection_add_n512":
        return "addv2_n512_g0p3_nn1_densmedium"
    if method == "high_nll_branch_prune_soft_score":
        return "high_nll_branch_prune"
    if method == "high_nll_parent_group_prune":
        return "hnllv2_q80_pop3_geommedium"
    if method == "rate_anchor_group_prune":
        return "ctxA_pop3_top10_maxg8"
    return method


def _parse_json_payload(value: object, fallback):
    if isinstance(value, (dict, list)):
        return value
    try:
        if value in ("", None):
            return fallback
        return json.loads(str(value))
    except Exception:
        return fallback


def _mask_for_scaled_node(coords: torch.Tensor, coord: Sequence[int], depth: int) -> torch.Tensor:
    target = torch.tensor(list(coord)[:3], device=coords.device, dtype=torch.long)
    if int(depth) <= 0:
        return (coords == target.reshape(1, 3)).all(dim=1)
    scaled = torch.div(coords, int(2 ** int(depth)), rounding_mode="floor")
    return (scaled == target.reshape(1, 3)).all(dim=1)


def _high_actual_bit_node_mask(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget_ratio: float,
    max_nodes: int,
    prefer_depths: Sequence[int] = (0, 1),
) -> Tuple[torch.Tensor, Dict[str, object]]:
    target = min(max(int(math.ceil(coords.shape[0] * float(budget_ratio))), 0), max(int(max_nodes), 0))
    drop_mask = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    rows = _parse_json_payload(base_stats.get("sparsepcgc_top_high_bit_nodes_json", ""), [])
    rows = [r for r in rows if bool(r.get("occupied", False))]
    depth_set = {int(d) for d in prefer_depths}
    rows.sort(
        key=lambda r: (
            0 if int(r.get("depth", 999)) in depth_set else 1,
            -float(r.get("bits", 0.0)),
        )
    )
    selected_bits: List[float] = []
    selected_depths: List[int] = []
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    for r in rows:
        if int(drop_mask.sum().item()) >= target:
            break
        depth = int(r.get("depth", 0))
        mask = _mask_for_scaled_node(coords, r.get("coord", []), depth) & (~drop_mask)
        if int(mask.sum().item()) <= 0:
            continue
        # Prefer nodes that are not isolated surface-critical singletons.
        if float(neigh[mask].mean().item()) < 1.0 and int(mask.sum().item()) <= 2:
            continue
        room = target - int(drop_mask.sum().item())
        idx = mask.nonzero(as_tuple=False).reshape(-1)
        if int(idx.numel()) > room:
            idx = idx[:room]
        drop_mask[idx] = True
        selected_bits.append(float(r.get("bits", 0.0)) * float(idx.numel()))
        selected_depths.extend([depth] * int(idx.numel()))
    bit_sum = float(sum(selected_bits))
    return drop_mask, {
        "candidate_family": "actual_bit_node",
        "selected_bit_sum": bit_sum,
        "selected_bit_mean": bit_sum / max(float(drop_mask.sum().item()), 1.0),
        "selected_bit_max": max(selected_bits) if selected_bits else 0.0,
        "selected_depth_hist_json": json.dumps({str(k): selected_depths.count(k) for k in sorted(set(selected_depths))}, sort_keys=True),
        "high_bit_node_count": len(selected_bits),
    }


def _high_actual_bit_parent_mask(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget_ratio: float,
    max_parents: int,
    block_filter: bool = False,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    target = min(max(int(math.ceil(coords.shape[0] * float(budget_ratio))), 0), coords.shape[0] - 1)
    top_nodes = [r for r in _parse_json_payload(base_stats.get("sparsepcgc_top_high_bit_nodes_json", ""), []) if bool(r.get("occupied", False))]
    top_blocks = _parse_json_payload(base_stats.get("sparsepcgc_bits_by_block_topk_json", ""), [])
    allowed_blocks = set()
    if block_filter:
        for item in top_blocks[:8]:
            text = str(item.get("block", ""))
            if ":" in text:
                d, rest = text.split(":", 1)
                if d == "d0":
                    try:
                        allowed_blocks.add(tuple(int(v) for v in rest.split(",")))
                    except Exception:
                        pass
    _up, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    parent = torch.div(coords, 2, rounding_mode="floor")
    parent_bits: Dict[Tuple[int, int, int], float] = {}
    parent_depths: Dict[Tuple[int, int, int], List[int]] = {}
    for r in top_nodes:
        depth = int(r.get("depth", 0))
        coord = torch.tensor(list(r.get("coord", []))[:3], device=coords.device, dtype=torch.long)
        if coord.numel() != 3:
            continue
        if depth <= 0:
            p = tuple(int(v) for v in torch.div(coord, 2, rounding_mode="floor").tolist())
        else:
            # Map coarser context back to the nearest fine parent bucket.
            p = tuple(int(v) for v in (coord * int(2 ** max(depth - 1, 0))).tolist())
        if block_filter and tuple(int(v // 32) for v in p) not in allowed_blocks:
            continue
        parent_bits[p] = parent_bits.get(p, 0.0) + float(r.get("bits", 0.0))
        parent_depths.setdefault(p, []).append(depth)
    order = sorted(parent_bits.items(), key=lambda item: float(item[1]), reverse=True)[: max(int(max_parents), 1)]
    drop_mask = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    selected_bits = 0.0
    selected_parents = 0
    selected_depths: List[int] = []
    for p, bits in order:
        if int(drop_mask.sum().item()) >= target:
            break
        p_t = torch.tensor(p, device=coords.device, dtype=torch.long)
        mask = (parent == p_t.reshape(1, 3)).all(dim=1) & (~drop_mask)
        if int(mask.sum().item()) <= 0:
            continue
        room = target - int(drop_mask.sum().item())
        idx = mask.nonzero(as_tuple=False).reshape(-1)
        if int(idx.numel()) > room:
            idx = idx[:room]
        drop_mask[idx] = True
        selected_bits += float(bits)
        selected_parents += 1
        selected_depths.extend(parent_depths.get(p, []))
    return drop_mask, {
        "candidate_family": "actual_bit_block_parent" if block_filter else "actual_bit_parent",
        "selected_bit_sum": float(selected_bits),
        "selected_bit_mean": float(selected_bits) / max(float(drop_mask.sum().item()), 1.0),
        "selected_bit_max": float(order[0][1]) if order else 0.0,
        "selected_parent_count": int(selected_parents),
        "selected_depth_hist_json": json.dumps({str(k): selected_depths.count(k) for k in sorted(set(selected_depths))}, sort_keys=True),
    }


def _high_bit_voxel_scores(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    max_pool: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map SparsePCGC high-bit debug nodes back to fine voxel candidates."""
    n = int(coords.shape[0])
    score = torch.full((n,), float("-inf"), device=coords.device, dtype=torch.float32)
    bit_score = torch.zeros((n,), device=coords.device, dtype=torch.float32)
    depth_score = torch.full((n,), -1, device=coords.device, dtype=torch.long)
    rows = [
        r
        for r in _parse_json_payload(base_stats.get("sparsepcgc_top_high_bit_nodes_json", ""), [])
        if bool(r.get("occupied", False))
    ][: max(int(max_pool), 0)]
    for r in rows:
        depth = int(r.get("depth", 0))
        bits = float(r.get("bits", 0.0))
        mask = _mask_for_scaled_node(coords, r.get("coord", []), depth)
        if int(mask.sum().item()) <= 0:
            continue
        update = mask & (bits > bit_score)
        bit_score[update] = float(bits)
        depth_score[update] = int(depth)
    finite = bit_score > 0
    if bool(finite.any().item()):
        # Slightly prefer finer levels when bit scores tie, but do not discard
        # coarser high-bit contexts because they can expand to useful budgets.
        depth_bonus = torch.where(depth_score >= 0, 1.0 / (depth_score.to(dtype=torch.float32) + 1.0), 0.0)
        score[finite] = bit_score[finite] + 0.05 * depth_bonus[finite]
    return score, bit_score, depth_score


def _hist_json_from_tensor(values: torch.Tensor) -> str:
    values = values.detach().to("cpu", dtype=torch.long)
    if values.numel() <= 0:
        return "{}"
    out: Dict[str, int] = {}
    for value in values.tolist():
        out[str(int(value))] = out.get(str(int(value)), 0) + 1
    return json.dumps(out, sort_keys=True)


def _aggressive_quality_proxy(
    coords: torch.Tensor,
    keep_mask: torch.Tensor,
    drop_mask: torch.Tensor,
    *,
    max_samples: int,
) -> Dict[str, float]:
    """Cheap D1/D2-like proxies for research sweeps.

    D1 is approximated as point-to-nearest-kept squared error over original
    points. D2 is a sampled point-to-local-plane proxy on kept neighbors.
    These are not MPEG pc_error numbers, so downstream reports must label them
    as proxies.
    """
    n = int(coords.shape[0])
    drop_count = int(drop_mask.sum().item())
    if n <= 0 or drop_count <= 0:
        return {
            "sampled_chamfer_proxy": 0.0,
            "d1_mse": 0.0,
            "d1_psnr": float("inf"),
            "d2_mse": 0.0,
            "d2_psnr": float("inf"),
            "d1_proxy": 0.0,
            "d2_proxy": 0.0,
            "normal_proxy": 0.0,
        }
    kept = coords[keep_mask].to(dtype=torch.float32)
    removed = coords[drop_mask].to(dtype=torch.float32)
    if int(kept.shape[0]) <= 0:
        peak = float((coords.amax(dim=0) - coords.amin(dim=0)).max().item() + 1.0)
        bad = peak * peak
        return {
            "sampled_chamfer_proxy": bad,
            "d1_mse": bad,
            "d1_psnr": 0.0,
            "d2_mse": bad,
            "d2_psnr": 0.0,
            "d1_proxy": bad,
            "d2_proxy": bad,
            "normal_proxy": 1.0,
        }
    max_removed = min(int(max_samples), int(removed.shape[0]))
    max_kept = min(max(int(max_samples), 512), int(kept.shape[0]))
    removed_sample = removed[:max_removed]
    kept_sample = kept[:max_kept]
    dist = torch.cdist(removed_sample, kept_sample)
    min_sq = dist.min(dim=1).values.pow(2)
    # Mean over all original points, with kept points contributing zero.
    d1_mse = float(min_sq.mean().item()) * (float(drop_count) / max(float(n), 1.0))
    peak = float((coords.amax(dim=0) - coords.amin(dim=0)).max().item() + 1.0)
    d1_psnr = 10.0 * math.log10((peak * peak) / max(d1_mse, 1e-12))

    # Point-to-plane proxy. Keep the sample tiny; this is for ranking/diagnosis,
    # not publication-grade quality measurement.
    d2_vals: List[float] = []
    normal_proxy_vals: List[float] = []
    plane_removed = removed_sample[: min(256, int(removed_sample.shape[0]))]
    if int(plane_removed.shape[0]) > 0 and int(kept_sample.shape[0]) >= 8:
        plane_dist = torch.cdist(plane_removed, kept_sample)
        nn = torch.topk(plane_dist, k=min(8, int(kept_sample.shape[0])), largest=False).indices
        for i in range(int(plane_removed.shape[0])):
            neigh = kept_sample.index_select(0, nn[i])
            center = neigh.mean(dim=0, keepdim=True)
            centered = neigh - center
            cov = centered.t().matmul(centered) / max(float(neigh.shape[0] - 1), 1.0)
            try:
                eigvals, eigvecs = torch.linalg.eigh(cov)
                normal = eigvecs[:, 0]
                d2_vals.append(float(torch.dot(plane_removed[i] - center.reshape(3), normal).pow(2).item()))
                normal_proxy_vals.append(float(eigvals[0].clamp_min(0).item() / eigvals.sum().clamp_min(1e-9).item()))
            except Exception:
                pass
    d2_mse = (float(sum(d2_vals)) / max(float(len(d2_vals)), 1.0)) * (float(drop_count) / max(float(n), 1.0))
    d2_psnr = 10.0 * math.log10((peak * peak) / max(d2_mse, 1e-12)) if d2_vals else float("nan")
    return {
        "sampled_chamfer_proxy": float(d1_mse),
        "d1_mse": float(d1_mse),
        "d1_psnr": float(d1_psnr),
        "d2_mse": float(d2_mse) if d2_vals else float("nan"),
        "d2_psnr": float(d2_psnr),
        "d1_proxy": float(d1_mse),
        "d2_proxy": float(d2_mse) if d2_vals else float("nan"),
        "normal_proxy": float(sum(normal_proxy_vals) / len(normal_proxy_vals)) if normal_proxy_vals else float("nan"),
    }


def _formal_or_sampled_quality(
    *,
    reference_xyz: torch.Tensor,
    decoded_xyz: torch.Tensor,
    coords: torch.Tensor,
    keep_mask: torch.Tensor,
    drop_mask: torch.Tensor,
    max_samples: int,
    formal_max_points: int,
    normal_max_points: int,
    pc_error_path: str = "",
    use_pc_error: bool = False,
) -> Dict[str, object]:
    proxy = _aggressive_quality_proxy(coords, keep_mask, drop_mask, max_samples=int(max_samples))
    out: Dict[str, object] = {
        **proxy,
        "chamfer": proxy.get("sampled_chamfer_proxy", float("nan")),
        "hausdorff_proxy": "",
        "point_to_plane_proxy": proxy.get("d2_mse", ""),
        "quality_eval_mode": "sampled_proxy",
    }
    if int(formal_max_points) <= 0:
        return out
    try:
        with tempfile.TemporaryDirectory(prefix="phase2h_quality_") as tmp:
            ref_path = Path(tmp) / "ref.ply"
            dec_path = Path(tmp) / "dec.ply"
            write_ascii_ply_xyz(ref_path, reference_xyz.detach().to("cpu").numpy().astype(np.float64, copy=False))
            write_ascii_ply_xyz(dec_path, decoded_xyz.detach().to("cpu").numpy().astype(np.float64, copy=False))
            metrics = evaluate_decoded_geometry(
                ref_path,
                dec_path,
                EvaluationConfig(
                    max_points=int(formal_max_points),
                    normal_max_points=int(normal_max_points),
                    emd_points=min(512, max(int(formal_max_points), 1)),
                    normal_k=16,
                    psnr_peak=0.0,
                ),
            )
        out.update({
            "mynet_d1_mse": float(metrics.get("p2point", float("nan"))),
            "mynet_d1_psnr": float(metrics.get("d1_psnr", float("nan"))),
            "mynet_d2_mse": float(metrics.get("p2plane", float("nan"))),
            "mynet_d2_psnr": float(metrics.get("d2_psnr", float("nan"))),
            "mynet_chamfer": float(metrics.get("cd", float("nan"))),
            "d1_mse": float(metrics.get("p2point", float("nan"))),
            "d1_psnr": float(metrics.get("d1_psnr", float("nan"))),
            "d2_mse": float(metrics.get("p2plane", float("nan"))),
            "d2_psnr": float(metrics.get("d2_psnr", float("nan"))),
            "chamfer": float(metrics.get("cd", float("nan"))),
            "sampled_chamfer_proxy": float(metrics.get("cd", proxy.get("sampled_chamfer_proxy", float("nan")))),
            "hausdorff_proxy": float(metrics.get("hd", float("nan"))),
            "point_to_plane_proxy": float(metrics.get("p2plane", float("nan"))),
            "quality_eval_mode": "myNet_evaluate_decoded_geometry",
        })
        if bool(use_pc_error):
            pc_path = Path(pc_error_path) if str(pc_error_path) else (Path(__file__).resolve().parents[2] / "compress/octree/SparsePCGC/extension/pc_error_d")
            if pc_path.exists():
                peak = float(np.ptp(reference_xyz.detach().to("cpu").numpy(), axis=0).max())
                resolution = max(int(math.ceil(peak)), 1)
                cmd = [
                    str(pc_path),
                    "-a",
                    str(ref_path),
                    "-b",
                    str(dec_path),
                    "--hausdorff=1",
                    f"--resolution={resolution}",
                    "-n",
                    str(ref_path),
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
                text = (proc.stdout or "") + "\n" + (proc.stderr or "")

                def grab(label: str) -> float:
                    for line in text.splitlines():
                        if label in line:
                            vals = []
                            for token in line.replace(":", " ").replace(",", " ").split():
                                try:
                                    vals.append(float(token))
                                except ValueError:
                                    pass
                            if vals:
                                return float(vals[-1])
                    return float("nan")

                pc_d1_mse = grab("mseF      (p2point)")
                pc_d1_psnr = grab("mseF,PSNR (p2point)")
                pc_d2_mse = grab("mseF      (p2plane)")
                pc_d2_psnr = grab("mseF,PSNR (p2plane)")
                out.update({
                    "pc_error_d1_mse": pc_d1_mse,
                    "pc_error_d1_psnr": pc_d1_psnr,
                    "pc_error_d2_mse": pc_d2_mse,
                    "pc_error_d2_psnr": pc_d2_psnr,
                    "pc_error_resolution": resolution,
                    "pc_error_returncode": int(proc.returncode),
                })
                if math.isfinite(pc_d1_psnr):
                    out["d1_mse"] = pc_d1_mse
                    out["d1_psnr"] = pc_d1_psnr
                if math.isfinite(pc_d2_psnr):
                    out["d2_mse"] = pc_d2_mse
                    out["d2_psnr"] = pc_d2_psnr
                    out["point_to_plane_proxy"] = pc_d2_mse
                out["quality_eval_mode"] = "pc_error_d+myNet_evaluate_decoded_geometry"
    except Exception as exc:
        out["quality_eval_mode"] = f"sampled_proxy_fallback:{type(exc).__name__}"
    return out


def _pc_error_metrics_for_paths(
    reference_path: str | Path,
    decoded_path: str | Path,
    *,
    pc_error_path: str = "",
) -> Dict[str, object]:
    pc_path = Path(pc_error_path) if str(pc_error_path) else (Path(__file__).resolve().parents[2] / "compress/octree/SparsePCGC/extension/pc_error_d")
    out: Dict[str, object] = {"pc_error_d_success": False}
    if not pc_path.exists():
        out["pc_error_error"] = "pc_error_d_missing"
        return out
    try:
        ref_points = read_ply_xyz(reference_path)
        peak = float(np.ptp(ref_points, axis=0).max()) if ref_points.size else 1.0
        resolution = max(int(math.ceil(peak)), 1)
        cmd = [
            str(pc_path),
            f"--fileA={reference_path}",
            f"--fileB={decoded_path}",
            "--hausdorff=1",
            f"--resolution={resolution}",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")

        def grab(label: str) -> float:
            for line in text.splitlines():
                if label in line:
                    vals = []
                    for token in line.replace(":", " ").replace(",", " ").split():
                        try:
                            vals.append(float(token))
                        except ValueError:
                            pass
                    if vals:
                        return float(vals[-1])
            return float("nan")

        out.update({
            "pc_error_d_success": proc.returncode == 0,
            "pc_error_returncode": int(proc.returncode),
            "pc_error_resolution": int(resolution),
            "pc_error_d1_mse": grab("mseF      (p2point)"),
            "pc_error_d1_psnr": grab("mseF,PSNR (p2point)"),
            "pc_error_d2_mse": grab("mseF      (p2plane)"),
            "pc_error_d2_psnr": grab("mseF,PSNR (p2plane)"),
        })
    except Exception as exc:
        out["pc_error_error"] = f"{type(exc).__name__}:{exc}"
    return out


def _quality_from_paths(
    reference_path: str | Path,
    decoded_path: str | Path,
    *,
    formal_max_points: int,
    normal_max_points: int,
    pc_error_path: str = "",
    use_pc_error: bool = True,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    try:
        metrics = evaluate_decoded_geometry(
            reference_path,
            decoded_path,
            EvaluationConfig(
                max_points=int(formal_max_points),
                normal_max_points=int(normal_max_points),
                emd_points=min(512, max(int(formal_max_points), 1)),
                normal_k=16,
                psnr_peak=0.0,
            ),
        )
        out.update({
            "mynet_eval_success": True,
            "mynet_d1_mse": float(metrics.get("p2point", float("nan"))),
            "mynet_d1_psnr": float(metrics.get("d1_psnr", float("nan"))),
            "mynet_d2_mse": float(metrics.get("p2plane", float("nan"))),
            "mynet_d2_psnr": float(metrics.get("d2_psnr", float("nan"))),
            "mynet_chamfer": float(metrics.get("cd", float("nan"))),
            "mynet_hausdorff": float(metrics.get("hd", float("nan"))),
        })
    except Exception as exc:
        out.update({"mynet_eval_success": False, "mynet_eval_error": f"{type(exc).__name__}:{exc}"})
    if bool(use_pc_error):
        out.update(_pc_error_metrics_for_paths(reference_path, decoded_path, pc_error_path=pc_error_path))
    # Prefer pc_error for D1/D2 if available; Chamfer comes from myNet evaluator.
    out["d1_mse"] = out.get("pc_error_d1_mse", out.get("mynet_d1_mse", ""))
    out["d1_psnr"] = out.get("pc_error_d1_psnr", out.get("mynet_d1_psnr", ""))
    out["d2_mse"] = out.get("pc_error_d2_mse", out.get("mynet_d2_mse", ""))
    out["d2_psnr"] = out.get("pc_error_d2_psnr", out.get("mynet_d2_psnr", ""))
    out["chamfer"] = out.get("mynet_chamfer", "")
    out["quality_eval_mode"] = "pc_error_d+myNet_path_eval" if bool(use_pc_error) else "myNet_path_eval"
    return out


def _coord_match_ratio_from_paths(reference_path: str | Path, decoded_path: str | Path) -> Tuple[int, float, bool]:
    try:
        ref = torch.as_tensor(read_ply_xyz(reference_path), dtype=torch.float32).round().to(dtype=torch.long)
        dec = torch.as_tensor(read_ply_xyz(decoded_path), dtype=torch.float32).round().to(dtype=torch.long)
        if ref.numel() <= 0:
            return int(dec.shape[0]), 1.0 if dec.numel() <= 0 else 0.0, dec.numel() <= 0
        keys, occupied = _coord_key_setup(ref)
        dec_keys = torch.unique(keys(dec), sorted=True)
        match = _lookup_occupied(keys(ref), dec_keys)
        ratio = float(match.to(dtype=torch.float32).mean().item())
        return int(dec.shape[0]), ratio, bool(ratio >= 0.999999 and int(dec.shape[0]) == int(ref.shape[0]))
    except Exception:
        return 0, float("nan"), False


def _select_aggressive_high_bit_mask(
    coords: torch.Tensor,
    base_stats: Mapping[str, object],
    *,
    budget_ratio: float,
    mode: str,
    max_pool: int,
    block_size: int,
    repair_fraction: float = 0.0,
    quality_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    target = min(max(int(math.ceil(float(coords.shape[0]) * float(budget_ratio))), 0), max(int(coords.shape[0]) - 1, 0))
    score, bit_score, depth_score = _high_bit_voxel_scores(coords, base_stats, max_pool=int(max_pool))
    valid = torch.isfinite(score) & (bit_score > 0)
    if target <= 0 or not bool(valid.any().item()):
        return torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool), {
            "candidate_family": mode,
            "selected_bit_sum": 0.0,
            "selected_bit_mean": 0.0,
            "selected_bit_max": 0.0,
            "selected_depth_hist_json": "{}",
            "selected_spatial_cell_count": 0,
            "phase2g_candidate_pool_count": int(valid.sum().item()),
        }

    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    unique_parent, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    block = torch.div(coords, int(block_size), rounding_mode="floor")
    unique_block, inverse_block = torch.unique(block, dim=0, sorted=True, return_inverse=True)
    block_counts = torch.bincount(inverse_block, minlength=int(unique_block.shape[0])).to(device=coords.device)
    cell = torch.div(coords, 16, rounding_mode="floor")
    unique_cell, inverse_cell = torch.unique(cell, dim=0, sorted=True, return_inverse=True)
    cell_counts = torch.bincount(inverse_cell, minlength=int(unique_cell.shape[0])).to(device=coords.device)

    adjusted = score.clone()
    parent_pop_per_voxel = parent_pop.index_select(0, inverse_parent).to(dtype=torch.float32)
    # Dense/flat-ish points are safer decimation targets; isolated/boundary
    # points get protected. Parent pop protects thin octree contexts.
    safe_bonus = 0.15 * neigh + 0.05 * parent_pop_per_voxel
    risk = 1.2 * (neigh <= 1).to(dtype=torch.float32) + 0.4 * (parent_pop_per_voxel <= 2).to(dtype=torch.float32)
    veto_mask = (neigh <= 1) | (parent_pop_per_voxel <= 1)
    vetoed_count = int((valid & veto_mask).sum().item())
    fill_from_lower_bit_count = 0
    veto_reason_counts = {
        "low_density": int((valid & (neigh <= 1)).sum().item()),
        "thin_parent": int((valid & (parent_pop_per_voxel <= 1)).sum().item()),
    }
    mode = {
        "high_bit_raw_prune": "high_actual_bit_node_prune",
        "high_bit_surface_aware_prune": "high_actual_bit_surface_safe_prune",
        "high_bit_context_blue_noise_prune": "high_actual_bit_blue_noise_prune",
        "high_bit_rd_score_prune": "high_bit_rd_score_prune",
        "high_bit_rate_first_quality_veto": "high_bit_quality_veto",
        "high_bit_parent_block_cap_prune": "high_bit_parent_block_cap",
        "high_bit_depth_balanced_prune": "high_bit_depth_balanced",
        "high_bit_prune_then_light_repair": "high_bit_quality_veto",
        "high_bit_prune_then_surface_repair": "high_actual_bit_surface_safe_prune",
    }.get(mode, mode)
    if mode == "high_actual_bit_node_prune":
        adjusted = score
    elif mode == "high_actual_bit_surface_safe_prune":
        adjusted = score + safe_bonus - risk
    elif mode == "high_actual_bit_blue_noise_prune":
        adjusted = score + 0.10 * neigh - risk
    elif mode == "high_actual_bit_decimation_prune":
        valid = valid & (neigh >= 4) & (parent_pop_per_voxel >= 3)
        adjusted = score + 0.25 * neigh - 0.2 * risk
    elif mode == "bit_budgeted_block_soft_prune":
        blocks = _parse_json_payload(base_stats.get("sparsepcgc_bits_by_block_topk_json", ""), [])
        allowed = set()
        for item in blocks[: max(8, min(64, int(len(blocks))))]:
            text = str(item.get("block", ""))
            if ":" not in text:
                continue
            depth_text, rest = text.split(":", 1)
            if depth_text != "d0":
                continue
            try:
                allowed.add(tuple(int(v) for v in rest.split(",")))
            except Exception:
                pass
        if allowed:
            allowed_mask = torch.tensor(
                [tuple(int(v) for v in b.tolist()) in allowed for b in block.detach().to("cpu")],
                device=coords.device,
                dtype=torch.bool,
            )
            valid = valid & allowed_mask
        adjusted = score + safe_bonus - 0.5 * risk
    elif mode == "high_bit_rd_score_prune":
        finite_bits = bit_score[valid]
        bit_norm = (bit_score - finite_bits.mean()) / finite_bits.std().clamp_min(1e-6)
        quality_risk = (
            0.45 * (1.0 / (neigh + 1.0))
            + 0.25 * (neigh <= 2).to(dtype=torch.float32)
            + 0.20 * (parent_pop_per_voxel <= 2).to(dtype=torch.float32)
            + 0.10 * (depth_score <= 0).to(dtype=torch.float32)
        )
        adjusted = bit_norm - float(quality_weight) * quality_risk
    elif mode == "high_bit_quality_veto":
        adjusted = score
    elif mode == "high_bit_parent_block_cap":
        adjusted = score + 0.03 * safe_bonus
    elif mode == "high_bit_depth_balanced":
        adjusted = score

    adjusted = torch.where(valid, adjusted, adjusted.new_full(adjusted.shape, -1e12))
    selected = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    if mode == "high_bit_quality_veto":
        safe_valid = valid & (~veto_mask)
        safe_adjusted = torch.where(safe_valid, adjusted, adjusted.new_full(adjusted.shape, -1e12))
        order = torch.argsort(safe_adjusted, descending=True)
        picked = order[:target]
        picked = picked[safe_valid.index_select(0, picked)]
        selected[picked] = True
        if int(selected.sum().item()) < target:
            fill_needed = target - int(selected.sum().item())
            fill_adjusted = torch.where(valid & (~selected), adjusted, adjusted.new_full(adjusted.shape, -1e12))
            fill_order = torch.argsort(fill_adjusted, descending=True)
            fill = fill_order[:fill_needed]
            fill = fill[(valid & (~selected)).index_select(0, fill)]
            selected[fill] = True
            fill_from_lower_bit_count = int(fill.numel())
    elif mode == "high_bit_parent_block_cap":
        parent_pick = _quota_select(
            adjusted,
            inverse_parent,
            target,
            group_counts=parent_pop.to(device=coords.device),
            quota_fraction=max(float(budget_ratio) * 1.5, 0.02),
            round_robin=False,
        )
        restricted = torch.where(parent_pick, adjusted, adjusted.new_full(adjusted.shape, -1e12))
        selected = _quota_select(
            restricted,
            inverse_block,
            target,
            group_counts=block_counts,
            quota_fraction=max(float(budget_ratio) * 2.0, 0.02),
            round_robin=False,
        )
    elif mode == "high_bit_depth_balanced":
        # Rate-first but avoid all selected voxels coming from one depth when
        # multiple high-bit depths are available. Use a 70/30 depth0/depth1
        # target and fill from the remaining high-bit pool.
        for depth, frac in ((0, 0.70), (1, 0.30)):
            need = target - int(selected.sum().item()) if depth == 1 else int(math.ceil(target * frac))
            if need <= 0:
                continue
            dvalid = valid & (depth_score == int(depth)) & (~selected)
            dadj = torch.where(dvalid, adjusted, adjusted.new_full(adjusted.shape, -1e12))
            order = torch.argsort(dadj, descending=True)
            pick = order[:need]
            pick = pick[dvalid.index_select(0, pick)]
            selected[pick] = True
        if int(selected.sum().item()) < target:
            need = target - int(selected.sum().item())
            fvalid = valid & (~selected)
            fadj = torch.where(fvalid, adjusted, adjusted.new_full(adjusted.shape, -1e12))
            order = torch.argsort(fadj, descending=True)
            pick = order[:need]
            pick = pick[fvalid.index_select(0, pick)]
            selected[pick] = True
            fill_from_lower_bit_count = int(pick.numel())
    elif mode in {"high_actual_bit_blue_noise_prune", "bit_budgeted_block_soft_prune"}:
        # Two-level quota: first per block, then per 16^3 spatial cell. This is
        # intentionally not a global top-k; it tests the user's distributed
        # prune hypothesis.
        block_pick = _quota_select(
            adjusted,
            inverse_block,
            target,
            group_counts=block_counts,
            quota_fraction=max(float(budget_ratio) * 1.8, 0.01),
            round_robin=True,
        )
        restricted = torch.where(block_pick, adjusted, adjusted.new_full(adjusted.shape, -1e12))
        selected = _quota_select(
            restricted,
            inverse_cell,
            target,
            group_counts=cell_counts,
            quota_fraction=max(float(budget_ratio) * 2.5, 0.01),
            round_robin=True,
        )
    else:
        order = torch.argsort(adjusted, descending=True)
        picked = order[:target]
        picked = picked[valid.index_select(0, picked)]
        selected[picked] = True

    if repair_fraction > 0.0 and int(selected.sum().item()) > 0:
        repair_count = min(int(math.ceil(int(selected.sum().item()) * float(repair_fraction))), int(selected.sum().item()))
        # "Repair" proxy: keep back the riskiest removed samples. This measures
        # how much quality can be recovered by sacrificing rate.
        selected_idx = selected.nonzero(as_tuple=False).reshape(-1)
        repair_score = risk.index_select(0, selected_idx) - 0.02 * bit_score.index_select(0, selected_idx)
        restore_idx = selected_idx.index_select(0, torch.argsort(repair_score, descending=True)[:repair_count])
        selected[restore_idx] = False

    selected_bits = bit_score[selected]
    selected_depth = depth_score[selected]
    selected_neigh = neigh[selected]
    selected_parent = torch.unique(inverse_parent[selected]) if bool(selected.any().item()) else torch.empty((0,), device=coords.device, dtype=torch.long)
    selected_block = torch.unique(inverse_block[selected]) if bool(selected.any().item()) else torch.empty((0,), device=coords.device, dtype=torch.long)
    selected_cell = torch.unique(inverse_cell[selected]) if bool(selected.any().item()) else torch.empty((0,), device=coords.device, dtype=torch.long)
    drop_count = int(selected.sum().item())
    density_risk = 1.0 / (selected_neigh + 1.0) if drop_count > 0 else torch.empty((0,), device=coords.device)
    debug = {
        "candidate_family": mode,
        "selected_bit_sum": float(selected_bits.sum().item()) if drop_count > 0 else 0.0,
        "selected_bit_mean": float(selected_bits.mean().item()) if drop_count > 0 else 0.0,
        "selected_bit_max": float(selected_bits.max().item()) if drop_count > 0 else 0.0,
        "selected_depth_hist_json": _hist_json_from_tensor(selected_depth[selected_depth >= 0]),
        "selected_parent_count": int(selected_parent.numel()),
        "selected_block_count": int(selected_block.numel()),
        "selected_spatial_cell_count": int(selected_cell.numel()),
        "phase2g_candidate_pool_count": int(valid.sum().item()),
        "per_parent_prune_cap": "",
        "per_block_prune_cap": max(float(budget_ratio) * 1.8, 0.01) if mode in {"high_actual_bit_blue_noise_prune", "bit_budgeted_block_soft_prune"} else "",
        "local_cell_prune_ratio_cap": max(float(budget_ratio) * 2.5, 0.01) if mode in {"high_actual_bit_blue_noise_prune", "bit_budgeted_block_soft_prune"} else "",
        "distributed_prune_score": float(selected_cell.numel()) / max(float(drop_count), 1.0),
        "blue_noise_min_distance": "",
        "density_drop_mean": float(density_risk.mean().item()) if drop_count > 0 else 0.0,
        "density_drop_max": float(density_risk.max().item()) if drop_count > 0 else 0.0,
        "hole_risk": float(density_risk.mean().item()) if drop_count > 0 else 0.0,
        "curvature_removed_mean": float((1.0 / (selected_neigh + 1.0)).mean().item()) if drop_count > 0 else 0.0,
        "boundary_removed_ratio": float((selected_neigh <= 2).to(dtype=torch.float32).mean().item()) if drop_count > 0 else 0.0,
        "local_over_prune_score": float(drop_count) / max(float(selected_cell.numel()), 1.0),
        "repair_add_count": int(max(0, target - drop_count)) if repair_fraction > 0.0 else 0,
        "repair_add_ratio": float(max(0, target - drop_count)) / max(float(coords.shape[0]), 1.0) if repair_fraction > 0.0 else 0.0,
        "quality_weight": float(quality_weight),
        "rate_weight": 1.0,
        "budget_reached": bool(drop_count >= max(int(target * 0.98), 1)),
        "candidate_pool_size": int(max_pool),
        "actual_prune_ratio": float(drop_count) / max(float(coords.shape[0]), 1.0),
        "same_parent_prune_max": int(torch.bincount(inverse_parent[selected], minlength=int(unique_parent.shape[0])).max().item()) if drop_count > 0 else 0,
        "same_block_prune_ratio_max": float((torch.bincount(inverse_block[selected], minlength=int(unique_block.shape[0])).to(dtype=torch.float32) / block_counts.to(dtype=torch.float32).clamp_min(1.0)).max().item()) if drop_count > 0 else 0.0,
        "vetoed_count": int(vetoed_count),
        "veto_reason_counts_json": json.dumps(veto_reason_counts, sort_keys=True),
        "fill_from_lower_bit_count": int(fill_from_lower_bit_count),
    }
    return selected, debug


def _evaluate_mask_for_phase2g(
    *,
    encoder,
    args,
    coords: torch.Tensor,
    drop_mask: torch.Tensor,
    meta,
    base_bits: float,
    base_stats: Mapping[str, object],
    method_debug: Mapping[str, object],
    block_size: int,
    max_geometry_samples: int,
) -> Dict[str, object]:
    cand_coords = torch.unique(coords[~drop_mask].to(dtype=torch.long), dim=0, sorted=True)
    result = _evaluate_coords_for_phase2f(
        encoder=encoder,
        args=args,
        coords=coords,
        cand_coords=cand_coords,
        meta=meta,
        base_bits=base_bits,
        base_stats=base_stats,
        method_debug=method_debug,
        block_size=int(block_size),
        max_geometry_samples=int(max_geometry_samples),
    )
    keep_mask = ~drop_mask
    result.update(_aggressive_quality_proxy(coords, keep_mask, drop_mask, max_samples=int(max_geometry_samples)))
    result["prune_ratio"] = float(drop_mask.sum().item()) / max(float(coords.shape[0]), 1.0)
    result["add_ratio"] = float(result.get("add_count", 0) or 0) / max(float(coords.shape[0]), 1.0)
    return result


def _evaluate_mask_for_phase2h(
    *,
    encoder,
    args,
    coords: torch.Tensor,
    drop_mask: torch.Tensor,
    meta,
    base_bits: float,
    base_stats: Mapping[str, object],
    method_debug: Mapping[str, object],
    block_size: int,
    max_geometry_samples: int,
    formal_max_points: int,
    normal_max_points: int,
    pc_error_path: str = "",
    use_pc_error: bool = False,
) -> Dict[str, object]:
    cand_coords = torch.unique(coords[~drop_mask].to(dtype=torch.long), dim=0, sorted=True)
    result = _evaluate_coords_for_phase2f(
        encoder=encoder,
        args=args,
        coords=coords,
        cand_coords=cand_coords,
        meta=meta,
        base_bits=base_bits,
        base_stats=base_stats,
        method_debug=method_debug,
        block_size=int(block_size),
        max_geometry_samples=int(max_geometry_samples),
    )
    keep_mask = ~drop_mask
    ref_xyz = _coords_to_xyz(coords, meta, args)
    dec_xyz = _coords_to_xyz(cand_coords, meta, args)
    result.update(_formal_or_sampled_quality(
        reference_xyz=ref_xyz,
        decoded_xyz=dec_xyz,
        coords=coords,
        keep_mask=keep_mask,
        drop_mask=drop_mask,
        max_samples=int(max_geometry_samples),
        formal_max_points=int(formal_max_points),
        normal_max_points=int(normal_max_points),
        pc_error_path=str(pc_error_path),
        use_pc_error=bool(use_pc_error),
    ))
    result["prune_ratio"] = float(drop_mask.sum().item()) / max(float(coords.shape[0]), 1.0)
    result["add_ratio"] = float(result.get("add_count", 0) or 0) / max(float(coords.shape[0]), 1.0)
    return result


def _evaluate_coords_for_phase2f(
    *,
    encoder,
    args,
    coords: torch.Tensor,
    cand_coords: torch.Tensor,
    meta,
    base_bits: float,
    base_stats: Mapping[str, object],
    method_debug: Mapping[str, object],
    block_size: int,
    max_geometry_samples: int,
) -> Dict[str, object]:
    keys, _occ = _coord_key_setup(coords)
    cand_keys = torch.unique(keys(cand_coords), sorted=True)
    keep_mask = _lookup_occupied(keys(coords), cand_keys)
    drop_mask = ~keep_mask
    cand_xyz = _coords_to_xyz(cand_coords, meta, args)
    stats = encoder.encode_bits(cand_xyz)
    raw_bit = float(stats.get("bit", 0.0))
    metrics = dict(context_metrics(coords, drop_mask, block_size=int(block_size)))
    geom = geometry_proxy(coords, keep_mask, drop_mask, max_samples=int(max_geometry_samples))
    est_delta = (
        _safe_float(stats.get("sparsepcgc_estimated_occupancy_bits"), float("nan"))
        - _safe_float(base_stats.get("sparsepcgc_estimated_occupancy_bits"), float("nan"))
    )
    out = {
        "actual_raw_percent": 100.0 * (raw_bit - base_bits) / max(base_bits, 1.0),
        "raw_bit": raw_bit,
        "base_bit": float(base_bits),
        "estimated_bits_total": stats.get("sparsepcgc_estimated_occupancy_bits", ""),
        "estimated_bits_delta": est_delta,
        "geometry_proxy": float(geom),
        "selected_voxel_count": int(drop_mask.sum().item()),
        "prune_count": int(drop_mask.sum().item()),
        "add_count": int(method_debug.get("add_count", 0) or 0),
        "sampled_chamfer_proxy": float(geom),
    }
    out.update(metrics)
    for key in SPARSEPCGC_RATE_DEBUG_KEYS:
        if key in stats:
            out[key] = stats.get(key, "")
    out.update(method_debug)
    return out


def _drop_mask_against_original(original: torch.Tensor, final_coords: torch.Tensor) -> torch.Tensor:
    keys, _occupied = _coord_key_setup(original)
    final_keys = torch.unique(keys(final_coords), sorted=True)
    kept = _lookup_occupied(keys(original), final_keys)
    return ~kept


def _merge_debug(first: Mapping[str, object], second: Mapping[str, object], *, name: str) -> Dict[str, object]:
    out = dict(second)
    out["operation_type"] = "composite"
    out["composite_operation_type"] = name
    out["component_operations"] = json.dumps([first.get("canonical_method", ""), second.get("canonical_method", "")])
    out["component_order"] = name
    out["add_count"] = int(_safe_float(first.get("add_count"), 0.0)) + int(_safe_float(second.get("add_count"), 0.0))
    out["move_count"] = int(_safe_float(first.get("move_count"), 0.0)) + int(_safe_float(second.get("move_count"), 0.0))
    out["merge_count"] = int(_safe_float(first.get("merge_count"), 0.0)) + int(_safe_float(second.get("merge_count"), 0.0))
    out["delta_pattern_nll"] = _safe_float(first.get("delta_pattern_nll"), 0.0) + _safe_float(second.get("delta_pattern_nll"), 0.0)
    out["occupied_nll_removed"] = _safe_float(first.get("occupied_nll_removed"), 0.0) + _safe_float(second.get("occupied_nll_removed"), 0.0)
    out["score_proxy"] = _safe_float(first.get("score_proxy"), 0.0) + _safe_float(second.get("score_proxy"), 0.0)
    out["added_point_nn_distance_mean"] = max(
        _safe_float(first.get("added_point_nn_distance_mean"), 0.0),
        _safe_float(second.get("added_point_nn_distance_mean"), 0.0),
    )
    out["added_point_nn_distance_max"] = max(
        _safe_float(first.get("added_point_nn_distance_max"), 0.0),
        _safe_float(second.get("added_point_nn_distance_max"), 0.0),
    )
    out["add_density_delta"] = max(
        _safe_float(first.get("add_density_delta"), 0.0),
        _safe_float(second.get("add_density_delta"), 0.0),
    )
    return out


def _coords_signature(coords: torch.Tensor) -> str:
    cpu = coords.detach().to("cpu", dtype=torch.long).contiguous()
    digest = hashlib.blake2b(cpu.numpy().tobytes(), digest_size=16)
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    return digest.hexdigest()


def _make_row(
    *,
    file_path: str,
    sequence: str,
    frame_id: str,
    beam_group_id: str,
    objective_mode: str,
    guard_mode: str,
    beam_iter: int,
    parent_rank: int,
    parent_state: BeamState,
    state_id: str,
    method: str,
    operation_type: str,
    cand_coords: torch.Tensor,
    drop_mask: torch.Tensor,
    method_debug: Mapping[str, object],
    metrics: Mapping[str, object],
    base_bits: float,
    raw_bit: float,
    actual_raw_percent: float,
    objective_j: float,
    edit_sequence: str,
    total_add_count: int,
    total_prune_count: int,
    total_move_count: int,
    total_merge_count: int,
    cumulative_add_ratio: float,
    cumulative_prune_ratio: float,
    sampled_chamfer_proxy: float,
    phase1_rule_raw: float,
    phase1_oracle_raw: float,
    block_only_raw: float,
    candidate_allowed: bool,
    candidate_guard_reason: str,
    candidate_rule_score: float,
    density_penalty_value: float,
    geometry_penalty_value: float,
) -> Dict[str, object]:
    return {
        "file": file_path,
        "sequence": sequence,
        "frame_id": frame_id,
        "beam_group_id": beam_group_id,
        "objective_mode": str(objective_mode),
        "high_nll_guard_mode": guard_mode,
        "beam_iter": int(beam_iter),
        "beam_rank": "",
        "parent_state_id": parent_state.state_id,
        "parent_beam_rank": int(parent_rank),
        "state_id": state_id,
        "operation_type": operation_type,
        "operation_family": _operation_family(method, operation_type),
        "operation_name": method,
        "candidate_variant": method,
        "composite_operation_type": method_debug.get("composite_operation_type", ""),
        "component_operations": method_debug.get("component_operations", ""),
        "component_order": method_debug.get("component_order", ""),
        "high_nll_guard_type": guard_mode if method.startswith("high_nll") else "",
        "soft_guard_score": float(candidate_rule_score) if method.startswith("high_nll") and math.isfinite(float(candidate_rule_score)) else "",
        "actual_raw_percent": float(actual_raw_percent),
        "delta_from_parent_raw_percent": float(actual_raw_percent) - float(parent_state.actual_raw_percent),
        "base_bit": float(base_bits),
        "raw_bit": float(raw_bit),
        "objective_J": float(objective_j),
        "density_penalty_value": float(density_penalty_value),
        "geometry_penalty_value": float(geometry_penalty_value),
        "selected_in_beam": False,
        "edit_sequence": edit_sequence,
        "total_add_count": int(total_add_count),
        "total_prune_count": int(total_prune_count),
        "total_move_count": int(total_move_count),
        "total_merge_count": int(total_merge_count),
        "cumulative_add_ratio": float(cumulative_add_ratio),
        "cumulative_prune_ratio": float(cumulative_prune_ratio),
        "added_point_nn_distance_mean": method_debug.get("added_point_nn_distance_mean", method_debug.get("add_geometry_proxy", 0.0)),
        "added_point_nn_distance_max": method_debug.get("added_point_nn_distance_max", method_debug.get("add_geometry_proxy", 0.0)),
        "add_density_delta": method_debug.get("add_density_delta", 0.0),
        "sampled_chamfer_proxy": float(sampled_chamfer_proxy),
        "pattern_projection_gain": method_debug.get("delta_pattern_nll", 0.0),
        "parent_pattern_nll_before": method_debug.get("parent_pattern_nll_before", ""),
        "parent_pattern_nll_after": method_debug.get("parent_pattern_nll_after", ""),
        "add_budget": method_debug.get("add_count", 0) if method.startswith(("addv2_", "pattern_projection_add")) else "",
        "partial_context_damage_ratio": metrics.get("partial_context_damage_ratio", 0.0),
        "parent_emptying_ratio": metrics.get("parent_emptying_ratio", 0.0),
        "delta_pattern_nll": method_debug.get("delta_pattern_nll", 0.0),
        "occupied_nll_removed": method_debug.get("occupied_nll_removed", 0.0),
        "move_distance_mean": method_debug.get("move_distance_mean", 0.0),
        "geometry_missing_nn_mean": method_debug.get("geometry_missing_nn_mean", 0.0),
        "block_only_baseline_raw_percent": float(block_only_raw),
        "phase1_rule_raw_percent": float(phase1_rule_raw),
        "phase1_oracle_raw_percent": float(phase1_oracle_raw),
        "candidate_allowed": bool(candidate_allowed),
        "candidate_guard_reason": str(candidate_guard_reason),
        "candidate_rule_score": float(candidate_rule_score) if math.isfinite(float(candidate_rule_score)) else "",
        "candidate_voxels": int(cand_coords.shape[0]),
        "drop_count": metrics.get("drop_count", 0),
        "actual_drop_ratio": metrics.get("actual_drop_ratio", 0.0),
        "selected_block_count": metrics.get("selected_block_count", 0),
        "drop_concentration_top1": metrics.get("drop_concentration_top1", 0.0),
        "drop_concentration_top5": metrics.get("drop_concentration_top5", 0.0),
        "add_count": method_debug.get("add_count", 0),
        "macro_count": method_debug.get("macro_drop_count", 0),
        "prune_count": int(metrics.get("drop_count", 0) or 0) if operation_type in {"prune", "small_macro", "composite"} else 0,
        "affected_parent_count": metrics.get("parent_emptying_count", 0),
        "same_parent_add_prune_ratio": "",
        "parent_group_id": "",
        "group_drop_count": metrics.get("drop_count", 0),
        "ancestor_reduction_count": method_debug.get("ancestor_reduction_count", ""),
        "local_pattern_nll_before": method_debug.get("parent_pattern_nll_before", ""),
        "local_pattern_nll_after_first_op": "",
        "local_pattern_nll_after_second_op": method_debug.get("parent_pattern_nll_after", ""),
        "delta_pattern_nll_total": method_debug.get("delta_pattern_nll", 0.0),
        "decomposed_block_id": "",
        "group_id": "",
        "group_raw_percent": "",
        "block_only_raw_percent": float(block_only_raw),
        "group_to_block_gain_ratio": (
            abs(float(actual_raw_percent)) / abs(float(block_only_raw))
            if abs(float(block_only_raw)) > 1e-12
            else ""
        ),
        "group_geometry_proxy": float(sampled_chamfer_proxy),
        "group_to_block_drop_ratio": "",
        "move_count": method_debug.get("move_count", 0),
        "merge_count": method_debug.get("merge_count", 0),
        "add_ratio": method_debug.get("add_ratio", 0.0),
        "sparsepcgc_estimated_occupancy_bits": method_debug.get("sparsepcgc_estimated_occupancy_bits", ""),
        "sparsepcgc_pred_occupancy_nll": method_debug.get("sparsepcgc_pred_occupancy_nll", ""),
        "sparsepcgc_prob_true_mean": method_debug.get("sparsepcgc_prob_true_mean", ""),
        "sparsepcgc_prob_true_low_ratio": method_debug.get("sparsepcgc_prob_true_low_ratio", ""),
        "sparsepcgc_occupied_low_prob_ratio": method_debug.get("sparsepcgc_occupied_low_prob_ratio", ""),
        "sparsepcgc_bits_by_depth_json": method_debug.get("sparsepcgc_bits_by_depth_json", ""),
        "sparsepcgc_candidates_by_depth_json": method_debug.get("sparsepcgc_candidates_by_depth_json", ""),
        "sparsepcgc_occupied_by_depth_json": method_debug.get("sparsepcgc_occupied_by_depth_json", ""),
        "sparsepcgc_low_prob_occupied_by_depth_json": method_debug.get("sparsepcgc_low_prob_occupied_by_depth_json", ""),
        "sparsepcgc_high_bit_nodes_by_depth_json": method_debug.get("sparsepcgc_high_bit_nodes_by_depth_json", ""),
        "sparsepcgc_bits_by_parent_popcount_json": method_debug.get("sparsepcgc_bits_by_parent_popcount_json", ""),
        "sparsepcgc_bits_by_child_pattern_topk_json": method_debug.get("sparsepcgc_bits_by_child_pattern_topk_json", ""),
        "sparsepcgc_bits_by_block_topk_json": method_debug.get("sparsepcgc_bits_by_block_topk_json", ""),
        "sparsepcgc_top_high_bit_nodes_json": method_debug.get("sparsepcgc_top_high_bit_nodes_json", ""),
        "best_beam_final_raw_percent": "",
        "best_beam_edit_sequence": "",
        "best_beam_total_add_count": "",
        "best_beam_total_prune_count": "",
        "best_beam_geometry_proxy": "",
        "best_beam_final_objective_J": "",
        "improvement_over_phase1_rule": "",
        "improvement_over_phase1_oracle": "",
        "improvement_over_phase2B_raw_only": "",
        "ratio_to_block_only_improvement": "",
        "selected_operation_counts": "",
    }


def _evaluate_candidate(
    *,
    method: str,
    coords: torch.Tensor,
    meta,
    args,
    encoder,
    amount: float,
    block_size: int,
    seed: int,
    max_operation_edits: int,
    bit_cache: Dict[str, float] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Mapping[str, object], float]:
    composite_steps = {
        "add_then_high_nll_local": (
            "addv2_n256_g0p3_nn1_densmedium",
            "high_nll_branch_prune",
        ),
        "high_nll_then_add_repair": (
            "high_nll_branch_prune",
            "addv2_n256_g0p3_nn1_densmedium",
        ),
        "small_macro_then_add_repair": (
            "smacro_cap0p0025_maxb1_top10_geommedium",
            "addv2_n256_g0p3_nn1_densmedium",
        ),
    }
    if method in composite_steps:
        first_method, second_method = composite_steps[method]
        first_coords, _first_mask, first_debug = build_candidate_coords(
            first_method,
            coords,
            float(amount),
            block_size=int(block_size),
            seed=int(seed),
            max_operation_edits=max(128, int(max_operation_edits) // 2),
        )
        first_coords = torch.unique(first_coords.to(dtype=torch.long), dim=0, sorted=True)
        second_coords, _second_mask, second_debug = build_candidate_coords(
            second_method,
            first_coords,
            float(amount),
            block_size=int(block_size),
            seed=int(seed) + 17,
            max_operation_edits=max(128, int(max_operation_edits) // 2),
        )
        cand_coords = torch.unique(second_coords.to(dtype=torch.long), dim=0, sorted=True)
        drop_mask = _drop_mask_against_original(coords, cand_coords)
        debug = _merge_debug(
            {**dict(first_debug), "canonical_method": first_method},
            {**dict(second_debug), "canonical_method": second_method},
            name=method,
        )
        debug.setdefault("candidate_variant", method)
        debug.setdefault("canonical_method", method)
        cache_key = _coords_signature(cand_coords)
        if bit_cache is not None and cache_key in bit_cache:
            bit = float(bit_cache[cache_key])
        else:
            cand_xyz = _coords_to_xyz(cand_coords, meta, args)
            stats = encoder.encode_bits(cand_xyz)
            bit = float(stats.get("bit", 0.0))
            for key in SPARSEPCGC_RATE_DEBUG_KEYS:
                if key in stats:
                    debug[key] = stats.get(key, "")
            if bit_cache is not None:
                bit_cache[cache_key] = bit
        return cand_coords, drop_mask, debug, bit

    build_method = _canonical_method(method)
    cand_coords, drop_mask, debug = build_candidate_coords(
        build_method,
        coords,
        float(amount),
        block_size=int(block_size),
        seed=int(seed),
        max_operation_edits=int(max_operation_edits),
    )
    cand_coords = torch.unique(cand_coords.to(dtype=torch.long), dim=0, sorted=True)
    debug = dict(debug)
    debug.setdefault("candidate_variant", method)
    debug.setdefault("canonical_method", build_method)
    cache_key = _coords_signature(cand_coords)
    if bit_cache is not None and cache_key in bit_cache:
        bit = float(bit_cache[cache_key])
    else:
        cand_xyz = _coords_to_xyz(cand_coords, meta, args)
        stats = encoder.encode_bits(cand_xyz)
        bit = float(stats.get("bit", 0.0))
        for key in SPARSEPCGC_RATE_DEBUG_KEYS:
            if key in stats:
                debug[key] = stats.get(key, "")
        if bit_cache is not None:
            bit_cache[cache_key] = bit
    return cand_coords, drop_mask, dict(debug), bit


def _phase1_baselines(
    *,
    coords: torch.Tensor,
    meta,
    args,
    encoder,
    amount: float,
    block_size: int,
    seed: int,
    max_operation_edits: int,
    base_bits: float,
    bit_cache: Dict[str, float] | None = None,
) -> Tuple[float, float, float]:
    rows: List[Dict[str, object]] = []
    for method in PHASE1_METHODS:
        if method == "noop":
            bit = float(base_bits)
            row: Dict[str, object] = {
                "method": method,
                "operation_type": "noop",
                "actual_raw_percent": 0.0,
                "score_proxy": 0.0,
            }
        else:
            cand_coords, drop_mask, debug, bit = _evaluate_candidate(
                method=method,
                coords=coords,
                meta=meta,
                args=args,
                encoder=encoder,
                amount=float(amount),
                block_size=int(block_size),
                seed=int(seed),
                max_operation_edits=int(max_operation_edits),
                bit_cache=bit_cache,
            )
            metrics = dict(context_metrics(coords, drop_mask, block_size=int(block_size)))
            actual_raw = 100.0 * (bit - base_bits) / max(base_bits, 1.0)
            row = {
                "method": method,
                "operation_type": debug.get("operation_type", "prune"),
                "actual_raw_percent": actual_raw,
                "score_proxy": debug.get("score_proxy", 0.0),
                "move_distance_mean": debug.get("move_distance_mean", 0.0),
                "add_count": debug.get("add_count", 0),
                "added_point_nn_distance_max": debug.get("added_point_nn_distance_max", debug.get("add_geometry_proxy", 0.0)),
                "add_geometry_proxy": debug.get("add_geometry_proxy", 0.0),
                "density_guard": debug.get("density_guard", ""),
            }
            row.update(metrics)
        eligible, score, reason = _phase1_rule_candidate_score(row)
        row["rule_candidate_eligible"] = eligible
        row["rule_candidate_score"] = score
        row["rule_candidate_reason"] = reason
        rows.append(row)
    block_raw = min(
        (_safe_float(r.get("actual_raw_percent"), float("nan")) for r in rows if r.get("method") == "block_only"),
        default=float("nan"),
    )
    nonblock = [r for r in rows if r.get("method") != "block_only"]
    oracle = min(nonblock, key=lambda r: (_safe_float(r.get("actual_raw_percent"), float("inf")), str(r.get("method"))))
    eligible_rows = [r for r in rows if bool(r.get("rule_candidate_eligible"))]
    selected = max(eligible_rows, key=lambda r: (_safe_float(r.get("rule_candidate_score"), float("-inf")), str(r.get("method"))))
    if selected.get("method") != "noop" and _safe_float(selected.get("actual_raw_percent"), 0.0) >= 0.0:
        selected = next((r for r in rows if r.get("method") == "noop"), selected)
    return (
        _safe_float(selected.get("actual_raw_percent"), 0.0),
        _safe_float(oracle.get("actual_raw_percent"), 0.0),
        float(block_raw),
    )


def _write_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _annotate_final_rows(rows: List[Dict[str, object]], group_id: str, final_state: BeamState) -> None:
    phase1_rule = _safe_float(rows[-1].get("phase1_rule_raw_percent"), 0.0) if rows else 0.0
    phase1_oracle = _safe_float(rows[-1].get("phase1_oracle_raw_percent"), 0.0) if rows else 0.0
    block_raw = _safe_float(rows[-1].get("block_only_baseline_raw_percent"), float("nan")) if rows else float("nan")
    ratio_to_block = (
        abs(float(final_state.actual_raw_percent)) / abs(block_raw)
        if math.isfinite(block_raw) and abs(block_raw) > 1e-12
        else float("nan")
    )
    for row in rows:
        if row.get("beam_group_id") != group_id:
            continue
        ops = [op for op in final_state.edit_sequence.split(">") if op and op != "start"]
        op_counts: Dict[str, int] = {}
        for op in ops:
            op_counts[op] = op_counts.get(op, 0) + 1
        row["best_beam_final_raw_percent"] = float(final_state.actual_raw_percent)
        row["best_beam_edit_sequence"] = final_state.edit_sequence
        row["best_beam_total_add_count"] = int(final_state.total_add_count)
        row["best_beam_total_prune_count"] = int(final_state.total_prune_count)
        row["best_beam_geometry_proxy"] = float(final_state.geometry_proxy_summary)
        row["best_beam_final_objective_J"] = float(final_state.objective_j)
        row["improvement_over_phase1_rule"] = float(final_state.actual_raw_percent) - phase1_rule
        row["improvement_over_phase1_oracle"] = float(final_state.actual_raw_percent) - phase1_oracle
        row["improvement_over_phase2B_raw_only"] = float(final_state.actual_raw_percent) - (-0.19206644223957692)
        row["ratio_to_block_only_improvement"] = ratio_to_block
        row["selected_operation_counts"] = json.dumps(op_counts, sort_keys=True)


def run_probe(cli) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2 RDO beam probe"),
            time.strftime("%Y%m%d"),
            time.strftime("%H%M%S"),
        )
    finally:
        sys.argv = old_argv
    args.compress = "SparsePCGC"
    args.compression_loss_backend = "sparsepcgc_surrogate"
    args.sparsepcgc_skip_decode = True
    args.sparsepcgc_worker_gpu_stats = False
    args.enable_sparsepcgc_occupancy_debug = True
    args.enable_sparsepcgc_exact_occupancy_teacher = False
    args.sparsepcgc_exact_occupancy_interval = 1

    methods = list(_parse_csv_text(cli.methods))
    diagnostic_methods = list(_parse_csv_text(cli.diagnostic_methods))
    guard_modes = list(_parse_csv_text(cli.high_nll_guard_modes))
    rows: List[Dict[str, object]] = []
    if bool(cli.append_output) and Path(cli.output_csv).exists():
        with open(cli.output_csv, newline="", encoding="utf-8") as f:
            rows.extend(dict(row) for row in csv.DictReader(f))
    encoder = build_actual_encoder(args)
    bit_cache: Dict[str, float] = {}
    try:
        for file_idx, file_path in enumerate(cli.files):
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            sequence = Path(file_path).parent.name
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, args)
            base_stats = encoder.encode_bits(base_xyz)
            base_bits = float(base_stats.get("bit", 0.0))
            amount = float(cli.amount)
            phase1_rule_raw, phase1_oracle_raw, block_only_raw = _phase1_baselines(
                coords=coords,
                meta=meta,
                args=args,
                encoder=encoder,
                amount=amount,
                block_size=int(cli.block_size),
                seed=int(cli.seed) + file_idx,
                max_operation_edits=int(cli.max_operation_edits),
                base_bits=base_bits,
                bit_cache=bit_cache,
            )
            for guard_mode in guard_modes:
                group_id = f"{sequence}:{frame_id}:amount={amount:.6f}:guard={guard_mode}:obj={cli.objective_mode}"
                init = BeamState(
                    state_id=f"{group_id}:s0",
                    coords=coords,
                    raw_bit=base_bits,
                    actual_raw_percent=0.0,
                    objective_j=0.0,
                    edit_sequence="start",
                )
                beam: List[BeamState] = [init]
                state_counter = 0
                group_row_start = len(rows)
                for beam_iter in range(1, int(cli.iterations) + 1):
                    candidates: List[Tuple[BeamState, Dict[str, object]]] = []
                    for parent_rank, state in enumerate(beam, start=1):
                        iter_methods = ["stop"] + methods + diagnostic_methods
                        for method in iter_methods:
                            state_counter += 1
                            state_id = f"{group_id}:i{beam_iter}:c{state_counter}"
                            if method == "stop":
                                cand_coords = state.coords
                                drop_mask = torch.zeros((state.coords.shape[0],), device=state.coords.device, dtype=torch.bool)
                                debug: Dict[str, object] = {"operation_type": "stop"}
                                metrics = dict(context_metrics(state.coords, drop_mask, block_size=int(cli.block_size)))
                                raw_bit = float(state.raw_bit)
                                actual_raw = float(state.actual_raw_percent)
                                sampled_chamfer = float(state.geometry_proxy_summary)
                                operation_type = "stop"
                                total_add = state.total_add_count
                                total_prune = state.total_prune_count
                                total_move = state.total_move_count
                                total_merge = state.total_merge_count
                                cum_add = state.cumulative_add_ratio
                                cum_prune = state.cumulative_prune_ratio
                                geom_summary = state.geometry_proxy_summary
                                density_summary = state.add_density_delta_mean
                            else:
                                cand_coords, drop_mask, debug, raw_bit = _evaluate_candidate(
                                    method=method,
                                    coords=state.coords,
                                    meta=meta,
                                    args=args,
                                    encoder=encoder,
                                    amount=amount,
                                    block_size=int(cli.block_size),
                                    seed=int(cli.seed) + file_idx + beam_iter * 100 + parent_rank,
                                    max_operation_edits=int(cli.max_operation_edits),
                                    bit_cache=bit_cache,
                                )
                                metrics = dict(context_metrics(state.coords, drop_mask, block_size=int(cli.block_size)))
                                actual_raw = 100.0 * (raw_bit - base_bits) / max(base_bits, 1.0)
                                operation_type = str(debug.get("operation_type", "prune"))
                                add_count = int(debug.get("add_count", 0) or 0)
                                prune_count = int(metrics.get("drop_count", 0) or 0) if operation_type in {"prune", "small_macro", "composite"} else 0
                                move_count = int(debug.get("move_count", 0) or 0)
                                merge_count = int(debug.get("merge_count", 0) or 0)
                                add_nn_mean = float(debug.get("added_point_nn_distance_mean", debug.get("add_geometry_proxy", 0.0)) or 0.0)
                                geom = geometry_proxy(
                                    state.coords,
                                    ~drop_mask,
                                    drop_mask,
                                    max_samples=int(cli.max_geometry_samples),
                                )
                                sampled_chamfer = (
                                    (float(geom) * float(metrics.get("drop_count", 0) or 0) + add_nn_mean * float(add_count))
                                    / max(float((metrics.get("drop_count", 0) or 0) + add_count), 1.0)
                                )
                                total_add = int(state.total_add_count) + add_count
                                total_prune = int(state.total_prune_count) + prune_count
                                total_move = int(state.total_move_count) + move_count
                                total_merge = int(state.total_merge_count) + merge_count
                                cum_add = float(state.cumulative_add_ratio) + float(add_count) / max(float(coords.shape[0]), 1.0)
                                cum_prune = float(state.cumulative_prune_ratio) + float(prune_count) / max(float(coords.shape[0]), 1.0)
                                touched = max(total_add + total_prune + total_move + total_merge, 1)
                                prev_touched = max(state.total_add_count + state.total_prune_count + state.total_move_count + state.total_merge_count, 0)
                                geom_summary = (
                                    (float(state.geometry_proxy_summary) * float(prev_touched) + sampled_chamfer * float(add_count + prune_count + move_count + merge_count))
                                    / float(touched)
                                )
                                density_summary = (
                                    (float(state.add_density_delta_mean) * float(max(state.total_add_count, 0)) + float(debug.get("add_density_delta", 0.0) or 0.0) * float(add_count))
                                    / float(max(total_add, 1))
                                )
                            objective = _objective_j(
                                actual_raw,
                                cumulative_add_ratio=cum_add,
                                geometry_proxy_value=geom_summary,
                                add_density_delta=density_summary,
                                lambda_add=float(cli.lambda_add),
                                lambda_geom=float(cli.lambda_geom),
                                lambda_density=float(cli.lambda_density),
                            )
                            density_penalty_value = float(cli.lambda_density) * float(density_summary)
                            geometry_penalty_value = float(cli.lambda_geom) * float(geom_summary)
                            temp_row = {
                                "method": method,
                                "operation_type": operation_type,
                                "partial_context_damage_ratio": metrics.get("partial_context_damage_ratio", 0.0),
                                "parent_emptying_ratio": metrics.get("parent_emptying_ratio", 0.0),
                                "move_distance_mean": debug.get("move_distance_mean", 0.0),
                                "add_count": debug.get("add_count", 0),
                                "add_ratio": float(debug.get("add_count", 0) or 0) / max(float(coords.shape[0]), 1.0),
                                "added_point_nn_distance_max": debug.get("added_point_nn_distance_max", debug.get("add_geometry_proxy", 0.0)),
                                "add_geometry_proxy": debug.get("add_geometry_proxy", 0.0),
                                "density_guard": debug.get("density_guard", ""),
                                "score_proxy": debug.get("score_proxy", 0.0),
                            }
                            allowed, rule_score, reason = _beam_candidate_allowed(
                                method,
                                temp_row,
                                state,
                                high_nll_guard_mode=guard_mode,
                                max_total_add_count=int(cli.max_total_add_count),
                                max_cumulative_add_ratio=float(cli.max_cumulative_add_ratio),
                            )
                            edit_sequence = state.edit_sequence if method == "stop" else f"{state.edit_sequence}>{method}"
                            row = _make_row(
                                file_path=str(file_path),
                                sequence=sequence,
                                frame_id=frame_id,
                                beam_group_id=group_id,
                                objective_mode=str(cli.objective_mode),
                                guard_mode=guard_mode,
                                beam_iter=beam_iter,
                                parent_rank=parent_rank,
                                parent_state=state,
                                state_id=state_id,
                                method=method,
                                operation_type=operation_type,
                                cand_coords=cand_coords,
                                drop_mask=drop_mask,
                                method_debug=debug,
                                metrics=metrics,
                                base_bits=base_bits,
                                raw_bit=raw_bit,
                                actual_raw_percent=actual_raw,
                                objective_j=objective,
                                edit_sequence=edit_sequence,
                                total_add_count=total_add,
                                total_prune_count=total_prune,
                                total_move_count=total_move,
                                total_merge_count=total_merge,
                                cumulative_add_ratio=cum_add,
                                cumulative_prune_ratio=cum_prune,
                                sampled_chamfer_proxy=geom_summary,
                                phase1_rule_raw=phase1_rule_raw,
                                phase1_oracle_raw=phase1_oracle_raw,
                                block_only_raw=block_only_raw,
                                candidate_allowed=allowed,
                                candidate_guard_reason=reason,
                                candidate_rule_score=rule_score,
                                density_penalty_value=density_penalty_value,
                                geometry_penalty_value=geometry_penalty_value,
                            )
                            rows.append(row)
                            if allowed:
                                candidates.append((
                                    BeamState(
                                        state_id=state_id,
                                        coords=cand_coords,
                                        raw_bit=raw_bit,
                                        actual_raw_percent=actual_raw,
                                        objective_j=objective,
                                        edit_sequence=edit_sequence,
                                        total_add_count=total_add,
                                        total_prune_count=total_prune,
                                        total_move_count=total_move,
                                        total_merge_count=total_merge,
                                        cumulative_add_ratio=cum_add,
                                        cumulative_prune_ratio=cum_prune,
                                        geometry_proxy_summary=geom_summary,
                                        add_density_delta_mean=density_summary,
                                    ),
                                    row,
                                ))
                    candidates.sort(key=lambda item: (item[0].objective_j, item[0].actual_raw_percent, item[0].edit_sequence))
                    next_beam: List[BeamState] = []
                    seen_signatures = set()
                    for rank, (state, row) in enumerate(candidates, start=1):
                        signature = (round(state.actual_raw_percent, 10), state.edit_sequence)
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)
                        if len(next_beam) >= int(cli.beam_width):
                            break
                        row["selected_in_beam"] = True
                        row["beam_rank"] = len(next_beam) + 1
                        next_beam.append(state)
                    if not next_beam:
                        break
                    beam = next_beam
                    _write_csv(cli.output_csv, rows)
                    print(json.dumps({
                        "group": group_id,
                        "iter": beam_iter,
                        "best_raw": beam[0].actual_raw_percent,
                        "best_seq": beam[0].edit_sequence,
                    }, sort_keys=True), flush=True)
                final_state = min(beam, key=lambda s: (s.objective_j, s.actual_raw_percent, s.edit_sequence))
                _annotate_final_rows(rows[group_row_start:], group_id, final_state)
                _write_csv(cli.output_csv, rows)
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
    return 0


def _load_phase2d_baseline() -> Dict[Tuple[str, str], float]:
    path = Path("/data/maejima/log/PHASE2D_rdo_beam_probe.csv")
    out: Dict[Tuple[str, str], float] = {}
    if not path.exists():
        return out
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                value = row.get("best_beam_final_raw_percent", "")
                if value in ("", None):
                    continue
                out[(str(row.get("sequence", "")), str(row.get("frame_id", "")))] = float(value)
    except Exception:
        return {}
    return out


def _phase2f_row(
    *,
    file_path: str,
    sequence: str,
    frame_id: str,
    candidate_name: str,
    candidate_family: str,
    budget: float,
    result: Mapping[str, object],
    base_stats: Mapping[str, object],
    phase2d_best: float,
    block_1p_raw: float,
    budget_block_raw: float,
) -> Dict[str, object]:
    raw = _safe_float(result.get("actual_raw_percent"), 0.0)
    geom = _safe_float(result.get("geometry_proxy"), 0.0)
    objective = raw + 0.005 * geom
    block_1p_ratio = abs(raw) / abs(block_1p_raw) if abs(block_1p_raw) > 1e-12 else ""
    budget_block_ratio = abs(raw) / abs(budget_block_raw) if abs(budget_block_raw) > 1e-12 else ""
    return {
        "file": file_path,
        "sequence": sequence,
        "frame_id": frame_id,
        "candidate_name": candidate_name,
        "candidate_family": candidate_family,
        "requested_budget_ratio": float(budget),
        "budget_ratio": float(budget),
        "actual_raw_percent": raw,
        "objective_J": float(objective),
        "raw_bit": result.get("raw_bit", ""),
        "base_bit": result.get("base_bit", ""),
        "estimated_bits_total": result.get("estimated_bits_total", result.get("sparsepcgc_estimated_occupancy_bits", "")),
        "estimated_bits_delta": result.get("estimated_bits_delta", ""),
        "geometry_proxy": geom,
        "prune_count": result.get("prune_count", 0),
        "add_count": result.get("add_count", 0),
        "selected_voxel_count": result.get("selected_voxel_count", result.get("drop_count", 0)),
        "selected_bit_sum": result.get("selected_bit_sum", 0.0),
        "selected_bit_mean": result.get("selected_bit_mean", 0.0),
        "selected_bit_max": result.get("selected_bit_max", 0.0),
        "selected_depth_hist_json": result.get("selected_depth_hist_json", "{}"),
        "selected_parent_count": result.get("selected_parent_count", ""),
        "selected_block_count": result.get("selected_block_count", ""),
        "selected_parent_bit_sum": result.get("selected_bit_sum", 0.0),
        "selected_block_bit_sum": result.get("selected_bit_sum", 0.0) if "block" in str(candidate_family) else "",
        "group_to_block_bit_ratio": "",
        "group_to_block_raw_gain_ratio": budget_block_ratio,
        "parent_emptying_ratio": result.get("parent_emptying_ratio", 0.0),
        "partial_context_damage_ratio": result.get("partial_context_damage_ratio", 0.0),
        "ancestor_reduction_count": result.get("ancestor_reduction_count", ""),
        "bits_by_depth_json": result.get("sparsepcgc_bits_by_depth_json", ""),
        "bits_by_depth_delta_json": "",
        "bits_by_block_topk_json": result.get("sparsepcgc_bits_by_block_topk_json", ""),
        "top_high_bit_nodes_json": result.get("sparsepcgc_top_high_bit_nodes_json", ""),
        "added_point_nn_distance_mean": result.get("added_point_nn_distance_mean", ""),
        "added_point_nn_distance_max": result.get("added_point_nn_distance_max", ""),
        "add_density_delta": result.get("add_density_delta", ""),
        "sampled_chamfer_proxy": result.get("sampled_chamfer_proxy", geom),
        "repair_parent_count": result.get("repair_parent_count", ""),
        "phase2d_best_raw_percent": phase2d_best,
        "block_only_1p_raw_percent": block_1p_raw,
        "block_only_budget_raw_percent": budget_block_raw,
        "ratio_to_block_only_1p": block_1p_ratio,
        "ratio_to_budgeted_block_only": budget_block_ratio,
        "base_bits_by_depth_json": base_stats.get("sparsepcgc_bits_by_depth_json", ""),
    }


def _load_phase2f_best() -> Dict[Tuple[str, str], float]:
    paths = [
        Path("/data/maejima/log/PHASE2F_actual_bit_guided_candidates.csv"),
        Path("/data/maejima/log/PHASE2F_actual_bit_guided_candidates_lr.csv"),
    ]
    out: Dict[Tuple[str, str], float] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("candidate_name") != "high_actual_bit_node_prune":
                        continue
                    key = (str(row.get("sequence", "")), str(row.get("frame_id", "")))
                    value = _safe_float(row.get("actual_raw_percent"), float("nan"))
                    if not math.isfinite(value):
                        continue
                    old = out.get(key, float("inf"))
                    if value < old:
                        out[key] = value
        except Exception:
            continue
    return out


def _phase2g_row(
    *,
    file_path: str,
    sequence: str,
    frame_id: str,
    candidate_name: str,
    candidate_family: str,
    budget: float,
    result: Mapping[str, object],
    base_stats: Mapping[str, object],
    phase2d_best: float,
    phase2f_high_bit: float,
    budget_block_raw: float,
    block_5p_raw: float,
) -> Dict[str, object]:
    raw = _safe_float(result.get("actual_raw_percent"), 0.0)
    geom = _safe_float(result.get("geometry_proxy"), 0.0)
    chamfer = _safe_float(result.get("sampled_chamfer_proxy"), geom)
    objective = raw + 0.003 * geom + 0.003 * chamfer
    ratio_budget = abs(raw) / abs(budget_block_raw) if abs(budget_block_raw) > 1e-12 else ""
    ratio_5p = abs(raw) / abs(block_5p_raw) if abs(block_5p_raw) > 1e-12 else ""
    return {
        "file": file_path,
        "sequence": sequence,
        "frame_id": frame_id,
        "candidate_name": candidate_name,
        "candidate_family": candidate_family,
        "requested_budget_ratio": float(budget),
        "budget_ratio": float(budget),
        "actual_raw_percent": raw,
        "raw_bit": result.get("raw_bit", ""),
        "base_bit": result.get("base_bit", ""),
        "estimated_bits_delta": result.get("estimated_bits_delta", ""),
        "prune_count": result.get("prune_count", result.get("selected_voxel_count", 0)),
        "prune_ratio": result.get("prune_ratio", 0.0),
        "add_count": result.get("add_count", 0),
        "add_ratio": result.get("add_ratio", 0.0),
        "objective_J": float(objective),
        "selected_bit_sum": result.get("selected_bit_sum", 0.0),
        "selected_bit_mean": result.get("selected_bit_mean", 0.0),
        "selected_bit_max": result.get("selected_bit_max", 0.0),
        "selected_depth_hist_json": result.get("selected_depth_hist_json", "{}"),
        "selected_parent_count": result.get("selected_parent_count", ""),
        "selected_block_count": result.get("selected_block_count", ""),
        "bits_by_depth_delta_json": "",
        "geometry_proxy": geom,
        "sampled_chamfer_proxy": chamfer,
        "d1_mse": result.get("d1_mse", ""),
        "d1_psnr": result.get("d1_psnr", ""),
        "d2_mse": result.get("d2_mse", ""),
        "d2_psnr": result.get("d2_psnr", ""),
        "mynet_d1_psnr": result.get("mynet_d1_psnr", ""),
        "mynet_d2_psnr": result.get("mynet_d2_psnr", ""),
        "pc_error_d1_psnr": result.get("pc_error_d1_psnr", ""),
        "pc_error_d2_psnr": result.get("pc_error_d2_psnr", ""),
        "pc_error_d1_mse": result.get("pc_error_d1_mse", ""),
        "pc_error_d2_mse": result.get("pc_error_d2_mse", ""),
        "pc_error_resolution": result.get("pc_error_resolution", ""),
        "pc_error_returncode": result.get("pc_error_returncode", ""),
        "mynet_d1_psnr": result.get("mynet_d1_psnr", ""),
        "mynet_d2_psnr": result.get("mynet_d2_psnr", ""),
        "pc_error_d1_psnr": result.get("pc_error_d1_psnr", ""),
        "pc_error_d2_psnr": result.get("pc_error_d2_psnr", ""),
        "pc_error_d1_mse": result.get("pc_error_d1_mse", ""),
        "pc_error_d2_mse": result.get("pc_error_d2_mse", ""),
        "pc_error_resolution": result.get("pc_error_resolution", ""),
        "pc_error_returncode": result.get("pc_error_returncode", ""),
        "d1_proxy": result.get("d1_proxy", ""),
        "d2_proxy": result.get("d2_proxy", ""),
        "normal_proxy": result.get("normal_proxy", ""),
        "density_drop_mean": result.get("density_drop_mean", ""),
        "density_drop_max": result.get("density_drop_max", ""),
        "hole_risk": result.get("hole_risk", ""),
        "curvature_removed_mean": result.get("curvature_removed_mean", ""),
        "boundary_removed_ratio": result.get("boundary_removed_ratio", ""),
        "local_over_prune_score": result.get("local_over_prune_score", ""),
        "per_parent_prune_cap": result.get("per_parent_prune_cap", ""),
        "per_block_prune_cap": result.get("per_block_prune_cap", ""),
        "local_cell_prune_ratio_cap": result.get("local_cell_prune_ratio_cap", ""),
        "distributed_prune_score": result.get("distributed_prune_score", ""),
        "blue_noise_min_distance": result.get("blue_noise_min_distance", ""),
        "selected_spatial_cell_count": result.get("selected_spatial_cell_count", ""),
        "repair_add_count": result.get("repair_add_count", ""),
        "repair_add_ratio": result.get("repair_add_ratio", ""),
        "repair_density_recovery": "",
        "repair_raw_cost": "",
        "repair_quality_gain": "",
        "phase2d_best_raw_percent": phase2d_best,
        "phase2f_high_bit_raw_percent": phase2f_high_bit,
        "block_only_budget_raw_percent": budget_block_raw,
        "ratio_to_block_only_budget": ratio_budget,
        "ratio_to_block_only_5p": ratio_5p,
        "base_bits_by_depth_json": base_stats.get("sparsepcgc_bits_by_depth_json", ""),
        "bits_by_depth_json": result.get("sparsepcgc_bits_by_depth_json", ""),
    }


def _phase2h_row(
    *,
    file_path: str,
    sequence: str,
    frame_id: str,
    candidate_name: str,
    candidate_family: str,
    budget: float,
    pool_size: int,
    result: Mapping[str, object],
    block_result: Mapping[str, object],
    rate_weight: float,
    quality_weight: float,
    repair_ratio: float,
) -> Dict[str, object]:
    raw = _safe_float(result.get("actual_raw_percent"), 0.0)
    block_raw = _safe_float(block_result.get("actual_raw_percent"), float("nan"))
    ratio_to_block = abs(raw) / abs(block_raw) if abs(block_raw) > 1e-12 else ""
    actual_prune_ratio = _safe_float(result.get("actual_prune_ratio", result.get("prune_ratio")), 0.0)
    budget_reached = bool(result.get("budget_reached", actual_prune_ratio >= float(budget) * 0.98))
    quality_risk = _safe_float(result.get("hole_risk"), 0.0) + _safe_float(result.get("boundary_removed_ratio"), 0.0)
    return {
        "file": file_path,
        "sequence": sequence,
        "frame_id": frame_id,
        "candidate_name": candidate_name,
        "candidate_family": candidate_family,
        "requested_budget_ratio": float(budget),
        "budget_ratio": float(budget),
        "actual_prune_ratio": actual_prune_ratio,
        "budget_reached": bool(budget_reached),
        "candidate_pool_size": int(pool_size),
        "actual_raw_percent": raw,
        "raw_bit": result.get("raw_bit", ""),
        "base_bit": result.get("base_bit", ""),
        "estimated_bits_delta": result.get("estimated_bits_delta", ""),
        "prune_count": result.get("prune_count", result.get("selected_voxel_count", 0)),
        "add_count": result.get("add_count", 0),
        "add_ratio": result.get("add_ratio", 0.0),
        "d1_mse": result.get("d1_mse", ""),
        "d1_psnr": result.get("d1_psnr", ""),
        "d2_mse": result.get("d2_mse", ""),
        "d2_psnr": result.get("d2_psnr", ""),
        "chamfer": result.get("chamfer", result.get("sampled_chamfer_proxy", "")),
        "sampled_chamfer_proxy": result.get("sampled_chamfer_proxy", ""),
        "geometry_proxy": result.get("geometry_proxy", ""),
        "hausdorff_proxy": result.get("hausdorff_proxy", ""),
        "point_to_plane_proxy": result.get("point_to_plane_proxy", result.get("d2_mse", "")),
        "quality_eval_mode": result.get("quality_eval_mode", ""),
        "density_drop_mean": result.get("density_drop_mean", ""),
        "density_drop_max": result.get("density_drop_max", ""),
        "hole_risk": result.get("hole_risk", ""),
        "curvature_removed_mean": result.get("curvature_removed_mean", ""),
        "boundary_removed_ratio": result.get("boundary_removed_ratio", ""),
        "pca_plane_residual_mean": result.get("normal_proxy", ""),
        "min_remaining_neighbors_mean": "",
        "local_over_prune_score": result.get("local_over_prune_score", ""),
        "selected_bit_sum": result.get("selected_bit_sum", 0.0),
        "selected_bit_mean": result.get("selected_bit_mean", 0.0),
        "selected_bit_max": result.get("selected_bit_max", 0.0),
        "selected_depth_hist_json": result.get("selected_depth_hist_json", "{}"),
        "selected_parent_count": result.get("selected_parent_count", ""),
        "selected_block_count": result.get("selected_block_count", ""),
        "same_parent_prune_max": result.get("same_parent_prune_max", ""),
        "same_block_prune_ratio_max": result.get("same_block_prune_ratio_max", ""),
        "vetoed_count": result.get("vetoed_count", ""),
        "veto_reason_counts_json": result.get("veto_reason_counts_json", ""),
        "fill_from_lower_bit_count": result.get("fill_from_lower_bit_count", ""),
        "per_parent_prune_cap": result.get("per_parent_prune_cap", ""),
        "per_block_prune_ratio_cap": result.get("per_block_prune_cap", ""),
        "rate_weight": float(rate_weight),
        "quality_weight": float(quality_weight),
        "surface_score_mean": "",
        "quality_risk_mean": quality_risk,
        "rd_score_mean": "",
        "repair_ratio": float(repair_ratio),
        "repair_add_count": result.get("repair_add_count", ""),
        "repair_raw_cost": result.get("repair_raw_cost", ""),
        "repair_quality_gain": result.get("repair_quality_gain", ""),
        "repair_density_recovery": result.get("repair_density_recovery", ""),
        "block_only_budget_raw_percent": block_raw,
        "block_only_budget_d1_psnr": block_result.get("d1_psnr", ""),
        "block_only_budget_d2_psnr": block_result.get("d2_psnr", ""),
        "block_only_budget_chamfer": block_result.get("chamfer", block_result.get("sampled_chamfer_proxy", "")),
        "ratio_to_block_only_budget": ratio_to_block,
    }


def run_phase2f_sweep(cli) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2F actual-bit-map candidate probe"),
            time.strftime("%Y%m%d"),
            time.strftime("%H%M%S"),
        )
    finally:
        sys.argv = old_argv
    args.compress = "SparsePCGC"
    args.compression_loss_backend = "sparsepcgc_surrogate"
    args.sparsepcgc_skip_decode = True
    args.sparsepcgc_worker_gpu_stats = False
    args.enable_sparsepcgc_occupancy_debug = True
    args.enable_sparsepcgc_exact_occupancy_teacher = False
    args.sparsepcgc_exact_occupancy_interval = 1

    budgets = [float(x) for x in _parse_csv_text(cli.phase2f_budgets)]
    rows: List[Dict[str, object]] = []
    phase2d = _load_phase2d_baseline()
    encoder = build_actual_encoder(args)
    try:
        for file_idx, file_path in enumerate(cli.files):
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            sequence = Path(file_path).parent.name
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, args)
            base_stats = encoder.encode_bits(base_xyz)
            base_bits = float(base_stats.get("bit", 0.0))
            phase2d_best = float(phase2d.get((sequence, frame_id), float("nan")))

            block_1p_raw = float("nan")
            block_raw_by_budget: Dict[float, float] = {}
            block_debug_by_budget: Dict[float, Dict[str, object]] = {}
            for budget in budgets:
                block_coords, _mask, block_debug = build_candidate_coords(
                    "block_only",
                    coords,
                    float(budget),
                    block_size=int(cli.block_size),
                    seed=int(cli.seed) + file_idx,
                    max_operation_edits=max(int(coords.shape[0]), int(cli.max_operation_edits)),
                )
                block_res = _evaluate_coords_for_phase2f(
                    encoder=encoder,
                    args=args,
                    coords=coords,
                    cand_coords=block_coords,
                    meta=meta,
                    base_bits=base_bits,
                    base_stats=base_stats,
                    method_debug=dict(block_debug),
                    block_size=int(cli.block_size),
                    max_geometry_samples=int(cli.max_geometry_samples),
                )
                block_raw_by_budget[float(budget)] = _safe_float(block_res.get("actual_raw_percent"), float("nan"))
                block_debug_by_budget[float(budget)] = block_res
                if abs(float(budget) - 0.010) < 1e-9:
                    block_1p_raw = block_raw_by_budget[float(budget)]

            if not math.isfinite(block_1p_raw):
                block_1p_raw = block_raw_by_budget.get(min(budgets, key=lambda b: abs(b - 0.010)), float("nan"))

            for budget in budgets:
                noop_res = {
                    "actual_raw_percent": 0.0,
                    "raw_bit": base_bits,
                    "base_bit": base_bits,
                    "estimated_bits_total": base_stats.get("sparsepcgc_estimated_occupancy_bits", ""),
                    "estimated_bits_delta": 0.0,
                    "geometry_proxy": 0.0,
                    "prune_count": 0,
                    "add_count": 0,
                    "selected_voxel_count": 0,
                }
                for key in SPARSEPCGC_RATE_DEBUG_KEYS:
                    if key in base_stats:
                        noop_res[key] = base_stats.get(key, "")
                rows.append(_phase2f_row(
                    file_path=str(file_path),
                    sequence=sequence,
                    frame_id=frame_id,
                    candidate_name="noop",
                    candidate_family="baseline",
                    budget=budget,
                    result=noop_res,
                    base_stats=base_stats,
                    phase2d_best=phase2d_best,
                    block_1p_raw=block_1p_raw,
                    budget_block_raw=block_raw_by_budget.get(float(budget), float("nan")),
                ))

                rows.append(_phase2f_row(
                    file_path=str(file_path),
                    sequence=sequence,
                    frame_id=frame_id,
                    candidate_name="budgeted_block_only",
                    candidate_family="block_only_baseline",
                    budget=budget,
                    result=block_debug_by_budget[float(budget)],
                    base_stats=base_stats,
                    phase2d_best=phase2d_best,
                    block_1p_raw=block_1p_raw,
                    budget_block_raw=block_raw_by_budget.get(float(budget), float("nan")),
                ))

                node_mask, node_debug = _high_actual_bit_node_mask(
                    coords,
                    base_stats,
                    budget_ratio=float(budget),
                    max_nodes=int(cli.max_operation_edits),
                )
                node_coords = torch.unique(coords[~node_mask], dim=0, sorted=True)
                node_res = _evaluate_coords_for_phase2f(
                    encoder=encoder,
                    args=args,
                    coords=coords,
                    cand_coords=node_coords,
                    meta=meta,
                    base_bits=base_bits,
                    base_stats=base_stats,
                    method_debug=node_debug,
                    block_size=int(cli.block_size),
                    max_geometry_samples=int(cli.max_geometry_samples),
                )
                rows.append(_phase2f_row(
                    file_path=str(file_path),
                    sequence=sequence,
                    frame_id=frame_id,
                    candidate_name="high_actual_bit_node_prune",
                    candidate_family="actual_bit_node",
                    budget=budget,
                    result=node_res,
                    base_stats=base_stats,
                    phase2d_best=phase2d_best,
                    block_1p_raw=block_1p_raw,
                    budget_block_raw=block_raw_by_budget.get(float(budget), float("nan")),
                ))

                parent_mask, parent_debug = _high_actual_bit_parent_mask(
                    coords,
                    base_stats,
                    budget_ratio=float(budget),
                    max_parents=int(cli.phase2f_parent_topk),
                    block_filter=False,
                )
                parent_coords = torch.unique(coords[~parent_mask], dim=0, sorted=True)
                parent_res = _evaluate_coords_for_phase2f(
                    encoder=encoder,
                    args=args,
                    coords=coords,
                    cand_coords=parent_coords,
                    meta=meta,
                    base_bits=base_bits,
                    base_stats=base_stats,
                    method_debug=parent_debug,
                    block_size=int(cli.block_size),
                    max_geometry_samples=int(cli.max_geometry_samples),
                )
                rows.append(_phase2f_row(
                    file_path=str(file_path),
                    sequence=sequence,
                    frame_id=frame_id,
                    candidate_name="high_actual_bit_parent_group_prune",
                    candidate_family="actual_bit_parent",
                    budget=budget,
                    result=parent_res,
                    base_stats=base_stats,
                    phase2d_best=phase2d_best,
                    block_1p_raw=block_1p_raw,
                    budget_block_raw=block_raw_by_budget.get(float(budget), float("nan")),
                ))

                block_parent_mask, block_parent_debug = _high_actual_bit_parent_mask(
                    coords,
                    base_stats,
                    budget_ratio=float(budget),
                    max_parents=int(cli.phase2f_parent_topk),
                    block_filter=True,
                )
                block_parent_coords = torch.unique(coords[~block_parent_mask], dim=0, sorted=True)
                block_parent_res = _evaluate_coords_for_phase2f(
                    encoder=encoder,
                    args=args,
                    coords=coords,
                    cand_coords=block_parent_coords,
                    meta=meta,
                    base_bits=base_bits,
                    base_stats=base_stats,
                    method_debug=block_parent_debug,
                    block_size=int(cli.block_size),
                    max_geometry_samples=int(cli.max_geometry_samples),
                )
                rows.append(_phase2f_row(
                    file_path=str(file_path),
                    sequence=sequence,
                    frame_id=frame_id,
                    candidate_name="high_actual_bit_block_decomposition",
                    candidate_family="actual_bit_block_parent",
                    budget=budget,
                    result=block_parent_res,
                    base_stats=base_stats,
                    phase2d_best=phase2d_best,
                    block_1p_raw=block_1p_raw,
                    budget_block_raw=block_raw_by_budget.get(float(budget), float("nan")),
                ))

                repair_base_coords = parent_coords
                repair_coords, _repair_mask, repair_debug = build_candidate_coords(
                    "addv2_n256_g0p3_nn1_densmedium",
                    repair_base_coords,
                    float(budget),
                    block_size=int(cli.block_size),
                    seed=int(cli.seed) + file_idx + 73,
                    max_operation_edits=256,
                )
                repair_debug = dict(repair_debug)
                repair_debug.update(parent_debug)
                repair_debug["add_count"] = int(repair_debug.get("add_count", 0) or 0)
                repair_debug["repair_parent_count"] = parent_debug.get("selected_parent_count", "")
                repair_res = _evaluate_coords_for_phase2f(
                    encoder=encoder,
                    args=args,
                    coords=coords,
                    cand_coords=torch.unique(repair_coords.to(dtype=torch.long), dim=0, sorted=True),
                    meta=meta,
                    base_bits=base_bits,
                    base_stats=base_stats,
                    method_debug=repair_debug,
                    block_size=int(cli.block_size),
                    max_geometry_samples=int(cli.max_geometry_samples),
                )
                rows.append(_phase2f_row(
                    file_path=str(file_path),
                    sequence=sequence,
                    frame_id=frame_id,
                    candidate_name="high_bit_prune_then_add_repair",
                    candidate_family="actual_bit_parent_repair",
                    budget=budget,
                    result=repair_res,
                    base_stats=base_stats,
                    phase2d_best=phase2d_best,
                    block_1p_raw=block_1p_raw,
                    budget_block_raw=block_raw_by_budget.get(float(budget), float("nan")),
                ))

                _write_csv(cli.output_csv, rows)
                print(json.dumps({
                    "phase2f": True,
                    "sequence": sequence,
                    "frame": frame_id,
                    "budget": budget,
                    "rows": len(rows),
                }, sort_keys=True), flush=True)
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
    _write_csv(cli.output_csv, rows)
    return 0


def run_phase2g_sweep(cli) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2G aggressive RD candidate probe"),
            time.strftime("%Y%m%d"),
            time.strftime("%H%M%S"),
        )
    finally:
        sys.argv = old_argv
    args.compress = "SparsePCGC"
    args.compression_loss_backend = "sparsepcgc_surrogate"
    args.sparsepcgc_skip_decode = True
    args.sparsepcgc_worker_gpu_stats = False
    args.enable_sparsepcgc_occupancy_debug = True
    args.enable_sparsepcgc_exact_occupancy_teacher = False
    args.sparsepcgc_exact_occupancy_interval = 1

    budgets = [float(x) for x in _parse_csv_text(cli.phase2g_budgets)]
    candidates = list(_parse_csv_text(cli.phase2g_candidates))
    rows: List[Dict[str, object]] = []
    phase2d = _load_phase2d_baseline()
    phase2f = _load_phase2f_best()
    encoder = build_actual_encoder(args)
    try:
        for file_idx, file_path in enumerate(cli.files):
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            sequence = Path(file_path).parent.name
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, args)
            base_stats = encoder.encode_bits(base_xyz)
            base_bits = float(base_stats.get("bit", 0.0))
            phase2d_best = float(phase2d.get((sequence, frame_id), float("nan")))
            phase2f_high_bit = float(phase2f.get((sequence, frame_id), float("nan")))

            block_raw_by_budget: Dict[float, float] = {}
            block_result_by_budget: Dict[float, Dict[str, object]] = {}
            for budget in budgets:
                block_coords, _mask, block_debug = build_candidate_coords(
                    "block_only",
                    coords,
                    float(budget),
                    block_size=int(cli.block_size),
                    seed=int(cli.seed) + file_idx,
                    max_operation_edits=max(int(coords.shape[0]), int(cli.max_operation_edits)),
                )
                block_res = _evaluate_coords_for_phase2f(
                    encoder=encoder,
                    args=args,
                    coords=coords,
                    cand_coords=block_coords,
                    meta=meta,
                    base_bits=base_bits,
                    base_stats=base_stats,
                    method_debug=dict(block_debug),
                    block_size=int(cli.block_size),
                    max_geometry_samples=int(cli.max_geometry_samples),
                )
                # Add the same proxy columns for fair table comparisons.
                keys, _occupied = _coord_key_setup(coords)
                block_keys = torch.unique(keys(block_coords), sorted=True)
                keep = _lookup_occupied(keys(coords), block_keys)
                block_res.update(_aggressive_quality_proxy(coords, keep, ~keep, max_samples=int(cli.max_geometry_samples)))
                block_res["prune_ratio"] = float((~keep).sum().item()) / max(float(coords.shape[0]), 1.0)
                block_raw_by_budget[float(budget)] = _safe_float(block_res.get("actual_raw_percent"), float("nan"))
                block_result_by_budget[float(budget)] = block_res

            block_5p_raw = block_raw_by_budget.get(
                0.050,
                block_raw_by_budget.get(max(budgets, key=lambda b: b), float("nan")),
            )

            for budget in budgets:
                noop = {
                    "actual_raw_percent": 0.0,
                    "raw_bit": base_bits,
                    "base_bit": base_bits,
                    "estimated_bits_delta": 0.0,
                    "geometry_proxy": 0.0,
                    "sampled_chamfer_proxy": 0.0,
                    "d1_mse": 0.0,
                    "d1_psnr": float("inf"),
                    "d2_mse": 0.0,
                    "d2_psnr": float("inf"),
                    "prune_count": 0,
                    "prune_ratio": 0.0,
                }
                rows.append(_phase2g_row(
                    file_path=str(file_path),
                    sequence=sequence,
                    frame_id=frame_id,
                    candidate_name="noop",
                    candidate_family="baseline",
                    budget=budget,
                    result=noop,
                    base_stats=base_stats,
                    phase2d_best=phase2d_best,
                    phase2f_high_bit=phase2f_high_bit,
                    budget_block_raw=block_raw_by_budget.get(float(budget), float("nan")),
                    block_5p_raw=block_5p_raw,
                ))
                rows.append(_phase2g_row(
                    file_path=str(file_path),
                    sequence=sequence,
                    frame_id=frame_id,
                    candidate_name="block_only",
                    candidate_family="block_only_baseline",
                    budget=budget,
                    result=block_result_by_budget[float(budget)],
                    base_stats=base_stats,
                    phase2d_best=phase2d_best,
                    phase2f_high_bit=phase2f_high_bit,
                    budget_block_raw=block_raw_by_budget.get(float(budget), float("nan")),
                    block_5p_raw=block_5p_raw,
                ))

                for candidate in candidates:
                    if candidate in {"noop", "block_only"}:
                        continue
                    mode = str(candidate)
                    repair_fraction = 0.0
                    if mode == "high_actual_bit_prune_then_add_surface_repair":
                        mode = "high_actual_bit_surface_safe_prune"
                        repair_fraction = float(cli.phase2g_repair_fraction)
                    drop_mask, debug = _select_aggressive_high_bit_mask(
                        coords,
                        base_stats,
                        budget_ratio=float(budget),
                        mode=mode,
                        max_pool=int(cli.phase2g_top_pool),
                        block_size=int(cli.block_size),
                        repair_fraction=repair_fraction,
                    )
                    debug = dict(debug)
                    debug["candidate_name"] = candidate
                    if candidate == "high_actual_bit_prune_then_add_surface_repair":
                        debug["candidate_family"] = "surface_repair"
                    result = _evaluate_mask_for_phase2g(
                        encoder=encoder,
                        args=args,
                        coords=coords,
                        drop_mask=drop_mask,
                        meta=meta,
                        base_bits=base_bits,
                        base_stats=base_stats,
                        method_debug=debug,
                        block_size=int(cli.block_size),
                        max_geometry_samples=int(cli.max_geometry_samples),
                    )
                    rows.append(_phase2g_row(
                        file_path=str(file_path),
                        sequence=sequence,
                        frame_id=frame_id,
                        candidate_name=candidate,
                        candidate_family=str(debug.get("candidate_family", mode)),
                        budget=budget,
                        result=result,
                        base_stats=base_stats,
                        phase2d_best=phase2d_best,
                        phase2f_high_bit=phase2f_high_bit,
                        budget_block_raw=block_raw_by_budget.get(float(budget), float("nan")),
                        block_5p_raw=block_5p_raw,
                    ))
                    _write_csv(cli.output_csv, rows)
                print(json.dumps({
                    "phase2g": True,
                    "sequence": sequence,
                    "frame": frame_id,
                    "budget": budget,
                    "rows": len(rows),
                }, sort_keys=True), flush=True)
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
    _write_csv(cli.output_csv, rows)
    return 0


def run_phase2h_sweep(cli) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2H formal RD high-bit probe"),
            time.strftime("%Y%m%d"),
            time.strftime("%H%M%S"),
        )
    finally:
        sys.argv = old_argv
    args.compress = "SparsePCGC"
    args.compression_loss_backend = "sparsepcgc_surrogate"
    args.sparsepcgc_skip_decode = True
    args.sparsepcgc_worker_gpu_stats = False
    args.enable_sparsepcgc_occupancy_debug = True
    args.enable_sparsepcgc_exact_occupancy_teacher = False
    args.sparsepcgc_exact_occupancy_interval = 1

    budgets = [float(x) for x in _parse_csv_text(cli.phase2h_budgets)]
    pools = [int(float(x)) for x in _parse_csv_text(cli.phase2h_pools)]
    candidates = list(_parse_csv_text(cli.phase2h_candidates))
    quality_weights = [float(x) for x in _parse_csv_text(cli.phase2h_quality_weights)]
    repair_ratios = [float(x) for x in _parse_csv_text(cli.phase2h_repair_ratios)]
    max_pool = max(pools) if pools else int(cli.phase2g_top_pool)
    args.sparsepcgc_occupancy_debug_topk_final = int(max_pool)
    args.sparsepcgc_occupancy_debug_topk_per_layer = max(1024, min(int(max_pool), 8192))

    rows: List[Dict[str, object]] = []
    encoder = build_actual_encoder(args)
    try:
        for file_idx, file_path in enumerate(cli.files):
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            sequence = Path(file_path).parent.name
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, args)
            base_stats = encoder.encode_bits(base_xyz)
            base_bits = float(base_stats.get("bit", 0.0))

            block_by_budget: Dict[float, Dict[str, object]] = {}
            for budget in budgets:
                block_coords, _mask, block_debug = build_candidate_coords(
                    "block_only",
                    coords,
                    float(budget),
                    block_size=int(cli.block_size),
                    seed=int(cli.seed) + file_idx,
                    max_operation_edits=max(int(coords.shape[0]), int(cli.max_operation_edits)),
                )
                block_drop = _drop_mask_against_original(coords, torch.unique(block_coords.to(dtype=torch.long), dim=0, sorted=True))
                block_res = _evaluate_mask_for_phase2h(
                    encoder=encoder,
                    args=args,
                    coords=coords,
                    drop_mask=block_drop,
                    meta=meta,
                    base_bits=base_bits,
                    base_stats=base_stats,
                    method_debug=dict(block_debug),
                    block_size=int(cli.block_size),
                    max_geometry_samples=int(cli.max_geometry_samples),
                    formal_max_points=int(cli.phase2h_quality_max_points),
                    normal_max_points=int(cli.phase2h_normal_max_points),
                    pc_error_path=str(getattr(cli, "pc_error_path", "")),
                    use_pc_error=bool(getattr(cli, "use_pc_error", False)),
                )
                block_by_budget[float(budget)] = block_res
                rows.append(_phase2h_row(
                    file_path=str(file_path),
                    sequence=sequence,
                    frame_id=frame_id,
                    candidate_name="block_only",
                    candidate_family="block_only_baseline",
                    budget=budget,
                    pool_size=max_pool,
                    result=block_res,
                    block_result=block_res,
                    rate_weight=1.0,
                    quality_weight=0.0,
                    repair_ratio=0.0,
                ))
                _write_csv(cli.output_csv, rows)

            for pool in pools:
                for budget in budgets:
                    block_res = block_by_budget[float(budget)]
                    for candidate in candidates:
                        if candidate in {"noop", "block_only"}:
                            continue
                        if candidate == "high_bit_rd_score_prune":
                            q_values = quality_weights
                            repair_values = [0.0]
                        elif candidate in {"high_bit_prune_then_surface_repair", "high_bit_prune_then_light_repair"}:
                            q_values = [0.5]
                            repair_values = repair_ratios
                        else:
                            q_values = [0.0]
                            repair_values = [0.0]
                        for q_weight in q_values:
                            for repair_ratio in repair_values:
                                mode = candidate
                                if candidate == "high_bit_prune_then_surface_repair":
                                    mode = "high_bit_prune_then_surface_repair"
                                drop_mask, debug = _select_aggressive_high_bit_mask(
                                    coords,
                                    base_stats,
                                    budget_ratio=float(budget),
                                    mode=mode,
                                    max_pool=int(pool),
                                    block_size=int(cli.block_size),
                                    repair_fraction=float(repair_ratio),
                                    quality_weight=float(q_weight),
                                )
                                debug = dict(debug)
                                result = _evaluate_mask_for_phase2h(
                                    encoder=encoder,
                                    args=args,
                                    coords=coords,
                                    drop_mask=drop_mask,
                                    meta=meta,
                                    base_bits=base_bits,
                                    base_stats=base_stats,
                                    method_debug=debug,
                                    block_size=int(cli.block_size),
                                    max_geometry_samples=int(cli.max_geometry_samples),
                                    formal_max_points=int(cli.phase2h_quality_max_points),
                                    normal_max_points=int(cli.phase2h_normal_max_points),
                                    pc_error_path=str(getattr(cli, "pc_error_path", "")),
                                    use_pc_error=bool(getattr(cli, "use_pc_error", False)),
                                )
                                name = candidate
                                if candidate == "high_bit_rd_score_prune":
                                    name = f"{candidate}_qw{q_weight:g}"
                                if candidate == "high_bit_prune_then_surface_repair":
                                    name = f"{candidate}_repair{repair_ratio:g}"
                                rows.append(_phase2h_row(
                                    file_path=str(file_path),
                                    sequence=sequence,
                                    frame_id=frame_id,
                                    candidate_name=name,
                                    candidate_family=candidate,
                                    budget=budget,
                                    pool_size=int(pool),
                                    result=result,
                                    block_result=block_res,
                                    rate_weight=1.0,
                                    quality_weight=float(q_weight),
                                    repair_ratio=float(repair_ratio),
                                ))
                                _write_csv(cli.output_csv, rows)
                    print(json.dumps({
                        "phase2h": True,
                        "sequence": sequence,
                        "frame": frame_id,
                        "budget": budget,
                        "pool": pool,
                        "rows": len(rows),
                    }, sort_keys=True), flush=True)
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
    _write_csv(cli.output_csv, rows)
    return 0


def _phase2j_row(
    *,
    file_path: str,
    sequence: str,
    frame_id: str,
    candidate_name: str,
    budget: float,
    pool: int,
    base_bits: float,
    baseline_stats: Mapping[str, object],
    processed_stats: Mapping[str, object],
    pre_quality: Mapping[str, object],
    baseline_quality: Mapping[str, object],
    processed_quality: Mapping[str, object],
    debug: Mapping[str, object],
    decoded_gt_path: str,
    decoded_processed_path: str,
    baseline_decode_count: int,
    processed_decode_count: int,
    baseline_match_ratio: float,
    processed_match_ratio: float,
    baseline_lossless: bool,
    processed_lossless: bool,
) -> Dict[str, object]:
    raw_bit = _safe_float(processed_stats.get("bit"), 0.0)
    actual_raw = 100.0 * (raw_bit - float(base_bits)) / max(float(base_bits), 1.0)
    base_d1_mse = _safe_float(baseline_quality.get("d1_mse"), float("nan"))
    proc_d1_mse = _safe_float(processed_quality.get("d1_mse"), float("nan"))
    base_d2_mse = _safe_float(baseline_quality.get("d2_mse"), float("nan"))
    proc_d2_mse = _safe_float(processed_quality.get("d2_mse"), float("nan"))
    base_cham = _safe_float(baseline_quality.get("chamfer"), float("nan"))
    proc_cham = _safe_float(processed_quality.get("chamfer"), float("nan"))
    base_d1_psnr = _safe_float(baseline_quality.get("d1_psnr"), float("nan"))
    proc_d1_psnr = _safe_float(processed_quality.get("d1_psnr"), float("nan"))
    base_d2_psnr = _safe_float(baseline_quality.get("d2_psnr"), float("nan"))
    proc_d2_psnr = _safe_float(processed_quality.get("d2_psnr"), float("nan"))
    return {
        "file": file_path,
        "sequence": sequence,
        "frame_id": frame_id,
        "candidate_name": candidate_name,
        "requested_budget_ratio": float(budget),
        "actual_prune_ratio": debug.get("actual_prune_ratio", debug.get("prune_ratio", 0.0)),
        "budget_reached": debug.get("budget_reached", ""),
        "candidate_pool_size": int(pool),
        "prune_count": processed_stats.get("prune_count", debug.get("prune_count", 0)),
        "add_count": processed_stats.get("add_count", debug.get("add_count", 0)),
        "baseline_bits": float(base_bits),
        "processed_bits": raw_bit,
        "actual_raw_percent": actual_raw,
        "raw_bit": raw_bit,
        "base_bit": float(base_bits),
        "estimated_bits_delta": _safe_float(processed_stats.get("sparsepcgc_estimated_occupancy_bits"), float("nan"))
        - _safe_float(baseline_stats.get("sparsepcgc_estimated_occupancy_bits"), float("nan")),
        "baseline_decoded_point_count": int(baseline_decode_count),
        "processed_decoded_point_count": int(processed_decode_count),
        "baseline_decode_coord_match_ratio": baseline_match_ratio,
        "processed_decode_coord_match_ratio": processed_match_ratio,
        "baseline_decode_lossless": bool(baseline_lossless),
        "processed_decode_lossless": bool(processed_lossless),
        "decoded_gt_path": decoded_gt_path,
        "decoded_processed_path": decoded_processed_path,
        "pre_d1_psnr": pre_quality.get("d1_psnr", ""),
        "pre_d2_psnr": pre_quality.get("d2_psnr", ""),
        "pre_chamfer": pre_quality.get("chamfer", ""),
        "baseline_decoded_d1_mse": baseline_quality.get("d1_mse", ""),
        "baseline_decoded_d1_psnr": baseline_quality.get("d1_psnr", ""),
        "baseline_decoded_d2_mse": baseline_quality.get("d2_mse", ""),
        "baseline_decoded_d2_psnr": baseline_quality.get("d2_psnr", ""),
        "baseline_decoded_chamfer": baseline_quality.get("chamfer", ""),
        "processed_decoded_d1_mse": processed_quality.get("d1_mse", ""),
        "processed_decoded_d1_psnr": processed_quality.get("d1_psnr", ""),
        "processed_decoded_d2_mse": processed_quality.get("d2_mse", ""),
        "processed_decoded_d2_psnr": processed_quality.get("d2_psnr", ""),
        "processed_decoded_chamfer": processed_quality.get("chamfer", ""),
        "delta_d1_mse": proc_d1_mse - base_d1_mse if math.isfinite(proc_d1_mse) and math.isfinite(base_d1_mse) else "",
        "delta_d2_mse": proc_d2_mse - base_d2_mse if math.isfinite(proc_d2_mse) and math.isfinite(base_d2_mse) else "",
        "delta_chamfer": proc_cham - base_cham if math.isfinite(proc_cham) and math.isfinite(base_cham) else "",
        "d1_psnr_drop": base_d1_psnr - proc_d1_psnr if math.isfinite(base_d1_psnr) and math.isfinite(proc_d1_psnr) else "",
        "d2_psnr_drop": base_d2_psnr - proc_d2_psnr if math.isfinite(base_d2_psnr) and math.isfinite(proc_d2_psnr) else "",
        "selected_bit_sum": debug.get("selected_bit_sum", ""),
        "selected_bit_mean": debug.get("selected_bit_mean", ""),
        "selected_depth_hist_json": debug.get("selected_depth_hist_json", ""),
        "selected_parent_count": debug.get("selected_parent_count", ""),
        "selected_block_count": debug.get("selected_block_count", ""),
        "quality_eval_mode": processed_quality.get("quality_eval_mode", ""),
        "pc_error_d_success": processed_quality.get("pc_error_d_success", ""),
        "mynet_eval_success": processed_quality.get("mynet_eval_success", ""),
    }


def run_phase2j_sweep(cli) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        base_args = parse_pugan_args(
            argparse.ArgumentParser(description="phase2J end-to-end RD probe"),
            time.strftime("%Y%m%d"),
            time.strftime("%H%M%S"),
        )
    finally:
        sys.argv = old_argv
    for attr, value in {
        "compress": "SparsePCGC",
        "compression_loss_backend": "sparsepcgc_surrogate",
        "sparsepcgc_worker_gpu_stats": False,
        "enable_sparsepcgc_occupancy_debug": True,
        "enable_sparsepcgc_exact_occupancy_teacher": False,
        "sparsepcgc_exact_occupancy_interval": 1,
    }.items():
        setattr(base_args, attr, value)
    debug_args = copy.copy(base_args)
    debug_args.sparsepcgc_skip_decode = True
    max_pool = max(int(float(x)) for x in _parse_csv_text(cli.phase2j_pools))
    debug_args.sparsepcgc_occupancy_debug_topk_final = int(max_pool)
    debug_args.sparsepcgc_occupancy_debug_topk_per_layer = max(1024, min(int(max_pool), 8192))
    decode_args = copy.copy(base_args)
    decode_args.sparsepcgc_skip_decode = False
    decode_args.enable_sparsepcgc_occupancy_debug = False
    decode_copy_dir = str(cli.phase2j_decoded_dir)
    Path(decode_copy_dir).mkdir(parents=True, exist_ok=True)
    decode_args.sparsepcgc_decoded_copy_dir = decode_copy_dir

    budgets = [float(x) for x in _parse_csv_text(cli.phase2j_budgets)]
    pools = [int(float(x)) for x in _parse_csv_text(cli.phase2j_pools)]
    candidates = list(_parse_csv_text(cli.phase2j_candidates))
    rows: List[Dict[str, object]] = []
    debug_encoder = build_actual_encoder(debug_args)
    decode_encoder = build_actual_encoder(decode_args)
    try:
        for file_idx, file_path in enumerate(cli.files):
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, base_args)
            sequence = Path(file_path).parent.name
            frame_id = Path(file_path).stem
            base_xyz = _coords_to_xyz(coords, meta, base_args)
            base_stats = debug_encoder.encode_bits(base_xyz)
            baseline_stats = decode_encoder.encode_bits(base_xyz)
            base_bits = float(baseline_stats.get("bit", base_stats.get("bit", 0.0)))
            decoded_gt_path = str(baseline_stats.get("decoded_copy_path", ""))
            baseline_count, baseline_match, baseline_lossless = _coord_match_ratio_from_paths(file_path, decoded_gt_path) if decoded_gt_path else (0, float("nan"), False)
            baseline_quality = _quality_from_paths(
                file_path,
                decoded_gt_path,
                formal_max_points=int(cli.phase2j_quality_max_points),
                normal_max_points=int(cli.phase2j_normal_max_points),
                pc_error_path=str(cli.pc_error_path),
                use_pc_error=bool(cli.use_pc_error),
            ) if decoded_gt_path else {}

            for pool in pools:
                for budget in budgets:
                    candidate_items: List[Tuple[str, torch.Tensor, Dict[str, object]]] = []
                    if "block_only" in candidates:
                        block_coords, _mask, block_debug = build_candidate_coords(
                            "block_only",
                            coords,
                            float(budget),
                            block_size=int(cli.block_size),
                            seed=int(cli.seed) + file_idx,
                            max_operation_edits=max(int(coords.shape[0]), int(cli.max_operation_edits)),
                        )
                        block_drop = _drop_mask_against_original(coords, torch.unique(block_coords.to(dtype=torch.long), dim=0, sorted=True))
                        block_debug = dict(block_debug)
                        block_debug["actual_prune_ratio"] = float(block_drop.sum().item()) / max(float(coords.shape[0]), 1.0)
                        block_debug["budget_reached"] = bool(block_debug["actual_prune_ratio"] >= float(budget) * 0.98)
                        candidate_items.append(("block_only", block_drop, block_debug))
                    if "high_bit_raw_prune" in candidates:
                        drop, debug = _select_aggressive_high_bit_mask(
                            coords,
                            base_stats,
                            budget_ratio=float(budget),
                            mode="high_bit_raw_prune",
                            max_pool=int(pool),
                            block_size=int(cli.block_size),
                        )
                        candidate_items.append(("high_bit_raw_prune", drop, dict(debug)))
                    for candidate_name, drop_mask, debug in candidate_items:
                        cand_coords = torch.unique(coords[~drop_mask].to(dtype=torch.long), dim=0, sorted=True)
                        cand_xyz = _coords_to_xyz(cand_coords, meta, base_args)
                        processed_stats = decode_encoder.encode_bits(cand_xyz)
                        processed_path = str(processed_stats.get("decoded_copy_path", ""))
                        processed_count, processed_match, processed_lossless = _coord_match_ratio_from_paths(file_path, processed_path) if processed_path else (0, float("nan"), False)
                        # Pre-encode quality uses a temporary processed PLY, not codec output.
                        with tempfile.TemporaryDirectory(prefix="phase2j_pre_") as tmp:
                            pre_path = Path(tmp) / "processed_pre.ply"
                            write_ascii_ply_xyz(pre_path, cand_xyz.detach().to("cpu").numpy().astype(np.float64, copy=False))
                            pre_quality = _quality_from_paths(
                                file_path,
                                pre_path,
                                formal_max_points=int(cli.phase2j_quality_max_points),
                                normal_max_points=int(cli.phase2j_normal_max_points),
                                pc_error_path=str(cli.pc_error_path),
                                use_pc_error=bool(cli.use_pc_error),
                            )
                        processed_quality = _quality_from_paths(
                            file_path,
                            processed_path,
                            formal_max_points=int(cli.phase2j_quality_max_points),
                            normal_max_points=int(cli.phase2j_normal_max_points),
                            pc_error_path=str(cli.pc_error_path),
                            use_pc_error=bool(cli.use_pc_error),
                        ) if processed_path else {}
                        processed_stats = dict(processed_stats)
                        processed_stats["prune_count"] = int(drop_mask.sum().item())
                        rows.append(_phase2j_row(
                            file_path=str(file_path),
                            sequence=sequence,
                            frame_id=frame_id,
                            candidate_name=candidate_name,
                            budget=budget,
                            pool=pool,
                            base_bits=base_bits,
                            baseline_stats=baseline_stats,
                            processed_stats=processed_stats,
                            pre_quality=pre_quality,
                            baseline_quality=baseline_quality,
                            processed_quality=processed_quality,
                            debug=debug,
                            decoded_gt_path=decoded_gt_path,
                            decoded_processed_path=processed_path,
                            baseline_decode_count=baseline_count,
                            processed_decode_count=processed_count,
                            baseline_match_ratio=baseline_match,
                            processed_match_ratio=processed_match,
                            baseline_lossless=baseline_lossless,
                            processed_lossless=processed_lossless,
                        ))
                        _write_csv(cli.output_csv, rows)
                    print(json.dumps({
                        "phase2j": True,
                        "sequence": sequence,
                        "frame": frame_id,
                        "budget": budget,
                        "pool": pool,
                        "rows": len(rows),
                    }, sort_keys=True), flush=True)
    finally:
        for enc in (debug_encoder, decode_encoder):
            close = getattr(enc, "close", None)
            if callable(close):
                close()
    _write_csv(cli.output_csv, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2 RDO beam-search probe for SparsePCGC edits")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--amount", type=float, default=0.010)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--diagnostic-methods", default=",".join(DEFAULT_DIAGNOSTIC_METHODS))
    parser.add_argument("--high-nll-guard-modes", default="hard,relaxed")
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-operation-edits", type=int, default=512)
    parser.add_argument("--max-geometry-samples", type=int, default=2048)
    parser.add_argument("--max-total-add-count", type=int, default=2048)
    parser.add_argument("--max-cumulative-add-ratio", type=float, default=0.003)
    parser.add_argument("--lambda-add", type=float, default=0.0)
    parser.add_argument("--lambda-geom", type=float, default=0.0)
    parser.add_argument("--lambda-density", type=float, default=0.0)
    parser.add_argument("--objective-mode", default="raw_only")
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--phase2f-sweep", action="store_true")
    parser.add_argument("--phase2f-budgets", default="0.005,0.010,0.020")
    parser.add_argument("--phase2f-parent-topk", type=int, default=64)
    parser.add_argument("--phase2g-sweep", action="store_true")
    parser.add_argument("--phase2g-budgets", default="0.010,0.020,0.030,0.050")
    parser.add_argument(
        "--phase2g-candidates",
        default="high_actual_bit_node_prune,high_actual_bit_surface_safe_prune,high_actual_bit_blue_noise_prune,high_actual_bit_decimation_prune,high_actual_bit_prune_then_add_surface_repair,bit_budgeted_block_soft_prune",
    )
    parser.add_argument("--phase2g-top-pool", type=int, default=8192)
    parser.add_argument("--phase2g-repair-fraction", type=float, default=0.10)
    parser.add_argument("--phase2h-sweep", action="store_true")
    parser.add_argument("--phase2i-sweep", action="store_true")
    parser.add_argument("--phase2j-sweep", action="store_true")
    parser.add_argument("--phase2h-budgets", default="0.020,0.030,0.050")
    parser.add_argument("--phase2h-pools", default="32768,65536")
    parser.add_argument(
        "--phase2h-candidates",
        default="high_bit_raw_prune,high_bit_rate_first_quality_veto,high_bit_parent_block_cap_prune,high_bit_depth_balanced_prune,high_bit_prune_then_light_repair",
    )
    parser.add_argument("--phase2h-quality-weights", default="0.0,0.5,1.0,2.0")
    parser.add_argument("--phase2h-repair-ratios", default="0.05,0.10,0.20")
    parser.add_argument("--phase2h-quality-max-points", type=int, default=12000)
    parser.add_argument("--phase2h-normal-max-points", type=int, default=12000)
    parser.add_argument("--use-pc-error", action="store_true")
    parser.add_argument("--pc-error-path", default="/home/maejima/MasterEx/compress/octree/SparsePCGC/extension/pc_error_d")
    parser.add_argument("--phase2j-budgets", default="0.020,0.030,0.050")
    parser.add_argument("--phase2j-pools", default="131072")
    parser.add_argument("--phase2j-candidates", default="block_only,high_bit_raw_prune")
    parser.add_argument("--phase2j-quality-max-points", type=int, default=3000)
    parser.add_argument("--phase2j-normal-max-points", type=int, default=3000)
    parser.add_argument("--phase2j-decoded-dir", default="/data/maejima/log/phase2j_decoded")
    parser.add_argument("--output-csv", required=True)
    return parser


def main() -> int:
    cli = build_parser().parse_args()
    if bool(cli.phase2j_sweep):
        return run_phase2j_sweep(cli)
    if bool(cli.phase2h_sweep) or bool(cli.phase2i_sweep):
        return run_phase2h_sweep(cli)
    if bool(cli.phase2g_sweep):
        return run_phase2g_sweep(cli)
    if bool(cli.phase2f_sweep):
        return run_phase2f_sweep(cli)
    return run_probe(cli)


if __name__ == "__main__":
    raise SystemExit(main())
