#!/usr/bin/env python
"""Context-aware SparsePCGC Where probe.

This is a research-only script. It does not import train.py or modify the
training policy. It builds fixed-ratio prune candidates from input PLY files,
encodes them with the SparsePCGC actual encoder, and writes CSV rows with
codec/raw-bit and octree-context diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.utils.config.args import parse_pugan_args
from models.utils.data.dataset import load_ply
from models.utils.loss.actual_encoder import build_actual_encoder
from models.utils.pointcloud.sparsepcgc_voxel import (
    quantize_sparsepcgc_coords,
    restore_points_from_voxel_coords,
)


def _parse_csv_floats(text: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(text).split(",") if item.strip())


def _parse_csv_text(text: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in str(text).split(",") if item.strip())


def _unique_coords(xyz_n3: torch.Tensor, args):
    coords_b3n, meta = quantize_sparsepcgc_coords(
        xyz_n3.transpose(0, 1).contiguous().unsqueeze(0),
        args=args,
        return_metadata=True,
    )
    coords = coords_b3n[0].transpose(0, 1).contiguous().to(dtype=torch.long)
    return torch.unique(coords, dim=0, sorted=True), meta


def _coords_to_xyz(coords_n3: torch.Tensor, meta, args):
    xyz, _info = restore_points_from_voxel_coords(
        coords_n3.transpose(0, 1).contiguous().unsqueeze(0),
        meta=meta,
        args=args,
        unique=True,
        dtype=torch.float32,
        device=coords_n3.device,
    )
    return xyz[0].contiguous()


def _hist(values: torch.Tensor, max_key: int | None = None) -> str:
    if values.numel() <= 0:
        return "{}"
    values = values.detach().to("cpu", dtype=torch.long).reshape(-1)
    if max_key is None:
        max_key = int(values.max().item()) if values.numel() else 0
    counts = torch.bincount(values.clamp_min(0), minlength=max_key + 1)
    payload = {str(idx): int(counts[idx].item()) for idx in range(int(counts.numel())) if int(counts[idx].item()) > 0}
    return json.dumps(payload, sort_keys=True)


def _block_info(coords: torch.Tensor, block_size: int):
    block = torch.div(coords, int(block_size), rounding_mode="floor")
    unique, inverse = torch.unique(block, dim=0, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=int(unique.shape[0])).to(device=coords.device)
    return unique, inverse, counts


def _parent_info(coords: torch.Tensor, parent_block: int = 2):
    parent = torch.div(coords, int(parent_block), rounding_mode="floor")
    unique, inverse = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    slots = (
        (coords[:, 0] & 1)
        + 2 * (coords[:, 1] & 1)
        + 4 * (coords[:, 2] & 1)
    ).to(dtype=torch.long)
    occ = torch.zeros((unique.shape[0], 8), device=coords.device, dtype=torch.bool)
    occ[inverse, slots] = True
    patterns = (occ.to(dtype=torch.long) * (2 ** torch.arange(8, device=coords.device, dtype=torch.long))).sum(dim=1)
    pop = occ.sum(dim=1).to(dtype=torch.long)
    return unique, inverse, slots, occ, patterns, pop


def _neighbor_count(coords: torch.Tensor) -> torch.Tensor:
    if coords.numel() <= 0:
        return torch.empty((0,), device=coords.device, dtype=torch.long)
    mins = coords.amin(dim=0) - 1
    span = (coords.amax(dim=0) - mins + 2).clamp_min(1)

    def keys(values):
        shifted = values - mins
        return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]

    occupied = torch.unique(keys(coords), sorted=True)
    offsets = torch.tensor(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        device=coords.device,
        dtype=torch.long,
    )
    counts = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.long)
    for offset in offsets:
        query = keys(coords + offset)
        pos = torch.searchsorted(occupied, query)
        in_bounds = pos < occupied.numel()
        safe = pos.clamp(max=max(int(occupied.numel()) - 1, 0))
        counts += (in_bounds & (occupied[safe] == query)).to(dtype=torch.long)
    return counts


def _quota_select(
    scores: torch.Tensor,
    group_inverse: torch.Tensor,
    target_count: int,
    *,
    group_counts: torch.Tensor,
    quota_fraction: float,
    round_robin: bool = True,
) -> torch.Tensor:
    n = int(scores.numel())
    target_count = min(max(int(target_count), 0), n)
    out = torch.zeros((n,), device=scores.device, dtype=torch.bool)
    if target_count <= 0 or n <= 0:
        return out
    quota = torch.ceil(group_counts.to(dtype=torch.float32) * float(quota_fraction)).to(dtype=torch.long).clamp_min(1)
    if not round_robin:
        adjusted = scores - group_inverse.to(dtype=torch.float32) * 0.0
        order = torch.argsort(adjusted, descending=True)
        picked_per_group = torch.zeros_like(group_counts)
        picked = []
        for idx in order.detach().cpu().tolist():
            g = int(group_inverse[idx].item())
            if int(picked_per_group[g].item()) >= int(quota[g].item()):
                continue
            picked.append(idx)
            picked_per_group[g] += 1
            if len(picked) >= target_count:
                break
        if picked:
            out[torch.tensor(picked, device=scores.device, dtype=torch.long)] = True
        return out

    group_order = torch.argsort(group_counts, descending=False).detach().cpu().tolist()
    per_group_indices: Dict[int, List[int]] = {}
    for g in group_order:
        idx = (group_inverse == int(g)).nonzero(as_tuple=False).reshape(-1)
        if idx.numel() <= 0:
            continue
        idx = idx.index_select(0, torch.argsort(scores.index_select(0, idx), descending=True))
        per_group_indices[int(g)] = idx.detach().cpu().tolist()
    picked_per_group = {int(g): 0 for g in per_group_indices}
    picked_total = 0
    while picked_total < target_count:
        changed = False
        for g in group_order:
            g = int(g)
            rows = per_group_indices.get(g, [])
            pos = picked_per_group.get(g, 0)
            if pos >= len(rows) or pos >= int(quota[g].item()):
                continue
            out[int(rows[pos])] = True
            picked_per_group[g] = pos + 1
            picked_total += 1
            changed = True
            if picked_total >= target_count:
                break
        if not changed:
            break
    return out


def _block_only_drop(coords: torch.Tensor, target: int, block_size: int) -> torch.Tensor:
    unique, inverse, counts = _block_info(coords, block_size)
    order = torch.argsort(counts, descending=False)
    drop_blocks = torch.zeros((unique.shape[0],), device=coords.device, dtype=torch.bool)
    dropped = 0
    for b in order.detach().cpu().tolist():
        if dropped >= target:
            break
        drop_blocks[int(b)] = True
        dropped += int(counts[int(b)].item())
    return drop_blocks.index_select(0, inverse)


def _random_scattered_drop(coords: torch.Tensor, target: int, seed: int) -> torch.Tensor:
    if target <= 0:
        return torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    # Deterministic integer hash; avoids non-reproducible torch random state.
    vals = coords[:, 0] * 73856093 + coords[:, 1] * 19349663 + coords[:, 2] * 83492791 + int(seed)
    order = torch.argsort(vals.remainder(2147483647), descending=True)
    out = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    out[order[:target]] = True
    return out


def _micro_codec_prior_drop(coords: torch.Tensor, target: int, block_size: int, quota_fraction: float) -> torch.Tensor:
    _unique, inverse, counts = _block_info(coords, block_size)
    # Sparse blocks receive high codec prior.
    score = 1.0 / counts.index_select(0, inverse).to(dtype=torch.float32).clamp_min(1.0)
    return _quota_select(score, inverse, target, group_counts=counts, quota_fraction=quota_fraction, round_robin=True)


def _parent_emptying_drop(
    coords: torch.Tensor,
    target: int,
    block_size: int,
    max_parent_pop: int,
    quota_fraction: float | None = None,
    geometry_guard: bool = False,
) -> torch.Tensor:
    unique_parent, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    unique_block, inverse_block, block_counts = _block_info(coords, block_size)
    parent_block = torch.div(unique_parent * 2, int(block_size), rounding_mode="floor")
    # Map parent blocks to block index.
    block_lookup = {tuple(row.detach().cpu().tolist()): i for i, row in enumerate(unique_block)}
    parent_block_idx = torch.tensor(
        [block_lookup.get(tuple(row.detach().cpu().tolist()), 0) for row in parent_block],
        device=coords.device,
        dtype=torch.long,
    )
    neigh = _neighbor_count(coords).to(dtype=torch.float32)
    parent_neigh = torch.zeros((unique_parent.shape[0],), device=coords.device, dtype=torch.float32)
    parent_neigh.scatter_add_(0, inverse_parent, neigh)
    parent_neigh = parent_neigh / parent_pop.to(dtype=torch.float32).clamp_min(1.0)
    block_sparse = 1.0 / block_counts.index_select(0, parent_block_idx).to(dtype=torch.float32).clamp_min(1.0)
    score = 3.0 / parent_pop.to(dtype=torch.float32).clamp_min(1.0) + block_sparse
    if geometry_guard:
        score = score - 0.20 * parent_neigh
    valid = parent_pop <= int(max_parent_pop)
    order = torch.argsort(torch.where(valid, score, score.new_full(score.shape, -1e9)), descending=True)
    drop_parent = torch.zeros((unique_parent.shape[0],), device=coords.device, dtype=torch.bool)
    per_block_drop = torch.zeros((unique_block.shape[0],), device=coords.device, dtype=torch.long)
    dropped = 0
    for p in order.detach().cpu().tolist():
        p = int(p)
        if not bool(valid[p].item()) or dropped >= target:
            break
        pop = int(parent_pop[p].item())
        if pop <= 0 or dropped + pop > max(target, 1) * 1.15:
            continue
        b = int(parent_block_idx[p].item())
        if quota_fraction is not None:
            quota = max(1, int(math.ceil(int(block_counts[b].item()) * float(quota_fraction))))
            if int(per_block_drop[b].item()) + pop > quota:
                continue
        drop_parent[p] = True
        per_block_drop[b] += pop
        dropped += pop
    return drop_parent.index_select(0, inverse_parent)


def _sibling_simplify_drop(coords: torch.Tensor, target: int) -> torch.Tensor:
    unique_parent, inverse_parent, slots, occ, patterns, parent_pop = _parent_info(coords)
    pattern_freq = torch.bincount(patterns, minlength=256).to(device=coords.device, dtype=torch.float32)
    score = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.float32)
    for idx in range(coords.shape[0]):
        p = int(inverse_parent[idx].item())
        pop = int(parent_pop[p].item())
        if pop <= 1:
            score[idx] = 3.0
            continue
        cur_pattern = int(patterns[p].item())
        slot = int(slots[idx].item())
        new_pattern = cur_pattern & ~(1 << slot)
        gain = float(pattern_freq[new_pattern].item() - pattern_freq[cur_pattern].item())
        # Prefer low-pop simplification and avoid dense-parent partial edits.
        score[idx] = gain / max(float(pattern_freq[cur_pattern].item()), 1.0) + 1.0 / float(pop)
        if pop >= 6:
            score[idx] -= 1.0
    order = torch.argsort(score, descending=True)
    out = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.bool)
    out[order[:target]] = True
    return out


def build_drop_mask(method: str, coords: torch.Tensor, ratio: float, *, block_size: int, seed: int) -> torch.Tensor:
    n = int(coords.shape[0])
    target = min(max(int(math.ceil(n * float(ratio))), 0), max(n - 1, 0))
    if target <= 0 or method == "noop":
        return torch.zeros((n,), device=coords.device, dtype=torch.bool)
    if method == "block_only":
        return _block_only_drop(coords, target, block_size)
    if method == "random_scattered_micro":
        return _random_scattered_drop(coords, target, seed)
    if method == "macro_micro_codec_prior":
        return _micro_codec_prior_drop(coords, target, block_size, quota_fraction=0.10)
    if method == "parent_emptying_prune":
        return _parent_emptying_drop(coords, target, block_size, max_parent_pop=3, quota_fraction=None)
    if method == "parent_emptying_with_geometry_guard":
        return _parent_emptying_drop(coords, target, block_size, max_parent_pop=3, quota_fraction=None, geometry_guard=True)
    if method == "sibling_pattern_simplify_prune":
        return _sibling_simplify_drop(coords, target)
    if method == "codec_prior_grouped_prune":
        return _parent_emptying_drop(coords, target, block_size, max_parent_pop=4, quota_fraction=0.20)
    if method == "codec_context_hybrid_prune":
        return _parent_emptying_drop(coords, target, block_size, max_parent_pop=4, quota_fraction=0.15, geometry_guard=True)
    if method == "grouped_micro_with_limited_scatter":
        return _parent_emptying_drop(coords, target, block_size, max_parent_pop=3, quota_fraction=0.10)
    raise ValueError(f"unknown method: {method}")


def context_metrics(coords: torch.Tensor, drop_mask: torch.Tensor, *, block_size: int) -> Mapping[str, object]:
    n = int(coords.shape[0])
    drop_count = int(drop_mask.sum().item())
    if drop_count <= 0:
        return {
            "drop_count": 0,
            "actual_drop_ratio": 0.0,
            "selected_block_count": 0,
            "max_drop_count_per_block": 0,
            "mean_drop_count_per_selected_block": 0.0,
            "drop_concentration_top1": 0.0,
            "drop_concentration_top5": 0.0,
            "drop_block_entropy": 0.0,
            "parent_emptying_count": 0,
            "parent_emptying_ratio": 0.0,
            "partial_context_damage_count": 0,
            "partial_context_damage_ratio": 0.0,
            "parent_occupancy_before_hist": "{}",
            "group_size_hist": "{}",
            "drop_neighbor_mean": 0.0,
        }
    _unique_block, inverse_block, _block_counts = _block_info(coords, block_size)
    drop_block_counts = torch.bincount(inverse_block[drop_mask], minlength=int(_unique_block.shape[0]))
    nonzero_block = drop_block_counts[drop_block_counts > 0]
    sorted_counts = torch.sort(nonzero_block, descending=True).values
    probs = sorted_counts.to(dtype=torch.float32) / float(drop_count)
    entropy = float((-(probs * torch.log2(probs.clamp_min(1e-12))).sum()).item()) if probs.numel() else 0.0

    _up, inverse_parent, _slots, _occ, _patterns, parent_pop = _parent_info(coords)
    drop_parent_counts = torch.bincount(inverse_parent[drop_mask], minlength=int(parent_pop.shape[0]))
    affected = drop_parent_counts > 0
    emptied = affected & (drop_parent_counts >= parent_pop)
    partial = affected & (drop_parent_counts < parent_pop)
    emptied_voxels = int(drop_parent_counts[emptied].sum().item())
    partial_voxels = int(drop_parent_counts[partial].sum().item())
    neigh = _neighbor_count(coords)
    drop_neigh = neigh[drop_mask].to(dtype=torch.float32)
    return {
        "drop_count": drop_count,
        "actual_drop_ratio": float(drop_count) / max(float(n), 1.0),
        "selected_block_count": int(nonzero_block.numel()),
        "max_drop_count_per_block": int(sorted_counts[0].item()) if sorted_counts.numel() else 0,
        "mean_drop_count_per_selected_block": float(nonzero_block.to(dtype=torch.float32).mean().item()) if nonzero_block.numel() else 0.0,
        "drop_concentration_top1": float(sorted_counts[:1].sum().item()) / float(drop_count),
        "drop_concentration_top5": float(sorted_counts[:5].sum().item()) / float(drop_count),
        "drop_block_entropy": entropy,
        "parent_emptying_count": int(emptied.sum().item()),
        "parent_emptying_ratio": float(emptied_voxels) / float(drop_count),
        "partial_context_damage_count": int(partial.sum().item()),
        "partial_context_damage_ratio": float(partial_voxels) / float(drop_count),
        "parent_occupancy_before_hist": _hist(parent_pop.index_select(0, inverse_parent[drop_mask]), max_key=8),
        "group_size_hist": _hist(drop_parent_counts[affected], max_key=8),
        "drop_neighbor_mean": float(drop_neigh.mean().item()) if drop_neigh.numel() else 0.0,
    }


def geometry_proxy(coords: torch.Tensor, keep_mask: torch.Tensor, drop_mask: torch.Tensor, max_samples: int) -> float:
    if int(drop_mask.sum().item()) <= 0 or int(keep_mask.sum().item()) <= 0:
        return 0.0
    dropped = coords[drop_mask].to(dtype=torch.float32)
    kept = coords[keep_mask].to(dtype=torch.float32)
    if dropped.shape[0] > max_samples:
        dropped = dropped[torch.linspace(0, dropped.shape[0] - 1, max_samples, device=coords.device).long()]
    if kept.shape[0] > max_samples:
        kept = kept[torch.linspace(0, kept.shape[0] - 1, max_samples, device=coords.device).long()]
    dists = torch.cdist(dropped, kept, p=2)
    return float(dists.min(dim=1).values.mean().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--amounts", default="0.005,0.010,0.020,0.030,0.040,0.050")
    parser.add_argument(
        "--methods",
        default=(
            "noop,block_only,macro_micro_codec_prior,random_scattered_micro,"
            "parent_emptying_prune,sibling_pattern_simplify_prune,codec_prior_grouped_prune,"
            "parent_emptying_with_geometry_guard,codec_context_hybrid_prune,grouped_micro_with_limited_scatter"
        ),
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-geometry-samples", type=int, default=2048)
    parser.add_argument("--enable-exact-occupancy", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    cli = parser.parse_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = parse_pugan_args(
            argparse.ArgumentParser(description="context-aware SparsePCGC Where probe"),
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
    args.enable_sparsepcgc_exact_occupancy_teacher = bool(cli.enable_exact_occupancy)
    args.sparsepcgc_exact_occupancy_interval = 1

    amounts = _parse_csv_floats(cli.amounts)
    methods = _parse_csv_text(cli.methods)
    rows: List[Mapping[str, object]] = []
    encoder = build_actual_encoder(args)
    try:
        for file_idx, file_path in enumerate(cli.files):
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            sequence = Path(file_path).parent.name
            base_xyz = _coords_to_xyz(coords, meta, args)
            base_stats = encoder.encode_bits(base_xyz)
            base_bits = float(base_stats.get("bit", 0.0))
            for amount in amounts:
                for method in methods:
                    drop_mask = build_drop_mask(
                        method,
                        coords,
                        amount,
                        block_size=int(cli.block_size),
                        seed=int(cli.seed) + file_idx,
                    )
                    keep_mask = ~drop_mask
                    if int(keep_mask.sum().item()) <= 0:
                        continue
                    cand_coords = torch.unique(coords[keep_mask], dim=0, sorted=True)
                    cand_xyz = _coords_to_xyz(cand_coords, meta, args)
                    stats = encoder.encode_bits(cand_xyz)
                    bit = float(stats.get("bit", 0.0))
                    metrics = dict(context_metrics(coords, drop_mask, block_size=int(cli.block_size)))
                    geom = geometry_proxy(coords, keep_mask, drop_mask, max_samples=int(cli.max_geometry_samples))
                    row = {
                        "file": str(file_path),
                        "sequence": sequence,
                        "method": method,
                        "target_amount": float(amount),
                        "input_voxels": int(coords.shape[0]),
                        "candidate_voxels": int(cand_coords.shape[0]),
                        "base_bit": base_bits,
                        "raw_bit": bit,
                        "actual_raw_percent": 100.0 * (bit - base_bits) / max(base_bits, 1.0),
                        "geometry_missing_nn_mean": geom,
                        "sparsepcgc_estimated_occupancy_bits": stats.get("sparsepcgc_estimated_occupancy_bits", ""),
                        "sparsepcgc_pred_occupancy_nll": stats.get("sparsepcgc_pred_occupancy_nll", ""),
                        "sparsepcgc_prob_true_mean": stats.get("sparsepcgc_prob_true_mean", ""),
                        "sparsepcgc_prob_true_low_ratio": stats.get("sparsepcgc_prob_true_low_ratio", ""),
                        "sparsepcgc_exact_estimated_bits": stats.get("sparsepcgc_exact_estimated_bits", ""),
                        "sparsepcgc_exact_occupancy_nll": stats.get("sparsepcgc_exact_occupancy_nll", ""),
                        "sparsepcgc_exact_prob_true_mean": stats.get("sparsepcgc_exact_prob_true_mean", ""),
                        "sparsepcgc_exact_low_prob_ratio": stats.get("sparsepcgc_exact_low_prob_ratio", ""),
                    }
                    row.update(metrics)
                    rows.append(row)
                    Path(cli.output_csv).parent.mkdir(parents=True, exist_ok=True)
                    with open(cli.output_csv, "w", newline="", encoding="utf-8") as f:
                        fieldnames = list(rows[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
