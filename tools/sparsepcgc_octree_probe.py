#!/usr/bin/env python
"""Probe SparsePCGC actual bits under simple octree/voxel transforms.

The goal is not to train the network.  This script measures which structural
distributions are actually cheaper for SparsePCGC on real training clouds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

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


def _unique_coords(xyz_n3: torch.Tensor, args):
    coords_b3n, meta = quantize_sparsepcgc_coords(
        xyz_n3.transpose(0, 1).contiguous().unsqueeze(0),
        args=args,
        return_metadata=True,
    )
    coords = coords_b3n[0].transpose(0, 1).contiguous().to(dtype=torch.long)
    coords = torch.unique(coords, dim=0, sorted=True)
    return coords, meta


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


def _neighbor_count(coords_n3: torch.Tensor) -> torch.Tensor:
    if coords_n3.numel() <= 0:
        return torch.empty((0,), device=coords_n3.device, dtype=torch.long)
    coord_set = {tuple(int(v) for v in row) for row in coords_n3.detach().cpu().tolist()}
    out = []
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for row in coords_n3.detach().cpu().tolist():
        x, y, z = (int(row[0]), int(row[1]), int(row[2]))
        out.append(sum((x + dx, y + dy, z + dz) in coord_set for dx, dy, dz in offsets))
    return torch.as_tensor(out, device=coords_n3.device, dtype=torch.long)


def _parent_child_popcount(coords_n3: torch.Tensor, block: int = 2):
    parent = torch.div(coords_n3, int(block), rounding_mode="floor")
    unique_parent, inverse = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=int(unique_parent.shape[0])).to(device=coords_n3.device)
    return counts[inverse], unique_parent, inverse, counts


def _occupied_key_membership(query_n3: torch.Tensor, occupied_n3: torch.Tensor) -> torch.Tensor:
    combined = torch.cat([query_n3, occupied_n3], dim=0)
    mins = combined.amin(dim=0)
    span = (combined.amax(dim=0) - mins + 1).clamp_min(1)

    def _keys(values):
        shifted = values - mins
        return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]

    occupied_keys = torch.unique(_keys(occupied_n3), sorted=True)
    query_keys = _keys(query_n3)
    pos = torch.searchsorted(occupied_keys, query_keys)
    in_bounds = pos < occupied_keys.numel()
    safe_pos = pos.clamp(max=max(int(occupied_keys.numel()) - 1, 0))
    return in_bounds & (occupied_keys[safe_pos] == query_keys)


def _hole_fill_candidates(coords_n3: torch.Tensor, max_add: int = 10000):
    if coords_n3.numel() <= 0:
        return []
    offsets = torch.tensor(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        device=coords_n3.device,
        dtype=torch.long,
    )
    query = (coords_n3[:, None, :] + offsets.view(1, -1, 3)).reshape(-1, 3)
    unique_query, inverse = torch.unique(query, dim=0, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=int(unique_query.shape[0])).to(device=coords_n3.device)
    occupied = _occupied_key_membership(unique_query, coords_n3)
    empty = unique_query[~occupied]
    empty_counts = counts[~occupied]
    if int(empty.shape[0]) <= 0:
        return []
    order = torch.argsort(empty_counts, descending=True)
    rows = []
    n = int(coords_n3.shape[0])
    for ratio in (0.0025, 0.005, 0.01, 0.02):
        add_count = min(max(1, int(math.ceil(n * ratio))), int(max_add), int(empty.shape[0]))
        add_coords = empty.index_select(0, order[:add_count])
        cand = torch.unique(torch.cat([coords_n3, add_coords], dim=0), dim=0, sorted=True)
        rows.append((f"fill_holes_top_{ratio:.4f}", cand))
    return rows


def transform_candidates(coords_n3: torch.Tensor, *, max_keep: int = 50000):
    coords_n3 = torch.unique(coords_n3.to(dtype=torch.long), dim=0, sorted=True)
    n = int(coords_n3.shape[0])
    candidates: list[tuple[str, torch.Tensor]] = [("original", coords_n3)]
    if n <= 8:
        return candidates

    neigh = _neighbor_count(coords_n3)
    parent_pop, unique_parent, inverse_parent, parent_counts = _parent_child_popcount(coords_n3, block=2)

    def _append(name: str, mask: torch.Tensor):
        mask = mask.to(device=coords_n3.device, dtype=torch.bool)
        if bool(mask.any().detach().cpu()) and int(mask.sum().item()) < n:
            cand = torch.unique(coords_n3[mask], dim=0, sorted=True)
            if 0 < int(cand.shape[0]) <= max_keep:
                candidates.append((name, cand))

    # Low-density/noise pruning: isolated and thin components are often expensive.
    for threshold in (1, 2, 3, 4, 5):
        _append(f"prune_neighbor_lt_{threshold}", neigh >= threshold)

    # Drop high-detail leaves inside sparse parents; approximates parent collapse.
    for max_child_pop in (1, 2, 3):
        _append(f"drop_parent_pop_le_{max_child_pop}", parent_pop > max_child_pop)

    # Keep dense parents only, increasingly aggressive.
    for min_child_pop in (2, 4, 6):
        keep_parent = parent_counts >= min_child_pop
        _append(f"keep_parent_pop_ge_{min_child_pop}", keep_parent[inverse_parent])

    for name, cand in _hole_fill_candidates(coords_n3, max_add=max(1, int(max_keep) - n) if max_keep > n else 10000):
        if 0 < int(cand.shape[0]) <= max_keep:
            candidates.append((name, cand))

    # Snap to coarser phases, then unique.  This tests whether smoother occupancy
    # lattices reduce MP-POV entropy enough to justify geometry loss.
    for block in (2, 4):
        snapped = torch.div(coords_n3, block, rounding_mode="floor") * block
        snapped = torch.unique(snapped, dim=0, sorted=True)
        if 0 < int(snapped.shape[0]) <= max_keep:
            candidates.append((f"snap_block_{block}", snapped))

    # Parent representative collapse: one voxel per coarse parent.
    for block in (2, 4, 8):
        parent = torch.div(coords_n3, block, rounding_mode="floor")
        unique_parent = torch.unique(parent, dim=0, sorted=True)
        collapsed = unique_parent * block
        if 0 < int(collapsed.shape[0]) <= max_keep:
            candidates.append((f"collapse_parent_block_{block}", collapsed))

    # Deterministic rate-distortion style top density keep.
    density_score = neigh.to(dtype=torch.float32) + parent_pop.to(dtype=torch.float32) * 0.5
    order = torch.argsort(density_score, descending=True)
    for keep_ratio in (0.95, 0.90, 0.80, 0.70, 0.50):
        keep_n = max(1, int(math.ceil(n * keep_ratio)))
        keep_idx = order[:keep_n]
        cand = torch.unique(coords_n3.index_select(0, keep_idx), dim=0, sorted=True)
        if 0 < int(cand.shape[0]) <= max_keep:
            candidates.append((f"keep_density_top_{keep_ratio:.2f}", cand))

    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--max-evals-per-file", type=int, default=18)
    parser.add_argument("--max-input-voxels", type=int, default=6000)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--blocks-per-file", type=int, default=0)
    parser.add_argument("--min-block-voxels", type=int, default=256)
    parser.add_argument("--max-block-voxels", type=int, default=4096)
    cli = parser.parse_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        file_day = time.strftime("%Y%m%d")
        file_time = time.strftime("%H%M%S")
        args = parse_pugan_args(argparse.ArgumentParser(description="SparsePCGC probe args"), file_day, file_time)
    finally:
        sys.argv = old_argv
    args.compress = "SparsePCGC"
    args.compression_loss_backend = "sparsepcgc_surrogate"
    args.sparsepcgc_skip_decode = True
    args.sparsepcgc_worker_gpu_stats = False
    args.enable_sparsepcgc_exact_occupancy_teacher = False
    encoder = build_actual_encoder(args)

    try:
        for file_path in cli.files:
            xyz = torch.as_tensor(load_ply(file_path, return_color=False), dtype=torch.float32)
            coords, meta = _unique_coords(xyz, args)
            if int(coords.shape[0]) > cli.max_input_voxels:
                step = max(1, int(math.ceil(int(coords.shape[0]) / float(cli.max_input_voxels))))
                coords = coords[::step].contiguous()
            coord_sets = [("full", coords)]
            if int(cli.block_size) > 0 and int(cli.blocks_per_file) > 0:
                block = int(cli.block_size)
                block_coords = torch.div(coords, block, rounding_mode="floor")
                unique_blocks, inverse = torch.unique(block_coords, dim=0, sorted=True, return_inverse=True)
                counts = torch.bincount(inverse, minlength=int(unique_blocks.shape[0]))
                order = torch.argsort(counts, descending=True)
                added = 0
                for block_idx in order.detach().cpu().tolist():
                    count = int(counts[int(block_idx)].detach().cpu())
                    if count < int(cli.min_block_voxels) or count > int(cli.max_block_voxels):
                        continue
                    mask = inverse == int(block_idx)
                    block_set = torch.unique(coords[mask], dim=0, sorted=True)
                    label = "block_" + "_".join(str(int(v)) for v in unique_blocks[int(block_idx)].detach().cpu().tolist())
                    coord_sets.append((label, block_set))
                    added += 1
                    if added >= int(cli.blocks_per_file):
                        break

            for scope, scoped_coords in coord_sets:
                rows = []
                for name, cand_coords in transform_candidates(scoped_coords, max_keep=cli.max_input_voxels):
                    if len(rows) >= int(cli.max_evals_per_file):
                        break
                    cand_xyz = _coords_to_xyz(cand_coords, meta, args)
                    stats = encoder.encode_bits(cand_xyz)
                    rows.append(
                        {
                            "op": name,
                            "voxels": int(cand_coords.shape[0]),
                            "bits": float(stats.get("bit", 0.0)),
                            "bpp": float(stats.get("bpp", 0.0)),
                            "node": float(stats.get("node", 0.0)),
                            "single": float(stats.get("single", 0.0)),
                        }
                    )
                if not rows:
                    continue
                base_bits = rows[0]["bits"]
                base_voxels = rows[0]["voxels"]
                for row in rows:
                    row["delta_bits"] = row["bits"] - base_bits
                    row["delta_percent"] = 100.0 * (row["bits"] - base_bits) / max(base_bits, 1.0)
                    row["voxel_delta_percent"] = 100.0 * (row["voxels"] - base_voxels) / max(base_voxels, 1)
                best = sorted(rows[1:], key=lambda item: item["delta_percent"])[:5]
                payload = {
                    "file": str(file_path),
                    "scope": scope,
                    "base": rows[0],
                    "best": best,
                    "all": rows,
                }
                print(json.dumps(payload, sort_keys=True), flush=True)
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
