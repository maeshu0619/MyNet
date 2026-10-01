#!/usr/bin/env python3
"""Measure GT versus preprocessed point-cloud bits with OctAttention and G-PCC."""

import argparse
import csv
import glob
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

MYNET_ROOT = Path(__file__).resolve().parents[1]
if str(MYNET_ROOT) not in sys.path:
    sys.path.insert(0, str(MYNET_ROOT))

from models.utils.data.dataset import load_ply
from models.utils.loss.actual_encoder import (
    _GPCCActualEncoder,
    _OctAttentionActualEncoder,
)


FIELDS = (
    "codec",
    "dataset",
    "sequence",
    "gt_path",
    "mine_path",
    "gt_points",
    "mine_points",
    "gt_bits",
    "mine_bits",
    "delta_bits",
    "delta_percent",
    "gt_codec_seconds",
    "mine_codec_seconds",
)


def _load_xyz(path):
    points = load_ply(str(path), return_color=False, loader="numpy")
    return torch.as_tensor(points, dtype=torch.float32).transpose(0, 1).contiguous()


def _pairs(profile_glob):
    pairs = []
    for profile_path in sorted(glob.glob(profile_glob)):
        name = Path(profile_path).stem
        prefix = "codec_compare_SparsePCGC_"
        if not name.startswith(prefix):
            continue
        dataset, sequence = name[len(prefix):].split("_", 1)
        with open(profile_path, newline="") as handle:
            row = next(csv.DictReader(handle))
        gt_path = Path(row["input_path"])
        mine_path = Path(row["output_path"])
        if not gt_path.is_file() or not mine_path.is_file():
            raise FileNotFoundError(f"Missing pair: GT={gt_path}, MINE={mine_path}")
        pairs.append((dataset, sequence, gt_path, mine_path))
    if not pairs:
        raise RuntimeError(f"No profiles matched: {profile_glob}")
    return pairs


def _build_encoder(codec, repo_root, qs):
    if codec == "OctAttention":
        args = SimpleNamespace(
            qs=qs,
            octattention_actualcode=True,
            octattention_tmp_dir="",
            octattention_ckpt=str(
                repo_root
                / "compress/octree/OctAttention/modelsave/obj/encoder_epoch_00800093.pth"
            ),
            octattention_teacher_device="cuda",
            bptt=1024,
        )
        return _OctAttentionActualEncoder(args)
    args = SimpleNamespace(
        qs=qs,
        gpcc_root=str(repo_root / "compress/octree/G-PCC"),
        gpcc_encoder_path=str(repo_root / "compress/octree/G-PCC/build/tmc3/tmc3"),
        gpcc_cfg_dir=str(
            repo_root
            / "compress/octree/G-PCC/cfg/octree-predlift/"
            "lossless-geom-lossless-attrs/longdress_vox10_1300"
        ),
        gpcc_tmp_dir="",
        gpcc_timeout=300.0,
        gpcc_effective_qs=qs,
        gpcc_prequantize=True,
        gpcc_disable_attribute_coding=True,
        gpcc_merge_duplicated_points=True,
    )
    return _GPCCActualEncoder(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qs", type=float, default=2.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    pairs = _pairs(args.profile_glob)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for codec in ("OctAttention", "G-PCC"):
            encoder = _build_encoder(codec, repo_root, args.qs)
            for dataset, sequence, gt_path, mine_path in pairs:
                gt = _load_xyz(gt_path)
                mine = _load_xyz(mine_path)
                if codec == "OctAttention" and torch.cuda.is_available():
                    gt = gt.cuda()
                    mine = mine.cuda()
                start = time.perf_counter()
                gt_stats = encoder.encode_bits(gt)
                gt_seconds = time.perf_counter() - start
                start = time.perf_counter()
                mine_stats = encoder.encode_bits(mine)
                mine_seconds = time.perf_counter() - start
                gt_bits = float(gt_stats["bit"])
                mine_bits = float(mine_stats["bit"])
                result = {
                    "codec": codec,
                    "dataset": dataset,
                    "sequence": sequence,
                    "gt_path": str(gt_path),
                    "mine_path": str(mine_path),
                    "gt_points": int(gt.shape[-1]),
                    "mine_points": int(mine.shape[-1]),
                    "gt_bits": gt_bits,
                    "mine_bits": mine_bits,
                    "delta_bits": mine_bits - gt_bits,
                    "delta_percent": 100.0 * (mine_bits - gt_bits) / max(gt_bits, 1.0),
                    "gt_codec_seconds": gt_seconds,
                    "mine_codec_seconds": mine_seconds,
                }
                writer.writerow(result)
                handle.flush()
                print(
                    f"{codec} {dataset}/{sequence}: "
                    f"GT={gt_bits:.0f}, MINE={mine_bits:.0f}, "
                    f"delta={result['delta_percent']:.6f}%",
                    flush=True,
                )


if __name__ == "__main__":
    main()
