#!/usr/bin/env python3
"""Actual encode/decode + MPEG D1/D2 + BD-rate. Edit defaults below or use CLI.

Example (holdout GT and its matching unit-normal PLY are explicitly selected):
  python -B test_BDrate.py --input /data/holdout/loot.ply \
      --normals /data/holdout/loot_normals.ply --resolution 1023 \
      --codec G-PCC --network-codec SparsePCGC --checkpoint /path/model.pth

Use --identity-control for a codec/metric self-test, NOT trained MINE evidence.
See reports/20261019_bdrate_evaluation.md for protocol and limitations.
"""
from pathlib import Path
import argparse
import json
import sys

# User-editable experiment settings; existing args.py is not modified.
model_date = "20260801"
model_time = "072958"
model_name = "best_loss_joint"
checkpoint = ""  # empty: resolve using existing args.py checkpoint convention
compression = "SparsePCGC"       # codec used for BOTH RD curves
network_compression = "SparsePCGC"  # architecture/config under which MINE was trained
network_dataset = "8i"  # Original test.py dataset family (8i / MVUB / UVG)
dataset_name = "explicit_holdout"
point_clouds = []  # Explicit held-out PLY paths. Never silently select training data.
normals_files = []  # Same order/coordinates as GT; omit only with d1_only=True.
geometry_resolution = None  # e.g. 1023 for a confirmed 10-bit geometry dataset
d1_only = False
rate_points = None  # None => codec's documented six-point preset; or list of dicts
coarse_point_indices = None  # Dense run: select six indices for interpolation audit
output_directory = ""  # empty: existing log_root/YYYYMMDD/MyNetwork_test_BDrate/codec/run
network_extra_args = []  # Original test.py args required to reproduce trained design


def parse_args(argv=None):
    from utils.bdrate.codecs import ROOT, default_points, validate_points
    from models.utils.config.args import _LOG_ROOT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", default=point_clouds, help="Explicit GT files (no recursive implicit dataset)")
    p.add_argument("--normals", nargs="*", default=normals_files, help="One unit-normal PLY per GT, in same order")
    p.add_argument("--resolution", type=float, default=geometry_resolution, help="MPEG intrinsic resolution, not guessed from bounding box")
    p.add_argument("--d1-only", action="store_true", default=d1_only)
    p.add_argument("--codec", choices=("G-PCC", "Draco", "OctAttention", "SparsePCGC"), default=compression)
    p.add_argument("--network-codec", default=network_compression, choices=("G-PCC", "Draco", "OctAttention", "SparsePCGC"))
    p.add_argument("--network-dataset", default=network_dataset, choices=("8i", "MVUB", "UVG"))
    p.add_argument("--model-date", default=model_date)
    p.add_argument("--model-time", default=model_time)
    p.add_argument("--model-name", default=model_name)
    p.add_argument("--checkpoint", default=checkpoint)
    p.add_argument("--dataset", default=dataset_name)
    p.add_argument("--points-json", help="JSON file containing explicit RD setting dicts; supports 10-15 points")
    p.add_argument("--coarse-indices", type=int, nargs="+", default=coarse_point_indices, help="Six zero-based dense-point indices")
    p.add_argument("--output", default=output_directory)
    p.add_argument("--log-root", default=str(_LOG_ROOT))
    p.add_argument("--minimum-overlap-db", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--codec-device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--network-python", default=sys.executable)
    p.add_argument("--oa-python", default=sys.executable)
    p.add_argument("--sparse-python", default="/home/maejima/miniconda3/envs/sparsepcgc/bin/python")
    p.add_argument("--identity-control", action="store_true", help="No Network; explicitly labelled identity control")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--gpcc-binary", default=str(ROOT / "G-PCC/build/tmc3/tmc3"))
    p.add_argument("--gpcc-config", default=str(ROOT / "G-PCC/cfg/octree-predlift/lossy-geom-lossy-attrs/longdress_vox10_1300/r01/encoder.cfg"))
    p.add_argument("--draco-encoder", default=str(ROOT / "Draco/build/draco_encoder"))
    p.add_argument("--draco-decoder", default=str(ROOT / "Draco/build/draco_decoder"))
    p.add_argument("--pc-error", default=str(ROOT / "SparsePCGC/extension/pc_error_d"))
    p.add_argument("--oa-checkpoint", default=str(ROOT / "OctAttention/modelsave/obj/encoder_epoch_00800093.pth"))
    p.add_argument("--sp-checkpoint", default=str(ROOT / "SparsePCGC/ckpts/dense/epoch_last.pth"))
    p.add_argument("--sp-ae-checkpoint", default=str(ROOT / "SparsePCGC/ckpts/dense_slne/epoch_last.pth"))
    p.add_argument("--sp-sr-checkpoint", default=str(ROOT / "SparsePCGC/ckpts/dense_1stage/epoch_last.pth"))
    # Everything after -- is forwarded to the original test.py argument parser.
    tokens = list(sys.argv[1:] if argv is None else argv)
    if "--" in tokens:
        split = tokens.index("--")
        tokens, extra = tokens[:split], tokens[split + 1:]
    else:
        extra = []
    args = p.parse_args(tokens)
    if args.checkpoint:
        import re
        match = re.search(r"/(\d{8})/(?:MyNetwork_train|Mynetwork_train)/(?:pretrained/)?(\d{6})_[^/]+/([^/]+)\.pth$", str(Path(args.checkpoint).expanduser().resolve()))
        if match:
            # Explicit checkpoint paths take precedence over the example dates;
            # never accidentally label an October checkpoint as the August one.
            args.model_date, args.model_time, args.model_name = match.groups()
    args.network_extra_args = list(network_extra_args) + extra
    if not args.input:
        p.error("Explicit held-out --input files are required")
    if args.resolution is None or not 0 < args.resolution < float("inf"):
        p.error("An explicit positive --resolution is required")
    if args.normals and len(args.normals) != len(args.input):
        p.error("--normals must have exactly one path per GT")
    if args.d1_only and args.normals:
        p.error("Choose D1-only OR explicit D2 normals, not both")
    if args.timeout <= 0 or args.threads < 1 or args.minimum_overlap_db < 0:
        p.error("Invalid timeout/threads/minimum overlap")
    args.points = json.loads(Path(args.points_json).read_text()) if args.points_json else (rate_points or default_points(args.codec))
    validate_points(args.codec, args.points)
    if args.coarse_indices is None:
        import numpy as np
        args.coarse_indices = np.linspace(0, len(args.points) - 1, min(6, len(args.points))).round().astype(int).tolist()
    if sorted(set(args.coarse_indices)) != args.coarse_indices or any(i < 0 or i >= len(args.points) for i in args.coarse_indices) or len(args.coarse_indices) < 4:
        p.error("Invalid coarse indices (need >=4 unique sorted in-range indices)")
    return args


def main(argv=None):
    from utils.bdrate.runner import run
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
