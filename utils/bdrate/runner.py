"""Streaming evaluation orchestration and durable per-point failure records."""
import csv
import datetime
import json
import os
import platform
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np
from .codecs import REPO, encode_decode
from .io import json_safe, read_vertices, run_command, save_json, sha256
from .metrics import evaluate, validate_normals
from .report import plot_curves, summarize, write_report


def _network_output(args, gt, folder, checkpoint):
    folder.mkdir()
    # PlyDirDataset requires .../<dataset>/<sequence>. Preserve that convention
    # for explicitly selected arbitrary GT locations, without scanning other data.
    inputs = folder / args.network_dataset / "eval"
    inputs.mkdir(parents=True)
    (inputs / gt.name).symlink_to(gt)
    # Do not let forwarded switches override the scientific identity/protected IO.
    reserved = {"--ckpt", "--checkpoint", "--input_dir_test", "--data_dir", "--save_ply_dir", "--save_test_ply", "--save_output_points",
                "--output_log", "--log_root", "--max_test_samples", "--compress", "--compression", "--trainORtest",
                "--test_inference_mode", "--use_uniform_noise", "--use_amp", "--num_workers", "--use_tf32", "--dataname"}
    if any(token.split("=")[0] in reserved for token in args.network_extra_args):
        raise ValueError("Forwarded test.py args override a protected inference/evaluation setting")
    switches = args.network_extra_args + [
        "--trainORtest", "test", "--checkpoint", checkpoint, "--compression", args.network_codec,
        "--dataname", args.network_dataset,
        "--input_dir_test", str(inputs), "--save_ply_dir", str(folder / "mine"), "--save_test_ply", "true",
        "--output_log", str(folder / "profile.csv"), "--log_root", str(folder / "network_log"),
        "--max_test_samples", "1", "--test_inference_mode", "full_cloud", "--num_workers", "0",
        "--use_amp", "false", "--use_tf32", "false", "--use_uniform_noise", "false"]
    save_json(folder / "request.json", dict(folder=str(folder), original_gt=str(gt), network_args=switches, threads=args.threads))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    run_command([args.network_python, str(Path(__file__).with_name("inference_worker.py")), str(folder / "request.json")],
                folder, "inference", args.timeout, cwd=REPO, env=env)
    with (folder / "profile.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one original test.py inference result")
    mine = Path(rows[0]["output_path"]).resolve()
    if not mine.is_file() or folder not in mine.parents:
        raise RuntimeError("Invalid test.py output path")
    return mine


def _versions():
    from importlib.metadata import version, PackageNotFoundError
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform())
    for package in ("numpy", "scipy", "openpyxl", "matplotlib", "torch", "plyfile"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def run(args):
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    run_name = now.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:8]
    folder = (Path(args.output).expanduser() if args.output else
              Path(args.log_root).expanduser() / now.strftime("%Y%m%d") / "MyNetwork_test_BDrate" / args.codec / run_name).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    print("BD-rate artifacts: " + str(folder), flush=True)
    clouds = [str(Path(p).expanduser().resolve()) for p in args.input]
    if len(set(clouds)) != len(clouds):
        raise ValueError("Duplicate Original GT path")
    protected = [REPO / p for p in ("train.py", "test.py", "models/network.py", "models/utils/config/args.py")]
    protected += [p for p in (REPO / "models").rglob("*.py") if p.is_file()]
    # Include the reused codec source/checkpoints/binaries, not just training code.
    codec_root = REPO.parent / "compress/octree"
    for name in ("G-PCC", "Draco", "OctAttention", "SparsePCGC"):
        protected += [p for p in (codec_root / name).rglob("*.py") if p.is_file()]
    for setting in ("gpcc_binary", "gpcc_config", "draco_encoder", "draco_decoder", "pc_error", "oa_checkpoint", "sp_checkpoint", "sp_ae_checkpoint", "sp_sr_checkpoint"):
        path = Path(getattr(args, setting))
        if path.is_file():
            protected.append(path)
    if (REPO / "repkpu_model/ckpt-best.pth").is_file():
        protected.append(REPO / "repkpu_model/ckpt-best.pth")
    before = {str(p): sha256(p) for p in protected}
    checkpoint = ""
    if not args.identity_control:
        from models.utils.config.args import _pretrained_checkpoint_path
        checkpoint = str(Path(args.checkpoint or _pretrained_checkpoint_path(
            args.model_date, args.model_time, args.network_codec, model_stem=args.model_name)).expanduser().resolve())
    metadata = dict(dataset=args.dataset, model_date=args.model_date, model_time=args.model_time, model_name=args.model_name,
                    checkpoint=checkpoint, codec=args.codec, network_codec=args.network_codec,
                    mode="IDENTITY_CONTROL_NOT_TRAINED_MINE" if args.identity_control else "trained_network",
                    created_at_jst=now.isoformat(), config=vars(args), versions=_versions(),
                    protected_source_hashes=before, source_sha256={str(p.relative_to(REPO)): sha256(p) for p in [REPO / "test_BDrate.py"] + list(Path(__file__).parent.glob("*.py"))},
                    evaluation_protocol="Original GT versus final decoded geometry; physical geometry streams including decoder side data; bpp denominator=Original GT vertex count",
                    network_rate_conditioning="one deterministic original test.py output per cloud, reused across codec rate settings",
                    caveats=["No automatic training/test split inference; explicit held-out selection is the researcher's responsibility",
                             "Custom OA/SP serialized wrapper overhead differs from native paper conventions; see codec_metadata",
                             "MPEG executable textual precision retained; no extra pre-integration rounding"])
    rows = []
    with (folder / "run.log").open("x", buffering=1) as log, (folder / "rd_points.jsonl").open("x", buffering=1) as journal:
        def emit(row):
            rows.append(row)
            journal.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush(); os.fsync(journal.fileno())
            message = "{} {} point {}: {}".format(Path(row["cloud"]).name, row["series"], row["rate_index"], row["status"])
            print(message, flush=True); log.write(message + "\n")
        save_json(folder / "configuration.json", metadata)
        try:
            if not Path(args.pc_error).is_file():
                raise FileNotFoundError("MPEG metric executable: " + args.pc_error)
            metadata["pc_error_sha256"] = sha256(args.pc_error)
            if checkpoint:
                metadata["checkpoint_sha256"] = sha256(checkpoint)
            for cloud_index, cloud in enumerate(clouds):
                cloud_folder = folder / ("cloud_{:04d}".format(cloud_index)); cloud_folder.mkdir()
                gt, mine, n_original, mine_error, n_mine, normal = Path(cloud), None, None, None, None, None
                try:
                    xyz, vertices = read_vertices(gt)
                    n_original = len(xyz)
                    has_normals = all(k in vertices.dtype.names for k in ("nx", "ny", "nz"))
                    del xyz, vertices
                    normal = str(Path(args.normals[cloud_index]).expanduser().resolve()) if args.normals else (str(gt) if has_normals and not args.d1_only else None)
                    if not args.d1_only:
                        if not normal:
                            raise ValueError("D2 requires supplied GT normals; use --normals or explicitly --d1-only")
                        validate_normals(gt, normal)
                    save_json(cloud_folder / "gt_provenance.json", dict(path=str(gt), sha256=sha256(gt), original_points=n_original,
                               normals=normal, normals_sha256=sha256(normal) if normal else None))
                    if args.identity_control:
                        mine = gt
                    else:
                        mine = _network_output(args, gt, cloud_folder / "network", checkpoint)
                    n_mine = len(read_vertices(mine)[0])
                except Exception as exc:
                    mine_error = "{}: {}".format(type(exc).__name__, exc)
                    log.write(traceback.format_exc() + "\n")
                for rate_index, point in enumerate(args.points):
                    for series, source in (("Baseline", gt), ("MINE", mine)):
                        point_folder = cloud_folder / ("{:02d}_{}".format(rate_index, series)); point_folder.mkdir()
                        row = dict(cloud=cloud, dataset=args.dataset, codec=args.codec, series=series,
                                   experiment_mode=metadata["mode"], model_date=args.model_date, model_time=args.model_time,
                                   rate_index=rate_index, parameters=point, original_points=n_original, mine_points=n_mine,
                                   status="error", resolution=args.resolution)
                        try:
                            if n_original is None or (not args.d1_only and not normal):
                                raise ValueError(mine_error or "GT/normal preflight failed")
                            if series == "MINE" and mine_error:
                                raise ValueError(mine_error)
                            measured = encode_decode(args.codec, source, point_folder, point, vars(args))
                            row.update(measured, bpp=measured["bits"] / n_original,
                                       source_path=str(source), source_sha256=sha256(source))
                            row.update(evaluate(gt, measured["decoded_path"], point_folder, args.pc_error,
                                                args.resolution, normal, not args.d1_only, args.timeout))
                            row["status"] = "ok"
                        except Exception as exc:
                            row["error"] = "{}: {}".format(type(exc).__name__, exc)
                            log.write(traceback.format_exc() + "\n")
                        save_json(point_folder / "measurement.json", row)
                        emit(row)
        except Exception as exc:
            metadata["fatal_error"] = "{}: {}".format(type(exc).__name__, exc)
            log.write(traceback.format_exc() + "\n")
        finally:
            metadata["protected_source_unchanged"] = all(Path(p).is_file() and sha256(p) == digest for p, digest in before.items())
            metadata["evaluation_source_unchanged"] = all(sha256(REPO / p) == digest for p, digest in metadata["source_sha256"].items())
            if checkpoint and "checkpoint_sha256" in metadata:
                metadata["checkpoint_unchanged"] = sha256(checkpoint) == metadata["checkpoint_sha256"]
            summary, diagnostics = summarize(rows, clouds, len(args.points), args.coarse_indices, args.minimum_overlap_db)
            for item in summary:
                item.update(dataset=args.dataset, codec=args.codec, experiment_mode=metadata["mode"],
                            model_name=args.model_name if checkpoint else None,
                            model_date=args.model_date if checkpoint else None,
                            model_time=args.model_time if checkpoint else None,
                            checkpoint_sha256=metadata.get("checkpoint_sha256"))
            if not metadata["protected_source_unchanged"] or not metadata["evaluation_source_unchanged"] or metadata.get("checkpoint_unchanged") is False:
                for item in summary:
                    item.update(status="invalid", bd_rate_percent=None, error="Source/checkpoint changed during evaluation")
            write_report(folder, rows, summary, diagnostics, metadata)
            if not args.no_plots:
                try:
                    plot_curves(folder, rows, clouds)
                except Exception as exc:
                    save_json(folder / "plot_error.json", dict(error=str(exc)))
            artifacts = {str(p.relative_to(folder)): sha256(p) for p in folder.rglob("*") if p.is_file() and p.name not in ("manifest.json", "run.log", "rd_points.jsonl")}
            save_json(folder / "manifest.json", artifacts)
    print("Excel: " + str(folder / "results.xlsx"), flush=True)
    required = [r for r in summary if r.get("cloud") == "DATASET_MEAN" and (not args.d1_only or r["metric"] == "d1")]
    return 0 if not metadata.get("fatal_error") and metadata["protected_source_unchanged"] and all(r["status"] == "ok" for r in required) else 2
