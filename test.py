import argparse
import csv
import datetime
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.network import Network
from models.utils.config.args import parse_pugan_args
from models.utils.data.dataset import PlyDirDataset
from models.utils.io.utils_ply import write_ply
from models.utils.pointcloud.utils_repkpu import rearrange
from models.utils.testing.utils import (
    _adapt_encoder_state_dict_for_sparse_input,
    _adapt_model_state_dict_for_sparse_input,
    _adapt_state_dict_to_model_shapes,
    _aggregate_structure_debug_chunks,
    _compute_drop_hardening,
    _downsample_input_batch,
    _run_named_inference_mode,
    _summarize_hardening_counts,
    _summarize_point_edits,
    _write_structure_decision_debug,
)
from record.write import Writing


def _sync_cuda(use_cuda):
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _to_float(value, default=0.0):
    try:
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=0):
    try:
        if torch.is_tensor(value):
            return int(round(float(value.detach().cpu())))
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _ratio(count, denom):
    return float(count) / max(float(denom), 1.0)


def _csv_fields():
    return [
        "sample_id",
        "input_path",
        "output_path",
        "inference_mode",
        "input_points",
        "output_points",
        "add_points",
        "delete_points",
        "adjust_points",
        "add_ratio",
        "delete_ratio",
        "adjust_ratio",
        "preserve_ratio",
        "same_voxel_adjust_count",
        "different_voxel_move_count",
        "delete_target_voxel_count",
        "add_target_voxel_count",
        "move_source_voxel_count",
        "move_target_voxel_count",
        "data_loading_time",
        "preprocess_time",
        "feature_time",
        "attribution_time",
        "structure_diagnosis_time",
        "point_edit_decision_time",
        "delete_time",
        "add_time",
        "adjust_time",
        "postprocess_time",
        "model_forward_total_time",
        "total_inference_time",
        "save_time",
    ]


def _output_point_path(args, step, input_path):
    output_dir = Path(os.path.abspath(os.path.expanduser(args.save_ply_dir)))
    stem = Path(input_path).stem
    return output_dir / f"{step:04d}_{stem}_edited.ply"


def _save_output_points(args, step, input_path, gen_pts):
    if not bool(getattr(args, "save_test_ply", False)):
        return ""
    output_path = _output_point_path(args, step, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = gen_pts.squeeze(0).transpose(0, 1).detach().cpu().numpy()
    xyz = out[:, :3].astype(np.float32)
    field_list = [xyz]
    field_names = ["x", "y", "z"]
    if out.shape[1] >= 6:
        rgb = np.clip(out[:, 3:6] * 255.0, 0, 255).astype(np.uint8)
        field_list.append(rgb)
        field_names.extend(["red", "green", "blue"])
    ok = write_ply(str(output_path), field_list, field_names)
    if not ok:
        raise RuntimeError(f"write_ply returned False: {output_path}")
    return str(output_path)


def _runtime_value(runtime_timing, *names):
    for name in names:
        if name in runtime_timing:
            return _to_float(runtime_timing[name])
    return 0.0


def _load_state_payload(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            state = payload.get(key)
            if isinstance(state, dict):
                return state
    return payload


def _load_trained_model(args, writer):
    model = Network(args, writer)

    repkpu_ckpt = os.path.join(os.path.dirname(__file__), "repkpu_model", "ckpt-best.pth")
    if os.path.exists(repkpu_ckpt):
        ckpt = torch.load(repkpu_ckpt, map_location="cpu")
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in ckpt.items()
            if k.startswith("encoder.")
        }
        encoder_state = _adapt_encoder_state_dict_for_sparse_input(model, encoder_state, writer=writer)
        encoder_state = _adapt_state_dict_to_model_shapes(
            encoder_state,
            model.encoder.state_dict(),
            writer=writer,
            label="RepKPU encoder",
        )
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.load_state_dict(encoder_state, strict=False)
        del ckpt, encoder_state

    model_state = _load_state_payload(args.ckpt)
    model_state = _adapt_model_state_dict_for_sparse_input(model, model_state, writer=writer)
    model_state = _adapt_state_dict_to_model_shapes(
        model_state,
        model.state_dict(),
        writer=writer,
        label="Model checkpoint",
    )
    model.load_state_dict(model_state, strict=False)
    del model_state
    gc.collect()

    if not bool(getattr(args, "cpu", False)) and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def _write_summary(rows, writer):
    if not rows:
        writer.write("InferenceProfileSummary: no samples processed")
        return
    numeric_keys = [
        "input_points",
        "output_points",
        "add_points",
        "delete_points",
        "adjust_points",
        "add_ratio",
        "delete_ratio",
        "adjust_ratio",
        "preserve_ratio",
        "data_loading_time",
        "preprocess_time",
        "feature_time",
        "attribution_time",
        "structure_diagnosis_time",
        "point_edit_decision_time",
        "delete_time",
        "add_time",
        "adjust_time",
        "postprocess_time",
        "model_forward_total_time",
        "total_inference_time",
        "save_time",
    ]
    writer.write("=== Inference Profile Summary ===")
    for key in numeric_keys:
        values = np.asarray([_to_float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
        writer.write(
            f"{key}: mean={float(values.mean()):.6g}, "
            f"min={float(values.min()):.6g}, max={float(values.max()):.6g}"
        )


def test(model, args, writer):
    model.eval()
    args.trainORtest = "test"
    # test.pyは推論profile専用にする。圧縮/品質評価は別スクリプトで実行する。
    args.test_compute_loss = False
    args.skip_actual_codec = True
    args.codec_eval_interval = 0
    args.use_uniform_noise = False
    args.debug_timing = True

    writer.write("Test Role: inference_only_profile")
    writer.write(f"checkpoint: {args.ckpt}")
    writer.write(f"input_dir: {args.input_dir_test}")
    writer.write(f"output_log: {args.output_log}")
    writer.write(f"save_output_points: {bool(getattr(args, 'save_test_ply', False))}")
    writer.write(f"output_dir: {args.save_ply_dir}")
    writer.write(
        "Disabled in test.py: compression codec eval, actual compression delta, "
        "before/after bits, BD-rate, D1/D2, Chamfer, point-to-plane, codec temp files."
    )

    dataset = PlyDirDataset(args, args.input_dir_test)
    use_cuda = next(model.parameters()).is_cuda
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": max(int(getattr(args, "num_workers", 0)), 0),
        "pin_memory": bool(use_cuda and getattr(args, "pin_memory", False)),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", False))
    loader = DataLoader(dataset, **loader_kwargs)

    output_log = Path(os.path.abspath(os.path.expanduser(args.output_log)))
    output_log.parent.mkdir(parents=True, exist_ok=True)
    max_samples = max(int(getattr(args, "max_test_samples", 0)), 0)
    requested_mode = str(getattr(args, "test_inference_mode", "full_cloud")).strip().lower()
    if requested_mode == "auto":
        requested_mode = "full_cloud"
        writer.write("InferenceMode: auto is treated as full_cloud to avoid benchmark-style double forward.")

    use_amp = bool(use_cuda and getattr(args, "use_amp", False))
    amp_dtype = torch.float16
    rows = []
    total_start = time.perf_counter()
    fetch_start = time.perf_counter()

    with output_log.open("w", newline="") as handle, torch.inference_mode():
        csv_writer = csv.DictWriter(handle, fieldnames=_csv_fields())
        csv_writer.writeheader()

        for step, pts in enumerate(loader):
            if max_samples > 0 and len(rows) >= max_samples:
                break

            sample_start = time.perf_counter()
            data_loading_time = sample_start - fetch_start
            input_path = str(dataset.files[step])
            preprocess_start = time.perf_counter()
            input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
            raw_points = int(input_pcd.shape[1])
            input_pcd = _downsample_input_batch(input_pcd, args, input_path)
            if use_cuda:
                input_pcd = input_pcd.cuda(non_blocking=True)
            input_pcd = rearrange(input_pcd, "b n c -> b c n").contiguous()
            _sync_cuda(use_cuda)
            preprocess_time = time.perf_counter() - preprocess_start

            profile_this_sample = bool(
                getattr(args, "profile_test", True)
                and (
                    step == 0
                    or (step + 1) % max(int(getattr(args, "profile_interval", 100)), 1) == 0
                )
            )
            args._log_this_step = profile_this_sample
            args._collect_sparsepcgc_debug = False

            forward_start = time.perf_counter()
            inference_result = _run_named_inference_mode(
                requested_mode,
                model,
                input_pcd,
                args,
                input_path,
                use_cuda,
                use_amp,
                amp_dtype,
                writer=writer,
            )
            _sync_cuda(use_cuda)
            model_forward_total_time = time.perf_counter() - forward_start

            post_start = time.perf_counter()
            gen_pts = inference_result["gen_pts"]
            final_w = inference_result["final_w"]
            pre_harden_gen_pts = gen_pts
            keep_mask, hardening_info = _compute_drop_hardening(final_w, args)
            if final_w is not None:
                keep_count = hardening_info["keep_count"]
                total_count = hardening_info["total_count"]
                if 0 < keep_count < total_count:
                    gen_pts = gen_pts[:, :, keep_mask].contiguous()

            edit_stats = _summarize_point_edits(
                input_xyz=input_pcd[:, :3, :],
                pre_harden_gen_pts=pre_harden_gen_pts,
                final_gen_pts=gen_pts,
                edit_ref_xyz=inference_result.get("edit_ref_xyz"),
                keep_mask=keep_mask,
                args=args,
            )
            base_model = model.module if hasattr(model, "module") else model
            structure_debug = _aggregate_structure_debug_chunks(
                inference_result.get("structure_debug_chunks", [])
            ) or getattr(base_model, "last_structure_debug", {}) or {}
            runtime_timing = getattr(base_model, "last_runtime_timing", {}) or {}
            hardening_counts = _summarize_hardening_counts(
                input_points=int(input_pcd.shape[-1]),
                pre_output_points=int(pre_harden_gen_pts.shape[-1]),
                keep_mask=keep_mask,
            )
            postprocess_time = time.perf_counter() - post_start

            save_start = time.perf_counter()
            output_path = _save_output_points(args, step, input_path, gen_pts)
            save_time = time.perf_counter() - save_start
            total_inference_time = time.perf_counter() - sample_start

            input_points = int(input_pcd.shape[-1])
            output_points = int(gen_pts.shape[-1])
            add_points = _to_int(
                structure_debug.get("add_actual_point_count", edit_stats.get("added_points", 0))
            )
            delete_points = _to_int(
                structure_debug.get("delete_removed_point_count", edit_stats.get("deleted_points", 0))
            )
            adjust_points = _to_int(edit_stats.get("adjusted_points", 0))
            preserve_ratio = _to_float(
                structure_debug.get(
                    "preserve_ratio",
                    max(0.0, 1.0 - _ratio(add_points + delete_points + adjust_points, input_points)),
                )
            )

            row = {
                "sample_id": step,
                "input_path": input_path,
                "output_path": output_path,
                "inference_mode": inference_result.get("mode", requested_mode),
                "input_points": input_points,
                "output_points": output_points,
                "add_points": add_points,
                "delete_points": delete_points,
                "adjust_points": adjust_points,
                "add_ratio": _ratio(add_points, input_points),
                "delete_ratio": _ratio(delete_points, input_points),
                "adjust_ratio": _ratio(adjust_points, input_points),
                "preserve_ratio": preserve_ratio,
                "same_voxel_adjust_count": _to_int(structure_debug.get("same_voxel_adjust_count", 0)),
                "different_voxel_move_count": _to_int(structure_debug.get("moved_different_voxel_count", 0)),
                "delete_target_voxel_count": _to_int(structure_debug.get("delete_target_voxel_count", 0)),
                "add_target_voxel_count": _to_int(structure_debug.get("add_target_voxel_count", 0)),
                "move_source_voxel_count": _to_int(structure_debug.get("move_source_voxel_count", 0)),
                "move_target_voxel_count": _to_int(structure_debug.get("move_target_voxel_count", 0)),
                "data_loading_time": data_loading_time,
                "preprocess_time": preprocess_time,
                "feature_time": _runtime_value(runtime_timing, "feature_extraction", "encode"),
                "attribution_time": _runtime_value(runtime_timing, "codec_cost_attribution"),
                "structure_diagnosis_time": _runtime_value(runtime_timing, "structure_diagnosis"),
                "point_edit_decision_time": _runtime_value(runtime_timing, "point_edit_decision"),
                "delete_time": _runtime_value(runtime_timing, "delete_module"),
                "add_time": _runtime_value(runtime_timing, "add_module"),
                "adjust_time": _runtime_value(runtime_timing, "adjust_move_module"),
                "postprocess_time": postprocess_time + _runtime_value(runtime_timing, "postprocess"),
                "model_forward_total_time": _runtime_value(
                    runtime_timing,
                    "total_forward",
                    "model_forward_total_time",
                ) or model_forward_total_time,
                "total_inference_time": total_inference_time,
                "save_time": save_time,
            }
            csv_writer.writerow(row)
            handle.flush()
            rows.append({key: row[key] for key in row})

            writer.write(
                "InferenceProfile: "
                f"sample={step}, input={input_points}, output={output_points}, "
                f"add={add_points}, delete={delete_points}, adjust={adjust_points}, "
                f"preserve_ratio={preserve_ratio:.6f}, total={total_inference_time:.6f}s"
            )
            if raw_points != input_points:
                writer.write(f"InputDownsample: {raw_points} -> {input_points}")
            if profile_this_sample:
                writer.write(
                    "ModuleTiming: "
                    f"feature={row['feature_time']:.6f}s, "
                    f"attribution={row['attribution_time']:.6f}s, "
                    f"diagnosis={row['structure_diagnosis_time']:.6f}s, "
                    f"decision={row['point_edit_decision_time']:.6f}s, "
                    f"delete={row['delete_time']:.6f}s, "
                    f"add={row['add_time']:.6f}s, "
                    f"adjust={row['adjust_time']:.6f}s, "
                    f"postprocess={row['postprocess_time']:.6f}s"
                )
                _write_structure_decision_debug(
                    writer,
                    f"TestStructureDecision sample={step}",
                    structure_debug,
                )
                writer.write(
                    "HardeningStats: "
                    f"mode={hardening_info.get('mode', 'none')}, "
                    f"kept_original={hardening_counts['kept_original']}, "
                    f"deleted_original={hardening_counts['deleted_original']}, "
                    f"kept_added={hardening_counts['kept_added']}, "
                    f"deleted_added={hardening_counts['deleted_added']}"
                )

            fetch_start = time.perf_counter()

    _write_summary(rows, writer)
    writer.write(
        f"TotalInferenceTiming: samples={len(rows)}, "
        f"total={time.perf_counter() - total_start:.6f}s"
    )
    writer.write(f"InferenceProfileCSV: {output_log}")
    if bool(getattr(args, "save_test_ply", False)):
        writer.write(f"OutputPointDir: {args.save_ply_dir}")


if __name__ == "__main__":
    file_day = datetime.datetime.now().strftime("%Y%m%d")
    file_time = datetime.datetime.now().strftime("%H%M%S")

    parser = argparse.ArgumentParser(description="Inference profiling arguments")
    parser.add_argument("--trainORtest", default="test", type=str, help="run mode")
    args = parse_pugan_args(parser, file_day, file_time)
    args.trainORtest = "test"

    if torch.cuda.is_available() and not args.cpu and args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    writer = Writing(
        args,
        file_day,
        file_time,
        filename="MyNetwork_test",
        flush_every=args.log_flush_every,
        sync_every=args.log_sync_every,
        log_root=args.log_root,
    )
    writer.write(f"Date of Testing: {file_day}-{file_time}")
    writer.write(f"Checkpoint Path: {args.ckpt}")
    writer.write(f"Profile CSV: {args.output_log}")

    model = _load_trained_model(args, writer)
    writer.write("Model checkpoint loaded. model.eval() is active.")

    start = time.perf_counter()
    writer.write("=== Start Inference Profiling ===")
    test(model, args, writer)
    elapsed = time.perf_counter() - start

    finish_date = datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S")
    writer.write(f"Testing time: {elapsed}")
    writer.write(f"Date of finishing testing: {finish_date}")
    writer.close()
