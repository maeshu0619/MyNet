import argparse
import csv
import datetime
import gc
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.network import Network
from models.utils.config.args import parse_pugan_args
from models.utils.data.dataset import PlyDirDataset, clear_ply_cache
from models.utils.io.utils_ply import read_ply, write_ply
from models.utils.pointcloud.voxel_collision import (
    compute_voxel_collision_stats_batch,
    flatten_voxel_collision_stats,
    format_voxel_collision_summary,
)
from models.utils.pointcloud.utils_repkpu import rearrange
from models.utils.pointcloud.utils_repkpu import configure_knn_backend
from models.utils.pointcloud.sparsepcgc_voxel import restore_points_from_voxel_coords
import models.network as network_module
from models.utils.testing.utils import (
    _adapt_encoder_state_dict_for_sparse_input,
    _adapt_model_state_dict_for_sparse_input,
    _adapt_state_dict_to_model_shapes,
    _downsample_input_batch,
    _run_named_inference_mode,
)
from models.utils.training.utils import summarize_point_edits as summarize_point_edits_train
from record.write import Writing

def terminal_log(message):
    print(str(message), flush=True)

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


_VOXEL_TEST_STAGES = ("input_gt", "model_output_raw", "saved_pre_write", "saved_ply")
_VOXEL_TEST_METRICS = (
    "raw_point_count",
    "finite_point_count",
    "unique_voxel_count",
    "duplicate_point_count",
    "duplicate_rate",
    "max_points_per_voxel",
    "point_reduction_rate",
)


def _voxel_collision_test_fields():
    fields = []
    for stage in _VOXEL_TEST_STAGES:
        for metric in _VOXEL_TEST_METRICS:
            fields.append(f"voxel_{stage}_{metric}")
    return fields


def _should_log_voxel_collision(args, step):
    if not bool(getattr(args, "enable_voxel_collision_log", False)):
        return False
    interval = max(int(getattr(args, "voxel_collision_log_interval", 100)), 1)
    return ((int(step) + 1) % interval) == 0


def _read_saved_ply_xyz(path):
    if not path or not os.path.exists(path):
        return None
    data = read_ply(path)
    xyz = np.vstack((data["x"], data["y"], data["z"])).astype(np.float32)
    return torch.from_numpy(xyz).unsqueeze(0)


def _collect_test_voxel_collision_stats(args, writer, step, stage_tensors):
    if not _should_log_voxel_collision(args, step):
        return {}
    voxel_size = float(getattr(args, "sparsepcgc_voxel_size", getattr(args, "octree_voxel", 1.0)))
    pos_q = int(getattr(args, "sparsepcgc_pos_quantscale", 1))
    max_points = int(getattr(args, "voxel_collision_max_points", 300000))
    first_only = bool(getattr(args, "voxel_collision_log_first_batch_only", True))
    flat = {}
    for stage in _VOXEL_TEST_STAGES:
        tensor = stage_tensors.get(stage)
        if tensor is None:
            writer.write(f"VoxelCollisionUnavailable[{stage}]: stage tensor is not available in test.py")
            continue
        with torch.no_grad():
            stats = compute_voxel_collision_stats_batch(
                tensor.detach(),
                voxel_size,
                pos_q,
                max_points=max_points,
                first_batch_only=first_only,
            )
        flat.update(flatten_voxel_collision_stats(f"voxel_{stage}", stats))
        writer.write(format_voxel_collision_summary(stage, stats))
        note = str(stats.get("sampling_note", ""))
        if note:
            writer.write(f"VoxelCollisionSampling[{stage}]: {note}")
    return flat


def _csv_fields():
    return [
        "sample_id",
        "step",
        "input_path",
        "output_path",
        "voxel_restored_output_path",
        "voxel_restored_output_status",
        "inference_mode",
        "input_points",
        "output_points",
        "add_points",
        "delete_points",
        "adjust_points",
        "add_ratio",
        "delete_ratio",
        "adjust_ratio",
        "data_loading_time",
        "preprocess_time",
        "postprocess_time",
        "model_forward_total_time",
        "total_inference_time",
        "save_time",
    ] + _voxel_collision_test_fields()


def _emit_table(title, headers, rows, writer):
    rows = [tuple(str(value) for value in row) for row in rows]
    headers = tuple(str(value) for value in headers)
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def _row(row):
        return "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"

    writer.write(title)
    writer.write(_row(headers))
    writer.write("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows:
        writer.write(_row(row))


def _write_step_table(row, writer):
    _emit_table(
        f"Step {int(row['step'])} Result",
        ["input_points", "output_points", "added", "deleted", "adjusted", "model_time_s", "total_time_s"],
        [
            (
                row["input_points"],
                row["output_points"],
                row["add_points"],
                row["delete_points"],
                row["adjust_points"],
                f"{row['model_forward_total_time']:.6f}",
                f"{row['total_inference_time']:.6f}",
            )
        ],
        writer,
    )


def _write_average_table(rows, writer):
    if not rows:
        writer.write("Average Result: no samples processed")
        return
    keys = [
        ("input_points", "input_points"),
        ("output_points", "output_points"),
        ("add_points", "added"),
        ("delete_points", "deleted"),
        ("adjust_points", "adjusted"),
        ("model_forward_total_time", "model_time_s"),
        ("total_inference_time", "total_time_s"),
    ]
    values = []
    for key, label in keys:
        arr = np.asarray([_to_float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
        values.append((label, f"{float(arr.mean()):.6f}"))
    _emit_table("Average Result", ["metric", "mean"], values, writer)


def _output_point_path(args, step, input_path):
    output_dir = Path(os.path.abspath(os.path.expanduser(args.save_ply_dir)))
    return output_dir / f"{step:04d}_Mine.ply"

def _output_voxel_restored_point_path(args, step, input_path):
    output_dir = Path(os.path.abspath(os.path.expanduser(args.save_ply_dir)))
    suffix = str(getattr(args, "voxel_restored_output_suffix", "_voxel_restored"))
    return output_dir / f"{step:04d}_Mine{suffix}.ply"


def _save_voxel_restored_output_points(args, step, input_path, model, fallback_dtype=torch.float32):
    """
    Phase7-5:
    test/inference時に、model.last_actuator_voxel_state['final_voxel_coords'] から復元した点群を別名で保存する。
    既存の _save_output_points は変更しない。
    """
    if not bool(getattr(args, "save_voxel_restored_output", False)):
        return "", "disabled"

    base_model = model.module if hasattr(model, "module") else model
    voxel_state = getattr(base_model, "last_actuator_voxel_state", None)
    require_state = bool(getattr(args, "voxel_restored_output_require_state", False))

    def _fallback(reason):
        if require_state:
            raise RuntimeError(f"VoxelRestoredOutput: {reason}")
        return "", reason

    if not isinstance(voxel_state, dict):
        return _fallback("last_actuator_voxel_state_missing")

    final_voxel_coords = voxel_state.get("final_voxel_coords", None)
    if not torch.is_tensor(final_voxel_coords):
        return _fallback("final_voxel_coords_missing")

    if final_voxel_coords.ndim != 3:
        return _fallback(f"invalid_final_voxel_coords_shape={tuple(final_voxel_coords.shape)}")

    if final_voxel_coords.shape[1] != 3 and final_voxel_coords.shape[-1] == 3:
        final_voxel_coords = final_voxel_coords.permute(0, 2, 1).contiguous()

    if final_voxel_coords.shape[1] != 3:
        return _fallback(f"invalid_final_voxel_coords_shape={tuple(final_voxel_coords.shape)}")

    final_voxel_valid_mask = voxel_state.get("final_voxel_valid_mask", None)
    coords = final_voxel_coords.detach().to(dtype=torch.long)

    if torch.is_tensor(final_voxel_valid_mask):
        valid_mask = final_voxel_valid_mask.detach().to(device=coords.device, dtype=torch.bool)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.squeeze(1)
    else:
        valid_mask = torch.ones((coords.shape[0], coords.shape[-1]), device=coords.device, dtype=torch.bool)

    if coords.shape[0] != 1:
        return _fallback(f"batch_size_not_supported_for_single_ply={coords.shape[0]}")

    coords_b = coords[0:1, :, valid_mask[0]]
    if coords_b.shape[-1] <= 0:
        return _fallback("empty_valid_final_voxel_coords")

    meta = {}
    voxel_step = voxel_state.get("voxel_step", None)
    voxel_offset = voxel_state.get("voxel_offset", None)
    if torch.is_tensor(voxel_step):
        meta["effective_qs_tensor"] = voxel_step.detach()[0:1].to(device=coords_b.device, dtype=fallback_dtype)
        meta["global_qs"] = meta["effective_qs_tensor"]
    if torch.is_tensor(voxel_offset):
        meta["global_offset_tensor"] = voxel_offset.detach()[0:1].to(device=coords_b.device, dtype=fallback_dtype)
        meta["global_offset"] = meta["global_offset_tensor"]

    restored_xyz, restore_info = restore_points_from_voxel_coords(
        coords_b,
        meta=meta if meta else None,
        args=args,
        center=bool(getattr(args, "sparsepcgc_dequantize_center", False)),
        unique=True,
        dtype=fallback_dtype,
        device=coords_b.device,
    )

    output_path = _output_voxel_restored_point_path(args, step, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xyz = restored_xyz.squeeze(0).transpose(0, 1).detach().cpu().numpy().astype(np.float32)
    ok = write_ply(str(output_path), [xyz], ["x", "y", "z"])
    if not ok:
        raise RuntimeError(f"write_ply returned False: {output_path}")

    return str(output_path), f"saved points={int(restore_info.get('restore_output_points', xyz.shape[0]))}"

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
    model_state = _load_state_payload(args.ckpt)
    checkpoint_has_encoder = any(str(key).startswith("encoder.") for key in model_state.keys())

    repkpu_ckpt = os.path.join(os.path.dirname(__file__), "repkpu_model", "ckpt-best.pth")
    if os.path.exists(repkpu_ckpt) and not checkpoint_has_encoder:
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


def _mp_start_method(args):
    method = str(getattr(args, "mp_start_method", "auto")).strip().lower()
    if method in {"", "none"}:
        method = "auto"
    if method != "auto" and method not in mp.get_all_start_methods():
        choices = ", ".join(["auto"] + mp.get_all_start_methods())
        raise ValueError(f"--mp_start_method must be one of: {choices}")
    return method


def _configure_mp_start_method(args):
    method = _mp_start_method(args)
    if method == "auto":
        return
    current = mp.get_start_method(allow_none=True)
    if current != method:
        mp.set_start_method(method, force=True)


def _build_test_loader_kwargs(args, use_cuda, writer):
    requested_workers = max(int(getattr(args, "num_workers", 0)), 0)
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": requested_workers,
        "pin_memory": bool(use_cuda and getattr(args, "pin_memory", False)),
    }
    if requested_workers <= 0:
        return loader_kwargs

    method = _mp_start_method(args)
    cuda_fork_unsafe = (
        use_cuda
        and torch.cuda.is_available()
        and torch.cuda.is_initialized()
        and method in {"auto", "fork"}
    )
    if cuda_fork_unsafe:
        loader_kwargs["num_workers"] = 0
        writer.write(
            "DataLoader workers were disabled for test.py because CUDA was already initialized "
            f"and fork workers can segfault ({requested_workers} requested). "
            "Use --mp_start_method spawn to keep worker loading enabled."
        )
        return loader_kwargs

    if method != "auto":
        loader_kwargs["multiprocessing_context"] = mp.get_context(method)
    loader_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", False))
    if bool(getattr(args, "clear_main_ply_cache_for_workers", True)):
        clear_ply_cache()
    return loader_kwargs


def _write_summary(rows, writer):
    _write_average_table(rows, writer)


def test(model, args, writer):
    model.eval()
    args.trainORtest = "test"
    # test.pyは推論・操作数ログ・PLY保存専用にする。
    args.test_compute_loss = False
    args.skip_actual_codec = True
    args.codec_eval_interval = 0
    args.use_uniform_noise = False
    args.test_compute_quality_metrics = False
    args.debug_timing = False
    args.verbose_step_logs = False
    args.log_step_time = False
    args.log_gpu_memory = False
    args._collect_structure_debug = False
    args._collect_octree_level_debug = False
    args._collect_sparsepcgc_debug = False
    if not bool(getattr(args, "_use_amp_cli_provided", False)):
        args.use_amp = False
    compress_key = str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "")
    if (
        compress_key == "sparsepcgc"
        and not bool(getattr(args, "sparsepcgc_enable_add_experiment", False))
        and not bool(getattr(args, "_add_cli_provided", False))
    ):
        writer.write("SparsePCGC Add follows args/checkpoint settings in test inference; no implicit test-only disable was applied.")

    writer.write("Test Role: inference_only")
    writer.write(f"checkpoint: {args.ckpt}")
    writer.write(f"input_dir: {args.input_dir_test}")
    writer.write(f"output_log: {args.output_log}")
    writer.write(f"save_output_points: {bool(getattr(args, 'save_test_ply', False))}")
    writer.write(f"output_dir: {args.save_ply_dir}")
    writer.write(
        "Test Operation Config: "
        f"add_enabled={bool(getattr(args, 'add', False))}, "
        f"target_add_ratio={float(getattr(args, 'target_add_ratio', 0.0)):.6f}, "
        f"max_add_ratio={float(getattr(args, 'max_add_ratio', 0.0)):.6f}, "
        f"target_drop_ratio={float(getattr(args, 'target_drop_ratio', 0.0)):.6f}, "
        f"max_drop_ratio={float(getattr(args, 'max_drop_ratio', 0.0)):.6f}, "
        f"target_move_ratio={float(getattr(args, 'target_move_ratio', 0.0)):.6f}, "
        f"max_move_ratio={float(getattr(args, 'max_move_ratio', 0.0)):.6f}"
    )
    terminal_log(f"Profile CSV: {args.output_log}")
    terminal_log(f"Save output points: {bool(getattr(args, 'save_test_ply', False))}")
    terminal_log(f"Output point directory: {args.save_ply_dir}")
    writer.write("Disabled in test.py: compression eval, shape quality metrics, detailed structure debug.")

    dataset = PlyDirDataset(args, args.input_dir_test)
    use_cuda = next(model.parameters()).is_cuda
    loader_kwargs = _build_test_loader_kwargs(args, use_cuda, writer)
    writer.write(
        "DataLoader: "
        f"files={len(dataset)}, workers={loader_kwargs['num_workers']}, "
        f"pin_memory={loader_kwargs['pin_memory']}, "
        f"ply_loader={getattr(args, 'ply_loader', 'numpy')}"
    )
    loader = DataLoader(dataset, **loader_kwargs)
    terminal_log(
        "DataLoader Ready: "
        f"files={len(dataset)}, workers={loader_kwargs['num_workers']}, "
        f"pin_memory={loader_kwargs['pin_memory']}, "
        f"ply_loader={getattr(args, 'ply_loader', 'numpy')}"
    )

    output_log = Path(os.path.abspath(os.path.expanduser(args.output_log)))
    output_log.parent.mkdir(parents=True, exist_ok=True)
    max_samples = max(int(getattr(args, "max_test_samples", 0)), 0)
    requested_mode = str(getattr(args, "test_inference_mode", "full_cloud")).strip().lower()
    if requested_mode == "auto":
        requested_mode = "full_cloud"
        writer.write(f"InferenceMode: auto resolved to {requested_mode}.")
    if requested_mode == "subtree_merge" and not bool(getattr(args, "test_allow_subtree_merge", False)):
        writer.write("InferenceMode: subtree_merge disabled; using pure full_cloud inference.")
        requested_mode = "full_cloud"

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

            total_files = len(dataset)
            if max_samples > 0:
                total_files = min(total_files, max_samples)

            terminal_log(f"Step Start: step={step + 1}/{total_files}")

            sample_start = time.perf_counter()
            data_loading_time = sample_start - fetch_start
            input_path = str(dataset.files[step])
            # ana_den6_online の固定特徴は元の GT PLY から構築するため、
            # train.py と同じ入力コンテキストを forward 前に渡す。
            args._current_input_file = str(Path(input_path).expanduser().resolve())
            # 推論はbootstrap teacherを使わず、学習後のNetwork residual主体で動かす。
            # 未設定のまま0扱いにすると初期den6 anchorへ誤って巻き戻る。
            args._den6_online_training_step_active = False
            args._global_train_step = max(
                int(getattr(args, "heuristic_guidance_anchor_steps", 0)),
                int(getattr(args, "heuristic_guidance_teacher_bootstrap_steps", 0)),
            )
            preprocess_start = time.perf_counter()
            input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
            raw_points = int(input_pcd.shape[1])
            input_pcd = _downsample_input_batch(input_pcd, args, input_path)
            if use_cuda:
                input_pcd = input_pcd.cuda(non_blocking=True)
            input_pcd = rearrange(input_pcd, "b n c -> b c n").contiguous()
            _sync_cuda(use_cuda)
            preprocess_time = time.perf_counter() - preprocess_start
            terminal_log(
                "Step Preprocess Done: "
                f"step={step + 1}/{total_files}, "
                f"points={int(input_pcd.shape[-1])}, "
                f"time={preprocess_time:.6f}s"
            )

            args._log_this_step = False
            args._collect_structure_debug = False
            args._collect_octree_level_debug = False
            args._collect_sparsepcgc_debug = False

            forward_start = time.perf_counter()
            terminal_log(
                "Step Forward Start: "
                f"step={step + 1}/{total_files}, mode={requested_mode}"
            )
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
            if (
                str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
                == "single_plan_student"
            ):
                base_model = model.module if hasattr(model, "module") else model
                contract_state = getattr(
                    base_model, "last_actuator_voxel_state", None
                )
                if not isinstance(contract_state, dict):
                    raise RuntimeError("Single-Plan推論の実行状態がない")
                expected_contract = {
                    "single_plan_network_forward_count": 1,
                    "single_plan_count": 1,
                    "single_plan_k_slot_count": 0,
                    "single_plan_critic_count": 0,
                    "single_plan_teacher_reference_count": 0,
                    "single_plan_cache_reference_count": 0,
                    "single_plan_den6_call_count": 0,
                    "single_plan_candidate_actual_encode_count": 0,
                }
                failures = [
                    "{}={}".format(name, _to_int(contract_state.get(name, -1)))
                    for name, expected in expected_contract.items()
                    if _to_int(contract_state.get(name, -1)) != expected
                ]
                mismatch = contract_state.get(
                    "single_plan_execution_count_mismatch", None
                )
                if torch.is_tensor(mismatch) and float(
                    mismatch.detach().abs().sum().cpu()
                ) != 0.0:
                    failures.append("executed_count_mismatch")
                if failures:
                    raise RuntimeError(
                        "Single-Plan cache-free推論契約違反: "
                        + ", ".join(failures)
                    )

            full_cloud_context = inference_result.get("full_octree_context", None)
            if isinstance(full_cloud_context, dict):
                global_voxel_coords = full_cloud_context.get("global_voxel_coords", None)
                if torch.is_tensor(global_voxel_coords) and global_voxel_coords.ndim == 3:
                    try:
                        voxel_count_value = int(global_voxel_coords.shape[-1])
                    except Exception:
                        voxel_count_value = 0
                    setattr(args, "_full_cloud_canonical_coords_count", voxel_count_value)
                    setattr(args, "_full_cloud_input_voxel_count", voxel_count_value)
                    setattr(args, "_full_cloud_voxel_count", voxel_count_value)
            terminal_log(
                "Step Forward Done: "
                f"step={step + 1}/{total_files}, "
                f"mode={inference_result.get('mode', requested_mode)}, "
                f"time={model_forward_total_time:.6f}s"
            )

            post_start = time.perf_counter()
            gen_pts = inference_result["gen_pts"]
            final_w = inference_result["final_w"]
            pre_harden_gen_pts = gen_pts
            keep_mask = None
            hardening_info = {
                "mode": "disabled",
                "threshold": float(getattr(args, "test_drop_threshold", 0.5)),
                "keep_count": None if gen_pts is None else int(gen_pts.shape[-1]),
                "total_count": None if gen_pts is None else int(gen_pts.shape[-1]),
            }
            if bool(getattr(args, "test_apply_post_hardening", False)):
                from models.utils.testing.utils import _compute_drop_hardening

                keep_mask, hardening_info = _compute_drop_hardening(final_w, args)
                if final_w is not None:
                    keep_count = hardening_info["keep_count"]
                    total_count = hardening_info["total_count"]
                    if 0 < keep_count < total_count:
                        gen_pts = gen_pts[:, :, keep_mask].contiguous()

            edit_stats = summarize_point_edits_train(
                input_xyz=input_pcd[:, :3, :],
                gen_pts=gen_pts,
                final_w=final_w,
                args=args,
                edit_ref_xyz=inference_result.get("edit_ref_xyz"),
            )
            if isinstance(edit_stats, dict):
                edit_stats["test_post_hardening_applied"] = bool(
                    getattr(args, "test_apply_post_hardening", False)
                )
                edit_stats["test_post_hardening_mode"] = str(hardening_info.get("mode", "disabled"))
            postprocess_time = time.perf_counter() - post_start
            terminal_log(
                "Step Postprocess Done: "
                f"step={step + 1}/{total_files}, "
                f"output_points={int(gen_pts.shape[-1])}, "
                f"time={postprocess_time:.6f}s"
            )

            save_start = time.perf_counter()
            output_path = _save_output_points(args, step, input_path, gen_pts)
            voxel_restored_output_path = ""
            voxel_restored_output_status = "disabled"
            if bool(getattr(args, "save_voxel_restored_output", False)):
                voxel_restored_output_path, voxel_restored_output_status = _save_voxel_restored_output_points(
                    args,
                    step,
                    input_path,
                    model,
                    fallback_dtype=gen_pts.dtype,
                )
                writer.write(
                    "VoxelRestoredOutput: "
                    f"path={voxel_restored_output_path or 'none'}, "
                    f"status={voxel_restored_output_status}"
                )
            save_time = time.perf_counter() - save_start
            total_inference_time = time.perf_counter() - sample_start
            saved_ply_xyz = None
            if output_path and bool(getattr(args, "enable_voxel_collision_log", False)):
                try:
                    saved_ply_xyz = _read_saved_ply_xyz(output_path)
                except Exception as exc:
                    writer.write(f"VoxelCollisionUnavailable[saved_ply]: failed to read saved PLY: {exc}")

            input_points = int(input_pcd.shape[-1])
            output_points = int(gen_pts.shape[-1])
            add_points = _to_int(edit_stats.get("added_points", 0))
            delete_points = _to_int(edit_stats.get("deleted_points", 0))
            adjust_points = _to_int(edit_stats.get("adjusted_points", 0))

            row = {
                "sample_id": step,
                "step": step + 1,
                "input_path": input_path,
                "output_path": output_path,
                "voxel_restored_output_path": voxel_restored_output_path,
                "voxel_restored_output_status": voxel_restored_output_status,
                "inference_mode": inference_result.get("mode", requested_mode),
                "input_points": input_points,
                "output_points": output_points,
                "add_points": add_points,
                "delete_points": delete_points,
                "adjust_points": adjust_points,
                "add_ratio": _ratio(add_points, input_points),
                "delete_ratio": _ratio(delete_points, input_points),
                "adjust_ratio": _ratio(adjust_points, input_points),
                "data_loading_time": data_loading_time,
                "preprocess_time": preprocess_time,
                "postprocess_time": postprocess_time,
                "model_forward_total_time": model_forward_total_time,
                "total_inference_time": total_inference_time,
                "save_time": save_time,
            }
            row.update(
                _collect_test_voxel_collision_stats(
                    args,
                    writer,
                    step,
                    {
                        "input_gt": input_pcd[:, :3, :],
                        "model_output_raw": pre_harden_gen_pts[:, :3, :],
                        "saved_pre_write": gen_pts[:, :3, :],
                        "saved_ply": saved_ply_xyz,
                    },
                )
            )
            csv_writer.writerow({key: row.get(key, None) for key in _csv_fields()})
            handle.flush()
            rows.append({key: row[key] for key in row})

            writer.write(
                "InferenceStep: "
                f"step={step + 1}, input={input_points}, output={output_points}, "
                f"add={add_points}, delete={delete_points}, adjust={adjust_points}, "
                f"model={model_forward_total_time:.6f}s, total={total_inference_time:.6f}s"
            )
            _write_step_table(row, writer)
            total_files = len(dataset)
            if max_samples > 0:
                total_files = min(total_files, max_samples)
            shown_step = step + 1

            terminal_log(
                "Progress: "
                f"step={shown_step}/{total_files}, "
                f"sample_id={step}, "
                f"input={input_points}, output={output_points}, "
                f"add={add_points}, delete={delete_points}, adjust={adjust_points}, "
                f"forward={row['model_forward_total_time']:.6f}s, "
                f"preprocess={row['preprocess_time']:.6f}s, "
                f"postprocess={row['postprocess_time']:.6f}s, "
                f"save={row['save_time']:.6f}s, "
                f"total={total_inference_time:.6f}s"
            )
            if raw_points != input_points:
                writer.write(f"InputDownsample: {raw_points} -> {input_points}")

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
    compress_key = str(getattr(args, "compress", "")).strip().lower().replace(
        "-", ""
    ).replace("_", "")
    if (
        compress_key == "sparsepcgc"
        and bool(getattr(args, "test_force_single_plan_student", True))
    ):
        # 推論ではTeacher/den6/cacheへ到達できないmodeを明示的に選ぶ。
        # checkpoint内Studentだけで未知入力から1 planを決定する。
        args.heuristic_guidance_mode = "single_plan_student"
        args.heuristic_guidance_enabled = False
        args.heuristic_guidance_network_only_inference = True
        args.single_plan_training_stage = "actual_calibration"
    _configure_mp_start_method(args)

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
    runtime_knn_backend = configure_knn_backend(args, writer=writer)
    network_module.KNN_BACKEND = runtime_knn_backend
    writer.write(f"Date of Testing: {file_day}-{file_time}")
    writer.write(f"Checkpoint Path: {args.ckpt}")
    writer.write(f"Profile CSV: {args.output_log}")

    terminal_log("=== Test Setup Start ===")
    terminal_log(f"Date of Testing: {file_day}-{file_time}")
    terminal_log(f"Checkpoint Path: {args.ckpt}")
    terminal_log(f"Profile CSV: {args.output_log}")

    model_load_start = time.perf_counter()
    model = _load_trained_model(args, writer)
    model_load_time = time.perf_counter() - model_load_start
    base_model = model.module if hasattr(model, "module") else model
    if (
        str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
        == "single_plan_student"
        and bool(getattr(args, "test_require_trained_single_plan_student", True))
    ):
        update_count = int(
            getattr(
                base_model,
                "single_plan_distillation_updates",
                torch.zeros((), dtype=torch.long),
            ).detach().cpu()
        )
        if update_count <= 0:
            raise RuntimeError(
                "checkpointのSingle-Plan Student蒸留更新回数が0である。"
                "未学習Studentを高速推論結果として使用できない"
            )
        actual_update_count = int(
            getattr(
                base_model,
                "single_plan_actual_training_updates",
                torch.zeros((), dtype=torch.long),
            ).detach().cpu()
        )
        if (
            bool(getattr(
                args,
                "test_require_actual_calibrated_single_plan_student",
                True,
            ))
            and actual_update_count <= 0
        ):
            raise RuntimeError(
                "このcheckpointはHeuristic planのActualだけで保存されており、"
                "Single-Plan Student自身のActual訓練履歴が0である。"
                "train.pyをsingle_plan_student modeで再開してから推論すること。"
                "診断目的で旧checkpointを実行する場合だけ"
                "--test_require_actual_calibrated_single_plan_student falseを指定できる"
            )
        writer.write(
            f"SinglePlanInferenceContract: distillation_updates={update_count}, "
            f"actual_training_updates={actual_update_count}, "
            "teacher=0, cache=0, den6=0, actual_candidate=0"
        )

    writer.write("Model checkpoint loaded. model.eval() is active.")
    terminal_log(f"=== Setup Complete === model_load_time={model_load_time:.3f}s")

    start = time.perf_counter()
    writer.write("=== Start Inference Profiling ===")
    terminal_log("=== Start Inference Profiling ===")

    test(model, args, writer)
    elapsed = time.perf_counter() - start

    finish_date = datetime.datetime.now().strftime("%Y/%m/%d - %H:%M:%S")
    writer.write(f"Testing time: {elapsed}")
    writer.write(f"Date of finishing testing: {finish_date}")

    terminal_log(f"=== Testing Finished === elapsed={elapsed:.3f}s")
    terminal_log(f"Date of finishing testing: {finish_date}")

    writer.close()
