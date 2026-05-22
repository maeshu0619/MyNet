import math
import os
import time
from contextlib import nullcontext
import torch
from ..training.correlation import format_corr
from ..training.correlation_debug import *
from ..training.sparsepcgc_controls import *
from ..training.compression_primary_loss import *
from ..training.case_debug import *
from ..training.metric_csv import *
from ..training.actual_codec_status import *
from ..training.metric_rows import *
from ..training.episode_metrics import *
from ..training.checkpoint_metrics import *
from ..training.scalar_utils import *
from ..training.noise_debug import *
from ..training.utils import *
from ..training.for_better_logging import (
    log_for_better_event,
    log_for_better_pretrain_complete,
    log_for_better_pretrain_step,
)
from ..pointcloud.utils_repkpu import *
from ..pointcloud.octree_subtree import *

def surrogate_param_norm(loss):
    surrogate = getattr(loss, "compression_surrogate", None)
    if surrogate is None:
        return None
    total_sq = 0.0
    with torch.no_grad():
        for param in surrogate.parameters():
            if param is None:
                continue
            try:
                total_sq += float(param.detach().float().pow(2).sum().cpu())
            except Exception:
                return None
    return math.sqrt(total_sq) if total_sq >= 0.0 else None

def infer_octree_depth_from_xyz(input_xyz, args):
    """
    input_xyz から点群全体をOctree分割したときの最大深さを推定する。
    量子化座標を想定し、最大座標スパンから ceil(log2(span + 1)) を計算する。
    失敗した場合は args 側の既存設定にフォールバックする。
    """
    fallback_candidates = [
        "octree_depth",
        "max_octree_depth",
        "max_depth",
        "depth",
        "bitdepth",
        "bit_depth",
        "coord_bit_depth",
        "train_subtree_level_max",
        "train_subtree_level_min",
    ]

    fallback_depth = None
    for name in fallback_candidates:
        value = getattr(args, name, None)
        try:
            value = int(value)
            if value > 0:
                fallback_depth = value
                break
        except (TypeError, ValueError):
            pass

    if input_xyz is None:
        return fallback_depth if fallback_depth is not None else 1

    try:
        with torch.no_grad():
            xyz = input_xyz.detach()
            if xyz.dim() == 2:
                # [3, N] or [N, 3] を想定
                if xyz.shape[0] == 3:
                    coord_min = xyz.amin(dim=1)
                    coord_max = xyz.amax(dim=1)
                else:
                    coord_min = xyz.amin(dim=0)
                    coord_max = xyz.amax(dim=0)
            elif xyz.dim() == 3:
                # [B, 3, N] を想定
                coord_min = xyz[:, :3, :].amin(dim=2)
                coord_max = xyz[:, :3, :].amax(dim=2)
            else:
                return fallback_depth if fallback_depth is not None else 1

            max_span = coord_max.sub(coord_min).amax()
            if torch.is_tensor(max_span):
                max_span = float(max_span.detach().cpu())

            if not math.isfinite(max_span) or max_span <= 0:
                return fallback_depth if fallback_depth is not None else 1

            # 座標範囲が 0〜1023 なら span=1023, span+1=1024, depth=10
            estimated_depth = int(math.ceil(math.log2(max(max_span + 1.0, 2.0))))
            estimated_depth = max(1, estimated_depth)

            # args側に明示的な最大depthがある場合は、それを超えないようにする
            if fallback_depth is not None:
                estimated_depth = min(estimated_depth, int(fallback_depth))

            return estimated_depth
    except Exception:
        return fallback_depth if fallback_depth is not None else 1



def with_pretrain_subtree_depth_overrides(args, callback, input_xyz=None):
    saved = {
        "train_subtree_level_min": getattr(args, "train_subtree_level_min", 0),
        "train_subtree_level_max": getattr(args, "train_subtree_level_max", 0),
        "train_subtree_randomize_level": getattr(args, "train_subtree_randomize_level", False),
        "train_subtree_depth_percent_curriculum": getattr(args, "train_subtree_depth_percent_curriculum", True),
        "train_subtree_depth_percent_start": getattr(args, "train_subtree_depth_percent_start", (0.0, 0.50)),
        "train_subtree_depth_percent_end": getattr(args, "train_subtree_depth_percent_end", (0.0, 0.50)),
        "_train_subtree_depth_cli_override": getattr(args, "_train_subtree_depth_cli_override", False),
    }
    try:
        if (
            int(getattr(args, "surrogate_pretrain_subtree_depth_min", -1)) > 0
            or int(getattr(args, "surrogate_pretrain_subtree_depth_max", -1)) > 0
        ):
            full_octree_depth = infer_octree_depth_from_xyz(input_xyz, args)
            depth_min = int(getattr(args, "surrogate_pretrain_subtree_depth_min", -1))
            depth_max = int(getattr(args, "surrogate_pretrain_subtree_depth_max", -1))
            if depth_min <= 0:
                depth_min = 1
            if depth_max <= 0:
                depth_max = full_octree_depth
            pct_min = float(depth_min) / float(max(full_octree_depth, 1))
            pct_max = float(depth_max) / float(max(full_octree_depth, 1))
            depth_min = max(1, min(int(depth_min), int(full_octree_depth)))
            depth_max = max(depth_min, min(int(depth_max), int(full_octree_depth)))
            if depth_min > depth_max:
                depth_min, depth_max = depth_max, depth_min
            args.train_subtree_level_min = int(depth_min)
            args.train_subtree_level_max = int(depth_max)
            args._train_subtree_depth_cli_override = True
        else:
            pct_min, pct_max = surrogate_pretrain_depth_percent_range(args)
            pct_min, pct_max = sorted((float(pct_min), float(pct_max)))
            args.train_subtree_level_min = 0
            args.train_subtree_level_max = 0
            args.train_subtree_depth_percent_curriculum = True
            args.train_subtree_depth_percent_start = (float(pct_min), float(pct_max))
            args.train_subtree_depth_percent_end = (float(pct_min), float(pct_max))
            args._train_subtree_depth_cli_override = False
            depth_min, depth_max = None, None

        # 通常trainと同じ sample_train_subtree_depth 経路から深さを選ばせる。
        if bool(getattr(args, "surrogate_pretrain_subtree_random_depth", True)):
            args.train_subtree_randomize_level = True
        else:
            args.train_subtree_randomize_level = False

        result = callback()
        if isinstance(result, dict):
            result["pretrain_depth_percent_range"] = (float(pct_min), float(pct_max))
            if depth_min is not None and depth_max is not None:
                result["pretrain_depth_absolute_range"] = (int(depth_min), int(depth_max))
            else:
                result["pretrain_depth_absolute_range"] = (
                    int(result.get("min_depth", 0)),
                    int(result.get("max_depth", 0)),
                )
        return result
    finally:
        for key, value in saved.items():
            setattr(args, key, value)

def build_surrogate_pretrain_subtree_sample(pts, args, cache_key, use_cuda, global_step):
    sample_t0 = time.perf_counter()
    input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0)
    input_pcd = downsample_input_batch(input_pcd, args, cache_key)
    if use_cuda:
        input_pcd = input_pcd.cuda(non_blocking=True)
    input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous()
    input_xyz = input_pcd[:, :3, :]
    input_attr_full = input_pcd[:, 3:, :].contiguous() if input_pcd.shape[1] > 3 else None

    def sample_depth_for_pretrain():
        return sample_train_subtree_depth(
            input_xyz,
            args,
            global_step=global_step,
            cache_key=cache_key,
        )

    subtree_depth_meta = with_pretrain_subtree_depth_overrides(
        args,
        sample_depth_for_pretrain,
        input_xyz=input_xyz,
    )
    min_subtree_points = max(int(getattr(args, "train_subtree_min_points", 1)), 1)
    requested_depth = max(int(subtree_depth_meta.get("depth", 1)), 1)
    group_state = build_octree_subtree_groups_with_retry(
        input_xyz,
        args,
        requested_depth=requested_depth,
        min_points=min_subtree_points,
        allow_largest_fallback=not bool(getattr(args, "surrogate_pretrain_skip_min_points_miss", False)),
    )
    retry_count = int(group_state.get("retry_count", 0))

    if not group_state.get("groups") and group_state.get("all_groups"):
        return {
            "skip_reason": "min_points_miss",
            "sampling_time": time.perf_counter() - sample_t0,
            "depth": requested_depth,
            "point_count": 0,
            "total_subtree_count": len(group_state.get("all_groups", [])),
            "eligible_subtree_count": 0,
            "selected_subtree_count": 0,
            "retry_count": retry_count,
        }

    if not group_state.get("groups"):
        return {
            "skip_reason": "no_valid_subtree",
            "sampling_time": time.perf_counter() - sample_t0,
            "depth": requested_depth,
            "point_count": 0,
            "total_subtree_count": 0,
            "eligible_subtree_count": 0,
            "selected_subtree_count": 0,
            "retry_count": retry_count,
        }

    subtree_ref = group_state["subtree_ref"]
    all_subtree_keys = group_state["unique_keys"]
    all_groups = group_state["all_groups"]
    group_source = group_state["groups"]
    selected_depth = int(group_state["depth"])
    skip_reason = str(group_state.get("selection_reason", "none"))
    eligible_count = int(group_state.get("eligible_count", len(group_source)))
    subtree_depth_meta = dict(subtree_depth_meta)
    subtree_depth_meta["depth"] = selected_depth
    total_subtree_count = int(all_subtree_keys.numel())

    selected_groups = group_source
    if bool(getattr(args, "surrogate_pretrain_subtree_reuse_train_sampler", True)):
        candidate_keys = all_subtree_keys.new_tensor([subtree_key for subtree_key, _ in group_source])
        selected_keys = select_octree_subtree_keys(candidate_keys, int(global_step), args)
        selected_key_set = set(selected_keys.detach().cpu().tolist())
        selected_groups = [
            (subtree_key, point_idx)
            for subtree_key, point_idx in group_source
            if subtree_key in selected_key_set
        ]
        if not selected_groups:
            selected_groups = [max(group_source, key=lambda item: int(item[1].numel()))]
            skip_reason = "sampler_empty_fallback_largest"
    else:
        selected_groups = [max(group_source, key=lambda item: int(item[1].numel()))]

    chosen_index = int(global_step) % max(len(selected_groups), 1)
    subtree_key, point_idx = selected_groups[chosen_index]
    subtree_xyz = input_xyz.index_select(2, point_idx).contiguous()
    subtree_attr = input_attr_full.index_select(2, point_idx).contiguous() if input_attr_full is not None else None
    point_count = int(subtree_xyz.shape[-1])
    if point_count <= 0:
        skip_reason = "empty_selected_subtree"

    bbox_min = subtree_xyz[:, :3, :].amin(dim=2).squeeze(0) if point_count > 0 else None
    bbox_max = subtree_xyz[:, :3, :].amax(dim=2).squeeze(0) if point_count > 0 else None
    subtree_cache_key = (
        f"{cache_key}|pretrain_subtree_depth={int(subtree_ref['depth'][0].item())}|subtree_key={int(subtree_key)}"
    )
    return {
        "input_xyz": input_xyz,
        "input_attr_full": input_attr_full,
        "subtree_xyz": subtree_xyz,
        "subtree_attr": subtree_attr,
        "subtree_ref": subtree_ref,
        "subtree_depth_meta": subtree_depth_meta,
        "subtree_cache_key": subtree_cache_key,
        "subtree_key": int(subtree_key),
        "point_count": point_count,
        "bbox_min": format_xyz_triplet(bbox_min),
        "bbox_max": format_xyz_triplet(bbox_max),
        "retry_count": retry_count,
        "skip_reason": skip_reason,
        "total_subtree_count": total_subtree_count,
        "eligible_subtree_count": int(eligible_count),
        "selected_subtree_count": int(len(selected_groups)),
        "depth": int(subtree_ref["depth"][0].item()),
        "requested_depth": int(requested_depth),
        "depth_percent_range": subtree_depth_meta.get("pretrain_depth_percent_range"),
        "depth_absolute_range": subtree_depth_meta.get("pretrain_depth_absolute_range"),
        "sampling_time": time.perf_counter() - sample_t0,
    }


def optimizer_lrs(optimizer):
    if optimizer is None:
        return []
    return [float(group.get("lr", 0.0)) for group in optimizer.param_groups]

def set_optimizer_lrs(optimizer, lrs):
    if optimizer is None:
        return
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = float(lr)


def run_surrogate_pretrain(
    *,
    model,
    args,
    loss,
    seq_datasets,
    loader_kwargs,
    metric_csv_paths,
    ckpt_dir,
    writer,
    plot=None,
    use_cuda,
    use_amp,
    amp_dtype,
    for_better_path=None,
):
    print(f"Surrogate pretrain step: {int(getattr(args, 'surrogate_step', 0))}")
    steps = max(int(getattr(args, "surrogate_step", 0)), 0)
    if steps <= 0:
        return
    backend = str(getattr(args, "compression_loss_backend", "proxy")).strip().lower()
    if not backend.endswith("_surrogate"):
        writer.write(f"SurrogatePretrain skipped: backend={backend} is not a surrogate backend.")
        return
    pretrain_mode = str(getattr(args, "surrogate_pretrain_mode", "full")).strip().lower()
    if pretrain_mode not in {"full", "subtree", "hybrid"}:
        raise ValueError("--surrogate_pretrain_mode must be one of: full, subtree, hybrid")

    refresh_interval = max(int(getattr(args, "surrogate_pretrain_actual_refresh_interval", 10)), 0)
    replay_enabled = bool(getattr(args, "surrogate_pretrain_use_replay", True))
    replay_steps = max(int(getattr(args, "surrogate_pretrain_replay_steps", 4)), 0)
    replay_batch = max(int(getattr(args, "surrogate_pretrain_replay_batch_size", 16)), 1)
    replay_buffer_size = max(int(getattr(args, "surrogate_pretrain_replay_buffer_size", 256)), 0)
    debug_interval = int(getattr(args, "surrogate_pretrain_sparsepcgc_debug_interval", 10))
    teacher_type = str(getattr(args, "surrogate_pretrain_subtree_teacher_type", "local_actual")).strip().lower()
    full_calibration_interval = max(int(getattr(args, "surrogate_pretrain_full_calibration_interval", 50)), 1)
    full_calibration_steps = max(int(getattr(args, "surrogate_pretrain_full_calibration_steps", 1)), 1)
    max_wall_time_sec = max(float(getattr(args, "surrogate_pretrain_max_wall_time_sec", 0.0)), 0.0)
    writer.write(
        "SurrogatePretrain start: "
        f"steps={steps}, lr={float(getattr(args, 'surrogate_pretrain_lr', 1e-4)):.6g}, "
        f"freeze_network={bool(getattr(args, 'surrogate_pretrain_freeze_network', True))}, "
        f"refresh_interval={refresh_interval}, "
        f"replay_enabled={replay_enabled}, replay_steps={replay_steps}, "
        f"replay_batch={replay_batch}, replay_buffer={replay_buffer_size}, "
        f"sparsepcgc_debug_interval={debug_interval}, mode={pretrain_mode}, "
        f"teacher_type={teacher_type}, full_calibration_interval={full_calibration_interval}, "
        f"full_calibration_steps={full_calibration_steps}, "
        f"subtree_steps_per_full={int(getattr(args, 'surrogate_pretrain_subtree_steps_per_full', full_calibration_interval))}, "
        f"subtree_depth_percent={float(getattr(args, 'surrogate_pretrain_subtree_depth_percent_min', 0.0)):.3g}-"
        f"{float(getattr(args, 'surrogate_pretrain_subtree_depth_percent_max', 0.50)):.3g}, "
        f"max_wall_time_sec={max_wall_time_sec:.1f}"
    )
    log_for_better_event(
        for_better_path,
        "surrogate_pretrain_start",
        steps=steps,
        mode=pretrain_mode,
        teacher_type=teacher_type,
        refresh_interval=refresh_interval,
        replay_enabled=replay_enabled,
        replay_steps=replay_steps,
        replay_batch=replay_batch,
        replay_buffer_size=replay_buffer_size,
        sparsepcgc_debug_interval=debug_interval,
        full_calibration_interval=full_calibration_interval,
        full_calibration_steps=full_calibration_steps,
        subtree_depth_percent_min=float(getattr(args, "surrogate_pretrain_subtree_depth_percent_min", 0.0)),
        subtree_depth_percent_max=float(getattr(args, "surrogate_pretrain_subtree_depth_percent_max", 0.50)),
        max_wall_time_sec=max_wall_time_sec,
    )
    if pretrain_mode == "full" and steps >= 1000:
        writer.write(
            "[WARN] surrogate_pretrain_mode=full with "
            f"{steps} steps can be extremely slow. Consider --surrogate_pretrain_mode subtree or hybrid."
        )
    elif pretrain_mode == "subtree":
        writer.write(
            "[SurrogatePretrain] mode=subtree uses "
            f"{teacher_type} teacher. Full SparsePCGC actual codec will not be called every step."
        )
        if teacher_type == "local_proxy":
            writer.write(
                "[WARN] surrogate_pretrain_subtree_teacher_type=local_proxy is NOT actual SparsePCGC bit; "
                "it uses differentiable local proxy scale. Local-proxy samples are not stored in actual replay by default."
            )
            log_for_better_event(
                for_better_path,
                "surrogate_pretrain_local_proxy_not_actual",
                message="subtree local_proxy trains on differentiable proxy terms, not actual SparsePCGC bit; local_proxy replay storage is disabled by default.",
                store_local_proxy_replay=bool(getattr(args, "surrogate_pretrain_store_local_proxy_replay", False)),
            )
    elif pretrain_mode == "hybrid":
        writer.write(
            "[SurrogatePretrain] mode=hybrid uses subtree steps plus full calibration. "
            f"Full actual calibration interval={full_calibration_interval}, steps_per_window={full_calibration_steps}."
        )
    if teacher_type == "local_actual":
        writer.write(
            "[WARN] surrogate_pretrain_subtree_teacher_type=local_actual encodes subtree-only point clouds. "
            "Subtree actual bit is a local teacher and is not identical to full-cloud SparsePCGC bit because "
            "bbox/origin/header/global density/context can differ."
        )
    if teacher_type == "inherited_full":
        writer.write(
            "[WARN] surrogate_pretrain_subtree_teacher_type=inherited_full assigns a full-cloud teacher/cache "
            "to subtree steps. This is biased and should be used only for calibration experiments."
        )

    model_was_training = model.training
    param_states = [(param, bool(param.requires_grad)) for param in model.parameters()]
    surrogate_optimizer = getattr(loss, "surrogate_optimizer", None)
    original_surrogate_lrs = optimizer_lrs(surrogate_optimizer)
    original_replay_max_entries = getattr(loss, "surrogate_replay_max_entries", None)
    pretrain_lr = float(getattr(args, "surrogate_pretrain_lr", 1e-4))
    if pretrain_lr > 0.0 and original_surrogate_lrs:
        set_optimizer_lrs(surrogate_optimizer, [pretrain_lr for _ in original_surrogate_lrs])

    saved_args = {
        "_global_train_step": getattr(args, "_global_train_step", 0),
        "_collect_sparsepcgc_debug": getattr(args, "_collect_sparsepcgc_debug", False),
        "_surrogate_pretrain_timing_enabled": getattr(args, "_surrogate_pretrain_timing_enabled", False),
        "_surrogate_pretrain_active": getattr(args, "_surrogate_pretrain_active", False),
        "_surrogate_pretrain_mode": getattr(args, "_surrogate_pretrain_mode", None),
        "_surrogate_pretrain_teacher_type": getattr(args, "_surrogate_pretrain_teacher_type", None),
        "_surrogate_pretrain_actual_scope": getattr(args, "_surrogate_pretrain_actual_scope", None),
        "_surrogate_pretrain_full_calibration": getattr(args, "_surrogate_pretrain_full_calibration", False),
        "compression_surrogate_refresh_interval": getattr(args, "compression_surrogate_refresh_interval", 0),
        "compression_surrogate_replay_steps": getattr(args, "compression_surrogate_replay_steps", 0),
        "compression_surrogate_replay_batch": getattr(args, "compression_surrogate_replay_batch", 1),
        "compression_surrogate_replay_entries": getattr(args, "compression_surrogate_replay_entries", 0),
        "compression_surrogate_reuse_last_target": getattr(args, "compression_surrogate_reuse_last_target", True),
    }
    corr_pairs = {}
    abs_error_history = []
    step_times = []
    fresh_actual_count = 0
    completed_steps = 0
    last_corr = None
    last_sign_match = None
    early_stop_hits = 0
    early_stop_reason = None
    eta_warned = False
    log_interval = max(int(getattr(args, "surrogate_pretrain_log_interval", 10)), 1)
    print_interval = max(int(getattr(args, "surrogate_pretrain_print_interval", log_interval)), 1)
    pretrain_start_time = time.perf_counter()
    last_log_time = 0.0

    try:
        if bool(getattr(args, "surrogate_pretrain_freeze_network", True)):
            for param, _old_state in param_states:
                param.requires_grad_(False)
        args._surrogate_pretrain_active = True
        args._surrogate_pretrain_timing_enabled = True
        args._surrogate_pretrain_mode = pretrain_mode
        args._surrogate_pretrain_teacher_type = teacher_type
        args._surrogate_pretrain_actual_scope = "full"
        args._surrogate_pretrain_full_calibration = False
        args.compression_surrogate_refresh_interval = refresh_interval
        args.compression_surrogate_replay_steps = replay_steps if replay_enabled else 0
        args.compression_surrogate_replay_batch = replay_batch
        args.compression_surrogate_replay_entries = replay_buffer_size
        args.compression_surrogate_reuse_last_target = bool(
            getattr(args, "surrogate_pretrain_allow_stale_target", True)
        )
        if original_replay_max_entries is not None:
            loss.surrogate_replay_max_entries = replay_buffer_size
            if replay_buffer_size > 0 and len(getattr(loss, "surrogate_replay", [])) > replay_buffer_size:
                loss.surrogate_replay = list(loss.surrogate_replay[-replay_buffer_size:])
                loss.surrogate_replay_next = len(loss.surrogate_replay) % replay_buffer_size
            elif replay_buffer_size <= 0:
                loss.surrogate_replay = []
                loss.surrogate_replay_next = 0
        model.train()

        while completed_steps < steps and early_stop_reason is None:
            progressed = False
            for _seq_dir, dataset in seq_datasets:
                loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
                data_wait_t0 = time.perf_counter()
                for local_step, pts in enumerate(loader):
                    surrogate_st = time.time()
                    data_time = time.perf_counter() - data_wait_t0
                    if completed_steps >= steps or early_stop_reason is not None:
                        break
                    progressed = True
                    step_zero = int(completed_steps)
                    step_number = step_zero + 1
                    step_t0 = data_wait_t0
                    file_path = dataset.files[local_step]
                    base_cache_key = f"surrogate_pretrain|{make_step_cache_key(file_path, args)}"
                    cache_key = base_cache_key
                    args._global_train_step = step_zero
                    full_window_pos = (step_number - 1) % full_calibration_interval
                    full_calibration = bool(
                        pretrain_mode == "hybrid"
                        and full_window_pos >= max(full_calibration_interval - full_calibration_steps, 0)
                    )
                    subtree_enabled = bool(pretrain_mode in {"subtree", "hybrid"} and not full_calibration)
                    actual_scope = "subtree" if subtree_enabled else "full"
                    effective_teacher_type = teacher_type if subtree_enabled else "full_actual"
                    if full_calibration:
                        cache_key = f"{base_cache_key}|hybrid_full_calibration"
                    if not subtree_enabled:
                        should_refresh_actual = (
                            full_calibration
                            or step_zero == 0
                            or refresh_interval == 1
                            or (refresh_interval > 1 and step_zero % refresh_interval == 0)
                        )
                    elif teacher_type == "local_actual":
                        should_refresh_actual = (
                            step_zero == 0
                            or refresh_interval == 1
                            or (refresh_interval > 1 and step_zero % refresh_interval == 0)
                        )
                    else:
                        should_refresh_actual = False
                    debug_collect = bool(
                        debug_interval > 0
                        and (step_number == 1 or step_number % debug_interval == 0)
                    )
                    args._collect_sparsepcgc_debug = debug_collect
                    args._surrogate_pretrain_mode = pretrain_mode
                    args._surrogate_pretrain_teacher_type = effective_teacher_type
                    args._surrogate_pretrain_actual_scope = actual_scope
                    args._surrogate_pretrain_full_calibration = full_calibration

                    model_t0 = time.perf_counter()
                    subtree_sampling_time = 0.0
                    subtree_meta = {
                        "depth": 0,
                        "point_count": 0,
                        "bbox_min": None,
                        "bbox_max": None,
                        "retry_count": 0,
                        "skip_reason": "none",
                        "subtree_key": None,
                        "total_subtree_count": 0,
                        "eligible_subtree_count": 0,
                        "selected_subtree_count": 0,
                        "requested_depth": 0,
                        "depth_percent_range": None,
                        "depth_absolute_range": None,
                    }
                    comp_debug = None
                    refresh_actual_arg = "always" if full_calibration else should_refresh_actual
                    if should_refresh_actual:
                        writer.write(
                            "[SurrogatePretrainActual] start "
                            f"mode={pretrain_mode} step={step_number}/{steps} "
                            f"scope={actual_scope} teacher={effective_teacher_type}"
                        )
                    with torch.enable_grad():
                        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                        if subtree_enabled:
                            subtree_sample = build_surrogate_pretrain_subtree_sample(
                                pts,
                                args,
                                base_cache_key,
                                use_cuda,
                                global_step=step_zero,
                            )
                            subtree_sampling_time = case_float(subtree_sample.get("sampling_time", 0.0), 0.0)
                            subtree_meta.update(
                                {
                                    "depth": case_int(subtree_sample.get("depth", 0)),
                                    "point_count": case_int(subtree_sample.get("point_count", 0)),
                                    "bbox_min": subtree_sample.get("bbox_min"),
                                    "bbox_max": subtree_sample.get("bbox_max"),
                                    "retry_count": case_int(subtree_sample.get("retry_count", 0)),
                                    "skip_reason": str(subtree_sample.get("skip_reason", "none")),
                                    "subtree_key": subtree_sample.get("subtree_key"),
                                    "total_subtree_count": case_int(subtree_sample.get("total_subtree_count", 0)),
                                    "eligible_subtree_count": case_int(subtree_sample.get("eligible_subtree_count", 0)),
                                    "selected_subtree_count": case_int(subtree_sample.get("selected_subtree_count", 0)),
                                    "requested_depth": case_int(subtree_sample.get("requested_depth", 0)),
                                    "depth_percent_range": subtree_sample.get("depth_percent_range"),
                                    "depth_absolute_range": subtree_sample.get("depth_absolute_range"),
                                }
                            )
                            if subtree_meta["point_count"] <= 0 or subtree_meta["skip_reason"] == "empty_selected_subtree":
                                model_time = time.perf_counter() - model_t0
                                comp_debug = {
                                    "teacher_mode": "skip",
                                    "teacher_skipped": True,
                                    "actual_value_source": f"subtree_skip:{subtree_meta['skip_reason']}",
                                    "surrogate_replay_size": len(getattr(loss, "surrogate_replay", [])),
                                    "surrogate_replay_sample_count": 0,
                                    "timing": {},
                                }
                            else:
                                subtree_xyz = subtree_sample["subtree_xyz"]
                                subtree_attr = subtree_sample.get("subtree_attr")
                                cache_key = subtree_sample["subtree_cache_key"]
                                with autocast_ctx:
                                    (
                                        gen_subtree_pts,
                                        _L_attr,
                                        _L_policy,
                                        _L_actuator,
                                        final_w,
                                        _Lp_out,
                                        _La_fit,
                                        _La_rep,
                                        _out_label,
                                    ) = model.forward(
                                        subtree_xyz,
                                        subtree_attr,
                                        cache_key=cache_key,
                                        return_attr_output=False,
                                    )
                                    gen_xyz = gen_subtree_pts[:, :3, :]
                                    final_w_for_loss = None
                                    if str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() != "hard":
                                        final_w_for_loss = final_w
                                    compression_gen_xyz, _noise_debug = prepare_compression_points(
                                        gen_xyz,
                                        args,
                                        model,
                                        collect_stats=debug_collect,
                                    )
                                    model_time = time.perf_counter() - model_t0
                                    saved_stale = getattr(args, "surrogate_pretrain_allow_stale_target", True)
                                    saved_reuse = getattr(args, "compression_surrogate_reuse_last_target", True)
                                    if teacher_type in {"local_proxy", "none"}:
                                        args.surrogate_pretrain_allow_stale_target = False
                                        args.compression_surrogate_reuse_last_target = False
                                    try:
                                        loss.get_compression_loss(
                                            args,
                                            gen_xyz=compression_gen_xyz,
                                            gt_xyz=subtree_xyz[:, :3, :],
                                            final_w=final_w_for_loss,
                                            cache_key=cache_key,
                                            refresh_actual_gen=refresh_actual_arg,
                                            actual_gen_xyz=gen_xyz,
                                        )
                                    finally:
                                        args.surrogate_pretrain_allow_stale_target = saved_stale
                                        args.compression_surrogate_reuse_last_target = saved_reuse
                        else:
                            input_xyz, patches, centroid_xyz, fd_xyz = prepare_whole_cloud_inputs(
                                pts,
                                args,
                                cache_key,
                                use_cuda,
                            )
                            with autocast_ctx:
                                gen_patches, _L_attr, _L_policy, _L_actuator, final_w, _Lp_out, _La_fit, _La_rep, _out_label = model.forward(
                                    patches,
                                    None,
                                    cache_key=cache_key,
                                    coord_scale=fd_xyz,
                                    return_attr_output=False,
                                )
                                gen_xyz = (centroid_xyz + gen_patches[:, :3, :] * fd_xyz).contiguous()
                                final_w_for_loss = None
                                if str(getattr(args, "discrete_loss_mode", "hard")).strip().lower() != "hard":
                                    final_w_for_loss = final_w
                                compression_gen_xyz, _noise_debug = prepare_compression_points(
                                    gen_xyz,
                                    args,
                                    model,
                                    collect_stats=True,
                                )
                                model_time = time.perf_counter() - model_t0
                                loss.get_compression_loss(
                                    args,
                                    gen_xyz=compression_gen_xyz,
                                    gt_xyz=input_xyz[:, :3, :],
                                    final_w=final_w_for_loss,
                                    cache_key=cache_key,
                                    refresh_actual_gen=refresh_actual_arg,
                                    actual_gen_xyz=gen_xyz,
                                )

                    if comp_debug is None:
                        comp_debug = dict(getattr(loss, "last_compression_debug", {}) or {})
                    comp_timing = comp_debug.get("timing", {}) or {}
                    actual_value = finite_float_or_none(comp_debug.get("actual_total_bit_percent", None))
                    pred_value = finite_float_or_none(comp_debug.get("surrogate_pred_bit", None))
                    teacher_mode_value = str(comp_debug.get("teacher_mode", "refresh" if should_refresh_actual else "skip"))
                    actual_source = str(comp_debug.get("actual_value_source", ""))
                    if actual_source.startswith(("missing", "target_missing", "local_proxy", "subtree_skip")):
                        actual_value = None
                    if actual_value is not None and pred_value is not None and is_fresh_actual(args, comp_debug):
                        fresh_actual_count += 1
                        last_corr, last_sign_match, _count = append_corr_pair(
                            corr_pairs,
                            "surrogate_actual",
                            pred_value,
                            actual_value,
                            max(int(getattr(args, "sparsepcgc_corr_window", 100)), 2),
                        )
                    abs_error = None
                    if actual_value is not None and pred_value is not None:
                        abs_error = abs(pred_value - actual_value)
                    if abs_error is not None and is_fresh_actual(args, comp_debug):
                        abs_error_history.append(float(abs_error))
                        max_abs_history = max(int(getattr(args, "sparsepcgc_corr_window", 100)), 2)
                        if len(abs_error_history) > max_abs_history:
                            del abs_error_history[:-max_abs_history]

                    step_time = time.perf_counter() - step_t0
                    step_times.append(float(step_time))
                    avg_step_time = sum(step_times) / max(len(step_times), 1)
                    eta_seconds = max(steps - step_number, 0) * avg_step_time
                    actual_eval_time = case_float(comp_timing.get("actual_encode", 0.0), 0.0)
                    if actual_eval_time <= 0.0 and bool(comp_debug.get("teacher_refresh", False)):
                        actual_eval_time = case_float(comp_debug.get("actual_encode_time_total", 0.0), 0.0)
                    if bool(comp_debug.get("teacher_refreshed", comp_debug.get("teacher_refresh", False))):
                        writer.write(
                            "[SurrogatePretrainActual] done "
                            f"mode={pretrain_mode} step={step_number}/{steps} "
                            f"scope={actual_scope} teacher={effective_teacher_type} "
                            f"time={actual_eval_time:.2f}s"
                        )
                    surrogate_update_time = (
                        case_float(comp_timing.get("surrogate_fit", 0.0), 0.0)
                        + case_float(comp_timing.get("surrogate_replay", 0.0), 0.0)
                        + (
                            case_float(comp_timing.get("target_cache", 0.0), 0.0)
                            if bool(comp_debug.get("teacher_stale", False))
                            else 0.0
                        )
                    )
                    gpu_alloc_mb = cuda_alloc_mb(use_cuda)
                    cpu_rss_mb = process_rss_mb()
                    current_lr = optimizer_lrs(surrogate_optimizer)
                    param_norm = surrogate_param_norm(loss)
                    row = {
                        "surrogate_pretrain_step": step_number,
                        "pretrain_mode": pretrain_mode,
                        "pretrain_teacher_type": effective_teacher_type,
                        "sample_name": os.path.basename(str(file_path)),
                        "codec": comp_debug.get("teacher_codec", getattr(args, "compress", "unknown")),
                        "backend": backend,
                        "surrogate_pretrain_loss": case_float(comp_debug.get("surrogate_train_loss", float("nan")), float("nan")),
                        "surrogate_pretrain_pred_bit_percent": pred_value,
                        "surrogate_pretrain_actual_bit_percent": actual_value,
                        "surrogate_pretrain_abs_error": abs_error,
                        "surrogate_pretrain_corr": last_corr,
                        "surrogate_pretrain_sign_match": last_sign_match,
                        "surrogate_pretrain_teacher_refresh": bool(comp_debug.get("teacher_refresh", False)),
                        "surrogate_pretrain_target_age": case_int(comp_debug.get("teacher_target_age", 0)),
                        "surrogate_pretrain_fresh_actual_count": fresh_actual_count,
                        "pretrain_step_time": step_time,
                        "pretrain_actual_eval_time": actual_eval_time,
                        "pretrain_surrogate_update_time": surrogate_update_time,
                        "pretrain_data_time": data_time,
                        "pretrain_subtree_sampling_time": subtree_sampling_time,
                        "pretrain_model_time": model_time,
                        "pretrain_log_time": last_log_time,
                        "pretrain_eta_seconds": eta_seconds,
                        "pretrain_gpu_alloc_mb": gpu_alloc_mb,
                        "pretrain_cpu_rss_mb": cpu_rss_mb,
                        "teacher_mode": teacher_mode_value,
                        "teacher_refreshed": bool(comp_debug.get("teacher_refreshed", comp_debug.get("teacher_refresh", False))),
                        "teacher_replayed": bool(comp_debug.get("teacher_replayed", False)),
                        "teacher_stale": bool(comp_debug.get("teacher_stale", False)),
                        "teacher_skipped": bool(comp_debug.get("teacher_skipped", False)),
                        "teacher_target_age": case_int(comp_debug.get("teacher_target_age", 0)),
                        "replay_buffer_size": case_int(comp_debug.get("surrogate_replay_size", len(getattr(loss, "surrogate_replay", [])))),
                        "replay_sample_count": case_int(comp_debug.get("surrogate_replay_sample_count", 0)),
                        "fresh_actual_count": fresh_actual_count,
                        "sparsepcgc_debug_collected": bool(comp_debug.get("sparsepcgc_debug_collected", False)),
                        "sparsepcgc_debug_time": case_float(comp_debug.get("sparsepcgc_debug_time", 0.0), 0.0),
                        "pretrain_subtree_enabled": bool(subtree_enabled),
                        "pretrain_subtree_depth": subtree_meta["depth"] if subtree_enabled else None,
                        "pretrain_subtree_point_count": subtree_meta["point_count"] if subtree_enabled else None,
                        "pretrain_subtree_bbox_min": subtree_meta["bbox_min"],
                        "pretrain_subtree_bbox_max": subtree_meta["bbox_max"],
                        "pretrain_subtree_retry_count": subtree_meta["retry_count"],
                        "pretrain_subtree_skip_reason": subtree_meta["skip_reason"],
                        "pretrain_subtree_requested_depth": subtree_meta["requested_depth"] if subtree_enabled else None,
                        "pretrain_subtree_depth_percent_range": subtree_meta["depth_percent_range"] if subtree_enabled else None,
                        "pretrain_subtree_depth_absolute_range": subtree_meta["depth_absolute_range"] if subtree_enabled else None,
                        "pretrain_subtree_key": subtree_meta["subtree_key"],
                        "pretrain_subtree_total_count": subtree_meta["total_subtree_count"],
                        "pretrain_subtree_eligible_count": subtree_meta["eligible_subtree_count"],
                        "pretrain_subtree_selected_count": subtree_meta["selected_subtree_count"],
                        "pretrain_full_calibration": bool(full_calibration),
                        "pretrain_actual_scope": actual_scope,
                        "surrogate_param_norm": param_norm,
                        "surrogate_pretrain_lr": current_lr[0] if current_lr else None,
                        "surrogate_pretrain_mean_error": finite_float_or_none(comp_debug.get("surrogate_mean_error",comp_debug.get("surrogate_mean_bit_error", None),)),
                    }

                    log_t0 = time.perf_counter()
                    should_print = step_number == 1 or step_number % print_interval == 0 or step_number >= steps
                    if should_print:
                        actual_text = "NA" if actual_value is None else f"{case_float(actual_value, float('nan')):.6f}"
                        pred_text = "NA" if pred_value is None else f"{case_float(pred_value, float('nan')):.6f}"
                        fit_loss_value = finite_float_or_none(comp_debug.get("surrogate_train_loss"))
                        loss_text = "NA" if fit_loss_value is None else f"{case_float(fit_loss_value, float('nan')):.6f}"
                        target_abs_error = finite_float_or_none(comp_debug.get("surrogate_abs_bit_error"))
                        error_label = "abs" if abs_error is not None else "target_abs"
                        error_value = abs_error if abs_error is not None else target_abs_error
                        error_text = "NA" if error_value is None else f"{case_float(error_value, float('nan')):.6f}"
                        writer.write(
                            "[SurrogatePretrain] "
                            f"mode={pretrain_mode} "
                            f"step={step_number}/{steps} "
                            f"teacher={row['teacher_mode']} "
                            f"depth={row['pretrain_subtree_depth'] if subtree_enabled else 'NA'} "
                            f"pts={row['pretrain_subtree_point_count'] if subtree_enabled else 'NA'} "
                            f"pred={pred_text} "
                            f"actual={actual_text} "
                            f"{error_label}={error_text} "
                            f"loss={loss_text} "
                            f"fresh={fresh_actual_count}"
                        )
                    row["pretrain_log_time"] = time.perf_counter() - log_t0
                    last_log_time = row["pretrain_log_time"]
                    log_for_better_pretrain_step(
                        for_better_path,
                        row,
                        comp_debug=comp_debug,
                        extra={
                            "full_calibration": bool(full_calibration),
                            "actual_scope": actual_scope,
                            "subtree_enabled": bool(subtree_enabled),
                        },
                    )
                    append_csv_row(
                        metric_csv_paths.get("surrogate_pretrain_step"),
                        SURROGATE_PRETRAIN_COLUMNS,
                        row,
                    )
                    if plot is not None and hasattr(plot, "record_surrogate_pretrain"):
                        plot.record_surrogate_pretrain(step_number, row)

                    if step_number in {1, 3} or (step_number % max(print_interval, 1) == 0):
                        estimated_total = avg_step_time * float(steps)
                        if estimated_total > 24.0 * 3600.0 and not eta_warned:
                            eta_warned = True
                            writer.write(
                                "[WARN] Surrogate pretrain estimated time is "
                                f"{estimated_total / 3600.0:.1f} hours. Consider increasing "
                                "--surrogate_pretrain_actual_refresh_interval, enabling replay, "
                                "using --surrogate_pretrain_mode subtree/hybrid, or reducing --surrogate_step."
                            )

                    min_corr = float(getattr(args, "surrogate_pretrain_min_corr", -1.0))
                    min_sign = float(getattr(args, "surrogate_pretrain_min_sign_match", -1.0))
                    min_abs_error = float(getattr(args, "surrogate_pretrain_min_abs_error", -1.0))
                    min_fresh = max(int(getattr(args, "surrogate_pretrain_min_fresh_samples", 30)), 0)
                    patience = max(int(getattr(args, "surrogate_pretrain_early_stop_patience", 0)), 0)
                    early_enabled = patience > 0 and (min_corr >= 0.0 or min_sign >= 0.0 or min_abs_error >= 0.0)
                    mean_abs_error = mean_finite(abs_error_history)
                    corr_ok_step = min_corr < 0.0 or (last_corr is not None and last_corr >= min_corr)
                    sign_ok_step = min_sign < 0.0 or (last_sign_match is not None and last_sign_match >= min_sign)
                    abs_ok_step = min_abs_error < 0.0 or (mean_abs_error is not None and mean_abs_error <= min_abs_error)
                    fresh_ok_step = fresh_actual_count >= min_fresh
                    if early_enabled and fresh_ok_step and corr_ok_step and sign_ok_step and abs_ok_step:
                        early_stop_hits += 1
                        if early_stop_hits >= patience:
                            early_stop_reason = (
                                f"corr={format_corr(last_corr, len(corr_pairs.get('surrogate_actual', [])))}, "
                                f"sign={case_float(last_sign_match, float('nan')):.6f}, "
                                f"mean_abs={case_float(mean_abs_error, float('nan')):.6f}, "
                                f"fresh={fresh_actual_count}, patience={patience}"
                            )
                            writer.write(f"SurrogatePretrainEarlyStop: {early_stop_reason}")
                    elif early_enabled:
                        early_stop_hits = 0

                    completed_steps += 1
                    surrogate_en = time.time()
                    print(f"Surrogate Step: {completed_steps}/{steps} | {surrogate_en - surrogate_st}sec")
                    if max_wall_time_sec > 0.0:
                        elapsed_wall = time.perf_counter() - pretrain_start_time
                        if elapsed_wall >= max_wall_time_sec:
                            early_stop_reason = (
                                f"max_wall_time_sec={max_wall_time_sec:.1f}, "
                                f"elapsed={elapsed_wall:.1f}"
                            )
                            writer.write(f"SurrogatePretrainEarlyStop: {early_stop_reason}")
                    data_wait_t0 = time.perf_counter()
                if completed_steps >= steps or early_stop_reason is not None:
                    break
            if not progressed:
                writer.write("SurrogatePretrain stopped early: no training samples were available.")
                break

        min_corr = float(getattr(args, "surrogate_pretrain_min_corr", -1.0))
        min_sign = float(getattr(args, "surrogate_pretrain_min_sign_match", -1.0))
        corr_ok = min_corr < 0.0 or (last_corr is not None and last_corr >= min_corr)
        sign_ok = min_sign < 0.0 or (last_sign_match is not None and last_sign_match >= min_sign)
        final_param_norm = surrogate_param_norm(loss)
        writer.write(
            "SurrogatePretrainSummary: "
            f"mode={pretrain_mode}, completed_steps={completed_steps}, fresh_actual_count={fresh_actual_count}, "
            f"corr={format_corr(last_corr, len(corr_pairs.get('surrogate_actual', [])))}, "
            f"sign_match={case_float(last_sign_match, float('nan')):.6f}, "
            f"mean_abs_error={case_float(mean_finite(abs_error_history), float('nan')):.6f}, "
            f"corr_ok={bool(corr_ok)}, sign_match_ok={bool(sign_ok)}, "
            f"early_stop={early_stop_reason or 'none'}, "
            f"surrogate_param_norm={case_float(final_param_norm, float('nan')):.6f}"
        )
        writer.write(
            "[SurrogatePretrain] complete "
            f"mode={pretrain_mode} steps={completed_steps} "
            f"surrogate_param_norm={case_float(final_param_norm, float('nan')):.6f} "
            f"lr={optimizer_lrs(surrogate_optimizer)[0] if optimizer_lrs(surrogate_optimizer) else 'NA'}"
        )
        log_for_better_pretrain_complete(
            for_better_path,
            mode=pretrain_mode,
            completed_steps=completed_steps,
            fresh_actual_count=fresh_actual_count,
            corr=last_corr,
            sign_match=last_sign_match,
            mean_abs_error=mean_finite(abs_error_history),
            early_stop_reason=early_stop_reason or "none",
            surrogate_param_norm=final_param_norm,
            surrogate_lr=optimizer_lrs(surrogate_optimizer),
        )
        if plot is not None and hasattr(plot, "plot_surrogate_pretrain_curve"):
            plot.plot_surrogate_pretrain_curve()
            writer.write(f"Saved surrogate pretrain plot/csv: {plot.save_dir}")
        if bool(getattr(args, "surrogate_pretrain_checkpoint", True)):
            path = os.path.join(ckpt_dir, "surrogate_pretrain.pth")
            torch.save(
                {
                    "compression_surrogate": loss.compression_surrogate.state_dict(),
                    "compression_surrogate_state_dict": loss.compression_surrogate.state_dict(),
                    "surrogate_optimizer_state_dict": None if surrogate_optimizer is None else surrogate_optimizer.state_dict(),
                    "surrogate_step": int(getattr(loss, "_surrogate_step", 0)),
                    "surrogate_feature_dim": int(getattr(loss, "surrogate_feature_dim", 0) or 0),
                    "surrogate_levels": list(getattr(loss, "surrogate_levels", []) or []),
                    "completed_steps": completed_steps,
                    "fresh_actual_count": fresh_actual_count,
                    "corr": last_corr,
                    "sign_match": last_sign_match,
                    "surrogate_param_norm": final_param_norm,
                    "early_stop_reason": early_stop_reason,
                },
                path,
            )
            writer.write(f"SurrogatePretrainCheckpoint: {path}")
    finally:
        for param, old_state in param_states:
            param.requires_grad_(old_state)
        for key, value in saved_args.items():
            setattr(args, key, value)
        if original_replay_max_entries is not None:
            loss.surrogate_replay_max_entries = original_replay_max_entries
        if original_surrogate_lrs:
            joint_scale = float(getattr(args, "surrogate_joint_lr_scale", 0.1))
            joint_lrs = [lr * joint_scale for lr in original_surrogate_lrs]
            set_optimizer_lrs(surrogate_optimizer, joint_lrs)
            writer.write(
                "SurrogatePretrainJointLR: "
                f"original={','.join(f'{lr:.6g}' for lr in original_surrogate_lrs)}, "
                f"scale={joint_scale:.6g}, "
                f"joint={','.join(f'{lr:.6g}' for lr in joint_lrs)}"
            )
            log_for_better_event(
                for_better_path,
                "surrogate_pretrain_joint_lr",
                original_lrs=original_surrogate_lrs,
                joint_scale=joint_scale,
                joint_lrs=joint_lrs,
            )
        if model_was_training:
            model.train()
        else:
            model.eval()