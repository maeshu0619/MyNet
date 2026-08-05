"""train.pyから分離した訓練補助処理。"""

import os
_TMPDIR = os.environ.get("TMPDIR") or "/dev/shm/mynet_tmp"
try:
    os.makedirs(_TMPDIR, exist_ok=True)
    os.environ["TMPDIR"] = _TMPDIR
    os.environ["TEMP"] = _TMPDIR
    os.environ["TMP"] = _TMPDIR
except OSError:
    pass

import torch
import torch.optim as optim
import argparse
import ctypes
import gc
import hashlib
import math
import csv
import numpy as np
from cfgs.utils import str2bool
import multiprocessing as mp
from collections import OrderedDict
import time
import datetime
from contextlib import nullcontext
from functools import partial

try:
    _LIBC_MALLOC_TRIM = ctypes.CDLL("libc.so.6").malloc_trim
    _LIBC_MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _LIBC_MALLOC_TRIM.restype = ctypes.c_int
except (OSError, AttributeError):
    _LIBC_MALLOC_TRIM = None


def _release_cpu_step_memory():
    """解放済みfull-cloud CPU offload領域をOSへ返す。"""
    gc.collect()
    if _LIBC_MALLOC_TRIM is not None:
        _LIBC_MALLOC_TRIM(0)

from models.network import Network
import models.network as network_module
from models.utils.loss.loss import Loss
from models.utils.loss.k_proposal_distillation import OfflineKProposalTeacherStore
from models.utils.loss.single_plan_distillation import SinglePlanTeacherStore
from models.utils.notify.mail_notify import TrainingMailNotifier
from record.write import Writing
from record.plot import PlotMaker
from models.utils.pointcloud.utils_repkpu import *
from models.utils.pointcloud.octree_subtree import *
from models.utils.pointcloud.sparsepcgc_voxel import (
    quantize_sparsepcgc_coords,
    attach_sparsepcgc_voxel_meta,
    restore_points_from_voxel_coords,
)
from models.utils.pointcloud.quant_noise import add_uniform_quantization_noise, resolve_uniform_noise_delta
from models.utils.pointcloud.ana_den6_online import (
    prefetch_ana_den6_online_guidance,
    shutdown_ana_den6_online_prefetch,
)
from models.utils.pointcloud.voxel_collision import (
    compute_voxel_collision_stats_batch,
    flatten_voxel_collision_stats,
    format_voxel_collision_summary,
)
from models.utils.data.dataset import *
from models.utils.patching.patch import *
from models.utils.compression.octree_stats import hard_octree_occupancy_stats
from models.utils.compression.edit_record import sparsepcgc_effective_edit_record_bit_scale
from models.utils.config.args import parse_pugan_args

from models.utils.training.full_cloud_actual_correction import (
    update_full_cloud_actual_correction_state,
    build_full_cloud_actual_correction_loss,
)
from models.utils.training.saved_tensor_offload import (
    selective_saved_tensor_cpu_offload,
    release_autograd_transient_references,
    release_saved_tensor_offload_payloads,
    saved_tensor_offload_stats,
)
from models.utils.training.memory_diagnostics import MemoryDiagnosticsCSV

from models.utils.training.utils import *
from models.utils.training.noise_debug import *
from models.utils.training.correlation import *
from models.utils.training.optim_amp import *
from models.utils.training.checkpointing import save_episode_checkpoint
from models.utils.training.log_step import *
from models.utils.training.log_epoch_episode import *
from models.utils.training.log_setup import log_training_setup
from models.utils.training.scalar_utils import *
from models.utils.training.correlation_debug import *
from models.utils.training.sparsepcgc_controls import *
from models.utils.training.compression_primary_loss import *
from models.utils.training.compression_primary_loss import (
    _compression_primary_support_balance,
    monotonic_support_scale,
)
from models.utils.training.full_context_subtree_loss import build_full_context_subtree_delta_loss
from models.utils.training.case_debug import *
from models.utils.training.metric_csv import *
from models.utils.training.metric_columns import (
    LOSS_GRAD_PROBE_COLUMNS,
    PHASE7_EVAL_SUMMARY_COLUMNS,
    PROPOSAL_CANDIDATE_COLUMNS,
    FULL_CLOUD_AMOUNT_CANDIDATE_COLUMNS,
    FULL_CLOUD_AMOUNT_SEQUENCE_SUMMARY_COLUMNS,
)
from models.utils.training.actual_codec_status import *
from models.utils.training.metric_rows import *
from models.utils.training.lr_control import apply_optimizer_lr_floor, step_scheduler_with_floor, optimizer_lrs_safe
from models.utils.training.episode_metrics import *
from models.utils.training.checkpoint_metrics import *
from models.utils.training.actual_compression_guard import apply_actual_compression_guard
from models.utils.training.for_better_logging import *
from models.utils.training.train_flow import * # train loopのStage固定、全点群入力、圧縮目的合成、Epoch窓選択を使う
from models.utils.training.loss_grad_probe import build_loss_grad_probe_rows, summarize_loss_grad_probe_rows

from models.utils.surrogate.pretrain import *

STEP_GRAD_COLUMNS = [
    "global_step",
    "episode",
    "epoch",
    "step",
    "stage",
    "loss_name",
    "loss_value",
    "target_group",
    "matched_param_count",
    "used_param_count",
    "none_grad_param_count",
    "grad_element_count",
    "grad_l2",
    "grad_abs_mean",
    "grad_abs_max",
    "grad_signed_mean",
    "param_name_sample",
]

def _limit_training_seq_dirs(seq_dirs, args):
    diagnostic_sequence = str(
        getattr(args, "network_k_diagnostic_sequence_name", "") or ""
    ).strip()
    if diagnostic_sequence:
        if str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() not in {
            "network_k_proposal_policy", "single_plan_student", "ana_den6_online"
        }:
            raise ValueError(
                "network_k_diagnostic_sequence_nameはK/Single/Exact-den6診断専用である"
            )
        selected = [
            path for path in seq_dirs
            if os.path.basename(os.path.normpath(str(path))) == diagnostic_sequence
        ]
        if len(selected) != 1:
            raise ValueError("K Proposal診断系列を一意に特定できない")
        return selected
    # 8iは従来既定では先頭3シーケンスのみを使うが、argsで4つ全部へ切替可能にする。
    if str(getattr(args, "dataname", "")).strip().lower() == "8i":
        mode = str(getattr(args, "train_8i_sequence_mode", "first3")).strip().lower()
        if mode == "all4":
            return list(seq_dirs)
        return list(seq_dirs[:3])
    return list(seq_dirs)


def _configure_streaming_exact_caches(args, seq_datasets, writer):
    """巡回幅より小さいLRUを止め、固定Tensorは厳密なdisk cacheへ移す。"""
    if not bool(getattr(args, "streaming_cache_auto_policy", True)):
        return
    train_limit = max(int(getattr(args, "train_frames_per_sequence", 0)), 0)
    paths = []
    for _, dataset in seq_datasets:
        source = list(getattr(dataset, "all_files", ()) or getattr(dataset, "files", ()))
        if train_limit > 0:
            source = source[:train_limit]
        paths.extend(os.path.realpath(str(path)) for path in source)
    paths = sorted(set(paths))
    if not paths:
        return
    total_disk_bytes = 0
    for path in paths:
        try:
            total_disk_bytes += int(os.path.getsize(path))
        except OSError:
            pass

    working_set = len(paths)
    dataset_entries = max(int(getattr(args, "dataset_cache_max_entries", 64)), 0)
    dataset_bytes = max(
        int(getattr(args, "dataset_cache_max_memory_mb", 1024)), 0
    ) * 1024 * 1024
    dataset_cannot_cover = (
        dataset_entries <= 0
        or working_set > dataset_entries
        or (dataset_bytes > 0 and total_disk_bytes > dataset_bytes)
    )

    episode_entries = int(getattr(args, "episode_input_common_cache_max_entries", 0))
    if episode_entries <= 0:
        episode_entries = working_set
    episode_bytes = max(
        int(getattr(args, "episode_input_common_cache_max_memory_mb", 512)), 0
    ) * 1024 * 1024
    # canonical contextは少なくとも入力座標と同じ規模になるため、
    # 圧縮済みPLY総量さえ上限を超える場合は全巡回を保持できない。
    episode_cannot_cover = (
        episode_entries <= 0
        or working_set > episode_entries
        or (episode_bytes > 0 and total_disk_bytes > episode_bytes)
    )

    if dataset_cannot_cover:
        args.dataset_cache = False
        clear_ply_cache()
        for _, dataset in seq_datasets:
            dataset.use_cache = False
    if episode_cannot_cover:
        args.episode_input_common_cache = False

    hot_entries = max(int(getattr(args, "streaming_cache_hot_entries", 4)), 0)
    if working_set > hot_entries:
        args.heuristic_guidance_cpu_tensor_cache_entries = min(
            max(int(getattr(
                args, "heuristic_guidance_cpu_tensor_cache_entries", 64
            )), 0),
            hot_entries,
        )
        args.structure_fixed_cache_max_entries = min(
            max(int(getattr(args, "structure_fixed_cache_max_entries", 64)), 0),
            hot_entries,
        )
    args.heuristic_guidance_tensor_disk_cache = True
    args.structure_fixed_disk_cache = True
    writer.write(
        "StreamingExactCachePolicy: "
        f"working_set={working_set}, "
        f"ply_disk_total_mb={total_disk_bytes / (1024.0 ** 2):.1f}, "
        f"dataset_ram_cache={int(bool(getattr(args, 'dataset_cache', False)))}, "
        f"episode_common_ram_cache={int(bool(getattr(args, 'episode_input_common_cache', False)))}, "
        f"guidance_hot_entries={int(getattr(args, 'heuristic_guidance_cpu_tensor_cache_entries', 0))}, "
        f"structure_hot_entries={int(getattr(args, 'structure_fixed_cache_max_entries', 0))}, "
        "guidance_disk_cache=1, structure_disk_cache=1"
    )


def _sparsepcgc_outcome_actual_percent(debug):
    """
    Outcome Weighted Imitation用にactual圧縮損失を取り出す。
    負なら改善、正なら悪化である。
    """
    if not isinstance(debug, dict):
        return None

    for key in (
        "compression_loss_raw",
        "actual_forward_raw_value",
        "actual_bit_percent_raw",
        "actual_bit_percent",
        "compression_loss_used",
        "actual_bit_percent_used_for_loss",
    ):
        value = debug.get(key, None)
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            return float(value)

    return None


def _sparsepcgc_actual_bit_objective_mode(args):
    mode = str(getattr(args, "sparsepcgc_actual_bit_objective", "raw")).strip().lower()
    if mode not in {"raw", "billed"}:
        mode = "raw"
    return mode


def _sparsepcgc_pick_objective_percent(args, raw_percent=None, billed_percent=None):
    mode = _sparsepcgc_actual_bit_objective_mode(args)
    raw_value = finite_float_or_none(raw_percent)
    billed_value = finite_float_or_none(billed_percent)
    if mode == "raw":
        if raw_value is not None:
            return float(raw_value), "raw"
        if billed_value is not None:
            return float(billed_value), "billed_fallback"
        return None, "raw_missing"
    if billed_value is not None:
        return float(billed_value), "billed"
    if raw_value is not None:
        return float(raw_value), "raw_fallback"
    return None, "billed_missing"








def _sparsepcgc_anchor_success_memory(args):
    memory = getattr(args, "_sparsepcgc_anchor_success_memory", None)
    if not isinstance(memory, OrderedDict):
        memory = OrderedDict()
        setattr(args, "_sparsepcgc_anchor_success_memory", memory)
    return memory


def _sparsepcgc_update_anchor_success_memory(
    args,
    *,
    cache_key,
    episode,
    global_step,
    anchor_debug,
    structure_debug,
    edit_stats,
):
    debug = {
        "anchor_success_teacher_saved": False,
        "anchor_success_teacher_percent": float("nan"),
        "anchor_success_teacher_amount": float("nan"),
        "anchor_success_memory_count": 0,
    }
    if (
        not bool(getattr(args, "sparsepcgc_anchor_success_teacher", True))
        or not cache_key
        or not isinstance(anchor_debug, dict)
    ):
        return debug

    anchor_actual_raw = finite_float_or_none(
        anchor_debug.get("compression_loss_raw", anchor_debug.get("actual_bit_percent_raw", anchor_debug.get("actual_raw_percent", None)))
    )
    if anchor_actual_raw is None or anchor_actual_raw >= -max(float(getattr(args, "sparsepcgc_anchor_success_margin", 1.0)), 0.0):
        memory = _sparsepcgc_anchor_success_memory(args)
        debug["anchor_success_memory_count"] = int(len(memory))
        return debug

    amount = finite_float_or_none(
        anchor_debug.get(
            "hard_drop_target_ratio_value",
            anchor_debug.get(
                "drop_ratio_hard",
                (edit_stats or {}).get("full_cloud_voxel_drop_ratio_percent", 0.0) / 100.0,
            ),
        )
    )
    amount = max(float(amount or 0.0), 0.0)
    ema = min(max(float(getattr(args, "sparsepcgc_anchor_success_ema", 0.20)), 1e-4), 1.0)
    memory = _sparsepcgc_anchor_success_memory(args)
    prev = memory.get(cache_key, {})
    prev_amount = finite_float_or_none(prev.get("amount", None))
    updated_amount = amount if prev_amount is None else (1.0 - ema) * float(prev_amount) + ema * amount
    memory[cache_key] = {
        "episode": int(episode) + 1,
        "global_step": int(global_step) + 1,
        "anchor_actual_raw_percent": float(anchor_actual_raw),
        "amount": float(updated_amount),
        "full_cloud_prune_ratio": finite_float_or_none((edit_stats or {}).get("full_cloud_voxel_drop_ratio_percent", None)),
        "local_prune_ratio": finite_float_or_none((edit_stats or {}).get("voxel_drop_ratio_percent", None)),
        "hard_drop_target_ratio_value": finite_float_or_none(anchor_debug.get("hard_drop_target_ratio_value", None)),
        "hard_drop_block_reason": str(anchor_debug.get("hard_drop_block_reason", (structure_debug or {}).get("hard_drop_block_reason", ""))),
        "hard_drop_target_ratio_source_id": int(
            finite_float_or_none(anchor_debug.get("hard_drop_target_ratio_source_id", (structure_debug or {}).get("hard_drop_target_ratio_source_id", 0))) or 0
        ),
        "selected_drop_count_hard": int(
            finite_float_or_none(anchor_debug.get("selected_drop_count_hard", (structure_debug or {}).get("selected_drop_count_hard", 0))) or 0
        ),
        "final_hard_drop_count": int(
            finite_float_or_none(anchor_debug.get("final_hard_drop_count", (structure_debug or {}).get("final_hard_drop_count", 0))) or 0
        ),
    }
    memory.move_to_end(cache_key)
    max_entries = max(int(getattr(args, "episode_input_common_cache_max_entries", 0)), 256)
    while len(memory) > max_entries:
        memory.popitem(last=False)
    debug.update(
        {
            "anchor_success_teacher_saved": True,
            "anchor_success_teacher_percent": float(anchor_actual_raw),
            "anchor_success_teacher_amount": float(updated_amount),
            "anchor_success_memory_count": int(len(memory)),
        }
    )
    return debug


def _sparsepcgc_anchor_success_teacher(args, cache_key):
    memory = _sparsepcgc_anchor_success_memory(args)
    entry = memory.get(cache_key, None)
    if not isinstance(entry, dict):
        return None
    return entry


def _sparsepcgc_surrogate_trust(args, comp_debug):
    debug = {
        "surrogate_trust_gate_used": False,
        "surrogate_bit_error_for_trust": float("nan"),
        "surrogate_trust_value": 1.0,
    }
    if not bool(getattr(args, "sparsepcgc_surrogate_trust_gate", True)):
        return 1.0, debug
    err = finite_float_or_none(comp_debug.get("surrogate_abs_bit_error", comp_debug.get("surrogate_bit_error", None)))
    if err is None:
        return 1.0, debug
    low = max(float(getattr(args, "sparsepcgc_surrogate_error_threshold", 10.0)), 0.0)
    high = max(float(getattr(args, "sparsepcgc_surrogate_error_disable_threshold", 13.0)), low)
    min_trust = min(max(float(getattr(args, "sparsepcgc_surrogate_min_trust", 0.0)), 0.0), 1.0)
    if err <= low:
        trust = 1.0
    elif err >= high:
        trust = min_trust
    else:
        ratio = (err - low) / max(high - low, 1e-12)
        trust = 1.0 + (min_trust - 1.0) * ratio
    debug.update(
        {
            "surrogate_trust_gate_used": True,
            "surrogate_bit_error_for_trust": float(err),
            "surrogate_trust_value": float(trust),
        }
    )
    return float(trust), debug


def _episode_input_common_cache_enabled(args):
    return bool(getattr(args, "episode_input_common_cache", False))


def _clone_input_common_cache_value_to_cpu(value, memo=None):
    """Tensor aliasを保ったまま入力共通値をCPUへ退避する。

    full cloud contextでは同じcanonical座標Tensorを互換用の複数keyから
    参照する。keyごとにcloneすると値は同じでも容量が3倍になり、全frameを
    保持できず逐次走査でLRU hitが永久に0になるため、同一objectを1回だけ複製する。
    """
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        identity = id(value)
        cached = memo.get(identity)
        if cached is not None:
            return cached
        cloned = value.detach().to(device="cpu").clone()
        memo[identity] = cloned
        return cloned
    if isinstance(value, dict):
        return {
            key: _clone_input_common_cache_value_to_cpu(item, memo)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_input_common_cache_value_to_cpu(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_input_common_cache_value_to_cpu(item, memo) for item in value)
    return value


def _clone_input_common_cache_value_to_device(value, device=None, memo=None):
    """CPU CacheのTensor aliasを壊さず、現在device用の独立値を返す。"""
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        identity = id(value)
        cached = memo.get(identity)
        if cached is not None:
            return cached
        out = value.detach().clone()
        if device is not None:
            out = out.to(device=device)
        memo[identity] = out
        return out
    if isinstance(value, dict):
        return {
            key: _clone_input_common_cache_value_to_device(item, device=device, memo=memo)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _clone_input_common_cache_value_to_device(item, device=device, memo=memo)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _clone_input_common_cache_value_to_device(item, device=device, memo=memo)
            for item in value
        )
    return value


def _estimate_input_common_cache_bytes(value, seen=None):
    """共有Tensor storageを一度だけ数え、Cache容量判定を実使用量へ合わせる。"""
    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        storage = value.untyped_storage() if hasattr(value, "untyped_storage") else value.storage()
        identity = (str(value.device), int(storage.data_ptr()))
        if identity in seen:
            return 0
        seen.add(identity)
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(
            _estimate_input_common_cache_bytes(item, seen) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_estimate_input_common_cache_bytes(item, seen) for item in value)
    return 0


def _episode_input_common_cache_store(args, cache_key, value):
    if (not _episode_input_common_cache_enabled(args)) or (not cache_key):
        return

    max_entries = int(getattr(args, "episode_input_common_cache_max_entries", 0))
    if max_entries <= 0:
        max_entries = int(getattr(args, "_episode_input_common_cache_auto_max_entries", 0))
    max_memory_mb = int(getattr(args, "episode_input_common_cache_max_memory_mb", 0))
    max_bytes = max(max_memory_mb, 0) * 1024 * 1024

    if max_entries <= 0 and max_bytes <= 0:
        return

    cache = getattr(args, "_episode_input_common_cache", None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        setattr(args, "_episode_input_common_cache", cache)

    bytes_used = int(getattr(args, "_episode_input_common_cache_bytes", 0) or 0)
    cpu_value = _clone_input_common_cache_value_to_cpu(value)
    entry_bytes = _estimate_input_common_cache_bytes(cpu_value)

    old_entry = cache.pop(cache_key, None)
    if isinstance(old_entry, dict):
        bytes_used -= int(old_entry.get("bytes", 0) or 0)

    cache[cache_key] = {
        "value": cpu_value,
        "bytes": int(entry_bytes),
    }
    cache.move_to_end(cache_key)
    bytes_used += int(entry_bytes)

    while cache and (
        (max_entries > 0 and len(cache) > max_entries)
        or (max_bytes > 0 and bytes_used > max_bytes)
    ):
        _, removed = cache.popitem(last=False)
        if isinstance(removed, dict):
            bytes_used -= int(removed.get("bytes", 0) or 0)

    setattr(args, "_episode_input_common_cache_bytes", max(int(bytes_used), 0))


def _episode_input_common_cache_fetch(args, cache_key, *, device=None, section="common"):
    if (not _episode_input_common_cache_enabled(args)) or (not cache_key):
        return None

    stats = getattr(args, "_episode_input_common_cache_stats", None)
    if not isinstance(stats, dict):
        stats = {}
        setattr(args, "_episode_input_common_cache_stats", stats)
    section_stats = stats.setdefault(section, {"hit": 0, "miss": 0})

    cache = getattr(args, "_episode_input_common_cache", None)
    if not isinstance(cache, OrderedDict):
        section_stats["miss"] += 1
        return None

    entry = cache.get(cache_key, None)
    if not isinstance(entry, dict) or "value" not in entry:
        section_stats["miss"] += 1
        return None

    cache.move_to_end(cache_key)
    section_stats["hit"] += 1
    return _clone_input_common_cache_value_to_device(entry["value"], device=device)


def _episode_input_common_cache_summary(args):
    cache = getattr(args, "_episode_input_common_cache", None)
    stats = getattr(args, "_episode_input_common_cache_stats", None)
    bytes_used = int(getattr(args, "_episode_input_common_cache_bytes", 0) or 0)

    total_hits = 0
    total_misses = 0
    section_parts = []
    if isinstance(stats, dict):
        for section_name in sorted(stats.keys()):
            item = stats.get(section_name, {}) or {}
            hit = int(item.get("hit", 0) or 0)
            miss = int(item.get("miss", 0) or 0)
            total_hits += hit
            total_misses += miss
            section_parts.append(f"{section_name}[hit={hit},miss={miss}]")

    return {
        "entries": len(cache) if isinstance(cache, OrderedDict) else 0,
        "bytes": int(bytes_used),
        "hits": int(total_hits),
        "misses": int(total_misses),
        "sections": section_parts,
    }


def _episode_input_common_cache_key(cache_key, section, **parts):
    items = [f"{cache_key}|episode_input_common={section}"]
    for name, value in parts.items():
        items.append(f"{name}={value}")
    return "|".join(items)












def _sparsepcgc_amount_outcome_memory(args):
    memory = getattr(args, "_sparsepcgc_amount_outcome_memory", None)
    if not isinstance(memory, OrderedDict):
        memory = OrderedDict()
        setattr(args, "_sparsepcgc_amount_outcome_memory", memory)
    return memory


def _sparsepcgc_amount_explore_ratios(args):
    raw_values = getattr(args, "sparsepcgc_amount_explore_ratio_values", None)
    values = []
    if isinstance(raw_values, (list, tuple)):
        for value in raw_values:
            try:
                ratio = float(value)
            except Exception:
                continue
            if math.isfinite(ratio):
                values.append(max(ratio, 0.0))
    if not values:
        values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    return sorted(set(values))




def _sparsepcgc_full_cloud_amount_memory_key(cache_key):
    return f"{str(cache_key)}|full_cloud_amount"


def _sparsepcgc_full_cloud_sequence_amount_memory(args):
    memory = getattr(args, "_sparsepcgc_full_cloud_sequence_amount_memory", None)
    if not isinstance(memory, OrderedDict):
        memory = OrderedDict()
        setattr(args, "_sparsepcgc_full_cloud_sequence_amount_memory", memory)
    return memory


def _sparsepcgc_full_cloud_sequence_amount_key(sequence_name):
    name = str(sequence_name or "").strip()
    return name if name else "__unknown_sequence__"


def _sparsepcgc_full_cloud_sequence_amount_topk(args, sequence_name):
    if not bool(getattr(args, "sparsepcgc_full_cloud_amount_sequence_memory_enable", True)):
        return []
    memory = _sparsepcgc_full_cloud_sequence_amount_memory(args)
    entry = memory.get(_sparsepcgc_full_cloud_sequence_amount_key(sequence_name), None)
    if not isinstance(entry, dict):
        return []
    items = entry.get("items", {})
    if not isinstance(items, dict) or not items:
        return []

    ranked = []
    for ratio_key, bucket in items.items():
        if not isinstance(bucket, dict):
            continue
        ratio = finite_float_or_none(bucket.get("ratio", ratio_key))
        if ratio is None or ratio <= 0.0:
            continue
        raw_score = finite_float_or_none(bucket.get("raw_score_ema", None))
        objective_score = finite_float_or_none(bucket.get("objective_score_ema", None))
        billed_score = finite_float_or_none(bucket.get("billed_score_ema", None))
        primary_score = raw_score
        primary_source = "raw"
        if primary_score is None:
            primary_score = objective_score
            primary_source = "objective"
        if primary_score is None:
            primary_score = billed_score
            primary_source = "billed"
        if primary_score is None:
            continue
        ranked.append(
            {
                "ratio": float(ratio),
                "base_class": int(case_int(bucket.get("base_class", 0), 0)),
                "base_bin": float(case_float(bucket.get("base_bin", 0.0), 0.0)),
                "residual": float(case_float(bucket.get("residual", 0.0), 0.0)),
                "objective_percent": objective_score,
                "raw_percent": raw_score,
                "billed_percent": billed_score,
                "selected_is_best": float(case_float(bucket.get("selected_is_best_ema", 0.0), 0.0)),
                "count": int(case_int(bucket.get("count", 0), 0)),
                "source_step": int(case_int(bucket.get("source_step", 0), 0)),
                "score": float(primary_score),
                "score_source": str(primary_source),
            }
        )
    ranked.sort(
        key=lambda item: (
            float(item.get("score", float("inf"))),
            -int(item.get("count", 0)),
            float(item.get("ratio", 0.0)),
        )
    )
    topk = max(int(getattr(args, "sparsepcgc_full_cloud_amount_sequence_memory_topk", 3)), 1)
    return ranked[:topk]


def _sparsepcgc_full_cloud_sequence_amount_best(args, sequence_name):
    topk = _sparsepcgc_full_cloud_sequence_amount_topk(args, sequence_name)
    return topk[0] if topk else None


def _sparsepcgc_full_cloud_sequence_baseline_memory(args):
    memory = getattr(args, "_sparsepcgc_full_cloud_sequence_baseline_memory", None)
    if not isinstance(memory, OrderedDict):
        memory = OrderedDict()
        setattr(args, "_sparsepcgc_full_cloud_sequence_baseline_memory", memory)
    return memory


def _sparsepcgc_full_cloud_sequence_baseline_get(args, sequence_name):
    memory = _sparsepcgc_full_cloud_sequence_baseline_memory(args)
    entry = memory.get(_sparsepcgc_full_cloud_sequence_amount_key(sequence_name), None)
    if not isinstance(entry, dict):
        return None
    value = finite_float_or_none(entry.get("baseline", None))
    return float(value) if value is not None else None


def _sparsepcgc_update_full_cloud_sequence_baseline(
    args,
    *,
    sequence_name,
    rd_score,
    global_step,
):
    rd_score = finite_float_or_none(rd_score)
    if rd_score is None:
        return None
    memory = _sparsepcgc_full_cloud_sequence_baseline_memory(args)
    key = _sparsepcgc_full_cloud_sequence_amount_key(sequence_name)
    entry = memory.get(key, None)
    if not isinstance(entry, dict):
        entry = {
            "baseline": float(rd_score),
            "count": 0,
            "source_step": 0,
        }
    momentum = min(
        max(float(getattr(args, "sparsepcgc_full_cloud_amount_sequence_baseline_momentum", 0.9)), 0.0),
        0.9999,
    )
    prev = finite_float_or_none(entry.get("baseline", None))
    if prev is None:
        entry["baseline"] = float(rd_score)
    else:
        entry["baseline"] = float(momentum) * float(prev) + (1.0 - float(momentum)) * float(rd_score)
    entry["count"] = int(case_int(entry.get("count", 0), 0)) + 1
    entry["source_step"] = int(global_step) + 1
    memory[key] = entry
    memory.move_to_end(key)
    max_entries = max(int(getattr(args, "episode_input_common_cache_max_entries", 0)), 256)
    while len(memory) > max_entries:
        memory.popitem(last=False)
    return float(entry["baseline"])


def _sparsepcgc_full_cloud_sequence_memory_seen_count(args, sequence_name, ratio):
    memory = _sparsepcgc_full_cloud_sequence_amount_memory(args)
    entry = memory.get(_sparsepcgc_full_cloud_sequence_amount_key(sequence_name), None)
    if not isinstance(entry, dict):
        return 0
    items = entry.get("items", {})
    if not isinstance(items, dict):
        return 0
    key = f"{float(max(min(float(ratio), 0.05), 0.0)):.6f}"
    bucket = items.get(key, None)
    if not isinstance(bucket, dict):
        return 0
    return int(case_int(bucket.get("count", 0), 0))


def _sparsepcgc_update_full_cloud_sequence_amount_memory(
    args,
    *,
    sequence_name,
    row,
    global_step,
):
    if not bool(getattr(args, "sparsepcgc_full_cloud_amount_sequence_memory_enable", True)):
        return None
    if not isinstance(row, dict) or bool(row.get("is_noop", False)):
        return None

    ratio = finite_float_or_none(row.get("final_ratio", None))
    if ratio is None or ratio <= 0.0:
        return None

    raw_percent = finite_float_or_none(
        row.get("actual_raw_percent", row.get("actual_objective_percent", None))
    )
    objective_percent = finite_float_or_none(
        row.get("actual_objective_percent", row.get("actual_percent", raw_percent))
    )
    billed_percent = finite_float_or_none(
        row.get("actual_percent", row.get("actual_objective_percent", raw_percent))
    )
    if raw_percent is None and objective_percent is None and billed_percent is None:
        return None

    memory = _sparsepcgc_full_cloud_sequence_amount_memory(args)
    sequence_key = _sparsepcgc_full_cloud_sequence_amount_key(sequence_name)
    entry = memory.get(sequence_key, None)
    if not isinstance(entry, dict):
        entry = {"items": {}}
    items = entry.setdefault("items", {})
    ratio_key = f"{float(max(min(float(ratio), 0.05), 0.0)):.6f}"
    bucket = items.get(ratio_key, None)
    if not isinstance(bucket, dict):
        bucket = {
            "ratio": float(ratio),
            "base_class": int(case_int(row.get("candidate_base_class", 0), 0)),
            "base_bin": float(case_float(row.get("candidate_base_bin", 0.0), 0.0)),
            "residual": float(case_float(row.get("candidate_residual", 0.0), 0.0)),
            "objective_score_ema": objective_percent,
            "raw_score_ema": raw_percent,
            "billed_score_ema": billed_percent,
            "selected_is_best_ema": float(bool(row.get("selected_is_best", False))),
            "count": 0,
            "source_step": 0,
        }

    momentum = min(
        max(float(getattr(args, "sparsepcgc_full_cloud_amount_sequence_memory_momentum", 0.7)), 0.0),
        0.9999,
    )

    def _ema_update(prev_value, new_value):
        prev_value = finite_float_or_none(prev_value)
        new_value = finite_float_or_none(new_value)
        if new_value is None:
            return prev_value
        if prev_value is None:
            return float(new_value)
        return float(momentum) * float(prev_value) + (1.0 - float(momentum)) * float(new_value)

    bucket["ratio"] = float(ratio)
    bucket["base_class"] = int(case_int(row.get("candidate_base_class", bucket.get("base_class", 0)), 0))
    bucket["base_bin"] = float(case_float(row.get("candidate_base_bin", bucket.get("base_bin", 0.0)), 0.0))
    bucket["residual"] = float(case_float(row.get("candidate_residual", bucket.get("residual", 0.0)), 0.0))
    bucket["objective_score_ema"] = _ema_update(bucket.get("objective_score_ema", None), objective_percent)
    bucket["raw_score_ema"] = _ema_update(bucket.get("raw_score_ema", None), raw_percent)
    bucket["billed_score_ema"] = _ema_update(bucket.get("billed_score_ema", None), billed_percent)
    bucket["selected_is_best_ema"] = _ema_update(
        bucket.get("selected_is_best_ema", None),
        float(bool(row.get("selected_is_best", False))),
    )
    bucket["count"] = int(case_int(bucket.get("count", 0), 0)) + 1
    bucket["source_step"] = int(global_step) + 1
    items[ratio_key] = bucket
    memory[sequence_key] = entry
    memory.move_to_end(sequence_key)
    max_entries = max(int(getattr(args, "episode_input_common_cache_max_entries", 0)), 256)
    while len(memory) > max_entries:
        memory.popitem(last=False)
    return bucket


def _sparsepcgc_quantize_amount_ratio(args, ratio):
    try:
        ratio_value = float(ratio)
    except Exception:
        return None
    if not math.isfinite(ratio_value):
        return None
    candidates = _sparsepcgc_amount_explore_ratios(args)
    if not candidates:
        return max(ratio_value, 0.0)
    return min(candidates, key=lambda candidate: abs(float(candidate) - ratio_value))


def _sparsepcgc_amount_outcome_teacher(args, memory_key):
    if not bool(getattr(args, "sparsepcgc_amount_outcome_memory", True)):
        return None
    if not memory_key:
        return None
    memory = _sparsepcgc_amount_outcome_memory(args)
    entry = memory.get(memory_key, None)
    if not isinstance(entry, dict):
        return None
    items = entry.get("items", {})
    if not isinstance(items, dict) or not items:
        return None

    min_good_count = max(int(getattr(args, "sparsepcgc_amount_memory_min_count_for_exploit", 1)), 1)
    best_entry = None
    best_tuple = None
    for ratio_key, bucket in items.items():
        if not isinstance(bucket, dict):
            continue
        try:
            ratio = float(bucket.get("ratio", ratio_key))
            success_ema = float(bucket.get("success_ema", 0.0))
            bad_ema = float(bucket.get("bad_ema", 0.0))
            good_count = int(bucket.get("good_count", 0))
            bad_count = int(bucket.get("bad_count", 0))
            count = int(bucket.get("count", good_count + bad_count))
            best_percent = float(bucket.get("best_percent", 0.0))
        except Exception:
            continue
        if not math.isfinite(ratio) or ratio <= 0.0 or good_count < min_good_count:
            continue
        score = success_ema - bad_ema
        ranking = (score, success_ema, good_count - bad_count, -best_percent, ratio)
        if best_tuple is None or ranking > best_tuple:
            best_tuple = ranking
            best_entry = {
                "ratio": float(ratio),
                "score": float(score),
                "good_count": int(good_count),
                "bad_count": int(bad_count),
                "count": int(count),
                "best_percent": float(best_percent),
                "bucket_count": int(len(items)),
            }
    return best_entry


def _sparsepcgc_apply_amount_outcome_context(args, *, memory_key=None, forward_key=None):
    teacher = _sparsepcgc_amount_outcome_teacher(args, memory_key)
    setattr(args, "_current_sparsepcgc_amount_memory_key", str(memory_key or ""))
    setattr(args, "_current_sparsepcgc_forward_key", str(forward_key or memory_key or ""))
    setattr(
        args,
        "_current_sparsepcgc_amount_outcome_teacher_ratio",
        float(teacher["ratio"]) if isinstance(teacher, dict) else float("nan"),
    )
    setattr(
        args,
        "_current_sparsepcgc_amount_outcome_teacher_score",
        float(teacher["score"]) if isinstance(teacher, dict) else float("nan"),
    )
    setattr(
        args,
        "_current_sparsepcgc_amount_outcome_teacher_count",
        int(teacher["good_count"]) if isinstance(teacher, dict) else 0,
    )
    setattr(
        args,
        "_current_sparsepcgc_amount_outcome_teacher_bad_count",
        int(teacher["bad_count"]) if isinstance(teacher, dict) else 0,
    )
    return teacher


def _sparsepcgc_update_amount_outcome_memory(args, memory_key, actual_percent, used_ratio):
    debug = {
        "amount_outcome_memory_saved": False,
        "amount_outcome_memory_label_id": 0,
        "amount_outcome_memory_used_ratio": float("nan"),
        "amount_outcome_memory_bucket_ratio": float("nan"),
        "amount_outcome_memory_best_ratio": float("nan"),
        "amount_outcome_memory_best_score": float("nan"),
        "amount_outcome_memory_best_count": 0,
        "amount_outcome_memory_good_count": 0,
        "amount_outcome_memory_bad_count": 0,
        "amount_outcome_memory_entry_count": 0,
    }
    if not bool(getattr(args, "sparsepcgc_amount_outcome_memory", True)):
        return debug
    if not memory_key:
        return debug

    try:
        actual_percent = float(actual_percent)
        used_ratio = float(used_ratio)
    except Exception:
        teacher = _sparsepcgc_amount_outcome_teacher(args, memory_key)
        if isinstance(teacher, dict):
            debug.update(
                {
                    "amount_outcome_memory_best_ratio": float(teacher.get("ratio", float("nan"))),
                    "amount_outcome_memory_best_score": float(teacher.get("score", float("nan"))),
                    "amount_outcome_memory_best_count": int(teacher.get("good_count", 0)),
                    "amount_outcome_memory_good_count": int(teacher.get("good_count", 0)),
                    "amount_outcome_memory_bad_count": int(teacher.get("bad_count", 0)),
                    "amount_outcome_memory_entry_count": int(teacher.get("bucket_count", 0)),
                }
            )
        return debug

    if not (math.isfinite(actual_percent) and math.isfinite(used_ratio) and used_ratio > 0.0):
        teacher = _sparsepcgc_amount_outcome_teacher(args, memory_key)
        if isinstance(teacher, dict):
            debug.update(
                {
                    "amount_outcome_memory_best_ratio": float(teacher.get("ratio", float("nan"))),
                    "amount_outcome_memory_best_score": float(teacher.get("score", float("nan"))),
                    "amount_outcome_memory_best_count": int(teacher.get("good_count", 0)),
                    "amount_outcome_memory_good_count": int(teacher.get("good_count", 0)),
                    "amount_outcome_memory_bad_count": int(teacher.get("bad_count", 0)),
                    "amount_outcome_memory_entry_count": int(teacher.get("bucket_count", 0)),
                }
            )
        return debug

    good_margin = max(float(getattr(args, "sparsepcgc_amount_outcome_good_margin", 0.25)), 0.0)
    bad_margin = max(float(getattr(args, "sparsepcgc_amount_outcome_bad_margin", 0.25)), 0.0)
    if actual_percent < -good_margin:
        label_id = 1
    elif actual_percent > bad_margin:
        label_id = 3
    else:
        label_id = 2

    bucket_ratio = _sparsepcgc_quantize_amount_ratio(args, used_ratio)
    debug["amount_outcome_memory_label_id"] = int(label_id)
    debug["amount_outcome_memory_used_ratio"] = float(used_ratio)
    debug["amount_outcome_memory_bucket_ratio"] = (
        float(bucket_ratio) if bucket_ratio is not None and math.isfinite(float(bucket_ratio)) else float("nan")
    )

    memory = _sparsepcgc_amount_outcome_memory(args)
    entry = memory.get(memory_key, None)
    if not isinstance(entry, dict):
        entry = {"items": {}}
    items = entry.setdefault("items", {})
    bucket_key = None if bucket_ratio is None else f"{float(bucket_ratio):.6f}"
    bucket = items.get(bucket_key, None) if bucket_key is not None else None
    if not isinstance(bucket, dict):
        bucket = {
            "ratio": float(bucket_ratio if bucket_ratio is not None else used_ratio),
            "success_ema": 0.0,
            "bad_ema": 0.0,
            "count": 0,
            "good_count": 0,
            "bad_count": 0,
            "best_percent": float("inf"),
            "last_percent": float("nan"),
        }

    ema = min(max(float(getattr(args, "sparsepcgc_amount_outcome_memory_ema", 0.20)), 1e-4), 1.0)
    bucket["ratio"] = float(bucket_ratio if bucket_ratio is not None else used_ratio)
    bucket["count"] = int(bucket.get("count", 0)) + 1
    bucket["last_percent"] = float(actual_percent)
    if label_id == 1:
        bucket["good_count"] = int(bucket.get("good_count", 0)) + 1
        bucket["success_ema"] = (1.0 - ema) * float(bucket.get("success_ema", 0.0)) + ema * 1.0
        bucket["bad_ema"] = (1.0 - ema) * float(bucket.get("bad_ema", 0.0))
        bucket["best_percent"] = min(float(bucket.get("best_percent", float("inf"))), float(actual_percent))
        debug["amount_outcome_memory_saved"] = True
    elif label_id == 3:
        bucket["bad_count"] = int(bucket.get("bad_count", 0)) + 1
        bucket["success_ema"] = (1.0 - ema) * float(bucket.get("success_ema", 0.0))
        bucket["bad_ema"] = (1.0 - ema) * float(bucket.get("bad_ema", 0.0)) + ema * 1.0
        bucket["best_percent"] = min(float(bucket.get("best_percent", float("inf"))), float(actual_percent))
        debug["amount_outcome_memory_saved"] = True
    items[bucket_key] = bucket
    memory[memory_key] = entry
    memory.move_to_end(memory_key)
    max_entries = max(int(getattr(args, "episode_input_common_cache_max_entries", 0)), 256)
    while len(memory) > max_entries:
        memory.popitem(last=False)

    teacher = _sparsepcgc_amount_outcome_teacher(args, memory_key)
    if isinstance(teacher, dict):
        debug.update(
            {
                "amount_outcome_memory_best_ratio": float(teacher.get("ratio", float("nan"))),
                "amount_outcome_memory_best_score": float(teacher.get("score", float("nan"))),
                "amount_outcome_memory_best_count": int(teacher.get("good_count", 0)),
                "amount_outcome_memory_good_count": int(teacher.get("good_count", 0)),
                "amount_outcome_memory_bad_count": int(teacher.get("bad_count", 0)),
                "amount_outcome_memory_entry_count": int(teacher.get("bucket_count", len(items))),
            }
        )
    else:
        debug["amount_outcome_memory_entry_count"] = int(len(items))

    return debug












def _sparsepcgc_full_cloud_amount_bins(args):
    values = getattr(args, "sparsepcgc_full_cloud_amount_bin_values", None)
    if not isinstance(values, (list, tuple)) or not values:
        values = (0.0, 0.015, 0.021, 0.026, 0.031, 0.038, 0.044, 0.05)
    out = [0.0]
    for value in values:
        try:
            out.append(min(max(float(value), 0.0), 0.05))
        except Exception:
            continue
    return tuple(sorted(set(out)))


def _sample_full_cloud_amount_geom_points(points, max_points):
    if not torch.is_tensor(points):
        return points
    try:
        max_points = int(max_points)
    except Exception:
        max_points = 0
    if max_points <= 0 or points.dim() < 3:
        return points
    n_points = int(points.shape[-1])
    if n_points <= max_points:
        return points
    # Deterministic uniform sampling keeps step-to-step comparisons stable.
    idx = torch.linspace(
        0,
        n_points - 1,
        steps=max_points,
        device=points.device,
    ).round().long().clamp_(0, n_points - 1)
    return points.index_select(-1, idx).contiguous()






def _build_sparsepcgc_full_cloud_amount_candidate_teacher_loss(
    args,
    amount_terms,
    *,
    compression_debug,
    structure_debug,
    loss_obj,
    base_model,
    full_cloud_context,
    gt_xyz,
    actual_percent,
    actual_available,
    cache_key,
    global_step,
    episode,
    epoch,
    step,
    sequence_name,
    input_points,
    drop_count,
    geom_loss=None,
):
    objective_mode = _sparsepcgc_actual_bit_objective_mode(args)
    debug = {
        "sparsepcgc_training_mode": str(getattr(args, "sparsepcgc_training_mode", "subtree_selector")),
        "full_cloud_amount_enabled": False,
        "full_cloud_amount_input_points": int(input_points or 0),
        "full_cloud_amount_bin": float("nan"),
        "full_cloud_amount_residual": float("nan"),
        "full_cloud_amount_pred_residual": float("nan"),
        "full_cloud_amount_pred_residual_raw": float("nan"),
        "full_cloud_amount_selected_base_bin": float("nan"),
        "full_cloud_amount_selected_residual": float("nan"),
        "full_cloud_amount_final_ratio": float("nan"),
        "full_cloud_amount_drop_count": int(drop_count or 0),
        "full_cloud_amount_noop_selected": False,
        "full_cloud_amount_candidate_count": 0,
        "full_cloud_amount_actual_eval_count": 0,
        "full_cloud_amount_actual_requested_count": 0,
        "full_cloud_amount_actual_finished_count": 0,
        "full_cloud_amount_teacher_source": "none",
        "full_cloud_amount_teacher_ratio": float("nan"),
        "full_cloud_amount_teacher_base_bin": float("nan"),
        "full_cloud_amount_teacher_residual": float("nan"),
        "full_cloud_amount_oracle_best_ratio": float("nan"),
        "full_cloud_amount_raw_oracle_best_ratio": float("nan"),
        "full_cloud_amount_oracle_best_actual_delta": float("nan"),
        "full_cloud_amount_selected_ratio": float("nan"),
        "full_cloud_amount_selected_actual_delta": float("nan"),
        "full_cloud_amount_oracle_gap": float("nan"),
        "full_cloud_amount_selected_is_best": False,
        "full_cloud_amount_selected_is_raw_best": False,
        "full_cloud_amount_raw_oracle_gap": float("nan"),
        "full_cloud_amount_actual_finished_nonselected_count": 0,
        "full_cloud_amount_wide_probe_due": False,
        "full_cloud_amount_wide_probe_actual_count": 0,
        "full_cloud_amount_sequence_memory_ratio": float("nan"),
        "amount_learning_mode": str(
            getattr(args, "sparsepcgc_full_cloud_amount_learning_mode", "network_selected_bandit")
        ),
        "selected_amount_class": -1,
        "selected_amount_bin": float("nan"),
        "selected_amount_ratio": float("nan"),
        "selected_action_log_prob": float("nan"),
        "amount_temperature": float("nan"),
        "amount_rd_score": float("nan"),
        "amount_policy_loss": 0.0,
        "amount_value_loss": 0.0,
        "amount_advantage": float("nan"),
        "sequence_amount_baseline": float("nan"),
        "amount_class_histogram": "",
        "amount_max_class_rate": float("nan"),
        "amount_selected_ratio_mean": float("nan"),
        "amount_selected_ratio_std": float("nan"),
        "full_cloud_amount_entropy": float("nan"),
        "full_cloud_amount_entropy_loss": 0.0,
        "full_cloud_amount_residual_loss": 0.0,
        "full_cloud_amount_residual_error": float("nan"),
        "full_cloud_amount_residual_enabled": bool(
            getattr(args, "sparsepcgc_full_cloud_amount_residual_enable", True)
        ),
        "full_cloud_amount_residual_max": float(
            getattr(args, "sparsepcgc_full_cloud_amount_residual_max", 0.0025)
        ),
        "full_cloud_amount_residual_teacher_clamped": False,
        "full_cloud_amount_ratio_hist_selected": "",
        "full_cloud_amount_ratio_hist_teacher": "",
        "full_cloud_amount_fine_probe_enabled": bool(
            getattr(args, "sparsepcgc_full_cloud_amount_fine_ratio_probe_enable", True)
        ),
        "full_cloud_amount_residual_probe_enabled": bool(
            getattr(args, "sparsepcgc_full_cloud_amount_residual_probe_enable", True)
        ),
        "full_cloud_amount_actual_wall_time_total": 0.0,
        "full_cloud_amount_actual_wall_time_max": 0.0,
        "full_cloud_amount_actual_dispatch_time": 0.0,
        "full_cloud_amount_actual_gather_time": 0.0,
        "full_cloud_amount_reuse_where_ranking": False,
        "full_cloud_amount_reuse_where_ranking_reason": "",
        "where_mode": str(getattr(args, "sparsepcgc_where_mode", "block_only")),
        "macro_ratio": 0.0,
        "micro_ratio": 0.0,
        "macro_selected_block_count": 0,
        "macro_drop_count": 0,
        "micro_drop_count": 0,
        "total_drop_count": 0,
        "selected_block_count": 0,
        "micro_selected_block_count": 0,
        "max_drop_count_per_block": 0,
        "mean_drop_count_per_selected_block": 0.0,
        "drop_concentration_top1_block_ratio": 0.0,
        "drop_concentration_top5_block_ratio": 0.0,
        "hard_where_uses_network_score": False,
        "heuristic_where_score_mean": 0.0,
        "heuristic_where_score_std": 0.0,
        "micro_quota_hit_block_count": 0,
        "micro_min_selected_blocks": 0,
        "micro_candidate_block_count": 0,
        "micro_min_blocks_satisfied": False,
        "micro_min_blocks_fallback_reason": "",
        "macro_micro_hybrid_fallback": False,
        "effective_where_mode": "",
        "macro_disabled_reason": "",
        "full_cloud_amount_predicted_delta": float("nan"),
        "full_cloud_amount_actual_delta": float("nan"),
        "full_cloud_amount_actual_objective_delta": float("nan"),
        "full_cloud_amount_surrogate_delta": float("nan"),
        "full_cloud_amount_geom_loss": float("nan"),
        "full_cloud_amount_cls_loss": 0.0,
        "full_cloud_amount_value_loss": 0.0,
        "full_cloud_amount_rank_loss": 0.0,
        "full_cloud_amount_geom_guard_loss": 0.0,
        "full_cloud_amount_ratio_reg_loss": 0.0,
        "full_cloud_amount_noop_guard_loss": 0.0,
        "full_cloud_amount_total_loss": 0.0,
        "full_cloud_verified_noop_guard_used": False,
        "sparsepcgc_actual_parallel_mode": str(getattr(args, "sparsepcgc_actual_parallel_mode", "single")),
        "sparsepcgc_actual_parallel_candidates": int(
            max(int(getattr(args, "sparsepcgc_actual_parallel_candidates", 1)), 1)
        ),
        "sparsepcgc_actual_worker_pool_used": False,
        "actual_bit_objective": str(objective_mode),
        "actual_objective_percent": float("nan"),
        "actual_objective_bit_source": "",
    }
    rows = []
    if str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() == "ana_den6_online":
        # ana_den6_onlineはActuatorが1つのAmount/Action/Where planを直接決める。
        # 旧full-cloud amount teacherのbest-of-N actual探索は主経路へ混入させない。
        debug["full_cloud_amount_teacher_source"] = "disabled_for_ana_den6_online_one_plan"
        return None, debug, rows
    if not isinstance(amount_terms, dict):
        return None, debug, rows

    amount_logits = amount_terms.get("full_cloud_amount_bin_logits", None)
    pred_per_amount = amount_terms.get("full_cloud_amount_predicted_delta_per_amount", None)
    residual_raw_tensor = amount_terms.get(
        "full_cloud_amount_residual_raw",
        amount_terms.get("amount_residual_raw", None),
    )
    if not (torch.is_tensor(amount_logits) and torch.is_tensor(pred_per_amount)):
        return None, debug, rows

    amount_logits = (
        amount_logits.reshape(-1)
        if amount_logits.dim() <= 1
        else amount_logits.reshape(-1, amount_logits.shape[-1])[0]
    )
    pred_per_amount = (
        pred_per_amount.reshape(-1)
        if pred_per_amount.dim() <= 1
        else pred_per_amount.reshape(-1, pred_per_amount.shape[-1])[0]
    )
    bins = _sparsepcgc_full_cloud_amount_bins(args)
    class_count = min(int(amount_logits.numel()), int(pred_per_amount.numel()), len(bins))
    if class_count <= 0:
        return None, debug, rows

    amount_logits = amount_logits[:class_count]
    pred_per_amount = pred_per_amount[:class_count]
    ref = amount_logits
    bin_tensor = ref.new_tensor(list(bins[:class_count]))
    learning_mode = str(
        getattr(args, "sparsepcgc_full_cloud_amount_learning_mode", "network_selected_bandit")
    ).strip().lower()
    if learning_mode not in {"multi_actual_teacher", "network_selected_bandit"}:
        learning_mode = "network_selected_bandit"
    selected_class = case_int(
        amount_terms.get("full_cloud_amount_selected_class", None),
        int(torch.argmax(amount_logits.detach()).item()),
    )
    selected_class = min(max(int(selected_class), 0), class_count - 1)
    selected_bin = bin_tensor[selected_class]
    action_sample_mode = str(
        amount_terms.get(
            "full_cloud_amount_action_sample_mode",
            getattr(args, "sparsepcgc_full_cloud_amount_action_sample_mode", "categorical"),
        )
    ).strip().lower()
    if action_sample_mode not in {"argmax", "categorical", "gumbel"}:
        action_sample_mode = "categorical"
    amount_temperature = case_float(
        amount_terms.get(
            "full_cloud_amount_action_temperature",
            getattr(args, "sparsepcgc_full_cloud_amount_exploration_temperature", 1.0),
        ),
        1.0,
    )
    amount_temperature = max(float(amount_temperature), 1e-6)
    policy_logits = amount_logits.float() / float(amount_temperature)
    policy_log_probs = torch.log_softmax(policy_logits, dim=0)
    policy_probs = torch.softmax(policy_logits, dim=0)
    selected_log_prob_t = policy_log_probs[selected_class]
    selected_action_log_prob = float(selected_log_prob_t.detach().cpu())
    amount_entropy_t = -(policy_probs * policy_log_probs).sum()
    amount_class_histogram = ",".join(
        f"{float(value):.4f}" for value in policy_probs.detach().cpu().tolist()
    )
    amount_max_class_rate = float(policy_probs.detach().max().cpu()) if policy_probs.numel() > 0 else float("nan")

    residual_enable = bool(getattr(args, "sparsepcgc_full_cloud_amount_residual_enable", True))
    residual_max = min(
        max(float(getattr(args, "sparsepcgc_full_cloud_amount_residual_max", 0.0025)), 0.0),
        0.01,
    )
    residual_loss_weight = max(
        float(getattr(args, "sparsepcgc_full_cloud_amount_residual_loss_weight", 1.0)),
        0.0,
    )
    if torch.is_tensor(residual_raw_tensor):
        pred_residual_raw_t = residual_raw_tensor.reshape(-1)[0].to(device=ref.device, dtype=ref.dtype)
    else:
        pred_residual_raw_t = ref.new_zeros(())
    pred_residual_raw_t = torch.nan_to_num(pred_residual_raw_t, nan=0.0, posinf=0.0, neginf=0.0)
    pred_residual_t = torch.tanh(pred_residual_raw_t) * float(residual_max)
    if selected_class == 0 or (not residual_enable):
        pred_residual_t = pred_residual_t * 0.0
    final_ratio = torch.clamp(selected_bin + pred_residual_t, 0.0, 0.05)
    if selected_class == 0:
        final_ratio = final_ratio * 0.0

    pred_residual_float = float(pred_residual_t.detach().float().cpu())
    pred_residual_raw_float = float(pred_residual_raw_t.detach().float().cpu())
    selected_ratio_value = float(final_ratio.detach().cpu())
    selected_surrogate = float(pred_per_amount.detach().flatten()[selected_class].cpu())
    actual_value = finite_float_or_none(actual_percent)
    actual_available = bool(actual_available and actual_value is not None and math.isfinite(float(actual_value)))
    noop_margin = max(float(getattr(args, "sparsepcgc_full_cloud_amount_noop_margin", 0.0)), 0.0)
    policy = str(
        getattr(
            args,
            "sparsepcgc_full_cloud_amount_actual_candidate_policy",
            "selected_plus_surrogate_topk",
        )
    ).strip().lower()
    multi_actual_enable = bool(getattr(args, "sparsepcgc_full_cloud_amount_multi_actual_enable", True))
    max_actual_default = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_max_actual_candidates_per_step", 2)),
        1,
    )
    warmup_actual_max = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_warmup_max_actual_candidates_per_step", 4)),
        1,
    )
    warmup_steps = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_multi_actual_warmup_steps", 100)),
        0,
    )
    in_multi_actual_warmup = int(global_step) < int(warmup_steps)
    max_actual = warmup_actual_max if in_multi_actual_warmup else max_actual_default
    if (not multi_actual_enable) or policy == "selected_only" or max_actual_default <= 1:
        max_actual = 1
    actual_topk = max(int(getattr(args, "sparsepcgc_full_cloud_amount_actual_topk", 2)), 0)
    wide_probe_enable = bool(
        getattr(args, "sparsepcgc_full_cloud_amount_wide_probe_enable", True)
    )
    wide_probe_ratios = tuple(
        getattr(args, "sparsepcgc_full_cloud_amount_wide_probe_ratio_values", ())
    )
    wide_probe_interval = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_wide_probe_interval", 50)),
        0,
    )
    wide_probe_sequence_head_steps = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_wide_probe_sequence_head_steps", 2)),
        0,
    )
    wide_probe_max_actual = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_wide_probe_max_actual", 3)),
        1,
    )
    wide_probe_due = bool(
        wide_probe_enable
        and multi_actual_enable
        and (
            int(step) < int(wide_probe_sequence_head_steps)
            or (
                wide_probe_interval > 0
                and ((int(global_step) + 1) % int(wide_probe_interval) == 0)
            )
        )
    )
    diagnostic_sweep_interval = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_diagnostic_sweep_interval", 0)),
        0,
    )
    diagnostic_sweep_due = bool(
        learning_mode == "network_selected_bandit"
        and diagnostic_sweep_interval > 0
        and ((int(global_step) + 1) % int(diagnostic_sweep_interval) == 0)
    )
    sequence_memory_enable = bool(
        getattr(args, "sparsepcgc_full_cloud_amount_sequence_memory_enable", True)
    )
    teacher_actual_priority = bool(
        getattr(args, "sparsepcgc_full_cloud_amount_teacher_actual_priority", True)
    )
    oracle_sweep_interval = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_oracle_sweep_interval", 0)),
        0,
    )
    oracle_sweep_max_bins = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_oracle_sweep_max_bins", class_count)),
        1,
    )
    oracle_sweep_due = bool(
        oracle_sweep_interval > 0 and ((int(global_step) + 1) % int(oracle_sweep_interval) == 0)
    )
    residual_probe_enable = bool(
        getattr(args, "sparsepcgc_full_cloud_amount_residual_probe_enable", True)
    )
    residual_probe_offsets = tuple(
        getattr(args, "sparsepcgc_full_cloud_amount_residual_probe_offset_values", (-1.0, 0.0, 1.0))
    )
    fine_ratio_probe_enable = bool(
        getattr(args, "sparsepcgc_full_cloud_amount_fine_ratio_probe_enable", True)
    )
    fine_ratio_values = tuple(
        getattr(args, "sparsepcgc_full_cloud_amount_fine_ratio_values", ())
    )
    fine_ratio_warmup_steps = max(
        int(getattr(args, "sparsepcgc_full_cloud_amount_fine_ratio_warmup_steps", 200)),
        0,
    )
    residual_teacher_mode = str(
        getattr(args, "sparsepcgc_full_cloud_amount_residual_teacher_mode", "candidate_ratio")
    ).strip().lower()
    if residual_teacher_mode not in {"candidate_ratio", "nearest_bin"}:
        residual_teacher_mode = "candidate_ratio"
    amount_memory_key = _sparsepcgc_full_cloud_amount_memory_key(cache_key)
    sequence_memory_entries = (
        _sparsepcgc_full_cloud_sequence_amount_topk(args, sequence_name)
        if sequence_memory_enable
        else []
    )
    sequence_memory_best_entry = sequence_memory_entries[0] if sequence_memory_entries else None
    amount_prob = torch.softmax(amount_logits.detach().float(), dim=0)

    nonnoop_bin_items = [
        (int(cls), float(bin_tensor.detach().flatten()[cls].cpu()))
        for cls in range(1, class_count)
    ]

    def _clamp_ratio(ratio):
        try:
            ratio_value = float(ratio)
        except Exception:
            return 0.0
        if not math.isfinite(ratio_value):
            return 0.0
        return min(max(ratio_value, 0.0), 0.05)

    def _best_base_for_ratio(ratio):
        ratio_value = _clamp_ratio(ratio)
        if ratio_value <= 0.0 or not nonnoop_bin_items:
            return 0, 0.0, 0.0
        best_cls = 0
        best_bin = 0.0
        best_gap = None
        for cls, base_bin in nonnoop_bin_items:
            gap = abs(base_bin - ratio_value)
            if best_gap is None or gap < best_gap:
                best_cls = int(cls)
                best_bin = float(base_bin)
                best_gap = float(gap)
        return int(best_cls), float(best_bin), float(ratio_value - best_bin)

    candidate_specs = OrderedDict()

    def _register_candidate_ratio(final_ratio_value, source, priority, *, base_class=None, base_bin=None, candidate_residual=None):
        ratio_value = _clamp_ratio(final_ratio_value)
        if ratio_value <= 0.0:
            key = "0:0.000000"
            entry = candidate_specs.get(key, None)
            if entry is None:
                candidate_specs[key] = {
                    "key": key,
                    "is_noop": True,
                    "final_ratio": 0.0,
                    "base_class": 0,
                    "base_bin": 0.0,
                    "candidate_residual": 0.0,
                    "sources": [str(source)],
                    "priority": int(priority),
                }
            else:
                if str(source) not in entry["sources"]:
                    entry["sources"].append(str(source))
                entry["priority"] = min(int(entry["priority"]), int(priority))
            return key

        if base_class is None or int(base_class) <= 0 or int(base_class) >= class_count:
            base_class, auto_base_bin, auto_residual = _best_base_for_ratio(ratio_value)
            if base_class <= 0:
                return _register_candidate_ratio(0.0, source, priority)
            if base_bin is None:
                base_bin = auto_base_bin
            if candidate_residual is None:
                candidate_residual = auto_residual
        else:
            base_class = min(max(int(base_class), 1), class_count - 1)
            if base_bin is None:
                base_bin = float(bin_tensor.detach().flatten()[base_class].cpu())
            if candidate_residual is None:
                candidate_residual = float(ratio_value - float(base_bin))

        base_bin = float(base_bin)
        candidate_residual = float(candidate_residual)
        key = f"{int(base_class)}:{ratio_value:.6f}"
        entry = candidate_specs.get(key, None)
        if entry is None:
            candidate_specs[key] = {
                "key": key,
                "is_noop": False,
                "final_ratio": float(ratio_value),
                "base_class": int(base_class),
                "base_bin": float(base_bin),
                "candidate_residual": float(candidate_residual),
                "sources": [str(source)],
                "priority": int(priority),
            }
        else:
            if str(source) not in entry["sources"]:
                entry["sources"].append(str(source))
            entry["priority"] = min(int(entry["priority"]), int(priority))
        return key

    selected_candidate_key = _register_candidate_ratio(0.0, "noop", 0)
    selected_candidate_key = _register_candidate_ratio(
        selected_ratio_value,
        "network_selected",
        0,
        base_class=selected_class,
        base_bin=float(selected_bin.detach().cpu()),
        candidate_residual=pred_residual_float,
    )

    def _register_bin_class(cls, source, priority):
        cls = min(max(int(cls), 0), class_count - 1)
        if cls <= 0:
            return _register_candidate_ratio(0.0, source, priority)
        return _register_candidate_ratio(
            float(bin_tensor.detach().flatten()[cls].cpu()),
            source,
            priority,
            base_class=cls,
            base_bin=float(bin_tensor.detach().flatten()[cls].cpu()),
            candidate_residual=0.0,
        )

    if policy in {
        "selected_plus_neighbors",
        "selected_plus_surrogate_topk",
        "selected_neighbors_memory_surrogate",
    } and selected_class > 0:
        _register_bin_class(max(selected_class - 1, 0), "neighbor_left", 4)
        _register_bin_class(min(selected_class + 1, class_count - 1), "neighbor_right", 4)

    if policy in {"selected_plus_surrogate_topk", "selected_neighbors_memory_surrogate", "all_bins"}:
        sorted_surrogate_classes = sorted(
            range(1, class_count),
            key=lambda cls: float(pred_per_amount.detach().flatten()[cls].cpu()),
        )
        surrogate_take = int(actual_topk) if policy == "selected_plus_surrogate_topk" else 1
        for cls in sorted_surrogate_classes[: max(surrogate_take, 0)]:
            _register_bin_class(int(cls), "surrogate_topk", 3)

    amount_teacher_entry = _sparsepcgc_amount_outcome_teacher(args, amount_memory_key)
    anchor_teacher_entry = _sparsepcgc_anchor_success_teacher(args, cache_key)
    if policy == "selected_neighbors_memory_surrogate":
        if isinstance(amount_teacher_entry, dict):
            memory_ratio = amount_teacher_entry.get("ratio", float("nan"))
            _register_candidate_ratio(memory_ratio, "memory_best", 2)
            if residual_probe_enable and residual_enable:
                for offset in residual_probe_offsets:
                    _register_candidate_ratio(
                        _clamp_ratio(float(memory_ratio) + float(offset) * float(residual_max)),
                        f"memory_residual_probe_{offset:+.2f}",
                        2,
                    )
        if isinstance(anchor_teacher_entry, dict):
            anchor_ratio = anchor_teacher_entry.get("amount", float("nan"))
            _register_candidate_ratio(anchor_ratio, "anchor_success", 2)
            if residual_probe_enable and residual_enable:
                for offset in residual_probe_offsets:
                    _register_candidate_ratio(
                        _clamp_ratio(float(anchor_ratio) + float(offset) * float(residual_max)),
                        f"anchor_residual_probe_{offset:+.2f}",
                        2,
                    )

    if sequence_memory_enable and sequence_memory_entries:
        for mem_rank, memory_entry in enumerate(sequence_memory_entries):
            memory_ratio = memory_entry.get("ratio", float("nan"))
            memory_base_class = memory_entry.get("base_class", None)
            memory_base_bin = memory_entry.get("base_bin", None)
            memory_residual = memory_entry.get("residual", None)
            source = "sequence_memory_best" if mem_rank == 0 else f"sequence_memory_top{mem_rank + 1}"
            _register_candidate_ratio(
                memory_ratio,
                source,
                1 if mem_rank == 0 else 2,
                base_class=memory_base_class,
                base_bin=memory_base_bin,
                candidate_residual=memory_residual,
            )

    if policy == "all_bins":
        for cls in range(1, class_count):
            _register_bin_class(int(cls), "all_bins", 5)

    if in_multi_actual_warmup:
        for warm_ratio in (0.015, 0.026, 0.031, 0.038, 0.05):
            _register_candidate_ratio(warm_ratio, "warmup_probe", 5)

    if residual_probe_enable and residual_enable and multi_actual_enable and selected_class > 0:
        for offset in residual_probe_offsets:
            _register_candidate_ratio(
                _clamp_ratio(float(selected_bin.detach().cpu()) + float(offset) * float(residual_max)),
                (
                    "selected_residual_probe_zero"
                    if abs(float(offset)) <= 1e-12
                    else ("selected_residual_probe_neg" if float(offset) < 0.0 else "selected_residual_probe_pos")
                ),
                1,
                base_class=selected_class,
                base_bin=float(selected_bin.detach().cpu()),
                candidate_residual=float(offset) * float(residual_max),
            )

    if fine_ratio_probe_enable and multi_actual_enable and residual_enable:
        for fine_ratio in fine_ratio_values:
            _register_candidate_ratio(float(fine_ratio), f"fine_ratio_probe_{float(fine_ratio):.3f}", 1)

    if wide_probe_due:
        for wide_ratio in wide_probe_ratios:
            _register_candidate_ratio(float(wide_ratio), f"wide_probe_{float(wide_ratio):.3f}", 2)

    if oracle_sweep_due:
        for cls in range(1, min(class_count, oracle_sweep_max_bins + 1)):
            _register_bin_class(int(cls), "oracle_sweep", 1)

    structure_debug = dict(structure_debug or {})
    compression_debug = dict(compression_debug or {})
    base_bit = finite_float_or_none(
        compression_debug.get("gt_actual_bit", compression_debug.get("gt_bit_abs", None))
    )
    if (base_bit is None or not math.isfinite(base_bit) or base_bit <= 0.0) and hasattr(loss_obj, "_get_cached_actual_gt"):
        cached_gt = loss_obj._get_cached_actual_gt(cache_key)
        if isinstance(cached_gt, dict):
            base_bit = finite_float_or_none(cached_gt.get("bit", None))
    if (base_bit is None or not math.isfinite(base_bit) or base_bit <= 0.0) and torch.is_tensor(gt_xyz):
        try:
            cached_gt = loss_obj._encode_actual_batch(args, gt_xyz)
            if isinstance(cached_gt, dict):
                base_bit = finite_float_or_none(cached_gt.get("bit", None))
        except Exception:
            base_bit = None
    if base_bit is None or not math.isfinite(base_bit) or base_bit <= 0.0:
        base_bit = None

    voxel_state = getattr(base_model, "last_actuator_voxel_state", None)
    actuator = getattr(base_model, "actuator", None)
    initial_voxel_coords = None
    if isinstance(voxel_state, dict):
        for key in ("initial_voxel_coords", "point_aligned_initial_voxel_coords"):
            value = voxel_state.get(key, None)
            if torch.is_tensor(value):
                initial_voxel_coords = value.detach().to(device=ref.device, dtype=torch.long)
                break
    codec_prior_score = amount_terms.get("codec_prune_prior_score_for_outcome", None)
    delete_prior_for_outcome = amount_terms.get("delete_prior_for_outcome", None)
    hard_delete_selection_mask = amount_terms.get("hard_delete_selection_mask_for_outcome", None)
    ranking_state = None
    reuse_where_ranking_used = False
    reuse_where_ranking_reason = ""
    where_mode = str(getattr(args, "sparsepcgc_where_mode", "block_only")).strip().lower()
    macro_micro_where = where_mode in {"macro_micro_heuristic", "macro_micro_hybrid"}
    if (
        multi_actual_enable
        and bool(getattr(args, "sparsepcgc_reuse_where_ranking_for_amounts", True))
        and not macro_micro_where
        and torch.is_tensor(initial_voxel_coords)
        and torch.is_tensor(codec_prior_score)
        and hasattr(actuator, "build_codec_block_drop_ranking")
        and hasattr(actuator, "mask_from_codec_block_ranking")
    ):
        try:
            ranking_state = actuator.build_codec_block_drop_ranking(
                initial_voxel_coords,
                codec_prior_score,
                block_size=max(int(structure_debug.get("codec_prune_prior_block_size", 4)), 1),
                selection_mask=hard_delete_selection_mask,
            )
            reuse_where_ranking_used = isinstance(ranking_state, dict)
        except Exception as exc:
            ranking_state = None
            reuse_where_ranking_reason = str(exc)
    elif macro_micro_where:
        reuse_where_ranking_reason = "macro_micro_where_mode"

    def _candidate_drop_mask_and_coords(candidate_ratio):
        if (
            not torch.is_tensor(initial_voxel_coords)
            or not torch.is_tensor(codec_prior_score)
            or actuator is None
            or candidate_ratio <= 0.0
        ):
            return None, None, 0, {}
        try:
            if macro_micro_where and hasattr(actuator, "build_macro_micro_where_mask"):
                hard_drop_mask, trace = actuator.build_macro_micro_where_mask(
                    initial_voxel_coords,
                    float(candidate_ratio),
                    codec_prior_score,
                    delete_prior=delete_prior_for_outcome,
                    selection_mask=hard_delete_selection_mask,
                    block_size=max(int(structure_debug.get("codec_prune_prior_block_size", 4)), 1),
                    args=args,
                    max_hard_count=int(getattr(args, "repair_max_hard_drop_voxels", 0)),
                    min_hard_count=0,
                )
            elif ranking_state is not None and reuse_where_ranking_used:
                hard_drop_mask, trace = actuator.mask_from_codec_block_ranking(
                    ranking_state,
                    target_drop_ratio=float(candidate_ratio),
                    max_hard_count=int(getattr(args, "repair_max_hard_drop_voxels", 0)),
                    min_hard_count=0,
                )
            else:
                hard_drop_mask = actuator._hard_codec_block_drop_mask(
                    initial_voxel_coords,
                    codec_prior_score,
                    block_size=max(int(structure_debug.get("codec_prune_prior_block_size", 4)), 1),
                    target_drop_ratio=float(candidate_ratio),
                    selection_mask=hard_delete_selection_mask,
                    max_hard_count=int(getattr(args, "repair_max_hard_drop_voxels", 0)),
                    min_hard_count=0,
                )
                trace = dict(getattr(actuator, "_last_hard_drop_count_trace", {}) or {})
        except Exception:
            return None, None, 0, {}
        if not torch.is_tensor(hard_drop_mask):
            return None, None, 0, trace
        kept_coords = []
        kept_counts = []
        for batch_idx in range(int(initial_voxel_coords.shape[0])):
            keep_mask_b = (~hard_drop_mask[batch_idx].squeeze(0).bool()).reshape(-1)
            coords_b = initial_voxel_coords[batch_idx : batch_idx + 1, :, keep_mask_b]
            kept_coords.append(coords_b)
            kept_counts.append(int(coords_b.shape[-1]))
        candidate_coords = None
        if kept_coords and len(set(kept_counts)) == 1:
            candidate_coords = torch.cat(kept_coords, dim=0).contiguous()
        drop_count_value = int(hard_drop_mask.detach().sum().item())
        return hard_drop_mask.detach(), candidate_coords, drop_count_value, trace

    def _candidate_geometry_penalty(unique_count_value, candidate_drop_count, candidate_ratio):
        ratio_target = min(
            max(float(getattr(args, "sparsepcgc_full_cloud_amount_ratio_reg_target", 0.05)), 0.0),
            0.05,
        )
        geom_penalty = _sparsepcgc_geometry_penalty_percent(
            args,
            unique_count_value,
            drop_count=int(candidate_drop_count),
        )
        ratio_penalty = max(float(candidate_ratio) - float(ratio_target), 0.0) ** 2
        return float(geom_penalty), float(ratio_penalty)

    def _fill_actual_row_fields(
        row,
        *,
        raw_bit,
        bit_source,
        edit_record_bits,
        actual_scope="full_cloud",
    ):
        row["actual_scope"] = str(actual_scope)
        row["actual_bit_source"] = str(bit_source or "")
        row["actual_bit_objective"] = str(objective_mode)
        row["actual_gt_bits"] = float(base_bit) if base_bit is not None else float("nan")
        row["actual_base_bits"] = float(base_bit) if base_bit is not None else float("nan")
        row["actual_total_bits_before"] = float(base_bit) if base_bit is not None else float("nan")
        row["actual_raw_bits"] = float(raw_bit) if raw_bit is not None and math.isfinite(float(raw_bit)) else float("nan")
        row["actual_mine_bits"] = row["actual_raw_bits"]
        row["actual_edit_record_bits"] = float(edit_record_bits)
        if raw_bit is not None and math.isfinite(float(raw_bit)):
            billed_bits = float(raw_bit) + float(edit_record_bits)
        else:
            billed_bits = float("nan")
        row["actual_billed_bits"] = billed_bits
        row["actual_total_bits_after"] = billed_bits
        if base_bit is not None and math.isfinite(case_float(row.get("actual_raw_percent", float("nan")), float("nan"))):
            raw_percent_value = case_float(row.get("actual_raw_percent", float("nan")), float("nan"))
            row["actual_raw_percent"] = float(raw_percent_value)
        if base_bit is not None and raw_bit is not None and math.isfinite(float(raw_bit)):
            raw_percent_value, billed_percent_value = _sparsepcgc_objective_percent_with_edit_record(
                args,
                float(raw_bit),
                float(base_bit),
                float(edit_record_bits),
            )
            row["actual_raw_percent"] = float(raw_percent_value)
            row["actual_percent"] = float(billed_percent_value)
        objective_percent_value, objective_source = _sparsepcgc_pick_objective_percent(
            args,
            row.get("actual_raw_percent", float("nan")),
            row.get("actual_percent", float("nan")),
        )
        row["actual_objective_percent"] = (
            float(objective_percent_value) if objective_percent_value is not None else float("nan")
        )
        row["actual_objective_bit_source"] = str(objective_source)
        row["teacher_compared_in_actual_pool"] = bool(
            bool(row.get("actual_finished", False))
            and math.isfinite(case_float(row.get("actual_objective_percent", float("nan")), float("nan")))
        )

    unique_count = int(initial_voxel_coords.shape[-1]) if torch.is_tensor(initial_voxel_coords) else int(
        structure_debug.get("input_voxel_count", input_points or 0)
    )

    def _candidate_sources(entry):
        return [str(source) for source in list((entry or {}).get("sources", []))]

    def _candidate_has_source(candidate_key, prefix):
        entry = candidate_specs.get(str(candidate_key), {}) or {}
        return any(str(source).startswith(str(prefix)) for source in _candidate_sources(entry))

    def _candidate_sort_key(candidate_key):
        entry = candidate_specs.get(str(candidate_key), {}) or {}
        base_class = int(entry.get("base_class", 0))
        primary = int(entry.get("priority", 99))
        predicted = float(pred_per_amount.detach().flatten()[base_class].cpu()) if base_class > 0 else 0.0
        ratio_gap = abs(float(entry.get("final_ratio", 0.0)) - float(selected_ratio_value))
        alt_prob = -float(amount_prob[base_class].detach().cpu()) if base_class > 0 else 0.0
        return (primary, predicted, alt_prob, ratio_gap, str(candidate_key))

    nonnoop_candidate_keys = [
        key for key, entry in candidate_specs.items()
        if not bool(entry.get("is_noop", False))
    ]
    sorted_candidate_keys = (
        sorted(nonnoop_candidate_keys, key=_candidate_sort_key)
        if teacher_actual_priority
        else sorted(
            nonnoop_candidate_keys,
            key=lambda key: float(
                pred_per_amount.detach().flatten()[int(candidate_specs[key]["base_class"])].cpu()
            ),
        )
    )
    candidate_priority_rank = {
        str(candidate_key): int(rank)
        for rank, candidate_key in enumerate(sorted_candidate_keys)
    }

    actual_candidate_keys = []
    actual_candidate_reason = {}
    use_enhanced_actual_selection = bool(
        wide_probe_enable or sequence_memory_enable or wide_probe_due
    ) and multi_actual_enable and max_actual > 1 and policy != "selected_only"

    def _append_actual_candidate(candidate_key, reason):
        candidate_key = str(candidate_key)
        if candidate_key not in candidate_specs:
            return False
        entry = candidate_specs.get(candidate_key, {}) or {}
        if bool(entry.get("is_noop", False)):
            return False
        if candidate_key in actual_candidate_keys:
            return False
        if len(actual_candidate_keys) >= int(max_actual):
            return False
        actual_candidate_keys.append(candidate_key)
        actual_candidate_reason[candidate_key] = str(reason)
        return True

    def _candidate_ratio_distance(candidate_key):
        entry = candidate_specs.get(str(candidate_key), {}) or {}
        return abs(float(entry.get("final_ratio", 0.0)) - float(selected_ratio_value))

    def _candidate_base_class(candidate_key):
        entry = candidate_specs.get(str(candidate_key), {}) or {}
        return int(entry.get("base_class", 0))

    def _pick_nonselected_predicted_candidate(excluded_keys):
        best_key = None
        best_tuple = None
        for candidate_key in sorted_candidate_keys:
            candidate_key = str(candidate_key)
            if candidate_key in excluded_keys or candidate_key == str(selected_candidate_key):
                continue
            entry = candidate_specs.get(candidate_key, {}) or {}
            if bool(entry.get("is_noop", False)):
                continue
            base_class = int(entry.get("base_class", 0))
            if base_class <= 0:
                continue
            ranking = (
                0 if base_class != int(selected_class) else 1,
                float(pred_per_amount.detach().flatten()[base_class].cpu()),
                int(entry.get("priority", 99)),
                -_candidate_ratio_distance(candidate_key),
                candidate_key,
            )
            if best_tuple is None or ranking < best_tuple:
                best_tuple = ranking
                best_key = candidate_key
        return best_key

    def _pick_uncertainty_candidate(excluded_keys):
        best_key = None
        best_tuple = None
        for candidate_key in sorted_candidate_keys:
            candidate_key = str(candidate_key)
            if candidate_key in excluded_keys or candidate_key == str(selected_candidate_key):
                continue
            entry = candidate_specs.get(candidate_key, {}) or {}
            if bool(entry.get("is_noop", False)):
                continue
            base_class = int(entry.get("base_class", 0))
            if base_class <= 0:
                continue
            ranking = (
                0 if base_class != int(selected_class) else 1,
                -float(amount_prob[base_class].detach().cpu()),
                float(pred_per_amount.detach().flatten()[base_class].cpu()),
                candidate_key,
            )
            if best_tuple is None or ranking < best_tuple:
                best_tuple = ranking
                best_key = candidate_key
        return best_key

    def _pick_wide_probe_candidates(excluded_keys):
        candidates = []
        for candidate_key in sorted_candidate_keys:
            candidate_key = str(candidate_key)
            if candidate_key in excluded_keys:
                continue
            entry = candidate_specs.get(candidate_key, {}) or {}
            if bool(entry.get("is_noop", False)) or not _candidate_has_source(candidate_key, "wide_probe_"):
                continue
            ratio_value = float(entry.get("final_ratio", 0.0))
            ranking = (
                int(_sparsepcgc_full_cloud_sequence_memory_seen_count(args, sequence_name, ratio_value)),
                min(abs(ratio_value - 0.010), abs(ratio_value - 0.040)),
                -abs(ratio_value - float(selected_ratio_value)),
                float(pred_per_amount.detach().flatten()[int(entry.get("base_class", 0))].cpu())
                if int(entry.get("base_class", 0)) > 0
                else 0.0,
                candidate_key,
            )
            candidates.append((ranking, candidate_key))
        candidates.sort(key=lambda item: item[0])
        return [candidate_key for _, candidate_key in candidates[: max(int(wide_probe_max_actual), 0)]]

    if selected_class > 0 and selected_candidate_key in candidate_specs:
        _append_actual_candidate(selected_candidate_key, "network_selected")

    if learning_mode == "network_selected_bandit":
        if diagnostic_sweep_due or wide_probe_due:
            diagnostic_key = _pick_uncertainty_candidate(set(actual_candidate_keys))
            wide_keys = _pick_wide_probe_candidates(set(actual_candidate_keys))
            if wide_keys:
                diagnostic_key = wide_keys[0]
            if diagnostic_key is not None:
                _append_actual_candidate(diagnostic_key, "diagnostic_sweep")
    elif use_enhanced_actual_selection:
        sequence_memory_best_key = None
        if isinstance(sequence_memory_best_entry, dict):
            best_ratio = sequence_memory_best_entry.get("ratio", float("nan"))
            best_ratio_key = None
            for candidate_key, entry in candidate_specs.items():
                if bool(entry.get("is_noop", False)):
                    continue
                if abs(float(entry.get("final_ratio", 0.0)) - float(best_ratio)) <= 1e-6:
                    best_ratio_key = str(candidate_key)
                    break
            sequence_memory_best_key = best_ratio_key
            if sequence_memory_best_key is not None:
                _append_actual_candidate(sequence_memory_best_key, "sequence_memory_best")

        predicted_key = _pick_nonselected_predicted_candidate(set(actual_candidate_keys))
        if predicted_key is not None:
            _append_actual_candidate(predicted_key, "predicted_top")

        uncertainty_key = _pick_uncertainty_candidate(set(actual_candidate_keys))
        if uncertainty_key is not None:
            _append_actual_candidate(uncertainty_key, "uncertainty_alt")

        if wide_probe_due:
            for candidate_key in _pick_wide_probe_candidates(set(actual_candidate_keys)):
                if len(actual_candidate_keys) >= int(max_actual):
                    break
                _append_actual_candidate(candidate_key, "wide_probe_due")

        for candidate_key in sorted_candidate_keys:
            if len(actual_candidate_keys) >= int(max_actual):
                break
            candidate_key = str(candidate_key)
            if candidate_key in actual_candidate_keys:
                continue
            too_close = any(
                abs(float(candidate_specs[candidate_key]["final_ratio"]) - float(candidate_specs[other_key]["final_ratio"])) <= 5e-4
                for other_key in actual_candidate_keys
            )
            if too_close and len(actual_candidate_keys) > 0:
                continue
            _append_actual_candidate(candidate_key, "fallback_diversified")
    else:
        for candidate_key in sorted_candidate_keys:
            if len(actual_candidate_keys) >= int(max_actual):
                break
            _append_actual_candidate(candidate_key, "legacy_priority")

    rows_by_key = {}
    candidate_encode_xyz = []
    candidate_encode_keys = []
    candidate_dispatch_t0 = time.time()
    selected_raw_bit = finite_float_or_none(
        compression_debug.get("gen_actual_bit", compression_debug.get("gen_bit_abs", None))
    )
    selected_billed_bit = finite_float_or_none(
        compression_debug.get(
            "gen_total_bit_with_edit_record",
            compression_debug.get("actual_total_bits", None),
        )
    )
    selected_edit_record_bits = finite_float_or_none(
        compression_debug.get("actual_edit_record_bits", None)
    )
    if selected_edit_record_bits is None or not math.isfinite(selected_edit_record_bits):
        selected_edit_record_bits = _sparsepcgc_edit_record_total_bits(
            args,
            unique_count,
            drop_count=int(drop_count or 0),
        )

    for cand_idx, (candidate_key, entry) in enumerate(candidate_specs.items()):
        candidate_key = str(candidate_key)
        is_noop = bool(entry.get("is_noop", False))
        base_class = int(entry.get("base_class", 0))
        base_bin = float(entry.get("base_bin", 0.0))
        candidate_residual = float(entry.get("candidate_residual", 0.0))
        cand_ratio = float(entry.get("final_ratio", 0.0))
        cand_surrogate = (
            float(pred_per_amount.detach().flatten()[base_class].cpu()) if base_class > 0 else 0.0
        )
        if is_noop:
            cand_drop_count = 0
            cand_geom_penalty = 0.0
            cand_ratio_penalty = 0.0
            cand_coords = None
            cand_where_trace = {}
            cand_requested = False
            cand_finished = True
            cand_error = ""
            cand_wall = 0.0
            compared_in_actual_pool = True
        else:
            _, cand_coords, cand_drop_count, cand_where_trace = _candidate_drop_mask_and_coords(cand_ratio)
            if candidate_key == str(selected_candidate_key) and int(drop_count or 0) > 0 and cand_drop_count <= 0:
                cand_drop_count = int(drop_count or 0)
            cand_geom_penalty, cand_ratio_penalty = _candidate_geometry_penalty(
                unique_count,
                cand_drop_count,
                cand_ratio,
            )
            cand_requested = bool(candidate_key in actual_candidate_keys)
            cand_finished = False
            cand_error = ""
            cand_wall = 0.0
            compared_in_actual_pool = False
            if cand_requested:
                if actual_available and candidate_key == str(selected_candidate_key):
                    cand_finished = True
                    compared_in_actual_pool = True
                elif torch.is_tensor(cand_coords):
                    candidate_encode_xyz.append(
                        _restore_codec_xyz_from_global_voxels(
                            args,
                            cand_coords,
                            full_cloud_context,
                            gt_xyz,
                        )
                    )
                    candidate_encode_keys.append(candidate_key)
                else:
                    cand_error = "candidate_coords_missing"

        row = {
            "_candidate_key": candidate_key,
            "_teacher_score": float("nan"),
            "_surrogate_teacher_score": float("nan"),
            "global_step": int(global_step) + 1,
            "episode": int(episode) + 1,
            "epoch": int(epoch) + 1,
            "step": int(step) + 1,
            "sample_key": str(cache_key),
            "candidate_id": int(cand_idx),
            "candidate_source": "+".join(entry.get("sources", [])),
            "candidate_priority_rank": int(candidate_priority_rank.get(candidate_key, len(candidate_specs))),
            "candidate_selected_for_actual_reason": str(actual_candidate_reason.get(candidate_key, "")),
            "is_wide_probe": bool(any(str(source).startswith("wide_probe_") for source in entry.get("sources", []))),
            "is_sequence_memory": bool(any(str(source).startswith("sequence_memory_") for source in entry.get("sources", []))),
            "candidate_base_class": int(base_class),
            "candidate_base_bin": float(base_bin),
            "candidate_residual": float(candidate_residual),
            "amount_bin": float(base_bin),
            "amount_residual": float(candidate_residual),
            "predicted_residual": float(pred_residual_float),
            "predicted_residual_raw": float(pred_residual_raw_float),
            "teacher_residual": float("nan"),
            "residual_error": float("nan"),
            "residual_teacher_clamped": False,
            "final_ratio": float(cand_ratio),
            "is_noop": bool(is_noop),
            "actual_requested": bool(cand_requested),
            "actual_finished": bool(cand_finished),
            "full_cloud_input_points": int(input_points or 0),
            "full_cloud_drop_count": int(cand_drop_count),
            "drop_count": int(cand_drop_count),
            "where_mode": str(cand_where_trace.get("where_mode", where_mode if not is_noop else "noop")),
            "effective_where_mode": str(cand_where_trace.get("effective_where_mode", cand_where_trace.get("where_mode", where_mode if not is_noop else "noop"))),
            "macro_ratio": case_float(cand_where_trace.get("macro_ratio", 0.0), 0.0),
            "micro_ratio": case_float(cand_where_trace.get("micro_ratio", 0.0), 0.0),
            "macro_selected_block_count": case_int(cand_where_trace.get("macro_selected_block_count", 0), 0),
            "macro_drop_count": case_int(cand_where_trace.get("macro_drop_count", 0), 0),
            "micro_drop_count": case_int(cand_where_trace.get("micro_drop_count", 0), 0),
            "total_drop_count": case_int(cand_where_trace.get("total_drop_count", cand_drop_count), int(cand_drop_count)),
            "selected_block_count": case_int(cand_where_trace.get("selected_block_count", 0), 0),
            "micro_selected_block_count": case_int(cand_where_trace.get("micro_selected_block_count", 0), 0),
            "max_drop_count_per_block": case_int(cand_where_trace.get("max_drop_count_per_block", 0), 0),
            "mean_drop_count_per_selected_block": case_float(
                cand_where_trace.get("mean_drop_count_per_selected_block", 0.0),
                0.0,
            ),
            "drop_concentration_top1_block_ratio": case_float(
                cand_where_trace.get("drop_concentration_top1_block_ratio", 0.0),
                0.0,
            ),
            "drop_concentration_top5_block_ratio": case_float(
                cand_where_trace.get("drop_concentration_top5_block_ratio", 0.0),
                0.0,
            ),
            "hard_where_uses_network_score": bool(cand_where_trace.get("hard_where_uses_network_score", False)),
            "heuristic_where_score_mean": case_float(cand_where_trace.get("heuristic_where_score_mean", 0.0), 0.0),
            "heuristic_where_score_std": case_float(cand_where_trace.get("heuristic_where_score_std", 0.0), 0.0),
            "micro_quota_hit_block_count": case_int(cand_where_trace.get("micro_quota_hit_block_count", 0), 0),
            "micro_min_selected_blocks": case_int(cand_where_trace.get("micro_min_selected_blocks", 0), 0),
            "micro_candidate_block_count": case_int(cand_where_trace.get("micro_candidate_block_count", 0), 0),
            "micro_min_blocks_satisfied": bool(cand_where_trace.get("micro_min_blocks_satisfied", False)),
            "micro_min_blocks_fallback_reason": str(cand_where_trace.get("micro_min_blocks_fallback_reason", "")),
            "macro_micro_hybrid_fallback": bool(cand_where_trace.get("macro_micro_hybrid_fallback", False)),
            "macro_disabled_reason": str(cand_where_trace.get("macro_disabled_reason", "")),
            "actual_percent": 0.0 if is_noop else float("nan"),
            "actual_raw_percent": 0.0 if is_noop else float("nan"),
            "actual_objective_percent": 0.0 if is_noop else float("nan"),
            "actual_bit_objective": str(objective_mode),
            "actual_objective_bit_source": "noop_baseline",
            "actual_total_bits_before": float(base_bit) if base_bit is not None else float("nan"),
            "actual_total_bits_after": float(base_bit) if (is_noop and base_bit is not None) else float("nan"),
            "actual_base_bits": float(base_bit) if base_bit is not None else float("nan"),
            "actual_raw_bits": float(base_bit) if (is_noop and base_bit is not None) else float("nan"),
            "actual_billed_bits": float(base_bit) if (is_noop and base_bit is not None) else float("nan"),
            "actual_edit_record_bits": 0.0,
            "actual_mine_bits": float(base_bit) if (is_noop and base_bit is not None) else float("nan"),
            "actual_gt_bits": float(base_bit) if base_bit is not None else float("nan"),
            "actual_bit_source": "noop_baseline",
            "surrogate_percent": float(cand_surrogate),
            "geom_loss": float(cand_geom_penalty),
            "predicted_delta": float(cand_surrogate),
            "teacher_compared_in_actual_pool": bool(compared_in_actual_pool),
            "teacher_is_best": False,
            "teacher_label": 0,
            "teacher_source": "none",
            "selected_is_best": False,
            "oracle_gap": float("nan"),
            "actual_worker_id": -1,
            "actual_wall_time": float(cand_wall),
            "actual_error_reason": str(cand_error),
            "actual_scope": "full_cloud",
        }
        if is_noop:
            row["_teacher_score"] = 0.0
            row["_surrogate_teacher_score"] = 0.0
        elif actual_available and candidate_key == str(selected_candidate_key):
            row["actual_requested"] = True
            row["actual_finished"] = True
            row["actual_percent"] = float(actual_value)
            row["actual_raw_percent"] = case_float(
                compression_debug.get("actual_raw_percent", float("nan")),
                float("nan"),
            )
            _fill_actual_row_fields(
                row,
                raw_bit=selected_raw_bit,
                bit_source=compression_debug.get("actual_value_source", "selected_existing_actual"),
                edit_record_bits=selected_edit_record_bits,
            )
        rows_by_key[candidate_key] = row
        rows.append(row)

    actual_dispatch_time = float(time.time() - candidate_dispatch_t0)
    actual_stats_by_key = {}
    if candidate_encode_keys:
        encode_candidates = []
        filtered_keys = []
        for candidate_key, xyz in zip(candidate_encode_keys, candidate_encode_xyz):
            if torch.is_tensor(xyz) and xyz.ndim == 3 and xyz.shape[1] == 3 and xyz.shape[-1] > 0:
                encode_candidates.append(xyz)
                filtered_keys.append(str(candidate_key))
            else:
                row = rows_by_key.get(str(candidate_key), None)
                if isinstance(row, dict):
                    row["actual_error_reason"] = "candidate_xyz_restore_failed"
        actual_gather_t0 = time.time()
        try:
            multi_stats = list(loss_obj._encode_actual_many(args, encode_candidates))
        except Exception as exc:
            multi_stats = []
            for candidate_key in filtered_keys:
                row = rows_by_key.get(str(candidate_key), None)
                if isinstance(row, dict):
                    row["actual_error_reason"] = f"encode_many_failed:{exc}"
        actual_gather_time = float(time.time() - actual_gather_t0)
        debug["full_cloud_amount_actual_gather_time"] = actual_gather_time
        for candidate_key, stats in zip(filtered_keys, multi_stats):
            row = rows_by_key.get(str(candidate_key), None)
            if not isinstance(row, dict):
                continue
            stats = dict(stats or {})
            row["actual_requested"] = bool(stats.get("actual_requested", True))
            row["actual_finished"] = bool(stats.get("actual_finished", False))
            row["actual_worker_id"] = case_int(stats.get("actual_worker_id", -1), -1)
            row["actual_wall_time"] = case_float(stats.get("actual_wall_time", 0.0), 0.0)
            row["actual_error_reason"] = str(stats.get("actual_error_reason", row.get("actual_error_reason", "")))
            raw_bit = finite_float_or_none(stats.get("bit", None))
            if row["actual_finished"] and base_bit is not None and raw_bit is not None and math.isfinite(raw_bit):
                _fill_actual_row_fields(
                    row,
                    raw_bit=raw_bit,
                    bit_source=stats.get("codec", "actual_encoder"),
                    edit_record_bits=_sparsepcgc_edit_record_total_bits(
                        args,
                        unique_count,
                        drop_count=int(row.get("full_cloud_drop_count", 0)),
                    ),
                )
                actual_stats_by_key[str(candidate_key)] = stats
            elif row["actual_finished"]:
                row["actual_error_reason"] = "base_bit_missing_for_candidate"
        debug["full_cloud_amount_actual_wall_time_total"] = float(
            sum(case_float(row.get("actual_wall_time", 0.0), 0.0) for row in rows)
        )
        debug["full_cloud_amount_actual_wall_time_max"] = float(
            max((case_float(row.get("actual_wall_time", 0.0), 0.0) for row in rows), default=0.0)
        )
        debug["sparsepcgc_actual_worker_pool_used"] = any(
            str(stats.get("actual_parallel_mode_effective", "single")) == "worker_pool"
            for stats in multi_stats
        )
    else:
        debug["full_cloud_amount_actual_gather_time"] = 0.0
    debug["full_cloud_amount_actual_dispatch_time"] = float(actual_dispatch_time)

    for row in rows:
        if bool(row.get("actual_requested", False)) and bool(row.get("actual_finished", False)) and not bool(row.get("is_noop", False)):
            _sparsepcgc_update_amount_outcome_memory(
                args,
                amount_memory_key,
                row.get("actual_objective_percent", row.get("actual_percent", float("nan"))),
                row.get("final_ratio", 0.0),
            )
            _sparsepcgc_update_full_cloud_sequence_amount_memory(
                args,
                sequence_name=sequence_name,
                row=row,
                global_step=global_step,
            )

    actual_pool_rows = []
    surrogate_pool_rows = []
    raw_actual_pool_rows = []
    for row in rows:
        if bool(row.get("is_noop", False)):
            row["_teacher_score"] = 0.0
            row["_surrogate_teacher_score"] = 0.0
            row["_raw_teacher_score"] = 0.0
            row["teacher_compared_in_actual_pool"] = True
            actual_pool_rows.append(row)
            surrogate_pool_rows.append(row)
            raw_actual_pool_rows.append(row)
            continue
        if bool(row.get("teacher_compared_in_actual_pool", False)) and math.isfinite(case_float(row.get("actual_objective_percent", float("nan")), float("nan"))):
            row["_teacher_score"] = (
                float(row.get("actual_objective_percent", row["actual_percent"]))
                + float(getattr(args, "sparsepcgc_full_cloud_amount_geom_penalty_weight", 0.1)) * float(row.get("geom_loss", 0.0))
                + float(getattr(args, "sparsepcgc_full_cloud_amount_ratio_reg_weight", 0.05))
                * (max(float(row.get("final_ratio", 0.0)) - float(getattr(args, "sparsepcgc_full_cloud_amount_ratio_reg_target", 0.05)), 0.0) ** 2)
            )
            actual_pool_rows.append(row)
        if bool(row.get("teacher_compared_in_actual_pool", False)) and math.isfinite(case_float(row.get("actual_raw_percent", float("nan")), float("nan"))):
            row["_raw_teacher_score"] = (
                float(row.get("actual_raw_percent", row.get("actual_objective_percent", float("nan"))))
                + float(getattr(args, "sparsepcgc_full_cloud_amount_geom_penalty_weight", 0.1)) * float(row.get("geom_loss", 0.0))
                + float(getattr(args, "sparsepcgc_full_cloud_amount_ratio_reg_weight", 0.05))
                * (max(float(row.get("final_ratio", 0.0)) - float(getattr(args, "sparsepcgc_full_cloud_amount_ratio_reg_target", 0.05)), 0.0) ** 2)
            )
            raw_actual_pool_rows.append(row)
        surrogate_score = float(row.get("surrogate_percent", float("nan")))
        if math.isfinite(surrogate_score):
            row["_surrogate_teacher_score"] = surrogate_score
            surrogate_pool_rows.append(row)

    teacher_row = rows_by_key.get("0:0.000000", None)
    teacher_row_key = str(teacher_row.get("_candidate_key")) if isinstance(teacher_row, dict) else "0:0.000000"
    teacher_class = 0
    teacher_delta = 0.0
    teacher_ratio_value = 0.0
    teacher_source = "surrogate_fallback"
    oracle_best_ratio = 0.0
    oracle_best_actual_delta = 0.0
    oracle_best_objective_delta = 0.0
    selected_row = rows_by_key.get(str(selected_candidate_key), None)
    selected_actual_delta = case_float(
        selected_row.get("actual_percent", float("nan")) if isinstance(selected_row, dict) else float("nan"),
        float("nan"),
    )
    selected_objective_delta = case_float(
        selected_row.get("actual_objective_percent", float("nan")) if isinstance(selected_row, dict) else float("nan"),
        float("nan"),
    )
    selected_score = case_float(
        selected_row.get("_teacher_score", float("nan")) if isinstance(selected_row, dict) else float("nan"),
        float("nan"),
    )
    selected_raw_score = case_float(
        selected_row.get("_raw_teacher_score", float("nan")) if isinstance(selected_row, dict) else float("nan"),
        float("nan"),
    )
    raw_oracle_best_ratio = float("nan")
    raw_oracle_gap = float("nan")
    selected_is_raw_best = False
    selected_is_best_value = False
    oracle_gap_value = float("nan")
    teacher_base_bin = 0.0
    teacher_residual_target = 0.0
    teacher_residual_clamped = False

    actual_nonnoop_rows = [row for row in actual_pool_rows if not bool(row.get("is_noop", False))]
    raw_actual_nonnoop_rows = [row for row in raw_actual_pool_rows if not bool(row.get("is_noop", False))]
    best_row = None
    if actual_pool_rows and actual_nonnoop_rows:
        best_row = min(actual_pool_rows, key=lambda row: float(row.get("_teacher_score", float("inf"))))
        oracle_best_ratio = float(best_row.get("final_ratio", 0.0))
        oracle_best_actual_delta = float(best_row.get("actual_percent", 0.0))
        oracle_best_objective_delta = float(best_row.get("actual_objective_percent", best_row.get("actual_percent", 0.0)))
        if bool(best_row.get("is_noop", False)):
            oracle_best_actual_delta = 0.0
            oracle_best_objective_delta = 0.0
        if oracle_sweep_due and len(actual_nonnoop_rows) > 1:
            teacher_source = "actual_full_cloud_oracle_sweep"
        elif len(actual_nonnoop_rows) > 1:
            teacher_source = "actual_full_cloud_multi"
        else:
            teacher_source = "actual_full_cloud_single"
    elif surrogate_pool_rows:
        best_row = min(surrogate_pool_rows, key=lambda row: float(row.get("_surrogate_teacher_score", float("inf"))))
        oracle_best_ratio = float(best_row.get("final_ratio", 0.0))
        oracle_best_actual_delta = float(best_row.get("surrogate_percent", 0.0))
        oracle_best_objective_delta = float(best_row.get("surrogate_percent", 0.0))
        teacher_source = "surrogate_fallback"

    if raw_actual_pool_rows and raw_actual_nonnoop_rows:
        raw_best_row = min(raw_actual_pool_rows, key=lambda row: float(row.get("_raw_teacher_score", float("inf"))))
        raw_oracle_best_ratio = float(raw_best_row.get("final_ratio", 0.0))
        raw_best_score = case_float(raw_best_row.get("_raw_teacher_score", float("nan")), float("nan"))
        if math.isfinite(raw_best_score) and math.isfinite(selected_raw_score):
            raw_oracle_gap = float(selected_raw_score - raw_best_score)
            selected_is_raw_best = bool(str(selected_candidate_key) == str(raw_best_row.get("_candidate_key", "")))

    entropy_decay_steps = max(int(getattr(args, "sparsepcgc_full_cloud_amount_entropy_decay_steps", 2000)), 0)
    min_entropy_weight = max(
        float(getattr(args, "sparsepcgc_full_cloud_amount_min_entropy_weight", 0.001)),
        0.0,
    )
    base_entropy_weight = max(
        float(getattr(args, "sparsepcgc_full_cloud_amount_entropy_weight", 0.01)),
        0.0,
    )
    entropy_decay = (
        max(0.0, 1.0 - float(global_step) / float(max(entropy_decay_steps, 1)))
        if entropy_decay_steps > 0
        else 0.0
    )
    entropy_weight_effective = max(
        float(min_entropy_weight),
        float(base_entropy_weight) * float(entropy_decay),
    )
    entropy_norm = amount_entropy_t / max(math.log(float(class_count)), 1e-6) if class_count > 1 else ref.new_zeros(())
    entropy_loss = (1.0 - entropy_norm.to(dtype=ref.dtype))
    cls_loss = ref.new_zeros(())
    value_loss = ref.new_zeros(())
    rank_loss = ref.new_zeros(())
    geom_loss_term = ref.new_zeros(())
    ratio_reg_loss = ref.new_zeros(())
    noop_guard_loss = ref.new_zeros(())
    residual_loss = ref.new_zeros(())
    amount_policy_loss = ref.new_zeros(())
    amount_rd_score_value = float("nan")
    amount_advantage_value = float("nan")
    debug_geom_cost_value = 0.0
    sequence_baseline_value = _sparsepcgc_full_cloud_sequence_baseline_get(args, sequence_name)
    bandit_aux_actual_teacher = bool(
        getattr(args, "sparsepcgc_full_cloud_amount_bandit_aux_actual_teacher", True)
    )

    if learning_mode == "network_selected_bandit":
        selected_actual_objective = case_float(
            selected_row.get("actual_objective_percent", float("nan")) if isinstance(selected_row, dict) else float("nan"),
            float("nan"),
        )
        selected_surrogate_objective = case_float(
            selected_row.get("surrogate_percent", float("nan")) if isinstance(selected_row, dict) else float("nan"),
            float("nan"),
        )
        selected_geom_cost = case_float(
            selected_row.get("geom_loss", 0.0) if isinstance(selected_row, dict) else 0.0,
            0.0,
        )
        debug_geom_cost_value = float(selected_geom_cost)
        selected_ratio_cost = float(selected_ratio_value)
        observed_objective = (
            selected_actual_objective
            if math.isfinite(selected_actual_objective)
            else selected_surrogate_objective
        )
        if not math.isfinite(observed_objective):
            observed_objective = 0.0
        amount_rd_score_value = (
            float(observed_objective)
            + float(getattr(args, "sparsepcgc_full_cloud_amount_geom_cost_weight", 0.0)) * float(selected_geom_cost)
            + float(getattr(args, "sparsepcgc_full_cloud_amount_ratio_cost_weight", 0.05)) * float(selected_ratio_cost)
        )
        if sequence_baseline_value is None or not math.isfinite(sequence_baseline_value):
            sequence_baseline_value = float(amount_rd_score_value)
        advantage_t = ref.new_tensor(-(float(amount_rd_score_value) - float(sequence_baseline_value)))
        amount_advantage_value = float(advantage_t.detach().cpu())
        value_loss = torch.nn.functional.smooth_l1_loss(
            pred_per_amount[int(selected_class)],
            ref.new_tensor(float(amount_rd_score_value)),
        )
        if action_sample_mode == "argmax":
            amount_policy_loss = ref.new_zeros(())
        else:
            amount_policy_loss = -(selected_log_prob_t.to(dtype=ref.dtype) * advantage_t.detach())
        if (
            isinstance(selected_row, dict)
            and bool(selected_row.get("actual_finished", False))
            and math.isfinite(selected_actual_objective)
        ):
            updated_baseline = _sparsepcgc_update_full_cloud_sequence_baseline(
                args,
                sequence_name=sequence_name,
                rd_score=amount_rd_score_value,
                global_step=global_step,
            )
            if updated_baseline is not None:
                sequence_baseline_value = float(updated_baseline)
        teacher_row = selected_row if isinstance(selected_row, dict) else teacher_row
        teacher_row_key = str(selected_candidate_key)
        teacher_class = int(selected_class)
        teacher_ratio_value = float(selected_ratio_value)
        teacher_base_bin = float(selected_bin.detach().cpu())
        teacher_residual_target = float(pred_residual_float) if residual_enable and selected_class > 0 else 0.0
        teacher_source = (
            "bandit_selected_actual"
            if isinstance(selected_row, dict)
            and bool(selected_row.get("actual_finished", False))
            and math.isfinite(selected_actual_objective)
            else "bandit_selected_surrogate"
        )
        selected_is_best_value = bool(
            isinstance(best_row, dict)
            and str(best_row.get("_candidate_key", "")) == str(selected_candidate_key)
        ) if best_row is not None else True
        oracle_gap_value = float("nan")
        if isinstance(best_row, dict) and math.isfinite(selected_score):
            best_teacher_score = case_float(
                best_row.get("_teacher_score", best_row.get("_surrogate_teacher_score", float("nan"))),
                float("nan"),
            )
            if math.isfinite(best_teacher_score):
                oracle_gap_value = float(selected_score - best_teacher_score)
        if bandit_aux_actual_teacher:
            actual_supervised_rows = [
                row for row in actual_pool_rows
                if math.isfinite(case_float(row.get("actual_objective_percent", float("nan")), float("nan")))
            ]
            actual_supervised_nonselected_rows = [
                row for row in actual_supervised_rows
                if not bool(row.get("is_noop", False))
                and str(row.get("_candidate_key", "")) != str(selected_candidate_key)
            ]
            if actual_supervised_rows and actual_supervised_nonselected_rows:
                bandit_best_row = min(
                    actual_supervised_rows,
                    key=lambda row: float(row.get("_teacher_score", float("inf"))),
                )
                bandit_best_score = case_float(
                    bandit_best_row.get("_teacher_score", float("nan")),
                    float("nan"),
                )
                teacher_row = bandit_best_row
                teacher_row_key = str(bandit_best_row.get("_candidate_key", "0:0.000000"))
                if (not math.isfinite(bandit_best_score)) or bandit_best_score >= float(noop_margin):
                    teacher_class = 0
                    teacher_ratio_value = 0.0
                else:
                    teacher_class = int(bandit_best_row.get("candidate_base_class", 0))
                    teacher_ratio_value = float(bandit_best_row.get("final_ratio", 0.0))
                teacher_class = min(max(int(teacher_class), 0), class_count - 1)
                if teacher_class > 0 and teacher_ratio_value > 0.0:
                    if residual_teacher_mode == "nearest_bin":
                        teacher_class, teacher_base_bin, teacher_residual_target = _best_base_for_ratio(teacher_ratio_value)
                    else:
                        teacher_class = int(case_int(bandit_best_row.get("candidate_base_class", teacher_class), teacher_class))
                        teacher_class = min(max(int(teacher_class), 1), class_count - 1)
                        teacher_base_bin = float(
                            case_float(
                                bandit_best_row.get(
                                    "candidate_base_bin",
                                    float(bin_tensor.detach().flatten()[teacher_class].cpu()),
                                ),
                                float(bin_tensor.detach().flatten()[teacher_class].cpu()),
                            )
                        )
                        teacher_residual_target = float(teacher_ratio_value - teacher_base_bin)
                    unclamped_teacher_residual = float(teacher_residual_target)
                    if residual_enable:
                        teacher_residual_target = min(
                            max(float(teacher_residual_target), -float(residual_max)),
                            float(residual_max),
                        )
                        teacher_residual_clamped = (
                            abs(float(unclamped_teacher_residual) - float(teacher_residual_target)) > 1e-9
                        )
                    else:
                        teacher_residual_target = 0.0
                        teacher_residual_clamped = abs(float(unclamped_teacher_residual)) > 1e-9
                else:
                    teacher_class = 0
                    teacher_base_bin = 0.0
                    teacher_residual_target = 0.0

                target = torch.tensor([teacher_class], device=amount_logits.device, dtype=torch.long)
                cls_loss = torch.nn.functional.cross_entropy(amount_logits.view(1, -1).float(), target)

                unique_supervised_rows = OrderedDict()
                for row in actual_supervised_rows:
                    cls = int(row.get("candidate_base_class", 0))
                    if cls < 0 or cls >= class_count:
                        continue
                    score_value = case_float(row.get("_teacher_score", float("nan")), float("inf"))
                    prev = unique_supervised_rows.get(cls, None)
                    prev_score = case_float(
                        prev.get("_teacher_score", float("nan")) if isinstance(prev, dict) else float("inf"),
                        float("inf"),
                    )
                    if prev is None or score_value < prev_score:
                        unique_supervised_rows[cls] = row
                value_terms = []
                for cls, row in unique_supervised_rows.items():
                    target_value = float(row.get("actual_objective_percent", 0.0))
                    value_terms.append(
                        torch.nn.functional.smooth_l1_loss(
                            pred_per_amount[int(cls)],
                            ref.new_tensor(float(target_value)),
                        )
                    )
                if value_terms:
                    value_loss = torch.stack(value_terms).mean()

                rank_terms = []
                rank_rows = list(unique_supervised_rows.values())
                margin = ref.new_tensor(0.1)
                for good_row in rank_rows:
                    for bad_row in rank_rows:
                        good_cls = int(good_row.get("candidate_base_class", 0))
                        bad_cls = int(bad_row.get("candidate_base_class", 0))
                        if good_cls == bad_cls:
                            continue
                        good_score = case_float(good_row.get("_teacher_score", float("nan")), float("nan"))
                        bad_score = case_float(bad_row.get("_teacher_score", float("nan")), float("nan"))
                        if not (math.isfinite(good_score) and math.isfinite(bad_score) and good_score + 1e-9 < bad_score):
                            continue
                        rank_terms.append(torch.relu(margin + pred_per_amount[good_cls] - pred_per_amount[bad_cls]))
                if rank_terms:
                    rank_loss = torch.stack(rank_terms).mean()

                if residual_enable and residual_loss_weight > 0.0 and teacher_class > 0:
                    residual_loss = torch.nn.functional.smooth_l1_loss(
                        pred_residual_t,
                        ref.new_tensor(float(teacher_residual_target)),
                    )
                teacher_source = "bandit_aux_actual_best"
                selected_is_best_value = bool(str(selected_candidate_key) == str(teacher_row_key))
                if math.isfinite(selected_score) and math.isfinite(bandit_best_score):
                    oracle_gap_value = float(selected_score - bandit_best_score)
    else:
        if actual_pool_rows and actual_nonnoop_rows:
            best_score = float(best_row.get("_teacher_score", float("inf"))) if isinstance(best_row, dict) else float("inf")
            teacher_row = best_row
            teacher_row_key = str(best_row.get("_candidate_key", "0:0.000000")) if isinstance(best_row, dict) else "0:0.000000"
            if best_score >= float(noop_margin):
                teacher_class = 0
                teacher_delta = 0.0
                teacher_ratio_value = 0.0
            else:
                teacher_class = int(best_row.get("candidate_base_class", 0))
                teacher_delta = float(best_row.get("actual_objective_percent", best_row.get("actual_percent", 0.0)))
                teacher_ratio_value = float(best_row.get("final_ratio", 0.0))
            if isinstance(selected_row, dict):
                selected_actual_delta = case_float(selected_row.get("actual_percent", float("nan")), float("nan"))
                selected_objective_delta = case_float(selected_row.get("actual_objective_percent", float("nan")), float("nan"))
                selected_score = case_float(selected_row.get("_teacher_score", float("nan")), float("nan"))
        elif surrogate_pool_rows:
            teacher_row = best_row
            teacher_row_key = str(best_row.get("_candidate_key", "0:0.000000")) if isinstance(best_row, dict) else "0:0.000000"
            best_score = float(best_row.get("_surrogate_teacher_score", float("inf"))) if isinstance(best_row, dict) else float("inf")
            if best_score >= float(noop_margin):
                teacher_class = 0
                teacher_delta = 0.0
                teacher_ratio_value = 0.0
            else:
                teacher_class = int(best_row.get("candidate_base_class", 0))
                teacher_delta = float(best_row.get("surrogate_percent", 0.0))
                teacher_ratio_value = float(best_row.get("final_ratio", 0.0))
            if isinstance(selected_row, dict):
                selected_actual_delta = case_float(selected_row.get("surrogate_percent", float("nan")), float("nan"))
                selected_objective_delta = case_float(selected_row.get("surrogate_percent", float("nan")), float("nan"))
                selected_score = case_float(selected_row.get("_surrogate_teacher_score", float("nan")), float("nan"))

        teacher_class = min(max(int(teacher_class), 0), class_count - 1)
        if teacher_class > 0 and teacher_ratio_value > 0.0:
            if residual_teacher_mode == "nearest_bin":
                teacher_class, teacher_base_bin, teacher_residual_target = _best_base_for_ratio(teacher_ratio_value)
            else:
                teacher_class = int(case_int(teacher_row.get("candidate_base_class", teacher_class), teacher_class)) if isinstance(teacher_row, dict) else teacher_class
                teacher_class = min(max(int(teacher_class), 1), class_count - 1)
                teacher_base_bin = float(
                    case_float(
                        teacher_row.get("candidate_base_bin", float(bin_tensor.detach().flatten()[teacher_class].cpu()))
                        if isinstance(teacher_row, dict)
                        else float(bin_tensor.detach().flatten()[teacher_class].cpu()),
                        float(bin_tensor.detach().flatten()[teacher_class].cpu()),
                    )
                )
                teacher_residual_target = float(teacher_ratio_value - teacher_base_bin)
            unclamped_teacher_residual = float(teacher_residual_target)
            if residual_enable:
                teacher_residual_target = min(
                    max(float(teacher_residual_target), -float(residual_max)),
                    float(residual_max),
                )
                teacher_residual_clamped = abs(float(unclamped_teacher_residual) - float(teacher_residual_target)) > 1e-9
            else:
                teacher_residual_target = 0.0
                teacher_residual_clamped = abs(float(unclamped_teacher_residual)) > 1e-9
        else:
            teacher_class = 0
            teacher_base_bin = 0.0
            teacher_residual_target = 0.0

        target = torch.tensor([teacher_class], device=amount_logits.device, dtype=torch.long)
        cls_loss = torch.nn.functional.cross_entropy(amount_logits.view(1, -1).float(), target)

        supervised_rows = [
            row for row in actual_pool_rows
            if math.isfinite(case_float(row.get("actual_objective_percent", float("nan")), float("nan")))
        ]
        supervised_source = "actual"
        if not supervised_rows:
            supervised_rows = [
                row for row in surrogate_pool_rows
                if math.isfinite(case_float(row.get("surrogate_percent", float("nan")), float("nan")))
            ]
            supervised_source = "surrogate"
        unique_supervised_rows = OrderedDict()
        for row in supervised_rows:
            cls = int(row.get("candidate_base_class", 0))
            if cls < 0 or cls >= class_count:
                continue
            score_value = case_float(
                row.get("_teacher_score", row.get("_surrogate_teacher_score", float("nan"))),
                float("inf"),
            )
            prev = unique_supervised_rows.get(cls, None)
            prev_score = case_float(
                prev.get("_teacher_score", prev.get("_surrogate_teacher_score", float("nan"))) if isinstance(prev, dict) else float("inf"),
                float("inf"),
            )
            if prev is None or score_value < prev_score:
                unique_supervised_rows[cls] = row
        value_terms = []
        for cls, row in unique_supervised_rows.items():
            target_value = (
                float(row.get("actual_objective_percent", float("nan")))
                if supervised_source == "actual" and math.isfinite(case_float(row.get("actual_objective_percent", float("nan")), float("nan")))
                else float(row.get("surrogate_percent", 0.0))
            )
            value_terms.append(
                torch.nn.functional.smooth_l1_loss(
                    pred_per_amount[int(cls)],
                    ref.new_tensor(float(target_value)),
                )
            )
        value_loss = torch.stack(value_terms).mean() if value_terms else ref.new_zeros(())

        rank_terms = []
        rank_rows = list(unique_supervised_rows.values())
        margin = ref.new_tensor(0.1)
        for good_row in rank_rows:
            for bad_row in rank_rows:
                good_cls = int(good_row.get("candidate_base_class", 0))
                bad_cls = int(bad_row.get("candidate_base_class", 0))
                if good_cls == bad_cls:
                    continue
                good_score = case_float(
                    good_row.get("_teacher_score", good_row.get("_surrogate_teacher_score", float("nan"))),
                    float("nan"),
                )
                bad_score = case_float(
                    bad_row.get("_teacher_score", bad_row.get("_surrogate_teacher_score", float("nan"))),
                    float("nan"),
                )
                if not (math.isfinite(good_score) and math.isfinite(bad_score) and good_score + 1e-9 < bad_score):
                    continue
                rank_terms.append(torch.relu(margin + pred_per_amount[good_cls] - pred_per_amount[bad_cls]))
        rank_loss = torch.stack(rank_terms).mean() if rank_terms else ref.new_zeros(())

        penalty_vector = ref.new_zeros((class_count,))
        for cls in range(class_count):
            if cls == 0:
                continue
            cand_ratio = float(bin_tensor.detach().flatten()[cls].cpu())
            cand_geom_penalty, _ = _candidate_geometry_penalty(unique_count, 0, cand_ratio)
            penalty_vector[cls] = float(cand_geom_penalty)
        geom_loss_term = (policy_probs * penalty_vector.float()).sum().to(dtype=ref.dtype)
        debug_geom_cost_value = float(geom_loss_term.detach().cpu())

        ratio_target = min(max(float(getattr(args, "sparsepcgc_full_cloud_amount_ratio_reg_target", 0.05)), 0.0), 0.05)
        ratio_reg_loss = torch.relu(final_ratio - ref.new_tensor(float(ratio_target))).pow(2)
        noop_guard_loss = -policy_log_probs[0] if teacher_class == 0 else ref.new_zeros(())
        if residual_enable and residual_loss_weight > 0.0 and teacher_class > 0:
            residual_loss = torch.nn.functional.smooth_l1_loss(
                pred_residual_t,
                ref.new_tensor(float(teacher_residual_target)),
            )
        amount_policy_loss = ref.new_zeros(())
        selected_is_best_value = bool(str(selected_candidate_key) == str(teacher_row_key))
        oracle_gap_value = float("nan")
        if isinstance(teacher_row, dict) and math.isfinite(selected_score):
            best_teacher_score = case_float(
                teacher_row.get("_teacher_score", teacher_row.get("_surrogate_teacher_score", float("nan"))),
                float("nan"),
            )
            if math.isfinite(best_teacher_score):
                oracle_gap_value = float(selected_score - best_teacher_score)

    for row in rows:
        row["teacher_is_best"] = bool(str(row.get("_candidate_key", "")) == str(teacher_row_key))
        row["teacher_label"] = int(teacher_class)
        row["teacher_source"] = str(teacher_source)
        row["teacher_residual"] = float(teacher_residual_target)
        row["residual_teacher_clamped"] = bool(teacher_residual_clamped)

    total_loss = (
        float(getattr(args, "sparsepcgc_full_cloud_amount_cls_loss_weight", 1.0)) * cls_loss
        + float(getattr(args, "sparsepcgc_full_cloud_amount_policy_loss_weight", 1.0)) * amount_policy_loss
        + float(getattr(args, "sparsepcgc_full_cloud_amount_value_loss_weight", 0.5)) * value_loss
        + float(getattr(args, "sparsepcgc_full_cloud_amount_rank_loss_weight", 0.2)) * rank_loss
        + float(getattr(args, "sparsepcgc_full_cloud_amount_geom_penalty_weight", 0.1)) * geom_loss_term
        + float(getattr(args, "sparsepcgc_full_cloud_amount_ratio_reg_weight", 0.05)) * ratio_reg_loss
        + float(getattr(args, "sparsepcgc_full_cloud_amount_noop_guard_weight", 0.5)) * noop_guard_loss
        + float(entropy_weight_effective) * entropy_loss
        + float(residual_loss_weight) * residual_loss
    )
    total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

    actual_requested_count = sum(
        1 for row in rows if bool(row.get("actual_requested", False)) and not bool(row.get("is_noop", False))
    )
    actual_finished_count = sum(
        1
        for row in rows
        if bool(row.get("actual_finished", False))
        and not bool(row.get("is_noop", False))
        and math.isfinite(case_float(row.get("actual_objective_percent", float("nan")), float("nan")))
    )
    actual_finished_nonselected_count = sum(
        1
        for row in rows
        if bool(row.get("actual_finished", False))
        and not bool(row.get("is_noop", False))
        and str(row.get("_candidate_key", "")) != str(selected_candidate_key)
        and math.isfinite(case_float(row.get("actual_objective_percent", float("nan")), float("nan")))
    )
    wide_probe_actual_count = sum(
        1
        for row in rows
        if bool(row.get("actual_finished", False))
        and not bool(row.get("is_noop", False))
        and bool(row.get("is_wide_probe", False))
        and math.isfinite(case_float(row.get("actual_objective_percent", float("nan")), float("nan")))
    )
    residual_error = float("nan")
    if teacher_class > 0 and residual_enable:
        residual_error = float(abs(pred_residual_float - float(teacher_residual_target)))
    selected_where_row = rows_by_key.get(str(selected_candidate_key), {}) if isinstance(rows_by_key, dict) else {}
    if not isinstance(selected_where_row, dict):
        selected_where_row = {}

    for row in rows:
        row["selected_is_best"] = bool(selected_is_best_value)
        row["oracle_gap"] = float(oracle_gap_value)
        if teacher_class > 0 and residual_enable:
            row["residual_error"] = float(abs(float(row.get("predicted_residual", 0.0)) - float(teacher_residual_target)))

    debug.update(
        {
            "full_cloud_amount_enabled": True,
            "full_cloud_amount_bin": float(selected_bin.detach().cpu()),
            "full_cloud_amount_residual": pred_residual_float,
            "full_cloud_amount_pred_residual": pred_residual_float,
            "full_cloud_amount_pred_residual_raw": pred_residual_raw_float,
            "full_cloud_amount_selected_base_bin": float(selected_bin.detach().cpu()),
            "full_cloud_amount_selected_residual": pred_residual_float,
            "full_cloud_amount_final_ratio": float(final_ratio.detach().cpu()),
            "full_cloud_amount_drop_count": int(drop_count or 0),
            "full_cloud_amount_noop_selected": bool(selected_class == 0),
            "full_cloud_amount_candidate_count": int(len(rows)),
            "full_cloud_amount_actual_eval_count": int(actual_finished_count),
            "full_cloud_amount_actual_requested_count": int(actual_requested_count),
            "full_cloud_amount_actual_finished_count": int(actual_finished_count),
            "full_cloud_amount_teacher_source": str(teacher_source),
            "full_cloud_amount_teacher_ratio": float(teacher_ratio_value),
            "full_cloud_amount_teacher_base_bin": float(teacher_base_bin),
            "full_cloud_amount_teacher_residual": float(teacher_residual_target),
            "full_cloud_amount_oracle_best_ratio": float(oracle_best_ratio),
            "full_cloud_amount_raw_oracle_best_ratio": float(raw_oracle_best_ratio),
            "full_cloud_amount_oracle_best_actual_delta": float(oracle_best_actual_delta),
            "full_cloud_amount_oracle_best_objective_delta": float(oracle_best_objective_delta),
            "full_cloud_amount_selected_ratio": float(selected_ratio_value),
            "full_cloud_amount_selected_actual_delta": float(selected_actual_delta),
            "full_cloud_amount_selected_objective_delta": float(selected_objective_delta),
            "full_cloud_amount_oracle_gap": float(oracle_gap_value),
            "full_cloud_amount_selected_is_best": bool(selected_is_best_value),
            "full_cloud_amount_selected_is_raw_best": bool(selected_is_raw_best),
            "full_cloud_amount_raw_oracle_gap": float(raw_oracle_gap),
            "full_cloud_amount_actual_finished_nonselected_count": int(actual_finished_nonselected_count),
            "full_cloud_amount_wide_probe_due": bool(wide_probe_due),
            "full_cloud_amount_wide_probe_actual_count": int(wide_probe_actual_count),
            "full_cloud_amount_sequence_memory_ratio": (
                float(sequence_memory_best_entry.get("ratio", float("nan")))
                if isinstance(sequence_memory_best_entry, dict)
                else float("nan")
            ),
            "amount_learning_mode": str(learning_mode),
            "selected_amount_class": int(selected_class),
            "selected_amount_bin": float(selected_bin.detach().cpu()),
            "selected_amount_ratio": float(selected_ratio_value),
            "selected_action_log_prob": float(selected_action_log_prob),
            "amount_temperature": float(amount_temperature),
            "amount_rd_score": float(amount_rd_score_value),
            "amount_policy_loss": float(amount_policy_loss.detach().cpu()),
            "amount_value_loss": float(value_loss.detach().cpu()),
            "amount_advantage": float(amount_advantage_value),
            "sequence_amount_baseline": float(sequence_baseline_value) if sequence_baseline_value is not None else float("nan"),
            "amount_class_histogram": str(amount_class_histogram),
            "amount_max_class_rate": float(amount_max_class_rate),
            "amount_selected_ratio_mean": float(selected_ratio_value),
            "amount_selected_ratio_std": 0.0,
            "where_mode": str(selected_where_row.get("where_mode", where_mode)),
            "effective_where_mode": str(selected_where_row.get("effective_where_mode", selected_where_row.get("where_mode", where_mode))),
            "macro_ratio": case_float(selected_where_row.get("macro_ratio", 0.0), 0.0),
            "micro_ratio": case_float(selected_where_row.get("micro_ratio", 0.0), 0.0),
            "macro_selected_block_count": case_int(selected_where_row.get("macro_selected_block_count", 0), 0),
            "macro_drop_count": case_int(selected_where_row.get("macro_drop_count", 0), 0),
            "micro_drop_count": case_int(selected_where_row.get("micro_drop_count", 0), 0),
            "total_drop_count": case_int(selected_where_row.get("total_drop_count", drop_count or 0), int(drop_count or 0)),
            "selected_block_count": case_int(selected_where_row.get("selected_block_count", 0), 0),
            "micro_selected_block_count": case_int(selected_where_row.get("micro_selected_block_count", 0), 0),
            "max_drop_count_per_block": case_int(selected_where_row.get("max_drop_count_per_block", 0), 0),
            "mean_drop_count_per_selected_block": case_float(
                selected_where_row.get("mean_drop_count_per_selected_block", 0.0),
                0.0,
            ),
            "drop_concentration_top1_block_ratio": case_float(
                selected_where_row.get("drop_concentration_top1_block_ratio", 0.0),
                0.0,
            ),
            "drop_concentration_top5_block_ratio": case_float(
                selected_where_row.get("drop_concentration_top5_block_ratio", 0.0),
                0.0,
            ),
            "hard_where_uses_network_score": bool(selected_where_row.get("hard_where_uses_network_score", False)),
            "heuristic_where_score_mean": case_float(selected_where_row.get("heuristic_where_score_mean", 0.0), 0.0),
            "heuristic_where_score_std": case_float(selected_where_row.get("heuristic_where_score_std", 0.0), 0.0),
            "micro_quota_hit_block_count": case_int(selected_where_row.get("micro_quota_hit_block_count", 0), 0),
            "micro_min_selected_blocks": case_int(selected_where_row.get("micro_min_selected_blocks", 0), 0),
            "micro_candidate_block_count": case_int(selected_where_row.get("micro_candidate_block_count", 0), 0),
            "micro_min_blocks_satisfied": bool(selected_where_row.get("micro_min_blocks_satisfied", False)),
            "micro_min_blocks_fallback_reason": str(selected_where_row.get("micro_min_blocks_fallback_reason", "")),
            "macro_micro_hybrid_fallback": bool(selected_where_row.get("macro_micro_hybrid_fallback", False)),
            "macro_disabled_reason": str(selected_where_row.get("macro_disabled_reason", "")),
            "full_cloud_amount_entropy": float(amount_entropy_t.detach().cpu()),
            "full_cloud_amount_entropy_loss": float(entropy_loss.detach().cpu()),
            "full_cloud_amount_residual_loss": float(residual_loss.detach().cpu()),
            "full_cloud_amount_residual_error": float(residual_error),
            "full_cloud_amount_residual_enabled": bool(residual_enable),
            "full_cloud_amount_residual_max": float(residual_max),
            "full_cloud_amount_residual_teacher_clamped": bool(teacher_residual_clamped),
            "full_cloud_amount_ratio_hist_selected": (
                f"base={float(selected_bin.detach().cpu()):.5f}|res={pred_residual_float:.5f}|final={selected_ratio_value:.5f}"
            ),
            "full_cloud_amount_ratio_hist_teacher": (
                f"base={float(teacher_base_bin):.5f}|res={float(teacher_residual_target):.5f}|final={float(teacher_ratio_value):.5f}"
            ),
            "full_cloud_amount_fine_probe_enabled": bool(fine_ratio_probe_enable),
            "full_cloud_amount_residual_probe_enabled": bool(residual_probe_enable),
            "full_cloud_amount_reuse_where_ranking": bool(reuse_where_ranking_used),
            "full_cloud_amount_reuse_where_ranking_reason": str(reuse_where_ranking_reason),
            "full_cloud_amount_predicted_delta": selected_surrogate,
            "full_cloud_amount_actual_delta": float(selected_actual_delta),
            "full_cloud_amount_actual_objective_delta": float(selected_objective_delta),
            "full_cloud_amount_surrogate_delta": selected_surrogate,
            "full_cloud_amount_geom_loss": float(debug_geom_cost_value),
            "full_cloud_amount_cls_loss": float(cls_loss.detach().cpu()),
            "full_cloud_amount_value_loss": float(value_loss.detach().cpu()),
            "full_cloud_amount_rank_loss": float(rank_loss.detach().cpu()),
            "full_cloud_amount_geom_guard_loss": float(geom_loss_term.detach().cpu()),
            "full_cloud_amount_ratio_reg_loss": float(ratio_reg_loss.detach().cpu()),
            "full_cloud_amount_noop_guard_loss": float(noop_guard_loss.detach().cpu()),
            "full_cloud_amount_total_loss": float(total_loss.detach().cpu()),
            "full_cloud_verified_noop_guard_used": bool(
                str(getattr(args, "sparsepcgc_proposal_inference_mode", "fast")).strip().lower() == "verified"
                and actual_available
                and selected_class != 0
                and float(actual_value) >= float(noop_margin)
            ),
            "actual_bit_objective": str(objective_mode),
            "actual_objective_percent": float(selected_objective_delta),
            "actual_objective_bit_source": str(
                selected_row.get("actual_objective_bit_source", objective_mode)
                if isinstance(selected_row, dict)
                else objective_mode
            ),
            "actual_train_objective_percent": float(selected_objective_delta),
            "actual_bit_percent_used_for_loss": float(selected_objective_delta),
            "actual_forward_value": float(selected_objective_delta),
            "compression_loss_used": float(selected_objective_delta),
        }
    )
    if not math.isfinite(debug["full_cloud_amount_actual_wall_time_total"]):
        debug["full_cloud_amount_actual_wall_time_total"] = 0.0
    if not math.isfinite(debug["full_cloud_amount_actual_wall_time_max"]):
        debug["full_cloud_amount_actual_wall_time_max"] = 0.0
    for row in rows:
        row.pop("_candidate_key", None)
        row.pop("_teacher_score", None)
        row.pop("_surrogate_teacher_score", None)
    return total_loss, debug, rows





def _log_sparsepcgc_restore_debug(args, writer, out_label, prefix="VoxelRestoreDebug"):
    # Phase2: canonical voxel coordsから復元した点群候補のdebugだけを出す。
    # 学習に使うgen_xyzはここでは変更しない。
    if not bool(getattr(args, "sparsepcgc_restore_points_debug", False)):
        return
    if not bool(getattr(args, "_log_this_step", True)):
        return
    if writer is None or not hasattr(writer, "write"):
        return
    if not isinstance(out_label, dict):
        return

    before_coords = out_label.get("canonical_voxel_coords_before", None)
    after_coords = out_label.get("canonical_voxel_coords_after", None)
    restored_xyz = out_label.get("restored_xyz_debug", None)
    restore_info = out_label.get("restore_info", {}) or {}

    def _shape(x):
        if torch.is_tensor(x):
            return tuple(x.shape)
        return None

    def _range_text(x):
        if not torch.is_tensor(x) or x.numel() == 0:
            return "n/a"
        x_det = x.detach()
        return (
            f"min={float(x_det.amin().float().cpu()):.6g}, "
            f"max={float(x_det.amax().float().cpu()):.6g}"
        )

    writer.write(
        f"{prefix}: "
        f"before_coords_shape={_shape(before_coords)}, "
        f"after_coords_shape={_shape(after_coords)}, "
        f"restored_xyz_shape={_shape(restored_xyz)}, "
        f"restored_xyz_range={_range_text(restored_xyz)}, "
        f"restore_input_points={restore_info.get('restore_input_points', 'n/a')}, "
        f"restore_output_points={restore_info.get('restore_output_points', 'n/a')}, "
        f"restore_center={restore_info.get('restore_center', 'n/a')}, "
        f"restore_unique={restore_info.get('restore_unique', 'n/a')}"
    )

def _build_full_cloud_octree_context_for_train(input_xyz, args, coord_scale=None):
    """
    full cloud anchor用の最小full_octree_contextを作る。
    Node/Voxel入力経路へ入れるため、global_voxel_coords/global_qs/global_offsetを必ず持たせる。
    """
    q_result = quantize_sparsepcgc_coords(
        input_xyz,
        args,
        coord_scale=coord_scale,
        offset=None,
        return_metadata=True,
    )

    if isinstance(q_result, tuple) and len(q_result) == 2:
        global_voxel_coords, voxel_meta = q_result
    else:
        global_voxel_coords = q_result
        voxel_meta = {}

    full_octree_context = attach_sparsepcgc_voxel_meta(
        {
            "octree_context_scope": "full_cloud",
            "octree_input_mode": "full_cloud",
        },
        global_voxel_coords.detach().to(dtype=torch.long),
        voxel_meta,
    )

    full_octree_context["full_global_voxel_coords"] = full_octree_context["global_voxel_coords"]
    full_octree_context["full_occupied_voxel_coords"] = full_octree_context["global_voxel_coords"]

    return full_octree_context

def _full_cloud_canonical_meta(full_cloud_canonical_context):
    """
    full cloud で一度だけ作った canonical voxel metadata を取り出す。
    Subtree / actual復元 / full-context loss は必ずこれを使う。
    """
    if not isinstance(full_cloud_canonical_context, dict):
        return {}

    meta = full_cloud_canonical_context.get("sparsepcgc_voxel_meta", None)
    if isinstance(meta, dict):
        return dict(meta)

    out = {}
    if "global_qs" in full_cloud_canonical_context:
        out["global_qs"] = full_cloud_canonical_context["global_qs"]
        out["effective_qs_tensor"] = full_cloud_canonical_context["global_qs"]
    if "global_offset" in full_cloud_canonical_context:
        out["global_offset"] = full_cloud_canonical_context["global_offset"]
        out["global_offset_tensor"] = full_cloud_canonical_context["global_offset"]
    return out

def _full_cloud_anchor_node_count_estimate(full_cloud_canonical_context, args):
    """
    FullCloud anchorで訓練graphを作るか判定するためのnode/voxel数推定値を返す。

    注意:
    ここではforward前なので、Network内部の厳密なnode数はまだ分からない。
    そのため、full cloud canonical voxel coords の点対応数を安全側の上限推定として使う。
    """
    if not isinstance(full_cloud_canonical_context, dict):
        return 0, "context_missing"

    key = str(
        getattr(args, "full_cloud_anchor_node_count_key", "global_voxel_coords")
    ).strip()

    coords = full_cloud_canonical_context.get(key, None)

    if not torch.is_tensor(coords):
        # 指定keyが無い場合は、既存の代表keyへfallbackする。
        for fallback_key in (
            "global_voxel_coords",
            "full_global_voxel_coords",
            "full_occupied_voxel_coords",
        ):
            coords = full_cloud_canonical_context.get(fallback_key, None)
            if torch.is_tensor(coords):
                key = fallback_key
                break

    if not torch.is_tensor(coords):
        return 0, "coords_missing"

    if coords.ndim == 3:
        return int(coords.shape[-1]), key

    if coords.ndim == 2:
        return int(coords.shape[0]), key

    return int(coords.numel()), key


def _resolve_full_cloud_anchor_no_grad(args, full_cloud_canonical_context):
    """
    FullCloud anchorで学習graphを作るか、no-grad teacher更新に落とすかを決める。

    基本方針:
    - full_cloud_anchor_allow_grad=False なら常にno-grad
    - full_cloud_anchor_grad_node_limit<=0 なら常にno-grad
    - node/voxel数推定値が上限を超えたらno-grad
    - 上限内のときだけgradを許可する
    """
    node_count, count_source = _full_cloud_anchor_node_count_estimate(
        full_cloud_canonical_context,
        args,
    )

    allow_grad = bool(getattr(args, "full_cloud_anchor_allow_grad", False))
    node_limit = int(getattr(args, "full_cloud_anchor_grad_node_limit", 50000))
    online_mode = (
        str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
        == "ana_den6_online"
    )
    if online_mode:
        # onlineはfull-cloud方策log-probへ勾配を流すことが要件である。
        # 条件不足時にno-gradへ黙って落とすと、訓練は進んでもWhere/Amount/Actionが学習されない。
        if not allow_grad:
            raise RuntimeError(
                "ana_den6_onlineではfull_cloud_anchor_allow_grad=Trueが必要である"
            )
        if node_limit <= 0:
            raise RuntimeError(
                "ana_den6_onlineではfull_cloud_anchor_grad_node_limitを正値にする必要がある"
            )
        if node_count <= 0:
            raise RuntimeError(
                "ana_den6_onlineでfull-cloud Voxel数を特定できない"
            )
        if node_count > node_limit:
            raise RuntimeError(
                "ana_den6_onlineのfull-cloud Voxel数が勾配上限を超えた: "
                f"{node_count}>{node_limit}。上限を明示的に増やすか入力設定を確認すること"
            )
        return False, f"ana_den6_online_grad_required:{node_count}<={node_limit}", node_count, count_source

    if not allow_grad:
        return True, "full_cloud_anchor_grad_disabled", node_count, count_source

    if node_limit <= 0:
        return True, "full_cloud_anchor_grad_node_limit_non_positive", node_count, count_source

    if node_count <= 0:
        return True, "full_cloud_anchor_node_count_unavailable", node_count, count_source

    if node_count > node_limit:
        return True, f"full_cloud_anchor_node_limit_exceeded:{node_count}>{node_limit}", node_count, count_source

    return False, f"full_cloud_anchor_grad_allowed:{node_count}<={node_limit}", node_count, count_source




def _select_actual_gen_xyz_from_voxel_state(
    args,
    writer,
    model,
    fallback_xyz,
    prefix="VoxelRestoredActual",
    canonical_context=None,
):
    """
    actual compression専用に、model.last_actuator_voxel_state['final_voxel_coords'] から点群を復元する。
    geometry loss用のgen_xyzは変更しない。
    flagがFalseなら完全に既存挙動を維持する。
    """
    if not bool(getattr(args, "use_voxel_restored_points_for_actual", False)):
        return fallback_xyz, {
            "used": False,
            "fallback": False,
            "reason": "disabled",
            "original_gen_points": int(fallback_xyz.shape[-1]) if torch.is_tensor(fallback_xyz) else 0,
            "restored_actual_points": 0,
            "final_voxel_coords_count": 0,
        }

    base_model = model.module if hasattr(model, "module") else model
    voxel_state = getattr(base_model, "last_actuator_voxel_state", None)

    require_state = bool(getattr(args, "voxel_restored_actual_require_state", False))

    def _fallback(reason, *, allow_even_if_required=False):
        if require_state and not allow_even_if_required:
            raise RuntimeError(f"{prefix}: {reason}")

        original_min, original_max = _phase7_tensor_range(fallback_xyz)
        return fallback_xyz, {
            "used": False,
            "fallback": True,
            "reason": reason,
            "original_gen_points": int(fallback_xyz.shape[-1]) if torch.is_tensor(fallback_xyz) else 0,
            "restored_actual_points": 0,
            "final_voxel_coords_count": 0,
            "original_gen_xyz_min": original_min,
            "original_gen_xyz_max": original_max,
            "restored_actual_xyz_min": 0.0,
            "restored_actual_xyz_max": 0.0,
        }

    if not isinstance(voxel_state, dict):
        return _fallback("last_actuator_voxel_state_missing")

    final_voxel_coords = voxel_state.get("final_voxel_coords", None)
    if not torch.is_tensor(final_voxel_coords):
        return _fallback("final_voxel_coords_missing")

    if final_voxel_coords.ndim != 3 or final_voxel_coords.shape[1] != 3:
        return _fallback(f"invalid_final_voxel_coords_shape={tuple(final_voxel_coords.shape)}")

    final_voxel_valid_mask = voxel_state.get("final_voxel_valid_mask", None)
    voxel_step = voxel_state.get("voxel_step", None)
    voxel_offset = voxel_state.get("voxel_offset", None)

    # ============================================================
    # 復元にも full cloud canonical metadata を優先して使う。
    # これにより final_voxel_coords → xyz の復元座標系も一意になる。
    # ============================================================
    meta = _full_cloud_canonical_meta(canonical_context)

    if not meta:
        meta = {}
        if torch.is_tensor(voxel_step):
            meta["effective_qs_tensor"] = voxel_step.detach().to(
                device=final_voxel_coords.device,
                dtype=fallback_xyz.dtype,
            )
            meta["global_qs"] = meta["effective_qs_tensor"]
        if torch.is_tensor(voxel_offset):
            meta["global_offset_tensor"] = voxel_offset.detach().to(
                device=final_voxel_coords.device,
                dtype=fallback_xyz.dtype,
            )
            meta["global_offset"] = meta["global_offset_tensor"]
    else:
        if "effective_qs_tensor" in meta and torch.is_tensor(meta["effective_qs_tensor"]):
            meta["effective_qs_tensor"] = meta["effective_qs_tensor"].detach().to(
                device=final_voxel_coords.device,
                dtype=fallback_xyz.dtype,
            )
            meta["global_qs"] = meta["effective_qs_tensor"]
        if "global_offset_tensor" in meta and torch.is_tensor(meta["global_offset_tensor"]):
            meta["global_offset_tensor"] = meta["global_offset_tensor"].detach().to(
                device=final_voxel_coords.device,
                dtype=fallback_xyz.dtype,
            )
            meta["global_offset"] = meta["global_offset_tensor"]

    coords = final_voxel_coords.detach().to(device=fallback_xyz.device, dtype=torch.long)

    if torch.is_tensor(final_voxel_valid_mask):
        valid_mask = final_voxel_valid_mask.detach().to(device=coords.device, dtype=torch.bool)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.squeeze(1)
    else:
        valid_mask = torch.ones(
            (coords.shape[0], coords.shape[2]),
            device=coords.device,
            dtype=torch.bool,
        )

    restored_list = []
    restored_counts = []

    for b in range(coords.shape[0]):
        valid_b = valid_mask[b]

        if valid_b.ndim != 1 or valid_b.numel() != coords.shape[2]:
            return _fallback(
                f"invalid_final_voxel_valid_mask_shape={tuple(valid_mask.shape)}, "
                f"coords_shape={tuple(coords.shape)}"
            )

        valid_count_b = int(valid_b.detach().bool().sum().cpu())
        if valid_count_b <= 0:
            if writer is not None and hasattr(writer, "write") and bool(getattr(args, "_log_this_step", True)):
                writer.write(
                    f"{prefix}: fallback=True, "
                    f"reason=empty_valid_final_voxel_coords, "
                    f"batch={b}, "
                    f"coords_shape={tuple(coords.shape)}, "
                    f"valid_mask_shape={tuple(valid_mask.shape)}"
                )

            return _fallback(
                "empty_valid_final_voxel_coords",
                allow_even_if_required=True,
            )

        coords_b = coords[b:b + 1, :, valid_b]

        meta_b = dict(meta)
        if "effective_qs_tensor" in meta_b and torch.is_tensor(meta_b["effective_qs_tensor"]):
            meta_b["effective_qs_tensor"] = meta_b["effective_qs_tensor"][b:b + 1]
            meta_b["global_qs"] = meta_b["effective_qs_tensor"]
        if "global_offset_tensor" in meta_b and torch.is_tensor(meta_b["global_offset_tensor"]):
            meta_b["global_offset_tensor"] = meta_b["global_offset_tensor"][b:b + 1]
            meta_b["global_offset"] = meta_b["global_offset_tensor"]

        restored_b, _ = restore_points_from_voxel_coords(
            coords_b,
            meta=meta_b if meta_b else None,
            args=args,
            center=bool(getattr(args, "sparsepcgc_dequantize_center", False)),
            unique=True,
            dtype=fallback_xyz.dtype,
            device=fallback_xyz.device,
        )
        restored_list.append(restored_b)
        restored_counts.append(int(restored_b.shape[-1]))

    if len(set(restored_counts)) != 1:
        return _fallback(f"variable_restored_counts={restored_counts}")

    restored_xyz = torch.cat(restored_list, dim=0).contiguous()

    if bool(getattr(args, "use_voxel_restored_points_for_actual_debug", True)):
        if writer is not None and hasattr(writer, "write") and bool(getattr(args, "_log_this_step", True)):
            restored_det = restored_xyz.detach()
            writer.write(
                f"{prefix}: used=True, "
                f"points={int(restored_xyz.shape[-1])}, "
                f"range_min={float(restored_det.amin().float().cpu()):.6g}, "
                f"range_max={float(restored_det.amax().float().cpu()):.6g}, "
                f"counts={restored_counts}"
            )

    original_min, original_max = _phase7_tensor_range(fallback_xyz)
    restored_min, restored_max = _phase7_tensor_range(restored_xyz)
    final_voxel_count = int(valid_mask.detach().bool().sum().cpu()) if torch.is_tensor(valid_mask) else int(coords.shape[-1])

    return restored_xyz, {
        "used": True,
        "fallback": False,
        "reason": "",
        "points": int(restored_xyz.shape[-1]),
        "counts": restored_counts,
        "original_gen_points": int(fallback_xyz.shape[-1]) if torch.is_tensor(fallback_xyz) else 0,
        "restored_actual_points": int(restored_xyz.shape[-1]),
        "final_voxel_coords_count": int(final_voxel_count),
        "original_gen_xyz_min": original_min,
        "original_gen_xyz_max": original_max,
        "restored_actual_xyz_min": restored_min,
        "restored_actual_xyz_max": restored_max,
    }


def _restore_codec_xyz_from_global_voxels(args, coords_b3n, context, like_xyz):
    if not torch.is_tensor(coords_b3n) or coords_b3n.ndim != 3 or coords_b3n.shape[1] != 3:
        return None
    meta = _full_cloud_canonical_meta(context)
    if not meta:
        meta = {}
        if isinstance(context, dict):
            for key in ("global_qs", "global_offset", "sparsepcgc_voxel_meta"):
                if key in context:
                    meta[key] = context[key]
    if "effective_qs_tensor" in meta and torch.is_tensor(meta["effective_qs_tensor"]):
        meta["effective_qs_tensor"] = meta["effective_qs_tensor"].detach().to(
            device=like_xyz.device,
            dtype=like_xyz.dtype,
        )
        meta["global_qs"] = meta["effective_qs_tensor"]
    if "global_offset_tensor" in meta and torch.is_tensor(meta["global_offset_tensor"]):
        meta["global_offset_tensor"] = meta["global_offset_tensor"].detach().to(
            device=like_xyz.device,
            dtype=like_xyz.dtype,
        )
        meta["global_offset"] = meta["global_offset_tensor"]
    restored, _ = restore_points_from_voxel_coords(
        coords_b3n.detach().to(device=like_xyz.device, dtype=torch.long),
        meta=meta if meta else None,
        args=args,
        center=bool(getattr(args, "sparsepcgc_dequantize_center", False)),
        unique=True,
        dtype=like_xyz.dtype,
        device=like_xyz.device,
    )
    return restored












def _ceil_log2_int(value):
    value = int(max(int(value), 1))
    return max(int(math.ceil(math.log2(float(value + 1)))), 1)


def _sparsepcgc_edit_record_leaf_bits(args, unique_count, edit_count):
    if not bool(getattr(args, "sparsepcgc_edit_record_bits_enabled", True)):
        return 0.0
    edit_count = max(int(edit_count), 0)
    if edit_count <= 0:
        return 0.0
    base_bits = max(float(getattr(args, "sparsepcgc_edit_record_base_bits", 8.0)), 0.0)
    count_bits = max(
        _ceil_log2_int(unique_count),
        int(getattr(args, "sparsepcgc_edit_record_count_bits_min", 4)),
    )
    # One leaf edit can be signaled as a coded node index plus child slot.
    address_bits = max(
        _ceil_log2_int(unique_count) + 3,
        int(getattr(args, "sparsepcgc_edit_record_leaf_address_bits_min", 10)),
    )
    return float(base_bits + count_bits + edit_count * address_bits)


def _sparsepcgc_edit_record_subtree_move_bits(args, unique_count, move_count, level_shift=1):
    if not bool(getattr(args, "sparsepcgc_edit_record_bits_enabled", True)):
        return 0.0
    move_count = max(int(move_count), 0)
    if move_count <= 0:
        return 0.0
    base_bits = max(float(getattr(args, "sparsepcgc_edit_record_base_bits", 8.0)), 0.0)
    count_bits = max(
        _ceil_log2_int(unique_count),
        int(getattr(args, "sparsepcgc_edit_record_count_bits_min", 4)),
    )
    # A subtree move records a coarse parent address, source slot, target slot, and shift.
    transform_bits = max(
        _ceil_log2_int(unique_count) + 3 + 3 + _ceil_log2_int(max(int(level_shift), 1)),
        int(getattr(args, "sparsepcgc_edit_record_subtree_move_bits_min", 16)),
    )
    return float(base_bits + count_bits + transform_bits)




def _sparsepcgc_edit_record_total_bits(
    args,
    unique_count,
    *,
    drop_count=0,
    add_count=0,
    subtree_move_count=0,
    subtree_move_level_shift=1,
):
    if not bool(getattr(args, "sparsepcgc_edit_record_bits_enabled", True)):
        return 0.0
    bits = 0.0
    bits += _sparsepcgc_edit_record_leaf_bits(args, unique_count, drop_count)
    bits += _sparsepcgc_edit_record_leaf_bits(args, unique_count, add_count)
    bits += _sparsepcgc_edit_record_subtree_move_bits(
        args,
        unique_count,
        subtree_move_count,
        level_shift=subtree_move_level_shift,
    )
    return float(bits * sparsepcgc_effective_edit_record_bit_scale(args))


def _sparsepcgc_objective_percent_with_edit_record(args, raw_bit, base_bit, edit_record_bits):
    base_bit = max(abs(float(base_bit)), 1.0)
    raw_percent = 100.0 * (float(raw_bit) - float(base_bit)) / base_bit
    billed_percent = 100.0 * (
        float(raw_bit) + float(max(edit_record_bits, 0.0)) - float(base_bit)
    ) / base_bit
    return float(raw_percent), float(billed_percent)




















def _sparsepcgc_geometry_penalty_percent(
    args,
    unique_count,
    *,
    drop_count=0,
    add_count=0,
    move_count=0,
    level_shift=1,
):
    lambda_geom = max(float(getattr(args, "sparsepcgc_actual_oracle_geometry_lambda", 0.05)), 0.0)
    if lambda_geom <= 0.0:
        return 0.0
    unique_count = max(int(unique_count), 1)
    edit_mass = float(max(int(drop_count), 0) + max(int(add_count), 0))
    if int(move_count) > 0:
        edit_mass += float(max(int(move_count), 0)) * math.sqrt(max(float(level_shift), 1.0))
    return float(lambda_geom * 100.0 * edit_mass / float(unique_count))































def _copy_sparsepcgc_actual_oracle_debug_for_metrics(target, debug):
    if not isinstance(target, dict) or not isinstance(debug, dict):
        return target
    target.update(
        {
            "actual_oracle_enabled": bool(debug.get("enabled", False)),
            "actual_oracle_used": bool(debug.get("used", False)),
            "actual_oracle_generated_candidate_count": int(debug.get("generated_candidate_count", debug.get("candidate_pool_count", 0)) or 0),
            "actual_oracle_accepted_candidate_count": int(debug.get("accepted_candidate_count", 0) or 0),
            "actual_oracle_accepted_prune_count": int(debug.get("accepted_prune_count", 0) or 0),
            "actual_oracle_accepted_add_count": int(debug.get("accepted_add_count", 0) or 0),
            "actual_oracle_accepted_adjust_count": int(debug.get("accepted_adjust_count", 0) or 0),
            "actual_oracle_accepted_subtree_move_count": int(debug.get("accepted_subtree_move_count", 0) or 0),
            "actual_oracle_accepted_parent_collapse_count": int(debug.get("accepted_parent_collapse_count", 0) or 0),
            "actual_oracle_accepted_pattern_canonicalize_count": int(debug.get("accepted_pattern_canonicalize_count", 0) or 0),
            "actual_oracle_noop_label_count": int(debug.get("noop_label_count", 0) or 0),
            "actual_oracle_noop_label_weight": float(debug.get("noop_label_weight", 0.0) or 0.0),
            "actual_oracle_high_rate_mppov_count": int(debug.get("high_rate_mppov_count", 0) or 0),
            "actual_oracle_low_prob_occupied_count": int(debug.get("low_prob_occupied_count", 0) or 0),
            "actual_oracle_single_child_chain_count": int(debug.get("single_child_chain_count", 0) or 0),
            "actual_oracle_context_pattern_candidate_count": int(debug.get("context_pattern_candidate_count", 0) or 0),
            "actual_oracle_eval_count": int(debug.get("tested_count", 0) or 0),
            "actual_oracle_eval_max_configured": int(debug.get("actual_eval_max_configured", debug.get("actual_eval_max", 0)) or 0),
            "actual_oracle_eval_max": int(debug.get("actual_eval_max", 0) or 0),
            "actual_oracle_eval_scope": str(debug.get("actual_eval_scope", "")),
            "actual_oracle_eval_full_coord_count": int(debug.get("actual_eval_full_coord_count", 0) or 0),
            "actual_oracle_full_cloud_teacher_required": bool(debug.get("full_cloud_teacher_required", False)),
            "actual_oracle_full_cloud_teacher_eval_available": bool(debug.get("full_cloud_teacher_eval_available", False)),
            "actual_oracle_time": float(debug.get("actual_oracle_time", 0.0) or 0.0),
            "actual_oracle_original_actual_cache_hit": bool(debug.get("original_actual_cache_hit", False)),
            "actual_oracle_original_actual_encode_time": float(debug.get("original_actual_encode_time", 0.0) or 0.0),
            "actual_oracle_candidate_actual_encode_time": float(debug.get("candidate_actual_encode_time", 0.0) or 0.0),
            "actual_oracle_released_main_cuda_cache": bool(debug.get("released_main_cuda_cache", False)),
            "actual_oracle_edit_record_bits": float(debug.get("selected_edit_record_bits", 0.0) or 0.0),
            "actual_oracle_best_edit_record_bits": float(debug.get("best_edit_record_bits", 0.0) or 0.0),
            "actual_oracle_raw_percent": float(debug.get("selected_raw_percent", 0.0) or 0.0),
            "actual_oracle_best_raw_percent": float(debug.get("best_raw_percent", 0.0) or 0.0),
            "actual_oracle_delta_actual_percent": float(debug.get("delta_actual_percent", 0.0) or 0.0),
            "actual_oracle_best_actual_percent": float(debug.get("best_actual_percent", 0.0) or 0.0),
            "actual_oracle_proxy_percent": float(debug.get("selected_proxy_percent", 0.0) or 0.0),
            "actual_oracle_best_proxy_percent": float(debug.get("best_proxy_percent", 0.0) or 0.0),
            "actual_oracle_geometry_percent": float(debug.get("selected_geometry_percent", 0.0) or 0.0),
            "actual_oracle_original_actual_bits": float(debug.get("original_actual_bits", 0.0) or 0.0),
            "actual_oracle_edited_actual_bits": float(debug.get("edited_actual_bits", 0.0) or 0.0),
            "actual_oracle_joint_tested_count": int(debug.get("joint_tested_count", 0) or 0),
            "actual_oracle_joint_improving_count": int(debug.get("joint_improving_count", 0) or 0),
            "actual_oracle_group_tested_count": int(debug.get("group_tested_count", 0) or 0),
            "actual_oracle_group_improving_count": int(debug.get("group_improving_count", 0) or 0),
            "actual_oracle_full_cloud_macro_fallback_triggered": bool(debug.get("full_cloud_macro_fallback_triggered", False)),
            "actual_oracle_full_cloud_macro_fail_extra_eval_max": int(debug.get("full_cloud_macro_fail_extra_eval_max", 0) or 0),
            "actual_oracle_full_cloud_macro_fallback_candidate_generation_enabled": bool(
                debug.get("full_cloud_macro_fallback_candidate_generation_enabled", False)
            ),
            "actual_oracle_full_cloud_macro_tested_count": int(debug.get("full_cloud_macro_tested_count", 0) or 0),
            "actual_oracle_full_cloud_macro_improving_count": int(debug.get("full_cloud_macro_improving_count", 0) or 0),
            "actual_oracle_full_cloud_macro_best_percent": float(debug.get("full_cloud_macro_best_percent", 0.0) or 0.0),
            "actual_oracle_full_cloud_macro_best_ratio": float(debug.get("full_cloud_macro_best_ratio", 0.0) or 0.0),
            "actual_oracle_full_cloud_macro_best_drop_count": int(debug.get("full_cloud_macro_best_drop_count", 0) or 0),
            "actual_oracle_macro_prune_tested_count": int(debug.get("macro_prune_tested_count", 0) or 0),
            "actual_oracle_macro_prune_improving_count": int(debug.get("macro_prune_improving_count", 0) or 0),
            "actual_oracle_macro_prune_best_percent": float(debug.get("macro_prune_best_percent", 0.0) or 0.0),
            "actual_oracle_macro_prune_best_ratio": float(debug.get("macro_prune_best_ratio", 0.0) or 0.0),
            "actual_oracle_macro_prune_best_drop_count": int(debug.get("macro_prune_best_drop_count", 0) or 0),
            "actual_oracle_macro_prune_best_variant": str(debug.get("macro_prune_best_variant", "")),
            "actual_oracle_macro_prune_best_proxy_percent": float(debug.get("macro_prune_best_proxy_percent", 0.0) or 0.0),
            "actual_oracle_parent_prune_tested_count": int(debug.get("parent_prune_tested_count", 0) or 0),
            "actual_oracle_parent_prune_improving_count": int(debug.get("parent_prune_improving_count", 0) or 0),
            "actual_oracle_pattern_plan_tested_count": int(debug.get("pattern_plan_tested_count", 0) or 0),
            "actual_oracle_pattern_plan_improving_count": int(debug.get("pattern_plan_improving_count", 0) or 0),
            "actual_oracle_subtree_move_tested_count": int(debug.get("subtree_move_tested_count", 0) or 0),
            "actual_oracle_subtree_move_improving_count": int(debug.get("subtree_move_improving_count", 0) or 0),
            "actual_oracle_operation": str(debug.get("reason", "")),
        }
    )
    return target


def _unwrap_train_model(model):
    # DataParallelで包まれている場合は中身のモデルを取り出す
    return model.module if hasattr(model, "module") else model


def _safe_scalar_for_grad_log(value):
    # CSV保存用にTensor/数値をfloatへ変換する
    if value is None:
        return None
    if not torch.is_tensor(value):
        try:
            return float(value)
        except Exception:
            return None
    try:
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu())
    except Exception:
            return None

def _phase7_debug_enabled(args, global_step):
    if bool(getattr(args, "compact_step_text_log", False)):
        return False
    if not bool(getattr(args, "phase7_debug", True)):
        return False
    interval = max(int(getattr(args, "phase7_debug_every", 10)), 1)
    return bool(getattr(args, "_log_this_step", False)) or (int(global_step) % interval == 0)


def _phase7_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            value = value.detach().float()
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return float(value.mean().cpu())
        return float(value)
    except Exception:
        return float(default)


def _phase7_tensor_range(x):
    if not torch.is_tensor(x) or x.numel() == 0:
        return 0.0, 0.0
    x_det = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    return float(x_det.amin().cpu()), float(x_det.amax().cpu())


def _phase7_final_voxel_count(model):
    base_model = model.module if hasattr(model, "module") else model
    voxel_state = getattr(base_model, "last_actuator_voxel_state", None)
    if not isinstance(voxel_state, dict):
        return 0
    valid_mask = voxel_state.get("final_voxel_valid_mask", None)
    coords = voxel_state.get("final_voxel_coords", None)
    if torch.is_tensor(valid_mask):
        return int(valid_mask.detach().bool().sum().cpu())
    if torch.is_tensor(coords):
        return int(coords.shape[-1])
    return 0


def _phase7_update_from_structure(comp_debug, structure_debug, *, is_anchor_step):
    if not isinstance(comp_debug, dict) or not isinstance(structure_debug, dict):
        return

    for key in (
        "network_voxel_node_input_requested",
        "network_voxel_node_input_used",
        "network_voxel_node_fallback",
        "network_voxel_node_fallback_reason",
        "network_voxel_node_source",
        "network_voxel_node_count",
        "network_voxel_node_feature_shape",
        "full_cloud_anchor_node_voxel_used",
        "subtree_node_voxel_used",

        # ============================================================
        # Phase5:
        # Phase4でNetwork側が出した構造整合性debugもcomp_debugへ渡す。
        # ============================================================
        "phase4_cost_attribution_input_mode",
        "phase4_cost_scores_requires_grad",
        "phase4_cost_logits_requires_grad",
        "phase4_cause_entropy",
        "phase4_aggregation_key_source",
        "phase4_aggregation_unit_count",
        "phase4_aggregation_min_unit_size",
        "phase4_aggregation_max_unit_size",
        "phase4_structural_key_source",
        "cause_aggregation_unit_mode",
        "local_recomputed",
        "structure_local_recomputed",
       "actuator_local_recomputed",

        # Section2:
        # leaf pattern candidate診断をmetric CSVへ流すためのdebug key。
        "leaf_pattern_available",
        "leaf_pattern_source",
        "leaf_pattern_reason",
        "leaf_unique_parent_count",
        "leaf_unique_pattern_count",
        "leaf_mean_child_count",
        "leaf_single_child_parent_ratio",
        "leaf_max_pattern_frequency",
        "leaf_candidate_available",
        "leaf_delete_gain_mean",
        "leaf_add_gain_mean",
        "leaf_move_gain_mean",
        "leaf_high_gain_candidate_ratio",

        # Section3:
        "leaf_feature_integration_used",
        "leaf_feature_best_gain_mean",
        "leaf_feature_best_gain_max",

        # Section4:
        "leaf_actuator_prior_enabled",
        "leaf_actuator_drop_prior_mean",
        "leaf_actuator_add_prior_mean",
        "leaf_actuator_move_prior_mean",
        "leaf_actuator_best_prior_mean",
        "leaf_actuator_best_prior_max",

        "leaf_target_direction_prior_enabled",
        "leaf_add_target_match_ratio",
        "leaf_move_target_match_ratio",
        "leaf_add_target_bias_mean",
        "leaf_move_target_bias_mean",

        "actual_oracle_force_no_edit_used",
        "actual_oracle_noop_label_count",
        "actual_oracle_noop_label_weight",
        "actual_oracle_has_drop",
        "prune_after_prior_mode",
        "phase0_network_prune_mode",
        "actual_gate_prune_enabled",
        "actual_gate_prune_allowed",
        "hard_prune_actual_allowed",
        "hard_drop_block_reason",
        "hard_drop_count_trace",
        "codec_prune_prior_phase",
        "raw_learned_drop_ratio",
        "learned_drop_ratio_before_floor",
        "learned_drop_ratio_after_floor",
        "learned_drop_ratio_before_gate",
        "learned_drop_ratio_after_gate",
        "effective_drop_ratio_for_hard_count",
        "drop_operation_gate",
        "voxel_count",
        "delete_candidate_count",
        "delete_candidate_point_count",
        "delete_candidate_empty_reason",
        "hard_delete_selection_count",
        "pre_round_target_count",
        "post_round_target_count",
        "min_hard_drop_count_floor_applied",
        "hard_mask_count",
        "final_hard_drop_count",
        "selected_drop_count_hard",
        "drop_ratio_hard",
        "phase0_network_mode_but_hard_drop_zero",
        "phase0_noop_only_collapse_detected",
        "collapse_reason",
    ):
        if key in structure_debug:
            comp_debug[key] = structure_debug.get(key)

    comp_debug["full_cloud_anchor_node_voxel_used"] = bool(
        is_anchor_step and bool(structure_debug.get("network_voxel_node_input_used", False))
    )
    comp_debug["subtree_node_voxel_used"] = bool(
        (not is_anchor_step) and bool(structure_debug.get("network_voxel_node_input_used", False))
    )
def _phase5_structure_safety_debug(args, structure_debug, *, is_anchor_step):
    """
    Phase5:
    Phase4でNetwork側が出したNode/Voxel・aggregation debugを、
    train.py側で監査できる形に正規化する。

    ここではTensorを保持しない。
    CSV/ログ用のbool, int, float, strだけを返す。
    """
    if not isinstance(structure_debug, dict):
        return {
            "phase5_structure_debug_available": False,
            "phase5_structure_safety_ok": False,
            "phase5_structure_safety_reason": "structure_debug_missing",
        }

    node_requested = bool(structure_debug.get("network_voxel_node_input_requested", False))
    node_used = bool(structure_debug.get("network_voxel_node_input_used", False))
    node_fallback = bool(structure_debug.get("network_voxel_node_fallback", False))
    node_fallback_reason = str(structure_debug.get("network_voxel_node_fallback_reason", ""))

    cost_input_mode = str(structure_debug.get("phase4_cost_attribution_input_mode", "unknown"))
    aggregation_key_source = str(structure_debug.get("phase4_aggregation_key_source", "unknown"))
    structural_key_source = str(structure_debug.get("phase4_structural_key_source", "unknown"))
    cause_unit_mode = str(structure_debug.get("cause_aggregation_unit_mode", "unknown"))

    unit_count = int(structure_debug.get("phase4_aggregation_unit_count", 0) or 0)
    min_unit_size = int(structure_debug.get("phase4_aggregation_min_unit_size", 0) or 0)
    max_unit_size = int(structure_debug.get("phase4_aggregation_max_unit_size", 0) or 0)

    valid_key_sources = {
        "full_unit_keys",
        "analysis_unit_keys",
        "structure.structural_voxel_key",
        "structure.point_feature_voxel_key",
        "structure_b.structural_voxel_key",
        "structure_b.point_feature_voxel_key",
        "canonical_subtree_tree.global_voxel_coords_hash",
        "full_octree_context.global_voxel_coords_hash",
    }

    valid_structural_sources = {
        "global_morton_keys",
        "global_voxel_coords_hash",
    }

    raw_structure_local_recomputed = bool(
        structure_debug.get("local_recomputed", False)
    )
    raw_cause_local_recomputed = (
        str(cause_unit_mode).strip().lower() == "local_recomputed"
    )
    raw_actuator_local_recomputed = bool(
        structure_debug.get("actuator_local_recomputed", False)
    )
    raw_structure_debug_local_recomputed = bool(
        structure_debug.get("structure_local_recomputed", False)
    )

    local_recomputed = bool(
        raw_structure_local_recomputed
        or raw_cause_local_recomputed
        or raw_actuator_local_recomputed
        or raw_structure_debug_local_recomputed
    )

    canonical_node_path_ok = bool(
        node_used
        and not node_fallback
        and cost_input_mode == "node_voxel"
        and aggregation_key_source in valid_key_sources
        and structural_key_source in valid_structural_sources
        and unit_count > 0
        and max_unit_size > 0
    )

    reasons = []

    if node_requested and not node_used:
        reasons.append("node_voxel_requested_but_not_used")

    if node_fallback:
        reasons.append(f"node_voxel_fallback:{node_fallback_reason}")

    if node_used and cost_input_mode not in {"node_voxel", "unknown"}:
        reasons.append(f"cost_attribution_input_mode_not_node_voxel:{cost_input_mode}")

    if aggregation_key_source not in valid_key_sources:
        reasons.append(f"invalid_aggregation_key_source:{aggregation_key_source}")

    if unit_count <= 0:
        reasons.append("aggregation_unit_count_zero")

    if max_unit_size <= 0:
        reasons.append("aggregation_max_unit_size_zero")

    if structural_key_source not in valid_structural_sources and node_used:
        reasons.append(f"invalid_structural_key_source:{structural_key_source}")

    if (
        local_recomputed
        and bool(getattr(args, "phase5_forbid_local_recompute", True))
        and not canonical_node_path_ok
    ):
        reasons.append("local_recomputed_detected")

    unit_collapse_warn = bool(unit_count == 1 and max_unit_size > 1)
    if (
        unit_collapse_warn
        and bool(getattr(args, "phase5_warn_unit_collapse", True))
        and bool(getattr(args, "phase5_guard_unit_collapse_as_error", False))
    ):
        reasons.append("aggregation_unit_collapse")

    ok = len(reasons) == 0

    return {
        "phase5_structure_debug_available": True,
        "phase5_structure_safety_ok": bool(ok),
        "phase5_structure_safety_reason": "ok" if ok else "|".join(reasons),
        "phase5_is_anchor_step": bool(is_anchor_step),
        "phase5_node_voxel_requested": bool(node_requested),
        "phase5_node_voxel_used": bool(node_used),
        "phase5_node_voxel_fallback": bool(node_fallback),
        "phase5_node_voxel_fallback_reason": str(node_fallback_reason),
        "phase5_cost_attribution_input_mode": str(cost_input_mode),
        "phase5_aggregation_key_source": str(aggregation_key_source),
        "phase5_structural_key_source": str(structural_key_source),
        "phase5_cause_aggregation_unit_mode": str(cause_unit_mode),
        "phase5_aggregation_unit_count": int(unit_count),
        "phase5_aggregation_min_unit_size": int(min_unit_size),
        "phase5_aggregation_max_unit_size": int(max_unit_size),
        "phase5_local_recomputed": bool(local_recomputed),
        "phase5_raw_structure_local_recomputed": bool(raw_structure_local_recomputed),
        "phase5_raw_cause_local_recomputed": bool(raw_cause_local_recomputed),
        "phase5_raw_actuator_local_recomputed": bool(raw_actuator_local_recomputed),
        "phase5_raw_structure_debug_local_recomputed": bool(raw_structure_debug_local_recomputed),
        "phase5_canonical_node_path_ok": bool(canonical_node_path_ok),
        "phase5_unit_collapse_warn": bool(unit_collapse_warn),
    }


def _phase5_apply_structure_guard(args, writer, phase5_debug, *, global_step):
    """
    Phase5:
    構造経路の異常を検出したとき、設定に応じて学習を止める。
    """
    if not isinstance(phase5_debug, dict):
        return

    if not bool(getattr(args, "phase5_structure_guard", True)):
        return

    if bool(phase5_debug.get("phase5_structure_safety_ok", False)):
        return

    reason = str(phase5_debug.get("phase5_structure_safety_reason", "unknown"))

    message = (
        "Phase5StructureGuard: "
        f"global_step={int(global_step)}, "
        f"ok=False, "
        f"reason={reason}, "
        f"node_used={bool(phase5_debug.get('phase5_node_voxel_used', False))}, "
        f"fallback={bool(phase5_debug.get('phase5_node_voxel_fallback', False))}, "
        f"cost_input={phase5_debug.get('phase5_cost_attribution_input_mode', 'unknown')}, "
        f"agg_source={phase5_debug.get('phase5_aggregation_key_source', 'unknown')}, "
        f"struct_source={phase5_debug.get('phase5_structural_key_source', 'unknown')}, "
        f"unit_count={int(phase5_debug.get('phase5_aggregation_unit_count', 0) or 0)}, "
        f"unit_size=[{int(phase5_debug.get('phase5_aggregation_min_unit_size', 0) or 0)}, "
        f"{int(phase5_debug.get('phase5_aggregation_max_unit_size', 0) or 0)}], "
        f"canonical_node_path_ok={bool(phase5_debug.get('phase5_canonical_node_path_ok', False))}, "
        f"raw_local_structure={bool(phase5_debug.get('phase5_raw_structure_local_recomputed', False))}, "
        f"raw_local_cause={bool(phase5_debug.get('phase5_raw_cause_local_recomputed', False))}, "
        f"raw_local_actuator={bool(phase5_debug.get('phase5_raw_actuator_local_recomputed', False))}, "
        f"raw_local_structure_debug={bool(phase5_debug.get('phase5_raw_structure_debug_local_recomputed', False))}"
    )

    if writer is not None and hasattr(writer, "write"):
        writer.write(message)

    if bool(getattr(args, "phase5_structure_guard_raise", True)):
        raise RuntimeError(message)

def _phase7_update_from_voxel_state(comp_debug, model):
    if not isinstance(comp_debug, dict):
        return

    base_model = model.module if hasattr(model, "module") else model
    voxel_state = getattr(base_model, "last_actuator_voxel_state", None)

    if not isinstance(voxel_state, dict):
        comp_debug["phase7_actuator_voxel_state_available"] = False
        return

    comp_debug["phase7_actuator_voxel_state_available"] = True

    key_map = {
        "drop_ratio_soft": "drop_ratio_soft",
        "drop_ratio_hard": "drop_ratio_hard",
        "add_ratio_soft": "add_ratio_soft",
        "add_ratio_hard": "add_ratio_hard",
        "move_ratio_soft": "move_ratio_soft",
        "move_ratio_hard": "move_ratio_hard",
        "add_ratio_loss_value": "add_ratio_loss_value",
        "add_consistency_loss_value": "add_consistency_loss_value",
        "voxel_soft_drop_mean": "voxel_soft_drop_mean",
        "voxel_soft_add_mean": "voxel_soft_add_mean",
        "voxel_soft_move_mean": "voxel_soft_move_mean",
        "voxel_edit_drop_count": "voxel_edit_drop_count",
        "voxel_edit_add_count": "voxel_edit_add_count",
        "voxel_edit_move_count": "voxel_edit_move_count",
        "same_voxel_move_rejected": "voxel_edit_same_voxel_move_rejected",
        "existing_target_rejected": "voxel_edit_existing_target_rejected",
        "duplicate_target_rejected": "voxel_edit_duplicate_target_rejected",
        "child_slot_rejected": "voxel_edit_child_slot_rejected",
        "empty_target_rejected": "voxel_edit_empty_target_rejected",
    }

    for out_key, state_key in key_map.items():
        comp_debug[out_key] = _phase7_float(
            voxel_state.get(state_key, None),
            0.0,
        )

    comp_debug["final_voxel_coords_count"] = int(
        _phase7_final_voxel_count(model)
    )

def _phase7_writer_line(args, writer, text):
    if bool(getattr(args, "compact_step_text_log", False)):
        return
    if writer is not None and hasattr(writer, "write"):
        writer.write(text)
    if bool(getattr(args, "phase7_debug_print", True)):
        print(text)

def _phase7_should_log_interval(args, global_step, every_attr, default_every):
    interval = max(int(getattr(args, every_attr, default_every)), 1)
    return bool(getattr(args, "_log_this_step", False)) or (int(global_step) % interval == 0)


def _phase7_apply_ablation_mode(args, writer):
    """
    Phase7-4:
    phase7_ablation_mode != none のときだけ既存argsを上書きする。
    none の場合は完全に既存挙動を維持する。
    """
    mode = str(getattr(args, "phase7_ablation_mode", "none")).strip().lower()
    if mode in {"", "none"}:
        setattr(args, "_phase7_ablation_applied", False)
        setattr(args, "_phase7_ablation_effective_mode", "none")
        return

    valid_modes = {
        "baseline",
        "voxel_actual_only",
        "full_context_only",
        "correction_only",
        "voxel_actual_full_context",
        "full_phase7",
        "debug_only",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported phase7_ablation_mode: {mode}")

    # 既存状態を記録する。ログ用であり、復元はしない。
    before = {
        "use_voxel_restored_points_for_actual": bool(getattr(args, "use_voxel_restored_points_for_actual", False)),
        "full_context_subtree_soft_proxy": bool(getattr(args, "full_context_subtree_soft_proxy", True)),
        "full_cloud_actual_correction_loss_enable": bool(getattr(args, "full_cloud_actual_correction_loss_enable", False)),
        "full_cloud_actual_correction_soft_proxy": bool(getattr(args, "full_cloud_actual_correction_soft_proxy", True)),
        "phase7_debug": bool(getattr(args, "phase7_debug", True)),
        "phase7_grad_debug": bool(getattr(args, "phase7_grad_debug", False)),
        "phase7_metric_columns": bool(getattr(args, "phase7_metric_columns", True)),
    }

    if mode == "baseline":
        args.use_voxel_restored_points_for_actual = False
        args.full_context_subtree_soft_proxy = False
        args.full_cloud_actual_correction_loss_enable = False
        args.full_cloud_actual_correction_soft_proxy = False

    elif mode == "voxel_actual_only":
        args.use_voxel_restored_points_for_actual = True
        args.full_context_subtree_soft_proxy = False
        args.full_cloud_actual_correction_loss_enable = False
        args.full_cloud_actual_correction_soft_proxy = False

    elif mode == "full_context_only":
        args.use_voxel_restored_points_for_actual = False
        args.full_context_subtree_soft_proxy = True
        args.full_cloud_actual_correction_loss_enable = False
        args.full_cloud_actual_correction_soft_proxy = False

    elif mode == "correction_only":
        args.use_voxel_restored_points_for_actual = False
        args.full_context_subtree_soft_proxy = False
        args.full_cloud_actual_correction_loss_enable = True
        args.full_cloud_actual_correction_soft_proxy = True

    elif mode == "voxel_actual_full_context":
        args.use_voxel_restored_points_for_actual = True
        args.full_context_subtree_soft_proxy = True
        args.full_cloud_actual_correction_loss_enable = False
        args.full_cloud_actual_correction_soft_proxy = False

    elif mode == "full_phase7":
        args.use_voxel_restored_points_for_actual = True
        args.full_context_subtree_soft_proxy = True
        args.full_cloud_actual_correction_loss_enable = True
        args.full_cloud_actual_correction_soft_proxy = True

    elif mode == "debug_only":
        args.use_voxel_restored_points_for_actual = False
        args.full_context_subtree_soft_proxy = False
        args.full_cloud_actual_correction_loss_enable = False
        args.full_cloud_actual_correction_soft_proxy = False
        args.phase7_debug = True
        args.phase7_grad_debug = True
        args.phase7_metric_columns = True
        args.phase7_debug_print = True

    after = {
        "use_voxel_restored_points_for_actual": bool(getattr(args, "use_voxel_restored_points_for_actual", False)),
        "full_context_subtree_soft_proxy": bool(getattr(args, "full_context_subtree_soft_proxy", True)),
        "full_cloud_actual_correction_loss_enable": bool(getattr(args, "full_cloud_actual_correction_loss_enable", False)),
        "full_cloud_actual_correction_soft_proxy": bool(getattr(args, "full_cloud_actual_correction_soft_proxy", True)),
        "phase7_debug": bool(getattr(args, "phase7_debug", True)),
        "phase7_grad_debug": bool(getattr(args, "phase7_grad_debug", False)),
        "phase7_metric_columns": bool(getattr(args, "phase7_metric_columns", True)),
    }

    setattr(args, "_phase7_ablation_applied", True)
    setattr(args, "_phase7_ablation_effective_mode", mode)
    setattr(args, "_phase7_ablation_before", before)
    setattr(args, "_phase7_ablation_after", after)

    if bool(getattr(args, "phase7_ablation_log", True)):
        _phase7_writer_line(
            args,
            writer,
            "Phase7AblationMode: "
            f"mode={mode}, "
            f"voxel_actual={after['use_voxel_restored_points_for_actual']}, "
            f"full_context_soft={after['full_context_subtree_soft_proxy']}, "
            f"correction_loss={after['full_cloud_actual_correction_loss_enable']}, "
            f"correction_soft={after['full_cloud_actual_correction_soft_proxy']}, "
            f"debug={after['phase7_debug']}, "
            f"grad_debug={after['phase7_grad_debug']}, "
            f"metric_columns={after['phase7_metric_columns']}"
        )

def _print_phase7_recommended_commands_and_exit():
    """
    Phase7-5:
    推奨軽量実験コマンドを表示して終了する。
    実験を自動実行しない。
    """
    base = (
        "python train.py "
        "--surrogate_step 0 "
        "--phase7_eval_summary True "
        "--phase7_eval_summary_every 1 "
        "--phase7_debug True "
        "--phase7_metric_columns True "
        "--print_rate 1 "
        "--max_train_steps 10"
    )

    commands = {
        "baseline": f"{base} --phase7_ablation_mode baseline",
        "voxel_actual_only": f"{base} --phase7_ablation_mode voxel_actual_only",
        "full_context_only": f"{base} --phase7_ablation_mode full_context_only",
        "correction_only": f"{base} --phase7_ablation_mode correction_only",
        "voxel_actual_full_context": f"{base} --phase7_ablation_mode voxel_actual_full_context",
        "full_phase7": f"{base} --phase7_ablation_mode full_phase7 --max_train_steps 30",
    }

    print("Phase7 recommended lightweight commands:")
    for name, command in commands.items():
        print(f"\n[{name}]")
        print(command)

def _phase7_grad_sanity_stats(model, zero_eps=1e-12):
    """
    Phase7-4:
    主要module/headのgrad状態を軽量に集計する。
    graphを保持しないため、必ずdetachしたgradだけを見る。
    """
    base_model = model.module if hasattr(model, "module") else model
    targets = {
        "drop_head": ["actuator.drop_head."],
        "add_head": ["actuator.add_head."],
        "move_head": ["actuator.move_voxel_head."],
        "operation_gate_head": ["actuator.operation_gate_head."],
        "drop_amount_head": ["actuator.drop_amount_head."],
        "add_amount_head": ["actuator.add_amount_head."],
        "move_amount_head": ["actuator.move_amount_head."],
        "policy": ["policy_module."],
        "cost_attr": ["cost_attributor."],
        "cause_agg": ["cause_aggregator."],
    }

    out = {}
    eps = float(zero_eps)

    for label, patterns in targets.items():
        matched = 0
        none_count = 0
        nan_count = 0
        grad_norm_sum = 0.0
        grad_max = 0.0

        for name, param in base_model.named_parameters():
            name_l = str(name).lower()
            if not any(pattern.lower() in name_l for pattern in patterns):
                continue

            matched += 1
            if param.grad is None:
                none_count += 1
                continue

            grad = param.grad.detach()
            if grad.numel() == 0:
                none_count += 1
                continue

            grad_f = grad.float().reshape(-1)
            finite_mask = torch.isfinite(grad_f)
            if not bool(finite_mask.all().item()):
                nan_count += int((~finite_mask).sum().detach().cpu().item())

            grad_clean = torch.nan_to_num(grad_f, nan=0.0, posinf=0.0, neginf=0.0)
            norm_value = float(torch.linalg.norm(grad_clean, ord=2).detach().cpu())
            max_value = float(grad_clean.abs().max().detach().cpu()) if grad_clean.numel() > 0 else 0.0

            grad_norm_sum += norm_value
            grad_max = max(grad_max, max_value)

        out[label] = {
            "matched_param_count": int(matched),
            "grad_norm": float(grad_norm_sum),
            "grad_is_none": bool(matched > 0 and none_count == matched),
            "grad_is_nan": bool(nan_count > 0),
            "grad_is_zero_like": bool(grad_norm_sum <= eps),
            "none_grad_param_count": int(none_count),
            "nan_grad_element_count": int(nan_count),
            "grad_abs_max": float(grad_max),
        }

    return out


def _phase7_log_grad_sanity(args, writer, model, comp_debug, global_step):
    if not bool(getattr(args, "phase7_grad_sanity_check", True)):
        return {}
    if not _phase7_should_log_interval(args, global_step, "phase7_grad_sanity_every", 10):
        return {}

    stats = _phase7_grad_sanity_stats(
        model,
        zero_eps=float(getattr(args, "phase7_grad_zero_eps", 1e-12)),
    )

    if isinstance(comp_debug, dict):
        key_map = {
            "drop_head": "phase7_grad_drop_head",
            "add_head": "phase7_grad_add_head",
            "move_head": "phase7_grad_move_head",
            "operation_gate_head": "phase7_grad_operation_gate_head",
            "policy": "phase7_grad_policy",
            "cost_attr": "phase7_grad_cost_attr",
        }
        for label, out_key in key_map.items():
            comp_debug[out_key] = float(stats.get(label, {}).get("grad_norm", 0.0))

        for label, values in stats.items():
            prefix = f"phase7_grad_sanity_{label}"
            comp_debug[f"{prefix}_norm"] = float(values.get("grad_norm", 0.0))
            comp_debug[f"{prefix}_is_none"] = bool(values.get("grad_is_none", False))
            comp_debug[f"{prefix}_is_nan"] = bool(values.get("grad_is_nan", False))
            comp_debug[f"{prefix}_is_zero_like"] = bool(values.get("grad_is_zero_like", False))

    parts = []
    for label in (
        "drop_head",
        "add_head",
        "move_head",
        "operation_gate_head",
        "drop_amount_head",
        "add_amount_head",
        "move_amount_head",
        "policy",
        "cost_attr",
        "cause_agg",
    ):
        values = stats.get(label, {})
        parts.append(
            f"{label}:norm={float(values.get('grad_norm', 0.0)):.6g},"
            f"none={bool(values.get('grad_is_none', False))},"
            f"nan={bool(values.get('grad_is_nan', False))},"
            f"zero={bool(values.get('grad_is_zero_like', False))}"
        )


    return stats


def _phase7_param_update_enabled(args, global_step):
    if not bool(getattr(args, "phase7_param_update_check", False)):
        return False
    return _phase7_should_log_interval(args, global_step, "phase7_param_update_every", 20)


def _phase7_take_param_snapshot(model):
    """
    Phase7-4:
    optimizer.step前の主要moduleパラメータをdetach cloneする。
    default Falseのdebug専用なので、重さは許容する。
    """
    base_model = model.module if hasattr(model, "module") else model
    targets = {
        "actuator": ["actuator."],
        "policy": ["policy_module."],
        "cost_attr": ["cost_attributor."],
        "cause_agg": ["cause_aggregator."],
    }

    snapshot = {key: [] for key in targets.keys()}

    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        name_l = str(name).lower()
        for label, patterns in targets.items():
            if any(pattern.lower() in name_l for pattern in patterns):
                snapshot[label].append((name, param.detach().clone()))
                break

    return snapshot


def _phase7_compare_param_snapshot(model, snapshot, zero_eps=1e-12):
    """
    Phase7-4:
    optimizer.step後に、snapshotとの差分を集計する。
    graphを保持しない。
    """
    base_model = model.module if hasattr(model, "module") else model
    current_params = {
        name: param.detach()
        for name, param in base_model.named_parameters()
        if param.requires_grad
    }

    out = {}
    eps = float(zero_eps)

    for label, items in (snapshot or {}).items():
        update_norm_sum = 0.0
        update_max = 0.0
        compared_count = 0

        for name, before in items:
            after = current_params.get(name, None)
            if after is None:
                continue
            diff = (after - before.to(device=after.device, dtype=after.dtype)).detach().float().reshape(-1)
            if diff.numel() == 0:
                continue
            diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            update_norm_sum += float(torch.linalg.norm(diff, ord=2).detach().cpu())
            update_max = max(update_max, float(diff.abs().max().detach().cpu()))
            compared_count += 1

        out[label] = {
            "param_update_norm": float(update_norm_sum),
            "param_update_max": float(update_max),
            "param_updated": bool(update_norm_sum > eps or update_max > eps),
            "compared_param_count": int(compared_count),
        }

    return out


def _phase7_log_param_update(args, writer, comp_debug, update_stats, global_step):
    if not update_stats:
        return

    if isinstance(comp_debug, dict):
        comp_debug["phase7_update_actuator"] = float(update_stats.get("actuator", {}).get("param_update_norm", 0.0))
        comp_debug["phase7_update_policy"] = float(update_stats.get("policy", {}).get("param_update_norm", 0.0))
        comp_debug["phase7_update_cost_attr"] = float(update_stats.get("cost_attr", {}).get("param_update_norm", 0.0))
        comp_debug["phase7_update_cause_agg"] = float(update_stats.get("cause_agg", {}).get("param_update_norm", 0.0))

        for label, values in update_stats.items():
            prefix = f"phase7_param_update_{label}"
            comp_debug[f"{prefix}_norm"] = float(values.get("param_update_norm", 0.0))
            comp_debug[f"{prefix}_max"] = float(values.get("param_update_max", 0.0))
            comp_debug[f"{prefix}_updated"] = bool(values.get("param_updated", False))

    _phase7_writer_line(
        args,
        writer,
        "Phase7ParamUpdate: "
        f"actuator_norm={float(update_stats.get('actuator', {}).get('param_update_norm', 0.0)):.6g}, "
        f"actuator_updated={bool(update_stats.get('actuator', {}).get('param_updated', False))}, "
        f"policy_norm={float(update_stats.get('policy', {}).get('param_update_norm', 0.0)):.6g}, "
        f"policy_updated={bool(update_stats.get('policy', {}).get('param_updated', False))}, "
        f"cost_attr_norm={float(update_stats.get('cost_attr', {}).get('param_update_norm', 0.0)):.6g}, "
        f"cost_attr_updated={bool(update_stats.get('cost_attr', {}).get('param_updated', False))}, "
        f"cause_agg_norm={float(update_stats.get('cause_agg', {}).get('param_update_norm', 0.0)):.6g}, "
        f"cause_agg_updated={bool(update_stats.get('cause_agg', {}).get('param_updated', False))}"
    )


def _phase7_add_ablation_summary_to_comp_debug(args, comp_debug):
    if not isinstance(comp_debug, dict):
        return

    mode = str(getattr(args, "_phase7_ablation_effective_mode", getattr(args, "phase7_ablation_mode", "none")))
    comp_debug["phase7_ablation_mode"] = mode
    comp_debug["phase7_voxel_actual_enabled"] = bool(getattr(args, "use_voxel_restored_points_for_actual", False))
    comp_debug["phase7_full_context_soft_enabled"] = bool(getattr(args, "full_context_subtree_soft_proxy", True))
    comp_debug["phase7_correction_loss_enabled"] = bool(getattr(args, "full_cloud_actual_correction_loss_enable", False))

    comp_debug["phase7_actual_input_points"] = int(comp_debug.get("original_gen_points", 0) or 0)
    comp_debug["phase7_restored_actual_points"] = int(comp_debug.get("restored_actual_points", 0) or 0)

    comp_debug["phase7_full_context_soft_proxy_loss"] = float(
        comp_debug.get(
            "full_context_soft_proxy_loss",
            comp_debug.get("full_context_subtree_soft_proxy_loss_value", 0.0),
        )
        or 0.0
    )
    comp_debug["phase7_correction_loss"] = float(
        comp_debug.get(
            "full_cloud_actual_correction_loss_value",
            comp_debug.get("full_cloud_corr_loss_value", 0.0),
        )
        or 0.0
    )
    comp_debug["phase7_full_cloud_actual_delta"] = float(
        comp_debug.get(
            "full_cloud_actual_delta",
            comp_debug.get("full_cloud_actual_percent", comp_debug.get("full_cloud_corr_last_full_actual_delta", 0.0)),
        )
        or 0.0
    )
    comp_debug["phase7_subtree_actual_delta"] = float(
        comp_debug.get(
            "subtree_actual_delta",
            comp_debug.get("subtree_teacher_percent", comp_debug.get("full_cloud_corr_last_subtree_actual_delta", 0.0)),
        )
        or 0.0
    )
    comp_debug["phase7_full_vs_subtree_gap"] = float(
        comp_debug.get(
            "full_vs_subtree_gap",
            comp_debug.get("full_cloud_corr_ema_full_vs_subtree_gap", 0.0),
        )
        or 0.0
    )

def _phase7_normalize_actual_debug(args, comp_debug):
    """
    Phase7-5:
    actual SparsePCGC / actual codec結果のkeyをPhase7評価summary用に正規化する。
    worker内部は変更せず、train.py側で既存keyを吸収する。
    """
    if not isinstance(comp_debug, dict):
        return {}

    scope = str(
        comp_debug.get(
            "actual_scope",
            getattr(args, "_current_teacher_scope", "")
        )
    )
    if not scope:
        scope = "unknown"

    input_source = str(
        comp_debug.get(
            "actual_input_source",
            "voxel_restored" if bool(comp_debug.get("voxel_restored_actual_used", False)) else "gen_xyz"
        )
    )

    total_bits = comp_debug.get(
        "actual_total_bits",
        comp_debug.get("gen_actual_bit", comp_debug.get("actual_sparsepcgc_bit", 0.0)),
    )

    actual_bpp = comp_debug.get(
        "actual_bpp",
        comp_debug.get("bpp", 0.0),
    )

    actual_delta = comp_debug.get(
        "actual_delta_percent",
        comp_debug.get("actual_total_bit_percent", comp_debug.get("total_bit", 0.0)),
    )

    lowprob_count = comp_debug.get(
        "actual_lowprob_count",
        comp_debug.get(
            "actual_lowprob_occupancy_count_after",
            comp_debug.get("low_prob_true_count", 0.0),
        ),
    )

    normalized = {
        "actual_scope": scope,
        "actual_input_source": input_source,
        "actual_used_voxel_restored_points": bool(comp_debug.get("voxel_restored_actual_used", False)),
        "actual_input_points": int(
            comp_debug.get(
                "actual_input_points",
                comp_debug.get("phase7_actual_input_points", comp_debug.get("gen_points", 0)),
            )
            or 0
        ),
        "actual_total_bits": _phase7_float(total_bits, 0.0),
        "actual_bpp": _phase7_float(actual_bpp, 0.0),
        "actual_delta_percent": _phase7_float(actual_delta, 0.0),
        "actual_occupancy_nll": _phase7_float(
            comp_debug.get(
                "actual_occupancy_nll",
                comp_debug.get("actual_occupancy_nll_after", comp_debug.get("sparsepcgc_exact_occupancy_nll", 0.0)),
            ),
            0.0,
        ),
        "actual_occupancy_nll_delta": _phase7_float(
            comp_debug.get(
                "actual_occupancy_nll_delta",
                comp_debug.get("sparsepcgc_exact_occupancy_nll_delta", 0.0),
            ),
            0.0,
        ),
        "actual_node_count": _phase7_float(
            comp_debug.get("actual_node_count", comp_debug.get("rate_proxy_after", comp_debug.get("gen_node", 0.0))),
            0.0,
        ),
        "actual_single_child_count": _phase7_float(
            comp_debug.get("actual_single_child_count", comp_debug.get("single_delta", 0.0)),
            0.0,
        ),
        "actual_lowprob_count": _phase7_float(lowprob_count, 0.0),
    }

    comp_debug.update(normalized)
    return normalized


def _phase7_eval_summary_path(args, plot):
    """
    Phase7-5:
    summary CSVの保存先を決める。
    既存metric CSVと同じrun配下へ置く。
    """
    name = str(getattr(args, "phase7_eval_summary_name", "phase7_eval_summary.csv")).strip()
    if not name:
        name = "phase7_eval_summary.csv"

    base_dir = getattr(plot, "log_dir", None)
    if base_dir is None:
        base_dir = getattr(args, "out_path", ".")

    return os.path.join(str(base_dir), name)


def _phase7_build_eval_summary_row(
    args,
    *,
    global_step,
    episode,
    epoch,
    step,
    stage,
    comp_debug,
    L_geom,
    L_com,
):
    """
    Phase7-5:
    compression_metric_rowより小さい、比較専用summary行を作る。
    """
    comp_debug = comp_debug if isinstance(comp_debug, dict) else {}
    _phase7_normalize_actual_debug(args, comp_debug)

    return {
        "global_step": int(global_step),
        "episode": int(episode),
        "epoch": int(epoch),
        "step": int(step),
        "stage": str(stage),

        "phase7_ablation_mode": str(
            comp_debug.get(
                "phase7_ablation_mode",
                getattr(args, "_phase7_ablation_effective_mode", getattr(args, "phase7_ablation_mode", "none")),
            )
        ),
        "voxel_restored_actual_used": bool(comp_debug.get("voxel_restored_actual_used", False)),
        "network_voxel_node_input_used": bool(comp_debug.get("network_voxel_node_input_used", False)),
        "network_voxel_node_fallback_reason": str(comp_debug.get("network_voxel_node_fallback_reason", "")),
        # ============================================================
        # Phase5:
        # Node/Voxel canonical経路の安全性summary
        # ============================================================
        "phase5_structure_safety_ok": bool(
            comp_debug.get("phase5_structure_safety_ok", False)
        ),
        "phase5_structure_safety_reason": str(
            comp_debug.get("phase5_structure_safety_reason", "")
        ),
        "phase5_cost_attribution_input_mode": str(
            comp_debug.get("phase5_cost_attribution_input_mode", "")
        ),
        "phase5_aggregation_key_source": str(
            comp_debug.get("phase5_aggregation_key_source", "")
        ),
        "phase5_structural_key_source": str(
            comp_debug.get("phase5_structural_key_source", "")
        ),
        "phase5_aggregation_unit_count": int(
            comp_debug.get("phase5_aggregation_unit_count", 0) or 0
        ),
        "phase5_aggregation_min_unit_size": int(
            comp_debug.get("phase5_aggregation_min_unit_size", 0) or 0
        ),
        "phase5_aggregation_max_unit_size": int(
            comp_debug.get("phase5_aggregation_max_unit_size", 0) or 0
        ),
        "phase5_local_recomputed": bool(
            comp_debug.get("phase5_local_recomputed", False)
        ),
        "phase5_unit_collapse_warn": bool(
            comp_debug.get("phase5_unit_collapse_warn", False)
        ),
        "L_geom": _phase7_float(L_geom, 0.0),
        "L_com": _phase7_float(L_com, 0.0),
        "full_context_subtree_hard_loss": _phase7_float(
            comp_debug.get("full_context_subtree_hard_loss", comp_debug.get("full_context_hard_loss", 0.0)),
            0.0,
        ),
        "full_context_subtree_soft_proxy_loss": _phase7_float(
            comp_debug.get("full_context_subtree_soft_proxy_loss", comp_debug.get("full_context_soft_proxy_loss", 0.0)),
            0.0,
        ),
        "full_cloud_actual_correction_loss": _phase7_float(
            comp_debug.get("full_cloud_actual_correction_loss", comp_debug.get("full_cloud_actual_correction_loss_value", 0.0)),
            0.0,
        ),

        "subtree_local_actual_delta": _phase7_float(
            comp_debug.get("subtree_local_actual_delta", comp_debug.get("phase7_subtree_actual_delta", 0.0)),
            0.0,
        ),
        "full_cloud_actual_delta": _phase7_float(
            comp_debug.get("full_cloud_actual_delta", comp_debug.get("phase7_full_cloud_actual_delta", 0.0)),
            0.0,
        ),
        "full_vs_subtree_gap": _phase7_float(
            comp_debug.get("full_vs_subtree_gap", comp_debug.get("phase7_full_vs_subtree_gap", 0.0)),
            0.0,
        ),
        "full_vs_context_gap": _phase7_float(comp_debug.get("full_vs_context_gap", 0.0), 0.0),

        "drop_ratio_soft": _phase7_float(comp_debug.get("drop_ratio_soft", 0.0), 0.0),
        "add_ratio_soft": _phase7_float(comp_debug.get("add_ratio_soft", 0.0), 0.0),
        "move_ratio_soft": _phase7_float(comp_debug.get("move_ratio_soft", 0.0), 0.0),
        "voxel_edit_drop_count": _phase7_float(comp_debug.get("voxel_edit_drop_count", 0.0), 0.0),
        "voxel_edit_add_count": _phase7_float(comp_debug.get("voxel_edit_add_count", 0.0), 0.0),
        "voxel_edit_move_count": _phase7_float(comp_debug.get("voxel_edit_move_count", 0.0), 0.0),

        "drop_grad_norm": _phase7_float(comp_debug.get("drop_grad_norm", comp_debug.get("phase7_grad_drop_head", 0.0)), 0.0),
        "add_grad_norm": _phase7_float(comp_debug.get("add_grad_norm", comp_debug.get("phase7_grad_add_head", 0.0)), 0.0),
        "move_grad_norm": _phase7_float(comp_debug.get("move_grad_norm", comp_debug.get("phase7_grad_move_head", 0.0)), 0.0),
        "operation_gate_grad_norm": _phase7_float(comp_debug.get("operation_gate_grad_norm", comp_debug.get("phase7_grad_operation_gate_head", 0.0)), 0.0),
        "policy_grad_norm": _phase7_float(comp_debug.get("policy_grad_norm", comp_debug.get("phase7_grad_policy", 0.0)), 0.0),
        "cost_attr_grad_norm": _phase7_float(comp_debug.get("cost_attr_grad_norm", comp_debug.get("phase7_grad_cost_attr", 0.0)), 0.0),

        "actual_total_bits": _phase7_float(comp_debug.get("actual_total_bits", 0.0), 0.0),
        "actual_bpp": _phase7_float(comp_debug.get("actual_bpp", 0.0), 0.0),
        "actual_occupancy_nll_delta": _phase7_float(comp_debug.get("actual_occupancy_nll_delta", 0.0), 0.0),

        "actual_scope": str(comp_debug.get("actual_scope", "")),
        "actual_input_source": str(comp_debug.get("actual_input_source", "")),
        "actual_used_voxel_restored_points": bool(comp_debug.get("actual_used_voxel_restored_points", False)),
        "actual_input_points": int(comp_debug.get("actual_input_points", 0) or 0),
        "actual_delta_percent": _phase7_float(comp_debug.get("actual_delta_percent", 0.0), 0.0),
        "actual_node_count": _phase7_float(comp_debug.get("actual_node_count", 0.0), 0.0),
        "actual_single_child_count": _phase7_float(comp_debug.get("actual_single_child_count", 0.0), 0.0),
        "actual_lowprob_count": _phase7_float(comp_debug.get("actual_lowprob_count", 0.0), 0.0),
    }


def _phase7_should_save_eval_summary(args, global_step):
    if not bool(getattr(args, "phase7_eval_summary", True)):
        return False
    interval = max(int(getattr(args, "phase7_eval_summary_every", 1)), 1)
    return int(global_step) % interval == 0

def _summarize_nonfinite_grads(model, limit=8):
    # Loss自体が有限でも、backward中に一部パラメータ勾配だけNaN/Infになることがある。
    base_model = _unwrap_train_model(model)
    bad_names = []
    bad_element_count = 0
    checked_param_count = 0
    checked_element_count = 0
    for name, param in base_model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        checked_param_count += 1
        grad = param.grad.detach()
        checked_element_count += int(grad.numel())
        finite_mask = torch.isfinite(grad)
        if bool(finite_mask.all().item()):
            continue
        bad_count = int((~finite_mask).sum().detach().cpu().item())
        bad_element_count += bad_count
        if len(bad_names) < int(limit):
            bad_names.append(f"{name}:{bad_count}")
    return {
        "has_nonfinite": bad_element_count > 0,
        "bad_element_count": int(bad_element_count),
        "checked_param_count": int(checked_param_count),
        "checked_element_count": int(checked_element_count),
        "bad_names": bad_names,
    }

def _phase7_named_grad_norms(model):
    base_model = model.module if hasattr(model, "module") else model

    targets = {
        "drop_grad_norm": [
            "actuator.drop_head.",
            "actuator.drop_amount_head.",
        ],
        "add_grad_norm": [
            "actuator.add_head.",
            "actuator.add_voxel_head.",
            "actuator.add_amount_head.",
        ],
        "move_grad_norm": [
            "actuator.move_voxel_head.",
            "actuator.move_amount_head.",
        ],
        "operation_gate_grad_norm": [
            "actuator.operation_gate_head.",
        ],
        "policy_grad_norm": [
            "policy_module.",
        ],
        "cost_attr_grad_norm": [
            "cost_attributor.",
        ],
        "cause_agg_grad_norm": [
            "cause_aggregator.",
        ],
    }

    out = {key: 0.0 for key in targets.keys()}

    for name, param in base_model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.numel() == 0:
            continue
        name_l = str(name).lower()
        # Phase7-3: Conv重みなど3次元以上のgradも扱えるように、必ず1次元へ平坦化してからL2 normを取る。
        grad_clean = torch.nan_to_num(
            grad.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).reshape(-1)

        grad_norm = float(torch.linalg.norm(grad_clean, ord=2).cpu())
        for out_key, patterns in targets.items():
            if any(pattern.lower() in name_l for pattern in patterns):
                out[out_key] += grad_norm

    return out


def _den6_online_grad_audit_enabled(args, global_step):
    if not bool(getattr(args, "heuristic_guidance_online_grad_audit", False)):
        return False
    interval = int(getattr(args, "heuristic_guidance_online_grad_audit_interval", 1))
    return interval > 0 and int(global_step) % interval == 0


def _den6_online_grad_norms(model):
    """Return the three learned decision gradients with one device sync.

    This deliberately inspects only the heads used for online Where, Amount,
    and Action decisions.  Unlike the legacy GradFlow debug path it does not
    walk every child module or construct per-parameter text statistics.
    """
    base_model = _unwrap_train_model(model)
    actuator = getattr(base_model, "actuator", None)
    groups = {
        "den6_online_where_grad_norm": (
            getattr(actuator, "drop_head", None),
            getattr(actuator, "add_head", None),
            getattr(actuator, "add_voxel_head", None),
            getattr(actuator, "move_voxel_head", None),
        ),
        "den6_online_amount_grad_norm": (
            getattr(actuator, "drop_amount_head", None),
            getattr(actuator, "add_amount_head", None),
            getattr(actuator, "move_amount_head", None),
        ),
        "den6_online_action_grad_norm": (
            getattr(actuator, "operation_gate_head", None),
            getattr(base_model, "policy_module", None),
        ),
    }
    norm_squares = []
    for modules in groups.values():
        terms = []
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.grad is not None:
                    grad = torch.nan_to_num(parameter.grad.detach().float())
                    terms.append(torch.sum(grad * grad))
        if terms:
            norm_squares.append(torch.stack(terms).sum())
        else:
            norm_squares.append(torch.zeros((), device=next(base_model.parameters()).device))
    values = torch.sqrt(torch.stack(norm_squares)).detach().cpu().tolist()
    return {name: float(value) for name, value in zip(groups.keys(), values)}

def _format_nonfinite_grad_summary(summary):
    if not summary or not summary.get("has_nonfinite", False):
        return "none"
    names = ",".join(summary.get("bad_names", []))
    if not names:
        names = "unlisted"
    return (
        f"bad_elements={int(summary.get('bad_element_count', 0))}, "
        f"checked_params={int(summary.get('checked_param_count', 0))}, "
        f"checked_elements={int(summary.get('checked_element_count', 0))}, "
        f"params={names}"
    )


def _format_soft_proxy_debug(args):
    merged = {}
    for attr_name in ("_soft_proxy_geom_debug", "_soft_proxy_com_debug"):
        value = getattr(args, attr_name, None)
        if isinstance(value, dict):
            merged.update(value)
    if not merged:
        return ""
    parts = []
    for key in (
        "soft_proxy_geom_requires_grad",
        "soft_proxy_com_requires_grad",
        "soft_proxy_prune_geom_requires_grad",
        "soft_proxy_prune_com_requires_grad",
        "drop_prob_requires_grad",
        "keep_prob_requires_grad",
        "drop_logit_mean",
        "drop_logit_min",
        "drop_logit_max",
        "drop_prob_mean",
        "drop_prob_min",
        "drop_prob_max",
        "drop_prob_proxy_mean",
        "drop_prob_proxy_min",
        "drop_prob_proxy_max",
        "keep_prob_mean",
        "keep_prob_min",
        "keep_prob_max",
        "drop_entropy",
        "selected_drop_count_hard",
        "soft_drop_mass",
        "prune_soft_geom_value",
        "prune_soft_rate_value",
        "prune_soft_node_value",
        "prune_soft_single_value",
        "prune_soft_bit_value",
    ):
        if key not in merged:
            continue
        value = merged[key]
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif value is None:
            parts.append(f"{key}=None")
        else:
            try:
                parts.append(f"{key}={float(value):.6g}")
            except Exception:
                parts.append(f"{key}={value}")
    return ", ".join(parts)


def _balance_actual_operation_head_gradients(args, model, structure_debug=None):
    """
    Operation head の optimizer.step 直前の実勾配を調整する。

    目的:
      - actual oracle教師がある場合は従来通り、そのoperationのheadを揃える。
      - minimal_loss_objective=True の場合は、
        Prune Where(drop_head) と Prune Amount(drop_amount_head) を
        teacher有無に関係なく目標normへ揃える。

    注意:
      - loss値やforward値は変えない。
      - backward後、optimizer.step前の param.grad だけを変更する。
      - loss_grad_probe の個別loss勾配ログとは別物である。
    """
    debug = {}

    if not bool(getattr(args, "repair_balance_operation_head_grads", True)):
        return debug

    structure_debug = structure_debug if isinstance(structure_debug, dict) else {}

    # ============================================================
    # 通常のoperation head balance目標値
    # ============================================================
    # 既存のactual oracle用target。
    # 明示指定されていない場合は1.0相当として扱う。
    # ============================================================
    default_target = max(
        float(getattr(args, "repair_operation_head_grad_target", 1.0)),
        0.0,
    )

    # ============================================================
    # Prune Where / Amount 専用の目標norm
    # ============================================================
    # 今回は args.py を追加編集せず、
    # 既存の grad_scale_operation_amount=200.0 を
    # Prune Where / Amount の最終grad targetとしても使う。
    #
    # これにより:
    #   drop_head        -> grad norm 約200
    #   drop_amount_head -> grad norm 約200
    # ============================================================
    online_one_plan = (
        str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
        == "ana_den6_online"
    )
    # The legacy path used a 200x Prune-only target.  In the online hybrid
    # path that makes the learned Amount/Where policy dominate Add/Adjust and
    # defeats the single-plan policy-gradient update.  Keep all decision heads
    # on the same configured target; this changes gradients only after
    # backward and never changes the den6 prior or the hard plan itself.
    prune_target = (
        default_target
        if online_one_plan
        else max(float(getattr(args, "grad_scale_operation_amount", default_target)), 0.0)
    )

    if default_target <= 0.0 and prune_target <= 0.0:
        return debug

    def _positive(*keys):
        for key in keys:
            try:
                if float(structure_debug.get(key, 0.0) or 0.0) > 0.0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    teacher_active = {
        "prune": _positive(
            "actual_oracle_accepted_prune_count",
            "actual_oracle_drop_bad_count",
            "actual_oracle_selected_drop_count",
        ),
        "add": _positive(
            "actual_oracle_accepted_add_count",
            "actual_oracle_add_bad_count",
            "actual_oracle_selected_add_count",
        ),
        "move": _positive(
            "actual_oracle_accepted_subtree_move_count",
            "actual_oracle_move_bad_count",
            "actual_oracle_selected_move_count",
        ),
    }

    # ============================================================
    # minimal_loss_objective中はPruneを常にbalance対象にする
    # ============================================================
    # 理由:
    #   現在の主損失は圧縮損失 + 幾何損失だけである。
    #   actual oracle教師の有無に依存させると、
    #   Prune Where / Amount の勾配norm調整が発火しないstepが出る。
    # ============================================================
    force_prune_balance = bool(getattr(args, "minimal_loss_objective", False))
    if force_prune_balance:
        teacher_active["prune"] = True
    if online_one_plan:
        # den6 online has no multi-plan oracle labels.  All three operation
        # heads are trained from the selected plan's policy gradient, so none
        # may be excluded merely because an oracle-specific counter is zero.
        teacher_active = {"prune": True, "add": True, "move": True}

    base_model = model.module if hasattr(model, "module") else model
    actuator = getattr(base_model, "actuator", None)
    if actuator is None:
        return debug

    groups = {
        "prune_where": ("prune", [getattr(actuator, "drop_head", None)]),
        "prune_amount": ("prune", [getattr(actuator, "drop_amount_head", None)]),
        "add_where": ("add", [getattr(actuator, "add_head", None)]),
        "add_direction": ("add", [getattr(actuator, "add_voxel_head", None)]),
        "add_amount": ("add", [getattr(actuator, "add_amount_head", None)]),
        "move_where": ("move", [getattr(actuator, "move_voxel_head", None)]),
        "move_amount": ("move", [getattr(actuator, "move_amount_head", None)]),
    }

    min_scale = max(
        float(getattr(args, "repair_operation_head_grad_min_scale", 1e-4)),
        0.0,
    )
    max_scale = max(
        float(getattr(args, "repair_operation_head_grad_max_scale", 100000.0)),
        min_scale,
    )

    for label, (operation, modules) in groups.items():
        if not teacher_active.get(operation, False):
            continue

        # ========================================================
        # Prune Where / Amountだけは200程度へ揃える
        # ========================================================
        if label in {"prune_where", "prune_amount"}:
            target = prune_target
        else:
            target = default_target

        if target <= 0.0:
            debug[f"{label}_grad_balance_status"] = "target_disabled"
            continue

        params = []
        seen = set()

        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                if param.grad is None or id(param) in seen:
                    continue
                seen.add(id(param))
                params.append(param)

        if not params:
            debug[f"{label}_grad_balance_status"] = "no_grad"
            continue

        norm_sq = 0.0
        for param in params:
            grad = torch.nan_to_num(
                param.grad.detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            norm_sq += float(torch.sum(grad * grad).cpu())

        norm_before = math.sqrt(max(norm_sq, 0.0))

        if not math.isfinite(norm_before) or norm_before <= 1e-12:
            debug[f"{label}_grad_balance_status"] = "zero_or_nonfinite"
            debug[f"{label}_grad_norm_before_balance"] = float(norm_before)
            continue

        scale = float(target) / float(norm_before)
        scale = min(max(scale, min_scale), max_scale)

        for param in params:
            if param.grad is not None:
                param.grad.mul_(float(scale))

        debug[f"{label}_grad_norm_before_balance"] = float(norm_before)
        debug[f"{label}_grad_balance_target"] = float(target)
        debug[f"{label}_grad_balance_scale"] = float(scale)
        debug[f"{label}_grad_norm_after_balance"] = float(norm_before * scale)
        debug[f"{label}_grad_balance_status"] = "scaled"

    if online_one_plan and default_target > 0.0:
        # The per-head pass above keeps legacy head groups healthy.  The
        # online policy, however, learns three *decisions* (Where, Amount,
        # Action).  Normalize their combined norms once more so one decision
        # cannot dominate solely because it owns more small MLP heads.
        decision_groups = {
            "where": [
                getattr(actuator, "drop_head", None),
                getattr(actuator, "add_head", None),
                getattr(actuator, "add_voxel_head", None),
                getattr(actuator, "move_voxel_head", None),
            ],
            "amount": [
                getattr(actuator, "drop_amount_head", None),
                getattr(actuator, "add_amount_head", None),
                getattr(actuator, "move_amount_head", None),
            ],
            "action": [
                getattr(actuator, "operation_gate_head", None),
                getattr(base_model, "policy_module", None),
            ],
        }
        for decision, modules in decision_groups.items():
            params = [
                param
                for module in modules
                if module is not None
                for param in module.parameters()
                if param.grad is not None
            ]
            if not params:
                debug[f"den6_online_{decision}_grad_balance_status"] = "no_grad"
                continue
            norm_sq = sum(
                torch.sum(torch.nan_to_num(param.grad.detach().float()) ** 2)
                for param in params
            )
            norm_before = float(torch.sqrt(norm_sq).detach().cpu())
            if not math.isfinite(norm_before) or norm_before <= 1e-12:
                debug[f"den6_online_{decision}_grad_balance_status"] = "zero_or_nonfinite"
                continue
            scale = min(max(default_target / norm_before, min_scale), max_scale)
            for param in params:
                param.grad.mul_(float(scale))
            debug[f"den6_online_{decision}_grad_norm_before_balance"] = norm_before
            debug[f"den6_online_{decision}_grad_norm_after_balance"] = norm_before * scale
            debug[f"den6_online_{decision}_grad_balance_status"] = "scaled"

    return debug

def _register_prune_where_head_grad_scale_hook(args, model, writer=None):
    """
    Prune Where(drop_head)へ入る勾配全体を倍率調整する。

    目的:
      - loss_grad_probe上でも、通常backward上でも、
        prune_where_drop_head の勾配を直接 1/6 程度へ下げる。
      - forward値、loss値、hard選択結果は変えない。
      - drop_amount_head には触らないため、Amount=200は維持される。

    注意:
      - optimizer.step直前のgrad balanceとは別である。
      - このhookはautogradからdrop_head parameterへ流れる勾配そのものに効く。
    """
    scale = max(
        float(getattr(args, "grad_scale_prune_where_head", 1.0)),
        0.0,
    )

    # 1.0なら何もしない。
    if abs(scale - 1.0) < 1e-12:
        if writer is not None and hasattr(writer, "write"):
            writer.write(
                "PruneWhereGradHook: disabled because grad_scale_prune_where_head=1.0"
            )
        return

    base_model = model.module if hasattr(model, "module") else model
    actuator = getattr(base_model, "actuator", None)
    if actuator is None:
        if writer is not None and hasattr(writer, "write"):
            writer.write("PruneWhereGradHook: skipped because actuator is missing")
        return

    drop_head = getattr(actuator, "drop_head", None)
    if drop_head is None:
        if writer is not None and hasattr(writer, "write"):
            writer.write("PruneWhereGradHook: skipped because actuator.drop_head is missing")
        return

    # 二重登録を防ぐ。
    if bool(getattr(base_model, "_prune_where_head_grad_scale_hook_registered", False)):
        if writer is not None and hasattr(writer, "write"):
            writer.write("PruneWhereGradHook: already registered")
        return

    handles = []

    for name, param in drop_head.named_parameters():
        if not param.requires_grad:
            continue

        # gradそのものをscale倍する。
        # forward値やloss値は変わらない。
        handle = param.register_hook(
            lambda grad, s=scale: grad * s if grad is not None else grad
        )
        handles.append(handle)

    setattr(base_model, "_prune_where_head_grad_scale_hook_registered", True)
    setattr(base_model, "_prune_where_head_grad_scale_hook_handles", handles)

    if writer is not None and hasattr(writer, "write"):
        writer.write(
            f"PruneWhereGradHook: registered, "
            f"scale={float(scale):.6g}, "
            f"param_count={len(handles)}"
        )
        
def _discrete_loss_mode_value(args):
    # parse_pugan_args が正規化する正式名を使う。旧 typo 名が残る実験設定だけ後方互換で読む。
    return str(
        getattr(args, "discrete_loss_mode", getattr(args, "discretelossmode", "hard"))
    ).strip().lower()


def _step_grad_group_specs():
    # 名前に基づいて、モジュール別・点操作別・head別の勾配集計対象を定義する。
    # actuator_all と op_* と head別グループは重複してよい。
    # 目的は「操作全体」「どこに」「どのくらい」を分けて確認することである。
    return [
        ("all_trainable", []),

        # ============================================================
        # モジュール単位
        # ============================================================
        ("encoder", ["encoder."]),
        ("structure_analyzer", ["structure_analyzer."]),
        ("cost_attributor", ["cost_attributor."]),
        ("cause_aggregator", ["cause_aggregator."]),
        ("policy_module", ["policy_module."]),
        ("actuator_all", ["actuator."]),
        ("full_cloud_amount_selector_total", ["full_cloud_amount_selector."]),

        # ============================================================
        # 操作単位：従来ログとの互換性を残す
        # ============================================================
        ("op_add", [
            "actuator.add_head.",
            "actuator.add_voxel_head.",
            "actuator.add_amount_head.",
        ]),
        ("op_prune_delete_drop", [
            "actuator.drop_head.",
            "actuator.drop_amount_head.",
        ]),
        ("op_adjust_move", [
            "actuator.move_voxel_head.",
            "actuator.subtree_move_source_head.",
            "actuator.move_amount_head.",
        ]),
        ("operation_gate_head", [
            "actuator.operation_gate_head.",
        ]),

        # ============================================================
        # 削除 Prune/Delete
        # ============================================================
        # どこを削除するか：削除位置scoreを出すhead
        ("prune_where_drop_head", [
            "actuator.drop_head.",
        ]),

        # どのくらい削除するか：削除割合を出すhead
        ("prune_amount_head", [
            "actuator.drop_amount_head.",
        ]),
        ("full_cloud_amount_bin_logits_weight", [
            "full_cloud_amount_selector.amount_bin_logits.weight",
        ]),
        ("full_cloud_amount_bin_logits_bias", [
            "full_cloud_amount_selector.amount_bin_logits.bias",
        ]),
        ("full_cloud_amount_residual_raw_weight", [
            "full_cloud_amount_selector.amount_residual_raw.weight",
        ]),
        ("full_cloud_amount_residual_raw_bias", [
            "full_cloud_amount_selector.amount_residual_raw.bias",
        ]),
        ("full_cloud_amount_predicted_delta_weight", [
            "full_cloud_amount_selector.predicted_delta.weight",
        ]),
        ("full_cloud_amount_predicted_delta_bias", [
            "full_cloud_amount_selector.predicted_delta.bias",
        ]),
        ("full_cloud_amount_predicted_delta_per_amount_weight", [
            "full_cloud_amount_selector.predicted_delta_per_amount.weight",
        ]),
        ("full_cloud_amount_predicted_delta_per_amount_bias", [
            "full_cloud_amount_selector.predicted_delta_per_amount.bias",
        ]),

        # ============================================================
        # 追加 Add
        # ============================================================
        # どの点を追加元候補にするか：add scoreを出すhead
        ("add_where_score_head", [
            "actuator.add_head.",
        ]),

        # どの近傍Voxelへ追加するか：追加方向を出すhead
        ("add_where_direction_head", [
            "actuator.add_voxel_head.",
        ]),

        # どのくらい追加するか：追加割合を出すhead
        ("add_amount_head", [
            "actuator.add_amount_head.",
        ]),

        # ============================================================
        # 調整 Adjust/Move
        # ============================================================
        # どの方向へ動かすか：26近傍方向logitを出すhead
        ("move_where_direction_head", [
            "actuator.move_voxel_head.",
        ]),

        ("move_where_source_head", [
            "actuator.subtree_move_source_head.",
        ]),

        # どのくらい動かすか：移動割合を出すhead
        ("move_amount_head", [
            "actuator.move_amount_head.",
        ]),

        # ============================================================
        # source位置は専用headに加え、policy/cost attributionにも依存する。
        # ============================================================
        ("move_source_policy_related", [
            "policy_module.",
            "cost_attributor.",
        ]),
    ]


def _match_param_names(named_params, keywords):
    # keywordsが空なら全学習可能パラメータを返す
    if not keywords:
        return [(name, param) for name, param in named_params]

    lowered_keywords = [str(key).lower() for key in keywords]
    matched = []
    for name, param in named_params:
        name_l = str(name).lower()
        if any(key in name_l for key in lowered_keywords):
            matched.append((name, param))
    return matched


def _grad_stats_from_named_grads(group_named_params, grad_by_name):
    grads = []
    none_count = 0
    elem_count = 0

    for name, _param in group_named_params:
        grad = grad_by_name.get(name, None)
        if grad is None:
            none_count += 1
            continue
        if not torch.is_tensor(grad):
            none_count += 1
            continue
        grad_det = grad.detach().float()
        if grad_det.numel() == 0:
            none_count += 1
            continue
        grads.append(grad_det.reshape(-1))
        elem_count += int(grad_det.numel())

    if not grads:
        return {
            "used_param_count": 0,
            "none_grad_param_count": int(none_count),
            "grad_element_count": 0,
            "grad_l2": 0.0,
            "grad_abs_mean": 0.0,
            "grad_abs_max": 0.0,
            "grad_signed_mean": 0.0,
        }

    flat = torch.cat(grads, dim=0)
    return {
        "used_param_count": int(len(grads)),
        "none_grad_param_count": int(none_count),
        "grad_element_count": int(elem_count),
        "grad_l2": float(torch.linalg.norm(flat, ord=2).detach().cpu()),
        "grad_abs_mean": float(flat.abs().mean().detach().cpu()),
        "grad_abs_max": float(flat.abs().max().detach().cpu()),
        "grad_signed_mean": float(flat.mean().detach().cpu()),
    }


def build_step_grad_rows(
    args,
    model,
    loss_items,
    *,
    global_step,
    episode,
    epoch,
    step,
    stage,
):
    """
    各損失項が各モジュール・点操作系パラメータへ流す勾配量をCSV行として作る。
    torch.autograd.gradを使うため、通常の .grad は汚さない。
    """
    enabled = bool(getattr(args, "step_grad_log", True))
    if not enabled:
        return []

    if bool(getattr(args, "step_grad_first_step_only", True)) and int(global_step) != 0:
        return []

    interval = max(int(getattr(args, "step_grad_log_interval", 1)), 1)
    if int(global_step) != 0 and (int(global_step) + 1) % interval != 0:
        return []

    base_model = _unwrap_train_model(model)
    named_params = [
        (name, param)
        for name, param in base_model.named_parameters()
        if param.requires_grad
    ]

    if not named_params:
        return []

    all_param_names = [name for name, _ in named_params]
    all_params = [param for _, param in named_params]
    group_specs = _step_grad_group_specs()

    rows = []

    for loss_name, loss_value in loss_items:
        if loss_value is None:
            continue
        if not torch.is_tensor(loss_value):
            continue
        if not loss_value.requires_grad:
            # detach済み・実Codec値・ログ専用値などはここに入る
            rows.append({
                "global_step": int(global_step),
                "episode": int(episode),
                "epoch": int(epoch),
                "step": int(step),
                "stage": str(stage),
                "loss_name": str(loss_name),
                "loss_value": _safe_scalar_for_grad_log(loss_value),
                "target_group": "no_grad_graph",
                "matched_param_count": 0,
                "used_param_count": 0,
                "none_grad_param_count": 0,
                "grad_element_count": 0,
                "grad_l2": 0.0,
                "grad_abs_mean": 0.0,
                "grad_abs_max": 0.0,
                "grad_signed_mean": 0.0,
                "param_name_sample": "",
            })
            continue

        if not torch.isfinite(loss_value.detach()).all().item():
            rows.append({
                "global_step": int(global_step),
                "episode": int(episode),
                "epoch": int(epoch),
                "step": int(step),
                "stage": str(stage),
                "loss_name": str(loss_name),
                "loss_value": _safe_scalar_for_grad_log(loss_value),
                "target_group": "non_finite_loss",
                "matched_param_count": 0,
                "used_param_count": 0,
                "none_grad_param_count": 0,
                "grad_element_count": 0,
                "grad_l2": 0.0,
                "grad_abs_mean": 0.0,
                "grad_abs_max": 0.0,
                "grad_signed_mean": 0.0,
                "param_name_sample": "",
            })
            continue

        try:
            grads = torch.autograd.grad(
                loss_value,
                all_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        except RuntimeError as exc:
            rows.append({
                "global_step": int(global_step),
                "episode": int(episode),
                "epoch": int(epoch),
                "step": int(step),
                "stage": str(stage),
                "loss_name": str(loss_name),
                "loss_value": _safe_scalar_for_grad_log(loss_value),
                "target_group": "autograd_error",
                "matched_param_count": 0,
                "used_param_count": 0,
                "none_grad_param_count": 0,
                "grad_element_count": 0,
                "grad_l2": 0.0,
                "grad_abs_mean": 0.0,
                "grad_abs_max": 0.0,
                "grad_signed_mean": 0.0,
                "param_name_sample": f"{type(exc).__name__}: {str(exc)[:160]}",
            })
            continue

        grad_by_name = {
            name: grad
            for name, grad in zip(all_param_names, grads)
        }

        for group_name, keywords in group_specs:
            group_named_params = _match_param_names(named_params, keywords)
            stats = _grad_stats_from_named_grads(group_named_params, grad_by_name)
            sample_names = [name for name, _ in group_named_params[:5]]

            rows.append({
                "global_step": int(global_step),
                "episode": int(episode),
                "epoch": int(epoch),
                "step": int(step),
                "stage": str(stage),
                "loss_name": str(loss_name),
                "loss_value": _safe_scalar_for_grad_log(loss_value),
                "target_group": str(group_name),
                "matched_param_count": int(len(group_named_params)),
                "used_param_count": int(stats["used_param_count"]),
                "none_grad_param_count": int(stats["none_grad_param_count"]),
                "grad_element_count": int(stats["grad_element_count"]),
                "grad_l2": float(stats["grad_l2"]),
                "grad_abs_mean": float(stats["grad_abs_mean"]),
                "grad_abs_max": float(stats["grad_abs_max"]),
                "grad_signed_mean": float(stats["grad_signed_mean"]),
                "param_name_sample": "|".join(sample_names),
            })

    return rows


def _voxel_collision_stage_set(args):
    raw = str(getattr(args, "voxel_collision_log_stages", "input_gt,model_output_raw,compression_input"))
    return {item.strip() for item in raw.split(",") if item.strip()}


def _should_log_voxel_collision(args, global_step):
    if not bool(getattr(args, "enable_voxel_collision_log", False)):
        return False
    interval = max(int(getattr(args, "voxel_collision_log_interval", 100)), 1)
    return ((int(global_step) + 1) % interval) == 0


def _collect_train_voxel_collision_stats(args, writer, global_step, stage_tensors):
    if not _should_log_voxel_collision(args, global_step):
        return {}
    stages = _voxel_collision_stage_set(args)
    voxel_size = float(getattr(args, "sparsepcgc_voxel_size", getattr(args, "octree_voxel", 1.0)))
    pos_q = int(getattr(args, "sparsepcgc_pos_quantscale", 1))
    max_points = int(getattr(args, "voxel_collision_max_points", 300000))
    first_only = bool(getattr(args, "voxel_collision_log_first_batch_only", True))
    flat = {}
    for stage in sorted(stages):
        tensor = stage_tensors.get(stage)
        if tensor is None:
            if hasattr(writer, "write") and not bool(getattr(args, "compact_step_text_log", False)):
                writer.write(f"VoxelCollisionUnavailable[{stage}]: stage tensor is not available in train.py")
            continue
        with torch.no_grad():
            stats = compute_voxel_collision_stats_batch(
                tensor.detach(),
                voxel_size,
                pos_q,
                max_points=max_points,
                first_batch_only=first_only,
            )
        flat.update(flatten_voxel_collision_stats(f"voxel_collision_{stage}", stats))
        if hasattr(writer, "write") and not bool(getattr(args, "compact_step_text_log", False)):
            writer.write(format_voxel_collision_summary(stage, stats))
            note = str(stats.get("sampling_note", ""))
            if note:
                writer.write(f"VoxelCollisionSampling[{stage}]: {note}")
    return flat

def _hard_occupancy_stats_mean_for_train(args, pts_b3n):
    """
    Actual Occupancyと同じ hard_octree_occupancy_stats をバッチ平均で計算する。
    この関数の値はhard統計なので、forward値・ログ値として使う。
    勾配はここからは流さない。
    """
    if pts_b3n is None or not torch.is_tensor(pts_b3n):
        return None
    if pts_b3n.ndim != 3 or pts_b3n.shape[1] != 3:
        return None

    compress_key = (
        str(getattr(args, "compress", ""))
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    if compress_key == "sparsepcgc":
        qs = float(getattr(args, "sparsepcgc_voxel_size", getattr(args, "octree_voxel", 1.0)))
        quant_mode = "sparsepcgc"
        pos_quantscale = int(getattr(args, "sparsepcgc_pos_quantscale", 1))
    else:
        qs = float(getattr(args, "qs", 1.0))
        quant_mode = "round"
        pos_quantscale = 1

    max_depth = int(getattr(args, "sparsepcgc_occupancy_max_depth", 0))

    stat_list = []
    with torch.no_grad():
        pts_det = pts_b3n.detach()
        for b in range(int(pts_det.shape[0])):
            stat_list.append(
                hard_octree_occupancy_stats(
                    pts_det[b, :3, :],
                    qs=qs,
                    max_depth=max_depth,
                    quant_mode=quant_mode,
                    pos_quantscale=pos_quantscale,
                )
            )

    if not stat_list:
        return None

    keys = (
        "occupancy_entropy",
        "occupancy_nll",
        "occupancy_pattern_count",
        "lowprob_occupancy_ratio",
        "occupancy_predictability",
        "node_count",
    )

    out = {}
    for key in keys:
        values = [float(stat.get(key, 0.0)) for stat in stat_list]
        out[key] = sum(values) / float(max(len(values), 1))

    return out


def _hard_occupancy_objective_for_train(args, before_xyz, after_xyz, device, dtype):
    """
    Actualと同じhard Occupancy統計から、学習用forward値を作る。
    ただし、この値自体はdetachされたhard値なので勾配は流れない。
    """
    before_stats = _hard_occupancy_stats_mean_for_train(args, before_xyz)
    after_stats = _hard_occupancy_stats_mean_for_train(args, after_xyz)

    if before_stats is None or after_stats is None:
        return None, {}

    entropy_delta = float(after_stats["occupancy_entropy"] - before_stats["occupancy_entropy"])
    nll_delta = float(after_stats["occupancy_nll"] - before_stats["occupancy_nll"])
    pattern_before = max(float(before_stats["occupancy_pattern_count"]), 1.0)
    pattern_delta_norm = float(after_stats["occupancy_pattern_count"] - before_stats["occupancy_pattern_count"]) / pattern_before
    lowprob_delta = float(after_stats["lowprob_occupancy_ratio"] - before_stats["lowprob_occupancy_ratio"])

    # 現在のoctree_stats.pyでは occupancy_nll は occupancy_entropy と同じ値で返る。
    # そのため、デフォルトではentropyを主成分にし、nllは重複を避けるため小さく扱う。
    w_entropy = float(getattr(args, "exact_occupancy_entropy_loss_weight", 1.0))
    w_nll = float(getattr(args, "exact_occupancy_nll_loss_weight", 0.0))
    w_pattern = float(getattr(args, "exact_occupancy_pattern_loss_weight", 0.25))
    w_lowprob = float(getattr(args, "exact_occupancy_lowprob_loss_weight", 1.0))

    hard_obj_value = (
        w_entropy * entropy_delta
        + w_nll * nll_delta
        + w_pattern * pattern_delta_norm
        + w_lowprob * lowprob_delta
    )

    hard_obj = torch.tensor(
        hard_obj_value,
        device=device,
        dtype=dtype,
    )

    debug = {
        "exact_occ_entropy_before": float(before_stats["occupancy_entropy"]),
        "exact_occ_entropy_after": float(after_stats["occupancy_entropy"]),
        "exact_occ_entropy_delta": float(entropy_delta),
        "exact_occ_nll_before": float(before_stats["occupancy_nll"]),
        "exact_occ_nll_after": float(after_stats["occupancy_nll"]),
        "exact_occ_nll_delta": float(nll_delta),
        "exact_occ_pattern_before": float(before_stats["occupancy_pattern_count"]),
        "exact_occ_pattern_after": float(after_stats["occupancy_pattern_count"]),
        "exact_occ_pattern_delta_norm": float(pattern_delta_norm),
        "exact_occ_lowprob_before": float(before_stats["lowprob_occupancy_ratio"]),
        "exact_occ_lowprob_after": float(after_stats["lowprob_occupancy_ratio"]),
        "exact_occ_lowprob_delta": float(lowprob_delta),
        "exact_occ_hard_objective": float(hard_obj_value),
        "actual_occupancy_predictability_after": float(after_stats["occupancy_predictability"]),
    }

    return hard_obj, debug


def _soft_occupancy_proxy_for_train(args, terms, model, out_label):
    """
    Actual Occupancy hard統計の代わりにbackwardへ使うsoft proxyを作る。
    forward値はhard側を使うため、この値は勾配用である。

    既存のcompression termsとActuator soft termsだけを使い、
    新しい重いpairwise計算は入れない。
    """
    soft_terms = []

    def _append_term(value, weight):
        if torch.is_tensor(value) and value.requires_grad:
            v = value
            if v.numel() != 1:
                v = v.mean()
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            soft_terms.append(float(weight) * v)

    # 圧縮proxy側。termsは loss.last_compression_terms 由来である。
    _append_term(terms.get("bit", None), float(getattr(args, "exact_occ_soft_bit_weight", 1.0)))
    _append_term(terms.get("node", None), float(getattr(args, "exact_occ_soft_node_weight", 1.0)))
    _append_term(terms.get("single", None), float(getattr(args, "exact_occ_soft_single_weight", 0.5)))
    _append_term(terms.get("op", None), float(getattr(args, "exact_occ_soft_op_weight", 0.25)))

    # キーが存在する実装ではOccupancy/lowprob系も使う。
    for key in (
        "lowprob",
        "lowprob_occupancy",
        "occupancy",
        "occupancy_nll",
        "sparsepcgc_aux",
        "sparsepcgc_aux_objective",
    ):
        _append_term(terms.get(key, None), float(getattr(args, "exact_occ_soft_extra_weight", 1.0)))

    # Actuator側のsoft termsも使う。
    actuator_soft_terms = {}

    base_model = model.module if hasattr(model, "module") else model
    model_soft_terms = getattr(base_model, "last_actuator_soft_terms", {})
    if isinstance(model_soft_terms, dict):
        actuator_soft_terms.update(model_soft_terms)

    if isinstance(out_label, dict):
        for key in (
            "drop_prob_proxy",
            "soft_drop_where_grad_base",
            "learned_drop_logit",
            "drop_logit",
            "prune_where_proxy",
            "prune_soft_bit",
            "prune_soft_node",
            "prune_soft_single",
            "prune_soft_rate",
        ):
            value = out_label.get(key, None)
            if torch.is_tensor(value):
                actuator_soft_terms[key] = value

    _append_term(
        actuator_soft_terms.get("prune_soft_bit", None),
        float(getattr(args, "exact_occ_soft_prune_bit_weight", 1.0)),
    )
    _append_term(
        actuator_soft_terms.get("prune_soft_node", None),
        float(getattr(args, "exact_occ_soft_prune_node_weight", 0.75)),
    )
    _append_term(
        actuator_soft_terms.get("prune_soft_single", None),
        float(getattr(args, "exact_occ_soft_prune_single_weight", 0.5)),
    )
    _append_term(
        actuator_soft_terms.get("prune_soft_rate", None),
        float(getattr(args, "exact_occ_soft_prune_rate_weight", 0.25)),
    )

    drop_prob_proxy = actuator_soft_terms.get("drop_prob_proxy", None)
    if torch.is_tensor(drop_prob_proxy) and drop_prob_proxy.requires_grad:
        drop_prob_safe = drop_prob_proxy.clamp(1e-6, 1.0 - 1e-6)
        drop_entropy = -(
            drop_prob_safe * drop_prob_safe.log()
            + (1.0 - drop_prob_safe) * (1.0 - drop_prob_safe).log()
        ).mean()
        _append_term(
            drop_entropy,
            float(getattr(args, "exact_occ_soft_drop_entropy_weight", 0.05)),
        )

    if not soft_terms:
        return None, {"exact_occ_soft_proxy_available": False}

    soft_proxy = soft_terms[0]
    for term in soft_terms[1:]:
        soft_proxy = soft_proxy + term

    soft_proxy = torch.nan_to_num(
        soft_proxy,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return soft_proxy, {
        "exact_occ_soft_proxy_available": True,
        "exact_occ_soft_proxy_term_count": int(len(soft_terms)),
    }


def _build_exact_occupancy_ste_term(args, terms, model, out_label, before_xyz, after_xyz):
    """
    Actual hard Occupancy値をforwardに使い、
    soft proxyをbackwardに使うSTE項を作る。

    返り値:
      ste_term
        forward値はActual hard Occupancy objective
        backwardはsoft proxyへ流れる
      debug
        CSVやログに残す値
    """
    if after_xyz is None or not torch.is_tensor(after_xyz):
        return None, {}

    weight = float(
        getattr(
            args,
            "exact_occupancy_ste_loss_weight",
            getattr(args, "sparsepcgc_exact_teacher_loss_weight", 0.0),
        )
    )
    if weight <= 0.0:
        return None, {"exact_occupancy_ste_used": False, "exact_occupancy_ste_disabled": True}

    hard_obj, hard_debug = _hard_occupancy_objective_for_train(
        args,
        before_xyz=before_xyz,
        after_xyz=after_xyz,
        device=after_xyz.device,
        dtype=after_xyz.dtype,
    )
    if hard_obj is None:
        return None, {"exact_occupancy_ste_used": False, "exact_occupancy_ste_reason": "hard_stats_unavailable"}

    soft_proxy, soft_debug = _soft_occupancy_proxy_for_train(
        args,
        terms=terms,
        model=model,
        out_label=out_label,
    )
    # ============================================================
    # Prune勾配リバランス中は exact occupancy のsoft勾配を止める
    # ============================================================
    # 目的:
    #   hardなActual occupancy値はforwardに残す。
    #   ただしbackwardだけはsoft_proxyへ流さない。
    #   これにより、Prune Whereへ残っている巨大勾配を切り分ける。
    # ============================================================
    if bool(getattr(args, "_prune_grad_rebalance_active", False)):
        if isinstance(soft_debug, dict):
            soft_debug["exact_occ_soft_proxy_suppressed_by_prune_rebalance"] = True
        soft_proxy = None
    debug = {}
    debug.update(hard_debug)
    debug.update(soft_debug)

    if soft_proxy is None or not (torch.is_tensor(soft_proxy) and soft_proxy.requires_grad):
        # soft proxyがない場合は、hard値だけをforwardに足す。
        # ただし勾配は流れない。
        ste_term = weight * hard_obj.detach()
        debug["exact_occupancy_ste_used"] = True
        debug["exact_occupancy_ste_grad_used"] = False
        debug["exact_occupancy_ste_weight"] = float(weight)
        return ste_term, debug

    soft_grad_weight = float(
        getattr(
            args,
            "exact_occupancy_ste_grad_weight",
            getattr(args, "sparsepcgc_exact_teacher_grad_weight", 1.0),
        )
    )

    # forwardはhard_obj、backwardはsoft_proxy。
    ste_term = weight * (
        hard_obj.detach()
        + soft_grad_weight * (soft_proxy - soft_proxy.detach())
    )

    debug["exact_occupancy_ste_used"] = True
    debug["exact_occupancy_ste_grad_used"] = True
    debug["exact_occupancy_ste_weight"] = float(weight)
    debug["exact_occupancy_ste_grad_weight"] = float(soft_grad_weight)

    return ste_term, debug

def run_episode_full_cloud_validation(
    *,
    model,
    args,
    loss,
    writer,
    seq_datasets,
    episode,
    global_step,
    use_cuda,
    use_amp,
    amp_dtype,
):
    if (
        str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
        == "single_plan_student"
        and str(getattr(
            args, "single_plan_training_stage", "actual_calibration"
        )).strip().lower() in {"representation", "fast_distillation"}
    ):
        return {"value": None, "count": 0, "sample_names": []}
    max_frames = max(int(getattr(args, "train_full_cloud_val_frames", 5)), 0)
    if max_frames <= 0:
        return {"value": None, "count": 0, "sample_names": []}
    source = str(getattr(args, "checkpoint_actual_source", "auto")).strip().lower()
    compress_key = str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "")
    backend = str(getattr(args, "compression_loss_backend", "")).strip().lower()
    sparsepcgc_backend = compress_key == "sparsepcgc" or backend.startswith("sparsepcgc_")
    if source not in {"auto", "full_cloud"} or not sparsepcgc_backend:
        return {"value": None, "count": 0, "sample_names": []}

    values = []
    sample_names = []
    was_training = bool(model.training)
    old_replay_max = getattr(loss, "surrogate_replay_max_entries", None)
    saved_args = {
        name: getattr(args, name, None)
        for name in (
            "_current_teacher_scope",
            "_current_teacher_anchor_reason",
            "_current_exact_teacher_mode",
            "_current_exact_teacher_uses_full_context",
            "_current_exact_teacher_fallback_reason",
            "_current_sample_name",
            "_current_input_file",
            "_current_subtree_id",
            "_log_this_step",
            "_collect_structure_debug",
            "_collect_sparsepcgc_debug",
            "_den6_online_training_step_active",
        )
    }
    model.eval()
    if old_replay_max is not None:
        loss.surrogate_replay_max_entries = 0
    try:
        for _, dataset in seq_datasets:
            if len(values) >= max_frames:
                break
            for idx in range(len(dataset)):
                if len(values) >= max_frames:
                    break
                file_path = dataset.files[idx]
                args._current_input_file = str(Path(file_path).expanduser().resolve())
                pts = dataset[idx]
                input_common_cache_key = make_step_cache_key(file_path, args)
                cache_key = f"{input_common_cache_key}|episode_full_cloud_validation"
                args._global_train_step = int(global_step)
                args._current_sample_name = os.path.basename(str(file_path))
                args._current_teacher_scope = "full_cloud"
                args._current_teacher_anchor_reason = "episode_full_cloud_validation"
                args._current_exact_teacher_mode = "full_cloud"
                args._current_exact_teacher_uses_full_context = False
                args._current_exact_teacher_fallback_reason = ""
                args._current_subtree_id = ""
                args._log_this_step = False
                args._collect_structure_debug = False
                args._collect_sparsepcgc_debug = False
                args._den6_online_training_step_active = False
                try:
                    input_pcd = prepare_full_cloud_input_pcd(pts, use_cuda)
                    input_xyz = input_pcd[:, :3, :]
                    input_attr = input_pcd[:, 3:, :].contiguous() if input_pcd.shape[1] > 3 else None
                    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                    with torch.no_grad(), autocast_ctx:
                        full_octree_context = _episode_input_common_cache_fetch(
                            args,
                            _episode_input_common_cache_key(input_common_cache_key, "full_cloud_canonical"),
                            device=input_xyz.device,
                            section="full_cloud_canonical",
                        )
                        if full_octree_context is None:
                            full_octree_context = _build_full_cloud_octree_context_for_train(
                                input_xyz,
                                args,
                                coord_scale=None,
                            )
                            _episode_input_common_cache_store(
                                args,
                                _episode_input_common_cache_key(input_common_cache_key, "full_cloud_canonical"),
                                full_octree_context,
                            )
                        full_cloud_canonical_context = full_octree_context
                        _sparsepcgc_apply_amount_outcome_context(
                            args,
                            memory_key=None,
                            forward_key=cache_key,
                        )
                        gen_pts, _, _, _, final_w, _, _, _, out_label = model.forward(
                            input_xyz,
                            input_attr,
                            cache_key=cache_key,
                            return_attr_output=False,
                            subtree_ref=None,
                            selected_subtree_keys=None,
                            subtree_tree=None,
                            full_octree_context=full_octree_context,
                            octree_input_mode="full_cloud",
                        )
                        gen_xyz = gen_pts[:, :3, :]
                        final_w_for_loss = None if _discrete_loss_mode_value(args) == "hard" else final_w
                        gen_xyz_for_actual, voxel_restored_actual_debug = _select_actual_gen_xyz_from_voxel_state(
                            args,
                            writer,
                            model,
                            gen_xyz,
                            prefix="VoxelRestoredActual[episode_full_cloud_validation]",
                            canonical_context=full_cloud_canonical_context,
                        )

                        validation_voxel_state_used = bool(
                            isinstance(voxel_restored_actual_debug, dict)
                            and voxel_restored_actual_debug.get("used", False)
                            and not voxel_restored_actual_debug.get("fallback", False)
                        )

                        validation_compression_source_xyz = gen_xyz_for_actual if validation_voxel_state_used else gen_xyz

                        compression_gen_xyz, _ = prepare_compression_points(
                            validation_compression_source_xyz,
                            args,
                            model,
                            collect_stats=False,
                        )
                        setattr(
                            args,
                            "_current_actual_uses_voxel_restored",
                            bool(voxel_restored_actual_debug.get("used", False)) if isinstance(voxel_restored_actual_debug, dict) else False,
                        )
                        # Phase7-3: actual codecへ渡す点群だけの切替debug。
                        # geometry lossのgen_xyzは変更しない。
                        if isinstance(voxel_restored_actual_debug, dict):
                            try:
                                setattr(args, "_last_voxel_restored_actual_debug", dict(voxel_restored_actual_debug))
                            except Exception:
                                pass
                        args._current_exact_teacher_mode = "full_cloud"
                        args._current_exact_teacher_uses_full_context = False
                        args._current_exact_teacher_fallback_reason = ""
                        loss.get_compression_loss(
                            args,
                            gen_xyz=compression_gen_xyz,
                            gt_xyz=input_xyz[:, :3, :],
                            final_w=final_w_for_loss,
                            cache_key=cache_key,
                            refresh_actual_gen="always",
                            actual_gen_xyz=gen_xyz_for_actual,
                            subtree_tree=None,
                            full_octree_context=full_octree_context,
                            octree_input_mode="full_cloud",
                        )
                    comp_debug = dict(getattr(loss, "last_compression_debug", {}) or {})
                    phase7_voxel_actual_debug = getattr(args, "_last_voxel_restored_actual_debug", {}) or {}
                    if isinstance(phase7_voxel_actual_debug, dict):
                        comp_debug.update(
                            {
                                "use_voxel_restored_points_for_actual": bool(getattr(args, "use_voxel_restored_points_for_actual", False)),
                                "voxel_restored_actual_used": bool(phase7_voxel_actual_debug.get("used", False)),
                                "voxel_restored_actual_fallback": bool(phase7_voxel_actual_debug.get("fallback", False)),
                                "voxel_restored_actual_fallback_reason": str(phase7_voxel_actual_debug.get("reason", "")),
                                "restored_actual_points": int(phase7_voxel_actual_debug.get("restored_actual_points", phase7_voxel_actual_debug.get("points", 0)) or 0),
                                "original_gen_points": int(phase7_voxel_actual_debug.get("original_gen_points", 0) or 0),
                                "restored_actual_xyz_min": float(phase7_voxel_actual_debug.get("restored_actual_xyz_min", 0.0) or 0.0),
                                "restored_actual_xyz_max": float(phase7_voxel_actual_debug.get("restored_actual_xyz_max", 0.0) or 0.0),
                                "original_gen_xyz_min": float(phase7_voxel_actual_debug.get("original_gen_xyz_min", 0.0) or 0.0),
                                "original_gen_xyz_max": float(phase7_voxel_actual_debug.get("original_gen_xyz_max", 0.0) or 0.0),
                                "final_voxel_coords_count": int(phase7_voxel_actual_debug.get("final_voxel_coords_count", comp_debug.get("final_voxel_coords_count", 0)) or 0),
                            }
                        )

                    if _phase7_debug_enabled(args, global_step):
                        _phase7_writer_line(
                            args,
                            writer,
                            "Phase7ActualInputDebug: "
                            f"use_voxel_restored={bool(comp_debug.get('use_voxel_restored_points_for_actual', False))}, "
                            f"used={bool(comp_debug.get('voxel_restored_actual_used', False))}, "
                            f"fallback={bool(comp_debug.get('voxel_restored_actual_fallback', False))}, "
                            f"reason={comp_debug.get('voxel_restored_actual_fallback_reason', '')}, "
                            f"original_points={int(comp_debug.get('original_gen_points', 0) or 0)}, "
                            f"restored_points={int(comp_debug.get('restored_actual_points', 0) or 0)}, "
                            f"final_voxel_count={int(comp_debug.get('final_voxel_coords_count', 0) or 0)}, "
                            f"orig_range=[{float(comp_debug.get('original_gen_xyz_min', 0.0) or 0.0):.6g}, {float(comp_debug.get('original_gen_xyz_max', 0.0) or 0.0):.6g}], "
                            f"restored_range=[{float(comp_debug.get('restored_actual_xyz_min', 0.0) or 0.0):.6g}, {float(comp_debug.get('restored_actual_xyz_max', 0.0) or 0.0):.6g}]"
                        )
                    value = finite_float_or_none(
                        comp_debug.get("full_cloud_actual_percent", comp_debug.get("actual_total_bit_percent"))
                    )
                    if value is not None:
                        values.append(float(value))
                        sample_names.append(os.path.basename(str(file_path)))
                except Exception as exc:
                    writer.write(
                        "FullCloudValidationWarning: "
                        f"episode={episode + 1}, sample={os.path.basename(str(file_path))}, "
                        f"error={type(exc).__name__}: {str(exc)[:300]}"
                    )
    finally:
        if old_replay_max is not None:
            loss.surrogate_replay_max_entries = old_replay_max
        for name, value in saved_args.items():
            setattr(args, name, value)
        if was_training:
            model.train()

    avg_value = sum(values) / float(len(values)) if values else None
    writer.write(
        "FullCloudValidationSummary: "
        f"episode={episode + 1}, count={len(values)}, "
        f"actual_percent={avg_value if avg_value is not None else 'n/a'}, "
        f"samples={','.join(sample_names[:8]) or 'none'}"
    )
    return {"value": avg_value, "count": len(values), "sample_names": sample_names}

def load_more_training_checkpoint(model, args, writer):
    # more_training=False の場合は、追加学習用checkpointを読まない
    if not bool(getattr(args, "more_training", False)):
        writer.write("MoreTraining: disabled. Start training from current initialized model.")
        return model

    ckpt_path = str(getattr(args, "more_training_ckpt", "")).strip()

    # more_training=True なのに読み込み先が空なら停止する
    if not ckpt_path:
        raise ValueError("MoreTraining: args.more_training_ckpt is empty, but args.more_training=True.")

    # 誤った初期値で学習を始めないため、存在しない場合は停止する
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"MoreTraining: checkpoint file not found: {ckpt_path}")

    writer.write("========== MoreTraining Resume ==========")
    writer.write(f"MoreTraining: enabled=True")
    writer.write(f"MoreTraining: load_model_path={ckpt_path}")
    writer.write(f"MoreTraining: pretrained_date={getattr(args, 'pretrained_date', '')}")
    writer.write(f"MoreTraining: pretrained_time={getattr(args, 'pretrained_time', '')}")
    writer.write(f"MoreTraining: compress={getattr(args, 'compress', '')}")
    writer.write(f"MoreTraining: method_com={getattr(args, 'method_com', 'not_in_args')}")

    # CPUへ読み込むことで、GPUメモリの一時使用量を抑える
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # checkpointの保存形式に合わせてstate_dictを取り出す
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            checkpoint_format = "model_state_dict"
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            checkpoint_format = "state_dict"
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            checkpoint_format = "model"
        elif "net" in checkpoint:
            state_dict = checkpoint["net"]
            checkpoint_format = "net"
        else:
            # save_episode_checkpoint が model.state_dict() を直接保存している場合を想定する
            state_dict = checkpoint
            checkpoint_format = "raw_state_dict"
    else:
        raise TypeError(f"MoreTraining: unsupported checkpoint type: {type(checkpoint).__name__}")

    # DataParallelで保存された場合の module. 接頭辞を除去する
    cleaned_state_dict = {}
    current_state_dict = model.state_dict()
    shape_mismatch_keys = []
    for key, value in state_dict.items():
        key_text = str(key)
        new_key = key_text[7:] if key_text.startswith("module.") else key_text
        current_value = current_state_dict.get(new_key)
        if (
            torch.is_tensor(value)
            and torch.is_tensor(current_value)
            and tuple(value.shape) != tuple(current_value.shape)
        ):
            # strict=Falseでもshape不一致はPyTorchが例外にする。互換性のない
            # legacy補助headだけを初期値のまま残し、Student重みは保持する。
            shape_mismatch_keys.append(
                "{}:{}->{}".format(
                    new_key, tuple(value.shape), tuple(current_value.shape)
                )
            )
            continue
        cleaned_state_dict[new_key] = value

    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)

    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

    writer.write(f"MoreTraining: checkpoint_format={checkpoint_format}")
    writer.write(f"MoreTraining: loaded_parameter_keys={len(cleaned_state_dict)}")
    writer.write(f"MoreTraining: missing_keys_count={len(missing_keys)}")
    writer.write(f"MoreTraining: unexpected_keys_count={len(unexpected_keys)}")
    writer.write(f"MoreTraining: shape_mismatch_skipped_count={len(shape_mismatch_keys)}")
    if shape_mismatch_keys:
        writer.write(
            "MoreTraining: shape_mismatch_skipped_detail="
            + ", ".join(shape_mismatch_keys[:50])
        )

    if missing_keys:
        writer.write("MoreTraining: missing_keys_detail=" + ", ".join(missing_keys[:50]))
        if len(missing_keys) > 50:
            writer.write(f"MoreTraining: missing_keys_detail_truncated=True total={len(missing_keys)}")

    if unexpected_keys:
        writer.write("MoreTraining: unexpected_keys_detail=" + ", ".join(unexpected_keys[:50]))
        if len(unexpected_keys) > 50:
            writer.write(f"MoreTraining: unexpected_keys_detail_truncated=True total={len(unexpected_keys)}")

    reset_full_cloud_amount_heads = bool(
        getattr(args, "sparsepcgc_full_cloud_amount_reset_heads_on_more_training", True)
    )
    training_mode = str(getattr(args, "sparsepcgc_training_mode", "subtree_selector")).strip().lower()
    amount_learning_mode = str(
        getattr(args, "sparsepcgc_full_cloud_amount_learning_mode", "network_selected_bandit")
    ).strip().lower()
    if (
        reset_full_cloud_amount_heads
        and training_mode == "full_cloud_amount"
        and amount_learning_mode == "network_selected_bandit"
    ):
        selector = getattr(model, "full_cloud_amount_selector", None)
        if selector is not None and hasattr(selector, "reset_amount_heads"):
            selector.reset_amount_heads()
            writer.write(
                "MoreTraining: reset full_cloud_amount_selector amount heads after checkpoint load "
                "(full_cloud_amount + network_selected_bandit)."
            )

    writer.write("MoreTraining: model parameters loaded. Training will continue from this checkpoint.")
    writer.write("=========================================")

    args._more_training_loaded = True
    args._more_training_ckpt_path = ckpt_path
    args._more_training_missing_keys = len(missing_keys)
    args._more_training_unexpected_keys = len(unexpected_keys)

    return model


def record_training_memory(
    memory_diagnostics,
    phase,
    *,
    args,
    model,
    loss,
    writer,
    episode=-1,
    epoch=-1,
    step=-1,
    global_step=-1,
    sample="",
):
    """診断失敗で学習を止めず、OOM直前の行だけ確実に残す。"""
    try:
        return memory_diagnostics.record(
            phase,
            args=args,
            model=model,
            loss=loss,
            episode=episode,
            epoch=epoch,
            step=step,
            global_step=global_step,
            sample=sample,
        )
    except Exception as exc:
        writer.write(
            "MemoryDiagnosticsWarning: "
            f"phase={phase}, error={type(exc).__name__}: {str(exc)[:200]}"
        )
        return {}


def audit_state_scalar(audit_state, key, default=0.0):
    value = audit_state.get(key, default) if isinstance(audit_state, dict) else default
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return float(value or default)


def audit_state_list(audit_state, key):
    value = audit_state.get(key, []) if isinstance(audit_state, dict) else []
    if torch.is_tensor(value):
        return value.detach().float().reshape(-1).cpu().tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


# private helperもtrain.pyの互換import対象に含める。
__all__ = [name for name in globals() if not name.startswith('__')]
