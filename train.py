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
from models.utils.training.utils_grad import *
from models.utils.config.args import parse_pugan_args

from models.utils.training.full_cloud_actual_correction import (
    update_full_cloud_actual_correction_state,
    build_full_cloud_actual_correction_loss,
)
from models.utils.training.saved_tensor_offload import (
    selective_saved_tensor_cpu_offload,
)
from models.utils.training.memory_diagnostics import MemoryDiagnosticsCSV

from models.utils.training.utils import *
from models.utils.training.noise_debug import *
from models.utils.training.correlation import *
from models.utils.training.optim_amp import *
from models.utils.training.checkpointing import save_episode_checkpoint
from models.utils.training.train_logging import *
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


def _sparsepcgc_scalar_tensor(value, reference):
    """
    Tensor/floatをreferenceと同じdevice/dtypeのscalar Tensorへ揃える。
    """
    if torch.is_tensor(value):
        value = value.to(device=reference.device, dtype=reference.dtype)
        return value.mean()
    try:
        return reference.new_tensor(float(value))
    except Exception:
        return reference.new_zeros(())


def _sparsepcgc_subtree_actual_filter_debug(raw_percent, used_percent, label_id, weight, used):
    return {
        "subtree_actual_filter_used": bool(used),
        "subtree_actual_filter_label_id": int(label_id),
        "subtree_actual_filter_weight": float(weight),
        "subtree_actual_filter_raw_percent": float(raw_percent) if raw_percent is not None and math.isfinite(float(raw_percent)) else float("nan"),
        "subtree_actual_filter_used_percent": float(used_percent) if used_percent is not None and math.isfinite(float(used_percent)) else float("nan"),
    }


def _resolve_subtree_actual_filter(args, subtree_comp_debug):
    raw_percent = _sparsepcgc_outcome_actual_percent(subtree_comp_debug)
    used_percent = finite_float_or_none(
        subtree_comp_debug.get(
            "compression_loss_used",
            subtree_comp_debug.get("actual_bit_percent_used_for_loss", raw_percent),
        )
    )
    if not bool(getattr(args, "sparsepcgc_subtree_actual_filter", True)):
        return 1.0, 0, "disabled", _sparsepcgc_subtree_actual_filter_debug(
            raw_percent, used_percent, 0, 1.0, False
        )

    good_margin = max(float(getattr(args, "sparsepcgc_subtree_good_margin", 0.25)), 0.0)
    bad_margin = max(float(getattr(args, "sparsepcgc_subtree_bad_margin", 0.25)), 0.0)
    if raw_percent is not None and raw_percent < -good_margin:
        weight = max(float(getattr(args, "sparsepcgc_subtree_good_compression_weight", 1.0)), 0.0)
        label_id, label = 1, "good"
    elif raw_percent is not None and raw_percent > bad_margin:
        weight = max(float(getattr(args, "sparsepcgc_subtree_bad_compression_weight", 0.0)), 0.0)
        label_id, label = 3, "bad"
    else:
        weight = max(float(getattr(args, "sparsepcgc_subtree_neutral_compression_weight", 0.25)), 0.0)
        label_id, label = 2, "neutral"
    return weight, label_id, label, _sparsepcgc_subtree_actual_filter_debug(
        raw_percent, used_percent, label_id, weight, True
    )


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


def _selected_metadata_input_common_cache_key(cache_key, selected_groups, subtree_depth, args):
    selected_keys = sorted(int(subtree_key) for subtree_key, _ in (selected_groups or []))
    selected_text = ",".join(str(key) for key in selected_keys) if selected_keys else "none"
    selected_hash = hashlib.sha1(selected_text.encode("utf-8")).hexdigest()[:16]
    repair_unit_level = int(getattr(args, "repair_unit_level", int(subtree_depth)))
    return _episode_input_common_cache_key(
        cache_key,
        "selected_metadata",
        depth=int(subtree_depth),
        repair_unit_level=int(repair_unit_level),
        selected_count=len(selected_keys),
        selected_hash=selected_hash,
    )


def _selected_subtree_runtime_input_common_cache_key(cache_key, subtree_key, subtree_depth, point_count):
    return _episode_input_common_cache_key(
        cache_key,
        "selected_subtree_runtime",
        depth=int(subtree_depth),
        subtree_key=int(subtree_key),
        point_count=int(point_count),
    )


def _subtree_potential_input_common_cache_key(cache_key, subtree_depth):
    return _episode_input_common_cache_key(
        cache_key,
        "subtree_potential_scores",
        depth=int(subtree_depth),
    )


def _build_selected_subtree_runtime_cache_entry(
    *,
    input_xyz,
    input_attr_full,
    full_cloud_canonical_context,
    subtree_tree,
    full_octree_context,
    point_idx,
    group_meta,
):
    subtree_xyz = input_xyz.index_select(2, point_idx).contiguous()
    subtree_attr = (
        input_attr_full.index_select(2, point_idx).contiguous()
        if torch.is_tensor(input_attr_full)
        else None
    )
    patched_subtree_tree, patched_full_context = _inject_full_cloud_canonical_into_subtree_metadata(
        subtree_tree=subtree_tree,
        full_octree_context=full_octree_context,
        full_cloud_canonical_context=full_cloud_canonical_context,
        point_idx=point_idx,
        device=input_xyz.device,
    )
    return {
        "subtree_xyz": subtree_xyz,
        "subtree_attr": subtree_attr,
        "subtree_tree": patched_subtree_tree,
        "full_octree_context": patched_full_context,
        "group_meta": dict(group_meta or {}),
        "point_count": int(point_idx.numel()),
    }


def _sparsepcgc_success_amount_memory_key(cache_key, subtree_key):
    return f"{str(cache_key)}|subtree={int(subtree_key)}"


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


def _sparsepcgc_amount_outcome_memory_key(cache_key, subtree_key):
    return _sparsepcgc_success_amount_memory_key(cache_key, subtree_key)


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


def _sparsepcgc_subtree_outcome_memory(args):
    memory = getattr(args, "_sparsepcgc_subtree_outcome_memory", None)
    if not isinstance(memory, OrderedDict):
        memory = OrderedDict()
        setattr(args, "_sparsepcgc_subtree_outcome_memory", memory)
    return memory


def _sparsepcgc_subtree_outcome_memory_key(cache_key, subtree_key):
    return f"{str(cache_key)}|subtree={int(subtree_key)}"


def _sparsepcgc_subtree_outcome_lookup(args, cache_key, subtree_key):
    if not bool(getattr(args, "sparsepcgc_subtree_outcome_selector", True)):
        return None
    key = _sparsepcgc_subtree_outcome_memory_key(cache_key, subtree_key)
    entry = _sparsepcgc_subtree_outcome_memory(args).get(key, None)
    return entry if isinstance(entry, dict) else None


def _sparsepcgc_update_subtree_outcome_memory(args, cache_key, subtree_key, actual_percent):
    debug = {
        "subtree_outcome_memory_saved": False,
        "subtree_outcome_memory_score": float("nan"),
        "subtree_outcome_memory_count": 0,
        "subtree_outcome_memory_good_count": 0,
        "subtree_outcome_memory_bad_count": 0,
    }
    if not bool(getattr(args, "sparsepcgc_subtree_outcome_selector", True)):
        return debug
    try:
        actual_percent = float(actual_percent)
        subtree_key = int(subtree_key)
    except Exception:
        return debug
    if not math.isfinite(actual_percent):
        return debug
    memory = _sparsepcgc_subtree_outcome_memory(args)
    key = _sparsepcgc_subtree_outcome_memory_key(cache_key, subtree_key)
    entry = memory.get(key, None)
    if not isinstance(entry, dict):
        entry = {
            "score_ema": 0.0,
            "count": 0,
            "good_count": 0,
            "bad_count": 0,
            "best_percent": float("inf"),
            "last_percent": float("nan"),
        }
    good_margin = max(float(getattr(args, "sparsepcgc_amount_outcome_good_margin", 0.25)), 0.0)
    bad_margin = max(float(getattr(args, "sparsepcgc_amount_outcome_bad_margin", 0.25)), 0.0)
    reward = 0.0
    if actual_percent < -good_margin:
        reward = min((-actual_percent - good_margin) / 5.0, 2.0)
        entry["good_count"] = int(entry.get("good_count", 0)) + 1
    elif actual_percent > bad_margin:
        reward = -min((actual_percent - bad_margin) / 5.0, 2.0)
        entry["bad_count"] = int(entry.get("bad_count", 0)) + 1
    ema = min(max(float(getattr(args, "sparsepcgc_subtree_outcome_memory_ema", 0.20)), 1e-4), 1.0)
    entry["score_ema"] = (1.0 - ema) * float(entry.get("score_ema", 0.0)) + ema * float(reward)
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["best_percent"] = min(float(entry.get("best_percent", float("inf"))), actual_percent)
    entry["last_percent"] = float(actual_percent)
    memory[key] = entry
    memory.move_to_end(key)
    max_entries = max(int(getattr(args, "episode_input_common_cache_max_entries", 0)), 256)
    while len(memory) > max_entries:
        memory.popitem(last=False)
    debug.update(
        {
            "subtree_outcome_memory_saved": True,
            "subtree_outcome_memory_score": float(entry.get("score_ema", 0.0)),
            "subtree_outcome_memory_count": int(entry.get("count", 0)),
            "subtree_outcome_memory_good_count": int(entry.get("good_count", 0)),
            "subtree_outcome_memory_bad_count": int(entry.get("bad_count", 0)),
        }
    )
    return debug


def _sparsepcgc_proposal_amount_bins(args):
    values = getattr(args, "sparsepcgc_proposal_amount_bin_values", None)
    if not isinstance(values, (list, tuple)) or not values:
        values = (0.0, 0.015, 0.021, 0.026, 0.031, 0.038, 0.044, 0.05)
    out = [0.0]
    for value in values:
        try:
            out.append(min(max(float(value), 0.0), 0.05))
        except Exception:
            continue
    return tuple(sorted(set(out)))


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


def _sparsepcgc_proposal_terms_for_subtree(args, subtree_key):
    terms = getattr(args, "_current_sparsepcgc_proposal_terms_by_key", None)
    if not isinstance(terms, dict):
        return None
    try:
        return terms.get(int(subtree_key), None)
    except Exception:
        return None


def _build_sparsepcgc_proposal_candidate_teacher_loss(
    args,
    proposal_terms,
    *,
    actual_percent,
    subtree_key,
    cache_key,
    global_step,
    episode,
    epoch,
    step,
    geom_loss=None,
):
    debug = {
        "proposal_selector_enabled": False,
        "proposal_candidate_count": 0,
        "proposal_actual_eval_count": 0,
        "proposal_surrogate_prefilter_count": 0,
        "proposal_applied_subtree_count": 0,
        "proposal_selected_subtree_count": 0,
        "proposal_noop_count": 0,
        "proposal_best_actual_percent": float("nan"),
        "proposal_chosen_actual_percent": float("nan"),
        "proposal_predicted_delta": float("nan"),
        "proposal_amount_bin": float("nan"),
        "proposal_amount_residual": float("nan"),
        "proposal_final_amount": float("nan"),
        "proposal_cls_loss": 0.0,
        "proposal_value_loss": 0.0,
        "proposal_rank_loss": 0.0,
        "proposal_geom_loss": 0.0,
        "proposal_total_loss": 0.0,
        "proposal_teacher_source": "none",
        "verified_noop_guard_used": False,
    }
    rows = []
    if not isinstance(proposal_terms, dict):
        return None, debug, rows
    amount_logits = proposal_terms.get("amount_bin_logits", None)
    pred_per_amount = proposal_terms.get("predicted_delta_per_amount", None)
    select_logit = proposal_terms.get("subtree_select_logit", None)
    subtree_pred_delta = proposal_terms.get("subtree_predicted_delta", None)
    residual_raw = proposal_terms.get("amount_residual_raw", None)
    if not (torch.is_tensor(amount_logits) and torch.is_tensor(pred_per_amount)):
        return None, debug, rows

    bins = _sparsepcgc_proposal_amount_bins(args)
    class_count = min(int(amount_logits.numel()), len(bins))
    if class_count <= 0:
        return None, debug, rows
    amount_logits = amount_logits.flatten()[:class_count]
    pred_per_amount = pred_per_amount.flatten()[:class_count]
    ref = amount_logits
    bin_tensor = ref.new_tensor(list(bins[:class_count]))
    selected_class = int(torch.argmax(amount_logits.detach()).item())
    selected_class = min(max(selected_class, 0), class_count - 1)
    residual_max = min(
        max(float(getattr(args, "sparsepcgc_proposal_amount_residual_max", 0.0025)), 0.0),
        0.01,
    )
    residual_enabled = bool(getattr(args, "sparsepcgc_proposal_amount_residual_enable", True))
    if torch.is_tensor(residual_raw):
        residual_tensor = torch.tanh(residual_raw.reshape(())) * float(residual_max)
    else:
        residual_tensor = ref.new_zeros(())
    if not residual_enabled or selected_class == 0:
        residual_tensor = residual_tensor * 0.0
    selected_bin_tensor = bin_tensor[selected_class]
    final_amount_tensor = torch.clamp(selected_bin_tensor + residual_tensor, 0.0, 0.05)
    if selected_class == 0:
        final_amount_tensor = final_amount_tensor * 0.0

    actual_value = finite_float_or_none(actual_percent)
    actual_available = actual_value is not None and math.isfinite(float(actual_value))
    noop_margin = max(float(getattr(args, "sparsepcgc_proposal_noop_margin", 0.0)), 0.0)
    if actual_available:
        if float(actual_value) < -float(noop_margin):
            teacher_class = int(selected_class)
            teacher_delta = float(actual_value)
        else:
            teacher_class = 0
            teacher_delta = 0.0
        teacher_source = "actual_subtree"
    else:
        teacher_class = int(torch.argmin(pred_per_amount.detach()).item())
        teacher_delta = float(pred_per_amount.detach().flatten()[teacher_class].cpu())
        teacher_source = "surrogate_fallback"
    teacher_class = min(max(int(teacher_class), 0), class_count - 1)

    candidate_classes = {0, selected_class}
    if bool(getattr(args, "sparsepcgc_proposal_eval_neighbor_amounts", True)):
        candidate_classes.add(max(selected_class - 1, 0))
        candidate_classes.add(min(selected_class + 1, class_count - 1))
    if bool(getattr(args, "sparsepcgc_proposal_use_surrogate_prefilter", True)):
        candidate_classes.add(int(torch.argmin(pred_per_amount.detach()).item()))
    candidate_classes = sorted(candidate_classes)

    candidate_teacher_values = {}
    for cls in candidate_classes:
        if cls == 0:
            candidate_teacher_values[int(cls)] = 0.0
        elif actual_available and cls == selected_class:
            candidate_teacher_values[int(cls)] = float(actual_value)
        else:
            candidate_teacher_values[int(cls)] = float(pred_per_amount.detach().flatten()[cls].cpu())
    if candidate_teacher_values:
        best_cls = min(candidate_teacher_values, key=lambda cls: candidate_teacher_values[cls])
        best_value = float(candidate_teacher_values[best_cls])
        if best_value < -float(noop_margin):
            teacher_class = int(best_cls)
            teacher_delta = best_value
            if actual_available and best_cls == selected_class:
                teacher_source = "actual_subtree"
            elif actual_available:
                teacher_source = "actual_subtree_mixed_surrogate"
            else:
                teacher_source = "surrogate_fallback"
        else:
            teacher_class = 0
            teacher_delta = 0.0
            teacher_source = "actual_subtree" if actual_available else "surrogate_fallback"

    for cand_idx, cls in enumerate(candidate_classes):
        is_noop = cls == 0
        cand_bin = float(bin_tensor.detach().flatten()[cls].cpu())
        cand_residual = 0.0 if is_noop else float(residual_tensor.detach().cpu())
        cand_amount = 0.0 if is_noop else min(max(cand_bin + cand_residual, 0.0), 0.05)
        cand_actual = 0.0 if is_noop else (float(actual_value) if cls == selected_class and actual_available else float("nan"))
        cand_source = "noop" if is_noop else ("network_selected" if cls == selected_class else "neighbor_or_surrogate")
        rows.append(
            {
                "global_step": int(global_step) + 1,
                "episode": int(episode) + 1,
                "epoch": int(epoch) + 1,
                "step": int(step) + 1,
                "sample_key": str(cache_key),
                "subtree_key": int(subtree_key),
                "candidate_id": int(cand_idx),
                "amount_bin": cand_bin,
                "amount_residual": cand_residual,
                "final_amount": cand_amount,
                "is_noop": bool(is_noop),
                "proposal_valid": True,
                "proposal_drop_count": 0,
                "actual_percent": cand_actual,
                "surrogate_percent": float(pred_per_amount.detach().flatten()[cls].cpu()),
                "predicted_delta": float(pred_per_amount.detach().flatten()[cls].cpu()),
                "teacher_is_best": bool(cls == teacher_class),
                "teacher_label": int(teacher_class),
                "candidate_source": cand_source,
                "actual_scope": "subtree" if actual_available else "none",
                "teacher_source": teacher_source,
            }
        )

    target = torch.tensor([teacher_class], device=amount_logits.device, dtype=torch.long)
    cls_loss = torch.nn.functional.cross_entropy(amount_logits.view(1, -1).float(), target)
    teacher_delta_tensor = ref.new_tensor(float(teacher_delta))
    pred_teacher = pred_per_amount[teacher_class]
    value_loss = torch.nn.functional.smooth_l1_loss(pred_teacher, teacher_delta_tensor)
    if torch.is_tensor(subtree_pred_delta):
        value_loss = value_loss + torch.nn.functional.smooth_l1_loss(
            subtree_pred_delta.reshape(()),
            teacher_delta_tensor,
        )
    selected_pred = pred_per_amount[selected_class]
    teacher_pred = pred_per_amount[teacher_class]
    rank_margin = ref.new_tensor(0.1)
    rank_loss = torch.relu(rank_margin + teacher_pred - selected_pred) if teacher_class != selected_class else ref.new_zeros(())
    if teacher_class == 0 and selected_class != 0:
        rank_loss = torch.relu(rank_margin + pred_per_amount[0] - selected_pred)
    if torch.is_tensor(select_logit):
        subtree_target = ref.new_tensor(0.0 if teacher_class == 0 else 1.0)
        select_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            select_logit.reshape(()),
            subtree_target,
        )
        cls_loss = cls_loss + select_loss
    if torch.is_tensor(geom_loss):
        geom_scalar = torch.nan_to_num(geom_loss.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        if torch.is_tensor(select_logit):
            geom_loss_term = torch.sigmoid(select_logit.reshape(())) * torch.relu(geom_scalar)
        else:
            geom_loss_term = torch.relu(geom_scalar) * 0.0
    else:
        geom_loss_term = ref.new_zeros(())

    total_loss = (
        float(getattr(args, "sparsepcgc_proposal_cls_loss_weight", 1.0)) * cls_loss
        + float(getattr(args, "sparsepcgc_proposal_value_loss_weight", 0.5)) * value_loss
        + float(getattr(args, "sparsepcgc_proposal_rank_loss_weight", 0.2)) * rank_loss
        + float(getattr(args, "sparsepcgc_proposal_geom_penalty_weight", 0.1)) * geom_loss_term
    )
    verified_guard = bool(
        str(getattr(args, "sparsepcgc_proposal_inference_mode", "fast")).strip().lower() == "verified"
        and actual_available
        and selected_class != 0
        and float(actual_value) >= -float(noop_margin)
    )
    debug.update(
        {
            "proposal_selector_enabled": True,
            "proposal_candidate_count": int(len(rows)),
            "proposal_actual_eval_count": 1 if actual_available else 0,
            "proposal_surrogate_prefilter_count": int(max(len(rows) - (2 if actual_available else 1), 0)),
            "proposal_applied_subtree_count": 0 if teacher_class == 0 else 1,
            "proposal_selected_subtree_count": 1,
            "proposal_noop_count": 1 if teacher_class == 0 else 0,
            "proposal_best_actual_percent": min(0.0, float(actual_value)) if actual_available else float("nan"),
            "proposal_chosen_actual_percent": float(actual_value) if actual_available else float("nan"),
            "proposal_predicted_delta": float(
                subtree_pred_delta.detach().cpu()
            ) if torch.is_tensor(subtree_pred_delta) else float(pred_per_amount.detach()[selected_class].cpu()),
            "proposal_amount_bin": float(selected_bin_tensor.detach().cpu()),
            "proposal_amount_residual": float(residual_tensor.detach().cpu()),
            "proposal_final_amount": float(final_amount_tensor.detach().cpu()),
            "proposal_cls_loss": float(cls_loss.detach().cpu()),
            "proposal_value_loss": float(value_loss.detach().cpu()),
            "proposal_rank_loss": float(rank_loss.detach().cpu()),
            "proposal_geom_loss": float(geom_loss_term.detach().cpu()),
            "proposal_total_loss": float(total_loss.detach().cpu()),
            "proposal_teacher_source": str(teacher_source),
            "verified_noop_guard_used": bool(verified_guard),
        }
    )
    return total_loss, debug, rows


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


def _sparsepcgc_update_success_amount_memory(args, memory_key, actual_percent, hard_ratio):
    """
    actualで改善したSubtreeのPrune量をEMAで記憶する。
    Amountが0へ逃げることを防ぐための教師として使う。
    """
    if not bool(getattr(args, "sparsepcgc_success_amount_memory", True)):
        return None

    if memory_key is None:
        return None

    try:
        actual_percent = float(actual_percent)
        hard_ratio = float(hard_ratio)
    except Exception:
        return None

    if not (math.isfinite(actual_percent) and math.isfinite(hard_ratio)):
        return None

    memory = getattr(args, "_sparsepcgc_success_amount_memory", None)
    if not isinstance(memory, dict):
        memory = {}
        setattr(args, "_sparsepcgc_success_amount_memory", memory)

    item = memory.get(memory_key, None)

    good_margin = max(float(getattr(args, "sparsepcgc_outcome_good_margin", 0.25)), 0.0)
    if actual_percent < -good_margin and hard_ratio > 0.0:
        ema = min(max(float(getattr(args, "sparsepcgc_success_amount_ema", 0.20)), 1e-4), 1.0)
        if isinstance(item, dict):
            old_ratio = float(item.get("ratio", hard_ratio))
            old_best = float(item.get("best_percent", actual_percent))
            count = int(item.get("count", 0)) + 1
            new_ratio = (1.0 - ema) * old_ratio + ema * hard_ratio
            best_percent = min(old_best, actual_percent)
        else:
            count = 1
            new_ratio = hard_ratio
            best_percent = actual_percent

        memory[memory_key] = {
            "ratio": float(new_ratio),
            "count": int(count),
            "best_percent": float(best_percent),
        }
        return float(new_ratio)

    if isinstance(item, dict):
        try:
            return float(item.get("ratio", 0.0))
        except Exception:
            return None

    return None


def _build_sparsepcgc_outcome_weighted_imitation_loss(
    args,
    *,
    actual_percent,
    actuator_terms,
    reference,
    memory_key=None,
):
    """
    actual結果に基づき、WhereとAmountを追加学習する。

    actual_percent < 0:
      実際に圧縮損失が下がった行動なので、hard_drop_maskを模倣する。

    actual_percent > 0:
      実際に悪化した行動なので、そのhard_drop_maskを避ける。

    注意:
      bad amount抑制を強くしすぎるとPrune量が0へ逃げる。
      そのためbad amount weightは小さくし、成功Amount memoryで下支えする。
    """
    zero = reference.new_zeros(())
    debug = {
        "outcome_imitation_used": False,
        "outcome_actual_percent": float("nan"),
        "outcome_good_weight": 0.0,
        "outcome_bad_weight": 0.0,
        "outcome_where_loss": 0.0,
        "outcome_amount_loss": 0.0,
        "outcome_amount_anticollapse_loss": 0.0,
        "outcome_success_amount_teacher": float("nan"),
        "bad_amount_loss_disabled_no_success_memory": False,
        "outcome_bad_amount_policy_id": 0,
    }

    if not bool(getattr(args, "sparsepcgc_outcome_imitation", True)):
        return zero, debug

    if not isinstance(actuator_terms, dict):
        return zero, debug

    try:
        actual_percent = float(actual_percent)
    except Exception:
        return zero, debug

    if not math.isfinite(actual_percent):
        return zero, debug

    debug["outcome_actual_percent"] = float(actual_percent)

    logit = actuator_terms.get("network_drop_logit_for_outcome", None)
    if logit is None:
        logit = actuator_terms.get("learned_drop_logit", None)

    hard_mask = actuator_terms.get("hard_drop_mask_for_outcome", None)
    candidate_mask = actuator_terms.get("hard_delete_selection_mask_for_outcome", None)

    if not torch.is_tensor(logit) or not torch.is_tensor(hard_mask):
        return zero, debug

    logit = logit.to(device=reference.device, dtype=reference.dtype)
    hard_target = hard_mask.to(device=reference.device, dtype=reference.dtype).detach()

    if hard_target.shape != logit.shape:
        try:
            hard_target = hard_target.reshape_as(logit)
        except Exception:
            return zero, debug

    if torch.is_tensor(candidate_mask):
        candidate = candidate_mask.to(device=reference.device, dtype=torch.bool)
        if candidate.shape != logit.shape:
            try:
                candidate = candidate.reshape_as(logit)
            except Exception:
                candidate = torch.ones_like(logit, dtype=torch.bool)
    else:
        candidate = torch.ones_like(logit, dtype=torch.bool)

    hard_bool = hard_target > 0.5
    good_margin = max(float(getattr(args, "sparsepcgc_outcome_good_margin", 0.25)), 0.0)
    bad_margin = max(float(getattr(args, "sparsepcgc_outcome_bad_margin", 0.25)), 0.0)
    scale = max(float(getattr(args, "sparsepcgc_outcome_weight_scale", 5.0)), 1e-6)
    max_weight = max(float(getattr(args, "sparsepcgc_outcome_max_weight", 2.0)), 0.0)

    good_weight = 0.0
    bad_weight = 0.0
    if actual_percent < -good_margin:
        good_weight = min((-actual_percent - good_margin) / scale, max_weight)
    elif actual_percent > bad_margin:
        bad_weight = min((actual_percent - bad_margin) / scale, max_weight)

    debug["outcome_good_weight"] = float(good_weight)
    debug["outcome_bad_weight"] = float(bad_weight)

    total_loss = zero
    where_loss = zero
    amount_loss = zero
    anticollapse_loss = zero

    # ------------------------------------------------------------
    # Where imitation / anti-imitation
    # ------------------------------------------------------------
    if good_weight > 0.0:
        # 良い行動では、選ばれたmaskを正例にする。
        # 非選択候補も弱い負例にするが、重みは小さくする。
        weight = candidate.to(dtype=reference.dtype) * 0.05 + hard_bool.to(dtype=reference.dtype) * 0.95
        if bool(weight.detach().sum().item() > 0):
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logit.float(),
                hard_target.float(),
                reduction="none",
            ).to(dtype=reference.dtype)
            where_loss = (raw * weight).sum() / weight.sum().clamp_min(1.0)
            total_loss = total_loss + (
                float(getattr(args, "sparsepcgc_outcome_where_weight", 0.05))
                * float(good_weight)
                * where_loss
            )

    elif bad_weight > 0.0 and bool(hard_bool.detach().any().item()):
        # 悪い行動では、実際に削った場所だけを負例にする。
        # 全候補を負例にするとPrune全体が潰れるので禁止。
        weight = hard_bool.to(dtype=reference.dtype)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            logit.float(),
            torch.zeros_like(logit, dtype=torch.float32),
            reduction="none",
        ).to(dtype=reference.dtype)
        where_loss = (raw * weight).sum() / weight.sum().clamp_min(1.0)
        total_loss = total_loss + (
            float(getattr(args, "sparsepcgc_outcome_bad_where_weight", 0.02))
            * float(bad_weight)
            * where_loss
        )

    # ------------------------------------------------------------
    # Amount imitation / anti-collapse
    # ------------------------------------------------------------
    raw_ratio = actuator_terms.get("raw_learned_drop_ratio_for_outcome", None)
    if raw_ratio is None:
        raw_ratio = actuator_terms.get("raw_learned_drop_ratio", None)

    hard_ratio = actuator_terms.get("drop_ratio_hard_for_outcome", None)
    if hard_ratio is None:
        hard_ratio = actuator_terms.get("drop_ratio_hard", None)

    if torch.is_tensor(raw_ratio):
        raw_ratio_t = _sparsepcgc_scalar_tensor(raw_ratio, reference)
        hard_ratio_t = _sparsepcgc_scalar_tensor(hard_ratio, reference).detach().clamp(0.0, 0.95)

        success_teacher = _sparsepcgc_update_success_amount_memory(
            args,
            memory_key,
            actual_percent,
            float(hard_ratio_t.detach().cpu()),
        )

        if success_teacher is not None and math.isfinite(float(success_teacher)):
            debug["outcome_success_amount_teacher"] = float(success_teacher)

        if good_weight > 0.0:
            # 改善したStepの実Prune量を教師にする。
            amount_loss = (raw_ratio_t - hard_ratio_t).pow(2)
            total_loss = total_loss + (
                float(getattr(args, "sparsepcgc_outcome_amount_weight", 0.05))
                * float(good_weight)
                * amount_loss
            )

        elif bad_weight > 0.0:
            bad_policy = str(
                getattr(args, "sparsepcgc_outcome_bad_amount_policy", "where_only")
            ).strip().lower()
            if bad_policy not in {"where_only", "success_guarded", "legacy"}:
                bad_policy = "where_only"
            debug["outcome_bad_amount_policy_id"] = {
                "where_only": 1,
                "success_guarded": 2,
                "legacy": 3,
            }.get(bad_policy, 0)
            if bad_policy == "where_only":
                pass
            elif (
                bool(getattr(args, "sparsepcgc_disable_bad_amount_when_no_success_memory", True))
                and not (success_teacher is not None and math.isfinite(float(success_teacher)))
            ):
                debug["bad_amount_loss_disabled_no_success_memory"] = True
            else:
                if bad_policy == "success_guarded" and success_teacher is not None and math.isfinite(float(success_teacher)):
                    min_keep = min(
                        max(float(getattr(args, "sparsepcgc_success_amount_min_keep", 0.60)), 0.0),
                        1.0,
                    )
                    success_floor = reference.new_tensor(float(success_teacher) * float(min_keep))
                    bad_target = torch.maximum(
                        (hard_ratio_t.detach() * 0.5).clamp(0.0, 0.95),
                        success_floor.clamp(0.0, 0.95),
                    )
                else:
                    bad_target = (hard_ratio_t.detach() * 0.5).clamp(0.0, 0.95)
                amount_loss = (raw_ratio_t - bad_target).pow(2)
                total_loss = total_loss + (
                    float(getattr(args, "sparsepcgc_outcome_bad_amount_weight", 0.005))
                    * float(bad_weight)
                    * amount_loss
                )

        if success_teacher is not None and math.isfinite(float(success_teacher)):
            # 成功Amountより下がりすぎる場合だけ戻す。
            # これにより、200Step以降にPrune量がどんどん0へ逃げる問題を抑える。
            min_keep = min(
                max(float(getattr(args, "sparsepcgc_success_amount_min_keep", 0.60)), 0.0),
                1.0,
            )
            target_min = reference.new_tensor(float(success_teacher) * float(min_keep))
            anticollapse_loss = torch.relu(target_min - raw_ratio_t).pow(2)
            total_loss = total_loss + (
                float(getattr(args, "sparsepcgc_success_amount_anticollapse_weight", 0.03))
                * anticollapse_loss
            )

    debug["outcome_where_loss"] = float(where_loss.detach().float().cpu()) if torch.is_tensor(where_loss) else 0.0
    debug["outcome_amount_loss"] = float(amount_loss.detach().float().cpu()) if torch.is_tensor(amount_loss) else 0.0
    debug["outcome_amount_anticollapse_loss"] = (
        float(anticollapse_loss.detach().float().cpu()) if torch.is_tensor(anticollapse_loss) else 0.0
    )
    debug["outcome_imitation_used"] = bool(
        torch.is_tensor(total_loss)
        and float(total_loss.detach().float().cpu()) != 0.0
    )

    return torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0), debug

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

def _slice_full_cloud_canonical_context(
    full_cloud_canonical_context,
    point_idx,
    *,
    device,
):
    """
    full cloud canonical voxel coords を point_idx で切り出し、
    Subtree入力点と1対1対応する subtree_tree 用contextを作る。
    """
    if not isinstance(full_cloud_canonical_context, dict):
        raise RuntimeError("full_cloud_canonical_context is missing.")

    full_coords = full_cloud_canonical_context.get("full_global_voxel_coords", None)
    if full_coords is None:
        full_coords = full_cloud_canonical_context.get("global_voxel_coords", None)

    if not torch.is_tensor(full_coords):
        raise RuntimeError("full cloud canonical global_voxel_coords is missing.")

    if full_coords.ndim != 3 or full_coords.shape[1] != 3:
        raise RuntimeError(
            f"full cloud canonical coords must be [B,3,N], got {tuple(full_coords.shape)}"
        )

    point_idx = point_idx.to(device=full_coords.device, dtype=torch.long)
    subtree_coords = full_coords.index_select(2, point_idx).detach().to(device=device, dtype=torch.long)

    out = {
        "octree_context_scope": "subtree_from_full_cloud_canonical",
        "octree_input_mode": "prebuilt_subtree_tree",
        "canonical_source": "full_cloud_canonical",
        "global_voxel_coords": subtree_coords,
        "subtree_global_voxel_coords": subtree_coords,
        "full_global_voxel_coords": full_coords.detach().to(device=device, dtype=torch.long),
        "full_occupied_voxel_coords": full_coords.detach().to(device=device, dtype=torch.long),
    }

    for key in (
        "global_qs",
        "global_offset",
        "sparsepcgc_voxel_meta",
    ):
        if key in full_cloud_canonical_context:
            out[key] = full_cloud_canonical_context[key]

    return out


def _inject_full_cloud_canonical_into_subtree_metadata(
    *,
    subtree_tree,
    full_octree_context,
    full_cloud_canonical_context,
    point_idx,
    device,
):
    """
    build_selected_group_octree_metadata() が返した metadata に対して、
    voxel座標系だけを full cloud canonical に強制的に差し替える。
    これにより、局所Subtree由来の再量子化を排除する。
    """
    canonical_subtree_context = _slice_full_cloud_canonical_context(
        full_cloud_canonical_context,
        point_idx,
        device=device,
    )

    patched_subtree_tree = dict(subtree_tree or {})
    patched_subtree_tree.update(canonical_subtree_context)

    patched_full_context = dict(full_octree_context or {})
    patched_full_context.update(
        {
            "canonical_source": "full_cloud_canonical",
            # current subtree入力に対応するcoords
            "global_voxel_coords": canonical_subtree_context["global_voxel_coords"],
            # full cloud全体のoccupied coords
            "full_global_voxel_coords": canonical_subtree_context["full_global_voxel_coords"],
            "full_occupied_voxel_coords": canonical_subtree_context["full_occupied_voxel_coords"],
        }
    )

    for key in (
        "global_qs",
        "global_offset",
        "sparsepcgc_voxel_meta",
    ):
        if key in canonical_subtree_context:
            patched_full_context[key] = canonical_subtree_context[key]

    return patched_subtree_tree, patched_full_context

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


def _sparsepcgc_actual_oracle_memory(args):
    if not bool(getattr(args, "sparsepcgc_actual_oracle_use_outcome_memory", True)):
        return None
    memory = getattr(args, "_sparsepcgc_actual_oracle_outcome_memory", None)
    if not isinstance(memory, dict):
        memory = OrderedDict()
        setattr(args, "_sparsepcgc_actual_oracle_outcome_memory", memory)
    return memory


def _sparsepcgc_actual_oracle_transition_key(op, current_code, child_slot, target_code):
    return f"{str(op)}:{int(current_code)}:{int(child_slot)}:{int(target_code)}"


def _sparsepcgc_actual_oracle_pair_key(drop_key, add_key):
    if not drop_key or not add_key:
        return ""
    return f"pair:{drop_key}|{add_key}"


def _sparsepcgc_actual_oracle_memory_bonus(args, key):
    memory = _sparsepcgc_actual_oracle_memory(args)
    if memory is None or not key or key not in memory:
        return 0.0, False, False
    item = memory.get(key, None)
    if not isinstance(item, dict):
        return 0.0, False, False
    ema_percent = float(item.get("ema_percent", 0.0))
    count = int(item.get("count", 0))
    scale = max(float(getattr(args, "sparsepcgc_actual_oracle_memory_score_scale", 0.5)), 1e-6)
    bonus = -ema_percent / scale
    min_count = max(int(getattr(args, "sparsepcgc_actual_oracle_memory_bad_min_count", 2)), 1)
    bad_threshold = float(getattr(args, "sparsepcgc_actual_oracle_memory_bad_skip_percent", 0.0))
    is_bad = (
        bool(getattr(args, "sparsepcgc_actual_oracle_memory_skip_bad", True))
        and count >= min_count
        and ema_percent >= bad_threshold
    )
    return float(bonus), bool(is_bad), True


def _sparsepcgc_actual_oracle_update_memory(args, key, percent):
    memory = _sparsepcgc_actual_oracle_memory(args)
    if memory is None or not key:
        return
    try:
        percent = float(percent)
    except Exception:
        return
    if not math.isfinite(percent):
        return
    alpha = min(max(float(getattr(args, "sparsepcgc_actual_oracle_memory_ema", 0.20)), 1e-4), 1.0)
    old = memory.get(key, None)
    if isinstance(old, dict):
        old_ema = float(old.get("ema_percent", percent))
        count = int(old.get("count", 0)) + 1
        ema = (1.0 - alpha) * old_ema + alpha * percent
    else:
        count = 1
        ema = percent
    memory[key] = {"ema_percent": float(ema), "count": int(count)}
    if isinstance(memory, OrderedDict):
        memory.move_to_end(key)
    max_entries = max(int(getattr(args, "sparsepcgc_actual_oracle_memory_max_entries", 4096)), 128)
    while len(memory) > max_entries:
        try:
            memory.popitem(last=False)
        except Exception:
            break


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


def _sparsepcgc_edit_record_structured_prune_bits(args, unique_count, block_size, drop_ratio):
    if not bool(getattr(args, "sparsepcgc_edit_record_bits_enabled", True)):
        return 0.0
    base_bits = max(float(getattr(args, "sparsepcgc_edit_record_base_bits", 8.0)), 0.0)
    transform_bits = max(
        int(getattr(args, "sparsepcgc_edit_record_structured_prune_bits_min", 32)),
        _ceil_log2_int(max(int(unique_count), 1)) + _ceil_log2_int(max(int(block_size), 2)) + 16,
    )
    raw_bits = float(base_bits + transform_bits)
    return float(raw_bits * sparsepcgc_effective_edit_record_bit_scale(args))


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


def _sparsepcgc_codec_proxy_neighbor_count(coords_n3):
    if coords_n3 is None or coords_n3.numel() <= 0:
        return torch.zeros((0,), device=coords_n3.device if torch.is_tensor(coords_n3) else "cpu", dtype=torch.long)
    offsets = torch.tensor(
        [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ],
        device=coords_n3.device,
        dtype=torch.long,
    )
    query = (coords_n3[:, None, :] + offsets.view(1, -1, 3)).reshape(-1, 3)
    combined = torch.cat([coords_n3, query], dim=0)
    mins = combined.amin(dim=0)
    span = (combined.amax(dim=0) - mins + 1).clamp_min(1)

    def _keys(values):
        shifted = values - mins
        return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]

    occupied_keys = torch.unique(_keys(coords_n3), sorted=True)
    query_keys = _keys(query)
    pos = torch.searchsorted(occupied_keys, query_keys)
    in_bounds = pos < occupied_keys.numel()
    safe_pos = pos.clamp(max=max(int(occupied_keys.numel()) - 1, 0))
    found = in_bounds & (occupied_keys[safe_pos] == query_keys)
    return found.view(coords_n3.shape[0], -1).sum(dim=1).to(dtype=torch.long)


def _sparsepcgc_axis_neighbor_count(coords_n3):
    if coords_n3 is None or coords_n3.numel() <= 0:
        return torch.zeros((0,), device=coords_n3.device if torch.is_tensor(coords_n3) else "cpu", dtype=torch.long)
    offsets = torch.tensor(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        device=coords_n3.device,
        dtype=torch.long,
    )
    query = (coords_n3[:, None, :] + offsets.view(1, -1, 3)).reshape(-1, 3)
    combined = torch.cat([coords_n3, query], dim=0)
    mins = combined.amin(dim=0)
    span = (combined.amax(dim=0) - mins + 1).clamp_min(1)

    def _keys(values):
        shifted = values - mins
        return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]

    occupied_keys = torch.unique(_keys(coords_n3), sorted=True)
    query_keys = _keys(query)
    pos = torch.searchsorted(occupied_keys, query_keys)
    in_bounds = pos < occupied_keys.numel()
    safe_pos = pos.clamp(max=max(int(occupied_keys.numel()) - 1, 0))
    found = in_bounds & (occupied_keys[safe_pos] == query_keys)
    return found.view(coords_n3.shape[0], -1).sum(dim=1).to(dtype=torch.long)


def _sparsepcgc_codec_proxy_profile(unique_coords, args):
    """
    Lightweight codec-aware proxy used only for greedy teacher ranking.

    SparsePCGC codes occupancy labels predicted from multiscale sparse-tensor
    context, so this proxy estimates per-node occupancy NLL from local context
    buckets instead of global 8-bit child-pattern frequency alone.  The actual
    encoder remains the final accept/reject gate.
    """
    if unique_coords is None or not torch.is_tensor(unique_coords) or unique_coords.numel() <= 0:
        device = unique_coords.device if torch.is_tensor(unique_coords) else torch.device("cpu")
        return {
            "enabled": False,
            "reason": "empty",
            "base_proxy_bits": 0.0,
            "leaf_occupied_bits": torch.zeros((0,), device=device, dtype=torch.float32),
            "leaf_occ_prob": torch.zeros((0,), device=device, dtype=torch.float32),
            "leaf_add_delta_bits": None,
            "leaf_empty_rate": None,
            "leaf_occupied_rate": None,
            "low_prob_occupied_count": 0,
            "high_rate_mppov_count": 0,
            "single_child_chain_count": 0,
            "context_pattern_candidate_count": 0,
        }

    coords = torch.unique(unique_coords.detach().to(dtype=torch.long), dim=0, sorted=True)
    device = coords.device
    smoothing = max(float(getattr(args, "sparsepcgc_codec_proxy_smoothing", 1.0)), 1e-6)
    low_prob_threshold = min(
        max(float(getattr(args, "sparsepcgc_proxy_low_prob_threshold", 0.15)), 1e-6),
        1.0 - 1e-6,
    )
    high_rate_threshold = max(float(getattr(args, "sparsepcgc_proxy_high_rate_bit_threshold", 2.0)), 0.0)
    max_levels = max(int(getattr(args, "sparsepcgc_codec_proxy_max_levels", 16)), 1)

    total_bits = 0.0
    single_child_chain_count = 0
    context_pattern_candidate_count = 0
    leaf_occupied_bits = torch.zeros((coords.shape[0],), device=device, dtype=torch.float32)
    leaf_occ_prob = torch.ones((coords.shape[0],), device=device, dtype=torch.float32)
    leaf_empty_rate = None
    leaf_occupied_rate = None
    leaf_add_delta_bits = None

    current = coords
    eps = 1e-12
    for level in range(max_levels):
        if current.numel() <= 0 or int(current.shape[0]) <= 1:
            break
        parent_coords = torch.div(current, 2, rounding_mode="floor")
        unique_parents, parent_inverse = torch.unique(
            parent_coords,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        parent_count = int(unique_parents.shape[0])
        if parent_count <= 0:
            break
        child_slot = (
            (current[:, 0] & 1)
            + 2 * (current[:, 1] & 1)
            + 4 * (current[:, 2] & 1)
        ).to(dtype=torch.long)
        occupancy = torch.zeros((parent_count, 8), device=device, dtype=torch.bool)
        occupancy[parent_inverse, child_slot] = True
        child_count = occupancy.sum(dim=1).to(dtype=torch.long)
        single_child_chain_count += int((child_count == 1).sum().detach().cpu())
        context_pattern_candidate_count += int(parent_count)

        parent_slot = (
            (unique_parents[:, 0] & 1)
            + 2 * (unique_parents[:, 1] & 1)
            + 4 * (unique_parents[:, 2] & 1)
        ).to(dtype=torch.long)
        neighbor_count = _sparsepcgc_codec_proxy_neighbor_count(unique_parents)
        neighbor_bucket = neighbor_count.clamp(0, 6)
        child_bucket = child_count.clamp(0, 8)
        context_id = (
            int(level) * 4096
            + parent_slot * 256
            + neighbor_bucket * 32
            + child_bucket
        ).to(dtype=torch.long)

        unique_context, context_inverse = torch.unique(
            context_id,
            sorted=True,
            return_inverse=True,
        )
        context_count = torch.bincount(
            context_inverse,
            minlength=int(unique_context.numel()),
        ).to(device=device, dtype=torch.float32)
        context_occ = torch.zeros(
            (int(unique_context.numel()), 8),
            device=device,
            dtype=torch.float32,
        )
        context_occ.scatter_add_(
            0,
            context_inverse.view(-1, 1).expand(-1, 8),
            occupancy.to(dtype=torch.float32),
        )
        parent_slot_prob = (
            context_occ.index_select(0, context_inverse) + float(smoothing)
        ) / (
            context_count.index_select(0, context_inverse).view(-1, 1)
            + 2.0 * float(smoothing)
        )
        parent_slot_prob = parent_slot_prob.clamp(min=eps, max=1.0 - eps)

        occupied_rate = -torch.log2(parent_slot_prob.clamp_min(eps))
        empty_rate = -torch.log2((1.0 - parent_slot_prob).clamp_min(eps))
        level_rate = torch.where(occupancy, occupied_rate, empty_rate).sum()
        total_bits += float(level_rate.detach().cpu())

        if level == 0:
            leaf_occupied_bits = occupied_rate[parent_inverse, child_slot].detach().clone()
            leaf_occ_prob = parent_slot_prob[parent_inverse, child_slot].detach().clone()
            leaf_empty_rate = empty_rate.detach().clone()
            leaf_occupied_rate = occupied_rate.detach().clone()
            leaf_add_delta_bits = (occupied_rate - empty_rate).detach().clone()

        current = unique_parents

    low_prob_occupied_count = int((leaf_occ_prob < float(low_prob_threshold)).sum().detach().cpu())
    high_rate_mppov_count = int((leaf_occupied_bits > float(high_rate_threshold)).sum().detach().cpu())
    return {
        "enabled": True,
        "reason": "ok",
        "base_proxy_bits": float(total_bits),
        "leaf_occupied_bits": leaf_occupied_bits,
        "leaf_occ_prob": leaf_occ_prob,
        "leaf_add_delta_bits": leaf_add_delta_bits,
        "leaf_empty_rate": leaf_empty_rate,
        "leaf_occupied_rate": leaf_occupied_rate,
        "low_prob_occupied_count": int(low_prob_occupied_count),
        "high_rate_mppov_count": int(high_rate_mppov_count),
        "single_child_chain_count": int(single_child_chain_count),
        "context_pattern_candidate_count": int(context_pattern_candidate_count),
    }


def _sparsepcgc_codec_proxy_bits(unique_coords, args):
    return float(_sparsepcgc_codec_proxy_profile(unique_coords, args).get("base_proxy_bits", 0.0))


def _sparsepcgc_proxy_delta_percent(candidate_coords, args, base_proxy_bits):
    base_proxy_bits = max(abs(float(base_proxy_bits)), 1.0)
    cand_bits = _sparsepcgc_codec_proxy_bits(candidate_coords, args)
    return float(cand_bits), float(100.0 * (cand_bits - base_proxy_bits) / base_proxy_bits)


def _sparsepcgc_coord_key_set(coords_n3):
    if coords_n3 is None or not torch.is_tensor(coords_n3) or coords_n3.numel() <= 0:
        return set()
    coords_cpu = torch.unique(coords_n3.detach().to(dtype=torch.long), dim=0, sorted=True).cpu()
    return {tuple(int(v) for v in row) for row in coords_cpu.tolist()}


def _sparsepcgc_coords_to_n3(coords):
    if coords is None or not torch.is_tensor(coords) or coords.numel() <= 0:
        return None
    if coords.ndim == 3 and coords.shape[1] == 3:
        return coords[0].transpose(0, 1).contiguous()
    if coords.ndim == 3 and coords.shape[-1] == 3:
        return coords[0].contiguous()
    if coords.ndim == 2 and coords.shape[0] == 3:
        return coords.transpose(0, 1).contiguous()
    if coords.ndim == 2 and coords.shape[-1] == 3:
        return coords.contiguous()
    return None


def _sparsepcgc_fast_diag_global_drop_set(full_coords_b3n, args):
    if (
        full_coords_b3n is None
        or not torch.is_tensor(full_coords_b3n)
        or full_coords_b3n.numel() <= 0
    ):
        return set(), {"available": False, "reason": "coords_missing"}
    full_coords = _sparsepcgc_coords_to_n3(full_coords_b3n)
    if full_coords is None:
        return set(), {"available": False, "reason": f"invalid_shape={tuple(full_coords_b3n.shape)}"}
    full_coords = torch.unique(full_coords.detach().to(dtype=torch.long), dim=0, sorted=True)
    if int(full_coords.shape[0]) <= 8:
        return set(), {"available": False, "reason": "too_few_voxels", "full_count": int(full_coords.shape[0])}

    threshold = max(int(getattr(args, "sparsepcgc_fast_diagnostic_neighbor_threshold", 3)), 1)
    axis_neigh = _sparsepcgc_axis_neighbor_count(full_coords).to(device=full_coords.device, dtype=torch.long)
    drop_coords = full_coords[axis_neigh < int(threshold)]
    drop_set = _sparsepcgc_coord_key_set(drop_coords)
    return drop_set, {
        "available": True,
        "reason": "ok",
        "threshold": int(threshold),
        "full_count": int(full_coords.shape[0]),
        "global_drop_count": int(len(drop_set)),
        "global_drop_ratio": float(len(drop_set)) / max(float(full_coords.shape[0]), 1.0),
    }


def _sparsepcgc_fast_diag_local_count(coords_n3, global_drop_set):
    if not global_drop_set or coords_n3 is None or not torch.is_tensor(coords_n3) or coords_n3.numel() <= 0:
        return 0, 0.0
    coords_set = _sparsepcgc_coord_key_set(coords_n3)
    if not coords_set:
        return 0, 0.0
    local_count = sum(1 for key in coords_set if key in global_drop_set)
    return int(local_count), float(local_count) / max(float(len(coords_set)), 1.0)


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


def _sparsepcgc_subtree_leaf_pattern_potential(coords_n3, args):
    """
    Cheap pre-oracle score for choosing a train Subtree.

    The actual oracle is expensive, so before running it we rank Subtrees by
    whether their leaf-level occupancy codes have plausible Add/Prune
    transitions toward more common codes.  This does not decide the edit; the
    actual SparsePCGC oracle still accepts/rejects candidates later.
    """
    if coords_n3 is None or coords_n3.numel() <= 0:
        return 0.0, {"reason": "empty"}

    coords_n3 = coords_n3.detach().to(dtype=torch.long)
    if coords_n3.ndim != 2 or coords_n3.shape[-1] != 3:
        return 0.0, {"reason": f"invalid_shape={tuple(coords_n3.shape)}"}

    unique_coords = torch.unique(coords_n3, dim=0, sorted=True)
    unique_count = int(unique_coords.shape[0])
    if unique_count <= 1:
        return 0.0, {"reason": "too_few_voxels", "unique": unique_count}

    device = unique_coords.device
    parent_coords = torch.div(unique_coords, 2, rounding_mode="floor")
    unique_parents, parent_inverse = torch.unique(
        parent_coords,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    parent_count = int(unique_parents.shape[0])
    if parent_count <= 0:
        return 0.0, {"reason": "no_parent", "unique": unique_count}

    child_slot = (
        (unique_coords[:, 0] & 1)
        + 2 * (unique_coords[:, 1] & 1)
        + 4 * (unique_coords[:, 2] & 1)
    ).to(dtype=torch.long)
    occupancy = torch.zeros((parent_count, 8), device=device, dtype=torch.bool)
    occupancy[parent_inverse, child_slot] = True

    pattern_weights = (2 ** torch.arange(8, device=device, dtype=torch.long)).view(1, 8)
    parent_code = (occupancy.to(dtype=torch.long) * pattern_weights).sum(dim=1).clamp(0, 255)
    smoothing = max(float(getattr(args, "leaf_pattern_candidate_smoothing", 1.0)), 0.0)
    code_hist = torch.bincount(parent_code, minlength=256).to(device=device, dtype=torch.float32)
    code_prob = code_hist + float(smoothing)
    code_prob = code_prob / code_prob.sum().clamp_min(torch.finfo(torch.float32).eps)
    code_nll = -torch.log2(code_prob.clamp_min(torch.finfo(torch.float32).eps))

    topk = max(int(getattr(args, "sparsepcgc_subtree_potential_candidate_topk", 4)), 1)
    current_code = parent_code.index_select(0, parent_inverse).clamp(0, 255)
    parent_child_count = occupancy.sum(dim=1).to(dtype=torch.float32).index_select(0, parent_inverse)
    bit_current = (1 << child_slot.clamp(0, 7)).to(device=device, dtype=torch.long)
    delete_code = torch.bitwise_and(current_code, torch.bitwise_not(bit_current)).clamp(0, 255)
    delete_gain = code_nll.index_select(0, current_code) - code_nll.index_select(0, delete_code)
    min_children_after = max(int(getattr(args, "leaf_pattern_delete_min_children_after", 1)), 0)
    delete_valid = (parent_child_count - 1.0) >= float(min_children_after)
    drop_values = torch.relu(delete_gain[delete_valid])
    if drop_values.numel() > 0:
        drop_score = float(torch.topk(drop_values, k=min(topk, int(drop_values.numel()))).values.sum().detach().cpu())
    else:
        drop_score = 0.0

    empty_parent_idx, empty_slot = (~occupancy).nonzero(as_tuple=True)
    add_score = 0.0
    if empty_parent_idx.numel() > 0:
        add_bit = (1 << empty_slot.clamp(0, 7)).to(device=device, dtype=torch.long)
        add_current = parent_code.index_select(0, empty_parent_idx).clamp(0, 255)
        add_code = torch.bitwise_or(add_current, add_bit).clamp(0, 255)
        add_gain = code_nll.index_select(0, add_current) - code_nll.index_select(0, add_code)
        add_values = torch.relu(add_gain)
        if add_values.numel() > 0:
            add_score = float(torch.topk(add_values, k=min(topk, int(add_values.numel()))).values.sum().detach().cpu())

    neigh = _sparsepcgc_codec_proxy_neighbor_count(unique_coords).to(device=device, dtype=torch.float32)
    density_score = neigh + parent_child_count * 0.5
    macro_ratio = min(
        max(float(getattr(args, "sparsepcgc_subtree_potential_macro_ratio", 0.20)), 0.0),
        0.80,
    )
    macro_drop_n = min(max(int(math.ceil(float(unique_count) * macro_ratio)), 1), unique_count - 1)
    density_order = torch.argsort(density_score, descending=False)
    low_density_idx = density_order[:macro_drop_n]
    density_mean = density_score.mean()
    low_density_mean = density_score.index_select(0, low_density_idx).mean()
    macro_density_score = float(
        torch.relu(density_mean - low_density_mean).detach().cpu()
    ) * math.sqrt(float(macro_drop_n))
    try:
        proxy_bits = float(_sparsepcgc_codec_proxy_bits(unique_coords, args))
    except Exception:
        proxy_bits = 0.0
    proxy_rate_score = proxy_bits / math.sqrt(max(float(unique_count), 1.0))

    drop_weight = max(float(getattr(args, "sparsepcgc_subtree_potential_drop_weight", 1.0)), 0.0)
    add_weight = max(float(getattr(args, "sparsepcgc_subtree_potential_add_weight", 1.0)), 0.0)
    macro_weight = max(float(getattr(args, "sparsepcgc_subtree_potential_macro_weight", 1.0)), 0.0)
    proxy_rate_weight = max(float(getattr(args, "sparsepcgc_subtree_potential_proxy_rate_weight", 0.02)), 0.0)
    size_weight = max(float(getattr(args, "sparsepcgc_subtree_potential_size_weight", 0.02)), 0.0)
    efficiency_weight = max(float(getattr(args, "sparsepcgc_subtree_potential_efficiency_weight", 2.0)), 0.0)
    small_tree_weight = max(float(getattr(args, "sparsepcgc_subtree_potential_small_tree_weight", 0.25)), 0.0)
    base_score = (
        drop_weight * drop_score
        + add_weight * add_score
        + macro_weight * macro_density_score
        + proxy_rate_weight * proxy_rate_score
    )
    edit_efficiency = base_score / math.sqrt(max(float(unique_count), 1.0))
    small_tree_score = 1.0 / math.sqrt(max(float(unique_count), 1.0))
    score = (
        base_score
        + size_weight * math.log1p(float(unique_count))
        + efficiency_weight * edit_efficiency
        + small_tree_weight * small_tree_score
    )
    return float(score), {
        "reason": "ok",
        "unique": unique_count,
        "parents": parent_count,
        "drop_score": float(drop_score),
        "add_score": float(add_score),
        "macro_density_score": float(macro_density_score),
        "proxy_rate_score": float(proxy_rate_score),
        "proxy_bits": float(proxy_bits),
        "edit_efficiency": float(edit_efficiency),
    }


def _select_sparsepcgc_potential_subtree_key(
    candidate_groups,
    candidate_subtree_keys,
    full_cloud_canonical_context,
    args,
    global_step,
    cache_key,
    model=None,
):
    try:
        setattr(args, "_current_sparsepcgc_proposal_terms_by_key", {})
        setattr(args, "_current_sparsepcgc_proposal_selection_meta", {"enabled": False})
    except Exception:
        pass
    if not bool(getattr(args, "sparsepcgc_subtree_potential_priority", True)):
        return None, {"enabled": False, "reason": "disabled"}
    compress_key = str(getattr(args, "compress", "")).strip().lower().replace("_", "").replace("-", "")
    if compress_key != "sparsepcgc":
        return None, {"enabled": False, "reason": "not_sparsepcgc"}
    if not candidate_groups:
        return None, {"enabled": True, "reason": "no_groups"}
    if not isinstance(full_cloud_canonical_context, dict):
        return None, {"enabled": True, "reason": "context_missing"}

    full_coords = full_cloud_canonical_context.get("full_global_voxel_coords", None)
    if full_coords is None:
        full_coords = full_cloud_canonical_context.get("global_voxel_coords", None)
    if not torch.is_tensor(full_coords) or full_coords.ndim != 3 or full_coords.shape[1] != 3:
        return None, {"enabled": True, "reason": "coords_missing"}
    fast_diag_drop_set = None
    fast_diag_global = {
        "global_drop_count": 0,
        "global_drop_ratio": 0.0,
    }
    fast_diag_weight = max(float(getattr(args, "sparsepcgc_subtree_potential_fast_diag_weight", 50.0)), 0.0)
    fast_diag_min_count = max(int(getattr(args, "sparsepcgc_subtree_potential_fast_diag_min_count", 1)), 0)

    group_by_key = {int(key): point_idx for key, point_idx in candidate_groups}
    pool_keys = [int(key) for key in candidate_subtree_keys.detach().cpu().tolist()]
    pool_keys = [key for key in pool_keys if key in group_by_key]
    if not pool_keys:
        return None, {"enabled": True, "reason": "empty_pool"}

    max_scan = max(int(getattr(args, "sparsepcgc_subtree_potential_max_scan", 256)), 1)
    if len(pool_keys) > max_scan:
        seed_text = f"{cache_key or ''}|potential_scan|step={int(global_step)}|seed={int(getattr(args, 'seed', 0))}"
        seed = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16], 16)
        pool_keys = sorted(
            pool_keys,
            key=lambda key: ((int(key) * 2654435761 + seed) & 0x7FFFFFFF),
        )[:max_scan]

    score_map = {}
    potential_cache_enabled = bool(
        _episode_input_common_cache_enabled(args)
        and getattr(args, "episode_input_subtree_potential_cache", True)
        and cache_key
    )
    potential_cache_key = _subtree_potential_input_common_cache_key(
        cache_key,
        int(full_cloud_canonical_context.get("global_depth", 0) or 0),
    )
    if potential_cache_enabled:
        cached_payload = _episode_input_common_cache_fetch(
            args,
            potential_cache_key,
            device=None,
            section="subtree_potential_scores",
        )
        if isinstance(cached_payload, dict):
            cached_scores = cached_payload.get("score_map", None)
            cached_fast_diag_global = cached_payload.get("fast_diag_global", None)
            if isinstance(cached_scores, dict):
                score_map = dict(cached_scores)
            if isinstance(cached_fast_diag_global, dict):
                fast_diag_global = dict(cached_fast_diag_global)

    missing_keys = [int(key) for key in pool_keys if int(key) not in score_map]
    if missing_keys:
        fast_diag_drop_set, fast_diag_global = _sparsepcgc_fast_diag_global_drop_set(full_coords, args)
        with torch.no_grad():
            for key in missing_keys:
                point_idx = group_by_key.get(int(key), None)
                if not torch.is_tensor(point_idx) or point_idx.numel() <= 0:
                    continue
                idx = point_idx.to(device=full_coords.device, dtype=torch.long)
                if int(idx.numel()) <= 1:
                    continue
                coords_n3 = full_coords[0].index_select(1, idx).transpose(0, 1).contiguous()
                score, detail = _sparsepcgc_subtree_leaf_pattern_potential(coords_n3, args)
                fast_local_count, fast_local_ratio = _sparsepcgc_fast_diag_local_count(coords_n3, fast_diag_drop_set)
                if fast_local_count >= fast_diag_min_count and fast_diag_weight > 0.0:
                    fast_score = fast_diag_weight * float(fast_local_count) / math.sqrt(max(float(coords_n3.shape[0]), 1.0))
                    score += float(fast_score)
                else:
                    fast_score = 0.0
                subtree_memory_bonus = 0.0
                subtree_memory_count = 0
                subtree_memory = _sparsepcgc_subtree_outcome_lookup(args, cache_key, key)
                if isinstance(subtree_memory, dict):
                    subtree_memory_count = int(subtree_memory.get("count", 0) or 0)
                    subtree_memory_bonus = (
                        float(getattr(args, "sparsepcgc_subtree_outcome_selector_weight", 20.0))
                        * float(subtree_memory.get("score_ema", 0.0) or 0.0)
                    )
                    score += float(subtree_memory_bonus)
                if isinstance(detail, dict):
                    detail = dict(detail)
                    detail["fast_diag_local_count"] = int(fast_local_count)
                    detail["fast_diag_local_ratio"] = float(fast_local_ratio)
                    detail["fast_diag_score"] = float(fast_score)
                    detail["fast_diag_global_drop_count"] = int(fast_diag_global.get("global_drop_count", 0) or 0)
                    detail["fast_diag_global_drop_ratio"] = float(fast_diag_global.get("global_drop_ratio", 0.0) or 0.0)
                    detail["subtree_outcome_memory_bonus"] = float(subtree_memory_bonus)
                    detail["subtree_outcome_memory_count"] = int(subtree_memory_count)
                score_map[int(key)] = {
                    "score": float(score),
                    "detail": detail,
                }
        if potential_cache_enabled:
            _episode_input_common_cache_store(
                args,
                potential_cache_key,
                {
                    "score_map": dict(score_map),
                    "fast_diag_global": dict(fast_diag_global),
                },
            )

    scored = []
    for key in pool_keys:
        cached_item = score_map.get(int(key), None)
        if not isinstance(cached_item, dict):
            continue
        score_value = float(cached_item.get("score", 0.0) or 0.0)
        detail = cached_item.get("detail", None)
        old_memory_bonus = (
            float(detail.get("subtree_outcome_memory_bonus", 0.0) or 0.0)
            if isinstance(detail, dict)
            else 0.0
        )
        current_memory_bonus = 0.0
        current_memory_count = 0
        subtree_memory = _sparsepcgc_subtree_outcome_lookup(args, cache_key, key)
        if isinstance(subtree_memory, dict):
            current_memory_count = int(subtree_memory.get("count", 0) or 0)
            current_memory_bonus = (
                float(getattr(args, "sparsepcgc_subtree_outcome_selector_weight", 20.0))
                * float(subtree_memory.get("score_ema", 0.0) or 0.0)
            )
        score_value = float(score_value) - float(old_memory_bonus) + float(current_memory_bonus)
        if isinstance(detail, dict):
            detail = dict(detail)
            detail["subtree_outcome_memory_bonus"] = float(current_memory_bonus)
            detail["subtree_outcome_memory_count"] = int(current_memory_count)
        scored.append(
            (
                float(score_value),
                int(key),
                detail,
            )
        )

    if not scored:
        return None, {"enabled": True, "reason": "no_scored_groups", "pool": len(pool_keys)}

    scored.sort(key=lambda item: item[0], reverse=True)

    proposal_selector_enabled = bool(
        getattr(args, "sparsepcgc_algorithmic_proposal_selector", True)
        and not getattr(args, "sparsepcgc_legacy_direct_actuator_train", False)
        and model is not None
    )
    if proposal_selector_enabled:
        try:
            base_model = _unwrap_train_model(model)
            score_fn = getattr(base_model, "score_algorithmic_proposal_subtrees", None)
        except Exception:
            base_model = None
            score_fn = None
        if callable(score_fn):
            proposal_topk = min(
                max(int(getattr(args, "sparsepcgc_proposal_topk_subtrees", 5)), 1),
                len(scored),
            )
            proposal_pool = scored[:proposal_topk]
            feature_rows = []
            pool_keys_for_terms = []
            for rank_idx, (score_value, key_value, detail_value) in enumerate(proposal_pool):
                point_idx = group_by_key.get(int(key_value), None)
                point_count = int(point_idx.numel()) if torch.is_tensor(point_idx) else 0
                detail_dict = detail_value if isinstance(detail_value, dict) else {}
                memory_count = float(detail_dict.get("subtree_outcome_memory_count", 0) or 0)
                feature_rows.append(
                    [
                        float(score_value) / 100.0,
                        math.log1p(max(float(point_count), 0.0)) / 10.0,
                        float(detail_dict.get("drop_score", 0.0) or 0.0) / 100.0,
                        float(detail_dict.get("add_score", 0.0) or 0.0) / 100.0,
                        float(detail_dict.get("macro_density_score", 0.0) or 0.0) / 100.0,
                        float(detail_dict.get("proxy_rate_score", 0.0) or 0.0) / 100.0,
                        float(detail_dict.get("fast_diag_local_ratio", 0.0) or 0.0),
                        float(detail_dict.get("fast_diag_score", 0.0) or 0.0) / 100.0,
                        float(detail_dict.get("subtree_outcome_memory_bonus", 0.0) or 0.0) / 100.0,
                        math.log1p(max(memory_count, 0.0)) / 10.0,
                        float(rank_idx) / max(float(proposal_topk - 1), 1.0),
                        1.0,
                    ]
                )
                pool_keys_for_terms.append(int(key_value))
            try:
                selector_device = next(base_model.parameters()).device
            except Exception:
                selector_device = full_coords.device
            feature_tensor = torch.tensor(
                feature_rows,
                device=selector_device,
                dtype=torch.float32,
            )
            selector_out = score_fn(feature_tensor)
            predicted_delta = selector_out["subtree_predicted_delta"]
            select_logit = selector_out["subtree_select_logit"]
            threshold = float(getattr(args, "sparsepcgc_proposal_accept_threshold", 0.0))
            max_apply = max(int(getattr(args, "sparsepcgc_proposal_max_apply_subtrees", 3)), 1)
            if not bool(getattr(args, "sparsepcgc_multi_subtree_train", False)):
                max_apply = 1
            order = torch.argsort(predicted_delta.detach(), dim=0, descending=False)
            accepted_indices = []
            for order_item in order.detach().cpu().tolist():
                idx = int(order_item)
                pred_value = float(predicted_delta.detach().flatten()[idx].cpu())
                if pred_value <= threshold:
                    accepted_indices.append(idx)
                if len(accepted_indices) >= max_apply:
                    break
            forced_probe = False
            if not accepted_indices and int(order.numel()) > 0:
                # Training still needs a concrete subtree to evaluate and teach no-op.
                accepted_indices = [int(order.detach().cpu().flatten()[0].item())]
                forced_probe = True

            selected_keys = [int(pool_keys_for_terms[idx]) for idx in accepted_indices]
            terms_by_key = {}
            for idx, key_value in enumerate(pool_keys_for_terms):
                terms_by_key[int(key_value)] = {
                    "subtree_select_logit": select_logit[idx],
                    "subtree_predicted_delta": predicted_delta[idx],
                    "amount_bin_logits": selector_out["amount_bin_logits"][idx],
                    "amount_residual_raw": selector_out["amount_residual_raw"][idx],
                    "predicted_delta_per_amount": selector_out["predicted_delta_per_amount"][idx],
                    "feature_tensor": feature_tensor[idx],
                    "pool_rank": int(idx),
                    "heuristic_score": float(proposal_pool[idx][0]),
                }
            setattr(args, "_current_sparsepcgc_proposal_terms_by_key", terms_by_key)
            setattr(
                args,
                "_current_sparsepcgc_proposal_selection_meta",
                {
                    "enabled": True,
                    "pool_count": int(len(proposal_pool)),
                    "selected_count": int(len(selected_keys)),
                    "selected_keys": ",".join(str(key) for key in selected_keys),
                    "forced_probe": bool(forced_probe),
                    "threshold": float(threshold),
                    "best_predicted_delta": float(predicted_delta.detach().flatten()[int(order[0])].cpu())
                    if int(order.numel()) > 0
                    else float("nan"),
                },
            )
            if selected_keys:
                selected = candidate_subtree_keys.new_tensor(selected_keys, dtype=candidate_subtree_keys.dtype)
                first_key = int(selected_keys[0])
                first_idx = pool_keys_for_terms.index(first_key)
                first_detail = proposal_pool[first_idx][2] if isinstance(proposal_pool[first_idx][2], dict) else {}
                meta = {
                    "enabled": True,
                    "reason": "network_proposal_selector",
                    "proposal_selector": True,
                    "proposal_pool_count": int(len(proposal_pool)),
                    "proposal_selected_count": int(len(selected_keys)),
                    "proposal_selected_keys": ",".join(str(key) for key in selected_keys),
                    "proposal_forced_probe": bool(forced_probe),
                    "proposal_best_predicted_delta": float(
                        predicted_delta.detach().flatten()[int(order[0])].cpu()
                    ) if int(order.numel()) > 0 else float("nan"),
                    "pool": len(pool_keys),
                    "scored": len(scored),
                    "rank": int(first_idx),
                    "score": float(proposal_pool[first_idx][0]),
                    "best_score": float(scored[0][0]),
                    "key": int(first_key),
                    "random": False,
                    "drop_score": float(first_detail.get("drop_score", 0.0)) if isinstance(first_detail, dict) else 0.0,
                    "add_score": float(first_detail.get("add_score", 0.0)) if isinstance(first_detail, dict) else 0.0,
                    "subtree_outcome_memory_bonus": float(
                        first_detail.get("subtree_outcome_memory_bonus", 0.0) or 0.0
                    ) if isinstance(first_detail, dict) else 0.0,
                    "subtree_outcome_memory_count": int(
                        first_detail.get("subtree_outcome_memory_count", 0) or 0
                    ) if isinstance(first_detail, dict) else 0,
                }
                return selected, meta

    # ============================================================
    # Multi-Subtree top-k selection
    # ============================================================
    # 既に計算したpotential scoreを使い、追加のactual評価なしで上位K個を選ぶ。
    # これによりSubtree選択自体の時間はほぼ増やさない。
    # ただし、選んだSubtreeを実際にForward/Lossする時間はKに応じて増える。
    # ============================================================
    if bool(getattr(args, "sparsepcgc_multi_subtree_train", False)):
        multi_k = max(int(getattr(args, "sparsepcgc_multi_subtree_topk", 3)), 1)
        max_total_points = max(
            int(getattr(args, "sparsepcgc_multi_subtree_max_total_points", 8192)),
            0,
        )

        selected_items = []
        selected_total_points = 0

        for score_value, key_value, detail_value in scored:
            point_idx = group_by_key.get(int(key_value), None)
            point_count = int(point_idx.numel()) if torch.is_tensor(point_idx) else 0

            if point_count <= 0:
                continue

            # 計算時間増加を抑えるため、選択Subtreeの総点数を制限する。
            # ただし1個も選ばれていない場合は、最大候補を必ず入れる。
            if (
                max_total_points > 0
                and selected_items
                and selected_total_points + point_count > max_total_points
            ):
                continue

            selected_items.append((float(score_value), int(key_value), detail_value, int(point_count)))
            selected_total_points += int(point_count)

            if len(selected_items) >= multi_k:
                break

        if not selected_items:
            score_value, key_value, detail_value = scored[0]
            point_idx = group_by_key.get(int(key_value), None)
            point_count = int(point_idx.numel()) if torch.is_tensor(point_idx) else 0
            selected_items = [(float(score_value), int(key_value), detail_value, int(point_count))]
            selected_total_points = int(point_count)

        selected_keys = [int(item[1]) for item in selected_items]
        selected = candidate_subtree_keys.new_tensor(
            selected_keys,
            dtype=candidate_subtree_keys.dtype,
        )

        first_detail = selected_items[0][2] if isinstance(selected_items[0][2], dict) else {}
        meta = {
            "enabled": True,
            "reason": "selected_topk",
            "multi_subtree": True,
            "requested_topk": int(multi_k),
            "selected_count": int(len(selected_items)),
            "selected_keys": ",".join(str(k) for k in selected_keys),
            "selected_total_points": int(selected_total_points),
            "pool": len(pool_keys),
            "scored": len(scored),
            "rank": 0,
            "score": float(selected_items[0][0]),
            "best_score": float(scored[0][0]),
            "key": int(selected_items[0][1]),
            "random": False,
            "drop_score": float(first_detail.get("drop_score", 0.0)) if isinstance(first_detail, dict) else 0.0,
            "add_score": float(first_detail.get("add_score", 0.0)) if isinstance(first_detail, dict) else 0.0,
            "macro_density_score": float(first_detail.get("macro_density_score", 0.0)) if isinstance(first_detail, dict) else 0.0,
            "proxy_rate_score": float(first_detail.get("proxy_rate_score", 0.0)) if isinstance(first_detail, dict) else 0.0,
            "proxy_bits": float(first_detail.get("proxy_bits", 0.0)) if isinstance(first_detail, dict) else 0.0,
            "fast_diag_local_count": int(first_detail.get("fast_diag_local_count", 0) or 0) if isinstance(first_detail, dict) else 0,
            "fast_diag_local_ratio": float(first_detail.get("fast_diag_local_ratio", 0.0) or 0.0) if isinstance(first_detail, dict) else 0.0,
            "fast_diag_score": float(first_detail.get("fast_diag_score", 0.0) or 0.0) if isinstance(first_detail, dict) else 0.0,
            "fast_diag_global_drop_count": int(fast_diag_global.get("global_drop_count", 0) or 0),
            "fast_diag_global_drop_ratio": float(fast_diag_global.get("global_drop_ratio", 0.0) or 0.0),
            "subtree_outcome_memory_bonus": float(first_detail.get("subtree_outcome_memory_bonus", 0.0) or 0.0) if isinstance(first_detail, dict) else 0.0,
            "subtree_outcome_memory_count": int(first_detail.get("subtree_outcome_memory_count", 0) or 0) if isinstance(first_detail, dict) else 0,
        }
        return selected, meta
    random_mix = min(max(float(getattr(args, "sparsepcgc_subtree_potential_random_mix", 0.05)), 0.0), 1.0)
    seed_text = f"{cache_key or ''}|potential_pick|step={int(global_step)}|seed={int(getattr(args, 'seed', 0))}"
    seed_value = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    use_random = random_mix > 0.0 and ((seed_value % 10000) / 10000.0) < random_mix
    if use_random:
        chosen_rank = seed_value % len(scored)
    else:
        topk = min(max(int(getattr(args, "sparsepcgc_subtree_potential_topk", 4)), 1), len(scored))
        if isinstance(scored[0][2], dict) and int(scored[0][2].get("fast_diag_local_count", 0) or 0) > 0:
            topk = 1
        chosen_rank = seed_value % topk

    chosen_score, chosen_key, chosen_detail = scored[chosen_rank]
    selected = candidate_subtree_keys.new_tensor([chosen_key], dtype=candidate_subtree_keys.dtype)
    meta = {
        "enabled": True,
        "reason": "selected",
        "pool": len(pool_keys),
        "scored": len(scored),
        "rank": int(chosen_rank),
        "score": float(chosen_score),
        "best_score": float(scored[0][0]),
        "key": int(chosen_key),
        "random": bool(use_random),
        "drop_score": float(chosen_detail.get("drop_score", 0.0)) if isinstance(chosen_detail, dict) else 0.0,
        "add_score": float(chosen_detail.get("add_score", 0.0)) if isinstance(chosen_detail, dict) else 0.0,
        "macro_density_score": (
            float(chosen_detail.get("macro_density_score", 0.0)) if isinstance(chosen_detail, dict) else 0.0
        ),
        "proxy_rate_score": (
            float(chosen_detail.get("proxy_rate_score", 0.0)) if isinstance(chosen_detail, dict) else 0.0
        ),
        "proxy_bits": float(chosen_detail.get("proxy_bits", 0.0)) if isinstance(chosen_detail, dict) else 0.0,
        "fast_diag_local_count": (
            int(chosen_detail.get("fast_diag_local_count", 0) or 0) if isinstance(chosen_detail, dict) else 0
        ),
        "fast_diag_local_ratio": (
            float(chosen_detail.get("fast_diag_local_ratio", 0.0) or 0.0) if isinstance(chosen_detail, dict) else 0.0
        ),
        "fast_diag_score": (
            float(chosen_detail.get("fast_diag_score", 0.0) or 0.0) if isinstance(chosen_detail, dict) else 0.0
        ),
        "fast_diag_global_drop_count": int(fast_diag_global.get("global_drop_count", 0) or 0),
        "fast_diag_global_drop_ratio": float(fast_diag_global.get("global_drop_ratio", 0.0) or 0.0),
        "subtree_outcome_memory_bonus": (
            float(chosen_detail.get("subtree_outcome_memory_bonus", 0.0) or 0.0) if isinstance(chosen_detail, dict) else 0.0
        ),
        "subtree_outcome_memory_count": (
            int(chosen_detail.get("subtree_outcome_memory_count", 0) or 0) if isinstance(chosen_detail, dict) else 0
        ),
    }
    return selected, meta


def _sparsepcgc_actual_oracle_candidate_indices(coords_n3, args, global_step, max_candidates, proxy_profile=None):
    if coords_n3.numel() <= 0 or int(max_candidates) <= 0:
        return [], None, None

    unique_coords, inverse = torch.unique(
        coords_n3.to(dtype=torch.long),
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    unique_count = int(unique_coords.shape[0])
    if unique_count <= 1:
        return [], unique_coords, inverse

    parent_coords = torch.div(unique_coords, 2, rounding_mode="floor")
    unique_parents, parent_inverse = torch.unique(
        parent_coords,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    child_slot = (
        (unique_coords[:, 0] & 1)
        + 2 * (unique_coords[:, 1] & 1)
        + 4 * (unique_coords[:, 2] & 1)
    ).to(dtype=torch.long)
    occupancy = torch.zeros(
        (unique_parents.shape[0], 8),
        device=coords_n3.device,
        dtype=torch.bool,
    )
    occupancy[parent_inverse, child_slot] = True
    parent_child_count = occupancy.sum(dim=1).to(dtype=torch.float32).index_select(0, parent_inverse)

    min_children_after = max(int(getattr(args, "leaf_pattern_delete_min_children_after", 1)), 0)
    valid = (parent_child_count - 1.0) >= float(min_children_after)
    valid_idx = valid.nonzero(as_tuple=False).reshape(-1)
    if valid_idx.numel() <= 0:
        return [], unique_coords, inverse

    pattern_weights = (2 ** torch.arange(8, device=coords_n3.device, dtype=torch.long)).view(1, 8)
    parent_code_unique = (occupancy.to(dtype=torch.long) * pattern_weights).sum(dim=1).clamp(0, 255)
    current_code = parent_code_unique.index_select(0, parent_inverse).clamp(0, 255)
    smoothing = max(float(getattr(args, "leaf_pattern_candidate_smoothing", 1.0)), 0.0)
    code_hist = torch.bincount(parent_code_unique, minlength=256).to(device=coords_n3.device, dtype=torch.float32)
    code_prob = (code_hist + float(smoothing))
    code_prob = code_prob / code_prob.sum().clamp_min(torch.finfo(torch.float32).eps)
    code_nll = -torch.log2(code_prob.clamp_min(torch.finfo(torch.float32).eps))

    bit_current = (1 << child_slot.clamp(0, 7)).to(device=coords_n3.device, dtype=torch.long)
    delete_code = torch.bitwise_and(current_code, torch.bitwise_not(bit_current)).clamp(0, 255)
    delete_gain = code_nll.index_select(0, current_code) - code_nll.index_select(0, delete_code)
    parent_nll = code_nll.index_select(0, current_code)
    if not isinstance(proxy_profile, dict) or not bool(proxy_profile.get("enabled", False)):
        proxy_profile = _sparsepcgc_codec_proxy_profile(unique_coords, args)
    proxy_weight = max(float(getattr(args, "sparsepcgc_codec_proxy_weight", 2.0)), 0.0)
    leaf_occupied_bits = proxy_profile.get("leaf_occupied_bits", None)
    if torch.is_tensor(leaf_occupied_bits) and leaf_occupied_bits.numel() == unique_count:
        proxy_drop_gain = leaf_occupied_bits.to(device=delete_gain.device, dtype=delete_gain.dtype)
    else:
        proxy_drop_gain = torch.zeros_like(delete_gain)

    selected = []
    seen = set()
    memory_weight = max(float(getattr(args, "sparsepcgc_actual_oracle_memory_weight", 0.75)), 0.0)
    memory_bonus = torch.zeros_like(delete_gain, dtype=torch.float32)
    memory_bad = torch.zeros_like(valid, dtype=torch.bool)
    memory_seen = torch.zeros_like(valid, dtype=torch.bool)
    if memory_weight > 0.0 or bool(getattr(args, "sparsepcgc_actual_oracle_memory_skip_bad", True)):
        for idx_item in valid_idx.detach().cpu().tolist():
            idx_int = int(idx_item)
            key = _sparsepcgc_actual_oracle_transition_key(
                "drop",
                int(current_code[idx_int].detach().cpu()),
                int(child_slot[idx_int].detach().cpu()),
                int(delete_code[idx_int].detach().cpu()),
            )
            bonus, is_bad, seen_memory = _sparsepcgc_actual_oracle_memory_bonus(args, key)
            memory_bonus[idx_int] = float(bonus)
            memory_bad[idx_int] = bool(is_bad)
            memory_seen[idx_int] = bool(seen_memory)

    order_valid = valid_idx
    filtered_valid = valid_idx[~memory_bad.index_select(0, valid_idx)]
    if filtered_valid.numel() > 0:
        order_valid = filtered_valid

    def _append_from_order(order_tensor, allow_memory_bad=False):
        nonlocal selected
        for item in order_tensor.detach().cpu().tolist():
            idx = int(item)
            if idx in seen:
                continue
            if not bool(valid[idx].detach().cpu()):
                continue
            key = _sparsepcgc_actual_oracle_transition_key(
                "drop",
                int(current_code[idx].detach().cpu()),
                int(child_slot[idx].detach().cpu()),
                int(delete_code[idx].detach().cpu()),
            )
            _bonus, is_bad, seen_memory = _sparsepcgc_actual_oracle_memory_bonus(args, key)
            if is_bad and not bool(allow_memory_bad):
                continue
            seen.add(idx)
            selected.append(
                {
                    "unique_idx": idx,
                    "memory_key": key,
                    "memory_seen": bool(seen_memory),
                    "score_hint": float(scored_gain[idx].detach().cpu()),
                    "proxy_delta_bits_hint": float(-proxy_drop_gain[idx].detach().cpu()),
                }
            )
            if len(selected) >= int(max_candidates):
                break

    scored_gain = (
        delete_gain
        + float(proxy_weight) * proxy_drop_gain
        + float(memory_weight) * memory_bonus.to(device=delete_gain.device, dtype=delete_gain.dtype)
    )
    valid_gain = scored_gain.index_select(0, order_valid)
    desc_order = order_valid.index_select(0, torch.argsort(valid_gain, descending=True))
    asc_order = order_valid.index_select(0, torch.argsort(valid_gain, descending=False))
    nll_order = order_valid.index_select(
        0,
        torch.argsort(parent_nll.index_select(0, order_valid), descending=True),
    )
    child_order = order_valid.index_select(
        0,
        torch.argsort(parent_child_count.index_select(0, order_valid), descending=True),
    )

    candidate_orders = (desc_order, asc_order, nll_order, child_order)
    for order in candidate_orders:
        if len(selected) >= int(max_candidates):
            break
        _append_from_order(order, allow_memory_bad=False)

    if len(selected) < int(max_candidates) and order_valid.numel() > 0:
        # Deterministic shuffle: 同じstep/同じSubtreeでは再現性を保ちつつ、候補の偏りを避ける。
        seed = int(global_step) * 1103515245 + unique_count * 12345
        noise = (
            (order_valid.to(dtype=torch.long) * 2654435761 + int(seed))
            & 0x7FFFFFFF
        )
        random_order = order_valid.index_select(0, torch.argsort(noise))
        _append_from_order(random_order, allow_memory_bad=False)

    if len(selected) < int(max_candidates) and bool(getattr(args, "sparsepcgc_actual_oracle_memory_fill_if_exhausted", True)):
        # メモリ上badな変換だけが残った場合でも候補探索を完全停止させない。
        # ここで補充した候補はactual評価と負例教師に回るため、探索は残しつつ採択はactual改善だけに保つ。
        for order in candidate_orders:
            if len(selected) >= int(max_candidates):
                break
            _append_from_order(order, allow_memory_bad=True)
        if len(selected) < int(max_candidates) and valid_idx.numel() > 0:
            seed = int(global_step) * 214013 + unique_count * 2531011
            noise = ((valid_idx.to(dtype=torch.long) * 1103515245 + int(seed)) & 0x7FFFFFFF)
            fallback_order = valid_idx.index_select(0, torch.argsort(noise))
            _append_from_order(fallback_order, allow_memory_bad=True)

    return selected[: int(max_candidates)], unique_coords, inverse


def _sparsepcgc_actual_oracle_add_candidates(unique_coords, args, global_step, max_candidates, proxy_profile=None):
    if unique_coords is None or unique_coords.numel() <= 0 or int(max_candidates) <= 0:
        return []

    unique_coords = unique_coords.to(dtype=torch.long)
    parent_coords = torch.div(unique_coords, 2, rounding_mode="floor")
    unique_parents, parent_inverse = torch.unique(
        parent_coords,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    if unique_parents.numel() <= 0:
        return []

    child_slot = (
        (unique_coords[:, 0] & 1)
        + 2 * (unique_coords[:, 1] & 1)
        + 4 * (unique_coords[:, 2] & 1)
    ).to(dtype=torch.long)
    occupancy = torch.zeros(
        (unique_parents.shape[0], 8),
        device=unique_coords.device,
        dtype=torch.bool,
    )
    occupancy[parent_inverse, child_slot] = True

    empty_parent_idx, empty_slot = (~occupancy).nonzero(as_tuple=True)
    if empty_parent_idx.numel() <= 0:
        return []

    pattern_weights = (2 ** torch.arange(8, device=unique_coords.device, dtype=torch.long)).view(1, 8)
    parent_code = (occupancy.to(dtype=torch.long) * pattern_weights).sum(dim=1).clamp(0, 255)
    smoothing = max(float(getattr(args, "leaf_pattern_candidate_smoothing", 1.0)), 0.0)
    code_hist = torch.bincount(parent_code, minlength=256).to(device=unique_coords.device, dtype=torch.float32)
    code_prob = (code_hist + float(smoothing))
    code_prob = code_prob / code_prob.sum().clamp_min(torch.finfo(torch.float32).eps)
    code_nll = -torch.log2(code_prob.clamp_min(torch.finfo(torch.float32).eps))

    add_bit = (1 << empty_slot.clamp(0, 7)).to(dtype=torch.long)
    current_code = parent_code.index_select(0, empty_parent_idx)
    add_code = torch.bitwise_or(current_code, add_bit).clamp(0, 255)
    add_gain = code_nll.index_select(0, current_code) - code_nll.index_select(0, add_code)
    parent_nll = code_nll.index_select(0, current_code)
    if not isinstance(proxy_profile, dict) or not bool(proxy_profile.get("enabled", False)):
        proxy_profile = _sparsepcgc_codec_proxy_profile(unique_coords, args)
    proxy_weight = max(float(getattr(args, "sparsepcgc_codec_proxy_weight", 2.0)), 0.0)
    leaf_add_delta_bits = proxy_profile.get("leaf_add_delta_bits", None)
    if torch.is_tensor(leaf_add_delta_bits) and leaf_add_delta_bits.ndim == 2:
        proxy_add_gain = -leaf_add_delta_bits.to(device=add_gain.device, dtype=add_gain.dtype)[
            empty_parent_idx,
            empty_slot,
        ]
    else:
        proxy_add_gain = torch.zeros_like(add_gain)
    memory_weight = max(float(getattr(args, "sparsepcgc_actual_oracle_memory_weight", 0.75)), 0.0)
    memory_bonus = torch.zeros_like(add_gain, dtype=torch.float32)
    memory_bad = torch.zeros_like(add_gain, dtype=torch.bool)
    if memory_weight > 0.0 or bool(getattr(args, "sparsepcgc_actual_oracle_memory_skip_bad", True)):
        for flat_item in range(int(empty_parent_idx.numel())):
            key = _sparsepcgc_actual_oracle_transition_key(
                "add",
                int(current_code[flat_item].detach().cpu()),
                int(empty_slot[flat_item].detach().cpu()),
                int(add_code[flat_item].detach().cpu()),
            )
            bonus, is_bad, _seen_memory = _sparsepcgc_actual_oracle_memory_bonus(args, key)
            memory_bonus[flat_item] = float(bonus)
            memory_bad[flat_item] = bool(is_bad)

    selected = []
    seen = set()

    def _child_bits(slot):
        slot = int(slot)
        return unique_coords.new_tensor([slot & 1, (slot >> 1) & 1, (slot >> 2) & 1])

    def _append_from_flat_order(order_tensor, allow_memory_bad=False):
        nonlocal selected
        for item in order_tensor.detach().cpu().tolist():
            flat_idx = int(item)
            parent_idx = int(empty_parent_idx[flat_idx].detach().cpu())
            target_slot = int(empty_slot[flat_idx].detach().cpu())
            key = (parent_idx, target_slot)
            if key in seen:
                continue
            memory_key = _sparsepcgc_actual_oracle_transition_key(
                "add",
                int(current_code[flat_idx].detach().cpu()),
                int(target_slot),
                int(add_code[flat_idx].detach().cpu()),
            )
            _bonus, is_bad, seen_memory = _sparsepcgc_actual_oracle_memory_bonus(args, memory_key)
            if is_bad and not bool(allow_memory_bad):
                continue

            src_candidates = (parent_inverse == parent_idx).nonzero(as_tuple=False).reshape(-1)
            if src_candidates.numel() <= 0:
                continue
            target_coord = unique_parents[parent_idx] * 2 + _child_bits(target_slot)
            dist = (unique_coords.index_select(0, src_candidates) - target_coord.view(1, 3)).abs().sum(dim=1)
            source_unique_idx = int(src_candidates[int(torch.argmin(dist).detach().cpu())].detach().cpu())

            seen.add(key)
            selected.append(
                {
                    "source_unique_idx": source_unique_idx,
                    "target_child_slot": target_slot,
                    "target_coord": target_coord.detach().clone(),
                    "score_hint": float(scored_add_gain[flat_idx].detach().cpu()),
                    "proxy_delta_bits_hint": float((-proxy_add_gain[flat_idx]).detach().cpu()),
                    "memory_key": memory_key,
                    "memory_seen": bool(seen_memory),
                }
            )
            if len(selected) >= int(max_candidates):
                break

    flat_idx = torch.arange(empty_parent_idx.numel(), device=unique_coords.device, dtype=torch.long)
    filtered_flat_idx = flat_idx[~memory_bad]
    if filtered_flat_idx.numel() <= 0:
        filtered_flat_idx = flat_idx
    scored_add_gain = (
        add_gain
        + float(proxy_weight) * proxy_add_gain
        + float(memory_weight) * memory_bonus.to(device=add_gain.device, dtype=add_gain.dtype)
    )
    gain_order = filtered_flat_idx.index_select(
        0,
        torch.argsort(scored_add_gain.index_select(0, filtered_flat_idx), descending=True),
    )
    nll_order = filtered_flat_idx.index_select(
        0,
        torch.argsort(parent_nll.index_select(0, filtered_flat_idx), descending=True),
    )
    seed = int(global_step) * 1664525 + int(unique_coords.shape[0]) * 1013904223
    noise = ((filtered_flat_idx * 2654435761 + int(seed)) & 0x7FFFFFFF)
    random_order = filtered_flat_idx.index_select(0, torch.argsort(noise))

    candidate_orders = (gain_order, nll_order, random_order)
    for order in candidate_orders:
        if len(selected) >= int(max_candidates):
            break
        _append_from_flat_order(order, allow_memory_bad=False)

    if len(selected) < int(max_candidates) and bool(getattr(args, "sparsepcgc_actual_oracle_memory_fill_if_exhausted", True)):
        for order in candidate_orders:
            if len(selected) >= int(max_candidates):
                break
            _append_from_flat_order(order, allow_memory_bad=True)
        if len(selected) < int(max_candidates) and flat_idx.numel() > 0:
            seed = int(global_step) * 22695477 + int(unique_coords.shape[0]) * 1_103_515_245
            noise = ((flat_idx * 2654435761 + int(seed)) & 0x7FFFFFFF)
            fallback_order = flat_idx.index_select(0, torch.argsort(noise))
            _append_from_flat_order(fallback_order, allow_memory_bad=True)

    return selected[: int(max_candidates)]


def _sparsepcgc_parse_float_list(raw_value, default_values):
    if isinstance(raw_value, str):
        values = []
        for item in raw_value.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError:
                continue
        return values or list(default_values)
    if isinstance(raw_value, (list, tuple)):
        values = []
        for item in raw_value:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
        return values or list(default_values)
    return list(default_values)


def _sparsepcgc_actual_oracle_macro_prune_candidates(
    unique_coords,
    args,
    max_candidates,
    proxy_profile=None,
    base_proxy_bits=None,
):
    if unique_coords is None or unique_coords.numel() <= 0 or int(max_candidates) <= 0:
        return []
    unique_coords = torch.unique(unique_coords.to(dtype=torch.long), dim=0, sorted=True)
    unique_count = int(unique_coords.shape[0])
    if unique_count <= 8:
        return []

    ratios = _sparsepcgc_parse_float_list(
        getattr(args, "sparsepcgc_actual_oracle_macro_prune_ratios", "0.05,0.10,0.15,0.20"),
        [0.05, 0.10, 0.15, 0.20],
    )
    max_ratio = min(
        max(float(getattr(args, "sparsepcgc_actual_oracle_macro_prune_max_ratio", 0.20)), 0.0),
        0.95,
    )
    ratios = sorted({min(max(float(ratio), 0.0), max_ratio) for ratio in ratios if float(ratio) > 0.0})
    if not ratios:
        return []

    min_voxels = max(int(getattr(args, "sparsepcgc_actual_oracle_macro_prune_min_voxels", 8)), 1)
    max_voxels = max(int(getattr(args, "sparsepcgc_actual_oracle_macro_prune_max_voxels", 512)), min_voxels)
    neigh = _sparsepcgc_codec_proxy_neighbor_count(unique_coords).to(device=unique_coords.device, dtype=torch.float32)
    parent = torch.div(unique_coords, 2, rounding_mode="floor")
    _unique_parent, parent_inverse = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    parent_pop = torch.bincount(parent_inverse, minlength=int(_unique_parent.shape[0])).to(
        device=unique_coords.device,
        dtype=torch.float32,
    )
    parent_pop_leaf = parent_pop.index_select(0, parent_inverse)
    if proxy_profile is not None and torch.is_tensor(proxy_profile.get("leaf_occupied_bits", None)):
        leaf_bits = proxy_profile["leaf_occupied_bits"].to(device=unique_coords.device, dtype=torch.float32)
        if leaf_bits.numel() != unique_count:
            leaf_bits = torch.zeros((unique_count,), device=unique_coords.device, dtype=torch.float32)
    else:
        leaf_bits = torch.zeros((unique_count,), device=unique_coords.device, dtype=torch.float32)
    leaf_bits_norm = leaf_bits / leaf_bits.detach().mean().clamp_min(1e-6)

    # Probe結果では「低密度voxelをまとまった割合で落とす」候補がactual bitを安定して下げた。
    # codec priority候補とは別に、単純な密度rank候補を必ずactual検証へ入れる。
    density_score = neigh + parent_pop_leaf * 0.5
    density_order = torch.argsort(density_score, descending=False)

    # High priority means cheap geometry removal and expensive occupancy coding:
    # isolated leaves, small parent populations, and high context NLL.
    codec_drop_priority = (
        (3.0 - neigh).clamp_min(0.0) * 1.50
        + (3.0 - parent_pop_leaf).clamp_min(0.0) * 0.75
        + leaf_bits_norm.clamp_min(0.0)
    )
    codec_order = torch.argsort(codec_drop_priority, descending=True)
    if base_proxy_bits is None:
        base_proxy_bits = _sparsepcgc_codec_proxy_bits(unique_coords, args)

    candidates = []
    seen_masks = set()

    def _append_candidate(ratio, drop_order, variant):
        drop_count = int(math.ceil(float(unique_count) * float(ratio)))
        drop_count = min(max(drop_count, min_voxels), max_voxels, unique_count - 1)
        if drop_count <= 0:
            return
        drop_idx = drop_order[:drop_count].to(device=unique_coords.device, dtype=torch.long)
        mask_key = tuple(sorted(int(v) for v in drop_idx.detach().cpu().tolist()))
        if mask_key in seen_masks:
            return
        seen_masks.add(mask_key)
        keep = torch.ones((unique_count,), device=unique_coords.device, dtype=torch.bool)
        keep[drop_idx] = False
        candidate_coords = torch.unique(unique_coords[keep], dim=0, sorted=True)
        if int(candidate_coords.shape[0]) <= 0:
            return
        proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
            candidate_coords,
            args,
            base_proxy_bits,
        )
        if str(variant) == "density":
            priority_gain = float((-density_score).index_select(0, drop_idx).mean().detach().cpu())
            variant_bonus = 100.0
        else:
            priority_gain = float(codec_drop_priority.index_select(0, drop_idx).mean().detach().cpu())
            variant_bonus = 0.0
        score = (
            variant_bonus
            + 10.0 * float(drop_count) / max(float(unique_count), 1.0)
            + priority_gain
            - max(float(getattr(args, "sparsepcgc_codec_proxy_weight", 2.0)), 0.0) * float(proxy_percent)
        )
        candidates.append(
            {
                "op": "macro_prune",
                "variant": str(variant),
                "unique_indices": [int(v) for v in drop_idx.detach().cpu().tolist()],
                "candidate_coords": candidate_coords.detach().clone(),
                "drop_count": int(drop_count),
                "drop_ratio": float(drop_count) / max(float(unique_count), 1.0),
                "score": float(score),
                "proxy_percent": float(proxy_percent),
                "proxy_bits": float(proxy_bits),
            }
        )

    for ratio in sorted(ratios, reverse=True):
        _append_candidate(ratio, density_order, "density")
    for ratio in sorted(ratios, reverse=True):
        _append_candidate(ratio, codec_order, "codec")

    candidates = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return candidates[: int(max_candidates)]


def _sparsepcgc_actual_oracle_full_cloud_macro_prune_candidates(
    full_coords,
    args,
    max_candidates,
    teacher_coords=None,
):
    if full_coords is None or not torch.is_tensor(full_coords) or full_coords.numel() <= 0 or int(max_candidates) <= 0:
        return []
    # full_eval_coords comes from the canonical full-cloud voxel context and is
    # already unique. Re-running torch.unique over 0.7M+ rows dominated teacher
    # generation on 8i sequences.
    full_coords = full_coords.to(dtype=torch.long).contiguous()
    full_count = int(full_coords.shape[0])
    if full_count <= 8:
        return []

    ratios = _sparsepcgc_parse_float_list(
        getattr(args, "sparsepcgc_actual_oracle_full_cloud_macro_prune_ratios", "0.02,0.05"),
        [0.02, 0.05],
    )
    max_ratio = min(
        max(float(getattr(args, "sparsepcgc_actual_oracle_full_cloud_macro_prune_max_ratio", 0.05)), 0.0),
        0.50,
    )
    ratios = sorted({min(max(float(ratio), 0.0), max_ratio) for ratio in ratios if float(ratio) > 0.0}, reverse=True)
    if not ratios:
        return []

    min_voxels = max(int(getattr(args, "sparsepcgc_actual_oracle_full_cloud_macro_prune_min_voxels", 128)), 1)
    max_voxels = max(
        int(getattr(args, "sparsepcgc_actual_oracle_full_cloud_macro_prune_max_voxels", 20000)),
        min_voxels,
    )
    candidates = []
    block_sizes = [
        max(int(round(value)), 2)
        for value in _sparsepcgc_parse_float_list(
            getattr(args, "sparsepcgc_actual_oracle_full_cloud_subtree_block_sizes", "32"),
            [32.0],
        )
    ]
    subtree_ratios = _sparsepcgc_parse_float_list(
        getattr(args, "sparsepcgc_actual_oracle_full_cloud_subtree_prune_ratios", "0.10,0.20,0.30"),
        [0.10, 0.20, 0.30],
    )
    target_ratio = min(
        max(float(getattr(args, "sparsepcgc_actual_oracle_full_cloud_subtree_target_ratio", 0.20)), 0.0),
        max_ratio,
    )
    min_target_fraction = min(
        max(
            float(
                getattr(
                    args,
                    "sparsepcgc_actual_oracle_full_cloud_subtree_min_target_fraction",
                    0.50,
                )
            ),
            0.0,
        ),
        1.0,
    )
    min_target_ratio = float(target_ratio) * float(min_target_fraction)
    auto_refine_blocks = bool(
        getattr(args, "sparsepcgc_actual_oracle_full_cloud_subtree_auto_refine_blocks", True)
    )
    min_refine_block_size = max(
        int(getattr(args, "sparsepcgc_actual_oracle_full_cloud_subtree_min_refine_block_size", 16)),
        2,
    )
    full_coords_cpu = full_coords.detach().to(device="cpu", dtype=torch.long).numpy()
    structured_has_target_like = False
    structured_seen_blocks = set()

    def _append_structured_candidates_for_block(block_size):
        nonlocal structured_has_target_like
        block_size = max(int(block_size), 2)
        if block_size in structured_seen_blocks:
            return
        structured_seen_blocks.add(block_size)
        block_coords_cpu = np.floor_divide(full_coords_cpu, int(block_size))
        unique_blocks_cpu, block_inverse_cpu, block_counts_cpu = np.unique(
            block_coords_cpu,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        if int(unique_blocks_cpu.shape[0]) <= 1:
            return
        block_order_cpu = np.argsort(block_counts_cpu, kind="stable")
        cumulative_counts_cpu = np.cumsum(block_counts_cpu[block_order_cpu], dtype=np.int64)
        ordered_subtree_ratios = sorted(
            subtree_ratios,
            key=lambda value: abs(float(value) - float(target_ratio)),
        )
        for ratio_raw in ordered_subtree_ratios:
            ratio = min(max(float(ratio_raw), 0.0), max_ratio)
            if ratio <= 0.0:
                continue
            target_drop = min(
                max(int(math.ceil(float(full_count) * ratio)), min_voxels),
                max_voxels,
                full_count - 1,
            )
            take = int(np.searchsorted(cumulative_counts_cpu, int(target_drop), side="left")) + 1
            take = min(max(take, 1), int(block_order_cpu.size) - 1)
            drop_blocks_cpu = block_order_cpu[:take].copy()
            drop_block_mask_cpu = np.zeros((unique_blocks_cpu.shape[0],), dtype=np.bool_)
            drop_block_mask_cpu[drop_blocks_cpu] = True
            drop_mask_cpu = drop_block_mask_cpu[block_inverse_cpu]
            drop_count = int(np.count_nonzero(drop_mask_cpu))
            if drop_count < min_voxels or drop_count > max_voxels or drop_count >= full_count:
                continue
            drop_mask = torch.from_numpy(drop_mask_cpu).to(device=full_coords.device)
            candidate_coords = full_coords[~drop_mask].contiguous()
            actual_ratio = float(drop_count) / max(float(full_count), 1.0)
            target_like = actual_ratio >= min_target_ratio
            structured_has_target_like = bool(structured_has_target_like or target_like)
            score = float(10000.0 - 1000.0 * abs(actual_ratio - target_ratio))
            if not target_like:
                # A too-coarse block can drop only a tiny fraction of the cloud.
                # Do not let that under-target candidate consume the only actual eval.
                score -= float(20000.0 + 1000.0 * max(min_target_ratio - actual_ratio, 0.0))
            candidates.append(
                {
                    "op": "full_cloud_subtree_prune",
                    "variant": f"block_{int(block_size)}_ratio_{actual_ratio:.6f}",
                    "candidate_coords": candidate_coords.detach().clone(),
                    "drop_coords": full_coords[drop_mask].detach().clone(),
                    "drop_count": int(drop_count),
                    "drop_block_count": int(take),
                    "drop_block_coords": torch.from_numpy(
                        unique_blocks_cpu[drop_blocks_cpu].copy()
                    ).to(device=full_coords.device, dtype=torch.long),
                    "block_size": int(block_size),
                    "drop_ratio": float(actual_ratio),
                    "target_like": bool(target_like),
                    "score": float(score),
                }
            )
            if len(candidates) >= int(max_candidates) and structured_has_target_like:
                break

    base_block_sizes = sorted(set(block_sizes))
    for block_size in base_block_sizes:
        _append_structured_candidates_for_block(block_size)
        if len(candidates) >= int(max_candidates) and structured_has_target_like:
            break

    if (
        auto_refine_blocks
        and not structured_has_target_like
        and target_ratio > 0.0
    ):
        refined_block_sizes = []
        seen_refined = set(base_block_sizes)
        for block_size in base_block_sizes:
            refine_block = max(int(block_size) // 2, 0)
            while refine_block >= min_refine_block_size:
                if refine_block not in seen_refined:
                    refined_block_sizes.append(refine_block)
                    seen_refined.add(refine_block)
                refine_block //= 2
        for block_size in refined_block_sizes:
            _append_structured_candidates_for_block(block_size)
            if len(candidates) >= int(max_candidates) and structured_has_target_like:
                break

    # Structured subtree candidates are intentionally ranked above scattered
    # voxel heuristics and already fill the proxy top-K budget. Avoid the much
    # more expensive full-cloud neighbor/parent scans when they cannot possibly
    # reach actual evaluation.
    if len(candidates) >= int(max_candidates) and structured_has_target_like:
        candidates = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return candidates[: int(max_candidates)]

    axis_neigh = _sparsepcgc_axis_neighbor_count(full_coords).to(device=full_coords.device, dtype=torch.long)
    thresholds = _sparsepcgc_parse_float_list(
        getattr(args, "sparsepcgc_actual_oracle_full_cloud_prune_neighbor_thresholds", "3"),
        [3.0],
    )
    for threshold_raw in sorted({int(float(value)) for value in thresholds if int(float(value)) > 0}):
        keep = axis_neigh >= int(threshold_raw)
        drop_count = int((~keep).sum().detach().cpu())
        if drop_count < min_voxels or drop_count > max_voxels or drop_count >= full_count:
            continue
        candidate_coords = torch.unique(full_coords[keep], dim=0, sorted=True)
        if int(candidate_coords.shape[0]) <= 0:
            continue
        candidates.append(
            {
                "op": "full_cloud_neighbor_prune",
                "variant": f"axis_neighbor_lt_{int(threshold_raw)}",
                "candidate_coords": candidate_coords.detach().clone(),
                "drop_coords": full_coords[~keep].detach().clone(),
                "drop_count": int(drop_count),
                "drop_ratio": float(drop_count) / max(float(full_count), 1.0),
                "score": float(100.0 - int(threshold_raw)),
            }
        )

    neigh = _sparsepcgc_codec_proxy_neighbor_count(full_coords).to(device=full_coords.device, dtype=torch.float32)
    parent = torch.div(full_coords, 2, rounding_mode="floor")
    unique_parent, parent_inverse = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    parent_pop = torch.bincount(parent_inverse, minlength=int(unique_parent.shape[0])).to(
        device=full_coords.device,
        dtype=torch.float32,
    )
    parent_pop_leaf = parent_pop.index_select(0, parent_inverse)
    density_score = neigh + parent_pop_leaf * 0.5
    order = torch.argsort(density_score, descending=False)

    seen_counts = set()
    for ratio in ratios:
        drop_count = int(math.ceil(float(full_count) * float(ratio)))
        drop_count = min(max(drop_count, min_voxels), max_voxels, full_count - 1)
        if drop_count <= 0 or drop_count in seen_counts:
            continue
        seen_counts.add(drop_count)
        drop_idx = order[:drop_count].to(device=full_coords.device, dtype=torch.long)
        keep = torch.ones((full_count,), device=full_coords.device, dtype=torch.bool)
        keep[drop_idx] = False
        candidate_coords = torch.unique(full_coords[keep], dim=0, sorted=True)
        if int(candidate_coords.shape[0]) <= 0:
            continue
        low_density = float(density_score.index_select(0, drop_idx).mean().detach().cpu())
        candidates.append(
            {
                "op": "full_cloud_macro_prune",
                "candidate_coords": candidate_coords.detach().clone(),
                "drop_coords": full_coords.index_select(0, drop_idx).detach().clone(),
                "drop_count": int(drop_count),
                "drop_ratio": float(drop_count) / max(float(full_count), 1.0),
                "score": float(-low_density + 10.0 * float(drop_count) / max(float(full_count), 1.0)),
            }
        )
    candidates = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return candidates[: int(max_candidates)]


def _sparsepcgc_fast_diagnostic_prune_indices(unique_coords, full_coords, args):
    if (
        unique_coords is None
        or full_coords is None
        or not torch.is_tensor(unique_coords)
        or not torch.is_tensor(full_coords)
        or unique_coords.numel() <= 0
        or full_coords.numel() <= 0
    ):
        return [], {}
    unique_coords = _sparsepcgc_coords_to_n3(unique_coords)
    full_coords = _sparsepcgc_coords_to_n3(full_coords)
    if unique_coords is None or full_coords is None:
        return [], {}
    unique_coords = torch.unique(unique_coords.to(dtype=torch.long), dim=0, sorted=True)
    full_coords = torch.unique(full_coords.to(device=unique_coords.device, dtype=torch.long), dim=0, sorted=True)
    if int(unique_coords.shape[0]) <= 0 or int(full_coords.shape[0]) <= 8:
        return [], {}

    threshold = max(int(getattr(args, "sparsepcgc_fast_diagnostic_neighbor_threshold", 3)), 1)
    mode = str(getattr(args, "sparsepcgc_fast_diagnostic_prune_mode", "axis_threshold")).strip().lower()
    if mode not in {"axis_threshold", "density_ratio", "hybrid"}:
        mode = "axis_threshold"
    target_global_ratio = min(
        max(float(getattr(args, "sparsepcgc_fast_diagnostic_target_global_ratio", 0.05)), 0.0),
        0.30,
    )
    target_local_ratio = min(
        max(float(getattr(args, "sparsepcgc_fast_diagnostic_target_local_ratio", 0.05)), 0.0),
        0.30,
    )
    parent_weight = max(
        float(getattr(args, "sparsepcgc_fast_diagnostic_density_parent_weight", 0.5)),
        0.0,
    )
    min_local = max(int(getattr(args, "sparsepcgc_fast_diagnostic_min_local_voxels", 1)), 1)
    max_local = max(int(getattr(args, "sparsepcgc_fast_diagnostic_max_local_voxels", 512)), min_local)

    def _density_score(coords_n3):
        axis = _sparsepcgc_axis_neighbor_count(coords_n3).to(device=coords_n3.device, dtype=torch.float32)
        parent = torch.div(coords_n3, 2, rounding_mode="floor")
        unique_parent, parent_inverse = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
        parent_pop = torch.bincount(parent_inverse, minlength=int(unique_parent.shape[0])).to(
            device=coords_n3.device,
            dtype=torch.float32,
        )
        parent_pop_leaf = parent_pop.index_select(0, parent_inverse)
        return axis + float(parent_weight) * parent_pop_leaf

    axis_neigh = _sparsepcgc_axis_neighbor_count(full_coords).to(device=full_coords.device, dtype=torch.long)
    axis_drop_mask = axis_neigh < int(threshold)
    density_drop_mask = torch.zeros_like(axis_drop_mask, dtype=torch.bool)
    density_score = _density_score(full_coords)
    if mode in {"density_ratio", "hybrid"} and target_global_ratio > 0.0:
        full_count = int(full_coords.shape[0])
        density_count = int(math.ceil(float(full_count) * float(target_global_ratio)))
        density_count = min(max(density_count, 1), max(full_count - 1, 1))
        density_order = torch.argsort(density_score, descending=False)
        density_drop_mask[density_order[:density_count]] = True

    if mode == "axis_threshold":
        drop_mask = axis_drop_mask
    elif mode == "hybrid":
        drop_mask = axis_drop_mask | density_drop_mask
    else:
        drop_mask = density_drop_mask

    global_drop_count = int(drop_mask.detach().sum().cpu())
    if global_drop_count <= 0:
        return [], {
            "diagnostic": "density_ratio_prune" if mode != "axis_threshold" else "axis_neighbor_prune",
            "mode": str(mode),
            "threshold": int(threshold),
            "global_drop_count": 0,
            "full_count": int(full_coords.shape[0]),
        }

    drop_coords = full_coords[drop_mask]
    full_mins = torch.minimum(unique_coords.amin(dim=0), drop_coords.amin(dim=0))
    full_maxs = torch.maximum(unique_coords.amax(dim=0), drop_coords.amax(dim=0))
    span = (full_maxs - full_mins + 1).clamp_min(1)

    def _keys(values):
        shifted = values - full_mins
        return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]

    drop_keys = torch.unique(_keys(drop_coords), sorted=True)
    unique_keys = _keys(unique_coords)
    pos = torch.searchsorted(drop_keys, unique_keys)
    in_bounds = pos < drop_keys.numel()
    safe_pos = pos.clamp(max=max(int(drop_keys.numel()) - 1, 0))
    local_mask = in_bounds & (drop_keys[safe_pos] == unique_keys)
    local_indices = local_mask.nonzero(as_tuple=False).reshape(-1)
    local_count = int(local_indices.numel())

    local_density = _density_score(unique_coords)
    desired_local_count = local_count
    if mode in {"density_ratio", "hybrid"} and target_local_ratio > 0.0:
        desired_local_count = int(math.ceil(float(unique_coords.shape[0]) * float(target_local_ratio)))
        desired_local_count = min(max(desired_local_count, min_local), max_local, max(int(unique_coords.shape[0]) - 1, 1))
        if (
            local_count < desired_local_count
            and bool(getattr(args, "sparsepcgc_fast_diagnostic_density_backfill_local", True))
        ):
            selected = torch.zeros((int(unique_coords.shape[0]),), device=unique_coords.device, dtype=torch.bool)
            if local_count > 0:
                selected[local_indices] = True
            local_order = torch.argsort(local_density, descending=False)
            need = max(int(desired_local_count) - int(local_count), 0)
            if need > 0:
                backfill = local_order[~selected.index_select(0, local_order)][:need]
                if backfill.numel() > 0:
                    local_indices = torch.cat([local_indices, backfill.to(device=local_indices.device)], dim=0)
                    local_indices = torch.unique(local_indices, sorted=True)
                    local_count = int(local_indices.numel())

    if local_count < min_local:
        return [], {
            "diagnostic": "density_ratio_prune" if mode != "axis_threshold" else "axis_neighbor_prune",
            "mode": str(mode),
            "threshold": int(threshold),
            "global_drop_count": int(global_drop_count),
            "full_count": int(full_coords.shape[0]),
            "local_drop_count": int(local_count),
            "reason": "below_min_local",
        }

    local_limit = int(max_local)
    if mode in {"density_ratio", "hybrid"} and target_local_ratio > 0.0:
        local_limit = min(local_limit, int(desired_local_count))
    if local_count > local_limit:
        selected_density = local_density.index_select(0, local_indices)
        order = torch.argsort(selected_density, descending=False)
        local_indices = local_indices.index_select(0, order[:local_limit])
        local_count = int(local_indices.numel())

    debug = {
        "diagnostic": "density_ratio_prune" if mode != "axis_threshold" else "axis_neighbor_prune",
        "mode": str(mode),
        "threshold": int(threshold),
        "global_drop_count": int(global_drop_count),
        "full_count": int(full_coords.shape[0]),
        "local_drop_count": int(local_count),
        "global_drop_ratio": float(global_drop_count) / max(float(full_coords.shape[0]), 1.0),
        "local_drop_ratio": float(local_count) / max(float(unique_coords.shape[0]), 1.0),
        "target_global_ratio": float(target_global_ratio),
        "target_local_ratio": float(target_local_ratio),
        "density_parent_weight": float(parent_weight),
        "desired_local_count": int(desired_local_count),
    }
    return [int(v) for v in local_indices.detach().cpu().tolist()], debug


def _sparsepcgc_fast_diagnostic_add_candidates(unique_coords, full_coords, args):
    if (
        not bool(getattr(args, "sparsepcgc_fast_diagnostic_add_teacher", True))
        or unique_coords is None
        or full_coords is None
        or not torch.is_tensor(unique_coords)
        or not torch.is_tensor(full_coords)
        or unique_coords.numel() <= 0
        or full_coords.numel() <= 0
    ):
        return [], {}
    max_local = max(int(getattr(args, "sparsepcgc_fast_diagnostic_add_max_local_voxels", 4)), 0)
    if max_local <= 0:
        return [], {"diagnostic": "dense_hole_add", "reason": "disabled_by_budget"}

    unique_coords = _sparsepcgc_coords_to_n3(unique_coords)
    full_coords = _sparsepcgc_coords_to_n3(full_coords)
    if unique_coords is None or full_coords is None:
        return [], {}
    unique_coords = torch.unique(unique_coords.to(dtype=torch.long), dim=0, sorted=True)
    full_coords = torch.unique(full_coords.to(device=unique_coords.device, dtype=torch.long), dim=0, sorted=True)
    if int(unique_coords.shape[0]) <= 0 or int(full_coords.shape[0]) <= 8:
        return [], {}

    threshold = min(
        max(int(getattr(args, "sparsepcgc_fast_diagnostic_add_neighbor_threshold", 6)), 1),
        6,
    )
    offsets = torch.tensor(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        device=full_coords.device,
        dtype=torch.long,
    )
    query = (full_coords[:, None, :] + offsets.view(1, -1, 3)).reshape(-1, 3)
    unique_query, inverse_query = torch.unique(query, dim=0, sorted=True, return_inverse=True)
    query_counts = torch.bincount(inverse_query, minlength=int(unique_query.shape[0])).to(
        device=full_coords.device,
        dtype=torch.long,
    )

    combined = torch.cat([unique_query, full_coords], dim=0)
    mins = combined.amin(dim=0)
    span = (combined.amax(dim=0) - mins + 1).clamp_min(1)

    def _keys(values):
        shifted = values - mins
        return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]

    full_keys = torch.unique(_keys(full_coords), sorted=True)
    query_keys = _keys(unique_query)
    pos = torch.searchsorted(full_keys, query_keys)
    in_bounds = pos < full_keys.numel()
    safe_pos = pos.clamp(max=max(int(full_keys.numel()) - 1, 0))
    occupied = in_bounds & (full_keys[safe_pos] == query_keys)
    empty_coords = unique_query[~occupied]
    empty_counts = query_counts[~occupied]
    dense_mask = empty_counts >= int(threshold)
    global_add_count = int(dense_mask.detach().sum().cpu())
    if global_add_count <= 0:
        return [], {
            "diagnostic": "dense_hole_add",
            "threshold": int(threshold),
            "global_add_count": 0,
            "full_count": int(full_coords.shape[0]),
        }

    target_coords = empty_coords[dense_mask]
    target_counts = empty_counts[dense_mask]
    local_parent = torch.div(unique_coords, 2, rounding_mode="floor")
    unique_parent, parent_inverse = torch.unique(local_parent, dim=0, sorted=True, return_inverse=True)
    parent_child_slot = (
        (unique_coords[:, 0] & 1)
        + 2 * (unique_coords[:, 1] & 1)
        + 4 * (unique_coords[:, 2] & 1)
    ).to(dtype=torch.long)
    occupancy = torch.zeros((unique_parent.shape[0], 8), device=unique_coords.device, dtype=torch.bool)
    occupancy[parent_inverse, parent_child_slot] = True

    target_parent = torch.div(target_coords, 2, rounding_mode="floor")
    parent_combined = torch.cat([target_parent, unique_parent], dim=0)
    parent_mins = parent_combined.amin(dim=0)
    parent_span = (parent_combined.amax(dim=0) - parent_mins + 1).clamp_min(1)

    def _parent_keys(values):
        shifted = values - parent_mins
        return shifted[:, 0] * parent_span[1] * parent_span[2] + shifted[:, 1] * parent_span[2] + shifted[:, 2]

    unique_parent_keys = torch.unique(_parent_keys(unique_parent), sorted=True)
    target_parent_keys = _parent_keys(target_parent)
    parent_pos = torch.searchsorted(unique_parent_keys, target_parent_keys)
    parent_in_bounds = parent_pos < unique_parent_keys.numel()
    safe_parent_pos = parent_pos.clamp(max=max(int(unique_parent_keys.numel()) - 1, 0))
    local_mask = parent_in_bounds & (unique_parent_keys[safe_parent_pos] == target_parent_keys)
    local_count = int(local_mask.detach().sum().cpu())
    if local_count <= 0:
        return [], {
            "diagnostic": "dense_hole_add",
            "threshold": int(threshold),
            "global_add_count": int(global_add_count),
            "local_add_count": 0,
            "full_count": int(full_coords.shape[0]),
        }

    local_target_idx = local_mask.nonzero(as_tuple=False).reshape(-1)
    local_order = local_target_idx.index_select(
        0,
        torch.argsort(target_counts.index_select(0, local_target_idx), descending=True),
    )
    selected = []
    used_sources = set()
    used_targets = set()
    for target_idx_raw in local_order.detach().cpu().tolist():
        target_idx = int(target_idx_raw)
        parent_idx = int(safe_parent_pos[target_idx].detach().cpu())
        target_coord = target_coords[target_idx].to(device=unique_coords.device, dtype=torch.long)
        target_slot = int(
            ((target_coord[0] & 1) + 2 * (target_coord[1] & 1) + 4 * (target_coord[2] & 1)).detach().cpu()
        )
        key = (int(parent_idx), int(target_slot))
        if key in used_targets or bool(occupancy[parent_idx, target_slot].detach().cpu()):
            continue
        source_candidates = (parent_inverse == int(parent_idx)).nonzero(as_tuple=False).reshape(-1)
        if source_candidates.numel() <= 0:
            continue
        dist = (unique_coords.index_select(0, source_candidates) - target_coord.view(1, 3)).abs().sum(dim=1)
        source_unique_idx = int(source_candidates[int(torch.argmin(dist).detach().cpu())].detach().cpu())
        if source_unique_idx in used_sources:
            continue
        used_sources.add(source_unique_idx)
        used_targets.add(key)
        selected.append(
            {
                "source_unique_idx": source_unique_idx,
                "target_child_slot": target_slot,
                "target_coord": target_coord.detach().clone(),
                "score_hint": float(target_counts[target_idx].detach().cpu()),
            }
        )
        if len(selected) >= int(max_local):
            break

    debug = {
        "diagnostic": "dense_hole_add",
        "threshold": int(threshold),
        "global_add_count": int(global_add_count),
        "local_add_count": int(len(selected)),
        "full_count": int(full_coords.shape[0]),
        "global_add_ratio": float(global_add_count) / max(float(full_coords.shape[0]), 1.0),
        "local_add_ratio": float(len(selected)) / max(float(unique_coords.shape[0]), 1.0),
    }
    return selected, debug


def _sparsepcgc_actual_oracle_subtree_move_candidates(unique_coords, args, global_step, max_candidates, base_proxy_bits=None):
    if unique_coords is None or unique_coords.numel() <= 0 or int(max_candidates) <= 0:
        return []

    unique_coords = unique_coords.to(dtype=torch.long)
    shifts = getattr(args, "sparsepcgc_actual_oracle_subtree_move_level_shifts", [1, 2])
    if isinstance(shifts, str):
        parsed = []
        for item in shifts.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                parsed.append(int(float(item)))
            except ValueError:
                continue
        shifts = parsed
    shifts = sorted({min(max(int(value), 1), 6) for value in shifts if int(value) >= 1})
    if not shifts:
        shifts = [1]

    min_voxels = max(int(getattr(args, "sparsepcgc_actual_oracle_subtree_move_min_voxels", 4)), 1)
    max_voxels = max(
        int(getattr(args, "sparsepcgc_actual_oracle_subtree_move_max_voxels", 64)),
        min_voxels,
    )
    smoothing = max(float(getattr(args, "leaf_pattern_candidate_smoothing", 1.0)), 0.0)
    memory_weight = max(float(getattr(args, "sparsepcgc_actual_oracle_memory_weight", 0.75)), 0.0)
    size_weight = max(float(getattr(args, "sparsepcgc_actual_oracle_subtree_move_size_weight", 0.02)), 0.0)
    proxy_weight = max(float(getattr(args, "sparsepcgc_codec_proxy_weight", 2.0)), 0.0)
    if base_proxy_bits is None:
        base_proxy_bits = _sparsepcgc_codec_proxy_bits(unique_coords, args)
    pattern_weights = (2 ** torch.arange(8, device=unique_coords.device, dtype=torch.long)).view(1, 8)

    candidate_specs = []
    seen_keys = set()

    for shift in shifts:
        block = int(1 << int(shift))
        node_coords = torch.div(unique_coords, block, rounding_mode="floor")
        super_coords = torch.div(node_coords, 2, rounding_mode="floor")
        child_slot = (
            (node_coords[:, 0] & 1)
            + 2 * (node_coords[:, 1] & 1)
            + 4 * (node_coords[:, 2] & 1)
        ).to(dtype=torch.long)

        unique_super, super_inverse = torch.unique(
            super_coords,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        if unique_super.numel() <= 0:
            continue

        occupancy = torch.zeros((unique_super.shape[0], 8), device=unique_coords.device, dtype=torch.bool)
        occupancy[super_inverse, child_slot] = True
        counts = torch.zeros((unique_super.shape[0], 8), device=unique_coords.device, dtype=torch.long)
        counts.index_put_(
            (super_inverse, child_slot),
            torch.ones_like(child_slot, dtype=torch.long),
            accumulate=True,
        )
        current_code = (occupancy.to(dtype=torch.long) * pattern_weights).sum(dim=1).clamp(0, 255)
        code_hist = torch.bincount(current_code, minlength=256).to(device=unique_coords.device, dtype=torch.float32)
        code_prob = (code_hist + float(smoothing))
        code_prob = code_prob / code_prob.sum().clamp_min(torch.finfo(torch.float32).eps)
        code_nll = -torch.log2(code_prob.clamp_min(torch.finfo(torch.float32).eps))

        for parent_idx in range(int(unique_super.shape[0])):
            code = int(current_code[parent_idx].detach().cpu())
            source_slots = occupancy[parent_idx].nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()
            target_slots = (~occupancy[parent_idx]).nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()
            if not source_slots or not target_slots:
                continue

            for source_slot in source_slots:
                moved_count = int(counts[parent_idx, int(source_slot)].detach().cpu())
                if moved_count < min_voxels or moved_count > max_voxels:
                    continue
                source_bits = unique_coords.new_tensor(
                    [int(source_slot) & 1, (int(source_slot) >> 1) & 1, (int(source_slot) >> 2) & 1],
                    dtype=torch.long,
                )
                source_node = unique_super[parent_idx] * 2 + source_bits
                source_mask = (node_coords == source_node.view(1, 3)).all(dim=1)
                if int(source_mask.sum().detach().cpu()) != moved_count:
                    continue

                for target_slot in target_slots:
                    target_bits = unique_coords.new_tensor(
                        [int(target_slot) & 1, (int(target_slot) >> 1) & 1, (int(target_slot) >> 2) & 1],
                        dtype=torch.long,
                    )
                    target_code = code & (~(1 << int(source_slot)))
                    target_code = target_code | (1 << int(target_slot))
                    key = _sparsepcgc_actual_oracle_transition_key(
                        f"subtree_move_s{int(shift)}",
                        code,
                        int(source_slot),
                        int(target_code),
                    )
                    key = f"{key}:to={int(target_slot)}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    bonus, is_bad, seen_memory = _sparsepcgc_actual_oracle_memory_bonus(args, key)
                    if is_bad:
                        continue

                    source_unique_indices = source_mask.nonzero(as_tuple=False).reshape(-1)

                    gain = float((code_nll[code] - code_nll[int(target_code)]).detach().cpu())
                    preliminary_score = (
                        gain
                        + size_weight * math.log1p(float(moved_count))
                        + memory_weight * float(bonus)
                    )
                    candidate_specs.append(
                        {
                            "op": "subtree_move",
                            "source_unique_indices": source_unique_indices.detach().clone(),
                            "delta": ((target_bits - source_bits).view(1, 3) * int(block)).detach().clone(),
                            "moved_count": int(moved_count),
                            "level_shift": int(shift),
                            "source_slot": int(source_slot),
                            "target_slot": int(target_slot),
                            "source_code": int(code),
                            "target_code": int(target_code),
                            "score": float(preliminary_score),
                            "nll_gain": float(gain),
                            "memory_key": key,
                            "memory_seen": bool(seen_memory),
                        }
                    )

    # Building and proxy-scoring every source/target translation is quadratic
    # in parent patterns. Local NLL is the cheap prefilter; only the handful of
    # candidates that can reach the actual gate materialize a full point cloud.
    candidate_specs.sort(key=lambda item: float(item["score"]), reverse=True)
    candidates = []
    for spec in candidate_specs[: int(max_candidates)]:
        source_unique_indices = spec["source_unique_indices"].to(
            device=unique_coords.device,
            dtype=torch.long,
        )
        transformed = unique_coords.clone()
        transformed[source_unique_indices] = (
            transformed.index_select(0, source_unique_indices)
            + spec["delta"].to(device=unique_coords.device, dtype=torch.long)
        )
        _cand_proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
            transformed,
            args,
            base_proxy_bits,
        )
        item = dict(spec)
        item.pop("delta", None)
        item["transformed_coords"] = transformed.detach()
        item["proxy_percent"] = float(proxy_percent)
        item["score"] = float(item["score"]) - proxy_weight * float(proxy_percent)
        candidates.append(item)
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates


def _sparsepcgc_actual_oracle_pattern_plan_candidates(unique_coords, args, global_step, max_candidates, base_proxy_bits=None):
    if unique_coords is None or unique_coords.numel() <= 0 or int(max_candidates) <= 0:
        return []

    unique_coords = unique_coords.to(dtype=torch.long)
    parent_coords = torch.div(unique_coords, 2, rounding_mode="floor")
    unique_parents, parent_inverse = torch.unique(
        parent_coords,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    if unique_parents.numel() <= 0:
        return []

    child_slot = (
        (unique_coords[:, 0] & 1)
        + 2 * (unique_coords[:, 1] & 1)
        + 4 * (unique_coords[:, 2] & 1)
    ).to(dtype=torch.long)
    occupancy = torch.zeros((unique_parents.shape[0], 8), device=unique_coords.device, dtype=torch.bool)
    occupancy[parent_inverse, child_slot] = True
    pattern_weights = (2 ** torch.arange(8, device=unique_coords.device, dtype=torch.long)).view(1, 8)
    parent_code = (occupancy.to(dtype=torch.long) * pattern_weights).sum(dim=1).clamp(0, 255)

    smoothing = max(float(getattr(args, "leaf_pattern_candidate_smoothing", 1.0)), 0.0)
    code_hist = torch.bincount(parent_code, minlength=256).to(device=unique_coords.device, dtype=torch.float32)
    code_prob = code_hist + float(smoothing)
    code_prob = code_prob / code_prob.sum().clamp_min(torch.finfo(torch.float32).eps)
    code_nll = -torch.log2(code_prob.clamp_min(torch.finfo(torch.float32).eps))

    target_topk = max(int(getattr(args, "sparsepcgc_actual_oracle_pattern_plan_target_topk", 16)), 1)
    max_edits = max(int(getattr(args, "sparsepcgc_actual_oracle_pattern_plan_max_edits", 16)), 1)
    max_edits = min(max_edits, max(int(getattr(args, "sparsepcgc_actual_oracle_max_selected_voxels", 16)), 1))
    min_gain = float(getattr(args, "sparsepcgc_actual_oracle_pattern_plan_min_nll_gain", 0.0))
    edit_penalty = max(float(getattr(args, "sparsepcgc_actual_oracle_pattern_plan_edit_penalty", 0.02)), 0.0)
    memory_weight = max(float(getattr(args, "sparsepcgc_actual_oracle_memory_weight", 0.75)), 0.0)
    proxy_weight = max(float(getattr(args, "sparsepcgc_codec_proxy_weight", 2.0)), 0.0)
    if base_proxy_bits is None:
        base_proxy_bits = _sparsepcgc_codec_proxy_bits(unique_coords, args)

    popular_codes = torch.argsort(code_prob, descending=True)
    popular_codes = [int(code) for code in popular_codes.detach().cpu().tolist() if int(code) > 0][:target_topk]
    if not popular_codes:
        return []

    def _slot_bits(slot, *, device):
        slot = int(slot)
        return torch.tensor([slot & 1, (slot >> 1) & 1, (slot >> 2) & 1], device=device, dtype=torch.long)

    candidates = []
    seen_keys = set()
    for parent_idx in range(int(unique_parents.shape[0])):
        source_code = int(parent_code[parent_idx].detach().cpu())
        if source_code <= 0:
            continue
        current_slots = [int(slot) for slot in occupancy[parent_idx].nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()]
        if not current_slots:
            continue
        current_slot_set = set(current_slots)
        parent_unique_indices = (parent_inverse == parent_idx).nonzero(as_tuple=False).reshape(-1)
        if parent_unique_indices.numel() <= 0:
            continue

        for target_code in popular_codes:
            if int(target_code) == int(source_code):
                continue
            target_slots = [slot for slot in range(8) if (int(target_code) & (1 << slot))]
            if not target_slots:
                continue
            target_slot_set = set(target_slots)
            drop_slots = sorted(current_slot_set - target_slot_set)
            add_slots = sorted(target_slot_set - current_slot_set)
            edit_count = len(drop_slots) + len(add_slots)
            if edit_count <= 0 or edit_count > max_edits:
                continue

            gain = float((code_nll[source_code] - code_nll[int(target_code)]).detach().cpu())
            if gain < min_gain:
                continue

            key = _sparsepcgc_actual_oracle_transition_key(
                "pattern_plan",
                source_code,
                edit_count,
                int(target_code),
            )
            key = f"{key}:drop={','.join(map(str, drop_slots))}:add={','.join(map(str, add_slots))}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bonus, is_bad, seen_memory = _sparsepcgc_actual_oracle_memory_bonus(args, key)
            if is_bad:
                continue

            drop_unique_indices = []
            for slot in drop_slots:
                slot_indices = parent_unique_indices[child_slot.index_select(0, parent_unique_indices) == int(slot)]
                if slot_indices.numel() > 0:
                    drop_unique_indices.append(int(slot_indices[0].detach().cpu()))
            if len(drop_unique_indices) != len(drop_slots):
                continue

            add_items = []
            source_pool_coords = unique_coords.index_select(0, parent_unique_indices)
            for slot in add_slots:
                target_coord = unique_parents[parent_idx] * 2 + _slot_bits(slot, device=unique_coords.device)
                dist = (source_pool_coords - target_coord.view(1, 3)).abs().sum(dim=1)
                nearest_local = int(torch.argmin(dist).detach().cpu())
                source_unique_idx = int(parent_unique_indices[nearest_local].detach().cpu())
                add_items.append(
                    {
                        "source_unique_idx": source_unique_idx,
                        "target_child_slot": int(slot),
                        "target_coord": target_coord.detach().clone(),
                    }
                )

            keep_unique = torch.ones((unique_coords.shape[0],), device=unique_coords.device, dtype=torch.bool)
            if drop_unique_indices:
                keep_unique[torch.as_tensor(drop_unique_indices, device=unique_coords.device, dtype=torch.long)] = False
            transformed = unique_coords[keep_unique]
            if add_items:
                transformed = torch.cat(
                    [transformed] + [item["target_coord"].view(1, 3) for item in add_items],
                    dim=0,
                )
                transformed = torch.unique(transformed, dim=0, sorted=True)
            if int(transformed.shape[0]) <= 0:
                continue

            _cand_proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                transformed,
                args,
                base_proxy_bits,
            )
            score = (
                gain
                - proxy_weight * float(proxy_percent)
                + memory_weight * float(bonus)
                - edit_penalty * float(edit_count)
            )
            candidates.append(
                {
                    "op": "pattern_plan",
                    "transformed_coords": transformed.detach().clone(),
                    "drop_unique_indices": drop_unique_indices,
                    "add_items": add_items,
                    "drop_count": len(drop_unique_indices),
                    "add_count": len(add_items),
                    "edit_count": int(edit_count),
                    "source_code": int(source_code),
                    "target_code": int(target_code),
                    "score": float(score),
                    "nll_gain": float(gain),
                    "proxy_percent": float(proxy_percent),
                    "memory_key": key,
                    "memory_seen": bool(seen_memory),
                }
            )

    multi_parent_max = max(
        int(getattr(args, "sparsepcgc_actual_oracle_pattern_plan_multi_parent_max", 8)),
        1,
    )
    if multi_parent_max > 1 and max_edits > 1:
        parent_plans_by_target = {int(code): [] for code in popular_codes}
        for parent_idx in range(int(unique_parents.shape[0])):
            source_code = int(parent_code[parent_idx].detach().cpu())
            if source_code <= 0:
                continue
            current_slots = [
                int(slot)
                for slot in occupancy[parent_idx].nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()
            ]
            if not current_slots:
                continue
            current_slot_set = set(current_slots)
            parent_unique_indices = (parent_inverse == parent_idx).nonzero(as_tuple=False).reshape(-1)
            if parent_unique_indices.numel() <= 0:
                continue
            source_pool_coords = unique_coords.index_select(0, parent_unique_indices)

            for target_code in popular_codes:
                if int(target_code) == int(source_code):
                    continue
                target_slots = [slot for slot in range(8) if (int(target_code) & (1 << slot))]
                if not target_slots:
                    continue
                target_slot_set = set(target_slots)
                drop_slots = sorted(current_slot_set - target_slot_set)
                add_slots = sorted(target_slot_set - current_slot_set)
                edit_count = len(drop_slots) + len(add_slots)
                if edit_count <= 0 or edit_count > max_edits:
                    continue
                gain = float((code_nll[source_code] - code_nll[int(target_code)]).detach().cpu())
                if gain < min_gain:
                    continue

                drop_unique_indices = []
                for slot in drop_slots:
                    slot_indices = parent_unique_indices[child_slot.index_select(0, parent_unique_indices) == int(slot)]
                    if slot_indices.numel() > 0:
                        drop_unique_indices.append(int(slot_indices[0].detach().cpu()))
                if len(drop_unique_indices) != len(drop_slots):
                    continue

                add_items = []
                for slot in add_slots:
                    target_coord = unique_parents[parent_idx] * 2 + _slot_bits(slot, device=unique_coords.device)
                    dist = (source_pool_coords - target_coord.view(1, 3)).abs().sum(dim=1)
                    nearest_local = int(torch.argmin(dist).detach().cpu())
                    source_unique_idx = int(parent_unique_indices[nearest_local].detach().cpu())
                    add_items.append(
                        {
                            "source_unique_idx": source_unique_idx,
                            "target_child_slot": int(slot),
                            "target_coord": target_coord.detach().clone(),
                        }
                    )

                parent_plans_by_target[int(target_code)].append(
                    {
                        "parent_idx": int(parent_idx),
                        "source_code": int(source_code),
                        "target_code": int(target_code),
                        "drop_unique_indices": drop_unique_indices,
                        "add_items": add_items,
                        "edit_count": int(edit_count),
                        "gain": float(gain),
                    }
                )

        for target_code, parent_plans in parent_plans_by_target.items():
            if len(parent_plans) < 2:
                continue
            parent_plans = sorted(parent_plans, key=lambda item: float(item["gain"]), reverse=True)
            selected_plans = []
            total_edits = 0
            total_gain = 0.0
            for plan in parent_plans:
                if len(selected_plans) >= multi_parent_max:
                    break
                if total_edits + int(plan["edit_count"]) > max_edits:
                    continue
                selected_plans.append(plan)
                total_edits += int(plan["edit_count"])
                total_gain += float(plan["gain"])
            if len(selected_plans) < 2 or total_edits <= 0:
                continue

            drop_unique_indices = []
            add_items = []
            source_codes = []
            for plan in selected_plans:
                drop_unique_indices.extend(int(v) for v in plan["drop_unique_indices"])
                add_items.extend(plan["add_items"])
                source_codes.append(int(plan["source_code"]))

            key = (
                f"pattern_plan_multi:target={int(target_code)}:"
                f"parents={len(selected_plans)}:edits={int(total_edits)}:"
                f"sources={','.join(map(str, source_codes[:8]))}"
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bonus, is_bad, seen_memory = _sparsepcgc_actual_oracle_memory_bonus(args, key)
            if is_bad:
                continue

            keep_unique = torch.ones((unique_coords.shape[0],), device=unique_coords.device, dtype=torch.bool)
            if drop_unique_indices:
                keep_unique[torch.as_tensor(drop_unique_indices, device=unique_coords.device, dtype=torch.long)] = False
            transformed = unique_coords[keep_unique]
            if add_items:
                transformed = torch.cat(
                    [transformed] + [item["target_coord"].view(1, 3) for item in add_items],
                    dim=0,
                )
                transformed = torch.unique(transformed, dim=0, sorted=True)
            if int(transformed.shape[0]) <= 0:
                continue

            _cand_proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                transformed,
                args,
                base_proxy_bits,
            )
            score = (
                total_gain
                - proxy_weight * float(proxy_percent)
                + memory_weight * float(bonus)
                - edit_penalty * float(total_edits)
            )
            candidates.append(
                {
                    "op": "pattern_plan",
                    "transformed_coords": transformed.detach().clone(),
                    "drop_unique_indices": drop_unique_indices,
                    "add_items": add_items,
                    "drop_count": len(drop_unique_indices),
                    "add_count": len(add_items),
                    "edit_count": int(total_edits),
                    "source_code": int(source_codes[0]) if source_codes else 0,
                    "target_code": int(target_code),
                    "score": float(score),
                    "nll_gain": float(total_gain),
                    "proxy_percent": float(proxy_percent),
                    "memory_key": key,
                    "memory_seen": bool(seen_memory),
                    "multi_parent_count": int(len(selected_plans)),
                }
            )

    candidates = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
    return candidates[: int(max_candidates)]


def _sparsepcgc_rows_membership_mask_fast(query_n3, table_n3):
    if (
        not torch.is_tensor(query_n3)
        or not torch.is_tensor(table_n3)
        or query_n3.numel() <= 0
        or table_n3.numel() <= 0
    ):
        device = query_n3.device if torch.is_tensor(query_n3) else torch.device("cpu")
        query_rows = int(query_n3.reshape(-1, 3).shape[0]) if torch.is_tensor(query_n3) and query_n3.numel() > 0 else 0
        return torch.zeros((query_rows,), device=device, dtype=torch.bool)

    query = query_n3.to(dtype=torch.long).reshape(-1, 3).contiguous()
    table = table_n3.to(device=query.device, dtype=torch.long).reshape(-1, 3).contiguous()
    if query.numel() <= 0 or table.numel() <= 0:
        return torch.zeros((query.shape[0],), device=query.device, dtype=torch.bool)

    both = torch.cat([query, table], dim=0)
    mins = both.amin(dim=0)
    span = (both.amax(dim=0) - mins + 1).clamp_min(1)

    def _key(values):
        shifted = values - mins
        return shifted[:, 0] * span[1] * span[2] + shifted[:, 1] * span[2] + shifted[:, 2]

    table_keys = torch.unique(_key(table), sorted=True)
    query_keys = _key(query)
    pos = torch.searchsorted(table_keys, query_keys)
    in_bounds = pos < table_keys.numel()
    safe_pos = pos.clamp(max=max(int(table_keys.numel()) - 1, 0))
    return in_bounds & (table_keys[safe_pos] == query_keys)


def _sparsepcgc_prepare_full_cloud_splice_base(full_coords_b3n, subtree_coords_n3):
    if not torch.is_tensor(full_coords_b3n) or not torch.is_tensor(subtree_coords_n3):
        return None

    if full_coords_b3n.ndim == 2:
        full_coords_b3n = (
            full_coords_b3n.transpose(0, 1).contiguous().unsqueeze(0)
            if full_coords_b3n.shape[-1] == 3
            else full_coords_b3n.unsqueeze(0)
        )
    if full_coords_b3n.ndim != 3 or full_coords_b3n.shape[1] != 3 or full_coords_b3n.shape[0] != 1:
        return None

    device = subtree_coords_n3.device
    full_coords = torch.unique(
        full_coords_b3n[0].transpose(0, 1).contiguous().to(device=device, dtype=torch.long),
        dim=0,
        sorted=True,
    )
    subtree_coords = torch.unique(
        subtree_coords_n3.to(device=device, dtype=torch.long).reshape(-1, 3).contiguous(),
        dim=0,
        sorted=True,
    )
    if full_coords.numel() <= 0 or subtree_coords.numel() <= 0:
        return None

    keep_mask = ~_sparsepcgc_rows_membership_mask_fast(full_coords, subtree_coords)
    full_without_subtree = full_coords[keep_mask]
    return {
        "full_coords": full_coords,
        "subtree_coords": subtree_coords,
        "full_without_subtree": full_without_subtree,
    }


def _sparsepcgc_splice_subtree_coords_into_full_cloud(full_coords_b3n, subtree_coords_n3, candidate_coords_n3, splice_base=None):
    if not torch.is_tensor(full_coords_b3n) or not torch.is_tensor(subtree_coords_n3) or not torch.is_tensor(candidate_coords_n3):
        return None
    device = candidate_coords_n3.device
    candidate_coords = torch.unique(
        candidate_coords_n3.to(device=device, dtype=torch.long).reshape(-1, 3).contiguous(),
        dim=0,
        sorted=True,
    )
    if candidate_coords.numel() <= 0:
        return None

    prepared = splice_base
    if not isinstance(prepared, dict):
        prepared = _sparsepcgc_prepare_full_cloud_splice_base(full_coords_b3n, subtree_coords_n3)
    if not isinstance(prepared, dict):
        return None

    full_without_subtree = prepared.get("full_without_subtree", None)
    if not torch.is_tensor(full_without_subtree):
        return None
    full_without_subtree = full_without_subtree.to(device=device, dtype=torch.long)
    spliced = torch.cat([full_without_subtree, candidate_coords], dim=0)
    spliced = torch.unique(spliced, dim=0, sorted=True)
    if int(spliced.shape[0]) <= 0:
        return None
    return spliced

def _attach_sparsepcgc_actual_oracle_drop(
    *,
    args,
    writer,
    loss,
    subtree_tree,
    full_octree_context,
    subtree_xyz,
    cache_key,
    global_step,
):
    if str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() == "ana_den6_online":
        raise RuntimeError(
            "ana_den6_online主経路にlegacy actual oracleは入れない。"
            "1 Stepにつきnetworkが決めた1 planだけをactual encodeすること。"
        )
    debug = {
        "enabled": False,
        "used": False,
        "candidate_count": 0,
        "candidate_pool_count": 0,
        "tested_count": 0,
        "bad_candidate_count": 0,
        "improving_candidate_count": 0,
        "combo_extra_count": 0,
        "joint_tested_count": 0,
        "joint_improving_count": 0,
        "group_tested_count": 0,
        "group_improving_count": 0,
        "parent_prune_tested_count": 0,
        "parent_prune_improving_count": 0,
        "pattern_plan_tested_count": 0,
        "pattern_plan_improving_count": 0,
        "subtree_move_tested_count": 0,
        "subtree_move_improving_count": 0,
        "selected_move_count": 0,
        "override_final_voxel_coords": None,
        "override_move_count": 0,
        "override_drop_count": 0,
        "override_subtree_prune_count": 0,
        "override_scope": "",
        "cached_edited_actual_stats": None,
        "best_percent": 0.0,
        "best_raw_percent": 0.0,
        "best_edit_record_bits": 0.0,
        "selected_raw_percent": 0.0,
        "selected_edit_record_bits": 0.0,
        "original_actual_bits": 0.0,
        "edited_actual_bits": 0.0,
        "delta_actual_percent": 0.0,
        "best_actual_percent": 0.0,
        "best_proxy_percent": 0.0,
        "selected_proxy_percent": 0.0,
        "selected_geometry_percent": 0.0,
        "generated_candidate_count": 0,
        "accepted_candidate_count": 0,
        "accepted_prune_count": 0,
        "accepted_add_count": 0,
        "accepted_adjust_count": 0,
        "accepted_subtree_move_count": 0,
        "accepted_parent_collapse_count": 0,
        "accepted_pattern_canonicalize_count": 0,
        "noop_label_count": 0,
        "noop_label_weight": float(getattr(args, "sparsepcgc_actual_oracle_noop_weight", 0.02)),
        "high_rate_mppov_count": 0,
        "low_prob_occupied_count": 0,
        "single_child_chain_count": 0,
        "context_pattern_candidate_count": 0,
        "actual_oracle_time": 0.0,
        "actual_eval_max": 0,
        "edit_record_effective_scale": float(sparsepcgc_effective_edit_record_bit_scale(args)),
        "reason": "disabled",
    }

    if not bool(getattr(args, "sparsepcgc_actual_oracle_edit", False)):
        return subtree_tree, full_octree_context, debug
    compress_key = str(getattr(args, "compress", "")).strip().lower().replace("_", "").replace("-", "")
    backend_key = str(getattr(args, "compression_loss_backend", "")).strip().lower().replace("_", "").replace("-", "")
    if compress_key != "sparsepcgc" and not backend_key.startswith("sparsepcgc"):
        debug["reason"] = "not_sparsepcgc"
        return subtree_tree, full_octree_context, debug

    interval = int(getattr(args, "sparsepcgc_actual_oracle_interval", 1))
    actual_validate_this_step = interval > 0 and ((int(global_step) + 1) % interval) == 0
    fast_diagnostic_enabled = bool(getattr(args, "sparsepcgc_fast_diagnostic_teacher", True))
    fast_diagnostic_unvalidated_teacher = bool(
        getattr(args, "sparsepcgc_fast_diagnostic_allow_unvalidated_teacher", False)
    )
    if (not actual_validate_this_step) and (not fast_diagnostic_enabled):
        debug["reason"] = "interval_skip"
        return subtree_tree, full_octree_context, debug
    if (not actual_validate_this_step) and fast_diagnostic_enabled and (not fast_diagnostic_unvalidated_teacher):
        debug["reason"] = "interval_skip_fast_diagnostic_requires_full_actual"
        debug["fast_diagnostic_enabled"] = True
        debug["fast_diagnostic_unvalidated_teacher_allowed"] = False
        return subtree_tree, full_octree_context, debug

    max_candidates = max(int(getattr(args, "sparsepcgc_actual_oracle_max_candidates", 0)), 0)
    if max_candidates <= 0:
        debug["reason"] = "max_candidates_zero"
        return subtree_tree, full_octree_context, debug

    if not isinstance(subtree_tree, dict):
        debug["reason"] = "subtree_tree_missing"
        return subtree_tree, full_octree_context, debug

    coords = subtree_tree.get("global_voxel_coords", None)
    if not torch.is_tensor(coords):
        debug["reason"] = "global_voxel_coords_missing"
        return subtree_tree, full_octree_context, debug
    if coords.ndim == 2:
        coords = coords.transpose(0, 1).contiguous().unsqueeze(0) if coords.shape[-1] == 3 else coords.unsqueeze(0)
    if coords.ndim != 3:
        debug["reason"] = f"invalid_coords_ndim={coords.ndim}"
        return subtree_tree, full_octree_context, debug
    if coords.shape[1] != 3 and coords.shape[-1] == 3:
        coords = coords.permute(0, 2, 1).contiguous()
    if coords.ndim != 3 or coords.shape[1] != 3 or coords.shape[0] != 1:
        debug["reason"] = f"invalid_coords_shape={tuple(coords.shape)}"
        return subtree_tree, full_octree_context, debug

    coords = coords.detach().to(device=subtree_xyz.device, dtype=torch.long)
    coords_n3 = coords[0].transpose(0, 1).contiguous()
    add_candidate_ratio = min(
        max(float(getattr(args, "sparsepcgc_actual_oracle_add_candidate_ratio", 0.50)), 0.0),
        1.0,
    )
    add_budget = int(round(float(max_candidates) * add_candidate_ratio))
    add_budget = min(max(add_budget, 0), max_candidates)
    drop_budget = max_candidates - add_budget
    if not bool(getattr(args, "sparsepcgc_actual_oracle_allow_add", True)):
        drop_budget = max_candidates
        add_budget = 0
    if not bool(getattr(args, "sparsepcgc_actual_oracle_allow_prune", True)):
        add_budget = max_candidates
        drop_budget = 0

    group_pool_voxels = min(
        max(int(getattr(args, "sparsepcgc_actual_oracle_group_voxels", 4)), 2),
        max(int(getattr(args, "sparsepcgc_actual_oracle_max_selected_voxels", 4)), 1),
    )
    group_candidate_max = max(int(getattr(args, "sparsepcgc_actual_oracle_group_candidate_max", 0)), 0)
    drop_pool_budget = drop_budget
    add_pool_budget = add_budget
    if group_candidate_max > 0:
        if bool(getattr(args, "sparsepcgc_actual_oracle_allow_prune", True)):
            drop_pool_budget = max(drop_pool_budget, group_pool_voxels)
        if bool(getattr(args, "sparsepcgc_actual_oracle_allow_add", True)):
            add_pool_budget = max(add_pool_budget, group_pool_voxels)

    unique_coords, inverse = torch.unique(
        coords_n3.to(dtype=torch.long),
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    oracle_splice_base = None
    proxy_profile = {
        "enabled": False,
        "reason": "skipped_fast_diagnostic_only",
        "base_proxy_bits": 0.0,
    }
    base_proxy_bits = 0.0
    candidate_pool = []
    candidate_indices = []
    add_candidate_pool = []
    add_candidates = []

    full_eval_coords = None
    oracle_eval_scope = "subtree_local"
    if bool(getattr(args, "sparsepcgc_actual_oracle_eval_full_cloud_splice", True)) and isinstance(full_octree_context, dict):
        splice_base_prepare_start = time.time()
        full_cloud_cache_key = str(full_octree_context.get("actual_oracle_full_cloud_cache_key", "") or "")
        splice_base_cache_key = _episode_input_common_cache_key(
            cache_key or full_cloud_cache_key,
            "actual_oracle_splice_base",
            subtree_points=int(unique_coords.shape[0]),
        )
        splice_base_cache_enabled = bool(
            _episode_input_common_cache_enabled(args)
            and getattr(args, "episode_input_actual_oracle_splice_cache", True)
            and splice_base_cache_key
        )
        if splice_base_cache_enabled:
            oracle_splice_base = _episode_input_common_cache_fetch(
                args,
                splice_base_cache_key,
                device=unique_coords.device,
                section="actual_oracle_splice_base",
            )
        if isinstance(oracle_splice_base, dict):
            debug["actual_oracle_splice_base_cache_hit"] = True
        else:
            oracle_splice_base = _sparsepcgc_prepare_full_cloud_splice_base(
                full_octree_context.get("full_global_voxel_coords", None),
                unique_coords,
            )
            debug["actual_oracle_splice_base_cache_hit"] = False
            if splice_base_cache_enabled and isinstance(oracle_splice_base, dict):
                _episode_input_common_cache_store(
                    args,
                    splice_base_cache_key,
                    oracle_splice_base,
                )
        debug["actual_oracle_splice_base_prepare_time"] = float(time.time() - splice_base_prepare_start)
        spliced_base = _sparsepcgc_splice_subtree_coords_into_full_cloud(
            full_octree_context.get("full_global_voxel_coords", None),
            unique_coords,
            unique_coords,
            splice_base=oracle_splice_base,
        )
        if torch.is_tensor(spliced_base) and int(spliced_base.shape[0]) > 0:
            full_eval_coords = spliced_base.detach()
            oracle_eval_scope = "full_cloud_splice"
    debug["actual_eval_scope"] = str(oracle_eval_scope)
    debug["actual_eval_full_coord_count"] = int(full_eval_coords.shape[0]) if torch.is_tensor(full_eval_coords) else 0
    full_cloud_teacher_required = bool(
        getattr(args, "sparsepcgc_require_full_cloud_actual_teacher", True)
    )
    missing_full_cloud_teacher_eval = bool(
        actual_validate_this_step
        and full_cloud_teacher_required
        and str(oracle_eval_scope) != "full_cloud_splice"
    )
    debug["full_cloud_teacher_required"] = bool(full_cloud_teacher_required)
    debug["full_cloud_teacher_eval_available"] = not bool(missing_full_cloud_teacher_eval)

    early_actual_eval_max = max(
        int(getattr(args, "sparsepcgc_actual_oracle_actual_eval_max", max_candidates)),
        0,
    )
    if early_actual_eval_max <= 0:
        early_actual_eval_max = max_candidates
    early_aux_probe_interval = max(
        int(getattr(args, "sparsepcgc_actual_oracle_aux_probe_interval", 6)),
        0,
    )
    early_aux_probe_due = (
        early_aux_probe_interval > 0
        and (int(global_step) + 1) % int(early_aux_probe_interval) == 0
    )
    early_full_cloud_macro_max = max(
        int(getattr(args, "sparsepcgc_actual_oracle_full_cloud_macro_prune_candidate_max", 1)),
        0,
    )
    early_full_macro_fail_extra_eval_max = max(
        int(getattr(args, "sparsepcgc_actual_oracle_full_macro_fail_extra_eval_max", 2)),
        0,
    )
    early_full_macro_fallback_enabled = bool(
        getattr(args, "sparsepcgc_actual_oracle_fallback_after_full_macro_fail", True)
        and early_full_cloud_macro_max > 0
        and early_full_macro_fail_extra_eval_max > 0
    )
    skip_unused_local_candidate_generation = bool(
        bool(getattr(args, "sparsepcgc_actual_oracle_skip_unused_local_candidates", True))
        and actual_validate_this_step
        and (not early_aux_probe_due)
        and torch.is_tensor(full_eval_coords)
        and bool(getattr(args, "sparsepcgc_actual_oracle_prioritize_full_cloud_macro", True))
        and early_full_cloud_macro_max > 0
        and int(early_actual_eval_max) <= int(early_full_cloud_macro_max)
    )
    debug["skip_unused_local_candidate_generation"] = bool(skip_unused_local_candidate_generation)
    debug["full_cloud_macro_fallback_candidate_generation_enabled"] = bool(early_full_macro_fallback_enabled)
    debug["full_cloud_macro_fail_extra_eval_max"] = int(early_full_macro_fail_extra_eval_max)

    local_candidate_generation_done = False
    
    def _ensure_local_candidate_generation():
        nonlocal candidate_pool, candidate_indices, add_candidate_pool, add_candidates
        nonlocal unique_coords, inverse, local_candidate_generation_done
        if local_candidate_generation_done:
            return
        local_candidate_generation_done = True
        local_candidate_start = time.time()
        candidate_pool, unique_coords_from_pool, inverse_from_pool = _sparsepcgc_actual_oracle_candidate_indices(
            coords_n3,
            args,
            global_step,
            drop_pool_budget,
            proxy_profile=proxy_profile,
        )
        if unique_coords_from_pool is not None and inverse_from_pool is not None:
            unique_coords = unique_coords_from_pool
            inverse = inverse_from_pool
        candidate_indices = candidate_pool[:drop_budget]
        add_candidate_pool = _sparsepcgc_actual_oracle_add_candidates(
            unique_coords,
            args,
            global_step,
            add_pool_budget,
            proxy_profile=proxy_profile,
        )
        add_candidates = add_candidate_pool[:add_budget]
        debug["local_candidate_generation_time"] = float(
            debug.get("local_candidate_generation_time", 0.0)
            + (time.time() - local_candidate_start)
        )
        debug["local_candidate_generation_lazy"] = bool(skip_unused_local_candidate_generation)
        debug["candidate_count"] = int(len(candidate_indices) + len(add_candidates))
        debug["candidate_pool_count"] = int(len(candidate_pool) + len(add_candidate_pool))
        if skip_unused_local_candidate_generation:
            debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + int(
                debug["candidate_pool_count"]
            )

    if actual_validate_this_step:
        proxy_profile = _sparsepcgc_codec_proxy_profile(unique_coords, args)
        base_proxy_bits = float(proxy_profile.get("base_proxy_bits", 0.0) or 0.0)
        debug["high_rate_mppov_count"] = int(proxy_profile.get("high_rate_mppov_count", 0) or 0)
        debug["low_prob_occupied_count"] = int(proxy_profile.get("low_prob_occupied_count", 0) or 0)
        debug["single_child_chain_count"] = int(proxy_profile.get("single_child_chain_count", 0) or 0)
        debug["context_pattern_candidate_count"] = int(proxy_profile.get("context_pattern_candidate_count", 0) or 0)

        if not skip_unused_local_candidate_generation:
            _ensure_local_candidate_generation()
    debug["enabled"] = True
    debug["candidate_count"] = int(len(candidate_indices) + len(add_candidates))
    debug["candidate_pool_count"] = int(len(candidate_pool) + len(add_candidate_pool))
    debug["generated_candidate_count"] = int(debug["candidate_pool_count"])

    def _oracle_actual_eval_coords(local_candidate_coords):
        if torch.is_tensor(full_eval_coords):
            spliced = _sparsepcgc_splice_subtree_coords_into_full_cloud(
                full_octree_context.get("full_global_voxel_coords", None) if isinstance(full_octree_context, dict) else None,
                unique_coords,
                local_candidate_coords,
                splice_base=oracle_splice_base,
            )
            if torch.is_tensor(spliced) and int(spliced.shape[0]) > 0:
                return spliced
        return local_candidate_coords

    def _oracle_actual_eval_xyz(local_candidate_coords):
        eval_coords = _oracle_actual_eval_coords(local_candidate_coords)
        return _restore_codec_xyz_from_global_voxels(
            args,
            eval_coords.transpose(0, 1).contiguous().unsqueeze(0),
            full_octree_context if isinstance(full_octree_context, dict) else subtree_tree,
            subtree_xyz,
        )

    edit_record_unique_count = int(full_eval_coords.shape[0]) if torch.is_tensor(full_eval_coords) else int(unique_coords.shape[0])

    point_mask = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.bool)
    score = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.float32)
    add_point_mask = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.bool)
    add_score = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.float32)
    add_child_slot = torch.full((1, coords.shape[-1]), -1, device=coords.device, dtype=torch.long)
    add_direction_index = torch.full((1, coords.shape[-1]), -1, device=coords.device, dtype=torch.long)
    move_point_mask = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.bool)
    move_score = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.float32)
    move_direction_index = torch.full((1, coords.shape[-1]), -1, device=coords.device, dtype=torch.long)
    bad_drop_point_mask = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.bool)
    bad_drop_score = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.float32)
    bad_add_point_mask = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.bool)
    bad_add_score = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.float32)
    bad_add_child_slot = torch.full((1, coords.shape[-1]), -1, device=coords.device, dtype=torch.long)
    bad_add_direction_index = torch.full((1, coords.shape[-1]), -1, device=coords.device, dtype=torch.long)
    bad_move_point_mask = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.bool)
    bad_move_score = torch.zeros((1, coords.shape[-1]), device=coords.device, dtype=torch.float32)
    bad_move_direction_index = torch.full((1, coords.shape[-1]), -1, device=coords.device, dtype=torch.long)
    bad_candidate_count = 0

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    neighbor_offset_to_index = {offset: idx for idx, offset in enumerate(neighbor_offsets)}

    def _neighbor_direction_index(delta):
        if not torch.is_tensor(delta) or delta.numel() != 3:
            return -1
        direction = tuple(int(v) for v in delta.detach().reshape(-1).sign().cpu().tolist())
        return int(neighbor_offset_to_index.get(direction, -1))

    fast_diagnostic_indices = []
    fast_diagnostic_debug = {}
    fast_diagnostic_add_items = []
    fast_diagnostic_add_debug = {}
    if fast_diagnostic_enabled and isinstance(full_octree_context, dict):
        fast_full_coords = full_octree_context.get("full_global_voxel_coords", None)
        if not torch.is_tensor(fast_full_coords):
            fast_full_coords = full_eval_coords
        fast_diagnostic_indices, fast_diagnostic_debug = _sparsepcgc_fast_diagnostic_prune_indices(
            unique_coords,
            fast_full_coords,
            args,
        )
        fast_diagnostic_add_items, fast_diagnostic_add_debug = _sparsepcgc_fast_diagnostic_add_candidates(
            unique_coords,
            fast_full_coords,
            args,
        )
        debug["fast_diagnostic_enabled"] = True
        debug["fast_diagnostic_name"] = str(fast_diagnostic_debug.get("diagnostic", ""))
        debug["fast_diagnostic_threshold"] = int(fast_diagnostic_debug.get("threshold", 0) or 0)
        debug["fast_diagnostic_full_drop_count"] = int(fast_diagnostic_debug.get("global_drop_count", 0) or 0)
        debug["fast_diagnostic_local_drop_count"] = int(fast_diagnostic_debug.get("local_drop_count", 0) or 0)
        debug["fast_diagnostic_full_drop_ratio"] = float(fast_diagnostic_debug.get("global_drop_ratio", 0.0) or 0.0)
        debug["fast_diagnostic_local_drop_ratio"] = float(fast_diagnostic_debug.get("local_drop_ratio", 0.0) or 0.0)
        debug["fast_diagnostic_add_name"] = str(fast_diagnostic_add_debug.get("diagnostic", ""))
        debug["fast_diagnostic_add_threshold"] = int(fast_diagnostic_add_debug.get("threshold", 0) or 0)
        debug["fast_diagnostic_full_add_count"] = int(fast_diagnostic_add_debug.get("global_add_count", 0) or 0)
        debug["fast_diagnostic_local_add_count"] = int(fast_diagnostic_add_debug.get("local_add_count", 0) or 0)
        debug["fast_diagnostic_full_add_ratio"] = float(fast_diagnostic_add_debug.get("global_add_ratio", 0.0) or 0.0)
        debug["fast_diagnostic_local_add_ratio"] = float(fast_diagnostic_add_debug.get("local_add_ratio", 0.0) or 0.0)
    else:
        debug["fast_diagnostic_enabled"] = False

    def _apply_fast_diagnostic_teacher():
        if not fast_diagnostic_indices and not fast_diagnostic_add_items:
            return False
        selected_indices = [int(v) for v in fast_diagnostic_indices]
        selected_index_set = set(selected_indices)
        strength = 1.0 + min(float(len(selected_indices)) / max(float(unique_coords.shape[0]), 1.0) * 100.0, 4.0)
        for unique_idx in selected_indices:
            if unique_idx < 0 or unique_idx >= int(unique_coords.shape[0]):
                continue
            point_indices = (inverse == int(unique_idx)).nonzero(as_tuple=False).reshape(-1)
            if point_indices.numel() <= 0:
                continue
            point_mask[0, point_indices] = True
            score[0, point_indices] = float(strength)
        selected_drop = int(point_mask.detach().sum().cpu())

        selected_add_items = []
        add_strength = 1.0 + min(
            float(len(fast_diagnostic_add_items)) / max(float(unique_coords.shape[0]), 1.0) * 100.0,
            2.0,
        )
        for add_item in fast_diagnostic_add_items:
            source_unique_idx = int(add_item.get("source_unique_idx", -1))
            target_slot = int(add_item.get("target_child_slot", -1))
            target_coord = add_item.get("target_coord", None)
            if (
                source_unique_idx < 0
                or source_unique_idx >= int(unique_coords.shape[0])
                or source_unique_idx in selected_index_set
                or target_slot < 0
                or target_slot > 7
                or not torch.is_tensor(target_coord)
            ):
                continue
            point_indices = (inverse == int(source_unique_idx)).nonzero(as_tuple=False).reshape(-1)
            if point_indices.numel() <= 0:
                continue
            add_point_mask[0, point_indices] = True
            add_score[0, point_indices] = torch.maximum(
                add_score[0, point_indices],
                add_score.new_full((int(point_indices.numel()),), float(add_strength)),
            )
            add_child_slot[0, point_indices] = int(target_slot)
            selected_add_items.append(
                {
                    "source_unique_idx": int(source_unique_idx),
                    "target_child_slot": int(target_slot),
                    "target_coord": target_coord.to(device=unique_coords.device, dtype=torch.long).view(1, 3),
                }
            )
        selected_add = int(add_point_mask.detach().sum().cpu())
        if selected_drop <= 0 and selected_add <= 0:
            return False
        keep_for_proxy = torch.ones((unique_coords.shape[0],), device=unique_coords.device, dtype=torch.bool)
        if selected_indices:
            keep_for_proxy[torch.as_tensor(selected_indices, device=unique_coords.device, dtype=torch.long)] = False
        edited_for_proxy = unique_coords[keep_for_proxy]
        if selected_add_items:
            add_coords_for_proxy = torch.cat([item["target_coord"] for item in selected_add_items], dim=0)
            edited_for_proxy = torch.unique(torch.cat([edited_for_proxy, add_coords_for_proxy], dim=0), dim=0, sorted=True)
        skip_fast_proxy_eval = bool(
            (not actual_validate_this_step)
            and getattr(args, "sparsepcgc_fast_diagnostic_skip_proxy_eval", True)
        )
        if skip_fast_proxy_eval:
            before_count = max(float(unique_coords.shape[0]), 1.0)
            after_count = float(edited_for_proxy.shape[0])
            proxy_bits = float(after_count)
            proxy_percent = float(100.0 * (after_count - before_count) / before_count)
        else:
            proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                edited_for_proxy,
                args,
                base_proxy_bits,
            )
        edit_record_bits = _sparsepcgc_edit_record_total_bits(
            args,
            int(unique_coords.shape[0]),
            drop_count=len(selected_indices),
            add_count=len(selected_add_items),
        )
        geometry_percent = _sparsepcgc_geometry_penalty_percent(
            args,
            int(unique_coords.shape[0]),
            drop_count=len(selected_indices),
            add_count=len(selected_add_items),
        )
        debug["used"] = True
        if selected_drop > 0 and selected_add > 0:
            debug["reason"] = f"fast_diagnostic_{debug.get('fast_diagnostic_name', 'prune')}+dense_hole_add"
        elif selected_add > 0:
            debug["reason"] = "fast_diagnostic_dense_hole_add"
        else:
            debug["reason"] = f"fast_diagnostic_{debug.get('fast_diagnostic_name', 'prune')}"
        debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + int(bool(selected_indices)) + int(bool(fast_diagnostic_add_items))
        debug["accepted_candidate_count"] = int(selected_drop > 0) + int(selected_add > 0)
        debug["accepted_prune_count"] = int(selected_drop)
        debug["accepted_add_count"] = int(selected_add)
        debug["selected_drop_count"] = int(selected_drop)
        debug["selected_add_count"] = int(selected_add)
        debug["best_percent"] = float(proxy_percent)
        debug["best_proxy_percent"] = float(proxy_percent)
        debug["selected_proxy_percent"] = float(proxy_percent)
        debug["selected_edit_record_bits"] = float(edit_record_bits)
        debug["selected_geometry_percent"] = float(geometry_percent)
        if not actual_validate_this_step:
            debug["actual_oracle_time"] = 0.0
            debug["tested_count"] = 0
            debug["actual_eval_max"] = 0
        debug["fast_diagnostic_used"] = True
        debug["fast_diagnostic_proxy_eval_skipped"] = bool(skip_fast_proxy_eval)
        debug["fast_diagnostic_proxy_bits"] = float(proxy_bits)
        debug["fast_diagnostic_proxy_percent"] = float(proxy_percent)
        return True
    
    if missing_full_cloud_teacher_eval:
        debug["reason"] = "full_cloud_splice_missing_for_required_teacher"
        debug["tested_count"] = 0
        debug["actual_eval_max"] = 0
        debug["actual_oracle_time"] = 0.0
    elif not actual_validate_this_step:
        if not _apply_fast_diagnostic_teacher():
            debug["reason"] = "interval_skip_no_fast_diagnostic_candidate"
    elif (
        (not skip_unused_local_candidate_generation)
        and (not candidate_pool and not add_candidate_pool)
    ) or unique_coords is None or inverse is None:
        debug["reason"] = "no_valid_actual_oracle_candidates"
    else:
        oracle_time_start = time.time()
        if (
            bool(getattr(args, "sparsepcgc_actual_oracle_release_cuda_cache", False))
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
            debug["released_main_cuda_cache"] = True
        actual_eval_max = max(
            int(getattr(args, "sparsepcgc_actual_oracle_actual_eval_max", max_candidates)),
            0,
        )
        if actual_eval_max <= 0:
            actual_eval_max = max_candidates
        debug["actual_eval_max"] = int(actual_eval_max)
        operation_cycle = [
            item.strip().lower()
            for item in str(
                getattr(args, "sparsepcgc_actual_oracle_operation_cycle", "add,move")
            ).replace(";", ",").split(",")
            if item.strip().lower() in {"add", "move"}
        ]
        if not operation_cycle:
            operation_cycle = ["add", "move"]
        aux_probe_interval = max(
            int(getattr(args, "sparsepcgc_actual_oracle_aux_probe_interval", 6)),
            0,
        )
        aux_probe_due = (
            aux_probe_interval > 0
            and (int(global_step) + 1) % int(aux_probe_interval) == 0
        )
        if aux_probe_due:
            aux_probe_index = max(((int(global_step) + 1) // int(aux_probe_interval)) - 1, 0)
            scheduled_operation = operation_cycle[int(aux_probe_index) % len(operation_cycle)]
            actual_eval_max = max(int(actual_eval_max), 2)
        else:
            scheduled_operation = "prune"
        debug["aux_probe_due"] = bool(aux_probe_due)
        debug["aux_probe_interval"] = int(aux_probe_interval)
        debug["scheduled_operation"] = str(scheduled_operation)
        prioritize_full_cloud_macro = bool(
            getattr(args, "sparsepcgc_actual_oracle_prioritize_full_cloud_macro", True)
            and torch.is_tensor(full_eval_coords)
            and int(getattr(args, "sparsepcgc_actual_oracle_full_cloud_macro_prune_candidate_max", 0)) > 0
        )
        actual_eval_max_configured = int(actual_eval_max)
        full_macro_fail_extra_eval_max = max(
            int(getattr(args, "sparsepcgc_actual_oracle_full_macro_fail_extra_eval_max", 2)),
            0,
        )
        full_macro_fail_fallback_enabled = bool(
            getattr(args, "sparsepcgc_actual_oracle_fallback_after_full_macro_fail", True)
            and prioritize_full_cloud_macro
            and scheduled_operation == "prune"
            and full_macro_fail_extra_eval_max > 0
        )
        actual_eval_limit = int(actual_eval_max_configured)
        if full_macro_fail_fallback_enabled:
            actual_eval_limit += int(full_macro_fail_extra_eval_max)
        debug["actual_eval_max_configured"] = int(actual_eval_max_configured)
        debug["actual_eval_max"] = int(actual_eval_limit)
        debug["full_cloud_macro_fail_fallback_enabled"] = bool(full_macro_fail_fallback_enabled)
        debug["full_cloud_macro_fail_extra_eval_max"] = int(full_macro_fail_extra_eval_max)
        single_eval_fraction = min(
            max(float(getattr(args, "sparsepcgc_actual_oracle_single_eval_fraction", 0.50)), 0.0),
            1.0,
        )
        if prioritize_full_cloud_macro or scheduled_operation in {"add", "move"}:
            single_eval_max = 0
        else:
            single_eval_max = max(1, int(math.ceil(float(actual_eval_limit) * single_eval_fraction)))
            single_eval_max = min(int(single_eval_max), int(actual_eval_limit))
        debug["single_eval_max"] = int(single_eval_max)
        debug["prioritize_full_cloud_macro"] = bool(prioritize_full_cloud_macro)

        def _actual_budget_exhausted(tested_count):
            return int(tested_count) >= int(actual_eval_limit)

        try:
            full_cloud_cache_key = str(
                full_octree_context.get("actual_oracle_full_cloud_cache_key", "")
                if isinstance(full_octree_context, dict)
                else ""
            )
            oracle_base_cache_key = (
                full_cloud_cache_key
                if full_cloud_cache_key and torch.is_tensor(full_eval_coords)
                else f"{cache_key or ''}|sparsepcgc_actual_oracle_voxel_base"
            )
            cached_gt = loss._get_cached_actual_gt(oracle_base_cache_key)
            base_encode_start = time.time()
            base_cache_hit = cached_gt is not None
            if cached_gt is None:
                base_xyz = _oracle_actual_eval_xyz(unique_coords)
                if base_xyz is None or base_xyz.shape[-1] <= 0:
                    base_xyz = subtree_xyz[:, :3, :]
                cached_gt = loss._encode_actual_batch(args, base_xyz)
                loss._store_cached_actual_gt(oracle_base_cache_key, cached_gt)
            debug["original_actual_cache_hit"] = bool(base_cache_hit)
            debug["original_actual_encode_time"] = float(time.time() - base_encode_start)
            base_bit = float(cached_gt.get("bit", 0.0))
            if not math.isfinite(base_bit) or base_bit <= 0.0:
                raise RuntimeError(f"invalid_base_bit={base_bit}")
            debug["original_actual_bits"] = float(base_bit)

            improving = []
            best_percent = 0.0
            best_raw_percent = 0.0
            best_actual_percent = 0.0
            best_proxy_percent = 0.0
            best_edit_record_bits = 0.0
            best_edited_actual_bits = float(base_bit)
            tested = 0
            single_tested = 0
            bad_candidate_count = 0
            improving_candidate_count = 0
            all_unique_idx = torch.arange(unique_coords.shape[0], device=coords.device, dtype=torch.long)
            unique_count = int(unique_coords.shape[0])

            def _candidate_objective(raw_bit, edit_record_bits, geometry_percent=0.0):
                raw_percent, billed_percent = _sparsepcgc_objective_percent_with_edit_record(
                    args,
                    raw_bit,
                    base_bit,
                    edit_record_bits,
                )
                return raw_percent, billed_percent, float(billed_percent + float(geometry_percent))

            def _update_best(raw_percent, actual_percent, objective_percent, edit_record_bits, raw_bit, proxy_percent):
                nonlocal best_percent, best_raw_percent, best_actual_percent
                nonlocal best_edit_record_bits, best_edited_actual_bits, best_proxy_percent
                if float(objective_percent) < float(best_percent):
                    best_percent = float(objective_percent)
                    best_raw_percent = float(raw_percent)
                    best_actual_percent = float(actual_percent)
                    best_edit_record_bits = float(edit_record_bits)
                    best_edited_actual_bits = float(raw_bit)
                    best_proxy_percent = float(proxy_percent)

            def _oracle_strength(percent, *, bad=False):
                if bad:
                    value = max(float(percent), 0.0)
                else:
                    value = abs(float(percent))
                return 1.0 + min(value / 10.0, 4.0)

            def _single_budget_exhausted():
                return int(single_tested) >= int(single_eval_max)

            for drop_candidate in candidate_indices:
                if scheduled_operation != "prune":
                    break
                if _actual_budget_exhausted(tested) or _single_budget_exhausted():
                    break
                if isinstance(drop_candidate, dict):
                    unique_idx = int(drop_candidate.get("unique_idx", -1))
                    drop_memory_key = str(drop_candidate.get("memory_key", ""))
                else:
                    unique_idx = int(drop_candidate)
                    drop_memory_key = ""
                if unique_idx < 0:
                    continue
                keep_unique = all_unique_idx != int(unique_idx)
                if int(keep_unique.sum().item()) <= 0:
                    continue
                candidate_coords = unique_coords[keep_unique]
                candidate_xyz = _oracle_actual_eval_xyz(candidate_coords)
                if candidate_xyz is None or candidate_xyz.shape[-1] <= 0:
                    continue
                stats = loss._encode_actual_batch(args, candidate_xyz)
                tested += 1
                single_tested += 1
                cand_bit = float(stats.get("bit", 0.0))
                edit_record_bits = _sparsepcgc_edit_record_total_bits(
                    args,
                    edit_record_unique_count,
                    drop_count=1,
                )
                proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                    unique_coords[keep_unique],
                    args,
                    base_proxy_bits,
                )
                geometry_percent = _sparsepcgc_geometry_penalty_percent(
                    args,
                    edit_record_unique_count,
                    drop_count=1,
                )
                raw_percent, actual_percent, cand_percent = _candidate_objective(
                    cand_bit,
                    edit_record_bits,
                    geometry_percent,
                )
                _sparsepcgc_actual_oracle_update_memory(args, drop_memory_key, cand_percent)
                _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                min_improve = max(float(getattr(args, "sparsepcgc_actual_oracle_min_improve_percent", 0.0)), 0.0)
                if cand_percent < -min_improve:
                    improving_candidate_count += 1
                    improving.append(
                        {
                            "op": "drop",
                            "unique_idx": unique_idx,
                            "percent": float(cand_percent),
                            "raw_percent": float(raw_percent),
                            "actual_percent": float(actual_percent),
                            "proxy_percent": float(proxy_percent),
                            "proxy_bits": float(proxy_bits),
                            "geometry_percent": float(geometry_percent),
                            "edit_record_bits": float(edit_record_bits),
                            "memory_key": drop_memory_key,
                            "score_hint": float(drop_candidate.get("score_hint", 0.0)) if isinstance(drop_candidate, dict) else 0.0,
                        }
                    )
                else:
                    bad_candidate_count += 1
                    mask = inverse == unique_idx
                    if bool(mask.any().detach().cpu()):
                        bad_drop_point_mask[0] |= mask
                        strength = _oracle_strength(cand_percent, bad=True)
                        bad_drop_score[0, mask] = torch.maximum(
                            bad_drop_score[0, mask],
                            bad_drop_score.new_full((int(mask.sum().item()),), float(strength)),
                        )

            for add_candidate in add_candidates:
                if scheduled_operation != "add":
                    break
                if _actual_budget_exhausted(tested) or _single_budget_exhausted():
                    break
                target_coord = add_candidate.get("target_coord", None)
                if not torch.is_tensor(target_coord):
                    continue
                target_coord = target_coord.to(device=unique_coords.device, dtype=torch.long).view(1, 3)
                candidate_coords = torch.cat([unique_coords, target_coord], dim=0)
                candidate_coords = torch.unique(candidate_coords, dim=0, sorted=True)
                if int(candidate_coords.shape[0]) <= int(unique_coords.shape[0]):
                    continue
                candidate_xyz = _oracle_actual_eval_xyz(candidate_coords)
                if candidate_xyz is None or candidate_xyz.shape[-1] <= 0:
                    continue
                stats = loss._encode_actual_batch(args, candidate_xyz)
                tested += 1
                single_tested += 1
                cand_bit = float(stats.get("bit", 0.0))
                edit_record_bits = _sparsepcgc_edit_record_total_bits(
                    args,
                    edit_record_unique_count,
                    add_count=1,
                )
                proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                    candidate_coords,
                    args,
                    base_proxy_bits,
                )
                geometry_percent = _sparsepcgc_geometry_penalty_percent(
                    args,
                    edit_record_unique_count,
                    add_count=1,
                )
                raw_percent, actual_percent, cand_percent = _candidate_objective(
                    cand_bit,
                    edit_record_bits,
                    geometry_percent,
                )
                add_memory_key = str(add_candidate.get("memory_key", ""))
                _sparsepcgc_actual_oracle_update_memory(args, add_memory_key, cand_percent)
                _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                min_improve = max(float(getattr(args, "sparsepcgc_actual_oracle_min_improve_percent", 0.0)), 0.0)
                if cand_percent < -min_improve:
                    improving_candidate_count += 1
                    improving.append(
                        {
                            "op": "add",
                            "source_unique_idx": int(add_candidate["source_unique_idx"]),
                            "target_child_slot": int(add_candidate["target_child_slot"]),
                            "target_coord": target_coord.detach().clone(),
                            "percent": float(cand_percent),
                            "raw_percent": float(raw_percent),
                            "actual_percent": float(actual_percent),
                            "proxy_percent": float(proxy_percent),
                            "proxy_bits": float(proxy_bits),
                            "geometry_percent": float(geometry_percent),
                            "edit_record_bits": float(edit_record_bits),
                            "memory_key": add_memory_key,
                            "score_hint": float(add_candidate.get("score_hint", 0.0)),
                        }
                    )
                else:
                    bad_candidate_count += 1
                    source_unique_idx = int(add_candidate["source_unique_idx"])
                    mask = inverse == source_unique_idx
                    if bool(mask.any().detach().cpu()):
                        bad_add_point_mask[0] |= mask
                        strength = _oracle_strength(cand_percent, bad=True)
                        bad_add_score[0, mask] = torch.maximum(
                            bad_add_score[0, mask],
                            bad_add_score.new_full((int(mask.sum().item()),), float(strength)),
                        )
                        bad_add_child_slot[0, mask] = int(add_candidate["target_child_slot"])
                        source_coord = unique_coords[source_unique_idx]
                        bad_add_direction_index[0, mask] = _neighbor_direction_index(
                            target_coord.reshape(3) - source_coord
                        )

            joint_tested = 0
            joint_improving_count = 0
            group_tested = 0
            group_improving_count = 0
            parent_prune_tested = 0
            parent_prune_improving_count = 0
            pattern_plan_tested = 0
            pattern_plan_improving_count = 0
            full_cloud_macro_tested = 0
            full_cloud_macro_improving_count = 0
            full_cloud_macro_best_percent = float("inf")
            full_cloud_macro_best_ratio = 0.0
            full_cloud_macro_best_drop_count = 0
            full_cloud_macro_max = max(
                int(getattr(args, "sparsepcgc_actual_oracle_full_cloud_macro_prune_candidate_max", 1)),
                0,
            )
            if not torch.is_tensor(full_eval_coords):
                full_cloud_macro_max = 0
            macro_prune_tested = 0
            macro_prune_improving_count = 0
            macro_prune_best_percent = float("inf")
            macro_prune_best_ratio = 0.0
            macro_prune_best_drop_count = 0
            macro_prune_best_variant = ""
            macro_prune_best_proxy_percent = 0.0
            macro_prune_max = max(
                int(getattr(args, "sparsepcgc_actual_oracle_macro_prune_candidate_max", 2)),
                0,
            )
            if scheduled_operation != "prune":
                macro_prune_max = 0
            max_joint_candidates = max(
                int(getattr(args, "sparsepcgc_actual_oracle_joint_candidate_max", 1)),
                0,
            )
            if scheduled_operation != "prune":
                max_joint_candidates = 0
            group_candidate_max = max(
                int(getattr(args, "sparsepcgc_actual_oracle_group_candidate_max", 2)),
                0,
            )
            if scheduled_operation not in {"prune", "add"}:
                group_candidate_max = 0
            parent_prune_max = max(
                int(getattr(args, "sparsepcgc_actual_oracle_parent_prune_candidate_max", 2)),
                0,
            )
            if scheduled_operation != "prune":
                parent_prune_max = 0
            pattern_plan_max = max(
                int(getattr(args, "sparsepcgc_actual_oracle_pattern_plan_candidate_max", 2)),
                0,
            )
            if scheduled_operation != "prune":
                pattern_plan_max = 0
            subtree_move_tested = 0
            subtree_move_improving_count = 0
            subtree_move_interval = max(
                int(getattr(args, "sparsepcgc_actual_oracle_subtree_move_interval", 4)),
                1,
            )
            subtree_move_allowed_this_step = (
                bool(getattr(args, "sparsepcgc_actual_oracle_allow_subtree_move", True))
                and scheduled_operation == "move"
            )
            subtree_move_max = max(
                int(getattr(args, "sparsepcgc_actual_oracle_subtree_move_candidate_max", 2)),
                0,
            )
            if not subtree_move_allowed_this_step:
                subtree_move_max = 0

            complex_eval_budget = max(int(actual_eval_limit) - int(single_eval_max), 0)
            configured_complex_eval_budget = max(
                int(actual_eval_max_configured) - int(single_eval_max),
                0,
            )
            full_cloud_macro_budget = (
                configured_complex_eval_budget
                if full_macro_fail_fallback_enabled
                else complex_eval_budget
            )
            full_cloud_macro_eval_limit = min(full_cloud_macro_max, full_cloud_macro_budget)
            remaining_complex_budget = max(complex_eval_budget - full_cloud_macro_eval_limit, 0)
            pattern_plan_eval_limit = min(pattern_plan_max, 1 if remaining_complex_budget > 0 else 0)
            remaining_complex_budget = max(remaining_complex_budget - pattern_plan_eval_limit, 0)
            macro_prune_eval_limit = min(macro_prune_max, remaining_complex_budget)
            remaining_complex_budget = max(remaining_complex_budget - macro_prune_eval_limit, 0)
            parent_prune_eval_limit = min(parent_prune_max, 1 if remaining_complex_budget > 0 else 0)
            remaining_complex_budget = max(remaining_complex_budget - parent_prune_eval_limit, 0)
            joint_eval_limit = min(max_joint_candidates, 1 if complex_eval_budget > 0 else 0)
            joint_eval_limit = min(joint_eval_limit, remaining_complex_budget)
            remaining_complex_budget = max(remaining_complex_budget - joint_eval_limit, 0)
            subtree_move_eval_limit = min(subtree_move_max, 1 if remaining_complex_budget > 0 else 0)
            remaining_complex_budget = max(remaining_complex_budget - subtree_move_eval_limit, 0)
            group_eval_limit = min(group_candidate_max, remaining_complex_budget)
            if (
                group_eval_limit <= 0
                and complex_eval_budget > 0
                and (
                    full_cloud_macro_eval_limit
                    + macro_prune_eval_limit
                    + joint_eval_limit
                    + parent_prune_eval_limit
                    + pattern_plan_eval_limit
                    + subtree_move_eval_limit
                ) <= 0
            ):
                group_eval_limit = min(group_candidate_max, complex_eval_budget)
            debug["full_cloud_macro_eval_max"] = int(full_cloud_macro_eval_limit)
            debug["macro_prune_eval_max"] = int(macro_prune_eval_limit)
            debug["joint_eval_max"] = int(joint_eval_limit)
            debug["group_eval_max"] = int(group_eval_limit)
            debug["parent_prune_eval_max"] = int(parent_prune_eval_limit)
            debug["pattern_plan_eval_max"] = int(pattern_plan_eval_limit)
            debug["subtree_move_eval_max"] = int(subtree_move_eval_limit)
            def _sparsepcgc_full_macro_candidate_cache_key():
                context_key = ""
                if isinstance(full_octree_context, dict):
                    context_key = str(full_octree_context.get("actual_oracle_full_cloud_cache_key", "") or "")
                if not context_key:
                    context_key = str(cache_key or "")

                ratios = str(getattr(args, "sparsepcgc_actual_oracle_full_cloud_macro_prune_ratios", ""))
                subtree_ratios = str(getattr(args, "sparsepcgc_actual_oracle_full_cloud_subtree_prune_ratios", ""))
                block_sizes = str(getattr(args, "sparsepcgc_actual_oracle_full_cloud_subtree_block_sizes", ""))
                target_ratio = str(getattr(args, "sparsepcgc_actual_oracle_full_cloud_subtree_target_ratio", ""))
                return (
                    f"{context_key}|full_macro_candidates"
                    f"|n={int(full_eval_coords.shape[0])}"
                    f"|max={int(full_cloud_macro_max)}"
                    f"|ratios={ratios}"
                    f"|subtree_ratios={subtree_ratios}"
                    f"|blocks={block_sizes}"
                    f"|target={target_ratio}"
                )

            def _get_or_build_full_macro_candidates_fast():
                use_cache = bool(getattr(args, "sparsepcgc_full_macro_candidate_cache", True))
                cache_max = max(int(getattr(args, "sparsepcgc_full_macro_candidate_cache_max_entries", 4)), 1)
                key = _sparsepcgc_full_macro_candidate_cache_key()

                if use_cache:
                    cache = getattr(args, "_sparsepcgc_full_macro_candidate_cache", None)
                    if not isinstance(cache, OrderedDict):
                        cache = OrderedDict()
                        setattr(args, "_sparsepcgc_full_macro_candidate_cache", cache)
                    cached = cache.get(key, None)
                    if cached is not None:
                        cache.move_to_end(key)
                        debug["full_cloud_macro_generate_cache_hit"] = True
                        return cached

                built = _sparsepcgc_actual_oracle_full_cloud_macro_prune_candidates(
                    full_eval_coords,
                    args,
                    full_cloud_macro_max,
                    teacher_coords=unique_coords,
                )

                if use_cache:
                    cache[key] = built
                    cache.move_to_end(key)
                    while len(cache) > cache_max:
                        try:
                            cache.popitem(last=False)
                        except Exception:
                            break
                    debug["full_cloud_macro_generate_cache_hit"] = False
                return built

            def _get_full_macro_local_indices_tensor(full_macro_item):
                op_name = str(full_macro_item.get("op", ""))
                if op_name == "full_cloud_subtree_prune":
                    drop_block_coords = full_macro_item.get("drop_block_coords", None)
                    block_size = max(int(full_macro_item.get("block_size", 32)), 2)
                    if not torch.is_tensor(drop_block_coords):
                        return unique_coords.new_zeros((0,), dtype=torch.long)
                    local_blocks = torch.div(unique_coords, block_size, rounding_mode="floor")
                    local_mask = _sparsepcgc_rows_membership_mask_fast(local_blocks, drop_block_coords)
                else:
                    drop_coords = full_macro_item.get("drop_coords", None)
                    if not torch.is_tensor(drop_coords):
                        return unique_coords.new_zeros((0,), dtype=torch.long)
                    local_mask = _sparsepcgc_rows_membership_mask_fast(unique_coords, drop_coords)
                return local_mask.nonzero(as_tuple=False).reshape(-1).to(device=unique_coords.device, dtype=torch.long)

            def _get_cached_or_encode_full_macro_actual(full_macro_item, candidate_coords):
                use_cache = bool(getattr(args, "sparsepcgc_full_macro_actual_cache", True))
                stats = None
                cache_hit = False

                context_key = ""
                if isinstance(full_octree_context, dict):
                    context_key = str(full_octree_context.get("actual_oracle_full_cloud_cache_key", "") or "")
                if not context_key:
                    context_key = str(cache_key or "")

                op_name = str(full_macro_item.get("op", "macro_prune"))
                variant = str(full_macro_item.get("variant", ""))
                block_size = int(full_macro_item.get("block_size", 0) or 0)
                drop_count = int(full_macro_item.get("drop_count", 0) or 0)
                drop_block_count = int(full_macro_item.get("drop_block_count", 0) or 0)
                drop_ratio = float(full_macro_item.get("drop_ratio", 0.0) or 0.0)
                encode_key = (
                    f"{context_key}|full_macro_actual"
                    f"|op={op_name}|variant={variant}"
                    f"|n={int(candidate_coords.shape[0])}"
                    f"|drop={drop_count}|drop_blocks={drop_block_count}"
                    f"|block={block_size}|ratio={drop_ratio:.8f}"
                )

                if use_cache and hasattr(loss, "_get_cached_actual_gt"):
                    try:
                        stats = loss._get_cached_actual_gt(encode_key)
                        cache_hit = stats is not None
                    except Exception:
                        stats = None
                        cache_hit = False

                if stats is None:
                    restore_start = time.time()
                    with torch.no_grad():
                        candidate_xyz = _restore_codec_xyz_from_global_voxels(
                            args,
                            candidate_coords.transpose(0, 1).contiguous().unsqueeze(0),
                            full_octree_context if isinstance(full_octree_context, dict) else subtree_tree,
                            subtree_xyz,
                        )
                    debug["full_cloud_macro_restore_time"] = float(
                        debug.get("full_cloud_macro_restore_time", 0.0) + (time.time() - restore_start)
                    )
                    if candidate_xyz is None or candidate_xyz.shape[-1] <= 0:
                        return None, False

                    candidate_encode_wall_start = time.time()
                    stats = loss._encode_actual_batch(args, candidate_xyz)
                    debug["candidate_actual_wall_time"] = float(
                        debug.get("candidate_actual_wall_time", 0.0) + (time.time() - candidate_encode_wall_start)
                    )
                    debug["candidate_actual_encode_time"] = float(
                        debug.get("candidate_actual_encode_time", 0.0)
                        + float(stats.get("encode_time", 0.0) or 0.0)
                    )

                    if use_cache and hasattr(loss, "_store_cached_actual_gt"):
                        try:
                            loss._store_cached_actual_gt(encode_key, stats)
                        except Exception:
                            pass

                debug["full_cloud_macro_actual_cache_hit_count"] = int(
                    debug.get("full_cloud_macro_actual_cache_hit_count", 0)
                ) + int(cache_hit)
                return stats, cache_hit

            if full_cloud_macro_eval_limit > 0 and torch.is_tensor(full_eval_coords):
                full_macro_generate_start = time.time()
                full_macro_candidates = _get_or_build_full_macro_candidates_fast()
                debug["full_cloud_macro_generate_time"] = float(time.time() - full_macro_generate_start)
                debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + int(len(full_macro_candidates))
                min_improve = max(float(getattr(args, "sparsepcgc_actual_oracle_min_improve_percent", 0.0)), 0.0)

                compute_local_proxy = bool(
                    getattr(args, "sparsepcgc_full_macro_compute_exact_local_proxy", False)
                )
                max_local_indices_for_record = max(
                    int(getattr(args, "sparsepcgc_actual_oracle_max_selected_voxels", 4)),
                    1,
                )

                for full_macro_item in full_macro_candidates:
                    if _actual_budget_exhausted(tested) or full_cloud_macro_tested >= full_cloud_macro_eval_limit:
                        break

                    candidate_coords = full_macro_item.get("candidate_coords", None)
                    if not torch.is_tensor(candidate_coords):
                        continue
                    candidate_coords = candidate_coords.to(device=full_eval_coords.device, dtype=torch.long).contiguous()

                    local_map_start = time.time()
                    with torch.no_grad():
                        local_unique_indices_tensor = _get_full_macro_local_indices_tensor(full_macro_item)
                    debug["full_cloud_macro_local_map_time"] = float(
                        debug.get("full_cloud_macro_local_map_time", 0.0) + (time.time() - local_map_start)
                    )

                    stats, actual_cache_hit = _get_cached_or_encode_full_macro_actual(full_macro_item, candidate_coords)
                    if not isinstance(stats, dict):
                        continue

                    tested += 1
                    full_cloud_macro_tested += 1
                    cand_bit = float(stats.get("bit", 0.0))
                    if not math.isfinite(cand_bit) or cand_bit <= 0.0:
                        continue

                    full_drop_count = int(
                        full_macro_item.get(
                            "drop_count",
                            int(local_unique_indices_tensor.numel()),
                        )
                    )

                    if str(full_macro_item.get("op", "")) == "full_cloud_subtree_prune":
                        edit_record_bits = _sparsepcgc_edit_record_structured_prune_bits(
                            args,
                            edit_record_unique_count,
                            int(full_macro_item.get("block_size", 32)),
                            float(full_macro_item.get("drop_ratio", 0.0)),
                        )
                    else:
                        edit_record_bits = _sparsepcgc_edit_record_total_bits(
                            args,
                            edit_record_unique_count,
                            drop_count=full_drop_count,
                        )

                    if compute_local_proxy:
                        local_proxy_start = time.time()
                        keep_local = torch.ones(
                            (unique_coords.shape[0],),
                            device=unique_coords.device,
                            dtype=torch.bool,
                        )
                        if local_unique_indices_tensor.numel() > 0:
                            keep_local[local_unique_indices_tensor] = False
                        local_candidate_coords = unique_coords[keep_local]
                        proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                            local_candidate_coords,
                            args,
                            base_proxy_bits,
                        )
                        debug["full_cloud_macro_local_proxy_time"] = float(
                            debug.get("full_cloud_macro_local_proxy_time", 0.0)
                            + (time.time() - local_proxy_start)
                        )
                    else:
                        proxy_bits = float(full_macro_item.get("proxy_bits", 0.0) or 0.0)
                        proxy_percent = float(full_macro_item.get("proxy_percent", 0.0) or 0.0)
                        debug["full_cloud_macro_local_proxy_skipped"] = True

                    geometry_percent = _sparsepcgc_geometry_penalty_percent(
                        args,
                        edit_record_unique_count,
                        drop_count=full_drop_count,
                    )
                    raw_percent, actual_percent, cand_percent = _candidate_objective(
                        cand_bit,
                        edit_record_bits,
                        geometry_percent,
                    )

                    if float(cand_percent) < float(full_cloud_macro_best_percent):
                        full_cloud_macro_best_percent = float(cand_percent)
                        full_cloud_macro_best_ratio = float(full_macro_item.get("drop_ratio", 0.0))
                        full_cloud_macro_best_drop_count = int(full_drop_count)

                    _update_best(
                        raw_percent,
                        actual_percent,
                        cand_percent,
                        edit_record_bits,
                        cand_bit,
                        proxy_percent,
                    )

                    if cand_percent < -min_improve:
                        full_cloud_macro_improving_count += 1
                        improving_candidate_count += 1

                        local_unique_indices = [
                            int(v)
                            for v in local_unique_indices_tensor[:max_local_indices_for_record]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                        improving.append(
                            {
                                "op": str(full_macro_item.get("op", "macro_prune")),
                                "unique_indices": local_unique_indices,
                                "percent": float(cand_percent),
                                "raw_percent": float(raw_percent),
                                "actual_percent": float(actual_percent),
                                "proxy_percent": float(proxy_percent),
                                "proxy_bits": float(proxy_bits),
                                "geometry_percent": float(geometry_percent),
                                "edit_record_bits": float(edit_record_bits),
                                "edited_actual_bits": float(cand_bit),
                                "full_cloud_drop_count": int(full_drop_count),
                                "drop_block_count": int(full_macro_item.get("drop_block_count", 0)),
                                "block_size": int(full_macro_item.get("block_size", 0)),
                                "drop_ratio": float(full_macro_item.get("drop_ratio", 0.0)),
                                "override_final_voxel_coords": candidate_coords.detach(),
                                "override_scope": "full_cloud",
                                "actual_stats": dict(stats),
                                "actual_cache_hit": bool(actual_cache_hit),
                            }
                        )
                    else:
                        bad_candidate_count += 1
                        

            if full_macro_fail_fallback_enabled:
                full_macro_fallback_triggered = bool(full_cloud_macro_improving_count <= 0)
                debug["full_cloud_macro_fallback_triggered"] = bool(full_macro_fallback_triggered)
                if not full_macro_fallback_triggered:
                    pattern_plan_eval_limit = 0
                    macro_prune_eval_limit = 0
                    parent_prune_eval_limit = 0
                    joint_eval_limit = 0
                    subtree_move_eval_limit = 0
                    group_eval_limit = 0
            else:
                debug["full_cloud_macro_fallback_triggered"] = False
            debug["macro_prune_eval_max"] = int(macro_prune_eval_limit)
            debug["joint_eval_max"] = int(joint_eval_limit)
            debug["group_eval_max"] = int(group_eval_limit)
            debug["parent_prune_eval_max"] = int(parent_prune_eval_limit)
            debug["pattern_plan_eval_max"] = int(pattern_plan_eval_limit)
            debug["subtree_move_eval_max"] = int(subtree_move_eval_limit)
            
            
            if macro_prune_eval_limit > 0:
                macro_candidates = _sparsepcgc_actual_oracle_macro_prune_candidates(
                    unique_coords,
                    args,
                    macro_prune_max,
                    proxy_profile=proxy_profile,
                    base_proxy_bits=base_proxy_bits,
                )
                debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + int(len(macro_candidates))
                min_improve = max(float(getattr(args, "sparsepcgc_actual_oracle_min_improve_percent", 0.0)), 0.0)
                for macro_item in macro_candidates:
                    if _actual_budget_exhausted(tested) or macro_prune_tested >= macro_prune_eval_limit:
                        break
                    candidate_coords = macro_item.get("candidate_coords", None)
                    if not torch.is_tensor(candidate_coords):
                        continue
                    candidate_xyz = _oracle_actual_eval_xyz(candidate_coords)
                    if candidate_xyz is None or candidate_xyz.shape[-1] <= 0:
                        continue
                    stats = loss._encode_actual_batch(args, candidate_xyz)
                    tested += 1
                    macro_prune_tested += 1
                    cand_bit = float(stats.get("bit", 0.0))
                    unique_indices = [int(v) for v in macro_item.get("unique_indices", [])]
                    edit_record_bits = _sparsepcgc_edit_record_total_bits(
                        args,
                        edit_record_unique_count,
                        drop_count=len(unique_indices),
                    )
                    proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                        candidate_coords,
                        args,
                        base_proxy_bits,
                    )
                    geometry_percent = _sparsepcgc_geometry_penalty_percent(
                        args,
                        edit_record_unique_count,
                        drop_count=len(unique_indices),
                    )
                    raw_percent, actual_percent, cand_percent = _candidate_objective(
                        cand_bit,
                        edit_record_bits,
                        geometry_percent,
                    )
                    if float(cand_percent) < float(macro_prune_best_percent):
                        macro_prune_best_percent = float(cand_percent)
                        macro_prune_best_ratio = float(macro_item.get("drop_ratio", 0.0))
                        macro_prune_best_drop_count = int(len(unique_indices))
                        macro_prune_best_variant = str(macro_item.get("variant", ""))
                        macro_prune_best_proxy_percent = float(proxy_percent)
                    _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                    if cand_percent < -min_improve:
                        macro_prune_improving_count += 1
                        improving_candidate_count += 1
                        improving.append(
                            {
                                "op": "macro_prune",
                                "unique_indices": unique_indices,
                                "percent": float(cand_percent),
                                "raw_percent": float(raw_percent),
                                "actual_percent": float(actual_percent),
                                "proxy_percent": float(proxy_percent),
                                "proxy_bits": float(proxy_bits),
                                "geometry_percent": float(geometry_percent),
                                "edit_record_bits": float(edit_record_bits),
                                "edited_actual_bits": float(cand_bit),
                            }
                        )
                    else:
                        bad_candidate_count += 1
                        for unique_idx in unique_indices:
                            mask = inverse == int(unique_idx)
                            if bool(mask.any().detach().cpu()):
                                bad_drop_point_mask[0] |= mask
            if joint_eval_limit > 0:
                _ensure_local_candidate_generation()
            if joint_eval_limit > 0 and candidate_indices and add_candidates:
                pair_candidates = []
                for drop_candidate in candidate_indices:
                    drop_unique_idx = int(
                        drop_candidate.get("unique_idx", -1)
                        if isinstance(drop_candidate, dict)
                        else drop_candidate
                    )
                    if drop_unique_idx < 0:
                        continue
                    drop_key = str(drop_candidate.get("memory_key", "")) if isinstance(drop_candidate, dict) else ""
                    drop_hint = float(drop_candidate.get("score_hint", 0.0)) if isinstance(drop_candidate, dict) else 0.0
                    for add_candidate in add_candidates:
                        source_unique_idx = int(add_candidate.get("source_unique_idx", -1))
                        if source_unique_idx < 0 or source_unique_idx == drop_unique_idx:
                            continue
                        target_coord = add_candidate.get("target_coord", None)
                        if not torch.is_tensor(target_coord):
                            continue
                        add_key = str(add_candidate.get("memory_key", ""))
                        pair_key = _sparsepcgc_actual_oracle_pair_key(drop_key, add_key)
                        pair_bonus, pair_is_bad, _pair_seen = _sparsepcgc_actual_oracle_memory_bonus(args, pair_key)
                        if pair_is_bad:
                            continue
                        pair_score = (
                            drop_hint
                            + float(add_candidate.get("score_hint", 0.0))
                            + max(float(getattr(args, "sparsepcgc_actual_oracle_memory_weight", 0.75)), 0.0)
                            * float(pair_bonus)
                        )
                        pair_candidates.append(
                            {
                                "drop_unique_idx": drop_unique_idx,
                                "drop_memory_key": drop_key,
                                "add_memory_key": add_key,
                                "pair_memory_key": pair_key,
                                "source_unique_idx": source_unique_idx,
                                "target_child_slot": int(add_candidate["target_child_slot"]),
                                "target_coord": target_coord.detach().clone(),
                                "score": float(pair_score),
                            }
                        )
                pair_candidates = sorted(pair_candidates, key=lambda item: float(item["score"]), reverse=True)
                debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + int(len(pair_candidates))
                for pair_item in pair_candidates[:max_joint_candidates]:
                    if _actual_budget_exhausted(tested) or joint_tested >= joint_eval_limit:
                        break
                    keep_unique = all_unique_idx != int(pair_item["drop_unique_idx"])
                    if int(keep_unique.sum().item()) <= 0:
                        continue
                    target_coord = pair_item["target_coord"].to(device=unique_coords.device, dtype=torch.long).view(1, 3)
                    candidate_coords = torch.cat([unique_coords[keep_unique], target_coord], dim=0)
                    candidate_coords = torch.unique(candidate_coords, dim=0, sorted=True)
                    candidate_xyz = _oracle_actual_eval_xyz(candidate_coords)
                    if candidate_xyz is None or candidate_xyz.shape[-1] <= 0:
                        continue
                    stats = loss._encode_actual_batch(args, candidate_xyz)
                    tested += 1
                    joint_tested += 1
                    cand_bit = float(stats.get("bit", 0.0))
                    edit_record_bits = _sparsepcgc_edit_record_total_bits(
                        args,
                        edit_record_unique_count,
                        drop_count=1,
                        add_count=1,
                    )
                    proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                        candidate_coords,
                        args,
                        base_proxy_bits,
                    )
                    geometry_percent = _sparsepcgc_geometry_penalty_percent(
                        args,
                        edit_record_unique_count,
                        drop_count=1,
                        add_count=1,
                    )
                    raw_percent, actual_percent, cand_percent = _candidate_objective(
                        cand_bit,
                        edit_record_bits,
                        geometry_percent,
                    )
                    _sparsepcgc_actual_oracle_update_memory(args, pair_item["pair_memory_key"], cand_percent)
                    _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                    min_improve = max(float(getattr(args, "sparsepcgc_actual_oracle_min_improve_percent", 0.0)), 0.0)
                    if cand_percent < -min_improve:
                        joint_improving_count += 1
                        improving_candidate_count += 1
                        improving.append(
                            {
                                "op": "drop_add",
                                "unique_idx": int(pair_item["drop_unique_idx"]),
                                "source_unique_idx": int(pair_item["source_unique_idx"]),
                                "target_child_slot": int(pair_item["target_child_slot"]),
                                "target_coord": target_coord.detach().clone(),
                                "percent": float(cand_percent),
                                "raw_percent": float(raw_percent),
                                "actual_percent": float(actual_percent),
                                "proxy_percent": float(proxy_percent),
                                "proxy_bits": float(proxy_bits),
                                "geometry_percent": float(geometry_percent),
                                "edit_record_bits": float(edit_record_bits),
                                "memory_key": str(pair_item["pair_memory_key"]),
                            }
                        )
                    else:
                        bad_candidate_count += 1

            if group_eval_limit > 0:
                _ensure_local_candidate_generation()
            if group_eval_limit > 0:
                group_voxels = max(
                    int(getattr(args, "sparsepcgc_actual_oracle_group_voxels", 4)),
                    2,
                )
                raw_group_size_list = getattr(args, "sparsepcgc_actual_oracle_group_size_list", [group_voxels])
                if isinstance(raw_group_size_list, str):
                    group_size_values = []
                    for item in raw_group_size_list.replace(";", ",").split(","):
                        item = item.strip()
                        if not item:
                            continue
                        try:
                            group_size_values.append(int(float(item)))
                        except ValueError:
                            continue
                elif isinstance(raw_group_size_list, (list, tuple)):
                    group_size_values = []
                    for item in raw_group_size_list:
                        try:
                            group_size_values.append(int(float(item)))
                        except (TypeError, ValueError):
                            continue
                else:
                    group_size_values = []
                max_group_voxels = min(
                    group_voxels,
                    max(int(getattr(args, "sparsepcgc_actual_oracle_max_selected_voxels", 4)), 1),
                )
                if not group_size_values:
                    group_size_values = [max_group_voxels]
                group_size_values = sorted(
                    {
                        min(max(int(size), 2), max_group_voxels)
                        for size in group_size_values
                        if int(size) >= 2
                    }
                )
                if not group_size_values and max_group_voxels >= 2:
                    group_size_values = [max_group_voxels]
                min_improve = max(float(getattr(args, "sparsepcgc_actual_oracle_min_improve_percent", 0.0)), 0.0)
                group_candidates_used = 0

                group_drop_indices_all = []
                if len(candidate_pool) >= 2:
                    for drop_candidate in candidate_pool:
                        idx = int(drop_candidate.get("unique_idx", -1) if isinstance(drop_candidate, dict) else drop_candidate)
                        if idx >= 0 and idx not in group_drop_indices_all:
                            group_drop_indices_all.append(idx)
                        if len(group_drop_indices_all) >= max_group_voxels:
                            break

                group_add_items_all = []
                if len(add_candidate_pool) >= 2:
                    seen_targets = set()
                    for add_candidate in add_candidate_pool:
                        target_coord = add_candidate.get("target_coord", None)
                        if not torch.is_tensor(target_coord):
                            continue
                        target_key = tuple(int(v) for v in target_coord.view(-1).detach().cpu().tolist())
                        if target_key in seen_targets:
                            continue
                        seen_targets.add(target_key)
                        group_add_items_all.append(add_candidate)
                        if len(group_add_items_all) >= max_group_voxels:
                            break

                used_drop_group_sizes = set()
                used_add_group_sizes = set()
                for requested_group_size in group_size_values:
                    if group_candidates_used >= group_eval_limit:
                        break

                    drop_group_size = min(int(requested_group_size), len(group_drop_indices_all), max_group_voxels)
                    if (
                        group_candidate_max > 0
                        and scheduled_operation == "prune"
                        and drop_group_size >= 2
                        and drop_group_size not in used_drop_group_sizes
                        and group_candidates_used < group_eval_limit
                    ):
                        used_drop_group_sizes.add(drop_group_size)
                        debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + 1
                        if _actual_budget_exhausted(tested):
                            break
                        group_drop_indices = group_drop_indices_all[:drop_group_size]
                        keep_unique = torch.ones((unique_coords.shape[0],), device=coords.device, dtype=torch.bool)
                        keep_unique[torch.as_tensor(group_drop_indices, device=coords.device, dtype=torch.long)] = False
                        if int(keep_unique.sum().item()) > 0:
                            candidate_coords = unique_coords[keep_unique]
                            candidate_xyz = _oracle_actual_eval_xyz(candidate_coords)
                            if candidate_xyz is not None and candidate_xyz.shape[-1] > 0:
                                stats = loss._encode_actual_batch(args, candidate_xyz)
                                tested += 1
                                group_tested += 1
                                group_candidates_used += 1
                                cand_bit = float(stats.get("bit", 0.0))
                                edit_record_bits = _sparsepcgc_edit_record_total_bits(
                                    args,
                                    edit_record_unique_count,
                                    drop_count=len(group_drop_indices),
                                )
                                proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                                    unique_coords[keep_unique],
                                    args,
                                    base_proxy_bits,
                                )
                                geometry_percent = _sparsepcgc_geometry_penalty_percent(
                                    args,
                                    edit_record_unique_count,
                                    drop_count=len(group_drop_indices),
                                )
                                raw_percent, actual_percent, cand_percent = _candidate_objective(
                                    cand_bit,
                                    edit_record_bits,
                                    geometry_percent,
                                )
                                _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                                if cand_percent < -min_improve:
                                    group_improving_count += 1
                                    improving_candidate_count += 1
                                    improving.append(
                                        {
                                            "op": "drop_group",
                                            "unique_indices": list(group_drop_indices),
                                            "percent": float(cand_percent),
                                            "raw_percent": float(raw_percent),
                                            "actual_percent": float(actual_percent),
                                            "proxy_percent": float(proxy_percent),
                                            "proxy_bits": float(proxy_bits),
                                            "geometry_percent": float(geometry_percent),
                                            "edit_record_bits": float(edit_record_bits),
                                        }
                                    )
                                else:
                                    bad_candidate_count += 1
                                    for unique_idx in group_drop_indices:
                                        mask = inverse == int(unique_idx)
                                        if bool(mask.any().detach().cpu()):
                                            bad_drop_point_mask[0] |= mask

                    add_group_size = min(int(requested_group_size), len(group_add_items_all), max_group_voxels)
                    if (
                        group_candidate_max > 0
                        and scheduled_operation == "add"
                        and add_group_size >= 2
                        and add_group_size not in used_add_group_sizes
                        and group_candidates_used < group_eval_limit
                    ):
                        used_add_group_sizes.add(add_group_size)
                        debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + 1
                        if _actual_budget_exhausted(tested):
                            break
                        group_add_items = group_add_items_all[:add_group_size]
                        target_coords = [
                            item["target_coord"].to(device=unique_coords.device, dtype=torch.long).view(1, 3)
                            for item in group_add_items
                        ]
                        candidate_coords = torch.cat([unique_coords] + target_coords, dim=0)
                        candidate_coords = torch.unique(candidate_coords, dim=0, sorted=True)
                        if int(candidate_coords.shape[0]) > int(unique_coords.shape[0]):
                            candidate_xyz = _oracle_actual_eval_xyz(candidate_coords)
                            if candidate_xyz is not None and candidate_xyz.shape[-1] > 0:
                                stats = loss._encode_actual_batch(args, candidate_xyz)
                                tested += 1
                                group_tested += 1
                                group_candidates_used += 1
                                cand_bit = float(stats.get("bit", 0.0))
                                edit_record_bits = _sparsepcgc_edit_record_total_bits(
                                    args,
                                    edit_record_unique_count,
                                    add_count=len(group_add_items),
                                )
                                proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                                    candidate_coords,
                                    args,
                                    base_proxy_bits,
                                )
                                geometry_percent = _sparsepcgc_geometry_penalty_percent(
                                    args,
                                    edit_record_unique_count,
                                    add_count=len(group_add_items),
                                )
                                raw_percent, actual_percent, cand_percent = _candidate_objective(
                                    cand_bit,
                                    edit_record_bits,
                                    geometry_percent,
                                )
                                _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                                if cand_percent < -min_improve:
                                    group_improving_count += 1
                                    improving_candidate_count += 1
                                    improving.append(
                                        {
                                            "op": "add_group",
                                            "source_unique_indices": [
                                                int(item["source_unique_idx"]) for item in group_add_items
                                            ],
                                            "target_child_slots": [
                                                int(item["target_child_slot"]) for item in group_add_items
                                            ],
                                            "target_coords": torch.cat(target_coords, dim=0).detach().clone(),
                                            "percent": float(cand_percent),
                                            "raw_percent": float(raw_percent),
                                            "actual_percent": float(actual_percent),
                                            "proxy_percent": float(proxy_percent),
                                            "proxy_bits": float(proxy_bits),
                                            "geometry_percent": float(geometry_percent),
                                            "edit_record_bits": float(edit_record_bits),
                                        }
                                    )
                                else:
                                    bad_candidate_count += 1
                                    for item in group_add_items:
                                        source_unique_idx = int(item["source_unique_idx"])
                                        mask = inverse == source_unique_idx
                                        if bool(mask.any().detach().cpu()):
                                            bad_add_point_mask[0] |= mask
                                            strength = _oracle_strength(cand_percent, bad=True)
                                            bad_add_score[0, mask] = torch.maximum(
                                                bad_add_score[0, mask],
                                                bad_add_score.new_full((int(mask.sum().item()),), float(strength)),
                                            )
                                            target_coord = item["target_coord"].to(
                                                device=unique_coords.device,
                                                dtype=torch.long,
                                            ).reshape(3)
                                            bad_add_direction_index[0, mask] = _neighbor_direction_index(
                                                target_coord - unique_coords[source_unique_idx]
                                            )

            if parent_prune_eval_limit > 0 and unique_count > 1:
                parent_coords = torch.div(unique_coords, 2, rounding_mode="floor")
                unique_parents, parent_inverse = torch.unique(
                    parent_coords,
                    dim=0,
                    sorted=True,
                    return_inverse=True,
                )
                if int(unique_parents.shape[0]) > 0:
                    child_slot_for_parent = (
                        (unique_coords[:, 0] & 1)
                        + 2 * (unique_coords[:, 1] & 1)
                        + 4 * (unique_coords[:, 2] & 1)
                    ).to(dtype=torch.long)
                    occupancy = torch.zeros(
                        (unique_parents.shape[0], 8),
                        device=unique_coords.device,
                        dtype=torch.bool,
                    )
                    occupancy[parent_inverse, child_slot_for_parent] = True
                    pattern_weights = (2 ** torch.arange(8, device=unique_coords.device, dtype=torch.long)).view(1, 8)
                    parent_code = (occupancy.to(dtype=torch.long) * pattern_weights).sum(dim=1).clamp(0, 255)
                    parent_counts = torch.bincount(
                        parent_inverse,
                        minlength=int(unique_parents.shape[0]),
                    ).to(device=unique_coords.device, dtype=torch.long)
                    smoothing = max(float(getattr(args, "leaf_pattern_candidate_smoothing", 1.0)), 0.0)
                    code_hist = torch.bincount(parent_code, minlength=256).to(device=unique_coords.device, dtype=torch.float32)
                    code_prob = code_hist + float(smoothing)
                    code_prob = code_prob / code_prob.sum().clamp_min(torch.finfo(torch.float32).eps)
                    code_nll = -torch.log2(code_prob.clamp_min(torch.finfo(torch.float32).eps))
                    min_parent_voxels = max(
                        int(getattr(args, "sparsepcgc_actual_oracle_parent_prune_min_voxels", 2)),
                        1,
                    )
                    max_parent_voxels = min(
                        max(int(getattr(args, "sparsepcgc_actual_oracle_parent_prune_max_voxels", 8)), min_parent_voxels),
                        max(int(getattr(args, "sparsepcgc_actual_oracle_max_selected_voxels", 4)), 1),
                    )
                    parent_items = []
                    for parent_idx in range(int(unique_parents.shape[0])):
                        drop_count = int(parent_counts[parent_idx].detach().cpu())
                        if drop_count < min_parent_voxels or drop_count > max_parent_voxels:
                            continue
                        if drop_count >= unique_count:
                            continue
                        code = int(parent_code[parent_idx].detach().cpu())
                        key = f"parent_prune:code={code}:count={drop_count}"
                        bonus, is_bad, seen_memory = _sparsepcgc_actual_oracle_memory_bonus(args, key)
                        if is_bad:
                            continue
                        parent_score = (
                            float(code_nll[code].detach().cpu())
                            + 0.25 * math.log1p(float(drop_count))
                            + max(float(getattr(args, "sparsepcgc_actual_oracle_memory_weight", 0.75)), 0.0) * float(bonus)
                        )
                        parent_items.append((float(parent_score), int(parent_idx), key, bool(seen_memory), int(drop_count)))
                    parent_items.sort(key=lambda item: item[0], reverse=True)
                    debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + int(
                        min(parent_prune_max, len(parent_items))
                    )
                    for _score, parent_idx, parent_key, _seen_memory, drop_count in parent_items[:parent_prune_max]:
                        if _actual_budget_exhausted(tested) or parent_prune_tested >= parent_prune_eval_limit:
                            break
                        parent_drop_indices = (parent_inverse == int(parent_idx)).nonzero(as_tuple=False).reshape(-1)
                        if int(parent_drop_indices.numel()) != int(drop_count):
                            continue
                        keep_unique = torch.ones((unique_coords.shape[0],), device=unique_coords.device, dtype=torch.bool)
                        keep_unique[parent_drop_indices] = False
                        if int(keep_unique.sum().item()) <= 0:
                            continue
                        candidate_coords = unique_coords[keep_unique]
                        candidate_xyz = _oracle_actual_eval_xyz(candidate_coords)
                        if candidate_xyz is None or candidate_xyz.shape[-1] <= 0:
                            continue
                        stats = loss._encode_actual_batch(args, candidate_xyz)
                        tested += 1
                        parent_prune_tested += 1
                        cand_bit = float(stats.get("bit", 0.0))
                        edit_record_bits = _sparsepcgc_edit_record_total_bits(
                            args,
                            edit_record_unique_count,
                            drop_count=int(parent_drop_indices.numel()),
                        )
                        proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                            unique_coords[keep_unique],
                            args,
                            base_proxy_bits,
                        )
                        geometry_percent = _sparsepcgc_geometry_penalty_percent(
                            args,
                            edit_record_unique_count,
                            drop_count=int(parent_drop_indices.numel()),
                        )
                        raw_percent, actual_percent, cand_percent = _candidate_objective(
                            cand_bit,
                            edit_record_bits,
                            geometry_percent,
                        )
                        _sparsepcgc_actual_oracle_update_memory(args, parent_key, cand_percent)
                        _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                        if cand_percent < -min_improve:
                            parent_prune_improving_count += 1
                            improving_candidate_count += 1
                            improving.append(
                                {
                                    "op": "parent_collapse",
                                    "unique_indices": [int(v) for v in parent_drop_indices.detach().cpu().tolist()],
                                    "percent": float(cand_percent),
                                    "raw_percent": float(raw_percent),
                                    "actual_percent": float(actual_percent),
                                    "proxy_percent": float(proxy_percent),
                                    "proxy_bits": float(proxy_bits),
                                    "geometry_percent": float(geometry_percent),
                                    "edit_record_bits": float(edit_record_bits),
                                    "memory_key": parent_key,
                                }
                            )
                        else:
                            bad_candidate_count += 1
                            for unique_idx in parent_drop_indices.detach().cpu().tolist():
                                mask = inverse == int(unique_idx)
                                if bool(mask.any().detach().cpu()):
                                    bad_drop_point_mask[0] |= mask

            if pattern_plan_eval_limit > 0:
                pattern_candidates = _sparsepcgc_actual_oracle_pattern_plan_candidates(
                    unique_coords,
                    args,
                    global_step,
                    pattern_plan_max,
                    base_proxy_bits=base_proxy_bits,
                )
                debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + int(len(pattern_candidates))
                for plan_item in pattern_candidates:
                    if _actual_budget_exhausted(tested) or pattern_plan_tested >= pattern_plan_eval_limit:
                        break
                    candidate_coords = plan_item.get("transformed_coords", None)
                    if not torch.is_tensor(candidate_coords):
                        continue
                    candidate_xyz = _oracle_actual_eval_xyz(candidate_coords)
                    if candidate_xyz is None or candidate_xyz.shape[-1] <= 0:
                        continue
                    stats = loss._encode_actual_batch(args, candidate_xyz)
                    tested += 1
                    pattern_plan_tested += 1
                    cand_bit = float(stats.get("bit", 0.0))
                    drop_indices = [int(v) for v in plan_item.get("drop_unique_indices", [])]
                    add_items = list(plan_item.get("add_items", []) or [])
                    edit_record_bits = _sparsepcgc_edit_record_total_bits(
                        args,
                        edit_record_unique_count,
                        drop_count=len(drop_indices),
                        add_count=len(add_items),
                    )
                    proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                        candidate_coords,
                        args,
                        base_proxy_bits,
                    )
                    geometry_percent = _sparsepcgc_geometry_penalty_percent(
                        args,
                        edit_record_unique_count,
                        drop_count=len(drop_indices),
                        add_count=len(add_items),
                    )
                    raw_percent, actual_percent, cand_percent = _candidate_objective(
                        cand_bit,
                        edit_record_bits,
                        geometry_percent,
                    )
                    plan_memory_key = str(plan_item.get("memory_key", ""))
                    _sparsepcgc_actual_oracle_update_memory(args, plan_memory_key, cand_percent)
                    _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                    if cand_percent < -min_improve:
                        pattern_plan_improving_count += 1
                        improving_candidate_count += 1
                        improving.append(
                            {
                                "op": "pattern_plan",
                                "drop_unique_indices": drop_indices,
                                "add_items": add_items,
                                "final_coords": candidate_coords.detach().clone(),
                                "percent": float(cand_percent),
                                "raw_percent": float(raw_percent),
                                "actual_percent": float(actual_percent),
                                "proxy_percent": float(proxy_percent),
                                "proxy_bits": float(proxy_bits),
                                "geometry_percent": float(geometry_percent),
                                "edit_record_bits": float(edit_record_bits),
                                "memory_key": plan_memory_key,
                                "score_hint": float(plan_item.get("score", 0.0)),
                            }
                        )
                    else:
                        bad_candidate_count += 1
                        for unique_idx in drop_indices:
                            mask = inverse == int(unique_idx)
                            if bool(mask.any().detach().cpu()):
                                bad_drop_point_mask[0] |= mask
                        for add_item in add_items:
                            source_unique_idx = int(add_item.get("source_unique_idx", -1))
                            if source_unique_idx < 0:
                                continue
                            mask = inverse == source_unique_idx
                            if bool(mask.any().detach().cpu()):
                                bad_add_point_mask[0] |= mask
                                target_slot = int(add_item.get("target_child_slot", -1))
                                if 0 <= target_slot <= 7:
                                    bad_add_child_slot[0, mask] = target_slot

            if subtree_move_eval_limit > 0:
                subtree_move_candidates = _sparsepcgc_actual_oracle_subtree_move_candidates(
                    unique_coords,
                    args,
                    global_step,
                    subtree_move_max,
                    base_proxy_bits=base_proxy_bits,
                )
                debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + int(len(subtree_move_candidates))
                for move_item in subtree_move_candidates:
                    if _actual_budget_exhausted(tested) or subtree_move_tested >= subtree_move_eval_limit:
                        break
                    candidate_coords = move_item.get("transformed_coords", None)
                    if not torch.is_tensor(candidate_coords):
                        continue
                    candidate_xyz = _oracle_actual_eval_xyz(candidate_coords)
                    if candidate_xyz is None or candidate_xyz.shape[-1] <= 0:
                        continue
                    stats = loss._encode_actual_batch(args, candidate_xyz)
                    tested += 1
                    subtree_move_tested += 1
                    cand_bit = float(stats.get("bit", 0.0))
                    edit_record_bits = _sparsepcgc_edit_record_total_bits(
                        args,
                        edit_record_unique_count,
                        subtree_move_count=int(move_item.get("moved_count", 0)),
                        subtree_move_level_shift=int(move_item.get("level_shift", 1)),
                    )
                    proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                        candidate_coords,
                        args,
                        base_proxy_bits,
                    )
                    geometry_percent = _sparsepcgc_geometry_penalty_percent(
                        args,
                        edit_record_unique_count,
                        move_count=int(move_item.get("moved_count", 0)),
                        level_shift=int(move_item.get("level_shift", 1)),
                    )
                    raw_percent, actual_percent, cand_percent = _candidate_objective(
                        cand_bit,
                        edit_record_bits,
                        geometry_percent,
                    )
                    move_memory_key = str(move_item.get("memory_key", ""))
                    _sparsepcgc_actual_oracle_update_memory(args, move_memory_key, cand_percent)
                    _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                    if cand_percent < -min_improve:
                        subtree_move_improving_count += 1
                        improving_candidate_count += 1
                        improving.append(
                            {
                                "op": "subtree_move",
                                "final_coords": candidate_coords.detach().clone(),
                                "source_unique_indices": move_item.get("source_unique_indices", None),
                                "moved_count": int(move_item.get("moved_count", 0)),
                                "level_shift": int(move_item.get("level_shift", 0)),
                                "source_slot": int(move_item.get("source_slot", -1)),
                                "target_slot": int(move_item.get("target_slot", -1)),
                                "percent": float(cand_percent),
                                "raw_percent": float(raw_percent),
                                "actual_percent": float(actual_percent),
                                "proxy_percent": float(proxy_percent),
                                "proxy_bits": float(proxy_bits),
                                "geometry_percent": float(geometry_percent),
                                "edit_record_bits": float(edit_record_bits),
                                "memory_key": move_memory_key,
                            }
                        )
                    else:
                        bad_candidate_count += 1
                        source_unique_indices = move_item.get("source_unique_indices", None)
                        if torch.is_tensor(source_unique_indices):
                            source_unique_indices = source_unique_indices.detach().reshape(-1).tolist()
                        source_slot = int(move_item.get("source_slot", 0))
                        target_slot = int(move_item.get("target_slot", 0))
                        source_bits = unique_coords.new_tensor(
                            [source_slot & 1, (source_slot >> 1) & 1, (source_slot >> 2) & 1]
                        )
                        target_bits = unique_coords.new_tensor(
                            [target_slot & 1, (target_slot >> 1) & 1, (target_slot >> 2) & 1]
                        )
                        direction_index = _neighbor_direction_index(target_bits - source_bits)
                        for source_unique_idx in source_unique_indices or []:
                            mask = inverse == int(source_unique_idx)
                            if bool(mask.any().detach().cpu()):
                                bad_move_point_mask[0] |= mask
                                strength = _oracle_strength(cand_percent, bad=True)
                                bad_move_score[0, mask] = torch.maximum(
                                    bad_move_score[0, mask],
                                    bad_move_score.new_full((int(mask.sum().item()),), float(strength)),
                                )
                                bad_move_direction_index[0, mask] = int(direction_index)

                debug["subtree_move_tested_count"] = int(subtree_move_tested)
                debug["subtree_move_improving_count"] = int(subtree_move_improving_count)
                debug["tested_count"] = int(tested)
                debug["best_percent"] = float(best_percent)
                debug["best_raw_percent"] = float(best_raw_percent)
                debug["best_edit_record_bits"] = float(best_edit_record_bits)
                debug["bad_candidate_count"] = int(bad_candidate_count)
            debug["bad_candidate_count"] = int(bad_candidate_count)
            debug["improving_candidate_count"] = int(improving_candidate_count)
            debug["joint_tested_count"] = int(joint_tested)
            debug["joint_improving_count"] = int(joint_improving_count)
            debug["full_cloud_macro_tested_count"] = int(full_cloud_macro_tested)
            debug["full_cloud_macro_improving_count"] = int(full_cloud_macro_improving_count)
            debug["full_cloud_macro_best_percent"] = (
                float(full_cloud_macro_best_percent) if math.isfinite(float(full_cloud_macro_best_percent)) else 0.0
            )
            debug["full_cloud_macro_best_ratio"] = float(full_cloud_macro_best_ratio)
            debug["full_cloud_macro_best_drop_count"] = int(full_cloud_macro_best_drop_count)
            debug["macro_prune_tested_count"] = int(macro_prune_tested)
            debug["macro_prune_improving_count"] = int(macro_prune_improving_count)
            debug["macro_prune_best_percent"] = (
                float(macro_prune_best_percent) if math.isfinite(float(macro_prune_best_percent)) else 0.0
            )
            debug["macro_prune_best_ratio"] = float(macro_prune_best_ratio)
            debug["macro_prune_best_drop_count"] = int(macro_prune_best_drop_count)
            debug["macro_prune_best_variant"] = str(macro_prune_best_variant)
            debug["macro_prune_best_proxy_percent"] = float(macro_prune_best_proxy_percent)
            debug["group_tested_count"] = int(group_tested)
            debug["group_improving_count"] = int(group_improving_count)
            debug["pattern_plan_tested_count"] = int(pattern_plan_tested)
            debug["pattern_plan_improving_count"] = int(pattern_plan_improving_count)
            debug["parent_prune_tested_count"] = int(parent_prune_tested)
            debug["parent_prune_improving_count"] = int(parent_prune_improving_count)

            if improving:
                improving_selection_start = time.time()
                improving = sorted(improving, key=lambda item: float(item["percent"]))
                max_selected = max(int(getattr(args, "sparsepcgc_actual_oracle_max_selected_voxels", 4)), 1)
                combo_validate_max_extra = max(
                    int(getattr(args, "sparsepcgc_actual_oracle_combo_validate_max_extra", 2)),
                    0,
                )
                selected_drop = 0
                selected_add = 0
                selected_move = 0
                combo_extra_count = 0
                dropped_unique = set()
                selected_add_sources = set()
                selected_add_targets = []
                override_final_voxel_coords = None
                override_drop_count = 0
                override_subtree_prune_count = 0
                override_scope = ""
                selected_full_cloud_override = False
                current_combo_percent = 0.0
                selected_raw_percent = 0.0
                selected_actual_percent = 0.0
                selected_proxy_percent = 0.0
                selected_geometry_percent = 0.0
                selected_edited_actual_bits = float(base_bit)
                selected_edit_record_bits = 0.0
                accepted_parent_collapse_count = 0
                accepted_pattern_canonicalize_count = 0

                def _combo_coords(drop_set, add_targets):
                    keep_unique = torch.ones((unique_coords.shape[0],), device=coords.device, dtype=torch.bool)
                    if drop_set:
                        drop_idx = torch.as_tensor(sorted(drop_set), device=coords.device, dtype=torch.long)
                        keep_unique[drop_idx] = False
                    combo = unique_coords[keep_unique]
                    if add_targets:
                        combo = torch.cat(
                            [combo]
                            + [
                                target.to(device=coords.device, dtype=torch.long).view(1, 3)
                                for target in add_targets
                            ],
                            dim=0,
                        )
                        combo = torch.unique(combo, dim=0, sorted=True)
                    return combo

                def _mark_drop(unique_idx, strength):
                    nonlocal selected_drop
                    dropped_unique.add(int(unique_idx))
                    mask = inverse == int(unique_idx)
                    point_mask[0] |= mask
                    if bool(mask.any().detach().cpu()):
                        score[0, mask] = max(
                            float(strength),
                            float(score[0, mask].max().detach().cpu()),
                        )
                    selected_drop += 1

                def _mark_drop_many(unique_indices, strength):
                    nonlocal selected_drop
                    if not unique_indices:
                        return
                    valid_indices = sorted(
                        {
                            int(value)
                            for value in unique_indices
                            if 0 <= int(value) < int(unique_coords.shape[0])
                            and int(value) not in dropped_unique
                        }
                    )
                    if not valid_indices:
                        return
                    index_tensor = torch.as_tensor(
                        valid_indices,
                        device=inverse.device,
                        dtype=torch.long,
                    )
                    selected_unique_mask = torch.zeros(
                        (unique_coords.shape[0],),
                        device=inverse.device,
                        dtype=torch.bool,
                    )
                    selected_unique_mask[index_tensor] = True
                    mask = selected_unique_mask.index_select(0, inverse)
                    point_mask[0] |= mask
                    score[0] = torch.where(
                        mask,
                        torch.maximum(score[0], score[0].new_full(score[0].shape, float(strength))),
                        score[0],
                    )
                    dropped_unique.update(valid_indices)
                    selected_drop += len(valid_indices)

                def _mark_add(source_unique_idx, target_child_slot, target_coord_item, strength):
                    nonlocal selected_add
                    mask = inverse == int(source_unique_idx)
                    add_point_mask[0] |= mask
                    if bool(mask.any().detach().cpu()):
                        add_score[0, mask] = max(
                            float(strength),
                            float(add_score[0, mask].max().detach().cpu()),
                        )
                        add_child_slot[0, mask] = int(target_child_slot)
                        add_direction_index[0, mask] = _neighbor_direction_index(
                            target_coord_item.reshape(3) - unique_coords[int(source_unique_idx)]
                        )
                    selected_add_sources.add(int(source_unique_idx))
                    selected_add_targets.append(target_coord_item.detach().clone())
                    selected_add += 1

                def _item_edited_actual_bits(item):
                    if not isinstance(item, dict):
                        return float(base_bit)
                    if "edited_actual_bits" in item:
                        return float(item.get("edited_actual_bits", base_bit))
                    raw_percent_value = float(item.get("raw_percent", 0.0) or 0.0)
                    return float(base_bit * (1.0 + raw_percent_value / 100.0))

                for item in improving:
                    if selected_move > 0:
                        break
                    if selected_drop + selected_add >= max_selected:
                        break
                    strength = 1.0 + min(abs(float(item["percent"])) / 10.0, 4.0)
                    op_name = str(item.get("op", ""))

                    if op_name == "subtree_move":
                        if selected_drop + selected_add + selected_move > 0:
                            continue
                        final_coords_item = item.get("final_coords", None)
                        if not torch.is_tensor(final_coords_item):
                            continue
                        override_final_voxel_coords = final_coords_item.detach().clone()
                        selected_move = int(item.get("moved_count", 0))
                        current_combo_percent = float(item["percent"])
                        selected_raw_percent = float(item.get("raw_percent", item["percent"]))
                        selected_actual_percent = float(item.get("actual_percent", item["percent"]))
                        selected_proxy_percent = float(item.get("proxy_percent", 0.0))
                        selected_geometry_percent = float(item.get("geometry_percent", 0.0))
                        selected_edited_actual_bits = _item_edited_actual_bits(item)
                        selected_edit_record_bits = float(item.get("edit_record_bits", 0.0))
                        source_unique_indices = item.get("source_unique_indices", None)
                        if torch.is_tensor(source_unique_indices):
                            for source_unique_idx in source_unique_indices.detach().reshape(-1).to(
                                device=inverse.device,
                                dtype=inverse.dtype,
                            ):
                                mask = inverse == source_unique_idx
                                if bool(mask.any().detach().cpu()):
                                    move_point_mask[0] |= mask
                                    move_score[0, mask] = max(
                                        strength,
                                        float(move_score[0, mask].max().detach().cpu()),
                                    )
                                    source_slot = int(item.get("source_slot", 0))
                                    target_slot = int(item.get("target_slot", 0))
                                    source_bits = unique_coords.new_tensor(
                                        [source_slot & 1, (source_slot >> 1) & 1, (source_slot >> 2) & 1]
                                    )
                                    target_bits = unique_coords.new_tensor(
                                        [target_slot & 1, (target_slot >> 1) & 1, (target_slot >> 2) & 1]
                                    )
                                    move_direction_index[0, mask] = _neighbor_direction_index(
                                        target_bits - source_bits
                                    )
                        continue

                    if op_name == "pattern_plan":
                        if selected_drop + selected_add > 0:
                            continue
                        drop_indices = [int(v) for v in item.get("drop_unique_indices", [])]
                        add_items = list(item.get("add_items", []) or [])
                        if len(drop_indices) + len(add_items) <= 0:
                            continue
                        if len(drop_indices) + len(add_items) > max_selected:
                            continue
                        accepted_pattern_canonicalize_count = 1
                        current_combo_percent = float(item["percent"])
                        selected_raw_percent = float(item.get("raw_percent", item["percent"]))
                        selected_actual_percent = float(item.get("actual_percent", item["percent"]))
                        selected_proxy_percent = float(item.get("proxy_percent", 0.0))
                        selected_geometry_percent = float(item.get("geometry_percent", 0.0))
                        selected_edited_actual_bits = _item_edited_actual_bits(item)
                        selected_edit_record_bits = float(item.get("edit_record_bits", 0.0))
                        for unique_idx in drop_indices:
                            _mark_drop(int(unique_idx), strength)
                        for add_item in add_items:
                            target_coord_item = add_item.get("target_coord", None)
                            if not torch.is_tensor(target_coord_item):
                                continue
                            _mark_add(
                                int(add_item.get("source_unique_idx", -1)),
                                int(add_item.get("target_child_slot", -1)),
                                target_coord_item,
                                strength,
                            )
                        continue

                    if op_name == "drop_add":
                        if selected_drop + selected_add > 0:
                            continue
                        if selected_drop + selected_add + 2 > max_selected:
                            continue
                        unique_idx = int(item["unique_idx"])
                        source_unique_idx = int(item["source_unique_idx"])
                        if unique_idx == source_unique_idx:
                            continue
                        target_coord_item = item.get("target_coord", None)
                        if not torch.is_tensor(target_coord_item):
                            continue
                        current_combo_percent = float(item["percent"])
                        selected_raw_percent = float(item.get("raw_percent", item["percent"]))
                        selected_actual_percent = float(item.get("actual_percent", item["percent"]))
                        selected_proxy_percent = float(item.get("proxy_percent", 0.0))
                        selected_geometry_percent = float(item.get("geometry_percent", 0.0))
                        selected_edited_actual_bits = _item_edited_actual_bits(item)
                        selected_edit_record_bits = float(item.get("edit_record_bits", 0.0))
                        _mark_drop(unique_idx, strength)
                        _mark_add(
                            source_unique_idx,
                            int(item["target_child_slot"]),
                            target_coord_item,
                            strength,
                        )
                        continue

                    if op_name in {
                        "drop_group",
                        "parent_collapse",
                        "macro_prune",
                        "full_cloud_subtree_prune",
                    }:
                        if selected_drop + selected_add > 0:
                            continue
                        unique_indices = [int(v) for v in item.get("unique_indices", [])]
                        if not unique_indices and op_name != "full_cloud_subtree_prune":
                            continue
                        if op_name in {"parent_collapse", "full_cloud_subtree_prune"}:
                            accepted_parent_collapse_count = 1
                        if op_name == "full_cloud_subtree_prune":
                            final_coords_item = item.get("override_final_voxel_coords", None)
                            if not torch.is_tensor(final_coords_item):
                                continue
                            override_final_voxel_coords = final_coords_item.detach().clone()
                            override_drop_count = int(item.get("full_cloud_drop_count", 0) or 0)
                            override_subtree_prune_count = int(item.get("drop_block_count", 0) or 0)
                            override_scope = str(item.get("override_scope", "full_cloud"))
                            selected_full_cloud_override = True
                            actual_stats_item = item.get("actual_stats", None)
                            if isinstance(actual_stats_item, dict):
                                debug["cached_edited_actual_stats"] = dict(actual_stats_item)
                        current_combo_percent = float(item["percent"])
                        selected_raw_percent = float(item.get("raw_percent", item["percent"]))
                        selected_actual_percent = float(item.get("actual_percent", item["percent"]))
                        selected_proxy_percent = float(item.get("proxy_percent", 0.0))
                        selected_geometry_percent = float(item.get("geometry_percent", 0.0))
                        selected_edited_actual_bits = _item_edited_actual_bits(item)
                        selected_edit_record_bits = float(item.get("edit_record_bits", 0.0))
                        remaining = max(max_selected - selected_drop - selected_add, 0)
                        if unique_indices and remaining > 0:
                            _mark_drop_many(unique_indices[:remaining], strength)
                        continue

                    if op_name == "add_group":
                        if selected_drop + selected_add > 0:
                            continue
                        source_indices = [int(v) for v in item.get("source_unique_indices", [])]
                        target_slots = [int(v) for v in item.get("target_child_slots", [])]
                        target_coords = item.get("target_coords", None)
                        if not source_indices or not torch.is_tensor(target_coords):
                            continue
                        current_combo_percent = float(item["percent"])
                        selected_raw_percent = float(item.get("raw_percent", item["percent"]))
                        selected_actual_percent = float(item.get("actual_percent", item["percent"]))
                        selected_proxy_percent = float(item.get("proxy_percent", 0.0))
                        selected_geometry_percent = float(item.get("geometry_percent", 0.0))
                        selected_edited_actual_bits = _item_edited_actual_bits(item)
                        selected_edit_record_bits = float(item.get("edit_record_bits", 0.0))
                        target_coords = target_coords.to(device=coords.device, dtype=torch.long).view(-1, 3)
                        for local_idx, source_unique_idx in enumerate(source_indices):
                            if selected_drop + selected_add >= max_selected:
                                break
                            if local_idx >= target_coords.shape[0] or local_idx >= len(target_slots):
                                break
                            _mark_add(
                                source_unique_idx,
                                target_slots[local_idx],
                                target_coords[local_idx : local_idx + 1],
                                strength,
                            )
                        continue

                    if op_name == "drop":
                        unique_idx = int(item["unique_idx"])
                        if unique_idx in selected_add_sources:
                            continue
                        if selected_drop + selected_add == 0:
                            current_combo_percent = float(item["percent"])
                            selected_raw_percent = float(item.get("raw_percent", item["percent"]))
                            selected_actual_percent = float(item.get("actual_percent", item["percent"]))
                            selected_proxy_percent = float(item.get("proxy_percent", 0.0))
                            selected_geometry_percent = float(item.get("geometry_percent", 0.0))
                            selected_edited_actual_bits = _item_edited_actual_bits(item)
                            selected_edit_record_bits = float(item.get("edit_record_bits", 0.0))
                            _mark_drop(unique_idx, strength)
                            continue
                        if combo_extra_count >= combo_validate_max_extra:
                            continue
                        if _actual_budget_exhausted(tested):
                            continue
                        trial_drop = set(dropped_unique)
                        trial_drop.add(unique_idx)
                        trial_coords = _combo_coords(trial_drop, selected_add_targets)
                        if int(trial_coords.shape[0]) <= 0:
                            continue
                        trial_xyz = _oracle_actual_eval_xyz(trial_coords)
                        if trial_xyz is None or trial_xyz.shape[-1] <= 0:
                            continue
                        trial_stats = loss._encode_actual_batch(args, trial_xyz)
                        tested += 1
                        combo_extra_count += 1
                        trial_bit = float(trial_stats.get("bit", 0.0))
                        trial_edit_record_bits = _sparsepcgc_edit_record_total_bits(
                            args,
                            edit_record_unique_count,
                            drop_count=len(trial_drop),
                            add_count=len(selected_add_targets),
                        )
                        trial_proxy_bits, trial_proxy_percent = _sparsepcgc_proxy_delta_percent(
                            trial_coords,
                            args,
                            base_proxy_bits,
                        )
                        trial_geometry_percent = _sparsepcgc_geometry_penalty_percent(
                            args,
                            edit_record_unique_count,
                            drop_count=len(trial_drop),
                            add_count=len(selected_add_targets),
                        )
                        trial_raw_percent, trial_actual_percent, trial_percent = _candidate_objective(
                            trial_bit,
                            trial_edit_record_bits,
                            trial_geometry_percent,
                        )
                        if trial_percent >= current_combo_percent:
                            continue
                        current_combo_percent = float(trial_percent)
                        selected_raw_percent = float(trial_raw_percent)
                        selected_actual_percent = float(trial_actual_percent)
                        selected_proxy_percent = float(trial_proxy_percent)
                        selected_geometry_percent = float(trial_geometry_percent)
                        selected_edited_actual_bits = float(trial_bit)
                        selected_edit_record_bits = float(trial_edit_record_bits)
                        _mark_drop(unique_idx, strength)
                    elif op_name == "add":
                        source_unique_idx = int(item["source_unique_idx"])
                        if source_unique_idx in dropped_unique:
                            continue
                        target_coord_item = item.get("target_coord", None)
                        if not torch.is_tensor(target_coord_item):
                            continue
                        if selected_drop + selected_add == 0:
                            current_combo_percent = float(item["percent"])
                            selected_raw_percent = float(item.get("raw_percent", item["percent"]))
                            selected_actual_percent = float(item.get("actual_percent", item["percent"]))
                            selected_proxy_percent = float(item.get("proxy_percent", 0.0))
                            selected_geometry_percent = float(item.get("geometry_percent", 0.0))
                            selected_edited_actual_bits = _item_edited_actual_bits(item)
                            selected_edit_record_bits = float(item.get("edit_record_bits", 0.0))
                            _mark_add(
                                source_unique_idx,
                                int(item["target_child_slot"]),
                                target_coord_item,
                                strength,
                            )
                            continue
                        if combo_extra_count >= combo_validate_max_extra:
                            continue
                        if _actual_budget_exhausted(tested):
                            continue
                        trial_targets = list(selected_add_targets) + [target_coord_item.detach().clone()]
                        trial_coords = _combo_coords(dropped_unique, trial_targets)
                        trial_xyz = _oracle_actual_eval_xyz(trial_coords)
                        if trial_xyz is None or trial_xyz.shape[-1] <= 0:
                            continue
                        trial_stats = loss._encode_actual_batch(args, trial_xyz)
                        tested += 1
                        combo_extra_count += 1
                        trial_bit = float(trial_stats.get("bit", 0.0))
                        trial_edit_record_bits = _sparsepcgc_edit_record_total_bits(
                            args,
                            edit_record_unique_count,
                            drop_count=len(dropped_unique),
                            add_count=len(trial_targets),
                        )
                        trial_proxy_bits, trial_proxy_percent = _sparsepcgc_proxy_delta_percent(
                            trial_coords,
                            args,
                            base_proxy_bits,
                        )
                        trial_geometry_percent = _sparsepcgc_geometry_penalty_percent(
                            args,
                            edit_record_unique_count,
                            drop_count=len(dropped_unique),
                            add_count=len(trial_targets),
                        )
                        trial_raw_percent, trial_actual_percent, trial_percent = _candidate_objective(
                            trial_bit,
                            trial_edit_record_bits,
                            trial_geometry_percent,
                        )
                        if trial_percent >= current_combo_percent:
                            continue
                        current_combo_percent = float(trial_percent)
                        selected_raw_percent = float(trial_raw_percent)
                        selected_actual_percent = float(trial_actual_percent)
                        selected_proxy_percent = float(trial_proxy_percent)
                        selected_geometry_percent = float(trial_geometry_percent)
                        selected_edited_actual_bits = float(trial_bit)
                        selected_edit_record_bits = float(trial_edit_record_bits)
                        _mark_add(
                            source_unique_idx,
                            int(item["target_child_slot"]),
                            target_coord_item,
                            strength,
                        )

                debug["tested_count"] = int(tested)
                debug["combo_extra_count"] = int(combo_extra_count)
                debug["best_percent"] = float(min(best_percent, current_combo_percent))
                debug["used"] = bool(
                    selected_drop > 0
                    or selected_add > 0
                    or selected_move > 0
                    or selected_full_cloud_override
                )
                debug["selected_drop_count"] = int(selected_drop)
                debug["selected_add_count"] = int(selected_add)
                debug["selected_move_count"] = int(selected_move)
                debug["accepted_candidate_count"] = int(1 if debug["used"] else 0)
                debug["accepted_prune_count"] = int(selected_drop)
                debug["accepted_add_count"] = int(selected_add)
                debug["accepted_adjust_count"] = 0
                debug["accepted_subtree_move_count"] = int(1 if selected_move > 0 else 0)
                debug["accepted_parent_collapse_count"] = int(accepted_parent_collapse_count)
                debug["accepted_pattern_canonicalize_count"] = int(accepted_pattern_canonicalize_count)
                debug["selected_raw_percent"] = float(selected_raw_percent)
                debug["delta_actual_percent"] = float(selected_actual_percent)
                debug["selected_proxy_percent"] = float(selected_proxy_percent)
                debug["selected_geometry_percent"] = float(selected_geometry_percent)
                debug["edited_actual_bits"] = float(selected_edited_actual_bits)
                debug["selected_edit_record_bits"] = float(selected_edit_record_bits)
                if override_final_voxel_coords is not None:
                    debug["override_final_voxel_coords"] = override_final_voxel_coords.detach().clone()
                    debug["override_move_count"] = int(selected_move)
                    debug["override_drop_count"] = int(override_drop_count)
                    debug["override_subtree_prune_count"] = int(override_subtree_prune_count)
                    debug["override_scope"] = str(override_scope)
                debug["improving_selection_time"] = float(time.time() - improving_selection_start)
                if not debug["used"]:
                    debug["reason"] = "no_actual_improving_combo_candidate"
                elif selected_move > 0:
                    debug["reason"] = "actual_improving_subtree_move_found"
                elif selected_full_cloud_override:
                    debug["reason"] = "actual_improving_full_cloud_override_found"
                elif selected_drop > 0 and selected_add > 0:
                    debug["reason"] = "actual_improving_drop_add_found"
                elif selected_add > 0:
                    debug["reason"] = "actual_improving_add_found"
                else:
                    debug["reason"] = "actual_improving_drop_found"
            else:
                debug["reason"] = "no_actual_improving_candidate"
            debug["tested_count"] = int(tested)
            debug["best_percent"] = float(best_percent)
            debug["best_raw_percent"] = float(best_raw_percent)
            debug["best_actual_percent"] = float(best_actual_percent)
            debug["best_proxy_percent"] = float(best_proxy_percent)
            debug["best_edit_record_bits"] = float(best_edit_record_bits)
            debug["best_edited_actual_bits"] = float(best_edited_actual_bits)
            if (
                not bool(debug.get("used", False))
                and bool(getattr(args, "sparsepcgc_actual_oracle_force_no_edit", False))
                and float(getattr(args, "sparsepcgc_actual_oracle_noop_weight", 0.0)) > 0.0
            ):
                debug["noop_label_count"] = int(unique_count)
                debug["noop_label_weight"] = float(getattr(args, "sparsepcgc_actual_oracle_noop_weight", 0.0))
            debug["actual_oracle_time"] = float(time.time() - oracle_time_start)
        except Exception as exc:
            debug["reason"] = f"oracle_error:{exc}"
            debug["actual_oracle_time"] = float(time.time() - oracle_time_start)

    if (
        actual_validate_this_step
        and (not bool(debug.get("used", False)))
        and fast_diagnostic_indices
        and bool(getattr(args, "sparsepcgc_actual_oracle_fast_fallback_after_reject", False))
    ):
        previous_reason = str(debug.get("reason", ""))
        if _apply_fast_diagnostic_teacher():
            debug["reason"] = f"fast_diagnostic_after_actual_reject:{previous_reason}"

    oracle_enabled = bool(
        debug["used"]
        or int(bad_candidate_count) > 0
        or bool(getattr(args, "sparsepcgc_actual_oracle_force_no_edit", False))
    )
    patched_values = {
        "actual_oracle_enabled": bool(oracle_enabled),
        "actual_oracle_drop_mask": point_mask.detach(),
        "actual_oracle_drop_score": score.detach(),
        "actual_oracle_drop_bad_mask": bad_drop_point_mask.detach(),
        "actual_oracle_drop_bad_score": bad_drop_score.detach(),
        "actual_oracle_drop_used": bool(point_mask.any().detach().cpu()),
        "actual_oracle_drop_best_percent": float(debug["best_percent"]),
        "actual_oracle_drop_tested_count": int(debug["tested_count"]),
        "actual_oracle_bad_candidate_count": int(debug["bad_candidate_count"]),
        "actual_oracle_improving_candidate_count": int(debug["improving_candidate_count"]),
        "actual_oracle_combo_extra_count": int(debug["combo_extra_count"]),
        "actual_oracle_generated_candidate_count": int(debug.get("generated_candidate_count", debug.get("candidate_pool_count", 0))),
        "actual_oracle_accepted_candidate_count": int(debug.get("accepted_candidate_count", 0)),
        "actual_oracle_accepted_prune_count": int(debug.get("accepted_prune_count", 0)),
        "actual_oracle_accepted_add_count": int(debug.get("accepted_add_count", 0)),
        "actual_oracle_accepted_adjust_count": int(debug.get("accepted_adjust_count", 0)),
        "actual_oracle_accepted_subtree_move_count": int(debug.get("accepted_subtree_move_count", 0)),
        "actual_oracle_accepted_parent_collapse_count": int(debug.get("accepted_parent_collapse_count", 0)),
        "actual_oracle_accepted_pattern_canonicalize_count": int(debug.get("accepted_pattern_canonicalize_count", 0)),
        "actual_oracle_noop_label_count": int(debug.get("noop_label_count", 0)),
        "actual_oracle_noop_label_weight": float(debug.get("noop_label_weight", 0.0)),
        "actual_oracle_high_rate_mppov_count": int(debug.get("high_rate_mppov_count", 0)),
        "actual_oracle_low_prob_occupied_count": int(debug.get("low_prob_occupied_count", 0)),
        "actual_oracle_single_child_chain_count": int(debug.get("single_child_chain_count", 0)),
        "actual_oracle_context_pattern_candidate_count": int(debug.get("context_pattern_candidate_count", 0)),
        "actual_oracle_eval_count": int(debug.get("tested_count", 0)),
        "actual_oracle_eval_max_configured": int(debug.get("actual_eval_max_configured", debug.get("actual_eval_max", 0))),
        "actual_oracle_eval_max": int(debug.get("actual_eval_max", 0)),
        "actual_oracle_eval_scope": str(debug.get("actual_eval_scope", "")),
        "actual_oracle_eval_full_coord_count": int(debug.get("actual_eval_full_coord_count", 0)),
        "actual_oracle_full_cloud_teacher_required": bool(debug.get("full_cloud_teacher_required", False)),
        "actual_oracle_full_cloud_teacher_eval_available": bool(debug.get("full_cloud_teacher_eval_available", False)),
        "actual_oracle_time": float(debug.get("actual_oracle_time", 0.0)),
        "actual_oracle_original_actual_cache_hit": bool(debug.get("original_actual_cache_hit", False)),
        "actual_oracle_original_actual_encode_time": float(debug.get("original_actual_encode_time", 0.0) or 0.0),
        "actual_oracle_candidate_actual_encode_time": float(debug.get("candidate_actual_encode_time", 0.0) or 0.0),
        "actual_oracle_released_main_cuda_cache": bool(debug.get("released_main_cuda_cache", False)),
        "actual_oracle_drop_reason": str(debug["reason"]),
        "actual_oracle_scheduled_operation": str(debug.get("scheduled_operation", "")),
        "actual_oracle_add_mask": add_point_mask.detach(),
        "actual_oracle_add_score": add_score.detach(),
        "actual_oracle_best_add_child_slot": add_child_slot.detach(),
        "actual_oracle_best_add_direction_index": add_direction_index.detach(),
        "actual_oracle_move_mask": move_point_mask.detach(),
        "actual_oracle_move_score": move_score.detach(),
        "actual_oracle_move_direction_index": move_direction_index.detach(),
        "actual_oracle_move_bad_mask": bad_move_point_mask.detach(),
        "actual_oracle_move_bad_score": bad_move_score.detach(),
        "actual_oracle_move_bad_direction_index": bad_move_direction_index.detach(),
        "actual_oracle_add_bad_mask": bad_add_point_mask.detach(),
        "actual_oracle_add_bad_score": bad_add_score.detach(),
        "actual_oracle_bad_add_child_slot": bad_add_child_slot.detach(),
        "actual_oracle_bad_add_direction_index": bad_add_direction_index.detach(),
        "actual_oracle_add_used": bool(add_point_mask.any().detach().cpu()),
        "actual_oracle_move_used": bool(move_point_mask.any().detach().cpu()),
        "actual_oracle_override_final_voxel_coords": (
            debug["override_final_voxel_coords"].detach()
            if torch.is_tensor(debug.get("override_final_voxel_coords", None))
            else None
        ),
        "actual_oracle_override_move_count": int(debug.get("override_move_count", 0) or 0),
        "actual_oracle_override_drop_count": int(debug.get("override_drop_count", 0) or 0),
        "actual_oracle_override_subtree_prune_count": int(
            debug.get("override_subtree_prune_count", 0) or 0
        ),
        "actual_oracle_override_scope": str(debug.get("override_scope", "") or ""),
        "actual_oracle_cached_edited_actual_stats": (
            dict(debug["cached_edited_actual_stats"])
            if isinstance(debug.get("cached_edited_actual_stats", None), dict)
            else None
        ),
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
        "actual_oracle_fast_diagnostic_used": bool(debug.get("fast_diagnostic_used", False)),
        "actual_oracle_fast_diagnostic_full_drop_count": int(debug.get("fast_diagnostic_full_drop_count", 0) or 0),
        "actual_oracle_fast_diagnostic_local_drop_count": int(debug.get("fast_diagnostic_local_drop_count", 0) or 0),
        "actual_oracle_fast_diagnostic_full_drop_ratio": float(debug.get("fast_diagnostic_full_drop_ratio", 0.0) or 0.0),
        "actual_oracle_fast_diagnostic_local_drop_ratio": float(debug.get("fast_diagnostic_local_drop_ratio", 0.0) or 0.0),
        "actual_oracle_fast_diagnostic_full_add_count": int(debug.get("fast_diagnostic_full_add_count", 0) or 0),
        "actual_oracle_fast_diagnostic_local_add_count": int(debug.get("fast_diagnostic_local_add_count", 0) or 0),
        "actual_oracle_fast_diagnostic_full_add_ratio": float(debug.get("fast_diagnostic_full_add_ratio", 0.0) or 0.0),
        "actual_oracle_fast_diagnostic_local_add_ratio": float(debug.get("fast_diagnostic_local_add_ratio", 0.0) or 0.0),
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
        "actual_oracle_operation": str(debug["reason"]),
    }

    patched_tree = dict(subtree_tree or {})
    patched_tree.update(patched_values)
    patched_context = dict(full_octree_context or {})
    patched_context.update(patched_values)

    if (
        bool(getattr(args, "sparsepcgc_actual_oracle_log", True))
        and not bool(getattr(args, "compact_step_text_log", False))
        and writer is not None
        and hasattr(writer, "write")
    ):
        if bool(getattr(args, "_log_this_step", False)) or bool(debug["used"]) or bool(debug["enabled"]):
            writer.write(
                "SparsePCGCActualOracle: "
                f"enabled={bool(debug['enabled'])}, "
                f"used={bool(debug['used'])}, "
                f"candidates={int(debug['candidate_count'])}, "
                f"candidate_pool={int(debug.get('candidate_pool_count', debug['candidate_count']))}, "
                f"tested={int(debug['tested_count'])}, "
                f"eval_max={int(debug.get('actual_eval_max', 0))}, "
                f"eval_scope={str(debug.get('actual_eval_scope', ''))}, "
                f"eval_full_coords={int(debug.get('actual_eval_full_coord_count', 0))}, "
                f"single_eval_max={int(debug.get('single_eval_max', 0))}, "
                f"macro_eval_max={int(debug.get('macro_prune_eval_max', 0))}, "
                f"joint_eval_max={int(debug.get('joint_eval_max', 0))}, "
                f"group_eval_max={int(debug.get('group_eval_max', 0))}, "
                f"parent_eval_max={int(debug.get('parent_prune_eval_max', 0))}, "
                f"pattern_eval_max={int(debug.get('pattern_plan_eval_max', 0))}, "
                f"subtree_eval_max={int(debug.get('subtree_move_eval_max', 0))}, "
                f"combo_extra={int(debug.get('combo_extra_count', 0))}, "
                f"full_macro_eval_max={int(debug.get('full_cloud_macro_eval_max', 0))}, "
                f"full_macro_tested={int(debug.get('full_cloud_macro_tested_count', 0))}, "
                f"full_macro_improving={int(debug.get('full_cloud_macro_improving_count', 0))}, "
                f"full_macro_best={float(debug.get('full_cloud_macro_best_percent', 0.0)):.6f}, "
                f"full_macro_best_ratio={float(debug.get('full_cloud_macro_best_ratio', 0.0)):.4f}, "
                f"full_macro_best_drop={int(debug.get('full_cloud_macro_best_drop_count', 0))}, "
                f"fast_diag_used={bool(debug.get('fast_diagnostic_used', False))}, "
                f"fast_diag={str(debug.get('fast_diagnostic_name', ''))}, "
                f"fast_diag_thr={int(debug.get('fast_diagnostic_threshold', 0))}, "
                f"fast_diag_full_drop={int(debug.get('fast_diagnostic_full_drop_count', 0))}, "
                f"fast_diag_local_drop={int(debug.get('fast_diagnostic_local_drop_count', 0))}, "
                f"fast_diag_add={str(debug.get('fast_diagnostic_add_name', ''))}, "
                f"fast_diag_add_thr={int(debug.get('fast_diagnostic_add_threshold', 0))}, "
                f"fast_diag_full_add={int(debug.get('fast_diagnostic_full_add_count', 0))}, "
                f"fast_diag_local_add={int(debug.get('fast_diagnostic_local_add_count', 0))}, "
                f"macro_prune_tested={int(debug.get('macro_prune_tested_count', 0))}, "
                f"macro_prune_improving={int(debug.get('macro_prune_improving_count', 0))}, "
                f"macro_best={float(debug.get('macro_prune_best_percent', 0.0)):.6f}, "
                f"macro_best_ratio={float(debug.get('macro_prune_best_ratio', 0.0)):.4f}, "
                f"macro_best_drop={int(debug.get('macro_prune_best_drop_count', 0))}, "
                f"macro_best_variant={str(debug.get('macro_prune_best_variant', ''))}, "
                f"macro_best_proxy={float(debug.get('macro_prune_best_proxy_percent', 0.0)):.6f}, "
                f"joint_tested={int(debug.get('joint_tested_count', 0))}, "
                f"joint_improving={int(debug.get('joint_improving_count', 0))}, "
                f"group_tested={int(debug.get('group_tested_count', 0))}, "
                f"group_improving={int(debug.get('group_improving_count', 0))}, "
                f"parent_prune_tested={int(debug.get('parent_prune_tested_count', 0))}, "
                f"parent_prune_improving={int(debug.get('parent_prune_improving_count', 0))}, "
                f"pattern_plan_tested={int(debug.get('pattern_plan_tested_count', 0))}, "
                f"pattern_plan_improving={int(debug.get('pattern_plan_improving_count', 0))}, "
                f"subtree_move_tested={int(debug.get('subtree_move_tested_count', 0))}, "
                f"subtree_move_improving={int(debug.get('subtree_move_improving_count', 0))}, "
                f"improving={int(debug.get('improving_candidate_count', 0))}, "
                f"bad={int(debug.get('bad_candidate_count', 0))}, "
                f"accepted={int(debug.get('accepted_candidate_count', 0))}, "
                f"noop_labels={int(debug.get('noop_label_count', 0))}, "
                f"noop_weight={float(debug.get('noop_label_weight', 0.0)):.4f}, "
                f"high_rate_mppov={int(debug.get('high_rate_mppov_count', 0))}, "
                f"low_prob_occ={int(debug.get('low_prob_occupied_count', 0))}, "
                f"single_chain={int(debug.get('single_child_chain_count', 0))}, "
                f"context_pattern={int(debug.get('context_pattern_candidate_count', 0))}, "
                f"memory={len(getattr(args, '_sparsepcgc_actual_oracle_outcome_memory', {}) or {})}, "
                f"orig_bits={float(debug.get('original_actual_bits', 0.0)):.3f}, "
                f"edited_bits={float(debug.get('edited_actual_bits', 0.0)):.3f}, "
                f"delta_actual={float(debug.get('delta_actual_percent', 0.0)):.6f}, "
                f"delta_proxy={float(debug.get('selected_proxy_percent', 0.0)):.6f}, "
                f"geometry={float(debug.get('selected_geometry_percent', 0.0)):.6f}, "
                f"best_percent={float(debug['best_percent']):.6f}, "
                f"best_raw_percent={float(debug.get('best_raw_percent', 0.0)):.6f}, "
                f"best_actual_percent={float(debug.get('best_actual_percent', 0.0)):.6f}, "
                f"best_proxy_percent={float(debug.get('best_proxy_percent', 0.0)):.6f}, "
                f"selected_raw_percent={float(debug.get('selected_raw_percent', 0.0)):.6f}, "
                f"edit_record_bits={float(debug.get('selected_edit_record_bits', 0.0)):.3f}, "
                f"edit_record_scale={float(debug.get('edit_record_effective_scale', 0.0)):.4f}, "
                f"selected_drop={int(debug.get('selected_drop_count', 0))}, "
                f"selected_add={int(debug.get('selected_add_count', 0))}, "
                f"selected_move={int(debug.get('selected_move_count', 0))}, "
                f"accepted_parent_collapse={int(debug.get('accepted_parent_collapse_count', 0))}, "
                f"accepted_pattern_canonicalize={int(debug.get('accepted_pattern_canonicalize_count', 0))}, "
                f"oracle_time={float(debug.get('actual_oracle_time', 0.0)):.4f}s, "
                f"macro_gen_time={float(debug.get('full_cloud_macro_generate_time', 0.0)):.4f}s, "
                f"macro_map_time={float(debug.get('full_cloud_macro_local_map_time', 0.0)):.4f}s, "
                f"candidate_wall_time={float(debug.get('candidate_actual_wall_time', 0.0)):.4f}s, "
                f"local_proxy_time={float(debug.get('full_cloud_macro_local_proxy_time', 0.0)):.4f}s, "
                f"selection_time={float(debug.get('improving_selection_time', 0.0)):.4f}s, "
                f"reason={debug['reason']}"
            )

    return patched_tree, patched_context, debug


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
    for key, value in state_dict.items():
        key_text = str(key)
        new_key = key_text[7:] if key_text.startswith("module.") else key_text
        cleaned_state_dict[new_key] = value

    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)

    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

    writer.write(f"MoreTraining: checkpoint_format={checkpoint_format}")
    writer.write(f"MoreTraining: loaded_parameter_keys={len(cleaned_state_dict)}")
    writer.write(f"MoreTraining: missing_keys_count={len(missing_keys)}")
    writer.write(f"MoreTraining: unexpected_keys_count={len(unexpected_keys)}")

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

def train(model, args, loss, writer, plot, notifier=None):
    """==========================================================="""
    """セットアップ"""
    """==========================================================="""
    """基本情報"""
    set_seed(args.seed, deterministic=getattr(args, "deterministic", False)) # ランスシードを固定し、学習結果の再現性を確保する
    best_loss = float('inf') # 後続の計算・ログのため
    raw_seq_dirs = collect_seq_dirs2(args.input_dir, dataset_name=args.dataname) # 入力ディレクトリから学習対象のシーケンスディレクトリ一覧を集める
    seq_dirs = _limit_training_seq_dirs(raw_seq_dirs, args) # 8iだけ先頭3シーケンスに制限し、4つ目は使わない
    if (
        _episode_input_common_cache_enabled(args)
        and bool(getattr(args, "episode_input_common_cache_enable_dataset_cache", True))
        and not bool(getattr(args, "dataset_cache", False))
    ):
        args.dataset_cache = True
    num_seq = len(seq_dirs)
    writer.write(f"Total seq directories: {num_seq}")
    if len(seq_dirs) != len(raw_seq_dirs):
        kept_names = ", ".join(os.path.basename(seq_dir) for seq_dir in seq_dirs)
        writer.write(
            "8i training sequence limit applied: "
            f"using {len(seq_dirs)} of {len(raw_seq_dirs)} sequence directories"
        )
        writer.write(f"8i kept sequence dirs: {kept_names}")
    seq_datasets = [(seq_dir, PlyDirDataset(args, seq_dir)) for seq_dir in seq_dirs] # 各シーケンス内のPLY点群ファイルを読み込むデータセットを作る
    single_plan_teacher_store = None
    if str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() == "single_plan_student":
        exact_cache_root = str(getattr(args, "exact_teacher_cache_root", "") or "").strip()
        if exact_cache_root and os.path.isdir(exact_cache_root):
            single_plan_teacher_store = SinglePlanTeacherStore.from_exact_cache_root(
                exact_cache_root
            )
        if single_plan_teacher_store is not None and single_plan_teacher_store.states:
            writer.write(
                "SinglePlanExactTeacher: layer_a_hit=1, states={}, plans={}, "
                "hard_plan_apply=0, inference_reference=0".format(
                    len(single_plan_teacher_store.states),
                    sum(len(rows) for rows in single_plan_teacher_store.states.values()),
                )
            )
        else:
            raise RuntimeError(
                "Single-Plan訓練用Layer A Cacheがない。"
                "tools/build_exact_teacher_cache.pyでfingerprint検証済みcacheを構築すること。"
                "旧datasetへのsilent fallbackは許可しない"
            )
    k_proposal_teacher_store = None
    k_offline_path = str(getattr(args, "network_k_offline_dataset", "") or "").strip()
    k_all_actual_enabled = bool(getattr(args, "network_k_all_actual_enabled", False))
    k_cache_free_required = bool(getattr(
        args, "network_k_require_cache_free_training", True
    ))
    if (
        k_all_actual_enabled
        and k_cache_free_required
        and (
            k_offline_path
            or int(getattr(args, "network_k_offline_bootstrap_steps", 0)) > 0
        )
    ):
        raise RuntimeError(
            "Network-only K訓練でoffline候補/cache/teacherが有効になっている"
        )
    if k_all_actual_enabled:
        unique_training_files = sorted({
            os.path.realpath(path)
            for _, dataset in seq_datasets
            for path in getattr(dataset, "files", ())
        })
        writer.write(
            "KAllActualMode: "
            f"state_count={len(unique_training_files)}, K={int(args.network_k_proposal_count)}, "
            f"offline_bootstrap_steps_per_state={int(getattr(args, 'network_k_offline_bootstrap_steps', 0))}, "
            f"offline_bootstrap_cadence={int(getattr(args, 'network_k_offline_bootstrap_cadence', 5))}, "
            f"cache_free_training={int(k_cache_free_required)}, "
            f"offline_teacher={int(bool(k_offline_path and int(getattr(args, 'network_k_offline_bootstrap_steps', 0)) > 0))}, "
            "cache_plan=0, den5=0, den6=0, actual_per_step=K, reward=absolute_actual"
        )
    elif str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() == "network_k_proposal_policy":
        writer.write(
            "KAllActualMode: disabled; Critic選択済み1 planだけをActual評価する。"
            "8 proposal全件Actual学習ではない。"
        )
    if (
        str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
        == "network_k_proposal_policy"
        and k_offline_path
        and (
            not k_all_actual_enabled
            or int(getattr(args, "network_k_offline_bootstrap_steps", 0)) > 0
        )
    ):
        if not os.path.isfile(k_offline_path):
            raise FileNotFoundError(f"network_k_offline_dataset not found: {k_offline_path}")
        k_proposal_teacher_store = OfflineKProposalTeacherStore(k_offline_path)
        writer.write(
            "KProposalOfflineTeacher: "
            f"path={k_offline_path}, split={args.network_k_offline_split}, "
            f"states={len(k_proposal_teacher_store.states)}, "
            f"bootstrap_only={bool(k_all_actual_enabled)}, "
            "runtime_den6=0, candidate_actual=0"
        )
    total_train_files = sum(len(dataset) for _, dataset in seq_datasets) # 全シーケンスに含まれる点群ファイル数を合計し、総Step数の見積もりなどに使用
    den6_prefetch_lookahead = max(
        int(getattr(args, "heuristic_guidance_online_prefetch_lookahead", 0)), 0
    )
    if str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() not in {
        "ana_den6_online", "ana_den6_residual"
    }:
        den6_prefetch_lookahead = 0
    if seq_datasets and den6_prefetch_lookahead > 0 and not k_all_actual_enabled:
        first_files = list(getattr(seq_datasets[0][1], "files", ()))[:den6_prefetch_lookahead]
        prefetch_state = prefetch_ana_den6_online_guidance(args, first_files)
        if int(prefetch_state.get("submitted", 0)) > 0:
            writer.write(
                "Den6OnlinePrefetch: "
                f"workers={int(getattr(args, 'heuristic_guidance_online_prefetch_workers', 0))}, "
                f"lookahead={den6_prefetch_lookahead}, submitted={int(prefetch_state['submitted'])}"
            )
    args._total_train_steps_estimate = max(int(getattr(args, "episodes", 1)), 1) * max(int(total_train_files), 1) # Episode数と点群ファイル数からそう学修Step数を概算
    if _episode_input_common_cache_enabled(args):
        setattr(args, "_episode_input_common_cache", OrderedDict())
        setattr(args, "_episode_input_common_cache_bytes", 0)
        setattr(args, "_episode_input_common_cache_stats", {})
        auto_max_entries = max(int(total_train_files), 1)
        setattr(args, "_episode_input_common_cache_auto_max_entries", auto_max_entries)
        configured_max_entries = int(getattr(args, "episode_input_common_cache_max_entries", 0))
        effective_max_entries = configured_max_entries if configured_max_entries > 0 else auto_max_entries
        effective_max_memory_mb = int(getattr(args, "episode_input_common_cache_max_memory_mb", 0))
        writer.write(
            "EpisodeInputCommonCache: "
            f"enabled=True, dataset_cache={bool(getattr(args, 'dataset_cache', False))}, "
            f"max_entries={effective_max_entries}, "
            f"max_memory_mb={effective_max_memory_mb}"
        )
    # Phase7-4:
    # ablation modeは学習前に一度だけ適用する。
    # phase7_ablation_mode='none' の場合は何も上書きしない。
    _phase7_apply_ablation_mode(args, writer)

    set_cache_expected = getattr(model, "set_expected_input_cache_entries", None) # モデル側に入力キャッシュ件数を設定する変数
    if callable(set_cache_expected):
        set_cache_expected(total_train_files) # モデルに学習ファイル総数を通知し、入力キャッシュの総低用量を設定
    patch_info_cache = OrderedDict() # パッチ分割結果を入力ファイルごとに再利用するため

    """圧縮予測と実圧縮"""
    sparsepcgc_proxy_actual_pairs = [] # Sparse PCGCのProxy推定値と実測値のペアの保存
    codec_actual_metric_pairs = {} # Codex Proxy値とActual Codec値の対応保存
    case_debug_path = init_case_debug_csv(args, plot, writer) # 圧縮効率が良い/悪いケースを後から分析するためのCSVの初期化
    case_debug_counts = {"good": 0, "bad": 0}
    metric_csv_paths = init_metric_csvs(args, plot, writer) # 圧縮メトリクス/点操作メトリクス/ChackPoint判定値などの書き込み
    if bool(getattr(args, "save_compression_metric_csv", True)):
        metric_csv_paths["full_cloud_amount_sequence_summary"] = os.path.join(
            plot.save_dir,
            f"{args.time}_full_cloud_amount_sequence_summary.csv",
        )
        init_csv_file(
            metric_csv_paths["full_cloud_amount_sequence_summary"],
            FULL_CLOUD_AMOUNT_SEQUENCE_SUMMARY_COLUMNS,
            writer,
            "FullCloudAmountSequenceSummaryCSV",
        )
    if bool(getattr(args, "phase7_eval_summary", True)):
        metric_csv_paths["phase7_eval_summary"] = _phase7_eval_summary_path(args, plot)
        init_csv_file(
            metric_csv_paths["phase7_eval_summary"],
            PHASE7_EVAL_SUMMARY_COLUMNS,
            writer,
            "Phase7EvalSummaryCSV",
        )
    # 各損失項が各モジュール・点操作へ流す勾配量を記録するCSV
    step_grad_dir = getattr(plot, "save_dir", None) or getattr(args, "out_path", ".")
    metric_csv_paths["step_grad"] = os.path.join(step_grad_dir, f"{args.time}_MyNetwork_step_grad.csv")
    if bool(getattr(args, "step_grad_log", True)):
        init_csv_file(metric_csv_paths["step_grad"], STEP_GRAD_COLUMNS, writer, "StepGradCSV")
        writer.write(
            "StepGradCSVMode: "
            f"first_step_only={bool(getattr(args, 'step_grad_first_step_only', True))}, "
            f"interval={int(getattr(args, 'step_grad_log_interval', 1))}"
        )
    else:
        writer.write(f"StepGradCSV: disabled path={metric_csv_paths['step_grad']}")

    """原因診断のためのログ"""
    memory_diagnostics_path = os.path.join(
        step_grad_dir, f"{args.time}_memory_diagnostics.csv"
    )
    memory_diagnostics = MemoryDiagnosticsCSV(memory_diagnostics_path)

    def _record_memory(
        phase, *, episode=-1, epoch=-1, step=-1, global_step=-1, sample=""
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

    writer.write(f"MemoryDiagnosticsCSV: enabled path={memory_diagnostics_path}")
    for_better_path = init_for_better_logger(args, plot, writer) # 改善・改悪要因を記録する詳細分析ログ
    checkpoint_gate_refs = {} # ChackPoint保存判定で使う基準値や過去値を保持
    best_trackers = None # 複数指標でBest CheckPointを追跡するための状態を初期化
    actual_guard_state = {"best_delta": float("inf"), "best_path": None, "bad_count": 0} # 実Codex評価が悪化したときに、巻き戻す

    """モデル保存先ファイルのセットアップ"""
    output_dir = os.path.join(args.out_path)
    ckpt_dir = os.path.join(output_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    """学習セットアップ"""
    optimizer, scheduler_steplr = build_optimizer_and_scheduler( model, args, writer) # モデルの重み更新に使うOptimizerと学習率を変えるStepLR schedduler
    apply_optimizer_lr_floor(optimizer, args, label="main", writer=writer, global_step=0, reason="train_start") # main LRが開始時点からfloor未満なら下限値へ戻す
    amp_state = setup_amp( model, args, writer) # CUDA利用可否
    use_cuda = amp_state["use_cuda"] # GPU使用の有無
    use_amp = amp_state["use_amp"] # 自動混合精度で計算するか否か
    amp_dtype = amp_state["amp_dtype"] # AMPで使う浮動小数点型の保存
    amp_scaler_enabled = amp_state["amp_scaler_enabled"] # GradScalerを使うのか否か
    scaler = amp_state["scaler"] # AMPのGradScaler。AMPでスケーリングされた勾配を逆スケーリングしてOptimizerに渡すために使う
    amp_overflow_patience = amp_state["amp_overflow_patience"] # AMPでオーバーフローが起きたときに、学習を安定させるためにOptimizerのステップをスキップする回数の設定
    consecutive_amp_skips = amp_state["consecutive_amp_skips"] # AMPでオーバーフローが起きたときにOptimizerのステップをスキップする回数のカウンタ
    consecutive_nonfinite_grad_skips = 0
    warmup_whole_cloud_caches(model, args, loss, seq_datasets, writer, use_cuda, use_amp, amp_dtype) # 全体点群処理で使う重い前処理やCodec関連情報を先に作り、学習中の初回Stepだけ極端に遅くなるのを抑える
    loader_kwargs = build_loader_kwargs( args, model, writer, use_cuda) # DataLoaderに渡すBatchSize等の設定

    """Surrogate事前学習セットアップ"""
    run_surrogate_pretrain(model=model, args=args, loss=loss, seq_datasets=seq_datasets, loader_kwargs=loader_kwargs, metric_csv_paths=metric_csv_paths, ckpt_dir=ckpt_dir, writer=writer, plot=plot, use_cuda=use_cuda, use_amp=use_amp, amp_dtype=amp_dtype, for_better_path=for_better_path)
    post_pretrain_norm = surrogate_param_norm(loss) # Surrogateのパタラメータノルムを計算し、事前学習後に重みが拘引されたか、以上に大きくないかを確認
    surrogate_optimizer = getattr(loss, "surrogate_optimizer", None) # Lossオブジェクト内にあるSurrogate用のOptimizerを取得
    apply_optimizer_lr_floor(surrogate_optimizer, args, label="surrogate", writer=writer, global_step=0, reason="after_surrogate_pretrain") # Surrogate LRが事前学習後にfloor未満なら下限値へ戻す
    surrogate_lrs = optimizer_lrs(surrogate_optimizer) # Surrogate用Optimizerの学習率一覧を取り出す
    pretrain_label = ( "start after surrogate pretrain" if int(getattr(args, "surrogate_step", 0)) > 0 else "start") # Surrogate事前学習を実行したか否かでログの表示名を変える
    writer.write( f"[Training] {pretrain_label} " f"surrogate_param_norm={case_float(post_pretrain_norm, float('nan')):.6f} " f"lr={surrogate_lrs[0] if surrogate_lrs else 'NA'}")
    log_for_better_event( for_better_path, "training_start_after_surrogate_pretrain", label=pretrain_label, surrogate_param_norm=post_pretrain_norm, surrogate_lrs=surrogate_lrs) # Surrogate事前学習後の状態を詳細分ん積ログへ保存し、本学修開始時の条件として後から確認できるようにする
    if (
        bool(getattr(args, "sparsepcgc_warmup_worker_before_train", True))
        and not bool(getattr(args, "disable_actual_codec_during_train", False))
        and str(getattr(args, "compression_loss_backend", "")).strip().lower().startswith("sparsepcgc")
        and not (
            str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
            == "single_plan_student"
            and str(getattr(
                args, "single_plan_training_stage", "actual_calibration"
            )).strip().lower() in {"representation", "fast_distillation"}
        )
    ):
        warmup_actual_encoder = getattr(loss, "warmup_actual_encoder", None)
        if callable(warmup_actual_encoder):
            actual_worker_warmup_start = time.time()
            warmup_actual_encoder(args)
            writer.write(
                "SparsePCGCWorkerWarmup: persistent=True, "
                f"elapsed={time.time() - actual_worker_warmup_start:.3f}s"
            )
    optimizer.zero_grad(set_to_none=True) # 本学習開始前にOptimizer内の勾配を削除

    """==========================================================="""
    """トレーニング"""
    """==========================================================="""
    prev_stage = None
    global_train_step = 0
    global_epoch = 0
    scheduler_step_count = 0
    # 候補そのものは保存せず、同一stateが次のθ領域へ進むための訪問回数だけを保持する。
    network_k_state_visit_counts = {}
    _record_memory("train_loop_start", global_step=global_train_step)
    for episode in range(args.episodes): # Episode開始
        writer.write(f"◆◆◆ Episode {episode + 1} / {args.episodes} ◆◆◆")
        _record_memory(
            "episode_start", episode=episode + 1, global_step=global_train_step
        )

        """Stage変更"""
        current_stage = resolve_compression_fixed_stage(args) # EpisodeでStageを切り替えず、圧縮損失が常に効くjoint Stageへ固定する
        args.training_stage = current_stage
        if current_stage != prev_stage: # 前EpisodeとStageが異なる場合
            stage_factors = stage_loss_factors(args) # 現在Stageでっ各損失をどの比率で扱うか取得する
            stage_factors, stage_guard_debug = sparsepcgc_stage_guard_factors(
                args,
                current_stage,
                stage_factors,
            )
            writer.write(f"Training Stage Switch: episode={episode + 1}, stage={current_stage}")
            writer.write( "Stage Loss Factors: " f"geom={stage_factors['geom']}, com={stage_factors['com']}, " f"attr={stage_factors['attr']}, policy={stage_factors['policy']}, repair={stage_factors['repair']}")
            log_for_better_event( for_better_path, "stage_switch", episode=episode + 1, stage=current_stage, stage_factors=stage_factors)
            if bool(stage_guard_debug.get("stage_switch_guard_used", False)):
                writer.write(
                    "StageSwitchGuard: "
                    f"stage={current_stage}, "
                    f"com={stage_guard_debug['compression_loss_factor_original']:.4f}->{stage_guard_debug['compression_loss_factor_effective']:.4f}, "
                    f"policy={stage_guard_debug['policy_loss_factor_original']:.4f}->{stage_guard_debug['policy_loss_factor_effective']:.4f}"
                )
            prev_stage = current_stage

        model.train()

        """変数の初期化"""
        episode_metric_sums = None
        episode_checkpoint_sums = new_checkpoint_metric_sum()
        episode_compression_sums = new_compression_episode_sum()
        episode_operation_sums = new_operation_episode_sum()
        episode_sequence_summary = OrderedDict()
        episode_optimizer_total_count = 0
        episode_optimizer_step_count = 0
        episode_nonfinite_grad_skip_count = 0
        episode_max_consecutive_nonfinite_grad_skips = 0

        for epoch, (seq_dir, dataset) in enumerate(seq_datasets): # Epoch開始
            writer.write(f"⦿⦿⦿ Epoch {epoch + 1}/{num_seq} : {seq_dir} ⦿⦿⦿")
            sequence_name = os.path.basename(os.path.normpath(str(seq_dir)))
            _record_memory(
                "epoch_before_loader",
                episode=episode + 1,
                epoch=epoch + 1,
                global_step=global_train_step,
                sample=sequence_name,
            )

            """基本情報のセットアップ"""
            active_dataset = apply_epoch_file_window(dataset, args, episode) # 各系列の訓練用150件内をEpisodeごとにmax_files件ずつ進める
            loader = torch.utils.data.DataLoader(active_dataset, **loader_kwargs) # 現在Epochの窓Datasetから点群ファイルを順に読み出す
            num_steps = len(active_dataset)
            active_files = list(getattr(active_dataset, "files", ()))
            if den6_prefetch_lookahead > 0:
                prefetch_ana_den6_online_guidance(args, active_files[:den6_prefetch_lookahead])
            epoch_has_optimizer_step = False
            epoch_metric_sums = None
            _record_memory(
                "epoch_loader_ready",
                episode=episode + 1,
                epoch=epoch + 1,
                global_step=global_train_step,
                sample=sequence_name,
            )

            for step, pts in enumerate(loader): # Step開始
                """基本情報のセットアップ"""
                st_step = time.time()
                if den6_prefetch_lookahead > 0:
                    next_prefetch_index = step + den6_prefetch_lookahead
                    if next_prefetch_index < len(active_files):
                        prefetch_ana_den6_online_guidance(
                            args, (active_files[next_prefetch_index],)
                        )
                optimizer.zero_grad(set_to_none=True) # 前Stepの勾配を必ず消し、条件分岐による勾配蓄積を防ぐ
                file_path = active_dataset.files[step]
                _record_memory(
                    "step_after_data_load",
                    episode=episode + 1,
                    epoch=epoch + 1,
                    step=step + 1,
                    global_step=global_train_step,
                    sample=os.path.basename(str(file_path)),
                )
                # den6 v2 manifestは入力PLYのSHA256とcodec設定で厳密照合する。
                # proxyや別frameへ黙ってfallbackさせないため、各Stepで実ファイルを明示する。
                args._current_input_file = str(Path(file_path).expanduser().resolve())
                cache_key = make_step_cache_key(file_path, args) # ファイルパスと設定から一意なキーを作り、前処理結果、Codec結果、Patch情報などのキャッシュ参照に使う
                raw_pts_num = int(pts.shape[1] if pts.dim() == 3 else pts.shape[0]) # 受け取ったデータの元点数を数え、点数比較やログに使用
                sparsepcgc_training_mode = str(
                    getattr(args, "sparsepcgc_training_mode", "subtree_selector")
                ).strip().lower()
                heuristic_mode = str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
                if heuristic_mode == "network_k_proposal_policy":
                    state_visit = int(network_k_state_visit_counts.get(cache_key, 0))
                    args._network_k_state_visit = state_visit
                    args._network_k_current_state_key = cache_key
                    network_k_state_visit_counts[cache_key] = state_visit + 1
                den6_online_full_cloud = heuristic_mode == "ana_den6_online"
                network_only_full_cloud = heuristic_mode in {
                    "network_only_codec_policy", "network_k_proposal_policy", "single_plan_student"
                }
                single_plan_cache_only_stage = bool(
                    heuristic_mode == "single_plan_student"
                    and str(getattr(
                        args, "single_plan_training_stage", "actual_calibration"
                    )).strip().lower() in {"representation", "fast_distillation"}
                )
                one_plan_full_cloud = den6_online_full_cloud or network_only_full_cloud
                full_cloud_amount_mode = bool(sparsepcgc_training_mode == "full_cloud_amount")
                if one_plan_full_cloud:
                    full_cloud_amount_mode = False
                    args._current_teacher_scope = "full_cloud"

                """ログ判定"""
                log_this_step = should_log_step(step + 1, num_steps, args.print_rate) # このStepで通常ログを出すか判定
                compact_step_text_log = bool(getattr(args, "compact_step_text_log", True))
                profile_this_step = should_log_step(global_train_step + 1, max(int(getattr(args, "_total_train_steps_estimate", num_steps)), 1), int(getattr(args, "profile_interval", 100))) # Profileログを出すStepあ否かの判定
                timing_enabled = bool(
                    (getattr(args, "debug_timing", False) and log_this_step)
                    or (
                        (
                            getattr(args, "log_step_time", True)
                            or getattr(args, "log_gpu_memory", True)
                        )
                        and profile_this_step
                    )
                )
                # Network/Actuator側も同じprofile Stepだけ詳細計測する。
                # debug_timingを常時Trueにせず、通常Stepへ同期コストを持ち込まない。
                args._profile_runtime_this_step = bool(timing_enabled)

                """ログ用の変数セット"""
                args._global_train_step = int(global_train_step) # 現在の累積Step番号を保存
                # Loss is a mixin object rather than nn.Module.  Mark this
                # exact train step explicitly so ana_den6_online can enforce
                # one fresh edited actual encode and report its counters.
                args._den6_online_training_step_active = bool(one_plan_full_cloud)
                args._current_sample_name = os.path.basename(str(file_path)) # teacher/debugログに点群ファイル名を残す
                args._current_teacher_scope = "full_cloud"
                args._sparsepcgc_full_cloud_actual_primary_active = False
                args._log_this_step = False
                sparsepcgc_csv_debug = ( str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "") == "sparsepcgc" and bool(getattr(args, "save_compression_metric_csv", True))) # Sparse PCGC専用ログ
                operation_csv_debug = bool( getattr(args, "save_operation_metric_csv", getattr(args, "save_operation_metrics_csv", True))) # 点操作メトリクスCSVを保存するか判定し、点移動量や追加/削除などのDebug収集条件に使用
                args._collect_sparsepcgc_debug = bool(
                    (not one_plan_full_cloud)
                    and sparsepcgc_csv_debug
                    and should_collect_sparsepcgc_hard_debug(
                        args,
                        log_this_step=log_this_step,
                        profile_this_step=profile_this_step,
                        global_step=global_train_step,
                    )
                )
                # ana_den6 onlineの通常学習では未使用の巨大debug Tensor/辞書を作らない。
                args._collect_structure_debug = bool(
                    (not den6_online_full_cloud)
                    and (log_this_step or profile_this_step or operation_csv_debug or sparsepcgc_add_experiment_active(args))
                )
                detail_log_this_step = False
                step_timing_breakdown = {}
                step_actual_oracle_metric_debug = {}
                k_all_actual_result = None

                """学習設定"""
                if timing_enabled and use_cuda and torch.cuda.is_available(): # GPU計測のためのリセット
                    torch.cuda.reset_peak_memory_stats()

                if timing_enabled: # 時間計測が有効なら入力整形処理の開始時刻を記録
                    sync_for_timing(use_cuda) # GPUを使用している場合は、正確な時間計測のためにGPUの処理が完了するのを待つ
                    timing_data_start = time.time() # 時間計測開始
                input_pcd = prepare_full_cloud_input_pcd(pts, use_cuda)
                input_xyz = input_pcd[:, :3, :]
                input_attr_full = input_pcd[:, 3:, :].contiguous() if input_pcd.shape[1] > 3 else None

                pcd_pts_num = input_xyz.shape[-1]
                # ============================================================
                # このStepで使う唯一の voxel 座標系を full cloud から一度だけ作る。
                # Network / actual / surrogate / debug は必ずこれを基準にする。
                # ============================================================
                full_cloud_canonical_start = time.time()
                full_cloud_canonical_context = _episode_input_common_cache_fetch(
                    args,
                    _episode_input_common_cache_key(cache_key, "full_cloud_canonical"),
                    device=input_xyz.device,
                    section="full_cloud_canonical",
                )
                if full_cloud_canonical_context is None:
                    full_cloud_canonical_context = _build_full_cloud_octree_context_for_train(
                        input_xyz[:, :3, :],
                        args,
                        coord_scale=None,
                    )
                    _episode_input_common_cache_store(
                        args,
                        _episode_input_common_cache_key(cache_key, "full_cloud_canonical"),
                        full_cloud_canonical_context,
                    )
                step_timing_breakdown["full_cloud_canonical_build_time"] = float(time.time() - full_cloud_canonical_start)

                try:
                    setattr(args, "_full_cloud_canonical_context", full_cloud_canonical_context)
                    setattr(args, "_full_cloud_canonical_coords_count", int(full_cloud_canonical_context["global_voxel_coords"].shape[-1]))
                except Exception:
                    pass

                if timing_enabled: # 時間計測が有効なStepなら
                    sync_for_timing(use_cuda) # CUDA処理の同期
                    timing_data_end = time.time() # 入力整形処理の終了時刻を記録
                    timing_model_start = timing_data_end # モデル処理の開始時刻の記録

                """学習基本情報セットアップ"""
                clear_policy_terms = getattr(model, "clear_discrete_policy_terms", None) # モデルが前Stepで保持した離散方策用の一次損失・Log Probability・報酬情報などを消す関数を持っているか否か
                if callable(clear_policy_terms):
                    clear_policy_terms() # 前Stepの離散方策関連の一時値を消す
                loss_mode = lossmode(args) # 損失モードの取得
                compression_primary_mode = loss_mode == "compression_primary" # 圧縮損失重視
                stage_factors = stage_loss_factors(args) # 現在の学習Stageに応じた損失項の比率
                stage_factors, stage_guard_debug = sparsepcgc_stage_guard_factors(
                    args,
                    current_stage,
                    stage_factors,
                )
                if compression_primary_mode and not bool(getattr(args, "cp_use_stage_factors", False)):
                    stage_factors = {name: 1.0 for name in stage_factors} # 全Stage係数を全て1.0にする
                    stage_guard_debug["compression_loss_factor_effective"] = 1.0
                    stage_guard_debug["policy_loss_factor_effective"] = 1.0
                compute_compression = True # StageやModeに関係なく毎Stepで圧縮損失を計算する
                actual_refresh_interval = max(int(getattr(args, "actual_eval_interval", 0)), 0)
                refresh_actual_gen = bool(
                    global_train_step == 0
                    or (actual_refresh_interval > 0 and global_train_step % actual_refresh_interval == 0)
                ) # 実Codec/Surrogateの出力側更新は間引いて計算時間を抑える
                full_cloud_amount_actual_interval_active = actual_refresh_interval
                full_cloud_amount_actual_step = False
                if full_cloud_amount_mode:
                    if bool(getattr(args, "sparsepcgc_full_cloud_amount_fresh_actual_every_step", True)):
                        full_cloud_amount_actual_interval_active = 1
                        refresh_actual_gen = True
                    else:
                        warmup_actual_steps = max(
                            int(getattr(args, "sparsepcgc_full_cloud_amount_warmup_steps", 20)),
                            0,
                        )
                        interval_name = (
                            "sparsepcgc_full_cloud_amount_warmup_actual_interval"
                            if int(global_train_step) < warmup_actual_steps
                            else "sparsepcgc_full_cloud_amount_actual_interval"
                        )
                        full_cloud_amount_actual_interval_active = max(int(getattr(args, interval_name, 5)), 1)
                        refresh_actual_gen = bool(
                            global_train_step == 0
                            or int(global_train_step) % int(full_cloud_amount_actual_interval_active) == 0
                        )
                    full_cloud_amount_actual_step = bool(refresh_actual_gen)
                    try:
                        setattr(args, "_full_cloud_amount_actual_interval_active", int(full_cloud_amount_actual_interval_active))
                        setattr(args, "_full_cloud_amount_actual_step", bool(full_cloud_amount_actual_step))
                    except Exception:
                        pass

                """変数の初期化と設定"""
                is_anchor_step = True
                anchor_reason = "full_cloud_only"
                compression_cache_key = cache_key # キャッシュキーの初期化
                compression_gt_pts = input_xyz # 圧縮損失で比較する教師側点群を入力点群にする
                compression_gen_xyz = None # 圧縮Lossへ渡した出力点群をVoxel衝突ログで参照する
                train_edit_stats = None # 点操作を見計算状態にする
                noise_debug = empty_noise_debug() # 圧縮損失用に量子化前の点群に加えるノイズのデバッグ情報を初期化
                voxel_collision_input_gt = input_xyz[:, :3, :]
                encoder_debug_chunks = [] if detail_log_this_step else None
                """損失項の初期化"""
                L_geom = input_xyz.new_zeros(())
                L_com = input_xyz.new_zeros(())
                L_attr = input_xyz.new_zeros(())
                L_policy = input_xyz.new_zeros(())
                L_actuator = input_xyz.new_zeros(())
                Lp_out = input_xyz.new_zeros(())
                La_fit = input_xyz.new_zeros(())
                La_rep = input_xyz.new_zeros(())
                loss_bit = input_xyz.new_zeros(())
                loss_single = input_xyz.new_zeros(())
                loss_nodes = input_xyz.new_zeros(())
                L_full_cloud_amount = input_xyz.new_zeros(())
                full_cloud_amount_debug = {}
                full_cloud_amount_candidate_rows = []
                gen_xyz = None
                final_w = None
                out_label = None
                full_cloud_anchor_no_grad = False
                full_cloud_anchor_no_grad_reason = ""
                full_cloud_primary_override_debug = {}
                full_cloud_geometry_teacher_debug = {}
                full_cloud_anchor_runtime_timing = {}

                """モデルの実行"""
                prev_log_flag = getattr(args, "_log_this_step", False)
                try:
                    args._log_this_step = bool(
                        (not compact_step_text_log)
                        and getattr(args, "verbose_step_logs", False)
                        and detail_log_this_step
                    ) # このfull-cloud処理内で詳細ログを出すか否か決定
                    if is_anchor_step:
                        """全点群の場合"""
                        full_cloud_anchor_block_start = time.time()
                        args._current_teacher_scope = "full_cloud" # full-cloud anchorでは実圧縮teacherも全点群基準として記録する
                        args._current_teacher_anchor_reason = str(anchor_reason) # full-cloudになった理由をteacherログへ渡す
                        args._current_exact_teacher_mode = "full_cloud" # exact occupancy teacherは全点群基準で走らせる
                        args._current_exact_teacher_uses_full_context = False # 全点群はSubtree文脈を使わない
                        args._current_exact_teacher_fallback_reason = "" # full-cloudではfallback理由なし
                        if not compact_step_text_log:
                            writer.write("Running full cloud Anchor step.") # Anchor Stepであることをログに出す

                        # FullCloud anchorは原則no-gradだが、明示的に許可され、
                        # かつnode/voxel数が上限以内のときだけ学習graphを作る。
                        (
                            full_cloud_anchor_no_grad,
                            full_cloud_anchor_no_grad_reason,
                            full_cloud_anchor_node_count,
                            full_cloud_anchor_node_count_source,
                        ) = _resolve_full_cloud_anchor_no_grad(
                            args,
                            full_cloud_canonical_context,
                        )
                        if full_cloud_amount_mode or den6_online_full_cloud or heuristic_mode == "single_plan_student":
                            full_cloud_anchor_no_grad = False
                            full_cloud_anchor_no_grad_reason = (
                                "ana_den6_online_full_cloud_requires_grad"
                                if den6_online_full_cloud
                                else (
                                    "single_plan_student_distillation_requires_grad"
                                    if heuristic_mode == "single_plan_student"
                                    else "full_cloud_amount_train_branch_requires_grad"
                                )
                            )

                        if not compact_step_text_log:
                            writer.write(
                                "FullCloudAnchorMode: "
                                f"no_grad={bool(full_cloud_anchor_no_grad)}, "
                                f"reason={full_cloud_anchor_no_grad_reason}, "
                                f"node_count={int(full_cloud_anchor_node_count)}, "
                                f"node_count_source={full_cloud_anchor_node_count_source}, "
                                f"grad_node_limit={int(getattr(args, 'full_cloud_anchor_grad_node_limit', 50000))}, "
                                f"allow_grad={bool(getattr(args, 'full_cloud_anchor_allow_grad', False))}"
                            )
                        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                        grad_ctx = torch.no_grad() if full_cloud_anchor_no_grad else nullcontext()
                        saved_tensor_threshold_mb = float(getattr(
                            args, "full_cloud_saved_tensor_cpu_offload_mb", 0.25
                        ))
                        model_saved_tensor_ctx = selective_saved_tensor_cpu_offload(
                            saved_tensor_threshold_mb,
                            pin_memory=bool(getattr(
                                args, "full_cloud_saved_tensor_pin_memory", False
                            )),
                            enabled=(
                                network_only_full_cloud
                                and use_cuda
                                and not full_cloud_anchor_no_grad
                            ),
                        )

                        with grad_ctx, autocast_ctx, model_saved_tensor_ctx: # 全体点群をno-gradでモデルに入力し、teacher更新用の出力だけ得る
                            """モデルの実行"""
                            # Step冒頭で作った full cloud canonical context をそのまま使う。
                            # ここで再量子化してはいけない。
                            full_octree_context = dict(full_cloud_canonical_context)
                            full_octree_context["octree_context_scope"] = "full_cloud"
                            full_octree_context["octree_input_mode"] = "full_cloud"
                            full_octree_context["canonical_source"] = "full_cloud_canonical"
                            full_octree_context["fast_full_cloud_oracle_anchor"] = False
                            _sparsepcgc_apply_amount_outcome_context(
                                args,
                                memory_key=None,
                                forward_key=cache_key,
                            )
                            # offline教師が存在する訓練frameだけ、実在sourceを勾配候補へ追加する。
                            # 自然shortlistは別途保持し、推論recallとして過大評価しない。
                            args._network_k_training_teacher_coords = None
                            args._network_k_training_teacher_target_coords = None
                            if (
                                heuristic_mode == "network_k_proposal_policy"
                                and isinstance(k_proposal_teacher_store, OfflineKProposalTeacherStore)
                                and not k_all_actual_enabled
                            ):
                                training_state_id = k_proposal_teacher_store.find_state_for_input(
                                    file_path,
                                    args,
                                    split=str(getattr(args, "network_k_offline_split", "train")),
                                )
                                if training_state_id is not None:
                                    args._network_k_training_teacher_coords = (
                                        k_proposal_teacher_store.training_source_coordinates(
                                            training_state_id
                                        )
                                    )
                                    target_set_cadence = max(int(getattr(
                                        args, "network_k_target_set_loss_cadence", 5
                                    )), 1)
                                    if global_train_step % target_set_cadence == 0:
                                        args._network_k_training_teacher_target_coords = (
                                            k_proposal_teacher_store.training_target_coordinates(
                                                training_state_id
                                            )
                                        )
                            gen_pts, L_attr, L_policy, L_actuator, final_w, Lp_out, La_fit, La_rep, out_label = model.forward(
                                input_xyz,
                                input_attr_full,
                                cache_key=cache_key,
                                return_attr_output=False,
                                compute_internal_losses=not bool(full_cloud_anchor_no_grad),
                                full_octree_context=full_octree_context,
                                octree_input_mode="full_cloud",
                            )
                            args._network_k_training_teacher_coords = None
                            args._network_k_training_teacher_target_coords = None
                        if network_only_full_cloud and use_cuda:
                            # Saved autograd tensors are already on CPU after
                            # leaving save_on_cpu. Return now-unused forward
                            # workspaces before the persistent codec worker
                            # allocates its encode buffers.
                            torch.cuda.empty_cache()
                        try:
                            full_cloud_anchor_runtime_timing = dict(
                                getattr(model.module if hasattr(model, "module") else model, "last_runtime_timing", {}) or {}
                            )
                        except Exception:
                            full_cloud_anchor_runtime_timing = {}
                        if final_w is not None and not torch.isfinite(final_w).all(): # final重みにNanやinfが混ざっていないか確認
                            writer.write( "Warning: final_w contains NaN/Inf. " "It will be sanitized before point-edit summary and losses.")
                            final_w = torch.nan_to_num(final_w, nan=0.0, posinf=1.0, neginf=0.0) # 変換
                            final_w = final_w.clamp(0.0, 1.0) # 変換
                        if detail_log_this_step:
                            base_model = model.module if hasattr(model, "module") else model # DataParallelで包まれているばあいは中身のモデルを取り出す
                            encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {})) # Encoder Debug情報をコピーして保存
                        gen_xyz = gen_pts[:, :3, :]
                        _log_sparsepcgc_restore_debug(args, writer, out_label)
                        edit_summary_t0 = time.time()
                        train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args) # 入力点群と出力点群を比較し、各操作の編集統計を計算
                        step_timing_breakdown["point_edit_summary_time"] = float(time.time() - edit_summary_t0)
                        final_w_for_loss = None
                        if _discrete_loss_mode_value(args) != "hard":
                            final_w_for_loss = final_w
                        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext() # 形状損失と圧縮損失の計算もAMP文脈で行うための設定を作る
                        loss_grad_ctx = torch.no_grad() if full_cloud_anchor_no_grad else nullcontext()
                        loss_saved_tensor_ctx = selective_saved_tensor_cpu_offload(
                            saved_tensor_threshold_mb,
                            pin_memory=bool(getattr(
                                args, "full_cloud_saved_tensor_pin_memory", False
                            )),
                            enabled=(
                                network_only_full_cloud
                                and use_cuda
                                and not full_cloud_anchor_no_grad
                            ),
                        )

                        with loss_grad_ctx, autocast_ctx, loss_saved_tensor_ctx:
                            """形状損失の計算"""
                            geometry_t0 = time.time()
                            if single_plan_cache_only_stage:
                                # Stage 1/2はCache教師だけで更新し、fresh geometry/Actualを呼ばない。
                                L_geom = input_xyz.new_zeros(())
                                full_cloud_geometry_teacher_debug.update({
                                    "single_plan_cache_only_geometry_skipped": True,
                                })
                            elif full_cloud_amount_mode:
                                geom_mode = str(
                                    getattr(args, "sparsepcgc_full_cloud_amount_geometry_mode", "sampled")
                                ).strip().lower()
                                if geom_mode == "off":
                                    L_geom = input_xyz.new_zeros(())
                                    full_cloud_geometry_teacher_debug.update(
                                        {
                                            "full_cloud_amount_geometry_mode": "off",
                                            "full_cloud_amount_geom_sample_points": 0,
                                        }
                                    )
                                else:
                                    run_full_geom = bool(
                                        geom_mode == "interval_full"
                                        and (
                                            int(global_train_step)
                                            % max(int(getattr(args, "sparsepcgc_full_cloud_amount_geom_interval", 20)), 1)
                                            == 0
                                        )
                                    )
                                    if geom_mode == "sampled" or not run_full_geom:
                                        geom_sample_points = max(
                                            int(getattr(args, "sparsepcgc_full_cloud_amount_geom_sample_points", 20000)),
                                            1,
                                        )
                                        geom_gen = _sample_full_cloud_amount_geom_points(gen_xyz, geom_sample_points)
                                        geom_gt = _sample_full_cloud_amount_geom_points(input_xyz[:, :3, :], geom_sample_points)
                                        geom_final_w = None
                                        L_geom = loss.get_geometry_loss(
                                            args,
                                            gen_pts=geom_gen,
                                            gt_pts=geom_gt,
                                            final_w=geom_final_w,
                                            out_label=out_label,
                                        )
                                        full_cloud_geometry_teacher_debug.update(
                                            {
                                                "full_cloud_amount_geometry_mode": "sampled",
                                                "full_cloud_amount_geom_sample_points": int(geom_sample_points),
                                            }
                                        )
                                    else:
                                        L_geom = loss.get_geometry_loss(
                                            args,
                                            gen_pts=gen_xyz,
                                            gt_pts=input_xyz[:, :3, :],
                                            final_w=final_w_for_loss,
                                            out_label=out_label,
                                        )
                                        full_cloud_geometry_teacher_debug.update(
                                            {
                                                "full_cloud_amount_geometry_mode": "interval_full",
                                                "full_cloud_amount_geom_sample_points": int(input_xyz.shape[-1]),
                                            }
                                        )
                            else:
                                L_geom = loss.get_geometry_loss(
                                    args,
                                    gen_pts=gen_xyz,
                                    gt_pts=input_xyz[:, :3, :],
                                    final_w=final_w_for_loss,
                                    out_label=out_label,
                                )
                            step_timing_breakdown["geometry_loss_time"] = float(time.time() - geometry_t0)

                            """圧縮損失の計算"""
                            if stage_factors["com"] != 0.0 and not single_plan_cache_only_stage:
                                compression_t0 = time.time()
                                gen_xyz_for_actual, voxel_restored_actual_debug = _select_actual_gen_xyz_from_voxel_state(
                                    args,
                                    writer,
                                    model,
                                    gen_xyz,
                                    prefix="VoxelRestoredActual[full_cloud_anchor]",
                                    canonical_context=full_cloud_canonical_context,
                                )

                                full_cloud_voxel_state_used = bool(
                                    isinstance(voxel_restored_actual_debug, dict)
                                    and voxel_restored_actual_debug.get("used", False)
                                    and not voxel_restored_actual_debug.get("fallback", False)
                                )

                                # voxel state 復元に成功した場合は、proxy側もactual側も同じ点群を使う。
                                # 復元に失敗した場合だけ従来のgen_xyzへfallbackする。
                                full_cloud_compression_source_xyz = gen_xyz_for_actual if full_cloud_voxel_state_used else gen_xyz

                                if k_all_actual_enabled:
                                    k_actual_t0 = time.time()
                                    k_model = _unwrap_train_model(model)
                                    proposal_output = getattr(
                                        k_model, "last_k_proposal_terms", None
                                    )
                                    actuator_voxel_state = getattr(
                                        k_model, "last_actuator_voxel_state", None
                                    )
                                    evaluator = getattr(
                                        loss, "evaluate_network_k_plans_actual", None
                                    )
                                    if not callable(evaluator):
                                        raise RuntimeError("K all-Actual evaluatorがLossに存在しない")
                                    k_all_actual_result = evaluator(
                                        args,
                                        proposal_output=proposal_output,
                                        voxel_state=actuator_voxel_state,
                                        gt_xyz=input_xyz[:, :3, :],
                                        cache_key=cache_key,
                                    )
                                    # 通常のselected-plan損失ではK評価済みの同一結果を再利用し、
                                    # 9回目の重複Actual encodeを発生させない。
                                    full_octree_context[
                                        "actual_oracle_cached_edited_actual_stats"
                                    ] = dict(k_all_actual_result["selected_stats"])
                                    full_octree_context[
                                        "actual_oracle_override_scope"
                                    ] = "full_cloud"
                                    full_octree_context[
                                        "network_k_all_actual_reuse"
                                    ] = True
                                    step_timing_breakdown["k_all_actual_time"] = float(
                                        time.time() - k_actual_t0
                                    )

                                compression_gen_xyz, noise_debug = prepare_compression_points(
                                    full_cloud_compression_source_xyz,
                                    args,
                                    model,
                                    collect_stats=bool(log_this_step or profile_this_step),
                                ) # 圧縮損失用の入力点群を作る

                                args._current_exact_teacher_mode = "full_cloud"
                                args._current_exact_teacher_uses_full_context = False
                                args._current_exact_teacher_fallback_reason = ""

                                L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss(
                                    args,
                                    gen_xyz=compression_gen_xyz,
                                    gt_xyz=input_xyz[:, :3, :],
                                    final_w=final_w_for_loss,
                                    cache_key=cache_key,
                                    refresh_actual_gen=refresh_actual_gen,
                                    actual_gen_xyz=gen_xyz_for_actual,
                                    full_octree_context=full_octree_context,
                                    octree_input_mode="full_cloud",
                                )
                                step_timing_breakdown["compression_loss_time"] = float(
                                    time.time() - compression_t0
                                )
                                if (
                                    bool(getattr(args, "sparsepcgc_actual_oracle_apply_full_override", False))
                                    and
                                    isinstance(step_actual_oracle_metric_debug, dict)
                                    and bool(step_actual_oracle_metric_debug.get("used", False))
                                    and str(step_actual_oracle_metric_debug.get("override_scope", "")) == "full_cloud"
                                ):
                                    oracle_billed_percent = finite_float_or_none(
                                        step_actual_oracle_metric_debug.get("delta_actual_percent", None)
                                    )
                                    edit_record_bits = max(
                                        float(step_actual_oracle_metric_debug.get("selected_edit_record_bits", 0.0) or 0.0),
                                        0.0,
                                    )
                                    if oracle_billed_percent is not None:
                                        billed_debug = dict(getattr(loss, "last_compression_debug", {}) or {})
                                        gt_actual_bit_for_override = finite_float_or_none(
                                            billed_debug.get(
                                                "gt_actual_bit",
                                                billed_debug.get("gt_bit_abs", None),
                                            )
                                        )
                                        final_encoded_bit = finite_float_or_none(
                                            billed_debug.get(
                                                "gen_actual_bit",
                                                billed_debug.get("gen_bit_abs", None),
                                            )
                                        )
                                        oracle_edited_bit = finite_float_or_none(
                                            step_actual_oracle_metric_debug.get("edited_actual_bits", None)
                                        )
                                        policy_final_raw_percent = None
                                        policy_final_billed_percent = None
                                        policy_final_total_bit_with_edit_record = None
                                        if (
                                            gt_actual_bit_for_override is not None
                                            and gt_actual_bit_for_override > 0.0
                                            and final_encoded_bit is not None
                                        ):
                                            policy_final_total_bit_with_edit_record = (
                                                float(final_encoded_bit) + float(edit_record_bits)
                                            )
                                            policy_final_raw_percent = 100.0 * (
                                                float(final_encoded_bit) - float(gt_actual_bit_for_override)
                                            ) / float(gt_actual_bit_for_override)
                                            policy_final_billed_percent = 100.0 * (
                                                float(final_encoded_bit)
                                                + float(edit_record_bits)
                                                - float(gt_actual_bit_for_override)
                                            ) / float(gt_actual_bit_for_override)
                                        if (
                                            gt_actual_bit_for_override is not None
                                            and gt_actual_bit_for_override > 0.0
                                            and oracle_edited_bit is not None
                                            and oracle_edited_bit > 0.0
                                        ):
                                            raw_percent = 100.0 * (
                                                float(oracle_edited_bit) - float(gt_actual_bit_for_override)
                                            ) / float(gt_actual_bit_for_override)
                                            billed_percent = float(oracle_billed_percent)
                                            edited_actual_bit_for_log = float(oracle_edited_bit)
                                            override_bit_source = "oracle_cached_candidate_encode"
                                        else:
                                            raw_percent = finite_float_or_none(
                                                step_actual_oracle_metric_debug.get("selected_raw_percent", None)
                                            )
                                            billed_percent = float(oracle_billed_percent)
                                            if oracle_edited_bit is not None and oracle_edited_bit > 0.0:
                                                edited_actual_bit_for_log = float(oracle_edited_bit)
                                                override_bit_source = "oracle_cached_candidate_encode"
                                            else:
                                                edited_actual_bit_for_log = float(final_encoded_bit or 0.0)
                                                override_bit_source = "fresh_final_full_cloud_encode_fallback"
                                        objective_percent, objective_bit_source = _sparsepcgc_pick_objective_percent(
                                            args,
                                            raw_percent,
                                            billed_percent,
                                        )
                                        if objective_percent is None:
                                            objective_percent = float(billed_percent)
                                            objective_bit_source = "billed_fallback_missing"
                                        objective_tensor = L_com.new_tensor(float(objective_percent))
                                        L_com = objective_tensor + (L_com - L_com.detach())
                                        loss_bit = objective_tensor + (loss_bit - loss_bit.detach())
                                        billed_debug.update(
                                            {
                                                "total_bit": float(billed_percent),
                                                "actual_total_bit_percent": float(billed_percent),
                                                "actual_train_objective_percent": float(objective_percent),
                                                "actual_objective_percent": float(objective_percent),
                                                "actual_bit_objective": str(_sparsepcgc_actual_bit_objective_mode(args)),
                                                "actual_objective_bit_source": str(objective_bit_source),
                                                "actual_bit_percent": float(billed_percent),
                                                "actual_delta_percent": float(billed_percent),
                                                "actual_raw_percent": float(raw_percent)
                                                if raw_percent is not None
                                                else float(billed_percent),
                                                "actual_edit_record_bits": float(edit_record_bits),
                                                "actual_total_bits": float(edited_actual_bit_for_log) + float(edit_record_bits),
                                                "gen_actual_bit": float(edited_actual_bit_for_log),
                                                "gen_total_bit_with_edit_record": float(edited_actual_bit_for_log)
                                                + float(edit_record_bits),
                                                "actual_target": float(objective_percent),
                                                "actual_forward_value": float(objective_percent),
                                                "actual_bit_percent_used_for_loss": float(objective_percent),
                                                "compression_loss_used": float(objective_percent),
                                                "compression_forward_teacher_percent": float(objective_percent),
                                                "forward_display_value": float(objective_percent),
                                                "policy_actual_percent": policy_final_billed_percent,
                                                "oracle_teacher_actual_percent": float(oracle_billed_percent),
                                                "policy_full_cloud_actual_bit_percent": policy_final_billed_percent,
                                                "policy_action_source": "actual_oracle_full_cloud_override",
                                                "oracle_full_cloud_raw_bit_percent": finite_float_or_none(
                                                    step_actual_oracle_metric_debug.get("selected_raw_percent", None)
                                                ),
                                                "oracle_full_cloud_actual_bit_percent": float(oracle_billed_percent),
                                                "oracle_full_cloud_override_used": True,
                                                "oracle_full_cloud_override_bit_source": str(override_bit_source),
                                                "policy_final_full_cloud_raw_bit_percent": policy_final_raw_percent,
                                                "policy_final_full_cloud_actual_bit_percent": policy_final_billed_percent,
                                                "policy_final_full_cloud_gt_bit": gt_actual_bit_for_override,
                                                "policy_final_full_cloud_gen_bit": final_encoded_bit,
                                                "policy_final_full_cloud_total_bit_with_edit_record": (
                                                    policy_final_total_bit_with_edit_record
                                                ),
                                            }
                                        )
                                        loss.last_compression_debug = billed_debug
                            else:
                                if not compact_step_text_log:
                                    writer.write(
                                        "Skipping fresh compression: cache-only Single-Plan stage"
                                        if single_plan_cache_only_stage
                                        else "Skipping compression loss due to stage factor"
                                    )
                                zero = input_xyz.new_zeros(())
                                L_com = zero
                                loss_bit = zero
                                loss_single = zero
                                loss_nodes = zero
                        step_timing_breakdown["full_cloud_anchor_block_time"] = float(
                            time.time() - full_cloud_anchor_block_start
                        )
                finally:
                    args._log_this_step = prev_log_flag
                if timing_enabled:
                    sync_for_timing(use_cuda)
                    timing_model_end = time.time()

                """損失の計算"""
                if timing_enabled:
                    timing_loss_start = time.time()
                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext() # Loss計算用のAMPコンテキストを作る
                with autocast_ctx:
                    final_w_for_loss = None # Lossに渡す点操作重みの初期化
                    if _discrete_loss_mode_value(args) != "hard": # 離散損失モードがHard以外か判定する
                        final_w_for_loss = locals().get("final_w", None)
                        # final_w_for_loss = final_w
                    if timing_enabled:
                        sync_for_timing(use_cuda)
                        timing_noise_start = time.time()
                    if timing_enabled:
                        sync_for_timing(use_cuda)
                        timing_noise_end = time.time()

                if compute_compression: # このStepで圧縮損失を計算した場合
                    comp_debug_for_noise = getattr(loss, "last_compression_debug", {}) or {} # 圧縮辞書の取得
                    comp_debug_for_noise.update( { "uniform_noise_enabled": bool(noise_debug.get("enabled", False)), "uniform_noise_applied": bool(noise_debug.get("applied", False)), "uniform_noise_delta": float(noise_debug.get("delta", 0.0)), "uniform_noise_mean_abs": float(noise_debug.get("mean_abs", 0.0)), "compression_input_noisy": bool(noise_debug.get("applied", False))}) # 平均絶対ノイズを追加
                    loss.last_compression_debug = comp_debug_for_noise # ノイズ情報を追記した圧縮Debug辞書をLossに保存しなおす

                """圧縮損失の合成"""
                if (
                    bool(getattr(args, "sparsepcgc_actual_oracle_apply_full_override", False))
                    and
                    torch.is_tensor(L_geom)
                    and isinstance(step_actual_oracle_metric_debug, dict)
                    and bool(step_actual_oracle_metric_debug.get("used", False))
                    and str(step_actual_oracle_metric_debug.get("override_scope", "")) == "full_cloud"
                ):
                    oracle_geometry_percent = finite_float_or_none(
                        step_actual_oracle_metric_debug.get("selected_geometry_percent", None)
                    )
                    geometry_before = finite_float_or_none(L_geom)
                    if oracle_geometry_percent is not None and geometry_before is not None:
                        geometry_grad_scale = min(
                            1.0,
                            max(abs(float(oracle_geometry_percent)), 1e-3)
                            / max(abs(float(geometry_before)), 1e-3),
                        )
                        L_geom = L_geom.new_tensor(float(oracle_geometry_percent)) + geometry_grad_scale * (
                            L_geom - L_geom.detach()
                        )
                        full_cloud_geometry_teacher_debug = {
                            "full_cloud_geometry_teacher_used": True,
                            "full_cloud_geometry_teacher_value": float(oracle_geometry_percent),
                            "full_cloud_geometry_shadow_before": float(geometry_before),
                            "full_cloud_geometry_grad_scale": float(geometry_grad_scale),
                        }

                # compression loss側で作られた微分可能な内訳を取得する。
                terms = dict(getattr(loss, "last_compression_terms", {}) or {})
                compression_debug_terms = dict(getattr(loss, "last_compression_debug", {}) or {})
                actual_total_bit_percent_term = compression_debug_terms.get(
                    "actual_total_bit_percent_fresh",
                    compression_debug_terms.get("actual_total_bit_percent", None),
                )
                if actual_total_bit_percent_term is not None:
                    if torch.is_tensor(L_com):
                        terms = dict(terms)
                        terms["actual_total_bit_percent"] = L_com.new_tensor(float(actual_total_bit_percent_term))
                        terms["actual_total_bit_percent_fresh"] = L_com.new_tensor(float(actual_total_bit_percent_term))
                    else:
                        terms = dict(terms)
                        terms["actual_total_bit_percent"] = float(actual_total_bit_percent_term)
                        terms["actual_total_bit_percent_fresh"] = float(actual_total_bit_percent_term)
                if torch.is_tensor(loss_bit):
                    terms = dict(terms)
                    terms["proxy_bit"] = loss_bit
                    
                L_com_objective = compose_train_compression_objective(args, terms, L_com, La_fit) # actual/surrogateではL_com直結と内訳合成を半々で混ぜる
                surrogate_trust_value, surrogate_trust_debug = _sparsepcgc_surrogate_trust(
                    args,
                    compression_debug_terms,
                )
                network_only_trust_gate = (
                    str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
                    in {"network_only_codec_policy", "network_k_proposal_policy", "single_plan_student"}
                )
                if network_only_trust_gate:
                    # A pretrained Surrogate from the legacy action
                    # distribution can initially be several percentage points
                    # wrong on Network-only plans.  Train that Surrogate on the
                    # fresh scalar as usual, but do not let its uncalibrated
                    # gradient steer the action policy.  The forward value is
                    # still the one edited-cloud Actual value.
                    surrogate_error = finite_float_or_none(
                        compression_debug_terms.get(
                            "surrogate_abs_bit_error",
                            compression_debug_terms.get("surrogate_bit_error", None),
                        )
                    )
                    trust_low = max(
                        float(getattr(args, "network_only_surrogate_trust_error", 0.05)),
                        0.0,
                    )
                    trust_high = max(
                        float(getattr(args, "network_only_surrogate_disable_error", 0.50)),
                        trust_low,
                    )
                    if surrogate_error is None:
                        surrogate_trust_value = 0.0
                    elif surrogate_error <= trust_low:
                        surrogate_trust_value = 1.0
                    elif surrogate_error >= trust_high:
                        surrogate_trust_value = 0.0
                    else:
                        surrogate_trust_value = 1.0 - (
                            (surrogate_error - trust_low)
                            / max(trust_high - trust_low, 1e-12)
                        )
                    surrogate_trust_debug.update({
                        "network_only_surrogate_trust_gate": True,
                        "surrogate_trust_value": float(surrogate_trust_value),
                        "network_only_surrogate_trust_error": float(trust_low),
                        "network_only_surrogate_disable_error": float(trust_high),
                    })
                surrogate_loss_before_trust = finite_float_or_none(L_com_objective)
                if float(surrogate_trust_value) < 1.0 and torch.is_tensor(L_com_objective):
                    if network_only_trust_gate:
                        # Teacher-STE: preserve the Actual forward scalar and
                        # scale only the Surrogate backward contribution.
                        L_com_objective = (
                            L_com_objective.detach()
                            + float(surrogate_trust_value)
                            * (L_com_objective - L_com_objective.detach())
                        )
                    else:
                        L_com_objective = (
                            float(surrogate_trust_value) * L_com_objective
                            + (1.0 - float(surrogate_trust_value)) * (float(getattr(args, "w_com", 1.0)) * L_com)
                        )
                surrogate_trust_debug["surrogate_loss_before_trust"] = (
                    float(surrogate_loss_before_trust)
                    if surrogate_loss_before_trust is not None
                    else float("nan")
                )
                surrogate_trust_debug["surrogate_loss_after_trust"] = (
                    float(finite_float_or_none(L_com_objective))
                    if finite_float_or_none(L_com_objective) is not None
                    else float("nan")
                )
                # ============================================================
                # 非有限損失の保険
                # ============================================================
                # Actuator内部で inf / nan が出ても L_total 全体を壊さないようにする。
                # 根本原因は structure_actuator.py 側で潰すが、train側でも防御する。
                # ============================================================
                L_actuator = torch.nan_to_num(
                    L_actuator,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                L_attr = torch.nan_to_num(
                    L_attr,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                L_policy = torch.nan_to_num(
                    L_policy,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                L_geom = torch.nan_to_num(
                    L_geom,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                L_com_objective = torch.nan_to_num(
                    L_com_objective,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                compression_tensor_debug = {
                    "compression_loss_tensor_value": finite_float_or_none(L_com),
                    "compression_loss_requires_grad": bool(torch.is_tensor(L_com) and L_com.requires_grad),
                    "compression_loss_grad_fn": (
                        type(L_com.grad_fn).__name__
                        if torch.is_tensor(L_com) and getattr(L_com, "grad_fn", None) is not None
                        else ""
                    ),
                    "compression_objective_tensor_value": finite_float_or_none(L_com_objective),
                    "compression_objective_requires_grad": bool(
                        torch.is_tensor(L_com_objective) and L_com_objective.requires_grad
                    ),
                    "compression_objective_grad_fn": (
                        type(L_com_objective.grad_fn).__name__
                        if torch.is_tensor(L_com_objective) and getattr(L_com_objective, "grad_fn", None) is not None
                        else ""
                    ),
                    "loss_bit_tensor_value": finite_float_or_none(loss_bit),
                    "loss_bit_requires_grad": bool(torch.is_tensor(loss_bit) and loss_bit.requires_grad),
                    "loss_bit_grad_fn": (
                        type(loss_bit.grad_fn).__name__
                        if torch.is_tensor(loss_bit) and getattr(loss_bit, "grad_fn", None) is not None
                        else ""
                    ),
                }
                compression_tensor_debug.update(full_cloud_geometry_teacher_debug)
                compression_tensor_debug.update(surrogate_trust_debug)
                if full_cloud_amount_mode:
                    base_model_for_full_cloud_amount = _unwrap_train_model(model)
                    full_cloud_amount_terms = dict(
                        getattr(base_model_for_full_cloud_amount, "last_actuator_soft_terms", {}) or {}
                    )
                    full_cloud_amount_structure_debug = dict(
                        getattr(base_model_for_full_cloud_amount, "last_structure_debug", {}) or {}
                    )
                    actual_percent_for_full_cloud_amount = _sparsepcgc_outcome_actual_percent(compression_debug_terms)
                    actual_available_for_full_cloud_amount = bool(
                        full_cloud_amount_actual_step
                        and actual_percent_for_full_cloud_amount is not None
                        and not bool(compression_debug_terms.get("actual_codec_fallback_to_proxy", False))
                    )
                    full_cloud_amount_drop_count = case_int(
                        full_cloud_amount_structure_debug.get(
                            "hard_drop_count",
                            full_cloud_amount_structure_debug.get(
                                "selected_drop_count_hard",
                                full_cloud_amount_structure_debug.get(
                                    "voxel_edit_drop_count",
                                    0,
                                ),
                            ),
                        ),
                        0,
                    )
                    (
                        L_full_cloud_amount,
                        full_cloud_amount_debug,
                        full_cloud_amount_candidate_rows,
                    ) = _build_sparsepcgc_full_cloud_amount_candidate_teacher_loss(
                        args,
                        full_cloud_amount_terms,
                        compression_debug=compression_debug_terms,
                        structure_debug=full_cloud_amount_structure_debug,
                        loss_obj=loss,
                        base_model=base_model_for_full_cloud_amount,
                        full_cloud_context=full_octree_context,
                        gt_xyz=input_xyz[:, :3, :],
                        actual_percent=actual_percent_for_full_cloud_amount,
                        actual_available=actual_available_for_full_cloud_amount,
                        cache_key=cache_key,
                        global_step=global_train_step,
                        episode=episode,
                        epoch=epoch,
                        step=step,
                        sequence_name=sequence_name,
                        input_points=int(input_xyz.shape[-1]),
                        drop_count=int(full_cloud_amount_drop_count),
                        geom_loss=L_geom,
                    )
                    if not torch.is_tensor(L_full_cloud_amount):
                        L_full_cloud_amount = input_xyz.new_zeros(())
                    if isinstance(full_cloud_amount_debug, dict):
                        full_cloud_amount_debug.update(
                            {
                                "sparsepcgc_training_mode": "full_cloud_amount",
                                "actual_scope": "full_cloud",
                                "full_cloud_amount_fresh_actual_every_step": bool(
                                    getattr(args, "sparsepcgc_full_cloud_amount_fresh_actual_every_step", True)
                                ),
                                "full_cloud_amount_actual_interval": int(full_cloud_amount_actual_interval_active),
                                "full_cloud_amount_actual_step": bool(full_cloud_amount_actual_step),
                            }
                        )
                        objective_value = finite_float_or_none(
                            full_cloud_amount_debug.get(
                                "actual_objective_percent",
                                full_cloud_amount_debug.get("actual_train_objective_percent", None),
                            )
                        )
                        if objective_value is not None:
                            if torch.is_tensor(L_com):
                                L_com = L_com.new_tensor(float(objective_value)) + (L_com - L_com.detach())
                            if torch.is_tensor(loss_bit):
                                loss_bit = loss_bit.new_tensor(float(objective_value)) + (loss_bit - loss_bit.detach())
                            if torch.is_tensor(L_com_objective):
                                L_com_objective = L_com_objective.new_tensor(float(objective_value)) + (
                                    L_com_objective - L_com_objective.detach()
                                )
                        compression_tensor_debug.update(full_cloud_amount_debug)
                    if full_cloud_amount_candidate_rows:
                        candidate_path = metric_csv_paths.get("full_cloud_amount_candidate_step")
                        for full_cloud_amount_candidate_row in full_cloud_amount_candidate_rows:
                            append_csv_row(
                                candidate_path,
                                FULL_CLOUD_AMOUNT_CANDIDATE_COLUMNS,
                                full_cloud_amount_candidate_row,
                            )

                """形状損失を合成"""
                legacy_L_downstream = (
                    stage_factors["geom"] * args.w_geom * L_geom
                    + stage_factors["com"] * float(getattr(args, "w_com", 10.0)) * L_com_objective
                ) # 形状損失と圧縮損失の合成

                """属性/方策/操作損失を合成"""
                legacy_L_total = ( legacy_L_downstream + stage_factors["attr"] * args.w_attr * L_attr + stage_factors["policy"] * args.w_policy * L_policy + stage_factors["repair"] * args.w_actuator * L_actuator)

                """損失の合成"""
                L = legacy_L_total
                L_downstream = legacy_L_downstream
                L_discrete_policy = L.new_zeros(())
                cp_debug = {} # compression primaryモード用のdebug情報を空辞書で初期化
                compression_support_anchor = L_com_objective
                if compression_primary_mode and not network_only_full_cloud: # legacy圧縮優先経路
                    L, L_com_objective, cp_debug = build_compression_primary_loss(
                        args,
                        terms=terms,
                        L_com=L_com,
                        L_geom=L_geom,
                        L_actuator=L_actuator,
                        global_train_step=global_train_step,
                        stage_factors=stage_factors,
                    )
                    compression_support_anchor = L_com_objective
                    # L_com_objective に後から足す gradient-only proxy を、
                    # 実際に backward される L にも反映するための蓄積変数である。
                    # forward値は0なので、損失値自体は変えない。
                    compression_extra_grad_delta = None

                    # ============================================================
                    # Compression Primary の勾配復帰
                    # ============================================================
                    # build_compression_primary_loss が hard actual bit だけを目的にした場合、
                    # L_com_objective が no_grad_graph になる。
                    # その場合、forward値は hard actual のまま維持し、
                    # backwardだけ loss_bit / loss_nodes / loss_single / op 由来の
                    # 微分可能proxyへ流す。
                    #
                    # 重要：
                    #   Surrogate予測値そのものは使わない。
                    #   terms["surrogate"] はここに入れない。
                    # ============================================================
                    if not (torch.is_tensor(L_com_objective) and L_com_objective.requires_grad):
                        # ============================================================
                        # Compression Primary の勾配復帰
                        # ============================================================
                        # forward値は L_com_objective の値を維持する。
                        # backwardだけ、微分可能な圧縮proxyへ流す。
                        # これにより、L_com が Add / Prune / Move の Where と Amount に届く。
                        # ============================================================

                        compression_grad_terms = []

                        bit_term = terms.get("bit", None)
                        if torch.is_tensor(bit_term) and bit_term.requires_grad:
                            compression_grad_terms.append(
                                float(getattr(args, "com_bit", 1.0)) * bit_term
                            )

                        node_term = terms.get("node", None)
                        if torch.is_tensor(node_term) and node_term.requires_grad:
                            compression_grad_terms.append(
                                float(getattr(args, "cp_lambda_nodes", 1.0)) * node_term
                            )

                        single_term = terms.get("single", None)
                        if torch.is_tensor(single_term) and single_term.requires_grad:
                            compression_grad_terms.append(
                                float(getattr(args, "cp_lambda_single", 1.0)) * single_term
                            )

                        op_term = terms.get("op", None)
                        if (
                            torch.is_tensor(op_term)
                            and op_term.requires_grad
                            and float(getattr(args, "cp_lambda_op", 0.0)) > 0.0
                        ):
                            compression_grad_terms.append(
                                float(getattr(args, "cp_lambda_op", 0.0)) * op_term
                            )

                        if compression_grad_terms:
                            compression_proxy_for_grad = compression_grad_terms[0]
                            for term in compression_grad_terms[1:]:
                                compression_proxy_for_grad = compression_proxy_for_grad + term

                            compression_proxy_for_grad = torch.nan_to_num(
                                compression_proxy_for_grad,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                            # ============================================================
                            # 圧縮proxy勾配のPrune Where倍率
                            # ============================================================
                            # 目的:
                            #   compression_primary_proxy_grad_weight は圧縮proxy全体の基本倍率である。
                            #   ただし現在の圧縮proxy勾配はほぼ Prune Where(drop_head) に集中している。
                            #
                            #   そのため、grad_scale_prune_where_compression をここで掛ける。
                            #
                            # 現在の目標:
                            #   prune_where_drop_head ≒ 1202
                            #   grad_scale_prune_where_compression = 0.17
                            #   1202 * 0.17 ≒ 204
                            # ============================================================
                            proxy_grad_weight = float(
                                getattr(args, "compression_primary_proxy_grad_weight", 0.10)
                            )

                            prune_where_compression_scale = max(
                                float(getattr(args, "grad_scale_prune_where_compression", 1.0)),
                                0.0,
                            )

                            proxy_grad_weight = proxy_grad_weight * prune_where_compression_scale

                            if torch.is_tensor(L_com_objective):
                                compression_proxy_grad_delta = proxy_grad_weight * (
                                    compression_proxy_for_grad - compression_proxy_for_grad.detach()
                                )

                                L_com_objective = L_com_objective + compression_proxy_grad_delta

                                if compression_extra_grad_delta is None:
                                    compression_extra_grad_delta = compression_proxy_grad_delta
                                else:
                                    compression_extra_grad_delta = compression_extra_grad_delta + compression_proxy_grad_delta
                            else:
                                L_com_objective = compression_proxy_for_grad.detach() + proxy_grad_weight * (
                                    compression_proxy_for_grad - compression_proxy_for_grad.detach()
                                )

                            # step_gradログ上でも L_com が同じ勾配経路を持つようにする
                            L_com = L_com_objective

                            if isinstance(cp_debug, dict):
                                cp_debug["compression_grad_fallback_used"] = True
                                cp_debug["compression_grad_fallback_source"] = "always_bit_node_single_op_proxy_ste"
                                cp_debug["compression_primary_proxy_grad_weight"] = proxy_grad_weight

                        else:
                            if isinstance(cp_debug, dict):
                                cp_debug["compression_grad_fallback_used"] = False
                                cp_debug["compression_grad_fallback_source"] = "no_grad_proxy_available"

                    # ============================================================
                    # Prune Where 専用の L_com 勾配復帰
                    # ============================================================
                    # 目的
                    # ・forward値は一切変えない
                    # ・backwardだけ Prune Where、つまり drop_head へ返す
                    # ・target_drop_ratio へ寄せるMSEは使わない
                    # ・SparsePCGCで有効な「bit/node/singleを減らす方向」のproxyを使う
                    # ============================================================

                    # ============================================================
                    # Prune勾配リバランス
                    # ============================================================
                    # 目的:
                    #   Whereへ偏った後付け勾配を止め、Amount anchorの効果を見る。
                    #
                    # 注意:
                    #   ここでは診断を優先し、Where anchor scaleは0にする。
                    #   後で安定したら 0.01 や 0.05 に戻してよい。
                    # ============================================================
                    prune_grad_rebalance = True
                    prune_where_anchor_scale = 0.0

                    actuator_soft_terms = {}

                    base_model_for_prune_proxy = _unwrap_train_model(model)
                    model_soft_terms = getattr(
                        base_model_for_prune_proxy,
                        "last_actuator_soft_terms",
                        {},
                    )
                    if isinstance(model_soft_terms, dict):
                        actuator_soft_terms.update(model_soft_terms)

                    if isinstance(out_label, dict):
                        for key in (
                            "prune_where_proxy",
                            "soft_drop_where_grad_base",
                            "soft_drop_prob_for_ste",
                            "learned_drop_logit",
                            "drop_logit",
                            "drop_prob_proxy",
                            "prune_soft_geom",
                            "prune_soft_rate",
                            "prune_soft_node",
                            "prune_soft_single",
                            "prune_soft_bit",
                        ):
                            value = out_label.get(key, None)
                            if torch.is_tensor(value):
                                actuator_soft_terms[key] = value

                    prune_where_grad_terms = []

                    # ------------------------------------------------------------
                    # bit/node/single/rateを減らす方向のPrune Where proxy
                    # ------------------------------------------------------------
                    # prune_soft_bit/node/single/rate は、削除すべき構造的に重い点を
                    # drop_prob_proxy 経由で学習させるための項である。
                    # ------------------------------------------------------------

                    prune_bit_term = actuator_soft_terms.get("prune_soft_bit", None)
                    if torch.is_tensor(prune_bit_term) and prune_bit_term.requires_grad:
                        prune_where_grad_terms.append(
                            float(getattr(args, "compression_soft_prune_bit_grad_weight", 30.0))
                            * prune_bit_term
                        )

                    prune_node_term = actuator_soft_terms.get("prune_soft_node", None)
                    if torch.is_tensor(prune_node_term) and prune_node_term.requires_grad:
                        prune_where_grad_terms.append(
                            float(getattr(args, "compression_soft_prune_node_grad_weight", 25.0))
                            * prune_node_term
                        )

                    prune_single_term = actuator_soft_terms.get("prune_soft_single", None)
                    if torch.is_tensor(prune_single_term) and prune_single_term.requires_grad:
                        prune_where_grad_terms.append(
                            float(getattr(args, "compression_soft_prune_single_grad_weight", 20.0))
                            * prune_single_term
                        )

                    prune_rate_term = actuator_soft_terms.get("prune_soft_rate", None)
                    if torch.is_tensor(prune_rate_term) and prune_rate_term.requires_grad:
                        prune_where_grad_terms.append(
                            float(getattr(args, "compression_soft_rate_point_weight", 0.25))
                            * prune_rate_term
                        )

                    # ------------------------------------------------------------
                    # 形状を壊すPruneは抑える
                    # ------------------------------------------------------------
                    # prune_soft_geom は「削ると形状的に危ない場所」に対するペナルティである。
                    # bit系proxyと同時に入れることで、単純な全削除方向を避ける。
                    # ------------------------------------------------------------

                    prune_geom_term = actuator_soft_terms.get("prune_soft_geom", None)
                    if torch.is_tensor(prune_geom_term) and prune_geom_term.requires_grad:
                        prune_where_grad_terms.append(
                            float(getattr(args, "compression_soft_prune_geom_guard_weight", 1.0))
                            * prune_geom_term
                        )

                    # ------------------------------------------------------------
                    # bit/node/single/rate proxyが取れない場合の最小保険
                    # ------------------------------------------------------------
                    # target_drop_ratioへ寄せるMSEは使わない。
                    # fallbackでは、Prune Where proxyに小さい勾配だけを返す。
                    # 符号は「削除候補を少し増やす」向きにして、Prune Whereが完全0で止まるのを防ぐ。
                    # ------------------------------------------------------------

                    if True:
                        fallback_proxy = None
                        fallback_source = "none"

                        for key in (
                            "drop_prob_proxy",
                            "learned_drop_logit",
                            "drop_logit",
                            "prune_where_proxy",
                            "soft_drop_where_grad_base",
                            "soft_drop_prob_for_ste",
                        ):
                            value = actuator_soft_terms.get(key, None)
                            if torch.is_tensor(value) and value.requires_grad:
                                fallback_proxy = value
                                fallback_source = key
                                break

                        if fallback_proxy is not None:
                            fallback_anchor = torch.nan_to_num(
                                fallback_proxy.float().mean(),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )

                            prune_where_grad_terms.append(
                                -float(getattr(args, "compression_soft_prune_logit_direct_grad_weight", 0.01))
                                * fallback_anchor
                            )

                            if isinstance(cp_debug, dict):
                                cp_debug["prune_where_grad_fallback_source"] = fallback_source
                        else:
                            if isinstance(cp_debug, dict):
                                cp_debug["prune_where_grad_fallback_source"] = "no_requires_grad_proxy"

                    # ------------------------------------------------------------
                    # L_com_objectiveへgradient-onlyで足す
                    # ------------------------------------------------------------
                    # forward値は0であり、損失値そのものは変えない。
                    # backwardだけ Prune Where proxy へ流す。
                    # ------------------------------------------------------------

                    if prune_where_grad_terms:
                        prune_where_proxy_for_grad = prune_where_grad_terms[0]
                        for term in prune_where_grad_terms[1:]:
                            prune_where_proxy_for_grad = prune_where_proxy_for_grad + term

                        prune_where_proxy_for_grad = torch.nan_to_num(
                            prune_where_proxy_for_grad,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )

                        prune_where_proxy_grad_weight = float(
                            getattr(args, "compression_soft_prune_where_proxy_grad_weight", 0.10)
                        )

                        if prune_grad_rebalance:
                            prune_where_proxy_grad_weight *= float(prune_where_anchor_scale)
                        prune_where_proxy_grad_max = max(
                            float(getattr(args, "compression_soft_prune_where_proxy_grad_max", 1.0)),
                            0.0,
                        )
                        prune_where_proxy_grad_weight = min(
                            max(prune_where_proxy_grad_weight, 0.0),
                            prune_where_proxy_grad_max,
                        )

                        if prune_where_proxy_grad_weight > 0.0:
                            prune_where_proxy_grad_delta = prune_where_proxy_grad_weight * (
                                prune_where_proxy_for_grad - prune_where_proxy_for_grad.detach()
                            )

                            L_com_objective = L_com_objective + prune_where_proxy_grad_delta
                            L_com = L_com_objective

                            if compression_extra_grad_delta is None:
                                compression_extra_grad_delta = prune_where_proxy_grad_delta
                            else:
                                compression_extra_grad_delta = (
                                    compression_extra_grad_delta + prune_where_proxy_grad_delta
                                )

                            if isinstance(cp_debug, dict):
                                cp_debug["prune_where_grad_proxy_used"] = True
                                cp_debug["prune_where_grad_proxy_weight"] = prune_where_proxy_grad_weight
                                cp_debug["prune_where_grad_proxy_source"] = "prune_soft_terms_or_fallback"
                    else:
                        if isinstance(cp_debug, dict):
                            cp_debug["prune_where_grad_proxy_used"] = False
                            cp_debug["prune_where_grad_proxy_source"] = "no_prune_soft_terms_available"

                    L_downstream = L_com_objective
                    # ============================================================
                    # Prune勾配リバランス状態を exact occupancy STE 側へ渡す
                    # ============================================================
                    # 目的:
                    #   Prune Where anchorを止めても、
                    #   exact occupancy STE のsoft proxyからWhereへ大きな勾配が残る。
                    #   そのため、診断中はexact occupancyのsoft勾配も止める。
                    # ============================================================
                    setattr(args, "_prune_grad_rebalance_active", bool(prune_grad_rebalance))
                    setattr(args, "_prune_where_anchor_scale", float(prune_where_anchor_scale))
                    exact_occ_ste_term, exact_occ_debug = _build_exact_occupancy_ste_term(
                        args,
                        terms=terms,
                        model=model,
                        out_label=out_label,
                        before_xyz=voxel_collision_input_gt,
                        after_xyz=gen_xyz,
                    )

                    if torch.is_tensor(exact_occ_ste_term):
                        L_com_objective = L_com_objective + exact_occ_ste_term
                        L_com = L_com_objective

                        if compression_extra_grad_delta is None:
                            compression_extra_grad_delta = exact_occ_ste_term
                        else:
                            compression_extra_grad_delta = compression_extra_grad_delta + exact_occ_ste_term

                    if isinstance(cp_debug, dict):
                        cp_debug.update(exact_occ_debug)

                    # そのため、実際に backward される L にも同じ差分を足す。
                    # 差分のforward値は0なので、損失値そのものは変わらない。
                    if torch.is_tensor(compression_extra_grad_delta) and compression_extra_grad_delta.requires_grad:
                        L = L + compression_extra_grad_delta

                    prune_where_direct_weight = float(
                        getattr(args, "compression_soft_prune_logit_direct_grad_weight", 0.01)
                    )

                    if prune_grad_rebalance:
                        prune_where_direct_weight *= float(prune_where_anchor_scale)

                    if prune_where_direct_weight > 0.0:
                        base_model_for_prune_proxy = _unwrap_train_model(model)
                        actuator_soft_terms = dict(
                            getattr(base_model_for_prune_proxy, "last_actuator_soft_terms", {}) or {}
                        )

                        # 念のためargs側にも保存されている場合は拾う
                        args_soft_terms = getattr(args, "_last_actuator_soft_terms", None)
                        if isinstance(args_soft_terms, dict):
                            actuator_soft_terms.update(args_soft_terms)

                        prune_where_proxy = None
                        prune_where_proxy_source = "none"

                        for key in (
                            "drop_prob_proxy",
                            "learned_drop_logit",
                            "drop_logit",
                            "soft_drop_where_grad_direct",
                            "soft_drop_prob_for_ste",
                            "prune_where_proxy",
                            "soft_drop_where_grad_base",
                        ):
                            value = actuator_soft_terms.get(key, None)
                            if torch.is_tensor(value) and value.requires_grad:
                                prune_where_proxy = value
                                prune_where_proxy_source = key
                                break

                        if prune_where_proxy is not None:
                            prune_where_anchor = torch.nan_to_num(
                                prune_where_proxy.float().mean(),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )

                            # forward値は0、backwardだけPrune Whereへ返す
                            prune_where_grad_delta = prune_where_direct_weight * (
                                prune_where_anchor - prune_where_anchor.detach()
                            )

                            L_com_objective = L_com_objective + prune_where_grad_delta
                            L_com = L_com_objective
                            L_downstream = L_com_objective

                            # ============================================================
                            # 実際にbackwardされるLにもPrune Where direct anchorを足す
                            # ============================================================
                            # L_com_objective / L_com / L_downstream だけを書き換えても、
                            # build_compression_primary_loss が返した L には後付けproxyが入らない。
                            # そのため、drop_headへ返すgradient-only項をL_totalにも明示的に足す。
                            # forward値は0なので、損失値そのものは変わらない。
                            # ============================================================
                            L = L + prune_where_grad_delta

                            if isinstance(cp_debug, dict):
                                cp_debug["prune_where_direct_anchor_used"] = True
                                cp_debug["prune_where_direct_anchor_source"] = prune_where_proxy_source
                                cp_debug["prune_where_direct_anchor_weight"] = prune_where_direct_weight
                        else:
                            if isinstance(cp_debug, dict):
                                cp_debug["prune_where_direct_anchor_used"] = False
                                cp_debug["prune_where_direct_anchor_source"] = "no_requires_grad_proxy"
                    
                    # ============================================================
                    # Prune Amount 専用の gradient-only anchor
                    # ============================================================
                    # 目的:
                    #   圧縮損失だけの訓練で、Whereだけでなく
                    #   prune_amount_head に明確な勾配を返す。
                    #
                    # 方針:
                    #   forward値は0にする。
                    #   backwardだけ learned_drop_ratio / raw_learned_drop_ratio へ返す。
                    #   これにより、損失値そのものは変えずにAmount headを起こす。
                    # ============================================================
                    if prune_grad_rebalance:
                        base_model_for_amount_proxy = _unwrap_train_model(model)

                        actuator_soft_terms = dict(
                            getattr(base_model_for_amount_proxy, "last_actuator_soft_terms", {}) or {}
                        )

                        args_soft_terms = getattr(args, "_last_actuator_soft_terms", None)
                        if isinstance(args_soft_terms, dict):
                            actuator_soft_terms.update(args_soft_terms)

                        if isinstance(out_label, dict):
                            for key in (
                                "learned_drop_ratio",
                                "raw_learned_drop_ratio",
                                "voxel_soft_drop_amount",
                                "soft_drop_mass",
                            ):
                                value = out_label.get(key, None)
                                if torch.is_tensor(value):
                                    actuator_soft_terms[key] = value

                        amount_proxy = None
                        amount_proxy_source = "none"

                        for key in (
                            "learned_drop_ratio",
                            "raw_learned_drop_ratio",
                            "voxel_soft_drop_amount",
                            "soft_drop_mass",
                        ):
                            value = actuator_soft_terms.get(key, None)
                            if torch.is_tensor(value) and value.requires_grad:
                                amount_proxy = value
                                amount_proxy_source = key
                                break

                        if amount_proxy is not None:
                            amount_value = torch.nan_to_num(
                                amount_proxy.float().mean(),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )

                            max_drop_ratio = max(
                                float(getattr(args, "max_drop_ratio", 0.30)),
                                1e-6,
                            )

                            # Amountの目標は、まず5%程度に固定する。
                            # これは最終性能用ではなく、Amount headが勾配を受け取れるかを確認する診断用である。
                            target_drop_ratio = min(
                                max(
                                    float(getattr(args, "repair_drop_ratio_floor", 0.03)),
                                    float(getattr(args, "repair_init_drop_ratio", 0.05)),
                                    0.05,
                                ),
                                max_drop_ratio,
                            )

                            if amount_proxy_source == "raw_learned_drop_ratio":
                                logit_scale = max(
                                    float(getattr(args, "repair_operation_amount_logit_scale", 6.0)),
                                    1e-6,
                                )
                                amount_ratio = torch.sigmoid(amount_value / logit_scale) * float(max_drop_ratio)
                            elif amount_proxy_source == "soft_drop_mass":
                                # soft_drop_mass は個数スケールの可能性があるため、
                                # ここでは診断用としてそのまま使わず、learned_drop_ratioが無い場合の最後の保険に留める。
                                amount_ratio = amount_value.clamp(0.0, float(max_drop_ratio))
                            else:
                                amount_ratio = amount_value.clamp(0.0, float(max_drop_ratio))

                            target_tensor = amount_ratio.new_tensor(float(target_drop_ratio))

                            amount_anchor_loss = torch.nn.functional.smooth_l1_loss(
                                amount_ratio,
                                target_tensor,
                                reduction="mean",
                            )

                            # ============================================================
                            # Prune Amount soft anchor
                            # ============================================================
                            # これは診断用である。
                            # 通常訓練ではAmountを人工的にtargetへ寄せず、
                            # actual / surrogate / hybrid priorから学習させる。
                            # ============================================================
                            amount_anchor_weight = (
                                max(float(getattr(args, "prune_amount_soft_anchor_weight", 0.0)), 0.0)
                                if bool(getattr(args, "prune_amount_soft_anchor_enable", False))
                                else 0.0
                            )

                            prune_amount_grad_delta = amount_anchor_weight * (
                                amount_anchor_loss - amount_anchor_loss.detach()
                            )

                            L_com_objective = L_com_objective + prune_amount_grad_delta
                            L_com = L_com_objective
                            L_downstream = L_com_objective

                            # 実際にbackwardされるLにも足す。
                            # forward値は0なので、損失値は変わらない。
                            L = L + prune_amount_grad_delta

                            if isinstance(cp_debug, dict):
                                cp_debug["prune_amount_anchor_used"] = True
                                cp_debug["prune_amount_anchor_source"] = amount_proxy_source
                                cp_debug["prune_amount_anchor_weight"] = float(amount_anchor_weight)
                                cp_debug["prune_amount_anchor_target_ratio"] = float(target_drop_ratio)
                                cp_debug["prune_amount_anchor_value"] = float(amount_ratio.detach().cpu())
                                cp_debug["prune_amount_anchor_loss"] = float(amount_anchor_loss.detach().cpu())
                        else:
                            if isinstance(cp_debug, dict):
                                cp_debug["prune_amount_anchor_used"] = False
                                cp_debug["prune_amount_anchor_source"] = "no_requires_grad_amount_proxy"
                        # ============================================================
                        # 保険: Prune Amount bias への直接gradient-only anchor
                        # ============================================================
                        # 目的:
                        #   learned_drop_ratio / raw_learned_drop_ratio が
                        #   drop_amount_head に接続されていない場合でも、
                        #   drop_amount_head.bias へ直接勾配を入れる。
                        #
                        # 方針:
                        #   loss = -bias.mean()
                        #   optimizerはlossを下げるため、biasは増える方向に更新される。
                        #   つまりPrune Amountが増える方向へ動く。
                        #
                        # 注意:
                        #   これは診断用である。
                        #   Amount headが動くことを確認した後は、重みを下げるか、
                        #   proxy接続の修正に置き換える。
                        # ============================================================
                        actuator_for_amount_bias = getattr(base_model_for_amount_proxy, "actuator", None)
                        drop_amount_head = getattr(actuator_for_amount_bias, "drop_amount_head", None)
                        drop_amount_bias = getattr(drop_amount_head, "bias", None)

                        if (
                            bool(getattr(args, "prune_amount_bias_anchor_enable", False))
                            and torch.is_tensor(drop_amount_bias)
                            and drop_amount_bias.requires_grad
                        ):
                            amount_bias_anchor_weight = max(
                                float(getattr(args, "grad_scale_operation_amount", 1.0)),
                                0.0,
                            )

                            amount_bias_anchor = -torch.nan_to_num(
                                drop_amount_bias.float().mean(),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )

                            prune_amount_bias_delta = amount_bias_anchor_weight * (
                                amount_bias_anchor - amount_bias_anchor.detach()
                            )

                            L_com_objective = L_com_objective + prune_amount_bias_delta
                            L_com = L_com_objective
                            L_downstream = L_com_objective

                            # 実際にbackwardされるLにも足す。
                            # forward値は0なので、損失値は変わらない。
                            L = L + prune_amount_bias_delta

                            if isinstance(cp_debug, dict):
                                cp_debug["prune_amount_bias_anchor_used"] = True
                                cp_debug["prune_amount_bias_anchor_weight"] = float(amount_bias_anchor_weight)
                                cp_debug["prune_amount_bias_anchor_value"] = float(drop_amount_bias.detach().float().mean().cpu())
                        else:
                            if isinstance(cp_debug, dict):
                                cp_debug["prune_amount_bias_anchor_used"] = False
                    tail_attr_block = stage_factors["attr"] * args.w_attr * L_attr
                    tail_policy_block = stage_factors["policy"] * args.w_policy * L_policy
                    tail_actuator_block = stage_factors["repair"] * args.w_actuator * L_actuator
                    tail_support_raw = tail_attr_block + tail_policy_block + tail_actuator_block
                    tail_balance = _compression_primary_support_balance(
                        args,
                        compression_support_anchor if torch.is_tensor(compression_support_anchor) else L,
                        tail_support_raw,
                        enabled=uses_actual_total_bit_objective(args),
                        target_ratio_name="compression_primary_tail_target_ratio",
                        min_scale_name="compression_primary_tail_balance_min_scale",
                        max_scale_name="compression_primary_tail_balance_max_scale",
                        disabled_reason="tail_balance_disabled",
                    )
                    tail_support_scale = float(tail_balance["scale"])
                    tail_support_scaled = tail_support_scale * tail_support_raw
                    L = L + tail_support_scaled

                    if isinstance(cp_debug, dict):
                        cp_debug["cp_support_tail_attr_raw"] = case_float(tail_attr_block, float("nan"))
                        cp_debug["cp_support_tail_policy_raw"] = case_float(tail_policy_block, float("nan"))
                        cp_debug["cp_support_tail_actuator_raw"] = case_float(tail_actuator_block, float("nan"))
                        cp_debug["cp_support_tail_attr_scaled"] = case_float(
                            tail_support_scale * tail_attr_block,
                            float("nan"),
                        )
                        cp_debug["cp_support_tail_policy_scaled"] = case_float(
                            tail_support_scale * tail_policy_block,
                            float("nan"),
                        )
                        cp_debug["cp_support_tail_actuator_scaled"] = case_float(
                            tail_support_scale * tail_actuator_block,
                            float("nan"),
                        )
                        cp_debug["cp_support_tail_raw"] = case_float(tail_support_raw, float("nan"))
                        cp_debug["cp_support_tail_scaled"] = case_float(tail_support_scaled, float("nan"))
                        cp_debug["cp_support_tail_scale"] = float(tail_support_scale)
                        cp_debug["cp_support_tail_reason"] = str(tail_balance.get("reason", ""))
                        cp_debug["cp_support_tail_target_ratio"] = (
                            float(tail_balance["target_ratio"])
                            if tail_balance.get("target_ratio", None) is not None
                            else float("nan")
                        )
                        cp_debug["cp_support_tail_primary_abs"] = (
                            float(tail_balance["primary_mag"])
                            if tail_balance.get("primary_mag", None) is not None
                            else float("nan")
                        )
                        cp_debug["cp_support_tail_support_abs"] = (
                            float(tail_balance["support_mag"])
                            if tail_balance.get("support_mag", None) is not None
                            else float("nan")
                        )
                        cp_debug["cp_support_tail_scaled_support_abs"] = (
                            float(tail_balance["scaled_support_mag"])
                            if tail_balance.get("scaled_support_mag", None) is not None
                            else float("nan")
                        )
                        cp_debug["cp_support_tail_dominant"] = str(
                            tail_balance.get("dominant", "neutral")
                        )
                        aux_scaled = float(cp_debug.get("cp_aux_block_scaled", 0.0))
                        support_total_scaled = aux_scaled + case_float(tail_support_scaled, 0.0)
                        main_block_value = float(cp_debug.get("cp_main_block", 0.0))
                        cp_debug["cp_support_total_scaled"] = float(support_total_scaled)
                        cp_debug["cp_support_total_ratio_to_main"] = (
                            abs(support_total_scaled) / max(abs(main_block_value), 1e-12)
                        )
                        cp_debug["cp_support_dominant"] = (
                            "compression"
                            if abs(main_block_value) + 1e-12 >= abs(support_total_scaled)
                            else "support"
                        )
                    L_discrete_policy = L.new_zeros(())
                elif (
                    compression_primary_mode
                    and heuristic_mode == "single_plan_student"
                    and single_plan_cache_only_stage
                ):
                    # Cache-only StageではActual/Surrogate forward値を要求しない。
                    # 後段で加えるTeacher蒸留lossだけがStudentを更新する。
                    L = L_geom.new_zeros(())
                    L_downstream = L
                    L_discrete_policy = L
                    L_attr = L_attr.detach()
                    L_policy = L_policy.detach()
                    L_actuator = L_actuator.detach()
                    cp_debug = {
                        "single_plan_cache_only_stage": True,
                        "fresh_actual_encode_count": 0,
                        "fresh_geometry_count": 0,
                    }
                elif compression_primary_mode and network_only_full_cloud:
                    # The Network-only objective is deliberately compact:
                    # Actual-forward/Surrogate-backward compression + geometry.
                    # Old Prune-only proxy anchors, attribution/policy teachers,
                    # and actuator imitation losses would reintroduce a legacy
                    # heuristic preference and retain their full-cloud graphs.
                    if not (torch.is_tensor(L_com_objective) and L_com_objective.requires_grad):
                        raise RuntimeError(
                            "network-only compression objective lost its Surrogate gradient"
                        )
                    geometry_weight = max(float(getattr(args, "cp_lambda_geom", 1.0)), 0.0)
                    compression_weight = max(
                        float(
                            getattr(
                                args,
                                "network_only_actual_surrogate_loss_weight",
                                1.0,
                            )
                        ),
                        0.0,
                    )
                    if k_all_actual_enabled:
                        # 全Kの実測絶対rewardを主信号にし、選択1案だけのSurrogateが
                        # 8専門slotを同じ方向へ引く影響は小さく残す。
                        compression_weight *= max(float(getattr(
                            args,
                            "network_k_all_actual_selected_surrogate_weight",
                            0.1,
                        )), 0.0)
                        if isinstance(k_all_actual_result, dict):
                            selected_index = int(k_all_actual_result.get("selected_slot", 0))
                            selected_actual_rows = k_all_actual_result.get(
                                "actual_compression_percent"
                            )
                            if torch.is_tensor(selected_actual_rows):
                                selected_actual_percent = float(
                                    selected_actual_rows.reshape(-1)[selected_index].detach().cpu()
                                )
                                # Surrogateの符号が未校正でも、Actualで悪化したplanを
                                # 微分可能proxyが正例として押し戻さないようにする。
                                if selected_actual_percent >= 0.0:
                                    compression_weight = 0.0
                    L = (
                        geometry_weight * L_geom
                        + stage_factors["com"] * compression_weight * L_com_objective
                    )
                    L_downstream = L_com_objective
                    L_discrete_policy = L.new_zeros(())
                    L_attr = L_attr.detach()
                    L_policy = L_policy.detach()
                    L_actuator = L_actuator.detach()
                    cp_debug = {
                        "network_only_objective": True,
                        "network_only_actual_surrogate_weight": float(compression_weight),
                        "network_only_geometry_weight": float(geometry_weight),
                        "network_only_legacy_proxy_loss": 0.0,
                        "network_only_behavior_cloning_loss": 0.0,
                    }
                elif _discrete_loss_mode_value(args) == "hard":
                    policy_loss_fn = getattr(model, "discrete_policy_loss", None) # モデルが保持しているHard離散方策用の損失関数を取得する
                    if callable(policy_loss_fn):
                        L_discrete_policy = policy_loss_fn(L_downstream.detach())
                        L = L + L_discrete_policy

                # ana_den6 onlineではcompression_primaryでもPolicy Gradientを必ず加える。
                # actual codecの結果は微分不能なので、Where/Amount/Actionのsample log-probへ
                # advantageを掛けて、1Stepで試した1planの成否を次Stepへ学習させる。
                heuristic_mode = str(
                    getattr(args, "heuristic_guidance_mode", "")
                ).strip().lower()
                if (
                    heuristic_mode in {
                        "ana_den6_online",
                        "network_only_codec_policy",
                        "network_k_proposal_policy",
                        "single_plan_student",
                    }
                    and not k_all_actual_enabled
                    and not single_plan_cache_only_stage
                ):
                    base_model_for_policy = model.module if hasattr(model, "module") else model
                    policy_loss_fn = getattr(base_model_for_policy, "discrete_policy_loss", None)
                    if not callable(policy_loss_fn):
                        raise RuntimeError(
                            f"{heuristic_mode}にはNetwork.discrete_policy_lossが必要である"
                        )
                    # 方策の成否は幾何等を含むL_downstreamではなく、このStepで
                    # 唯一実行したfull-cloud Actual圧縮率だけで判定する。
                    # Surrogateは従来どおり微分可能な主損失として別経路で逆伝播する。
                    actual_policy_value = finite_float_or_none(
                        compression_debug_terms.get(
                            "actual_total_bit_percent_fresh",
                            compression_debug_terms.get("actual_total_bit_percent", None),
                        )
                    )
                    if actual_policy_value is None:
                        raise RuntimeError(
                            f"{heuristic_mode}の毎Step policy更新にfull-cloud Actual圧縮率がない"
                        )
                    online_policy_objective = L_downstream.new_tensor(
                        float(actual_policy_value)
                    )
                    online_policy_loss = policy_loss_fn(
                        online_policy_objective,
                        geometry=L_geom.detach(),
                    )
                    if not torch.is_tensor(online_policy_loss):
                        raise RuntimeError(
                            f"{heuristic_mode}のdiscrete_policy_lossがTensorを返していない"
                        )
                    if heuristic_mode == "single_plan_student":
                        # 既定0。Actual policy gradientは蒸留Gate通過後の限定ablationだけで使う。
                        online_policy_loss = online_policy_loss * max(float(getattr(
                            args, "single_plan_policy_gradient_weight", 0.0
                        )), 0.0)
                    elif heuristic_mode == "ana_den6_online":
                        # 現在のforward係数0.1はLoss図をPolicyで支配しないため維持する。
                        # backwardだけ063943時の実効係数1.0相当へ戻し、Actual/Geometryの
                        # 相対評価をWhere/Amount/Actionへ十分に伝える。
                        policy_backward_scale = max(float(getattr(
                            args,
                            "heuristic_guidance_online_policy_backward_scale",
                            10.0,
                        )), 0.0)
                        online_policy_loss = (
                            online_policy_loss.detach()
                            + policy_backward_scale
                            * (online_policy_loss - online_policy_loss.detach())
                        )
                        compression_debug_terms[
                            "den6_online_policy_backward_scale"
                        ] = float(policy_backward_scale)
                        latest_policy_debug = dict(
                            getattr(loss, "last_compression_debug", {}) or {}
                        )
                        latest_policy_debug[
                            "den6_online_policy_backward_scale"
                        ] = float(policy_backward_scale)
                        loss.last_compression_debug = latest_policy_debug
                    policy_debug = dict(
                        getattr(base_model_for_policy, "last_discrete_policy_debug", {}) or {}
                    )
                    if not policy_debug:
                        raise RuntimeError(
                            f"{heuristic_mode}のsingle-proposal policy項が生成されていない"
                        )
                    compression_debug_terms.update({
                        f"den6_online_policy_{key}": value
                        for key, value in policy_debug.items()
                    })
                    latest_compression_debug = dict(
                        getattr(loss, "last_compression_debug", {}) or {}
                    )
                    latest_compression_debug.update({
                        f"den6_online_policy_{key}": value
                        for key, value in policy_debug.items()
                    })
                    loss.last_compression_debug = latest_compression_debug
                    # hard分岐で既に足している場合の二重加算を避ける。
                    if _discrete_loss_mode_value(args) == "hard" and not compression_primary_mode:
                        L = L - L_discrete_policy
                    L_discrete_policy = online_policy_loss
                    L = L + L_discrete_policy
                    if heuristic_mode in {"network_only_codec_policy", "network_k_proposal_policy", "single_plan_student"}:
                        plan_gain_loss_fn = getattr(
                            base_model_for_policy, "network_only_plan_gain_loss", None
                        )
                        if not callable(plan_gain_loss_fn):
                            raise RuntimeError("network-only plan gain predictor loss is missing")
                        L_plan_gain = plan_gain_loss_fn(online_policy_objective)
                        if not torch.is_tensor(L_plan_gain) or not L_plan_gain.requires_grad:
                            raise RuntimeError("network-only plan gain loss has no gradient")
                        L = L + L_plan_gain
                        compression_debug_terms["network_only_plan_gain_loss"] = float(
                            L_plan_gain.detach().cpu()
                        )
                        plan_gain_debug = dict(
                            getattr(base_model_for_policy, "last_discrete_policy_debug", {}) or {}
                        )
                        compression_debug_terms.update({
                            f"den6_online_policy_{key}": value
                            for key, value in plan_gain_debug.items()
                        })
                        latest_compression_debug = dict(
                            getattr(loss, "last_compression_debug", {}) or {}
                        )
                        latest_compression_debug.update({
                            f"den6_online_policy_{key}": value
                            for key, value in plan_gain_debug.items()
                        })
                        loss.last_compression_debug = latest_compression_debug

                    if (
                        heuristic_mode == "network_k_proposal_policy"
                        and isinstance(k_proposal_teacher_store, OfflineKProposalTeacherStore)
                        and (
                            not k_all_actual_enabled
                            or global_train_step < int(getattr(
                                args, "network_k_offline_bootstrap_steps", 0
                            ))
                        )
                    ):
                        offline_state_id = k_proposal_teacher_store.find_state_for_input(
                            file_path,
                            args,
                            split=str(getattr(args, "network_k_offline_split", "train")),
                        )
                        if offline_state_id is not None:
                            proposal_output = getattr(
                                base_model_for_policy, "last_k_proposal_terms", None
                            )
                            actuator_state = getattr(
                                base_model_for_policy, "last_actuator_voxel_state", None
                            )
                            initial_voxel_coords = (
                                actuator_state.get("initial_voxel_coords")
                                if isinstance(actuator_state, dict) else None
                            )
                            if not isinstance(proposal_output, dict) or not torch.is_tensor(initial_voxel_coords):
                                raise RuntimeError(
                                    "offline K proposal loss requires current proposal and canonical voxels"
                                )
                            offline_teacher = k_proposal_teacher_store.teacher_for_output(
                                offline_state_id,
                                proposal_output,
                                initial_voxel_coords,
                                split=str(getattr(args, "network_k_offline_split", "train")),
                            )
                            offline_loss_fn = getattr(
                                base_model_for_policy,
                                "k_proposal_offline_distillation_loss",
                                None,
                            )
                            if not callable(offline_loss_fn):
                                raise RuntimeError("K proposal offline set loss is missing")
                            offline_losses = offline_loss_fn(offline_teacher)
                            offline_weight = max(
                                float(getattr(args, "network_k_offline_loss_weight", 1.0)),
                                0.0,
                            )
                            weighted_offline_total = offline_losses["total"] * offline_weight
                            if not weighted_offline_total.requires_grad:
                                raise RuntimeError("K proposal offline set loss has no gradient")
                            L = L + weighted_offline_total
                            compression_debug_terms.update({
                                "k_proposal_offline_state_id": offline_state_id,
                                "k_proposal_offline_loss": float(weighted_offline_total.detach().cpu()),
                                "k_proposal_offline_loss_weight": offline_weight,
                                "k_proposal_offline_dominance_ratio": float(offline_losses["dominance_ratio"]),
                                "k_proposal_offline_dominance_warning": bool(offline_losses["dominance_warning"]),
                                "k_proposal_offline_add_where_teacher_available": bool(
                                    offline_teacher.get("add_where_teacher_available", False)
                                ),
                                "k_proposal_shortlist_natural_recall": float(
                                    offline_teacher["shortlist_natural_recall"][
                                        offline_teacher["shortlist_natural_recall_mask"]
                                    ].mean().detach().cpu()
                                ) if bool(offline_teacher[
                                    "shortlist_natural_recall_mask"
                                ].any()) else float("nan"),
                                "k_proposal_shortlist_training_recall": float(
                                    offline_teacher["shortlist_training_recall"][
                                        offline_teacher["shortlist_training_recall_mask"]
                                    ].mean().detach().cpu()
                                ) if bool(offline_teacher[
                                    "shortlist_training_recall_mask"
                                ].any()) else float("nan"),
                                "k_proposal_target_reachable_recall": float(
                                    offline_teacher["target_reachable_recall"][
                                        offline_teacher["target_reachable_recall_mask"]
                                    ].mean().detach().cpu()
                                ) if bool(offline_teacher[
                                    "target_reachable_recall_mask"
                                ].any()) else float("nan"),
                            })
                            for metric_name, metric_value in offline_losses.get("metrics", {}).items():
                                if torch.is_tensor(metric_value):
                                    metric_value = float(metric_value.detach().cpu())
                                compression_debug_terms[
                                    f"k_proposal_offline_metric_{metric_name}"
                                ] = metric_value
                            for loss_name, raw_value in offline_losses["raw"].items():
                                compression_debug_terms[
                                    f"k_proposal_offline_{loss_name}_raw"
                                ] = float(raw_value.detach().cpu())
                            for loss_name, weighted_value in offline_losses["weighted"].items():
                                compression_debug_terms[
                                    f"k_proposal_offline_{loss_name}_weighted"
                                ] = float(weighted_value.detach().cpu()) * offline_weight
                            latest_compression_debug = dict(
                                getattr(loss, "last_compression_debug", {}) or {}
                            )
                            latest_compression_debug.update(compression_debug_terms)
                            loss.last_compression_debug = latest_compression_debug

                if (
                    heuristic_mode == "single_plan_student"
                    and isinstance(single_plan_teacher_store, SinglePlanTeacherStore)
                ):
                    setting_id = (
                        "native_vs{}_pq{}_ae{}_sr{}_m{}".format(
                            float(getattr(args, "sparsepcgc_voxel_size", 1.0)),
                            int(getattr(args, "sparsepcgc_pos_quantscale", 1)),
                            int(getattr(args, "sparsepcgc_scale_ae", 0)),
                            int(getattr(args, "sparsepcgc_scale_sr", 2)),
                            int(getattr(args, "sparsepcgc_scale_m", 8)),
                        ).replace("vs1.0", "vs1")
                    )
                    teacher_state_id = single_plan_teacher_store.find(file_path, setting_id)
                    if teacher_state_id is not None:
                        teacher_record = single_plan_teacher_store.supervision_record(
                            teacher_state_id, global_train_step
                        )
                        base_student = model.module if hasattr(model, "module") else model
                        distill_fn = getattr(
                            base_student, "single_plan_teacher_distillation_loss", None
                        )
                        if not callable(distill_fn):
                            raise RuntimeError("Single-Plan蒸留lossがない")
                        single_distill = distill_fn(teacher_record)
                        if not single_distill.requires_grad:
                            raise RuntimeError("Single-Plan蒸留lossの勾配が切れている")
                        L = L + single_distill
                        compression_debug_terms.update({
                            "single_plan_teacher_state_id": teacher_state_id,
                            "single_plan_teacher_plan_key": str(teacher_record["plan_key"]),
                            "single_plan_teacher_actual_gain": float(
                                teacher_record["actual_gain_percent"]
                            ),
                            "single_plan_distillation_loss": float(single_distill.detach().cpu()),
                        })
                        compression_debug_terms.update({
                            "single_plan_distill_{}".format(key): value
                            for key, value in dict(getattr(
                                base_student, "last_single_plan_distillation_debug", {}
                            )).items()
                        })
                        latest_compression_debug = dict(
                            getattr(loss, "last_compression_debug", {}) or {}
                        )
                        latest_compression_debug.update(compression_debug_terms)
                        loss.last_compression_debug = latest_compression_debug

                if (
                    heuristic_mode == "ana_den6_online"
                    and bool(getattr(args, "single_plan_shadow_distillation", True))
                ):
                    # このStepで実行したExact+Network residualの1 planだけを、
                    # 同じ入力を見たSingle-Plan Studentへ蒸留する。未実行Poolや
                    # cache planをStudent forwardへ注入せず、Actual回数も増やさない。
                    base_student = model.module if hasattr(model, "module") else model
                    shadow_state = getattr(
                        base_student, "last_actuator_voxel_state", None
                    )
                    shadow_debug = (
                        shadow_state.get("ana_den6_exact_residual_plan_debug", {})
                        if isinstance(shadow_state, dict) else {}
                    )
                    shadow_teacher = (
                        dict(shadow_debug.get("single_plan_shadow_teacher") or {})
                        if isinstance(shadow_debug, dict) else {}
                    )
                    if not shadow_teacher:
                        raise RuntimeError(
                            "ana_den6 online実行planからSingle-Plan shadow教師を作れない"
                        )
                    shadow_actual = finite_float_or_none(
                        compression_debug_terms.get(
                            "actual_total_bit_percent_fresh",
                            compression_debug_terms.get(
                                "actual_total_bit_percent", None
                            ),
                        )
                    )
                    if shadow_actual is None:
                        raise RuntimeError(
                            "Single-Plan shadow蒸留に実行planのActual値がない"
                        )
                    shadow_teacher["actual_gain_percent"] = -float(shadow_actual)
                    shadow_geometry = case_float(L_geom, float("nan"))
                    if not math.isfinite(shadow_geometry):
                        raise RuntimeError(
                            "Single-Plan shadow蒸留にGeometry値がない"
                        )
                    shadow_teacher["geometry"] = {
                        "D1_loss_db": float(shadow_geometry),
                        "D2_loss_db": float(shadow_geometry),
                    }
                    distill_fn = getattr(
                        base_student, "single_plan_teacher_distillation_loss", None
                    )
                    if not callable(distill_fn):
                        raise RuntimeError("Single-Plan shadow蒸留lossがない")
                    shadow_distill_raw = distill_fn(shadow_teacher)
                    if not shadow_distill_raw.requires_grad:
                        raise RuntimeError("Single-Plan shadow蒸留lossの勾配が切れている")
                    # Student蒸留は維持する。ただし生損失をそのまま全モデル共通の
                    # gradient clipへ入れず、圧縮主目的に対する比率で正規化する。
                    shadow_balance = _compression_primary_support_balance(
                        args,
                        (
                            compression_support_anchor
                            if torch.is_tensor(compression_support_anchor)
                            else L_com_objective
                        ),
                        shadow_distill_raw,
                        enabled=True,
                        target_ratio_name="single_plan_shadow_target_ratio",
                        min_scale_name="single_plan_shadow_balance_min_scale",
                        max_scale_name="single_plan_shadow_balance_max_scale",
                        disabled_reason="single_plan_shadow_balance_disabled",
                    )
                    proposed_shadow_scale = float(shadow_balance["scale"])
                    shadow_scale_state = getattr(
                        base_student, "single_plan_shadow_balance_scale", None
                    )
                    previous_shadow_scale = float("nan")
                    if torch.is_tensor(shadow_scale_state):
                        previous_shadow_scale = float(
                            shadow_scale_state.detach().float().cpu()
                        )
                    shadow_distill_scale = monotonic_support_scale(
                        previous_shadow_scale,
                        proposed_shadow_scale,
                    )
                    if torch.is_tensor(shadow_scale_state):
                        with torch.no_grad():
                            shadow_scale_state.fill_(shadow_distill_scale)
                    if not math.isfinite(previous_shadow_scale):
                        shadow_scale_reason = "initial_calibration"
                    elif shadow_distill_scale < previous_shadow_scale:
                        shadow_scale_reason = "budget_tightened"
                    else:
                        shadow_scale_reason = "convergence_preserved"
                    shadow_distill = shadow_distill_scale * shadow_distill_raw
                    L = L + shadow_distill
                    compression_debug_terms.update({
                        "single_plan_shadow_distillation": True,
                        "single_plan_shadow_plan_key": str(
                            shadow_teacher["plan_key"]
                        ),
                        "single_plan_shadow_actual_gain": float(
                            shadow_teacher["actual_gain_percent"]
                        ),
                        "single_plan_shadow_loss": float(
                            shadow_distill.detach().cpu()
                        ),
                        "single_plan_shadow_loss_raw": float(
                            shadow_distill_raw.detach().cpu()
                        ),
                        "single_plan_shadow_loss_scale": float(
                            shadow_distill_scale
                        ),
                        "single_plan_shadow_loss_scale_proposed": float(
                            proposed_shadow_scale
                        ),
                        "single_plan_shadow_target_ratio": float(
                            shadow_balance.get("target_ratio", 0.0) or 0.0
                        ),
                        "single_plan_shadow_balance_reason": str(
                            shadow_scale_reason
                        ),
                        "single_plan_shadow_update_count": int(
                            base_student.single_plan_distillation_updates.detach().cpu()
                        ),
                    })
                    compression_debug_terms.update({
                        "single_plan_shadow_{}".format(key): value
                        for key, value in dict(getattr(
                            base_student,
                            "last_single_plan_distillation_debug",
                            {},
                        )).items()
                    })
                    latest_compression_debug = dict(
                        getattr(loss, "last_compression_debug", {}) or {}
                    )
                    latest_compression_debug.update(compression_debug_terms)
                    loss.last_compression_debug = latest_compression_debug

                if k_all_actual_enabled:
                    if heuristic_mode != "network_k_proposal_policy":
                        raise RuntimeError("K all-Actual lossはK proposal mode専用である")
                    if not isinstance(k_all_actual_result, dict):
                        raise RuntimeError("K all-Actual評価結果が学習損失へ届いていない")
                    base_model_for_policy = model.module if hasattr(model, "module") else model
                    # 163件の保存Actualは初期化期間だけ使用する。teacher座標を
                    # shortlistへ注入せず、現在Networkが自然に出した候補へ教師化する。
                    bootstrap_state_id = None
                    bootstrap_active = False
                    if isinstance(k_proposal_teacher_store, OfflineKProposalTeacherStore):
                        bootstrap_state_id = k_proposal_teacher_store.find_state_for_input(
                            file_path,
                            args,
                            split=str(getattr(args, "network_k_offline_split", "train")),
                        )
                        bootstrap_counts = getattr(
                            args, "_network_k_offline_bootstrap_state_steps", None
                        )
                        if not isinstance(bootstrap_counts, dict):
                            bootstrap_counts = {}
                            args._network_k_offline_bootstrap_state_steps = bootstrap_counts
                        bootstrap_encounters = getattr(
                            args, "_network_k_offline_bootstrap_state_encounters", None
                        )
                        if not isinstance(bootstrap_encounters, dict):
                            bootstrap_encounters = {}
                            args._network_k_offline_bootstrap_state_encounters = bootstrap_encounters
                        encounter_index = int(bootstrap_encounters.get(
                            bootstrap_state_id, 0
                        )) if bootstrap_state_id is not None else 0
                        bootstrap_cadence = max(int(getattr(
                            args, "network_k_offline_bootstrap_cadence", 5
                        )), 1)
                        bootstrap_active = bool(
                            bootstrap_state_id is not None
                            and int(bootstrap_counts.get(bootstrap_state_id, 0))
                            < int(getattr(args, "network_k_offline_bootstrap_steps", 0))
                            and encounter_index % bootstrap_cadence == 0
                        )
                        if bootstrap_state_id is not None:
                            bootstrap_encounters[bootstrap_state_id] = encounter_index + 1
                        if bootstrap_active:
                            bootstrap_t0 = time.time()
                            proposal_output = getattr(
                                base_model_for_policy, "last_k_proposal_terms", None
                            )
                            actuator_state = getattr(
                                base_model_for_policy, "last_actuator_voxel_state", None
                            )
                            initial_voxel_coords = (
                                actuator_state.get("initial_voxel_coords")
                                if isinstance(actuator_state, dict) else None
                            )
                            if not isinstance(proposal_output, dict) or not torch.is_tensor(initial_voxel_coords):
                                raise RuntimeError("K Actual bootstrapにproposal/canonical voxelがない")
                            bootstrap_teacher = k_proposal_teacher_store.teacher_for_output(
                                bootstrap_state_id,
                                proposal_output,
                                initial_voxel_coords,
                                split=str(getattr(args, "network_k_offline_split", "train")),
                            )
                            bootstrap_losses = base_model_for_policy.k_proposal_offline_distillation_loss(
                                bootstrap_teacher
                            )
                            bootstrap_weight = max(float(getattr(
                                args, "network_k_offline_bootstrap_loss_weight", 1.0
                            )), 0.0)
                            bootstrap_loss = bootstrap_losses["total"] * bootstrap_weight
                            if not bootstrap_loss.requires_grad:
                                raise RuntimeError("163候補bootstrap lossの勾配が切れている")
                            L = L + bootstrap_loss
                            compression_debug_terms.update({
                                "k_all_actual_offline_bootstrap_active": True,
                                "k_all_actual_offline_bootstrap_state_id": bootstrap_state_id,
                                "k_all_actual_offline_bootstrap_loss": float(
                                    bootstrap_loss.detach().cpu()
                                ),
                                "k_all_actual_offline_bootstrap_state_step": int(
                                    bootstrap_counts.get(bootstrap_state_id, 0)
                                ),
                                "k_all_actual_offline_bootstrap_encounter": encounter_index,
                                "k_all_actual_offline_bootstrap_cadence": bootstrap_cadence,
                                "k_all_actual_offline_bootstrap_dense_target_active": True,
                                "k_all_actual_offline_bootstrap_time": float(
                                    time.time() - bootstrap_t0
                                ),
                                "k_all_actual_shortlist_natural_recall": float(
                                    bootstrap_teacher["shortlist_natural_recall"][
                                        bootstrap_teacher["shortlist_natural_recall_mask"]
                                    ].mean().detach().cpu()
                                ) if bool(bootstrap_teacher[
                                    "shortlist_natural_recall_mask"
                                ].any()) else float("nan"),
                            })
                            for metric_name, metric_value in bootstrap_losses.get(
                                "metrics", {}
                            ).items():
                                if torch.is_tensor(metric_value):
                                    metric_value = float(metric_value.detach().cpu())
                                compression_debug_terms[
                                    f"k_all_actual_bootstrap_{metric_name}"
                                ] = metric_value
                            bootstrap_counts[bootstrap_state_id] = int(
                                bootstrap_counts.get(bootstrap_state_id, 0)
                            ) + 1
                        elif (
                            bootstrap_state_id is not None
                            and int(bootstrap_counts.get(bootstrap_state_id, 0))
                            < int(getattr(args, "network_k_offline_bootstrap_steps", 0))
                        ):
                            compression_debug_terms.update({
                                "k_all_actual_offline_bootstrap_active": False,
                                "k_all_actual_offline_bootstrap_deferred": True,
                                "k_all_actual_offline_bootstrap_encounter": encounter_index,
                                "k_all_actual_offline_bootstrap_cadence": bootstrap_cadence,
                            })
                        elif bootstrap_state_id is None:
                            compression_debug_terms.update({
                                "k_all_actual_offline_bootstrap_active": False,
                                "k_all_actual_offline_bootstrap_miss": True,
                            })
                    all_actual_loss_fn = getattr(
                        base_model_for_policy, "k_proposal_all_actual_loss", None
                    )
                    if not callable(all_actual_loss_fn):
                        raise RuntimeError("K all-Actual policy lossがNetworkに存在しない")
                    L_k_all_actual = all_actual_loss_fn(
                        k_all_actual_result["actual_compression_percent"],
                        state_key=cache_key,
                    )
                    if not torch.is_tensor(L_k_all_actual) or not L_k_all_actual.requires_grad:
                        raise RuntimeError("K all-Actual policy lossの勾配が切れている")
                    L = L + L_k_all_actual
                    L_discrete_policy = L_k_all_actual
                    k_actual_debug = dict(
                        getattr(base_model_for_policy, "last_k_all_actual_debug", {}) or {}
                    )
                    compression_debug_terms.update({
                        f"k_all_actual_{key}": value
                        for key, value in k_actual_debug.items()
                    })
                    compression_debug_terms.update({
                        "k_all_actual_proposal_count": int(
                            k_all_actual_result["proposal_actual_encode_count"]
                        ),
                        "k_all_actual_proposal_aux_stats_count": int(
                            k_all_actual_result.get("proposal_aux_stats_count", 0)
                        ),
                        "k_all_actual_baseline_bits": float(
                            k_all_actual_result["baseline_bits"]
                        ),
                        "k_all_actual_baseline_scalar_cache_hit": bool(
                            k_all_actual_result["baseline_scalar_cache_hit"]
                        ),
                        "den6_online_baseline_actual_encode_count": int(
                            k_all_actual_result["baseline_actual_encode_count"]
                        ),
                        "den6_online_edited_actual_encode_count": int(
                            k_all_actual_result["edited_actual_encode_count"]
                        ),
                        "den6_online_candidate_actual_encode_count": 0,
                    })
                    latest_compression_debug = dict(
                        getattr(loss, "last_compression_debug", {}) or {}
                    )
                    latest_compression_debug.update(compression_debug_terms)
                    loss.last_compression_debug = latest_compression_debug
                    loss._den6_online_actual_audit = {
                        "baseline": int(k_all_actual_result["baseline_actual_encode_count"]),
                        "edited": int(k_all_actual_result["edited_actual_encode_count"]),
                        "candidate": 0,
                        "proposal": int(k_all_actual_result["proposal_actual_encode_count"]),
                        "worker_request_count": int(k_all_actual_result["proposal_actual_encode_count"]),
                        "edited_result_cache_hit": False,
                    }

                if full_cloud_amount_mode and torch.is_tensor(L_full_cloud_amount):
                    L = L + L_full_cloud_amount
                    if isinstance(full_cloud_amount_debug, dict):
                        full_cloud_amount_debug["full_cloud_amount_loss_added_to_total"] = True
                        full_cloud_amount_debug["full_cloud_amount_loss_requires_grad"] = bool(
                            L_full_cloud_amount.requires_grad
                        )

                """情報精査"""
                comp_debug = dict(getattr(loss, "last_compression_debug", {}) or {}) # 直前の圧縮Debug情報を取り出す
                if isinstance(full_cloud_amount_debug, dict) and full_cloud_amount_debug:
                    comp_debug.update(full_cloud_amount_debug)
                    comp_debug["actual_scope"] = "full_cloud" if full_cloud_amount_mode else comp_debug.get("actual_scope", "")
                    comp_debug["teacher_scope"] = (
                        "full_cloud_amount"
                        if full_cloud_amount_mode
                        else comp_debug.get("teacher_scope", "")
                    )
                # ============================================================
                # Direct Network Prune debug
                # ============================================================
                if bool(getattr(args, "direct_network_prune", False)):
                    comp_debug["direct_network_prune"] = True
                    comp_debug["direct_prune_use_raw_compression_loss"] = bool(
                        getattr(args, "direct_prune_use_raw_compression_loss", True)
                    )
                    comp_debug["direct_prune_expected_no_full_cloud_primary"] = True
                base_model_for_phase7 = model.module if hasattr(model, "module") else model
                phase7_structure_debug = getattr(base_model_for_phase7, "last_structure_debug", {}) or {}
                _phase7_update_from_structure(
                    comp_debug,
                    phase7_structure_debug,
                    is_anchor_step=True,
                )
                _phase7_update_from_voxel_state(comp_debug, model)
                # Phase7-4:
                # ablation modeと短時間判定用summaryをcomp_debugへ集約する。
                _phase7_add_ablation_summary_to_comp_debug(args, comp_debug)
                if isinstance(step_timing_breakdown, dict) and step_timing_breakdown:
                    comp_debug.update(step_timing_breakdown)
                    comp_debug["octree_build_time"] = float(
                        step_timing_breakdown.get("full_cloud_canonical_build_time", 0.0)
                    )
                    if full_cloud_amount_mode:
                        comp_debug["full_cloud_amount_step_time"] = float(
                            step_timing_breakdown.get("full_cloud_anchor_block_time", 0.0)
                        )
                if isinstance(full_cloud_anchor_runtime_timing, dict) and full_cloud_anchor_runtime_timing:
                    comp_debug["full_cloud_anchor_runtime_timing"] = dict(full_cloud_anchor_runtime_timing)
                    for runtime_key, runtime_value in full_cloud_anchor_runtime_timing.items():
                        try:
                            comp_debug[f"full_cloud_anchor_runtime_{runtime_key}"] = float(runtime_value)
                        except Exception:
                            pass
                if isinstance(step_actual_oracle_metric_debug, dict) and step_actual_oracle_metric_debug:
                    _copy_sparsepcgc_actual_oracle_debug_for_metrics(comp_debug, step_actual_oracle_metric_debug)
                oracle_actions_applied = bool(
                    getattr(args, "sparsepcgc_actual_oracle_apply_teacher_actions", False)
                    or getattr(args, "sparsepcgc_actual_oracle_apply_full_override", False)
                )
                policy_full_actual = finite_float_or_none(
                    comp_debug.get(
                        "full_cloud_actual_bit_percent",
                        comp_debug.get("actual_total_bit_percent", None),
                    )
                )
                if (
                    not oracle_actions_applied
                    and policy_full_actual is not None
                    and str(comp_debug.get("actual_scope", "")) == "full_cloud"
                ):
                    comp_debug["policy_full_cloud_actual_bit_percent"] = float(policy_full_actual)
                    comp_debug["oracle_full_cloud_override_used"] = False
                    comp_debug["policy_action_source"] = "network_actuator"

                comp_debug.update(
                    {
                        "is_anchor_refresh_step": True,
                        "is_subtree_step": False,
                        "stage_switch_guard_used": bool(stage_guard_debug.get("stage_switch_guard_used", False)),
                        "stage_original": str(stage_guard_debug.get("stage_original", current_stage)),
                        "stage_effective": str(stage_guard_debug.get("stage_effective", current_stage)),
                        "compression_loss_factor_original": float(
                            stage_guard_debug.get("compression_loss_factor_original", stage_factors.get("com", 1.0))
                        ),
                        "compression_loss_factor_effective": float(
                            stage_guard_debug.get("compression_loss_factor_effective", stage_factors.get("com", 1.0))
                        ),
                        "policy_loss_factor_original": float(
                            stage_guard_debug.get("policy_loss_factor_original", stage_factors.get("policy", 1.0))
                        ),
                        "policy_loss_factor_effective": float(
                            stage_guard_debug.get("policy_loss_factor_effective", stage_factors.get("policy", 1.0))
                        ),
                    }
                )

                anchor_debug_source = (
                    comp_debug if str(comp_debug.get("actual_scope", "")) == "full_cloud" else None
                )
                anchor_success_update_debug = {
                    "anchor_success_teacher_saved": False,
                    "anchor_success_teacher_percent": float("nan"),
                    "anchor_success_teacher_amount": float("nan"),
                    "anchor_success_memory_count": 0,
                }
                if (
                    isinstance(anchor_debug_source, dict)
                    and str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
                    != "ana_den6_online"
                ):
                    anchor_success_update_debug = _sparsepcgc_update_anchor_success_memory(
                        args,
                        cache_key=cache_key,
                        episode=episode,
                        global_step=global_train_step,
                        anchor_debug=anchor_debug_source,
                        structure_debug=phase7_structure_debug,
                        edit_stats=train_edit_stats,
                    )
                comp_debug.update(anchor_success_update_debug)


                if cp_debug: # Compression Primaryモード用のDebug情報が存在するか判定
                    comp_debug.update(cp_debug) # 圧縮目的のDebug情報を追加
                    loss.last_compression_debug = comp_debug # 統合後のcomp_debugをLossに保存

                if isinstance(compression_tensor_debug, dict):
                    compression_tensor_debug.update(
                        {
                            "compression_loss_tensor_value": finite_float_or_none(L_com),
                            "compression_loss_requires_grad": bool(torch.is_tensor(L_com) and L_com.requires_grad),
                            "compression_loss_grad_fn": (
                                type(L_com.grad_fn).__name__
                                if torch.is_tensor(L_com) and getattr(L_com, "grad_fn", None) is not None
                                else ""
                            ),
                            "compression_objective_tensor_value": finite_float_or_none(L_com_objective),
                            "compression_objective_requires_grad": bool(
                                torch.is_tensor(L_com_objective) and L_com_objective.requires_grad
                            ),
                            "compression_objective_grad_fn": (
                                type(L_com_objective.grad_fn).__name__
                                if torch.is_tensor(L_com_objective) and getattr(L_com_objective, "grad_fn", None) is not None
                                else ""
                            ),
                            "loss_bit_tensor_value": finite_float_or_none(loss_bit),
                            "loss_bit_requires_grad": bool(torch.is_tensor(loss_bit) and loss_bit.requires_grad),
                            "loss_bit_grad_fn": (
                                type(loss_bit.grad_fn).__name__
                                if torch.is_tensor(loss_bit) and getattr(loss_bit, "grad_fn", None) is not None
                                else ""
                            ),
                        }
                    )
                    comp_debug.update(compression_tensor_debug)
                    if compression_tensor_debug.get("compression_objective_tensor_value") is not None:
                        comp_debug["compression_objective"] = compression_tensor_debug.get("compression_objective_tensor_value")
                        comp_debug["lcom_objective"] = compression_tensor_debug.get("compression_objective_tensor_value")

                loss.last_compression_debug = comp_debug

                base_model = model.module if hasattr(model, "module") else model # DataParallelで包まれている場合は中身のモデルを取り出す
                structure_debug = getattr(base_model, "last_structure_debug", {}) or {} # モデル内部で記録された構造解析・構造修復のDebug情報を取得
                if isinstance(structure_debug, dict):
                    structure_debug = dict(structure_debug)
                    structure_debug["actual_oracle_full_cloud_teacher_required"] = bool(
                        getattr(args, "sparsepcgc_require_full_cloud_actual_teacher", True)
                    )
                    if isinstance(step_actual_oracle_metric_debug, dict) and step_actual_oracle_metric_debug:
                        _copy_sparsepcgc_actual_oracle_debug_for_metrics(structure_debug, step_actual_oracle_metric_debug)
                # ============================================================
                # Phase5:
                # Network内部のNode/Voxel・aggregation整合性をtrain.py側で監査する。
                # ============================================================
                phase5_structure_debug = _phase5_structure_safety_debug(
                    args,
                    structure_debug,
                    is_anchor_step=is_anchor_step,
                )

                if isinstance(comp_debug, dict):
                    comp_debug.update(phase5_structure_debug)
                    loss.last_compression_debug = comp_debug

                _phase5_apply_structure_guard(
                    args,
                    writer,
                    phase5_structure_debug,
                    global_step=global_train_step,
                )
                for debug_key in ( # 圧縮CSVからもfull-cloud構造入力を追えるように必要項目だけを転記する
                    "octree_input_mode",
                    "structural_voxel_mode",
                    "point_feature_voxel_mode",
                    "structural_voxel_key_available",
                    "point_feature_voxel_key_available",
                    "global_depth",
                    "enable_sparsepcgc_exact_occupancy_teacher",
                    "sparsepcgc_exact_teacher_mode",
                    "exact_teacher_uses_full_context",
                    "exact_teacher_fallback_reason",
                    "actuator_voxel_mode",
                    "actuator_local_recomputed",
                    "actuator_full_octree_context_available",
                    "actuator_parent_occupancy_code",
                    "actuator_sibling_count",
                    "actuator_ancestor_count",
                    "actuator_full_context_bonus_mean",
                    "before_occupied_voxel_count",
                    "after_occupied_voxel_count",
                    "occupied_voxel_delta",
                    "actuator_voxel_state_saved",
                    "actuator_final_voxel_state_available",
                    "final_voxel_update_mode",
                    "final_voxel_recomputed_from_pts_out",
                    "network_voxel_node_input_requested",
                    "network_voxel_node_input_used",
                    "network_voxel_node_fallback",
                    "network_voxel_node_fallback_reason",
                    "network_voxel_node_count",
                    "network_voxel_node_source",
                    "network_voxel_node_feature_shape",
                ):
                    if debug_key in structure_debug and debug_key not in comp_debug:
                        comp_debug[debug_key] = structure_debug.get(debug_key)
                if (
                    bool(getattr(args, "network_voxel_node_debug", True))
                    and bool(getattr(args, "_log_this_step", True))
                    and isinstance(structure_debug, dict)
                    and bool(structure_debug.get("network_voxel_node_input_requested", False))
                ):
                    writer.write(
                        "VoxelNodeInputDebug: "
                        f"used={bool(structure_debug.get('network_voxel_node_input_used', False))}, "
                        f"fallback={bool(structure_debug.get('network_voxel_node_fallback', False))}, "
                        f"reason={structure_debug.get('network_voxel_node_fallback_reason', '')}, "
                        f"node_count={int(structure_debug.get('network_voxel_node_count', 0) or 0)}, "
                        f"source={structure_debug.get('network_voxel_node_source', 'none')}, "
                        f"feature_shape={structure_debug.get('network_voxel_node_feature_shape', '')}, "
                        f"phase5_ok={bool(comp_debug.get('phase5_structure_safety_ok', False))}, "
                        f"phase5_reason={comp_debug.get('phase5_structure_safety_reason', '')}, "
                        f"cost_input={structure_debug.get('phase4_cost_attribution_input_mode', 'unknown')}, "
                        f"agg_source={structure_debug.get('phase4_aggregation_key_source', 'unknown')}, "
                        f"struct_source={structure_debug.get('phase4_structural_key_source', 'unknown')}, "
                        f"unit_count={int(structure_debug.get('phase4_aggregation_unit_count', 0) or 0)}, "
                        f"unit_size=[{int(structure_debug.get('phase4_aggregation_min_unit_size', 0) or 0)}, "
                        f"{int(structure_debug.get('phase4_aggregation_max_unit_size', 0) or 0)}]"
                    )

                operation_entropy_value = finite_float_or_none(structure_debug.get("operation_entropy")) # 探索多様性の移動平均を出すために現在値を取り出す
                if operation_entropy_value is not None:
                    operation_entropy_history = list(getattr(args, "_operation_entropy_history", [])) # 直近の操作entropy履歴を取得する
                    operation_entropy_history.append(float(operation_entropy_value)) # 現在Stepのentropyを履歴へ追加する
                    operation_entropy_window = max(int(getattr(args, "lr_decay_actual_window", 100)), 2) # actual診断と同じ窓幅で探索の生存状況を見る
                    operation_entropy_history = operation_entropy_history[-operation_entropy_window:] # 履歴が肥大化しないよう窓幅へ切る
                    args._operation_entropy_history = operation_entropy_history # 次Step以降のために履歴を保持する
                    comp_debug["operation_entropy_moving_avg"] = sum(operation_entropy_history) / float(max(len(operation_entropy_history), 1)) # 操作entropyの移動平均をCSVへ渡す
                if train_edit_stats is None:
                    train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args) # 入力/出力点群を比較し、操作を計算
                # 念のため、未設定時はfull cloudへ戻す。
                # 通常はStep開始時に設定され、Subtree学習時は選択Subtreeに差し替わる。
                if voxel_collision_input_gt is None:
                    voxel_collision_input_gt = input_xyz[:, :3, :]

                voxel_collision_debug = _collect_train_voxel_collision_stats(
                    args,
                    writer,
                    global_train_step,
                    {
                        "input_gt": voxel_collision_input_gt,
                        "model_output_raw": gen_xyz,
                        "compression_input": compression_gen_xyz,
                    },
                )
                if voxel_collision_debug:
                    comp_debug.update(voxel_collision_debug)
                    loss.last_compression_debug = comp_debug
                skip_optimizer_reason = None
                corr_debug = update_actual_correlation_debug(args, comp_debug, L_com, codec_actual_metric_pairs) # 圧縮推定値と実圧縮値の対応更新
                if corr_debug: # 相関診断結果が得られたら
                    comp_debug.update(corr_debug) # 診断情報の追加
                    loss.last_compression_debug = comp_debug # 相関診断を追加したcomp_debugを保存しなおす
                    corr_value = finite_float_or_none(corr_debug.get("corr_surrogate_actual")) # Surrogateと実圧縮の相関地を取り出す
                    if (
                        log_this_step
                        and not compact_step_text_log
                        and bool(getattr(args, "surrogate_realign_on_low_corr", False))
                        and corr_value is not None
                        and corr_value < float(getattr(args, "surrogate_realign_min_corr", 0.3))
                    ):
                        writer.write( "SurrogateRealignNotice: " f"corr_surrogate_actual={corr_value:.6f} below " f"{float(getattr(args, 'surrogate_realign_min_corr', 0.3)):.6f}; " f"realign_steps={int(getattr(args, 'surrogate_realign_steps', 0))} " "(current implementation logs the trigger; extra realign steps are not run unless added later).")
                    if bool(is_anchor_step):
                        comp_debug["full_cloud_anchor_no_grad"] = bool(full_cloud_anchor_no_grad)
                        comp_debug["full_cloud_anchor_no_grad_reason"] = str(full_cloud_anchor_no_grad_reason)
                        comp_debug["full_cloud_anchor_node_count"] = int(
                            locals().get("full_cloud_anchor_node_count", 0)
                        )
                        comp_debug["full_cloud_anchor_node_count_source"] = str(
                            locals().get("full_cloud_anchor_node_count_source", "")
                        )
                        comp_debug["full_cloud_anchor_grad_node_limit"] = int(
                            getattr(args, "full_cloud_anchor_grad_node_limit", 50000)
                        )
                        comp_debug["full_cloud_anchor_allow_grad"] = bool(
                            getattr(args, "full_cloud_anchor_allow_grad", False)
                        )

                    if (
                        bool(is_anchor_step)
                        and bool(full_cloud_anchor_no_grad)
                    ):
                        skip_optimizer_reason = "full_cloud_anchor_no_grad"
                        comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                        loss.last_compression_debug = comp_debug

                    elif ( bool(getattr(args, "skip_optimizer_on_actual_fallback", True)) and bool(comp_debug.get("actual_codec_fallback_to_proxy", False))):
                        skip_optimizer_reason = "actual_codec_fallback_to_proxy"
                        comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                        loss.last_compression_debug = comp_debug

                """CSV"""
                compression_metric_row = build_compression_metric_row(
                    args,
                    global_step=global_train_step,
                    episode=episode,
                    epoch=epoch,
                    step=step,
                    stage=current_stage,
                    comp_debug=comp_debug,
                    L_com=L_com,
                    sequence_name=sequence_name,
                    sequence_step=step,
                ) # 圧縮StepCSVに書き込む1行を作る
                if bool(getattr(args, "phase7_metric_columns", True)) and isinstance(comp_debug, dict):
                    for key in (
                        # SparsePCGC worker GPU stats
                        "sparsepcgc_worker_cuda_available",
                        "sparsepcgc_worker_cuda_device",
                        "sparsepcgc_worker_cuda_allocated_mb",
                        "sparsepcgc_worker_cuda_reserved_mb",
                        "sparsepcgc_worker_cuda_max_allocated_mb",
                        "sparsepcgc_worker_cuda_max_reserved_mb",
                        "sparsepcgc_worker_cuda_allocated_delta_mb",
                        "sparsepcgc_worker_cuda_reserved_delta_mb",

                        "sparsepcgc_worker_before_cuda_allocated_mb",
                        "sparsepcgc_worker_before_cuda_reserved_mb",
                        "sparsepcgc_worker_before_cuda_max_allocated_mb",
                        "sparsepcgc_worker_before_cuda_max_reserved_mb",
                        "sparsepcgc_worker_after_cuda_allocated_mb",
                        "sparsepcgc_worker_after_cuda_reserved_mb",
                        "sparsepcgc_worker_after_cuda_max_allocated_mb",
                        "sparsepcgc_worker_after_cuda_max_reserved_mb",

                        "actual_sparsepcgc_worker_cuda_allocated_mb",
                        "actual_sparsepcgc_worker_cuda_reserved_mb",
                        "actual_sparsepcgc_worker_cuda_max_allocated_mb",
                        "actual_sparsepcgc_worker_cuda_max_reserved_mb",
                        "actual_sparsepcgc_worker_cuda_allocated_delta_mb",
                        "actual_sparsepcgc_worker_cuda_reserved_delta_mb",
                        
                        "network_voxel_node_input_used",
                        "network_voxel_node_fallback",
                        "network_voxel_node_fallback_reason",
                        "network_voxel_node_source",
                        "network_voxel_node_count",
                        "network_voxel_node_feature_shape",
                        "full_cloud_anchor_node_voxel_used",
                        "full_cloud_anchor_actual_total_bit_percent",
                        "full_cloud_anchor_actual_bit_percent",
                        "full_cloud_anchor_teacher_type",
                        "full_cloud_anchor_full_cloud_teacher_used",
                        "full_cloud_anchor_point_count_before",
                        "full_cloud_anchor_point_count_after",
                        "full_cloud_anchor_unique_coord_before",
                        "full_cloud_anchor_unique_coord_after",
                        "subtree_node_voxel_used",

                        "voxel_restored_actual_used",
                        "voxel_restored_actual_fallback",
                        "voxel_restored_actual_fallback_reason",
                        "restored_actual_points",
                        "original_gen_points",
                        "restored_actual_xyz_min",
                        "restored_actual_xyz_max",
                        "original_gen_xyz_min",
                        "original_gen_xyz_max",
                        "final_voxel_coords_count",

                        "full_context_hard_loss",
                        "full_context_soft_proxy_loss",
                        "full_context_subtree_loss_total",
                        "full_cloud_actual_correction_loss_value",
                        "full_cloud_actual_correction_loss_enabled",
                        "full_cloud_actual_correction_soft_proxy_used",
                        "full_vs_subtree_gap",
                        "full_vs_context_gap",
                        "ema_full_vs_subtree_gap",
                        "ema_full_vs_context_gap",

                        "drop_ratio_soft",
                        "drop_ratio_hard",
                        "add_ratio_soft",
                        "add_ratio_hard",
                        "move_ratio_soft",
                        "move_ratio_hard",
                        "voxel_soft_drop_mean",
                        "voxel_soft_add_mean",
                        "voxel_soft_move_mean",
                        "voxel_edit_drop_count",
                        "voxel_edit_add_count",
                        "voxel_edit_move_count",
                        "same_voxel_move_rejected",
                        "existing_target_rejected",
                        "duplicate_target_rejected",
                        "child_slot_rejected",
                        "empty_target_rejected",

                        "drop_grad_norm",
                        "add_grad_norm",
                        "move_grad_norm",
                        "operation_gate_grad_norm",
                        "policy_grad_norm",
                        "cost_attr_grad_norm",
                        "cause_agg_grad_norm",
                        # Phase7-4 ablation summary
                        "phase7_ablation_mode",
                        "phase7_voxel_actual_enabled",
                        "phase7_full_context_soft_enabled",
                        "phase7_correction_loss_enabled",

                        # Phase7-4 grad sanity
                        "phase7_grad_drop_head",
                        "phase7_grad_add_head",
                        "phase7_grad_move_head",
                        "phase7_grad_operation_gate_head",
                        "phase7_grad_policy",
                        "phase7_grad_cost_attr",
                        "phase7_grad_sanity_drop_head_norm",
                        "phase7_grad_sanity_add_head_norm",
                        "phase7_grad_sanity_move_head_norm",
                        "phase7_grad_sanity_operation_gate_head_norm",
                        "phase7_grad_sanity_drop_amount_head_norm",
                        "phase7_grad_sanity_add_amount_head_norm",
                        "phase7_grad_sanity_move_amount_head_norm",
                        "phase7_grad_sanity_policy_norm",
                        "phase7_grad_sanity_cost_attr_norm",
                        "phase7_grad_sanity_cause_agg_norm",
                        "phase7_grad_sanity_drop_head_is_none",
                        "phase7_grad_sanity_add_head_is_none",
                        "phase7_grad_sanity_move_head_is_none",
                        "phase7_grad_sanity_operation_gate_head_is_none",
                        "phase7_grad_sanity_policy_is_none",
                        "phase7_grad_sanity_cost_attr_is_none",
                        "phase7_grad_sanity_cause_agg_is_none",
                        "phase7_grad_sanity_drop_head_is_nan",
                        "phase7_grad_sanity_add_head_is_nan",
                        "phase7_grad_sanity_move_head_is_nan",
                        "phase7_grad_sanity_operation_gate_head_is_nan",
                        "phase7_grad_sanity_policy_is_nan",
                        "phase7_grad_sanity_cost_attr_is_nan",
                        "phase7_grad_sanity_cause_agg_is_nan",
                        "phase7_grad_sanity_drop_head_is_zero_like",
                        "phase7_grad_sanity_add_head_is_zero_like",
                        "phase7_grad_sanity_move_head_is_zero_like",
                        "phase7_grad_sanity_operation_gate_head_is_zero_like",
                        "phase7_grad_sanity_policy_is_zero_like",
                        "phase7_grad_sanity_cost_attr_is_zero_like",
                        "phase7_grad_sanity_cause_agg_is_zero_like",

                        # Phase7-4 parameter update
                        "phase7_update_actuator",
                        "phase7_update_policy",
                        "phase7_update_cost_attr",
                        "phase7_update_cause_agg",
                        "phase7_param_update_actuator_norm",
                        "phase7_param_update_policy_norm",
                        "phase7_param_update_cost_attr_norm",
                        "phase7_param_update_cause_agg_norm",
                        "phase7_param_update_actuator_max",
                        "phase7_param_update_policy_max",
                        "phase7_param_update_cost_attr_max",
                        "phase7_param_update_cause_agg_max",
                        "phase7_param_update_actuator_updated",
                        "phase7_param_update_policy_updated",
                        "phase7_param_update_cost_attr_updated",
                        "phase7_param_update_cause_agg_updated",

                        # Phase7-4 short-run判定
                        "phase7_actual_input_points",
                        "phase7_restored_actual_points",
                        "phase7_full_context_soft_proxy_loss",
                        "phase7_correction_loss",
                        "phase7_full_cloud_actual_delta",
                        "phase7_subtree_actual_delta",
                        "phase7_full_vs_subtree_gap",
                    ):
                        if key in comp_debug:
                            compression_metric_row[key] = comp_debug[key]
                if isinstance(comp_debug, dict):
                    for key in (
                        "full_cloud_corr_update_used",
                        "full_cloud_corr_update_reason",
                        "full_cloud_corr_loss_used",
                        "full_cloud_corr_loss_reason",
                        "full_cloud_corr_loss_value",
                        "full_cloud_corr_loss_enabled",
                        "full_cloud_corr_loss_added_to_total",
                        "full_cloud_corr_loss_weight_used",
                        "full_cloud_corr_loss_requires_grad",
                        "full_cloud_corr_loss_severity",
                        "full_cloud_corr_ema_full_vs_subtree_gap",
                        "full_cloud_corr_ema_full_vs_context_gap",
                        "full_cloud_corr_ema_full_vs_proxy_gap",
                        "full_cloud_corr_ema_full_actual_delta",
                        "full_cloud_corr_last_full_actual_delta",
                        "full_cloud_corr_last_subtree_actual_delta",
                        "full_cloud_corr_last_full_context_delta",
                        "full_cloud_corr_last_subtree_proxy_delta",
                        "full_cloud_corr_last_update_step",
                        "full_cloud_corr_move_count",
                        "full_cloud_corr_add_count",
                        "full_cloud_corr_drop_count",
                        "full_cloud_corr_same_voxel_move_rejected",
                        "full_cloud_corr_existing_target_rejected",
                        "full_cloud_corr_duplicate_target_rejected",
                        "full_cloud_corr_child_slot_rejected",
                        "full_cloud_corr_empty_target_rejected",
                    ):
                        if key in comp_debug:
                            compression_metric_row[key] = comp_debug[key]
                if (
                    bool(getattr(args, "full_cloud_actual_correction_debug", True))
                    and not compact_step_text_log
                    and bool(getattr(args, "_log_this_step", True))
                    and isinstance(comp_debug, dict)
                    and (
                        comp_debug.get("full_cloud_corr_update_used", False)
                        or comp_debug.get("full_cloud_corr_loss_used", False)
                    )
                ):
                    writer.write(
                        "FullCloudActualCorrection: "
                        f"update_used={bool(comp_debug.get('full_cloud_corr_update_used', False))}, "
                        f"update_reason={comp_debug.get('full_cloud_corr_update_reason', 'none')}, "
                        f"loss_used={bool(comp_debug.get('full_cloud_corr_loss_used', False))}, "
                        f"loss_enabled={bool(comp_debug.get('full_cloud_corr_loss_enabled', False))}, "
                        f"loss={float(comp_debug.get('full_cloud_corr_loss_value', 0.0) or 0.0):.6g}, "
                        f"ema_full_delta={float(comp_debug.get('full_cloud_corr_ema_full_actual_delta', 0.0) or 0.0):.6g}, "
                        f"gap_full_subtree={float(comp_debug.get('full_cloud_corr_ema_full_vs_subtree_gap', 0.0) or 0.0):.6g}, "
                        f"gap_full_context={float(comp_debug.get('full_cloud_corr_ema_full_vs_context_gap', 0.0) or 0.0):.6g}, "
                        f"gap_full_proxy={float(comp_debug.get('full_cloud_corr_ema_full_vs_proxy_gap', 0.0) or 0.0):.6g}, "
                        f"move={float(comp_debug.get('full_cloud_corr_move_count', 0.0) or 0.0):.0f}, "
                        f"add={float(comp_debug.get('full_cloud_corr_add_count', 0.0) or 0.0):.0f}, "
                        f"drop={float(comp_debug.get('full_cloud_corr_drop_count', 0.0) or 0.0):.0f}, "
                        f"move_reject_same={float(comp_debug.get('full_cloud_corr_same_voxel_move_rejected', 0.0) or 0.0):.0f}, "
                        f"move_reject_existing={float(comp_debug.get('full_cloud_corr_existing_target_rejected', 0.0) or 0.0):.0f}, "
                        f"move_reject_duplicate={float(comp_debug.get('full_cloud_corr_duplicate_target_rejected', 0.0) or 0.0):.0f}, "
                        f"move_reject_child_slot={float(comp_debug.get('full_cloud_corr_child_slot_rejected', 0.0) or 0.0):.0f}, "
                        f"move_reject_empty={float(comp_debug.get('full_cloud_corr_empty_target_rejected', 0.0) or 0.0):.0f}"
                    )

                if isinstance(comp_debug, dict):
                    for key in (
                        "full_context_subtree_delta_used",
                        "full_context_subtree_delta_reason",
                        "full_context_subtree_delta_value",
                        "full_context_subtree_delta_before_nodes",
                        "full_context_subtree_delta_after_nodes",
                        "full_context_subtree_delta_node_delta_norm",
                        "full_context_subtree_delta_before_single",
                        "full_context_subtree_delta_after_single",
                        "full_context_subtree_delta_single_delta",
                        "full_context_subtree_delta_before_entropy",
                        "full_context_subtree_delta_after_entropy",
                        "full_context_subtree_delta_entropy_delta",
                        "full_context_subtree_delta_before_lowprob",
                        "full_context_subtree_delta_after_lowprob",
                        "full_context_subtree_delta_lowprob_delta",
                        "full_context_subtree_delta_before_nll",
                        "full_context_subtree_delta_after_nll",
                        "full_context_subtree_delta_nll_delta",
                        "full_context_subtree_delta_before_count",
                        "full_context_subtree_delta_after_count",
                        "full_context_subtree_delta_count_delta_norm",
                        "full_context_subtree_delta_before_isolated",
                        "full_context_subtree_delta_after_isolated",
                        "full_context_subtree_delta_isolated_delta",
                        "full_context_subtree_delta_grad_used",
                        "full_context_subtree_delta_weight",
                        "cp_full_context_subtree_delta",
                        "cp_full_context_subtree_delta_requires_grad",
                    ):
                        if key in comp_debug:
                            compression_metric_row[key] = comp_debug[key]
                # ============================================================
                # Actual hard Occupancy値はActual列・exact列にだけ入れる
                # Predicted列はsoft proxy側の値を残す
                # ============================================================
                if isinstance(comp_debug, dict):
                    if "exact_occ_entropy_delta" in comp_debug:
                        compression_metric_row["actual_occupancy_entropy_delta"] = comp_debug["exact_occ_entropy_delta"]
                        compression_metric_row["exact_hard_occupancy_entropy_delta"] = comp_debug["exact_occ_entropy_delta"]

                        pred = compression_metric_row.get("predicted_occupancy_entropy_delta", None)
                        if pred is not None:
                            try:
                                compression_metric_row["gap_occupancy_entropy_delta"] = (
                                    float(pred) - float(comp_debug["exact_occ_entropy_delta"])
                                )
                            except Exception:
                                pass

                    if "exact_occ_nll_delta" in comp_debug:
                        compression_metric_row["actual_occupancy_nll_delta"] = comp_debug["exact_occ_nll_delta"]
                        compression_metric_row["exact_hard_occupancy_nll_delta"] = comp_debug["exact_occ_nll_delta"]

                        pred = compression_metric_row.get("predicted_occupancy_nll_delta", None)
                        if pred is not None:
                            try:
                                compression_metric_row["gap_occupancy_nll_delta"] = (
                                    float(pred) - float(comp_debug["exact_occ_nll_delta"])
                                )
                            except Exception:
                                pass

                    if "exact_occ_pattern_delta_norm" in comp_debug:
                        compression_metric_row["actual_occupancy_pattern_delta"] = comp_debug["exact_occ_pattern_delta_norm"]
                        compression_metric_row["exact_hard_occupancy_pattern_delta_norm"] = comp_debug["exact_occ_pattern_delta_norm"]

                        pred = compression_metric_row.get("predicted_occupancy_pattern_delta", None)
                        if pred is not None:
                            try:
                                compression_metric_row["gap_occupancy_pattern_delta"] = (
                                    float(pred) - float(comp_debug["exact_occ_pattern_delta_norm"])
                                )
                            except Exception:
                                pass

                    if "exact_occ_lowprob_after" in comp_debug:
                        compression_metric_row["actual_lowprob_occupancy_ratio_after"] = comp_debug["exact_occ_lowprob_after"]
                        compression_metric_row["exact_hard_lowprob_occupancy_ratio_after"] = comp_debug["exact_occ_lowprob_after"]

                        pred = compression_metric_row.get("predicted_lowprob_occupancy_ratio", None)
                        if pred is not None:
                            try:
                                compression_metric_row["gap_lowprob_occupancy_ratio"] = (
                                    float(pred) - float(comp_debug["exact_occ_lowprob_after"])
                                )
                            except Exception:
                                pass

                    if "exact_occupancy_ste_weight" in comp_debug:
                        compression_metric_row["training_exact_occupancy_ste_weight"] = comp_debug["exact_occupancy_ste_weight"]

                    if "exact_occupancy_ste_grad_used" in comp_debug:
                        compression_metric_row["training_exact_occupancy_ste_grad_used"] = comp_debug["exact_occupancy_ste_grad_used"]
                operation_metric_row = build_operation_metric_row(
                    args,
                    global_step=global_train_step,
                    episode=episode,
                    epoch=epoch,
                    step=step,
                    stage=current_stage,
                    comp_debug=comp_debug,
                    structure_debug=structure_debug,
                    edit_stats=train_edit_stats,
                    sequence_name=sequence_name,
                    sequence_step=step,
                ) # 点操作StepCSVに書き込む1行を作る
                operation_metric_row["actual_oracle_full_cloud_teacher_required"] = bool(
                    getattr(args, "sparsepcgc_require_full_cloud_actual_teacher", True)
                )

                """ログ"""
                if log_this_step:
                    if compact_step_text_log:
                        log_compact_step_summary(
                            writer,
                            step,
                            num_steps,
                            args,
                            loss,
                            comp_debug,
                            structure_debug,
                            train_edit_stats,
                            L=L,
                            L_geom=L_geom,
                            L_com=L_com,
                            L_com_objective=L_com_objective,
                            L_attr=L_attr,
                            L_policy=L_policy,
                            L_actuator=L_actuator,
                            loss_bit=loss_bit,
                            loss_single=loss_single,
                            loss_nodes=loss_nodes,
                            stage_factors=stage_factors,
                            step_completed=None,
                        )
                    else:
                        log_step_loss( writer, step, num_steps, L, L_geom, L_com, L_com_objective, L_attr, L_policy, L_actuator, Lp_out, La_fit, La_rep, L_discrete_policy, loss_bit, loss_single, loss_nodes)
                        if cp_debug and bool(getattr(args, "cp_log_grad_terms", True)):
                            log_compression_primary_terms(writer, step, num_steps, cp_debug)
                        log_compression_stats( writer, step, num_steps, comp_debug)
                        before_node, after_node, before_single, after_single = log_compression_train_debug( writer, step, num_steps, args, comp_debug, loss, L_com)
                        log_codec_actual_correlation( writer, step, num_steps, args, comp_debug, codec_actual_metric_pairs, before_node, after_node, before_single, after_single)
                        log_sparsepcgc_train_debug( writer, step, num_steps, args, comp_debug, sparsepcgc_proxy_actual_pairs)
                        soft_proxy_debug_text = _format_soft_proxy_debug(args)
                        if soft_proxy_debug_text:
                            writer.write(f"SoftProxyGradDebug: {soft_proxy_debug_text}")
                        if structure_debug:
                            log_structure_debug( writer, structure_debug, step, num_steps)
                            write_structure_decision_debug( writer, f"StructureDecision step={step + 1}/{num_steps}", structure_debug)
                if timing_enabled:
                    sync_for_timing(use_cuda)
                    timing_loss_end = time.time()

                """勾配確認"""
                step_grad_loss_items = [
                    ("L_total", L),
                    ("L_downstream", L_downstream),
                    ("L_geom", L_geom),
                    ("L_com", L_com),
                    ("L_com_objective", L_com_objective),
                    ("full_cloud_amount", L_full_cloud_amount),
                    ("L_attr", L_attr),
                    ("L_policy", L_policy),
                    ("L_actuator", L_actuator),
                    ("weighted_L_attr", stage_factors["attr"] * args.w_attr * L_attr),
                    ("weighted_L_policy", stage_factors["policy"] * args.w_policy * L_policy),
                    ("weighted_L_actuator", stage_factors["repair"] * args.w_actuator * L_actuator),
                    ("loss_bit", loss_bit),
                    ("loss_nodes", loss_nodes),
                    ("loss_single", loss_single),
                    ("surrogate_loss_for_grad", terms.get("surrogate", None)),
                    ("L_discrete_policy", L_discrete_policy),
                ]
                if torch.is_tensor(La_fit) and La_fit.requires_grad:
                    step_grad_loss_items.append(("La_fit", La_fit))
                sparsepcgc_aux_term = terms.get("sparsepcgc", None)
                if torch.is_tensor(sparsepcgc_aux_term) and sparsepcgc_aux_term.requires_grad:
                    step_grad_loss_items.append(("sparsepcgc_aux_objective", sparsepcgc_aux_term))
                if (
                    bool(is_anchor_step)
                    and bool(full_cloud_anchor_no_grad)
                ):
                    step_grad_rows = []
                    if not compact_step_text_log:
                        writer.write("StepGradProbe: skipped because full_cloud_anchor_no_grad=True")
                else:
                    step_grad_rows = build_step_grad_rows(
                        args,
                        model,
                        step_grad_loss_items,
                        global_step=global_train_step,
                        episode=episode,
                        epoch=epoch,
                        step=step,
                        stage=current_stage,
                    )
                if step_grad_rows:
                    append_count = 0
                    for step_grad_row in step_grad_rows:
                        append_csv_row(
                            metric_csv_paths.get("step_grad"),
                            STEP_GRAD_COLUMNS,
                            step_grad_row,
                        )
                        append_count += 1
                    if not compact_step_text_log:
                        writer.write(
                            "StepGradProbe: "
                            f"rows={append_count}, "
                            f"path={metric_csv_paths.get('step_grad')}"
                        )

                """勾配を流す"""
                step_completed = False # Optimizer更新が成功したかのフラグ
                total_loss_finite = bool(torch.isfinite(L.detach()).all().item()) and skip_optimizer_reason is None # LがNanなどでないか否かの判定
                param_update_snapshots = None # 更新前パラメータの記録を見作成で初期化
                network_only_param_before = None
                network_only_head_audit_due = bool(
                    global_train_step == 0
                    or global_train_step % max(int(getattr(
                        args, "network_only_head_audit_interval", 10
                    )), 1) == 0
                )
                if network_only_full_cloud and total_loss_finite and network_only_head_audit_due:
                    audit_model = _unwrap_train_model(model)
                    policy_module_for_audit = (
                        audit_model.network_k_proposal_policy
                        if heuristic_mode == "network_k_proposal_policy"
                        else audit_model.single_plan_student
                        if heuristic_mode == "single_plan_student"
                        else audit_model.network_only_codec_policy
                    )
                    network_only_param_before = {
                        name: parameter.detach().clone()
                        for name, parameter in policy_module_for_audit.named_parameters()
                        if parameter.requires_grad
                    }
                amp_info = { "enabled": bool(amp_scaler_enabled), "found_inf": None, "scale_before": None, "scale_after": None, "consecutive_amp_skips": int(consecutive_amp_skips)} # AMPの状態を記録する辞書を作る
                last_nonfinite_grad_summary = None
                if total_loss_finite: # 総損失がInfでないとき、更新前パラメータを記録
                    param_update_snapshots = capture_param_update_snapshots( args, model, step + 1, num_steps)
                # cuDNN backward用workspaceが、連続full-cloud Stepで断片化した
                # allocator cacheに阻まれないよう未使用blockだけを返却する。
                # 生きているFP32 Tensorとautograd graphには触れない。
                if one_plan_full_cloud and use_cuda and torch.cuda.is_available():
                    reserved_mb = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
                    cache_threshold_mb = float(getattr(
                        args, "full_cloud_empty_cache_threshold_mb", 8192.0
                    ))
                    if cache_threshold_mb <= 0.0 or reserved_mb >= cache_threshold_mb:
                        torch.cuda.empty_cache()
                if skip_optimizer_reason is not None: # Optimizer更新を止める必要があるか否かの判定
                    writer.write(
                        f"Skip Optimizing!!! reason={skip_optimizer_reason}; "
                        f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}"
                    ) # Skip理由と位置を同じ行に出す

                    if skip_optimizer_reason == "actual_codec_fallback_to_proxy":
                        writer.write(
                            "Skipped optimizer step because actual codec teacher fell back to proxy at "
                            f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}; "
                            "this prevents proxy-only updates from replacing real-compression imitation."
                        )
                    elif skip_optimizer_reason == "full_cloud_anchor_no_grad":
                        writer.write(
                            "Skipped optimizer step because FullCloud anchor is used only for "
                            "no-grad calibration / teacher update / actual evaluation. "
                            f"reason={full_cloud_anchor_no_grad_reason}, "
                            f"node_count={int(locals().get('full_cloud_anchor_node_count', 0))}, "
                            f"node_count_source={str(locals().get('full_cloud_anchor_node_count_source', ''))}, "
                            f"grad_node_limit={int(getattr(args, 'full_cloud_anchor_grad_node_limit', 50000))}"
                        )
                elif not total_loss_finite:
                    skip_optimizer_reason = "non_finite_total_loss"
                    comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                    loss.last_compression_debug = comp_debug
                    writer.write( f"Skip Optimizing!!! reason=non_finite_total_loss; " f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}, L={float(L.detach().float().mean().cpu()) if torch.is_tensor(L) else float('nan'):.6g}") # 非有限Lossの理由と値を同じ行に出す
                    writer.write( f"Skipped optimizer step due to non-finite total loss at " f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}.")
                elif amp_scaler_enabled: # AMP用の逆伝播・更新処理へ進む
                    """AMP更新/勾配"""
                    scale_before = float(scaler.get_scale()) # BackWard前のAMP loss caleを取得
                    amp_info["scale_before"] = scale_before # AMP Debug情報に更新前ぉssSacleを保存
                    scaler.scale(L).backward() # LをAMP用にスケーリングしてから逆伝播
                    scaler.unscale_(optimizer) # Optimizer内の勾配を元のスケールへ戻す
                    operation_grad_balance_debug = _balance_actual_operation_head_gradients(
                        args,
                        model,
                        structure_debug,
                    )
                    comp_debug.update(operation_grad_balance_debug)
                    if den6_online_full_cloud and _den6_online_grad_audit_enabled(args, global_train_step):
                        comp_debug.update(_den6_online_grad_norms(model))
                    # Phase7-4:
                    # unscale後の実gradを対象にsanity checkする。
                    _phase7_log_grad_sanity(
                        args,
                        writer,
                        model,
                        comp_debug,
                        global_train_step,
                    )

                    if bool(getattr(args, "phase7_grad_debug", False)):
                        phase7_grad_debug = _phase7_named_grad_norms(model)
                        comp_debug.update(phase7_grad_debug)
                        if _phase7_debug_enabled(args, global_train_step):
                            _phase7_writer_line(
                                args,
                                writer,
                                "Phase7GradDebug: "
                                f"drop={phase7_grad_debug.get('drop_grad_norm', 0.0):.6g}, "
                                f"add={phase7_grad_debug.get('add_grad_norm', 0.0):.6g}, "
                                f"move={phase7_grad_debug.get('move_grad_norm', 0.0):.6g}, "
                                f"policy={phase7_grad_debug.get('policy_grad_norm', 0.0):.6g}, "
                                f"cost_attr={phase7_grad_debug.get('cost_attr_grad_norm', 0.0):.6g}, "
                                f"cause_agg={phase7_grad_debug.get('cause_agg_grad_norm', 0.0):.6g}"
                            )
                    if _phase7_debug_enabled(args, global_train_step):
                        _phase7_writer_line(
                            args,
                            writer,
                            "Phase7ShortRunDebug: "
                            f"mode={comp_debug.get('phase7_ablation_mode', 'none')}, "
                            f"voxel_actual={bool(comp_debug.get('phase7_voxel_actual_enabled', False))}, "
                            f"full_context_soft={bool(comp_debug.get('phase7_full_context_soft_enabled', False))}, "
                            f"correction_loss_enabled={bool(comp_debug.get('phase7_correction_loss_enabled', False))}, "
                            f"actual_points={int(comp_debug.get('phase7_actual_input_points', 0) or 0)}, "
                            f"restored_points={int(comp_debug.get('phase7_restored_actual_points', 0) or 0)}, "
                            f"full_context_soft_loss={float(comp_debug.get('phase7_full_context_soft_proxy_loss', 0.0) or 0.0):.6g}, "
                            f"correction_loss={float(comp_debug.get('phase7_correction_loss', 0.0) or 0.0):.6g}, "
                            f"full_delta={float(comp_debug.get('phase7_full_cloud_actual_delta', 0.0) or 0.0):.6g}, "
                            f"subtree_delta={float(comp_debug.get('phase7_subtree_actual_delta', 0.0) or 0.0):.6g}, "
                            f"gap={float(comp_debug.get('phase7_full_vs_subtree_gap', 0.0) or 0.0):.6g}"
                        )

                    if bool(getattr(args, "debug_grad_flow", False)):
                        log_grad_flow(args, writer, model, step + 1, num_steps, global_step=global_train_step) # 各層・各モジュールに勾配が届いているか否かの判定ログ
                    nonfinite_grad_summary = _summarize_nonfinite_grads(
                        model,
                        limit=int(getattr(args, "nonfinite_grad_log_param_limit", 8)),
                    )
                    last_nonfinite_grad_summary = nonfinite_grad_summary
                    if (
                        bool(getattr(args, "skip_optimizer_on_nonfinite_grad", True))
                        and bool(nonfinite_grad_summary.get("has_nonfinite", False))
                    ):
                        skip_optimizer_reason = "non_finite_grad"
                        comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                        comp_debug["nonfinite_grad_summary"] = _format_nonfinite_grad_summary(nonfinite_grad_summary)
                        loss.last_compression_debug = comp_debug
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        scale_after = float(scaler.get_scale())
                        amp_info["found_inf"] = float(nonfinite_grad_summary.get("bad_element_count", 0))
                        amp_info["scale_after"] = scale_after
                        writer.write(
                            "Skip Optimizing!!! reason=non_finite_grad; "
                            f"{comp_debug['nonfinite_grad_summary']}; "
                            f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}"
                        )
                        consecutive_amp_skips += 1
                    else:
                        grad_clip = float(getattr(args, "train_grad_clip", 0.0)) # 勾配ノルムの上限値を設定から取得する
                        if grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in model.parameters() if p.requires_grad],
                                max_norm=grad_clip,
                            )

                        phase7_param_snapshot = None
                        if _phase7_param_update_enabled(args, global_train_step):
                            phase7_param_snapshot = _phase7_take_param_snapshot(model)

                        scaler.step(optimizer) # Optimizer更新

                        phase7_param_update_stats = {}
                        if phase7_param_snapshot is not None:
                            phase7_param_update_stats = _phase7_compare_param_snapshot(
                                model,
                                phase7_param_snapshot,
                                zero_eps=float(getattr(args, "phase7_grad_zero_eps", 1e-12)),
                            )

                        # Phase7-4:
                        # GradScalerの内部属性 _per_optimizer_states はPyTorchの版によって存在しない。
                        # そのため、AMP skip判定は公開APIのscale変化で行う。
                        # scaler.step() がoverflowでoptimizer.stepをskipした場合、多くの環境ではscale_after < scale_before になる。
                        scaler.update() # GradScalerのLoss Scaleを更新
                        scale_after = float(scaler.get_scale()) # 更新後Loss Scaleを取得

                        found_inf = 1.0 if scale_after < scale_before else 0.0
                        amp_info["found_inf"] = found_inf
                        amp_info["scale_after"] = scale_after

                        step_completed = scale_after >= scale_before
                        if step_completed: # 成功した場合の処理
                            consecutive_amp_skips = 0
                            if phase7_param_update_stats:
                                _phase7_log_param_update(
                                    args,
                                    writer,
                                    comp_debug,
                                    phase7_param_update_stats,
                                    global_train_step,
                                )
                        else:
                            skip_optimizer_reason = "amp_found_inf_or_scale_drop"
                            comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                            loss.last_compression_debug = comp_debug
                            writer.write( f"Skip Optimizing!!! reason=amp_found_inf_or_scale_drop; " f"found_inf={found_inf:.6g}, scale_before={scale_before:.6g}, scale_after={scale_after:.6g}, " f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}") # AMP skipの理由とscale状態を同じ行に出す
                            consecutive_amp_skips += 1 # Skipの連続回数を1回増やす
                            if consecutive_amp_skips >= amp_overflow_patience: # AMP Overflowが設定回数以上連続したかの判定
                                consecutive_amp_skips = 0
                                if use_cuda and cuda_bf16_ops_safe():
                                    amp_dtype = torch.bfloat16
                                    amp_scaler_enabled = False
                                    writer.write( "float16 AMP overflow persisted; switched AMP autocast to bfloat16.")
                                else:
                                    use_amp = False
                                    amp_scaler_enabled = False
                                    scaler = torch.cuda.amp.GradScaler(enabled=False)
                                    writer.write( "float16 AMP overflow persisted; disabled AMP and continue in float32.")
                else:
                    L.backward() # 通常の勾配を流す
                    operation_grad_balance_debug = _balance_actual_operation_head_gradients(
                        args,
                        model,
                        structure_debug,
                    )
                    comp_debug.update(operation_grad_balance_debug)
                    if den6_online_full_cloud and _den6_online_grad_audit_enabled(args, global_train_step):
                        comp_debug.update(_den6_online_grad_norms(model))
                    # Phase7-4:
                    # backward直後の実gradを対象にsanity checkする。
                    _phase7_log_grad_sanity(
                        args,
                        writer,
                        model,
                        comp_debug,
                        global_train_step,
                    )
                    if bool(getattr(args, "phase7_grad_debug", False)):
                        phase7_grad_debug = _phase7_named_grad_norms(model)
                        comp_debug.update(phase7_grad_debug)
                        if _phase7_debug_enabled(args, global_train_step):
                            _phase7_writer_line(
                                args,
                                writer,
                                "Phase7GradDebug: "
                                f"drop={phase7_grad_debug.get('drop_grad_norm', 0.0):.6g}, "
                                f"add={phase7_grad_debug.get('add_grad_norm', 0.0):.6g}, "
                                f"move={phase7_grad_debug.get('move_grad_norm', 0.0):.6g}, "
                                f"policy={phase7_grad_debug.get('policy_grad_norm', 0.0):.6g}, "
                                f"cost_attr={phase7_grad_debug.get('cost_attr_grad_norm', 0.0):.6g}, "
                                f"cause_agg={phase7_grad_debug.get('cause_agg_grad_norm', 0.0):.6g}"
                            )
                    if bool(getattr(args, "debug_grad_flow", False)):
                        log_grad_flow(args, writer, model, step + 1, num_steps, global_step=global_train_step) # 各モジュールの勾配状態をログに出す
                    nonfinite_grad_summary = _summarize_nonfinite_grads(
                        model,
                        limit=int(getattr(args, "nonfinite_grad_log_param_limit", 8)),
                    )
                    last_nonfinite_grad_summary = nonfinite_grad_summary
                    if (
                        bool(getattr(args, "skip_optimizer_on_nonfinite_grad", True))
                        and bool(nonfinite_grad_summary.get("has_nonfinite", False))
                    ):
                        skip_optimizer_reason = "non_finite_grad"
                        comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                        comp_debug["nonfinite_grad_summary"] = _format_nonfinite_grad_summary(nonfinite_grad_summary)
                        loss.last_compression_debug = comp_debug
                        optimizer.zero_grad(set_to_none=True)
                        writer.write(
                            "Skip Optimizing!!! reason=non_finite_grad; "
                            f"{comp_debug['nonfinite_grad_summary']}; "
                            f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}"
                        )
                    else:
                        grad_clip = float(getattr(args, "train_grad_clip", 0.0)) # 勾配クリップの上限値取得
                        phase7_param_snapshot = None
                        if _phase7_param_update_enabled(args, global_train_step):
                            phase7_param_snapshot = _phase7_take_param_snapshot(model)

                        optimizer.step() # モデルパラメータの更新
                        step_completed = True # 更新フラグをTrueにする
                        consecutive_amp_skips = 0 # AMP loss scale連続Skip回数を0に戻す

                        if phase7_param_snapshot is not None:
                            phase7_param_update_stats = _phase7_compare_param_snapshot(
                                model,
                                phase7_param_snapshot,
                                zero_eps=float(getattr(args, "phase7_grad_zero_eps", 1e-12)),
                            )
                            _phase7_log_param_update(
                                args,
                                writer,
                                comp_debug,
                                phase7_param_update_stats,
                                global_train_step,
                            )
                episode_optimizer_total_count += 1
                if step_completed:
                    episode_optimizer_step_count += 1
                    consecutive_nonfinite_grad_skips = 0
                elif skip_optimizer_reason == "non_finite_grad":
                    episode_nonfinite_grad_skip_count += 1
                    consecutive_nonfinite_grad_skips += 1
                    episode_max_consecutive_nonfinite_grad_skips = max(
                        episode_max_consecutive_nonfinite_grad_skips,
                        consecutive_nonfinite_grad_skips,
                    )
                optimizer_success_ratio = episode_optimizer_step_count / float(max(episode_optimizer_total_count, 1))
                if last_nonfinite_grad_summary:
                    comp_debug["nonfinite_grad_bad_element_count"] = int(last_nonfinite_grad_summary.get("bad_element_count", 0))
                    comp_debug["nonfinite_grad_checked_param_count"] = int(last_nonfinite_grad_summary.get("checked_param_count", 0))
                    comp_debug["nonfinite_grad_checked_element_count"] = int(last_nonfinite_grad_summary.get("checked_element_count", 0))
                    if bool(last_nonfinite_grad_summary.get("has_nonfinite", False)) and "nonfinite_grad_summary" not in comp_debug:
                        comp_debug["nonfinite_grad_summary"] = _format_nonfinite_grad_summary(last_nonfinite_grad_summary)
                comp_debug["optimizer_step"] = bool(step_completed)
                comp_debug["optimizer_skip_reason"] = str(skip_optimizer_reason or "")
                comp_debug["optimizer_step_success_rate_episode"] = float(optimizer_success_ratio)
                comp_debug["consecutive_nonfinite_grad_skips"] = int(consecutive_nonfinite_grad_skips)
                loss.last_compression_debug = comp_debug
                compression_metric_row.update(
                    {
                        "optimizer_step": bool(step_completed),
                        "optimizer_skip_reason": str(skip_optimizer_reason or ""),
                        "optimizer_step_success_rate_episode": float(optimizer_success_ratio),
                        "nonfinite_grad_bad_element_count": int(comp_debug.get("nonfinite_grad_bad_element_count", 0)),
                        "nonfinite_grad_checked_param_count": int(comp_debug.get("nonfinite_grad_checked_param_count", 0)),
                        "nonfinite_grad_checked_element_count": int(comp_debug.get("nonfinite_grad_checked_element_count", 0)),
                        "consecutive_nonfinite_grad_skips": int(consecutive_nonfinite_grad_skips),
                        "nonfinite_grad_summary": str(comp_debug.get("nonfinite_grad_summary", "")),
                    }
                )
                if step_completed: # Optimizer更新が成功したら差分ログを出す
                    log_param_updates( args, writer, model, param_update_snapshots, step + 1, num_steps)
                network_only_head_audit = {}
                if network_only_full_cloud and isinstance(network_only_param_before, dict):
                    audit_model = _unwrap_train_model(model)
                    grouped = {
                        "where": (
                            "local_trunk", "local_cost_head", "shared_local_trunk",
                            "policy.local_trunk", "policy.local_cost_head",
                            "shared_basis_head", "fixed_codec_basis_head", "plan_tokens",
                            "token_mixer", "coefficient_head",
                            "coefficient_scale_head", "priority_head", "threshold_head",
                            "order_head", "variant_head", "slot_order_bias", "slot_variant_bias",
                        ),
                        "amount": (
                            "amount_head", "amount_scale_head", "share_head", "share_scale_head",
                            "policy.amount_head", "policy.share_head",
                            "slot_ratio_bias", "slot_share_bias", "plan_tokens",
                        ),
                        "action": (
                            "gate_head", "enable_head", "plan_tokens", "policy.gate_head",
                        ),
                        "direction": (
                            "direction_field_head", "shared_direction_head", "direction_delta_head",
                            "plan_tokens", "policy.direction_field_head",
                        ),
                        "interaction": (
                            "interaction_head", "critic", "critic_interaction_head", "critic_gain_head",
                            "utility_head",
                        ),
                    }
                    policy_module_for_audit = (
                        audit_model.network_k_proposal_policy
                        if heuristic_mode == "network_k_proposal_policy"
                        else audit_model.single_plan_student
                        if heuristic_mode == "single_plan_student"
                        else audit_model.network_only_codec_policy
                    )
                    named_now = dict(policy_module_for_audit.named_parameters())
                    for group_name, prefixes in grouped.items():
                        grad_sq = 0.0
                        update_sq = 0.0
                        for name, parameter in named_now.items():
                            if not name.startswith(prefixes):
                                continue
                            if parameter.grad is not None:
                                grad_sq += float(parameter.grad.detach().float().pow(2).sum().cpu())
                            before = network_only_param_before.get(name)
                            if torch.is_tensor(before):
                                update_sq += float(
                                    (parameter.detach() - before).float().pow(2).sum().cpu()
                                )
                        network_only_head_audit[f"{group_name}_grad_norm"] = grad_sq ** 0.5
                        network_only_head_audit[f"{group_name}_update_norm"] = update_sq ** 0.5
                    comp_debug.update({
                        f"network_only_{key}": value
                        for key, value in network_only_head_audit.items()
                    })
                    loss.last_compression_debug = comp_debug
                network_only_param_before = None
                if timing_enabled:
                    sync_for_timing(use_cuda)
                    timing_step_end = time.time()
                epoch_has_optimizer_step = epoch_has_optimizer_step or step_completed # このEpoch内で一回でも更新が成功したかを記録
                if skip_optimizer_reason is not None or not total_loss_finite:
                    args._last_grad_flow = {} # backwardしていないskip stepでは前stepの勾配値をCSVへ持ち越さない
                operation_metric_row = attach_grad_flow_to_operation_row(operation_metric_row, args) # backward後に得られた各操作headの勾配normをOperation CSV行へ反映する
                if log_this_step and compact_step_text_log:
                    log_compact_step_grad(writer, step, num_steps, args)
                if _phase7_should_save_eval_summary(args, global_train_step):
                    phase7_eval_summary_row = _phase7_build_eval_summary_row(
                        args,
                        global_step=global_train_step,
                        episode=episode,
                        epoch=epoch,
                        step=step,
                        stage=current_stage,
                        comp_debug=comp_debug,
                        L_geom=L_geom,
                        L_com=L_com,
                    )
                    append_csv_row(
                        metric_csv_paths.get("phase7_eval_summary"),
                        PHASE7_EVAL_SUMMARY_COLUMNS,
                        phase7_eval_summary_row,
                    )
                append_csv_row( metric_csv_paths.get("compression_step"), COMPRESSION_METRIC_COLUMNS, compression_metric_row) # 圧縮メトリクスのStep単位CSV1行追記
                accumulate_compression_episode(episode_compression_sums, compression_metric_row) # Step単位の圧縮メトリクスをEpisode累積器へ加算する
                append_csv_row( metric_csv_paths.get("operation_step"), OPERATION_METRIC_COLUMNS, operation_metric_row) # 点操作メトリクスのStep単位CSVへ1行追記
                accumulate_operation_episode(episode_operation_sums, operation_metric_row) # Step単位の点操作メトリクスをEpisode累積器へ加算
                if str(getattr(args, "sparsepcgc_training_mode", "subtree_selector")).strip().lower() == "full_cloud_amount":
                    seq_summary = episode_sequence_summary.get(sequence_name, None)
                    if seq_summary is None:
                        seq_summary = {
                            "episode": int(episode) + 1,
                            "epoch": int(epoch) + 1,
                            "sequence_name": str(sequence_name),
                            "step_count": 0,
                            "_actual_sum": 0.0,
                            "_actual_count": 0,
                            "_compression_loss_sum": 0.0,
                            "_compression_loss_count": 0,
                            "_ratio_sum": 0.0,
                            "_ratio_count": 0,
                            "_teacher_ratio_sum": 0.0,
                            "_teacher_ratio_count": 0,
                            "_oracle_ratio_sum": 0.0,
                            "_oracle_ratio_count": 0,
                            "_selected_ratio_sum": 0.0,
                            "_selected_ratio_count": 0,
                            "_raw_oracle_ratio_sum": 0.0,
                            "_raw_oracle_ratio_count": 0,
                            "_selected_best_sum": 0.0,
                            "_selected_best_count": 0,
                            "_selected_raw_best_sum": 0.0,
                            "_selected_raw_best_count": 0,
                            "_oracle_gap_sum": 0.0,
                            "_oracle_gap_count": 0,
                            "_raw_oracle_gap_sum": 0.0,
                            "_raw_oracle_gap_count": 0,
                            "_wide_probe_actual_count_sum": 0.0,
                            "_wide_probe_actual_count_count": 0,
                            "_sequence_memory_ratio_sum": 0.0,
                            "_sequence_memory_ratio_count": 0,
                            "_amount_rd_score_sum": 0.0,
                            "_amount_rd_score_count": 0,
                            "_amount_temperature_sum": 0.0,
                            "_amount_temperature_count": 0,
                            "_sequence_amount_baseline_sum": 0.0,
                            "_sequence_amount_baseline_count": 0,
                            "_selected_action_log_prob_sum": 0.0,
                            "_selected_action_log_prob_count": 0,
                            "_amount_entropy_sum": 0.0,
                            "_amount_entropy_count": 0,
                            "_amount_policy_loss_sum": 0.0,
                            "_amount_policy_loss_count": 0,
                            "_amount_value_loss_sum": 0.0,
                            "_amount_value_loss_count": 0,
                            "_amount_advantage_sum": 0.0,
                            "_amount_advantage_count": 0,
                            "_selected_amount_class_sum": 0.0,
                            "_selected_amount_class_count": 0,
                            "_amount_max_class_rate_sum": 0.0,
                            "_amount_max_class_rate_count": 0,
                            "_selected_ratio_sq_sum": 0.0,
                            "_selected_ratio_sq_count": 0,
                            "_amount_class_histogram_last": "",
                        }
                        episode_sequence_summary[sequence_name] = seq_summary
                    seq_summary["step_count"] += 1
                    row_actual = case_float(
                        compression_metric_row.get("actual_train_objective_percent", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_actual):
                        seq_summary["_actual_sum"] += float(row_actual)
                        seq_summary["_actual_count"] += 1
                    row_compression_loss = case_float(
                        compression_metric_row.get("compression_loss_used", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_compression_loss):
                        seq_summary["_compression_loss_sum"] += float(row_compression_loss)
                        seq_summary["_compression_loss_count"] += 1
                    row_ratio = case_float(
                        compression_metric_row.get("full_cloud_amount_final_ratio", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_ratio):
                        seq_summary["_ratio_sum"] += float(row_ratio)
                        seq_summary["_ratio_count"] += 1
                    row_selected_ratio = case_float(
                        compression_metric_row.get("full_cloud_amount_selected_ratio", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_selected_ratio):
                        seq_summary["_selected_ratio_sum"] += float(row_selected_ratio)
                        seq_summary["_selected_ratio_count"] += 1
                        seq_summary["_selected_ratio_sq_sum"] += float(row_selected_ratio) * float(row_selected_ratio)
                        seq_summary["_selected_ratio_sq_count"] += 1
                    row_teacher_ratio = case_float(
                        compression_metric_row.get("full_cloud_amount_teacher_ratio", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_teacher_ratio):
                        seq_summary["_teacher_ratio_sum"] += float(row_teacher_ratio)
                        seq_summary["_teacher_ratio_count"] += 1
                    row_oracle_ratio = case_float(
                        compression_metric_row.get("full_cloud_amount_oracle_best_ratio", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_oracle_ratio):
                        seq_summary["_oracle_ratio_sum"] += float(row_oracle_ratio)
                        seq_summary["_oracle_ratio_count"] += 1
                    row_raw_oracle_ratio = case_float(
                        compression_metric_row.get("full_cloud_amount_raw_oracle_best_ratio", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_raw_oracle_ratio):
                        seq_summary["_raw_oracle_ratio_sum"] += float(row_raw_oracle_ratio)
                        seq_summary["_raw_oracle_ratio_count"] += 1
                    seq_summary["_selected_best_sum"] += float(
                        bool(compression_metric_row.get("full_cloud_amount_selected_is_best", False))
                    )
                    seq_summary["_selected_best_count"] += 1
                    row_selected_raw_best = compression_metric_row.get(
                        "full_cloud_amount_selected_is_raw_best",
                        None,
                    )
                    if row_selected_raw_best is not None:
                        seq_summary["_selected_raw_best_sum"] += float(bool(row_selected_raw_best))
                        seq_summary["_selected_raw_best_count"] += 1
                    row_oracle_gap = case_float(
                        compression_metric_row.get("full_cloud_amount_oracle_gap", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_oracle_gap):
                        seq_summary["_oracle_gap_sum"] += float(row_oracle_gap)
                        seq_summary["_oracle_gap_count"] += 1
                    row_raw_oracle_gap = case_float(
                        compression_metric_row.get("full_cloud_amount_raw_oracle_gap", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_raw_oracle_gap):
                        seq_summary["_raw_oracle_gap_sum"] += float(row_raw_oracle_gap)
                        seq_summary["_raw_oracle_gap_count"] += 1
                    row_wide_probe_actual = case_float(
                        compression_metric_row.get("full_cloud_amount_wide_probe_actual_count", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_wide_probe_actual):
                        seq_summary["_wide_probe_actual_count_sum"] += float(row_wide_probe_actual)
                        seq_summary["_wide_probe_actual_count_count"] += 1
                    row_sequence_memory_ratio = case_float(
                        compression_metric_row.get("full_cloud_amount_sequence_memory_ratio", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_sequence_memory_ratio):
                        seq_summary["_sequence_memory_ratio_sum"] += float(row_sequence_memory_ratio)
                        seq_summary["_sequence_memory_ratio_count"] += 1
                    row_amount_rd_score = case_float(
                        compression_metric_row.get("amount_rd_score", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_amount_rd_score):
                        seq_summary["_amount_rd_score_sum"] += float(row_amount_rd_score)
                        seq_summary["_amount_rd_score_count"] += 1
                    row_amount_temperature = case_float(
                        compression_metric_row.get("amount_temperature", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_amount_temperature):
                        seq_summary["_amount_temperature_sum"] += float(row_amount_temperature)
                        seq_summary["_amount_temperature_count"] += 1
                    row_sequence_baseline = case_float(
                        compression_metric_row.get("sequence_amount_baseline", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_sequence_baseline):
                        seq_summary["_sequence_amount_baseline_sum"] += float(row_sequence_baseline)
                        seq_summary["_sequence_amount_baseline_count"] += 1
                    row_selected_log_prob = case_float(
                        compression_metric_row.get("selected_action_log_prob", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_selected_log_prob):
                        seq_summary["_selected_action_log_prob_sum"] += float(row_selected_log_prob)
                        seq_summary["_selected_action_log_prob_count"] += 1
                    row_amount_entropy = case_float(
                        compression_metric_row.get("full_cloud_amount_entropy", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_amount_entropy):
                        seq_summary["_amount_entropy_sum"] += float(row_amount_entropy)
                        seq_summary["_amount_entropy_count"] += 1
                    row_amount_policy_loss = case_float(
                        compression_metric_row.get("amount_policy_loss", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_amount_policy_loss):
                        seq_summary["_amount_policy_loss_sum"] += float(row_amount_policy_loss)
                        seq_summary["_amount_policy_loss_count"] += 1
                    row_amount_value_loss = case_float(
                        compression_metric_row.get("amount_value_loss", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_amount_value_loss):
                        seq_summary["_amount_value_loss_sum"] += float(row_amount_value_loss)
                        seq_summary["_amount_value_loss_count"] += 1
                    row_amount_advantage = case_float(
                        compression_metric_row.get("amount_advantage", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_amount_advantage):
                        seq_summary["_amount_advantage_sum"] += float(row_amount_advantage)
                        seq_summary["_amount_advantage_count"] += 1
                    row_selected_amount_class = case_float(
                        compression_metric_row.get("selected_amount_class", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_selected_amount_class):
                        seq_summary["_selected_amount_class_sum"] += float(row_selected_amount_class)
                        seq_summary["_selected_amount_class_count"] += 1
                    row_amount_max_class_rate = case_float(
                        compression_metric_row.get("amount_max_class_rate", float("nan")),
                        float("nan"),
                    )
                    if math.isfinite(row_amount_max_class_rate):
                        seq_summary["_amount_max_class_rate_sum"] += float(row_amount_max_class_rate)
                        seq_summary["_amount_max_class_rate_count"] += 1
                    seq_summary["_amount_class_histogram_last"] = str(
                        compression_metric_row.get("amount_class_histogram", "")
                    )
                maybe_record_case_debug( args, writer, case_debug_path, case_debug_counts, global_step=global_train_step, episode=episode, epoch=epoch, step=step, file_path=file_path, comp_debug=comp_debug, structure_debug=structure_debug, edit_stats=train_edit_stats, L=L, L_geom=L_geom, L_com=L_com, L_actuator=L_actuator) # 圧縮改善が良いケース・悪いケースを条件に応じてCase Debag CSVへ保存

                """損失ログの記録"""
                if epoch_metric_sums is None:
                    epoch_metric_sums = new_metric_sums(L.device, plot.num_loss) # Epoch内で初めのStepなら損失累積器を作る
                surrogate_compression_metric = surrogate_compression_plot_metric(loss, L_com, L.device) # Surrogate予測の(Mine-GT)*100/GTを通常plotへ渡す
                actual_compression_metric = actual_compression_plot_metric(loss, L.device) # 実codecで測った(Mine-GT)*100/GTを通常plotへ渡す
                policy_actual_metric = policy_actual_compression_plot_metric(loss, L.device) # Network自身の最終出力actualを通常plotへ渡す
                oracle_teacher_metric = oracle_teacher_compression_plot_metric(loss, L.device) # Oracle teacher actualを通常plotへ渡す
                if den6_online_full_cloud:
                    source_model = model.module if hasattr(model, "module") else model
                    source_state = getattr(source_model, "last_actuator_voxel_state", {})
                    source_plan = (
                        source_state.get("ana_den6_exact_residual_plan_debug", {})
                        if isinstance(source_state, dict) else {}
                    )
                    performance_source = str(source_plan.get("performance_source", ""))
                    # Exact anchorのActualをNetwork-only性能として図へ混ぜない。
                    if performance_source == "exact_teacher_anchor":
                        oracle_teacher_metric = actual_compression_metric
                        policy_actual_metric = None
                    elif not bool(source_plan.get("network_only_performance", False)):
                        policy_actual_metric = None
                actual_compression_ratio_metric = actual_compression_ratio_plot_metric(loss, L.device) # 実codecで測った100*Mine/GTを通常plotへ渡す
                surrogate_metrics = surrogate_plot_metrics(loss) # Surrogate教師学習の誤差系列を通常plotへ渡す
                metric_values = [ L, L_geom, surrogate_compression_metric, actual_compression_metric, policy_actual_metric, oracle_teacher_metric, L_attr, L_policy, loss_single, loss_nodes, Lp_out, La_fit, La_rep, L_actuator, *surrogate_metrics, actual_compression_ratio_metric] # plot列順にStep損失をまとめる
                add_metric_sums( epoch_metric_sums, metric_values, L.device) # 現在Stepの損失値をEpoch累積器へ加算
                if episode_metric_sums is None:
                    episode_metric_sums = new_metric_sums(L.device, plot.num_loss) # Episode内で初めのEpochなら損失累積器を作る
                step_metric_values = metric_values # Step/Episode/Checkpointで同じ列順のmetricを使う
                add_metric_sums(episode_metric_sums, step_metric_values, L.device) # 現在Stepの損失一覧
                accumulate_checkpoint_metrics( episode_checkpoint_sums, compression_metric_row, operation_metric_row, step_metric_values) # ChackPoint判定用メトリクス
                if train_edit_stats is None:
                    train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args) # 点操作情報を計算
                plot_edit_stats = dict(train_edit_stats or {})
                if one_plan_full_cloud:
                    # online主経路のpoint-edit CSV/図は3操作だけに固定する。
                    plot_edit_stats = {
                        key: value
                        for key, value in plot_edit_stats.items()
                        if key in {"added_ratio_percent", "deleted_ratio_percent", "adjusted_ratio_percent"}
                    }
                else:
                    plot_edit_stats["oracle_full_cloud_prune_ratio_percent"] = operation_metric_row.get(
                        "oracle_full_cloud_prune_ratio_percent",
                        0.0,
                    )
                plot.record_point_edits("step", global_train_step + 1, plot_edit_stats) # 点操作統計をCSVに記録
                plot.record_occupancy_metrics("step", global_train_step + 1, compression_metric_row) # 図用の占有統計を保持する（詳細テキストログとは独立）
                plot.record_voxel_collision_metrics("step", global_train_step + 1, compression_metric_row) # 図用のVoxel衝突統計を保持する
                plot_step_info = plot.record_metrics("step", global_train_step + 1, step_metric_values) # Step単位の損失値をCSVに保存
                if den6_online_full_cloud:
                    base_model_for_audit = model.module if hasattr(model, "module") else model
                    audit_voxel_state = getattr(base_model_for_audit, "last_actuator_voxel_state", {})
                    audit_plan = (
                        audit_voxel_state.get("ana_den6_exact_residual_plan_debug", {})
                        if isinstance(audit_voxel_state, dict) else {}
                    )
                    audit_compression = dict(getattr(loss, "last_compression_debug", {}) or {})
                    actual_audit = getattr(loss, "_den6_online_actual_audit", {})
                    if not isinstance(actual_audit, dict):
                        actual_audit = {}
                    cache_stats = getattr(args, "_ana_den6_online_cache_stats", {})
                    static_node_cache = {}
                    static_cache_stats_fn = getattr(base_model_for_audit, "input_cache_stats", None)
                    if callable(static_cache_stats_fn):
                        static_node_cache = static_cache_stats_fn()
                    audit_runtime = dict(
                        getattr(base_model_for_audit, "last_runtime_timing", {}) or {}
                    )
                    # one-plan online方式の設計条件を単なるログではなく実行時契約にする。
                    # 旧実装のようにAction/Amountが徐々に0へ落ちても学習を継続する
                    # silent failureを禁止し、最初に壊れたStepで原因を残す。
                    online_invariant_failures = []
                    if int(audit_plan.get("plan_count", 0) or 0) != 1:
                        online_invariant_failures.append("plan_count!=1")
                    if int(audit_plan.get("pool_reference_count", 0) or 0) != 1:
                        online_invariant_failures.append("pool_reference_count!=1")
                    if int(actual_audit.get(
                        "candidate",
                        audit_compression.get("den6_online_candidate_actual_encode_count", 0),
                    ) or 0) != 0:
                        online_invariant_failures.append("candidate_actual_encode_count!=0")
                    if int(actual_audit.get(
                        "edited",
                        audit_compression.get("den6_online_edited_actual_encode_count", 0),
                    ) or 0) != 1:
                        online_invariant_failures.append("edited_actual_encode_count!=1")
                    if list(audit_plan.get("selected_action_mask", [])) != [1, 1, 1]:
                        online_invariant_failures.append("selected_action_mask!=[1,1,1]")
                    selected_counts = dict(audit_plan.get("selected_counts") or {})
                    selected_amounts = dict(audit_plan.get("selected_amount_ratios") or {})
                    for operation in ("Prune", "Add", "Adjust"):
                        if int(selected_counts.get(operation, 0) or 0) <= 0:
                            online_invariant_failures.append(f"{operation}_count<=0")
                        if float(selected_amounts.get(operation, 0.0) or 0.0) <= 0.0:
                            online_invariant_failures.append(f"{operation}_amount<=0")
                    # Exact anchor中はhard forwardがTeacherそのものなので方策grad=0が正しい。
                    # anchor後も、明示的なgrad auditを有効にしたStepだけ同期して検査する。
                    anchor_active = str(audit_plan.get("performance_source", "")) == "exact_teacher_anchor"
                    grad_audit_active = bool(getattr(args, "heuristic_guidance_online_grad_audit", False))
                    if (not anchor_active) and grad_audit_active:
                        for head_name, debug_name in (
                            ("Where", "den6_online_where_grad_norm_before_balance"),
                            ("Amount", "den6_online_amount_grad_norm_before_balance"),
                            ("Action", "den6_online_action_grad_norm_before_balance"),
                            ("Surrogate", "surrogate_grad_norm"),
                        ):
                            if float(audit_compression.get(debug_name, 0.0) or 0.0) <= 0.0:
                                online_invariant_failures.append(f"{head_name}_grad<=0")
                    if online_invariant_failures:
                        raise RuntimeError(
                            "ana_den6 one-plan invariant violation: "
                            + ", ".join(online_invariant_failures)
                        )
                    audit_phase_timing = {}
                    if timing_enabled:
                        audit_phase_timing = {
                            "data": float(timing_data_end - timing_data_start),
                            "model": float(timing_model_end - timing_model_start),
                            "loss": float(timing_loss_end - timing_loss_start),
                            "backward_opt": float(timing_step_end - timing_loss_end),
                        }
                    writer.write(
                        "Den6OnlineAudit: "
                        f"cache={dict(cache_stats) if isinstance(cache_stats, dict) else {}}, "
                        f"plan_count={int(audit_plan.get('plan_count', 0) or 0)}, "
                        f"pool_reference_count={int(audit_plan.get('pool_reference_count', 0) or 0)}, "
                        f"guidance_cpu_hit={bool(audit_plan.get('guidance_cpu_tensor_cache_hit', False))}, "
                        f"static_compatibility={bool(audit_plan.get('static_candidate_compatibility_used', False))}, "
                        f"proposal_source={str(audit_plan.get('proposal_source', ''))}, "
                        f"performance_source={str(audit_plan.get('performance_source', ''))}, "
                        f"network_only_performance={bool(audit_plan.get('network_only_performance', False))}, "
                        f"selected_action_index={int(audit_plan.get('selected_action_index', -1))}, "
                        f"selected_action_mask={list(audit_plan.get('selected_action_mask', []))}, "
                        f"selected_action_count={int(audit_plan.get('selected_action_count', 0) or 0)}, "
                        f"teacher_bootstrap={bool(audit_plan.get('teacher_bootstrap_active', False))}, "
                        f"teacher_bc_loss={case_float(audit_plan.get('teacher_behavior_clone_loss', 0.0), 0.0):.6g}, "
                        f"prior_plan_hash={str(audit_plan.get('plan_hash', ''))}, "
                        f"final_voxel_hash={str(audit_plan.get('final_voxel_hash', ''))}, "
                        f"expected_final_voxel_hash={str(audit_plan.get('expected_final_voxel_hash', ''))}, "
                        f"selected_counts={dict(audit_plan.get('selected_counts') or {})}, "
                        f"selected_amount_ratios={dict(audit_plan.get('selected_amount_ratios') or {})}, "
                        f"add_selection={dict(audit_plan.get('add_selection_diagnostics') or {})}, "
                        f"selected_coord_hashes={dict(audit_plan.get('selected_coord_hashes') or {})}, "
                        f"operation_order={str(audit_plan.get('operation_order', ''))}, "
                        f"amount_mode={str(audit_plan.get('amount_mode', ''))}, "
                        f"amount_bin_ratio={float(audit_plan.get('amount_bin_ratio', 0.0) or 0.0):.7f}, "
                        f"amount_fine_log_residual={float(audit_plan.get('amount_fine_log_residual', 0.0) or 0.0):.7f}, "
                        f"operation_amount_log_residuals={dict(audit_plan.get('operation_amount_log_residuals') or {})}, "
                        f"operation_amount_mean_log_residuals={dict(audit_plan.get('operation_amount_mean_log_residuals') or {})}, "
                        f"amount_total_ratio={float(audit_plan.get('amount_total_ratio_before_count', 0.0) or 0.0):.7f}, "
                        f"residual_alpha={float(audit_plan.get('residual_alpha', 0.0)):.6f}, "
                        f"where_residual_weight={float(audit_plan.get('where_residual_weight', 0.0) or 0.0):.6f}, "
                        f"policy_baseline_source={str(audit_compression.get('den6_online_policy_objective_baseline_source', ''))}, "
                        f"policy_baseline={float(audit_compression.get('den6_online_policy_objective_baseline', 0.0) or 0.0):.6f}, "
                        f"policy_advantage={float(audit_compression.get('den6_online_policy_advantage', 0.0) or 0.0):.6f}, "
                        f"policy_backward_scale={float(audit_compression.get('den6_online_policy_backward_scale', 1.0) or 1.0):.3f}, "
                        f"geometry_policy_source={str(audit_compression.get('den6_online_policy_geometry_policy_baseline_source', ''))}, "
                        f"geometry_policy_advantage={float(audit_compression.get('den6_online_policy_geometry_policy_advantage', 0.0) or 0.0):.6f}, "
                        f"geometry_policy_weighted={float(audit_compression.get('den6_online_policy_geometry_policy_weighted', 0.0) or 0.0):.6f}, "
                        f"residual_candidate_delta=(count={int(audit_plan.get('residual_changed_candidate_count', 0) or 0)}, "
                        f"ratio={float(audit_plan.get('residual_changed_candidate_ratio', 0.0) or 0.0):.6f}), "
                        f"actual_encodes=(baseline={int(actual_audit.get('baseline', audit_compression.get('den6_online_baseline_actual_encode_count', 0)))}, "
                        f"edited={int(actual_audit.get('edited', audit_compression.get('den6_online_edited_actual_encode_count', 0)))}, "
                        f"candidate={int(actual_audit.get('candidate', audit_compression.get('den6_online_candidate_actual_encode_count', 0)))}, "
                        f"worker_requests={int(actual_audit.get('worker_request_count', 0))}, "
                        f"edited_cache_hit={bool(actual_audit.get('edited_result_cache_hit', False))})"
                        f", codec_bits=(baseline={float(audit_compression.get('gt_actual_bit', 0.0)):.1f}, "
                        f"edited={float(audit_compression.get('gen_actual_bit', 0.0)):.1f})"
                        f", static_node_cache=(entries={int(static_node_cache.get('entries', 0) or 0)}, "
                        f"bytes={int(static_node_cache.get('bytes', 0) or 0)}, "
                        f"working_set_bypassed={int(static_node_cache.get('working_set_bypassed', 0) or 0)})"
                        f", cuda_cache_released_before_actual="
                        f"{int(getattr(args, '_den6_online_cuda_cache_released_bytes', 0) or 0) / (1024 ** 2):.1f}MiB"
                        f", grad_norms=(where={float(audit_compression.get('den6_online_where_grad_norm', 0.0) or 0.0):.6g}, "
                        f"amount={float(audit_compression.get('den6_online_amount_grad_norm', 0.0) or 0.0):.6g}, "
                        f"action={float(audit_compression.get('den6_online_action_grad_norm', 0.0) or 0.0):.6g}, "
                        f"surrogate={float(audit_compression.get('surrogate_grad_norm', 0.0) or 0.0):.6g})"
                        f", grad_norms_pre_decision_balance=(where={float(audit_compression.get('den6_online_where_grad_norm_before_balance', 0.0) or 0.0):.6g}, "
                        f"amount={float(audit_compression.get('den6_online_amount_grad_norm_before_balance', 0.0) or 0.0):.6g}, "
                        f"action={float(audit_compression.get('den6_online_action_grad_norm_before_balance', 0.0) or 0.0):.6g})"
                        f", policy=(objective={float(audit_compression.get('den6_online_policy_objective', 0.0) or 0.0):.6g}, "
                        f"baseline={float(audit_compression.get('den6_online_policy_objective_baseline', 0.0) or 0.0):.6g}, "
                        f"advantage={float(audit_compression.get('den6_online_policy_advantage', 0.0) or 0.0):.6g}, "
                        f"log_prob={float(audit_compression.get('den6_online_policy_log_prob', 0.0) or 0.0):.6g}, "
                        f"loss={float(audit_compression.get('den6_online_policy_policy_loss', 0.0) or 0.0):.6g})"
                        f", shadow=(raw={float(audit_compression.get('single_plan_shadow_loss_raw', 0.0) or 0.0):.6g}, "
                        f"scaled={float(audit_compression.get('single_plan_shadow_loss', 0.0) or 0.0):.6g}, "
                        f"scale={float(audit_compression.get('single_plan_shadow_loss_scale', 0.0) or 0.0):.6g}, "
                        f"proposed={float(audit_compression.get('single_plan_shadow_loss_scale_proposed', 0.0) or 0.0):.6g}, "
                        f"reason={str(audit_compression.get('single_plan_shadow_balance_reason', ''))})"
                        f", timing=(step_before_audit={float(time.time() - st_step):.3f}s, "
                        f"actual_total={float(audit_compression.get('actual_encode_time_total', 0.0) or 0.0):.3f}s, "
                        f"actual_gt={float(audit_compression.get('gt_actual_encode_time', 0.0) or 0.0):.3f}s, "
                        f"actual_edited={float(audit_compression.get('gen_actual_encode_time', 0.0) or 0.0):.3f}s, "
                        f"actual_worker={float(audit_compression.get('actual_worker_roundtrip_time', 0.0) or 0.0):.3f}s, "
                        f"actual_transfer={float(audit_compression.get('actual_input_prepare_time', 0.0) or 0.0):.3f}s, "
                        f"actual_ply={float(audit_compression.get('actual_ply_write_time', 0.0) or 0.0):.3f}s), "
                        f"phase={audit_phase_timing}, network={audit_runtime}"
                    )
                if network_only_full_cloud:
                    audit_model = _unwrap_train_model(model)
                    audit_state = getattr(audit_model, "last_actuator_voxel_state", {})
                    audit_plan = (
                        audit_state.get("ana_den6_exact_residual_plan_debug", {})
                        if isinstance(audit_state, dict) else {}
                    )
                    audit_compression = dict(getattr(loss, "last_compression_debug", {}) or {})
                    actual_audit = getattr(loss, "_den6_online_actual_audit", {})
                    if not isinstance(actual_audit, dict):
                        actual_audit = {}
                    baseline_encodes = int(
                        actual_audit.get(
                            "baseline",
                            audit_compression.get("den6_online_baseline_actual_encode_count", 0),
                        ) or 0
                    )
                    edited_encodes = int(
                        actual_audit.get(
                            "edited",
                            audit_compression.get("den6_online_edited_actual_encode_count", 0),
                        ) or 0
                    )
                    candidate_encodes = int(
                        actual_audit.get(
                            "candidate",
                            audit_compression.get("den6_online_candidate_actual_encode_count", 0),
                        ) or 0
                    )
                    contract = {
                        "network_forward_count": int(audit_plan.get("network_forward_count", 0) or 0),
                        "plan_count": int(audit_plan.get("plan_count", 0) or 0),
                        "den6_call_count": int(audit_plan.get("den6_call_count", 0) or 0),
                        "candidate_object_count": int(audit_plan.get("candidate_object_count", 0) or 0),
                        "pool_reference_count": int(audit_plan.get("pool_reference_count", 0) or 0),
                        "behavior_cloning_loss": float(audit_plan.get("behavior_cloning_loss", 0.0) or 0.0),
                        "teacher_plan_reference_count": int(audit_plan.get("teacher_plan_reference_count", 0) or 0),
                        "baseline_actual_encode_count": baseline_encodes,
                        "edited_actual_encode_count": edited_encodes,
                        "candidate_actual_encode_count": candidate_encodes,
                        "total_actual_encode_count": baseline_encodes + edited_encodes + candidate_encodes,
                        "proposal_actual_encode_count": int(
                            actual_audit.get("proposal", 0) or 0
                        ),
                    }
                    expected_edited_actual = (
                        int(getattr(args, "network_k_proposal_count", 8))
                        if k_all_actual_enabled else 1
                    )
                    if single_plan_cache_only_stage:
                        expected_edited_actual = 0
                    failures = []
                    for key, expected in (
                        ("network_forward_count", 1),
                        ("plan_count", 1),
                        ("den6_call_count", 0),
                        ("candidate_object_count", 0),
                        ("pool_reference_count", 0),
                        ("teacher_plan_reference_count", 0),
                        ("edited_actual_encode_count", expected_edited_actual),
                        ("candidate_actual_encode_count", 0),
                    ):
                        if int(contract[key]) != int(expected):
                            failures.append(f"{key}!={expected}")
                    if float(contract["behavior_cloning_loss"]) != 0.0:
                        failures.append("behavior_cloning_loss!=0")
                    if (
                        k_all_actual_enabled
                        and int(contract["proposal_actual_encode_count"])
                        != expected_edited_actual
                    ):
                        failures.append(
                            f"proposal_actual_encode_count!={expected_edited_actual}"
                        )
                    if failures:
                        raise RuntimeError(
                            "network-only one-plan invariant violation: " + ", ".join(failures)
                        )

                    def _audit_scalar(key, default=0.0):
                        value = audit_state.get(key, default) if isinstance(audit_state, dict) else default
                        if torch.is_tensor(value):
                            return float(value.detach().float().mean().cpu())
                        return float(value or default)

                    def _audit_list(key):
                        value = audit_state.get(key, []) if isinstance(audit_state, dict) else []
                        if torch.is_tensor(value):
                            return value.detach().float().reshape(-1).cpu().tolist()
                        return list(value) if isinstance(value, (list, tuple)) else []

                    diversity = getattr(args, "_network_only_diversity", None)
                    if not isinstance(diversity, dict):
                        diversity = {
                            "plan_hashes": [], "ratios": [], "shares": [],
                            "priorities": [], "last_coords": None,
                        }
                        args._network_only_diversity = diversity
                    plan_hash = str(audit_plan.get("plan_hash", ""))
                    current_coords = audit_plan.get("selected_coord_key_set", set())
                    if not isinstance(current_coords, set):
                        current_coords = set(current_coords or [])
                    previous_coords = diversity.get("last_coords")
                    if isinstance(previous_coords, set):
                        union_count = len(previous_coords | current_coords)
                        where_jaccard_distance = (
                            1.0 - len(previous_coords & current_coords) / float(union_count)
                            if union_count > 0 else 0.0
                        )
                    else:
                        where_jaccard_distance = float("nan")
                    diversity["last_coords"] = current_coords
                    diversity["plan_hashes"].append(plan_hash)
                    diversity["ratios"].append(_audit_scalar("network_only_total_ratio_unconstrained"))
                    diversity["shares"].append(_audit_list("network_only_shares_raw"))
                    diversity["priorities"].append(tuple(audit_plan.get("priority_order", [])))
                    history_limit = 1000
                    for history_key in ("plan_hashes", "ratios", "shares", "priorities"):
                        diversity[history_key] = diversity[history_key][-history_limit:]
                    valid_hashes = [value for value in diversity["plan_hashes"] if value]
                    unique_plan_rate = (
                        len(set(valid_hashes)) / float(len(valid_hashes))
                        if valid_hashes else 0.0
                    )
                    same_plan_repeat_rate = 1.0 - unique_plan_rate
                    ratio_std = float(np.std(diversity["ratios"])) if diversity["ratios"] else 0.0
                    shares_array = np.asarray(diversity["shares"], dtype=np.float64)
                    share_std = (
                        np.std(shares_array, axis=0).tolist()
                        if shares_array.ndim == 2 and shares_array.shape[0] > 0 else []
                    )
                    add_direction_hist = np.bincount(
                        np.asarray(audit_plan.get("add_direction_indices", []), dtype=np.int64),
                        minlength=26,
                    ).tolist()
                    adjust_direction_hist = np.bincount(
                        np.asarray(audit_plan.get("adjust_direction_indices", []), dtype=np.int64),
                        minlength=26,
                    ).tolist()
                    compression_weight_for_audit = max(
                        float(getattr(args, "network_only_actual_surrogate_loss_weight", 1.0)),
                        0.0,
                    )
                    geometry_weight_for_audit = max(float(getattr(args, "cp_lambda_geom", 1.0)), 0.0)
                    raw_loss_magnitudes = {
                        "geometry": abs(float(finite_float_or_none(L_geom) or 0.0)),
                        "actual_surrogate_ste": abs(float(finite_float_or_none(L_com_objective) or 0.0)),
                        "surrogate_prediction": abs(float(finite_float_or_none(loss_bit) or 0.0)),
                        "policy_gradient": abs(float(audit_compression.get("den6_online_policy_policy_core_raw", 0.0) or 0.0)),
                        "entropy": abs(float(audit_compression.get("den6_online_policy_entropy_raw", 0.0) or 0.0)),
                        "adaptive_amount_entropy": abs(float(audit_compression.get("den6_online_policy_adaptive_amount_entropy_raw", 0.0) or 0.0)),
                        "interaction_huber": abs(float(audit_compression.get("den6_online_policy_plan_gain_huber", 0.0) or 0.0)),
                    }
                    weighted_loss_magnitudes = {
                        "geometry": geometry_weight_for_audit * raw_loss_magnitudes["geometry"],
                        "actual_surrogate_ste": compression_weight_for_audit * raw_loss_magnitudes["actual_surrogate_ste"],
                        "policy_gradient": abs(float(audit_compression.get("den6_online_policy_policy_core_weighted", 0.0) or 0.0)),
                        "entropy": abs(float(audit_compression.get("den6_online_policy_entropy_weighted", 0.0) or 0.0)),
                        "adaptive_amount_entropy": abs(float(audit_compression.get("den6_online_policy_adaptive_amount_entropy_weighted", 0.0) or 0.0)),
                        "interaction_huber": abs(float(audit_compression.get("den6_online_policy_plan_gain_huber_weighted", 0.0) or 0.0)),
                    }
                    nonzero_weighted = {
                        key: value for key, value in weighted_loss_magnitudes.items()
                        if math.isfinite(value) and value > 1e-12
                    }
                    loss_dominance_ratio = (
                        max(nonzero_weighted.values()) / max(min(nonzero_weighted.values()), 1e-12)
                        if len(nonzero_weighted) >= 2 else 1.0
                    )
                    loss_dominance_warning = loss_dominance_ratio > 100.0

                    k_all_actual_full_log = dict(
                        getattr(audit_model, "last_k_all_actual_debug", {}) or {}
                    )
                    if bool(getattr(args, "network_only_audit_verbose", False)):
                        k_all_actual_text_log = k_all_actual_full_log
                    else:
                        k_all_actual_text_log = {
                            key: k_all_actual_full_log.get(key)
                            for key in (
                                "actual_best_compression_percent",
                                "actual_mean_compression_percent",
                                "actual_improving_plan_count",
                                "actual_zero_plan_count",
                                "actual_best_slot",
                                "critic_selected_slot",
                                "critic_regret_percent",
                                "critic_mae_percent",
                                "critic_sign_match",
                                "unique_executable_plan_count",
                                "exploration_temperature",
                                "exploration_anneal_blocked",
                                "positive_experience_count",
                            )
                            if key in k_all_actual_full_log
                        }
                    writer.write(
                        "NetworkOnlyAudit: "
                        f"counters={contract}, "
                        f"k_all_actual={k_all_actual_text_log}, "
                        f"counts={dict(audit_plan.get('selected_counts') or {})}, "
                        f"ratios={dict(audit_plan.get('selected_amount_ratios') or {})}, "
                        f"shares_raw={_audit_list('network_only_shares_raw')}, "
                        f"shares_hard={_audit_list('network_only_shares')}, "
                        f"shares_mean={_audit_list('network_only_shares_mean')}, "
                        f"total_ratio_raw={_audit_scalar('network_only_total_ratio_raw'):.8f}, "
                        f"total_ratio_unconstrained={_audit_scalar('network_only_total_ratio_unconstrained'):.8f}, "
                        f"total_ratio_hard={_audit_scalar('network_only_total_ratio'):.8f}, "
                        f"total_ratio_mean={_audit_scalar('network_only_total_ratio_mean'):.8f}, "
                        f"gates={_audit_list('operation_gate_hard')}, "
                        f"priorities={_audit_list('network_only_priorities')}, "
                        f"temperature={_audit_scalar('network_only_temperature'):.6f}, "
                        f"threshold={_audit_list('network_only_where_threshold')}, "
                        f"exploration_fraction={_audit_scalar('network_only_exploration_fraction'):.6f}, "
                        f"diversity=(plan_hash={plan_hash}, unique_rate={unique_plan_rate:.6f}, "
                        f"repeat_rate={same_plan_repeat_rate:.6f}, jaccard_distance={where_jaccard_distance:.6f}, "
                        f"ratio_std={ratio_std:.8f}, share_std={share_std}, "
                        f"priority_order={list(audit_plan.get('priority_order', []))}, "
                        f"add_direction_hist={add_direction_hist}, adjust_direction_hist={adjust_direction_hist}), "
                        f"entropy=(where={_audit_scalar('network_only_where_entropy'):.6g}, "
                        f"amount={_audit_scalar('network_only_amount_entropy'):.6g}, "
                        f"action={_audit_scalar('network_only_action_entropy'):.6g}, "
                        f"direction={_audit_scalar('network_only_direction_entropy'):.6g}), "
                        f"gain=(local={_audit_scalar('network_only_predicted_local_gain_sum'):.6g}, "
                        f"interaction={_audit_scalar('network_only_interaction_correction'):.6g}, "
                        f"plan={_audit_scalar('network_only_predicted_plan_gain'):.6g}, "
                        f"actual={float(audit_compression.get('actual_total_bit_percent_fresh', audit_compression.get('actual_total_bit_percent', 0.0)) or 0.0):.6g}), "
                        f"surrogate=(mae={float(audit_compression.get('surrogate_abs_bit_error', 0.0) or 0.0):.6g}, "
                        f"sign_match={audit_compression.get('sign_match_surrogate_actual', '')}), "
                        f"loss_scale=(raw={raw_loss_magnitudes}, weighted={weighted_loss_magnitudes}, "
                        f"dominance_ratio={loss_dominance_ratio:.6g}, warning={loss_dominance_warning}), "
                        f"grad_update={network_only_head_audit}, "
                        f"timing=(step={float(time.time() - st_step):.3f}, "
                        f"actual={float(audit_compression.get('actual_encode_time_total', 0.0) or 0.0):.3f}, "
                        f"network={dict(getattr(audit_model, 'last_runtime_timing', {}) or {})})"
                    )
                    if heuristic_mode == "single_plan_student":
                        single_distill_debug = dict(getattr(
                            audit_model, "last_single_plan_distillation_debug", {}
                        ) or {})
                        writer.write(
                            "SinglePlanDistillationAudit: "
                            f"teacher_hard_apply=0, actual_encode={edited_encodes}, "
                            f"loss={float(single_distill_debug.get('weighted', 0.0) or 0.0):.6g}, "
                            f"prune_reachable={float(single_distill_debug.get('prune_source_reachable', 0.0) or 0.0):.6g}, "
                            f"adjust_reachable={float(single_distill_debug.get('adjust_source_reachable', 0.0) or 0.0):.6g}, "
                            f"prune_raw_recall={float(single_distill_debug.get('prune_raw_topk_recall', 0.0) or 0.0):.6g}, "
                            f"adjust_raw_recall={float(single_distill_debug.get('adjust_raw_topk_recall', 0.0) or 0.0):.6g}, "
                            f"prune_fixed_oracle={float(single_distill_debug.get('prune_fixed_feature_oracle_recall', 0.0) or 0.0):.6g}, "
                            f"adjust_fixed_oracle={float(single_distill_debug.get('adjust_fixed_feature_oracle_recall', 0.0) or 0.0):.6g}, "
                            f"prune_recall={float(single_distill_debug.get('prune_source_recall', 0.0) or 0.0):.6g}, "
                            f"add_target_recall={float(single_distill_debug.get('add_target_recall', 0.0) or 0.0):.6g}, "
                            f"adjust_recall={float(single_distill_debug.get('adjust_source_recall', 0.0) or 0.0):.6g}, "
                            f"direction_recall={float(single_distill_debug.get('adjust_direction_recall', 0.0) or 0.0):.6g}, "
                            f"teacher_role={single_distill_debug.get('teacher_role', '')}"
                        )
                    if heuristic_mode == "network_k_proposal_policy":
                        selected_slot_value = int(round(_audit_scalar("k_proposal_selected_slot")))
                        writer.write(
                            "KProposalAudit: "
                            f"shared_encoder_forward_count={int(audit_state.get('shared_encoder_forward_count', 0) or 0)}, "
                            f"shared_basis_forward_count={int(audit_state.get('shared_basis_forward_count', 0) or 0)}, "
                            f"proposal_count={int(audit_state.get('proposal_count', 0) or 0)}, "
                            f"critic_batch_count={int(audit_state.get('critic_batch_count', 0) or 0)}, "
                            f"selected_plan_count={int(audit_state.get('selected_plan_count', 0) or 0)}, "
                            f"selected_slot={selected_slot_value}, "
                            f"den6={int(audit_state.get('den6_call_count', 0) or 0)}, "
                            f"cache={int(audit_state.get('cache_reference_count', 0) or 0)}, "
                            f"teacher={int(audit_state.get('teacher_reference_count', 0) or 0)}, "
                            f"probe={int(audit_state.get('sparsepcgc_probe_count', 0) or 0)}, "
                            f"candidate_actual={int(audit_state.get('candidate_actual_encode_count', 0) or 0)}, "
                            f"unique_executable={int(audit_state.get('k_proposal_unique_executable_plan_count', -1))}, "
                            f"expected_counts={_audit_list('k_proposal_expected_count')}, "
                            f"executed_counts={_audit_list('k_proposal_executed_count')}, "
                            f"execution_count_mismatch={_audit_list('k_proposal_execution_count_mismatch')}, "
                            f"critic_gain={_audit_list('k_proposal_predicted_gain')}, "
                            f"critic_geometry={_audit_list('k_proposal_predicted_geometry')}, "
                            f"critic_interaction={_audit_list('k_proposal_predicted_interaction')}, "
                            f"critic_uncertainty={_audit_list('k_proposal_uncertainty')}"
                        )
                        offline_state = audit_compression.get(
                            "k_proposal_offline_state_id", ""
                        )
                        if offline_state:
                            offline_raw = {
                                name: float(audit_compression.get(
                                    f"k_proposal_offline_{name}_raw", 0.0
                                ) or 0.0)
                                for name in (
                                    "mode_matching", "theta_supervision", "coverage", "teacher_soft_best",
                                    "voxel_relative_value", "target_set", "direction",
                                    "candidate_value", "ranking", "hard_negative",
                                    "critic_selection", "high_value_diversity", "geometry",
                                    "interaction", "uncertainty_calibration",
                                    "actual_replay_value", "actual_elite_imitation",
                                )
                            }
                            offline_weighted = {
                                name: float(audit_compression.get(
                                    f"k_proposal_offline_{name}_weighted", 0.0
                                ) or 0.0)
                                for name in offline_raw
                            }
                            writer.write(
                                "KProposalOfflineLoss: "
                                f"state={offline_state}, "
                                f"total={float(audit_compression.get('k_proposal_offline_loss', 0.0) or 0.0):.6g}, "
                                f"weight={float(audit_compression.get('k_proposal_offline_loss_weight', 0.0) or 0.0):.6g}, "
                                f"raw={offline_raw}, weighted={offline_weighted}, "
                                f"dominance_ratio={float(audit_compression.get('k_proposal_offline_dominance_ratio', 0.0) or 0.0):.6g}, "
                                f"warning={bool(audit_compression.get('k_proposal_offline_dominance_warning', False))}, "
                                f"add_where_teacher_available={bool(audit_compression.get('k_proposal_offline_add_where_teacher_available', False))}, "
                                f"shortlist_natural_recall={float(audit_compression.get('k_proposal_shortlist_natural_recall', float('nan'))):.6g}, "
                                f"shortlist_training_recall={float(audit_compression.get('k_proposal_shortlist_training_recall', float('nan'))):.6g}, "
                                f"target_reachable_recall={float(audit_compression.get('k_proposal_target_reachable_recall', float('nan'))):.6g}, "
                                f"actual_k_oracle={audit_compression.get('k_proposal_offline_metric_actual_k_oracle', None)}"
                            )
                    if loss_dominance_warning:
                        writer.write(
                            "NetworkOnlyLossScaleWarning: a weighted loss term exceeds another "
                            f"nonzero term by {loss_dominance_ratio:.3f}x; terms={weighted_loss_magnitudes}"
                        )
                if timing_enabled:
                    sync_for_timing(use_cuda)
                    en_step = time.time()

                    if not compact_step_text_log:
                        log_step_timing( writer=writer, args=args, step=step, num_steps=num_steps, epoch=epoch, global_train_step=global_train_step, use_cuda=use_cuda, st_step=st_step, timing_data_start=timing_data_start, timing_data_end=timing_data_end, timing_model_start=timing_model_start, timing_model_end=timing_model_end, timing_noise_start=timing_noise_start, timing_noise_end=timing_noise_end, timing_loss_start=timing_loss_start, timing_loss_end=timing_loss_end, timing_step_end=timing_step_end, en_step=en_step, loss=loss, model=model, KNN_BACKEND=KNN_BACKEND)
                else:
                    en_step = time.time()
                if log_this_step:
                    if not compact_step_text_log:
                        log_point_edit_stats( writer, train_edit_stats, step, num_steps)
                    print( f"Epi{episode + 1}/Epo{epoch + 1}/Step{step + 1}:" f"{en_step-st_step:.4f}s   |   " f"{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
                amp_info["consecutive_amp_skips"] = int(consecutive_amp_skips)
                full_cloud_meta_for_better = {
                    "enabled": True,
                    "input_scope": "full_cloud",
                    "point_count": int(raw_pts_num),
                    "is_anchor_step": True,
                    "anchor_reason": anchor_reason,
                    "loss_scope": "full_cloud_output_vs_full_cloud_input",
                }
                log_for_better_step( for_better_path, args=args, model=model, loss_obj=loss, optimizer=optimizer, global_step=global_train_step, episode=episode, epoch=epoch, step=step, stage=current_stage, stage_factors=stage_factors, compression_row=compression_metric_row, operation_row=operation_metric_row, comp_debug=comp_debug, structure_debug=structure_debug, edit_stats=train_edit_stats, subtree_meta=full_cloud_meta_for_better, loss_values={ "L": L, "L_geom": L_geom, "L_com": L_com, "L_com_objective": L_com_objective, "L_attr": L_attr, "L_policy": L_policy, "L_actuator": L_actuator, "loss_bit": loss_bit, "loss_single": loss_single, "loss_nodes": loss_nodes}, step_completed=step_completed, total_loss_finite=total_loss_finite, amp_info=amp_info, timing={"step_seconds": en_step - st_step})
                # このStepのbackward・計測・記録がすべて終わった後だけ参照を切る。
                # 計算内容は変えず、前Stepのfull-cloud Tensorと次Stepのforwardが
                # 同時にGPU上へ存在することを防ぐ。
                if den6_online_full_cloud:
                    base_model_for_release = _unwrap_train_model(model)
                    release_step_state = getattr(base_model_for_release, "release_step_transient_state", None)
                    if callable(release_step_state):
                        release_step_state()
                    gen_pts = None
                    gen_xyz = None
                    compression_gen_xyz = None
                    final_w = None
                    out_label = None
                    structure_debug = None
                    comp_debug = None
                    L = None
                    L_geom = None
                    L_com = None
                    L_com_objective = None
                    L_attr = None
                    L_policy = None
                    L_actuator = None
                    Lp_out = None
                    La_fit = None
                    La_rep = None
                    loss_bit = None
                    loss_single = None
                    loss_nodes = None
                    final_w_for_loss = None
                    gen_xyz_for_actual = None
                    voxel_restored_actual_debug = None
                    # full-cloud canonical/context Tensorは次Stepで再生成する。
                    # 旧contextを保持したまま次frameを構築すると、点数が異なる
                    # frameごとに数GiBの一時重複が発生する。
                    input_xyz = None
                    pts = None
                    input_pcd = None
                    input_attr_full = None
                    compression_gt_pts = None
                    voxel_collision_input_gt = None
                    full_cloud_canonical_context = None
                    full_octree_context = None
                    # scalar Tensorでもgrad_fnからfull graphを参照するため、
                    # backward・全ログ完了後に内訳の別名もまとめて切る。
                    terms = {}
                    compression_debug_terms = {}
                    compression_grad_terms = {}
                    compression_tensor_debug = {}
                    phase3_terms = {}
                    cp_debug = {}
                    actuator_terms = {}
                    actuator_soft_terms = {}
                    model_soft_terms = {}
                    args_soft_terms = {}
                    full_cloud_amount_terms = None
                    param_update_snapshots = None
                    compression_metric_row = None
                    operation_metric_row = None
                    train_edit_stats = None
                    # train() is one large Python function, so loop-local
                    # autograd scalars otherwise survive into the next step.
                    # Even a scalar grad_fn retains the complete full-cloud
                    # forward graph (several GiB).  Backward, optimizer update,
                    # metrics and logging are complete here, so only references
                    # are released; no arithmetic or gradient is changed.
                    value = None
                    term = None
                    legacy_L_downstream = None
                    legacy_L_total = None
                    L_downstream = None
                    L_discrete_policy = None
                    fallback_proxy = None
                    fallback_anchor = None
                    prune_where_proxy_for_grad = None
                    prune_bit_term = None
                    prune_node_term = None
                    prune_single_term = None
                    prune_rate_term = None
                    prune_geom_term = None
                    amount_proxy = None
                    amount_value = None
                    amount_ratio = None
                    amount_anchor_loss = None
                    prune_amount_grad_delta = None
                    tail_attr_block = None
                    tail_policy_block = None
                    tail_actuator_block = None
                    tail_support_raw = None
                    tail_support_scaled = None
                    compression_support_anchor = None
                    online_policy_loss = None
                    prune_where_grad_terms = []
                    step_grad_loss_items = []
                    audit_voxel_state = {}
                    audit_plan_debug = {}
                    audit_plan = {}
                    metric_values = []
                    step_metric_values = []
                    surrogate_metrics = []
                    # saved-tensor offloadのbackward復元blockと前Stepのloss
                    # bridgeを次のfull-cloud forwardへ持ち越さない。
                    loss.last_geometry_debug = {}
                    loss.last_compression_terms = {}
                    loss.last_compression_debug = {}
                    setattr(args, "_current_sparsepcgc_proposal_terms_by_key", {})
                    setattr(args, "_current_sparsepcgc_proposal_selection_meta", {"enabled": False})
                    setattr(args, "_last_voxel_restored_actual_debug", {})
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _release_cpu_step_memory()
                _record_memory(
                    "step_after_cleanup",
                    episode=episode + 1,
                    epoch=epoch + 1,
                    step=step + 1,
                    global_step=global_train_step,
                    sample=os.path.basename(str(file_path)),
                )
                global_train_step += 1
                max_train_steps = int(getattr(args, "max_train_steps", 0))
                if max_train_steps > 0 and global_train_step >= max_train_steps:
                    writer.write(f"MaxTrainSteps reached: {global_train_step}/{max_train_steps}; stopping debug run.")
                    log_for_better_event( for_better_path, "max_train_steps_reached", global_step=global_train_step, max_train_steps=max_train_steps)
                    writer.flush()
                    return

            """lr scheduler"""
            if epoch_has_optimizer_step:
                scheduler_event = step_scheduler_with_floor( scheduler_steplr, optimizer, args, writer=writer, global_epoch=global_epoch + 1, global_step=global_train_step) # StepLRを進める場合でもLR floorを必ず適用する
                if scheduler_event.get("scheduler_stepped"):
                    scheduler_step_count += 1
                scheduler_event["scheduler_step_count"] = scheduler_step_count
                scheduler_event["current_lr_main"] = optimizer_lrs_safe(optimizer)
                scheduler_event["current_lr_surrogate"] = optimizer_lrs_safe(getattr(loss, "surrogate_optimizer", None))
                log_for_better_event( for_better_path, "scheduler_lr_step", **scheduler_event)
            else:
                writer.write("No successful optimizer step in this epoch; lr_scheduler.step() was skipped.")

            """ログの記録"""
            if epoch_metric_sums is not None: # このEpoch内でStep損失が1回以上累積されているか判定
                epoch_avgs = metric_avgs_to_floats(epoch_metric_sums) # Epoch内で累積した損失合計を件数で割り、PythonのFloatリストへ変換
                plot.epo_avg = epoch_avgs # 計算下Epoch平均損失をPlot管理機に保存
                plot_epoch_info = plot.record_metrics("epo", global_epoch + 1, epoch_avgs) # Epoch単位の平均損失をPlot用CSVへ記録
                log_plot_skip_epoch( writer, plot_epoch_info, global_epoch) # Epoch単位の平均損失をCSVに記録
                writer.write(format_metric_summary("EpochAvg", plot.metric_keys, epoch_avgs))
            epoch_edit_info = plot.record_point_edits("epo", global_epoch + 1) # Epoch内で記録されたStep単位の点編集統計を集計
            plot.record_occupancy_metrics("epo", global_epoch + 1) # Epoch内の占有統計を図用に集計
            plot.record_voxel_collision_metrics("epo", global_epoch + 1) # Epoch内のVoxel衝突統計を図用に集計
            log_epoch_point_edit_average( writer, epoch_edit_info, global_epoch) # Epoch単位の点ん操作統計をログに記録
            global_epoch += 1
            # Epochごとの図生成は通常の詳細Stepログ削減とは独立して残す。
            _record_memory(
                "epoch_before_plots",
                episode=episode + 1,
                epoch=epoch + 1,
                global_step=global_train_step,
                sample=sequence_name,
            )
            plot.plot_loss_curve("step")
            plot.plot_loss_curve("epo")
            plot.plot_point_edit_curve("step")
            plot.plot_point_edit_curve("epo")
            plot.plot_occupancy_curve("step")
            plot.plot_occupancy_curve("epo")
            plot.plot_voxel_collision_curve("step")
            plot.plot_voxel_collision_curve("epo")
            _record_memory(
                "epoch_after_plots",
                episode=episode + 1,
                epoch=epoch + 1,
                global_step=global_train_step,
                sample=sequence_name,
            )
            writer.write(f"Saved step/epoch plots/csv: {plot.save_dir}")
            writer.flush()
        if episode_metric_sums is not None:
            plot.epi_avg = metric_avgs_to_floats(episode_metric_sums)
            plot_episode_info = plot.record_metrics("epi", episode + 1, plot.epi_avg)
            log_plot_skip_episode( writer, plot_episode_info, episode)
        else:
            plot.epi_avg = [None for _ in range(plot.num_loss)]
        writer.write(format_metric_summary("EpisodeAvg", plot.metric_keys, plot.epi_avg))
        episode_edit_info = plot.record_point_edits("epi", episode + 1)
        plot.record_occupancy_metrics("epi", episode + 1)
        plot.record_voxel_collision_metrics("epi", episode + 1)
        log_episode_point_edit_average( writer, episode_edit_info, episode)
        _record_memory(
            "episode_before_plots",
            episode=episode + 1,
            global_step=global_train_step,
        )
        plot.plot_loss_curve("epi")
        plot.plot_point_edit_curve("epi")
        plot.plot_occupancy_curve("epi")
        plot.plot_voxel_collision_curve("epi")
        _record_memory(
            "episode_after_plots",
            episode=episode + 1,
            global_step=global_train_step,
        )
        writer.write(f"Saved episode plots/csv: {plot.save_dir}")
        if _episode_input_common_cache_enabled(args):
            cache_summary = _episode_input_common_cache_summary(args)
            writer.write(
                "EpisodeInputCommonCacheSummary: "
                f"episode={episode + 1}, "
                f"entries={int(cache_summary['entries'])}, "
                f"memory={format_bytes(int(cache_summary['bytes']))}, "
                f"hits={int(cache_summary['hits'])}, "
                f"misses={int(cache_summary['misses'])}, "
                f"sections={'; '.join(cache_summary['sections']) if cache_summary['sections'] else 'none'}"
            )
        writer.flush()
        checkpoint_metrics = finalize_checkpoint_metrics( args, current_stage, episode, plot, episode_checkpoint_sums, checkpoint_gate_refs)
        _record_memory(
            "episode_before_full_cloud_validation",
            episode=episode + 1,
            global_step=global_train_step,
        )
        full_cloud_val = run_episode_full_cloud_validation(
            model=model,
            args=args,
            loss=loss,
            writer=writer,
            seq_datasets=seq_datasets,
            episode=episode,
            global_step=global_train_step,
            use_cuda=use_cuda,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        _record_memory(
            "episode_after_full_cloud_validation",
            episode=episode + 1,
            global_step=global_train_step,
        )
        checkpoint_metrics["full_cloud_val_actual_percent"] = full_cloud_val.get("value")
        checkpoint_metrics["full_cloud_val_actual_count"] = int(full_cloud_val.get("count") or 0)
        if (
            str(checkpoint_metrics.get("checkpoint_actual_source", "")).strip().lower() == "full_cloud"
            and full_cloud_val.get("value") is not None
            and int(full_cloud_val.get("count") or 0) > 0
        ):
            checkpoint_metrics["full_cloud_actual_delta"] = float(full_cloud_val["value"])
            checkpoint_metrics["full_cloud_actual_count"] = int(full_cloud_val["count"])
            checkpoint_metrics["checkpoint_actual_delta"] = float(full_cloud_val["value"])
            checkpoint_metrics["checkpoint_actual_count"] = int(full_cloud_val["count"])
            checkpoint_metrics["checkpoint_eligible"] = True
            checkpoint_metrics["checkpoint_ineligible_reason"] = ""
        optimizer_success_ratio = episode_optimizer_step_count / float(max(episode_optimizer_total_count, 1))
        min_optimizer_success_ratio = float(getattr(args, "checkpoint_min_optimizer_step_ratio", 0.20))
        optimizer_success_ok = optimizer_success_ratio >= min_optimizer_success_ratio
        nonfinite_consecutive_ok = episode_max_consecutive_nonfinite_grad_skips < 2
        checkpoint_reasons = []
        existing_reason = str(checkpoint_metrics.get("checkpoint_ineligible_reason") or "").strip()
        if existing_reason:
            checkpoint_reasons.append(existing_reason)
        if not optimizer_success_ok:
            checkpoint_reasons.append("optimizer_step_success_ratio_low")
        if not nonfinite_consecutive_ok:
            checkpoint_reasons.append("consecutive_nonfinite_grad")
        checkpoint_metrics.update(
            {
                "optimizer_step_count": int(episode_optimizer_step_count),
                "optimizer_total_step_count": int(episode_optimizer_total_count),
                "optimizer_step_success_ratio": float(optimizer_success_ratio),
                "optimizer_success_ok": bool(optimizer_success_ok),
                "episode_nonfinite_grad_skip_count": int(episode_nonfinite_grad_skip_count),
                "episode_max_consecutive_nonfinite_grad_skips": int(episode_max_consecutive_nonfinite_grad_skips),
                "nonfinite_consecutive_ok": bool(nonfinite_consecutive_ok),
                "checkpoint_eligible": bool(
                    checkpoint_metrics.get("checkpoint_eligible", False)
                    and optimizer_success_ok
                    and nonfinite_consecutive_ok
                ),
                "checkpoint_ineligible_reason": ",".join(dict.fromkeys(checkpoint_reasons)),
            }
        )
        writer.write(
            "EpisodeOptimizerSummary: "
            f"episode={episode + 1}, "
            f"optimizer_steps={episode_optimizer_step_count}/{episode_optimizer_total_count}, "
            f"success_ratio={optimizer_success_ratio:.6f}, "
            f"nonfinite_grad_skips={episode_nonfinite_grad_skip_count}, "
            f"max_consecutive_nonfinite_grad_skips={episode_max_consecutive_nonfinite_grad_skips}, "
            f"checkpoint_eligible={checkpoint_metrics['checkpoint_eligible']}, "
            f"reason={checkpoint_metrics.get('checkpoint_ineligible_reason') or 'none'}"
        )
        append_csv_row( metric_csv_paths.get("checkpoint_episode"), CHECKPOINT_METRIC_COLUMNS, checkpoint_metrics)
        compression_episode_metrics = finalize_compression_episode_metrics( episode, current_stage, episode_compression_sums)
        append_csv_row( metric_csv_paths.get("compression_episode"), COMPRESSION_EPISODE_METRIC_COLUMNS, compression_episode_metrics)
        if episode_sequence_summary:
            for seq_summary in episode_sequence_summary.values():
                current_sequence_memory_best = _sparsepcgc_full_cloud_sequence_amount_best(
                    args,
                    seq_summary.get("sequence_name", ""),
                )
                append_csv_row(
                    metric_csv_paths.get("full_cloud_amount_sequence_summary"),
                    FULL_CLOUD_AMOUNT_SEQUENCE_SUMMARY_COLUMNS,
                    {
                        "episode": int(seq_summary.get("episode", episode + 1)),
                        "epoch": int(seq_summary.get("epoch", 0)),
                        "sequence_name": str(seq_summary.get("sequence_name", "")),
                        "step_count": int(seq_summary.get("step_count", 0)),
                        "mean_actual_train_objective_percent": (
                            seq_summary["_actual_sum"] / max(seq_summary["_actual_count"], 1)
                            if int(seq_summary.get("_actual_count", 0)) > 0
                            else None
                        ),
                        "mean_compression_loss_used": (
                            seq_summary["_compression_loss_sum"] / max(seq_summary["_compression_loss_count"], 1)
                            if int(seq_summary.get("_compression_loss_count", 0)) > 0
                            else None
                        ),
                        "mean_full_cloud_amount_final_ratio": (
                            seq_summary["_ratio_sum"] / max(seq_summary["_ratio_count"], 1)
                            if int(seq_summary.get("_ratio_count", 0)) > 0
                            else None
                        ),
                        "mean_selected_ratio": (
                            seq_summary["_selected_ratio_sum"] / max(seq_summary["_selected_ratio_count"], 1)
                            if int(seq_summary.get("_selected_ratio_count", 0)) > 0
                            else None
                        ),
                        "mean_teacher_ratio": (
                            seq_summary["_teacher_ratio_sum"] / max(seq_summary["_teacher_ratio_count"], 1)
                            if int(seq_summary.get("_teacher_ratio_count", 0)) > 0
                            else None
                        ),
                        "mean_oracle_best_ratio": (
                            seq_summary["_oracle_ratio_sum"] / max(seq_summary["_oracle_ratio_count"], 1)
                            if int(seq_summary.get("_oracle_ratio_count", 0)) > 0
                            else None
                        ),
                        "mean_raw_oracle_best_ratio": (
                            seq_summary["_raw_oracle_ratio_sum"] / max(seq_summary["_raw_oracle_ratio_count"], 1)
                            if int(seq_summary.get("_raw_oracle_ratio_count", 0)) > 0
                            else None
                        ),
                        "selected_is_best_rate": (
                            seq_summary["_selected_best_sum"] / max(seq_summary["_selected_best_count"], 1)
                            if int(seq_summary.get("_selected_best_count", 0)) > 0
                            else None
                        ),
                        "selected_is_raw_best_rate": (
                            seq_summary["_selected_raw_best_sum"] / max(seq_summary["_selected_raw_best_count"], 1)
                            if int(seq_summary.get("_selected_raw_best_count", 0)) > 0
                            else None
                        ),
                        "mean_oracle_gap": (
                            seq_summary["_oracle_gap_sum"] / max(seq_summary["_oracle_gap_count"], 1)
                            if int(seq_summary.get("_oracle_gap_count", 0)) > 0
                            else None
                        ),
                        "mean_raw_oracle_gap": (
                            seq_summary["_raw_oracle_gap_sum"] / max(seq_summary["_raw_oracle_gap_count"], 1)
                            if int(seq_summary.get("_raw_oracle_gap_count", 0)) > 0
                            else None
                        ),
                        "sequence_memory_best_ratio": (
                            float(current_sequence_memory_best.get("ratio", float("nan")))
                            if isinstance(current_sequence_memory_best, dict)
                            else (
                                seq_summary["_sequence_memory_ratio_sum"] / max(seq_summary["_sequence_memory_ratio_count"], 1)
                                if int(seq_summary.get("_sequence_memory_ratio_count", 0)) > 0
                                else None
                            )
                        ),
                        "wide_probe_actual_count": (
                            seq_summary["_wide_probe_actual_count_sum"] / max(seq_summary["_wide_probe_actual_count_count"], 1)
                            if int(seq_summary.get("_wide_probe_actual_count_count", 0)) > 0
                            else None
                        ),
                        "mean_amount_rd_score": (
                            seq_summary["_amount_rd_score_sum"] / max(seq_summary["_amount_rd_score_count"], 1)
                            if int(seq_summary.get("_amount_rd_score_count", 0)) > 0
                            else None
                        ),
                        "mean_amount_temperature": (
                            seq_summary["_amount_temperature_sum"] / max(seq_summary["_amount_temperature_count"], 1)
                            if int(seq_summary.get("_amount_temperature_count", 0)) > 0
                            else None
                        ),
                        "mean_sequence_amount_baseline": (
                            seq_summary["_sequence_amount_baseline_sum"] / max(seq_summary["_sequence_amount_baseline_count"], 1)
                            if int(seq_summary.get("_sequence_amount_baseline_count", 0)) > 0
                            else None
                        ),
                        "mean_selected_action_log_prob": (
                            seq_summary["_selected_action_log_prob_sum"] / max(seq_summary["_selected_action_log_prob_count"], 1)
                            if int(seq_summary.get("_selected_action_log_prob_count", 0)) > 0
                            else None
                        ),
                        "mean_amount_entropy": (
                            seq_summary["_amount_entropy_sum"] / max(seq_summary["_amount_entropy_count"], 1)
                            if int(seq_summary.get("_amount_entropy_count", 0)) > 0
                            else None
                        ),
                        "mean_amount_policy_loss": (
                            seq_summary["_amount_policy_loss_sum"] / max(seq_summary["_amount_policy_loss_count"], 1)
                            if int(seq_summary.get("_amount_policy_loss_count", 0)) > 0
                            else None
                        ),
                        "mean_amount_value_loss": (
                            seq_summary["_amount_value_loss_sum"] / max(seq_summary["_amount_value_loss_count"], 1)
                            if int(seq_summary.get("_amount_value_loss_count", 0)) > 0
                            else None
                        ),
                        "mean_amount_advantage": (
                            seq_summary["_amount_advantage_sum"] / max(seq_summary["_amount_advantage_count"], 1)
                            if int(seq_summary.get("_amount_advantage_count", 0)) > 0
                            else None
                        ),
                        "mean_selected_amount_class": (
                            seq_summary["_selected_amount_class_sum"] / max(seq_summary["_selected_amount_class_count"], 1)
                            if int(seq_summary.get("_selected_amount_class_count", 0)) > 0
                            else None
                        ),
                        "amount_class_histogram_last": str(seq_summary.get("_amount_class_histogram_last", "")),
                        "amount_max_class_rate_mean": (
                            seq_summary["_amount_max_class_rate_sum"] / max(seq_summary["_amount_max_class_rate_count"], 1)
                            if int(seq_summary.get("_amount_max_class_rate_count", 0)) > 0
                            else None
                        ),
                        "amount_selected_ratio_std": (
                            math.sqrt(
                                max(
                                    0.0,
                                    seq_summary["_selected_ratio_sq_sum"] / max(seq_summary["_selected_ratio_sq_count"], 1)
                                    - (
                                        seq_summary["_selected_ratio_sum"] / max(seq_summary["_selected_ratio_count"], 1)
                                    ) ** 2,
                                )
                            )
                            if int(seq_summary.get("_selected_ratio_sq_count", 0)) > 0
                            else None
                        ),
                    },
                )
        operation_episode_metrics = finalize_operation_episode_metrics( episode, current_stage, episode_operation_sums)
        append_csv_row( metric_csv_paths.get("operation_episode"), OPERATION_EPISODE_METRIC_COLUMNS, operation_episode_metrics)
        writer.write(
            "EpisodeCompressionDiagnostics: "
            f"episode={episode + 1}, "
            f"anchor_raw={case_float(compression_episode_metrics.get('mean_anchor_actual_raw', float('nan')), float('nan')):.6f}, "
            f"subtree_raw={case_float(compression_episode_metrics.get('mean_subtree_actual_raw', float('nan')), float('nan')):.6f}, "
            f"subtree_good={int(case_float(compression_episode_metrics.get('subtree_good_count', 0), 0))}, "
            f"subtree_neutral={int(case_float(compression_episode_metrics.get('subtree_neutral_count', 0), 0))}, "
            f"subtree_bad={int(case_float(compression_episode_metrics.get('subtree_bad_count', 0), 0))}, "
            f"outcome_good={int(case_float(compression_episode_metrics.get('outcome_good_count', 0), 0))}, "
            f"outcome_bad={int(case_float(compression_episode_metrics.get('outcome_bad_count', 0), 0))}, "
            f"surrogate_trust_mean={case_float(compression_episode_metrics.get('surrogate_trust_mean', float('nan')), float('nan')):.6f}, "
            f"anchor_success_memory_count={int(case_float(compression_episode_metrics.get('anchor_success_memory_count', 0), 0))}"
        )

        # 毎エピソードと最高スコアのモデルを保存
        best_loss, model_path, best_trackers = save_episode_checkpoint( model=model, ckpt_dir=ckpt_dir, plot=plot, writer=writer, episode=episode, best_loss=best_loss, args=args, stage=current_stage, checkpoint_metrics=checkpoint_metrics, best_trackers=best_trackers, loss=loss)
        if bool(getattr(args, "phase7_eval_summary", True)):
            try:
                latest_phase7_summary = {
                    "episode": int(episode),
                    "stage": str(current_stage),
                    "model_path": str(model_path),
                    "phase7_ablation_mode": str(
                        getattr(args, "_phase7_ablation_effective_mode", getattr(args, "phase7_ablation_mode", "none"))
                    ),
                    "checkpoint_metrics": checkpoint_metrics,
                }
                phase7_json_path = os.path.join(str(ckpt_dir), "phase7_latest_checkpoint_summary.json")
                with open(phase7_json_path, "w", encoding="utf-8") as handle:
                    import json
                    json.dump(latest_phase7_summary, handle, ensure_ascii=False, indent=2, default=str)

                if model_path:
                    best_phase7_json_path = os.path.join(str(ckpt_dir), "phase7_best_checkpoint_summary.json")
                    with open(best_phase7_json_path, "w", encoding="utf-8") as handle:
                        import json
                        json.dump(latest_phase7_summary, handle, ensure_ascii=False, indent=2, default=str)
            except Exception as exc:
                writer.write(f"Phase7EvalSummaryCheckpointSaveWarning: {type(exc).__name__}: {exc}")
                
        guard_event = apply_actual_compression_guard( args=args, model=model, loss=loss, optimizer=optimizer, writer=writer, guard_state=actual_guard_state, checkpoint_metrics=checkpoint_metrics, ckpt_dir=ckpt_dir, episode=episode)
        if guard_event:
            guard_event["global_step"] = global_train_step
            guard_event["current_lr_main"] = optimizer_lrs_safe(optimizer)
            guard_event["current_lr_surrogate"] = optimizer_lrs_safe(getattr(loss, "surrogate_optimizer", None))
            guard_event["L_total"] =    (L) if "L" in locals() else None
            guard_event["L_com"] = finite_float_or_none(L_com) if "L_com" in locals() else None
            # guard_event["L_total"] = scalar_value(L) if "L" in locals() else None
            # guard_event["L_com"] = scalar_value(L_com) if "L_com" in locals() else None
            log_for_better_event( for_better_path, "actual_compression_guard", episode=episode, stage=current_stage, **guard_event)
        log_for_better_episode( for_better_path, args=args, episode=episode, stage=current_stage, checkpoint_metrics=checkpoint_metrics, compression_episode_metrics=compression_episode_metrics, operation_episode_metrics=operation_episode_metrics, best_trackers=best_trackers, model_path=model_path)
        if notifier is not None:
            notifier.episode_finished( episode=episode + 1, total_episodes=args.episodes, loss_value=float(plot.epi_loss_return()), model_path=model_path, log_path=getattr(writer, "file_path", None))
    _record_memory(
        "train_complete",
        episode=int(args.episodes),
        global_step=global_train_step,
    )
    memory_diagnostics.close()
    return best_loss

if __name__ == '__main__':
    """=== セットアップ ==="""
    setup_t0 = time.time()
    # トレーニングInfoのセットアップ
    file_day = datetime.datetime.now().strftime('%Y%m%d')
    file_time = datetime.datetime.now().strftime('%H%M%S')

    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--trainORtest', default="train", type=str, help='date')
    args = parse_pugan_args(parser, file_day, file_time)
    if bool(getattr(args, "print_phase7_recommended_commands", False)):
        _print_phase7_recommended_commands_and_exit()
        raise SystemExit(0)
    requested_mp_method = str(getattr(args, "mp_start_method", "auto")).strip().lower()
    if requested_mp_method != "auto":
        current_mp_method = mp.get_start_method(allow_none=True)
        if current_mp_method != requested_mp_method:
            mp.set_start_method(requested_mp_method, force=True)

    if torch.cuda.is_available() and not args.cpu and args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        variable_length_full_cloud = (
            str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
            in {"ana_den6_online", "network_only_codec_policy", "network_k_proposal_policy", "single_plan_student"}
        )
        # 8iの点数はframeごとに変わる。benchmark=Trueだと巨大Conv1dの
        # algorithm/workspace探索が形状ごとに増え続けるため、この経路では
        # 固定algorithmを使う。dtype・入力・学習計算は変更しない。
        torch.backends.cudnn.benchmark = bool(
            not getattr(args, "deterministic", False)
            and not variable_length_full_cloud
        )
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    # ログのセットアップ
    writer = Writing( args, file_day, file_time, filename="MyNetwork_train", flush_every=args.log_flush_every, sync_every=args.log_sync_every, log_root=args.log_root)
    writer.write(f"SetupTiming: writer_init={time.time() - setup_t0:.3f}s")
    runtime_knn_backend = configure_knn_backend(args, writer=writer)
    globals()["KNN_BACKEND"] = runtime_knn_backend
    network_module.KNN_BACKEND = runtime_knn_backend
    setup_plot_t0 = time.time()
    plot = PlotMaker(args)
    writer.write(f"SetupTiming: plot_init={time.time() - setup_plot_t0:.3f}s")

    log_training_setup( writer, args, file_day, file_time)
    # ============================================================
    # Direct Network Prune 起動確認
    # ============================================================
    if bool(getattr(args, "direct_network_prune", False)):
        writer.write(
            "DirectNetworkPrune: ACTIVE, "
            f"prune_after_prior_mode={getattr(args, 'sparsepcgc_prune_after_prior_mode', '')}, "
            f"codec_prior={getattr(args, 'sparsepcgc_codec_prune_prior', None)}, "
            f"actual_gate_prune={getattr(args, 'sparsepcgc_actual_gate_prune', None)}, "
            f"noop_guard={getattr(args, 'sparsepcgc_policy_actual_noop_guard', None)}, "
            f"full_cloud_primary={getattr(args, 'sparsepcgc_full_cloud_actual_primary', None)}, "
            f"full_cloud_correction={getattr(args, 'full_cloud_actual_correction_loss_enable', None)}"
        )
    else:
        writer.write(
            "DirectNetworkPrune: INACTIVE. "
            "この状態ではPhase0後にoracle/gateでPruneが止まる可能性がある。"
        )
    notifier = TrainingMailNotifier.from_args(args, writer=writer)

    setup_model_t0 = time.time()
    model = Network(args, writer)
    writer.write(f"SetupTiming: model_init={time.time() - setup_model_t0:.3f}s")

    setup_ckpt_t0 = time.time()
    repkpu_ckpt = os.path.join(os.path.dirname(__file__), "repkpu_model", "ckpt-best.pth")
    ckpt = torch.load(repkpu_ckpt, map_location="cpu")
    encoder_state = { k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
    encoder_state = adapt_encoder_state_dict_for_sparse_input(model, encoder_state, writer=writer)
    model.encoder.load_state_dict(encoder_state, strict=False)
    for p in model.encoder.parameters():
        # Single-Plan representation段階だけはargs正規化でencoder_0grad=Falseに
        # している。Stage 2/3と既存modeでは従来どおり固定する。
        p.requires_grad = not bool(getattr(args, "encoder_0grad", True))
    writer.write("RepKPU encoder loaded: repkpu_model/ckpt-best.pth")
    writer.write(f"SetupTiming: encoder_ckpt_load={time.time() - setup_ckpt_t0:.3f}s")

    # more_training=Trueなら、追加学習用checkpointからモデル全体のパラメータを読み込む
    setup_more_training_t0 = time.time()
    model = load_more_training_checkpoint(model, args, writer)
    writer.write(f"SetupTiming: more_training_ckpt_load={time.time() - setup_more_training_t0:.3f}s")

    if args.cpu is False and torch.cuda.is_available():
        setup_cuda_t0 = time.time()
        model = model.cuda()
        writer.write(f"SetupTiming: model_to_cuda={time.time() - setup_cuda_t0:.3f}s")

    _register_prune_where_head_grad_scale_hook(args, model, writer=writer)

    setup_loss_t0 = time.time()
    loss = Loss(args, file_day + "-" + file_time, writer)
    writer.write(f"SetupTiming: loss_init={time.time() - setup_loss_t0:.3f}s")
    writer.write(f"SetupTiming: total_before_train={time.time() - setup_t0:.3f}s")

    st = time.time()
    writer.write("=== Start Training ===")
    notifier.training_started( start_date=datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'), log_path=getattr(writer, "file_path", None))
    best_loss = None
    try:
        best_loss = train(model, args, loss, writer, plot, notifier=notifier)
        en = time.time()
        finish_date = datetime.datetime.now().strftime('%Y/%m/%d - %H:%M:%S')
        writer.write(f"Training time: {en - st}")
        writer.write(f"Date of finishing training: {finish_date}")
        notifier.training_finished( elapsed_sec=en - st, finish_date=finish_date, best_loss=best_loss, log_path=getattr(writer, "file_path", None))
    except Exception as exc:
        try:
            writer.write(f"Training error: {type(exc).__name__}: {exc}")
        finally:
            notifier.training_error(exc, log_path=getattr(writer, "file_path", None))
        raise
    finally:
        shutdown_ana_den6_online_prefetch(wait=False)
        writer.close()
